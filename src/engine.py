"""Shared helpers: data loaders, Optuna studies, MLflow tracking, evaluation and plots."""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config as C
from src.data import corruptions as K
from src.data.pets import ManifestDataset
from src.metrics import l1, psnr, ssim


# ----------------------------------------------------------------------------- loaders
def loader(ds, batch_size=None, shuffle=False, batch_sampler=None, drop_last=False):
    kw = dict(num_workers=C.NUM_WORKERS, pin_memory=C.DEVICE.type == "cuda",
              persistent_workers=C.NUM_WORKERS > 0)
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    drop_last = drop_last and len(ds) >= batch_size   # never produce an empty epoch
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kw)


# ----------------------------------------------------------------------------- tracking
def mlflow_setup(experiment):
    mlflow.set_tracking_uri((C.OUT / "mlruns").resolve().as_uri())
    mlflow.set_experiment(experiment)


def log_artifact(path, sub=None):
    try:
        mlflow.log_artifact(str(path), sub)
    except Exception as e:  # tracking must never kill a training run
        print("[mlflow] artifact not logged:", e)


# ----------------------------------------------------------------------------- optuna
def make_study(name, direction="maximize"):
    storage = f"sqlite:///{C.out('optuna', name + '.db').resolve().as_posix()}"
    return optuna.create_study(
        study_name=name, storage=storage, direction=direction, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))


def run_study(study, objective, n_trials):
    done = len([t for t in study.trials if t.state.is_finished()])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, gc_after_trial=True)
    return summarize_study(study)


def summarize_study(study):
    name = study.study_name
    df = study.trials_dataframe()
    df.to_csv(C.out("tables", f"{name}_trials.csv"), index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    summary = {"study": name, "direction": study.direction.name, "n_trials": len(study.trials),
               "n_complete": len(complete), "n_pruned": len(pruned),
               "best_trial": study.best_trial.number, "best_value": study.best_value,
               "best_params": study.best_params}
    C.save_json(summary, C.out("tables", f"{name}_best.json"))
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(study)
        ax.figure.savefig(C.out("figures", f"{name}_history.png"), dpi=130, bbox_inches="tight")
        plt.close("all")
        if len(complete) >= 3:
            ax = plot_param_importances(study)
            ax.figure.savefig(C.out("figures", f"{name}_importance.png"), dpi=130, bbox_inches="tight")
            plt.close("all")
    except Exception as e:
        print("[optuna] plot skipped:", e)
    print(f"[optuna] {name}: best #{summary['best_trial']} = {summary['best_value']:.4f} {summary['best_params']}")
    return summary


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_restorer(restore_fn, images, entries, batch_size=128, classes=None, keep=False):
    """Run restore_fn over manifest entries. restore_fn(noisy[B,3,H,W] on device, labels) ->
    dict with 'out' plus optional per-sample extras (tensors with leading B dimension).

    Returns a per-entry DataFrame: cls, severity, PSNR/SSIM/L1 of the corrupted input and
    of the restoration, plus extras (e.g. predicted class, routing weights)."""
    ds = ManifestDataset(images, entries, classes)
    rows, kept = [], {}
    for noisy, clean, lab, j in loader(ds, batch_size):
        noisy, clean, lab = noisy.to(C.DEVICE), clean.to(C.DEVICE), lab.to(C.DEVICE)
        res = restore_fn(noisy, lab)
        out = res["out"].clamp(0, 1)
        m = {"psnr_in": psnr(noisy, clean, reduce=False), "ssim_in": ssim(noisy, clean, reduce=False),
             "psnr_out": psnr(out, clean, reduce=False), "ssim_out": ssim(out, clean, reduce=False),
             "l1_out": l1(out, clean, reduce=False)}
        m = {k: v.cpu().numpy() for k, v in m.items()}
        extras = {k: v.detach().cpu().numpy() for k, v in res.items() if k != "out"}
        for b in range(len(j)):
            e = ds.entries[int(j[b])]
            r = {"entry": int(j[b]), "index": e["index"], "cls": e["cls"], "severity": e["severity"]}
            r.update({k: float(v[b]) for k, v in m.items()})
            for k, v in extras.items():
                if v.ndim == 1:
                    r[k] = v[b].item()
                else:
                    for c, val in enumerate(v[b].reshape(-1)):
                        r[f"{k}_{c}"] = float(val)
            rows.append(r)
        if keep:
            for b in range(len(j)):
                kept[int(j[b])] = (noisy[b].cpu(), out[b].cpu(), clean[b].cpu())
    df = pd.DataFrame(rows)
    return (df, kept) if keep else df


def group_table(df, cols=("psnr_in", "psnr_out", "ssim_in", "ssim_out", "l1_out")):
    """Mean metrics per corruption type and severity + per type + overall."""
    cols = [c for c in cols if c in df]
    t1 = df.groupby(["cls", "severity"])[list(cols)].mean()
    t2 = df.groupby("cls")[list(cols)].mean()
    t2.index = pd.MultiIndex.from_tuples([(c, "all") for c in t2.index])
    t3 = pd.DataFrame(df[list(cols)].mean()).T
    t3.index = pd.MultiIndex.from_tuples([("ALL", "all")])
    t = pd.concat([t1, t2, t3]).sort_index()
    t.index.names = ["cls", "severity"]
    return t.round(4)


@torch.no_grad()
def restore_entries(restore_fn, images, entries, ids):
    """Recompute specific entries (for figures). Returns list of (noisy, out, clean, extras)."""
    ds = ManifestDataset(images, entries)
    res = []
    for i in ids:
        noisy, clean, lab, _ = ds[i]
        r = restore_fn(noisy[None].to(C.DEVICE), torch.tensor([lab], device=C.DEVICE))
        ex = {k: v[0].cpu() for k, v in r.items() if k != "out"}
        res.append((noisy, r["out"][0].clamp(0, 1).cpu(), clean, ex))
    return res


def pick_examples(df, n_per_group=1, seed=C.SEED):
    """One random entry per (cls, severity) group, plus extras up to 12 total."""
    rng = np.random.default_rng(seed)
    ids = []
    for _, g in df.groupby(["cls", "severity"]):
        ids += rng.choice(g["entry"].values, size=min(n_per_group, len(g)), replace=False).tolist()
    rest = df[~df["entry"].isin(ids)]
    if len(ids) < 12 and len(rest):
        ids += rng.choice(rest["entry"].values, size=min(12 - len(ids), len(rest)), replace=False).tolist()
    return ids[:12]


def pick_failures(df, n=4):
    """Worst restoration per corruption type (lowest output SSIM), plus worst gain overall."""
    ids = []
    for cls in ["salt", "blur", "occlusion"]:
        g = df[df.cls == cls]
        if len(g):
            ids.append(int(g.sort_values("ssim_out").iloc[0]["entry"]))
    g = df[~df.entry.isin(ids)].assign(gain=lambda d: d.psnr_out - d.psnr_in).sort_values("gain")
    ids += g["entry"].head(n - len(ids)).astype(int).tolist()
    return ids[:n]


# ----------------------------------------------------------------------------- plots
def to_img(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def restoration_grid(items, path, titles=None, row_labels=None):
    """items: list of (noisy, out, clean[, extras]). Columns: clean, input, output, |error|."""
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(8, 2.1 * n))
    axes = np.atleast_2d(axes)
    heads = ["Clean target", "Corrupted input", "Restored", "|Error| (x3)"]
    for r, it in enumerate(items):
        noisy, out, clean = it[:3]
        err = (out - clean).abs().mean(0).numpy()
        ims = [to_img(clean), to_img(noisy), to_img(out), None]
        for c in range(4):
            ax = axes[r, c]
            if c == 3:
                ax.imshow(np.clip(err * 3, 0, 1), cmap="inferno", vmin=0, vmax=1)
            else:
                ax.imshow(ims[c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(heads[c], fontsize=9)
        if row_labels:
            axes[r, 0].set_ylabel(row_labels[r], fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def curves(history, path, keys=None, title=""):
    keys = keys or [k for k in history if k != "epoch"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        v = history[k]
        if isinstance(v, dict):
            for lab, vv in v.items():
                ax.plot(range(1, len(vv) + 1), vv, label=lab)
            ax.legend(fontsize=7)
        else:
            ax.plot(range(1, len(v) + 1), v)
        ax.set_title(k, fontsize=9); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def heatmap(mat, rows, cols, path, title="", fmt="{:.2f}", cmap="viridis", vmin=0, vmax=1):
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 0.5 * len(rows) + 1.5))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v - vmin) / max(vmax - vmin, 1e-9) < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t


def save_ckpt(model, name, hparams):
    path = C.out("checkpoints", f"{name}.pt")
    torch.save({"state_dict": model.state_dict(), "hparams": hparams}, path)
    return path


def load_ckpt(name):
    return torch.load(C.out("checkpoints", f"{name}.pt"), map_location="cpu", weights_only=False)
