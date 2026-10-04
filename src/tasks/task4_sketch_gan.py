"""Task 4 - Style-conditioned face-to-sketch cGAN (U-Net generator + PatchGAN discriminator).

L_D = 0.5 * [BCE(D(x, y, s), 1) + BCE(D(x, G(x, s), s), 0)]
L_G = BCE(D(x, G(x, s), s), 1) + lambda_L1 * |y - G(x, s)|_1
Optuna (short trials) tunes lr_G, lr_D, batch, base channels, dropout, embedding dim, lambda_L1;
the best configuration is retrained for the full schedule. The official test set is used
only once, at the end.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F

from src import config as C
from src import engine as E
from src.data.fs2k import PairDataset, get_fs2k
from src.metrics import l1, psnr, ssim
from src.models.cgan import PatchDiscriminator, UNetGenerator
from src.models.restoration import count_params

NAME = "task4_sketch_cgan"
to01 = lambda t: (t + 1) / 2


@torch.no_grad()
def validate(G, split, bs=32):
    G.eval()
    ph, sk, st = split
    rows = []
    for x, y, s in E.loader(PairDataset(ph, sk, st), bs):
        x, y, s = x.to(C.DEVICE), y.to(C.DEVICE), s.to(C.DEVICE)
        g = to01(G(x, s)).clamp(0, 1)
        t = to01(y)
        rows.append(torch.stack([l1(g, t, False), ssim(g, t, reduce=False), psnr(g, t, reduce=False),
                                 s.float()], 1).cpu())
    r = torch.cat(rows).numpy()
    G.train()
    return pd.DataFrame(r, columns=["l1", "ssim", "psnr", "style"])


def objective_value(df):
    """Lower is better: pixel error plus structural dissimilarity."""
    return float(df.l1.mean() + 0.5 * (1 - df.ssim.mean()))


@torch.no_grad()
def sample_grid(G, fixed, path, title):
    G.eval()
    x, y, s = fixed
    g = to01(G(x.to(C.DEVICE), s.to(C.DEVICE))).clamp(0, 1).cpu()
    G.train()
    n = len(s)
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for j in range(n):
        for i, (im, lab) in enumerate([(to01(x[j]), "photo"), (to01(y[j]), "real sketch"), (g[j], "generated")]):
            axes[i, j].imshow(im.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(lab, fontsize=8)
        axes[0, j].set_title(f"style {int(s[j]) + 1}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return g


def fixed_val_batch(val, n=6):
    """Same validation photos every time (two per style when possible)."""
    ph, sk, st = val
    ids = []
    for s in range(3):
        ids += np.where(st == s)[0][:2].tolist()
    ids = (ids + list(range(len(st))))[:n]
    ds = PairDataset(ph, sk, st)
    xs, ys, ss = zip(*[ds[i] for i in ids])
    return torch.stack(xs), torch.stack(ys), torch.tensor(ss)


def train(hp, data, epochs, trial=None, log=True, sample_every=None):
    G = UNetGenerator(hp["base"], hp["emb_dim"], hp["dropout"]).to(C.DEVICE)
    D = PatchDiscriminator(hp["base"], hp["emb_dim"]).to(C.DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=hp["lr_g"], betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=hp["lr_d"], betas=(0.5, 0.999))
    # linear decay over the second half (pix2pix schedule)
    decay = lambda ep: 1.0 if ep < epochs // 2 else max(0.0, 1 - (ep - epochs // 2) / max(epochs - epochs // 2, 1))
    schG = torch.optim.lr_scheduler.LambdaLR(optG, decay)
    schD = torch.optim.lr_scheduler.LambdaLR(optD, decay)
    bce = torch.nn.BCEWithLogitsLoss()
    dl = E.loader(PairDataset(*data["train"], augment=True), hp["batch_size"], shuffle=True, drop_last=True)
    fixed = fixed_val_batch(data["val"])
    keys = ["d_real", "d_fake", "g_adv", "g_l1"]
    hist = {k: [] for k in keys} | {"val_l1": [], "val_ssim": [], "val_objective": []}
    best, best_state, snaps = 1e9, None, []
    for ep in range(epochs):
        agg, n = np.zeros(4), 0
        for x, y, s in dl:
            x, y, s = x.to(C.DEVICE), y.to(C.DEVICE), s.to(C.DEVICE)
            fake = G(x, s)
            # ---- discriminator step
            pr, pf = D(x, y, s), D(x, fake.detach(), s)
            d_real, d_fake = bce(pr, torch.ones_like(pr)), bce(pf, torch.zeros_like(pf))
            optD.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            optD.step()
            # ---- generator step
            pg = D(x, fake, s)
            g_adv = bce(pg, torch.ones_like(pg))
            g_l1 = F.l1_loss(fake, y)
            optG.zero_grad(set_to_none=True)
            (g_adv + hp["lambda_l1"] * g_l1).backward()
            optG.step()
            agg += np.array([d_real.item(), d_fake.item(), g_adv.item(), g_l1.item()]); n += 1
        schG.step(); schD.step()
        agg /= max(n, 1)
        v = validate(G, data["val"])
        obj = objective_value(v)
        for k, val in zip(keys, agg):
            hist[k].append(float(val))
        hist["val_l1"].append(float(v.l1.mean())); hist["val_ssim"].append(float(v.ssim.mean()))
        hist["val_objective"].append(obj)
        if log:
            mlflow.log_metrics({**dict(zip(keys, agg)), "val_l1": v.l1.mean(), "val_ssim": v.ssim.mean(),
                                "val_psnr": v.psnr.mean(), "val_objective": obj}, step=ep)
        print(f"  ep {ep + 1}/{epochs} D_real {agg[0]:.3f} D_fake {agg[1]:.3f} G_adv {agg[2]:.3f} "
              f"G_L1 {agg[3]:.3f} | val L1 {v.l1.mean():.4f} SSIM {v.ssim.mean():.4f}")
        if sample_every and ((ep + 1) % sample_every == 0 or ep == 0):
            p = C.out("figures", f"task4_samples_epoch{ep + 1:03d}.png")
            snaps.append((ep + 1, sample_grid(G, fixed, p, f"epoch {ep + 1}")))
            E.log_artifact(p, "samples")
        if obj < best:
            best, best_state = obj, {k: t.detach().cpu().clone() for k, t in G.state_dict().items()}
        if trial is not None:
            trial.report(obj, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    G.load_state_dict(best_state)
    return G, hist, best, snaps, fixed


def tune(data):
    E.mlflow_setup(f"{NAME}_optuna")

    def objective(trial):
        hp = {"lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
              "lr_d": trial.suggest_float("lr_d", 2e-5, 5e-4, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
              "base": trial.suggest_categorical("base", [32, 48, 64]),
              "dropout": trial.suggest_float("dropout", 0.0, 0.5),
              "emb_dim": trial.suggest_categorical("emb_dim", [8, 16, 32]),
              "lambda_l1": trial.suggest_float("lambda_l1", 10, 200, log=True)}
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(hp)
            _, _, best, _, _ = train(hp, data, C.CFG["t4_trial_epochs"], trial)
            mlflow.log_metric("best_val_objective", best)
        return best

    return E.run_study(E.make_study(NAME, direction="minimize"), objective, C.CFG["t4_trials"])


def run():
    C.seed_everything()
    data = get_fs2k()
    print("FS2K sizes:", {k: len(v[2]) for k, v in data.items() if k != "test_names"})
    s = tune(data)
    hp = s["best_params"]
    E.mlflow_setup(f"{NAME}_final")
    with mlflow.start_run(run_name="final"):
        mlflow.log_params({**hp, "epochs": C.CFG["t4_epochs"]})
        G, hist, _, snaps, fixed = train(hp, data, C.CFG["t4_epochs"], sample_every=C.CFG["t4_sample_every"])
        mlflow.log_metric("generator_params", count_params(G))
        E.log_artifact(E.save_ckpt(G, "sketch_generator", hp), "checkpoints")
        E.curves({"discriminator": {"D_real": hist["d_real"], "D_fake": hist["d_fake"]},
                  "generator": {"G_adv": hist["g_adv"]}, "G_L1 (train)": hist["g_l1"],
                  "validation": {"L1": hist["val_l1"], "SSIM": hist["val_ssim"]}},
                 C.out("figures", "task4_curves.png"), title="Task 4 cGAN losses")

        # training progression on the same validation photos
        if snaps:
            pick = snaps if len(snaps) <= 6 else [snaps[i] for i in np.linspace(0, len(snaps) - 1, 6).astype(int)]
            n = fixed[0].shape[0]
            fig, axes = plt.subplots(len(pick) + 1, n, figsize=(1.6 * n, 1.7 * (len(pick) + 1)))
            for j in range(n):
                axes[0, j].imshow(to01(fixed[1][j]).permute(1, 2, 0).clamp(0, 1).numpy())
                for i, (ep, g) in enumerate(pick, start=1):
                    axes[i, j].imshow(g[j].permute(1, 2, 0).numpy())
                    if j == 0:
                        axes[i, j].set_ylabel(f"ep {ep}", fontsize=8)
            axes[0, 0].set_ylabel("target", fontsize=8)
            for a in axes.flat:
                a.set_xticks([]); a.set_yticks([])
            plt.tight_layout(); fig.savefig(C.out("figures", "task4_progression.png"), dpi=110); plt.close(fig)

        # ---- final test evaluation (official test set, used once)
        df = validate(G, data["test"])
        df["style"] = df["style"].astype(int) + 1
        t = pd.concat([df.groupby("style")[["l1", "ssim", "psnr"]].mean(),
                       pd.DataFrame(df[["l1", "ssim", "psnr"]].mean()).T.rename(index={0: "all"})]).round(4)
        t.to_csv(C.out("tables", "task4_test_results.csv"))
        print(t)
        mlflow.log_metrics({"test_l1": df.l1.mean(), "test_ssim": df.ssim.mean(), "test_psnr": df.psnr.mean()})

        ph, sk, st = data["test"]
        G.eval()

        def show(ids, path, title):
            ds = PairDataset(ph, sk, st)
            xs, ys, ss = zip(*[ds[i] for i in ids])
            sample_grid(G, (torch.stack(xs), torch.stack(ys), torch.tensor(ss)), path, title)

        rng = np.random.default_rng(C.SEED)
        show(rng.choice(len(st), size=min(12, len(st)), replace=False).tolist(),
             C.out("figures", "task4_test_examples.png"), "Test examples")
        show(df.sort_values("ssim").index[:4].tolist(), C.out("figures", "task4_failures.png"),
             "Lowest-SSIM test cases")
        # style control: same photo rendered in all three styles
        ids = rng.choice(len(st), size=min(4, len(st)), replace=False)
        fig, axes = plt.subplots(len(ids), 5, figsize=(8, 1.8 * len(ids)))
        axes = np.atleast_2d(axes)
        with torch.no_grad():
            for r, i in enumerate(ids):
                x, y, s = PairDataset(ph, sk, st)[int(i)]
                gens = to01(G(x[None].repeat(3, 1, 1, 1).to(C.DEVICE), torch.arange(3, device=C.DEVICE))).clamp(0, 1).cpu()
                ims = [to01(x), to01(y)] + list(gens)
                heads = ["photo", f"real (style {s + 1})", "style 1", "style 2", "style 3"]
                for c in range(5):
                    axes[r, c].imshow(ims[c].permute(1, 2, 0).clamp(0, 1).numpy())
                    axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                    if r == 0:
                        axes[r, c].set_title(heads[c], fontsize=8)
        plt.tight_layout(); fig.savefig(C.out("figures", "task4_style_control.png"), dpi=110); plt.close(fig)
        for f in ["task4_curves.png", "task4_progression.png", "task4_test_examples.png", "task4_failures.png",
                  "task4_style_control.png"]:
            if C.out("figures", f).exists():
                E.log_artifact(C.out("figures", f), "figures")
        E.log_artifact(C.out("tables", "task4_test_results.csv"))


if __name__ == "__main__":
    run()
