"""Task 2 - Corruption classifier + hard-routed specialist autoencoders.

Stages:
  a) classifier: Optuna (lr, batch, channels, dropout, weight decay) -> final -> metrics + confusion matrix
  b) specialists: ONE shared Optuna search (a trial trains all three briefly, objective = mean score)
     -> three independently trained specialists with the shared best architecture
  c) evaluation in oracle-routing and predicted-routing modes + misrouting analysis
"""
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config as C
from src import engine as E
from src.data.corruptions import CLASSES
from src.data.pets import (BalancedBatchSampler, ManifestDataset, RandomCorruptionDataset, get_arrays,
                           get_manifests)
from src.metrics import restoration_loss, restoration_score
from src.models.restoration import CorruptionClassifier, DenoisingAE

NAME_C = "task2_classifier"
NAME_S = "task2_specialists"
EXPERTS = ["salt", "blur", "occlusion"]


# ============================================================================ classifier
@torch.no_grad()
def predict(model, images, entries, bs=256):
    model.eval()
    ys, ps = [], []
    for x, _, y, _ in E.loader(ManifestDataset(images, entries), bs):
        ps.append(F.softmax(model(x.to(C.DEVICE)), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train_classifier(hp, data, val_entries, epochs, trial=None, log=True):
    model = CorruptionClassifier(base=hp["base"], dropout=hp["dropout"]).to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    # balanced batches: exactly batch/4 images of every class in every batch
    dl = E.loader(RandomCorruptionDataset(data["train"]),
                  batch_sampler=BalancedBatchSampler(len(data["train"]), hp["batch_size"]))
    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        tot, cor, n = 0.0, 0, 0
        for x, _, y in dl:
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); cor += (logits.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        yt, pr = predict(model, data["val"], val_entries)
        acc, f1 = accuracy_score(yt, pr.argmax(1)), f1_score(yt, pr.argmax(1), average="macro")
        for k, v in zip(hist, [tot / n, cor / n, acc, f1]):
            hist[k].append(float(v))
        if log:
            mlflow.log_metrics({"train_loss": tot / n, "train_acc": cor / n, "val_acc": acc, "val_macro_f1": f1}, step=ep)
        print(f"  ep {ep + 1}/{epochs} loss {tot / n:.4f} val acc {acc:.4f} F1 {f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            trial.report(f1, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    return model, hist, best


def classifier_report(model, images, entries, prefix):
    yt, pr = predict(model, images, entries)
    yp = pr.argmax(1)
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(yt, yp, labels=range(4), normalize="true")
    pd.DataFrame(rep).T.round(4).to_csv(C.out("tables", f"{prefix}_report.csv"))
    E.heatmap(cm, [f"true {c}" for c in CLASSES], [f"pred {c}" for c in CLASSES],
              C.out("figures", f"{prefix}_confusion.png"), title="Normalised confusion matrix", cmap="Blues")
    # accuracy by corruption severity (where are the hard cases?)
    sev = pd.DataFrame({"cls": [e["cls"] for e in entries], "severity": [e["severity"] for e in entries],
                        "correct": yt == yp}).groupby(["cls", "severity"]).correct.mean().round(4)
    sev.to_csv(C.out("tables", f"{prefix}_acc_by_severity.csv"))
    res = {"accuracy": rep["accuracy"], "macro_precision": rep["macro avg"]["precision"],
           "macro_recall": rep["macro avg"]["recall"], "macro_f1": rep["macro avg"]["f1-score"]}
    C.save_json(res, C.out("tables", f"{prefix}_summary.json"))
    print(prefix, res)
    return res, yp, pr


def tune_classifier(data, man):
    E.mlflow_setup(f"{NAME_C}_optuna")

    def objective(trial):
        hp = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
              "base": trial.suggest_categorical("base", [16, 32, 48]),
              "dropout": trial.suggest_float("dropout", 0.0, 0.5),
              "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(hp)
            _, _, best = train_classifier(hp, data, man["val"], C.CFG["t2c_trial_epochs"], trial)
            mlflow.log_metric("best_val_macro_f1", best)
        return best

    return E.run_study(E.make_study(NAME_C), objective, C.CFG["t2c_trials"])


# ============================================================================ specialists
def train_specialist(cls, hp, data, val_entries, epochs, trial=None, step0=0, log=True):
    """Specialist trained ONLY on its own corruption (fresh corruption every load)."""
    model = DenoisingAE(base=hp["base"], z_ch=hp["z_ch"], dropout=hp.get("dropout", 0.0)).to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    dl = E.loader(RandomCorruptionDataset(data["train"], fixed_cls=cls), hp["batch_size"], shuffle=True, drop_last=True)
    hist = {"train_loss": [], "val_psnr": [], "val_ssim": []}
    best, best_state = -1e9, None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for noisy, clean, _ in dl:
            noisy, clean = noisy.to(C.DEVICE), clean.to(C.DEVICE)
            loss = restoration_loss(model(noisy), clean, hp["alpha"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(noisy); n += len(noisy)
        sched.step()
        model.eval()
        df = E.evaluate_restorer(lambda x, y: {"out": model(x)}, data["val"], val_entries, classes=[cls])
        p, s = df.psnr_out.mean(), df.ssim_out.mean()
        score = restoration_score(p, s)
        for k, v in zip(hist, [tot / n, p, s]):
            hist[k].append(float(v))
        if log:
            mlflow.log_metrics({f"{cls}_train_loss": tot / n, f"{cls}_val_psnr": p, f"{cls}_val_ssim": s}, step=ep)
        print(f"  [{cls}] ep {ep + 1}/{epochs} loss {tot / n:.4f} val PSNR {p:.2f} SSIM {s:.4f}")
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            trial.report(score, step0 + ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    return model, hist, best


def tune_specialists(data, man):
    E.mlflow_setup(f"{NAME_S}_optuna")

    def objective(trial):
        hp = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
              "z_ch": trial.suggest_categorical("z_ch", [8, 16, 32]),
              "base": trial.suggest_categorical("base", [16, 32, 48]),
              "alpha": trial.suggest_float("alpha", 0.5, 0.95),
              "dropout": 0.0}
        ep = C.CFG["t2s_trial_epochs"]
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(hp)
            scores = []
            for i, cls in enumerate(EXPERTS):
                # only the first specialist reports to the pruner (steps must be comparable across trials)
                _, _, s = train_specialist(cls, hp, data, man["val"], ep, trial if i == 0 else None)
                scores.append(s)
            mlflow.log_metric("mean_val_score", float(np.mean(scores)))
        return float(np.mean(scores))

    return E.run_study(E.make_study(NAME_S), objective, C.CFG["t2s_trials"])


# ============================================================================ routing
def load_models():
    ck = E.load_ckpt("classifier")
    clf = CorruptionClassifier(base=ck["hparams"]["base"], dropout=ck["hparams"]["dropout"])
    clf.load_state_dict(ck["state_dict"])
    experts = {}
    for cls in EXPERTS:
        ck = E.load_ckpt(f"specialist_{cls}")
        m = DenoisingAE(base=ck["hparams"]["base"], z_ch=ck["hparams"]["z_ch"])
        m.load_state_dict(ck["state_dict"])
        experts[cls] = m
    return clf.to(C.DEVICE).eval(), {k: v.to(C.DEVICE).eval() for k, v in experts.items()}


def hard_route(clf, experts, x, route=None):
    """route: tensor of class ids (oracle) or None (use classifier argmax)."""
    probs = F.softmax(clf(x), 1)
    r = probs.argmax(1) if route is None else route
    out = x.clone()                       # identity bypass for 'clean'
    for k, cls in enumerate(CLASSES[1:], start=1):
        sel = r == k
        if sel.any():
            out[sel] = experts[cls](x[sel])
    return {"out": out, "route": r, "probs": probs}


def run():
    C.seed_everything()
    data = get_arrays()
    man = get_manifests(len(data["val"]), len(data["test"]))

    # ---- a) classifier
    s = tune_classifier(data, man)
    E.mlflow_setup(f"{NAME_C}_final")
    with mlflow.start_run(run_name="final"):
        hp = s["best_params"]
        mlflow.log_params({**hp, "epochs": C.CFG["t2c_epochs"]})
        clf, hist, _ = train_classifier(hp, data, man["val"], C.CFG["t2c_epochs"])
        E.log_artifact(E.save_ckpt(clf, "classifier", hp), "checkpoints")
        E.curves(hist, C.out("figures", "task2_classifier_curves.png"), title="Task 2 classifier")
        res, _, _ = classifier_report(clf, data["test"], man["test"], "task2_classifier_test")
        mlflow.log_metrics({f"test_{k}": v for k, v in res.items()})
        for f in ["task2_classifier_curves.png", "task2_classifier_test_confusion.png"]:
            E.log_artifact(C.out("figures", f), "figures")

    # ---- b) specialists
    s = tune_specialists(data, man)
    hp = {**s["best_params"], "dropout": 0.0}
    E.mlflow_setup(f"{NAME_S}_final")
    hists = {}
    with mlflow.start_run(run_name="final"):
        mlflow.log_params({**hp, "epochs": C.CFG["t2s_epochs"]})
        for cls in EXPERTS:
            torch.manual_seed(C.SEED + EXPERTS.index(cls))      # independent initialisations
            m, h, _ = train_specialist(cls, hp, data, man["val"], C.CFG["t2s_epochs"])
            hists[cls] = h
            E.log_artifact(E.save_ckpt(m, f"specialist_{cls}", hp), "checkpoints")
        merged = {k: {c: hists[c][k] for c in EXPERTS} for k in ["train_loss", "val_psnr", "val_ssim"]}
        E.curves(merged, C.out("figures", "task2_specialist_curves.png"), title="Task 2 specialists")
        E.log_artifact(C.out("figures", "task2_specialist_curves.png"), "figures")

        # ---- c) oracle vs predicted routing on the test manifest
        clf, experts = load_models()
        oracle = lambda x, y: hard_route(clf, experts, x, route=y)
        pred = lambda x, y: hard_route(clf, experts, x)
        d_or = E.evaluate_restorer(oracle, data["test"], man["test"])
        d_pr = E.evaluate_restorer(pred, data["test"], man["test"])
        d_or.to_csv(C.out("tables", "task2_oracle_per_image.csv"), index=False)
        d_pr.to_csv(C.out("tables", "task2_predicted_per_image.csv"), index=False)
        table = pd.concat({"oracle": E.group_table(d_or), "predicted": E.group_table(d_pr)}, axis=1)
        table.to_csv(C.out("tables", "task2_routing_results.csv"))
        print(table)
        mlflow.log_metrics({"test_psnr_oracle": d_or.psnr_out.mean(), "test_psnr_predicted": d_pr.psnr_out.mean(),
                            "test_ssim_oracle": d_or.ssim_out.mean(), "test_ssim_predicted": d_pr.ssim_out.mean()})

        # misrouting analysis: classifier errors and the restoration quality they cost
        true_id = d_pr.cls.map({c: i for i, c in enumerate(CLASSES)})
        mis = d_pr[d_pr.route != true_id].copy()
        mis["pred_cls"] = mis.route.map(dict(enumerate(CLASSES)))
        mis = mis.merge(d_or[["entry", "psnr_out", "ssim_out"]], on="entry", suffixes=("", "_oracle"))
        mis["psnr_loss_vs_oracle"] = mis.psnr_out_oracle - mis.psnr_out
        mis.sort_values("psnr_loss_vs_oracle", ascending=False).to_csv(
            C.out("tables", "task2_misrouted.csv"), index=False)
        pd.crosstab([mis.cls, mis.severity], mis.pred_cls).to_csv(C.out("tables", "task2_misrouting_counts.csv"))
        worst = mis.sort_values("psnr_loss_vs_oracle", ascending=False).entry.head(6).astype(int).tolist()
        if worst:
            items = E.restore_entries(pred, data["test"], man["test"], worst)
            labels = [f"true {man['test'][i]['cls']}/{man['test'][i]['severity']} -> "
                      f"{CLASSES[int(it[3]['route'])]}" for i, it in zip(worst, items)]
            E.restoration_grid(items, C.out("figures", "task2_misrouting_failures.png"), row_labels=labels)
            E.log_artifact(C.out("figures", "task2_misrouting_failures.png"), "figures")
        ex = E.pick_examples(d_pr)
        items = E.restore_entries(pred, data["test"], man["test"], ex)
        labels = [f"{man['test'][i]['cls']}/{man['test'][i]['severity']} -> {CLASSES[int(it[3]['route'])]}"
                  for i, it in zip(ex, items)]
        E.restoration_grid(items, C.out("figures", "task2_examples.png"), row_labels=labels)
        E.log_artifact(C.out("tables", "task2_routing_results.csv"))


if __name__ == "__main__":
    run()
