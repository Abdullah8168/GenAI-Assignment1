"""Task 1 - Universal multi-corruption denoising autoencoder.

Stages: Optuna search -> final training with best params -> skip-connection ablation
-> test evaluation (per corruption & severity) -> figures (12 examples, 4 failures).
"""
import mlflow
import optuna
import torch

from src import config as C
from src import engine as E
from src.data.pets import ManifestDataset, RandomCorruptionDataset, get_arrays, get_manifests
from src.metrics import restoration_loss, restoration_score
from src.models.restoration import DenoisingAE, count_params

NAME = "task1_universal_dae"


def build(hp, skip=False):
    return DenoisingAE(base=hp["base"], z_ch=hp["z_ch"], dropout=hp["dropout"], skip=skip)


def train(hp, data, val_entries, epochs, trial=None, skip=False, log=True):
    """Train one DAE. Reports the validation score to Optuna each epoch (for pruning)."""
    model = build(hp, skip).to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    train_dl = E.loader(RandomCorruptionDataset(data["train"]), hp["batch_size"], shuffle=True, drop_last=True)
    hist = {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_score": []}
    best, best_state = -1e9, None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for noisy, clean, _ in train_dl:
            noisy, clean = noisy.to(C.DEVICE, non_blocking=True), clean.to(C.DEVICE, non_blocking=True)
            loss = restoration_loss(model(noisy), clean, hp["alpha"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(noisy); n += len(noisy)
        sched.step()
        model.eval()
        df = E.evaluate_restorer(lambda x, y: {"out": model(x)}, data["val"], val_entries)
        p, s = df.psnr_out.mean(), df.ssim_out.mean()
        score = restoration_score(p, s)
        for k, v in zip(hist, [tot / max(n, 1), p, s, score]):
            hist[k].append(float(v))
        if log:
            mlflow.log_metrics({"train_loss": tot / max(n, 1), "val_psnr": p, "val_ssim": s, "val_score": score}, step=ep)
        print(f"  ep {ep + 1}/{epochs} loss {tot / max(n, 1):.4f} val PSNR {p:.2f} SSIM {s:.4f}")
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            trial.report(score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    return model, hist, best


def tune(data, man):
    E.mlflow_setup(f"{NAME}_optuna")

    def objective(trial):
        hp = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "z_ch": trial.suggest_categorical("z_ch", [8, 16, 32]),          # latent = z_ch * 8 * 8
            "base": trial.suggest_categorical("base", [16, 32, 48]),         # encoder channels base,2b,4b,8b
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),                # L1 vs (1-SSIM) weight
        }
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(hp)
            _, _, best = train(hp, data, man["val"], C.CFG["t1_trial_epochs"], trial=trial)
            mlflow.log_metric("best_val_score", best)
        return best

    return E.run_study(E.make_study(NAME), objective, C.CFG["t1_trials"])


def run():
    C.seed_everything()
    data = get_arrays()
    man = get_manifests(len(data["val"]), len(data["test"]))
    summary = tune(data, man)
    hp = summary["best_params"]

    # ---- final model
    E.mlflow_setup(f"{NAME}_final")
    with mlflow.start_run(run_name="final"):
        mlflow.log_params({**hp, "epochs": C.CFG["t1_epochs"], "skip": False, "profile": C.PROFILE})
        model, hist, _ = train(hp, data, man["val"], C.CFG["t1_epochs"])
        mlflow.log_metric("n_params", count_params(model))
        mlflow.log_metric("latent_dim", model.latent_dim)
        ck = E.save_ckpt(model, "universal_dae", hp)
        E.log_artifact(ck, "checkpoints")
        E.curves(hist, C.out("figures", "task1_curves.png"), title="Task 1 universal DAE")
        E.log_artifact(C.out("figures", "task1_curves.png"))

        # ---- test evaluation
        model.eval()
        fn = lambda x, y: {"out": model(x)}
        df = E.evaluate_restorer(fn, data["test"], man["test"])
        df.to_csv(C.out("tables", "task1_test_per_image.csv"), index=False)
        table = E.group_table(df)
        table.to_csv(C.out("tables", "task1_test_results.csv"))
        print(table)
        mlflow.log_metrics({"test_psnr": df.psnr_out.mean(), "test_ssim": df.ssim_out.mean()})
        E.log_artifact(C.out("tables", "task1_test_results.csv"))

        ex = E.pick_examples(df)
        items = E.restore_entries(fn, data["test"], man["test"], ex)
        labels = [f"{man['test'][i]['cls']}/{man['test'][i]['severity']}" for i in ex]
        E.restoration_grid(items, C.out("figures", "task1_examples.png"), row_labels=labels)
        fail = E.pick_failures(df)
        items = E.restore_entries(fn, data["test"], man["test"], fail)
        labels = [f"{man['test'][i]['cls']}/{man['test'][i]['severity']}" for i in fail]
        E.restoration_grid(items, C.out("figures", "task1_failures.png"), row_labels=labels)
        C.save_json({"examples": ex, "failures": fail}, C.out("tables", "task1_figure_entries.json"))
        for f in ["task1_examples.png", "task1_failures.png"]:
            E.log_artifact(C.out("figures", f), "figures")

    # ---- ablation: same config with one limited skip connection (64x64)
    with mlflow.start_run(run_name="ablation_skip64"):
        ep = C.CFG["t1_ablation_epochs"]
        mlflow.log_params({**hp, "epochs": ep, "skip": True})
        rows = {}
        for skip in [False, True]:
            m, _, _ = train(hp, data, man["val"], ep, skip=skip, log=False)
            m.eval()
            d = E.evaluate_restorer(lambda x, y: {"out": m(x)}, data["test"], man["test"])
            rows["skip64" if skip else "no_skip"] = d.groupby("cls")[["psnr_in", "psnr_out", "ssim_out"]].mean()
        import pandas as pd
        abl = pd.concat(rows, names=["variant"]).round(4)
        abl.to_csv(C.out("tables", "task1_skip_ablation.csv"))
        E.log_artifact(C.out("tables", "task1_skip_ablation.csv"))
        print(abl)


if __name__ == "__main__":
    run()
