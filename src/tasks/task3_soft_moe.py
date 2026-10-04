"""Task 3 - Jointly trained soft mixture-of-experts.

Initialisation: gate <- Task 2 classifier, experts <- Task 2 specialists.
Stage 1 (warm-up): experts frozen, only the gate is trained.
Stage 2 (joint):   everything unfrozen; experts use a 10x smaller learning rate than the gate.
Loss: l1*L1 + ls*(1-SSIM) + lc*CE(gate logits, corruption label) + lb*balance(w).
Optuna tunes joint lr, temperature tau, lc, lb and the L1/SSIM weighting; trials whose
routing collapses (a branch's mean weight < 2%) are pruned.
"""
import copy

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F

from src import config as C
from src import engine as E
from src.data import corruptions as K
from src.data.corruptions import CLASSES
from src.data.pets import BalancedBatchSampler, RandomCorruptionDataset, get_arrays, get_manifests, to_tensor
from src.metrics import l1 as l1_metric, psnr, restoration_score, ssim
from src.models.restoration import SoftMoE, balance_loss
from src.tasks.task2_hard_routing import hard_route, load_models

NAME = "task3_soft_moe"
BRANCHES = ["identity (clean)", "salt expert", "blur expert", "occlusion expert"]
COLLAPSE = 0.02


def build(tau):
    clf, experts = load_models()
    return SoftMoE(copy.deepcopy(clf), *[copy.deepcopy(experts[c]) for c in ["salt", "blur", "occlusion"]],
                   tau=tau).to(C.DEVICE)


def moe_fn(model):
    def f(x, y):
        out, w, _, _ = model(x)
        return {"out": out, "w": w}
    return f


def train(hp, data, val_entries, warmup, epochs, trial=None, log=True):
    model = build(hp["tau"])
    dl = E.loader(RandomCorruptionDataset(data["train"]),
                  batch_sampler=BalancedBatchSampler(len(data["train"]), hp["batch_size"]))
    hist = {"loss": [], "l1": [], "ce": [], "balance": [], "val_psnr": [], "val_ssim": [],
            "val_mean_w": {b: [] for b in BRANCHES}}
    best, best_state, step = -1e9, None, 0
    for stage, n_ep in [("warmup", warmup), ("joint", epochs)]:
        for p in model.experts.parameters():
            p.requires_grad_(stage == "joint")
        if stage == "warmup":
            opt = torch.optim.Adam(model.gate.parameters(), lr=hp["gate_lr"])
        else:
            opt = torch.optim.Adam([{"params": model.gate.parameters(), "lr": hp["joint_lr"]},
                                    {"params": model.experts.parameters(), "lr": hp["joint_lr"] * 0.1}])
        for ep in range(n_ep):
            model.train()
            if stage == "warmup":
                model.experts.eval()      # frozen experts: keep BN statistics fixed
            agg = np.zeros(4); n = 0
            for x, clean, y in dl:
                x, clean, y = x.to(C.DEVICE), clean.to(C.DEVICE), y.to(C.DEVICE)
                out, w, logits, _ = model(x)
                l1 = F.l1_loss(out, clean)
                ls = 1 - ssim(out, clean)
                ce = F.cross_entropy(logits, y)
                lb = balance_loss(w)
                loss = hp["l1"] * l1 + (1 - hp["l1"]) * ls + hp["lc"] * ce + hp["lb"] * lb
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                agg += np.array([loss.item(), l1.item(), ce.item(), lb.item()]) * len(y); n += len(y)
            model.eval()
            df = E.evaluate_restorer(moe_fn(model), data["val"], val_entries)
            p, s = df.psnr_out.mean(), df.ssim_out.mean()
            mean_w = [df[f"w_{k}"].mean() for k in range(4)]
            score = restoration_score(p, s)
            for k, v in zip(["loss", "l1", "ce", "balance"], agg / n):
                hist[k].append(float(v))
            hist["val_psnr"].append(float(p)); hist["val_ssim"].append(float(s))
            for b, v in zip(BRANCHES, mean_w):
                hist["val_mean_w"][b].append(float(v))
            if log:
                mlflow.log_metrics({"loss": agg[0] / n, "l1": agg[1] / n, "ce": agg[2] / n, "balance": agg[3] / n,
                                    "val_psnr": p, "val_ssim": s,
                                    **{f"val_w_{c}": v for c, v in zip(CLASSES, mean_w)}}, step=step)
            print(f"  [{stage}] ep {ep + 1}/{n_ep} loss {agg[0] / n:.4f} val PSNR {p:.2f} SSIM {s:.4f} "
                  f"w={np.round(mean_w, 3)}")
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if trial is not None:
                if min(mean_w) < COLLAPSE:
                    trial.set_user_attr("pruned_reason", f"routing collapse w={np.round(mean_w, 3).tolist()}")
                    raise optuna.TrialPruned()
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            step += 1
    model.load_state_dict(best_state)
    return model, hist, best


def tune(data, man):
    E.mlflow_setup(f"{NAME}_optuna")

    def objective(trial):
        hp = {"joint_lr": trial.suggest_float("joint_lr", 1e-5, 5e-4, log=True),
              "tau": trial.suggest_float("tau", 0.3, 3.0, log=True),
              "lc": trial.suggest_float("lc", 0.01, 1.0, log=True),
              "lb": trial.suggest_float("lb", 1e-3, 0.5, log=True),
              "l1": trial.suggest_float("l1", 0.5, 0.95),      # l_s = 1 - l1
              "gate_lr": 1e-4, "batch_size": 64}
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(hp)
            _, _, best = train(hp, data, man["val"], C.CFG["t3_trial_warmup"], C.CFG["t3_trial_epochs"], trial)
            mlflow.log_metric("best_val_score", best)
        return best

    return E.run_study(E.make_study(NAME), objective, C.CFG["t3_trials"])


def mixed_corruption_test(models, images, n=300):
    """Extra stress test: TWO corruptions at once (outside the training distribution).
    Shows where soft routing can blend experts and hard routing must pick one."""
    rng = np.random.default_rng(C.SEED)
    combos = [("salt", "blur"), ("blur", "occlusion"), ("salt", "occlusion")]
    rows = []
    n = min(n, len(images))
    for i in range(n):
        a, b = combos[i % 3]
        clean = images[i].astype(np.float32) / 255
        x = K.apply(clean, a, K.level_params(a, "medium", rng))
        x = K.apply(x, b, K.level_params(b, "medium", rng))
        xt, ct = to_tensor(x)[None].to(C.DEVICE), to_tensor(clean)[None].to(C.DEVICE)
        r = {"combo": f"{a}+{b}", "psnr_in": psnr(xt, ct).item()}
        with torch.no_grad():
            for name, fn in models.items():
                res = fn(xt, None)
                r[f"psnr_{name}"] = psnr(res["out"].clamp(0, 1), ct).item()
                r[f"ssim_{name}"] = ssim(res["out"].clamp(0, 1), ct).item()
                if "w" in res:
                    for k in range(4):
                        r[f"w_{CLASSES[k]}"] = res["w"][0, k].item()
        rows.append(r)
    df = pd.DataFrame(rows)
    t = df.groupby("combo").mean(numeric_only=True).round(4)
    t.to_csv(C.out("tables", "task3_mixed_corruptions.csv"))
    print(t)
    return t


def run():
    C.seed_everything()
    data = get_arrays()
    man = get_manifests(len(data["val"]), len(data["test"]))
    s = tune(data, man)
    hp = {**s["best_params"], "gate_lr": 1e-4, "batch_size": 64}

    E.mlflow_setup(f"{NAME}_final")
    with mlflow.start_run(run_name="final"):
        mlflow.log_params({**hp, "warmup": C.CFG["t3_warmup"], "epochs": C.CFG["t3_epochs"]})
        model, hist, _ = train(hp, data, man["val"], C.CFG["t3_warmup"], C.CFG["t3_epochs"])
        E.log_artifact(E.save_ckpt(model, "soft_moe", hp), "checkpoints")
        E.curves(hist, C.out("figures", "task3_curves.png"), title="Task 3 soft MoE (warm-up then joint)")
        model.eval()

        # ---- reconstruction results
        fn = moe_fn(model)
        df = E.evaluate_restorer(fn, data["test"], man["test"])
        df.to_csv(C.out("tables", "task3_test_per_image.csv"), index=False)
        E.group_table(df).to_csv(C.out("tables", "task3_test_results.csv"))
        mlflow.log_metrics({"test_psnr": df.psnr_out.mean(), "test_ssim": df.ssim_out.mean()})

        # ---- routing analysis: mean weights per true corruption & severity (heatmap)
        wcols = [f"w_{k}" for k in range(4)]
        rw = df.groupby(["cls", "severity"])[wcols].mean()
        rw.columns = BRANCHES
        rw.round(4).to_csv(C.out("tables", "task3_routing_weights.csv"))
        E.heatmap(rw.values, [f"{a}/{b}" for a, b in rw.index], BRANCHES,
                  C.out("figures", "task3_routing_heatmap.png"), title="Mean routing weight by true corruption")
        # entropy of the weights: low = one expert dominates, high = distributed
        W = df[wcols].values
        df["entropy"] = -(W * np.log(W + 1e-9)).sum(1)
        df["dominant"] = W.argmax(1)
        health = {
            "mean_weight_per_branch": dict(zip(CLASSES, W.mean(0).round(4).tolist())),
            "fraction_dominant_per_branch": dict(zip(CLASSES, (np.bincount(df.dominant, minlength=4) / len(df)).round(4).tolist())),
            "inactive_branches(<2% mean weight)": [CLASSES[k] for k in range(4) if W[:, k].mean() < COLLAPSE],
            # weight a branch receives on inputs it is NOT responsible for
            "mean_weight_on_unrelated_inputs": {CLASSES[k]: round(float(df[df.cls != CLASSES[k]][f"w_{k}"].mean()), 4)
                                                for k in range(4)},
            "routing_accuracy(argmax==true)": round(float((df.dominant == df.cls.map(
                {c: i for i, c in enumerate(CLASSES)})).mean()), 4),
            "mean_entropy": round(float(df.entropy.mean()), 4), "max_entropy": round(float(np.log(4)), 4),
        }
        C.save_json(health, C.out("tables", "task3_routing_health.json"))
        print(health)
        # weight distribution plot
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
        for k, ax in enumerate(axes):
            for cls in CLASSES:
                ax.hist(df[df.cls == cls][f"w_{k}"], bins=20, range=(0, 1), alpha=0.5, label=cls)
            ax.set_title(BRANCHES[k], fontsize=9); ax.set_xlabel("weight")
        axes[0].legend(fontsize=7, title="true input"); axes[0].set_ylabel("count")
        plt.tight_layout(); fig.savefig(C.out("figures", "task3_weight_distribution.png"), dpi=130); plt.close(fig)

        # dominant vs distributed examples
        dom = df.sort_values("entropy").entry.head(4).astype(int).tolist()
        dis = df.sort_values("entropy", ascending=False).entry.head(4).astype(int).tolist()
        for tag, ids in [("dominant", dom), ("distributed", dis)]:
            items = E.restore_entries(fn, data["test"], man["test"], ids)
            labels = [f"{man['test'][i]['cls']}/{man['test'][i]['severity']} w=" +
                      ",".join(f"{v:.2f}" for v in it[3]["w"].tolist()) for i, it in zip(ids, items)]
            E.restoration_grid(items, C.out("figures", f"task3_{tag}_examples.png"), row_labels=labels)
        fail = E.pick_failures(df)
        items = E.restore_entries(fn, data["test"], man["test"], fail)
        labels = [f"{man['test'][i]['cls']}/{man['test'][i]['severity']} w=" +
                  ",".join(f"{v:.2f}" for v in it[3]["w"].tolist()) for i, it in zip(fail, items)]
        E.restoration_grid(items, C.out("figures", "task3_failures.png"), row_labels=labels)

        # ---- comparison of all three systems + mixed-corruption stress test
        clf, experts = load_models()
        from src.models.restoration import DenoisingAE
        ck = E.load_ckpt("universal_dae")
        uni = DenoisingAE(base=ck["hparams"]["base"], z_ch=ck["hparams"]["z_ch"]).to(C.DEVICE).eval()
        uni.load_state_dict(ck["state_dict"])
        systems = {"universal": lambda x, y: {"out": uni(x)},
                   "hard": lambda x, y: hard_route(clf, experts, x),
                   "soft": fn}
        mixed_corruption_test(systems, data["test"])
        comp = {}
        for name, path in [("universal", "task1_test_per_image.csv"), ("hard_predicted", "task2_predicted_per_image.csv"),
                           ("hard_oracle", "task2_oracle_per_image.csv"), ("soft_moe", "task3_test_per_image.csv")]:
            p = C.out("tables", path)
            if p.exists():
                comp[name] = pd.read_csv(p).groupby("cls")[["psnr_out", "ssim_out"]].mean()
        if comp:
            pd.concat(comp, axis=1).round(4).to_csv(C.out("tables", "comparison_all_systems.csv"))
        for f in ["task3_routing_heatmap.png", "task3_weight_distribution.png", "task3_curves.png",
                  "task3_dominant_examples.png", "task3_distributed_examples.png", "task3_failures.png"]:
            E.log_artifact(C.out("figures", f), "figures")
        for f in ["task3_test_results.csv", "task3_routing_weights.csv", "task3_routing_health.json",
                  "task3_mixed_corruptions.csv"]:
            E.log_artifact(C.out("tables", f))


if __name__ == "__main__":
    run()
