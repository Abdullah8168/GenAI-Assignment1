"""Global configuration: paths, seeds, device and run profiles.

Select a profile with the environment variable GENAI_PROFILE:
  smoke - a few minutes on CPU, tiny subsets; only checks that every code path runs
  quick - default; ~2-3 h on a single Kaggle T4/P100 GPU
  full  - longer schedules / more Optuna trials if you have more GPU time
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GENAI_OUT", ROOT / "outputs"))
DATA = Path(os.environ.get("GENAI_DATA", ROOT / "data"))
FS2K_ROOT = os.environ.get("FS2K_ROOT")          # folder containing FS2K (photo/, sketch/, anno_*.json)
SYNTHETIC = os.environ.get("GENAI_SYNTHETIC") == "1"  # random images instead of datasets (code test only)

SEED = 42
IMG = 128
CLASSES = ["clean", "salt", "blur", "occlusion"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(os.environ.get("GENAI_WORKERS", min(4, os.cpu_count() or 1)))

PROFILES = {
    "smoke": dict(
        max_train=64, max_val=32, max_test=8,
        t1_trials=2, t1_trial_epochs=1, t1_epochs=1, t1_ablation_epochs=1,
        t2c_trials=2, t2c_trial_epochs=1, t2c_epochs=1,
        t2s_trials=2, t2s_trial_epochs=1, t2s_epochs=1,
        t3_trials=2, t3_trial_warmup=1, t3_trial_epochs=1, t3_warmup=1, t3_epochs=1,
        t4_trials=2, t4_trial_epochs=1, t4_epochs=2, t4_sample_every=1,
    ),
    "quick": dict(
        max_train=None, max_val=None, max_test=None,
        t1_trials=12, t1_trial_epochs=5, t1_epochs=40, t1_ablation_epochs=15,
        t2c_trials=10, t2c_trial_epochs=4, t2c_epochs=20,
        t2s_trials=8, t2s_trial_epochs=4, t2s_epochs=30,
        t3_trials=8, t3_trial_warmup=1, t3_trial_epochs=2, t3_warmup=3, t3_epochs=15,
        t4_trials=8, t4_trial_epochs=12, t4_epochs=120, t4_sample_every=10,
    ),
    "full": dict(
        max_train=None, max_val=None, max_test=None,
        t1_trials=25, t1_trial_epochs=8, t1_epochs=80, t1_ablation_epochs=30,
        t2c_trials=20, t2c_trial_epochs=6, t2c_epochs=30,
        t2s_trials=15, t2s_trial_epochs=6, t2s_epochs=60,
        t3_trials=15, t3_trial_warmup=1, t3_trial_epochs=3, t3_warmup=5, t3_epochs=30,
        t4_trials=15, t4_trial_epochs=20, t4_epochs=200, t4_sample_every=10,
    ),
}
PROFILE = os.environ.get("GENAI_PROFILE", "quick")
CFG = PROFILES[PROFILE]

SUBDIRS = ["checkpoints", "onnx", "figures", "tables", "optuna", "manifests", "samples", "cache"]


def out(sub, name=None):
    p = OUT / sub
    p.mkdir(parents=True, exist_ok=True)
    return p / name if name else p


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
