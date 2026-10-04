"""Oxford-IIIT Pet loading, the fixed 80/20 split, corruption manifests and datasets (Tasks 1-3).

Images are decoded once, converted to RGB, resized to 128x128 and cached as a
uint8 .npy array, so training never touches JPEG decoding again.
Corruptions are NOT cached: training samples a new corruption every time an
image is loaded (RandomCorruptionDataset); val/test use the stored manifests.
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src import config as C
from src.data import corruptions as K

CLS2ID = {c: i for i, c in enumerate(K.CLASSES)}


# ----------------------------------------------------------------------------- raw images
def _pets_root():
    """Return a folder containing images/ and annotations/ (download if needed)."""
    candidates = [os.environ.get("PETS_ROOT"), C.DATA / "oxford-iiit-pet", "/kaggle/input/oxford-iiit-pet"]
    for c in candidates:
        if c and (Path(c) / "annotations" / "trainval.txt").exists():
            return Path(c)
    from torchvision.datasets import OxfordIIITPet  # downloads ~800 MB from robots.ox.ac.uk
    OxfordIIITPet(root=str(C.DATA), split="trainval", download=True)
    return C.DATA / "oxford-iiit-pet"


def _synthetic(n, seed):
    """Smooth random colour images - only for testing the code without the dataset."""
    rng = np.random.default_rng(seed)
    low = rng.random((n, 8, 8, 3)).astype(np.float32)
    t = torch.from_numpy(low).permute(0, 3, 1, 2)
    up = torch.nn.functional.interpolate(t, size=(C.IMG, C.IMG), mode="bicubic", align_corners=False)
    return (up.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def load_split_images(split):
    """uint8 array [N, 128, 128, 3] for the official 'trainval' or 'test' list."""
    if C.SYNTHETIC:
        return _synthetic(200 if split == "trainval" else 60, 0 if split == "trainval" else 1)
    cache = C.DATA / f"pets_{split}_{C.IMG}.npy"
    if cache.exists():
        return np.load(cache)
    root = _pets_root()
    names = [l.split()[0] for l in open(root / "annotations" / f"{split}.txt") if l.strip()]
    arr = np.zeros((len(names), C.IMG, C.IMG, 3), np.uint8)
    for i, n in enumerate(names):
        im = Image.open(root / "images" / f"{n}.jpg").convert("RGB").resize((C.IMG, C.IMG), Image.BICUBIC)
        arr[i] = np.asarray(im)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def get_split():
    """80/20 split of the official trainval list with seed 42 (shared by Tasks 1-3)."""
    path = C.out("manifests", "split.json")
    if path.exists():
        return C.load_json(path)
    n = len(load_split_images("trainval"))
    idx = np.random.default_rng(C.SEED).permutation(n)
    cut = int(0.8 * n)
    split = {"seed": C.SEED, "train": sorted(idx[:cut].tolist()), "val": sorted(idx[cut:].tolist())}
    C.save_json(split, path)
    return split


def get_arrays():
    """Returns dict of uint8 arrays: train, val, test (subsampled in the smoke profile)."""
    tv, test = load_split_images("trainval"), load_split_images("test")
    s = get_split()
    tr, va = tv[s["train"]], tv[s["val"]]
    lim = lambda a, k: a[: C.CFG[k]] if C.CFG[k] else a
    return {"train": lim(tr, "max_train"), "val": lim(va, "max_val"), "test": lim(test, "max_test")}


# ----------------------------------------------------------------------------- manifests
def build_val_manifest(n_val):
    """One deterministic condition per validation image, classes exactly balanced."""
    rng = np.random.default_rng(C.SEED + 1)
    classes = np.array([K.CLASSES[i % 4] for i in range(n_val)])
    rng.shuffle(classes)
    entries = []
    for i, cls in enumerate(classes):
        seed = 1_000_000 + i
        params = K.sample_params(cls, np.random.default_rng(seed))
        entries.append({"index": i, "cls": str(cls), "severity": K.severity_of(cls, params),
                        "params": params, "seed": seed})
    return entries


def build_test_manifest(n_test):
    """Per test image: 1 clean + 3 corruptions x 3 fixed severities = 10 entries."""
    entries = []
    for i in range(n_test):
        entries.append({"index": i, "cls": "clean", "severity": "none", "params": {}, "seed": None})
        k = 0
        for cls in ["salt", "blur", "occlusion"]:
            for level in ["low", "medium", "high"]:
                seed = 2_000_000 + i * 10 + k
                k += 1
                params = K.level_params(cls, level, np.random.default_rng(seed))
                entries.append({"index": i, "cls": cls, "severity": level, "params": params, "seed": seed})
    return entries


def get_manifests(n_val, n_test):
    paths = {"val": C.out("manifests", "val_manifest.json"), "test": C.out("manifests", "test_manifest.json")}
    out = {}
    for name, n, fn in [("val", n_val, build_val_manifest), ("test", n_test, build_test_manifest)]:
        p = paths[name]
        if p.exists():
            m = C.load_json(p)
            if m["n_images"] == n:
                out[name] = m["entries"]
                continue
        entries = fn(n)
        C.save_json({"n_images": n, "seed_rule": "see src/data/pets.py", "entries": entries}, p)
        out[name] = entries
    return out


# ----------------------------------------------------------------------------- datasets
def to_tensor(hwc):
    return torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1).float()


class RandomCorruptionDataset(Dataset):
    """Training data: a NEW corruption (type + severity) is sampled on every load.

    - fixed_cls=None   -> class drawn uniformly from the 4 conditions (Task 1)
    - fixed_cls='blur' -> only that corruption (Task 2 specialists)
    - index given as (i, class_id) tuple -> class forced by BalancedBatchSampler
    """

    def __init__(self, images, fixed_cls=None, hflip=True):
        self.images, self.fixed_cls, self.hflip = images, fixed_cls, hflip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        # torch seeds each DataLoader worker differently -> different corruptions per worker
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        if isinstance(key, (tuple, list)):
            i, cls = key[0], K.CLASSES[key[1]]
        else:
            i = key
            cls = self.fixed_cls or K.CLASSES[int(rng.integers(0, 4))]
        clean = self.images[i].astype(np.float32) / 255.0
        if self.hflip and rng.random() < 0.5:
            clean = clean[:, ::-1]
        params = K.sample_params(cls, rng)
        noisy = K.apply(np.ascontiguousarray(clean), cls, params)
        return to_tensor(noisy), to_tensor(clean), CLS2ID[cls]


class ManifestDataset(Dataset):
    """Deterministic val/test data reproduced from a manifest (no corrupted files stored)."""

    def __init__(self, images, entries, classes=None):
        self.images = images
        self.entries = [e for e in entries if classes is None or e["cls"] in classes]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, j):
        e = self.entries[j]
        clean = self.images[e["index"]].astype(np.float32) / 255.0
        noisy = K.apply(clean, e["cls"], e["params"])
        return to_tensor(noisy), to_tensor(clean), CLS2ID[e["cls"]], j


class BalancedBatchSampler(Sampler):
    """Every batch contains exactly batch_size/4 samples of each class.

    Yields lists of (image_index, class_id); the dataset applies the forced class.
    """

    def __init__(self, n_images, batch_size, n_classes=4):
        assert batch_size % n_classes == 0, "batch size must be divisible by 4"
        self.n, self.bs, self.k = n_images, batch_size, n_classes

    def __len__(self):
        return max(1, self.n // self.bs)

    def __iter__(self):
        perm = torch.randperm(self.n).tolist()
        for b in range(len(self)):
            idx = perm[b * self.bs:(b + 1) * self.bs]
            if len(idx) < self.bs:  # tiny datasets (smoke profile)
                idx = (idx * self.bs)[: self.bs]
            labels = torch.arange(self.bs) % self.k
            labels = labels[torch.randperm(self.bs)].tolist()
            yield list(zip(idx, labels))
