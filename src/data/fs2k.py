"""FS2K photo->sketch pairs (Task 4).

Expected layout (official release, https://github.com/DengPingFan/FS2K):
  FS2K_ROOT/
    anno_train.json, anno_test.json      # official train/test definition, one entry per pair
    photo/photo1/image0001.jpg ...       # photos (extensions vary .jpg/.png)
    sketch/sketch1/sketch0001.jpg ...    # paired sketches
Each annotation entry has "image_name" like "photo1/image0001" and "style" in {0,1,2}.
The loader searches recursively and matches files by name stem, so small layout
differences (extra parent folder, .png vs .jpg) still work.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src import config as C

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _find_root():
    cands = [C.FS2K_ROOT, C.DATA / "FS2K", "/kaggle/input/fs2k", "/kaggle/input"]
    for c in cands:
        if not c or not Path(c).exists():
            continue
        hits = list(Path(c).rglob("anno_train.json"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        "FS2K not found. Download it (see README) and set FS2K_ROOT to the folder that "
        "contains anno_train.json, anno_test.json, photo/ and sketch/.")


def _index_files(root):
    """Map 'photo1/image0001' (lower-case, no extension) -> file path."""
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx[f"{p.parent.name}/{p.stem}".lower()] = p
    return idx


def _load(path, mode="RGB"):
    return np.asarray(Image.open(path).convert(mode).resize((C.IMG, C.IMG), Image.BICUBIC))


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    low = torch.from_numpy(rng.random((n, 3, 8, 8)).astype(np.float32))
    ph = F.interpolate(low, size=(C.IMG, C.IMG), mode="bicubic", align_corners=False).clamp(0, 1)
    gray = ph.mean(1, keepdim=True)
    edge = (gray - F.avg_pool2d(gray, 5, 1, 2)).abs() * 8
    sk = (1 - edge.clamp(0, 1)).repeat(1, 3, 1, 1)
    to8 = lambda t: (t.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return to8(ph), to8(sk), rng.integers(0, 3, n)


def load_official(split):
    """Returns photos [N,128,128,3] uint8, sketches [N,128,128,3] uint8, styles [N] int64, names."""
    if C.SYNTHETIC:
        n = 90 if split == "train" else 30
        ph, sk, st = _synthetic(n, 0 if split == "train" else 1)
        return ph, sk, st, [f"syn{i}" for i in range(n)]
    cache = C.DATA / f"fs2k_{split}_{C.IMG}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["photos"], z["sketches"], z["styles"], list(z["names"])
    root = _find_root()
    files = _index_files(root)
    anno = json.load(open(root / f"anno_{split}.json"))
    photos, sketches, styles, names, missing = [], [], [], [], 0
    for e in anno:
        name = e["image_name"].replace("\\", "/").lower()
        name = "/".join(name.split("/")[-2:])
        name = os.path.splitext(name)[0]
        sk_name = name.replace("photo", "sketch").replace("image", "sketch")
        if name not in files or sk_name not in files:
            missing += 1
            continue
        photos.append(_load(files[name]))
        sketches.append(_load(files[sk_name]))
        styles.append(int(e["style"]))
        names.append(name)
    if missing:
        print(f"[fs2k] warning: {missing} {split} pairs not found on disk")
    if not photos:
        raise RuntimeError(f"No FS2K pairs found under {root}; check the layout described in src/data/fs2k.py")
    photos, sketches, styles = np.stack(photos), np.stack(sketches), np.array(styles, np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, photos=photos, sketches=sketches, styles=styles, names=np.array(names))
    return photos, sketches, styles, names


def get_fs2k():
    """Official test set + official train split into 85% train / 15% val, stratified by style, seed 42."""
    from sklearn.model_selection import train_test_split
    ph, sk, st, names = load_official("train")
    tph, tsk, tst, tnames = load_official("test")
    path = C.out("manifests", "fs2k_split.json")
    if path.exists() and C.load_json(path)["n_train_official"] == len(st):
        s = C.load_json(path)
        tr, va = np.array(s["train"]), np.array(s["val"])
    else:
        tr, va = train_test_split(np.arange(len(st)), test_size=0.15, random_state=C.SEED, stratify=st)
        C.save_json({"seed": C.SEED, "n_train_official": len(st), "train": sorted(tr.tolist()),
                     "val": sorted(va.tolist()),
                     "val_style_counts": np.bincount(st[va], minlength=3).tolist()}, path)
    lim = lambda a, k: a[: C.CFG[k]] if C.CFG[k] else a
    tr, va = lim(np.sort(tr), "max_train"), lim(np.sort(va), "max_val")
    nt = C.CFG["max_test"] or len(tst)
    return {
        "train": (ph[tr], sk[tr], st[tr]),
        "val": (ph[va], sk[va], st[va]),
        "test": (tph[:nt], tsk[:nt], tst[:nt]),
        "test_names": tnames[:nt],
    }


def to_pm1(a):
    """uint8 HWC (batch or single) -> float CHW in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(a)).float() / 127.5 - 1
    return t.permute(0, 3, 1, 2) if t.dim() == 4 else t.permute(2, 0, 1)


class PairDataset(Dataset):
    """Photo/sketch pairs. Augmentation (resize-jitter crop + horizontal flip) is applied to
    the photo and sketch CONCATENATED along channels, so both receive the identical spatial
    transform and pixel correspondence is preserved."""

    def __init__(self, photos, sketches, styles, augment=False, jitter=142):
        self.p, self.s, self.st = photos, sketches, styles
        self.augment, self.jitter = augment, jitter

    def __len__(self):
        return len(self.st)

    def __getitem__(self, i):
        x, y = to_pm1(self.p[i]), to_pm1(self.s[i])
        if self.augment:
            pair = torch.cat([x, y], 0)[None]                       # [1, 6, H, W]
            pair = F.interpolate(pair, size=(self.jitter, self.jitter), mode="bilinear", align_corners=False)[0]
            top, left = torch.randint(0, self.jitter - C.IMG + 1, (2,)).tolist()
            pair = pair[:, top:top + C.IMG, left:left + C.IMG]
            if torch.rand(1).item() < 0.5:
                pair = pair.flip(-1)
            x, y = pair[:3], pair[3:]
        return x, y, int(self.st[i])
