"""Corruption pipeline shared by training, manifests, evaluation AND the FastAPI backend.

Pure NumPy on purpose: the backend container can import this exact file without
PyTorch, so the corruptions seen in the app are bit-identical to the training ones.

Images are float32 arrays in HWC layout with values in [0, 1].
Every corruption is described by (class name, params dict). Params are JSON
serialisable so they can be stored in the validation/test manifests.
"""
import numpy as np

CLASSES = ["clean", "salt", "blur", "occlusion"]

# Training ranges (assignment table)
SALT_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_COVERAGE_RANGE = (0.10, 0.35)
OCC_RECTS_RANGE = (1, 3)

# Fixed test severities (assignment): low / medium / high
TEST_LEVELS = {
    "salt": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {"low": {"k": 3, "sigma": 0.7}, "medium": {"k": 5, "sigma": 1.5}, "high": {"k": 7, "sigma": 2.5}},
    "occlusion": {"low": {"coverage": 0.10, "n": 1}, "medium": {"coverage": 0.20, "n": 2},
                  "high": {"coverage": 0.35, "n": 3}},
}


# ----------------------------------------------------------------------------- params
def sample_rects(rng, n, target, H=128, W=128, tol=0.005, max_tries=3000):
    """Place n random black rectangles whose UNION covers target*H*W (+- tol).

    The target area is split between rectangles with a Dirichlet(3) draw (no
    near-empty rectangles), each
    rectangle gets a random aspect ratio in [0.5, 2] and a random position.
    Because rectangles may overlap we measure the real union and retry.
    """
    best, best_err = None, 1e9
    for _ in range(max_tries):
        shares = rng.dirichlet(np.full(n, 3.0)) * target * H * W
        mask = np.zeros((H, W), bool)
        rects = []
        for a in shares:
            ar = rng.uniform(0.5, 2.0)
            w = int(np.clip(round(np.sqrt(a * ar)), 4, W))
            h = int(np.clip(round(a / max(w, 1)), 4, H))
            x = int(rng.integers(0, W - w + 1))
            y = int(rng.integers(0, H - h + 1))
            rects.append([x, y, w, h])
            mask[y:y + h, x:x + w] = True
        err = abs(mask.mean() - target)
        if err < best_err:
            best, best_err = (rects, float(mask.mean())), err
        if err <= tol:
            break
    return best


def sample_params(cls, rng, H=128, W=128):
    """Random training-time parameters for one corruption class."""
    if cls == "clean":
        return {}
    if cls == "salt":
        return {"p": float(rng.uniform(*SALT_RANGE)), "seed": int(rng.integers(0, 2**31 - 1))}
    if cls == "blur":
        return {"k": int(rng.choice(BLUR_KERNELS)), "sigma": float(rng.uniform(*BLUR_SIGMA_RANGE))}
    if cls == "occlusion":
        n = int(rng.integers(OCC_RECTS_RANGE[0], OCC_RECTS_RANGE[1] + 1))
        lo, hi = OCC_COVERAGE_RANGE
        # target shrunk by the placement tolerance so the measured union stays inside [10%, 35%]
        rects, cov = sample_rects(rng, n, rng.uniform(lo + 0.005, hi - 0.005), H, W)
        return {"rects": rects, "coverage": cov}
    raise ValueError(cls)


def level_params(cls, level, rng, H=128, W=128):
    """Parameters for one of the fixed test severities (low/medium/high)."""
    if cls == "clean":
        return {}
    cfg = TEST_LEVELS[cls][level]
    if cls == "salt":
        return {"p": cfg["p"], "seed": int(rng.integers(0, 2**31 - 1))}
    if cls == "blur":
        return dict(cfg)
    rects, cov = sample_rects(rng, cfg["n"], cfg["coverage"], H, W)
    return {"rects": rects, "coverage": cov}


def custom_params(cls, rng, p=None, k=None, sigma=None, coverage=None, n=None, H=128, W=128):
    """User-chosen parameters from the app (missing values are sampled)."""
    base = sample_params(cls, rng, H, W)
    if cls == "salt" and p is not None:
        base["p"] = float(np.clip(p, 0.0, 0.5))
    if cls == "blur":
        if k is not None:
            base["k"] = int(k) if int(k) % 2 == 1 else int(k) + 1
        if sigma is not None:
            base["sigma"] = float(max(sigma, 0.1))
    if cls == "occlusion" and (coverage is not None or n is not None):
        n = int(np.clip(n if n is not None else len(base["rects"]), 1, 3))
        cov = float(np.clip(coverage if coverage is not None else base["coverage"], 0.01, 0.6))
        base["rects"], base["coverage"] = sample_rects(rng, n, cov, H, W)
    return base


def severity_of(cls, params):
    """Map continuous parameters to low/medium/high (tertiles of the training range)."""
    if cls == "clean":
        return "none"
    if cls == "salt":
        v, (lo, hi) = params["p"], SALT_RANGE
    elif cls == "blur":
        v, (lo, hi) = params["sigma"], BLUR_SIGMA_RANGE
    else:
        v, (lo, hi) = params["coverage"], OCC_COVERAGE_RANGE
    t = (v - lo) / (hi - lo)
    return "low" if t < 1 / 3 else ("medium" if t < 2 / 3 else "high")


# ----------------------------------------------------------------------------- ops
def gaussian_kernel1d(k, sigma):
    # Same construction as torchvision.transforms.functional.gaussian_blur
    x = np.linspace(-(k - 1) / 2.0, (k - 1) / 2.0, k)
    pdf = np.exp(-0.5 * (x / sigma) ** 2)
    return (pdf / pdf.sum()).astype(np.float32)


def gaussian_blur(img, k, sigma):
    """Separable Gaussian blur with reflect padding (matches torchvision)."""
    ker = gaussian_kernel1d(k, sigma)
    r = k // 2
    H, W = img.shape[:2]
    p = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="reflect")
    tmp = sum(ker[i] * p[i:i + H] for i in range(k))
    p = np.pad(tmp, ((0, 0), (r, r), (0, 0)), mode="reflect")
    return sum(ker[i] * p[:, i:i + W] for i in range(k)).astype(np.float32)


def salt_and_pepper(img, p, seed):
    """Each pixel is selected with probability p and set to black or white (50/50).
    The same value is written to all three channels (impulse noise on the pixel)."""
    g = np.random.default_rng(seed)
    H, W = img.shape[:2]
    sel = g.random((H, W)) < p
    val = (g.random((H, W)) < 0.5).astype(np.float32)
    out = img.copy()
    out[sel] = val[sel][:, None]
    return out


def occlude(img, rects):
    out = img.copy()
    for x, y, w, h in rects:
        out[y:y + h, x:x + w] = 0.0
    return out


def apply(img, cls, params):
    """Apply corruption `cls` with `params` to an HWC float image in [0, 1]."""
    if cls == "clean":
        return img.copy()
    if cls == "salt":
        return salt_and_pepper(img, params["p"], params["seed"])
    if cls == "blur":
        return gaussian_blur(img, params["k"], params["sigma"])
    if cls == "occlusion":
        return occlude(img, params["rects"])
    raise ValueError(cls)
