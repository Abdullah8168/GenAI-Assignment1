"""RestoreLab FastAPI backend - serves all four tasks from ONNX models.

Endpoints
  GET  /api/health              model status + system info
  GET  /api/samples             sample gallery (unseen test images)
  POST /api/corrupt             apply a runtime corruption (same code as training)
  POST /api/restore/universal   Task 1
  POST /api/restore/hard        Task 2 (classifier -> specialist / identity)
  POST /api/restore/soft        Task 3 (soft mixture of experts)
  POST /api/sketch              Task 4 (style-conditioned generator)
"""
import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app import corruptions as K

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/onnx"))
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", "/models/samples"))
IMG = 128
MAX_BYTES = 10 * 2**20
CLASSES = K.CLASSES
MODEL_NAMES = ["universal_dae", "classifier", "specialist_salt", "specialist_blur", "specialist_occlusion",
               "soft_moe", "sketch_generator"]

app = FastAPI(title="RestoreLab API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
SESSIONS: dict = {}
INFO: dict = {}
STARTED = time.time()


@app.on_event("startup")
def load_models():
    """Load every ONNX model once; missing files leave the API running in 'degraded' mode."""
    import json
    info_path = MODEL_DIR / "model_info.json"
    if info_path.exists():
        INFO.update(json.loads(info_path.read_text()))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for name in MODEL_NAMES:
        p = MODEL_DIR / f"{name}.onnx"
        if p.exists():
            SESSIONS[name] = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
            print(f"loaded {p.name}")
        else:
            print(f"MISSING {p}")


# ----------------------------------------------------------------------------- helpers
def need(*names):
    missing = [n for n in names if n not in SESSIONS]
    if missing:
        raise HTTPException(503, f"Model(s) not loaded: {', '.join(missing)}. Put the ONNX files in {MODEL_DIR}.")


async def read_image(file: UploadFile) -> np.ndarray:
    """Validate upload, convert to RGB, resize to 128x128 (bicubic, as in training), HWC float [0,1]."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 10 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Unsupported or corrupted image file (use PNG, JPEG, WEBP or BMP).")
    im = im.resize((IMG, IMG), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def data_url(hwc) -> str:
    arr = (np.clip(hwc, 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def nchw(hwc):
    return np.ascontiguousarray(hwc.transpose(2, 0, 1)[None], dtype=np.float32)


def hwc(nchw_arr):
    return nchw_arr[0].transpose(1, 2, 0)


def run(name, feeds):
    t = time.perf_counter()
    out = SESSIONS[name].run(None, feeds)
    return out, (time.perf_counter() - t) * 1000


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim(a, b):
    """SSIM with an 11x11 Gaussian window (sigma 1.5), reflect padding."""
    f = lambda z: K.gaussian_blur(z, 11, 1.5)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    va, vb, cov = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


_STOPS = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], np.float32) / 255


def error_map(out, ref):
    """|out - ref| averaged over channels, x3 gain, rendered with an inferno-like colormap."""
    e = np.clip(np.abs(out - ref).mean(2) * 3, 0, 1) * (len(_STOPS) - 1)
    i = np.clip(e.astype(int), 0, len(_STOPS) - 2)
    t = (e - i)[..., None]
    return _STOPS[i] * (1 - t) + _STOPS[i + 1] * t


def make_corruption(img, corruption, severity, seed, salt_p, blur_k, blur_sigma, occ_coverage, occ_rects):
    """Returns (model input, corruption info or None). corruption='none' -> image used as uploaded."""
    if corruption in (None, "", "none"):
        return img, None
    if corruption not in CLASSES:
        raise HTTPException(422, f"corruption must be one of none, {', '.join(CLASSES)}")
    seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    severity = severity or "medium"
    if corruption == "clean":
        params, severity = {}, "none"
    elif severity in ("low", "medium", "high"):
        params = K.level_params(corruption, severity, rng)
    elif severity == "custom":
        params = K.custom_params(corruption, rng, p=salt_p, k=blur_k, sigma=blur_sigma,
                                 coverage=occ_coverage, n=occ_rects)
    else:
        raise HTTPException(422, "severity must be low, medium, high or custom")
    noisy = K.apply(img, corruption, params)
    return noisy, {"type": corruption, "severity": severity, "params": params, "seed": seed}


def restoration_response(clean, noisy, out, info, model, times, **extra):
    has_ref = info is not None
    return {
        "input_image": data_url(noisy),
        "output_image": data_url(out),
        "reference_image": data_url(clean) if has_ref else None,
        "error_map": data_url(error_map(out, clean)) if has_ref else None,
        "metrics": {"psnr_input": psnr(noisy, clean), "psnr_output": psnr(out, clean),
                    "ssim_input": ssim(noisy, clean), "ssim_output": ssim(out, clean)} if has_ref else None,
        "corruption": info,
        "inference_ms": {k: round(v, 2) for k, v in times.items()},
        "model": model,
        **extra,
    }


# ----------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    models = {}
    for n in MODEL_NAMES:
        p = MODEL_DIR / f"{n}.onnx"
        s = SESSIONS.get(n)
        models[n] = {"loaded": s is not None, "file": p.name,
                     "size_mb": round(p.stat().st_size / 2**20, 2) if p.exists() else None,
                     "inputs": [i.name for i in s.get_inputs()] if s else [],
                     "outputs": [o.name for o in s.get_outputs()] if s else []}
    return {"status": "ok" if len(SESSIONS) == len(MODEL_NAMES) else "degraded", "models": models,
            "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
            "image_size": IMG, "uptime_s": round(time.time() - STARTED, 1),
            "temperature": INFO.get("models", {}).get("soft_moe", {}).get("temperature")}


@app.get("/api/samples")
def samples():
    res = {}
    for group in ["pets", "faces"]:
        d = SAMPLE_DIR / group
        files = sorted(p.name for p in d.glob("*.png")) if d.exists() else []
        res[group] = [{"name": f, "url": f"/api/samples/{group}/{f}"} for f in files]
    return res


@app.get("/api/samples/{group}/{name}")
def sample_file(group: str, name: str):
    if group not in ("pets", "faces") or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404, "Not found")
    p = SAMPLE_DIR / group / name
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")


@app.post("/api/corrupt")
async def corrupt(file: UploadFile = File(...), corruption: str = Form(...), severity: str = Form("medium"),
                  seed: Optional[int] = Form(None), salt_p: Optional[float] = Form(None),
                  blur_k: Optional[int] = Form(None), blur_sigma: Optional[float] = Form(None),
                  occ_coverage: Optional[float] = Form(None), occ_rects: Optional[int] = Form(None)):
    clean = await read_image(file)
    noisy, info = make_corruption(clean, corruption, severity, seed, salt_p, blur_k, blur_sigma, occ_coverage, occ_rects)
    return {"clean_image": data_url(clean), "corrupted_image": data_url(noisy), "corruption": info}


@app.post("/api/restore/universal")
async def universal(file: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"),
                    seed: Optional[int] = Form(None), salt_p: Optional[float] = Form(None),
                    blur_k: Optional[int] = Form(None), blur_sigma: Optional[float] = Form(None),
                    occ_coverage: Optional[float] = Form(None), occ_rects: Optional[int] = Form(None)):
    need("universal_dae")
    clean = await read_image(file)
    noisy, info = make_corruption(clean, corruption, severity, seed, salt_p, blur_k, blur_sigma, occ_coverage, occ_rects)
    (out,), ms = run("universal_dae", {"input": nchw(noisy)})
    return restoration_response(clean, noisy, hwc(out), info, "universal_dae.onnx", {"total": ms})


@app.post("/api/restore/hard")
async def hard(file: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"),
               mode: str = Form("predicted"),
               seed: Optional[int] = Form(None), salt_p: Optional[float] = Form(None),
               blur_k: Optional[int] = Form(None), blur_sigma: Optional[float] = Form(None),
               occ_coverage: Optional[float] = Form(None), occ_rects: Optional[int] = Form(None)):
    need("classifier", "specialist_salt", "specialist_blur", "specialist_occlusion")
    clean = await read_image(file)
    noisy, info = make_corruption(clean, corruption, severity, seed, salt_p, blur_k, blur_sigma, occ_coverage, occ_rects)
    x = nchw(noisy)
    (probs,), t_cls = run("classifier", {"input": x})
    probs = probs[0]
    predicted = CLASSES[int(probs.argmax())]
    if mode == "oracle":
        if info is None:
            raise HTTPException(422, "Oracle routing needs a known corruption (choose one instead of 'none').")
        route = info["type"]
    else:
        mode, route = "predicted", predicted
    times = {"classifier": t_cls}
    if route == "clean":
        out = noisy                                   # identity bypass: no expert is run
        times["expert"] = 0.0
    else:
        (o,), times["expert"] = run(f"specialist_{route}", {"input": x})
        out = hwc(o)
    times["total"] = times["classifier"] + times["expert"]
    return restoration_response(
        clean, noisy, out, info, "classifier.onnx + specialist_*.onnx", times,
        probabilities={c: float(p) for c, p in zip(CLASSES, probs)}, predicted=predicted,
        expert="identity" if route == "clean" else route, mode=mode)


@app.post("/api/restore/soft")
async def soft(file: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"),
               seed: Optional[int] = Form(None), salt_p: Optional[float] = Form(None),
               blur_k: Optional[int] = Form(None), blur_sigma: Optional[float] = Form(None),
               occ_coverage: Optional[float] = Form(None), occ_rects: Optional[int] = Form(None)):
    need("soft_moe")
    clean = await read_image(file)
    noisy, info = make_corruption(clean, corruption, severity, seed, salt_p, blur_k, blur_sigma, occ_coverage, occ_rects)
    (out, w, logits, branches), ms = run("soft_moe", {"input": nchw(noisy)})
    w = w[0]
    return restoration_response(
        clean, noisy, hwc(out), info, "soft_moe.onnx", {"total": ms},
        weights={c: float(v) for c, v in zip(CLASSES, w)}, dominant=CLASSES[int(w.argmax())],
        temperature=INFO.get("models", {}).get("soft_moe", {}).get("temperature"),
        expert_images={c: data_url(branches[0, k].transpose(1, 2, 0)) for k, c in enumerate(CLASSES)})


@app.post("/api/sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(1)):
    need("sketch_generator")
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3")
    photo = await read_image(file)
    (y,), ms = run("sketch_generator", {"photo": nchw(photo * 2 - 1), "style": np.array([style - 1], np.int64)})
    return {"input_image": data_url(photo), "output_image": data_url((hwc(y) + 1) / 2), "style": style,
            "inference_ms": {"total": round(ms, 2)}, "model": "sketch_generator.onnx"}
