"""Export every inference model to ONNX and verify ONNX Runtime == PyTorch outputs.

Writes outputs/onnx/*.onnx, outputs/onnx/model_info.json (used by the backend) and
outputs/tables/onnx_verification.csv (max/mean absolute difference per output).
Also copies a few unseen test images to outputs/samples/ for the app gallery.
"""
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src import config as C
from src import engine as E
from src.data import corruptions as K
from src.models.cgan import UNetGenerator
from src.models.restoration import CorruptionClassifier, DenoisingAE, SoftMoE

OPSET = 17


class WithSoftmax(nn.Module):
    """Classifier exported with its softmax so the app receives probabilities directly."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), 1)


def dae_from(name):
    ck = E.load_ckpt(name)
    m = DenoisingAE(base=ck["hparams"]["base"], z_ch=ck["hparams"]["z_ch"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["hparams"]


def export(model, args, path, input_names, output_names):
    dyn = {n: {0: "batch"} for n in input_names + output_names}
    kw = dict(opset_version=OPSET, input_names=input_names, output_names=output_names, dynamic_axes=dyn)
    try:
        torch.onnx.export(model, args, str(path), dynamo=False, **kw)   # classic exporter (torch >= 2.5)
    except TypeError:
        torch.onnx.export(model, args, str(path), **kw)


def verify(model, args, path, input_names, output_names):
    # torch.onnx.export restores the module's previous train/eval flag on the WHOLE tree
    # afterwards, so force eval again: BatchNorm must use running statistics, as in ONNX.
    model.eval()
    with torch.no_grad():
        ref = model(*args)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: a.numpy() for n, a in zip(input_names, args)})
    rows = []
    for name, r, g in zip(output_names, ref, got):
        d = np.abs(r.numpy() - g)
        rows.append({"model": path.name, "output": name, "shape": list(g.shape),
                     "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                     "pass(<1e-3)": bool(d.max() < 1e-3)})
    return rows


def test_inputs(n=8):
    """Real corrupted test images (one of each corruption) rather than random noise."""
    from src.data.pets import load_split_images
    imgs = load_split_images("test")[:n].astype(np.float32) / 255
    rng = np.random.default_rng(0)
    xs = []
    for i, im in enumerate(imgs):
        cls = K.CLASSES[i % 4]
        xs.append(K.apply(im, cls, K.sample_params(cls, rng)))
    return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()


def run():
    torch.set_grad_enabled(False)
    onnx_dir = C.out("onnx")
    x = test_inputs()
    rows, info = [], {"image_size": C.IMG, "classes": K.CLASSES, "opset": OPSET, "models": {}}

    def do(name, model, args, ins, outs, extra=None):
        p = onnx_dir / f"{name}.onnx"
        export(model, args, p, ins, outs)
        rows.extend(verify(model, args, p, ins, outs))
        info["models"][name] = {"file": p.name, "inputs": ins, "outputs": outs,
                                "size_mb": round(p.stat().st_size / 2**20, 2), **(extra or {})}
        print("exported", p.name)

    norm01 = {"input_range": "[0,1] RGB float32 NCHW", "output_range": "[0,1]"}
    have = lambda *names: all(C.out("checkpoints", f"{n}.pt").exists() for n in names)
    skipped = []

    if have('universal_dae'):
        m, hp = dae_from("universal_dae")
        do("universal_dae", m, (x,), ["input"], ["output"], {**norm01, "hparams": hp})
    else:
        skipped.append('universal_dae')

    if have('classifier'):
        ck = E.load_ckpt("classifier")
        clf = CorruptionClassifier(base=ck["hparams"]["base"], dropout=ck["hparams"]["dropout"])
        clf.load_state_dict(ck["state_dict"])
        do("classifier", WithSoftmax(clf).eval(), (x,), ["input"], ["probabilities"],
           {"input_range": "[0,1]", "classes": K.CLASSES, "hparams": ck["hparams"]})
    else:
        skipped.append('classifier')

    if have('specialist_salt', 'specialist_blur', 'specialist_occlusion'):
        for cls in ["salt", "blur", "occlusion"]:
            m, hp = dae_from(f"specialist_{cls}")
            do(f"specialist_{cls}", m, (x,), ["input"], ["output"], {**norm01, "hparams": hp})
    else:
        skipped.append('specialist_salt')

    if have('soft_moe', 'specialist_salt'):
        ck = E.load_ckpt("soft_moe")
        sd = ck["state_dict"]
        gate = CorruptionClassifier(base=sd["gate.features.0.0.weight"].shape[0])
        sp = E.load_ckpt("specialist_salt")["hparams"]
        experts = [DenoisingAE(base=sp["base"], z_ch=sp["z_ch"]) for _ in range(3)]
        moe = SoftMoE(gate, *experts, tau=ck["hparams"]["tau"])
        moe.load_state_dict(sd)
        do("soft_moe", moe.eval(), (x,), ["input"], ["output", "weights", "logits", "branches"],
           {**norm01, "temperature": float(ck["hparams"]["tau"]),
            "branches": ["clean(identity)", "salt", "blur", "occlusion"], "hparams": ck["hparams"]})
    else:
        skipped.append('soft_moe')

    if have('sketch_generator'):
        ck = E.load_ckpt("sketch_generator")
        hp = ck["hparams"]
        G = UNetGenerator(hp["base"], hp["emb_dim"], hp["dropout"])
        G.load_state_dict(ck["state_dict"])
        photo = x * 2 - 1
        style = torch.tensor([i % 3 for i in range(len(photo))], dtype=torch.long)
        do("sketch_generator", G.eval(), (photo, style), ["photo", "style"], ["sketch"],
           {"input_range": "photo [-1,1] RGB NCHW; style int64 in {0,1,2}", "output_range": "[-1,1]",
            "hparams": hp})
    else:
        skipped.append('sketch_generator')
    if skipped:
        print("[export] not trained yet, skipped:", skipped)

    df = pd.DataFrame(rows)
    df.to_csv(C.out("tables", "onnx_verification.csv"), index=False)
    print(df.to_string())
    C.save_json(info, onnx_dir / "model_info.json")
    export_samples()


def export_samples(n=8):
    """A few official TEST images (never used in training) for the app's sample gallery."""
    from src.data.pets import load_split_images
    d = C.out("samples") / "pets"
    d.mkdir(parents=True, exist_ok=True)
    imgs = load_split_images("test")
    for i, k in enumerate(np.random.default_rng(7).choice(len(imgs), n, replace=False)):
        Image.fromarray(imgs[k]).save(d / f"pet_{i}.png")
    try:
        from src.data.fs2k import load_official
        ph, _, _, _ = load_official("test")
        d = C.out("samples") / "faces"
        d.mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(np.random.default_rng(7).choice(len(ph), min(n, len(ph)), replace=False)):
            Image.fromarray(ph[k]).save(d / f"face_{i}.png")
    except Exception as e:
        print("[samples] faces skipped:", e)


if __name__ == "__main__":
    run()
