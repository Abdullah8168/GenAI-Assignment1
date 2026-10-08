"""Exercise every demo feature through the running app's API and save the results.

    python tools/demo_run.py            (app must be running: docker compose up)

Writes images + summary.json to demo_results/ - the same files the app's Download buttons produce.
"""
import base64
import json
import time
import urllib.request
import uuid
from pathlib import Path

API = "http://localhost:8080/api"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_results"
OUT.mkdir(exist_ok=True)


def get(path):
    return json.load(urllib.request.urlopen(API + path, timeout=60))


def post(path, file_path, **fields):
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(file_path).name}\"\r\n"
             f"Content-Type: image/png\r\n\r\n").encode() + Path(file_path).read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(API + path, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, **json.load(e)}


def save(data_url, name):
    if data_url:
        (OUT / name).write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
    return name


def short(d):
    keep = ["metrics", "corruption", "inference_ms", "predicted", "expert", "mode", "probabilities",
            "weights", "dominant", "temperature", "style", "model", "http_error", "detail"]
    return {k: d[k] for k in keep if k in d}


samples = ROOT / "outputs" / "samples"
pet, face = samples / "pets" / "pet_0.png", samples / "faces" / "face_0.png"
summary = {"health": get("/health")}
summary["health"].pop("models", None)
log = lambda name, d: (summary.__setitem__(name, short(d)), print(f"{name:32s}", json.dumps(short(d))[:180]))

# 1) runtime corruption of an uploaded clean image (3 types x 3 severities)
for c in ["salt", "blur", "occlusion"]:
    for s in ["low", "medium", "high"]:
        d = post("/corrupt", pet, corruption=c, severity=s, seed=7)
        save(d["corrupted_image"], f"1_corrupt_{c}_{s}.png")
        log(f"corrupt/{c}/{s}", d)
save(d["clean_image"], "1_clean_input.png")

# 2-4) universal, hard (predicted + oracle), soft for each corruption
for c in ["clean", "salt", "blur", "occlusion"]:
    for ep in ["universal", "hard", "soft"]:
        d = post(f"/restore/{ep}", pet, corruption=c, severity="high", seed=7)
        for k in ["input_image", "output_image", "error_map"]:
            save(d.get(k), f"{ep}_{c}_{k.replace('_image', '')}.png")
        for k, v in (d.get("expert_images") or {}).items():
            save(v, f"soft_{c}_branch_{k}.png")
        log(f"{ep}/{c}", d)
d = post("/restore/hard", pet, corruption="blur", severity="low", mode="oracle", seed=7)
log("hard/blur-low/oracle", d)

# already-corrupted upload (corruption=none): the model is not told anything
noisy = OUT / "1_corrupt_salt_high.png"
for ep in ["universal", "hard", "soft"]:
    d = post(f"/restore/{ep}", noisy, corruption="none")
    save(d["output_image"], f"{ep}_uploaded_corrupted_output.png")
    log(f"{ep}/uploaded-corrupted", d)

# 5) face-to-sketch in all three styles
for st in [1, 2, 3]:
    d = post("/sketch", face, style=st)
    save(d["output_image"], f"sketch_style{st}.png")
    log(f"sketch/style{st}", d)
save(d["input_image"], "sketch_input_photo.png")

# 6) input validation
d = urllib.request.Request(API + "/sketch")
log("validation/bad-file", post("/corrupt", ROOT / "README.md", corruption="salt"))

# 7) experiment tracking records in MLflow
try:
    exps = json.load(urllib.request.urlopen("http://localhost:5000/api/2.0/mlflow/experiments/search?max_results=50"))
    rows = {}
    for e in exps["experiments"]:
        req = urllib.request.Request("http://localhost:5000/api/2.0/mlflow/runs/search",
                                     json.dumps({"experiment_ids": [e["experiment_id"]], "max_results": 1000}).encode(),
                                     {"Content-Type": "application/json"})
        rows[e["name"]] = len(json.load(urllib.request.urlopen(req)).get("runs", []))
    summary["mlflow_runs_per_experiment"] = rows
    print("mlflow", rows)
except Exception as e:
    summary["mlflow_error"] = str(e)
    print("mlflow error", e)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nsaved {len(list(OUT.glob('*.png')))} images + summary.json to {OUT}")
