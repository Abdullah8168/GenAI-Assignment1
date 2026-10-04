# RestoreLab — Generative AI Assignment 1

Four generative systems, trained with PyTorch + Optuna + MLflow, exported to ONNX and served
through one React/Tailwind + FastAPI application running in Docker Compose.

| # | Workspace in the app | Model(s) | Data |
|---|---|---|---|
| 1 | **Universal Restoration** | One convolutional denoising autoencoder with a compressed latent bottleneck | Oxford-IIIT Pet |
| 2 | **Hard-Routed Restoration** | 4-class corruption classifier → salt / blur / occlusion specialist AEs (identity bypass for clean) | Oxford-IIIT Pet |
| 3 | **Soft Mixture-of-Experts Restoration** | Gate (initialised from the classifier) + 3 experts (from the specialists) + identity branch, trained jointly | Oxford-IIIT Pet |
| 4 | **Face-to-Sketch Generator** | Style-conditioned U-Net generator + PatchGAN discriminator (learned style embedding) | FS2K |

---

## 1. Run the application (evaluator quick start)

Requirements: **Docker Desktop** (with Compose v2) and Git.

```bash
git clone <this-repo-url> genai-assignment1
cd genai-assignment1
# 1) get the trained models: download outputs.zip (link below) and unzip it into ./outputs
#    -> ./outputs/onnx/*.onnx, ./outputs/samples/, ./outputs/mlruns/ ...
# 2) start everything
docker compose up --build
```

| URL | What |
|---|---|
| http://localhost:8080 | The application (all four workspaces + System page) |
| http://localhost:8000/docs | FastAPI interactive API documentation |
| http://localhost:5000 | MLflow experiment-tracking records (all Optuna trials and final runs) |

**Trained models / outputs download:** `<PUT YOUR GOOGLE DRIVE / GITHUB RELEASE LINK TO outputs.zip HERE>`

Stop with `Ctrl+C` or `docker compose down`. No Python, Node or IDE is needed on the host.

### API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, loaded ONNX models (inputs/outputs/size), runtime info |
| GET | `/api/samples` | unseen official-test images for the gallery |
| POST | `/api/corrupt` | apply a runtime corruption (same code as training) |
| POST | `/api/restore/universal` | Task 1 |
| POST | `/api/restore/hard` | Task 2 (`mode` = predicted / oracle) |
| POST | `/api/restore/soft` | Task 3 (returns the four routing weights and branch outputs) |
| POST | `/api/sketch` | Task 4 (`style` = 1, 2, 3) |

---

## 2. Reproduce training

### On Kaggle (recommended, free GPU)
1. Create a new Kaggle notebook and import `kaggle/genai_train.ipynb` (File → Import Notebook).
2. Settings: **Accelerator GPU T4/P100**, **Internet ON**.
3. In the first code cell set `REPO_URL` to this GitHub repository (or attach the repo zip as a dataset).
4. **Save Version → Save & Run All (Commit)**. The `quick` profile takes ~2–3 h.
5. Download `outputs.zip` from the version's *Output* tab and unzip into `./outputs`.

Datasets are downloaded automatically: Oxford-IIIT Pet through `torchvision`, FS2K from the
official Google Drive link of https://github.com/DengPingFan/FS2K.

### Locally
```bash
pip install -r requirements.txt                       # + a CUDA build of torch if you have a GPU
GENAI_PROFILE=quick FS2K_ROOT=/path/to/FS2K python run_all.py              # everything
python run_all.py --stages t1 t2                                           # selected stages
```

### Smoke test (checks every code path in a few minutes, CPU, no datasets)
```bash
docker build -f tools/train.Dockerfile -t genai-train .
docker run --rm -e GENAI_PROFILE=smoke -e GENAI_SYNTHETIC=1 -e GENAI_WORKERS=0 \
           -e GENAI_DATA=/w/outputs/cache -v "$PWD/outputs_smoke:/w/outputs" genai-train
```

### Profiles (`src/config.py`)
| profile | Optuna trials (T1/T2c/T2s/T3/T4) | final epochs (T1/T2c/T2s/T3/T4) |
|---|---|---|
| smoke | 2 each | 1–2 |
| quick | 12 / 10 / 8 / 8 / 8 | 40 / 20 / 30 / 3+15 / 120 |
| full | 25 / 20 / 15 / 15 / 15 | 80 / 30 / 60 / 5+30 / 200 |

---

## 3. Repository layout

```
src/
  config.py                 paths, seeds, run profiles
  data/corruptions.py       corruption definitions (pure NumPy, shared with the backend)
  data/pets.py              Pet loading, 80/20 split (seed 42), manifests, balanced batch sampler
  data/fs2k.py              FS2K pairs, stratified 85/15 split, paired augmentation
  models/restoration.py     DenoisingAE, CorruptionClassifier, SoftMoE, balance loss
  models/cgan.py            UNetGenerator, PatchDiscriminator (style embeddings)
  metrics.py                SSIM / PSNR / L1, restoration loss
  engine.py                 loaders, Optuna studies, MLflow, evaluation tables, plots
  tasks/task1_universal.py  Optuna → final training → skip ablation → test evaluation
  tasks/task2_hard_routing.py classifier, specialists, oracle vs predicted routing
  tasks/task3_soft_moe.py   warm-up + joint training, routing analysis, mixed corruptions
  tasks/task4_sketch_gan.py cGAN training, losses logged separately, samples over time
  export_onnx.py            ONNX export + ONNX Runtime vs PyTorch verification
run_all.py                  pipeline entry point
kaggle/genai_train.ipynb    one-click Kaggle training notebook
backend/                    FastAPI app + Dockerfile
frontend/                   React + Vite + Tailwind app + Dockerfile (nginx)
tools/                      MLflow UI image, mlruns path fixer, CPU training image
docker-compose.yml          backend + frontend + MLflow
outputs/                    (generated) onnx/, checkpoints/, figures/, tables/, optuna/*.db,
                            manifests/, mlruns/, samples/
```

After training, the small evidence files are committed:
`git add -f outputs/optuna outputs/tables outputs/figures outputs/manifests`.
Large files (`outputs/onnx`, `outputs/checkpoints`, `outputs/mlruns`) are distributed via the download link.

---

## 4. Key design decisions (summary — details in the report)

* **Runtime corruption.** Every `__getitem__` samples clean / salt / blur / occlusion with equal
  probability and a fresh severity; nothing corrupted is stored. Validation and test use JSON manifests
  (`outputs/manifests/`) holding type, severity, parameters, rectangle coordinates and seed. Test: per image
  1 clean + 3 corruptions × 3 fixed levels.
* **Bottleneck.** The encoder downsamples 128→8; a 1×1 conv squeezes to `z_ch` channels, so the latent has
  `z_ch·64` values (e.g. 1024 vs 49 152 input values). No skip connections in the submitted models; a one-skip
  (64×64) variant is trained as an ablation (`task1_skip_ablation.csv`).
* **Loss.** `α·L1 + (1−α)(1−SSIM)`, α tuned by Optuna. SSIM is implemented in `src/metrics.py`.
* **Balanced classifier batches.** `BalancedBatchSampler` puts exactly `batch/4` samples of each class in
  every batch.
* **Soft MoE.** Gate copied from the classifier, experts from the specialists; warm-up trains only the gate,
  joint fine-tuning uses a 10× smaller learning rate for experts. Optuna prunes trials with routing collapse
  (any branch's mean weight < 2 %).
* **cGAN.** Learned `nn.Embedding` style vector tiled into the input of G and D and added at G's bottleneck.
  Paired augmentation (jitter-crop + flip) is applied to the concatenated photo+sketch tensor so both get the
  identical transform. D-real, D-fake, G-adv and G-L1 are logged separately; samples from the same validation
  photos are logged every N epochs.
* **ONNX.** All seven inference models are exported (opset 17) and compared against PyTorch on real corrupted
  images (`outputs/tables/onnx_verification.csv`, max abs diff < 1e-6).
* **One corruption implementation.** `src/data/corruptions.py` is NumPy-only and is copied into the backend image,
  so app corruptions are identical to training corruptions.

## 5. Acknowledgements
Oxford-IIIT Pet (Parkhi et al., 2012); FS2K (Fan et al., 2022); pix2pix (Isola et al., 2017) for the U-Net/PatchGAN
design; SSIM (Wang et al., 2004). AI assistance (Claude Code) was used for code scaffolding and debugging — see the
AI-use appendix of the report.
