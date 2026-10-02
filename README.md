# Generative AI, Assignment 1: restoration autoencoders and face-to-sketch cGAN

Four models, one application.

| Task | Model | App workspace |
|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder (UDAE) | Universal Restoration |
| 2 | Corruption classifier + 3 hard-routed specialist autoencoders | Hard-Routed Restoration |
| 3 | Jointly trained soft mixture-of-experts (gate + identity + 3 experts) | Soft Mixture-of-Experts Restoration |
| 4 | Style-conditioned face-to-sketch conditional GAN (U-Net + PatchGAN) | Face-to-Sketch Generator |

Stack: PyTorch, Optuna (all four tasks), MLflow tracking, ONNX Runtime inference, FastAPI backend, React + Tailwind frontend, Docker Compose.

```
src/genai/          data pipeline, models, losses, task1..task4 (train + Optuna), evaluate, export, report_assets
scripts/            run_all.sh, download_models.py, make_samples.py, make_diagrams.py
app/backend         FastAPI + ONNX Runtime (Dockerfile, main.py)
app/frontend        React + Tailwind (Vite), nginx Dockerfile
notebooks/          colab_train.ipynb  (full GPU pipeline)
report/             IEEE LaTeX report (main.tex, generated tables and figures)
docs/               stitch_prompt.md, demo_script.md, ai_use_log.md, FINISHING_CHECKLIST.md
tests/              pytest checks of the assignment's hard requirements
```

## 1. Train (needs a GPU: Colab / Kaggle / local)

The easiest way is `notebooks/colab_train.ipynb`. Locally:

```bash
pip install -r requirements.txt
export PYTHONPATH=src                      # Windows PowerShell: $env:PYTHONPATH="src"
python -m genai.prepare --pets             # downloads Oxford-IIIT Pet, caches 128x128, 80/20 split (seed 42), manifests
# FS2K: download from https://github.com/DengPingFan/FS2K and extract to data/FS2K (photo/, sketch/, anno_*.json)
python -m genai.prepare --fs2k
python -m genai.task1 --mode all           # Optuna study + final training (checkpoints/task1_udae.pt)
python -m genai.task2 --part both --mode all
python -m genai.task3 --mode all           # needs the Task 2 checkpoints
python -m genai.task4 --mode all
python -m genai.evaluate --task all        # test-set metrics + figures -> outputs/results, outputs/figures
python -m genai.export                     # models/*.onnx + outputs/onnx_parity.json (PyTorch vs ONNX check)
python scripts/make_samples.py             # clean sample images for the app
```

`scripts/run_all.sh` runs everything. Every script has `--smoke` (tiny CPU run) and budget flags (`--trials`, `--trial-epochs`, `--final-epochs`).
Defaults take roughly 2 to 4 hours on a T4. Optuna studies are stored in `studies/*.db` and resume after a disconnect.

Experiment tracking (MLflow, SQLite backend): `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Data rules enforced in code: Pets official test split is only read by `evaluate.py`; FS2K official test split likewise; corruptions for training are generated at load time; validation and test corruptions come from stored JSON manifests.

## 2. Run the application (Docker Compose)

```bash
# the ONNX files must be in ./models (created by `python -m genai.export`, or downloaded):
#   edit models/MODEL_LINKS.json with your download links, then
python scripts/download_models.py
docker compose up --build
```

Open http://localhost:8080. The API docs are at http://localhost:8000/docs. Nothing else needs to be started: no VS Code, no Python scripts.
Webcam capture works on `http://localhost` (browsers require https or localhost).

Without Docker (development): `cd app/backend && MODELS_DIR=../../models PYTHONPATH=../../src uvicorn main:app --port 8000` and `cd app/frontend && npm install && npm run dev`.

## 3. Model files

Large files are not committed. Either use Git LFS (`.gitattributes` tracks `models/*.onnx` and `*.pt`), or upload the seven `.onnx` files (or one zip) to Google Drive / a GitHub release and put the links in `models/MODEL_LINKS.json`.
Needed files: `universal_udae.onnx`, `classifier.onnx`, `specialist_salt.onnx`, `specialist_blur.onnx`, `specialist_occlusion.onnx`, `soft_moe.onnx`, `generator.onnx`.

## 4. Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## 5. Report

`report/main.tex` (IEEE conference format). After training: `python -m genai.report_assets` fills the tables and copies the figures, then `cd report && pdflatex main.tex` (twice). Search for `TODO` in the PDF: those are the places that need your own analysis, URLs and screenshots. See `docs/FINISHING_CHECKLIST.md`.
