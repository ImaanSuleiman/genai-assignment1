# What is done and what only you can do

## Already built and tested (CPU smoke tests with synthetic data)
- Data pipeline: runtime corruptions, deterministic validation/test manifests, Pets 80/20 split (seed 42), FS2K stratified 85/15 split, paired augmentation.
- Tasks 1 to 4 training code with Optuna (all four), MLflow logging, checkpoints, resumable studies.
- Evaluation (tables, 12 examples, failure cases, error maps, confusion matrix, routing heatmap, mixed-corruption test, GAN style check).
- ONNX export with PyTorch-vs-ONNX parity check for all 7 models.
- FastAPI backend, React + Tailwind frontend (4 workspaces, webcam), Dockerfiles, Docker Compose. Browser-tested.
- IEEE LaTeX report with generated tables and diagrams, Colab notebook, README, tests.

## Things that could NOT be done in the session (no GPU, no dataset access, no Docker daemon)
1. **Train on a GPU**: run `notebooks/colab_train.ipynb` (about 2 to 4 hours on a T4). No real numbers exist yet.
2. **Download FS2K** (Google Drive link on https://github.com/DengPingFan/FS2K) and put it at `data/FS2K`.
3. **Test the Docker build on your PC**: `docker compose up --build`. I could not run Docker here; the backend was checked in a clean virtual environment that mimics the image, and the frontend was built with `npm run build`. If a Docker build step fails, send me the error.
4. **Google Stitch design**: paste `docs/stitch_prompt.md` into Stitch, keep screenshots as `report/figures/stitch_design.png`. I cannot operate Stitch for you and the evidence has to be genuine.
5. **GitHub repo**: create it, push (Git LFS for `.onnx`/`.pt` or Drive links in `models/MODEL_LINKS.json`), put the URL in the report.
6. **Demo video** (5 to 7 min, YouTube): follow `docs/demo_script.md`, put the link in the report.
7. **Report**: after training run `python -m genai.report_assets`, then replace every red TODO (analysis paragraphs, GPU model, epochs, URLs, screenshots, AI-use appendix). The analysis must describe YOUR results; send me `outputs/results/*.json` and the figures and I will help you write it.
8. **Verify the references** in `report/main.tex` (I wrote them from memory of the literature) and add the sources you actually consulted.
9. **Read and understand the code.** The assignment says you may be asked to explain or modify any component and to run unseen images. See `docs/VIVA_NOTES.md`.

## Deadline
The PDF says March 16, 2024, which cannot be right for a Sept/Oct 2026 semester. Check Google Classroom: late submissions are not accepted.
