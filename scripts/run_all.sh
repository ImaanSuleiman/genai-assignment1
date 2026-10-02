#!/usr/bin/env bash
# Full pipeline on a GPU machine (Colab/Kaggle/local). A few hours on a T4 with the default budgets.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m genai.prepare --pets --fs2k          # --fs2k needs data/FS2K (manual download)
python -m genai.task1 --mode all
python -m genai.task2 --part both --mode all
python -m genai.task3 --mode all
python -m genai.task4 --mode all
python -m genai.evaluate --task all
python -m genai.export
python scripts/make_samples.py
echo "Done. ONNX in ./models, figures in ./outputs/figures, tables in ./outputs/results"
