"""Download the trained ONNX models into ./models.

Usage:
    python scripts/download_models.py                    # reads models/MODEL_LINKS.json
    python scripts/download_models.py --zip <URL>        # one zip containing the .onnx files

models/MODEL_LINKS.json  ->  {"universal_udae.onnx": "https://...", "classifier.onnx": "https://...", ...}
(Google Drive links work if you `pip install gdown`.)
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
REQUIRED = ["universal_udae.onnx", "classifier.onnx", "specialist_salt.onnx", "specialist_blur.onnx",
            "specialist_occlusion.onnx", "soft_moe.onnx", "generator.onnx"]


def fetch(url, dest):
    print("downloading", dest.name)
    if "drive.google.com" in url:
        try:
            import gdown
            gdown.download(url, str(dest), quiet=False, fuzzy=True)
            return
        except ImportError:
            print("tip: pip install gdown for Google Drive links")
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    if a.zip:
        z = MODELS / "models.zip"
        fetch(a.zip, z)
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if n.endswith(".onnx"):
                    with f.open(n) as src, open(MODELS / Path(n).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        z.unlink()
    else:
        links = json.load(open(MODELS / "MODEL_LINKS.json"))
        for name, url in links.items():
            if url and not url.startswith("PASTE"):
                fetch(url, MODELS / name)
    missing = [m for m in REQUIRED if not (MODELS / m).exists()]
    print("missing:", missing or "none")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
