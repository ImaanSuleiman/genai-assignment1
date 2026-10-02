"""Shared paths, seeding and small helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("GENAI_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("GENAI_DATA", ROOT / "data"))
CKPT_DIR = Path(os.environ.get("GENAI_CKPT", ROOT / "checkpoints"))
OUT_DIR = Path(os.environ.get("GENAI_OUT", ROOT / "outputs"))
MODELS_DIR = Path(os.environ.get("GENAI_MODELS", ROOT / "models"))
STUDY_DIR = Path(os.environ.get("GENAI_STUDIES", ROOT / "studies"))

CLASSES = ["clean", "salt", "blur", "occlusion"]
CLASS_TITLES = {"clean": "Clean", "salt": "Salt-and-pepper", "blur": "Gaussian blur", "occlusion": "Occlusion"}
SEED = 42
IMG = 128


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mkdirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def worker_count(default: int = 2) -> int:
    try:
        return max(0, min(default, os.cpu_count() or 1))
    except Exception:
        return 0
