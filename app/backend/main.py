"""FastAPI backend: validates uploads, preprocesses, runs the ONNX models and returns results + timing.

Endpoints
    GET  /api/health         model availability
    GET  /api/samples        clean sample images (for the "select a clean sample" workflow)
    GET  /api/samples/{id}   sample as PNG
    POST /api/universal      Task 1  Universal Restoration
    POST /api/hard-route     Task 2  Hard-Routed Restoration
    POST /api/soft-moe       Task 3  Soft Mixture-of-Experts Restoration
    POST /api/face2sketch    Task 4  Face-to-Sketch Generator
"""
from __future__ import annotations

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
from fastapi.responses import Response
from PIL import Image

from genai import corruptions as C
from genai.synthetic import synthetic_images

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", Path(__file__).parent / "samples"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
IMG = 128
CLASSES = ["clean", "salt", "blur", "occlusion"]
EXPERTS = ["identity (bypass)", "salt-and-pepper expert", "blur expert", "occlusion expert"]
SPEC_FILES = {"salt": "specialist_salt.onnx", "blur": "specialist_blur.onnx", "occlusion": "specialist_occlusion.onnx"}
LEVELS = {"low": 0, "medium": 1, "high": 2}
MODEL_FILES = {"universal": "universal_udae.onnx", "classifier": "classifier.onnx", "soft_moe": "soft_moe.onnx",
               "generator": "generator.onnx", **{f"specialist_{k}": v for k, v in SPEC_FILES.items()}}

app = FastAPI(title="GenAI Assignment 1 - Restoration & Sketch Studio", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

_sessions: dict[str, ort.InferenceSession] = {}


# ----------------------------------------------------------------------------- helpers
def session(name: str) -> ort.InferenceSession:
    if name not in _sessions:
        path = MODELS_DIR / MODEL_FILES[name]
        if not path.exists():
            raise HTTPException(503, f"Model file {path.name} not found in {MODELS_DIR}. "
                                     "Run `python scripts/download_models.py` (see README) and restart the containers.")
        _sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions[name]


def run(name: str, feeds: dict):
    s = session(name)
    t0 = time.perf_counter()
    out = s.run(None, feeds)
    return out, (time.perf_counter() - t0) * 1000.0


def decode_image(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    if len(data) == 0:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_BYTES // 1024 // 1024} MB).")
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()
        im = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Not a valid image file (use JPEG, PNG or WebP).")
    if im.format not in ("JPEG", "PNG", "WEBP"):
        raise HTTPException(400, f"Unsupported image format {im.format}. Use JPEG, PNG or WebP.")
    orig = im.size
    im = im.convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8), orig


def samples() -> list[np.ndarray]:
    files = sorted([p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]) if SAMPLES_DIR.exists() else []
    if files:
        return [np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG), Image.BICUBIC)) for p in files]
    return list(synthetic_images(8, seed=7)[0])  # fallback so the UI always works


async def get_input(file: Optional[UploadFile], sample_id: Optional[int]):
    if file is not None and file.filename:
        img, orig = decode_image(await file.read())
        return img, {"source": "upload", "original_size": list(orig)}
    if sample_id is not None:
        s = samples()
        if not (0 <= sample_id < len(s)):
            raise HTTPException(400, "Unknown sample id.")
        return s[sample_id], {"source": f"sample {sample_id}", "original_size": [IMG, IMG]}
    raise HTTPException(400, "Provide an image file or a sample_id.")


def to_nchw(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def from_nchw(a: np.ndarray) -> np.ndarray:
    return (np.clip(a[0], 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)


def b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(a: np.ndarray, b: np.ndarray) -> str:
    e = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1) / 255.0
    e = np.clip(e / 0.3, 0, 1)
    r = np.clip(1.5 * e, 0, 1)
    g = np.clip(2 * e - 0.6, 0, 1)
    bl = np.clip(0.7 - 2 * np.abs(e - 0.25), 0, 1)
    return b64((np.stack([r, g, bl], -1) * 255).astype(np.uint8))


def apply_corruption(img: np.ndarray, kind: str, level: str, seed: int):
    """kind in none | already_corrupted | salt | blur | occlusion. Returns (input_image, settings, has_reference)."""
    if kind in ("none", "already_corrupted"):
        return img, {"corruption": kind}, False
    if kind not in ("salt", "blur", "occlusion"):
        raise HTTPException(400, f"Unknown corruption '{kind}'.")
    if level not in LEVELS:
        raise HTTPException(400, "level must be low, medium or high.")
    params = C.fixed_params(kind, LEVELS[level], np.random.default_rng(seed))
    settings = {"corruption": kind, "level": level, "seed": seed, **{k: v for k, v in params.items() if k != "type"}}
    return C.apply(img, params), settings, True


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))


# ----------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"status": "ok", "models_dir": str(MODELS_DIR),
            "models": {k: {"file": v, "available": (MODELS_DIR / v).exists(), "loaded": k in _sessions} for k, v in MODEL_FILES.items()}}


@app.get("/api/samples")
def list_samples():
    return {"count": len(samples()), "ids": list(range(len(samples())))}


@app.get("/api/samples/{sid}")
def get_sample(sid: int):
    s = samples()
    if not 0 <= sid < len(s):
        raise HTTPException(404, "No such sample")
    buf = io.BytesIO()
    Image.fromarray(s[sid]).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/api/universal")
async def universal(file: Optional[UploadFile] = File(None), sample_id: Optional[int] = Form(None),
                    corruption: str = Form("none"), level: str = Form("medium"), seed: int = Form(42)):
    clean, src = await get_input(file, sample_id)
    noisy, settings, has_ref = apply_corruption(clean, corruption, level, seed)
    (out,), ms = run("universal", {"input": to_nchw(noisy)})
    restored = from_nchw(out)
    res = {"input_image": b64(noisy), "restored_image": b64(restored), "error_map": error_map(restored, clean if has_ref else noisy),
           "settings": settings, "source": src, "inference_ms": round(ms, 2), "model": "Universal UDAE (ONNX)"}
    if has_ref:
        res["clean_reference"] = b64(clean)
        res["metrics"] = {"psnr_input": psnr(noisy, clean), "psnr_restored": psnr(restored, clean)}
    return res


@app.post("/api/hard-route")
async def hard_route(file: Optional[UploadFile] = File(None), sample_id: Optional[int] = Form(None),
                     corruption: str = Form("none"), level: str = Form("medium"), seed: int = Form(42)):
    clean, src = await get_input(file, sample_id)
    noisy, settings, has_ref = apply_corruption(clean, corruption, level, seed)
    x = to_nchw(noisy)
    (probs,), ms_cls = run("classifier", {"input": x})
    probs = probs[0]
    k = int(np.argmax(probs))
    ms_exp = 0.0
    if k == 0:  # identity bypass: a clean prediction never touches an expert
        restored = noisy
    else:
        (out,), ms_exp = run(f"specialist_{CLASSES[k]}", {"input": x})
        restored = from_nchw(out)
    res = {"input_image": b64(noisy), "restored_image": b64(restored), "error_map": error_map(restored, clean if has_ref else noisy),
           "settings": settings, "source": src,
           "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)}, "predicted_corruption": CLASSES[k],
           "selected_expert": EXPERTS[k], "identity_bypass": k == 0,
           "inference_ms": round(ms_cls + ms_exp, 2), "classifier_ms": round(ms_cls, 2), "expert_ms": round(ms_exp, 2)}
    if has_ref:
        res["clean_reference"] = b64(clean)
        res["metrics"] = {"psnr_input": psnr(noisy, clean), "psnr_restored": psnr(restored, clean)}
        res["classifier_correct"] = CLASSES[k] == corruption
    return res


@app.post("/api/soft-moe")
async def soft_moe(file: Optional[UploadFile] = File(None), sample_id: Optional[int] = Form(None),
                   corruption: str = Form("none"), level: str = Form("medium"), seed: int = Form(42),
                   second_corruption: str = Form("none")):
    """second_corruption lets the user stack two corruptions (the case soft routing is designed for)."""
    clean, src = await get_input(file, sample_id)
    noisy, settings, has_ref = apply_corruption(clean, corruption, level, seed)
    if second_corruption not in ("none", corruption):
        noisy, s2, r2 = apply_corruption(noisy, second_corruption, level, seed + 1)
        settings = {**settings, "second": s2}
        has_ref = has_ref or r2
    out, ms = run("soft_moe", {"input": to_nchw(noisy)})
    restored, w = from_nchw(out[0]), out[1][0]
    k = int(np.argmax(w))
    res = {"input_image": b64(noisy), "restored_image": b64(restored), "error_map": error_map(restored, clean if has_ref else noisy),
           "settings": settings, "source": src,
           "weights": {name: float(v) for name, v in zip(EXPERTS, w)}, "dominant_expert": EXPERTS[k],
           "contribution_percent": [round(float(v) * 100, 1) for v in w], "inference_ms": round(ms, 2)}
    if has_ref:
        res["clean_reference"] = b64(clean)
        res["metrics"] = {"psnr_input": psnr(noisy, clean), "psnr_restored": psnr(restored, clean)}
    return res


@app.post("/api/face2sketch")
async def face2sketch(file: Optional[UploadFile] = File(None), sample_id: Optional[int] = Form(None), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(400, "style must be 1, 2 or 3.")
    photo, src = await get_input(file, sample_id)
    x = (to_nchw(photo) * 2 - 1).astype(np.float32)
    (out,), ms = run("generator", {"photo": x, "style": np.array([style - 1], dtype=np.int64)})
    sketch = (((np.clip(out[0], -1, 1) + 1) / 2).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return {"photo_image": b64(photo), "sketch_image": b64(sketch), "style": f"Style {style}", "source": src,
            "inference_ms": round(ms, 2), "model": "cGAN generator (ONNX)"}
