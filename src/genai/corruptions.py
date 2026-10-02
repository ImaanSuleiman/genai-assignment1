"""Corruption pipeline for Tasks 1-3 (assignment specification).

Every corruption is described by a plain dict ("params"). apply() is a pure function of
(image, params), so a manifest (list of params) reproduces corruptions exactly.

Training: params are sampled on the fly (sample_params).
Validation / test: params are generated once and stored in JSON manifests.
"""
from __future__ import annotations

import cv2
import numpy as np

TYPES = ["clean", "salt", "blur", "occlusion"]

# Fixed test severities required by the assignment: (low, medium, high)
TEST_SALT = [0.03, 0.08, 0.15]
TEST_BLUR = [(3, 0.7), (5, 1.5), (7, 2.5)]
TEST_OCC = [(1, 0.10), (2, 0.20), (3, 0.35)]  # (num rectangles, area fraction)
LEVEL_NAMES = ["low", "medium", "high"]

# Training ranges
SALT_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA = (0.5, 2.5)
OCC_N = (1, 3)
OCC_FRAC = (0.10, 0.35)


# ----------------------------------------------------------------------------- occlusion
def make_rects(n: int, frac: float, h: int, w: int, rng: np.random.Generator, tol: float = 0.02, tries: int = 60):
    """n black rectangles whose union covers ~frac of the image. Returns (rects, coverage)."""
    target = frac * h * w
    best, best_err, best_cov = None, 1e9, 0.0
    for _ in range(tries):
        shares = rng.dirichlet(np.ones(n) * 4.0)
        rects, mask = [], np.zeros((h, w), dtype=bool)
        for a in shares * target:
            ar = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            rw = int(round(np.sqrt(a * ar)))
            rw = min(max(rw, 1), w)
            rh = min(max(int(round(a / rw)), 1), h)
            x0 = int(rng.integers(0, w - rw + 1))
            y0 = int(rng.integers(0, h - rh + 1))
            rects.append([x0, y0, x0 + rw, y0 + rh])
            mask[y0:y0 + rh, x0:x0 + rw] = True
        cov = float(mask.mean())
        err = abs(cov - frac)
        if err < best_err:
            best, best_err, best_cov = rects, err, cov
        if err <= tol:
            break
    return best, best_cov


# ----------------------------------------------------------------------------- sampling
def sample_params(ctype: str, rng: np.random.Generator, h: int = 128, w: int = 128) -> dict:
    """Random training corruption (uniform within the ranges of the assignment table)."""
    if ctype == "clean":
        return {"type": "clean"}
    if ctype == "salt":
        return {"type": "salt", "p": float(rng.uniform(*SALT_RANGE)), "seed": int(rng.integers(2**31 - 1))}
    if ctype == "blur":
        return {"type": "blur", "k": int(rng.choice(BLUR_KERNELS)), "sigma": float(rng.uniform(*BLUR_SIGMA))}
    if ctype == "occlusion":
        n = int(rng.integers(OCC_N[0], OCC_N[1] + 1))
        frac = float(rng.uniform(*OCC_FRAC))
        rects, cov = make_rects(n, frac, h, w, rng)
        return {"type": "occlusion", "n": n, "rects": rects, "coverage": cov}
    raise ValueError(ctype)


def fixed_params(ctype: str, level: int, rng: np.random.Generator, h: int = 128, w: int = 128) -> dict:
    """Test-time corruption at one of the three fixed severities (level 0/1/2)."""
    if ctype == "clean":
        return {"type": "clean"}
    if ctype == "salt":
        return {"type": "salt", "p": TEST_SALT[level], "seed": int(rng.integers(2**31 - 1))}
    if ctype == "blur":
        k, s = TEST_BLUR[level]
        return {"type": "blur", "k": k, "sigma": s}
    if ctype == "occlusion":
        n, frac = TEST_OCC[level]
        rects, cov = make_rects(n, frac, h, w, rng, tol=0.01, tries=200)
        return {"type": "occlusion", "n": n, "rects": rects, "coverage": cov}
    raise ValueError(ctype)


def severity_bucket(params: dict) -> str | None:
    """low / medium / high: tertiles of the training range (used to report validation results)."""
    t = params["type"]
    if t == "clean":
        return None
    if t == "salt":
        v, lo, hi = params["p"], *SALT_RANGE
    elif t == "blur":
        v, lo, hi = params["sigma"], *BLUR_SIGMA
    else:
        v, lo, hi = params["coverage"], *OCC_FRAC
    r = (v - lo) / (hi - lo)
    return LEVEL_NAMES[0 if r < 1 / 3 else (1 if r < 2 / 3 else 2)]


# ----------------------------------------------------------------------------- apply
def apply(img: np.ndarray, params: dict) -> np.ndarray:
    """img: HxWx3 uint8. Returns a corrupted copy (uint8)."""
    t = params["type"]
    if t == "clean":
        return img.copy()
    h, w = img.shape[:2]
    if t == "salt":
        rng = np.random.default_rng(params["seed"])
        selected = rng.random((h, w)) < params["p"]
        white = rng.random((h, w)) < 0.5  # equal probability black / white
        out = img.copy()
        out[selected & white] = 255
        out[selected & ~white] = 0
        return out
    if t == "blur":
        k = int(params["k"])
        return cv2.GaussianBlur(img, (k, k), sigmaX=float(params["sigma"]), borderType=cv2.BORDER_REFLECT_101)
    if t == "occlusion":
        out = img.copy()
        for x0, y0, x1, y1 in params["rects"]:
            out[y0:y1, x0:x1] = 0
        return out
    raise ValueError(t)


# ----------------------------------------------------------------------------- manifests
def make_val_manifest(n_images: int, seed: int = 42, fixed_type: str | None = None, indices=None) -> list[dict]:
    """One deterministic corruption per validation image, classes balanced (i % 4 after a seeded shuffle)."""
    rng = np.random.default_rng(seed)
    indices = list(range(n_images)) if indices is None else list(indices)
    order = rng.permutation(len(indices))
    out = []
    for rank, pos in enumerate(order):
        ctype = fixed_type or TYPES[rank % 4]
        params = sample_params(ctype, rng)
        out.append({"index": int(indices[pos]), "corruption": ctype, "level": severity_bucket(params), "params": params})
    out.sort(key=lambda m: m["index"])
    return out


def make_test_manifest(n_images: int, seed: int = 1234) -> list[dict]:
    """Every test image x (clean + 3 corruptions x 3 fixed severities) = 10 entries per image."""
    out = []
    for i in range(n_images):
        rng = np.random.default_rng(seed * 100003 + i)
        out.append({"index": i, "corruption": "clean", "level": None, "params": {"type": "clean"}})
        for ctype in TYPES[1:]:
            for lv in range(3):
                out.append({"index": i, "corruption": ctype, "level": LEVEL_NAMES[lv],
                            "params": fixed_params(ctype, lv, rng)})
    return out
