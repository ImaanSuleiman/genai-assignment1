"""Procedural images used for smoke tests and as fallback demo samples (NOT for the real experiments)."""
from __future__ import annotations

import cv2
import numpy as np

IMG = 128


def synthetic_images(n: int, seed: int = 0, size: int = IMG):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, size, size, 3), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    for i in range(n):
        c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
        t = (xx * rng.uniform(-1, 1) + yy * rng.uniform(-1, 1))[..., None]
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        img = (c0 * (1 - t) + c1 * t).astype(np.uint8)
        for _ in range(int(rng.integers(2, 6))):
            col = tuple(int(v) for v in rng.uniform(0, 255, 3))
            ctr = (int(rng.integers(10, size - 10)), int(rng.integers(10, size - 10)))
            if rng.random() < 0.5:
                cv2.circle(img, ctr, int(rng.integers(6, 30)), col, -1)
            else:
                cv2.rectangle(img, ctr, (min(size - 1, ctr[0] + int(rng.integers(8, 40))),
                                         min(size - 1, ctr[1] + int(rng.integers(8, 40)))), col, -1)
        out[i] = img
    return out, [f"synthetic_{seed}_{i:04d}" for i in range(n)]


def synthetic_fs2k(n: int, seed: int = 0, size: int = IMG):
    """Fake (photo, sketch, style) triples: sketch = style-dependent edge map of the photo."""
    photos, _ = synthetic_images(n, seed, size)
    sketches = np.zeros_like(photos)
    styles = np.zeros(n, np.int64)
    for i in range(n):
        s = int(i % 3)
        styles[i] = s
        g = cv2.cvtColor(photos[i], cv2.COLOR_RGB2GRAY)
        if s == 0:
            e = 255 - cv2.Canny(g, 60, 120)
        elif s == 1:
            e = cv2.GaussianBlur(g, (5, 5), 0)
        else:
            e = 255 - cv2.Laplacian(g, cv2.CV_8U)
        sketches[i] = np.stack([e] * 3, -1)
    return photos, sketches, styles
