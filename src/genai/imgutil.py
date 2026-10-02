"""Tiny image helpers (no torchvision dependency)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def make_grid(batch: torch.Tensor, nrow: int, padding: int = 2) -> torch.Tensor:
    """batch (N,C,H,W) in [0,1] -> single (C,H',W') grid."""
    n, c, h, w = batch.shape
    ncol = int(np.ceil(n / nrow))
    grid = torch.ones(c, ncol * (h + padding) + padding, nrow * (w + padding) + padding)
    for i in range(n):
        r, col = divmod(i, nrow)
        y, x = padding + r * (h + padding), padding + col * (w + padding)
        grid[:, y:y + h, x:x + w] = batch[i]
    return grid


def to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.detach().cpu().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a if a.shape[2] == 3 else a[:, :, 0])


def save_image(t: torch.Tensor, path) -> None:
    to_pil(t).save(path)
