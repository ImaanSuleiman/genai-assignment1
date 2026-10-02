"""Differentiable SSIM, PSNR and the combined losses used by the assignment."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(win: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(win, dtype=dtype, device=device) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g[:, None] * g[None, :])[None, None]
    return w2d.expand(channels, 1, win, win).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, win: int = 11, sigma: float = 1.5,
         per_image: bool = False) -> torch.Tensor:
    """Mean SSIM between images in [0, data_range], shape (N,C,H,W)."""
    c = x.shape[1]
    w = _gaussian_window(win, sigma, c, x.device, x.dtype)
    pad = win // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    s_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    s_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_xy + c1) * (2 * s_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (s_x + s_y + c2))
    return m.flatten(1).mean(1) if per_image else m.mean()


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB, shape (N,)."""
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


def udae_loss(x_hat, x, alpha: float):
    """L_UDAE = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    return alpha * F.l1_loss(x_hat, x) + (1 - alpha) * (1 - ssim(x_hat, x))


def balance_loss(w: torch.Tensor) -> torch.Tensor:
    """sum_k (mean_batch(w_k) - 1/K)^2 for routing weights w of shape (N,K)."""
    k = w.shape[1]
    return ((w.mean(0) - 1.0 / k) ** 2).sum()
