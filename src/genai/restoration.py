"""Shared training / validation code for restoration models (Task 1 UDAE and Task 2 specialists)."""
from __future__ import annotations

import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F

from .common import CKPT_DIR, mkdirs
from .losses import psnr, ssim, udae_loss


@torch.no_grad()
def validate(model, loader, device, max_batches: int | None = None) -> dict:
    """Validation L1 / SSIM / PSNR. obj = L1 + (1 - SSIM): fixed combination of reconstruction quality and
    structural similarity, independent of the loss weight alpha being tuned."""
    model.eval()
    l1s, ss, ps, n = 0.0, 0.0, 0.0, 0
    for b, (xn, xc, _, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        xn, xc = xn.to(device), xc.to(device)
        out = model(xn)
        k = xn.shape[0]
        l1s += F.l1_loss(out, xc, reduction="none").flatten(1).mean(1).sum().item()
        ss += ssim(out, xc, per_image=True).sum().item()
        ps += psnr(out, xc).sum().item()
        n += k
    l1, s, p = l1s / n, ss / n, ps / n
    return {"val_l1": l1, "val_ssim": s, "val_psnr": p, "val_obj": l1 + (1 - s)}


def train_restoration(model, train_loader, val_loader, *, lr: float, alpha: float, epochs: int, device,
                      tracker=None, trial: optuna.Trial | None = None, ckpt_path=None, ckpt_extra=None,
                      weight_decay: float = 0.0, max_batches: int | None = None, prefix: str = "", log=print):
    """Adam + cosine schedule, loss = alpha*L1 + (1-alpha)*(1-SSIM). Saves best-val checkpoint."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, history = float("inf"), []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b, (xn, xc, _, _) in enumerate(train_loader):
            if max_batches and b >= max_batches:
                break
            xn, xc = xn.to(device, non_blocking=True), xc.to(device, non_blocking=True)
            loss = udae_loss(model(xn), xc, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * xn.shape[0]
            n += xn.shape[0]
        sched.step()
        val = validate(model, val_loader, device, max_batches)
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"], **val}
        history.append(rec)
        if tracker:
            tracker.metrics({prefix + k: v for k, v in rec.items() if k != "epoch"}, step=ep + 1)
        log(f"[{prefix or 'restore'}] ep {ep + 1}/{epochs} loss {rec['train_loss']:.4f} val_obj {val['val_obj']:.4f} "
            f"ssim {val['val_ssim']:.3f} psnr {val['val_psnr']:.2f} ({time.time() - t0:.0f}s)")
        if val["val_obj"] < best:
            best = val["val_obj"]
            if ckpt_path:
                mkdirs(CKPT_DIR)
                torch.save({"cfg": model.cfg, "state": model.state_dict(), "val": val, **(ckpt_extra or {})}, ckpt_path)
        if trial is not None:
            trial.report(val["val_obj"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, history
