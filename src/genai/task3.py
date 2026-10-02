"""Task 3: jointly trained soft mixture-of-experts restoration.

Gate = Task-2 classifier, experts = Task-2 specialists, plus an identity branch for clean images.
Stage A (warm-up): experts frozen, only the gate trains.   Stage B: everything fine-tunes with a smaller LR.

    python -m genai.task3 --mode all
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F

from . import data_pets as dp
from .common import CKPT_DIR, CLASSES, OUT_DIR, SEED, count_params, get_device, mkdirs, save_json, seed_everything, worker_count
from .losses import balance_loss, psnr, ssim
from .models import SoftMoE
from .optuna_utils import make_study, run_study
from .task2 import SPECIALISTS, load_classifier, load_specialist
from .tracking import Tracker

SEARCH_SPACE = {
    "lr": "loguniform[1e-5, 1e-3]  (fine-tuning LR; warm-up LR = 3 x lr)",
    "tau": "loguniform[0.3, 3.0]",
    "lam_ce": "loguniform[0.01, 1.0]",
    "lam_bal": "loguniform[1e-3, 1.0]",
    "lam_l1": "uniform[0.5, 0.95]  (L1 weight; SSIM weight = 1 - lam_l1)",
}


def build_moe(tau: float, device) -> SoftMoE:
    gate = load_classifier(device=device)
    experts = [load_specialist(c, device) for c in SPECIALISTS]
    return SoftMoE(gate, experts, tau).to(device)


@torch.no_grad()
def validate_moe(model, loader, device, max_batches=None) -> dict:
    model.eval()
    l1s = ss = ps = 0.0
    n = 0
    wsum = torch.zeros(4)
    wcls = torch.zeros(4, 4)
    cnt = torch.zeros(4)
    correct = 0
    for b, (xn, xc, y, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        xn, xc = xn.to(device), xc.to(device)
        out, w, logits = model(xn)
        l1s += F.l1_loss(out, xc, reduction="none").flatten(1).mean(1).sum().item()
        ss += ssim(out, xc, per_image=True).sum().item()
        ps += psnr(out, xc).sum().item()
        n += len(y)
        wsum += w.sum(0).cpu()
        correct += (logits.argmax(1).cpu() == y).sum().item()
        for c in range(4):
            m = y == c
            if m.any():
                wcls[c] += w[m.to(device)].sum(0).cpu()
                cnt[c] += m.sum()
    l1, s, p = l1s / n, ss / n, ps / n
    return {"val_l1": l1, "val_ssim": s, "val_psnr": p, "val_obj": l1 + (1 - s), "val_gate_acc": correct / n,
            "mean_weights": (wsum / n).tolist(), "weights_by_class": (wcls / cnt.clamp_min(1)[:, None]).tolist()}


def moe_loss(out, w, logits, xc, y, c):
    rec = c["lam_l1"] * F.l1_loss(out, xc) + (1 - c["lam_l1"]) * (1 - ssim(out, xc))
    ce = F.cross_entropy(logits, y)
    bal = balance_loss(w)
    return rec + c["lam_ce"] * ce + c["lam_bal"] * bal, {"rec": rec.item(), "ce": ce.item(), "bal": bal.item()}


def loaders(data, bs, workers):
    ds = dp.RestorationDataset(data.train, train=True)
    tr = dp.make_loader(ds, bs, workers=workers, batch_sampler=dp.BalancedBatchSampler(len(ds), bs))  # balanced batches
    va = dp.make_loader(dp.RestorationDataset(data.trainval, manifest=data.val_manifest), 64, workers=workers)
    return tr, va


def train_moe(cfg, data, device, workers, warmup_epochs, epochs, tracker=None, trial=None, ckpt=None, max_batches=None,
              prune_collapse=False, log=print):
    bs = cfg.get("batch_size", 32)
    tr, va = loaders(data, bs, workers)
    model = build_moe(cfg["tau"], device)
    best, hist, step = float("inf"), [], 0
    stages = [("warmup", warmup_epochs, 3 * cfg["lr"], list(model.gate.parameters())),
              ("finetune", epochs, cfg["lr"], list(model.parameters()))]
    for name, n_ep, lr, params in stages:
        if n_ep == 0:
            continue
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
        for ep in range(n_ep):
            model.train()
            if name == "warmup":  # frozen experts keep BN/dropout in eval mode
                model.experts.eval()
            t0, tot, n, parts = time.time(), 0.0, 0, {"rec": 0.0, "ce": 0.0, "bal": 0.0}
            for b, (xn, xc, y, _) in enumerate(tr):
                if max_batches and b >= max_batches:
                    break
                xn, xc, y = xn.to(device), xc.to(device), y.to(device)
                out, w, logits = model(xn)
                loss, pr = moe_loss(out, w, logits, xc, y, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(y)
                n += len(y)
                for k in parts:
                    parts[k] += pr[k] * len(y)
            sched.step()
            step += 1
            val = validate_moe(model, va, device, max_batches)
            rec = {"epoch": step, "stage": name, "train_loss": tot / max(n, 1), **{f"train_{k}": v / max(n, 1) for k, v in parts.items()},
                   **{k: v for k, v in val.items() if not isinstance(v, list)}}
            for i, c in enumerate(CLASSES):
                rec[f"w_{c}"] = val["mean_weights"][i]
            hist.append(rec)
            if tracker:
                tracker.metrics({"t3/" + k: v for k, v in rec.items() if isinstance(v, float)}, step=step)
            log(f"[moe/{name}] ep {step} loss {rec['train_loss']:.4f} val_obj {val['val_obj']:.4f} gate_acc {val['val_gate_acc']:.3f} "
                f"w={np.round(val['mean_weights'], 2).tolist()} ({time.time() - t0:.0f}s)")
            if val["val_obj"] < best:
                best = val["val_obj"]
                if ckpt:
                    mkdirs(CKPT_DIR)
                    torch.save({"state": model.state_dict(), "hparams": cfg, "val": val,
                                "gate_cfg": model.gate.cfg, "tau": cfg["tau"],
                                "expert_cfgs": [e.cfg for e in model.experts]}, ckpt)
            if trial is not None:
                trial.report(val["val_obj"], step)
                if prune_collapse and (max(val["mean_weights"]) > 0.7 or min(val["mean_weights"]) < 0.03):
                    raise optuna.TrialPruned()  # routing collapse: one expert takes everything or one is dead
                if trial.should_prune():
                    raise optuna.TrialPruned()
    return best, hist


def load_moe(path=None, device="cpu") -> SoftMoE:
    from .models import CorruptionClassifier, UDAE
    ck = torch.load(path or CKPT_DIR / "task3_moe.pt", map_location=device)
    gate = CorruptionClassifier(tuple(ck["gate_cfg"]["channels"]), ck["gate_cfg"]["dropout"])
    experts = [UDAE(**c) for c in ck["expert_cfgs"]]
    m = SoftMoE(gate, experts, ck["tau"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["optuna", "final", "all"], default="all")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial-epochs", type=int, default=4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--final-epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=worker_count(2))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.trials, a.trial_epochs, a.warmup_epochs, a.final_epochs, a.workers = 2, 1, 1, 1, 0
    mb = 3 if a.smoke else None
    seed_everything()
    device = get_device()
    data = dp.load_pets()
    mkdirs(CKPT_DIR, OUT_DIR)
    study = make_study("task3_moe")

    def objective(trial):
        cfg = {"lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
               "tau": trial.suggest_float("tau", 0.3, 3.0, log=True),
               "lam_ce": trial.suggest_float("lam_ce", 0.01, 1.0, log=True),
               "lam_bal": trial.suggest_float("lam_bal", 1e-3, 1.0, log=True),
               "lam_l1": trial.suggest_float("lam_l1", 0.5, 0.95), "batch_size": 32}
        seed_everything(SEED + trial.number)
        with Tracker("task3_moe", f"trial_{trial.number}", nested=True) as t:
            t.params(cfg)
            best, _ = train_moe(cfg, data, device, a.workers, a.warmup_epochs, a.trial_epochs, t, trial, max_batches=mb,
                                prune_collapse=True)
            t.metrics({"best_val_obj": best})
        return best

    parent = Tracker("task3_moe", "optuna_study")
    if a.mode in ("optuna", "all"):
        s = run_study(study, objective, a.trials, SEARCH_SPACE,
                      {"trial_epochs": a.trial_epochs, "warmup_epochs": a.warmup_epochs,
                       "objective": "min val(L1 + (1 - SSIM)); pruned on routing collapse (max mean weight > 0.7 or min < 0.03)"})
        parent.metrics({"best_value": s["best_value"]})
        print("Best MoE:", s["best_params"], s["best_value"])
    parent.end()
    if a.mode in ("final", "all"):
        best = {**study.best_params, "batch_size": 32}
        seed_everything()
        ck = CKPT_DIR / "task3_moe.pt"
        with Tracker("task3_moe", "final_training") as t:
            t.params({**best, "warmup_epochs": a.warmup_epochs, "epochs": a.final_epochs})
            val, hist = train_moe(best, data, device, a.workers, a.warmup_epochs, a.final_epochs, t, ckpt=ck, max_batches=mb)
            t.artifact(ck)
        save_json({"best_params": best, "warmup_epochs": a.warmup_epochs, "epochs": a.final_epochs, "best_val_obj": val, "history": hist},
                  OUT_DIR / "task3_final.json")


if __name__ == "__main__":
    main()
