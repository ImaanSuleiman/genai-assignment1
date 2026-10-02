"""Task 4: style-conditioned face-to-sketch conditional GAN (FS2K).

    python -m genai.task4 --mode all
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data_fs2k as df
from .common import CKPT_DIR, OUT_DIR, SEED, count_params, get_device, mkdirs, save_json, seed_everything, worker_count
from .losses import psnr, ssim
from .models import PatchDiscriminator, UNetGenerator, init_weights
from .optuna_utils import make_study, run_study
from .tracking import Tracker

SEARCH_SPACE = {
    "lr_g": "loguniform[1e-4, 1e-3]",
    "lr_d": "loguniform[5e-5, 1e-3]",
    "batch_size": "categorical[4, 8, 16, 32]",
    "base": "categorical[32, 48, 64]  (base channel count of G and D)",
    "dropout": "uniform[0.0, 0.5]  (generator decoder dropout)",
    "style_dim": "categorical[8, 16, 32]  (style-embedding dimension)",
    "lam_l1": "loguniform[10, 200]  (L1 reconstruction weight, initial value 100)",
}


def to01(x):
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def validate_gan(G, loader, device, max_batches=None) -> dict:
    G.eval()
    l1 = ss = ps = 0.0
    n = 0
    for b, (p, s, st) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        p, s, st = p.to(device), s.to(device), st.to(device)
        fake = G(p, st)
        a, c = to01(fake), to01(s)
        l1 += F.l1_loss(a, c, reduction="none").flatten(1).mean(1).sum().item()
        ss += ssim(a, c, per_image=True).sum().item()
        ps += psnr(a, c).sum().item()
        n += len(st)
    l1, ss, ps = l1 / n, ss / n, ps / n
    return {"val_l1": l1, "val_ssim": ss, "val_psnr": ps, "val_obj": l1 + (1 - ss)}


@torch.no_grad()
def sample_grid(G, photos, device, n_styles=3):
    """Rows: fixed validation photos; columns: photo | Style 1 | Style 2 | Style 3 (for the fixed-interval log)."""
    from .imgutil import make_grid
    G.eval()
    p = photos.to(device)
    cols = [to01(p)]
    for s in range(n_styles):
        cols.append(to01(G(p, torch.full((len(p),), s, dtype=torch.long, device=device))))
    rows = torch.stack(cols, 1).flatten(0, 1)
    return make_grid(rows.cpu(), nrow=n_styles + 1, padding=2)


def make_loaders(bs, workers, data=None):
    tr_p, tr_s, tr_st = data["train"] if data else df.load_split("train")
    va_p, va_s, va_st = data["val"] if data else df.load_split("val")
    tr = DataLoader(df.PairedSketchDataset(tr_p, tr_s, tr_st, augment=True), batch_size=bs, shuffle=True,
                    drop_last=len(tr_p) > bs, num_workers=workers)
    va = DataLoader(df.PairedSketchDataset(va_p, va_s, va_st), batch_size=32, num_workers=workers)
    return tr, va


def train_gan(cfg, epochs, device, workers, tracker=None, trial=None, ckpt=None, last_ckpt=None, max_batches=None,
              sample_every: int = 0, sample_dir=None, log=print):
    tr, va = make_loaders(cfg["batch_size"], workers)
    fixed = torch.stack([va.dataset[i][0] for i in range(min(4, len(va.dataset)))])  # same photos every time
    G = UNetGenerator(cfg["base"], 3, cfg["style_dim"], cfg["dropout"]).to(device)
    D = PatchDiscriminator(cfg["base"], 3, cfg["style_dim"]).to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    oG = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()
    best, hist = float("inf"), []
    for ep in range(epochs):
        G.train()
        D.train()
        acc = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        n, t0 = 0, time.time()
        for b, (p, s, st) in enumerate(tr):
            if max_batches and b >= max_batches:
                break
            p, s, st = p.to(device), s.to(device), st.to(device)
            fake = G(p, st)
            # ---- discriminator: real pair -> 1, generated pair -> 0
            dr = D(p, s, st)
            df_ = D(p, fake.detach(), st)
            l_real = bce(dr, torch.ones_like(dr))
            l_fake = bce(df_, torch.zeros_like(df_))
            oD.zero_grad(set_to_none=True)
            (0.5 * (l_real + l_fake)).backward()
            oD.step()
            # ---- generator: fool D + stay close to the paired ground-truth sketch
            dg = D(p, fake, st)
            l_adv = bce(dg, torch.ones_like(dg))
            l_l1 = F.l1_loss(fake, s)
            oG.zero_grad(set_to_none=True)
            (l_adv + cfg["lam_l1"] * l_l1).backward()
            oG.step()
            k = len(st)
            n += k
            for key, v in zip(acc, (l_real, l_fake, l_adv, l_l1)):
                acc[key] += v.item() * k
        val = validate_gan(G, va, device, max_batches)
        rec = {"epoch": ep + 1, **{k: v / max(n, 1) for k, v in acc.items()}, **val}
        hist.append(rec)
        if tracker:
            tracker.metrics({"t4/" + k: v for k, v in rec.items() if k != "epoch"}, step=ep + 1)
        log(f"[gan] ep {ep + 1}/{epochs} D(real) {rec['d_real']:.3f} D(fake) {rec['d_fake']:.3f} G_adv {rec['g_adv']:.3f} "
            f"G_L1 {rec['g_l1']:.3f} | val L1 {val['val_l1']:.4f} SSIM {val['val_ssim']:.3f} ({time.time() - t0:.0f}s)")
        if sample_every and sample_dir and ((ep + 1) % sample_every == 0 or ep == 0):
            from .imgutil import save_image
            mkdirs(sample_dir)
            path = f"{sample_dir}/epoch_{ep + 1:03d}.png"
            save_image(sample_grid(G, fixed, device), path)
            if tracker:
                tracker.artifact(path, "samples")
        pack = lambda: {"cfg": G.cfg, "state": G.state_dict(), "d_cfg": D.cfg, "d_state": D.state_dict(), "hparams": cfg, "val": val, "epoch": ep + 1}
        if val["val_obj"] < best:
            best = val["val_obj"]
            if ckpt:
                mkdirs(CKPT_DIR)
                torch.save(pack(), ckpt)
        if last_ckpt:
            torch.save(pack(), last_ckpt)
        if trial is not None:
            trial.report(val["val_obj"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, hist


def load_generator(path=None, device="cpu") -> UNetGenerator:
    ck = torch.load(path or CKPT_DIR / "task4_generator.pt", map_location=device)
    g = UNetGenerator(**ck["cfg"])
    g.load_state_dict(ck["state"])
    return g.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["optuna", "final", "all"], default="all")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--trial-epochs", type=int, default=10)
    ap.add_argument("--final-epochs", type=int, default=150)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=worker_count(2))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.trials, a.trial_epochs, a.final_epochs, a.workers, a.sample_every = 2, 1, 1, 0, 1
    mb = 3 if a.smoke else None
    seed_everything()
    device = get_device()
    mkdirs(CKPT_DIR, OUT_DIR)
    study = make_study("task4_cgan")

    def objective(trial):
        cfg = {"lr_g": trial.suggest_float("lr_g", 1e-4, 1e-3, log=True),
               "lr_d": trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
               "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
               "base": trial.suggest_categorical("base", [32, 48, 64]),
               "dropout": trial.suggest_float("dropout", 0.0, 0.5),
               "style_dim": trial.suggest_categorical("style_dim", [8, 16, 32]),
               "lam_l1": trial.suggest_float("lam_l1", 10, 200, log=True)}
        seed_everything(SEED + trial.number)
        with Tracker("task4_cgan", f"trial_{trial.number}", nested=True) as t:
            t.params(cfg)
            best, _ = train_gan(cfg, a.trial_epochs, device, a.workers, t, trial, max_batches=mb)
            t.metrics({"best_val_obj": best})
        return best

    parent = Tracker("task4_cgan", "optuna_study")
    if a.mode in ("optuna", "all"):
        s = run_study(study, objective, a.trials, SEARCH_SPACE,
                      {"trial_epochs": a.trial_epochs, "objective": "min val(L1 + (1 - SSIM)) of generated vs paired sketch"})
        parent.metrics({"best_value": s["best_value"]})
        print("Best cGAN:", s["best_params"], s["best_value"])
    parent.end()
    if a.mode in ("final", "all"):
        best = study.best_params
        seed_everything()
        ck = CKPT_DIR / "task4_generator.pt"
        with Tracker("task4_cgan", "final_training") as t:
            t.params({**best, "epochs": a.final_epochs})
            val, hist = train_gan(best, a.final_epochs, device, a.workers, t, ckpt=ck, last_ckpt=CKPT_DIR / "task4_last.pt",
                                  max_batches=mb, sample_every=a.sample_every, sample_dir=str(OUT_DIR / "task4_samples"))
            t.artifact(ck)
        save_json({"best_params": best, "epochs": a.final_epochs, "best_val_obj": val, "history": hist},
                  OUT_DIR / "task4_final.json")


if __name__ == "__main__":
    main()
