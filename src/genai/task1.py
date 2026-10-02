"""Task 1: universal multi-corruption denoising autoencoder (UDAE).

    python -m genai.task1 --mode all            # Optuna study, then final training
    python -m genai.task1 --mode all --smoke    # tiny CPU check
"""
from __future__ import annotations

import argparse

import optuna
import torch

from . import data_pets as dp
from .common import CKPT_DIR, OUT_DIR, SEED, count_params, get_device, mkdirs, save_json, seed_everything, worker_count
from .models import UDAE
from .optuna_utils import make_study, run_study
from .restoration import train_restoration
from .tracking import Tracker

SEARCH_SPACE = {
    "lr": "loguniform[1e-4, 3e-3]",
    "batch_size": "categorical[16, 32, 64, 128]",
    "latent_ch": "categorical[8, 16, 32, 64]  (latent = latent_ch x 8 x 8)",
    "base": "categorical[16, 32, 48]  (encoder channels base, 2*base, 4*base, 8*base)",
    "dropout": "uniform[0.0, 0.3]",
    "alpha": "uniform[0.5, 0.95]  (L1 weight; SSIM weight = 1 - alpha)",
}


def loaders(data: dp.PetsData, batch_size: int, workers: int):
    tr = dp.RestorationDataset(data.train, train=True)
    va = dp.RestorationDataset(data.trainval, manifest=data.val_manifest)
    return dp.make_loader(tr, batch_size, shuffle=True, workers=workers), dp.make_loader(va, 64, workers=workers)


def suggest(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "latent_ch": trial.suggest_categorical("latent_ch", [8, 16, 32, 64]),
        "base": trial.suggest_categorical("base", [16, 32, 48]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "alpha": trial.suggest_float("alpha", 0.5, 0.95),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["optuna", "final", "all"], default="all")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--trial-epochs", type=int, default=6)
    ap.add_argument("--final-epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=worker_count(2))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limited-skip", action="store_true", help="ablation: one narrow skip connection")
    a = ap.parse_args(argv)
    if a.smoke:
        a.trials, a.trial_epochs, a.final_epochs, a.workers = 2, 1, 1, 0
    mb = 3 if a.smoke else None
    seed_everything()
    device = get_device()
    data = dp.load_pets()
    mkdirs(CKPT_DIR, OUT_DIR)
    study_name = "task1_udae" + ("_smoke" if a.smoke else "")
    study = make_study(study_name)
    parent = Tracker("task1_udae", "optuna_study")

    def objective(trial):
        cfg = suggest(trial)
        seed_everything(SEED + trial.number)
        tr, va = loaders(data, cfg["batch_size"], a.workers)
        model = UDAE(cfg["base"], cfg["latent_ch"], cfg["dropout"], limited_skip=a.limited_skip)
        with Tracker("task1_udae", f"trial_{trial.number}", nested=True) as t:
            t.params({**cfg, "params": count_params(model), "latent_dim": model.latent_dim})
            best, _ = train_restoration(model, tr, va, lr=cfg["lr"], alpha=cfg["alpha"], epochs=a.trial_epochs,
                                        device=device, tracker=t, trial=trial, max_batches=mb, prefix="t1/")
            t.metrics({"best_val_obj": best})
        return best

    if a.mode in ("optuna", "all"):
        summary = run_study(study, objective, a.trials, SEARCH_SPACE,
                            {"trial_epochs": a.trial_epochs, "objective": "min val(L1 + (1 - SSIM)) on val manifest"})
        parent.params({"n_trials": summary["n_trials"]})
        parent.metrics({"best_value": summary["best_value"]})
        print("Best:", summary["best_params"], summary["best_value"])
        parent.end()
    if a.mode in ("final", "all"):
        best = study.best_params
        seed_everything()
        tr, va = loaders(data, best["batch_size"], a.workers)
        model = UDAE(best["base"], best["latent_ch"], best["dropout"], limited_skip=a.limited_skip)
        ck = CKPT_DIR / ("task1_udae_smoke.pt" if a.smoke else "task1_udae.pt")
        with Tracker("task1_udae", "final_training") as t:
            t.params({**best, "epochs": a.final_epochs, "params": count_params(model), "latent_dim": model.latent_dim})
            val, hist = train_restoration(model, tr, va, lr=best["lr"], alpha=best["alpha"], epochs=a.final_epochs,
                                          device=device, tracker=t, ckpt_path=ck, max_batches=mb, prefix="t1/",
                                          ckpt_extra={"hparams": best})
            t.artifact(ck)
        save_json({"best_params": best, "epochs": a.final_epochs, "best_val_obj": val, "history": hist,
                   "params": count_params(model), "latent_dim": model.latent_dim},
                  OUT_DIR / ("task1_final_smoke.json" if a.smoke else "task1_final.json"))


def load_udae(path, device="cpu") -> UDAE:
    ck = torch.load(path, map_location=device)
    m = UDAE(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


if __name__ == "__main__":
    main()
