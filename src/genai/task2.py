"""Task 2: corruption classifier + three hard-routed specialist autoencoders.

    python -m genai.task2 --part classifier --mode all
    python -m genai.task2 --part specialists --mode all
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import data_pets as dp
from .common import CKPT_DIR, CLASSES, OUT_DIR, SEED, count_params, get_device, mkdirs, save_json, seed_everything, worker_count
from .models import CLASSIFIER_CONFIGS, CorruptionClassifier, UDAE
from .optuna_utils import make_study, run_study
from .restoration import train_restoration, validate
from .tracking import Tracker

SPECIALISTS = ["salt", "blur", "occlusion"]

CLS_SPACE = {
    "lr": "loguniform[1e-4, 3e-3]",
    "batch_size": "categorical[16, 32, 64, 128] (balanced: batch/4 per class)",
    "channels": "categorical[small(16-128), medium(32-256), large(32-256x2)]",
    "dropout": "uniform[0.0, 0.5]",
    "weight_decay": "loguniform[1e-6, 1e-2]",
}
SPEC_SPACE = {
    "lr": "loguniform[1e-4, 3e-3]",
    "latent_ch": "categorical[8, 16, 32, 64]",
    "base": "categorical[16, 32, 48]",
    "batch_size": "categorical[16, 32, 64, 128]",
    "alpha": "uniform[0.5, 0.95]  (L1 weight; SSIM weight = 1 - alpha)",
}


# ------------------------------------------------------------------------------------ classifier
@torch.no_grad()
def predict_all(model, loader, device, max_batches=None):
    model.eval()
    ys, ps, probs = [], [], []
    for b, (xn, _, y, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        p = F.softmax(model(xn.to(device)), 1).cpu()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def classification_report(y, p) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=range(4), average="macro", zero_division=0)
    cm = confusion_matrix(y, p, labels=range(4))
    return {"accuracy": float(accuracy_score(y, p)), "macro_precision": float(mp), "macro_recall": float(mr),
            "macro_f1": float(mf),
            "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": (cm / cm.sum(1, keepdims=True).clip(1)).tolist()}


def cls_loaders(data, bs, workers):
    tr_ds = dp.RestorationDataset(data.train, train=True)
    sampler = dp.BalancedBatchSampler(len(tr_ds), bs)
    tr = dp.make_loader(tr_ds, bs, workers=workers, batch_sampler=sampler)
    va = dp.make_loader(dp.RestorationDataset(data.trainval, manifest=data.val_manifest), 64, workers=workers)
    return tr, va


def train_classifier(cfg, data, epochs, device, workers, tracker=None, trial=None, ckpt=None, max_batches=None, log=print):
    tr, va = cls_loaders(data, cfg["batch_size"], workers)
    model = CorruptionClassifier(CLASSIFIER_CONFIGS[cfg["channels"]], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, hist = -1.0, []
    for ep in range(epochs):
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for b, (xn, _, y, _) in enumerate(tr):
            if max_batches and b >= max_batches:
                break
            xn, y = xn.to(device), y.to(device)
            loss = F.cross_entropy(model(xn), y)  # multiclass cross-entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv, _ = predict_all(model, va, device, max_batches)
        rep = classification_report(yv, pv)
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1), "val_acc": rep["accuracy"], "val_macro_f1": rep["macro_f1"]}
        hist.append(rec)
        if tracker:
            tracker.metrics({"cls/" + k: v for k, v in rec.items() if k != "epoch"}, step=ep + 1)
        log(f"[cls] ep {ep + 1}/{epochs} loss {rec['train_loss']:.4f} val_acc {rec['val_acc']:.4f} f1 {rec['val_macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        if rep["macro_f1"] > best:
            best = rep["macro_f1"]
            if ckpt:
                mkdirs(CKPT_DIR)
                torch.save({"cfg": model.cfg, "state": model.state_dict(), "hparams": cfg, "val": rep}, ckpt)
        if trial is not None:
            trial.report(rec["val_macro_f1"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, hist


def run_classifier(a, data, device):
    study = make_study("task2_classifier", "maximize")
    mb = 3 if a.smoke else None

    def objective(trial):
        cfg = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
               "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
               "channels": trial.suggest_categorical("channels", list(CLASSIFIER_CONFIGS)),
               "dropout": trial.suggest_float("dropout", 0.0, 0.5),
               "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
        seed_everything(SEED + trial.number)
        with Tracker("task2_classifier", f"trial_{trial.number}", nested=True) as t:
            t.params(cfg)
            best, _ = train_classifier(cfg, data, a.trial_epochs, device, a.workers, t, trial, max_batches=mb)
            t.metrics({"best_val_macro_f1": best})
        return best

    parent = Tracker("task2_classifier", "optuna_study")
    if a.mode in ("optuna", "all"):
        s = run_study(study, objective, a.trials, CLS_SPACE, {"trial_epochs": a.trial_epochs, "objective": "max val macro-F1"})
        parent.metrics({"best_value": s["best_value"]})
        print("Best classifier:", s["best_params"], s["best_value"])
    parent.end()
    if a.mode in ("final", "all"):
        best = study.best_params
        seed_everything()
        ck = CKPT_DIR / "task2_classifier.pt"
        with Tracker("task2_classifier", "final_training") as t:
            t.params({**best, "epochs": a.final_epochs})
            _, hist = train_classifier(best, data, a.final_epochs, device, a.workers, t, ckpt=ck, max_batches=mb)
            t.artifact(ck)
        save_json({"best_params": best, "epochs": a.final_epochs, "history": hist}, OUT_DIR / "task2_classifier_final.json")


# ------------------------------------------------------------------------------------ specialists
def spec_loaders(data, ctype, bs, workers):
    tr = dp.make_loader(dp.RestorationDataset(data.train, fixed_type=ctype, train=True), bs, shuffle=True, workers=workers)
    va = dp.make_loader(dp.RestorationDataset(data.trainval, manifest=data.val_for_type(ctype)), 64, workers=workers)
    return tr, va


def run_specialists(a, data, device):
    study = make_study("task2_specialists")
    mb = 3 if a.smoke else None

    def objective(trial):
        cfg = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
               "latent_ch": trial.suggest_categorical("latent_ch", [8, 16, 32, 64]),
               "base": trial.suggest_categorical("base", [16, 32, 48]),
               "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
               "alpha": trial.suggest_float("alpha", 0.5, 0.95)}
        scores = []
        with Tracker("task2_specialists", f"trial_{trial.number}", nested=True) as t:
            t.params(cfg)
            for k, ctype in enumerate(SPECIALISTS):  # shared architecture evaluated on all three specialists
                seed_everything(SEED + trial.number * 10 + k)
                tr, va = spec_loaders(data, ctype, cfg["batch_size"], a.workers)
                m = UDAE(cfg["base"], cfg["latent_ch"], 0.1)
                best, _ = train_restoration(m, tr, va, lr=cfg["lr"], alpha=cfg["alpha"], epochs=a.trial_epochs, device=device,
                                            tracker=t, max_batches=mb, prefix=f"t2/{ctype}/")
                scores.append(best)
                t.metrics({f"best_{ctype}": best})
                trial.report(float(np.mean(scores)), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores))

    parent = Tracker("task2_specialists", "optuna_study")
    if a.mode in ("optuna", "all"):
        s = run_study(study, objective, a.trials, SPEC_SPACE,
                      {"trial_epochs": a.trial_epochs, "objective": "min mean over 3 specialists of val(L1 + (1 - SSIM))"})
        parent.metrics({"best_value": s["best_value"]})
        print("Best specialist architecture:", s["best_params"], s["best_value"])
    parent.end()
    if a.mode in ("final", "all"):
        best = study.best_params
        out = {}
        for k, ctype in enumerate(SPECIALISTS):  # three INDEPENDENT trainings (different seeds, separate weights)
            seed_everything(SEED + 100 + k)
            tr, va = spec_loaders(data, ctype, best["batch_size"], a.workers)
            m = UDAE(best["base"], best["latent_ch"], 0.1)
            ck = CKPT_DIR / f"task2_specialist_{ctype}.pt"
            with Tracker("task2_specialists", f"final_{ctype}") as t:
                t.params({**best, "type": ctype, "epochs": a.final_epochs, "params": count_params(m)})
                val, hist = train_restoration(m, tr, va, lr=best["lr"], alpha=best["alpha"], epochs=a.final_epochs, device=device,
                                              tracker=t, ckpt_path=ck, max_batches=mb, prefix=f"t2/{ctype}/",
                                              ckpt_extra={"hparams": best, "type": ctype})
                t.artifact(ck)
            out[ctype] = {"best_val_obj": val, "history": hist}
        save_json({"best_params": best, "epochs": a.final_epochs, "specialists": out}, OUT_DIR / "task2_specialists_final.json")


def load_classifier(path=None, device="cpu"):
    ck = torch.load(path or CKPT_DIR / "task2_classifier.pt", map_location=device)
    m = CorruptionClassifier(tuple(ck["cfg"]["channels"]), ck["cfg"]["dropout"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def load_specialist(ctype, device="cpu"):
    ck = torch.load(CKPT_DIR / f"task2_specialist_{ctype}.pt", map_location=device)
    m = UDAE(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["classifier", "specialists", "both"], default="both")
    ap.add_argument("--mode", choices=["optuna", "final", "all"], default="all")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial-epochs", type=int, default=5)
    ap.add_argument("--final-epochs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=worker_count(2))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.trials, a.trial_epochs, a.final_epochs, a.workers = 2, 1, 1, 0
    seed_everything()
    device = get_device()
    data = dp.load_pets()
    mkdirs(CKPT_DIR, OUT_DIR)
    if a.part in ("classifier", "both"):
        run_classifier(a, data, device)
    if a.part in ("specialists", "both"):
        run_specialists(a, data, device)


if __name__ == "__main__":
    main()
