"""FS2K face-sketch data for Task 4.

Expected folder (as released by the dataset authors), placed in data/FS2K:
    FS2K/photo/photo1|photo2|photo3/*.jpg
    FS2K/sketch/sketch1|sketch2|sketch3/*
    FS2K/anno_train.json   FS2K/anno_test.json     (fields incl. image_name, style)

Official train/test definitions are used. 15% of the official training portion becomes the validation set
(stratified by sketch style, seed 42). The official test set is only used by evaluate.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import DATA_DIR, IMG, SEED, load_json, mkdirs, save_json

FS2K_RAW = DATA_DIR / "FS2K"
FS2K_DIR = DATA_DIR / "fs2k_cache"
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def _index_dir(d: Path) -> dict:
    """number-in-filename -> path, for one sketch/photo sub-folder."""
    idx = {}
    if d.is_dir():
        for p in d.iterdir():
            if p.suffix in EXTS:
                nums = re.findall(r"\d+", p.stem)
                if nums:
                    idx[int(nums[-1])] = p
    return idx


def read_pairs(root: Path, split: str):
    """Returns list of (photo_path, sketch_path, style) for the official split."""
    anno = load_json(root / f"anno_{split}.json")
    if isinstance(anno, dict):
        anno = list(anno.values())
    cache, pairs, missing = {}, [], 0
    for a in anno:
        name = a["image_name"].replace("\\", "/")
        name = re.sub(r"\.(jpg|jpeg|png)$", "", name, flags=re.I)
        sub, stem = name.split("/")[-2], name.split("/")[-1]
        num = int(re.findall(r"\d+", stem)[-1])
        sk_sub = sub.replace("photo", "sketch")
        for d in (root / "photo" / sub, root / "sketch" / sk_sub):
            cache.setdefault(d, _index_dir(d))
        p = cache[root / "photo" / sub].get(num)
        s = cache[root / "sketch" / sk_sub].get(num)
        if p is None or s is None:
            missing += 1
            continue
        pairs.append((p, s, int(a["style"])))
    if missing:
        print(f"[FS2K] WARNING: {missing} annotation entries had no matching photo/sketch file in {split}")
    return pairs


def _load_pairs(pairs):
    from PIL import Image
    n = len(pairs)
    ph = np.zeros((n, IMG, IMG, 3), np.uint8)
    sk = np.zeros((n, IMG, IMG, 3), np.uint8)
    st = np.zeros(n, np.int64)
    for i, (p, s, style) in enumerate(pairs):
        ph[i] = np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG), Image.BICUBIC))
        sk[i] = np.asarray(Image.open(s).convert("RGB").resize((IMG, IMG), Image.BICUBIC))
        st[i] = style
    return ph, sk, st


def prepare(synthetic: bool = False) -> None:
    from sklearn.model_selection import train_test_split
    mkdirs(FS2K_DIR)
    if synthetic:
        from .synthetic import synthetic_fs2k
        ph, sk, st = synthetic_fs2k(150, seed=3)
        tph, tsk, tst = synthetic_fs2k(45, seed=4)
    else:
        if not (FS2K_RAW / "anno_train.json").exists():
            raise SystemExit(f"FS2K not found at {FS2K_RAW}. Download it from the FS2K GitHub page "
                             "(DengPingFan/FS2K) and extract so that anno_train.json is in that folder.")
        ph, sk, st = _load_pairs(read_pairs(FS2K_RAW, "train"))
        tph, tsk, tst = _load_pairs(read_pairs(FS2K_RAW, "test"))
    idx = np.arange(len(ph))
    tr, va = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=st)
    np.savez_compressed(FS2K_DIR / "train.npz", photo=ph[np.sort(tr)], sketch=sk[np.sort(tr)], style=st[np.sort(tr)])
    np.savez_compressed(FS2K_DIR / "val.npz", photo=ph[np.sort(va)], sketch=sk[np.sort(va)], style=st[np.sort(va)])
    np.savez_compressed(FS2K_DIR / "test.npz", photo=tph, sketch=tsk, style=tst)
    save_json({"train": len(tr), "val": len(va), "test": len(tph),
               "style_counts_train": np.bincount(st[np.sort(tr)], minlength=3).tolist(),
               "style_counts_val": np.bincount(st[np.sort(va)], minlength=3).tolist(),
               "style_counts_test": np.bincount(tst, minlength=3).tolist()}, FS2K_DIR / "split_info.json")
    print("FS2K:", load_json(FS2K_DIR / "split_info.json"))


def load_split(split: str):
    d = np.load(FS2K_DIR / f"{split}.npz")
    return d["photo"], d["sketch"], d["style"]


# --------------------------------------------------------------------------- dataset
def paired_augment(p: torch.Tensor, s: torch.Tensor, rng: np.random.Generator):
    """Spatial augmentation applied IDENTICALLY to photo and sketch (flip + random crop-resize)."""
    if rng.random() < 0.5:
        p, s = p.flip(-1), s.flip(-1)
    scale = float(rng.uniform(0.8, 1.0))
    h = w = p.shape[-1]
    ch, cw = int(round(h * scale)), int(round(w * scale))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    p = F.interpolate(p[None, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)[0]
    s = F.interpolate(s[None, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)[0]
    return p, s


class PairedSketchDataset(Dataset):
    """Returns (photo, sketch, style) with images scaled to [-1, 1]."""

    def __init__(self, photos, sketches, styles, augment: bool = False):
        self.p, self.s, self.st, self.augment = photos, sketches, styles, augment
        self._ctr = 0

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        p = torch.from_numpy(self.p[i]).permute(2, 0, 1).float() / 255.0
        s = torch.from_numpy(self.s[i]).permute(2, 0, 1).float() / 255.0
        if self.augment:
            self._ctr += 1
            rng = np.random.default_rng((torch.initial_seed() + self._ctr * 1000003 + i) % (2**32))
            p, s = paired_augment(p, s, rng)
        return p * 2 - 1, s * 2 - 1, int(self.st[i])
