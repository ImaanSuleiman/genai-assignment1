"""Oxford-IIIT Pet data for Tasks 1-3.

prepare():  downloads the dataset (torchvision), converts to RGB 128x128 uint8 arrays, makes the
            80/20 train/val split (seed 42) and writes the deterministic corruption manifests.
            The official test split is only ever used by evaluate.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import corruptions as C
from .common import CLASSES, DATA_DIR, IMG, SEED, load_json, mkdirs, save_json

PETS_DIR = DATA_DIR / "pets"


# --------------------------------------------------------------------------- preparation
def _load_split_images(root: Path, split: str, limit: int | None = None):
    from PIL import Image
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root=str(root), split=split, target_types="category", download=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    arr = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    names = []
    for i in range(n):
        img = Image.open(ds._images[i]).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        arr[i] = np.asarray(img)
        names.append(Path(ds._images[i]).stem)
    return arr, names


def prepare(synthetic: bool = False, limit: int | None = None) -> None:
    mkdirs(PETS_DIR, PETS_DIR / "manifests")
    if synthetic:
        from .synthetic import synthetic_images
        trainval, tv_names = synthetic_images(120, seed=1)
        test, te_names = synthetic_images(40, seed=2)
    else:
        trainval, tv_names = _load_split_images(PETS_DIR / "raw", "trainval", limit)
        test, te_names = _load_split_images(PETS_DIR / "raw", "test", limit)
    np.save(PETS_DIR / "trainval_128.npy", trainval)
    np.save(PETS_DIR / "test_128.npy", test)
    save_json({"trainval": tv_names, "test": te_names}, PETS_DIR / "names.json")

    # 80/20 split, seed 42, on the official trainval collection
    perm = np.random.RandomState(SEED).permutation(len(trainval))
    n_train = int(round(0.8 * len(trainval)))
    train_idx, val_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    save_json({"train": train_idx.tolist(), "val": val_idx.tolist(), "seed": SEED}, PETS_DIR / "splits.json")

    # deterministic manifests: val (1 balanced corruption per image) and test (10 entries per image)
    val_manifest = C.make_val_manifest(len(trainval), seed=SEED, indices=val_idx)
    save_json(val_manifest, PETS_DIR / "manifests" / "val_manifest.json")
    test_manifest = C.make_test_manifest(len(test))
    save_json(test_manifest, PETS_DIR / "manifests" / "test_manifest.json")
    print(f"Pets: train={len(train_idx)} val={len(val_idx)} test={len(test)} "
          f"val_manifest={len(val_manifest)} test_manifest={len(test_manifest)}")


# --------------------------------------------------------------------------- loading
@dataclass
class PetsData:
    trainval: np.ndarray
    test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_manifest: list
    test_manifest: list

    @property
    def train(self):
        return self.trainval[self.train_idx]

    def val_for_type(self, ctype: str):
        """Deterministic validation manifest containing only one corruption type."""
        return C.make_val_manifest(len(self.trainval), seed=SEED + 7, fixed_type=ctype, indices=self.val_idx)


def load_pets(load_test: bool = False) -> PetsData:
    sp = load_json(PETS_DIR / "splits.json")
    trainval = np.load(PETS_DIR / "trainval_128.npy")
    test = np.load(PETS_DIR / "test_128.npy") if load_test else np.zeros((0, IMG, IMG, 3), np.uint8)
    val_manifest = load_json(PETS_DIR / "manifests" / "val_manifest.json")
    test_manifest = load_json(PETS_DIR / "manifests" / "test_manifest.json") if load_test else []
    return PetsData(trainval, test, np.array(sp["train"]), np.array(sp["val"]), val_manifest, test_manifest)


# --------------------------------------------------------------------------- datasets
def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)


class RestorationDataset(Dataset):
    """Returns (corrupted, clean, label, image_index).

    * manifest given   -> deterministic corruption read from the manifest (validation / test)
    * no manifest      -> NEW corruption type and severity sampled every time an image is loaded
                          (runtime corruption, nothing is stored on disk)
    * __getitem__((idx, cls)) forces the corruption class (used by the balanced sampler)
    * fixed_type       -> always the same corruption type (specialist training)
    """

    def __init__(self, images: np.ndarray, manifest: list | None = None, fixed_type: str | None = None,
                 train: bool = False):
        self.images, self.manifest, self.fixed_type, self.train = images, manifest, fixed_type, train
        self._ctr = 0

    def __len__(self):
        return len(self.manifest) if self.manifest is not None else len(self.images)

    def __getitem__(self, i):
        if self.manifest is not None:
            m = self.manifest[i]
            clean = self.images[m["index"]]
            noisy = C.apply(clean, m["params"])
            return to_tensor(noisy), to_tensor(clean), CLASSES.index(m["corruption"]), m["index"]
        forced = None
        if isinstance(i, tuple):
            i, forced = i
        self._ctr += 1
        rng = np.random.default_rng((torch.initial_seed() + self._ctr * 1000003 + int(i)) % (2**32))
        clean = self.images[i]
        if self.train and rng.random() < 0.5:
            clean = clean[:, ::-1]
        if forced is not None:
            label = forced
        elif self.fixed_type is not None:
            label = CLASSES.index(self.fixed_type)
        else:
            label = int(rng.integers(4))  # equal probability for the four input conditions
        params = C.sample_params(CLASSES[label], rng)
        noisy = C.apply(np.ascontiguousarray(clean), params)
        return to_tensor(noisy), to_tensor(clean), label, int(i)


class BalancedBatchSampler:
    """Every batch holds exactly batch_size/4 images of each corruption class."""

    def __init__(self, n_images: int, batch_size: int, seed: int = SEED):
        assert batch_size % 4 == 0, "batch size must be divisible by 4 for balanced batches"
        self.n, self.bs, self.seed, self.epoch = n_images, batch_size, seed, 0

    def __len__(self):
        return max(1, self.n // self.bs)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perm = rng.permutation(self.n)
        for b in range(len(self)):
            ids = perm[(np.arange(self.bs) + b * self.bs) % self.n]
            yield [(int(ids[k]), k % 4) for k in range(self.bs)]


def make_loader(ds, batch_size, shuffle=False, workers=2, batch_sampler=None):
    from torch.utils.data import DataLoader
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) > batch_size, **kw)
