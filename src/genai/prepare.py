"""Data preparation entry point.

    python -m genai.prepare --pets            # downloads Oxford-IIIT Pets, builds cache, splits, manifests
    python -m genai.prepare --fs2k            # builds FS2K cache from data/FS2K (manual download)
    python -m genai.prepare --pets --fs2k --synthetic   # tiny fake data for smoke tests only
"""
import argparse

from . import data_fs2k, data_pets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pets", action="store_true")
    ap.add_argument("--fs2k", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N pet images per split (debug)")
    a = ap.parse_args()
    if not (a.pets or a.fs2k):
        a.pets = a.fs2k = True
    if a.pets:
        data_pets.prepare(synthetic=a.synthetic, limit=a.limit)
    if a.fs2k:
        data_fs2k.prepare(synthetic=a.synthetic)


if __name__ == "__main__":
    main()
