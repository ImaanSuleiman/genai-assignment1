"""Copy 8 clean Pets TEST images into app/backend/samples so the 'select a clean sample' workflow
shows real images. Run after `python -m genai.prepare --pets`."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genai.common import DATA_DIR, ROOT

arr = np.load(DATA_DIR / "pets" / "test_128.npy")
out = ROOT / "app" / "backend" / "samples"
out.mkdir(parents=True, exist_ok=True)
for i, idx in enumerate(np.linspace(0, len(arr) - 1, 8).astype(int)):
    Image.fromarray(arr[idx]).save(out / f"sample_{i}.png")
print("wrote 8 samples to", out)
