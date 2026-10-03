"""Combine the four app screenshots into report/figures/app_screens.png (2x2).

Put app_universal.png, app_hard.png, app_soft.png, app_face.png in report/figures first.
    python scripts/montage_screens.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
NAMES = [("app_universal.png", "Task 1: Universal Restoration"), ("app_hard.png", "Task 2: Hard-Routed Restoration"),
         ("app_soft.png", "Task 3: Soft Mixture-of-Experts"), ("app_face.png", "Task 4: Face-to-Sketch Generator")]
W = 960
tiles = []
for fn, title in NAMES:
    im = Image.open(FIG / fn).convert("RGB")
    im = im.resize((W, int(im.height * W / im.width)))
    tiles.append((im, title))
H = max(t[0].height for t in tiles) + 34
sheet = Image.new("RGB", (2 * W + 30, 2 * H + 30), "white")
d = ImageDraw.Draw(sheet)
for i, (im, title) in enumerate(tiles):
    x, y = 10 + (i % 2) * (W + 10), 10 + (i // 2) * (H + 10)
    d.text((x + 4, y + 8), title, fill=(30, 30, 30))
    sheet.paste(im, (x, y + 30))
sheet.save(FIG / "app_screens.png")
print("wrote", FIG / "app_screens.png", sheet.size)
