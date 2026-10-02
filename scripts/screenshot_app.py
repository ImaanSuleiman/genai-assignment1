"""Capture the four workspaces of the RUNNING app (docker compose up) into report/figures/app_screens.png.

    pip install playwright pillow && playwright install chromium
    python scripts/screenshot_app.py [--url http://localhost:8080] [--face path/to/face.jpg]
"""
import argparse
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--face", default=None, help="a face photo for the Task 4 screenshot")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shots = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1360, "height": 900})
        pg.goto(a.url)
        pg.wait_for_timeout(1500)
        for label, wait in (("Universal Restoration", "Inference time"), ("Hard-Routed Restoration", "Classifier probabilities"),
                            ("Soft Mixture-of-Experts", "Routing weights")):
            pg.click(f"text={label}")
            pg.click("img[alt='sample 2']")
            pg.click("text=Restore image")
            pg.wait_for_selector(f"text={wait}", timeout=30000)
            path = f"/tmp/shot_{len(shots)}.png"
            pg.screenshot(path=path)
            shots.append(path)
        pg.click("text=Face-to-Sketch Generator")
        if a.face:
            pg.set_input_files("input[type=file]", a.face)
            pg.click("text=Generate sketch")
            pg.wait_for_selector("text=Output size", timeout=30000)
        path = f"/tmp/shot_{len(shots)}.png"
        pg.screenshot(path=path)
        shots.append(path)
        b.close()
    ims = [Image.open(s).convert("RGB") for s in shots]
    w, h = ims[0].size
    sheet = Image.new("RGB", (2 * w, 2 * h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 2) * w, (i // 2) * h))
    sheet.resize((w, h)).save(OUT / "app_screens.png")
    print("saved", OUT / "app_screens.png")


if __name__ == "__main__":
    main()
