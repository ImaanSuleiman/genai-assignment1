"""Export every inference model to ONNX and verify parity with PyTorch.

    python -m genai.export            # writes models/*.onnx and outputs/onnx_parity.json

Models (all with a dynamic batch axis, opset 17):
    universal_udae.onnx        input -> output
    classifier.onnx            input -> probs (softmax of the 4 class logits)
    specialist_{salt,blur,occlusion}.onnx
    soft_moe.onnx              input -> output, weights   (temperature baked in)
    generator.onnx             photo [-1,1], style int64 {0,1,2} -> sketch [-1,1]
"""
from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CKPT_DIR, MODELS_DIR, OUT_DIR, mkdirs, save_json
from .task1 import load_udae
from .task2 import SPECIALISTS, load_classifier, load_specialist
from .task3 import load_moe
from .task4 import load_generator


class ClassifierProbs(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return F.softmax(self.m(x), dim=1)


class MoEWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y, w, _ = self.m(x)
        return y, w


def _export(model, args, path, input_names, output_names, dynamic):
    model.eval()
    kw = dict(input_names=input_names, output_names=output_names, opset_version=17,
              dynamic_axes={n: {0: "batch"} for n in input_names + output_names})
    try:
        torch.onnx.export(model, args, str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, args, str(path), **kw)


def _parity(name, torch_fn, path, feeds: dict, tol=1e-3):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
    with torch.no_grad():
        t_out = torch_fn(*feeds.values())
    t_out = [t_out] if isinstance(t_out, torch.Tensor) else list(t_out)
    diffs = [float(np.abs(o - t.numpy()).max()) for o, t in zip(onnx_out, t_out)]
    ok = max(diffs) < tol
    print(f"[onnx] {name:<22} max|diff| = {max(diffs):.2e}  {'OK' if ok else 'MISMATCH'}")
    return {"model": name, "file": str(path.name), "max_abs_diff": max(diffs), "tolerance": tol, "ok": ok,
            "size_mb": round(path.stat().st_size / 1e6, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    mkdirs(MODELS_DIR, OUT_DIR)
    torch.manual_seed(0)
    x = torch.rand(3, 3, 128, 128)
    report = []

    def want(n):
        return a.only is None or n in a.only

    if want("udae") and (CKPT_DIR / "task1_udae.pt").exists():
        m = load_udae(CKPT_DIR / "task1_udae.pt")
        p = MODELS_DIR / "universal_udae.onnx"
        _export(m, (x,), p, ["input"], ["output"], True)
        report.append(_parity("universal_udae", m, p, {"input": x}))
    if want("classifier") and (CKPT_DIR / "task2_classifier.pt").exists():
        m = ClassifierProbs(load_classifier()).eval()
        p = MODELS_DIR / "classifier.onnx"
        _export(m, (x,), p, ["input"], ["probs"], True)
        report.append(_parity("classifier", m, p, {"input": x}))
    for c in SPECIALISTS:
        if want("specialists") and (CKPT_DIR / f"task2_specialist_{c}.pt").exists():
            m = load_specialist(c)
            p = MODELS_DIR / f"specialist_{c}.onnx"
            _export(m, (x,), p, ["input"], ["output"], True)
            report.append(_parity(f"specialist_{c}", m, p, {"input": x}))
    if want("moe") and (CKPT_DIR / "task3_moe.pt").exists():
        m = MoEWrapper(load_moe()).eval()
        p = MODELS_DIR / "soft_moe.onnx"
        _export(m, (x,), p, ["input"], ["output", "weights"], True)
        report.append(_parity("soft_moe", m, p, {"input": x}))
    if want("generator") and (CKPT_DIR / "task4_generator.pt").exists():
        g = load_generator()
        photo = torch.rand(3, 3, 128, 128) * 2 - 1
        style = torch.tensor([0, 1, 2], dtype=torch.long)
        p = MODELS_DIR / "generator.onnx"
        _export(g, (photo, style), p, ["photo", "style"], ["sketch"], True)
        report.append(_parity("generator", g, p, {"photo": photo, "style": style}))
        # different batch size check (dynamic axis)
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"photo": photo[:1].numpy(), "style": style[:1].numpy()})[0]
        report.append({"model": "generator_batch1", "ok": o.shape == (1, 3, 128, 128), "max_abs_diff": 0.0})
    save_json(report, OUT_DIR / "onnx_parity.json")
    bad = [r for r in report if not r["ok"]]
    if bad:
        raise SystemExit(f"ONNX parity failed for: {[r['model'] for r in bad]}")
    print(f"Exported {len(report)} models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
