"""Fast checks of the assignment's hard requirements. Run:  PYTHONPATH=src python -m pytest -q tests"""
import numpy as np
import pytest
import torch

from genai import corruptions as C
from genai.data_pets import BalancedBatchSampler, RestorationDataset
from genai.losses import ssim
from genai.models import CorruptionClassifier, PatchDiscriminator, SoftMoE, UDAE, UNetGenerator

IMG = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)


def test_salt_and_pepper_probability_and_values():
    p = {"type": "salt", "p": 0.1, "seed": 1}
    out = C.apply(IMG, p)
    changed = (out != IMG).any(-1)
    assert 0.07 < changed.mean() < 0.11            # ~10% of pixels (some already black/white)
    vals = out[(out != IMG).any(-1)]
    assert set(np.unique(vals)) <= {0, 255}


def test_blur_matches_opencv_and_is_deterministic():
    p = {"type": "blur", "k": 5, "sigma": 1.5}
    assert np.array_equal(C.apply(IMG, p), C.apply(IMG, p))
    assert (C.apply(IMG, p).astype(int) - IMG).std() > 5


@pytest.mark.parametrize("n,frac", [(1, 0.10), (2, 0.20), (3, 0.35)])
def test_fixed_occlusion_levels(n, frac):
    p = C.fixed_params("occlusion", [0.10, 0.20, 0.35].index(frac), np.random.default_rng(3))
    assert p["n"] == n and abs(p["coverage"] - frac) < 0.02
    out = C.apply(IMG, p)
    assert abs((out == 0).all(-1).mean() - p["coverage"]) < 0.01


def test_training_ranges():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = C.sample_params("salt", rng); assert 0.02 <= s["p"] <= 0.15
        b = C.sample_params("blur", rng); assert b["k"] in (3, 5, 7) and 0.5 <= b["sigma"] <= 2.5
        o = C.sample_params("occlusion", rng); assert 1 <= o["n"] <= 3 and 0.08 <= o["coverage"] <= 0.37


def test_manifests_are_deterministic_and_complete():
    a, b = C.make_test_manifest(5), C.make_test_manifest(5)
    assert a == b and len(a) == 50                 # 10 entries per image
    levels = {(m["corruption"], m["level"]) for m in a}
    assert len(levels) == 10
    assert [m["params"]["p"] for m in a if m["corruption"] == "salt" and m["index"] == 0] == [0.03, 0.08, 0.15]
    assert C.make_val_manifest(40) == C.make_val_manifest(40)
    counts = np.bincount([["clean", "salt", "blur", "occlusion"].index(m["corruption"]) for m in C.make_val_manifest(40)])
    assert counts.tolist() == [10, 10, 10, 10]


def test_runtime_corruption_changes_every_load():
    ds = RestorationDataset(np.stack([IMG] * 4), train=False)
    types = {tuple(ds[0][0].flatten()[:50].tolist()) for _ in range(30)}
    assert len(types) > 5                          # new type/severity sampled per load


def test_balanced_sampler():
    s = BalancedBatchSampler(100, 32)
    batch = next(iter(s))
    assert np.bincount([c for _, c in batch], minlength=4).tolist() == [8, 8, 8, 8]


def test_udae_has_real_bottleneck_and_no_skip():
    m = UDAE(32, 16, 0.1)
    x = torch.rand(2, 3, 128, 128)
    z, feats = m.encode(x)
    assert z.numel() // 2 == 16 * 8 * 8 < 3 * 128 * 128 / 40
    assert m(x).shape == x.shape and m.skip_proj is None


def test_soft_moe_weights_sum_to_one_and_reconstruction_is_weighted_sum():
    gate = CorruptionClassifier((16, 32), 0.0)
    ex = [UDAE(16, 8, 0.0) for _ in range(3)]
    m = SoftMoE(gate, ex, tau=0.7).eval()
    x = torch.rand(2, 3, 128, 128)
    y, w, _ = m(x)
    assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)
    manual = w[:, :1, None, None] * x + sum(w[:, k + 1, None, None, None] * ex[k].eval()(x) for k in range(3))
    assert torch.allclose(y, manual, atol=1e-5)


def test_gan_shapes_and_style_is_used_by_G_and_D():
    G, D = UNetGenerator(16, style_dim=8, dropout=0.0).eval(), PatchDiscriminator(16, style_dim=8).eval()
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    # make the (zero-initialised) FiLM layers non-trivial so the effect of the style is measurable
    for m in G.modules():
        if hasattr(m, "fc") and hasattr(m, "norm"):
            torch.nn.init.normal_(m.fc.weight, 0, 0.1)
    a, b = G(x, torch.tensor([0, 0])), G(x, torch.tensor([1, 1]))
    assert a.shape == x.shape and (a - b).abs().mean() > 1e-4
    assert D(x, a, torch.tensor([0, 0])).shape[1:] == (1, 14, 14)
    assert (D(x, a, torch.tensor([0, 0])) - D(x, a, torch.tensor([2, 2]))).abs().mean() > 1e-6


def test_ssim_identity_and_range():
    x = torch.rand(2, 3, 64, 64)
    assert abs(ssim(x, x).item() - 1) < 1e-4
    assert ssim(x, torch.rand_like(x)).item() < 0.5
