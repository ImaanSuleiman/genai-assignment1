"""Final evaluation on the held-out TEST data (never used for training or tuning).

    python -m genai.evaluate --task all
    python -m genai.evaluate --task 1 --limit 200      # quick subset

Writes outputs/results/*.json and outputs/figures/*.png (used by the LaTeX report).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import corruptions as C
from . import data_fs2k as df
from . import data_pets as dp
from .common import CKPT_DIR, CLASS_TITLES, CLASSES, OUT_DIR, get_device, load_json, mkdirs, save_json, worker_count
from .losses import psnr, ssim
from .task1 import load_udae
from .task2 import SPECIALISTS, classification_report, load_classifier, load_specialist
from .task3 import load_moe
from .task4 import load_generator, to01

RES, FIG = OUT_DIR / "results", OUT_DIR / "figures"


# ============================================================================ helpers
def metrics_batch(out, clean):
    return (psnr(out, clean).cpu().numpy(), ssim(out, clean, per_image=True).cpu().numpy(),
            F.l1_loss(out, clean, reduction="none").flatten(1).mean(1).cpu().numpy())


@torch.no_grad()
def run_manifest(fn, ds, device, bs=128, workers=2):
    """fn(noisy)->(output, extra dict of per-sample tensors). Returns per-entry arrays aligned with the manifest."""
    loader = dp.make_loader(ds, bs, workers=workers)
    acc = defaultdict(list)
    for xn, xc, y, _ in loader:
        xn, xc = xn.to(device), xc.to(device)
        out, extra = fn(xn)
        p, s, l = metrics_batch(out, xc)
        pi, si, li = metrics_batch(xn, xc)
        for k, v in dict(psnr=p, ssim=s, l1=l, in_psnr=pi, in_ssim=si, in_l1=li).items():
            acc[k].append(v)
        for k, v in extra.items():
            acc[k].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def aggregate(res, manifest, keys=("psnr", "ssim", "l1", "in_psnr", "in_ssim", "in_l1")):
    """Mean metrics per corruption type and per (type, severity level)."""
    by_type, by_level = defaultdict(list), defaultdict(list)
    for i, m in enumerate(manifest):
        by_type[m["corruption"]].append(i)
        by_level[(m["corruption"], m["level"] or "-")].append(i)
    f = lambda idx: {k: float(np.mean(res[k][idx])) for k in keys} | {"n": len(idx)}
    allcorr = [i for i, m in enumerate(manifest) if m["corruption"] != "clean"]
    return {"by_type": {c: f(v) for c, v in by_type.items()},
            "by_level": {f"{c}/{l}": f(v) for (c, l), v in by_level.items()},
            "all_corrupted": f(allcorr), "overall": f(list(range(len(manifest))))}


def fmt_w(w):
    return '[' + ' '.join(f'{float(v):.2f}' for v in w) + ']'


def to_img(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def example_figure(rows, path, title=None):
    """rows: list of dict(clean, noisy, out, caption). Columns: clean target | input | output | |error|.
    More than 6 rows are laid out as two side-by-side blocks to keep the figure compact."""
    n = len(rows)
    blocks = 2 if n > 6 else 1
    per = int(np.ceil(n / blocks))
    fig, axes = plt.subplots(per, 4 * blocks, figsize=(7.2 * blocks, 1.9 * per), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for k, row in enumerate(rows):
        r, b = k % per, k // per
        err = (row["out"] - row["clean"]).abs().mean(0).cpu().numpy()
        ims = [to_img(row["clean"]), to_img(row["noisy"]), to_img(row["out"]), err]
        for c, im in enumerate(ims):
            ax = axes[r, 4 * b + c]
            ax.imshow(im, vmin=0, vmax=0.3, cmap="inferno") if c == 3 else ax.imshow(im)
            if r == 0:
                ax.set_title(["Clean target", "Input", "Output", "Abs. error"][c], fontsize=8)
        axes[r, 4 * b].text(-0.04, 0.5, row["caption"], transform=axes[r, 4 * b].transAxes, ha="right", va="center", fontsize=6)
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    mkdirs(path.parent)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pick_examples(res, manifest, k=12):
    """k representative examples: spread over the 3 corruption types x 3 levels (+ clean) by median-quality entries."""
    groups = [("clean", None)] + [(c, l) for c in CLASSES[1:] for l in C.LEVEL_NAMES]
    chosen = []
    for g in groups:
        idx = [i for i, m in enumerate(manifest) if (m["corruption"], m["level"]) == g]
        idx = sorted(idx, key=lambda i: res["psnr"][i])
        chosen.append(idx[len(idx) // 2])
    return chosen[:k] + [i for i in range(len(manifest)) if i not in chosen][: max(0, k - len(chosen))]


def failure_cases(res, manifest, k=4):
    """Worst PSNR among corrupted inputs, at most 2 per corruption type so the failures are diverse."""
    order = np.argsort(res["psnr"])
    cnt, out = defaultdict(int), []
    for i in order:
        c = manifest[i]["corruption"]
        if c == "clean" or cnt[c] >= 2:
            continue
        cnt[c] += 1
        out.append(int(i))
        if len(out) == k:
            break
    return out


def history_plot(hist, keys, path, title, ylabel=""):
    fig, ax = plt.subplots(figsize=(5, 3))
    for k, lab in keys:
        ax.plot([h["epoch"] for h in hist], [h[k] for h in hist], label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    mkdirs(path.parent)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def show_rows(ds, idxs, res, caption_fn, out_fn, device):
    rows = []
    for i in idxs:
        xn, xc, _, _ = ds[i]
        with torch.no_grad():
            out = out_fn(xn[None].to(device))[0].cpu()
        rows.append({"clean": xc, "noisy": xn, "out": out, "caption": caption_fn(i)})
    return rows


def caption(manifest, res, i):
    m = manifest[i]
    lv = f"/{m['level']}" if m["level"] else ""
    return f"{CLASS_TITLES[m['corruption']]}{lv}\nPSNR {res['psnr'][i]:.1f} dB"


# ============================================================================ Task 1
def eval_task1(data, device, workers, limit):
    model = load_udae(CKPT_DIR / "task1_udae.pt", device)
    manifest = data.test_manifest if not limit else [m for m in data.test_manifest if m["index"] < limit]
    ds = dp.RestorationDataset(data.test, manifest=manifest)
    res = run_manifest(lambda x: (model(x), {}), ds, device, workers=workers)
    agg = aggregate(res, manifest)
    save_json(agg, RES / "task1_test.json")
    np.savez(RES / "task1_per_sample.npz", **res)
    ex = pick_examples(res, manifest, 12)
    example_figure(show_rows(ds, ex, res, lambda i: caption(manifest, res, i), lambda x: model(x), device),
                   FIG / "task1_examples.png", "Task 1: universal denoising autoencoder (12 test examples)")
    fc = failure_cases(res, manifest, 4)
    example_figure(show_rows(ds, fc, res, lambda i: caption(manifest, res, i), lambda x: model(x), device),
                   FIG / "task1_failures.png", "Task 1: failure cases (lowest PSNR)")
    save_json({"failure_indices": fc, "failures": [manifest[i] | {"psnr": float(res["psnr"][i])} for i in fc]}, RES / "task1_failures.json")
    f = OUT_DIR / "task1_final.json"
    if f.exists():
        h = load_json(f)["history"]
        history_plot(h, [("train_loss", "train loss"), ("val_obj", "val L1+(1-SSIM)")], FIG / "task1_curves.png", "Task 1 training")
    print("Task 1:", {k: round(v["psnr"], 2) for k, v in agg["by_type"].items()})
    return agg


# ============================================================================ Task 2
def eval_task2(data, device, workers, limit):
    clf = load_classifier(device=device)
    specs = [load_specialist(c, device) for c in SPECIALISTS]
    manifest = data.test_manifest if not limit else [m for m in data.test_manifest if m["index"] < limit]
    ds = dp.RestorationDataset(data.test, manifest=manifest)

    @torch.no_grad()
    def both_modes(x):
        probs = F.softmax(clf(x), 1)
        pred = probs.argmax(1)
        outs = torch.stack([x] + [s(x) for s in specs], 1)  # clean = identity bypass (never processed by an expert)
        return outs, probs, pred

    # classifier metrics + two routing modes in one pass
    loader = dp.make_loader(ds, 128, workers=workers)
    acc = defaultdict(list)
    for xn, xc, y, _ in loader:
        xn, xc, y = xn.to(device), xc.to(device), y.to(device)
        outs, probs, pred = both_modes(xn)
        ar = torch.arange(len(y), device=device)
        o_or, o_pr = outs[ar, y], outs[ar, pred]
        for name, o in (("oracle", o_or), ("pred", o_pr)):
            p, s, l = metrics_batch(o, xc)
            acc[f"{name}_psnr"].append(p), acc[f"{name}_ssim"].append(s), acc[f"{name}_l1"].append(l)
        pi, si, li = metrics_batch(xn, xc)
        acc["in_psnr"].append(pi), acc["in_ssim"].append(si), acc["in_l1"].append(li)
        acc["label"].append(y.cpu().numpy()), acc["pred"].append(pred.cpu().numpy()), acc["probs"].append(probs.cpu().numpy())
    res = {k: np.concatenate(v) for k, v in acc.items()}
    rep = classification_report(res["label"], res["pred"])
    out = {"classifier_test": rep}
    for mode in ("oracle", "pred"):
        view = {"psnr": res[f"{mode}_psnr"], "ssim": res[f"{mode}_ssim"], "l1": res[f"{mode}_l1"],
                "in_psnr": res["in_psnr"], "in_ssim": res["in_ssim"], "in_l1": res["in_l1"]}
        out[f"routing_{mode}"] = aggregate(view, manifest)
    wrong = np.where(res["label"] != res["pred"])[0]
    delta = res["oracle_psnr"] - res["pred_psnr"]
    worst = wrong[np.argsort(-delta[wrong])][:4] if len(wrong) else []
    out["misrouted"] = {"count": int(len(wrong)), "fraction": float(len(wrong) / len(res["label"])),
                        "mean_psnr_drop_when_misrouted": float(delta[wrong].mean()) if len(wrong) else 0.0,
                        "worst_cases": [{"index": int(i), "true": CLASSES[res["label"][i]], "pred": CLASSES[res["pred"][i]],
                                         "level": manifest[i]["level"], "psnr_oracle": float(res["oracle_psnr"][i]),
                                         "psnr_pred": float(res["pred_psnr"][i])} for i in worst]}
    save_json(out, RES / "task2_test.json")
    # confusion matrix figure (normalized)
    cm = np.array(rep["confusion_matrix_normalized"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4)), ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=30), ax.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title("Normalized confusion matrix (test)", fontsize=10)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    mkdirs(FIG)
    plt.savefig(FIG / "task2_confusion.png", dpi=150)
    plt.close(fig)
    # misrouting failure figure (predicted routing output)
    if len(worst):
        def routed(x):
            o, _, p = both_modes(x)
            return o[torch.arange(len(x)), p]
        rows = show_rows(ds, [int(i) for i in worst], {"psnr": res["pred_psnr"]},
                         lambda i: f"{CLASS_TITLES[CLASSES[res['label'][i]]]} -> {CLASSES[res['pred'][i]]}\nPSNR {res['pred_psnr'][i]:.1f} dB",
                         routed, device)
        example_figure(rows, FIG / "task2_misrouted.png", "Task 2: classifier errors that break restoration")
    f = OUT_DIR / "task2_classifier_final.json"
    if f.exists():
        history_plot(load_json(f)["history"], [("train_loss", "train loss"), ("val_acc", "val accuracy"), ("val_macro_f1", "val macro-F1")],
                     FIG / "task2_classifier_curves.png", "Task 2 classifier training")
    f = OUT_DIR / "task2_specialists_final.json"
    if f.exists():
        fig, ax = plt.subplots(figsize=(5, 3))
        for c, d in load_json(f)["specialists"].items():
            ax.plot([h["epoch"] for h in d["history"]], [h["val_obj"] for h in d["history"]], label=c)
        ax.set_xlabel("epoch"), ax.set_ylabel("val L1+(1-SSIM)"), ax.legend(), ax.grid(alpha=0.3)
        ax.set_title("Task 2 specialist training", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / "task2_specialist_curves.png", dpi=150)
        plt.close(fig)
    print("Task 2: acc", round(rep["accuracy"], 4), "oracle", round(out["routing_oracle"]["all_corrupted"]["psnr"], 2),
          "pred", round(out["routing_pred"]["all_corrupted"]["psnr"], 2))
    return out


# ============================================================================ Task 3
def mixed_manifest(n, seed=99):
    """Stress test: TWO corruptions on the same image (blur then salt-and-pepper, or occlusion then blur)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        combo = ["blur", "salt"] if i % 2 == 0 else ["occlusion", "blur"]
        out.append({"index": i, "corruption": "+".join(combo), "level": None,
                    "params": [C.fixed_params(c, 1, rng) for c in combo]})
    return out


class MixedDS(torch.utils.data.Dataset):
    def __init__(self, images, manifest):
        self.images, self.manifest = images, manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        m = self.manifest[i]
        clean = self.images[m["index"]]
        x = clean
        for p in m["params"]:
            x = C.apply(x, p)
        return dp.to_tensor(x), dp.to_tensor(clean), 0, m["index"]


def eval_task3(data, device, workers, limit, t1=None, t2=None):
    moe = load_moe(device=device)
    udae = load_udae(CKPT_DIR / "task1_udae.pt", device)
    clf = load_classifier(device=device)
    specs = [load_specialist(c, device) for c in SPECIALISTS]
    manifest = data.test_manifest if not limit else [m for m in data.test_manifest if m["index"] < limit]
    ds = dp.RestorationDataset(data.test, manifest=manifest)

    def fn(x):
        out, w, _ = moe(x)
        return out, {"w": w}

    res = run_manifest(fn, ds, device, workers=workers)
    agg = aggregate(res, manifest)
    # routing statistics
    W = res["w"]
    wt = defaultdict(list)
    for i, m in enumerate(manifest):
        wt[(m["corruption"], m["level"] or "-")].append(W[i])
    table = {f"{c}/{l}": np.mean(v, 0).tolist() for (c, l), v in wt.items()}
    branch = ["identity", "salt-expert", "blur-expert", "occlusion-expert"]
    health = {"mean_weight_per_branch": W.mean(0).tolist(),
              "frac_images_branch_dominant_gt_0.5": (W > 0.5).mean(0).tolist(),
              "argmax_share": (np.bincount(W.argmax(1), minlength=4) / len(W)).tolist(),
              "inactive_branches": [branch[k] for k in range(4) if W[:, k].max() < 0.1],
              "branch_names": branch}
    # does an expert dominate unrelated inputs?  (mean weight of branch k on inputs of a different type)
    unrelated = {}
    for k, c in enumerate(CLASSES):
        other = [i for i, m in enumerate(manifest) if m["corruption"] != c]
        unrelated[branch[k]] = float(W[other, k].mean())
    health["mean_weight_on_other_types"] = unrelated
    # two-corruption stress test: universal AE vs hard (predicted) vs soft MoE
    n_mix = min(len(data.test), 300)
    mm = mixed_manifest(n_mix)
    mds = MixedDS(data.test, mm)
    ml = dp.make_loader(mds, 64, workers=workers)
    macc = defaultdict(list)
    with torch.no_grad():
        for xn, xc, _, _ in ml:
            xn, xc = xn.to(device), xc.to(device)
            outs = torch.stack([xn] + [s(xn) for s in specs], 1)
            pred = F.softmax(clf(xn), 1).argmax(1)
            hard = outs[torch.arange(len(xn)), pred]
            for name, o in (("input", xn), ("universal_ae", udae(xn)), ("hard_routed_pred", hard), ("soft_moe", moe(xn)[0])):
                p, s, l = metrics_batch(o, xc)
                macc[name + "_psnr"].append(p), macc[name + "_ssim"].append(s)
    mixed = {k: float(np.concatenate(v).mean()) for k, v in macc.items()}
    save_json({**agg, "routing_weights_by_condition": table, "routing_health": health, "mixed_corruption_test": mixed},
              RES / "task3_test.json")
    # routing heatmap
    keys = list(table)
    M = np.array([table[k] for k in keys])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(keys))), ax.set_yticklabels(keys, fontsize=7)
    ax.set_xticks(range(4)), ax.set_xticklabels(["identity", "salt", "blur", "occl."], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < 0.6 else "black")
    ax.set_title("Mean routing weights per test condition", fontsize=10)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "task3_routing_heatmap.png", dpi=150)
    plt.close(fig)
    # weight distribution (box plot of each branch weight for each true type)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
    for k, c in enumerate(CLASSES):
        idx = [i for i, m in enumerate(manifest) if m["corruption"] == c]
        axes[k].boxplot([W[idx, j] for j in range(4)], showfliers=False)
        axes[k].set_xticks([1, 2, 3, 4]), axes[k].set_xticklabels(["id", "S", "B", "O"])
        axes[k].set_title(f"true: {c}", fontsize=9)
    axes[0].set_ylabel("gate weight")
    plt.tight_layout()
    plt.savefig(FIG / "task3_weight_distribution.png", dpi=150)
    plt.close(fig)
    # examples: dominant expert vs distributed weights
    dom = int(np.argmax(W.max(1) * [m["corruption"] != "clean" for m in manifest]))
    dist = int(np.argmin(np.where([m["corruption"] != "clean" for m in manifest], W.max(1), 9)))
    ex = pick_examples(res, manifest, 12)
    rows = show_rows(ds, ex, res, lambda i: caption(manifest, res, i) + "\nw=" + fmt_w(W[i]),
                     lambda x: moe(x)[0], device)
    example_figure(rows, FIG / "task3_examples.png", "Task 3: soft mixture-of-experts (12 test examples)")
    rows = show_rows(ds, [dom, dist], res, lambda i: caption(manifest, res, i) + "\nw=" + fmt_w(W[i]),
                     lambda x: moe(x)[0], device)
    example_figure(rows, FIG / "task3_dominant_vs_distributed.png", "Top: one expert dominates. Bottom: weights distributed")
    fc = failure_cases(res, manifest, 4)
    example_figure(show_rows(ds, fc, res, lambda i: caption(manifest, res, i) + "\nw=" + fmt_w(W[i]),
                             lambda x: moe(x)[0], device), FIG / "task3_failures.png", "Task 3: failure cases")
    # mixed corruption examples
    rows = []
    for i in range(4):
        xn, xc, _, _ = mds[i]
        with torch.no_grad():
            o, w, _ = moe(xn[None].to(device))
        rows.append({"clean": xc, "noisy": xn, "out": o[0].cpu(), "caption": f"{mm[i]['corruption']}\nw={fmt_w(w[0].cpu().numpy())}"})
    example_figure(rows, FIG / "task3_mixed_corruption.png", "Two simultaneous corruptions")
    f = OUT_DIR / "task3_final.json"
    if f.exists():
        h = load_json(f)["history"]
        history_plot(h, [("train_loss", "train loss"), ("val_obj", "val L1+(1-SSIM)")], FIG / "task3_curves.png", "Task 3 training")
        history_plot(h, [(f"w_{c}", f"mean w ({c})") for c in CLASSES], FIG / "task3_weight_evolution.png", "Mean gate weight during training")
    print("Task 3:", {k: round(v["psnr"], 2) for k, v in agg["by_type"].items()}, "mixed", {k: round(v, 2) for k, v in mixed.items() if "psnr" in k})
    return agg


# ============================================================================ Task 4
@torch.no_grad()
def eval_task4(device, workers, limit):
    G = load_generator(device=device)
    P, S, ST = df.load_split("test")
    if limit:
        P, S, ST = P[:limit], S[:limit], ST[:limit]
    ds = df.PairedSketchDataset(P, S, ST)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)
    own, mism, base = defaultdict(list), [], defaultdict(list)
    for p, s, st in loader:
        p, s, st = p.to(device), s.to(device), st.to(device)
        fake = G(p, st)
        a, c = to01(fake), to01(s)
        for k, v in zip(("psnr", "ssim", "l1"), (psnr(a, c), ssim(a, c, per_image=True), F.l1_loss(a, c, reduction="none").flatten(1).mean(1))):
            own[k].append(v.cpu().numpy())
        own["style"].append(st.cpu().numpy())
        wrong = (st + 1 + torch.randint(0, 2, st.shape, device=st.device)) % 3  # a different style
        mism.append(F.l1_loss(to01(G(p, wrong)), c, reduction="none").flatten(1).mean(1).cpu().numpy())
        gray = to01(p).mean(1, keepdim=True).expand(-1, 3, -1, -1)  # trivial baseline: grayscale photo
        for k, v in zip(("psnr", "ssim", "l1"), (psnr(gray, c), ssim(gray, c, per_image=True), F.l1_loss(gray, c, reduction="none").flatten(1).mean(1))):
            base[k].append(v.cpu().numpy())
    own = {k: np.concatenate(v) for k, v in own.items()}
    base = {k: np.concatenate(v) for k, v in base.items()}
    mism = np.concatenate(mism)
    out = {"overall": {k: float(own[k].mean()) for k in ("psnr", "ssim", "l1")},
           "per_style": {f"Style {s + 1}": {k: float(own[k][own["style"] == s].mean()) for k in ("psnr", "ssim", "l1")} | {"n": int((own["style"] == s).sum())}
                         for s in range(3) if (own["style"] == s).any()},
           "baseline_grayscale_photo": {k: float(base[k].mean()) for k in ("psnr", "ssim", "l1")},
           "style_conditioning_check": {"l1_matched_style": float(own["l1"].mean()), "l1_mismatched_style": float(mism.mean())},
           "n_test": int(len(P))}
    save_json(out, RES / "task4_test.json")
    # figure: photo | GT sketch | own style | style 1 | style 2 | style 3
    pick = list(np.linspace(0, len(P) - 1, min(8, len(P))).astype(int))
    worst = list(np.argsort(-own["l1"])[:4])
    for name, idxs, ttl in (("task4_examples.png", pick, "Task 4: test photos, ground-truth sketch and generated sketches"),
                            ("task4_failures.png", worst, "Task 4: failure cases (highest L1)")):
        fig, axes = plt.subplots(len(idxs), 5, figsize=(8, 1.7 * len(idxs)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(idxs):
            p, s, st = ds[int(i)]
            gens = [to01(G(p[None].to(device), torch.tensor([k], device=device))[0]).cpu() for k in range(3)]
            ims = [to01(p), to01(s)] + gens
            for c, im in enumerate(ims):
                axes[r, c].imshow(im.permute(1, 2, 0).numpy())
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(["Photo", f"GT (style)", "Gen: Style 1", "Gen: Style 2", "Gen: Style 3"][c], fontsize=8)
            axes[r, 1].text(0.5, -0.12, f"GT = Style {int(st) + 1}", transform=axes[r, 1].transAxes, ha="center", fontsize=6)
        fig.suptitle(ttl, fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / name, dpi=140, bbox_inches="tight")
        plt.close(fig)
    f = OUT_DIR / "task4_final.json"
    if f.exists():
        h = load_json(f)["history"]
        history_plot(h, [("d_real", "D real loss"), ("d_fake", "D fake loss"), ("g_adv", "G adversarial")], FIG / "task4_gan_losses.png", "Task 4 adversarial losses")
        history_plot(h, [("g_l1", "G reconstruction L1 (train)"), ("val_l1", "val L1")], FIG / "task4_l1_curves.png", "Task 4 L1")
    # development of the generator over time: stack the fixed-photo grids logged during training
    sd = OUT_DIR / "task4_samples"
    shots = sorted(sd.glob("epoch_*.png")) if sd.exists() else []
    if len(shots) >= 2:
        from PIL import Image
        pick_s = sorted({shots[round(i * (len(shots) - 1) / 3)] for i in range(4)})
        ims = [Image.open(q).convert("RGB") for q in pick_s]
        w = max(i.width for i in ims)
        canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
        y = 0
        for q, im in zip(pick_s, ims):
            canvas.paste(im, (0, y))
            y += im.height
        canvas.save(FIG / "task4_evolution.png")
        save_json({"epochs_shown": [q.stem for q in pick_s]}, RES / "task4_evolution.json")
    print("Task 4:", out["overall"], "baseline", out["baseline_grayscale_photo"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["1", "2", "3", "4", "all"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N test images")
    ap.add_argument("--workers", type=int, default=worker_count(2))
    a = ap.parse_args(argv)
    device = get_device()
    mkdirs(RES, FIG)
    need_pets = a.task in ("1", "2", "3", "all")
    data = dp.load_pets(load_test=True) if need_pets else None
    if a.task in ("1", "all"):
        eval_task1(data, device, a.workers, a.limit)
    if a.task in ("2", "all"):
        eval_task2(data, device, a.workers, a.limit)
    if a.task in ("3", "all"):
        eval_task3(data, device, a.workers, a.limit)
    if a.task in ("4", "all"):
        eval_task4(device, a.workers, a.limit)


if __name__ == "__main__":
    main()
