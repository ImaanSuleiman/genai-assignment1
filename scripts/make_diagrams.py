"""Architecture diagrams for the report (matplotlib only). Writes report/figures/arch_*.png.

    python scripts/make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"io": "#e2e8f0", "enc": "#c7d2fe", "lat": "#fde68a", "dec": "#bbf7d0", "gate": "#fbcfe8", "exp": "#bae6fd", "net": "#ddd6fe", "loss": "#fecaca"}


def box(ax, x, y, w, h, text, kind="io", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=COL[kind], ec="#334155", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def arrow(ax, a, b, side="r", text=None, rad=0.0, fs=6):
    ax_, ay, aw, ah = a
    bx, by, bw, bh = b
    p0 = {"r": (ax_ + aw, ay + ah / 2), "b": (ax_ + aw / 2, ay), "t": (ax_ + aw / 2, ay + ah)}[side]
    p1 = {"r": (bx, by + bh / 2), "b": (bx + bw / 2, by + bh), "t": (bx + bw / 2, by)}[side]
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#334155", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.12, text, ha="center", fontsize=fs, color="#475569")


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold")
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def udae():
    fig, ax = canvas(7.2, 2.6, "Task 1: universal denoising autoencoder (UDAE)")
    items = [("Corrupted\n$\\tilde{x}$\n3x128x128", "io"), ("Enc 1\n32 ch\n64x64", "enc"), ("Enc 2\n64 ch\n32x32", "enc"), ("Enc 3\n128 ch\n16x16", "enc"),
             ("Enc 4\n256 ch\n8x8", "enc"), ("Latent z\n1x1 conv\nC x 8x8", "lat"), ("Dec 1\n128 ch\n16x16", "dec"), ("Dec 2\n64 ch\n32x32", "dec"),
             ("Dec 3\n32 ch\n64x64", "dec"), ("Dec 4+out\n3x128x128\nsigmoid", "dec")]
    prev, x = None, 0.1
    for t, k in items:
        b = box(ax, x, 0.9, 0.62, 0.95, t, k, fs=5.5)
        if prev:
            arrow(ax, prev, b)
        prev, x = b, x + 0.71
    ax.text(3.6, 2.2, "stride-2 conv blocks (Conv-BN-ReLU x2, Dropout2d) | nearest upsampling + conv blocks | NO skip connections", ha="center", fontsize=6.5)
    ax.text(3.6, 0.45, "$\\mathcal{L}=\\alpha\\,\\mathcal{L}_{L1}(x,\\hat{x})+(1-\\alpha)(1-\\mathrm{SSIM}(x,\\hat{x}))$   |   latent size = C x 64 values (e.g. 1024 for C=16, 48x compression)",
            ha="center", fontsize=6.5)
    save(fig, "arch_udae.png")


def hard():
    fig, ax = canvas(7.2, 3.0, "Task 2: classifier + hard-routed specialists")
    inp = box(ax, 0.1, 1.2, 0.9, 0.7, "Input $\\tilde{x}$", "io")
    clf = box(ax, 1.4, 1.2, 1.3, 0.7, "CNN classifier $C$\n$p=[p_c,p_s,p_b,p_o]$", "gate")
    arrow(ax, inp, clf)
    rt = box(ax, 3.1, 1.2, 1.0, 0.7, "$r=\\arg\\max p$\n(router)", "net")
    arrow(ax, clf, rt)
    outs = [(2.25, "Identity bypass ($r$ = clean)", "io"), (1.5, "$A_{salt}$ (salt-and-pepper)", "exp"), (0.75, "$A_{blur}$ (Gaussian blur)", "exp"), (0.0, "$A_{occ}$ (occlusion)", "exp")]
    bs = []
    for y, t, k in outs:
        b = box(ax, 4.6, y + 0.05, 1.6, 0.5, t, k, fs=6)
        arrow(ax, rt, b, rad=0.0)
        bs.append(b)
    out = box(ax, 6.45, 1.2, 0.65, 0.7, "$\\hat{x}$", "dec")
    for b in bs:
        arrow(ax, b, out)
    ax.text(3.6, 2.95 - 0.1, "oracle mode: true label from the test manifest replaces $r$", ha="center", fontsize=6.5)
    save(fig, "arch_hard.png")


def soft():
    fig, ax = canvas(7.2, 3.7, "Task 3: jointly trained soft mixture-of-experts")
    inp = box(ax, 0.1, 1.7, 0.8, 0.7, "Input $\\tilde{x}$", "io")
    gate = box(ax, 1.3, 2.75, 1.9, 0.6, "Gate $G$ (init. from Task 2 classifier)\nlogits / $\\tau$ -> softmax", "gate", fs=6)
    arrow(ax, inp, gate, side="t", rad=0.25)
    ex = []
    for y, t, k in ((2.05, "identity $\\tilde{x}$", "io"), (1.45, "$A_{salt}$ (init. Task 2)", "exp"), (0.85, "$A_{blur}$", "exp"), (0.25, "$A_{occ}$", "exp")):
        b = box(ax, 3.4, y, 1.5, 0.45, t, k, fs=6)
        arrow(ax, inp, b, rad=0.0)
        ex.append(b)
    mix = box(ax, 5.4, 1.5, 1.0, 0.8, "$\\hat{x}=\\sum_k w_k\\,y_k$", "net")
    for i, b in enumerate(ex):
        arrow(ax, b, mix, text=f"$w_{i}$")
    arrow(ax, gate, mix, side="r", rad=-0.1)
    out = box(ax, 6.6, 1.6, 0.55, 0.6, "$\\hat{x}$", "dec")
    arrow(ax, mix, out)
    ax.text(3.6, 3.55, "$\\mathcal{L}=\\lambda_1 L_1+\\lambda_s(1-\\mathrm{SSIM})+\\lambda_c CE+\\lambda_b\\sum_k(\\bar w_k-\\frac{1}{4})^2$ ; warm-up: gate only, then joint fine-tuning", ha="center", fontsize=6.3)
    save(fig, "arch_soft.png")


def cgan():
    fig, ax = canvas(7.2, 3.6, "Task 4: style-conditioned cGAN (U-Net generator + PatchGAN discriminator)")
    photo = box(ax, 0.1, 2.3, 0.9, 0.6, "Photo $x$", "io")
    style = box(ax, 0.1, 1.3, 0.9, 0.6, "Style $s$\nEmbedding", "lat")
    g = box(ax, 1.5, 1.6, 2.0, 1.4, "Generator $G(x,s)$\nU-Net: 6 down / 6 up\nskips + FiLM(style)\nstyle also as input channels", "enc", fs=6)
    arrow(ax, photo, g)
    arrow(ax, style, g)
    fake = box(ax, 3.9, 2.4, 0.9, 0.55, "Generated $\\hat{y}$", "dec")
    arrow(ax, g, fake)
    real = box(ax, 3.9, 0.55, 0.9, 0.55, "Real sketch $y$", "io")
    d = box(ax, 5.2, 1.4, 1.5, 1.2, "PatchDiscriminator\n$D(x,\\cdot,s)$\nstyle embedding\nas input channels", "gate", fs=6)
    arrow(ax, fake, d)
    arrow(ax, real, d)
    ax.text(3.6, 3.45, "$\\mathcal{L}_G=\\mathcal{L}_{adv}+\\lambda_{L1}\\|y-G(x,s)\\|_1$ ;  $\\mathcal{L}_D=\\frac{1}{2}(BCE(D(x,y,s),1)+BCE(D(x,\\hat{y},s),0))$", ha="center", fontsize=6.3)
    ax.text(6.0, 0.9, "patch logits (14x14)", ha="center", fontsize=6)
    save(fig, "arch_cgan.png")


def app():
    fig, ax = canvas(7.2, 2.4, "Application architecture (Docker Compose)")
    br = box(ax, 0.1, 0.9, 1.3, 0.9, "Browser\nReact + Tailwind\n(designed in Stitch)", "io", fs=6)
    ng = box(ax, 1.9, 0.9, 1.3, 0.9, "nginx container\nstatic bundle\n/api proxy", "net", fs=6)
    api = box(ax, 3.7, 0.9, 1.5, 0.9, "FastAPI container\nvalidate, preprocess,\ncorruptions, timing", "enc", fs=6)
    onx = box(ax, 5.7, 0.9, 1.4, 0.9, "ONNX Runtime\n7 .onnx models\n(volume ./models)", "exp", fs=6)
    arrow(ax, br, ng, text=":8080")
    arrow(ax, ng, api, text="/api")
    arrow(ax, api, onx)
    ax.text(3.6, 0.35, "endpoints: /api/health | /api/universal | /api/hard-route | /api/soft-moe | /api/face2sketch", ha="center", fontsize=6.5)
    save(fig, "arch_app.png")


if __name__ == "__main__":
    for f in (udae, hard, soft, cgan, app):
        f()
    print("diagrams written to", OUT)
