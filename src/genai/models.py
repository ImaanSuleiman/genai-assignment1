"""All network architectures.

Task 1/2  UDAE                     conv encoder -> compressed latent -> conv decoder (specialists reuse it)
Task 2    CorruptionClassifier     4-class CNN (clean / salt / blur / occlusion)
Task 3    SoftMoE                  gate (classifier) + identity + 3 specialist autoencoders
Task 4    UNetGenerator, PatchDiscriminator   style-conditioned pix2pix-style cGAN
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


# =============================================================================== Task 1 / 2
class UDAE(nn.Module):
    """Universal denoising autoencoder.

    Encoder: `depth` stride-2 stages (channels base, 2*base, 4*base, ...) -> 128 / 2^depth spatial size.
    Latent : 1x1 conv to `latent_ch` channels  => latent_ch * (128/2^depth)^2 numbers (default 16*8*8 = 1024,
             48x smaller than the 49 152 input values).
    Decoder: nearest-neighbour upsampling + convs back to 128x128x3 (sigmoid).
    limited_skip: optional single narrow skip (4 channels at 32x32). Off by default: the assignment forbids
                  unrestricted skips, so this is only studied as an ablation.
    """

    def __init__(self, base: int = 32, latent_ch: int = 16, dropout: float = 0.1, depth: int = 4,
                 limited_skip: bool = False, skip_ch: int = 4):
        super().__init__()
        self.cfg = dict(base=base, latent_ch=latent_ch, dropout=dropout, depth=depth, limited_skip=limited_skip)
        chs = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        prev = 3
        for c in chs:
            self.enc.append(nn.Sequential(conv_block(prev, c, 3, 2), conv_block(c, c, 3, 1), nn.Dropout2d(dropout)))
            prev = c
        self.to_latent = nn.Conv2d(chs[-1], latent_ch, 1)
        self.from_latent = nn.Sequential(nn.Conv2d(latent_ch, chs[-1], 1), nn.BatchNorm2d(chs[-1]), nn.ReLU(inplace=True))
        dec_out = list(reversed(chs[:-1])) + [chs[0]]
        self.skip_stage = 1 if limited_skip else -1
        self.skip_proj = nn.Conv2d(chs[1], skip_ch, 1) if limited_skip else None
        self.dec = nn.ModuleList()
        cin = chs[-1]
        for j, co in enumerate(dec_out):
            extra = skip_ch if j == self.skip_stage else 0
            self.dec.append(nn.ModuleList([conv_block(cin + extra, co), conv_block(co, co)]))
            cin = co
        self.out = nn.Conv2d(cin, 3, 3, padding=1)

    @property
    def latent_dim(self) -> int:
        d = self.cfg["depth"]
        return self.cfg["latent_ch"] * (128 // 2 ** d) ** 2

    def encode(self, x):
        feats = []
        for blk in self.enc:
            x = blk(x)
            feats.append(x)
        return self.to_latent(x), feats

    def decode(self, z, feats=None):
        h = self.from_latent(z)
        for j, (c1, c2) in enumerate(self.dec):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            if j == self.skip_stage and feats is not None:
                h = torch.cat([h, self.skip_proj(feats[1])], 1)
            h = c2(c1(h))
        return torch.sigmoid(self.out(h))

    def forward(self, x):
        z, feats = self.encode(x)
        return self.decode(z, feats)


# =============================================================================== Task 2
CLASSIFIER_CONFIGS = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "large": (32, 64, 128, 256, 256),
}


class CorruptionClassifier(nn.Module):
    """Conv classifier; avg+max global pooling so both global (blur) and local (noise, mask) evidence is used."""

    def __init__(self, channels=(32, 64, 128, 256), dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.cfg = dict(channels=list(channels), dropout=dropout)
        layers, prev = [], 3
        for c in channels:
            layers += [conv_block(prev, c), conv_block(c, c), nn.MaxPool2d(2)]
            prev = c
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * prev, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([F.adaptive_avg_pool2d(h, 1).flatten(1), F.adaptive_max_pool2d(h, 1).flatten(1)], 1)
        return self.fc(self.drop(h))  # logits


# =============================================================================== Task 3
class SoftMoE(nn.Module):
    """w = softmax(G(x)/tau);  x_hat = w0*x + w1*A_salt(x) + w2*A_blur(x) + w3*A_occ(x)."""

    def __init__(self, gate: nn.Module, experts: list[nn.Module], tau: float = 1.0):
        super().__init__()
        assert len(experts) == 3
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x):
        logits = self.gate(x)
        w = F.softmax(logits / self.tau, dim=1)
        outs = [x] + [e(x) for e in self.experts]
        y = sum(w[:, k].view(-1, 1, 1, 1) * outs[k] for k in range(4))
        return y, w, logits


# =============================================================================== Task 4
class FiLMNorm(nn.Module):
    """InstanceNorm followed by a style-dependent scale and shift (conditions the generator at every decoder stage)."""

    def __init__(self, c: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(c, affine=False)
        self.fc = nn.Linear(style_dim, 2 * c)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, e):
        g, b = self.fc(e).chunk(2, dim=1)
        return self.norm(x) * (1 + g[:, :, None, None]) + b[:, :, None, None]


class UpBlock(nn.Module):
    def __init__(self, cin, cout, style_dim, dropout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
        self.norm = FiLMNorm(cout, style_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, e):
        return F.relu(self.drop(self.norm(self.conv(x), e)))


class UNetGenerator(nn.Module):
    """U-Net generator G(x, s). 128x128 -> 2x2 bottleneck -> 128x128, learned style embedding injected
    (a) as extra input channels and (b) through FiLM scale/shift in every decoder block."""

    def __init__(self, base: int = 64, n_styles: int = 3, style_dim: int = 16, dropout: float = 0.3, depth: int = 6):
        super().__init__()
        self.cfg = dict(base=base, n_styles=n_styles, style_dim=style_dim, dropout=dropout, depth=depth)
        self.emb = nn.Embedding(n_styles, style_dim)
        chs = [min(base * 2 ** i, base * 8) for i in range(depth)]
        self.down = nn.ModuleList()
        prev = 3 + style_dim
        for i, c in enumerate(chs):
            layers = [nn.Conv2d(prev, c, 4, 2, 1, bias=(i == 0))]
            if i > 0:
                layers.append(nn.InstanceNorm2d(c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.down.append(nn.Sequential(*layers))
            prev = c
        self.up = nn.ModuleList()
        cin = chs[-1]
        for j in range(depth - 1):
            co = chs[-2 - j]
            self.up.append(UpBlock(cin, co, style_dim, dropout if j < 3 else 0.0))
            cin = 2 * co
        self.final = nn.ConvTranspose2d(cin, 3, 4, 2, 1)

    def forward(self, x, s):
        e = self.emb(s)
        h = torch.cat([x, e[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1)
        skips = []
        for d in self.down:
            h = d(h)
            skips.append(h)
        h = skips.pop()
        for u in self.up:
            h = u(h, e)
            h = torch.cat([h, skips.pop()], 1)
        return torch.tanh(self.final(h))


class PatchDiscriminator(nn.Module):
    """PatchGAN D(x, y, s): judges local photo-sketch-style patches (output: grid of logits)."""

    def __init__(self, base: int = 64, n_styles: int = 3, style_dim: int = 16):
        super().__init__()
        self.cfg = dict(base=base, n_styles=n_styles, style_dim=style_dim)
        self.emb = nn.Embedding(n_styles, style_dim)
        b = base
        self.net = nn.Sequential(
            nn.Conv2d(6 + style_dim, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b, 2 * b, 4, 2, 1, bias=False), nn.InstanceNorm2d(2 * b), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * b, 4 * b, 4, 2, 1, bias=False), nn.InstanceNorm2d(4 * b), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * b, 8 * b, 4, 1, 1, bias=False), nn.InstanceNorm2d(8 * b), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * b, 1, 4, 1, 1),
        )

    def forward(self, x, y, s):
        e = self.emb(s)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, y, e], 1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and getattr(m, "weight", None) is not None:
        if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
            return  # FiLM layers start at identity
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
