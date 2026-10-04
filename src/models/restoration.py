"""Models for Tasks 1-3: denoising autoencoder, corruption classifier, soft mixture-of-experts."""
import torch
import torch.nn as nn


def conv_bn(cin, cout, stride=1, k=3, act="relu"):
    pad = (k - 1) // 2 if stride == 1 else 1
    layers = [nn.Conv2d(cin, cout, k if stride == 1 else 4, stride, pad, bias=False), nn.BatchNorm2d(cout)]
    layers.append(nn.LeakyReLU(0.2, inplace=True) if act == "lrelu" else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DenoisingAE(nn.Module):
    """Convolutional encoder -> compressed latent -> convolutional decoder.

    Encoder: 128 -> 64 -> 32 -> 16 -> 8 with channels base, 2b, 4b, 8b.
    Bottleneck: a 1x1 conv squeezes 8b channels to `z_ch` channels at 8x8, so the
    latent has z_ch*64 numbers (e.g. 16*64 = 1024 vs 49 152 input values -> 48x
    compression). Dropout is applied to the latent code.
    Decoder: nearest-neighbour upsampling + 3x3 convs (avoids the checkerboard
    artefacts of transposed convolutions), sigmoid output in [0, 1].

    skip=True adds ONE limited skip connection at 64x64 (used only for the ablation
    study; the submitted model has skip=False, i.e. no path around the bottleneck).
    """

    def __init__(self, base=32, z_ch=16, dropout=0.1, skip=False):
        super().__init__()
        b = base
        self.skip = skip
        self.e1 = nn.Sequential(conv_bn(3, b, 2, act="lrelu"), conv_bn(b, b, act="lrelu"))            # 64
        self.e2 = nn.Sequential(conv_bn(b, 2 * b, 2, act="lrelu"), conv_bn(2 * b, 2 * b, act="lrelu"))  # 32
        self.e3 = nn.Sequential(conv_bn(2 * b, 4 * b, 2, act="lrelu"), conv_bn(4 * b, 4 * b, act="lrelu"))  # 16
        self.e4 = nn.Sequential(conv_bn(4 * b, 8 * b, 2, act="lrelu"), conv_bn(8 * b, 8 * b, act="lrelu"))  # 8
        self.to_z = nn.Conv2d(8 * b, z_ch, 1)
        self.drop = nn.Dropout2d(dropout)
        self.from_z = nn.Sequential(nn.Conv2d(z_ch, 8 * b, 1), nn.ReLU(inplace=True))

        def up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cin, cout), conv_bn(cout, cout))

        self.d4 = up(8 * b, 4 * b)   # 16
        self.d3 = up(4 * b, 2 * b)   # 32
        self.d2 = up(2 * b, b)       # 64
        self.d1 = up(b * (2 if skip else 1), b)  # 128
        self.out = nn.Sequential(nn.Conv2d(b, 3, 3, 1, 1), nn.Sigmoid())
        self.latent_dim = z_ch * 8 * 8

    def encode(self, x):
        h1 = self.e1(x)
        z = self.to_z(self.e4(self.e3(self.e2(h1))))
        return z, h1

    def decode(self, z, h1=None):
        h = self.d2(self.d3(self.d4(self.from_z(self.drop(z)))))
        if self.skip:
            h = torch.cat([h, h1], 1)
        return self.out(self.d1(h))

    def forward(self, x):
        z, h1 = self.encode(x)
        return self.decode(z, h1)


class CorruptionClassifier(nn.Module):
    """4-way CNN: clean / salt / blur / occlusion. Returns logits.

    The first block keeps full resolution because blur and salt-and-pepper are
    high-frequency cues that early downsampling would destroy.
    """

    def __init__(self, base=32, dropout=0.3):
        super().__init__()
        chs = [base, 2 * base, 4 * base, 8 * base]
        layers, cin = [], 3
        for c in chs:
            layers += [conv_bn(cin, c), conv_bn(c, c), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, 4))

    def forward(self, x):
        return self.head(self.features(x))


class SoftMoE(nn.Module):
    """Differentiable soft mixture of experts (Task 3).

        w = softmax(G(x) / tau)
        x_hat = w0 * x + w1 * A_salt(x) + w2 * A_blur(x) + w3 * A_occ(x)

    Branch order matches the classifier class order (clean, salt, blur, occlusion),
    so the gate can be initialised from the Task 2 classifier and the experts from
    the Task 2 specialists. Returns (reconstruction, weights, gate logits, branch outputs).
    """

    def __init__(self, gate, salt, blur, occ, tau=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList([salt, blur, occ])
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)
        branches = torch.stack([x] + [e(x) for e in self.experts], 1)  # [B, 4, 3, H, W]
        y = (w[:, :, None, None, None] * branches).sum(1)
        return y, w, logits, branches


def balance_loss(w):
    """sum_k (mean_batch(w_k) - 1/4)^2  - penalises routing collapse onto one branch."""
    return ((w.mean(0) - 1.0 / w.shape[1]) ** 2).sum()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
