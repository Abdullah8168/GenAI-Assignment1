"""Style-conditioned pix2pix-style cGAN for face-to-sketch (Task 4).

Generator  G(x, s): U-Net, 128x128 photo + style id -> 128x128 sketch in [-1, 1].
Discriminator D(x, y, s): 70x70-style PatchGAN giving a 14x14 map of real/fake logits.
The style id (0, 1, 2) goes through a LEARNED nn.Embedding in both networks:
  * tiled over the image and concatenated with the input channels, and
  * (generator only) projected and added to the 1x1 bottleneck,
so the condition affects both local texture and global appearance.
"""
import torch
import torch.nn as nn


def init_weights(m):
    # pix2pix initialisation: N(0, 0.02) for convs, N(1, 0.02) for BN scale
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def tile(e, ref):
    """[B, D] embedding -> [B, D, H, W] matching ref's spatial size."""
    return e[:, :, None, None].expand(-1, -1, ref.shape[2], ref.shape[3])


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.f = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.f(x), skip], 1)


class UNetGenerator(nn.Module):
    def __init__(self, base=64, emb_dim=16, dropout=0.5, n_styles=3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(n_styles, emb_dim)
        self.d1 = Down(3 + emb_dim, b, norm=False)   # 64
        self.d2 = Down(b, 2 * b)                      # 32
        self.d3 = Down(2 * b, 4 * b)                  # 16
        self.d4 = Down(4 * b, 8 * b)                  # 8
        self.d5 = Down(8 * b, 8 * b)                  # 4
        self.d6 = Down(8 * b, 8 * b)                  # 2
        self.d7 = nn.Sequential(nn.Conv2d(8 * b, 8 * b, 4, 2, 1), nn.ReLU(inplace=True))  # 1x1 bottleneck
        self.style_proj = nn.Linear(emb_dim, 8 * b)
        self.u1 = Up(8 * b, 8 * b, dropout)           # 2   (+d6)
        self.u2 = Up(16 * b, 8 * b, dropout)          # 4   (+d5)
        self.u3 = Up(16 * b, 8 * b, dropout)          # 8   (+d4)
        self.u4 = Up(16 * b, 4 * b)                   # 16  (+d3)
        self.u5 = Up(8 * b, 2 * b)                    # 32  (+d2)
        self.u6 = Up(4 * b, b)                        # 64  (+d1)
        self.final = nn.Sequential(nn.ConvTranspose2d(2 * b, 3, 4, 2, 1), nn.Tanh())
        self.apply(init_weights)

    def forward(self, x, style):
        e = self.emb(style)
        h1 = self.d1(torch.cat([x, tile(e, x)], 1))
        h2 = self.d2(h1)
        h3 = self.d3(h2)
        h4 = self.d4(h3)
        h5 = self.d5(h4)
        h6 = self.d6(h5)
        z = self.d7(h6) + self.style_proj(e)[:, :, None, None]
        u = self.u1(z, h6)
        u = self.u2(u, h5)
        u = self.u3(u, h4)
        u = self.u4(u, h3)
        u = self.u5(u, h2)
        u = self.u6(u, h1)
        return self.final(u)


class PatchDiscriminator(nn.Module):
    """Input: photo (3) + sketch (3) + tiled style embedding. Output: [B, 1, 14, 14] logits;
    each logit judges one ~70x70 receptive field of the photo/sketch pair."""

    def __init__(self, base=64, emb_dim=16, n_styles=3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(n_styles, emb_dim)
        self.net = nn.Sequential(
            nn.Conv2d(6 + emb_dim, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),            # 64
            nn.Conv2d(b, 2 * b, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * b), nn.LeakyReLU(0.2, True),  # 32
            nn.Conv2d(2 * b, 4 * b, 4, 2, 1, bias=False), nn.BatchNorm2d(4 * b), nn.LeakyReLU(0.2, True),  # 16
            nn.Conv2d(4 * b, 8 * b, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * b), nn.LeakyReLU(0.2, True),  # 15
            nn.Conv2d(8 * b, 1, 4, 1, 1),                                                     # 14
        )
        self.apply(init_weights)

    def forward(self, x, y, style):
        return self.net(torch.cat([x, y, tile(self.emb(style), x)], 1))
