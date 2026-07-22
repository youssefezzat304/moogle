"""joint_fusion.py — from-scratch shared-latent WAC<->geomap model (512 / token 32).

ONE self-contained model. No pretrained loading. Both encoders learn a SHARED
latent space from epoch 1 (joint training), so there is no separate-spaces
alignment problem to retrofit.

Config (fixed): image 512, token 32 -> 16x16 = 256 tokens, hidden_dim 512.

Components:
  E_w : WAC encoder        (1 channel  -> tokens)
  E_g : geomap encoder     (num_classes one-hot -> tokens)
  D_w : WAC decoder        (tokens -> 1-channel WAC, continuous)
  D_g : geomap decoder     (tokens -> num_classes logits, CATEGORICAL)

Four paths share the one latent:
  wac->wac, geo->geo, wac->geo (GOAL), geo->wac (reverse).
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
# Shared ViT-style encoder (channels configurable for WAC=1 vs geo=num_classes)
# ──────────────────────────────────────────────────────────────────────────
class ViTEncoder(nn.Module):
    def __init__(self, in_channels: int, image_size: int = 512,
                 patch_size: int = 32, hidden_dim: int = 512,
                 nheads: int = 8, num_layers: int = 6):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2     # 16*16 = 256

        self.conv_proj = nn.Conv2d(in_channels, hidden_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv_proj(x)                               # (B, D, 16, 16)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)  # (B, 256, D)
        x = x + self.pos_embed
        return self.transformer(x)                          # (B, 256, D)


# ──────────────────────────────────────────────────────────────────────────
# Decoders: 16 -> 32 -> 64 -> 128 -> 256 -> 512  (five clean 2x stages)
#
# Uses RESIZE-CONV (bilinear upsample + 3x3 conv) instead of ConvTranspose2d.
# Reason: ConvTranspose2d with stride==kernel has non-overlapping receptive
# fields and prints periodic CHECKERBOARD artifacts (Odena et al.). For lunar
# imagery a grid artifact masquerades as surface texture, and for a categorical
# geomap it creates spurious boundary flicker. Bilinear-upsample-then-conv has
# no learned-weight imbalance to go wrong, so it upsamples cleanly.
# ──────────────────────────────────────────────────────────────────────────
class _UpBlock(nn.Module):
    """One clean 2x upsample: bilinear resize -> 3x3 conv -> BN -> GELU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(self.up(x))))


class _UpStack(nn.Module):
    """Shared upsampling trunk 16->512 via five clean 2x stages.

    16 -> 32 -> 64 -> 128 -> 256 -> 512, ending at 32 feature channels.
    """
    def __init__(self, hidden_dim=512, grid=16):
        super().__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 1), nn.BatchNorm2d(256), nn.GELU())
        # five 2x stages: channels taper 256->128->96->64->48->32
        self.blocks = nn.Sequential(
            _UpBlock(256, 128),   # 16 -> 32
            _UpBlock(128,  96),   # 32 -> 64
            _UpBlock( 96,  64),   # 64 -> 128
            _UpBlock( 64,  48),   # 128 -> 256
            _UpBlock( 48,  32),   # 256 -> 512
        )

    def forward(self, z):
        n, s, d = z.shape
        x = z.reshape(n, self.grid, self.grid, d).permute(0, 3, 1, 2)
        x = self.bottleneck(x)
        x = self.blocks(x)                                   # (B, 32, 512, 512)
        return x


class WACDecoder(nn.Module):
    """tokens -> (B,1,512,512) continuous WAC (no activation, normalised range)."""
    def __init__(self, hidden_dim=512, grid=16):
        super().__init__()
        self.trunk = _UpStack(hidden_dim, grid)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1))
    def forward(self, z):
        return self.head(self.trunk(z))


class GeoDecoder(nn.Module):
    """tokens -> (B,num_classes,512,512) class logits (categorical)."""
    def __init__(self, num_classes, hidden_dim=512, grid=16):
        super().__init__()
        self.trunk = _UpStack(hidden_dim, grid)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, num_classes, 1))
    def forward(self, z):
        return self.head(self.trunk(z))


# ──────────────────────────────────────────────────────────────────────────
# Joint fusion model
# ──────────────────────────────────────────────────────────────────────────
class JointFusion(nn.Module):
    def __init__(self, num_classes, image_size=512, patch_size=32,
                 hidden_dim=512, nheads=8, num_layers=6):
        super().__init__()
        grid = image_size // patch_size                       # 16
        self.enc_w = ViTEncoder(1, image_size, patch_size, hidden_dim,
                                nheads, num_layers)
        self.enc_g = ViTEncoder(num_classes, image_size, patch_size, hidden_dim,
                                nheads, num_layers)
        self.dec_w = WACDecoder(hidden_dim, grid)
        self.dec_g = GeoDecoder(num_classes, hidden_dim, grid)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # individual paths
    def wac2wac(self, wac):            return self.dec_w(self.enc_w(wac))
    def geo2geo(self, geo):            return self.dec_g(self.enc_g(geo))
    def wac2geo(self, wac):            return self.dec_g(self.enc_w(wac))   # GOAL
    def geo2wac(self, geo):            return self.dec_w(self.enc_g(geo))

    def forward(self, wac, geo_onehot):
        z_w = self.enc_w(wac)
        z_g = self.enc_g(geo_onehot)
        return {
            "wac_rec": self.dec_w(z_w),
            "geo_rec": self.dec_g(z_g),
            "wac2geo": self.dec_g(z_w),    # the goal
            "geo2wac": self.dec_w(z_g),
            "z_w": z_w, "z_g": z_g,
        }


if __name__ == "__main__":
    NC = 50
    m = JointFusion(num_classes=NC)
    wac = torch.rand(2, 1, 512, 512)
    geo = torch.zeros(2, NC, 512, 512); geo[:, 0] = 1.0
    out = m(wac, geo)
    for k, v in out.items():
        print(f"{k:8s}: {tuple(v.shape)}")
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params: {n:.1f}M")
