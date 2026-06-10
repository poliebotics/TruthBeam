"""Coordinate-aware Feature Pyramid Network.

Standard FPN top-down pathway with 1×1 lateral connections, plus normalized
(x, y) coordinate channels appended at each output level. The coordinate
channels give the network explicit position information — useful for sparse
CFA inputs where the spatial position of each pixel within the 2×2 Bayer
block matters.

Encoder must produce 4 stages of features at strides [4, 8, 16, 32]. Output
levels are P2 (stride 4) … P5 (stride 32), each with `out_channels + 2` channels.

Channel dim convention:
    encoder_dims = (C0, C1, C2, C3)   # ConvNeXt-Tiny: (96, 192, 384, 768)
                                       # ConvNeXt-Large: (192, 384, 768, 1536)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _coord_channels(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """Return (b, 2, h, w) with x ∈ [-1, 1] in channel 0, y ∈ [-1, 1] in channel 1."""
    if w > 1:
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    else:
        x = torch.zeros(1, device=device, dtype=dtype)
    if h > 1:
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    else:
        y = torch.zeros(1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1).contiguous()
    return coord


class CoordAwareFPN(nn.Module):
    """Top-down FPN over 4 encoder stages with coordinate channels appended.

    forward(feats) -> list[P2, P3, P4, P5] each (B, out_channels+2, h, w).
    """

    def __init__(self, encoder_dims: tuple[int, int, int, int], out_channels: int = 256):
        super().__init__()
        c0, c1, c2, c3 = encoder_dims
        self.lat0 = nn.Conv2d(c0, out_channels, kernel_size=1)
        self.lat1 = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        # 3x3 smoothing convs after merge (standard FPN)
        self.smooth0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_channels = out_channels
        self.out_channels_with_coords = out_channels + 2

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        f0, f1, f2, f3 = feats  # strides 4, 8, 16, 32
        p3 = self.lat3(f3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="nearest")
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="nearest")
        p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="nearest")

        p0 = self.smooth0(p0)
        p1 = self.smooth1(p1)
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)

        out = []
        for p in (p0, p1, p2, p3):
            b, _, h, w = p.shape
            coords = _coord_channels(b, h, w, p.device, p.dtype)
            out.append(torch.cat([p, coords], dim=1))
        return out  # [P2_coord, P3_coord, P4_coord, P5_coord]
