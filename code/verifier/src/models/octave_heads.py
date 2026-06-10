"""Per-octave XOF prediction heads (audit-conformant).

Each head consumes one FPN level (operator-chosen mapping based on stride
proximity to the octave's grid resolution), adaptively pools to the octave's
spatial size, applies 1×1 conv to 3 channels, and **tanh activation** so the
output is in [-1, 1] — matching the centered training target convention
`(byte - 127.5) / 127.5`.

Octave shapes (3 channels = R, G, B) per protocol v8:
    oct0:  17 × 30
    oct1:  34 × 60
    oct2:  68 × 120
    oct3: 135 × 240

FPN level mapping (P5 stride 32 ... P2 stride 4 on a packed-CFA-half-res
2300×2660 input):
    P5 native ≈ (71, 83)     — closest to oct0 (17, 30) and oct1 (34, 60)
    P4 native ≈ (143, 166)   — closest to oct2 (68, 120) and oct3 (135, 240)

Each head adaptively pools the chosen FPN level to the exact octave size.
Pre-projection a 3×3 conv with GELU+GroupNorm gives a small refinement before
the 1×1 RGB projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OCTAVE_SHAPES = ((17, 30), (34, 60), (68, 120), (135, 240))
# Spec mapping (audit A5): oct0→P5, oct1→P4, oct2→P3, oct3→P2.
# CoordAwareFPN.forward returns [P2, P3, P4, P5] (stride 4..32), so the FPN
# index per octave is the reverse of the P-level number minus 2:
#   oct0 → P5 → fpn_levels[3]
#   oct1 → P4 → fpn_levels[2]
#   oct2 → P3 → fpn_levels[1]
#   oct3 → P2 → fpn_levels[0]
OCTAVE_FPN_LEVEL = (3, 2, 1, 0)


def _gn_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class _OctaveHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_h: int, out_w: int):
        super().__init__()
        g = _gn_groups(hidden)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(g, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(g, hidden),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(hidden, 3, kernel_size=1)
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, fpn_feature: torch.Tensor) -> torch.Tensor:
        # Pool to octave grid first (cheaper if octave is much smaller than FPN feature).
        x = F.adaptive_avg_pool2d(fpn_feature, output_size=(self.out_h, self.out_w))
        x = self.refine(x)
        x = self.proj(x)
        return torch.tanh(x)  # ∈ [-1, 1]


class OctaveHeads(nn.Module):
    """Predicts 4 octaves from FPN features.

    `enabled_octaves` lets A0 (oct0+oct1 only) skip oct2/oct3 — the unused heads
    are simply not built and the forward returns `None` for skipped octaves.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        enabled_octaves: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.enabled = tuple(sorted(set(enabled_octaves)))
        self.heads = nn.ModuleDict()
        for i in self.enabled:
            h, w = OCTAVE_SHAPES[i]
            self.heads[f"oct{i}"] = _OctaveHead(in_channels, hidden, h, w)

    def forward(self, fpn_levels: list[torch.Tensor]) -> list[torch.Tensor | None]:
        outs: list[torch.Tensor | None] = [None, None, None, None]
        for i in self.enabled:
            level = OCTAVE_FPN_LEVEL[i]
            outs[i] = self.heads[f"oct{i}"](fpn_levels[level])
        return outs
