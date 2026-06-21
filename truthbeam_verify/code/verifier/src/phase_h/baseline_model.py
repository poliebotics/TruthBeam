"""Phase H — supervised baseline discriminator architecture.

Tests whether a small supervised CNN can learn a fast version of the
chain-coupling evidence Phase G/Item 1 validate. Output is a baseline,
not a production verifier — that requires F-A behavior, raw realism, and
calibration work that's separate.

Architecture (per operator spec):
    - 4 conv blocks: 3x3 conv → BN → ReLU → 2x2 max pool, channels [64, 128, 256, 512]
    - Global average pool
    - 2-layer FC head: 512 → 128 → 1
    - Sigmoid output (handled outside in BCE-with-logits training)

Input: 7 channels = 4 packed CFA + 3 RGB E, at Phase G eval resolution
       (768×1024 — same as DiffusionDiagnosticUNet).

After 4 ×2 maxpools: 768/16 = 48, 1024/16 = 64. Feature map: (B, 512, 48, 64).
GAP → (B, 512). FC → (B, 128) → (B, 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """3x3 conv → BN → ReLU → 2x2 maxpool."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


class PhaseHBaseline(nn.Module):
    """Binary classifier: real (C, E) vs not-real (shuffled or perturbed).

    Returns LOGITS — apply sigmoid externally for probability or use
    BCEWithLogitsLoss for training.
    """
    def __init__(self, in_channels: int = 7,
                 channels: tuple[int, ...] = (64, 128, 256, 512),
                 fc_hidden: int = 128):
        super().__init__()
        self.in_channels = in_channels
        layers = []
        prev = in_channels
        for c in channels:
            layers.append(ConvBlock(prev, c))
            prev = c
        self.conv_blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels[-1], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 7, 768, 1024)
        h = self.conv_blocks(x)        # (B, 512, 48, 64)
        h = self.gap(h)                # (B, 512, 1, 1)
        h = h.flatten(1)               # (B, 512)
        h = F.relu(self.fc1(h), inplace=True)
        return self.fc2(h).squeeze(-1)  # (B,) — logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Smoke test
    m = PhaseHBaseline()
    print(f"params: {count_params(m)/1e6:.2f} M")
    x = torch.randn(2, 7, 768, 1024)
    y = m(x)
    print(f"input {tuple(x.shape)} → output {tuple(y.shape)} dtype={y.dtype}")
    assert y.shape == (2,), f"expected (2,), got {y.shape}"
    print("OK")
