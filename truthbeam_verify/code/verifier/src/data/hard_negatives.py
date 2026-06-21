"""Hard-negative composition for InfoNCE training.

Spec calls for batch-level negative composition:
    30% easy:    cross-session XOF
    30% medium:  same-session distant rows (|Δrow| > 1200)
    25% hard:    same-session neighbors  (1 ≤ |Δrow| ≤ 10)
    15% hardest: same-row XOF with octave 3 fully replaced with uniform
                 random uint8

For exp001a smoke test we use plain in-batch InfoNCE (positives on diagonal,
negatives off-diagonal). The full composition above is added once the
smoke-test loop is validated.

This module currently provides the octave-3 perturbation primitive only.
"""
from __future__ import annotations

import numpy as np
import torch

from .xof_generation import OCTAVE_SHAPES


def perturb_octave3_random(
    octaves: list[torch.Tensor],
    rng: np.random.Generator,
) -> list[torch.Tensor]:
    """Replace octave 3 with uniform random uint8/255 noise. Returns new list."""
    h, w = OCTAVE_SHAPES[3]
    noise_np = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
    noise = torch.from_numpy(noise_np).float() / 255.0
    return [octaves[0], octaves[1], octaves[2], noise]
