"""Per-channel median + IQR/1.349 normalization (audit §5).

Stats are fitted on **training rows only** of one session. They are stored in
`<exp>/normalization_stats.json` and applied at training time.

For experiments A0/A1/A2/A4/A7 the stats source matches the session being
loaded (D2 rows use D2-train stats; V10 rows use V10-train stats).

For experiment A6 (transfer test), the stats source is **D2-train ONLY** for
both D2-val and ALL V10 evaluation bins.

Observability metrics (mean/std/saturation/contrast — see `src/eval/observability.py`)
must be computed on the PRE-normalized packed CFA, not on the normalized tensor,
to preserve the photon regime per row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from .packed_cfa import HALF_H, HALF_W, load_packed_cfa_cached, cache_path_for

EPS = 1e-6
GAUSSIAN_IQR_FACTOR = 1.349   # IQR(N(0, 1)) ≈ 1.349; divide IQR by this to estimate σ


def fit_stats_from_cached(
    cache_root: Path,
    session: str,
    train_rows: Iterable[int],
    *,
    sample_pixels_per_frame: int = 65536,
    seed: int = 0,
) -> dict:
    """Fit per-channel median + IQR/1.349 from cached packed-CFA frames.

    To keep memory bounded for sessions with thousands of frames, we sample
    `sample_pixels_per_frame` pixel positions per frame (deterministic via seed).
    For a 2300×2660 frame, 65536 ≈ 1% of pixels per channel — enough to estimate
    median+IQR robustly.

    Returns:
        {
          "method": "median_iqr",
          "center": [c0..c3],   # per-channel median (raw byte units)
          "scale":  [s0..s3],   # per-channel IQR / 1.349, floored at EPS
          "rows_used": [t, ...],
          "session": session,
          "sample_pixels_per_frame": int,
          "seed": int,
        }
    """
    rng = torch.Generator().manual_seed(seed)
    train_rows = list(train_rows)
    if not train_rows:
        raise ValueError("train_rows is empty")

    n_pix_total = sample_pixels_per_frame * len(train_rows)
    samples = torch.empty((4, n_pix_total), dtype=torch.int32)
    pos = 0
    for t in train_rows:
        cp = cache_path_for(cache_root, session, t)
        if not cp.exists():
            raise FileNotFoundError(f"cached packed CFA missing: {cp}")
        cfa = load_packed_cfa_cached(cp).to(torch.int32)  # (4, 2300, 2660)
        flat = cfa.reshape(4, -1)
        idx = torch.randint(0, flat.shape[1], (sample_pixels_per_frame,), generator=rng)
        samples[:, pos:pos + sample_pixels_per_frame] = flat[:, idx]
        pos += sample_pixels_per_frame

    centers = []
    scales = []
    for c in range(4):
        ch = samples[c].float()
        med = float(ch.median().item())
        q25 = float(torch.quantile(ch, 0.25).item())
        q75 = float(torch.quantile(ch, 0.75).item())
        iqr = max(q75 - q25, EPS)
        scale = max(iqr / GAUSSIAN_IQR_FACTOR, EPS)
        centers.append(med)
        scales.append(scale)

    return {
        "method": "median_iqr",
        "center": centers,
        "scale": scales,
        "rows_used": train_rows,
        "session": session,
        "sample_pixels_per_frame": sample_pixels_per_frame,
        "seed": seed,
    }


def save_stats(stats: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_normalization(
    packed_cfa: torch.Tensor,
    stats: dict,
) -> torch.Tensor:
    """Apply median+IQR/1.349 normalization. `packed_cfa` is (4, H, W) uint8 or float."""
    if stats.get("method") != "median_iqr":
        raise ValueError(f"unsupported normalization method: {stats.get('method')}")
    x = packed_cfa.to(torch.float32)
    center = torch.tensor(stats["center"], dtype=torch.float32).view(4, 1, 1)
    scale = torch.tensor(stats["scale"], dtype=torch.float32).view(4, 1, 1)
    return (x - center) / scale
