"""Per-evaluated-row observability logger (audit §6).

Writes one JSONL record per evaluated frame to `<exp>/observability.jsonl`.
Metrics are computed on **raw or black-level-subtracted pre-normalized** packed
CFA — never on the experiment's normalized tensor — so that the photon regime
of each frame is preserved across experiments.

Schema (one record per row, append-only):

    {
      "experiment_id":   str,
      "session":         "D2" | "V10",
      "row":             int,        # capture row index t
      "split":           "train" | "val" | "eval",
      "v10_bin":         "early" | "mid" | "late" | null,
      "mean_intensity_raw":             float,   # over the 4 packed CFA channels
      "std_intensity_raw":              float,
      "saturation_rate":                float,   # fraction of pixels at sensor max (255 for BayerRG8)
      "low_frequency_contrast_proxy":   float,   # std of 8x8-blockmean image, normalized
      "normalization_stats_source":     str,     # e.g. "D2-train" or "V10-train" or "none"
      "offset_used":                    int,     # target_chain_row - capture_row signed offset
    }

The schema is fixed; downstream analysis pivots on these fields.

Note: `v10_bin` mapping (early/mid/late row ranges) is not yet operator-confirmed.
The default mapping below is a reasonable equal-thirds split that the caller may
override via `bin_edges`. Document the chosen split in the run manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

import torch

SENSOR_MAX = 255.0  # BayerRG8


def _v10_bin(row: int, n_total: int = 3743, edges: tuple[int, int] | None = None) -> str:
    """Return early|mid|late for a V10 row index.

    `edges` defaults to equal thirds: early = [0, n/3), mid = [n/3, 2n/3), late = [2n/3, n).
    Override with operator-confirmed boundaries when available.
    """
    if edges is None:
        third = n_total // 3
        edges = (third, 2 * third)
    e0, e1 = edges
    if row < e0:
        return "early"
    if row < e1:
        return "mid"
    return "late"


def compute_row_observability(
    packed_cfa_pre_norm: torch.Tensor,
    *,
    sensor_max: float = SENSOR_MAX,
    block_size: int = 8,
) -> dict[str, float]:
    """`packed_cfa_pre_norm` is (4, H, W) raw bytes (uint8 ok) or float pre-norm.

    For sparse-native captures (3/4 of each channel is exact 0), use only nonzero
    positions for mean/std/saturation; otherwise zeros dominate and reported
    intensity will be a quarter of true.
    """
    x = packed_cfa_pre_norm.to(torch.float32)
    # Detect sparse-native vs dense by zero-fraction per channel.
    nonzero_per_channel = (x > 0).reshape(x.shape[0], -1).float().mean(dim=1)
    sparse = bool((nonzero_per_channel < 0.30).any().item())  # heuristic, native is ~25% nonzero
    if sparse:
        mask = x > 0
        mean = x[mask].mean().item()
        std = x[mask].std().item()
        saturation = (x[mask] >= sensor_max).float().mean().item()
    else:
        mean = x.mean().item()
        std = x.std().item()
        saturation = (x >= sensor_max).float().mean().item()

    # Low-frequency contrast proxy: std of block-mean intensity (across all 4 channels collapsed).
    intensity = x.mean(dim=0)  # (H, W)
    H, W = intensity.shape
    Hb, Wb = H // block_size, W // block_size
    if Hb > 0 and Wb > 0:
        cropped = intensity[: Hb * block_size, : Wb * block_size]
        block_means = cropped.unfold(0, block_size, block_size).unfold(1, block_size, block_size).mean(dim=(-1, -2))
        lf_contrast = block_means.std().item() / max(mean, 1.0)
    else:
        lf_contrast = 0.0

    return {
        "mean_intensity_raw": mean,
        "std_intensity_raw": std,
        "saturation_rate": saturation,
        "low_frequency_contrast_proxy": lf_contrast,
    }


class ObservabilityLogger:
    """Append-only JSONL observability log."""

    def __init__(self, path: Path, experiment_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def log(
        self,
        *,
        session: Literal["D2", "V10"],
        row: int,
        split: Literal["train", "val", "eval"],
        v10_bin: str | None,
        normalization_stats_source: str,
        offset_used: int,
        observability: dict[str, float],
    ) -> None:
        record = {
            "experiment_id": self.experiment_id,
            "session": session,
            "row": int(row),
            "split": split,
            "v10_bin": v10_bin,
            **observability,
            "normalization_stats_source": normalization_stats_source,
            "offset_used": int(offset_used),
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()


def log_batch_observability(
    logger: ObservabilityLogger,
    packed_cfa_pre_norm_batch: torch.Tensor,  # (B, 4, H, W)
    rows: Iterable[int],
    sessions: Iterable[Literal["D2", "V10"]],
    split: Literal["train", "val", "eval"],
    *,
    normalization_stats_source: str,
    offset_used: int,
    v10_n_total: int = 3743,
    v10_edges: tuple[int, int] | None = None,
) -> None:
    rows = list(rows)
    sessions = list(sessions)
    assert len(rows) == len(sessions) == packed_cfa_pre_norm_batch.shape[0]
    for i, (row, sess) in enumerate(zip(rows, sessions)):
        obs = compute_row_observability(packed_cfa_pre_norm_batch[i])
        v10_bin = _v10_bin(row, v10_n_total, v10_edges) if sess == "V10" else None
        logger.log(
            session=sess,
            row=row,
            split=split,
            v10_bin=v10_bin,
            normalization_stats_source=normalization_stats_source,
            offset_used=offset_used,
            observability=obs,
        )
