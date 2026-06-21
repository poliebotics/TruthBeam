"""Run-manifest writer (audit §2, §12).

Each training run writes a single `<exp_dir>/manifest.json` capturing the
declared experiment configuration plus the audit-mandated provenance fields:

- `offset_convention`: from §2; the offset used to map capture_row → target_chain_row.
- `camera_photometry_locked`: §12; true for these recordings.
- `projector_pipeline_lock_status`: §12; "unconfirmed" until operator confirms.
- `xof_stored_representation`: §3; "raw_unsigned_byte" or "spec_centered_minus_128".
- `normalization_stats_source`: §5; e.g. "D2-train" or "V10-train" or "D2-train (A6 transfer)".
- `d_search_max`: §7; from config, never inferred.
- `negatives_seed`: §7 invariant 4; fixed per run.
- `git_commit`: best-effort.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _hash_files(paths: list[Path]) -> str:
    """SHA-256 over sorted paths' contents — deterministic identity for a code or cache set."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
        except (FileNotFoundError, IsADirectoryError):
            continue
    return h.hexdigest()


def compute_code_hash(src_dir: Path) -> str:
    src_dir = Path(src_dir)
    py_files = sorted(src_dir.rglob("*.py"))
    return _hash_files(py_files)


def compute_target_cache_hash(cache_root: Path) -> str:
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return "missing"
    pt_files = sorted(cache_root.rglob("*.pt"))
    return _hash_files(pt_files[:1024])  # bound the work; cache is hundreds-of-thousands of files


def write_run_manifest(
    out_dir: Path,
    *,
    experiment_id: str,
    config: dict[str, Any],
    offset_convention: dict[str, Any],
    xof_stored_representation: str,
    normalization_stats_source: str,
    d_search_max: int | None,
    negatives_seed: int,
    splits: dict[str, Any],
    bayer_channel_order: str = "RGGB",
    black_level_handling: str = "no_measurement_found / no_subtraction",
    target_cache_hash: str | None = None,
    code_hash: str | None = None,
    candidate_negative_families: list[str] | None = None,
    fmr_calibration_split: str = "D2-val first half",
    fmr_report_split: str = "D2-val second half",
    stn_guardrails: dict[str, Any] | None = None,
    ddp_config: dict[str, Any] | None = None,
    projector_pipeline_lock_status: str = "unconfirmed",
    camera_photometry_locked: bool = True,
) -> Path:
    """Write `<out_dir>/manifest.json` with the audit-mandated fields (A17)."""
    if xof_stored_representation not in ("raw_unsigned_byte", "spec_centered_minus_128"):
        raise ValueError(
            f"xof_stored_representation must be 'raw_unsigned_byte' or "
            f"'spec_centered_minus_128', got {xof_stored_representation!r}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "code_hash": code_hash or "not_computed",
        "config": config,
        "splits": splits,
        "offset_convention": offset_convention,
        "winning_offset": offset_convention.get("winning_offset"),
        "winning_offset_convention": offset_convention.get("convention",
                                                            "target_chain_row = capture_row + offset"),
        "bayer_channel_order": bayer_channel_order,
        "black_level_handling": black_level_handling,
        "target_cache_hash": target_cache_hash or "not_computed",
        "xof_stored_representation": xof_stored_representation,
        "candidate_negative_families": candidate_negative_families
            or ["delay_window", "near_shift", "same_session_random", "cross_session_random"],
        "d_search_max": d_search_max,
        "negatives_seed": negatives_seed,
        "fmr_calibration_split": fmr_calibration_split,
        "fmr_report_split": fmr_report_split,
        "stn_guardrails": stn_guardrails,
        "ddp_config": ddp_config,
        "normalization_stats_source": normalization_stats_source,
        "camera_photometry_locked": camera_photometry_locked,
        "projector_pipeline_lock_status": projector_pipeline_lock_status,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
