"""YAML config loader with deep `extends:` merging.

Resolves `extends: <relative_path>` by recursively loading the parent and
deep-merging the child on top. Multiple extends entries (list) are merged
left-to-right; later parents override earlier.

Required Phase D fields are validated after merging — missing required keys
raise `KeyError` with a clear message rather than failing later with
NoneType errors deep in the trainer.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = (
    ("data", "d2_train_start"),
    ("data", "d2_train_end"),
    ("data", "d2_val_start"),
    ("data", "d2_val_end"),
    ("data", "d2_val_calib_start"),
    ("data", "d2_val_calib_end"),
    ("data", "d2_val_report_start"),
    ("data", "d2_val_report_end"),
    ("data", "v10_train_start"),
    ("data", "v10_train_end"),
    ("data", "v10_val_start"),
    ("data", "v10_val_end"),
    ("data", "v10_val_calib_start"),
    ("data", "v10_val_calib_end"),
    ("data", "v10_val_report_start"),
    ("data", "v10_val_report_end"),
    ("data", "v10_early_start"),
    ("data", "v10_early_end"),
    ("data", "v10_mid_start"),
    ("data", "v10_mid_end"),
    ("data", "v10_late_start"),
    ("data", "v10_late_end"),
    ("data", "cache_root"),
    ("data", "black_level"),
    ("offset", "value"),                      # may be null until diagnostic runs
    ("offset", "convention"),
    ("xof", "stored_representation"),
    ("candidate_ranking", "d_search_max"),
    ("provenance", "camera_photometry_locked"),
    ("provenance", "projector_pipeline_lock_status"),
    ("model", "class"),
    ("train", "lr"),
    ("train", "batch_size"),
    ("train", "epochs"),
    ("train", "warmup_steps"),
    ("train", "grad_clip"),
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`; both dicts. Returns a new dict."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path) -> dict:
    """Load a YAML config, resolving `extends:` (string or list) recursively."""
    config_path = Path(config_path)
    raw = _load_yaml(config_path)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ValueError(f"`extends` must be a string or list of strings, got {type(extends)} in {config_path}")

    merged: dict = {}
    for ext in extends:
        ext_path = (config_path.parent / ext).resolve() if not Path(ext).is_absolute() else Path(ext)
        parent = load_config(ext_path)  # recursive
        merged = _deep_merge(merged, parent)
    return _deep_merge(merged, raw)


def validate_required_keys(cfg: dict, required: tuple[tuple[str, ...], ...] = REQUIRED_KEYS) -> None:
    """Raise KeyError listing all missing required keys."""
    missing: list[str] = []
    for path in required:
        node: Any = cfg
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if not ok:
            missing.append(".".join(path))
    if missing:
        raise KeyError(
            f"Phase D config missing required keys (after extends merge): {missing}"
        )


def load_and_validate(config_path: Path) -> dict:
    cfg = load_config(config_path)
    validate_required_keys(cfg)
    return cfg
