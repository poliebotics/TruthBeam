"""session_logs.py — CSV writers for a session's per-step logs.

Scope: format of chain_log.csv and capture_log.csv, nothing else. Column
names, ordering, comment-line prefix, fsync discipline — all live here.

Adapt this module for:
- Adding a new column to either CSV (must also update verifier assumptions
  in any downstream project; CSV hashes are pinned into manifest_hash_final)
- Changing the per-row write strategy (currently: append + fsync per row,
  trading throughput for crash durability)

Do NOT put here:
- Chain / hashing logic — that's chain.py / session_schema.py
- The session manifest — that's session_schema.SessionManifest
- Post-session log-hash computation — that's session_finalize.py

====================================================================
v9 B++ SCHEMA CHANGE (step 2 of v9 landing, 2026-04-24)
====================================================================

Row-local attribution of emission-adjacent fields now uses LIVE
semantics: row t commits the emission that was ON THE PROJECTOR
during cap_t, not the emission queued AFTER cap_t for cap_{t+1}.

Changes from v8:

- `tile_pixel_sha_hex`   → `emission_live_pixel_blake3_hex`
  Hash is over the pixel bytes of E_t (the emission live during
  cap_t), not E_{t+1}. Renamed to (a) reflect the semantic flip,
  (b) drop legacy "tile" projector-side language in favour of
  capture-side "emission live" language, and (c) spell out the
  hash algorithm so no downstream reader mistakes it for a
  different SHA family.

- `tile_queued_wall_ns`  → `emission_live_queued_wall_ns`
  Wall-clock time at which E_t was queued to the DMD. Under v8
  this column recorded the queue time of E_{t+1} (the new
  emission queued inside iter t); under v9 it records the queue
  time of the emission LIVE during cap_t (queued inside iter
  t-1, or at bootstrap for row 0). Same rename rationale.

- `emission_png_path`    (name unchanged)
  Points to `derived/Emissions/tile_{t:06d}.png` whose bytes are
  E_t under v9 (E_{t+1} under v8). The filename convention still
  uses "tile_" for backwards-grep-ability.

- `xof_seed_R_hex`, `xof_seed_G_hex`, `xof_seed_B_hex`
  REMOVED. These are deterministic functions of S_t via
  session_schema.compute_xof_seeds. Persisting them added a
  consistency surface (writer could drift from derivation; a
  "tile_pixel_sha_hex committed the wrong emission" class of
  trap). Verifier re-derives on read.

Unchanged from v8:

- `t`, `S_t_hex`, `bayer_blake3_hex`, `capture_frame_id`,
  `aravis_device_timestamp_ns`, `capture_wall_ns`, `meta_hex`

The per-row chain advance folds in the drand round, so its domain tag is
ROW_DOMAIN_TAG = b"TB:ROW:v9" (a real cryptographic change; a v8 chain_log
is not re-walkable under v9). The XOF-seed and final_root domain tags retain
their v8 label because that math is unchanged. See the
session_schema.py:PROTOCOL_VERSION comment block.

Note on wiring: tb_loop.py's _open_logs duplicates this field list
inline (not imported) and is the code path that actually drives
writes. CHAIN_LOG_FIELDS below is the canonical source of truth;
keep the tb_loop.py inline copy in sync with any change here.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


# v9 B++ schema. See module docstring for v8 → v9 change summary.
# v9 additions (drand per-row): drand_round_number + drand_round_value_hex
# committed per row; both are folded into S_{t+1} per
# session_schema.compute_chain_row (ROW_DOMAIN_TAG=TB:ROW:v9).
CHAIN_LOG_FIELDS = (
    "t", "S_t_hex",
    "bayer_blake3_hex",
    "capture_frame_id",
    "aravis_device_timestamp_ns",
    "capture_wall_ns",
    "emission_live_pixel_blake3_hex",
    "emission_live_queued_wall_ns",
    "meta_hex",
    "emission_png_path",
    "drand_round_number",       # v9: u64 (0 = unavailable sentinel)
    "drand_round_value_hex",    # v9: 64-hex (all-zeros hex = unavailable sentinel)
    "drand_staleness_ms",       # v9: age of cached round at this row (-1 = never fetched)
)

CAPTURE_LOG_FIELDS = (
    "capture_idx", "capture_frame_id",
    "aravis_device_timestamp_ns", "capture_wall_ns",
    "bayer_blake3_hex", "raw_path",
    "consumed_into_chain", "consumed_as_t",
)


@dataclass
class SessionLogger:
    chain_log_path: Path
    capture_log_path: Path
    session_iso_utc: str


def open_logs(session_dir: Path, session_iso_utc: str) -> SessionLogger:
    """Create chain_log.csv and capture_log.csv with their header lines.

    Each CSV begins with a `# session_iso_utc=<...>` comment line followed
    by the DictWriter header row. The files are created with `x` mode so
    a session dir that already has logs errors out rather than silently
    appending.
    """
    logger = SessionLogger(
        chain_log_path=Path(session_dir) / "chain_log.csv",
        capture_log_path=Path(session_dir) / "capture_log.csv",
        session_iso_utc=session_iso_utc,
    )
    with open(logger.chain_log_path, "x", newline="") as f:
        f.write(f"# session_iso_utc={session_iso_utc}\n")
        w = csv.DictWriter(f, fieldnames=list(CHAIN_LOG_FIELDS))
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    with open(logger.capture_log_path, "x", newline="") as f:
        f.write(f"# session_iso_utc={session_iso_utc}\n")
        w = csv.DictWriter(f, fieldnames=list(CAPTURE_LOG_FIELDS))
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    return logger


def append_chain_row(logger: SessionLogger, row: dict) -> None:
    """Append one chain_log row, flush + fsync for per-row durability."""
    with open(logger.chain_log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CHAIN_LOG_FIELDS))
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def append_capture_row(logger: SessionLogger, row: dict) -> None:
    """Append one capture_log row, flush + fsync for per-row durability."""
    with open(logger.capture_log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CAPTURE_LOG_FIELDS))
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
