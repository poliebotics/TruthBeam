"""session_finalize.py — post-session, best-effort processing.

Scope: work that runs AFTER the chain has finished — reconciling the
capture_log against the chain_log, encoding debayered preview PNGs from
the raw bayer frames, and any other artifact-shaping that's not on the
chain-integrity critical path.

Adapt this module for:
- Skipping preview encoding entirely on storage-constrained rigs
- Higher-quality debayering (Malvar-He-Cutler vs. the simple nearest-
  neighbour path here)
- Additional per-session reports (photometric summaries, timing summaries)
  — this is the right home for them, not the recording loop

Do NOT put here:
- Chain-hash computation — manifest_hash_final, capture_log_hash,
  chain_log_hash are produced by session_schema helpers called from
  tb_main.main() during the same finalize pass
- The terminal anchor transaction — that's the interior publisher's
  fire_final_anchor() method
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


def save_tile_png_safe(tile_array, out_path, t: int) -> None:
    """Best-effort PNG write of a (1080, 1920, 3) uint8 tile. Called
    inside the async disk-worker path; failures are logged and swallowed
    so a PNG encoder error never aborts a session."""
    try:
        Image.fromarray(tile_array, mode="RGB").save(
            out_path, compress_level=1
        )
    except Exception as e:
        print(f"[png] WARN: tile_{t:06d}.png save failed: {e!r}", flush=True)


def debayer_bayerrg_nn(raw8: np.ndarray) -> np.ndarray:
    """Nearest-neighbour debayer of a BayerRG8 frame to RGB. Preview-only.
    Shape in: (H, W) uint8. Shape out: (H, W, 3) uint8."""
    R = raw8[0::2, 0::2]
    G = raw8[0::2, 1::2]
    B = raw8[1::2, 1::2]
    return np.stack([
        np.repeat(np.repeat(R, 2, axis=0), 2, axis=1),
        np.repeat(np.repeat(G, 2, axis=0), 2, axis=1),
        np.repeat(np.repeat(B, 2, axis=0), 2, axis=1),
    ], axis=-1)


def offline_encode_previews(session_dir, real_roi):
    """After the session ends, walk Recordings/*.raw and write a
    debayered PNG into derived/Recordings_previews/ for each. Returns
    (n_encoded, elapsed_s)."""
    rx, ry, rw, rh = real_roi
    session_dir = Path(session_dir)
    recordings_dir = session_dir / "Recordings"
    previews_dir = session_dir / "derived" / "Recordings_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.monotonic()
    for raw_path in sorted(recordings_dir.glob("frame_*.raw")):
        data = raw_path.read_bytes()
        raw8 = np.frombuffer(data, dtype=np.uint8).reshape(rh, rw)
        rgb = debayer_bayerrg_nn(raw8)
        out_path = previews_dir / (raw_path.stem + ".png")
        Image.fromarray(rgb, mode="RGB").save(out_path, compress_level=1)
        n += 1
    return n, time.monotonic() - t0


def reconcile_capture_log(session_dir, chain_log_path, capture_log_path):
    """After the session ends, rewrite capture_log.csv with the true
    consumed_into_chain / consumed_as_t columns for each row. Returns
    (M, per_bayer_blake3_t_map). Pure file rewrite; the chain_log rows are
    authoritative and unchanged."""
    bayer_to_t = {}
    with open(chain_log_path) as f:
        f.readline()  # comment header
        reader = csv.DictReader(f)
        for row in reader:
            bayer_to_t[row["bayer_blake3_hex"]] = int(row["t"])

    if not capture_log_path.exists():
        return 0, bayer_to_t
    with open(capture_log_path) as f:
        header_comment = f.readline()
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda r: int(r["capture_idx"]))
    fieldnames = reader.fieldnames
    tmp_path = capture_log_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="") as f:
        f.write(header_comment)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            consumed_t = bayer_to_t.get(r["bayer_blake3_hex"])
            r["consumed_into_chain"] = "true" if consumed_t is not None else "false"
            r["consumed_as_t"] = consumed_t if consumed_t is not None else -1
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, capture_log_path)
    try:
        dfd = os.open(session_dir, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return len(rows), bayer_to_t
