"""Phase G — merge per-shard eval outputs into a single summary.json + raw NPZ.

Reads:
  <eval_dir>/eval_d2_raw_shard0.npz
  <eval_dir>/eval_d2_raw_shard1.npz
  ...
  <eval_dir>/eval_v10_raw_shard{N}.npz

Writes:
  <eval_dir>/eval_d2_raw.npz   (concatenated)
  <eval_dir>/eval_v10_raw.npz  (concatenated)
  <eval_dir>/summary.json      (recomputed from merged data)

Usage:
  python scripts/phase_g/merge_eval_shards.py --eval-dir <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_g"))

# We re-use summarize() and the constants from the eval script for consistency.
import eval_diffusion_diagnostic as ed  # noqa: E402


def merge_session(eval_dir: Path, session: str, expected_num_shards: int | None) -> dict | None:
    pattern = f"eval_{session.lower()}_raw_shard*.npz"
    shard_files = sorted(eval_dir.glob(pattern))
    if not shard_files:
        print(f"[skip] {session}: no shard files matching {pattern}")
        return None

    # Verify exactly the expected shard IDs are present (0..N-1, no gaps, no dups)
    if expected_num_shards is not None:
        import re
        found_ids = []
        for f in shard_files:
            m = re.search(r"_shard(\d+)\.npz$", f.name)
            if m:
                found_ids.append(int(m.group(1)))
        found_ids.sort()
        expected_ids = list(range(expected_num_shards))
        if found_ids != expected_ids:
            raise SystemExit(
                f"[merge:{session}] ERROR — expected shards {expected_ids} but found "
                f"{found_ids}. Missing or duplicate shard files. Aborting; will not "
                f"silently produce a partial summary.json."
            )

    # Load all shards
    cond_arrays: dict[str, list[np.ndarray]] = {}
    rows_all: list[np.ndarray] = []
    block_idx_all: list[np.ndarray] = []
    for f in shard_files:
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("cond_"):
                cond_label = k[len("cond_"):]
                cond_arrays.setdefault(cond_label, []).append(z[k])
        rows_all.append(z["rows"])
        block_idx_all.append(z["block_idx"])

    # Concatenate along frame axis
    merged_results = {k: np.concatenate(v, axis=0) for k, v in cond_arrays.items()}
    merged_rows = np.concatenate(rows_all, axis=0)
    merged_blocks = np.concatenate(block_idx_all, axis=0)

    # Sort by row (frame index) so block order is preserved
    order = np.argsort(merged_rows)
    for k in merged_results:
        merged_results[k] = merged_results[k][order]
    merged_rows = merged_rows[order]
    merged_blocks = merged_blocks[order]

    print(f"[merge] {session}: {len(shard_files)} shards → "
          f"{merged_rows.size} total frames")

    # Save merged raw NPZ
    np.savez(eval_dir / f"eval_{session.lower()}_raw.npz",
             **{f"cond_{k}": v for k, v in merged_results.items()},
             rows=merged_rows, block_idx=merged_blocks)

    # Build the eval_data structure summarize() expects
    eval_data = {
        "session": session,
        "rows": merged_rows.tolist(),
        "block_idx": merged_blocks.tolist(),
        "results": merged_results,
        "timesteps": list(ed.ALL_TIMESTEPS),
        "wrong_offsets": list(ed.WRONG_OFFSETS),
        "lag_ks": list(ed.LAG_SWEEP_KS),
        "K_noise": ed.K_NOISE,
    }
    return ed.summarize(eval_data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--num-shards", type=int, default=None,
                    help="Expected number of shards per session. If set, the merge "
                         "will fail loudly if any shard ID 0..N-1 is missing.")
    args = ap.parse_args()

    out: dict = {"sessions": {}}
    for sess in ("D2", "V10"):
        s = merge_session(args.eval_dir, sess, args.num_shards)
        if s is not None:
            out["sessions"][sess] = s

    (args.eval_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.eval_dir}/summary.json")
    for sess, sd in out["sessions"].items():
        print(f"  {sess}: n={sd.get('n_frames')}, "
              f"AUROC vs wrong = {sd.get('auroc',{}).get('correct_vs_wrong_avg',float('nan')):.3f}")


if __name__ == "__main__":
    main()
