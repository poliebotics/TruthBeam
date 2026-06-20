"""Stage 0 — merge per-shard NPZs into a single summary.json + raw NPZ.

Usage:
  python scripts/stage_0/merge_shards.py --eval-dir <path> --num-shards 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_0"))

# Reuse summarize from stage_0/eval.py (same module-local function)
from eval import summarize  # noqa: E402


def merge_session(eval_dir: Path, session: str, expected_num_shards: int) -> dict | None:
    pattern = f"stage0_{session.lower()}_raw_shard*.npz"
    shard_files = sorted(eval_dir.glob(pattern))
    if not shard_files:
        print(f"[skip] {session}: no shard files matching {pattern}")
        return None

    # Verify expected shard IDs present
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
            f"{found_ids}. Aborting; will not silently produce partial summary.")

    cond_arrays: dict[str, list[np.ndarray]] = {}
    rows_all: list[np.ndarray] = []
    block_idx_all: list[np.ndarray] = []
    c_fake_all: list[np.ndarray] = []   # optional, only present if --save-c-fake
    for f in shard_files:
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("cond_"):
                cond_arrays.setdefault(k[len("cond_"):], []).append(z[k])
        rows_all.append(z["rows"])
        block_idx_all.append(z["block_idx"])
        if "c_fake_tensor" in z.files:
            c_fake_all.append(z["c_fake_tensor"])

    merged_results = {k: np.concatenate(v, axis=0) for k, v in cond_arrays.items()}
    merged_rows = np.concatenate(rows_all, axis=0)
    merged_blocks = np.concatenate(block_idx_all, axis=0)
    order = np.argsort(merged_rows)
    for k in merged_results:
        merged_results[k] = merged_results[k][order]
    merged_rows = merged_rows[order]
    merged_blocks = merged_blocks[order]
    extras = {}
    if c_fake_all:
        if len(c_fake_all) != len(shard_files):
            # silent partial-merge would produce
            # truncated c_fake output that doesn't match `rows` length. Halt
            # loudly instead.
            raise SystemExit(
                f"[merge:{session}] ERROR — c_fake_tensor present in "
                f"{len(c_fake_all)}/{len(shard_files)} shards. Inconsistent "
                "shard runs detected. Re-run all shards with the same "
                "--save-c-fake setting before merging.")
        merged_c_fake = np.concatenate(c_fake_all, axis=0)
        merged_c_fake = merged_c_fake[order]
        extras["c_fake_tensor"] = merged_c_fake

    print(f"[merge] {session}: {len(shard_files)} shards → {merged_rows.size} frames"
          f"{'  (c_fake persisted)' if 'c_fake_tensor' in extras else ''}")

    np.savez(eval_dir / f"stage0_{session.lower()}_raw.npz",
             **{f"cond_{k}": v for k, v in merged_results.items()},
             rows=merged_rows, block_idx=merged_blocks,
             **extras)

    eval_data = {
        "session": session,
        "rows": merged_rows.tolist(),
        "block_idx": merged_blocks.tolist(),
        "results": merged_results,
    }
    return summarize(eval_data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()

    out: dict = {"sessions": {}}
    for sess in ("D2", "V10"):
        s = merge_session(args.eval_dir, sess, args.num_shards)
        if s is not None:
            out["sessions"][sess] = s

    (args.eval_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.eval_dir}/summary.json")
    for sess, sd in out["sessions"].items():
        n = sd.get("n_frames")
        au = sd.get("auroc", {}).get("real_correct_vs_fake_correct", float("nan"))
        gap = sd.get("deltas", {}).get("paired_gap_correct", {}).get("mean", float("nan"))
        print(f"  {sess}: n={n}, AUROC real-vs-fake = {au:.3f}, paired_gap_correct = {gap:+.5f}")


if __name__ == "__main__":
    main()
