"""Phase G — recompute summary.json from existing raw NPZs.

Use after fixing eval_diffusion_diagnostic.summarize() / auroc_pooled() to
regenerate per-run summary.json without re-running the (expensive) eval.

Usage:
  python scripts/phase_g/resummarize_eval.py --eval-dir <path>
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

import eval_diffusion_diagnostic as ed  # noqa: E402


def resummarize_session(eval_dir: Path, session: str) -> dict | None:
    npz_path = eval_dir / f"eval_{session.lower()}_raw.npz"
    if not npz_path.exists():
        print(f"[skip] {session}: no {npz_path}")
        return None
    z = np.load(npz_path, allow_pickle=False)
    results = {k[len("cond_"):]: z[k] for k in z.files if k.startswith("cond_")}
    rows = z["rows"]
    block_idx = z["block_idx"]
    eval_data = {
        "session": session,
        "rows": rows.tolist(),
        "block_idx": block_idx.tolist(),
        "results": results,
        "timesteps": list(ed.ALL_TIMESTEPS),
        "wrong_offsets": list(ed.WRONG_OFFSETS),
        "lag_ks": list(ed.LAG_SWEEP_KS),
        "K_noise": ed.K_NOISE,
    }
    s = ed.summarize(eval_data)
    print(f"[done] {session}: n={s['n_frames']} "
          f"AUROC vs wrong = {s.get('auroc',{}).get('correct_vs_wrong_avg',float('nan')):.4f}  "
          f"Δ_wrong = {s.get('deltas',{}).get('delta_wrong',{}).get('mean',float('nan')):+.6f}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    args = ap.parse_args()
    out: dict = {"sessions": {}}
    for sess in ("D2", "V10"):
        s = resummarize_session(args.eval_dir, sess)
        if s is not None:
            out["sessions"][sess] = s
    (args.eval_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.eval_dir}/summary.json")


if __name__ == "__main__":
    main()
