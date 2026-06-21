"""Bit-exact validation: CPU vs GPU output of render_E_for_phase_h.

The bitexact_renderer is designed device-agnostic with integer math and claims
bit-exact output across devices. This script verifies empirically by rendering
the same (session, row, condition) inputs on both devices and comparing.

Acceptance: max abs diff == 0 across all (session, row, condition) tested.
Anything > 0 indicates a non-deterministic op or a bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.xof_perturb import (  # noqa: E402
    render_E_for_phase_h, load_chain_log,
    TRAIN_POOL_LABELS, HELDOUT_POOL_LABELS, _spec_by_label,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"))
    ap.add_argument("--v10-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"))
    ap.add_argument("--n-frames", type=int, default=5,
                    help="frames per session to validate")
    ap.add_argument("--gpu-device", type=str, default="cuda:0")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA")
        sys.exit(2)

    chain_d2 = load_chain_log(args.d2_dir)
    chain_v10 = load_chain_log(args.v10_dir) if args.v10_dir.exists() else {}

    sessions = []
    if chain_d2:
        sessions.append(("D2", chain_d2, chain_v10))
    if chain_v10:
        sessions.append(("V10", chain_v10, chain_d2))

    # Conditions to test: identity, shuffled (via cross-session), 4 train (one per type),
    # 2 held-out
    test_conditions = [
        "identity",
        "shuffled",  # synthesized below
        "xof_t1_global_k64",
        "xof_t2_oct1_k16",
        "xof_t4_swap_oct2",
        "xof_t5_swap_G",  # heldout
        "xof_t3_region_k64",  # heldout
        "xof_t6_replace_general",
    ]

    rows_per_session = {}
    for sess, chain, _ in sessions:
        keys = sorted(chain.keys())
        # Spread out picks across the session
        idxs = [int(len(keys) * f) for f in (0.15, 0.30, 0.50, 0.70, 0.85)][:args.n_frames]
        rows_per_session[sess] = [keys[i] for i in idxs]

    n_total = 0
    n_exact = 0
    failures = []

    for sess, chain, other_chain in sessions:
        rows = rows_per_session[sess]
        for r in rows:
            for cond in test_conditions:
                # Build kwargs to match eval_baseline.py exactly
                if cond == "identity":
                    kw_cpu = dict(session=sess, target_frame_id=r,
                                  condition_label="identity", chain_log=chain)
                elif cond == "shuffled":
                    if not other_chain:
                        continue
                    other_keys = sorted(other_chain.keys())
                    prow = other_keys[r % len(other_chain)]
                    other_sess = "V10" if sess == "D2" else "D2"
                    kw_cpu = dict(session=other_sess, target_frame_id=prow,
                                  condition_label="identity", chain_log=other_chain)
                else:
                    spec = _spec_by_label(cond)
                    if spec.needs_donor():
                        keys = sorted(chain.keys())
                        idx = keys.index(r)
                        donor_row = keys[(idx + len(keys) // 4) % len(keys)]
                        kw_cpu = dict(session=sess, target_frame_id=r,
                                      condition_label=cond, chain_log=chain,
                                      donor_chain_log=chain,
                                      donor_frame_id=donor_row)
                    else:
                        kw_cpu = dict(session=sess, target_frame_id=r,
                                      condition_label=cond, chain_log=chain)

                # Render on CPU and GPU
                E_cpu = render_E_for_phase_h(**kw_cpu, device="cpu")
                E_gpu = render_E_for_phase_h(**kw_cpu, device=args.gpu_device)

                # Compare — both should be CPU tensors (cv2 resize is CPU)
                # and bit-exact float32
                if E_gpu.device.type != "cpu":
                    print(f"  WARN: E_gpu on {E_gpu.device}; expected cpu after .cpu()")
                    E_gpu = E_gpu.cpu()
                diff = (E_cpu.float() - E_gpu.float()).abs()
                max_diff = float(diff.max())
                mean_diff = float(diff.mean())
                bit_exact = bool((E_cpu == E_gpu).all().item())
                n_total += 1
                if bit_exact:
                    n_exact += 1
                else:
                    failures.append({
                        "session": sess, "row": r, "cond": cond,
                        "max_diff": max_diff, "mean_diff": mean_diff,
                    })
                tag = "EXACT" if bit_exact else f"DIFF max={max_diff:.6e}"
                print(f"  {sess} r={r:6d} {cond:30s} {tag}", flush=True)

    print()
    print(f"=== SUMMARY ===")
    print(f"n_total = {n_total}")
    print(f"n_exact = {n_exact}")
    print(f"n_failures = {len(failures)}")
    if failures:
        print()
        print("Failures:")
        for f in failures[:10]:
            print(f"  {f}")
        sys.exit(1)
    print()
    print("VALIDATION PASS — CPU and GPU renders are bit-exact.")
    sys.exit(0)


if __name__ == "__main__":
    main()
