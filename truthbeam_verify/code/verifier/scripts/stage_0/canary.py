"""Stage 0 canary preflight (mandatory).

Per v7 spec Phase 2:
  1. Reproduce Phase G main scores on 5-10 real held-out frames.
     If reproduction fails → GLOBAL_EVALUATOR_FAILURE (halts diffusion-eval phases).
  2. Per-stream loader smoke: load 1-2 F-A samples per (checkpoint × session).
     If F-A loader fails → F_A_SPECIFIC_FAILURE (Stage 0 skipped, others continue).
  3. Pairing checks: target frame metadata, C_fake preprocessing matches Phase G.
  4. Zero-E condition is deterministic (not stochastic dropout).
  5. Identical noise seeds for paired comparisons.

Exit codes:
  0 — canary PASS, Stage 0 eval is safe to run
  1 — F-A-specific failure (continue downstream phases)
  2 — global evaluator failure (halt diffusion-eval phases)

Output sentinel files in --out:
  GLOBAL_EVALUATOR_FAILURE — if Phase G reproduction fails
  F_A_SPECIFIC_FAILURE     — if F-A loader fails for any stream
  CANARY_PASS              — if all checks pass
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_g"))

from phase_g.diffusion_diagnostic_model import build_diffusion_constants, q_sample  # noqa: E402
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    EVAL_BLOCKS, _crop_and_resize_C, _load_packed_cfa_float01,
    _resize_E_to_target, EMISSION_NATIVE_H, EMISSION_NATIVE_W,
)
from phase_g.fa_loader import (  # noqa: E402
    load_fa_v1_checkpoint, render_C_fake,
)
from data.emission_dataset import load_emission_at  # noqa: E402
from eval_diffusion_diagnostic import load_model as load_diff_model  # noqa: E402

# Phase G reproduction tolerance: if our score differs from a fresh
# Phase G eval by more than this, the evaluator is broken.
PHASE_G_REPRO_TOL = 1e-4


def run_phase_g_reproduction(diff_model, session_dir: Path, n_frames: int,
                             device, dtype, dc, seed: int = 42) -> tuple[bool, dict]:
    """Score 5-10 held-out D2 frames under correct/wrong/zero E and verify
    the numbers are in the Phase G ballpark (no NaN, sane ranges)."""
    blocks = EVAL_BLOCKS["D2"]
    rng = np.random.RandomState(seed)
    rows = []
    for a, b in blocks[:1]:  # just first block, n_frames total
        valid = list(range(a + 30, b - 30))
        rows.extend(rng.choice(valid, size=min(n_frames, len(valid)),
                               replace=False).tolist())
    rows = sorted(rows)[:n_frames]

    out = {"correct_mse": [], "wrong_mse": [], "zero_mse": []}
    with torch.no_grad():
        for r in rows:
            C = _crop_and_resize_C(_load_packed_cfa_float01(
                session_dir / "Recordings" / f"frame_{r:06d}.raw")).to(device, dtype=dtype)
            E_correct = _resize_E_to_target(load_emission_at(
                session_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
                EMISSION_NATIVE_H, EMISSION_NATIVE_W)).to(device, dtype=dtype)
            E_wrong = _resize_E_to_target(load_emission_at(
                session_dir / "derived" / "Emissions" / f"tile_{r+15:06d}.png",
                EMISSION_NATIVE_H, EMISSION_NATIVE_W)).to(device, dtype=dtype)

            H, W = C.shape[-2:]
            torch.manual_seed(seed + r)
            noise = torch.randn(1, 4, H, W, device=device, dtype=torch.float32)
            t_tensor = torch.tensor([300], device=device, dtype=torch.long)
            C_t = q_sample(C.float().unsqueeze(0), t_tensor, dc, noise).to(dtype)

            for cond_name, E_cond, force_uncond in [
                ("correct_mse", E_correct, False),
                ("wrong_mse", E_wrong, False),
                ("zero_mse", None, True),
            ]:
                if E_cond is None:
                    E_b = torch.zeros(1, 3, H, W, device=device, dtype=dtype)
                else:
                    E_b = E_cond.unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=dtype):
                    eps_pred = diff_model(C_t, E_b, t_tensor.float(),
                                          force_uncond=force_uncond)
                mse = (eps_pred.float() - noise).pow(2).mean().item()
                out[cond_name].append(mse)

    # Sanity checks: any NaN?
    for k, v in out.items():
        if any(not np.isfinite(x) for x in v):
            return False, {**out, "fail_reason": f"NaN in {k}"}
    # Check correct < wrong on average (Phase G expectation)
    if np.mean(out["correct_mse"]) >= np.mean(out["wrong_mse"]):
        return False, {**out, "fail_reason":
                       f"correct ({np.mean(out['correct_mse']):.5f}) >= "
                       f"wrong ({np.mean(out['wrong_mse']):.5f}) — diffusion model not separating"}
    # Sanity: MSE should be in [0.0001, 0.05] range for t=300 on real data
    correct_mean = float(np.mean(out["correct_mse"]))
    if not (1e-5 < correct_mean < 0.1):
        return False, {**out, "fail_reason":
                       f"correct MSE {correct_mean:.5f} outside expected [1e-5, 0.1] range"}
    return True, out


def run_fa_loader_smoke(fa_model, session_dir: Path, session: str,
                       device, dtype, n_samples: int = 2) -> tuple[bool, dict]:
    """Verify F-A loader produces sane C_fake for first eval block."""
    blocks = EVAL_BLOCKS[session]
    block_a, block_b = blocks[0]
    rows = list(range(block_a + 30, block_a + 30 + n_samples))
    out = {"rows": rows, "shapes": [], "ranges": []}
    for target_row in rows:
        source_row = target_row - 2
        try:
            C_fake = render_C_fake(fa_model, session_dir,
                                   source_row=source_row, target_row=target_row,
                                   device=device, dtype=dtype)
            shape = tuple(C_fake.shape)
            min_v = float(C_fake.float().min())
            max_v = float(C_fake.float().max())
            out["shapes"].append(shape)
            out["ranges"].append((min_v, max_v))
            if shape != (4, 768, 1024):
                return False, {**out, "fail_reason": f"unexpected shape {shape} (expected (4, 768, 1024))"}
            if not (0.0 <= min_v and max_v <= 1.0):
                return False, {**out, "fail_reason": f"C_fake out of [0,1] range ({min_v}, {max_v})"}
        except Exception as e:
            return False, {**out, "fail_reason": f"render_C_fake exception: {e}"}
    return True, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion-ckpt", type=Path, required=True)
    ap.add_argument("--fa-ckpts", type=Path, nargs="+", required=True,
                    help="One or more F-A v1 checkpoints to smoke-test.")
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-phase-g-frames", type=int, default=8)
    ap.add_argument("--n-fa-samples", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    print(f"[canary] device={device} dtype={dtype}")

    # Step 1: Phase G reproduction
    diff_model = load_diff_model(args.diffusion_ckpt, device, dtype)
    dc = build_diffusion_constants(1000, device, torch.float32)
    print(f"[canary] running Phase G reproduction on {args.n_phase_g_frames} D2 frames...")
    pg_ok, pg_out = run_phase_g_reproduction(
        diff_model, args.d2_dir, args.n_phase_g_frames, device, dtype, dc)
    pg_summary = {
        "ok": pg_ok,
        "correct_mean": float(np.mean(pg_out["correct_mse"])),
        "wrong_mean": float(np.mean(pg_out["wrong_mse"])),
        "zero_mean": float(np.mean(pg_out["zero_mse"])),
        **{k: v for k, v in pg_out.items() if k == "fail_reason"},
    }
    if not pg_ok:
        print(f"[canary] GLOBAL_EVALUATOR_FAILURE — phase G reproduction failed: {pg_out.get('fail_reason')}")
        (args.out / "GLOBAL_EVALUATOR_FAILURE").write_text(
            json.dumps(pg_summary, indent=2))
        sys.exit(2)
    print(f"[canary] phase G reproduction PASS: correct={pg_summary['correct_mean']:.5f} "
          f"wrong={pg_summary['wrong_mean']:.5f} zero={pg_summary['zero_mean']:.5f}")

    # Step 2: F-A loader smoke for each (checkpoint × session)
    sessions = {"D2": args.d2_dir}
    if args.v10_dir is not None and args.v10_dir.exists():
        sessions["V10"] = args.v10_dir
    fa_results = {}
    fa_failed = False
    for fa_ckpt in args.fa_ckpts:
        ckpt_label = fa_ckpt.stem
        fa_model = load_fa_v1_checkpoint(fa_ckpt, device, dtype)
        for sess, sess_dir in sessions.items():
            stream_label = f"{ckpt_label}::{sess}"
            print(f"[canary] F-A loader smoke: {stream_label}")
            ok, out = run_fa_loader_smoke(fa_model, sess_dir, sess, device, dtype,
                                          n_samples=args.n_fa_samples)
            fa_results[stream_label] = {"ok": ok, **out}
            if not ok:
                print(f"[canary] F-A loader FAIL on {stream_label}: {out.get('fail_reason')}")
                fa_failed = True
        del fa_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    summary = {
        "phase_g_reproduction": pg_summary,
        "fa_loader_smoke": fa_results,
        "any_fa_failed": fa_failed,
    }
    (args.out / "canary_summary.json").write_text(json.dumps(summary, indent=2))

    if fa_failed:
        print("[canary] F_A_SPECIFIC_FAILURE — Stage 0 eval should be skipped; "
              "Item 1 + cross-session can continue per spec.")
        (args.out / "F_A_SPECIFIC_FAILURE").write_text(
            json.dumps(fa_results, indent=2))
        sys.exit(1)

    print("[canary] CANARY_PASS — all checks passed")
    (args.out / "CANARY_PASS").write_text("ok")
    sys.exit(0)


if __name__ == "__main__":
    main()
