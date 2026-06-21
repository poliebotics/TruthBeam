"""Phase E threshold calibration for Phase F evaluation pre-registration.

For each Phase E binder ckpt, run inference on the threshold_calibration_normal
slice (D2 [5392, 5992)) and compute the matched-pair score distribution.
Output frozen thresholds:

  τ95_psnr = 5th percentile of matched_psnr  (95% TAR threshold, higher-better convention)
  τ99_psnr = 1st percentile of matched_psnr  (99% TAR threshold)
  τ95_mse  = 95th percentile of matched_mse  (same threshold, lower-better convention)
  τ99_mse  = 99th percentile of matched_mse

These are pre-registered BEFORE any attack evaluation. The Phase F editor's
attack pairs are accepted by binder b iff their score crosses τ for that binder.

This script processes one binder at a time (`--binder <name>`). Run multiple
in parallel via different GPUs by setting `CUDA_VISIBLE_DEVICES`.

Output per binder:
  experiments/phase_e/<binder>/thresholds.json

Aggregation:
  experiments/phase_e/PHASE_E_THRESHOLDS.json (combines all binders)

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/phase_f/compute_phase_e_thresholds.py \
    --binder e1 --bf16 \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --out-root /path/to/poliebotics_phase_b/experiments/phase_e
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2
except Exception:
    EmissionPredictorV2 = None


# Threshold-calibration slice per `experiments/phase_f_prep/three_role_split.md`.
THRESHOLD_CALIB_SLICE_D2 = (5392, 5992)


# Binder registry: maps short name → spec
def get_binder_specs(experiments_root: Path) -> dict:
    """Phase E binders + exp001c. Resolution and arch per binder."""
    return {
        "exp001c": {
            "ckpt": experiments_root / "exp001c/checkpoints/ep027.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1": {
            "ckpt": experiments_root / "phase_e/e1/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_bf16_sg": {
            "ckpt": experiments_root / "phase_e/e1_bf16_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_fp16_ddp": {
            "ckpt": experiments_root / "phase_e/e1_fp16_ddp/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_fp16_sg": {
            "ckpt": experiments_root / "phase_e/e1_fp16_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_lr2e4_sg": {
            "ckpt": experiments_root / "phase_e/e1_lr2e4_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_no_pretrained_sg": {
            "ckpt": experiments_root / "phase_e/e1_no_pretrained_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_no_warmup_sg": {
            "ckpt": experiments_root / "phase_e/e1_no_warmup_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e1_seed42_sg": {
            "ckpt": experiments_root / "phase_e/e1_seed42_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
        },
        "e2": {
            "ckpt": experiments_root / "phase_e/e2/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 2300, "capture_w": 2660,
        },
        "e2_fp16": {
            "ckpt": experiments_root / "phase_e/e2_fp16/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictor", "capture_h": 2300, "capture_w": 2660,
        },
        "e3r": {
            "ckpt": experiments_root / "phase_e/e3r/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictorV2", "capture_h": 1150, "capture_w": 1330,
        },
        "e3r_fp16_ddp": {
            "ckpt": experiments_root / "phase_e/e3r_fp16_ddp/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictorV2", "capture_h": 1150, "capture_w": 1330,
        },
        "e3r_fp16_sg": {
            "ckpt": experiments_root / "phase_e/e3r_fp16_sg/checkpoints/best_by_psnr.pt",
            "arch": "EmissionPredictorV2", "capture_h": 1150, "capture_w": 1330,
        },
    }


def load_binder(spec: dict, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    if spec["arch"] == "EmissionPredictor":
        m = EmissionPredictor(emission_h=1080, emission_w=1920, pretrained=False)
    elif spec["arch"] == "EmissionPredictorV2":
        if EmissionPredictorV2 is None:
            raise RuntimeError("EmissionPredictorV2 import failed")
        m = EmissionPredictorV2(emission_h=1080, emission_w=1920, pretrained=False)
    else:
        raise ValueError(spec["arch"])
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m = m.to(device)
    return m


def psnr01(p: torch.Tensor, t: torch.Tensor) -> float:
    mse = ((p - t) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder", required=True, help="Binder name (key in registry)")
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/phase_e"))
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    specs = get_binder_specs(args.experiments_root)
    if args.binder not in specs:
        raise SystemExit(f"unknown binder {args.binder!r}; choices: {list(specs.keys())}")
    spec = specs[args.binder]
    if not spec["ckpt"].exists():
        raise SystemExit(f"ckpt missing: {spec['ckpt']}")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"[init] binder={args.binder} arch={spec['arch']} "
          f"capture_hw={spec['capture_h']}x{spec['capture_w']}", flush=True)
    binder = load_binder(spec, device, dtype)

    # threshold_calibration_normal slice on D2
    rs, re = THRESHOLD_CALIB_SLICE_D2
    ds = EmissionDataset(
        session_dir=args.d2_dir, row_start=rs, row_end=re,
        capture_h=spec["capture_h"], capture_w=spec["capture_w"],
        emission_h=1080, emission_w=1920, session_id="D2", augment=False,
    )
    print(f"[init] threshold_calibration slice rows=[{rs}, {re})  n={len(ds)}", flush=True)

    psnrs: list[float] = []
    mses: list[float] = []
    per_row: list[dict] = []
    t0 = time.time()
    for i in range(len(ds)):
        sample = ds[i]
        cap = sample["capture"].unsqueeze(0).to(device, dtype=dtype)
        em_target = sample["emission"].to(device, dtype=torch.float32)  # (3, 1080, 1920)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            pred = binder(cap).float().clamp(0, 1).squeeze(0)  # (3, 1080, 1920)
        mse = ((pred - em_target) ** 2).mean().item()
        psnr = float("inf") if mse < 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))
        psnrs.append(psnr)
        mses.append(mse)
        per_row.append({"t": int(sample["t"]), "matched_psnr": psnr, "matched_mse": mse})
        if (i + 1) % 50 == 0:
            print(f"  row {i+1}/{len(ds)}  rolling matched_psnr_mean={np.mean(psnrs):.3f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    psnr_arr = np.array([p for p in psnrs if not math.isinf(p)], dtype=np.float64)
    mse_arr = np.array(mses, dtype=np.float64)

    # Percentiles for both conventions
    def pct(arr, p):
        return float(np.percentile(arr, p))

    out = {
        "binder": args.binder,
        "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()},
        "slice": {"session": "D2", "row_start": rs, "row_end": re,
                  "role": "threshold_calibration_normal"},
        "n_rows": int(len(per_row)),
        "n_finite_psnr": int(psnr_arr.size),
        "psnr_distribution": {
            "mean": float(psnr_arr.mean()),
            "std":  float(psnr_arr.std()),
            "min":  float(psnr_arr.min()),
            "max":  float(psnr_arr.max()),
            "p1":   pct(psnr_arr, 1),
            "p5":   pct(psnr_arr, 5),
            "p10":  pct(psnr_arr, 10),
            "p25":  pct(psnr_arr, 25),
            "p50":  pct(psnr_arr, 50),
            "p75":  pct(psnr_arr, 75),
            "p90":  pct(psnr_arr, 90),
            "p95":  pct(psnr_arr, 95),
            "p99":  pct(psnr_arr, 99),
        },
        "mse_distribution": {
            "mean": float(mse_arr.mean()),
            "std":  float(mse_arr.std()),
            "min":  float(mse_arr.min()),
            "max":  float(mse_arr.max()),
            "p1":   pct(mse_arr, 1),
            "p5":   pct(mse_arr, 5),
            "p10":  pct(mse_arr, 10),
            "p25":  pct(mse_arr, 25),
            "p50":  pct(mse_arr, 50),
            "p75":  pct(mse_arr, 75),
            "p90":  pct(mse_arr, 90),
            "p95":  pct(mse_arr, 95),
            "p99":  pct(mse_arr, 99),
        },
        "thresholds": {
            "tau95_psnr_higher_better": pct(psnr_arr, 5),
            "tau99_psnr_higher_better": pct(psnr_arr, 1),
            "tau95_mse_lower_better":   pct(mse_arr, 95),
            "tau99_mse_lower_better":   pct(mse_arr, 99),
        },
        "interpretation": (
            "tau95_psnr_higher_better = score above which 95% of REAL pairs lie. "
            "Equivalent: tau95_mse_lower_better = score below which 95% of REAL pairs lie. "
            "Phase F editor attacks must clear this threshold against held-out binders."
        ),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # Per-binder output
    binder_out_dir = args.out_root / args.binder
    binder_out_dir.mkdir(parents=True, exist_ok=True)
    (binder_out_dir / "thresholds.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {binder_out_dir}/thresholds.json", flush=True)
    print(f"  τ95_psnr = {out['thresholds']['tau95_psnr_higher_better']:.3f} dB", flush=True)
    print(f"  τ99_psnr = {out['thresholds']['tau99_psnr_higher_better']:.3f} dB", flush=True)
    print(f"  matched_psnr_mean = {out['psnr_distribution']['mean']:.3f} dB", flush=True)


if __name__ == "__main__":
    main()
