"""V2 verifier audit 2: error decomposition (geometric vs photometric).

For each held-out (C, E) pair:
  1. Compute baseline binder error: per-region MSE(binder(C), E)
  2. For each shift δ ∈ {-3,...,3} px in (dy, dx) (49 shifts):
     - Roll binder(C) by δ
     - Recompute per-region MSE
  3. Find best shift per region
  4. Report baseline error, best-shift error, improvement, best shift

Geometric error component: how much error reduces under simple translation.
Photometric component: residual error after best translation.

CPU only.
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

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from compute_phase_e_thresholds import get_binder_specs, load_binder  # noqa: E402


SHIFT_RANGE = 3
VAL_RANGE_D2 = (4792, 5392)


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate (3, H, W) image by (dy, dx) px. Borders cropped."""
    H, W = img.shape[1:]
    out = np.zeros_like(img)
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = img[:, src_y0:src_y1, src_x0:src_x1]
    return out


def region_mask_from_brightness(em: np.ndarray) -> dict:
    """Per-pixel region membership based on E's brightness."""
    em_g = em.mean(axis=0)  # (H, W)
    masks = {
        "subject_high":  (em_g > 0.6).astype(np.float32),
        "subject_medium":((em_g > 0.3) & (em_g <= 0.6)).astype(np.float32),
        "background":    ((em_g > 0.1) & (em_g <= 0.3)).astype(np.float32),
        "low_energy":    (em_g <= 0.1).astype(np.float32),
    }
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder", default="exp001c")
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--n-frames", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    specs = get_binder_specs(args.experiments_root)
    spec = specs[args.binder]
    print(f"[init] binder={args.binder}", flush=True)
    binder = load_binder(spec, device, dtype)

    rs, re = VAL_RANGE_D2
    ds = EmissionDataset(
        session_dir=args.d2_dir, row_start=rs, row_end=re,
        capture_h=spec["capture_h"], capture_w=spec["capture_w"],
        emission_h=1080, emission_w=1920, session_id="D2", augment=False,
    )
    rng = np.random.RandomState(0)
    sample_indices = sorted(rng.choice(len(ds),
                                       size=min(args.n_frames, len(ds)),
                                       replace=False).tolist())
    print(f"[init] {len(sample_indices)} frames sampled", flush=True)

    # Per-region accumulators of baseline_mse, best_mse, best_shift_dy, best_shift_dx
    region_data = {r: {"baseline_mses": [], "best_mses": [], "best_dy": [], "best_dx": []}
                   for r in ("subject_high", "subject_medium", "background", "low_energy")}

    t0 = time.time()
    shifts = [(dy, dx) for dy in range(-SHIFT_RANGE, SHIFT_RANGE + 1)
                       for dx in range(-SHIFT_RANGE, SHIFT_RANGE + 1)]

    for fi, ds_idx in enumerate(sample_indices):
        sample = ds[ds_idx]
        cap = sample["capture"].unsqueeze(0)
        em_real = sample["emission"].numpy()  # (3, H, W)
        with torch.no_grad():
            pred_em = binder(cap).float().clamp(0, 1).squeeze(0).numpy()

        masks = region_mask_from_brightness(em_real)

        for region, mask in masks.items():
            mask_sum = mask.sum() + 1e-8
            # Baseline (no shift)
            sq_err = ((pred_em - em_real) ** 2).mean(axis=0)
            base_mse = float((sq_err * mask).sum() / mask_sum)
            # Try all shifts
            best_mse = base_mse
            best_dy, best_dx = 0, 0
            for dy, dx in shifts:
                if dy == 0 and dx == 0:
                    continue
                shifted = shift_image(pred_em, dy, dx)
                sq_err = ((shifted - em_real) ** 2).mean(axis=0)
                mse = float((sq_err * mask).sum() / mask_sum)
                if mse < best_mse:
                    best_mse = mse
                    best_dy, best_dx = dy, dx
            region_data[region]["baseline_mses"].append(base_mse)
            region_data[region]["best_mses"].append(best_mse)
            region_data[region]["best_dy"].append(best_dy)
            region_data[region]["best_dx"].append(best_dx)

        if (fi + 1) % 10 == 0:
            print(f"  frame {fi+1}/{len(sample_indices)}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    summary = {}
    for region, d in region_data.items():
        if not d["baseline_mses"]:
            continue
        base = np.array(d["baseline_mses"])
        best = np.array(d["best_mses"])
        improvement = (base - best) / np.maximum(base, 1e-8)
        # Best shift magnitude
        shift_mag = np.sqrt(np.array(d["best_dy"]) ** 2 + np.array(d["best_dx"]) ** 2)
        summary[region] = {
            "n_frames": len(base),
            "baseline_rmse_median": float(np.sqrt(np.median(base))),
            "best_shift_rmse_median": float(np.sqrt(np.median(best))),
            "improvement_pct_median": float(100 * np.median(improvement)),
            "improvement_pct_p95": float(100 * np.percentile(improvement, 95)),
            "best_shift_mag_px_median": float(np.median(shift_mag)),
            "best_shift_mag_px_p95": float(np.percentile(shift_mag, 95)),
            "frac_with_zero_shift": float((shift_mag == 0).mean()),
        }

    out = {
        "binder": args.binder,
        "n_frames": len(sample_indices),
        "shift_range_px": SHIFT_RANGE,
        "region_summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "audit_2_error_decomposition.json").write_text(json.dumps(out, indent=2))

    md = [
        "# V2 verifier audit 2 — error decomposition",
        "",
        f"Binder: `{args.binder}`",
        f"Frames analyzed: {len(sample_indices)}",
        f"Shift search: ±{SHIFT_RANGE} px ({(2*SHIFT_RANGE+1)**2} candidate shifts).",
        "",
        "## Per-region error decomposition",
        "",
        "| region | n_frames | baseline RMSE | post-shift RMSE | median improvement % | p95 improvement % | median best-shift mag (px) | frac_no_shift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("subject_high", "subject_medium", "background", "low_energy"):
        if region not in summary:
            continue
        s = summary[region]
        md.append(
            f"| {region} | {s['n_frames']} | "
            f"{s['baseline_rmse_median']:.4f} | "
            f"{s['best_shift_rmse_median']:.4f} | "
            f"{s['improvement_pct_median']:.1f}% | "
            f"{s['improvement_pct_p95']:.1f}% | "
            f"{s['best_shift_mag_px_median']:.1f} | "
            f"{s['frac_with_zero_shift']:.2f} |"
        )
    md += [
        "",
        "## Reading the table",
        "",
        "- **baseline RMSE** = error at zero shift = the binder's natural error.",
        "- **post-shift RMSE** = error after best 7×7-grid translation.",
        "- **median improvement %** = (baseline² - best²)/baseline² × 100. High = error is geometric (alignment-correctable).",
        "- **best-shift mag** = pixel distance of best shift from origin. Reveals systematic mis-alignment.",
        "- **frac_no_shift** = fraction of frames where baseline (zero shift) was already optimal.",
        "",
        "Interpretation:",
        "- High improvement (>30%) = error is dominantly geometric. Canonicalization-refiner viable.",
        "- Low improvement (<10%) = error is dominantly photometric. Alignment refinement won't help much; v2 architecture needs photometric correction not just spatial.",
        "",
        f"Elapsed: {out['elapsed_sec']}s",
    ]
    (args.out / "audit_2_error_decomposition.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/audit_2_error_decomposition.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()
