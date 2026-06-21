"""V2 verifier audit 4: shift-stability control on binder.

Tests for aliasing in the binder's downsampling stack. If the binder is a
suitable canonicalizer, its output should be smooth under sub-pixel shifts
of the input. Discontinuous jumps at integer pixel boundaries indicate
aliasing — anti-aliased downsampling needed in any v2 architecture
built on top of this binder.

Methodology:
  1. Take real C frames (held-out)
  2. For each shift magnitude δ ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0} px:
     - Apply sub-pixel x-axis shift via bilinear interpolation
     - Run shifted C through binder
     - Compute RMSE relative to unshifted output
  3. Plot RMSE vs shift magnitude
  4. Report whether RMSE grows smoothly (good) or jumps at integer boundaries (aliasing)

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
import torch.nn.functional as F

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from compute_phase_e_thresholds import get_binder_specs, load_binder  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


SHIFT_MAGNITUDES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
VAL_RANGE_D2 = (4792, 5392)


def subpixel_shift_x(img: torch.Tensor, dx: float) -> torch.Tensor:
    """Apply sub-pixel x-axis shift via bilinear interpolation. (1, C, H, W) → (1, C, H, W)."""
    _, _, H, W = img.shape
    # Build affine grid for translation
    theta = torch.tensor([[1.0, 0.0, -2.0 * dx / W],
                          [0.0, 1.0, 0.0]], device=img.device,
                         dtype=img.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder", default="exp001c")
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--n-frames", type=int, default=30)
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
    print(f"[init] {len(sample_indices)} frames; {len(SHIFT_MAGNITUDES)} shift magnitudes",
          flush=True)

    rmse_per_mag = {m: [] for m in SHIFT_MAGNITUDES}
    t0 = time.time()
    for fi, ds_idx in enumerate(sample_indices):
        sample = ds[ds_idx]
        cap = sample["capture"].unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            base_pred = binder(cap).float().clamp(0, 1)

        for m in SHIFT_MAGNITUDES:
            if m == 0.0:
                rmse_per_mag[m].append(0.0)
                continue
            cap_shifted = subpixel_shift_x(cap, m)
            with torch.no_grad():
                pred_shifted = binder(cap_shifted).float().clamp(0, 1)
            rmse = float(torch.sqrt(((pred_shifted - base_pred) ** 2).mean()).item())
            rmse_per_mag[m].append(rmse)

        if (fi + 1) % 5 == 0:
            print(f"  frame {fi+1}/{len(sample_indices)}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    summary = {}
    for m, rmses in rmse_per_mag.items():
        summary[m] = {
            "n_frames": len(rmses),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_median": float(np.median(rmses)),
            "rmse_p95": float(np.percentile(rmses, 95)),
        }

    # Detect aliasing: discontinuity at integer boundaries
    # If RMSE at 1.0 is much larger than what linear interp from 0.75 to 1.25 predicts → aliasing
    rmse_vals = [summary[m]["rmse_mean"] for m in SHIFT_MAGNITUDES]
    aliasing_score = None
    if len(SHIFT_MAGNITUDES) >= 5:
        # Check 1.0 px point against neighbors
        idx_1 = SHIFT_MAGNITUDES.index(1.0)
        if 0 < idx_1 < len(SHIFT_MAGNITUDES) - 1:
            r_left = summary[SHIFT_MAGNITUDES[idx_1 - 1]]["rmse_mean"]
            r_right = summary[SHIFT_MAGNITUDES[idx_1 + 1]]["rmse_mean"]
            r_at_1 = summary[1.0]["rmse_mean"]
            interp = (r_left + r_right) / 2
            aliasing_score = (r_at_1 - interp) / (interp + 1e-8)  # how anomalous is 1.0 vs neighbors

    out = {
        "binder": args.binder,
        "n_frames": len(sample_indices),
        "shift_magnitudes": SHIFT_MAGNITUDES,
        "summary": summary,
        "aliasing_score_at_1px": aliasing_score,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "audit_4_shift_stability.json").write_text(json.dumps(out, indent=2))

    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(8, 4))
        means = [summary[m]["rmse_mean"] for m in SHIFT_MAGNITUDES]
        p95s = [summary[m]["rmse_p95"] for m in SHIFT_MAGNITUDES]
        ax.plot(SHIFT_MAGNITUDES, means, "o-", label="mean RMSE")
        ax.plot(SHIFT_MAGNITUDES, p95s, "s--", label="p95 RMSE", alpha=0.6)
        ax.set_xlabel("Sub-pixel shift magnitude (px)")
        ax.set_ylabel(f"RMSE (binder output vs unshifted) — {args.binder}")
        ax.set_title(f"Shift stability — aliasing_score@1px={aliasing_score:.3f}"
                     if aliasing_score is not None else "Shift stability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Mark integer boundaries
        for x in [1.0, 2.0]:
            ax.axvline(x, color="r", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(args.out / "shift_stability_plot.png", dpi=130)
        plt.close(fig)

    md = [
        "# V2 verifier audit 4 — shift-stability control",
        "",
        f"Binder: `{args.binder}`",
        f"Frames analyzed: {len(sample_indices)}",
        f"Sub-pixel shifts (x-axis only): {SHIFT_MAGNITUDES} px",
        "",
        "## Output RMSE vs shift magnitude",
        "",
        "| shift (px) | mean RMSE | median RMSE | p95 RMSE |",
        "|---:|---:|---:|---:|",
    ]
    for m in SHIFT_MAGNITUDES:
        s = summary[m]
        md.append(
            f"| {m:.2f} | {s['rmse_mean']:.5f} | {s['rmse_median']:.5f} | {s['rmse_p95']:.5f} |"
        )
    md += [
        "",
        f"## Aliasing diagnostic",
        "",
        f"`aliasing_score_at_1px` = {aliasing_score:.3f} if aliasing_score is not None else 'n/a'",
        "(deviation of RMSE@1.0px from linear interp of neighbors;"
        " > 0.3 suggests integer-pixel discontinuity = aliasing in downsampling)",
        "",
        "## Interpretation",
        "",
        "If RMSE grows smoothly with shift magnitude → binder is shift-stable, suitable canonicalizer base.",
        "If RMSE jumps discontinuously at 1.0 or 2.0 px boundaries → aliasing in downsampling stack;"
        " v2 architecture needs anti-aliased downsampling.",
        "",
        f"Plot: `shift_stability_plot.png`",
        "",
        f"Elapsed: {out['elapsed_sec']}s",
    ]
    (args.out / "audit_4_shift_stability.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/audit_4_shift_stability.{{json,md,png}}", flush=True)


if __name__ == "__main__":
    main()
