"""V2 verifier audit 1: feature-space residual alignment test.

For best Phase E binder (defaults to exp001c — half-res, fastest on CPU):
1. Run binder on held-out (C, E) pairs from D2 selection_gate_normal [4792, 5392)
2. Take binder's predicted emission as the "E-shaped representation"
3. Pick N informative patches per frame (high-variance regions of E)
4. For each patch, compute local correlation surface over ±4 px shift window
5. Record peak offset (local bias) and FWHM (local precision)
6. Aggregate by region (high / medium / low brightness via E's intensity)

Output: audit_1_alignment.{json,md} with median + P95 peak offset and
FWHM per region.

CPU only. Estimated 30-60 min for 100 frames.
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


PATCH_SIZE = 32
SHIFT_RANGE = 4   # ±4 px window
N_PATCHES_PER_FRAME = 200
VAL_RANGE_D2 = (4792, 5392)


def pick_informative_patches(em: torch.Tensor, n: int, patch_size: int,
                              shift_pad: int, rng: np.random.RandomState) -> list[tuple[int, int]]:
    """Return list of (y, x) top-left positions of patches in informative regions.
    Informative = high variance in the patch. Avoids edges by `shift_pad`."""
    em_g = em.mean(dim=0).numpy()  # (H, W) grayscale
    H, W = em_g.shape
    margin = patch_size // 2 + shift_pad + 1
    valid_h = range(margin, H - margin)
    valid_w = range(margin, W - margin)

    # Compute local variance map at coarse stride
    stride = 16
    h_positions = list(valid_h)[::stride]
    w_positions = list(valid_w)[::stride]
    var_map = np.zeros((len(h_positions), len(w_positions)))
    for i, y in enumerate(h_positions):
        for j, x in enumerate(w_positions):
            patch = em_g[y - patch_size // 2:y + patch_size // 2,
                         x - patch_size // 2:x + patch_size // 2]
            var_map[i, j] = float(patch.var())

    # Pick top-N variance positions; randomly perturb to avoid grid bias
    flat = var_map.flatten()
    top_k = min(n * 3, len(flat))  # oversample then random subset
    top_idx = np.argpartition(-flat, top_k - 1)[:top_k]
    chosen = rng.choice(top_idx, size=min(n, len(top_idx)), replace=False)
    positions = []
    for idx in chosen:
        i, j = idx // var_map.shape[1], idx % var_map.shape[1]
        y = h_positions[i] + rng.randint(-stride // 2, stride // 2 + 1)
        x = w_positions[j] + rng.randint(-stride // 2, stride // 2 + 1)
        y = max(margin, min(H - margin, y))
        x = max(margin, min(W - margin, x))
        positions.append((y, x))
    return positions


def local_correlation_surface(pred_em: np.ndarray, real_em: np.ndarray,
                              cy: int, cx: int, patch_size: int, shift_range: int) -> np.ndarray:
    """Compute correlation between a patch in real_em and shifted patches of pred_em.

    Returns (2*shift_range+1, 2*shift_range+1) correlation map.
    Higher correlation = better alignment at that shift.
    """
    h, w = patch_size // 2, patch_size // 2
    real_patch = real_em[cy - h:cy + h, cx - w:cx + w].flatten()
    real_centered = real_patch - real_patch.mean()
    real_norm = np.linalg.norm(real_centered) + 1e-8

    surf = np.zeros((2 * shift_range + 1, 2 * shift_range + 1), dtype=np.float32)
    for dy in range(-shift_range, shift_range + 1):
        for dx in range(-shift_range, shift_range + 1):
            patch = pred_em[cy + dy - h:cy + dy + h,
                            cx + dx - w:cx + dx + w].flatten()
            patch_centered = patch - patch.mean()
            patch_norm = np.linalg.norm(patch_centered) + 1e-8
            corr = float(np.dot(real_centered, patch_centered)) / (real_norm * patch_norm)
            surf[dy + shift_range, dx + shift_range] = corr
    return surf


def peak_and_fwhm(surf: np.ndarray, shift_range: int) -> tuple[float, float, float, float]:
    """From correlation surface, return (peak_dy, peak_dx, peak_value, fwhm_px)."""
    iy, ix = np.unravel_index(int(surf.argmax()), surf.shape)
    peak_dy = iy - shift_range
    peak_dx = ix - shift_range
    peak_value = float(surf[iy, ix])
    # FWHM in pixels: count cells in the surface above peak/2
    half = peak_value / 2
    above = (surf >= half).sum()
    # Approximate FWHM as sqrt of count (cells are 1 px each since shift step = 1)
    fwhm = math.sqrt(max(above, 1))
    return float(peak_dy), float(peak_dx), peak_value, fwhm


def categorize_region(em: torch.Tensor, cy: int, cx: int, patch_size: int) -> str:
    """Brightness-based region category from real E patch."""
    h = patch_size // 2
    patch = em[:, cy - h:cy + h, cx - h:cx + h].mean().item()
    # Rough thresholds based on emission intensity. Emissions are in [0,1].
    if patch > 0.6:
        return "subject_high"
    elif patch > 0.3:
        return "subject_medium"
    elif patch > 0.1:
        return "background"
    else:
        return "low_energy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder", default="exp001c", help="Phase E binder name")
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--n-patches-per-frame", type=int, default=N_PATCHES_PER_FRAME)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    specs = get_binder_specs(args.experiments_root)
    if args.binder not in specs:
        raise SystemExit(f"unknown binder {args.binder}")
    spec = specs[args.binder]
    if not Path(spec["ckpt"]).exists():
        # Try local g1a path (we mirrored the experiments tree)
        local_alt = ROOT / "experiments" / "phase_e" / args.binder / "checkpoints" / "best_by_psnr.pt"
        if local_alt.exists():
            spec["ckpt"] = local_alt
        else:
            raise SystemExit(f"binder ckpt missing: {spec['ckpt']}")
    print(f"[init] binder={args.binder} arch={spec['arch']} "
          f"capture_hw={spec['capture_h']}x{spec['capture_w']}", flush=True)
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
    print(f"[init] {len(sample_indices)} frames sampled from val", flush=True)

    region_data = {}  # region -> list of (peak_offset_px, fwhm_px, peak_value)
    t0 = time.time()
    for fi, ds_idx in enumerate(sample_indices):
        sample = ds[ds_idx]
        cap = sample["capture"].unsqueeze(0)
        em_real = sample["emission"]  # (3, 1080, 1920)
        with torch.no_grad():
            pred_em = binder(cap).float().clamp(0, 1).squeeze(0)  # (3, 1080, 1920)

        em_real_g = em_real.mean(dim=0).numpy()
        pred_em_g = pred_em.mean(dim=0).numpy()

        positions = pick_informative_patches(em_real, args.n_patches_per_frame,
                                             PATCH_SIZE, SHIFT_RANGE, rng)

        for cy, cx in positions:
            surf = local_correlation_surface(
                pred_em_g, em_real_g, cy, cx, PATCH_SIZE, SHIFT_RANGE)
            peak_dy, peak_dx, peak_value, fwhm = peak_and_fwhm(surf, SHIFT_RANGE)
            peak_offset = math.sqrt(peak_dy ** 2 + peak_dx ** 2)
            region = categorize_region(em_real, cy, cx, PATCH_SIZE)
            region_data.setdefault(region, []).append({
                "peak_offset_px": peak_offset,
                "peak_dy": peak_dy, "peak_dx": peak_dx,
                "fwhm_px": fwhm,
                "peak_value": peak_value,
            })

        if (fi + 1) % 10 == 0:
            print(f"  frame {fi+1}/{len(sample_indices)}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    summary = {}
    for region, rows in region_data.items():
        offsets = [r["peak_offset_px"] for r in rows]
        fwhms = [r["fwhm_px"] for r in rows]
        peaks = [r["peak_value"] for r in rows]
        summary[region] = {
            "n_patches": len(rows),
            "peak_offset_px_median": float(np.median(offsets)),
            "peak_offset_px_p95": float(np.percentile(offsets, 95)),
            "peak_offset_px_mean": float(np.mean(offsets)),
            "fwhm_px_median": float(np.median(fwhms)),
            "fwhm_px_p95": float(np.percentile(fwhms, 95)),
            "peak_value_median": float(np.median(peaks)),
        }

    out = {
        "binder": args.binder,
        "n_frames": len(sample_indices),
        "n_patches_per_frame": args.n_patches_per_frame,
        "patch_size_px": PATCH_SIZE,
        "shift_range_px": SHIFT_RANGE,
        "region_summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "audit_1_alignment.json").write_text(json.dumps(out, indent=2))

    md = [
        f"# V2 verifier audit 1 — feature-space residual alignment",
        "",
        f"Binder: `{args.binder}` (arch: {spec['arch']}, capture {spec['capture_h']}×{spec['capture_w']})",
        f"Frames analyzed: {len(sample_indices)} from D2 selection_gate_normal [{rs}, {re})",
        f"Patches per frame: {args.n_patches_per_frame} (chosen at high-variance positions)",
        f"Patch size: {PATCH_SIZE}×{PATCH_SIZE} px; shift range ±{SHIFT_RANGE} px.",
        "",
        "## Per-region peak offset + FWHM",
        "",
        "| region | n_patches | median offset (px) | p95 offset (px) | mean offset (px) | median FWHM (px) | p95 FWHM (px) | median peak corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("subject_high", "subject_medium", "background", "low_energy"):
        if region not in summary:
            continue
        s = summary[region]
        md.append(
            f"| {region} | {s['n_patches']} | "
            f"{s['peak_offset_px_median']:.2f} | "
            f"{s['peak_offset_px_p95']:.2f} | "
            f"{s['peak_offset_px_mean']:.2f} | "
            f"{s['fwhm_px_median']:.2f} | "
            f"{s['fwhm_px_p95']:.2f} | "
            f"{s['peak_value_median']:.3f} |"
        )
    md += [
        "",
        "## Reading the table",
        "",
        "- **median offset** = typical pixel distance between binder's local feature peak and ground truth. Lower is better.",
        "- **p95 offset** = worst-case offset (95th percentile). Sets the canonicalization-floor target.",
        "- **FWHM** = full-width-half-max of the local correlation peak in pixels. Smaller = sharper localization.",
        "- **peak corr** = correlation value at the peak. Higher = stronger feature matching.",
        "",
        "Target for v2 verifier (per CGPT round-7 calibration):",
        "- Subject regions: P95 offset ≤ 2 px → 'high quality' canonicalization tier",
        "- Background: P95 ≤ 4 px acceptable",
        "- Low-energy: best-effort, low canonicalization weight",
        "",
        f"Elapsed: {out['elapsed_sec']}s",
    ]
    (args.out / "audit_1_alignment.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/audit_1_alignment.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()
