"""Track 2A — resolution-stability comparison: 16×16 pilot vs 32×32 rerun.

Per operator's spec 2026-05-04:

    Comparison: per-frame correlation between 16×16 and 32×32 saliency
    maps (downsample 32×32 to 16×16). Report Pearson + SSIM.

    Acceptance: per-frame correlation > 0.7 supports 16×16
    resolution-stability. Below that, pilot maps need a caveat in
    manuscript.

Inputs:
  - 16×16 pilot npz: experiments/paper_analyses/spatial_heatmaps/
      pilot_full/heatmap_data.npz
  - 32×32 rerun npz: experiments/paper_analyses/spatial_heatmaps/
      resolution_stability_32x32/heatmap_data.npz

Comparison strategy:

  For each (session, frame_id, source) shared between the two npz files
  (the 32×32 rerun is on a 4 D2 + 4 V10 stratified subset of the pilot's
  10+10):

    map_16x16   = pilot[f"compat__{sess}__f{frame}__{source}"]    # (16,16)
    map_32x32   = rerun[f"compat__{sess}__f{frame}__{source}"]    # (32,32)
    map_32_to_16 = block_average(map_32x32, factor=2)              # (16,16)
    pearson = pearsonr(map_16x16.flatten(), map_32_to_16.flatten())
    ssim    = ssim(map_16x16, map_32_to_16, data_range=...)

  Same for detection_contribution maps.

  Report aggregates per-(frame, source), per-frame, and overall.

Exit codes:
  0 — acceptance pass: per-frame mean Pearson > 0.7 across all frames.
  1 — acceptance fail: at least one frame's mean Pearson ≤ 0.7.
  2 — infrastructure error: missing/mismatched npz files, no shared keys.

Run as:
  .venv/bin/python scripts/paper_analyses/compare_resolution_stability.py \
      --pilot-npz <path>/pilot_full/heatmap_data.npz \
      --rerun-npz <path>/resolution_stability_32x32/heatmap_data.npz \
      --output-dir <path>/resolution_stability_32x32/comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# downsampling: factor-2 block average. 32×32 → 16×16 by averaging each
# disjoint 2×2 cell. This is the natural area downsample for grid-cell
# saliency maps (each pilot cell corresponds to a 2×2 block of finer cells).
# ---------------------------------------------------------------------

def block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping (factor × factor) blocks. arr must be 2-D
    and have shape divisible by factor on both axes.

    Caveat on grid-cell correspondence (verified for the pilot's input
    resolution):
        spatial_heatmap_pilot.grid_cells partitions the input H×W via
        np.linspace(0, H, grid+1).round(). For PHASE_G_INPUT_H=768 and
        PHASE_G_INPUT_W=1024, both 16 and 32 divide H and W exactly:
            768 / 16 = 48     768 / 32 = 24
            1024 / 16 = 64    1024 / 32 = 32
        So 16-grid cell boundaries are an exact subset of 32-grid
        boundaries, and a 2×2 block of 32×32 cells corresponds exactly
        to one 16×16 cell. block_average is therefore the correct
        downsample for our pilot inputs.

        For OTHER input resolutions where H % grid != 0 or W % grid != 0,
        np.linspace+round can produce 16- and 32-grid boundaries that
        DON'T align — block_average would then be an approximation. This
        comparison script is scoped to the pilot's input HW; the shape
        assertion in compare_one (16×16 in, 32×32 in) catches mismatches
        for any other use.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got {arr.shape}")
    h, w = arr.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            f"shape {arr.shape} not divisible by factor {factor}"
        )
    h2, w2 = h // factor, w // factor
    return arr.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


# ---------------------------------------------------------------------
# Pearson + SSIM
# ---------------------------------------------------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Pearson r over flattened a, b. Returns NaN if either has
    zero variance (avoids DivisionByZero in numpy.corrcoef)."""
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    if af.size != bf.size:
        raise ValueError(f"size mismatch: {af.size} vs {bf.size}")
    if af.std() == 0 or bf.std() == 0:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on a pair of 2-D maps. Uses skimage.metrics.structural_similarity
    with data_range computed from the joint range of the two maps. We
    intentionally avoid windowing for these tiny 16×16 maps; SSIM is
    computed over the entire array.

    Returns NaN if either map has zero variance (SSIM is undefined)."""
    from skimage.metrics import structural_similarity as ssim
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    joint_min = float(min(a.min(), b.min()))
    joint_max = float(max(a.max(), b.max()))
    data_range = joint_max - joint_min
    if data_range == 0:
        return float("nan")
    # 16×16 is small; pick win_size = 7 (largest odd ≤ min(H,W)) to keep
    # SSIM well-defined. Caller already guarantees 16×16 (compare_one
    # asserts), so a smaller-array fallback would be dead code; keep
    # win_size at 7.
    win = 7
    return float(ssim(a, b, data_range=data_range, win_size=win))


# ---------------------------------------------------------------------
# key parsing — matches `spatial_heatmap_pilot`'s npz convention:
#     compat__{sess}__f{r}__{label}
#     compat_baseline__{sess}__f{r}__{label}
#     detection_contribution__{sess}__f{r}__{label}
# ---------------------------------------------------------------------

def parse_compat_keys(npz_files: list[str]) -> set[tuple[str, int, str]]:
    """Return the set of (session, frame_id, label) triples present as
    `compat__...` keys."""
    out: set[tuple[str, int, str]] = set()
    for k in npz_files:
        if not k.startswith("compat__"):
            continue
        if k.startswith("compat_baseline__"):
            continue
        parts = k.split("__")
        # expected: ["compat", "{sess}", "f{r}", "{label}"]  (exactly 4 parts)
        if len(parts) != 4:
            continue
        sess = parts[1]
        f_part = parts[2]
        label = parts[3]
        if not f_part.startswith("f"):
            continue
        try:
            frame = int(f_part[1:])
        except ValueError:
            continue
        out.add((sess, frame, label))
    return out


def parse_detection_keys(npz_files: list[str]) -> set[tuple[str, int, str]]:
    out: set[tuple[str, int, str]] = set()
    for k in npz_files:
        if not k.startswith("detection_contribution__"):
            continue
        parts = k.split("__")
        if len(parts) != 4:
            continue
        sess = parts[1]
        f_part = parts[2]
        label = parts[3]
        if not f_part.startswith("f"):
            continue
        try:
            frame = int(f_part[1:])
        except ValueError:
            continue
        out.add((sess, frame, label))
    return out


# ---------------------------------------------------------------------
# main comparison
# ---------------------------------------------------------------------

def compare_one(map16: np.ndarray, map32: np.ndarray) -> dict:
    if map16.shape != (16, 16):
        raise ValueError(f"expected 16×16, got {map16.shape}")
    if map32.shape != (32, 32):
        raise ValueError(f"expected 32×32, got {map32.shape}")
    map32_down = block_average(map32, factor=2)
    return {
        "pearson": pearson_corr(map16, map32_down),
        "ssim": ssim_2d(map16, map32_down),
        "map16_minmax": (float(map16.min()), float(map16.max())),
        "map32down_minmax": (float(map32_down.min()), float(map32_down.max())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-npz", type=Path, required=True,
                    help="16×16 pilot heatmap_data.npz")
    ap.add_argument("--rerun-npz", type=Path, required=True,
                    help="32×32 rerun heatmap_data.npz")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="where to write comparison_report.md + JSON")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="acceptance threshold on per-frame mean Pearson "
                         "(default 0.7)")
    args = ap.parse_args()

    if not args.pilot_npz.exists():
        print(f"[compare] INFRA: pilot npz missing: {args.pilot_npz}")
        return 2
    if not args.rerun_npz.exists():
        print(f"[compare] INFRA: rerun npz missing: {args.rerun_npz}")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pilot = np.load(args.pilot_npz, allow_pickle=False)
    rerun = np.load(args.rerun_npz, allow_pickle=False)

    pilot_keys = list(pilot.files)
    rerun_keys = list(rerun.files)

    pilot_compat = parse_compat_keys(pilot_keys)
    rerun_compat = parse_compat_keys(rerun_keys)
    shared_compat = pilot_compat & rerun_compat
    pilot_det = parse_detection_keys(pilot_keys)
    rerun_det = parse_detection_keys(rerun_keys)
    shared_det = pilot_det & rerun_det

    if not shared_compat:
        print("[compare] INFRA: no shared compat keys between pilot and rerun")
        print(f"  pilot compat keys ({len(pilot_compat)}): "
              f"{sorted(pilot_compat)[:5]}...")
        print(f"  rerun compat keys ({len(rerun_compat)}): "
              f"{sorted(rerun_compat)[:5]}...")
        return 2

    print(f"[compare] shared compat triples: {len(shared_compat)}")
    print(f"[compare] shared detection triples: {len(shared_det)}")

    # Per-(sess, frame, source) results
    rows: list[dict] = []
    for sess, frame, label in sorted(shared_compat):
        compat_key = f"compat__{sess}__f{frame}__{label}"
        m16 = pilot[compat_key]
        m32 = rerun[compat_key]
        try:
            stats = compare_one(m16, m32)
        except ValueError as e:
            print(f"  [skip] {sess} f={frame} {label}: {e}")
            continue
        rows.append({
            "kind": "compat",
            "session": sess, "frame": frame, "source": label,
            **stats,
        })

    for sess, frame, label in sorted(shared_det):
        det_key = f"detection_contribution__{sess}__f{frame}__{label}"
        m16 = pilot[det_key]
        m32 = rerun[det_key]
        try:
            stats = compare_one(m16, m32)
        except ValueError as e:
            print(f"  [skip] det {sess} f={frame} {label}: {e}")
            continue
        rows.append({
            "kind": "detection",
            "session": sess, "frame": frame, "source": label,
            **stats,
        })

    # Per-frame aggregation. We enumerate ALL (sess, frame) pairs that
    # appear in any row — including those whose rows are all NaN — so
    # they cannot disappear from the acceptance gate. NaN means
    # collapse to NaN; the gate treats NaN as failure.
    all_frame_keys = sorted({(r["session"], r["frame"]) for r in rows})
    per_frame_pearson_mean: dict[str, float] = {}
    per_frame_ssim_mean: dict[str, float] = {}
    for sess, frame in all_frame_keys:
        key = f"{sess}_f{frame}"
        pearson_vals = [r["pearson"] for r in rows
                        if r["session"] == sess and r["frame"] == frame
                        and not np.isnan(r["pearson"])]
        ssim_vals = [r["ssim"] for r in rows
                     if r["session"] == sess and r["frame"] == frame
                     and not np.isnan(r["ssim"])]
        per_frame_pearson_mean[key] = (
            float(np.mean(pearson_vals)) if pearson_vals else float("nan")
        )
        per_frame_ssim_mean[key] = (
            float(np.mean(ssim_vals)) if ssim_vals else float("nan")
        )

    overall_pearson_mean = (
        float(np.mean([r["pearson"] for r in rows
                       if not np.isnan(r["pearson"])]))
        if rows else float("nan")
    )
    overall_ssim_mean = (
        float(np.mean([r["ssim"] for r in rows
                       if not np.isnan(r["ssim"])]))
        if rows else float("nan")
    )

    # Acceptance per operator: per-frame mean Pearson > threshold for ALL
    # frames. NaN-only frames count as failures (cannot demonstrate
    # stability if no valid correlation could be computed).
    failing_frames = {
        k: v for k, v in per_frame_pearson_mean.items()
        if np.isnan(v) or v <= args.threshold
    }
    acceptance_pass = bool(per_frame_pearson_mean) and not failing_frames

    summary = {
        "threshold": args.threshold,
        "n_rows": len(rows),
        "n_frames": len(per_frame_pearson_mean),
        "overall_pearson_mean": overall_pearson_mean,
        "overall_ssim_mean": overall_ssim_mean,
        "per_frame_pearson_mean": per_frame_pearson_mean,
        "per_frame_ssim_mean": per_frame_ssim_mean,
        "acceptance_pass": acceptance_pass,
        "failing_frames": failing_frames,
    }

    # Emit JSON
    json_path = args.output_dir / "comparison_summary.json"
    json_path.write_text(json.dumps({
        "summary": summary,
        "rows": rows,
    }, indent=2, default=float))
    print(f"[compare] wrote {json_path}")

    # Emit markdown report
    md_lines = [
        "# 16×16 vs 32×32 spatial heatmap resolution-stability comparison",
        "",
        f"- pilot npz: `{args.pilot_npz}`",
        f"- 32×32 rerun npz: `{args.rerun_npz}`",
        f"- threshold (per-frame mean Pearson): {args.threshold}",
        f"- shared compat triples: {len(shared_compat)}",
        f"- shared detection triples: {len(shared_det)}",
        "",
        f"**Overall mean Pearson**: {overall_pearson_mean:.4f}",
        f"**Overall mean SSIM**: {overall_ssim_mean:.4f}",
        "",
        f"**Acceptance**: {'PASS' if acceptance_pass else 'FAIL'} "
        f"(per-frame mean Pearson > {args.threshold} on all frames)",
        "",
        "## Per-frame mean Pearson (across compat + detection sources)",
        "",
        "| frame | mean Pearson | mean SSIM |",
        "|---|---|---|",
    ]
    for k in sorted(per_frame_pearson_mean.keys()):
        p = per_frame_pearson_mean[k]
        s = per_frame_ssim_mean.get(k, float("nan"))
        md_lines.append(f"| {k} | {p:.4f} | {s:.4f} |")
    md_lines += [
        "",
        "## Per-(session, frame, source) detail",
        "",
        "| kind | session | frame | source | Pearson | SSIM | "
        "16×16 [min,max] | 32→16 [min,max] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['kind']} | {r['session']} | {r['frame']} | {r['source']} "
            f"| {r['pearson']:.4f} | {r['ssim']:.4f} "
            f"| [{r['map16_minmax'][0]:+.4f}, {r['map16_minmax'][1]:+.4f}] "
            f"| [{r['map32down_minmax'][0]:+.4f}, "
            f"{r['map32down_minmax'][1]:+.4f}] |"
        )
    if failing_frames:
        md_lines += [
            "",
            "## Frames below threshold",
            "",
        ]
        for k, v in failing_frames.items():
            md_lines.append(f"- {k}: mean Pearson = {v:.4f}")
    md_lines += [
        "",
        "## Methodology notes",
        "",
        "- `block_average` averages 2×2 disjoint blocks (32×32 → 16×16).",
        "  Each pilot cell corresponds exactly to a 2×2 block of finer cells",
        "  for the pilot's input resolution (768/16=48, 768/32=24, 1024/16=64,",
        "  1024/32=32 — both 16- and 32-grid boundaries align via",
        "  `np.linspace(0, dim, grid+1).round()`). For other input resolutions",
        "  where dim % grid != 0, block_average would be an approximation.",
        "- Pearson computed on flattened maps. NaN if either has zero variance.",
        "- SSIM uses `skimage.metrics.structural_similarity` with",
        "  `data_range = max(map16.max(), map32.max()) - min(map16.min(), map32.min())`",
        "  and `win_size = 7` (the largest odd ≤ min(H,W) for 16×16 maps).",
        "- Per-frame mean folds BOTH compat and detection-contribution",
        "  rows together for a single per-frame statistic. The operator's",
        "  spec phrase \"per-frame correlation\" is ambiguous between",
        "  per-(frame,source) and per-frame-aggregate; this script reports",
        "  both (per-(frame,source) detail table + per-frame mean for the",
        "  threshold gate). When reporting results to operator, call this",
        "  out so they can read the per-(frame,source) detail if needed.",
        "- Acceptance: per-frame mean Pearson must EXCEED the threshold",
        "  (default 0.7) for ALL frames; NaN-only frames count as failures.",
        "  Below threshold = methodology caveat needed in manuscript.",
        "",
    ]
    md_path = args.output_dir / "comparison_report.md"
    md_path.write_text("\n".join(md_lines))
    print(f"[compare] wrote {md_path}")

    print(f"\n[compare] overall mean Pearson = {overall_pearson_mean:.4f}")
    print(f"[compare] overall mean SSIM    = {overall_ssim_mean:.4f}")
    print(f"[compare] per-frame Pearson means:")
    for k, v in sorted(per_frame_pearson_mean.items()):
        flag = "  " if v > args.threshold else "!!"
        print(f"  {flag} {k}: {v:.4f}")
    print(f"\n[compare] acceptance = {'PASS' if acceptance_pass else 'FAIL'}")
    return 0 if acceptance_pass else 1


if __name__ == "__main__":
    sys.exit(main())
