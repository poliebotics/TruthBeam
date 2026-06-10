"""Track 2B — cross-fill comparison: mean-fill pilot vs blur-fill σ ∈ {2, 4, 8}.

Per operator's spec 2026-05-04:

    Comparison panel: fixed-fill (pilot) vs blur-fill (σ=2,4,8) side by
    side for representative frames.

This is the methodology-validation summary for Track 2B. Per-σ panels
already exist (each blur-fill run produces its own comparison_panel.png
showing real + F-A v1 ckpts under that σ). What's missing — and what
this script produces — is the cross-fill view: do the mean-fill pilot
maps and the three blur-fill maps tell the same saliency story?

Methodology:

  For each (session, frame_id, source) shared across the pilot and all
  three blur-fill outputs, we have four 16×16 saliency maps (mean / σ=2
  / σ=4 / σ=8). For each blur-fill σ we compute:

    pearson(σ) = Pearson(map_mean, map_blur_σ)
    ssim(σ)    = SSIM(map_mean, map_blur_σ)

  Aggregated per-frame and overall.

  Acceptance: per-frame mean Pearson > 0.7 against every blur-fill σ
  separately (same threshold as T2A resolution-stability). Below
  threshold → manuscript caveat needed on fill-mode choice for that
  frame.

Side-by-side panel: a compact figure showing the four fill modes for a
deterministically stratified subset of frames (1 low + 2 near-median +
1 high gap-quantile per session = 4 D2 + 4 V10 = 8 frames), one row of
4 maps per (frame, source).

Inputs:
  - mean-fill pilot npz: spatial_heatmaps/pilot_full/heatmap_data.npz
  - blur σ=2 npz: spatial_heatmaps/blur_fill_control/sigma_2/heatmap_data.npz
  - blur σ=4 npz: spatial_heatmaps/blur_fill_control/sigma_4/heatmap_data.npz
  - blur σ=8 npz: spatial_heatmaps/blur_fill_control/sigma_8/heatmap_data.npz

Exit codes:
  0 — acceptance pass: per-frame mean Pearson > threshold for all (frame, σ)
  1 — acceptance fail: at least one (frame, σ) below threshold
  2 — infrastructure error: missing/mismatched npz files, no shared keys

Run as:
  .venv/bin/python scripts/paper_analyses/compare_blur_fill.py \\
      --pilot-npz <path>/pilot_full/heatmap_data.npz \\
      --sigma2-npz <path>/blur_fill_control/sigma_2/heatmap_data.npz \\
      --sigma4-npz <path>/blur_fill_control/sigma_4/heatmap_data.npz \\
      --sigma8-npz <path>/blur_fill_control/sigma_8/heatmap_data.npz \\
      --output-dir <path>/blur_fill_control/cross_fill_comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Pearson + SSIM (mirroring compare_resolution_stability conventions)
# ---------------------------------------------------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over flattened a, b. Returns NaN if either has zero
    variance (avoids DivisionByZero)."""
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    if af.size != bf.size:
        raise ValueError(f"size mismatch: {af.size} vs {bf.size}")
    if af.std() == 0 or bf.std() == 0:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on a pair of 2-D maps. Uses skimage's structural_similarity
    with data_range computed from the joint range. Returns NaN if either
    map has zero variance.

    For 16×16 inputs we hardcode win_size=7 (largest odd ≤ min(H,W) that
    keeps SSIM well-defined). Same convention as
    compare_resolution_stability.ssim_2d.
    """
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
    return float(ssim(a, b, data_range=data_range, win_size=7))


# ---------------------------------------------------------------------
# key parsing — matches `spatial_heatmap_pilot` and `heatmap_blur_fill`
# npz conventions:
#     compat__{sess}__f{r}__{label}
#     compat_baseline__{sess}__f{r}__{label}
#     detection_contribution__{sess}__f{r}__{label}
#
# The pilot's keys and the blur-fill keys use the SAME naming: only the
# fill mode differs in PNG filenames, not in npz keys. So the keys are
# directly comparable across npz files.
# ---------------------------------------------------------------------

def parse_compat_keys(npz_files: list[str]) -> set[tuple[str, int, str]]:
    out: set[tuple[str, int, str]] = set()
    for k in npz_files:
        if not k.startswith("compat__"):
            continue
        if k.startswith("compat_baseline__"):
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
# representative subset for the side-by-side panel
# ---------------------------------------------------------------------

def select_representative_frames(
    shared_frames_per_sess: dict[str, list[tuple[int, float]]],
) -> list[tuple[str, int]]:
    """Stratified pick: 1 low + 2 near-median + 1 high gap-quantile per
    session (same convention as T2A resolution-stability). Input maps
    session → sorted list of (frame, gap_score) pairs (descending by
    gap)."""
    out: list[tuple[str, int]] = []
    for sess in ("D2", "V10"):
        ranked = shared_frames_per_sess.get(sess, [])
        n = len(ranked)
        if n < 4:
            # If fewer than 4 shared frames, take what we have.
            out.extend((sess, f) for f, _ in ranked)
            continue
        # Highest gap (rank 0)
        out.append((sess, ranked[0][0]))
        # 2 nearest the median (rank n//2 and n//2 - 1)
        med1 = n // 2
        med2 = max(0, med1 - 1)
        for r in (med1, med2):
            cand = ranked[r][0]
            if (sess, cand) not in out:
                out.append((sess, cand))
                if sum(1 for s, _ in out if s == sess) >= 3:
                    break
        # Lowest gap (rank n-1)
        out.append((sess, ranked[-1][0]))
    return out


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def compare_pair(map_mean: np.ndarray, map_blur: np.ndarray) -> dict:
    if map_mean.shape != (16, 16):
        raise ValueError(f"expected 16×16 mean-fill map, got {map_mean.shape}")
    if map_blur.shape != (16, 16):
        raise ValueError(f"expected 16×16 blur-fill map, got {map_blur.shape}")
    return {
        "pearson": pearson_corr(map_mean, map_blur),
        "ssim": ssim_2d(map_mean, map_blur),
        "mean_minmax": (float(map_mean.min()), float(map_mean.max())),
        "blur_minmax": (float(map_blur.min()), float(map_blur.max())),
    }


def render_side_by_side(
    output_dir: Path,
    representative: list[tuple[str, int]],
    pilot: np.lib.npyio.NpzFile,
    by_sigma: dict[int, np.lib.npyio.NpzFile],
    sources_to_show: list[str],
) -> Path:
    """Render the cross-fill side-by-side comparison panel for the
    representative subset.

    Layout: rows = (frame, source) pairs; cols = 4 fill modes (mean,
    σ=2, σ=4, σ=8). Each cell shows the compatibility heatmap with a
    consistent diverging colormap and per-cell vmin/vmax derived from
    the joint range across the four fill modes for that (frame, source)
    pair (so colors are comparable across the row).
    """
    import matplotlib.pyplot as plt
    sigmas_in_order = sorted(by_sigma.keys())
    n_rows = len(representative) * len(sources_to_show)
    n_cols = 1 + len(sigmas_in_order)  # mean + σ values
    if n_rows == 0:
        raise ValueError("no representative (frame, source) pairs to render")

    # Row height tuned for inspection: at 1.0 inch/row × dpi=110, an 8-frame
    # × 5-source panel (40 rows) is 4400 px tall — practical to scroll. The
    # earlier 2.5 inch/row produced an 11000 px figure that was unwieldy.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 1.0 * n_rows),
        dpi=110, squeeze=False,
    )

    row_i = 0
    col_labels = ["mean_fill"] + [f"blur σ={s}" for s in sigmas_in_order]
    for sess, frame in representative:
        for src in sources_to_show:
            key = f"compat__{sess}__f{frame}__{src}"
            if key not in pilot.files:
                # Skip the row if pilot doesn't have it; should not happen.
                continue
            map_mean = pilot[key]
            blur_maps = []
            valid = True
            for s in sigmas_in_order:
                if key not in by_sigma[s].files:
                    valid = False
                    break
                blur_maps.append(by_sigma[s][key])
            if not valid:
                # skip
                row_i += 1
                continue

            # Joint range across the 4 maps for consistent colorbar.
            stacked = np.stack([map_mean] + blur_maps)
            vmax = float(np.max(np.abs(stacked)))
            if vmax == 0:
                vmax = 1.0

            for col_i, (lab, m) in enumerate(zip(
                col_labels, [map_mean] + blur_maps,
            )):
                ax = axes[row_i, col_i]
                ax.imshow(m, cmap="RdYlGn", vmin=-vmax, vmax=+vmax,
                          interpolation="nearest")
                if row_i == 0:
                    ax.set_title(lab, fontsize=9)
                if col_i == 0:
                    ax.set_ylabel(f"{sess} f={frame}\n{src}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            row_i += 1

    # Trim any unused rows.
    for unused_row in range(row_i, n_rows):
        for c in range(n_cols):
            axes[unused_row, c].axis("off")

    fig.suptitle("Cross-fill comparison — mean fill vs blur fill σ ∈ {2, 4, 8}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = output_dir / "cross_fill_comparison_panel.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-npz", type=Path, required=True,
                    help="mean-fill pilot heatmap_data.npz (10 D2 + 10 V10)")
    ap.add_argument("--sigma2-npz", type=Path, required=True)
    ap.add_argument("--sigma4-npz", type=Path, required=True)
    ap.add_argument("--sigma8-npz", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="where to write report + panel PNG")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="acceptance threshold on per-(frame, σ) mean Pearson "
                         "(default 0.7, matches T2A resolution-stability)")
    args = ap.parse_args()

    for label, p in [("pilot", args.pilot_npz),
                     ("sigma2", args.sigma2_npz),
                     ("sigma4", args.sigma4_npz),
                     ("sigma8", args.sigma8_npz)]:
        if not p.exists():
            print(f"[compare] INFRA: {label} npz missing: {p}")
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pilot = np.load(args.pilot_npz, allow_pickle=False)
    s2 = np.load(args.sigma2_npz, allow_pickle=False)
    s4 = np.load(args.sigma4_npz, allow_pickle=False)
    s8 = np.load(args.sigma8_npz, allow_pickle=False)
    by_sigma = {2: s2, 4: s4, 8: s8}

    pilot_compat = parse_compat_keys(list(pilot.files))
    pilot_det = parse_detection_keys(list(pilot.files))
    shared_compat = set(pilot_compat)
    shared_det = set(pilot_det)
    for sigma_label, sigma_npz in by_sigma.items():
        shared_compat &= parse_compat_keys(list(sigma_npz.files))
        shared_det &= parse_detection_keys(list(sigma_npz.files))

    if not shared_compat:
        print("[compare] INFRA: no shared compat keys between pilot and all "
              "blur sigmas")
        print(f"  pilot compat triples: {len(pilot_compat)}")
        for s, npz in by_sigma.items():
            print(f"  σ={s}: {len(parse_compat_keys(list(npz.files)))}")
        return 2

    print(f"[compare] shared compat triples (across mean + 3 sigmas): "
          f"{len(shared_compat)}")
    print(f"[compare] shared detection triples: {len(shared_det)}")

    # Build per-(frame, σ) Pearson + SSIM rows.
    rows: list[dict] = []
    for sess, frame, src in sorted(shared_compat):
        map_mean = pilot[f"compat__{sess}__f{frame}__{src}"]
        for sigma in sorted(by_sigma.keys()):
            map_blur = by_sigma[sigma][f"compat__{sess}__f{frame}__{src}"]
            try:
                stats = compare_pair(map_mean, map_blur)
            except ValueError as e:
                print(f"  [skip] {sess} f={frame} {src} σ={sigma}: {e}")
                continue
            rows.append({
                "kind": "compat",
                "session": sess, "frame": frame, "source": src,
                "sigma": sigma,
                **stats,
            })

    for sess, frame, src in sorted(shared_det):
        det_key = f"detection_contribution__{sess}__f{frame}__{src}"
        map_mean = pilot[det_key]
        for sigma in sorted(by_sigma.keys()):
            map_blur = by_sigma[sigma][det_key]
            try:
                stats = compare_pair(map_mean, map_blur)
            except ValueError as e:
                print(f"  [skip] det {sess} f={frame} {src} σ={sigma}: {e}")
                continue
            rows.append({
                "kind": "detection",
                "session": sess, "frame": frame, "source": src,
                "sigma": sigma,
                **stats,
            })

    # Aggregate per-(frame, σ): mean Pearson across all sources for that
    # (sess, frame, σ). NaN-only collapses are explicit failures (cannot
    # demonstrate stability without a valid correlation).
    all_frame_sigma_keys = sorted({(r["session"], r["frame"], r["sigma"])
                                    for r in rows})
    per_pair_pearson: dict[str, float] = {}
    per_pair_ssim: dict[str, float] = {}
    for sess, frame, sigma in all_frame_sigma_keys:
        key = f"{sess}_f{frame}_σ{sigma}"
        pearson_vals = [r["pearson"] for r in rows
                        if r["session"] == sess and r["frame"] == frame
                        and r["sigma"] == sigma
                        and not np.isnan(r["pearson"])]
        ssim_vals = [r["ssim"] for r in rows
                     if r["session"] == sess and r["frame"] == frame
                     and r["sigma"] == sigma
                     and not np.isnan(r["ssim"])]
        per_pair_pearson[key] = (
            float(np.mean(pearson_vals)) if pearson_vals else float("nan")
        )
        per_pair_ssim[key] = (
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

    # Acceptance: every (frame, σ) pair's mean Pearson > threshold; NaN counts
    # as failure.
    failing = {
        k: v for k, v in per_pair_pearson.items()
        if np.isnan(v) or v <= args.threshold
    }
    acceptance_pass = bool(per_pair_pearson) and not failing

    # Per-σ aggregate: mean Pearson across all (frame, source) for each σ.
    per_sigma_pearson_mean = {}
    per_sigma_ssim_mean = {}
    for sigma in sorted(by_sigma.keys()):
        ps = [r["pearson"] for r in rows
              if r["sigma"] == sigma and not np.isnan(r["pearson"])]
        ss = [r["ssim"] for r in rows
              if r["sigma"] == sigma and not np.isnan(r["ssim"])]
        per_sigma_pearson_mean[sigma] = (
            float(np.mean(ps)) if ps else float("nan")
        )
        per_sigma_ssim_mean[sigma] = (
            float(np.mean(ss)) if ss else float("nan")
        )

    # Representative-subset selection for the panel: rank shared frames
    # by gap-score proxy. We don't have explicit gap scores here, but the
    # pilot's per-frame compat__{sess}__f{frame}__real baseline magnitudes
    # are a proxy. Use mean abs(real-compat-map) per frame as the
    # ranking signal (higher → "more salient" frames).
    ranked: dict[str, list[tuple[int, float]]] = {"D2": [], "V10": []}
    seen_frames: set[tuple[str, int]] = set()
    for sess, frame, src in shared_compat:
        if src != "real":
            continue
        if (sess, frame) in seen_frames:
            continue
        seen_frames.add((sess, frame))
        m = pilot[f"compat__{sess}__f{frame}__real"]
        ranked.setdefault(sess, []).append((frame, float(np.mean(np.abs(m)))))
    for sess in ranked:
        ranked[sess].sort(key=lambda t: t[1], reverse=True)

    representative = select_representative_frames(ranked)

    # Sources to show in the panel: real + 4 F-A v1 ckpts (full set).
    sources_to_show = ["real", "fa_v1_step_5000", "fa_v1_step_25000",
                       "fa_v1_step_70000", "fa_v1_step_100000"]

    panel_path = render_side_by_side(
        args.output_dir, representative, pilot, by_sigma, sources_to_show,
    )
    print(f"[compare] wrote {panel_path}")

    # Emit JSON
    summary = {
        "threshold": args.threshold,
        "n_rows": len(rows),
        "n_pairs": len(per_pair_pearson),
        "overall_pearson_mean": overall_pearson_mean,
        "overall_ssim_mean": overall_ssim_mean,
        "per_sigma_pearson_mean": per_sigma_pearson_mean,
        "per_sigma_ssim_mean": per_sigma_ssim_mean,
        "per_pair_pearson_mean": per_pair_pearson,
        "per_pair_ssim_mean": per_pair_ssim,
        "acceptance_pass": acceptance_pass,
        "failing_pairs": failing,
        "representative_frames_for_panel": [
            {"session": s, "frame": f} for s, f in representative
        ],
    }
    json_path = args.output_dir / "cross_fill_summary.json"
    json_path.write_text(json.dumps({
        "summary": summary,
        "rows": rows,
    }, indent=2, default=float))
    print(f"[compare] wrote {json_path}")

    # Markdown report
    md = [
        "# Cross-fill comparison — mean-fill pilot vs blur-fill σ ∈ {2, 4, 8}",
        "",
        f"- pilot npz : `{args.pilot_npz}`",
        f"- σ=2 npz   : `{args.sigma2_npz}`",
        f"- σ=4 npz   : `{args.sigma4_npz}`",
        f"- σ=8 npz   : `{args.sigma8_npz}`",
        f"- threshold (per-(frame, σ) mean Pearson): {args.threshold}",
        f"- shared compat triples: {len(shared_compat)}",
        f"- shared detection triples: {len(shared_det)}",
        "",
        f"**Overall mean Pearson** (all rows, all σ): {overall_pearson_mean:.4f}",
        f"**Overall mean SSIM**:    {overall_ssim_mean:.4f}",
        "",
        f"**Acceptance**: {'PASS' if acceptance_pass else 'FAIL'} "
        f"(per-(frame, σ) mean Pearson > {args.threshold} on all pairs)",
        "",
        "## Per-σ aggregate",
        "",
        "| σ | mean Pearson (vs mean-fill) | mean SSIM |",
        "|---|---|---|",
    ]
    for sigma in sorted(by_sigma.keys()):
        md.append(
            f"| {sigma} | {per_sigma_pearson_mean[sigma]:.4f} | "
            f"{per_sigma_ssim_mean[sigma]:.4f} |"
        )
    md += [
        "",
        "## Per-(frame, σ) mean Pearson",
        "",
        "| session | frame | σ | mean Pearson | mean SSIM |",
        "|---|---|---|---|---|",
    ]
    # Sort by (session, frame, sigma) for stable, scannable output.
    pair_rows = []
    for sess, frame, sigma in all_frame_sigma_keys:
        key = f"{sess}_f{frame}_σ{sigma}"
        pair_rows.append((sess, frame, sigma,
                           per_pair_pearson[key],
                           per_pair_ssim.get(key, float("nan"))))
    pair_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    for sess, frame, sigma, p, s in pair_rows:
        md.append(f"| {sess} | {frame} | {sigma} | {p:.4f} | {s:.4f} |")
    if failing:
        md += [
            "",
            "## (frame, σ) below threshold",
            "",
        ]
        for k, v in failing.items():
            md.append(f"- {k}: mean Pearson = {v:.4f}")
    md += [
        "",
        "## Methodology notes",
        "",
        "- Compares mean-fill pilot maps against blur-fill maps at σ ∈ {2,",
        "  4, 8}. All four fill modes share the same 16×16 grid resolution",
        "  and the same 10 D2 + 10 V10 frame set, so npz keys are directly",
        "  compatible.",
        "- Pearson computed on flattened compatibility / detection-",
        "  contribution maps. NaN if either has zero variance.",
        "- SSIM uses skimage.metrics.structural_similarity with",
        "  `data_range = max(joint) - min(joint)` and `win_size = 7`",
        "  (largest odd ≤ min(H,W) for 16×16 maps). Same convention as",
        "  T2A resolution-stability.",
        "- Per-(frame, σ) mean folds compat + detection rows together for",
        "  a single per-pair statistic. The rows table preserves",
        "  per-(frame, source, σ) detail.",
        "- Acceptance: per-(frame, σ) mean Pearson must EXCEED the",
        "  threshold (default 0.7) for ALL (frame, σ) pairs; NaN counts",
        "  as failure. Below threshold = methodology caveat needed in",
        "  manuscript on fill-mode choice for that frame.",
        "- Side-by-side panel: 1 low + 2 near-median + 1 high gap-quantile",
        "  per session (4 D2 + 4 V10), one row per (frame, source) for the",
        "  5 sources (real + F-A v1 @ 5k/25k/70k/100k). Each row shows",
        "  4 fill modes (mean, σ=2, σ=4, σ=8) at a shared per-row",
        "  diverging colormap.",
        "",
    ]
    md_path = args.output_dir / "cross_fill_report.md"
    md_path.write_text("\n".join(md))
    print(f"[compare] wrote {md_path}")

    print(f"\n[compare] overall mean Pearson = {overall_pearson_mean:.4f}")
    print(f"[compare] overall mean SSIM    = {overall_ssim_mean:.4f}")
    print("[compare] per-σ Pearson means:")
    for sigma in sorted(by_sigma.keys()):
        print(f"  σ={sigma}: {per_sigma_pearson_mean[sigma]:.4f}")
    print(f"\n[compare] acceptance = {'PASS' if acceptance_pass else 'FAIL'}")
    return 0 if acceptance_pass else 1


if __name__ == "__main__":
    sys.exit(main())
