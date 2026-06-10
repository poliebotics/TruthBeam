"""Derive canonical crop bbox for current Phase F / Phase E rig.

Same methodology as `4.7_fresh_start/training/bbox_v3/calibration/bbox_newrig.json`:
range-map manual threshold, largest connected component. Operates at packed CFA
resolution (2300×2660) for speed; reports bbox at packed AND at BayerRG8 raw
(4600×5320) coords.

Procedure:
  1. Sample N frames spread across D2's 5992 captures
  2. Load each as packed CFA, take G channel (avg of G1, G2)
  3. Temporal range map: max - min over time, per pixel
  4. Normalize to [0, 1]
  5. Sweep thresholds {0.03, 0.05, 0.08, 0.10, 0.15, 0.20} → bbox of largest CC
  6. Save bbox_current_rig.json + crop_spec.md + range_map.png + bbox_overlay.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa  # noqa: E402


# Threshold sweep matching 4.7_fresh_start methodology
THRESHOLD_SWEEP = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
PRIMARY_THRESHOLD = 0.08


def load_g_channel(raw_path: Path) -> np.ndarray:
    """Load BayerRG8 raw, return G channel (avg of G1, G2) at packed-CFA half-res (2300, 2660)."""
    cfa = bayer_rg8_to_packed_cfa(raw_path.read_bytes())
    g = ((cfa[1].astype(np.float32) + cfa[2].astype(np.float32)) * 0.5)
    return g  # (2300, 2660) float32 in [0, 255]


def largest_blob_bbox(mask: np.ndarray) -> tuple[int, int, int, int, int, float]:
    """Return (y_min, y_max, x_min, x_max, area, fraction_of_frame) of the
    largest 4-connected blob in a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4)
    if n <= 1:
        return 0, 0, 0, 0, 0, 0.0
    # stats[0] is background; pick largest non-background CC
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    area = int(stats[idx, cv2.CC_STAT_AREA])
    # Bbox of the blob within the frame (inclusive y_min, exclusive y_max — Python convention)
    frame_area = mask.size
    return y, y + h, x, x + w, area, area / frame_area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/data/d2"))
    ap.add_argument("--n-frames", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raws = sorted((args.d2_dir / "Recordings").glob("frame_*.raw"))
    rng = np.random.RandomState(args.seed)
    # Skip first 30 / last 30 (avoid potential dead zones at session start/end)
    margin = 30
    candidates = list(range(margin, len(raws) - margin))
    sampled = sorted(rng.choice(candidates, size=min(args.n_frames, len(candidates)),
                                replace=False).tolist())
    print(f"[init] {len(raws)} raws total; sampling {len(sampled)} for range map", flush=True)

    # Compute temporal range map
    H, W = 2300, 2660
    pmin = np.full((H, W), 255.0, dtype=np.float32)
    pmax = np.zeros((H, W), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(sampled):
        g = load_g_channel(raws[t])
        np.minimum(pmin, g, out=pmin)
        np.maximum(pmax, g, out=pmax)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sampled)} elapsed={time.time()-t0:.0f}s", flush=True)

    range_map = pmax - pmin  # (H, W) float32 in [0, 255]
    range_map_n = range_map / max(float(range_map.max()), 1e-8)  # normalize to [0, 1]
    print(f"[range_map] max={range_map.max():.1f}  mean={range_map.mean():.1f}  "
          f"normalized: max={range_map_n.max():.3f} mean={range_map_n.mean():.3f}",
          flush=True)

    # Threshold sweep
    sweep = {}
    for thr in THRESHOLD_SWEEP:
        mask = (range_map_n >= thr)
        y0, y1, x0, x1, area, frac = largest_blob_bbox(mask)
        sweep[thr] = {
            "y_min": int(y0), "y_max": int(y1),
            "x_min": int(x0), "x_max": int(x1),
            "dimensions_h_w": [int(y1 - y0), int(x1 - x0)],
            "area_px": int(area),
            "fraction_of_frame": float(frac),
        }
        print(f"  threshold={thr:.2f}  bbox y=[{y0},{y1}) x=[{x0},{x1})  "
              f"size={y1-y0}×{x1-x0}  frac={frac:.4f}", flush=True)

    primary = sweep[PRIMARY_THRESHOLD]

    # Save bbox JSON in the bbox_newrig.json format
    bbox_json = {
        "y_min": primary["y_min"],
        "y_max": primary["y_max"],
        "x_min": primary["x_min"],
        "x_max": primary["x_max"],
        "dimensions": primary["dimensions_h_w"],
        "area_px": primary["area_px"],
        "fraction_of_frame": primary["fraction_of_frame"],
        "method": f"range_map_manual_threshold_{PRIMARY_THRESHOLD:.2f}_largest_blob",
        "threshold_value": PRIMARY_THRESHOLD,
        "margin_fraction": 0.0,
        "authoritative": True,
        "source_session": "D2 (cittadel_first_human_20260425_020819)",
        "source_session_full_path": str(args.d2_dir),
        "rig_hash": "3984699a504fb79821e6bcd1d81e8f7e3c64a7a99761680dac9aea104795038b",
        "rig_hash_match_with_4_7_fresh_start": True,
        "rig_hash_4_7_short": "3984699a504fb798",
        "frame_resolution_packed_cfa": [2300, 2660],
        "frame_resolution_bayer_rg8_raw": [4600, 5320],
        "n_frames_used": len(sampled),
        "n_frames_total": len(raws),
        "applicable_to": "All recordings on this rig (D2 + V10 confirmed same rig_hash). Bbox is in PACKED CFA coords (2300×2660). For BayerRG8 raw (4600×5320), 2× both axes.",
        "bayer_rg8_raw_bbox": {
            "y_min": primary["y_min"] * 2,
            "y_max": primary["y_max"] * 2,
            "x_min": primary["x_min"] * 2,
            "x_max": primary["x_max"] * 2,
            "dimensions": [primary["dimensions_h_w"][0] * 2, primary["dimensions_h_w"][1] * 2],
        },
        "alternatives": {f"threshold_{thr:.2f}": v for thr, v in sweep.items()},
        "note": ("Derived 2026-05-01 by re-running the same range-map+threshold "
                 "methodology used in 4.7_fresh_start/training/bbox_v3/calibration/"
                 "bbox_newrig.json on D2 captures at current rig's native packed CFA "
                 "resolution. Same rig_hash (3984699a504fb798), same physical projector "
                 "position, but capture resolution differs (this calibration is at "
                 "packed CFA 2300×2660; the 4.7 calibration was at 360×640 / 720×1280)."),
    }
    bbox_path = ROOT / "calibration" / "bbox_current_rig.json"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_path.write_text(json.dumps(bbox_json, indent=2))
    (args.out / "bbox_current_rig.json").write_text(json.dumps(bbox_json, indent=2))

    # Visualization: range map + primary bbox overlay
    rm_u8 = np.clip(range_map_n * 255, 0, 255).astype(np.uint8)
    rm_color = cv2.applyColorMap(rm_u8, cv2.COLORMAP_JET)
    cv2.rectangle(rm_color, (primary["x_min"], primary["y_min"]),
                  (primary["x_max"], primary["y_max"]), (0, 255, 0), 4)
    # Draw alternates as thinner lines
    for thr, v in sweep.items():
        if thr == PRIMARY_THRESHOLD:
            continue
        col = (255, 255, 255) if thr < PRIMARY_THRESHOLD else (180, 180, 180)
        cv2.rectangle(rm_color, (v["x_min"], v["y_min"]),
                      (v["x_max"], v["y_max"]), col, 1)
    cv2.imwrite(str(args.out / "range_map_with_bboxes.png"), rm_color,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # Bbox overlay on a representative real frame
    rep_t = sampled[len(sampled) // 2]
    g_rep = load_g_channel(raws[rep_t])
    g_rep_u8 = np.clip(g_rep, 0, 255).astype(np.uint8)
    g_rep_color = cv2.cvtColor(g_rep_u8, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(g_rep_color, (primary["x_min"], primary["y_min"]),
                  (primary["x_max"], primary["y_max"]), (0, 255, 0), 4)
    cv2.imwrite(str(args.out / "bbox_overlay_frame.png"), g_rep_color,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # Crop spec markdown
    md_lines = [
        "# Canonical local crop — current Phase F / Phase E rig",
        "",
        f"Derived 2026-05-01 via range-map manual-threshold method (matching "
        f"4.7_fresh_start methodology in `bbox_newrig.json`).",
        "",
        "## Rig context",
        f"- Rig hash: `3984699a504fb798…` (matches 4.7_fresh_start; same physical procam)",
        f"- Capture native: BayerRG8 raw 5320×4600",
        f"- Packed CFA: (4, 2300, 2660); G channel used for range-map computation",
        f"- D2 verified rig hash; V10 verified same rig hash",
        "",
        "## Procedure",
        f"1. Sampled {len(sampled)} D2 frames (skipping first/last {margin} to avoid session edges)",
        "2. Computed per-pixel temporal range map (max − min) on G channel at packed CFA resolution",
        "3. Normalized to [0, 1]",
        f"4. Swept thresholds {{{', '.join(str(t) for t in THRESHOLD_SWEEP)}}}",
        "5. Took bbox of largest 4-connected blob at each threshold",
        f"6. Selected primary threshold = {PRIMARY_THRESHOLD} (matches 4.7 methodology)",
        "",
        "## Authoritative bbox at packed CFA (2300×2660)",
        "",
        f"```",
        f"y_min = {primary['y_min']}",
        f"y_max = {primary['y_max']}",
        f"x_min = {primary['x_min']}",
        f"x_max = {primary['x_max']}",
        f"dimensions = {primary['dimensions_h_w'][0]} × {primary['dimensions_h_w'][1]}  (h × w)",
        f"area_px = {primary['area_px']}",
        f"fraction_of_frame = {primary['fraction_of_frame']:.4f}",
        f"```",
        "",
        "## Same bbox at BayerRG8 raw (4600×5320) — 2× scale",
        "",
        f"```",
        f"y_min = {primary['y_min'] * 2}",
        f"y_max = {primary['y_max'] * 2}",
        f"x_min = {primary['x_min'] * 2}",
        f"x_max = {primary['x_max'] * 2}",
        f"dimensions = {primary['dimensions_h_w'][0] * 2} × {primary['dimensions_h_w'][1] * 2}  (h × w)",
        f"```",
        "",
        "## Threshold sweep results (packed CFA coords)",
        "",
        "| threshold | y_min | y_max | x_min | x_max | h × w | frac |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for thr in THRESHOLD_SWEEP:
        v = sweep[thr]
        h_w = f"{v['dimensions_h_w'][0]}×{v['dimensions_h_w'][1]}"
        marker = " **(primary)**" if thr == PRIMARY_THRESHOLD else ""
        md_lines.append(
            f"| {thr:.2f}{marker} | {v['y_min']} | {v['y_max']} | "
            f"{v['x_min']} | {v['x_max']} | {h_w} | {v['fraction_of_frame']:.4f} |"
        )
    md_lines += [
        "",
        "## Files",
        "- `bbox_current_rig.json` — bbox metadata in same format as 4.7's `bbox_newrig.json`",
        "- `range_map_with_bboxes.png` — temporal range map (jet) with all swept bboxes overlaid (primary in green, others in white/grey)",
        "- `bbox_overlay_frame.png` — primary bbox drawn on a representative G-channel frame",
        "",
        "## Diffusion diagnostic resize spec",
        "Per `4.7_fresh_start/training/README.md:104-107`: resize cropped C and E symmetrically to 512×1024.",
        f"From this bbox: cropped frame is {primary['dimensions_h_w'][0]}×{primary['dimensions_h_w'][1]}.",
        f"Aspect ratio: {primary['dimensions_h_w'][0] / primary['dimensions_h_w'][1]:.3f} (target 512/1024 = 0.5).",
    ]
    aspect_ratio = primary['dimensions_h_w'][0] / max(primary['dimensions_h_w'][1], 1)
    if abs(aspect_ratio - 0.5) > 0.05:
        md_lines.append(
            f"\n**Aspect note**: cropped aspect {aspect_ratio:.3f} differs from 512/1024 (0.5) "
            f"by {abs(aspect_ratio - 0.5):.3f}. Symmetric resize will distort. "
            f"Consider 512×{int(512/aspect_ratio)} or {int(1024*aspect_ratio)}×1024 to preserve aspect.")
    (args.out / "crop_spec.md").write_text("\n".join(md_lines))

    print(f"\n[done] primary bbox: y=[{primary['y_min']},{primary['y_max']}) "
          f"x=[{primary['x_min']},{primary['x_max']}) "
          f"size={primary['dimensions_h_w'][0]}×{primary['dimensions_h_w'][1]} "
          f"frac={primary['fraction_of_frame']:.4f}", flush=True)
    print(f"[done] saved bbox_current_rig.json to {bbox_path} and {args.out}/", flush=True)
    print(f"[done] visualization: {args.out}/range_map_with_bboxes.png + bbox_overlay_frame.png", flush=True)


if __name__ == "__main__":
    main()
