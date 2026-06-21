"""Visualize what the F-A mini editor is actually producing.

Grid layout (rows × cols):
  - 1 fixed source row at top: C_source | E_source | C_target_real
  - N target rows below: E_target_i | C_pred_i | |C_pred_i - C_source|×5

The diff column amplifies the per-pixel difference between C_pred and C_source
by 5× so visual inspection can catch sub-percentile perturbations even when
the diversity metric reads ~10⁻⁵.

Operator question: with diversity = 0.00001 across targets, are outputs truly
identical or differing in ways the metric isn't catching? The diff column
answers this directly.

Run on Lambda (where the editor_final.pt lives):
  CUDA_VISIBLE_DEVICES=1 python scripts/phase_f/visualize_mini_predictions.py \
    --ckpt /path/to/poliebotics_phase_b/mini_experiment/editor_final.pt \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --out /path/to/poliebotics_phase_b/experiments/phase_f_prep/mini_experiment/visuals
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa, packed_cfa_to_bayer_rg8  # noqa: E402
from phase_f.editor_model import Editor  # noqa: E402
from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402


def packed_cfa_to_rgb(cfa_float01: torch.Tensor) -> np.ndarray:
    """packed CFA (4, 2300, 2660) float [0,1] → demosaiced RGB uint8 (1150, 1330, 3).

    For visualization speed we run cv2 demosaic on a half-resolution recombined
    Bayer (so output is 1150 × 1330 RGB instead of 2300 × 2660). That's good
    enough for a visual grid.
    """
    cfa_u8 = (cfa_float01.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    # packed_cfa_to_bayer_rg8 returns 24M bytes; reshape to (4600, 5320)
    bayer_bytes = packed_cfa_to_bayer_rg8(cfa_u8)
    bayer = np.frombuffer(bayer_bytes, dtype=np.uint8).reshape(4600, 5320)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB_EA)  # (4600, 5320, 3)
    # Downsample for grid display
    rgb_small = cv2.resize(rgb, (rgb.shape[1] // 4, rgb.shape[0] // 4),
                            interpolation=cv2.INTER_AREA)
    return rgb_small


def emission_to_rgb(em_float01: torch.Tensor) -> np.ndarray:
    """emission (3, H, W) float [0,1] → RGB uint8 (H', W', 3) downsampled to ~1150×1330."""
    rgb = (em_float01.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))  # CHW → HWC
    h, w, _ = rgb.shape
    target_h = 1150
    target_w = int(w * target_h / h)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def diff_amplify(a: torch.Tensor, b: torch.Tensor, scale: float = 5.0) -> np.ndarray:
    """|a - b| * scale → uint8 grayscale-as-RGB at half-res for grid display."""
    diff = (a - b).abs().mean(dim=0)  # (H, W) avg over channels
    diff_np = diff.cpu().numpy()
    diff_u8 = np.clip(diff_np * scale * 255.0, 0, 255).astype(np.uint8)
    diff_rgb = cv2.cvtColor(diff_u8, cv2.COLOR_GRAY2RGB)
    return cv2.resize(diff_rgb, (diff_rgb.shape[1] // 4, diff_rgb.shape[0] // 4),
                       interpolation=cv2.INTER_AREA)


def label_image(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def make_grid(rows: list[list[np.ndarray]], pad: int = 6,
              bg: tuple[int, int, int] = (24, 24, 28)) -> np.ndarray:
    """Tile rows of images into a single grid, padded by `pad` px."""
    cell_h = max(im.shape[0] for r in rows for im in r)
    cell_w = max(im.shape[1] for r in rows for im in r)
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    canvas = np.full(
        (n_rows * cell_h + (n_rows + 1) * pad,
         n_cols * cell_w + (n_cols + 1) * pad, 3),
        bg, dtype=np.uint8,
    )
    for ri, row in enumerate(rows):
        for ci, im in enumerate(row):
            ih, iw = im.shape[:2]
            y0 = pad + ri * (cell_h + pad)
            x0 = pad + ci * (cell_w + pad)
            # Top-left placement; if image smaller than cell, leave bg around.
            canvas[y0:y0 + ih, x0:x0 + iw] = im
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--source-t", type=int, default=5526)
    ap.add_argument("--source-k", type=int, default=1, help="C_target = capture[source_t + k] is the matched real target")
    ap.add_argument("--target-ts", nargs="+", type=int,
                    default=[5431, 5484, 5495, 5527, 5600, 5701, 5800, 5900])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Load editor at native packed CFA resolution
    editor = Editor(capture_h=2300, capture_w=2660, emission_h=1080, emission_w=1920,
                    init_mode="random").to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexp = editor.load_state_dict(state, strict=False)
    print(f"[init] editor loaded, missing={len(missing)} unexp={len(unexp)}", flush=True)
    editor.eval()

    # Load source capture + emission, target capture + the matched-target emission
    source_t = args.source_t
    target_real_t = source_t + args.source_k

    cap_source = load_capture_at(args.d2_dir / "Recordings" / f"frame_{source_t:06d}.raw",
                                  2300, 2660).unsqueeze(0).to(device)
    em_source = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{source_t:06d}.png",
                                  1080, 1920).unsqueeze(0).to(device)
    em_target_matched = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{target_real_t:06d}.png",
                                         1080, 1920).unsqueeze(0).to(device)
    cap_target_real = load_capture_at(args.d2_dir / "Recordings" / f"frame_{target_real_t:06d}.raw",
                                      2300, 2660).unsqueeze(0).to(device)

    # Run forward with the matched E_target (this is the legitimate scenario)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
        cap_pred_matched = editor(cap_source, em_source, em_target_matched).float().clamp(0, 1)

    # Run forward for each off-target E_target_i
    preds: list[tuple[int, torch.Tensor, torch.Tensor]] = []  # (target_t, em_target, cap_pred)
    for t in args.target_ts:
        em_path = args.d2_dir / "derived" / "Emissions" / f"tile_{t:06d}.png"
        if not em_path.exists():
            print(f"[skip] missing emission tile_{t:06d}.png", flush=True)
            continue
        em_target = load_emission_at(em_path, 1080, 1920).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = editor(cap_source, em_source, em_target).float().clamp(0, 1)
        preds.append((t, em_target, pred))

    # Build grid
    print(f"[viz] building grid for source_t={source_t}, {len(preds)} alt-target predictions",
          flush=True)
    cap_source_rgb = label_image(packed_cfa_to_rgb(cap_source.squeeze(0)),
                                 f"C_source (t={source_t})")
    em_source_rgb = label_image(emission_to_rgb(em_source.squeeze(0)),
                                f"E_source (t={source_t})")
    em_target_matched_rgb = label_image(emission_to_rgb(em_target_matched.squeeze(0)),
                                        f"E_target matched (t={target_real_t})")
    cap_target_real_rgb = label_image(packed_cfa_to_rgb(cap_target_real.squeeze(0)),
                                      f"C_target REAL (t={target_real_t})")
    cap_pred_matched_rgb = label_image(packed_cfa_to_rgb(cap_pred_matched.squeeze(0)),
                                       f"C_pred (matched E_target)")
    diff_matched_rgb = label_image(diff_amplify(cap_pred_matched.squeeze(0),
                                                cap_source.squeeze(0), scale=5.0),
                                   "|C_pred - C_source| × 5")

    rows = [
        # Anchor row: source + matched-target reference
        [cap_source_rgb, em_source_rgb, em_target_matched_rgb,
         cap_target_real_rgb, cap_pred_matched_rgb, diff_matched_rgb],
    ]
    # One row per off-target E_target
    for t, em_target, pred in preds:
        rows.append([
            np.zeros_like(cap_source_rgb),  # blank where C_source repeats
            np.zeros_like(em_source_rgb),
            label_image(emission_to_rgb(em_target.squeeze(0)),
                        f"E_target alt (t={t})"),
            np.zeros_like(cap_target_real_rgb),  # no real target for off-source
            label_image(packed_cfa_to_rgb(pred.squeeze(0)),
                        f"C_pred (alt E_target)"),
            label_image(diff_amplify(pred.squeeze(0),
                                     cap_source.squeeze(0), scale=5.0),
                        "|C_pred - C_source| × 5"),
        ])

    grid = make_grid(rows, pad=6)
    out_path = args.out / "f_a_mini_visualization.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[done] wrote {out_path}  ({grid.shape[1]}×{grid.shape[0]})", flush=True)

    # Also save a metric block
    pred_stats = []
    for t, _, pred in preds:
        delta = (pred.squeeze(0) - cap_source.squeeze(0)).abs().mean().item()
        diff_to_matched = (pred.squeeze(0) - cap_pred_matched.squeeze(0)).abs().mean().item()
        pred_stats.append({
            "target_t": t,
            "mean_abs_delta_from_source": float(delta),
            "mean_abs_diff_from_matched_pred": float(diff_to_matched),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std()),
        })
    import json
    metric_payload = {
        "source_t": source_t,
        "target_real_t": target_real_t,
        "pred_matched_mean": float(cap_pred_matched.mean()),
        "pred_matched_std": float(cap_pred_matched.std()),
        "pred_matched_mean_abs_delta_from_source": float(
            (cap_pred_matched.squeeze(0) - cap_source.squeeze(0)).abs().mean()),
        "alternates": pred_stats,
    }
    (args.out / "f_a_mini_visualization.json").write_text(json.dumps(metric_payload, indent=2))
    print(f"[done] wrote {args.out}/f_a_mini_visualization.json", flush=True)


if __name__ == "__main__":
    main()
