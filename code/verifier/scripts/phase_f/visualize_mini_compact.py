"""Compact viewable version of the F-A mini visualization.

Layout: 2 rows of 5 panels each, ~600px tall per row, total ~1500 wide.
  Top row:    C_source | E_source | E_target_matched | C_target_REAL | C_pred (matched)
  Bottom row: 4 alt E_target predictions + diff strip below

This is a "see at a glance" companion to the full-res 54 MB grid.
Pulls demosaic + emission images already laid out in the full grid via
np.array slicing — fast and lossless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import packed_cfa_to_bayer_rg8  # noqa: E402
from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402


def cfa_to_rgb_small(cfa_float01: torch.Tensor, target_w: int = 360) -> np.ndarray:
    cfa_u8 = (cfa_float01.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    bayer_bytes = packed_cfa_to_bayer_rg8(cfa_u8)
    bayer = np.frombuffer(bayer_bytes, dtype=np.uint8).reshape(4600, 5320)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB_EA)
    h_orig, w_orig = rgb.shape[:2]
    target_h = int(h_orig * target_w / w_orig)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def emission_to_rgb_small(em_float01: torch.Tensor, target_w: int = 360) -> np.ndarray:
    rgb = (em_float01.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))
    h_orig, w_orig = rgb.shape[:2]
    target_h = int(h_orig * target_w / w_orig)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def amp_diff_small(a: torch.Tensor, b: torch.Tensor, scale: float = 8.0,
                    target_w: int = 360) -> np.ndarray:
    diff = (a - b).abs().mean(dim=0)
    diff_np = diff.cpu().numpy()
    diff_u8 = np.clip(diff_np * scale * 255.0, 0, 255).astype(np.uint8)
    diff_rgb = cv2.cvtColor(diff_u8, cv2.COLOR_GRAY2RGB)
    h_orig, w_orig = diff_rgb.shape[:2]
    target_h = int(h_orig * target_w / w_orig)
    return cv2.resize(diff_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--source-t", type=int, default=5526)
    ap.add_argument("--source-k", type=int, default=1)
    ap.add_argument("--target-ts", nargs="+", type=int, default=[5431, 5495, 5600, 5800])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    from phase_f.editor_model import Editor
    editor = Editor(capture_h=2300, capture_w=2660, emission_h=1080, emission_w=1920,
                    init_mode="random").to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    editor.load_state_dict(state, strict=False)
    editor.eval()

    source_t = args.source_t
    target_real_t = source_t + args.source_k

    cap_source = load_capture_at(args.d2_dir / "Recordings" / f"frame_{source_t:06d}.raw",
                                  2300, 2660).unsqueeze(0).to(device)
    em_source = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{source_t:06d}.png",
                                  1080, 1920).unsqueeze(0).to(device)
    em_target_matched = load_emission_at(
        args.d2_dir / "derived" / "Emissions" / f"tile_{target_real_t:06d}.png",
        1080, 1920).unsqueeze(0).to(device)
    cap_target_real = load_capture_at(args.d2_dir / "Recordings" / f"frame_{target_real_t:06d}.raw",
                                       2300, 2660).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cap_pred_matched = editor(cap_source, em_source, em_target_matched).float().clamp(0, 1)

    alt_data = []
    for t in args.target_ts:
        em_path = args.d2_dir / "derived" / "Emissions" / f"tile_{t:06d}.png"
        em_target = load_emission_at(em_path, 1080, 1920).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = editor(cap_source, em_source, em_target).float().clamp(0, 1)
        alt_data.append((t, em_target, pred))

    W = 360
    # Top row
    top = [
        label(cfa_to_rgb_small(cap_source.squeeze(0), W), f"C_source t={source_t}"),
        label(emission_to_rgb_small(em_source.squeeze(0), W), f"E_source t={source_t}"),
        label(emission_to_rgb_small(em_target_matched.squeeze(0), W), f"E_target matched t={target_real_t}"),
        label(cfa_to_rgb_small(cap_target_real.squeeze(0), W), f"C_target REAL t={target_real_t}"),
        label(cfa_to_rgb_small(cap_pred_matched.squeeze(0), W), "C_pred matched"),
    ]

    # Diff for matched
    diff_matched = label(
        amp_diff_small(cap_pred_matched.squeeze(0), cap_source.squeeze(0), 8.0, W),
        "|C_pred - C_source|x8"
    )

    # Alt rows: each row = E_target | C_pred | diff_from_source | diff_from_matched_pred
    alt_rows = []
    for t, em_target, pred in alt_data:
        alt_rows.append([
            label(emission_to_rgb_small(em_target.squeeze(0), W), f"E_target alt t={t}"),
            label(cfa_to_rgb_small(pred.squeeze(0), W), f"C_pred alt"),
            label(amp_diff_small(pred.squeeze(0), cap_source.squeeze(0), 8.0, W),
                  "|C_pred-C_source|x8"),
            label(amp_diff_small(pred.squeeze(0), cap_pred_matched.squeeze(0), 100.0, W),
                  "|C_pred-C_pred(matched)|x100"),
        ])

    # Build canvas: top row full width (5 panels), then 1 row for matched diff (centered),
    # then alt rows (4 panels each)
    pad = 8
    bg = (24, 24, 28)
    panel_h = top[0].shape[0]

    def stack_horiz(panels: list[np.ndarray], total_cols: int) -> np.ndarray:
        cell_w = panels[0].shape[1] if panels else W
        canvas = np.full((panel_h, total_cols * cell_w + (total_cols + 1) * pad, 3),
                         bg, dtype=np.uint8)
        for i, p in enumerate(panels):
            x0 = pad + i * (cell_w + pad)
            canvas[:p.shape[0], x0:x0 + p.shape[1]] = p
        return canvas

    # Top row: 5 columns
    top_strip = stack_horiz(top, 5)
    # Second strip: 1 panel diff_matched (in the rightmost-corresponding column)
    matched_diff_strip = stack_horiz([diff_matched], 1)
    # Pad matched_diff_strip to width of top
    if matched_diff_strip.shape[1] < top_strip.shape[1]:
        padded = np.full((matched_diff_strip.shape[0], top_strip.shape[1], 3), bg, dtype=np.uint8)
        # Place at column index 4 (matched C_pred column)
        x_target = pad + 4 * (W + pad)
        padded[:, x_target:x_target + matched_diff_strip.shape[1] - pad] = matched_diff_strip[:, pad:]
        matched_diff_strip = padded

    # Alt rows: 4 columns
    alt_strips = [stack_horiz(r, 4) for r in alt_rows]
    # Pad alt strips to top width
    for i, s in enumerate(alt_strips):
        if s.shape[1] < top_strip.shape[1]:
            padded = np.full((s.shape[0], top_strip.shape[1], 3), bg, dtype=np.uint8)
            padded[:, :s.shape[1]] = s
            alt_strips[i] = padded

    full = np.vstack([top_strip] + [matched_diff_strip] + alt_strips +
                      [np.full((pad, top_strip.shape[1], 3), bg, dtype=np.uint8)])
    out_path = args.out / "f_a_mini_compact.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"[done] wrote {out_path}  shape={full.shape}", flush=True)


if __name__ == "__main__":
    main()
