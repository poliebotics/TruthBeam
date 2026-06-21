"""Side-by-side example crops at thresholds 0.08, 0.12, 0.15 on a few D2 frames."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa  # noqa: E402

OUT = ROOT / "results" / "crop_derivation"
OUT.mkdir(parents=True, exist_ok=True)

bbox_json = json.loads((ROOT / "calibration" / "bbox_current_rig.json").read_text())
alts = bbox_json["alternatives"]
SHOW = [("0.08 (primary)", alts["threshold_0.08"]),
        ("0.12",          alts["threshold_0.12"]),
        ("0.15",          alts["threshold_0.15"])]

raws = sorted((ROOT / "data" / "d2" / "Recordings").glob("frame_*.raw"))
sample_idxs = [200, 1500, 3500, 5500]


def load_g(p: Path) -> np.ndarray:
    cfa = bayer_rg8_to_packed_cfa(p.read_bytes())
    g = ((cfa[1].astype(np.float32) + cfa[2].astype(np.float32)) * 0.5)
    return np.clip(g, 0, 255).astype(np.uint8)


# Build grid: rows = frames, cols = [full-frame w/ all bboxes, crop@0.08, crop@0.12, crop@0.15]
rows = []
LABEL_H = 28
TARGET_H = 360  # display height per cell

for ti, idx in enumerate(sample_idxs):
    g = load_g(raws[idx])
    full = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    # Draw all three bboxes on full frame
    colors = [(0, 255, 0), (0, 200, 255), (0, 0, 255)]  # green, yellow, red
    for (lbl, bb), col in zip(SHOW, colors):
        cv2.rectangle(full, (bb["x_min"], bb["y_min"]),
                      (bb["x_max"], bb["y_max"]), col, 6)
    # Resize full to target height
    fh, fw = full.shape[:2]
    full_disp = cv2.resize(full, (int(fw * TARGET_H / fh), TARGET_H))

    cells = [full_disp]
    for (lbl, bb), col in zip(SHOW, colors):
        crop = g[bb["y_min"]:bb["y_max"], bb["x_min"]:bb["x_max"]]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        ch, cw = crop_bgr.shape[:2]
        crop_disp = cv2.resize(crop_bgr, (int(cw * TARGET_H / ch), TARGET_H))
        # Color border to match full-frame bbox color
        cv2.rectangle(crop_disp, (1, 1), (crop_disp.shape[1] - 2, crop_disp.shape[0] - 2), col, 3)
        cells.append(crop_disp)

    # Pad cells to common height (already same), concat horizontally with gaps
    gap = np.full((TARGET_H, 8, 3), 30, dtype=np.uint8)
    row = []
    for j, c in enumerate(cells):
        if j > 0:
            row.append(gap)
        row.append(c)
    row_img = np.concatenate(row, axis=1)
    rows.append(row_img)

# Common width
max_w = max(r.shape[1] for r in rows)
padded_rows = []
for r in rows:
    if r.shape[1] < max_w:
        pad = np.full((r.shape[0], max_w - r.shape[1], 3), 30, dtype=np.uint8)
        r = np.concatenate([r, pad], axis=1)
    padded_rows.append(r)

# Header row with column labels
header_h = 36
header = np.full((header_h, max_w, 3), 50, dtype=np.uint8)
col_w = max_w // 4
labels = ["full frame (all bboxes)", "crop @ 0.08 (green)",
          "crop @ 0.12 (yellow)",     "crop @ 0.15 (red)"]
for ci, lbl in enumerate(labels):
    cv2.putText(header, lbl, (ci * col_w + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Side label per row (frame idx)
LABEL_W = 80
labeled = []
for r, idx in zip(padded_rows, sample_idxs):
    side = np.full((r.shape[0], LABEL_W, 3), 50, dtype=np.uint8)
    cv2.putText(side, f"frame", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(side, f"{idx}", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    labeled.append(np.concatenate([side, r], axis=1))

header_with_pad = np.concatenate([np.full((header_h, LABEL_W, 3), 50, dtype=np.uint8),
                                  header], axis=1)
gap_v = np.full((6, header_with_pad.shape[1], 3), 30, dtype=np.uint8)
out = [header_with_pad, gap_v]
for r in labeled:
    out.append(r)
    out.append(gap_v)
grid = np.concatenate(out, axis=0)
cv2.imwrite(str(OUT / "crop_threshold_examples.png"), grid,
            [cv2.IMWRITE_PNG_COMPRESSION, 6])

print(f"[done] saved {OUT/'crop_threshold_examples.png'}  shape={grid.shape}")
print(f"frames used: {sample_idxs}")
for lbl, bb in SHOW:
    print(f"  {lbl}: y=[{bb['y_min']},{bb['y_max']}) x=[{bb['x_min']},{bb['x_max']}) "
          f"size={bb['dimensions_h_w'][0]}x{bb['dimensions_h_w'][1]} "
          f"frac={bb['fraction_of_frame']:.4f}")
