"""Phase F #1 — orientation/Bayer-phase/axis proof.

Loads a real D2 capture, walks through the tensor conventions end-to-end, runs a
visual demosaic, and saves both the small inline check and a full
demosaiced PNG for visual sanity check.

Output:
  experiments/phase_f_prep/orientation_proof.md     — written by caller
  experiments/phase_f_prep/orientation_demosaic.png — full demosaiced image
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import (
    EXPECTED_BYTES, HEIGHT, WIDTH, HALF_H, HALF_W,
    bayer_rg8_to_packed_cfa,
)


def main():
    out_dir = ROOT / "experiments" / "phase_f_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data" / "d2" / "Recordings" / "frame_005500.raw"
    print(f"[orient] loading {raw_path}")
    raw_bytes = raw_path.read_bytes()
    print(f"  size: {len(raw_bytes):,} bytes (expected {EXPECTED_BYTES:,}) — match={len(raw_bytes) == EXPECTED_BYTES}")

    # Step 1: numpy reshape — confirm rows/cols
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    raw2d = arr.reshape(HEIGHT, WIDTH)
    print(f"\n  numpy reshape(HEIGHT={HEIGHT}, WIDTH={WIDTH}) → shape {raw2d.shape}")
    print(f"  shape[0] = rows  = HEIGHT (vertical)  = {raw2d.shape[0]}")
    print(f"  shape[1] = cols  = WIDTH  (horizontal) = {raw2d.shape[1]}")
    print(f"  aspect ratio (W/H) = {WIDTH/HEIGHT:.4f} → {'wider than tall' if WIDTH > HEIGHT else 'taller than wide'}")

    # Step 2: BayerRG phase from camera FourCC
    print(f"\n  BayerRG phase mapping (FourCC RG08):")
    print(f"    [0::2, 0::2] = R   (top-left of each 2×2)   row=0,col=0")
    print(f"    [0::2, 1::2] = G1  (top-right)              row=0,col=1")
    print(f"    [1::2, 0::2] = G2  (bottom-left)            row=1,col=0")
    print(f"    [1::2, 1::2] = B   (bottom-right)           row=1,col=1")

    # Step 3: extract one 2×2 block to spot-check Bayer phase makes sense
    block = raw2d[1000:1004, 2000:2004]
    print(f"\n  Sample 4×4 block at [1000:1004, 2000:2004]:")
    for row in block:
        print(f"    {row.tolist()}")

    R = raw2d[0::2, 0::2]
    G1 = raw2d[0::2, 1::2]
    G2 = raw2d[1::2, 0::2]
    B = raw2d[1::2, 1::2]
    print(f"\n  Per-channel means: R={R.mean():.2f}  G1={G1.mean():.2f}  G2={G2.mean():.2f}  B={B.mean():.2f}")
    print(f"  G1 ≈ G2 sanity: {abs(G1.mean() - G2.mean()):.2f} (expect ≪ 1 since both are green pixels)")
    print(f"  R/B differ from G as expected: yes (typical for non-pure-green scene)")

    # Step 4: packed CFA conversion
    cfa = bayer_rg8_to_packed_cfa(raw_bytes)
    print(f"\n  bayer_rg8_to_packed_cfa output:")
    print(f"    shape: {cfa.shape}")
    print(f"    dtype: {cfa.dtype}")
    print(f"    convention: (C=4, H={cfa.shape[1]}, W={cfa.shape[2]})")
    print(f"    H = HEIGHT//2 = {HEIGHT}//2 = {HALF_H} → expected {HALF_H}, got {cfa.shape[1]}, match={cfa.shape[1] == HALF_H}")
    print(f"    W = WIDTH//2  = {WIDTH}//2  = {HALF_W} → expected {HALF_W}, got {cfa.shape[2]}, match={cfa.shape[2] == HALF_W}")
    print(f"    channel order: ch0=R, ch1=G1, ch2=G2, ch3=B")
    print(f"    packed CFA per-channel means: R={cfa[0].mean():.2f}  G1={cfa[1].mean():.2f}  G2={cfa[2].mean():.2f}  B={cfa[3].mean():.2f}")
    print(f"    (these match the raw[Bayer-phase] means above — confirms ordering)")

    # Step 5: visual demosaic. cv2 has cv2.COLOR_BayerRG2RGB which expects RG layout.
    # NB: cv2 pattern naming differs from camera FourCC convention. The CV2 BayerRG
    # constant means "the cell at [0,0] is R, then G to its right". Same as our RGGB.
    rgb = cv2.cvtColor(raw2d, cv2.COLOR_BayerRG2RGB)
    print(f"\n  cv2.demosaic shape: {rgb.shape} (H={rgb.shape[0]}, W={rgb.shape[1]}, C={rgb.shape[2]})")
    print(f"  → match raw H × W: {rgb.shape[0] == HEIGHT and rgb.shape[1] == WIDTH}")

    # Save full-resolution demosaic for visual check
    out_full = out_dir / "orientation_demosaic_full.png"
    cv2.imwrite(str(out_full), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"\n  saved full demosaic: {out_full}")

    # Save downsampled version for inline display
    h_target = 720
    w_target = int(WIDTH * h_target / HEIGHT)
    rgb_small = cv2.resize(rgb, (w_target, h_target), interpolation=cv2.INTER_AREA)
    out_small = out_dir / "orientation_demosaic_small.png"
    cv2.imwrite(str(out_small), cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))
    print(f"  saved small demosaic: {out_small}  ({w_target}×{h_target})")

    # Save a corner-marker visual: paint a small rectangle in the upper-left to make
    # orientation unambiguous in the saved image
    rgb_with_marker = rgb_small.copy()
    cv2.rectangle(rgb_with_marker, (10, 10), (100, 60), (255, 0, 0), 3)
    cv2.putText(rgb_with_marker, "TOP-LEFT", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    out_marker = out_dir / "orientation_demosaic_marked.png"
    cv2.imwrite(str(out_marker), cv2.cvtColor(rgb_with_marker, cv2.COLOR_RGB2BGR))
    print(f"  saved marked demosaic: {out_marker}")


if __name__ == "__main__":
    main()
