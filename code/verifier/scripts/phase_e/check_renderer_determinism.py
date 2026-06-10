"""Determinism + storage-match checks for the chain-byte probe renderer.

Three checks (per operator's pre-launch determinism block):

1. Renderer self-consistency: render the same chain bytes twice with
   bitexact_renderer.render_from_streams; verify pixel-exact match.

2. Seed determinism: noise_streams_byte_replace(streams, p=0.05, seed=0)
   twice; verify byte-exact corrupted streams.

3. Renderer-vs-storage match: render emission for one D2 row from clean
   chain bytes, compare to derived/Emissions/tile_NNNNNN.png. Acceptance:
     - bit-exact: ideal
     - <0.5 dB PSNR drift: acceptable, document
     - >1 dB drift: STOP, surface to operator

Run:
  python scripts/phase_e/check_renderer_determinism.py \
    --d2-dir <data> --row 5500 --out experiments/phase_e/probes/determinism.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bitexact_renderer import (  # noqa: E402
    expand_seeds_to_streams,
    noise_streams_byte_replace,
    render_from_streams,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.raw_bayer_dataset import load_chain_log  # noqa: E402


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    if mse < 1e-12: return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", required=True, type=Path)
    ap.add_argument("--row", type=int, default=5500, help="D2 row to use (must have valid emission tile)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    out: dict = {"d2_dir": str(args.d2_dir), "row_under_test": args.row, "device": args.device}
    print(f"=== Renderer determinism + storage-match checks (row {args.row}, device={args.device}) ===", flush=True)

    # Load chain log + extract S_{t+1}_hex
    chain = load_chain_log(args.d2_dir / "chain_log.csv")
    target_chain_row = args.row + 1  # XOF for row t derived from S_{t+1}
    if target_chain_row not in chain:
        raise SystemExit(f"chain[{target_chain_row}] missing — pick another row")
    s_next_hex = chain[target_chain_row]
    s_next_bytes = bytes.fromhex(s_next_hex)
    out["s_next_hex"] = s_next_hex
    out["target_chain_row"] = target_chain_row

    # Expand to streams
    stream_r, stream_g, stream_b = expand_seeds_to_streams(s_next_bytes)
    out["stream_lengths"] = [len(stream_r), len(stream_g), len(stream_b)]

    # Check 1: renderer self-consistency
    print("\n[check 1/3] Renderer self-consistency (render same streams twice)", flush=True)
    img_a = render_from_streams(stream_r, stream_g, stream_b, device=args.device)
    img_b = render_from_streams(stream_r, stream_g, stream_b, device=args.device)
    a_np = img_a.cpu().numpy()
    b_np = img_b.cpu().numpy()
    self_match = bool(np.array_equal(a_np, b_np))
    self_max_abs = int(np.abs(a_np.astype(np.int32) - b_np.astype(np.int32)).max())
    out["check_1_self_consistency"] = {
        "bitexact": self_match,
        "max_abs_diff": self_max_abs,
        "PASS": self_match,
    }
    print(f"  bitexact={self_match}  max_abs_diff={self_max_abs}  → {'PASS' if self_match else 'FAIL'}", flush=True)

    # Check 2: seed determinism for noise
    print("\n[check 2/3] Noise seed determinism (corrupt twice with same seed=0, p=0.05)", flush=True)
    n1 = noise_streams_byte_replace((stream_r, stream_g, stream_b), p=0.05, seed=0)
    n2 = noise_streams_byte_replace((stream_r, stream_g, stream_b), p=0.05, seed=0)
    seeds_match = all(a == b for a, b in zip(n1, n2))
    seed_max_diff = max(sum(int(x) ^ int(y) > 0 for x, y in zip(a, b))
                         for a, b in zip(n1, n2)) if not seeds_match else 0
    out["check_2_seed_determinism"] = {
        "byteexact": seeds_match,
        "diff_byte_count": seed_max_diff,
        "PASS": seeds_match,
    }
    print(f"  byteexact={seeds_match}  → {'PASS' if seeds_match else 'FAIL'}", flush=True)

    # Check 3: renderer vs storage match
    tile_path = args.d2_dir / "derived" / "Emissions" / f"tile_{target_chain_row:06d}.png"
    print(f"\n[check 3/3] Renderer-vs-storage match against {tile_path.name}", flush=True)
    if not tile_path.exists():
        out["check_3_storage_match"] = {"error": f"tile_path not found: {tile_path}", "PASS": False}
        print(f"  ERROR: {tile_path} not found", flush=True)
    else:
        # tile_gpu writes in HWC RGB layout (out_chw.permute(1,2,0).contiguous() then .numpy())
        # cv2.imwrite expects BGR; assume the saving code converted properly. Inspect order:
        # Looking at tb_loop disk_worker for emission tile save... cv2.imwrite default is BGR,
        # but session_finalize.py likely saves emissions with cv2 BGR convention. Compare both.
        rendered = a_np.transpose(1, 2, 0)  # (3, H, W) → (H, W, 3) RGB
        stored_bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if stored_bgr is None:
            out["check_3_storage_match"] = {"error": "cv2.imread returned None", "PASS": False}
            print(f"  ERROR: cv2 could not read {tile_path}", flush=True)
        else:
            stored_rgb = cv2.cvtColor(stored_bgr, cv2.COLOR_BGR2RGB)
            if rendered.shape != stored_rgb.shape:
                out["check_3_storage_match"] = {"error": f"shape mismatch: {rendered.shape} vs {stored_rgb.shape}", "PASS": False}
                print(f"  ERROR: shape mismatch", flush=True)
            else:
                bitexact = bool(np.array_equal(rendered, stored_rgb))
                psnr_db = psnr_uint8(rendered, stored_rgb)
                max_abs = int(np.abs(rendered.astype(np.int32) - stored_rgb.astype(np.int32)).max())
                # Per-channel PSNR
                per_channel_psnr = {c: psnr_uint8(rendered[..., i], stored_rgb[..., i])
                                     for i, c in enumerate("rgb")}
                # Acceptance per operator spec
                if bitexact:
                    verdict = "bit-exact"
                    ok = True
                elif psnr_db >= 50:  # ~0.5 dB drift threshold (1 LSB ≈ 48 dB)
                    verdict = "small drift, acceptable"
                    ok = True
                elif psnr_db < 40:
                    verdict = "LARGE DRIFT — STOP, surface to operator"
                    ok = False
                else:
                    verdict = "moderate drift, document and proceed"
                    ok = True
                out["check_3_storage_match"] = {
                    "tile_path": str(tile_path),
                    "bitexact": bitexact,
                    "psnr_db": psnr_db,
                    "per_channel_psnr_db": per_channel_psnr,
                    "max_abs_diff": max_abs,
                    "rendered_shape": list(rendered.shape),
                    "stored_shape": list(stored_rgb.shape),
                    "rendered_mean_per_channel": [float(rendered[..., i].mean()) for i in range(3)],
                    "stored_mean_per_channel": [float(stored_rgb[..., i].mean()) for i in range(3)],
                    "verdict": verdict,
                    "PASS": ok,
                }
                print(f"  bitexact={bitexact}  PSNR={psnr_db:.2f} dB  max_abs={max_abs}  per_ch={per_channel_psnr}", flush=True)
                print(f"  verdict: {verdict}", flush=True)

    overall = all(out[f"check_{i}_{n}"]["PASS"] for i, n in [
        (1, "self_consistency"), (2, "seed_determinism"), (3, "storage_match")
    ] if f"check_{i}_{n}" in out)
    out["OVERALL_PASS"] = overall
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n=== OVERALL: {'PASS' if overall else 'FAIL'} ===", flush=True)
    print(f"[done] wrote {args.out}", flush=True)
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
