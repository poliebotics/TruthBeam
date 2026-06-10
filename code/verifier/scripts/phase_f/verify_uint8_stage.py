"""Phase F #11 — verify uint8 stage of editor pipeline is bit-exact.

The editor outputs float32 [0, 1] packed CFA. To produce a chain-consistent
fake, that float CFA must be:
  float [0,1] → clip to [0,1] → multiply by 255 → round → cast uint8
  → packed_cfa_to_bayer_rg8 (bytes)
  → blake3 (chain-byte hash)

The bayer_rg8_to_packed_cfa(raw_bytes) → packed CFA roundtrip MUST be
bit-exact at the uint8 stage (only the float→uint8 step is lossy; the byte
representation is the protocol commitment).

This script validates that bit-exactness on synthetic editor outputs across
several distributions:
  - Uniform random in [0, 1]
  - Real capture cast to float
  - Edge cases: all-zero, all-one, half (rounding boundary)
  - Real capture + small additive noise (simulates editor delta)

Output: experiments/phase_f_prep/uint8_stage_verification.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import (
    HALF_H, HALF_W, HEIGHT, WIDTH,
    bayer_rg8_to_packed_cfa, packed_cfa_to_bayer_rg8,
    quantize_packed_cfa_to_uint8, verify_uint8_stage_bitexact,
)


def case_uniform_random(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(4, HALF_H, HALF_W).astype(np.float32)


def case_real_capture_as_float() -> np.ndarray:
    raw = (ROOT / "data/d2/Recordings/frame_005500.raw").read_bytes()
    cfa_uint8 = bayer_rg8_to_packed_cfa(raw)
    return cfa_uint8.astype(np.float32) / 255.0


def case_real_plus_noise(noise_std: float, seed: int) -> np.ndarray:
    base = case_real_capture_as_float()
    rng = np.random.RandomState(seed)
    return base + rng.randn(*base.shape).astype(np.float32) * noise_std


def case_all_zero() -> np.ndarray:
    return np.zeros((4, HALF_H, HALF_W), dtype=np.float32)


def case_all_one() -> np.ndarray:
    return np.ones((4, HALF_H, HALF_W), dtype=np.float32)


def case_half() -> np.ndarray:
    """Rounding-boundary check: 0.5 should round to 128 (round-half-to-even
    via numpy's .round() but our impl uses .round() then .astype(uint8)
    which is round-half-to-even for numpy)."""
    return np.full((4, HALF_H, HALF_W), 0.5, dtype=np.float32)


def case_below_zero_above_one() -> np.ndarray:
    """Edge: float values OUTSIDE [0, 1] (editor sigmoid won't produce these,
    but if a caller forgets to clip we should still produce valid uint8)."""
    rng = np.random.RandomState(0)
    arr = rng.rand(4, HALF_H, HALF_W).astype(np.float32) * 1.4 - 0.2
    return arr  # values in [-0.2, 1.2]


def main():
    out_dir = ROOT / "experiments/phase_f_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("uniform_random_seed=0",       case_uniform_random(0)),
        ("uniform_random_seed=42",      case_uniform_random(42)),
        ("real_capture_as_float",       case_real_capture_as_float()),
        ("real_capture_plus_0.01_noise", case_real_plus_noise(0.01, 0)),
        ("real_capture_plus_0.05_noise", case_real_plus_noise(0.05, 1)),
        ("all_zero",                    case_all_zero()),
        ("all_one",                     case_all_one()),
        ("all_half_(rounding_boundary)", case_half()),
        ("out_of_range_-0.2_to_+1.2",   case_below_zero_above_one()),
    ]

    results = []
    all_ok = True
    print(f"=== Phase F #11 — uint8 stage bit-exact verification ===\n")
    for name, cfa in cases:
        v = verify_uint8_stage_bitexact(cfa)
        v["case"] = name
        v["input_dtype"] = str(cfa.dtype)
        v["input_min"] = float(cfa.min())
        v["input_max"] = float(cfa.max())
        v["input_mean"] = float(cfa.mean())
        ok = v["uint8_stage_bitexact"]
        all_ok = all_ok and ok
        results.append(v)
        print(f"  {name:<42} dtype={cfa.dtype} min={float(cfa.min()):+.3f} max={float(cfa.max()):+.3f} "
              f"→ bitexact={ok} max_abs_diff={v['max_abs_diff']}")

    # Also test an explicit round-trip on real_capture: should be bit-exact since
    # the input IS already uint8 cast to float.
    raw_orig = (ROOT / "data/d2/Recordings/frame_005500.raw").read_bytes()
    cfa_orig = bayer_rg8_to_packed_cfa(raw_orig)
    cfa_float = cfa_orig.astype(np.float32) / 255.0
    cfa_back_uint8 = quantize_packed_cfa_to_uint8(cfa_float)
    raw_back = packed_cfa_to_bayer_rg8(cfa_back_uint8)
    orig_match_ok = (raw_orig == raw_back)
    print(f"\n  ORIGINAL real raw → float → quantize → bayer8 → byte-compare to original raw: "
          f"bitexact={orig_match_ok}")

    json_path = out_dir / "uint8_stage_verification.json"
    json_path.write_text(json.dumps({
        "all_ok": all_ok,
        "real_round_trip_to_orig_bytes": orig_match_ok,
        "results": results,
    }, indent=2))

    md_lines = [
        "# Phase F #11 — uint8 stage bit-exact verification",
        "",
        "Tests that the editor's float [0, 1] packed CFA → quantize → uint8 → bayer raw bytes",
        "→ unpack → uint8 packed CFA stage is bit-exact.",
        "",
        "If this stage isn't bit-exact, the editor's predicted output cannot produce a",
        "chain-consistent `bayer_hash` (the protocol's commitment is over uint8 raw bytes;",
        "any drift between quantize-uint8 and roundtrip-uint8 means the chain check will fail).",
        "",
        f"**Overall: {'PASS' if all_ok else 'FAIL'} ({sum(1 for r in results if r['uint8_stage_bitexact'])}/{len(results)} cases bit-exact)**",
        "",
        f"**Real-capture extension: original raw bytes == quantize(real/255)→bayer8 ? {orig_match_ok}**",
        "",
        "## Per-case results",
        "",
        "| case | input dtype | input range | bit-exact at uint8 stage | max abs diff |",
        "|---|---|---|---|---:|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['case']} | {r['input_dtype']} | "
            f"[{r['input_min']:+.3f}, {r['input_max']:+.3f}] | "
            f"{r['uint8_stage_bitexact']} | {r['max_abs_diff']} |"
        )
    md_lines += [
        "",
        "## What the cases cover",
        "",
        "- `uniform_random_*`: representative editor outputs (sigmoid-bounded, full range).",
        "- `real_capture_as_float`: real D2 frame cast to float32 / 255. If our quantize is",
        "  the inverse of the original /255, this should produce raw bytes byte-identical to",
        "  the on-disk `.raw` file (verified separately as `real_round_trip_to_orig_bytes`).",
        "- `real_capture_plus_*_noise`: simulates editor learning a small delta on top of",
        "  the source capture. Tests the quantize survives small additive perturbations.",
        "- `all_zero` / `all_one`: floor/ceiling of the float range.",
        "- `all_half_(rounding_boundary)`: 0.5 input → uint8 output; numpy `.round()` is",
        "  round-half-to-even, so 0.5*255=127.5 → 128. All pixels should land on 128.",
        "- `out_of_range_-0.2_to_+1.2`: caller forgot to sigmoid; quantize must clip first.",
        "",
        "## How to use in F-A pipeline",
        "",
        "```python",
        "from phase_f.cfa_roundtrip import (",
        "    quantize_packed_cfa_to_uint8,    # float [0,1] → uint8",
        "    packed_cfa_to_bayer_rg8,         # uint8 (4, 2300, 2660) → 24,472,000 bytes",
        "    editor_output_to_chain_bytes,    # one-shot helper: float → uint8 → bytes",
        ")",
        "from blake3 import blake3",
        "",
        "raw_bytes, cfa_uint8 = editor_output_to_chain_bytes(C_pred)",
        "bayer_hash = blake3(raw_bytes).digest()",
        "# bayer_hash is what the chain-walk verifier will recompute and check against",
        "# the chain_log.csv entry for this row.",
        "```",
        "",
        "## Files",
        "",
        "- `src/phase_f/cfa_roundtrip.py` — module with `quantize_packed_cfa_to_uint8`,",
        "  `editor_output_to_chain_bytes`, `verify_uint8_stage_bitexact`",
        "- `scripts/phase_f/verify_uint8_stage.py` — this verification script",
        "- `experiments/phase_f_prep/uint8_stage_verification.{md,json}` — outputs",
    ]
    (out_dir / "uint8_stage_verification.md").write_text("\n".join(md_lines))
    print(f"\n[done] all_ok={all_ok}  real_round_trip_to_orig_bytes={orig_match_ok}")
    print(f"  wrote {json_path}")
    print(f"  wrote {out_dir / 'uint8_stage_verification.md'}")


if __name__ == "__main__":
    main()
