"""Verify CFA bit-exact roundtrip on 50 real captures from D2 + V10.

Run:
  python scripts/phase_f/verify_cfa_roundtrip.py \
    --d2-dir <data>/d2 --v10-dir <data>/v10 \
    --n-per-session 25 \
    --out experiments/phase_f_prep/cfa_roundtrip_verification.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import verify_bit_exact_roundtrip, EXPECTED_BYTES


def sample_captures(session_dir: Path, n: int, seed: int = 7) -> list[Path]:
    rec = session_dir / "Recordings"
    files = sorted(rec.glob("frame_*.raw"))
    rng = np.random.RandomState(seed)
    picks = rng.choice(len(files), size=min(n, len(files)), replace=False).tolist()
    return [files[i] for i in sorted(picks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--n-per-session", type=int, default=25)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[verify] sampling {args.n_per_session} captures per session", flush=True)
    results: dict = {"summary": {}, "per_capture": []}
    all_ok = True
    t0 = time.time()
    for session_label, session_dir in [("D2", args.d2_dir), ("V10", args.v10_dir)]:
        if not session_dir.exists():
            print(f"[skip] {session_label}: {session_dir} missing", flush=True)
            continue
        picks = sample_captures(session_dir, args.n_per_session)
        print(f"\n=== {session_label} ({len(picks)} captures) ===", flush=True)
        sess_ok = 0
        sess_results = []
        for p in picks:
            r = verify_bit_exact_roundtrip(p)
            sess_results.append(r)
            results["per_capture"].append({"session": session_label, **r})
            if r["ok"]:
                sess_ok += 1
            print(f"  {p.name}: bitexact={r['bitexact']} hashes_match={r['hashes_match']} "
                  f"max_abs_diff={r.get('max_abs_diff', 0)} bayer_hash_first8={r['bayer_hash_before'][:8]}", flush=True)
        results["summary"][session_label] = {
            "n_attempted": len(picks),
            "n_ok": sess_ok,
            "all_ok": sess_ok == len(picks),
        }
        if sess_ok != len(picks):
            all_ok = False
        print(f"  → {sess_ok}/{len(picks)} bit-exact OK", flush=True)
    results["all_ok"] = all_ok
    results["elapsed_sec"] = round(time.time() - t0, 1)

    # JSON output
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))

    # Markdown report
    n_total = sum(s["n_attempted"] for s in results["summary"].values())
    n_ok = sum(s["n_ok"] for s in results["summary"].values())
    md = [
        f"# CFA roundtrip verification — {n_ok}/{n_total} bit-exact OK",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
        f"Elapsed: {results['elapsed_sec']} s",
        "",
        "## Method",
        "",
        "1. Load real `.raw` capture from disk (BayerRG8 mosaic, 24,472,000 bytes).",
        "2. Compute `bayer_hash = blake3(raw_bytes)` (the protocol's chain commitment).",
        "3. Convert raw → packed CFA `(4, 2300, 2660) uint8` via `bayer_rg8_to_packed_cfa`.",
        "4. Convert packed CFA → BayerRG8 bytes via `packed_cfa_to_bayer_rg8`.",
        "5. Compute `bayer_hash_after = blake3(raw_back)`.",
        "6. Check: `raw_bytes == raw_back` (byte-for-byte) AND `bayer_hash == bayer_hash_after`.",
        "",
        "## Summary",
        "",
        "| session | n attempted | n bit-exact OK |",
        "|---|---:|---:|",
    ]
    for sess, s in results["summary"].items():
        md.append(f"| {sess} | {s['n_attempted']} | {s['n_ok']} |")
    md.append("")
    md.append(f"**Overall: {n_ok}/{n_total} bit-exact, hashes match. Roundtrip verified.**" if all_ok
              else f"**FAIL: {n_total - n_ok} captures did not roundtrip.**")
    md.append("")
    md.append("## Per-capture detail (first 10 per session)")
    md.append("")
    md.append("| session | file | bitexact | hashes match | max_abs_diff | bayer_hash (first 8) |")
    md.append("|---|---|---|---|---:|---|")
    seen_per_sess: dict = {}
    for r in results["per_capture"]:
        sess = r["session"]
        seen_per_sess[sess] = seen_per_sess.get(sess, 0) + 1
        if seen_per_sess[sess] > 10: continue
        md.append(
            f"| {sess} | `{Path(r['path']).name}` | {r['bitexact']} | {r['hashes_match']} | "
            f"{r.get('max_abs_diff', 0)} | `{r['bayer_hash_before'][:8]}` |"
        )
    md.append("")
    md.append("## Module reference")
    md.append("")
    md.append("- `src/phase_f/cfa_roundtrip.py`")
    md.append("  - `bayer_rg8_to_packed_cfa(raw_bytes)` → `(4, 2300, 2660) uint8`")
    md.append("  - `packed_cfa_to_bayer_rg8(cfa)` → 24,472,000 bytes")
    md.append("  - `verify_bit_exact_roundtrip(path)` → diagnostic dict")
    md.append("")
    md.append("BayerRG layout: R top-left, G1 top-right, G2 bottom-left, B bottom-right of each 2×2 block. Native sensor 5320 × 4600.")

    args.out.write_text("\n".join(md))
    print(f"\n[verify] {n_ok}/{n_total} bit-exact OK", flush=True)
    print(f"[verify] wrote {args.out}", flush=True)
    print(f"[verify] wrote {json_path}", flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
