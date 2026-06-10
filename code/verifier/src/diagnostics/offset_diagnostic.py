"""Offset diagnostic — empirically verify the capture↔chain row alignment.

Procedure (audit Q2, operator-confirmed):

1. Sweep offsets in [-16, +16] inclusive (33 candidates) per capture frame.
2. For each capture frame at row `t`:
     a. Downsample the captured green channel (G1+G2 averaged at half-res then
        further area-resized) to 64×64.
     b. For each candidate offset `o`, load the rendered emission tile for
        chain_row = t + o; convert to grayscale via green channel; downsample
        to 64×64.
     c. Compute Pearson correlation between (a) and (b).
     d. Argmax over `o` → winning offset for this capture.

3. Across each evaluation bin, take the **mode** of the winning offsets.
4. Pass criterion (per bin):
     - ≥80% of captures' winning offset is within ±1 of the bin mode.
5. Cross-bin pass: bin modes are all the same OR differ by at most ±1.

If pass: training MUST use the consistent winning offset. The convention is
`target_chain_row = capture_row + winning_offset`.

If fail: STOP and write to QUESTIONS.md. Inconsistent offsets indicate a
pairing or timing-drift bug in the recording pipeline.

Bin row ranges (operator-confirmed):
    D2-mid:    rows [2500, 3000), sample 50 captures uniformly
    V10-early: rows [ 100,  300), sample 30 captures uniformly
    V10-mid:   rows [1500, 1800), sample 30 captures uniformly
    V10-late:  rows [3200, 3500), sample 30 captures uniformly
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from data.raw_bayer_dataset import load_chain_log
from preprocessing.packed_cfa import HEIGHT, WIDTH, EXPECTED_BYTES, split_cfa_rggb

OFFSETS = tuple(range(-16, 17))   # -16..+16 inclusive

BIN_RANGES = {
    "D2-mid":    (2500, 3000),
    "V10-early": (100, 300),
    "V10-mid":   (1500, 1800),
    "V10-late":  (3200, 3500),
}
BIN_SAMPLES = {
    "D2-mid": 50,
    "V10-early": 30,
    "V10-mid": 30,
    "V10-late": 30,
}


def _capture_green_64x64(raw_path: Path) -> np.ndarray:
    """Load raw, average G1+G2, downsample to 64x64 float32."""
    raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
    if raw_bytes.size != EXPECTED_BYTES:
        raise ValueError(f"unexpected raw size {raw_bytes.size}: {raw_path}")
    raw2d = raw_bytes.reshape(HEIGHT, WIDTH)
    cfa = split_cfa_rggb(raw2d).astype(np.float32)  # (4, H/2, W/2)
    g_mean = 0.5 * (cfa[1] + cfa[2])  # (H/2, W/2) average green
    return cv2.resize(g_mean, (64, 64), interpolation=cv2.INTER_AREA)


def _emission_green_64x64(tile_path: Path) -> np.ndarray:
    """Load 1080×1920 RGB tile, take green channel, downsample to 64x64 float32."""
    img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(tile_path)
    # cv2 returns BGR; index 1 is green
    g = img[:, :, 1].astype(np.float32)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)


def best_offset_for_frame(
    *,
    capture_path: Path,
    emissions_dir: Path,
    chain_max_t: int,
    capture_t: int,
    offsets: Iterable[int] = OFFSETS,
) -> tuple[int, dict[int, float]]:
    """Return (winning_offset, {offset: correlation}) for one capture."""
    cap = _capture_green_64x64(capture_path)
    scores: dict[int, float] = {}
    for o in offsets:
        target = capture_t + o
        if target < 0 or target > chain_max_t:
            scores[o] = float("nan")
            continue
        tile = emissions_dir / f"tile_{target:06d}.png"
        if not tile.exists():
            scores[o] = float("nan")
            continue
        emi = _emission_green_64x64(tile)
        scores[o] = _pearson(cap, emi)
    valid = {o: s for o, s in scores.items() if not np.isnan(s)}
    winning = max(valid, key=valid.get)
    return winning, scores


def run_bin_diagnostic(
    *,
    bin_name: str,
    session_dir: Path,
    seed: int = 0,
) -> dict:
    """Run the diagnostic on one bin and return a structured result."""
    if bin_name not in BIN_RANGES:
        raise ValueError(f"unknown bin {bin_name!r}; expected one of {list(BIN_RANGES)}")
    start, end = BIN_RANGES[bin_name]
    n_samples = BIN_SAMPLES[bin_name]
    rng = np.random.default_rng(seed)

    chain = load_chain_log(session_dir / "chain_log.csv")
    chain_max_t = max(chain.keys())
    recordings = session_dir / "Recordings"
    emissions_dir = session_dir / "derived" / "Emissions"

    candidate_rows = [t for t in range(start, end) if (recordings / f"frame_{t:06d}.raw").exists()]
    if len(candidate_rows) < n_samples:
        n_samples = len(candidate_rows)
    sampled_rows = sorted(rng.choice(candidate_rows, size=n_samples, replace=False).tolist())

    per_frame: list[dict] = []
    for t in sampled_rows:
        cap_path = recordings / f"frame_{t:06d}.raw"
        winning, scores = best_offset_for_frame(
            capture_path=cap_path,
            emissions_dir=emissions_dir,
            chain_max_t=chain_max_t,
            capture_t=t,
        )
        per_frame.append({"t": t, "winning_offset": winning,
                          "scores": {str(k): v for k, v in scores.items()}})

    winners = [pf["winning_offset"] for pf in per_frame]
    counter = Counter(winners)
    mode_offset, mode_count = counter.most_common(1)[0]
    within_pm1 = sum(1 for w in winners if abs(w - mode_offset) <= 1)
    pass_rate = within_pm1 / len(winners) if winners else 0.0

    return {
        "bin": bin_name,
        "session_dir": str(session_dir),
        "row_range": [start, end],
        "n_sampled": len(sampled_rows),
        "winners": winners,
        "mode_offset": mode_offset,
        "mode_count": mode_count,
        "within_pm1_fraction": pass_rate,
        "passed_within_bin": pass_rate >= 0.8,
        "per_frame": per_frame,
    }


def cross_bin_consistency(bin_results: list[dict]) -> dict:
    """Check whether all bin modes agree within ±1."""
    modes = [r["mode_offset"] for r in bin_results]
    spread = max(modes) - min(modes)
    return {
        "modes": modes,
        "spread": int(spread),
        "consistent": spread <= 1,
        "consensus_offset": int(round(sum(modes) / len(modes))) if modes else None,
    }


def _write_questions_failure(questions_md_path: Path, summary: dict) -> None:
    """Append a structured failure block to QUESTIONS.md."""
    questions_md_path = Path(questions_md_path)
    questions_md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["", "## Offset diagnostic FAILED — training cannot proceed",
             "",
             f"Run at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             "",
             "Per-bin results:",
             ""]
    for b in summary["bins"]:
        lines.append(f"- **{b['bin']}** (rows {b['row_range'][0]}..{b['row_range'][1]}, n={b['n_sampled']}): "
                     f"mode={b['mode_offset']} ({b['mode_count']}/{b['n_sampled']}), "
                     f"within ±1: {b['within_pm1_fraction']:.0%}, "
                     f"passed: {b['passed_within_bin']}")
    lines.append("")
    lines.append(f"Cross-bin modes: {summary['cross_bin']['modes']} "
                 f"(spread {summary['cross_bin']['spread']}, "
                 f"consistent: {summary['cross_bin']['consistent']})")
    lines.append("")
    lines.append("**No consistent winning offset found.** Inconsistent offsets across "
                 "bins indicate a pairing or timing-drift bug in the recording pipeline. "
                 "Training is BLOCKED until this is resolved.")
    lines.append("")
    with open(questions_md_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def run_all(
    *,
    d2_session_dir: Path,
    v10_session_dir: Path,
    out_dir: Path,
    seed: int = 0,
    questions_md_path: Path | None = None,
    raise_on_failure: bool = True,
) -> dict:
    """Run all four bins, write diagnostic JSON, return summary.

    If `overall_pass` is false:
      - write a failure block to `questions_md_path` (default: out_dir/../QUESTIONS.md)
      - if `raise_on_failure` is True (default), raise SystemExit(1) so the
        sanity orchestrator (and the cloud script via PIPESTATUS check) halts.
    """
    bin_results = []
    bin_results.append(run_bin_diagnostic(
        bin_name="D2-mid", session_dir=d2_session_dir, seed=seed
    ))
    for bin_name in ("V10-early", "V10-mid", "V10-late"):
        bin_results.append(run_bin_diagnostic(
            bin_name=bin_name, session_dir=v10_session_dir, seed=seed
        ))

    consistency = cross_bin_consistency(bin_results)
    all_within_bin = all(r["passed_within_bin"] for r in bin_results)
    overall_pass = bool(all_within_bin and consistency["consistent"])

    summary = {
        "bins": bin_results,
        "cross_bin": consistency,
        "all_bins_pass_within_pm1": all_within_bin,
        "overall_pass": overall_pass,
        "winning_offset": consistency["consensus_offset"] if overall_pass else None,
        "convention": "target_chain_row = capture_row + offset",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offset_diagnostic.json").write_text(json.dumps(summary, indent=2))

    if not overall_pass:
        qpath = Path(questions_md_path) if questions_md_path else (out_dir.parent / "QUESTIONS.md")
        _write_questions_failure(qpath, summary)
        if raise_on_failure:
            raise SystemExit(
                f"offset diagnostic FAILED — see {out_dir / 'offset_diagnostic.json'} "
                f"and {qpath}; training BLOCKED."
            )
    return summary
