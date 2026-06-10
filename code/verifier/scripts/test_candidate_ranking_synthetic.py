"""Phase D synthetic test — verify candidate-ranking invariants + Window-FMR@95
on hand-crafted inputs without needing the cloud or trained model.

Run: `.venv/bin/python scripts/test_candidate_ranking_synthetic.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from eval.candidate_ranking import (
    FAMILIES,
    NEAR_SHIFT_OFFSETS,
    assemble_candidate_set,
    compute_tau95,
    per_family_window_fmr,
    score_candidate_xof,
    score_variants_for_experiment,
)


# ---------- assemble_candidate_set invariants ----------

def test_invariants_basic():
    cs = assemble_candidate_set(
        capture_session="D2", capture_row=1000, offset=1,
        same_session_chain_rows=list(range(0, 5992)),
        other_session_chain_rows=list(range(0, 3743)),
        other_session_id="V10",
        d_search_max=32, n_same_session_random=192, n_cross_session_random=192,
        seed=42,
    )
    # Invariant 1: positive appears exactly once
    matched = [c for c in cs.candidates if c.family == "matched"]
    assert len(matched) == 1, f"expected 1 matched, got {len(matched)}"
    assert matched[0].session == "D2" and matched[0].row == 1001
    # Invariant 2: dedup within each family
    for fam in FAMILIES:
        keys = [(c.session, c.row) for c in cs.candidates if c.family == fam]
        assert len(keys) == len(set(keys)), f"family {fam} has duplicates"
    # Invariant: positive not in any negative family
    pos_key = (matched[0].session, matched[0].row)
    for c in cs.candidates:
        if c.family != "matched":
            assert (c.session, c.row) != pos_key, f"positive leaked into {c.family}"
    # Invariant: cross_session_random preserves session_id (V10 here)
    cross = [c for c in cs.candidates if c.family == "cross_session_random"]
    assert all(c.session == "V10" for c in cross), "cross-session row had wrong session id"
    # Same-session has the right session
    same = [c for c in cs.candidates if c.family == "same_session_random"]
    assert all(c.session == "D2" for c in same), "same-session candidate had wrong session id"
    print(f"  PASS — invariants check (n_candidates={len(cs.candidates)}, families: "
          f"{ {f: sum(1 for c in cs.candidates if c.family == f) for f in FAMILIES} })")


def test_near_shift_offsets():
    cs = assemble_candidate_set(
        capture_session="D2", capture_row=1000, offset=1,
        same_session_chain_rows=list(range(0, 5992)),
        other_session_chain_rows=list(range(0, 3743)),
        other_session_id="V10",
        d_search_max=32, n_same_session_random=192, n_cross_session_random=192,
        seed=42,
    )
    near = [c for c in cs.candidates if c.family == "near_shift"]
    near_offsets = sorted(c.row - 1001 for c in near)
    expected = sorted(NEAR_SHIFT_OFFSETS)
    assert near_offsets == expected, f"near_shift offsets {near_offsets} != {expected}"
    print(f"  PASS — near_shift = {near_offsets}")


def test_determinism():
    cs1 = assemble_candidate_set(
        capture_session="D2", capture_row=2000, offset=1,
        same_session_chain_rows=list(range(0, 5992)),
        other_session_chain_rows=list(range(0, 3743)),
        other_session_id="V10",
        d_search_max=32, n_same_session_random=192, n_cross_session_random=192,
        seed=99,
    )
    cs2 = assemble_candidate_set(
        capture_session="D2", capture_row=2000, offset=1,
        same_session_chain_rows=list(range(0, 5992)),
        other_session_chain_rows=list(range(0, 3743)),
        other_session_id="V10",
        d_search_max=32, n_same_session_random=192, n_cross_session_random=192,
        seed=99,
    )
    keys1 = [(c.session, c.row, c.family) for c in cs1.candidates]
    keys2 = [(c.session, c.row, c.family) for c in cs2.candidates]
    assert keys1 == keys2, "same seed produced different candidate set"
    print("  PASS — same seed → identical candidate set")


# ---------- compute_tau95 on known distributions ----------

def test_tau95_uniform():
    rng = np.random.default_rng(0)
    # Uniform [0, 1]: 95th percentile is ~0.95
    scores = rng.uniform(0.0, 1.0, size=10000).tolist()
    tau95 = compute_tau95(scores)
    assert abs(tau95 - 0.95) < 0.01, f"uniform tau95={tau95}, expected ≈0.95"
    print(f"  PASS — uniform[0,1] tau95={tau95:.4f}")


def test_tau95_constant():
    scores = [0.5] * 100
    tau95 = compute_tau95(scores)
    assert abs(tau95 - 0.5) < 1e-9, f"constant tau95={tau95}, expected 0.5"
    print(f"  PASS — constant=0.5 tau95={tau95:.4f}")


def test_tau95_empty():
    tau95 = compute_tau95([])
    assert np.isnan(tau95), f"empty tau95={tau95}, expected NaN"
    print(f"  PASS — empty list → NaN")


# ---------- per_family_window_fmr known cases ----------

def test_window_fmr_perfect_separation():
    """All matched scores low, all neg scores high — tau95 cleanly above all negatives → FMR = 0."""
    matched = [0.0] * 100
    tau95 = compute_tau95(matched)  # = 0.0
    neg_per_frame = [{
        "delay_window": [0.5, 0.6, 0.7],
        "near_shift": [0.5],
        "same_session_random": [0.6],
        "cross_session_random": [0.7],
    }] * 100
    out = per_family_window_fmr(
        matched_scores_by_frame=matched,
        negative_scores_by_frame_by_family=neg_per_frame,
        tau95=tau95,
    )
    assert all(v == 0.0 for v in out["per_family"].values()), out["per_family"]
    assert out["worst_family_value"] == 0.0
    assert out["pooled"] == 0.0
    print(f"  PASS — perfect separation → all FMR=0 (worst={out['worst_family_value']})")


def test_window_fmr_complete_overlap():
    """Matched and negative scores both uniform [0,1] — tau95 ≈ 0.95, ~95% of frames have at least one neg ≤ 0.95 → FMR very high (≈1)."""
    rng = np.random.default_rng(0)
    matched = rng.uniform(0, 1, size=200).tolist()
    tau95 = compute_tau95(matched)  # ≈ 0.95
    neg_per_frame = []
    for _ in range(200):
        neg_per_frame.append({
            "delay_window": rng.uniform(0, 1, size=10).tolist(),
            "near_shift": rng.uniform(0, 1, size=10).tolist(),
            "same_session_random": rng.uniform(0, 1, size=10).tolist(),
            "cross_session_random": rng.uniform(0, 1, size=10).tolist(),
        })
    out = per_family_window_fmr(
        matched_scores_by_frame=matched,
        negative_scores_by_frame_by_family=neg_per_frame,
        tau95=tau95,
    )
    # With 10 negs per family per frame, P(min_neg ≤ 0.95) = 1 - 0.05^10 ≈ 1.0
    assert all(v > 0.95 for v in out["per_family"].values()), out["per_family"]
    print(f"  PASS — complete overlap → per_family FMR≈1 (got {out['per_family']})")


def test_window_fmr_partial_overlap():
    """Tunable: matched in [0, 0.5], negatives in [0.3, 1.0]. tau95 ≈ 0.475 cuts the negative distribution at the 0.475-tail; ~12% of negatives below."""
    rng = np.random.default_rng(0)
    matched = rng.uniform(0, 0.5, size=200).tolist()
    tau95 = compute_tau95(matched)
    neg_per_frame = []
    for _ in range(200):
        neg_per_frame.append({
            "delay_window": rng.uniform(0.3, 1.0, size=10).tolist(),
            "near_shift": rng.uniform(0.3, 1.0, size=10).tolist(),
            "same_session_random": rng.uniform(0.3, 1.0, size=10).tolist(),
            "cross_session_random": rng.uniform(0.3, 1.0, size=10).tolist(),
        })
    out = per_family_window_fmr(
        matched_scores_by_frame=matched,
        negative_scores_by_frame_by_family=neg_per_frame,
        tau95=tau95,
    )
    # P(any of 10 in [0.3, 1.0] below tau95≈0.475) = 1 - ((1.0-0.475)/0.7)^10 ≈ 0.95
    # So FMR per family should be ~0.9-0.99, plenty different from 0 or 1.
    for fam, v in out["per_family"].items():
        assert 0.5 < v < 1.0, f"family {fam} FMR={v} unexpected"
    print(f"  PASS — partial overlap → per_family FMR plausible "
          f"(min={min(out['per_family'].values()):.3f}, max={max(out['per_family'].values()):.3f}, "
          f"worst_family={out['worst_family_name']})")


# ---------- score_candidate_xof variant correctness ----------

def test_score_variants_a0_vs_full():
    """A0 should report exactly 2 variants; A1/A2/A6/A7 should report 4."""
    a0 = score_variants_for_experiment("exp001h_a0")
    assert a0 == ("oct0_only", "oct0_plus_oct1"), a0
    full = score_variants_for_experiment("exp001h_a1")
    assert full == ("oct0_only", "oct0_plus_oct1", "oct0_plus_oct1_plus_oct2", "all_octaves"), full
    print(f"  PASS — A0 variants = {a0}; A1+ variants = {full}")


def test_score_zero_when_pred_equals_target():
    """Predicting exactly the target gives score 0 (modulo numerical floor)."""
    octs = [torch.zeros(3, 17, 30), torch.zeros(3, 34, 60),
            torch.zeros(3, 68, 120), torch.zeros(3, 135, 240)]
    s = score_candidate_xof(octs, octs, "all_octaves")
    assert s < 1e-6, f"score for identical pred/target = {s}, expected ~0"
    # And nonzero when target differs
    octs2 = [t + 0.5 for t in octs]
    s2 = score_candidate_xof(octs, octs2, "all_octaves")
    assert s2 > 0.1, f"score for shifted target = {s2}, expected nonzero"
    print(f"  PASS — score(p=t)=0 ({s:.2e}); score(p,t+0.5)={s2:.4f}")


# ---------- run ----------

if __name__ == "__main__":
    print("== assemble_candidate_set invariants ==")
    test_invariants_basic()
    test_near_shift_offsets()
    test_determinism()

    print()
    print("== compute_tau95 ==")
    test_tau95_uniform()
    test_tau95_constant()
    test_tau95_empty()

    print()
    print("== per_family_window_fmr ==")
    test_window_fmr_perfect_separation()
    test_window_fmr_complete_overlap()
    test_window_fmr_partial_overlap()

    print()
    print("== score_candidate_xof ==")
    test_score_variants_a0_vs_full()
    test_score_zero_when_pred_equals_target()

    print()
    print("ALL SYNTHETIC TESTS PASSED")
