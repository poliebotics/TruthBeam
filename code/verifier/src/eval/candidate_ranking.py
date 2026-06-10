"""Candidate-ranking + Window-FMR@95 primitives (audit A12, A13, A14).

Family taxonomy (audit-confirmed):

    delay_window        — offsets within d_search_max (default 32) excluding
                          the positive (target_chain_row).
    near_shift          — fixed offsets ±1, ±2, ±4, ±8, ±16 (excluding 0
                          and any that collide with the positive).
    same_session_random — `n_same_session_random` (default 192) random rows
                          from the SAME session, outside the delay window.
    cross_session_random— `n_cross_session_random` (default 192) random rows
                          from the OTHER session (session-id preserved).

Each candidate is a `(session_id, target_chain_row)` pair. Cross-session
candidates retain their session, so scoring uses the correct row from the
correct session — row 1234 from V10 ≠ row 1234 from D2.

Invariants enforced:
  1. Positive never in any negative family.
  2. Within-family deduplication.
  3. Family membership tracked separately even if same row appears in
     multiple families (we drop the duplicate from later families).
  4. Fixed RNG seed per evaluated frame, recorded in manifest.
  5. Exact candidate row indices (with session) logged per evaluated frame.

Scoring (centered continuous space, NO rounding):
    candidate_oct = (byte - 127.5) / 127.5
    score = Σ_o α_o * mean(SmoothL1(pred_o, candidate_oct_o))
    Lower = better. Octave weights `α` taken from the loss's OCTAVE_WEIGHTS.

Score variants:
    A0:        ("oct0_only", "oct0_plus_oct1")
    A1/A2/A6/A7: full set ("oct0_only", "oct0_plus_oct1",
                            "oct0_plus_oct1_plus_oct2", "all_octaves")

Window-FMR@95 (audit A12):
    τ95 = 95th percentile of POSITIVE scores from the calibration set.
    Window-FMR = fraction of frames where ANY wrong candidate's score ≤ τ95.
    (Recall: lower = better, so a wrong candidate ≤ τ95 means it scored at
    least as well as the worst 95% of true positives — a false match.)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

FAMILIES = ("matched", "delay_window", "near_shift", "same_session_random", "cross_session_random")
NEAR_SHIFT_OFFSETS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)

# Indexed by octave number; matches losses/huber_xof.OCTAVE_WEIGHTS.
OCTAVE_WEIGHTS = (1.0, 1.0, 0.5, 0.25)


@dataclass
class Candidate:
    session: str            # "D2" or "V10"
    row: int                # target_chain_row in that session
    family: str             # one of FAMILIES


@dataclass
class CandidateSet:
    candidates: list[Candidate] = field(default_factory=list)
    capture_session: str = ""
    capture_row: int = -1
    target_chain_row: int = -1
    seed: int = 0


def assemble_candidate_set(
    *,
    capture_session: str,
    capture_row: int,
    offset: int,
    same_session_chain_rows: Sequence[int],
    other_session_chain_rows: Sequence[int],
    other_session_id: str,
    d_search_max: int,
    n_same_session_random: int = 192,
    n_cross_session_random: int = 192,
    seed: int = 0,
) -> CandidateSet:
    """Build the full candidate set for one evaluated frame.

    `same_session_chain_rows` and `other_session_chain_rows` are the chain
    `t` indices available in each session (typically the union of train+val
    chain rows that have valid emission tiles).
    """
    target = capture_row + offset
    same_set = set(int(r) for r in same_session_chain_rows)
    other_set = set(int(r) for r in other_session_chain_rows)
    rng = random.Random(seed)
    cs = CandidateSet(
        capture_session=capture_session,
        capture_row=capture_row,
        target_chain_row=target,
        seed=seed,
    )
    cs.candidates.append(Candidate(capture_session, target, "matched"))

    used: set[tuple[str, int]] = {(capture_session, target)}

    # delay_window: every offset in [-d_search_max, +d_search_max] except 0 (= matched)
    for d in range(-d_search_max, d_search_max + 1):
        if d == 0:
            continue
        r = target + d
        key = (capture_session, r)
        if key in used:
            continue
        if r in same_set:
            cs.candidates.append(Candidate(capture_session, r, "delay_window"))
            used.add(key)

    # near_shift: fixed offsets ±1, ±2, ±4, ±8, ±16 (already inside delay_window, but
    # tracked separately as a distinct family — we MOVE them to near_shift, removing
    # the delay_window duplicates).
    near_rows = {target + d for d in NEAR_SHIFT_OFFSETS}
    new_candidates: list[Candidate] = []
    for c in cs.candidates:
        if c.family == "delay_window" and c.row in near_rows and c.session == capture_session:
            new_candidates.append(Candidate(c.session, c.row, "near_shift"))
        else:
            new_candidates.append(c)
    cs.candidates = new_candidates

    # same_session_random: rows from same session, outside the delay window
    delay_keys = {(capture_session, target + d) for d in range(-d_search_max, d_search_max + 1)}
    same_pool = [r for r in same_set if (capture_session, r) not in delay_keys]
    same_pool = list({r: None for r in same_pool})  # dedup, preserve order
    rng.shuffle(same_pool)
    n_taken = 0
    for r in same_pool:
        key = (capture_session, r)
        if key in used:
            continue
        cs.candidates.append(Candidate(capture_session, r, "same_session_random"))
        used.add(key)
        n_taken += 1
        if n_taken >= n_same_session_random:
            break

    # cross_session_random: rows from the OTHER session
    other_pool = list({int(r): None for r in other_set})
    rng.shuffle(other_pool)
    n_taken = 0
    for r in other_pool:
        key = (other_session_id, r)
        if key in used:
            continue
        cs.candidates.append(Candidate(other_session_id, r, "cross_session_random"))
        used.add(key)
        n_taken += 1
        if n_taken >= n_cross_session_random:
            break

    return cs


def _octave_weights_for_variant(variant: str) -> tuple[float, ...]:
    """Return per-octave weights to use when computing this variant's score."""
    base = OCTAVE_WEIGHTS
    if variant == "oct0_only":
        return (base[0], 0.0, 0.0, 0.0)
    if variant == "oct0_plus_oct1":
        return (base[0], base[1], 0.0, 0.0)
    if variant == "oct0_plus_oct1_plus_oct2":
        return (base[0], base[1], base[2], 0.0)
    if variant == "all_octaves":
        return base
    raise ValueError(f"unknown variant {variant!r}")


def _byte_to_centered(byte_tensor: torch.Tensor) -> torch.Tensor:
    return (byte_tensor.to(torch.float32) - 127.5) / 127.5


def score_candidate_xof(
    pred_octaves: list[torch.Tensor | None],
    candidate_target_octaves_centered: list[torch.Tensor | None],
    variant: str,
    beta: float = 1.0,
) -> float:
    """SmoothL1 sum-over-octaves with variant weights. Lower = better."""
    weights = _octave_weights_for_variant(variant)
    score = 0.0
    for i, (p, t) in enumerate(zip(pred_octaves, candidate_target_octaves_centered)):
        if p is None or t is None or weights[i] == 0.0:
            continue
        score += weights[i] * F.smooth_l1_loss(p, t, beta=beta).item()
    return float(score)


def score_variants_for_experiment(experiment_id: str) -> tuple[str, ...]:
    """A0 supports oct0_only and oct0_plus_oct1 only; others get the full set."""
    if "a0" in experiment_id.lower():
        return ("oct0_only", "oct0_plus_oct1")
    return ("oct0_only", "oct0_plus_oct1", "oct0_plus_oct1_plus_oct2", "all_octaves")


def per_family_window_fmr(
    *,
    matched_scores_by_frame: list[float],
    negative_scores_by_frame_by_family: list[dict[str, list[float]]],
    tau95: float,
) -> dict:
    """Compute per-family Window-FMR@τ95 + aggregations.

    Args:
        matched_scores_by_frame: scores of the matched candidate per frame (used to
            sanity-check; tau95 is computed elsewhere from the calibration half).
        negative_scores_by_frame_by_family: one entry per frame; each is a dict
            family -> list of negative-candidate scores.
        tau95: 95th percentile of positive scores from the calibration half.

    Returns dict with:
        per_family[family]: fraction of frames where any negative in that family ≤ tau95
        worst_family: max over families (PRIMARY metric)
        macro_average: mean over families (secondary)
        pooled: fraction of frames where any negative across ALL families ≤ tau95 (tertiary)
        n_frames: total frames
    """
    n = len(negative_scores_by_frame_by_family)
    if n == 0:
        return {"per_family": {}, "worst_family": float("nan"),
                "macro_average": float("nan"), "pooled": float("nan"), "n_frames": 0}

    families = [f for f in FAMILIES if f != "matched"]
    per_family: dict[str, float] = {}
    pooled_count = 0
    for f in families:
        hits = 0
        for frame_dict in negative_scores_by_frame_by_family:
            scores = frame_dict.get(f, [])
            if any(s <= tau95 for s in scores):
                hits += 1
        per_family[f] = hits / n

    for frame_dict in negative_scores_by_frame_by_family:
        all_neg = [s for fam_scores in frame_dict.values() for s in fam_scores]
        if any(s <= tau95 for s in all_neg):
            pooled_count += 1

    return {
        "per_family": per_family,
        "worst_family_value": max(per_family.values()),
        "worst_family_name": max(per_family, key=per_family.get),
        "macro_average": float(np.mean(list(per_family.values()))),
        "pooled": pooled_count / n,
        "n_frames": n,
        "tau95": tau95,
    }


def compute_tau95(positive_scores: Iterable[float]) -> float:
    arr = np.asarray(list(positive_scores), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, 0.95))
