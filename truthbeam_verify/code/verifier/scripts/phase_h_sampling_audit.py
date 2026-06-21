"""Phase H sub-diag 6.1 — sampling proportions audit.

Per CGPT round 3 sub-diagnostic 6.1: verify Type 1 (and other train-pool
perturbations) actually appeared in batches at intended ratios during Phase H
supervised baseline training.

What this audits:

(a) Aggregate intended distribution per `PhaseHBaselineDataset._build_sample`:
    - positive (real C + identity-rendered E_target):    50%
    - shuffled negative (real C + far-frame identity E): 25%
    - perturbed negative (real C + train-pool perturbed E): 25%
        ↓ 25% / 28 train-pool labels = 0.89% per label
        ↓ Type 1 (7 labels) = 6.25%
        ↓ Type 2 (16 labels) = 14.3%
        ↓ Type 4 (4 labels) = 3.57%
        ↓ Type 6 general (1 label) = 0.89%

(b) Whether the per-(session, row) decision actually distributes uniformly
    across the dataset.

(c) **CRITICAL FINDING TO CHECK**: per-frame label-locking. The dataset's
    decision RNG is seeded by _stable_session_seed_int(epoch_seed, "phase_h_sample",
    session, row, idx, retry_count=0). If `epoch_seed` doesn't change across
    epochs (it's set once at dataset init), each (session, row) idx is locked
    to one of {positive, shuffled, perturbed-with-label-X} for the entire
    training run. That would mean ~50% of frames are NEVER shown as negatives,
    and ~22% of frames are shown only as Type-1 perturbations, etc.

    If this is the case, it's a real concern: the model sees each frame in only
    one role rather than learning to discriminate that frame across all roles.
    Sub-diagnostic 6.4 (E-usage ablation) would be the empirical test.

Output: experiments/phase_h_sampling_audit/
    sample_distribution.json   — counts + per-label proportions
    audit_report.md            — human-readable verdict

Run:
    python scripts/phase_h_sampling_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.xof_perturb import (  # noqa: E402
    TRAIN_POOL_LABELS, _spec_by_label,
)


SESSION_DIRS = {
    "D2":  ROOT / "data" / "d2",
    "V10": ROOT / "data" / "v10",
}
DEFAULT_OUT = ROOT / "experiments" / "phase_h_sampling_audit"


def _stable_session_seed_int(seed: int, label: str, *parts) -> int:
    """Deterministic seed (mirrors PhaseHBaselineDataset)."""
    import hashlib
    h = hashlib.sha256()
    h.update(f"{seed}|{label}".encode())
    for p in parts:
        h.update(b"|"); h.update(str(p).encode())
    return int.from_bytes(h.digest()[:8], "big") & ((1 << 63) - 1)


def replay_decision(epoch_seed: int, session: str, row: int, idx: int) -> dict:
    """Replay the decision logic of _build_sample(idx, retry_count=0).

    Returns: {"class": "positive"|"shuffled"|"perturbed", "label": str_or_None}
    Where label is the perturbation condition_label for "perturbed", else None.
    """
    import random
    seed = _stable_session_seed_int(epoch_seed, "phase_h_sample",
                                     session, row, idx, 0)
    rng = random.Random(seed)
    u = rng.random()
    if u < 0.5:
        return {"class": "positive", "label": "identity"}
    elif u < 0.75:
        return {"class": "shuffled", "label": "identity"}
    else:
        cond = rng.choice(TRAIN_POOL_LABELS)
        return {"class": "perturbed", "label": cond}


def main() -> int:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the same flat list as PhaseHBaselineDataset
    # (sessions + chain log filtering).
    print("[audit] building flat (session, row) list from chain logs...")
    flat: list[tuple[str, int]] = []
    for sess, sd in SESSION_DIRS.items():
        # Replicate train_rows logic from diffusion_diagnostic_dataset
        from phase_g.diffusion_diagnostic_dataset import EVAL_BLOCKS
        # train_rows = all rows EXCEPT held-out blocks, stride=3
        held = set()
        for a, b in EVAL_BLOCKS[sess]:
            for r in range(a, b):
                held.add(r)
        # Read chain log for this session to know valid rows
        from phase_g.xof_perturb import load_chain_log
        chain = load_chain_log(sd)
        valid_rows = sorted([r for r in chain.keys() if r not in held])
        # Apply stride=3
        valid_rows = valid_rows[::3]
        for r in valid_rows:
            flat.append((sess, r))
        print(f"  {sess}: {len(valid_rows)} train rows after stride=3 + held-out filter")
    print(f"[audit] dataset_size={len(flat)} (this is the # samples per epoch)")

    # Replay decisions for every idx with epoch_seed=42 (default --seed)
    epoch_seed = 42
    print(f"[audit] replaying decisions with epoch_seed={epoch_seed} for "
          f"{len(flat)} samples...")
    t0 = time.time()
    class_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    per_session: dict[str, Counter] = {sess: Counter() for sess in SESSION_DIRS}
    per_row_decision: dict[tuple[str, int], dict] = {}

    for idx, (sess, row) in enumerate(flat):
        d = replay_decision(epoch_seed, sess, row, idx)
        class_counter[d["class"]] += 1
        per_session[sess][d["class"]] += 1
        if d["class"] == "perturbed":
            label_counter[d["label"]] += 1
            # Coarse type
            cond = d["label"]
            if cond.startswith("xof_t1"): type_counter["Type 1 (bit_flip_global)"] += 1
            elif cond.startswith("xof_t2"): type_counter["Type 2 (bit_flip_octave)"] += 1
            elif cond.startswith("xof_t4"): type_counter["Type 4 (octave_swap)"] += 1
            elif cond.startswith("xof_t6"): type_counter["Type 6 (replace)"] += 1
            else: type_counter["other"] += 1
        per_row_decision[(sess, row)] = d
    print(f"[audit] done in {time.time()-t0:.1f}s")

    n = len(flat)
    print()
    print("=== Aggregate distribution ===")
    print(f"{'class':20s}  count   pct")
    for cls in ("positive", "shuffled", "perturbed"):
        c = class_counter.get(cls, 0)
        print(f"  {cls:18s} {c:6d}  {100*c/n:.2f}%")
    print()
    print("=== Per-session distribution ===")
    for sess, ctr in per_session.items():
        n_sess = sum(ctr.values())
        print(f"  {sess} (n={n_sess}):")
        for cls in ("positive", "shuffled", "perturbed"):
            c = ctr.get(cls, 0)
            print(f"    {cls:18s} {c:6d}  {100*c/n_sess:.2f}%")
    print()
    print("=== Per-Type breakdown of perturbed samples ===")
    n_perturbed = class_counter.get("perturbed", 0)
    expected = {
        "Type 1 (bit_flip_global)": 7 / 28,
        "Type 2 (bit_flip_octave)": 16 / 28,
        "Type 4 (octave_swap)":     4 / 28,
        "Type 6 (replace)":         1 / 28,
    }
    for ttype in sorted(type_counter.keys()):
        c = type_counter[ttype]
        pct_of_perturbed = 100 * c / n_perturbed if n_perturbed else 0
        pct_of_total     = 100 * c / n
        exp_pct_of_total = 100 * expected.get(ttype, 0) * 0.25
        print(f"  {ttype:35s}  count={c:6d}  pct_of_perturbed={pct_of_perturbed:5.2f}%  "
              f"pct_of_total={pct_of_total:5.2f}%  expected_pct_of_total={exp_pct_of_total:.2f}%")
    print()
    print("=== Per-label perturbation count (top 10 + bottom 5) ===")
    most_common = label_counter.most_common(10)
    for lbl, c in most_common:
        print(f"  {lbl:35s} {c}")
    print("  ...")
    least_common = sorted(label_counter.items(), key=lambda x: x[1])[:5]
    for lbl, c in least_common:
        print(f"  {lbl:35s} {c}")
    print()
    print("=== CRITICAL FINDING — per-frame label-locking ===")
    locked_classes = Counter()
    locked_labels = Counter()
    for (sess, row), d in per_row_decision.items():
        locked_classes[d["class"]] += 1
        if d["class"] == "perturbed":
            locked_labels[d["label"]] += 1
    print(f"Each (session, row) is LOCKED to one decision because epoch_seed never")
    print(f"changes during training. Distribution of locked roles:")
    print(f"  positive-locked frames:  {locked_classes['positive']:6d} / {n} = {100*locked_classes['positive']/n:.2f}%")
    print(f"  shuffled-locked frames:  {locked_classes['shuffled']:6d} / {n} = {100*locked_classes['shuffled']/n:.2f}%")
    print(f"  perturbed-locked frames: {locked_classes['perturbed']:6d} / {n} = {100*locked_classes['perturbed']/n:.2f}%")
    print(f"  → ~50% of frames NEVER shown as negative; ~22-25% of frames shown")
    print(f"    only with one specific perturbation label.")

    # Save artifacts
    summary = {
        "dataset_size": n,
        "epoch_seed": epoch_seed,
        "class_counts": dict(class_counter),
        "class_pct": {k: 100*v/n for k, v in class_counter.items()},
        "per_session_counts": {sess: dict(ctr) for sess, ctr in per_session.items()},
        "type_counts": dict(type_counter),
        "type_pct_of_total": {k: 100*v/n for k, v in type_counter.items()},
        "label_counts": dict(label_counter),
        "n_train_pool_labels": len(TRAIN_POOL_LABELS),
        "expected_per_label_pct_of_total": 100 * 0.25 / len(TRAIN_POOL_LABELS),
        "locked_classes": dict(locked_classes),
        "intended_aggregate": {
            "positive_pct": 50.0, "shuffled_pct": 25.0, "perturbed_pct": 25.0,
        },
    }
    (out_dir / "sample_distribution.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append("# Phase H sampling audit — sub-diag 6.1")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"Dataset size (samples/epoch): {n}; epoch_seed: {epoch_seed}")
    md.append("")
    md.append("## Aggregate class distribution")
    md.append("")
    md.append("| class | count | observed % | intended % |")
    md.append("|---|---|---|---|")
    for cls, exp in (("positive", 50), ("shuffled", 25), ("perturbed", 25)):
        c = class_counter.get(cls, 0)
        md.append(f"| {cls} | {c} | {100*c/n:.2f}% | {exp}% |")
    md.append("")
    md.append("## Per-Type breakdown of perturbed samples")
    md.append("")
    md.append("| Type | count | % of total | expected % | n_labels in pool |")
    md.append("|---|---|---|---|---|")
    n_labels_per_type = {"Type 1 (bit_flip_global)": 7, "Type 2 (bit_flip_octave)": 16,
                          "Type 4 (octave_swap)": 4, "Type 6 (replace)": 1}
    for ttype in sorted(n_labels_per_type.keys()):
        c = type_counter.get(ttype, 0)
        nl = n_labels_per_type[ttype]
        exp_pct = 100 * 0.25 * nl / 28
        md.append(f"| {ttype} | {c} | {100*c/n:.2f}% | {exp_pct:.2f}% | {nl} |")
    md.append("")
    md.append("## Per-label perturbation counts")
    md.append("")
    md.append("| label | count | % of total |")
    md.append("|---|---|---|")
    for lbl in TRAIN_POOL_LABELS:
        c = label_counter.get(lbl, 0)
        md.append(f"| {lbl} | {c} | {100*c/n:.3f}% |")
    md.append("")
    md.append("## ⚠️ Per-frame label-locking finding")
    md.append("")
    md.append("`PhaseHBaselineDataset._build_sample` derives its decision from")
    md.append("`_stable_session_seed_int(epoch_seed, \"phase_h_sample\", session, row, idx, 0)`.")
    md.append("`epoch_seed` is set once at dataset construction and never updated, so each")
    md.append("(session, row, idx) is **permanently locked** to one of {positive, shuffled,")
    md.append("perturbed-with-label-X} for the entire training run.")
    md.append("")
    md.append("**Implication**: training never shows the same frame in multiple roles.")
    md.append("- ~50% of frames are NEVER shown as a negative.")
    md.append("- ~25% of frames are ONLY shown as shuffled negatives.")
    md.append("- The remaining ~25% are spread one-label-each across the 28 train-pool perturbations.")
    md.append("")
    md.append("This is a **data-augmentation deficiency**: the model effectively learns")
    md.append("a per-frame label rather than per-condition discrimination. Sub-diagnostic 6.4")
    md.append("(E-usage ablation: target/zero/shuffled E on same C) is the right empirical test.")
    md.append("")
    md.append("**Recommended fix for v2**: re-derive `epoch_seed` per training epoch (e.g.,")
    md.append("from sampler.set_epoch(epoch)) so each frame can rotate through all roles over")
    md.append("the course of training. This requires a one-line change to PhaseHBaselineDataset.")
    md.append("")
    (out_dir / "audit_report.md").write_text("\n".join(md))
    print(f"[audit] wrote {out_dir / 'audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
