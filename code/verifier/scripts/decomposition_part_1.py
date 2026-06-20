"""Decomposition Part 1 — diffusion-feature probe with logreg.

Tests whether a small logistic regression on Phase G diffusion verifier scores
(from existing Stage 0 NPZ outputs) can separate real captures from F-A v1
fakes, and how well that classifier generalises across F-A checkpoints.

METHOD NOTE (2026-05-04):
    Diffusion-feature probe: logistic regression on existing Stage 0 scores.
    Three probes (Raw, Coupling, All); train on fake@{5k,25k,70k}, test on
    fake@100k for the headline, plus leave-one-checkpoint-out (LOOCV) folds.
    CPU-only, runs in minutes.

    Data:
        For each of {5k, 25k, 70k, 100k} F-A checkpoints, per session, per
        frame: 8 mean-MSE values (averaged over 5 timesteps × 4 K_noise),
        4 from real C and 4 from F-A-fake C, under conditions
        {correct, shuffled, source, zero}.

    Sample schema:
        - Each frame contributes one "real" sample (label=0) with features
          derived from cond_real_* values. These features are CHECKPOINT-
          INDEPENDENT so each real sample appears once per fold.
        - Each frame contributes one "fake" sample (label=1) PER F-A
          checkpoint, with features derived from cond_fake_* at that ckpt.

    Probes (feature definitions):
        Raw     (4 features): the 4 condition MSEs (correct, shuffled, source,
                              zero) for whichever C type the sample is from.
        Coupling (3 features): contrasts (shuffled - correct), (source - correct),
                              (zero - correct). Captures coupling structure.
        All     (7 features): Raw + Coupling concatenated.

    Two evaluation modes:
        Headline: train on fake samples from {5k,25k,70k} + all real,
                  test on fake samples from 100k + all real.
                  Real-sample frame-leakage controlled by 80/20 frame split:
                  train uses 80% of frames' real samples; test uses 20%.
        LOOCV:   4 folds. In each, one ckpt is held-out for fake_test;
                  other 3 are train. Same 80/20 frame split for real.

    Train/test for "real" samples is by frame index (deterministic 80/20
    split, seed=0). Train/test for "fake" samples is by checkpoint membership.

    Per probe, AUROC is reported per session and combined.

Output: experiments/decomposition/part_1_diffusion_probe/
    decomp_part1_results.json     — all metrics + per-probe coefs
    decomp_part1_report.md        — human-readable
    raw_features.npz              — features + labels (saved for reuse)

Run: python scripts/decomposition_part_1.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# -------- defaults / paths --------

DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/experiments/decomposition/part_1_diffusion_probe")
DEFAULT_STAGE_0 = Path("/path/to/poliebotics_phase_b/experiments/stage_0/eval")

# Strict path-mode resolution
_PROBE = [DEFAULT_STAGE_0]
_AVAIL = sum(int(p.exists()) for p in _PROBE)
if _AVAIL == len(_PROBE):
    pass
elif _AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_STAGE_0 = LOCAL_ROOT / "experiments" / "stage_0" / "eval"
    DEFAULT_OUT = LOCAL_ROOT / "experiments" / "decomposition" / "part_1_diffusion_probe"
else:
    raise SystemExit(f"[decomp_part1] mixed root state: {_AVAIL}/{len(_PROBE)} Lambda paths exist.")

CKPT_STEPS = (5000, 25000, 70000, 100000)
SESSIONS = ("d2", "v10")
COND_NAMES = ("correct", "shuffled", "source", "zero")
HEADLINE_TRAIN_CKPTS = (5000, 25000, 70000)
HEADLINE_TEST_CKPT = 100000


# -------- feature builders --------

def npz_path(stage_0_root: Path, ckpt: int, sess: str) -> Path:
    return stage_0_root / f"step_{ckpt:08d}" / f"stage0_{sess}_raw.npz"


def load_per_frame_means(npz: dict) -> dict:
    """Average over (timestep, K_noise) per condition. Returns
    {cond_name: (n_frames,) means} for both real_* and fake_* conds.
    """
    out = {}
    for ctype in ("real", "fake"):
        for cond in COND_NAMES:
            key = f"cond_{ctype}_{cond}"
            arr = npz[key]               # (n_frames, 5, 4)
            out[f"{ctype}_{cond}"] = arr.mean(axis=(1, 2))   # (n_frames,)
    return out


def features_raw(per_cond: dict, ctype: str) -> np.ndarray:
    """4 features: the 4 condition mean MSEs."""
    cols = [per_cond[f"{ctype}_{c}"] for c in COND_NAMES]
    return np.stack(cols, axis=1)  # (n_frames, 4)


def features_coupling(per_cond: dict, ctype: str) -> np.ndarray:
    """3 features: (shuffled-correct), (source-correct), (zero-correct)."""
    correct = per_cond[f"{ctype}_correct"]
    cols = [
        per_cond[f"{ctype}_shuffled"] - correct,
        per_cond[f"{ctype}_source"] - correct,
        per_cond[f"{ctype}_zero"] - correct,
    ]
    return np.stack(cols, axis=1)


def features_all(per_cond: dict, ctype: str) -> np.ndarray:
    return np.concatenate([features_raw(per_cond, ctype),
                           features_coupling(per_cond, ctype)], axis=1)


PROBE_FNS = {
    "Raw":      features_raw,
    "Coupling": features_coupling,
    "All":      features_all,
}


# -------- data assembly --------

def assemble_dataset(stage_0_root: Path) -> dict:
    """For each session, load all 4 ckpts and build the full sample matrix.

    Returns {session: {ckpt: per_cond_dict}}.
    """
    out = {}
    for sess in SESSIONS:
        out[sess] = {}
        for ckpt in CKPT_STEPS:
            f = npz_path(stage_0_root, ckpt, sess)
            if not f.exists():
                raise SystemExit(f"missing NPZ at {f}")
            z = np.load(f, allow_pickle=False)
            # Sanity check: frame count matches across ckpts (real conditions
            # are checkpoint-independent — fakes vary).
            out[sess][ckpt] = {
                "per_cond": load_per_frame_means(z),
                "rows": z["rows"],
                "block_idx": z["block_idx"],
            }
    # Verify (rows, block_idx) match across ckpts within session
    for sess in SESSIONS:
        ref_rows = out[sess][CKPT_STEPS[0]]["rows"]
        for ckpt in CKPT_STEPS[1:]:
            if not np.array_equal(out[sess][ckpt]["rows"], ref_rows):
                raise SystemExit(
                    f"row mismatch in {sess} ckpt {ckpt} — schema integrity violated")
    return out


def build_samples(probe_name: str, data: dict, train_frame_mask: np.ndarray,
                  ckpt_membership: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, group) where group encodes (session, sample_type, ckpt).

    train_frame_mask: dict {sess: bool array (n_frames,) — True = train fold}
    ckpt_membership:  dict {ckpt: "train" | "test"} — for fake samples

    Returns concatenated (X, y, group_str_array).
    """
    feat_fn = PROBE_FNS[probe_name]
    X_list, y_list, g_list = [], [], []
    for sess in SESSIONS:
        # Real samples — checkpoint-independent. Use ckpt[0] data.
        per_cond_first = data[sess][CKPT_STEPS[0]]["per_cond"]
        X_real_full = feat_fn(per_cond_first, "real")  # (n_frames, n_feat)
        for is_train_fold, mask_role in [(True, "train"), (False, "test")]:
            mask = train_frame_mask[sess] if is_train_fold else ~train_frame_mask[sess]
            X_list.append(X_real_full[mask])
            y_list.append(np.zeros(int(mask.sum())))
            g_list.extend([f"{sess}|real|{mask_role}"] * int(mask.sum()))
        # Fake samples — one per checkpoint per frame
        for ckpt in CKPT_STEPS:
            per_cond = data[sess][ckpt]["per_cond"]
            X_fake_full = feat_fn(per_cond, "fake")
            ckpt_role = ckpt_membership[ckpt]
            mask = train_frame_mask[sess] if ckpt_role == "train" else ~train_frame_mask[sess]
            X_list.append(X_fake_full[mask])
            y_list.append(np.ones(int(mask.sum())))
            g_list.extend([f"{sess}|fake_ckpt{ckpt}|{ckpt_role}"] * int(mask.sum()))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    g = np.array(g_list)
    return X, y, g


def split_train_test(X: np.ndarray, y: np.ndarray, g: np.ndarray,
                     ckpt_membership: dict) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """Split rows into train/test based on group string suffix '|train' or '|test'."""
    # Frame-fold-based split for real: rows ending with '|real|train' or '|real|test'
    # Ckpt-based split for fake: rows ending with '|fake_ckptK|<role>'
    is_train = np.array([s.endswith("|train") for s in g])
    is_test  = np.array([s.endswith("|test")  for s in g])
    return X[is_train], y[is_train], X[is_test], y[is_test]


def run_logreg_eval(probe_name: str, data: dict,
                     ckpt_membership: dict, frame_mask: dict, seed: int = 0) -> dict:
    """Train logreg on train set, eval on test set.

    Returns dict with per-session AUROC + combined.
    """
    X, y, g = build_samples(probe_name, data, frame_mask, ckpt_membership)
    Xtr, ytr, Xte, yte = split_train_test(X, y, g, ckpt_membership)
    if len(np.unique(ytr)) < 2:
        return {"error": "train set has only one class"}
    if len(np.unique(yte)) < 2:
        return {"error": "test set has only one class"}
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(Xtr_s, ytr)
    score_te = lr.predict_proba(Xte_s)[:, 1]
    auroc = float(roc_auc_score(yte, score_te))

    # Per-session AUROC on test set
    g_train_mask = np.array([s.endswith("|train") for s in g])
    g_test = g[~g_train_mask]
    sess_aurocs = {}
    for sess in SESSIONS:
        sess_mask = np.array([s.startswith(f"{sess}|") for s in g_test])
        if sess_mask.sum() == 0 or len(np.unique(yte[sess_mask])) < 2:
            sess_aurocs[sess] = float("nan")
        else:
            sess_aurocs[sess] = float(roc_auc_score(yte[sess_mask],
                                                     score_te[sess_mask]))
    return {
        "n_train": int(len(ytr)),
        "n_test":  int(len(yte)),
        "auroc_combined": auroc,
        "auroc_per_session": sess_aurocs,
        "n_features": int(Xtr.shape[1]),
        "coefs":    lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
    }


# -------- main --------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-0-root", type=Path, default=DEFAULT_STAGE_0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frame-test-frac", type=float, default=0.2,
                    help="Fraction of frames held out for real-sample test set.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[decomp_part1] loading Stage 0 NPZs from {args.stage_0_root}")
    data = assemble_dataset(args.stage_0_root)
    rs = np.random.RandomState(args.seed)
    frame_mask = {}
    for sess in SESSIONS:
        n_frames = len(data[sess][CKPT_STEPS[0]]["rows"])
        n_train = int((1 - args.frame_test_frac) * n_frames)
        idx = rs.permutation(n_frames)
        train_idx = set(idx[:n_train].tolist())
        frame_mask[sess] = np.array([i in train_idx for i in range(n_frames)])
        print(f"  {sess}: {n_frames} frames, train_mask sum={frame_mask[sess].sum()}")

    # ---- HEADLINE: train 5k/25k/70k → test 100k ----
    print("\n[decomp_part1] HEADLINE: train fake@{5k,25k,70k} → test fake@100k")
    headline_membership = {ck: ("train" if ck != HEADLINE_TEST_CKPT else "test")
                           for ck in CKPT_STEPS}
    headline_results = {}
    for probe in PROBE_FNS:
        res = run_logreg_eval(probe, data, headline_membership, frame_mask)
        headline_results[probe] = res
        if "error" in res:
            print(f"  {probe}: ERROR {res['error']}")
        else:
            print(f"  {probe}: AUROC combined={res['auroc_combined']:.4f}  "
                  f"D2={res['auroc_per_session']['d2']:.4f}  "
                  f"V10={res['auroc_per_session']['v10']:.4f}  "
                  f"(n_feat={res['n_features']}, "
                  f"n_tr={res['n_train']}, n_te={res['n_test']})")

    # ---- LOOCV: 4 folds across checkpoints ----
    print("\n[decomp_part1] LOOCV: 4 folds (hold out one ckpt each)")
    loocv_results = {probe: {} for probe in PROBE_FNS}
    for held_out in CKPT_STEPS:
        membership = {ck: ("train" if ck != held_out else "test") for ck in CKPT_STEPS}
        for probe in PROBE_FNS:
            res = run_logreg_eval(probe, data, membership, frame_mask)
            loocv_results[probe][f"holdout_{held_out}"] = res
            if "error" in res:
                continue
            print(f"  fold(held_out={held_out}) {probe}: "
                  f"AUROC={res['auroc_combined']:.4f}  "
                  f"(D2={res['auroc_per_session']['d2']:.4f}, "
                  f"V10={res['auroc_per_session']['v10']:.4f})")

    # ---- save ----
    summary = {
        "headline": {
            "train_ckpts": list(HEADLINE_TRAIN_CKPTS),
            "test_ckpt":   HEADLINE_TEST_CKPT,
            "frame_test_frac": args.frame_test_frac,
            "results":     headline_results,
        },
        "loocv":    loocv_results,
        "interpretation_notes": [
            "Probes: Raw=4 cond MSEs; Coupling=3 contrasts to 'correct' baseline; All=Raw+Coupling.",
            "Real samples are checkpoint-independent; train/test split via 80/20 frame mask (seed=0).",
            "Fake samples come from the corresponding F-A checkpoint; train/test split by ckpt membership.",
            "AUROC: positive class = fake (label=1). 1.0 = perfect separation, 0.5 = chance.",
            "Probe coefficients in Raw probe correspond to (correct, shuffled, source, zero) MSE weights.",
        ],
        "schema_version": "decomp_part1_v0_cc_interpretation_2026-05-04",
    }
    (args.out / "decomp_part1_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[decomp_part1] decomp_part1_results.json saved")

    write_report(args.out, headline_results, loocv_results)
    print(f"[decomp_part1] report → {args.out / 'decomp_part1_report.md'}")
    return 0


def write_report(out_dir: Path, headline: dict, loocv: dict):
    md = []
    md.append("# Decomposition Part 1 — diffusion-feature probe results")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append("")
    md.append("**Method note**: probe design and probe/feature choices are recorded in `decomp_part1_results.json` under `interpretation_notes`.")
    md.append("")
    md.append("## Headline — train fake@{5k,25k,70k} → test fake@100k")
    md.append("")
    md.append("| probe | n_features | AUROC combined | AUROC D2 | AUROC V10 | n_train | n_test |")
    md.append("|---|---|---|---|---|---|---|")
    for probe, res in headline.items():
        if "error" in res:
            md.append(f"| {probe} | — | — | — | — | — | — | (error: {res['error']}) |")
            continue
        md.append(f"| {probe} | {res['n_features']} | {res['auroc_combined']:.4f} | "
                  f"{res['auroc_per_session']['d2']:.4f} | "
                  f"{res['auroc_per_session']['v10']:.4f} | "
                  f"{res['n_train']} | {res['n_test']} |")
    md.append("")
    md.append("## LOOCV across 4 F-A checkpoints")
    md.append("")
    for probe in loocv:
        md.append(f"### {probe} probe")
        md.append("")
        md.append("| held-out | AUROC combined | AUROC D2 | AUROC V10 |")
        md.append("|---|---|---|---|")
        for fold, res in loocv[probe].items():
            if "error" in res:
                md.append(f"| {fold} | — | — | — |  *(error)*")
                continue
            md.append(f"| {fold} | {res['auroc_combined']:.4f} | "
                      f"{res['auroc_per_session']['d2']:.4f} | "
                      f"{res['auroc_per_session']['v10']:.4f} |")
        md.append("")
    md.append("## Interpretation guide")
    md.append("- **AUROC ≈ 1.0**: real-vs-fake distributions are well-separated in this feature space at the held-out checkpoint.")
    md.append("- **AUROC ≈ 0.5**: classifier cannot distinguish real from fake — F-A is producing fakes that look real to the diffusion verifier in this feature space.")
    md.append("- **Generalisation drop (low headline AUROC, high LOOCV early-ckpt AUROC)**: F-A learned to evade the diffusion-feature signature by step 100k.")
    md.append("- **Probe ordering**: if `Coupling` ≈ `All` >> `Raw`, the discriminative signal lies in conditional contrasts, not absolute MSE values.")
    md.append("")
    (out_dir / "decomp_part1_report.md").write_text("\n".join(md))


if __name__ == "__main__":
    raise SystemExit(main())
