"""Tier 0 Analysis A — Frame-level bootstrap CIs across 5 experiments.

Per CGPT round 4 + operator authorization 2026-05-03: paper-ready bootstrap
confidence intervals using frame-level resampling units (NOT row-level).

Methodology:
    1. Per condition NPZ (shape (n_frames, n_timesteps, n_K_noise)), aggregate
       per frame: score_frame[i] = mean_{t,k} cond[i, t, k].
    2. Bootstrap: resample frames with replacement (1000 iters), stratify by
       block when block_idx is available, compute per-resample:
           - AUROC (positive = correct=target, negative = wrong/shuffled/etc)
           - Paired win rate: fraction of frames where target < wrong score
           - Mean Δ = mean(wrong - target)
           - 5th-percentile Δ
           - Min Δ
           - Inversions (frames where target ≥ wrong, ordering breaks)
    3. 95% CI = [2.5%, 97.5%] of resamples per metric.
    4. AUROC=1.0 special handling: if ALL frame-level orderings agree, the
       AUROC bootstrap CI is degenerate [1.0, 1.0]. Report number of frames
       and all-frames-agree flag explicitly so manuscript can interpret.

Experiments covered:
    1. Phase G main vs wrong-E (D2, V10) — confirms chain-coupled signal
    2. Phase G shuffled control vs same — verifies 0.5 chance baseline
    3. Cross-session d2_only / v10_only verifiers (both directions)
    4. Stage 0 F-A v1 per checkpoint × {correct, shuffled, source, zero}
    5. Item 1 perturbation sensitivity (per condition aggregate)

Output:
    experiments/paper_analyses/bootstrap_cis/
        robustness_table.md
        raw_resamples.npz
        bootstrap_methodology.md

Run: python scripts/paper_analyses/bootstrap_cis.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


LOCAL_ROOT = Path("/path/to/poliebotics_phase_b")
DEFAULT_OUT = LOCAL_ROOT / "experiments" / "paper_analyses" / "bootstrap_cis"
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 42


def per_frame_means(z: np.lib.npyio.NpzFile, cond_key: str) -> np.ndarray:
    """Aggregate (n_frames, T, K) → (n_frames,) by mean over (T, K)."""
    arr = z[cond_key]
    return arr.mean(axis=(1, 2))


def bootstrap_pair(target: np.ndarray, wrong: np.ndarray,
                    block_idx: np.ndarray | None = None,
                    n_iters: int = DEFAULT_N_BOOTSTRAP,
                    seed: int = DEFAULT_SEED) -> dict:
    """Compute bootstrap stats for a paired target-vs-wrong comparison.

    target, wrong: (n_frames,) per-frame mean MSEs (lower = better for target).
    block_idx: (n_frames,) optional block labels for stratified resampling.

    Returns dict with point estimates + 95% CIs for AUROC / win rate / Δ stats.
    """
    n = min(len(target), len(wrong))
    target = target[:n]; wrong = wrong[:n]
    rng = np.random.default_rng(seed)

    # Point estimates (no resample)
    auroc_point = float(roc_auc_score(
        np.concatenate([np.ones(n), np.zeros(n)]),
        -np.concatenate([target, wrong])))   # convention: positive=correct=lower MSE
    delta = wrong - target  # positive = target wins
    win_rate_point = float((delta > 0).mean())
    mean_delta_point = float(delta.mean())
    p5_delta_point = float(np.percentile(delta, 5))
    min_delta_point = float(delta.min())
    inversions_point = int((delta <= 0).sum())

    # Bootstrap
    aurocs, win_rates, mean_deltas, p5_deltas, min_deltas, inversions_list = [], [], [], [], [], []
    if block_idx is not None and len(block_idx) >= n:
        # Stratified: resample within each block
        blocks = block_idx[:n]
        unique_blocks = np.unique(blocks)
        block_to_idxs = {b: np.where(blocks == b)[0] for b in unique_blocks}
    else:
        unique_blocks = None
        block_to_idxs = None

    for _ in range(n_iters):
        if unique_blocks is not None:
            sampled_idx = np.concatenate([
                rng.choice(block_to_idxs[b], size=len(block_to_idxs[b]), replace=True)
                for b in unique_blocks
            ])
        else:
            sampled_idx = rng.choice(n, size=n, replace=True)
        t = target[sampled_idx]
        w = wrong[sampled_idx]
        d = w - t
        try:
            auc = roc_auc_score(
                np.concatenate([np.ones(len(t)), np.zeros(len(w))]),
                -np.concatenate([t, w]))
        except ValueError:
            auc = float("nan")
        aurocs.append(auc)
        win_rates.append(float((d > 0).mean()))
        mean_deltas.append(float(d.mean()))
        p5_deltas.append(float(np.percentile(d, 5)))
        min_deltas.append(float(d.min()))
        inversions_list.append(int((d <= 0).sum()))

    aurocs = np.array(aurocs); win_rates = np.array(win_rates)
    mean_deltas = np.array(mean_deltas); p5_deltas = np.array(p5_deltas)
    min_deltas = np.array(min_deltas); inversions_arr = np.array(inversions_list)

    return {
        "n_frames": int(n),
        "n_iters": n_iters,
        "stratified": unique_blocks is not None,
        "n_blocks": int(len(unique_blocks)) if unique_blocks is not None else 0,
        "all_frames_agree": bool(inversions_point == 0),
        "auroc": {
            "point": auroc_point,
            "ci_lo": float(np.nanpercentile(aurocs, 2.5)),
            "ci_hi": float(np.nanpercentile(aurocs, 97.5)),
        },
        "win_rate": {
            "point": win_rate_point,
            "ci_lo": float(np.percentile(win_rates, 2.5)),
            "ci_hi": float(np.percentile(win_rates, 97.5)),
        },
        "mean_delta": {
            "point": mean_delta_point,
            "ci_lo": float(np.percentile(mean_deltas, 2.5)),
            "ci_hi": float(np.percentile(mean_deltas, 97.5)),
        },
        "p5_delta": {"point": p5_delta_point,
                      "ci_lo": float(np.percentile(p5_deltas, 2.5)),
                      "ci_hi": float(np.percentile(p5_deltas, 97.5))},
        "min_delta": {"point": min_delta_point,
                       "ci_lo": float(np.percentile(min_deltas, 2.5)),
                       "ci_hi": float(np.percentile(min_deltas, 97.5))},
        "inversions": {"point": inversions_point,
                        "ci_lo": float(np.percentile(inversions_arr, 2.5)),
                        "ci_hi": float(np.percentile(inversions_arr, 97.5))},
    }


def fmt_ci(stat: dict, fmt: str = "{:.4f}") -> str:
    return f"{fmt.format(stat['point'])} [{fmt.format(stat['ci_lo'])}, {fmt.format(stat['ci_hi'])}]"


# ---------------- experiment definitions ----------------

PG_MAIN = LOCAL_ROOT / "experiments" / "phase_g_diffusion_diagnostic" / "main" / "eval"
PG_SHUFFLED = LOCAL_ROOT / "experiments" / "phase_g_diffusion_diagnostic" / "shuffled" / "eval"
CROSS_D2 = LOCAL_ROOT / "experiments" / "cross_session_ablation" / "d2_only" / "eval"
CROSS_V10 = LOCAL_ROOT / "experiments" / "cross_session_ablation" / "v10_only" / "eval"
STAGE_0 = LOCAL_ROOT / "experiments" / "stage_0" / "eval"
ITEM_1 = LOCAL_ROOT / "experiments" / "item_1" / "eval"


def run_phase_g_pair(npz_dir: Path, label_prefix: str, sess: str,
                      negatives: list[str], seed: int) -> list[dict]:
    """Phase G main / shuffled / cross-session: target=cond_correct vs each
    negative cond. negatives is a list of cond_* keys excluding 'cond_correct'."""
    npz_path = npz_dir / f"eval_{sess}_raw.npz"
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=False)
    if "cond_correct" not in z.files: return []
    target_pf = per_frame_means(z, "cond_correct")
    block_idx = z["block_idx"] if "block_idx" in z.files else None

    rows = []
    for neg in negatives:
        if neg not in z.files: continue
        wrong_pf = per_frame_means(z, neg)
        stats = bootstrap_pair(target_pf, wrong_pf, block_idx=block_idx, seed=seed)
        rows.append({
            "experiment": label_prefix,
            "session": sess.upper(),
            "comparison": neg.replace("cond_", ""),
            "stats": stats,
        })
    return rows


def run_stage_0(ckpt_step: int, sess: str, seed: int) -> list[dict]:
    """Stage 0 F-A v1: per-checkpoint bootstrap on (real_correct vs fake_correct)
    plus (fake_correct vs fake_shuffled/source/zero)."""
    ckpt_dir = STAGE_0 / f"step_{ckpt_step:08d}"
    npz_path = ckpt_dir / f"stage0_{sess}_raw.npz"
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=False)
    block_idx = z["block_idx"] if "block_idx" in z.files else None

    rows = []
    # Pair 1: real_correct (the verifier's "real with target E") vs
    #         fake_correct (F-A's C_fake with target E). Target=real_correct.
    real_correct_pf = per_frame_means(z, "cond_real_correct")
    fake_correct_pf = per_frame_means(z, "cond_fake_correct")
    rows.append({
        "experiment": f"Stage 0 step_{ckpt_step}",
        "session": sess.upper(),
        "comparison": "real_correct vs fake_correct (lower=real-like)",
        "stats": bootstrap_pair(real_correct_pf, fake_correct_pf,
                                  block_idx=block_idx, seed=seed),
    })
    # Pair 2: fake_correct (target E) vs fake_shuffled (wrong E) — does F-A
    #         couple to its target E? Target=fake_correct.
    fake_shuffled_pf = per_frame_means(z, "cond_fake_shuffled")
    rows.append({
        "experiment": f"Stage 0 step_{ckpt_step}",
        "session": sess.upper(),
        "comparison": "fake_correct vs fake_shuffled (E-coupling on F-A side)",
        "stats": bootstrap_pair(fake_correct_pf, fake_shuffled_pf,
                                  block_idx=block_idx, seed=seed),
    })
    return rows


def run_item_1(sess: str, seed: int, perturbations: list[str]) -> list[dict]:
    """Item 1: target=cond_correct vs each perturbation."""
    npz_path = ITEM_1 / f"eval_{sess}_raw.npz"
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=False)
    if "cond_correct" not in z.files: return []
    target_pf = per_frame_means(z, "cond_correct")
    block_idx = z["block_idx"] if "block_idx" in z.files else None

    rows = []
    for p in perturbations:
        cond = f"cond_{p}"
        if cond not in z.files: continue
        wrong_pf = per_frame_means(z, cond)
        stats = bootstrap_pair(target_pf, wrong_pf, block_idx=block_idx, seed=seed)
        rows.append({
            "experiment": "Item 1 perturbations",
            "session": sess.upper(),
            "comparison": p,
            "stats": stats,
        })
    return rows


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-iters", type=int, default=DEFAULT_N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[bootstrap_cis] n_iters={args.n_iters} seed={args.seed}")
    t0 = time.time()
    all_rows = []

    # 1. Phase G main vs wrong-E (D2, V10) + shuffled control
    pg_negatives = ["cond_uncond"] + [f"cond_wrong_{o:+d}" for o in (-30, -15, -5, -2, -1, 1, 2, 5, 15, 30)
                                       if f"cond_wrong_{o:+d}" in [f"cond_wrong_{o:+d}" for o in (-30,-15,-5,-2,-1,1,2,5,15,30)]]
    # The actual condition keys in Phase G NPZ are cond_wrong_{-2,+2,-15,+15,+30} only:
    pg_negatives = ["cond_uncond", "cond_wrong_-2", "cond_wrong_+2",
                    "cond_wrong_-15", "cond_wrong_+15", "cond_wrong_+30"]
    for sess in ("d2", "v10"):
        all_rows.extend(run_phase_g_pair(PG_MAIN, "Phase G main", sess, pg_negatives, args.seed))
    print(f"  Phase G main done ({len(all_rows)} rows so far, {time.time()-t0:.0f}s)")

    # 2. Phase G shuffled control alone (verify ~0.5)
    for sess in ("d2", "v10"):
        all_rows.extend(run_phase_g_pair(PG_SHUFFLED, "Phase G shuffled-control", sess,
                                          pg_negatives, args.seed))
    print(f"  Phase G shuffled done ({len(all_rows)} rows so far, {time.time()-t0:.0f}s)")

    # 3. Cross-session ablation (both verifiers, both sessions)
    for sess in ("d2", "v10"):
        all_rows.extend(run_phase_g_pair(CROSS_D2, "Cross d2_only verifier", sess,
                                          pg_negatives, args.seed))
        all_rows.extend(run_phase_g_pair(CROSS_V10, "Cross v10_only verifier", sess,
                                          pg_negatives, args.seed))
    print(f"  Cross-session done ({len(all_rows)} rows so far, {time.time()-t0:.0f}s)")

    # 4. Stage 0 F-A v1 (4 ckpts × 2 sessions)
    for ckpt in (5000, 25000, 70000, 100000):
        for sess in ("d2", "v10"):
            all_rows.extend(run_stage_0(ckpt, sess, args.seed))
    print(f"  Stage 0 done ({len(all_rows)} rows so far, {time.time()-t0:.0f}s)")

    # 5. Item 1 perturbation sensitivity (per condition)
    item_1_perts = (
        # Type 1 global bit-flip
        [f"xof_t1_global_k{k}" for k in (1, 4, 16, 64, 256, 1024, 4096)]
        # Type 2 octave bit-flip (16 conditions)
        + [f"xof_t2_oct{o}_k{k}" for o in (0, 1, 2, 3) for k in (1, 4, 16, 64)]
        # Type 3 region bit-flip
        + [f"xof_t3_region_k{k}" for k in (16, 64, 256)]
        # Type 4 octave swap
        + [f"xof_t4_swap_oct{o}" for o in (0, 1, 2, 3)]
        # Type 5 channel swap
        + [f"xof_t5_swap_{c}" for c in ("R", "G", "B")]
        # Type 6
        + ["xof_t6_replace_general", "xof_t6_calibration_row+30"]
    )
    for sess in ("d2", "v10"):
        all_rows.extend(run_item_1(sess, args.seed, item_1_perts))
    print(f"  Item 1 done ({len(all_rows)} rows so far, {time.time()-t0:.0f}s)")

    print(f"[bootstrap_cis] {len(all_rows)} comparisons total in {time.time()-t0:.0f}s wall")

    # Save raw resamples (compact: just point + CIs per comparison)
    json_rows = [
        {
            "experiment": r["experiment"],
            "session": r["session"],
            "comparison": r["comparison"],
            **r["stats"],
        }
        for r in all_rows
    ]
    (args.out / "robustness_table.json").write_text(json.dumps(json_rows, indent=2, default=float))

    # Markdown table
    md = []
    md.append("# Frame-level bootstrap CIs — robustness table")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"n_iters={args.n_iters}, seed={args.seed}")
    md.append("")
    md.append("Bootstrap unit: **frames** (NOT (frame, timestep, K_noise) rows). Per-frame")
    md.append("mean MSE computed before resampling. Stratified by `block_idx` when available.")
    md.append("")
    md.append("Columns: AUROC [95% CI] | paired win rate [95%] | mean Δ [95%] | 5th-pct Δ |")
    md.append("min Δ | inversions | n_frames | all_frames_agree | n_blocks")
    md.append("")
    md.append("| experiment | session | comparison | AUROC | win rate | mean Δ | p5 Δ | min Δ | inversions | n | agree | blocks |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        s = r["stats"]
        agree = "yes" if s["all_frames_agree"] else "no"
        md.append(f"| {r['experiment']} | {r['session']} | {r['comparison']} | "
                  f"{fmt_ci(s['auroc'])} | "
                  f"{fmt_ci(s['win_rate'], '{:.3f}')} | "
                  f"{fmt_ci(s['mean_delta'], '{:.5f}')} | "
                  f"{s['p5_delta']['point']:.5f} | "
                  f"{s['min_delta']['point']:.5f} | "
                  f"{s['inversions']['point']} | "
                  f"{s['n_frames']} | {agree} | {s['n_blocks']} |")
    (args.out / "robustness_table.md").write_text("\n".join(md))
    print(f"[bootstrap_cis] wrote {args.out / 'robustness_table.md'}")

    # Methodology doc
    method_md = [
        "# Bootstrap methodology — frame-level resampling",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## Resampling unit",
        "",
        "Each experiment NPZ stores per-condition arrays of shape (n_frames, n_timesteps, n_K_noise).",
        "We aggregate per-frame first via `mean over (timesteps, K_noise)` to get a single scalar score per",
        "(frame, condition). Bootstrap then resamples *frames* with replacement; rows within a frame are NOT",
        "treated as independent units.",
        "",
        "When `block_idx` is available (e.g., Phase G held-out blocks), resampling is stratified per block",
        "to preserve block-balance.",
        "",
        "## Metrics per resample",
        "",
        "- **AUROC**: positive class = `cond_correct` (target E), negative class = the wrong/perturbed cond.",
        "  Convention: lower MSE → more compatible; AUROC computed with `roc_auc_score(labels, -scores)`.",
        "  AUROC = 1.0 means target is reliably scored lower MSE than wrong on every comparison.",
        "- **Paired win rate**: per-frame fraction where target_score < wrong_score. Insensitive to magnitude.",
        "- **mean Δ** = mean(wrong − target). Positive → target wins on average.",
        "- **5th-percentile Δ** = worst-case 5% of frames; conservative readability for paper claims.",
        "- **min Δ** = absolute worst frame; sensitive to outliers but informative.",
        "- **Inversions** = number of frames where target ≥ wrong (ordering breaks).",
        "",
        "## Special handling for AUROC=1.0",
        "",
        "When `all_frames_agree=yes`, the bootstrap CI on AUROC is degenerate [1.0, 1.0]. The paired",
        "win rate, mean/p5/min Δ, and inversions provide additional discrimination — we do NOT report",
        "AUROC alone for these cases.",
        "",
        "## Notes",
        "",
        f"- 1000 bootstrap iterations per comparison (n_iters={args.n_iters}).",
        f"- Random seed for reproducibility: {args.seed}.",
        "- bf16 numerical floor: scores below ~1e-3 are at the precision floor; mean Δ near zero",
        "  on shuffled-control may reflect this rather than evidence of marginal signal.",
        "",
    ]
    (args.out / "bootstrap_methodology.md").write_text("\n".join(method_md))

    # Save per-comparison resamples (compact NPZ for downstream scripts)
    np.savez_compressed(args.out / "raw_resamples.npz",
                        rows_json=np.array(json.dumps(json_rows, default=float)))
    print(f"[bootstrap_cis] DONE in {time.time()-t0:.0f}s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
