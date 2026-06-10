"""Condition-level comparison matrix — Phase G vs Phase H, normalized by ΔE.

Per CGPT round 3 directive (operator 2026-05-03): replace competitive "Phase G
vs Phase H" framing with condition-by-condition comparison normalized by Δ
rendered E. Both detectors may share blind spots — the matrix surfaces this
honestly rather than scoring one against the other.

Outputs (one row per condition × session):
    condition       — short label
    type            — baseline / wrong / lag / xof_t1..t6
    delta_rendered_E — mean L2 between E_correct and E_perturbed (in [0,1] pixel-space)
    phase_g_auroc   — AUROC of (cond_correct vs cond_X) on Phase G diffusion-verifier scores
    phase_g_delta_mse — mean (per-frame mean MSE for cond_X) - mean (per-frame MSE for cond_correct)
    phase_h_auroc   — same condition, Phase H supervised classifier (NaN until step_25000 eval lands)
    in_phase_h_train_pool — TRUE if condition is among Phase H training negatives (Type 1, 2, 4, 6 general)
    n_frames_d2 / n_frames_v10 — counts

Three categories are derived at the end of the report:
    "detected by both"        — high AUROC in both
    "diffusion stronger"       — Phase G high, Phase H low
    "missed by both"           — Phase G low AND Phase H low

Sorted by delta_rendered_E ascending so blind spots cluster at the top.

Run: python scripts/condition_comparison_matrix.py
Output: experiments/condition_comparison/condition_matrix.csv + .md report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# Local g1a paths
LOCAL_ROOT = Path("/path/to/poliebotics_phase_b")
SESSION_DIRS = {
    "D2":  LOCAL_ROOT / "data" / "d2",
    "V10": LOCAL_ROOT / "data" / "v10",
}
ITEM_1_EVAL = LOCAL_ROOT / "experiments" / "item_1" / "eval"
PHASE_H_EVAL = LOCAL_ROOT / "experiments" / "phase_h_supervised_baseline" / "eval"
DEFAULT_OUT = LOCAL_ROOT / "experiments" / "condition_comparison"

SESSIONS = ("d2", "v10")
SESSION_LABELS = {"d2": "D2", "v10": "V10"}


# Phase H training pool — per src/phase_g/xof_perturb.py TRAIN_POOL_LABELS
PHASE_H_TRAIN_POOL = (
    [f"xof_t1_global_k{k}" for k in (1, 4, 16, 64, 256, 1024, 4096)]
    + [f"xof_t2_oct{o}_k{k}" for o in (0, 1, 2, 3) for k in (1, 4, 16, 64)]
    + [f"xof_t4_swap_oct{o}" for o in (0, 1, 2, 3)]
    + ["xof_t6_replace_general"]
)


def cond_type(cond: str) -> str:
    """Coarse type label."""
    if cond == "correct": return "baseline"
    if cond.startswith("wrong_"): return "wrong-frame"
    if cond == "uncond": return "uncond"
    if cond.startswith("lag_"): return "lag"
    if cond.startswith("xof_t1"): return "xof_t1_bit_flip_global"
    if cond.startswith("xof_t2"): return "xof_t2_bit_flip_octave"
    if cond.startswith("xof_t3"): return "xof_t3_region"
    if cond.startswith("xof_t4"): return "xof_t4_octave_swap"
    if cond.startswith("xof_t5"): return "xof_t5_channel_swap"
    if cond.startswith("xof_t6"): return "xof_t6_replace"
    return "other"


def list_conditions(item_1_npz_d2: Path) -> list[str]:
    """Read condition list from D2 NPZ. Each cond_X key gives one condition X."""
    z = np.load(item_1_npz_d2, allow_pickle=False)
    out = [k[len("cond_"):] for k in z.files if k.startswith("cond_")]
    return sorted(out)


def phase_g_metrics_for_condition(z: np.lib.npyio.NpzFile, cond: str
                                   ) -> dict | None:
    """Compute Phase G AUROC + ΔMSE of (correct vs cond) on per-frame mean MSEs.

    Each cond_* array has shape (n_frames, 5, 4) — 5 timesteps × 4 K_noise.
    We average over (timestep, K) to get one scalar per frame, then AUROC of
    (correct < cond) — AUROC=1.0 means correct is reliably lower MSE than cond.
    """
    if cond == "correct":
        return None
    correct_key = "cond_correct"
    cond_key = f"cond_{cond}"
    if correct_key not in z.files or cond_key not in z.files:
        return None
    correct_pf = z[correct_key].mean(axis=(1, 2))
    cond_pf    = z[cond_key].mean(axis=(1, 2))
    # Equal lengths required
    n = min(len(correct_pf), len(cond_pf))
    correct_pf = correct_pf[:n]
    cond_pf = cond_pf[:n]
    scores = np.concatenate([correct_pf, cond_pf])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    if not np.isfinite(scores).all():
        return None
    try:
        # Convention: positive = correct (lower MSE). For AUROC where higher
        # score means positive, score = -MSE.
        auroc = float(roc_auc_score(labels, -scores))
    except ValueError:
        return None
    return {
        "auroc": auroc,
        "delta_mse_mean": float(cond_pf.mean() - correct_pf.mean()),
        "correct_mean_mse": float(correct_pf.mean()),
        "cond_mean_mse": float(cond_pf.mean()),
        "n_frames": int(n),
    }


def render_E_for_condition(session: str, target_row: int, cond: str,
                            chain_log: dict[int, str]) -> torch.Tensor:
    """Render the E tensor for a given condition at (session, target_row).

    Returns (3, 1080, 1920) float32 in [0,1]. Uses the bit-exact render path
    via xof_perturb.render_E_for_phase_h for XOF perturbations + identity, and
    direct file load for wrong/lag conditions.

    Conditions handled:
        correct                            — render identity from chain[target_row]
        wrong_+N / wrong_-N (N != 0)       — render identity from chain[target_row + N]
        lag_+N / lag_-N (N != 0)           — render identity from chain[target_row + N]
                                             (wrong vs lag: different sets, same mechanic)
        uncond                             — N/A (zero E used; we return None)
        xof_t* labels                      — render via xof_perturb.render_E_for_phase_h
    """
    from phase_g.xof_perturb import render_E_for_phase_h, expand_streams_from_s_t, render_streams_to_tile

    sd = SESSION_DIRS[SESSION_LABELS[session]]

    if cond == "correct":
        return render_E_for_phase_h(SESSION_LABELS[session], target_row, "identity",
                                     chain_log, device="cpu")
    if cond == "uncond":
        return None  # zero E, ΔE undefined here
    if cond.startswith("wrong_") or cond.startswith("lag_"):
        offset_str = cond.split("_", 1)[1]   # "+N" or "-N"
        try:
            offset = int(offset_str)
        except ValueError:
            return None
        partner_row = target_row + offset
        if partner_row not in chain_log:
            return None
        return render_E_for_phase_h(SESSION_LABELS[session], partner_row, "identity",
                                     chain_log, device="cpu")
    if cond.startswith("xof_t"):
        # Need donor for type 4/5/6
        from phase_g.xof_perturb import _spec_by_label
        spec = _spec_by_label(cond)
        donor_row = None; donor_chain = None
        if spec.needs_donor():
            # Type 4/5/6 use a far-frame donor in same session (chain length-aware)
            rows_sorted = sorted(chain_log.keys())
            n = len(rows_sorted)
            idx = rows_sorted.index(target_row)
            donor_row = rows_sorted[(idx + n // 2) % n]
            donor_chain = chain_log
        return render_E_for_phase_h(SESSION_LABELS[session], target_row, cond,
                                     chain_log,
                                     donor_chain_log=donor_chain,
                                     donor_frame_id=donor_row, device="cpu")
    return None


def compute_delta_rendered_E(session: str, conditions: list[str], chain_log: dict,
                              n_sample_frames: int = 8, seed: int = 0) -> dict[str, float]:
    """Per condition, compute mean L2 distance between E_correct and E_cond
    over a small sample of frames. Returns {cond: mean_L2}.

    Uses 8 sample frames per session (same set across conditions for fair
    comparison). Frames sampled deterministically from chain_log keys.
    """
    rs = np.random.RandomState(seed)
    rows = sorted(chain_log.keys())
    # Pick frames inside Phase G held-out blocks if possible (matches Item 1)
    # Insetting from min/max by 30 to keep wrong/lag offsets in-range:
    valid = [r for r in rows if (rows[0] + 60 <= r <= rows[-1] - 60)]
    n = min(n_sample_frames, len(valid))
    if n == 0:
        return {c: float("nan") for c in conditions}
    sampled = rs.choice(valid, size=n, replace=False)

    delta_cache: dict[str, list[float]] = {c: [] for c in conditions if c != "correct"}
    for target_row in sampled:
        try:
            e_correct = render_E_for_condition(session, int(target_row), "correct", chain_log)
        except Exception as ex:
            print(f"  [ΔE] {session} row {target_row} correct render failed: {ex}")
            continue
        if e_correct is None:
            continue
        for cond in conditions:
            if cond == "correct" or cond == "uncond": continue
            try:
                e_cond = render_E_for_condition(session, int(target_row), cond, chain_log)
            except Exception as ex:
                print(f"  [ΔE] {session} row {target_row} {cond} render failed: {ex}")
                continue
            if e_cond is None: continue
            l2 = float(((e_correct.float() - e_cond.float()) ** 2).mean().sqrt())
            delta_cache[cond].append(l2)
    out: dict[str, float] = {"correct": 0.0, "uncond": float("nan")}
    for cond, vals in delta_cache.items():
        out[cond] = float(np.mean(vals)) if vals else float("nan")
    return out


def phase_h_metrics_for_condition(eval_dir: Path, sess: str, cond: str
                                   ) -> dict | None:
    """Phase H AUROC for condition `cond` from
    eval/{step_X}/summary.json[sessions][SESS][metrics][cond]."""
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists(): return None
    s = json.loads(summary_path.read_text())
    sess_label = SESSION_LABELS[sess]
    body = s.get("sessions", {}).get(sess_label, {})
    metrics = body.get("metrics", {})
    if cond not in metrics: return None
    return {"auroc": float(metrics[cond]["auroc"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--phase-h-step", type=str, default="step_10000",
                    help="Which Phase H eval ckpt to read AUROCs from "
                    "(symlink 'final' or 'step_25000' once landed).")
    ap.add_argument("--n-delta-e-frames", type=int, default=8,
                    help="# frames per session to average ΔE over.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[matrix] Phase H source: {PHASE_H_EVAL / args.phase_h_step}")
    npz_paths = {sess: ITEM_1_EVAL / f"eval_{sess}_raw.npz" for sess in SESSIONS}
    for sess, p in npz_paths.items():
        if not p.exists():
            raise SystemExit(f"missing {p}")

    # Get conditions list (D2 should match V10 — verify)
    conds_d2 = list_conditions(npz_paths["d2"])
    conds_v10 = list_conditions(npz_paths["v10"])
    if conds_d2 != conds_v10:
        print(f"[matrix] WARN: D2 has {len(conds_d2)} conditions, V10 has "
              f"{len(conds_v10)}. Using union.")
    conditions = sorted(set(conds_d2) | set(conds_v10))
    print(f"[matrix] {len(conditions)} unique conditions across sessions")

    # ---- Compute Phase G metrics ----
    pg = {}
    for sess in SESSIONS:
        z = np.load(npz_paths[sess], allow_pickle=False)
        pg[sess] = {}
        for cond in conditions:
            m = phase_g_metrics_for_condition(z, cond)
            pg[sess][cond] = m

    # ---- Compute ΔE for each condition (per session) ----
    print(f"[matrix] computing ΔE over {args.n_delta_e_frames} sample frames per session...")
    from phase_g.xof_perturb import load_chain_log
    delta_e: dict[str, dict[str, float]] = {}
    for sess in SESSIONS:
        sd = SESSION_DIRS[SESSION_LABELS[sess]]
        chain = load_chain_log(sd)
        t0 = time.time()
        delta_e[sess] = compute_delta_rendered_E(sess, conditions, chain,
                                                  n_sample_frames=args.n_delta_e_frames)
        print(f"  {sess}: ΔE done in {time.time()-t0:.0f}s, "
              f"{sum(1 for v in delta_e[sess].values() if not np.isnan(v))} valid")

    # ---- Phase H metrics (if eval present) ----
    ph_eval_dir = PHASE_H_EVAL / args.phase_h_step
    ph = {sess: {} for sess in SESSIONS}
    if ph_eval_dir.exists():
        for sess in SESSIONS:
            for cond in conditions:
                m = phase_h_metrics_for_condition(ph_eval_dir, sess, cond)
                ph[sess][cond] = m
    else:
        print(f"[matrix] Phase H eval dir missing ({ph_eval_dir}); leaving Phase H AUROCs as NaN")

    # ---- Assemble rows ----
    rows = []
    for cond in conditions:
        for sess in SESSIONS:
            row = {
                "condition": cond,
                "session": SESSION_LABELS[sess],
                "type": cond_type(cond),
                "delta_rendered_E": delta_e[sess].get(cond, float("nan")),
                "phase_g_auroc": (pg[sess][cond] or {}).get("auroc", float("nan")),
                "phase_g_delta_mse": (pg[sess][cond] or {}).get("delta_mse_mean", float("nan")),
                "phase_g_n_frames": (pg[sess][cond] or {}).get("n_frames", 0),
                "phase_h_auroc": (ph[sess].get(cond) or {}).get("auroc", float("nan")),
                "in_phase_h_train_pool": cond in PHASE_H_TRAIN_POOL,
            }
            rows.append(row)

    # ---- Write CSV ----
    import csv
    csv_path = args.out / "condition_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[matrix] wrote {csv_path} ({len(rows)} rows)")

    # ---- Categorize ----
    HIGH = 0.90
    LOW = 0.65
    cats = {"detected_by_both": [], "diffusion_stronger": [],
            "missed_by_both": [], "phase_h_only": [], "uncategorized": []}
    for r in rows:
        if r["condition"] == "correct": continue
        pg_auc = r["phase_g_auroc"]
        ph_auc = r["phase_h_auroc"]
        if not np.isfinite(pg_auc): continue
        ph_finite = np.isfinite(ph_auc)
        if pg_auc >= HIGH and ph_finite and ph_auc >= HIGH:
            cats["detected_by_both"].append(r)
        elif pg_auc >= HIGH and ph_finite and ph_auc < LOW:
            cats["diffusion_stronger"].append(r)
        elif pg_auc < LOW and ph_finite and ph_auc < LOW:
            cats["missed_by_both"].append(r)
        elif pg_auc < LOW and ph_finite and ph_auc >= HIGH:
            cats["phase_h_only"].append(r)
        else:
            cats["uncategorized"].append(r)

    # ---- Markdown report ----
    rows_sorted = sorted(rows, key=lambda r: r["delta_rendered_E"]
                          if np.isfinite(r["delta_rendered_E"]) else 1e9)
    md = []
    md.append("# Condition-level comparison matrix — Phase G vs Phase H")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"Phase H source: `{ph_eval_dir.name}`")
    md.append(f"# conditions × sessions: {len(rows)}; ΔE sample frames per session: {args.n_delta_e_frames}")
    md.append("")
    md.append("**Methodology**: AUROC normalized by Δ rendered E (L2 distance between E_correct "
              "and E_perturbed in pixel space). Sorted by ΔE ascending so blind-spot conditions "
              "cluster at the top.")
    md.append("")
    md.append("Per CGPT round 3 directive: replace competitive Phase G vs Phase H framing with "
              "honest condition-level comparison. Both detectors may share blind spots.")
    md.append("")
    md.append("## Categorisation")
    md.append("")
    md.append("Thresholds: HIGH ≥ 0.90, LOW < 0.65, applied to AUROC of (correct vs cond).")
    md.append("")
    md.append("| category | count | examples (first 5) |")
    md.append("|---|---|---|")
    for cat, rows_in_cat in cats.items():
        examples = ", ".join(sorted(set(f"{r['condition']}@{r['session']}" for r in rows_in_cat))[:5])
        md.append(f"| {cat} | {len(rows_in_cat)} | {examples} |")
    md.append("")
    md.append("## Full table (sorted by Δ rendered E)")
    md.append("")
    md.append("| condition | session | type | ΔE | PG AUROC | PG ΔMSE | PH AUROC | in PH train pool |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows_sorted:
        de = r["delta_rendered_E"]
        de_s = f"{de:.4f}" if np.isfinite(de) else "—"
        pg_a = r["phase_g_auroc"]
        pg_a_s = f"{pg_a:.3f}" if np.isfinite(pg_a) else "—"
        pg_d = r["phase_g_delta_mse"]
        pg_d_s = f"{pg_d:.5f}" if np.isfinite(pg_d) else "—"
        ph_a = r["phase_h_auroc"]
        ph_a_s = f"{ph_a:.3f}" if np.isfinite(ph_a) else "pending"
        md.append(f"| {r['condition']} | {r['session']} | {r['type']} | "
                  f"{de_s} | {pg_a_s} | {pg_d_s} | {ph_a_s} | "
                  f"{'yes' if r['in_phase_h_train_pool'] else 'no'} |")
    md.append("")
    (args.out / "condition_matrix_report.md").write_text("\n".join(md))
    print(f"[matrix] wrote {args.out / 'condition_matrix_report.md'}")

    # JSON dump for downstream scripts
    (args.out / "condition_matrix.json").write_text(
        json.dumps({"rows": rows, "categorisation_summary": {k: len(v) for k,v in cats.items()},
                    "phase_h_step_used": args.phase_h_step,
                    "thresholds_used": {"HIGH": HIGH, "LOW": LOW}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
