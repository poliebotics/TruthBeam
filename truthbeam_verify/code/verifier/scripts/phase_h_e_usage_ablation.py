"""Sub-diag 6.4 — Phase H E-usage ablation. CRITICAL.

Per the project's 2026-05-03 directive: determine whether Phase H step_25000 actually
*uses* E or is C-only. Compare classifier scores on (C, E_target) vs (C, E_zero)
vs (C, E_shuffled) vs (C, E_source).

Pass criterion: target/zero/shuffled/source scores are distinguishable.
Fail criterion: scores ≈ identical → Phase H is C-only → SURFACE IMMEDIATELY,
operator wants to halt fresh binder phase momentarily for a discussion.

Methodology:
    - Load Phase H step_25000.pt (PhaseHBaseline, in_channels=7).
    - Sample 100 D2 + 100 V10 held-out frames (Phase G EVAL_BLOCKS, inset 30).
    - For each frame, compute classifier logit under 4 E variants:
        E_target:   render_E_for_phase_h(session, row, "identity", chain)
        E_zero:     torch.zeros((3, 768, 1024))
        E_shuffled: render_E_for_phase_h(other_session, partner_row, "identity")
        E_source:   render_E_for_phase_h(session, row - 2, "identity", chain)
                    (where source = target - 2 per F-A v1 SOURCE_LAG convention)
    - Compute AUROC of (target vs zero), (target vs shuffled), (target vs source).
    - Report per-frame logit deltas + score distributions.

Output:
    experiments/phase_h_e_usage_ablation/
        e_usage_report.md
        per_frame_logits.npz
        verdict.json (with overall_pass + verdict text for operator)

Single GPU (cuda:7 default); ~10-20 min.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_h.baseline_dataset import (  # noqa: E402
    _load_packed_cfa_float01, _crop_and_resize_C,
)
from phase_g.xof_perturb import (  # noqa: E402
    load_chain_log, render_E_for_phase_h, verify_identity_render_parity,
)
from phase_g.diffusion_diagnostic_dataset import EVAL_BLOCKS  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "phase_h"))
from train_baseline import PhaseHBaseline  # noqa: E402


DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/experiments/phase_h_e_usage_ablation")
DEFAULT_CKPT = Path(
    "/path/to/poliebotics_phase_b/experiments/phase_h_supervised_baseline/checkpoints/step_00025000.pt")
SESSION_DIRS = {
    "D2":  Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"),
}

# Strict path-mode resolution (Lambda or local — never half-mounted)
_PROBE = [SESSION_DIRS["D2"], SESSION_DIRS["V10"], DEFAULT_CKPT]
_AVAIL = sum(int(p.exists()) for p in _PROBE)
if _AVAIL == len(_PROBE):
    pass
elif _AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    SESSION_DIRS = {
        "D2":  LOCAL_ROOT / "data" / "d2",
        "V10": LOCAL_ROOT / "data" / "v10",
    }
    DEFAULT_CKPT = (LOCAL_ROOT / "experiments" / "phase_h_supervised_baseline"
                    / "checkpoints" / "step_00025000.pt")
    DEFAULT_OUT = LOCAL_ROOT / "experiments" / "phase_h_e_usage_ablation"
else:
    raise SystemExit(f"[6.4] mixed root state: {_AVAIL}/{len(_PROBE)}")


SOURCE_LAG = 2


def auroc_pos_higher(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC where positive class has higher logit. Convention: target_logit
    treated as 'positive' (model thinks this is a real coupling), other condition
    is 'negative'. AUROC = 1.0 means target's logit is reliably higher than
    the other condition's logit."""
    from sklearn.metrics import roc_auc_score
    n = min(len(pos), len(neg))
    pos = pos[:n]; neg = neg[:n]
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        return float("nan")
    try:
        return float(roc_auc_score(
            np.concatenate([np.ones(n), np.zeros(n)]),
            np.concatenate([pos, neg])))
    except ValueError:
        return float("nan")


def sample_frames(session: str, n_target: int, seed: int = 0) -> list[int]:
    sd = SESSION_DIRS[session]
    chain = load_chain_log(sd)
    rs = np.random.RandomState(seed)
    rows = []
    for a, b in EVAL_BLOCKS[session]:
        valid = []
        for r in range(a + 30, b - 30):
            if r in chain and (r - SOURCE_LAG) in chain:
                if (sd / "Recordings" / f"frame_{r:06d}.raw").exists():
                    valid.append(r)
        rows.extend(valid)
    if len(rows) > n_target:
        rows = sorted(rs.choice(rows, size=n_target, replace=False).tolist())
    return rows


def pick_shuffled_partner(session: str, row: int, other_chain: dict,
                           rs: np.random.RandomState) -> int:
    """Pick partner row from OTHER session (cross-session shuffled) for stronger
    distinguishability test."""
    keys = sorted(other_chain.keys())
    return int(rs.choice(keys))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-frames", type=int, default=100,
                    help="frames per session")
    ap.add_argument("--device", type=str, default="cuda:7")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    print(f"[6.4] device={device} dtype={dtype} ckpt={args.ckpt}")

    # Identity-render parity gate
    print("[6.4] identity-render parity gate...")
    chains = {}
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        chains[sess] = chain
        if not chain: continue
        samples = [f for f in (100, 1500, 3000) if f in chain][:3] or [sorted(chain.keys())[len(chain) // 2]]
        ok, details = verify_identity_render_parity(sd, chain, samples, tol=1e-6)
        print(f"  {sess}: max={details['max_overall']:.2e}  ok={ok}")
        if not ok: return 1

    # Load model + validate ckpt step matches expected
    print(f"[6.4] loading Phase H ckpt...")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # validate ckpt identity to prevent silent
    # wrong-ckpt verdict.
    ckpt_step = int(ck.get("step", -1))
    print(f"  ckpt step={ckpt_step}, max_steps_arg={ck.get('args', {}).get('max_steps', '?')}")
    if "step_00025000" in args.ckpt.name and ckpt_step != 25000:
        raise SystemExit(
            f"[6.4] ckpt filename suggests step_25000 but ck['step']={ckpt_step}; "
            "refusing to score the wrong checkpoint.")
    model = PhaseHBaseline(in_channels=7).to(device, dtype=dtype)
    model.load_state_dict(ck["model"])
    model.eval()

    rs_partner = np.random.RandomState(args.seed + 1)

    # Sample frames per session
    sampled = {}
    for sess in ("D2", "V10"):
        sampled[sess] = sample_frames(sess, args.n_frames, seed=args.seed)
        print(f"  {sess}: {len(sampled[sess])} frames sampled")

    # Score every (frame, E_condition)
    conditions = ["target", "zero", "shuffled", "source"]
    logits: dict[str, dict[str, list[float]]] = {sess: {c: [] for c in conditions}
                                                   for sess in ("D2", "V10")}
    rows_per_sess: dict[str, list[int]] = {sess: [] for sess in ("D2", "V10")}

    overall_t0 = time.time()
    for sess in ("D2", "V10"):
        sd = SESSION_DIRS[sess]
        other_sess = "V10" if sess == "D2" else "D2"
        other_chain = chains[other_sess]
        chain = chains[sess]
        print(f"[6.4] scoring {sess} ({len(sampled[sess])} frames × 4 E variants)...")
        for fi, row in enumerate(sampled[sess]):
            # Load C
            C = _crop_and_resize_C(_load_packed_cfa_float01(
                sd / "Recordings" / f"frame_{row:06d}.raw"))
            C_b = C.unsqueeze(0).to(device, dtype=dtype)

            # E variants — all (3, 768, 1024)
            E_target = render_E_for_phase_h(sess, row, "identity", chain).unsqueeze(0).to(device, dtype=dtype)
            E_zero = torch.zeros_like(E_target)
            partner = pick_shuffled_partner(sess, row, other_chain, rs_partner)
            E_shuffled = render_E_for_phase_h(other_sess, partner, "identity",
                                                other_chain).unsqueeze(0).to(device, dtype=dtype)
            E_source = render_E_for_phase_h(sess, row - SOURCE_LAG, "identity",
                                             chain).unsqueeze(0).to(device, dtype=dtype)

            for cond_name, E in (("target", E_target), ("zero", E_zero),
                                  ("shuffled", E_shuffled), ("source", E_source)):
                x = torch.cat([C_b, E], dim=1)  # (1, 7, H, W)
                with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                         dtype=dtype, enabled=(dtype != torch.float32)):
                    logit = model(x).float().cpu().item()
                logits[sess][cond_name].append(logit)
            rows_per_sess[sess].append(row)
            if (fi + 1) % 20 == 0:
                elapsed = time.time() - overall_t0
                print(f"  {sess} {fi+1}/{len(sampled[sess])}  elapsed={elapsed:.0f}s",
                      flush=True)
    print(f"[6.4] all scoring done in {time.time()-overall_t0:.0f}s")

    # Compute headline metrics
    metrics = {}
    for sess in ("D2", "V10"):
        L = {c: np.array(logits[sess][c], dtype=np.float64) for c in conditions}
        metrics[sess] = {
            "n_frames": len(L["target"]),
            "mean_logit": {c: float(np.mean(L[c])) for c in conditions},
            "std_logit":  {c: float(np.std(L[c]))  for c in conditions},
            "auroc_target_vs_zero":     auroc_pos_higher(L["target"], L["zero"]),
            "auroc_target_vs_shuffled": auroc_pos_higher(L["target"], L["shuffled"]),
            "auroc_target_vs_source":   auroc_pos_higher(L["target"], L["source"]),
            "delta_target_minus_zero":     float(np.mean(L["target"] - L["zero"])),
            "delta_target_minus_shuffled": float(np.mean(L["target"] - L["shuffled"])),
            "delta_target_minus_source":   float(np.mean(L["target"] - L["source"])),
            "frac_target_gt_zero":     float(np.mean(L["target"] > L["zero"])),
            "frac_target_gt_shuffled": float(np.mean(L["target"] > L["shuffled"])),
            "frac_target_gt_source":   float(np.mean(L["target"] > L["source"])),
        }

    # Verdict thresholds.
    # target-vs-zero is an OOD probe (Phase H never
    # saw zero-E during training, so its response to E_zero may be arbitrary).
    # PASS must require IN-DISTRIBUTION contrasts (target vs shuffled OR target vs source).
    # target-vs-zero is reported as a supplemental diagnostic, not load-bearing for PASS.
    THRESH_PASS = 0.70
    THRESH_C_ONLY = 0.55
    pass_target_vs_shuffled = (metrics["D2"]["auroc_target_vs_shuffled"] >= THRESH_PASS and
                                metrics["V10"]["auroc_target_vs_shuffled"] >= THRESH_PASS)
    pass_target_vs_source = (metrics["D2"]["auroc_target_vs_source"] >= THRESH_PASS and
                              metrics["V10"]["auroc_target_vs_source"] >= THRESH_PASS)
    pass_target_vs_zero = (metrics["D2"]["auroc_target_vs_zero"] >= THRESH_PASS and
                            metrics["V10"]["auroc_target_vs_zero"] >= THRESH_PASS)

    # PASS requires at least one IN-DISTRIBUTION comparison (shuffled or source) to clear.
    overall_pass = pass_target_vs_shuffled or pass_target_vs_source

    # C-only: BOTH in-distribution comparisons fail to distinguish in BOTH sessions.
    c_only = (metrics["D2"]["auroc_target_vs_shuffled"] <= THRESH_C_ONLY and
              metrics["V10"]["auroc_target_vs_shuffled"] <= THRESH_C_ONLY and
              metrics["D2"]["auroc_target_vs_source"] <= THRESH_C_ONLY and
              metrics["V10"]["auroc_target_vs_source"] <= THRESH_C_ONLY)

    if c_only:
        verdict_text = ("PHASE H IS C-ONLY (FAIL). target/zero/shuffled/source logits are "
                        "indistinguishable → Phase H ignores E entirely; its AUROC=1.0 on shuffled-pair "
                        "must be coming from C-side features alone (somehow detecting E presence/absence "
                        "in C? or a confound). Major manuscript-framing finding — operator wants to halt "
                        "fresh binder phase before further compute.")
    elif overall_pass:
        verdict_text = ("PHASE H USES E (PASS). target/(zero or shuffled) is distinguishable at AUROC ≥ "
                        f"{THRESH_PASS:.2f} in both sessions. Phase H actually consumes the E channels "
                        "for its decisions. Continue queue uninterrupted.")
    else:
        verdict_text = ("PHASE H E-USAGE INTERMEDIATE. Logits are weakly distinguishable but no comparison "
                        f"clears AUROC ≥ {THRESH_PASS:.2f} in both sessions. Surface to operator for "
                        "judgment — could be partial E-usage, or confound. NOT halting fresh binder "
                        "phase by default but flagging for review.")

    verdict = {
        "overall_pass": overall_pass,
        "c_only": c_only,
        "verdict_text": verdict_text,
        "thresholds_used": {"pass": THRESH_PASS, "c_only": THRESH_C_ONLY},
        "metrics": metrics,
        "ckpt": str(args.ckpt),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    # Save raw logits
    np.savez_compressed(args.out / "per_frame_logits.npz",
                        d2_rows=np.array(rows_per_sess["D2"]),
                        v10_rows=np.array(rows_per_sess["V10"]),
                        d2_target=np.array(logits["D2"]["target"]),
                        d2_zero=np.array(logits["D2"]["zero"]),
                        d2_shuffled=np.array(logits["D2"]["shuffled"]),
                        d2_source=np.array(logits["D2"]["source"]),
                        v10_target=np.array(logits["V10"]["target"]),
                        v10_zero=np.array(logits["V10"]["zero"]),
                        v10_shuffled=np.array(logits["V10"]["shuffled"]),
                        v10_source=np.array(logits["V10"]["source"]))

    # Markdown report
    md = []
    md.append("# Phase H E-usage ablation — sub-diag 6.4")
    md.append("")
    md.append(f"Generated: {verdict['generated_utc']}")
    md.append(f"Ckpt: `{args.ckpt}`")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    md.append(f"**{'✅ PASS' if overall_pass else ('❌ FAIL (C-ONLY)' if c_only else '⚠️ INTERMEDIATE')}**")
    md.append("")
    md.append(verdict_text)
    md.append("")
    md.append("## Per-session metrics")
    md.append("")
    md.append("| session | mean target | mean zero | mean shuffled | mean source | AUROC(t v zero) | AUROC(t v shuf) | AUROC(t v src) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for sess in ("D2", "V10"):
        m = metrics[sess]
        md.append(f"| {sess} | {m['mean_logit']['target']:+.3f} | "
                  f"{m['mean_logit']['zero']:+.3f} | "
                  f"{m['mean_logit']['shuffled']:+.3f} | "
                  f"{m['mean_logit']['source']:+.3f} | "
                  f"{m['auroc_target_vs_zero']:.4f} | "
                  f"{m['auroc_target_vs_shuffled']:.4f} | "
                  f"{m['auroc_target_vs_source']:.4f} |")
    md.append("")
    md.append("## Per-frame deltas (target − comparison, mean)")
    md.append("")
    md.append("| session | Δ(target − zero) | Δ(target − shuffled) | Δ(target − source) | frac t > zero | frac t > shuf | frac t > src |")
    md.append("|---|---|---|---|---|---|---|")
    for sess in ("D2", "V10"):
        m = metrics[sess]
        md.append(f"| {sess} | {m['delta_target_minus_zero']:+.4f} | "
                  f"{m['delta_target_minus_shuffled']:+.4f} | "
                  f"{m['delta_target_minus_source']:+.4f} | "
                  f"{m['frac_target_gt_zero']:.3f} | "
                  f"{m['frac_target_gt_shuffled']:.3f} | "
                  f"{m['frac_target_gt_source']:.3f} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- **AUROC(target vs zero) high**: Phase H discriminates `(C, E_target)` from `(C, E_zero)`")
    md.append("  → Phase H actually uses E. Without E, classifier confidence drops.")
    md.append("- **AUROC(target vs zero) ≈ 0.5**: Phase H ignores E entirely. C-only verifier.")
    md.append("  Means Phase H's perfect AUROC on shuffled-pair training came from C-side features.")
    md.append("- **AUROC(target vs source) interesting if ~ AUROC(target vs zero)**: Phase H either")
    md.append("  uses E broadly (no specific frame-pair binding), or treats source-E as noise. Hard")
    md.append("  to distinguish without further probes.")
    md.append("")
    (args.out / "e_usage_report.md").write_text("\n".join(md))
    print(f"[6.4] report → {args.out / 'e_usage_report.md'}")
    print(f"[6.4] VERDICT: {verdict_text[:80]}")
    # exit code MUST reflect c_only so wrapper
    # scripts can halt fresh binder phase. PASS: rc=0; c_only: rc=2 (distinct
    # from generic failure rc=1); intermediate: rc=3.
    if c_only:
        return 2
    if overall_pass:
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
