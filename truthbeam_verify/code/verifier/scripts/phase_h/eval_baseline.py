"""Phase H — supervised baseline evaluation.

Loads a Phase H baseline checkpoint and computes AUROC on:
  - Real (positive) vs shuffled (negative class A) — matches Phase G shuffled
    control.
  - Real vs each in-training perturbation condition (in-family supervised
    sensitivity).
  - Real vs each held-out perturbation condition (generalization sensitivity).

Per spec: 100 frames per session × all 51 condition labels (35 perturbation +
correct + shuffled offsets + zero).

This eval uses the held-out FRAMES (Phase G eval blocks), not the training
rows used by train_baseline.py.

Usage:
  python scripts/phase_h/eval_baseline.py \\
      --ckpt <path>/model_final.pt \\
      --d2-dir <path>/data/d2 --v10-dir <path>/data/v10 \\
      --out <path>/eval [--n-frames 100]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_g"))

from phase_h.baseline_model import PhaseHBaseline  # noqa: E402
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    EVAL_BLOCKS, _crop_and_resize_C, _load_packed_cfa_float01,
)
from phase_g.xof_perturb import (  # noqa: E402
    render_E_for_phase_h, load_chain_log,
    TRAIN_POOL_LABELS, HELDOUT_POOL_LABELS, _spec_by_label,
)
from eval_diffusion_diagnostic import block_bootstrap_ci  # noqa: E402


def auroc_from_logits(pos_logits: np.ndarray,
                      neg_logits: np.ndarray) -> float:
    """AUROC where higher logit = more positive class. Tie-aware Mann-Whitney
    via average ranks. NOT the Phase G `auroc_pooled` — that one negates
    scores (designed for MSE where LOWER means more positive). Using
    auroc_pooled here would silently invert AUROC (~0 instead of ~1)."""
    pos = pos_logits.flatten()
    neg = neg_logits.flatten()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    n1, n2 = pos.size, neg.size
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="stable")
    sorted_scores = all_scores[order]
    ranks_sorted = np.empty(len(all_scores), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    R1 = ranks[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2
    return float(U / (n1 * n2))


def load_ckpt(ckpt_path: Path, device: torch.device,
              dtype: torch.dtype) -> PhaseHBaseline:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"]
    m = PhaseHBaseline(in_channels=7).to(device, dtype=dtype)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def score_batch(model: PhaseHBaseline, x: torch.Tensor,
                device: torch.device, dtype: torch.dtype) -> np.ndarray:
    """Return per-sample logits (NumPy)."""
    x = x.to(device, dtype=dtype)
    with torch.amp.autocast("cuda", dtype=dtype):
        logits = model(x)
    return logits.float().cpu().numpy()


def evaluate_session(model: PhaseHBaseline, session: str, session_dir: Path,
                     chain_log: dict[int, str],
                     other_chain: dict[int, str] | None,
                     other_session_dir: Path | None,
                     other_session: str | None,
                     n_frames: int, bs: int,
                     device: torch.device, dtype: torch.dtype,
                     seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    blocks = EVAL_BLOCKS[session]
    per_block = max(1, n_frames // len(blocks))
    sampled = []
    for a, b in blocks:
        valid = list(range(a + 30, b - 30))
        valid = [r for r in valid if r in chain_log]
        chosen = (rng.choice(valid, size=min(per_block, len(valid)), replace=False)
                  .tolist() if len(valid) > 0 else [])
        sampled.extend(sorted(chosen))
    print(f"[eval-h] session={session} n={len(sampled)}", flush=True)

    # Conditions: identity (positive), shuffled (negative class A),
    # plus all 35 perturbation labels.
    all_conditions = ["identity", "shuffled"] + TRAIN_POOL_LABELS + HELDOUT_POOL_LABELS
    # NOTE: shuffled is built per-frame using a deterministic partner

    # Per-frame logit storage
    n = len(sampled)
    logits_by_cond: dict[str, np.ndarray] = {
        cond: np.full(n, np.nan, dtype=np.float64) for cond in all_conditions
    }

    # Score each frame under each condition (sub-batched by bs)
    for i_start in range(0, n, bs):
        batch_rows = sampled[i_start:i_start + bs]
        # Load C for this batch once
        C_batch = torch.stack([
            _crop_and_resize_C(_load_packed_cfa_float01(
                session_dir / "Recordings" / f"frame_{r:06d}.raw"))
            for r in batch_rows
        ])

        for cond in all_conditions:
            # Build E batch under this condition
            E_list = []
            for r in batch_rows:
                if cond == "identity":
                    E = render_E_for_phase_h(session, r, "identity", chain_log,
                                             device=str(device))
                elif cond == "shuffled":
                    # Cross-session shuffled if available; else within-session offset by N/2
                    if other_chain is not None and other_session_dir is not None:
                        # Pick a partner from other session, deterministic by row
                        psess = other_session
                        prow = sorted(other_chain.keys())[r % len(other_chain)]
                        E = render_E_for_phase_h(psess, prow, "identity", other_chain,
                                                 device=str(device))
                    else:
                        # within-session, offset by half session
                        keys = sorted(chain_log.keys())
                        prow = keys[(keys.index(r) + len(keys) // 2) % len(keys)]
                        E = render_E_for_phase_h(session, prow, "identity", chain_log,
                                                 device=str(device))
                else:
                    spec = _spec_by_label(cond)
                    if spec.needs_donor():
                        # Pick same-session donor ~N/4 away
                        keys = sorted(chain_log.keys())
                        idx = keys.index(r)
                        donor_row = keys[(idx + len(keys) // 4) % len(keys)]
                        E = render_E_for_phase_h(session, r, cond, chain_log,
                                                 device=str(device),
                                                 donor_chain_log=chain_log,
                                                 donor_frame_id=donor_row)
                    else:
                        E = render_E_for_phase_h(session, r, cond, chain_log,
                                                 device=str(device))
                E_list.append(E)
            E_batch = torch.stack(E_list)
            x = torch.cat([C_batch, E_batch], dim=1)  # (B, 7, H, W)
            logits = score_batch(model, x, device, dtype)
            logits_by_cond[cond][i_start:i_start + len(batch_rows)] = logits
        if (i_start // bs) % 5 == 0:
            print(f"  [eval-h {session}] {i_start + len(batch_rows)}/{n}", flush=True)

    # Compute AUROCs
    pos_logits = logits_by_cond["identity"]
    metrics: dict[str, dict] = {}
    block_lengths = [len(sampled)]  # single-block aggregation for simplicity
    for cond in all_conditions:
        if cond == "identity":
            continue
        # AUROC: positive = identity logits (real), negative = condition logits.
        au = auroc_from_logits(pos_logits, logits_by_cond[cond])
        # Pooled mean logit
        pos_mean = float(np.mean(pos_logits))
        neg_mean = float(np.mean(logits_by_cond[cond]))
        metrics[cond] = {
            "auroc": float(au),
            "pos_mean_logit": pos_mean,
            "neg_mean_logit": neg_mean,
            "delta": pos_mean - neg_mean,
        }

    return {
        "session": session,
        "n_frames": n,
        "rows": sampled,
        "metrics": metrics,
        "raw_logits": {k: v.tolist() for k, v in logits_by_cond.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    print(f"[init] device={device} dtype={dtype} ckpt={args.ckpt}")

    model = load_ckpt(args.ckpt, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params/1e6:.2f}M")

    chain_d2 = load_chain_log(args.d2_dir)
    chain_v10 = load_chain_log(args.v10_dir) if args.v10_dir else None

    out: dict = {"sessions": {}, "ckpt": str(args.ckpt)}
    if chain_d2:
        d2_eval = evaluate_session(
            model, "D2", args.d2_dir, chain_d2,
            other_chain=chain_v10, other_session_dir=args.v10_dir,
            other_session="V10",
            n_frames=args.n_frames, bs=args.bs,
            device=device, dtype=dtype, seed=args.seed,
        )
        out["sessions"]["D2"] = d2_eval
    if chain_v10:
        v10_eval = evaluate_session(
            model, "V10", args.v10_dir, chain_v10,
            other_chain=chain_d2, other_session_dir=args.d2_dir,
            other_session="D2",
            n_frames=args.n_frames, bs=args.bs,
            device=device, dtype=dtype, seed=args.seed + 1,
        )
        out["sessions"]["V10"] = v10_eval

    (args.out / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}/summary.json")
    # Print a quick headline
    for sess in out["sessions"]:
        print(f"  {sess}:")
        m = out["sessions"][sess]["metrics"]
        for cond in ("shuffled", "xof_t1_global_k64", "xof_t3_region_k64",
                     "xof_t6_replace_general"):
            if cond in m:
                print(f"    {cond:30s} AUROC={m[cond]['auroc']:.3f}")


if __name__ == "__main__":
    main()
