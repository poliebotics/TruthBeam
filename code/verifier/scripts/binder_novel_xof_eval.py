"""Binder MSE eval on novel-XOF E values — pre-flight for F-A v2 viability.

Tests whether the existing 13 Phase E binders correctly discriminate novel-XOF
E values when paired with real C captures. Question: do binders give appropri-
ately HIGH MSE for novel E (correctly rejecting), or LOW MSE (failing to
discriminate)?

Per operator's preflight spec 2026-05-03:
    - 100 real C frames stratified across D2/V10 held-out blocks
    - 22 E conditions per frame: target, shuffled (lag > 60), 20 novel
    - 13 binders (Phase E exp001c family loaded from PHASE_E_THRESHOLDS.json)
    - Outputs: per-binder + aggregate AUROC for (target vs novel) and
      (target vs shuffled), distributional comparison, interpretation matrix

Forward amortization: each binder forward is on C only — it produces a single
(3, 1080, 1920) prediction, then we MSE against 22 E variants. So actual
forward count is 13 × 100 = 1,300 (not 28,600); MSE is cheap.

Sanity gate (HARD): runs first on 5 frames × 1 binder × 3 conds. Halts on:
    - m_target > m_shuffled (binder fundamentally broken)
    - NaN/Inf in any score
    - render parity failure on identity E

Outputs:
    seeds.json            — novel-S_t hex values + nonces (deterministic)
    raw_scores.npz        — frames × conditions × binders MSE matrix
    per_binder_summary.json — per-binder AUROC (target vs novel, target vs shuffled)
    aggregate_summary.json  — across-binder distribution stats + interpretation
    binder_novel_eval_report.md — human-readable report

Run:
    python scripts/binder_novel_xof_eval.py [--device cuda:7|cpu] [--bf16]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402
from phase_g.xof_perturb import (  # noqa: E402
    load_chain_log, expand_streams_from_s_t, render_streams_to_tile,
    verify_identity_render_parity,
)
from phase_g.diffusion_diagnostic_dataset import EVAL_BLOCKS  # noqa: E402

from models.emission_predictor import EmissionPredictor  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
except ImportError:
    EmissionPredictorV2 = None


# -------- defaults / paths --------

DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/experiments/binder_novel_eval")
DEFAULT_THRESHOLDS = Path(
    "/path/to/poliebotics_phase_b/experiments/phase_e/PHASE_E_THRESHOLDS.json")
SESSION_DIRS = {
    "D2":  Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"),
}

# Strict path-mode resolution (Lambda or local — never half-mounted).
_PROBE = [SESSION_DIRS["D2"], SESSION_DIRS["V10"], DEFAULT_THRESHOLDS]
_AVAIL = sum(int(p.exists()) for p in _PROBE)
if _AVAIL == len(_PROBE):
    pass  # Lambda mode
elif _AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    SESSION_DIRS = {
        "D2":  LOCAL_ROOT / "data" / "d2",
        "V10": LOCAL_ROOT / "data" / "v10",
    }
    DEFAULT_THRESHOLDS = (LOCAL_ROOT / "experiments" / "phase_e"
                          / "PHASE_E_THRESHOLDS.json")
    DEFAULT_OUT = LOCAL_ROOT / "experiments" / "binder_novel_eval"
else:
    raise SystemExit(
        f"[binder_novel_eval] mixed root state: {_AVAIL}/{len(_PROBE)} "
        "Lambda paths exist. Refusing to run.")


N_FRAMES_DEFAULT = 100      # total real C frames across both sessions
N_NOVEL_SEEDS    = 20        # number of novel chain seeds
LAG_MIN_SHUFFLED = 60        # same-session shuffled partner constraint
EMISSION_H_NATIVE, EMISSION_W_NATIVE = 1080, 1920


# -------- frame sampling --------

def sample_held_out_frames(n_total: int, seed: int = 0) -> list[tuple[str, int]]:
    """Stratified sample across D2 + V10 held-out blocks.

    Distribution: 50 D2 (split across 3 blocks ≈ 17 each) +
                  50 V10 (split across 2 blocks ≈ 25 each).
    Deterministic via numpy.random.RandomState(seed).
    Filters to rows present in chain_log AND with .raw on disk.

    NOTE (Codex audit MED 2026-05-04): this is a balanced 50/50 D2:V10 sample,
    NOT a representative sample of the true held-out distribution (held-out
    block sizes differ across sessions). The balance is deliberate for
    discrimination preflight — we want symmetric per-session signal so that
    failure modes specific to either session aren't masked by the other.
    """
    rs = np.random.RandomState(seed)
    n_d2 = n_total // 2
    n_v10 = n_total - n_d2
    out: list[tuple[str, int]] = []

    for sess, n_target in (("D2", n_d2), ("V10", n_v10)):
        sd = SESSION_DIRS[sess]
        chain = load_chain_log(sd)
        blocks = EVAL_BLOCKS[sess]
        # Per-block valid rows (in chain log + recordings dir).
        per_block: list[list[int]] = []
        for a, b in blocks:
            valid = []
            for r in range(a, b):
                if r in chain and (sd / "Recordings" / f"frame_{r:06d}.raw").exists():
                    valid.append(r)
            per_block.append(valid)
        # Distribute n_target evenly across blocks
        n_per = [n_target // len(blocks)] * len(blocks)
        for i in range(n_target % len(blocks)):
            n_per[i] += 1
        for bi, (n_b, valid) in enumerate(zip(n_per, per_block)):
            if not valid:
                print(f"[sample] {sess} block {bi}: no valid rows, skipping")
                continue
            picked = rs.choice(valid, size=min(n_b, len(valid)), replace=False)
            for r in sorted(picked.tolist()):
                out.append((sess, int(r)))
    return out


def pick_shuffled_partner(session: str, row: int, chain: dict, rs: np.random.RandomState,
                          session_dir: Path, lag_min: int = LAG_MIN_SHUFFLED) -> int:
    """Pick a partner row with |lag|>lag_min that has BOTH a chain entry and an
    emission tile PNG on disk. (Codex audit MED 2026-05-04: previous version
    only checked chain, so missing PNGs would surface as FileNotFoundError
    much later in the eval loop.)
    """
    em_dir = session_dir / "derived" / "Emissions"
    rows = sorted(chain.keys())
    for _ in range(128):
        cand = int(rs.choice(rows))
        if abs(cand - row) > lag_min and (em_dir / f"tile_{cand:06d}.png").exists():
            return cand
    # Fallback: deterministic walk
    for off in range(lag_min + 1, len(rows)):
        cand = (row + off) % max(rows)
        if (cand in chain and abs(cand - row) > lag_min
                and (em_dir / f"tile_{cand:06d}.png").exists()):
            return cand
    raise RuntimeError(f"no shuffled partner with PNG found for {session} row {row}")


# -------- novel-seed generation --------

def generate_novel_s_t_hex(n: int, seed_bytes: bytes | None = None) -> tuple[list[str], list[str]]:
    """Returns (s_t_hex_list, nonces_hex_list).

    Deterministic if seed_bytes provided. Otherwise os.urandom (no record-back).
    """
    if seed_bytes is None:
        nonces = [os.urandom(32) for _ in range(n)]
    else:
        from blake3 import blake3
        nonces = [blake3(seed_bytes + i.to_bytes(8, "big")).digest(length=32)
                  for i in range(n)]
    s_t_hex = [n_.hex() for n_ in nonces]   # nonces ARE the S_t-equivalent values
    nonces_hex = [n_.hex() for n_ in nonces]
    return s_t_hex, nonces_hex


def render_E_native(s_t_hex: str) -> torch.Tensor:
    """Render a 32-byte S_t-equivalent into (3, 1080, 1920) float32 [0,1] tensor."""
    streams = expand_streams_from_s_t(s_t_hex)
    tile = render_streams_to_tile(streams, device="cpu")  # (3, 1080, 1920) uint8
    return tile.float() / 255.0


# -------- binder loading --------

def load_binder(spec: dict, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    arch = spec["arch"]
    if arch == "EmissionPredictor":
        m = EmissionPredictor(emission_h=EMISSION_H_NATIVE, emission_w=EMISSION_W_NATIVE,
                              pretrained=False)
    elif arch == "EmissionPredictorV2":
        if EmissionPredictorV2 is None:
            raise RuntimeError("EmissionPredictorV2 import failed")
        m = EmissionPredictorV2(emission_h=EMISSION_H_NATIVE, emission_w=EMISSION_W_NATIVE,
                                pretrained=False)
    else:
        raise ValueError(f"unknown arch: {arch}")
    ckpt = Path(spec["ckpt"])
    if not ckpt.exists():
        raise RuntimeError(f"ckpt missing: {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    # Codex audit HIGH 2026-05-04: capture missing/unexpected so we can fail
    # loudly if a binder ckpt is incompatible with the model code (e.g. arch
    # drift). Empty sets = clean load. Anything non-empty is a hard error.
    missing, unexpected = m.load_state_dict(state, strict=False)
    if len(missing) or len(unexpected):
        raise RuntimeError(
            f"binder ckpt {ckpt.name} state_dict mismatch: "
            f"missing={list(missing)[:5]}{'...' if len(missing) > 5 else ''} "
            f"unexpected={list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


# -------- AUROC (tie-aware Mann-Whitney) --------

def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Returns P(pos < neg) when 'positive=correct=lower-MSE' convention.

    For (target vs novel/shuffled): positive class = target (low MSE expected).
                                    negative class = novel/shuffled (high MSE).
    AUROC ≈ 1 means binder cleanly separates target as lower MSE than negatives.
    """
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    n_pos, n_neg = pos.size, neg.size
    # Concatenate, rank, count pos<neg ties weighted by 0.5
    all_scores = np.concatenate([pos, neg])
    labels     = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(all_scores, kind="mergesort")
    # Standard ranks (average for ties)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = all_scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[labels == 1].sum()
    U = sum_pos - n_pos * (n_pos + 1) / 2.0
    # Convention: AUROC P(score_neg > score_pos). For "lower MSE = correct
    # match", we want P(pos < neg) = 1 - U/(n_pos n_neg), but conventional
    # auroc lib has P(pos > neg). Adjust based on lower-better:
    # auroc_lower_better = 1 - (U / (n_pos * n_neg))
    return 1.0 - (U / (n_pos * n_neg))


# -------- main --------

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--thresholds-json", type=Path, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES_DEFAULT)
    ap.add_argument("--n-novel", type=int, default=N_NOVEL_SEEDS)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for frame sampling + shuffled-partner picking.")
    ap.add_argument("--novel-seed-bytes", type=str, default=None,
                    help="Hex bytes for deterministic novel S_t generation. "
                    "Default: os.urandom (logged into seeds.json).")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda:N or cpu. Default: GPU 7 if available, else CPU.")
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 autocast for binder forward (faster on A100).")
    ap.add_argument("--sanity-only", action="store_true",
                    help="Run just the sanity check then exit (5 frames × 1 binder).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- pick device ----
    # CRITICAL: Phase H trains DDP=8 across ALL GPUs. Default to CPU and
    # require explicit `--device cuda:N` if the operator wants GPU. This
    # prevents accidental contention with running training. (Codex audit
    # CRITICAL finding 2026-05-04.)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    print(f"[binder_novel_eval] device={device} dtype={dtype}  "
          f"(set --device cuda:N for GPU; default is CPU to avoid Phase H DDP contention)")

    # ---- identity-render parity gate (HARD) ----
    print("[binder_novel_eval] identity-render parity gate...")
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        if not chain:
            print(f"  WARN: {sess} chain log empty; skipping")
            continue
        samples = [f for f in (100, 1500, 3000) if f in chain][:3]
        if not samples:
            samples = [sorted(chain.keys())[len(chain) // 2]]
        ok, details = verify_identity_render_parity(sd, chain, samples, tol=1e-6)
        print(f"  {sess}: max_overall={details['max_overall']:.2e}  ok={ok}")
        if not ok:
            print(f"  HARD GATE FAIL — aborting (details={details})")
            return 1
    print("[binder_novel_eval] parity OK")

    # ---- load binder registry ----
    th = json.loads(args.thresholds_json.read_text())
    binders_meta = th["binders"]
    binder_names = sorted(binders_meta.keys())
    print(f"[binder_novel_eval] {len(binder_names)} binders: {binder_names}")

    # ---- generate novel seeds ----
    seed_bytes = bytes.fromhex(args.novel_seed_bytes) if args.novel_seed_bytes else None
    novel_s_t_hex, novel_nonces_hex = generate_novel_s_t_hex(args.n_novel, seed_bytes)
    seeds_doc = {
        "n_novel": args.n_novel,
        "deterministic": seed_bytes is not None,
        "seed_bytes_hex": args.novel_seed_bytes,
        "novel_s_t_hex": novel_s_t_hex,
        "nonces_hex": novel_nonces_hex,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "seeds.json").write_text(json.dumps(seeds_doc, indent=2))
    print(f"[binder_novel_eval] generated {args.n_novel} novel S_t values; "
          f"first={novel_s_t_hex[0][:16]}…")

    # Sanity: no collisions with real chain.
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        existing = set(chain.values())
        cols = sum(1 for s in novel_s_t_hex if s in existing)
        if cols > 0:
            print(f"  HALT: {cols} novel seeds collide with {sess} chain")
            return 2

    # ---- pre-render the 22 E variants per frame (independent of binder) ----
    # Lifecycle: per-frame E variants are loaded/rendered once and reused
    # across all binders, since binders only consume C and predict E.
    print(f"[binder_novel_eval] sampling {args.n_frames} frames...")
    sampled = sample_held_out_frames(args.n_frames, seed=args.seed)
    n_actual = len(sampled)
    if n_actual < args.n_frames:
        print(f"  WARN: only {n_actual}/{args.n_frames} frames sampled "
              f"(some held-out blocks under-populated)")
    print(f"[binder_novel_eval] {n_actual} frames sampled "
          f"({sum(1 for s,_ in sampled if s == 'D2')} D2 / "
          f"{sum(1 for s,_ in sampled if s == 'V10')} V10)")

    print("[binder_novel_eval] rendering 20 novel E tensors (shared across all frames)...")
    novel_E = []
    t0 = time.time()
    for i, sth in enumerate(novel_s_t_hex):
        novel_E.append(render_E_native(sth))      # (3, 1080, 1920) f32 cpu
    print(f"  {len(novel_E)} novel E rendered in {time.time()-t0:.1f}s")

    # Per-frame target/shuffled E (lazy load once each)
    print("[binder_novel_eval] preparing per-frame target/shuffled E...")
    rs_partner = np.random.RandomState(args.seed + 1)
    chains_cache = {sess: load_chain_log(sd) for sess, sd in SESSION_DIRS.items()}
    frame_meta = []  # parallel to sampled
    for ri, (sess, row) in enumerate(sampled):
        sd = SESSION_DIRS[sess]
        chain = chains_cache[sess]
        partner_row = pick_shuffled_partner(sess, row, chain, rs_partner, sd)
        frame_meta.append({"session": sess, "row": row, "partner_row": partner_row})
    # Don't materialise all E targets in RAM (100 × 3 × 1080 × 1920 × 4 bytes ≈ 2.4 GB).
    # We'll load them per binder loop, but each E comes from disk only once via cache.

    # ---- sanity check (5 frames × 1 binder × 3 conditions) ----
    print("\n[binder_novel_eval] SANITY CHECK: 5 frames × 1 binder × 3 conditions...")
    sanity_binder_name = "exp001c" if "exp001c" in binders_meta else binder_names[0]
    sanity_spec = binders_meta[sanity_binder_name]["spec"]
    sanity_binder = load_binder(sanity_spec, device, dtype)
    n_sanity = min(5, n_actual)
    sanity_target = []
    sanity_shuffled = []
    sanity_novel = []
    for i in range(n_sanity):
        sess = frame_meta[i]["session"]
        row = frame_meta[i]["row"]
        partner_row = frame_meta[i]["partner_row"]
        sd = SESSION_DIRS[sess]
        cap = load_capture_at(sd / "Recordings" / f"frame_{row:06d}.raw",
                              sanity_spec["capture_h"], sanity_spec["capture_w"])
        cap = cap.unsqueeze(0).to(device, dtype=dtype)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                 dtype=dtype if dtype != torch.float32 else torch.float32,
                                 enabled=(dtype != torch.float32)):
            pred = sanity_binder(cap).float().clamp(0, 1).squeeze(0).cpu()
        E_target = load_emission_at(sd / "derived" / "Emissions" / f"tile_{row:06d}.png",
                                     EMISSION_H_NATIVE, EMISSION_W_NATIVE)
        E_shuffled = load_emission_at(
            sd / "derived" / "Emissions" / f"tile_{partner_row:06d}.png",
            EMISSION_H_NATIVE, EMISSION_W_NATIVE)
        E_novel = novel_E[0]   # use first novel seed for sanity
        m_t = float(((pred - E_target) ** 2).mean())
        m_s = float(((pred - E_shuffled) ** 2).mean())
        m_n = float(((pred - E_novel) ** 2).mean())
        if not all(np.isfinite([m_t, m_s, m_n])):
            print(f"  HALT: NaN/Inf in sanity scores for {sess} row {row} "
                  f"(t={m_t}, s={m_s}, n={m_n})")
            return 3
        sanity_target.append(m_t); sanity_shuffled.append(m_s); sanity_novel.append(m_n)
        print(f"  {sess} row {row}: m_target={m_t:.5f}  m_shuffled={m_s:.5f}  "
              f"m_novel={m_n:.5f}")

    mean_t = float(np.mean(sanity_target))
    mean_s = float(np.mean(sanity_shuffled))
    mean_n = float(np.mean(sanity_novel))
    print(f"[sanity] {sanity_binder_name}: "
          f"mean target={mean_t:.5f}  shuffled={mean_s:.5f}  novel={mean_n:.5f}")
    # Codex audit HIGH 2026-05-04: check per-frame proportion AND aggregate
    # mean, not just the mean. A binder that flips on individual frames but
    # averages OK should NOT pass. Require ≥80% per-frame correctness.
    n_target_lt_shuffled = sum(1 for t, s in zip(sanity_target, sanity_shuffled) if t < s)
    n_target_lt_novel    = sum(1 for t, n in zip(sanity_target, sanity_novel)    if t < n)
    frac_ts = n_target_lt_shuffled / len(sanity_target)
    frac_tn = n_target_lt_novel    / len(sanity_target)
    print(f"[sanity] per-frame: target<shuffled in {n_target_lt_shuffled}/{n_sanity} "
          f"({frac_ts:.1%}); target<novel in {n_target_lt_novel}/{n_sanity} ({frac_tn:.1%})")
    if mean_t >= mean_s:
        print(f"  HALT: mean m_target ({mean_t:.5f}) >= mean m_shuffled "
              f"({mean_s:.5f}) — binder fundamentally broken or scoring inverted")
        return 4
    if frac_ts < 0.8:
        print(f"  HALT: only {frac_ts:.0%} of sanity frames have target < shuffled "
              "(threshold 80%) — binder discrimination too noisy to trust")
        return 4
    if frac_tn < 0.8:
        print(f"  HALT: only {frac_tn:.0%} of sanity frames have target < novel "
              "(threshold 80%) — binder cannot reject novel-XOF E (concerning A); "
              "running full eval would just confirm the failure mode")
        return 4
    print(f"[sanity] PASS — mean target < shuffled and ≥80% per-frame consistency")
    del sanity_binder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.sanity_only:
        print("[binder_novel_eval] --sanity-only set; exiting after sanity")
        return 0

    # ---- full eval matrix: frames × conditions × binders ----
    n_conds = 2 + args.n_novel  # target, shuffled, 20 novel
    cond_labels = ["target", "shuffled"] + [f"novel_{i:02d}" for i in range(args.n_novel)]
    n_b = len(binder_names)
    mse_mat = np.full((n_actual, n_conds, n_b), np.nan, dtype=np.float64)

    print(f"\n[binder_novel_eval] FULL EVAL: {n_actual} frames × {n_conds} conds × "
          f"{n_b} binders = {n_actual * n_b} forwards + "
          f"{n_actual * n_conds * n_b} MSEs")

    overall_t0 = time.time()
    for bi, bname in enumerate(binder_names):
        spec = binders_meta[bname]["spec"]
        try:
            binder = load_binder(spec, device, dtype)
        except RuntimeError as ex:
            print(f"  [{bname}] load failed: {ex}; recording NaN row, continuing")
            continue
        cap_h, cap_w = spec["capture_h"], spec["capture_w"]
        print(f"  [{bi+1}/{n_b}] {bname} arch={spec['arch']} "
              f"cap={cap_h}x{cap_w}", flush=True)
        b_t0 = time.time()
        for fi, fm in enumerate(frame_meta):
            sess = fm["session"]; row = fm["row"]; partner_row = fm["partner_row"]
            sd = SESSION_DIRS[sess]
            cap = load_capture_at(sd / "Recordings" / f"frame_{row:06d}.raw",
                                  cap_h, cap_w)
            cap_b = cap.unsqueeze(0).to(device, dtype=dtype)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                     dtype=dtype if dtype != torch.float32 else torch.float32,
                                     enabled=(dtype != torch.float32)):
                pred = binder(cap_b).float().clamp(0, 1).squeeze(0).cpu()
            E_target = load_emission_at(
                sd / "derived" / "Emissions" / f"tile_{row:06d}.png",
                EMISSION_H_NATIVE, EMISSION_W_NATIVE)
            E_shuffled = load_emission_at(
                sd / "derived" / "Emissions" / f"tile_{partner_row:06d}.png",
                EMISSION_H_NATIVE, EMISSION_W_NATIVE)
            mse_mat[fi, 0, bi] = float(((pred - E_target) ** 2).mean())
            mse_mat[fi, 1, bi] = float(((pred - E_shuffled) ** 2).mean())
            for ni, En in enumerate(novel_E):
                mse_mat[fi, 2 + ni, bi] = float(((pred - En) ** 2).mean())
            if (fi + 1) % 20 == 0:
                print(f"    frame {fi+1}/{n_actual}  "
                      f"elapsed={time.time()-b_t0:.0f}s", flush=True)
        del binder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        b_elapsed = time.time() - b_t0
        # Per-binder summary print
        m_t = np.nanmean(mse_mat[:, 0, bi])
        m_s = np.nanmean(mse_mat[:, 1, bi])
        m_n = np.nanmean(mse_mat[:, 2:, bi])
        print(f"    [{bname}] done in {b_elapsed:.0f}s  "
              f"target={m_t:.5f} shuffled={m_s:.5f} novel={m_n:.5f}")
    overall_elapsed = time.time() - overall_t0
    print(f"[binder_novel_eval] full eval done in {overall_elapsed:.0f}s")

    # ---- check for NaN ----
    if not np.isfinite(mse_mat).any():
        print("[binder_novel_eval] HALT: all-NaN result matrix")
        return 5
    if np.isnan(mse_mat).all():
        print("[binder_novel_eval] HALT: complete NaN — no valid binder loaded")
        return 5

    # ---- save raw matrix ----
    np.savez_compressed(args.out / "raw_scores.npz",
                        mse_mat=mse_mat,
                        binder_names=np.array(binder_names),
                        cond_labels=np.array(cond_labels),
                        sessions=np.array([fm["session"] for fm in frame_meta]),
                        rows=np.array([fm["row"] for fm in frame_meta]),
                        partner_rows=np.array([fm["partner_row"] for fm in frame_meta]))
    print(f"[binder_novel_eval] raw_scores.npz saved (shape={mse_mat.shape})")

    # ---- per-binder + aggregate metrics ----
    per_binder = {}
    for bi, bname in enumerate(binder_names):
        col = mse_mat[:, :, bi]
        if np.isnan(col).all():
            per_binder[bname] = {"loaded": False, "reason": "all-NaN (load failed)"}
            continue
        # Codex audit HIGH 2026-05-04: partial NaN should also fail-loud, not
        # silently flow into nanmean and produce misleading per-binder stats.
        # If any frame for this binder has any NaN/Inf score, mark as failed.
        if not np.isfinite(col).all():
            n_bad = int((~np.isfinite(col)).sum())
            per_binder[bname] = {
                "loaded": False,
                "reason": f"partial NaN/Inf ({n_bad} non-finite scores)"
            }
            print(f"  [{bname}] FAILED: {n_bad} non-finite scores in matrix; "
                  "marking as not-loaded for AUROC/aggregate")
            continue
        m_target_per_frame = col[:, 0]
        m_shuffled_per_frame = col[:, 1]
        m_novel_flat = col[:, 2:].reshape(-1)
        # AUROC: target (positive=lower MSE = correct match) vs negatives (higher MSE)
        auc_target_vs_novel    = auroc(m_target_per_frame, m_novel_flat)
        auc_target_vs_shuffled = auroc(m_target_per_frame, m_shuffled_per_frame)
        per_binder[bname] = {
            "loaded": True,
            "arch": binders_meta[bname]["spec"]["arch"],
            "n_frames": int((~np.isnan(col[:, 0])).sum()),
            "mean_mse": {
                "target":   float(np.nanmean(m_target_per_frame)),
                "shuffled": float(np.nanmean(m_shuffled_per_frame)),
                "novel":    float(np.nanmean(m_novel_flat)),
            },
            "std_mse": {
                "target":   float(np.nanstd(m_target_per_frame)),
                "shuffled": float(np.nanstd(m_shuffled_per_frame)),
                "novel":    float(np.nanstd(m_novel_flat)),
            },
            "ratio": {
                "shuffled_over_target": float(np.nanmean(m_shuffled_per_frame) /
                                               np.nanmean(m_target_per_frame))
                                          if np.nanmean(m_target_per_frame) > 0 else float("inf"),
                "novel_over_target":    float(np.nanmean(m_novel_flat) /
                                               np.nanmean(m_target_per_frame))
                                          if np.nanmean(m_target_per_frame) > 0 else float("inf"),
                "novel_over_shuffled":  float(np.nanmean(m_novel_flat) /
                                               np.nanmean(m_shuffled_per_frame))
                                          if np.nanmean(m_shuffled_per_frame) > 0 else float("inf"),
            },
            "auroc": {
                "target_vs_novel":    float(auc_target_vs_novel),
                "target_vs_shuffled": float(auc_target_vs_shuffled),
            },
        }
    (args.out / "per_binder_summary.json").write_text(
        json.dumps(per_binder, indent=2))
    print(f"[binder_novel_eval] per_binder_summary.json saved "
          f"({sum(1 for v in per_binder.values() if v.get('loaded'))} loaded)")

    # ---- aggregate metrics ----
    loaded = [v for v in per_binder.values() if v.get("loaded")]
    if not loaded:
        print("[binder_novel_eval] HALT: no binders loaded successfully")
        return 6
    aurocs_tn = np.array([v["auroc"]["target_vs_novel"] for v in loaded])
    aurocs_ts = np.array([v["auroc"]["target_vs_shuffled"] for v in loaded])
    means_t = np.array([v["mean_mse"]["target"] for v in loaded])
    means_s = np.array([v["mean_mse"]["shuffled"] for v in loaded])
    means_n = np.array([v["mean_mse"]["novel"] for v in loaded])

    # Interpretation classification
    # Best:   AUROC(t v n) > 0.95 AND AUROC(t v s) > 0.95
    #         AND novel/shuffled ratio close to 1 (in [0.7, 1.5])
    # ConcA:  AUROC(t v n) ≈ 0.5 (within [0.4, 0.6])  → binders pattern-match any E
    # ConcB:  AUROC(t v n) variance high OR mean far from [0.5, 0.95]
    # Inter:  partial — neither best nor concerning extremes
    auroc_tn_mean = float(np.nanmean(aurocs_tn))
    auroc_ts_mean = float(np.nanmean(aurocs_ts))
    auroc_tn_std  = float(np.nanstd(aurocs_tn))
    nov_over_shuf_mean = float(np.nanmean(means_n / np.maximum(means_s, 1e-12)))

    # Codex audit MED 2026-05-04: "best" must require EVERY loaded binder
    # to pass the bounds individually, not just the aggregate mean — otherwise
    # one or two strong binders can mask a fleet of failed ones.
    per_binder_passes_best = all(
        v["auroc"]["target_vs_novel"] >= 0.90
        and v["auroc"]["target_vs_shuffled"] >= 0.90
        and 0.5 <= (v["mean_mse"]["novel"] / max(v["mean_mse"]["shuffled"], 1e-12)) <= 2.0
        for v in loaded
    )
    if (auroc_tn_mean >= 0.95 and auroc_ts_mean >= 0.95
            and 0.7 <= nov_over_shuf_mean <= 1.5
            and per_binder_passes_best):
        case = "best"
    elif 0.4 <= auroc_tn_mean <= 0.6:
        case = "concerning_A_pattern_match_any_E"
    elif auroc_tn_std > 0.15 or not (0.5 <= auroc_tn_mean <= 0.99):
        case = "concerning_B_unpredictable"
    else:
        case = "intermediate"

    aggregate = {
        "n_binders_loaded": len(loaded),
        "n_frames": int(n_actual),
        "n_conds": int(n_conds),
        "auroc_target_vs_novel": {
            "mean": auroc_tn_mean,
            "std":  auroc_tn_std,
            "min":  float(np.nanmin(aurocs_tn)),
            "max":  float(np.nanmax(aurocs_tn)),
        },
        "auroc_target_vs_shuffled": {
            "mean": auroc_ts_mean,
            "std":  float(np.nanstd(aurocs_ts)),
            "min":  float(np.nanmin(aurocs_ts)),
            "max":  float(np.nanmax(aurocs_ts)),
        },
        "ratio_novel_over_shuffled": {
            "mean": nov_over_shuf_mean,
            "min":  float(np.nanmin(means_n / np.maximum(means_s, 1e-12))),
            "max":  float(np.nanmax(means_n / np.maximum(means_s, 1e-12))),
        },
        "interpretation_case": case,
    }
    (args.out / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))
    print(f"[binder_novel_eval] aggregate_summary.json saved (case={case})")

    # ---- report ----
    write_report(args.out, per_binder, aggregate, binders_meta, args.n_novel,
                 sampled, novel_s_t_hex, overall_elapsed)
    print(f"[binder_novel_eval] report → {args.out / 'binder_novel_eval_report.md'}")
    return 0


def write_report(out_dir: Path, per_binder: dict, aggregate: dict,
                 binders_meta: dict, n_novel: int,
                 sampled: list, novel_s_t_hex: list, eval_time_s: float):
    case = aggregate["interpretation_case"]
    case_text = {
        "best": "**BEST CASE** — binders cleanly discriminate novel-XOF E. "
                "F-A v2 binder loss provides meaningful training signal.",
        "concerning_A_pattern_match_any_E":
                "**CONCERNING A** — binders give similar MSE for novel and target "
                "E. They pattern-match any E-like input and won't constrain F-A v2.",
        "concerning_B_unpredictable":
                "**CONCERNING B** — binder behaviour on novel E is inconsistent "
                "or unpredictable. F-A v2 plan needs rethinking.",
        "intermediate":
                "**INTERMEDIATE** — partial discrimination. F-A v2 viable but "
                "training signal partially compromised; tune loss weights/curriculum.",
    }[case]
    md = []
    md.append("# Binder novel-XOF discrimination preflight")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"Eval wall: {eval_time_s:.0f} s")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    md.append(case_text)
    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| binders loaded | {aggregate['n_binders_loaded']} |")
    md.append(f"| frames sampled | {aggregate['n_frames']} ({sum(1 for s,_ in sampled if s == 'D2')} D2 / {sum(1 for s,_ in sampled if s == 'V10')} V10) |")
    md.append(f"| novel seeds | {n_novel} |")
    md.append(f"| AUROC(target vs novel) mean ± std | "
              f"{aggregate['auroc_target_vs_novel']['mean']:.3f} ± "
              f"{aggregate['auroc_target_vs_novel']['std']:.3f} |")
    md.append(f"| AUROC(target vs novel) min/max | "
              f"{aggregate['auroc_target_vs_novel']['min']:.3f} / "
              f"{aggregate['auroc_target_vs_novel']['max']:.3f} |")
    md.append(f"| AUROC(target vs shuffled) mean ± std | "
              f"{aggregate['auroc_target_vs_shuffled']['mean']:.3f} ± "
              f"{aggregate['auroc_target_vs_shuffled']['std']:.3f} |")
    md.append(f"| MSE(novel) / MSE(shuffled) mean | "
              f"{aggregate['ratio_novel_over_shuffled']['mean']:.3f} |")
    md.append("")
    md.append("**Interpretation guide**:")
    md.append("- AUROC ≈ 1.0: binder reliably places target at lower MSE than negatives.")
    md.append("- AUROC ≈ 0.5: binder gives comparable MSE regardless of E (failure).")
    md.append("- novel/shuffled ratio ≈ 1: novel E is rejected as much as a shuffled real E.")
    md.append("- novel/shuffled ratio << 1: novel E somehow scores BETTER than a shuffled real E (worrying).")
    md.append("- novel/shuffled ratio >> 1: novel E scores worse than shuffled (binder over-attributes mismatch to novelty).")
    md.append("")
    md.append("## Per-binder")
    md.append("")
    md.append("| binder | arch | mean MSE target | mean MSE shuffled | mean MSE novel | "
              "AUROC(t v n) | AUROC(t v s) | shuffled/target | novel/target | novel/shuffled |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for bname in sorted(per_binder.keys()):
        v = per_binder[bname]
        if not v.get("loaded"):
            md.append(f"| {bname} | — | — | — | — | — | — | — | — | — |  *(load failed)* ")
            continue
        md.append(f"| {bname} | {v['arch']} | "
                  f"{v['mean_mse']['target']:.5f} | {v['mean_mse']['shuffled']:.5f} | "
                  f"{v['mean_mse']['novel']:.5f} | "
                  f"{v['auroc']['target_vs_novel']:.3f} | "
                  f"{v['auroc']['target_vs_shuffled']:.3f} | "
                  f"{v['ratio']['shuffled_over_target']:.2f} | "
                  f"{v['ratio']['novel_over_target']:.2f} | "
                  f"{v['ratio']['novel_over_shuffled']:.2f} |")
    md.append("")
    md.append("## Methodology")
    md.append("")
    md.append("- 100 frames stratified across D2 + V10 Phase G held-out blocks.")
    md.append("- For each frame: real C (resized to per-binder capture HW), real E_target, "
              "shuffled E (lag>60 same session), and 20 novel E values rendered from random "
              "32-byte chain seeds (BLAKE3-XOF expansion + bit-exact renderer).")
    md.append(f"- Each binder forward predicts (3, 1080, 1920) E from C; we then compute MSE "
              f"against each of {n_novel + 2} E variants.")
    md.append("- AUROC computed per-binder (positive class = target, negative class = novel "
              "or shuffled). Convention: lower MSE = better → AUROC = 1 is perfect "
              "discrimination, 0.5 is chance.")
    md.append("- Sanity gate: 5 frames × 1 binder × 3 conds verified m_target < m_shuffled "
              "before launching full eval.")
    md.append("")
    md.append("## Source files")
    md.append("")
    md.append(f"- `seeds.json` — {n_novel} novel S_t hex values (deterministic if "
              f"`--novel-seed-bytes` was set; default = os.urandom).")
    md.append("- `raw_scores.npz` — full mse_mat[frames, conds, binders] + axis labels.")
    md.append("- `per_binder_summary.json` — AUROCs + ratios + means per binder.")
    md.append("- `aggregate_summary.json` — across-binder stats + interpretation case.")
    md.append("")
    (out_dir / "binder_novel_eval_report.md").write_text("\n".join(md))


if __name__ == "__main__":
    raise SystemExit(main())
