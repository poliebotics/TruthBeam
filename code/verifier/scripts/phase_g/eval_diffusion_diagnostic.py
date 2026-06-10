"""Phase G — evaluation harness for the diffusion diagnostic.

For each held-out frame from D2 and V10:
  - Add fixed noise at timesteps {50, 150, 300, 500, 750}
  - K=4 noise samples per timestep
  - Score eps_pred vs true noise under conditions:
      correct E, wrong E with offsets ±2, ±15, +30 (within same eval block),
      zero E (force_uncond=True)
  - SAME noise sample paired across all conditions for each (frame, timestep, k)

Lag sweep:
  - For each held-out frame, evaluate at E_{r+k} for k ∈ {-30,-15,-5,-2,-1,0,
    +1,+2,+5,+15,+30}
  - Pair against the same noise sample
  - Plot mean MSE vs k → real chain coupling peaks at k=0

Metrics (computed per held-out condition, D2 and V10 separately):
  - Δ_wrong   = mean(MSE(wrong E)) - mean(MSE(correct E))
  - Δ_uncond  = mean(MSE(zero E))  - mean(MSE(correct E))
  - AUROC of correct vs wrong (paired comparison: per-frame correct vs each wrong)
  - Lag curve (mean MSE vs k)
  - Block-bootstrap 95% CIs (resample whole eval blocks)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/phase_g/eval_diffusion_diagnostic.py \
    --ckpt <path>/model_final.pt \
    --d2-dir <path>/data/d2 --v10-dir <path>/data/v10 \
    --out <path>/eval --n-d2 200 --n-v10 200 --bs 4 --bf16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    DiffusionDiagnosticDataset, EVAL_BLOCKS, _crop_and_resize_C,
    _load_packed_cfa_float01, _resize_E_to_target, EMISSION_NATIVE_H,
    EMISSION_NATIVE_W,
)
from data.emission_dataset import load_emission_at  # noqa: E402


PRIMARY_TIMESTEPS = (150, 300, 500)
SECONDARY_TIMESTEPS = (50, 750)
ALL_TIMESTEPS = (50, 150, 300, 500, 750)
K_NOISE = 4
WRONG_OFFSETS = (-2, +2, -15, +15, +30)
LAG_SWEEP_KS = (-30, -15, -5, -2, -1, 0, +1, +2, +5, +15, +30)


def block_bootstrap_ci(values: np.ndarray, block_lengths: list[int],
                       n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float, float]:
    """Block bootstrap of `values` arranged so the first `block_lengths[0]`
    entries belong to block 0, next `block_lengths[1]` to block 1, etc.

    Returns (mean, lower_q, upper_q) of the bootstrap distribution of the
    grand mean.
    """
    rng = np.random.RandomState(seed)
    boundaries = [0]
    for L in block_lengths:
        boundaries.append(boundaries[-1] + L)
    blocks = [values[boundaries[i]:boundaries[i+1]] for i in range(len(block_lengths))]
    boot_means = np.zeros(n_boot, dtype=np.float64)
    n_blocks = len(blocks)
    for b in range(n_boot):
        idx = rng.randint(0, n_blocks, size=n_blocks)
        cat = np.concatenate([blocks[i] for i in idx]) if n_blocks > 0 else np.zeros(0)
        boot_means[b] = cat.mean() if cat.size > 0 else float("nan")
    return float(np.nanmean(values)), float(np.nanquantile(boot_means, alpha / 2)), \
           float(np.nanquantile(boot_means, 1 - alpha / 2))


def auroc_pooled(correct_mse: np.ndarray, wrong_mse: np.ndarray) -> float:
    """Pooled (Mann-Whitney) AUROC where the positive label is "correct E"
    (lower MSE = stronger 'correct' signal). Tie-aware: uses average ranks
    so that perfectly identical scores (e.g. shuffled control giving 0.00391
    for every condition to 5 decimals) return AUROC = 0.5 exactly rather
    than a noise-biased value. NOT a paired metric — pools all correct-frame
    MSEs vs all wrong-frame MSEs."""
    correct = correct_mse.flatten()
    wrong = wrong_mse.flatten()
    # Filter NaN
    correct = correct[np.isfinite(correct)]
    wrong = wrong[np.isfinite(wrong)]
    if correct.size == 0 or wrong.size == 0:
        return float("nan")
    # Score: -MSE (so higher = more "correct-like")
    s_correct = -correct
    s_wrong = -wrong
    n1, n2 = s_correct.size, s_wrong.size
    all_scores = np.concatenate([s_correct, s_wrong])
    # Average-rank tie handling (Mann-Whitney convention).
    # Equivalent to scipy.stats.rankdata(all_scores, method="average") but no scipy dep.
    order = np.argsort(all_scores, kind="stable")
    sorted_scores = all_scores[order]
    ranks_sorted = np.empty(len(all_scores), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    R1 = ranks[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2
    return float(U / (n1 * n2))


def load_model(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> DiffusionDiagnosticUNet:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    args = ck.get("args", {}) if isinstance(ck, dict) else {}
    base_ch = args.get("base_ch", 96)
    mults = tuple(args.get("mults", (1, 2, 4, 4)))
    attn_at = args.get("attn_at")
    if attn_at is None:
        # Default matches the trainer's default (last-only). If a checkpoint
        # was trained with a different config, it's stored in args["attn_at"]
        # and we use that instead.
        attn_at = tuple(i == len(mults) - 1 for i in range(len(mults)))
    else:
        attn_at = tuple(bool(x) for x in attn_at)
    model = DiffusionDiagnosticUNet(
        in_ch=4, base_ch=base_ch,
        channel_mults=mults,
        attn_at=attn_at,
        cond_drop_prob=0.0,  # eval: no dropout
        hint_in_ch=11,
    ).to(device, dtype=dtype)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_session(model: DiffusionDiagnosticUNet, session: str, session_dir: Path,
                     device: torch.device, dtype: torch.dtype, dc: dict,
                     n_frames: int, bs: int, seed: int = 0,
                     shard: int = 0, num_shards: int = 1,
                     extended_perturbations: bool = False) -> dict:
    """Evaluate per-frame MSE under all conditions for one session.

    If num_shards > 1, only process frames where (global_frame_idx % num_shards) == shard.
    Each shard gets ~n_frames/num_shards frames; shards write disjoint outputs
    that can be merged by `merge_eval_shards.py`.
    """
    rng = np.random.RandomState(seed)
    blocks = EVAL_BLOCKS[session]
    # Sample n_frames evenly across blocks (deterministic via seed).
    per_block = max(1, n_frames // len(blocks))
    sampled_all: list[tuple[int, int]] = []  # (block_idx, frame_row)
    for bi, (a, b) in enumerate(blocks):
        valid = list(range(a + 30, b - 30))  # inset for wrong-E offsets up to ±30
        if len(valid) <= per_block:
            chosen = valid
        else:
            chosen = rng.choice(valid, size=per_block, replace=False).tolist()
        for r in sorted(chosen):
            sampled_all.append((bi, r))
    # Apply shard filter — preserve the GLOBAL pre-shard index for use as the
    # noise seed, so every (frame, t, k) gets a unique deterministic noise
    # tensor regardless of which shard it ran on. (Without this, each shard's
    # local fi=0 would re-use the same noise tensor → cross-frame noise dups.)
    sampled_with_gi: list[tuple[int, int, int]] = [
        (gi, bi, row) for gi, (bi, row) in enumerate(sampled_all)
    ]
    if num_shards > 1:
        sampled_with_gi = [t for t in sampled_with_gi if (t[0] % num_shards) == shard]
        print(f"[eval] sharded: shard={shard}/{num_shards}  "
              f"{len(sampled_with_gi)}/{len(sampled_all)} frames", flush=True)
    sampled = [(bi, row) for _, bi, row in sampled_with_gi]
    global_indices = [gi for gi, _, _ in sampled_with_gi]

    # Helpers to load assets per-row.
    def load_C(row: int) -> torch.Tensor:
        return _crop_and_resize_C(_load_packed_cfa_float01(
            session_dir / "Recordings" / f"frame_{row:06d}.raw"))

    def load_E(row: int) -> torch.Tensor:
        return _resize_E_to_target(load_emission_at(
            session_dir / "derived" / "Emissions" / f"tile_{row:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W))

    # Per-frame storage.
    # results[frame_idx][condition_label][t_idx][k_noise_idx] = scalar MSE
    n_frames_actual = len(sampled)
    cond_labels = (["correct"] +
                   [f"wrong_{off:+d}" for off in WRONG_OFFSETS] +
                   ["uncond"] +
                   [f"lag_{k:+d}" for k in LAG_SWEEP_KS])

    # Extended perturbations (Item 1): pre-load chain log + spec list
    perturbation_specs = []
    chain_log = None
    if extended_perturbations:
        from phase_g import xof_perturb as xp
        perturbation_specs = xp.all_perturbation_specs()
        chain_log = xp.load_chain_log(session_dir)
        cond_labels.extend([s.label for s in perturbation_specs])
        print(f"[eval] extended_perturbations: +{len(perturbation_specs)} conditions",
              flush=True)
    results = {lbl: np.full((n_frames_actual, len(ALL_TIMESTEPS), K_NOISE), np.nan,
                            dtype=np.float64)
               for lbl in cond_labels}
    block_assignment = [bi for bi, _ in sampled]

    print(f"[eval] session={session} n_frames={n_frames_actual} "
          f"timesteps={ALL_TIMESTEPS} K={K_NOISE}", flush=True)

    t_start = time.time()
    for fi, (bi, row) in enumerate(sampled):
        # Load C and E batch for this frame (we'll batch over noise samples).
        C0 = load_C(row).to(device, dtype=dtype)
        # Pre-load all needed E tiles for this frame:
        #   - correct: E[row]
        #   - wrong_*: E[row + off] where off ∈ WRONG_OFFSETS (within same block)
        #   - lag_*:   E[row + k]  for k in LAG_SWEEP_KS (within same block)
        E_cache: dict[int, torch.Tensor] = {}
        E_cache[row] = load_E(row).to(device, dtype=dtype)
        block_a, block_b = blocks[bi]

        def safe_offset(off: int) -> int:
            r = row + off
            r = max(block_a, min(block_b - 1, r))
            return r

        for off in WRONG_OFFSETS:
            r = safe_offset(off)
            if r not in E_cache:
                E_cache[r] = load_E(r).to(device, dtype=dtype)
        for k in LAG_SWEEP_KS:
            r = safe_offset(k)
            if r not in E_cache:
                E_cache[r] = load_E(r).to(device, dtype=dtype)

        # Extended perturbations: render perturbed E for each spec.
        # Cached per-spec by spec.label (string keys; doesn't collide with row ints).
        perturb_E: dict[str, torch.Tensor] = {}
        if extended_perturbations:
            from phase_g import xof_perturb as xp
            if chain_log is None:
                raise RuntimeError(
                    "extended_perturbations requires a loaded chain_log")
            if row not in chain_log:
                raise RuntimeError(
                    f"row {row} missing from chain_log for {session}; cannot derive "
                    f"real XOF streams. Aborting rather than silently degrading.")
            if len(perturbation_specs) != 35:
                raise RuntimeError(
                    f"expected 35 perturbation specs (34 + 1 calibration), "
                    f"got {len(perturbation_specs)}")
            base_streams = xp.expand_streams_from_s_t(chain_log[row])
            session_total_rows = max(chain_log.keys()) + 1
            # General donor (lag>60), used by Type 4/5/6_replace
            donor_row_gen = xp.cross_frame_partner(
                session, row, session_total_rows, salt="general", min_gap=60)
            # Per-octave donors (Type 4)
            donor_rows_oct = {
                oct_idx: xp.cross_frame_partner(
                    session, row, session_total_rows, salt=f"oct{oct_idx}", min_gap=60)
                for oct_idx in xp.TYPE4_OCTAVES
            }
            # Per-channel donors (Type 5)
            donor_rows_chan = {
                ch_idx: xp.cross_frame_partner(
                    session, row, session_total_rows, salt=f"chan{ch_idx}", min_gap=60)
                for ch_idx in xp.TYPE5_CHANNELS
            }
            # Calibration donor (Type 6 calibration). Clamp to chain_log range.
            calib_row = max(0, min(session_total_rows - 1, row + 30))

            def streams_for(r_donor: int) -> tuple[bytes, bytes, bytes]:
                if r_donor not in chain_log:
                    raise ValueError(
                        f"donor row {r_donor} missing from chain_log; cannot "
                        f"silently degrade Type 4/5/6 perturbations.")
                return xp.expand_streams_from_s_t(chain_log[r_donor])

            for spec in perturbation_specs:
                if spec.type == "type4_octave_swap":
                    donor = streams_for(donor_rows_oct[spec.octave_idx])
                elif spec.type == "type5_channel_swap":
                    donor = streams_for(donor_rows_chan[spec.channel_idx])
                elif spec.type == "type6_replace":
                    donor = streams_for(donor_row_gen)
                elif spec.type == "type6_calibration":
                    donor = streams_for(calib_row)
                else:
                    donor = None
                streams_p = xp.apply_spec(spec, base_streams, donor, session, row)
                E_p = xp.render_perturbed_E(streams_p, device="cpu").to(device, dtype=dtype)
                perturb_E[spec.label] = E_p
            # Verify all spec labels rendered (catch silent skips). Explicit
            # if-not-raise (NOT assert) so this check survives `python -O`.
            for spec in perturbation_specs:
                if spec.label not in perturb_E:
                    raise RuntimeError(
                        f"spec {spec.label} did not render; refusing to score "
                        f"as uncond by default.")

        # Build a fixed (timesteps × K_noise) noise tensor for this frame.
        # Seed by the GLOBAL frame index so different shards produce independent
        # noise tensors and merging shards yields cross-frame independent noise.
        with torch.no_grad():
            # noise: (T_idx, K, 4, H, W)
            H, W = C0.shape[-2:]
            gi = global_indices[fi]
            torch.manual_seed(seed + gi)
            noise = torch.randn(len(ALL_TIMESTEPS), K_NOISE, 4, H, W,
                                device=device, dtype=torch.float32)

            # Stack batches across (timesteps × K) for this frame.
            for ti, t_val in enumerate(ALL_TIMESTEPS):
                t_tensor = torch.full((K_NOISE,), t_val, device=device, dtype=torch.long)
                # q_sample expects float32 + dc constants
                C_rep = C0.float().unsqueeze(0).expand(K_NOISE, -1, -1, -1)
                C_t = q_sample(C_rep, t_tensor, dc, noise[ti]).to(dtype)
                t_float = t_tensor.float()

                # Score under each condition. Run model once per condition.
                # Collect (cond_label, E_tensor_or_uncond_flag) tuples.
                cond_list: list[tuple[str, torch.Tensor | None]] = []
                cond_list.append(("correct", E_cache[row]))
                for off in WRONG_OFFSETS:
                    r = safe_offset(off)
                    cond_list.append((f"wrong_{off:+d}", E_cache[r]))
                cond_list.append(("uncond", None))
                for k in LAG_SWEEP_KS:
                    r = safe_offset(k)
                    cond_list.append((f"lag_{k:+d}", E_cache[r]))
                # Extended perturbations
                for spec in perturbation_specs:
                    cond_list.append((spec.label, perturb_E.get(spec.label)))

                # Sub-batch by `bs` to keep memory bounded.
                for cond_label, E_tensor in cond_list:
                    # Build E batch (K, 3, H, W) — same E for each of the K noise samples.
                    if E_tensor is None:
                        E_batch = torch.zeros(K_NOISE, 3, H, W, device=device, dtype=dtype)
                        force_uncond = True
                    else:
                        E_batch = E_tensor.unsqueeze(0).expand(K_NOISE, -1, -1, -1).contiguous()
                        force_uncond = False

                    mses = []
                    for s in range(0, K_NOISE, bs):
                        e = min(s + bs, K_NOISE)
                        with torch.amp.autocast("cuda", dtype=dtype):
                            eps_pred = model(C_t[s:e], E_batch[s:e], t_float[s:e],
                                             force_uncond=force_uncond)
                        diff = (eps_pred.float() - noise[ti, s:e].float())
                        # Mean MSE per sample.
                        mse_each = diff.pow(2).mean(dim=(1, 2, 3))
                        mses.append(mse_each.cpu().numpy())
                    mses_np = np.concatenate(mses)
                    results[cond_label][fi, ti, :] = mses_np

        if fi % 5 == 0 or fi == n_frames_actual - 1:
            elapsed = time.time() - t_start
            print(f"  [eval {session}] frame {fi+1}/{n_frames_actual} "
                  f"row={row} elapsed={elapsed:.0f}s", flush=True)

    return {
        "session": session,
        "rows": [r for _, r in sampled],
        "block_idx": block_assignment,
        "results": results,
        "timesteps": list(ALL_TIMESTEPS),
        "wrong_offsets": list(WRONG_OFFSETS),
        "lag_ks": list(LAG_SWEEP_KS),
        "K_noise": K_NOISE,
    }


def summarize(eval_data: dict) -> dict:
    """Compute Δ_wrong, Δ_uncond, AUROC, lag curve with block-bootstrap CIs."""
    res = eval_data["results"]
    block_idx = np.array(eval_data["block_idx"])
    wrong_offsets = eval_data["wrong_offsets"]
    lag_ks = eval_data["lag_ks"]
    timesteps = eval_data["timesteps"]
    n_frames = res["correct"].shape[0]

    # block_lengths in the order rows are stored.
    n_blocks = int(block_idx.max() + 1) if n_frames > 0 else 0
    block_lengths = [int((block_idx == bi).sum()) for bi in range(n_blocks)]

    # Average over (timesteps, K) to get one scalar per frame per condition.
    def per_frame(cond: str) -> np.ndarray:
        return res[cond].mean(axis=(1, 2))  # (n_frames,)

    correct = per_frame("correct")
    uncond = per_frame("uncond")
    wrongs = {off: per_frame(f"wrong_{off:+d}") for off in wrong_offsets}
    wrongs_avg = np.stack(list(wrongs.values()), axis=1).mean(axis=1)

    # Δ_wrong, Δ_uncond per frame
    delta_wrong = wrongs_avg - correct
    delta_uncond = uncond - correct

    summary: dict = {
        "n_frames": int(n_frames),
        "block_sizes": block_lengths,
        "by_condition": {},
        "deltas": {},
        "auroc": {},
        "lag_curve": {},
        "by_timestep": {},
    }

    if n_frames == 0:
        return summary

    # Mean MSE per condition (with block-bootstrap CI).
    for cond in ["correct", "uncond"] + [f"wrong_{off:+d}" for off in wrong_offsets]:
        v = per_frame(cond)
        m, lo, hi = block_bootstrap_ci(v, block_lengths)
        summary["by_condition"][cond] = {"mean": m, "ci_low": lo, "ci_high": hi}

    # Deltas
    m, lo, hi = block_bootstrap_ci(delta_wrong, block_lengths)
    summary["deltas"]["delta_wrong"] = {"mean": m, "ci_low": lo, "ci_high": hi}
    m, lo, hi = block_bootstrap_ci(delta_uncond, block_lengths)
    summary["deltas"]["delta_uncond"] = {"mean": m, "ci_low": lo, "ci_high": hi}

    # AUROC (paired). For each frame, "correct" gives one MSE, "wrong_avg" gives one.
    # AUROC over the pooled distribution.
    auroc = auroc_pooled(correct, wrongs_avg)
    summary["auroc"]["correct_vs_wrong_avg"] = float(auroc)
    # Per-offset AUROC
    for off, w in wrongs.items():
        summary["auroc"][f"correct_vs_wrong_{off:+d}"] = float(auroc_pooled(correct, w))
    summary["auroc"]["correct_vs_uncond"] = float(auroc_pooled(correct, uncond))

    # Lag curve: mean MSE vs k
    lag_curve = {}
    for k in lag_ks:
        v = res[f"lag_{k:+d}"].mean(axis=(1, 2))  # (n_frames,)
        m, lo, hi = block_bootstrap_ci(v, block_lengths)
        lag_curve[str(k)] = {"mean": m, "ci_low": lo, "ci_high": hi}
    summary["lag_curve"] = lag_curve

    # Per-timestep breakdown for primary t's
    for ti, t_val in enumerate(timesteps):
        if t_val not in PRIMARY_TIMESTEPS:
            continue
        c_t = res["correct"][:, ti, :].mean(axis=1)
        w_avg_t = np.stack([res[f"wrong_{off:+d}"][:, ti, :].mean(axis=1)
                            for off in wrong_offsets], axis=1).mean(axis=1)
        u_t = res["uncond"][:, ti, :].mean(axis=1)
        d_w = w_avg_t - c_t
        d_u = u_t - c_t
        m_dw, lo_dw, hi_dw = block_bootstrap_ci(d_w, block_lengths)
        m_du, lo_du, hi_du = block_bootstrap_ci(d_u, block_lengths)
        summary["by_timestep"][f"t={t_val}"] = {
            "delta_wrong": {"mean": m_dw, "ci_low": lo_dw, "ci_high": hi_dw},
            "delta_uncond": {"mean": m_du, "ci_low": lo_du, "ci_high": hi_du},
            "auroc_correct_vs_wrong_avg": float(auroc_pooled(c_t, w_avg_t)),
        }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-d2", type=int, default=200)
    ap.add_argument("--n-v10", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="Frame shard index (0..num_shards-1). When num_shards>1, "
                         "this process only handles frames where (idx %% num_shards) == shard.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="If > 1, output paths get a _shard{N} suffix and a separate "
                         "merge_eval_shards.py step combines them.")
    ap.add_argument("--extended-perturbations", action="store_true",
                    help="Item 1: add 35 XOF-perturbation conditions per frame "
                         "(Type 1 global bit-flip × 7 k-values, Type 2 octave-localized "
                         "× 16, Type 3 spatial-region × 3, Type 4 octave swap × 4, "
                         "Type 5 channel swap × 3, Type 6 replace + calibration × 2).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    model = load_model(args.ckpt, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] loaded ckpt={args.ckpt}  params={n_params/1e6:.1f}M  "
          f"device={device} dtype={dtype}", flush=True)

    dc = build_diffusion_constants(args.T, device, torch.float32)

    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    out: dict = {"sessions": {}}
    if args.n_d2 > 0:
        eval_d2 = evaluate_session(model, "D2", args.d2_dir, device, dtype, dc,
                                   n_frames=args.n_d2, bs=args.bs, seed=args.seed,
                                   shard=args.shard, num_shards=args.num_shards,
                                   extended_perturbations=args.extended_perturbations)
        np.savez(args.out / f"eval_d2_raw{suffix}.npz",
                 **{f"cond_{k}": v for k, v in eval_d2["results"].items()},
                 rows=np.array(eval_d2["rows"]),
                 block_idx=np.array(eval_d2["block_idx"]))
        if args.num_shards == 1:
            out["sessions"]["D2"] = summarize(eval_d2)
    if args.n_v10 > 0 and args.v10_dir is not None and args.v10_dir.exists():
        eval_v10 = evaluate_session(model, "V10", args.v10_dir, device, dtype, dc,
                                    n_frames=args.n_v10, bs=args.bs, seed=args.seed + 1,
                                    shard=args.shard, num_shards=args.num_shards,
                                   extended_perturbations=args.extended_perturbations)
        np.savez(args.out / f"eval_v10_raw{suffix}.npz",
                 **{f"cond_{k}": v for k, v in eval_v10["results"].items()},
                 rows=np.array(eval_v10["rows"]),
                 block_idx=np.array(eval_v10["block_idx"]))
        if args.num_shards == 1:
            out["sessions"]["V10"] = summarize(eval_v10)

    if args.num_shards == 1:
        (args.out / "summary.json").write_text(json.dumps(out, indent=2))
        print(f"\n[done] wrote {args.out}/summary.json + raw NPZ files", flush=True)
    else:
        print(f"\n[done] shard {args.shard}/{args.num_shards} wrote raw NPZs only — "
              f"run merge_eval_shards.py to compute summary.json", flush=True)


if __name__ == "__main__":
    main()
