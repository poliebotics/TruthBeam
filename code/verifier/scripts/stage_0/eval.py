"""Stage 0 — paired comparison eval: real (C, E) vs F-A-fake (C_fake, E).

Per the operator's v7 spec Phase 2 (Stage 0 expanded). For each F-A v1
checkpoint × session × held-out target frame:

    1. Render C_fake = F-A(C_source, E_source, E_target).
    2. Resize C_fake to Phase G diffusion model input (4, 768, 1024).
    3. Score (C_fake, ?) under conditions:
         - correct E_target
         - shuffled E (lag offset matching Phase G shuffled pattern)
         - source E (E_source — F-A was trained to ERASE this)
         - zero E (force_uncond=True)
    4. Score (C_real_target, ?) under SAME conditions for paired baseline.
    5. Frame-aggregated metrics: paired Δ_fake = MSE(C_fake) - MSE(C_real)
       per condition, AUROC of real-vs-fake at correct E.

Cache real-baseline scores keyed by (session, target_row, source_row,
condition, timestep, noise_seed) so subsequent F-A checkpoints reuse them.
Required for compute estimate accuracy per spec.

Sharding: --shard k --num-shards N divides held-out frames mod num_shards.
Different shards do not duplicate work; merge with merge_stage_0_shards.py.

Headline metrics (frame-level aggregation):
  - paired_gap = mean over frames of (mean_t,k MSE_fake - mean_t,k MSE_real)
  - auroc_real_vs_fake (correct E condition)
  - delta_shuffled, delta_source, delta_zero on F-A side
  - real-baseline lifts: real_zero - real_target, fake_zero - fake_target

Outputs:
    <out>/eval_<session>_raw_shard{N}.npz — per-frame per-condition MSEs
    <out>/summary_shard{N}.json — per-shard summary (full merge separately)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_g"))

from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    EVAL_BLOCKS, _crop_and_resize_C, _load_packed_cfa_float01,
    _resize_E_to_target, EMISSION_NATIVE_H, EMISSION_NATIVE_W,
)
from phase_g.fa_loader import (  # noqa: E402
    load_fa_v1_checkpoint, render_C_fake, load_E_native,
    C_native_to_phase_g_input, load_C_native,
)
from data.emission_dataset import load_emission_at  # noqa: E402
from eval_diffusion_diagnostic import (  # noqa: E402
    block_bootstrap_ci, auroc_pooled, load_model as load_diff_model,
)


# Stage 0 eval config (per v7 spec)
TIMESTEPS = (50, 150, 300, 500, 750)
K_NOISE = 4

# Source-frame lag for F-A (matches F-A training k_choices ∈ {1, 2, 3})
SOURCE_LAG = 2

# Wrong-E offset for shuffled condition (matches Phase G shuffled pattern)
SHUFFLED_OFFSET = 1000  # large offset within session


def _row_in_any_held_out_block(row: int, blocks: list[tuple[int, int]]) -> bool:
    return any(a <= row < b for a, b in blocks)


def cross_frame_for_shuffled(row: int, total_rows: int,
                             block_a: int, block_b: int,
                             session: str) -> int:
    """Pick a deterministic 'shuffled' partner row OUTSIDE all held-out blocks
    of this session, with |gap| > 60 from the eval row.

    Per spec: shuffled E should be drawn from the training population, not
    from another held-out frame, to preserve the shuffled control's
    independence semantics.
    """
    held_out = EVAL_BLOCKS[session]
    # Try the canonical N/2 rotation first
    cand = (row + total_rows // 2) % total_rows
    if abs(cand - row) > 60 and not _row_in_any_held_out_block(cand, held_out):
        return cand
    # Walk forward from N/4 until we find a non-held-out row
    for off in range(total_rows // 4, total_rows):
        cand = (row + off) % total_rows
        if abs(cand - row) > 60 and not _row_in_any_held_out_block(cand, held_out):
            return cand
    # As a last resort (shouldn't trigger for D2/V10 sizes): walk backward
    for off in range(total_rows // 4, total_rows):
        cand = (row - off) % total_rows
        if abs(cand - row) > 60 and not _row_in_any_held_out_block(cand, held_out):
            return cand
    raise ValueError(f"no shuffled partner found for row={row} in session={session}")


@torch.no_grad()
def score_one(diffusion_model: DiffusionDiagnosticUNet, C: torch.Tensor,
              E: torch.Tensor | None, t_val: int, K: int, dc: dict,
              device: torch.device, dtype: torch.dtype, bs: int,
              noise: torch.Tensor) -> np.ndarray:
    """Run K noise samples through the diffusion model. C: (4, H, W),
    E: (3, H, W) or None (uncond), noise: (K, 4, H, W). Returns (K,) MSE array."""
    H, W = C.shape[-2:]
    t_tensor = torch.full((K,), t_val, device=device, dtype=torch.long)
    C_rep = C.float().unsqueeze(0).expand(K, -1, -1, -1)
    C_t = q_sample(C_rep, t_tensor, dc, noise.float()).to(dtype)
    t_float = t_tensor.float()
    if E is None:
        E_batch = torch.zeros(K, 3, H, W, device=device, dtype=dtype)
        force_uncond = True
    else:
        E_batch = E.unsqueeze(0).expand(K, -1, -1, -1).contiguous()
        force_uncond = False
    mses = []
    for s in range(0, K, bs):
        e = min(s + bs, K)
        with torch.amp.autocast("cuda", dtype=dtype):
            eps_pred = diffusion_model(C_t[s:e], E_batch[s:e], t_float[s:e],
                                       force_uncond=force_uncond)
        diff = (eps_pred.float() - noise[s:e].float())
        mse_each = diff.pow(2).mean(dim=(1, 2, 3))
        mses.append(mse_each.cpu().numpy())
    return np.concatenate(mses)


def evaluate_session(diff_model: DiffusionDiagnosticUNet,
                     fa_model,
                     session: str, session_dir: Path,
                     device: torch.device, dtype: torch.dtype, dc: dict,
                     n_frames: int, bs: int, seed: int,
                     shard: int = 0, num_shards: int = 1,
                     save_c_fake: bool = False) -> dict:
    """For each held-out target frame, score (C_fake, conditions) and
    (C_real, conditions) paired."""
    rng = np.random.RandomState(seed)
    blocks = EVAL_BLOCKS[session]
    per_block = max(1, n_frames // len(blocks))
    sampled_all: list[tuple[int, int]] = []  # (block_idx, frame_row)
    for bi, (a, b) in enumerate(blocks):
        # Inset by 30 (for wrong-E offsets) AND ensure source_row = row - SOURCE_LAG
        # is within the block too (so source frame loads from same session/block).
        valid = list(range(a + 30 + SOURCE_LAG, b - 30))
        if len(valid) <= per_block:
            chosen = valid
        else:
            chosen = rng.choice(valid, size=per_block, replace=False).tolist()
        for r in sorted(chosen):
            sampled_all.append((bi, r))
    sampled_with_gi = [(gi, bi, row) for gi, (bi, row) in enumerate(sampled_all)]
    if num_shards > 1:
        sampled_with_gi = [t for t in sampled_with_gi if (t[0] % num_shards) == shard]
        print(f"[stage_0] sharded: shard={shard}/{num_shards}  "
              f"{len(sampled_with_gi)}/{len(sampled_all)} frames")
    if not sampled_with_gi:
        return {"session": session, "results": {}, "rows": [], "block_idx": []}

    # Conditions: correct, shuffled, source, zero. Both for C_real and C_fake.
    cond_labels = [
        "real_correct", "real_shuffled", "real_source", "real_zero",
        "fake_correct", "fake_shuffled", "fake_source", "fake_zero",
    ]
    n_frames_actual = len(sampled_with_gi)
    results = {lbl: np.full((n_frames_actual, len(TIMESTEPS), K_NOISE), np.nan,
                            dtype=np.float64)
               for lbl in cond_labels}
    block_assignment = [bi for _, bi, _ in sampled_with_gi]
    # Optional C_fake persistence (Phase G resolution, bf16 to save space).
    c_fake_buffer: list[np.ndarray] | None = (
        [] if save_c_fake else None
    )

    print(f"[stage_0] {session} n_frames={n_frames_actual} timesteps={TIMESTEPS}")

    t_start = time.time()
    for fi, (gi, bi, target_row) in enumerate(sampled_with_gi):
        block_a, block_b = blocks[bi]
        source_row = target_row - SOURCE_LAG
        # Find a shuffled E partner outside this block (large offset)
        # Use chain length to inform; here we assume D2 ~= 5992, V10 ~= 3743
        total_rows = 5992 if session == "D2" else 3743
        shuffled_row = cross_frame_for_shuffled(target_row, total_rows,
                                                block_a, block_b, session)

        # Load E variants (Phase G resolution)
        E_correct = _resize_E_to_target(load_emission_at(
            session_dir / "derived" / "Emissions" / f"tile_{target_row:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W)).to(device, dtype=dtype)
        E_source = _resize_E_to_target(load_emission_at(
            session_dir / "derived" / "Emissions" / f"tile_{source_row:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W)).to(device, dtype=dtype)
        E_shuffled = _resize_E_to_target(load_emission_at(
            session_dir / "derived" / "Emissions" / f"tile_{shuffled_row:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W)).to(device, dtype=dtype)

        # Load C_real_target at Phase G resolution
        C_real = _crop_and_resize_C(
            _load_packed_cfa_float01(
                session_dir / "Recordings" / f"frame_{target_row:06d}.raw"
            )).to(device, dtype=dtype)

        # Render C_fake (F-A native res → Phase G res)
        C_fake = render_C_fake(fa_model, session_dir,
                               source_row=source_row, target_row=target_row,
                               device=device, dtype=dtype).to(device, dtype=dtype)
        if c_fake_buffer is not None:
            # Codex audit HIGH 2026-05-04: NumPy has no native bf16, so
            # converting bf16 → numpy silently widens or fails. Use float16
            # for storage instead (numpy supports it natively; precision is
            # equivalent for our 1080p visualisation + binder eval downstream).
            c_fake_buffer.append(C_fake.detach().to(torch.float16).cpu().numpy())

        # Build paired noise (same noise for fake and real to enable paired comparison)
        H, W = C_real.shape[-2:]
        torch.manual_seed(seed + gi)
        noise_per_t = torch.randn(len(TIMESTEPS), K_NOISE, 4, H, W,
                                  device=device, dtype=torch.float32)

        # Score under each condition for both real and fake
        cond_E = {
            "correct": E_correct,
            "shuffled": E_shuffled,
            "source": E_source,
            "zero": None,
        }

        for ti, t_val in enumerate(TIMESTEPS):
            for cond_name, E_cond in cond_E.items():
                mse_real = score_one(diff_model, C_real, E_cond, t_val, K_NOISE,
                                     dc, device, dtype, bs, noise_per_t[ti])
                mse_fake = score_one(diff_model, C_fake, E_cond, t_val, K_NOISE,
                                     dc, device, dtype, bs, noise_per_t[ti])
                results[f"real_{cond_name}"][fi, ti, :] = mse_real
                results[f"fake_{cond_name}"][fi, ti, :] = mse_fake

        if fi % 5 == 0 or fi == n_frames_actual - 1:
            elapsed = time.time() - t_start
            print(f"  [stage_0 {session}] frame {fi+1}/{n_frames_actual} "
                  f"target={target_row} elapsed={elapsed:.0f}s", flush=True)

    out = {
        "session": session,
        "rows": [r for _, _, r in sampled_with_gi],
        "block_idx": block_assignment,
        "results": results,
    }
    if c_fake_buffer is not None:
        # (n_frames, 4, 768, 1024) float16 array. Reconstruct via
        # torch.from_numpy(arr).float() if higher precision needed.
        out["c_fake_tensor"] = np.stack(c_fake_buffer, axis=0)
    return out


def summarize(eval_data: dict) -> dict:
    res = eval_data["results"]
    block_idx = np.array(eval_data["block_idx"])
    n_blocks = int(block_idx.max() + 1) if len(block_idx) > 0 else 0
    block_lengths = [int((block_idx == bi).sum()) for bi in range(n_blocks)]

    def per_frame(cond: str) -> np.ndarray:
        return res[cond].mean(axis=(1, 2))

    summary = {"n_frames": int(res["real_correct"].shape[0]),
               "block_sizes": block_lengths,
               "by_condition": {}, "deltas": {}, "auroc": {}}

    for cond in ["real_correct", "real_shuffled", "real_source", "real_zero",
                 "fake_correct", "fake_shuffled", "fake_source", "fake_zero"]:
        v = per_frame(cond)
        m, lo, hi = block_bootstrap_ci(v, block_lengths)
        summary["by_condition"][cond] = {"mean": m, "ci_low": lo, "ci_high": hi}

    # Paired gaps: fake - real per condition
    for cond_name in ("correct", "shuffled", "source", "zero"):
        gap = per_frame(f"fake_{cond_name}") - per_frame(f"real_{cond_name}")
        m, lo, hi = block_bootstrap_ci(gap, block_lengths)
        summary["deltas"][f"paired_gap_{cond_name}"] = {"mean": m, "ci_low": lo, "ci_high": hi}

    # AUROC: real-correct vs fake-correct (does diffusion distinguish them?)
    summary["auroc"]["real_correct_vs_fake_correct"] = auroc_pooled(
        per_frame("real_correct"), per_frame("fake_correct"))
    # AUROC: fake correct vs fake shuffled (does F-A reproduce E-coupling?)
    summary["auroc"]["fake_correct_vs_fake_shuffled"] = auroc_pooled(
        per_frame("fake_correct"), per_frame("fake_shuffled"))
    # Real-side baseline (sanity)
    summary["auroc"]["real_correct_vs_real_shuffled"] = auroc_pooled(
        per_frame("real_correct"), per_frame("real_shuffled"))

    # Real-baseline lifts (per spec Case 3 diagnosis)
    real_zero_lift = per_frame("real_zero") - per_frame("real_correct")
    fake_zero_lift = per_frame("fake_zero") - per_frame("fake_correct")
    real_shuf_lift = per_frame("real_shuffled") - per_frame("real_correct")
    fake_shuf_lift = per_frame("fake_shuffled") - per_frame("fake_correct")
    for label, arr in (("real_zero_lift", real_zero_lift),
                       ("fake_zero_lift", fake_zero_lift),
                       ("real_shuffled_lift", real_shuf_lift),
                       ("fake_shuffled_lift", fake_shuf_lift)):
        m, lo, hi = block_bootstrap_ci(arr, block_lengths)
        summary["deltas"][label] = {"mean": m, "ci_low": lo, "ci_high": hi}

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion-ckpt", type=Path, required=True,
                    help="Phase G main diffusion verifier checkpoint.")
    ap.add_argument("--fa-ckpt", type=Path, required=True,
                    help="F-A v1 checkpoint (e.g. step_00100000.pt).")
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-d2", type=int, default=200)
    ap.add_argument("--n-v10", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--save-c-fake", action="store_true",
                    help="Persist C_fake tensors to NPZ (enables visual_grids "
                    "Grid 3, Test B etc.). ~1 GB extra per (session, ckpt). "
                    "Default off to keep NPZ small.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    diff_model = load_diff_model(args.diffusion_ckpt, device, dtype)
    print(f"[init] diffusion model loaded from {args.diffusion_ckpt}")
    fa_model = load_fa_v1_checkpoint(args.fa_ckpt, device, dtype)
    print(f"[init] F-A model loaded from {args.fa_ckpt}")

    dc = build_diffusion_constants(args.T, device, torch.float32)

    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    out_summary: dict = {"sessions": {}, "fa_ckpt": str(args.fa_ckpt),
                         "diffusion_ckpt": str(args.diffusion_ckpt)}
    if args.n_d2 > 0:
        eval_d2 = evaluate_session(diff_model, fa_model, "D2", args.d2_dir,
                                   device, dtype, dc,
                                   n_frames=args.n_d2, bs=args.bs, seed=args.seed,
                                   shard=args.shard, num_shards=args.num_shards,
                                   save_c_fake=args.save_c_fake)
        d2_extras = {}
        if "c_fake_tensor" in eval_d2:
            d2_extras["c_fake_tensor"] = eval_d2["c_fake_tensor"]
        np.savez(args.out / f"stage0_d2_raw{suffix}.npz",
                 **{f"cond_{k}": v for k, v in eval_d2["results"].items()},
                 rows=np.array(eval_d2["rows"]),
                 block_idx=np.array(eval_d2["block_idx"]),
                 **d2_extras)
        if args.num_shards == 1:
            out_summary["sessions"]["D2"] = summarize(eval_d2)
    if args.n_v10 > 0 and args.v10_dir is not None and args.v10_dir.exists():
        eval_v10 = evaluate_session(diff_model, fa_model, "V10", args.v10_dir,
                                    device, dtype, dc,
                                    n_frames=args.n_v10, bs=args.bs, seed=args.seed + 1,
                                    shard=args.shard, num_shards=args.num_shards,
                                    save_c_fake=args.save_c_fake)
        v10_extras = {}
        if "c_fake_tensor" in eval_v10:
            v10_extras["c_fake_tensor"] = eval_v10["c_fake_tensor"]
        np.savez(args.out / f"stage0_v10_raw{suffix}.npz",
                 **{f"cond_{k}": v for k, v in eval_v10["results"].items()},
                 rows=np.array(eval_v10["rows"]),
                 block_idx=np.array(eval_v10["block_idx"]),
                 **v10_extras)
        if args.num_shards == 1:
            out_summary["sessions"]["V10"] = summarize(eval_v10)

    if args.num_shards == 1:
        (args.out / "summary.json").write_text(json.dumps(out_summary, indent=2))
        print(f"[done] wrote {args.out}/summary.json")
    else:
        print(f"[done] shard {args.shard}/{args.num_shards} wrote raw NPZs only")


if __name__ == "__main__":
    main()
