"""Closure Task 4: Exact emitted-RGB → XOF upper bound.

Trains a Tiny + FPN + per-octave heads model that takes the EXACT pre-projection
emission RGB tile (rendered from chain log s_next_hex) and predicts XOF byte
octaves. No projector, no camera, no body — just generator → RGB → XOF.

If oct0/oct1 fail here too, the generator design itself is broken (bytes
aren't identifiable from emitted RGB regardless of optics). If they recover
well, the optical channel is the bottleneck.

Architecture: same as A1 (Tiny + CoordAwareFPN + OctaveHeads, all 4 octaves)
EXCEPT input is 3-channel RGB (no CFA stem adaptation).

Splits: same as A1 — D2 train [0, 4592), val_calib [4792, 5392),
val_report [5392, 5992).

Run:
  python scripts/closure/closure_task4_exact_rgb.py \
    --d2-dir /path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2 \
    --out-dir experiments/closure_package/exact_rgb_xof \
    --max-steps 5000 --bs 4 --bf16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from blake3 import blake3
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from data.xof_generation import (  # noqa: E402
    OCTAVE_BYTES_PER_CHANNEL,
    OCTAVE_SHAPES,
    SEED_DOMAIN_TAG_R,
    SEED_DOMAIN_TAG_G,
    SEED_DOMAIN_TAG_B,
    TOTAL_BYTES_PER_CHANNEL,
    expand_seed_to_octaves,
)
from eval.candidate_ranking import (  # noqa: E402
    assemble_candidate_set,
    compute_tau95,
    per_family_window_fmr,
    score_candidate_xof,
    score_variants_for_experiment,
)
from data.packed_cfa_dataset import xof_octaves_centered_from_hex  # noqa: E402
from losses.huber_xof import huber_xof_loss  # noqa: E402
from models.coord_aware_fpn import CoordAwareFPN  # noqa: E402
from models.octave_heads import OctaveHeads  # noqa: E402

TILE_H, TILE_W = 1080, 1920
NUM_OCTAVES = 4
ENC_DIMS_TINY = (96, 192, 384, 768)


def render_emission_rgb(s_next_hex: str) -> torch.Tensor:
    """Render exact emission RGB tile from chain s_next_hex.

    Faithful to truth_beam.protocol.tile_gpu.gen_rgb_tile_cuda:
      per-octave: stream bytes → reshape (3, gh, gw) → center to [-128, 127]
                  → bilinear upsample to (3, TILE_H, TILE_W) → divide by 2^o
                  → accumulate.
      final = accumulator + 128 ∈ [0, 255] (clamped), / 255 → float [0,1].

    Bit-exact int math is not needed for this self-recovery test; float
    bilinear matches up to ~1 LSB which is well below model precision.
    Returns (3, TILE_H, TILE_W) float in [0, 1].
    """
    s_next = bytes.fromhex(s_next_hex)
    seed_r = blake3(SEED_DOMAIN_TAG_R + s_next).digest()
    seed_g = blake3(SEED_DOMAIN_TAG_G + s_next).digest()
    seed_b = blake3(SEED_DOMAIN_TAG_B + s_next).digest()

    accum = torch.zeros((3, TILE_H, TILE_W), dtype=torch.float32)
    for ch_idx, seed in enumerate((seed_r, seed_g, seed_b)):
        raw = blake3(seed).digest(length=TOTAL_BYTES_PER_CHANNEL)
        arr = np.frombuffer(raw, dtype=np.uint8)
        pos = 0
        for o, ((gh, gw), nbytes) in enumerate(zip(OCTAVE_SHAPES, OCTAVE_BYTES_PER_CHANNEL)):
            slab = torch.from_numpy(arr[pos:pos + nbytes].reshape(gh, gw).astype(np.int16).copy())
            pos += nbytes
            slab = slab.float() - 128.0   # center to [-128, 127]
            up = F.interpolate(slab.unsqueeze(0).unsqueeze(0),
                                size=(TILE_H, TILE_W), mode="bilinear", align_corners=True)
            accum[ch_idx] += up.squeeze() / (2 ** o)
    rgb = (accum + 128.0).clamp(0, 255) / 255.0
    return rgb


class ExactRGBDataset(Dataset):
    def __init__(self, chain: dict[int, str], rows: list[int]):
        self.chain = chain
        self.rows = [r for r in rows if r in chain and (r + 1) in chain]
        if len(self.rows) == 0:
            raise ValueError("no rows with valid chain entry + next-row XOF")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = self.rows[i]
        # Emission for row t comes from chain[t+1] (target_chain_row offset=+1)
        target_chain_row = t + 1
        rgb = render_emission_rgb(self.chain[target_chain_row])
        # Centered XOF target for the same target_chain_row
        xof_octs = xof_octaves_centered_from_hex(self.chain[target_chain_row])
        return {
            "rgb": rgb,
            "t": t,
            "target_chain_row": target_chain_row,
            "xof_oct0": xof_octs[0],
            "xof_oct1": xof_octs[1],
            "xof_oct2": xof_octs[2],
            "xof_oct3": xof_octs[3],
        }


class ExactRGBDecoder(nn.Module):
    def __init__(self, fpn_out_channels=256, head_hidden=128):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3), in_chans=3,
        )
        self.fpn = CoordAwareFPN(ENC_DIMS_TINY, out_channels=fpn_out_channels)
        self.octave_heads = OctaveHeads(
            in_channels=self.fpn.out_channels_with_coords,
            hidden=head_hidden, enabled_octaves=(0, 1, 2, 3),
        )

    def forward(self, x):
        feats = self.encoder(x)
        fpn = self.fpn(feats)
        return self.octave_heads(fpn)


def collate(batch):
    rgb = torch.stack([b["rgb"] for b in batch])
    out = {
        "rgb": rgb,
        "t": torch.tensor([b["t"] for b in batch]),
        "target_chain_row": torch.tensor([b["target_chain_row"] for b in batch]),
    }
    for i in range(4):
        out[f"xof_oct{i}"] = torch.stack([b[f"xof_oct{i}"] for b in batch])
    return out


@torch.no_grad()
def evaluate(model, val_ds, device, autocast_dtype, n_eval=64):
    model.eval()
    n = min(n_eval, len(val_ds))
    matched_scores = {v: [] for v in ("oct0_only", "oct0_plus_oct1",
                                       "oct0_plus_oct1_plus_oct2", "all_octaves")}
    bit_recovery = {i: [] for i in range(4)}
    for i in range(n):
        sample = val_ds[i]
        x = sample["rgb"].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            preds = model(x)
        preds_cpu = [p.float().squeeze(0).cpu() if p is not None else None for p in preds]
        true_octs = [sample[f"xof_oct{i}"] for i in range(4)]
        for v in matched_scores:
            aligned = []
            for j in range(4):
                if preds_cpu[j] is None:
                    aligned.append(None)
                else:
                    aligned.append(true_octs[j])
            matched_scores[v].append(score_candidate_xof(preds_cpu, aligned, v))
        # Per-octave bit recovery: convert to bytes (rounded), compare exact match
        for j in range(4):
            if preds_cpu[j] is None:
                continue
            pred_bytes = (preds_cpu[j] * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            true_bytes = (true_octs[j] * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            bit_recovery[j].append(((pred_bytes == true_bytes).float().mean()).item())
    summary = {
        "matched_score": {v: float(np.mean(matched_scores[v])) for v in matched_scores},
        "byte_match_per_octave": {f"oct{j}": float(np.mean(bit_recovery[j])) if bit_recovery[j] else None
                                   for j in range(4)},
        "n_eval": n,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=4592)
    ap.add_argument("--val-calib-start", type=int, default=4792)
    ap.add_argument("--val-calib-end", type=int, default=5392)
    ap.add_argument("--val-report-start", type=int, default=5392)
    ap.add_argument("--val-report-end", type=int, default=5992)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    chain = load_chain_log(args.d2_dir / "chain_log.csv")
    print(f"[init] D2 chain log loaded: {len(chain)} rows", flush=True)

    train_ds = ExactRGBDataset(chain, list(range(args.train_start, args.train_end)))
    val_calib = ExactRGBDataset(chain, list(range(args.val_calib_start, args.val_calib_end)))
    val_report = ExactRGBDataset(chain, list(range(args.val_report_start, args.val_report_end)))
    print(f"[init] train={len(train_ds)} val_calib={len(val_calib)} val_report={len(val_report)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = ExactRGBDecoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params/1e6:.1f}M  bf16={args.bf16}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=max(args.warmup, 1)),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.max_steps - args.warmup, 1))],
        milestones=[args.warmup],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    history = []
    t0 = time.time()
    step = 0
    train_iter = iter(train_loader)
    best_loss = math.inf
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["rgb"].to(device, non_blocking=True)
        targets = {f"xof_oct{i}": batch[f"xof_oct{i}"].to(device, non_blocking=True) for i in range(4)}

        model.train()
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            preds = model(x)
            loss, parts = huber_xof_loss(preds, [targets[f"xof_oct{i}"] for i in range(4)])

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if step % 10 == 0:
            print(f"[step {step}] loss={parts['total']:.4f} lr={opt.param_groups[0]['lr']:.2e} {parts}", flush=True)
            history.append({"step": step, **parts})

        if parts["total"] < best_loss:
            best_loss = parts["total"]
            ckpt_path = args.out_dir / "checkpoints" / "best_by_loss.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"model": model.state_dict(), "step": step,
                        "loss": parts["total"]}, tmp)
            tmp.replace(ckpt_path)

        if step > 0 and step % args.val_every == 0:
            summary = evaluate(model, val_calib, device, autocast_dtype, n_eval=64)
            print(f"[val step {step}] {summary}", flush=True)
            (args.out_dir / "val_history.jsonl").open("a").write(
                json.dumps({"step": step, **summary}) + "\n")

        step += 1

    final_ckpt = args.out_dir / "checkpoints" / "final_step.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step}, final_ckpt)

    # Final eval on report half
    final = evaluate(model, val_report, device, autocast_dtype, n_eval=min(len(val_report), 256))
    print(f"[final] report-half: {final}", flush=True)
    (args.out_dir / "final_report.json").write_text(json.dumps({
        "report": final,
        "max_steps": args.max_steps,
        "best_loss": best_loss,
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2))
    (args.out_dir / "loss_history.jsonl").write_text(
        "\n".join(json.dumps(h) for h in history) + "\n")
    print(f"[done] elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
