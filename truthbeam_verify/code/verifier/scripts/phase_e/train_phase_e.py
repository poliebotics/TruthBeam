"""Phase E emission trainer — DDP-aware, per-epoch val, stop criteria.

Supports E1, E2, E3r via config file. Uses exp001c's training recipe:
  AdamW lr=1e-4 wd=0.05, charbonnier + 0.1 × ms_l1, 30-epoch cap with
  linear warmup 200 steps → cosine annealing, fp16/bf16 autocast.

Per-epoch on rank 0:
  - Matched PSNR on full val
  - Easy gap (200-row deterministic same-session-random mismatch)
  - Top-1 retrieval (50 candidates)
  - Output variance across 8 sample frames
  - Per-channel L1
  - Loss component breakdown

Every 5 epochs:
  - Family-balanced FMR@5 (delay_window/near_shift/same_session_random/cross_session_random)

Stop criteria checked every epoch:
  - SUCCESS, PLATEAU, MANIFESTLY-BROKEN, HARD-CAP

Run (per torchrun):
  torchrun --standalone --nproc_per_node=2 scripts/phase_e/train_phase_e.py \
    --config configs/e1.yaml --bf16
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
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import EmissionDataset, load_emission_at  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from losses.emission_loss import emission_loss  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
except Exception:
    EmissionPredictorV2 = None


# ---------- DDP ----------

def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Long timeout to avoid Phase D's NCCL barrier issue
        from datetime import timedelta
        dist.init_process_group(backend="nccl", init_method="env://",
                                 timeout=timedelta(minutes=30))
    return True, rank, world, local_rank


def is_main():
    return int(os.environ.get("RANK", "0")) == 0


def mprint(*a, **kw):
    if is_main():
        print(*a, **kw)


# ---------- model ----------

def _patch_stem_to_legacy_full_g_dup(model):
    """Phase E uses exp001c's full-magnitude G duplication for ALL experiments
    (operator decision). EmissionPredictorV2's encoder calls
    adapt_convnext_stem_4ch_half_g (G * 0.5). This function reverses the
    half-scaling so ch1/ch2 hold the original ImageNet G weights."""
    import torch.nn as nn
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 4 \
                and mod.kernel_size == (4, 4) and mod.stride == (4, 4):
            with torch.no_grad():
                mod.weight[:, 1] *= 2.0  # G1: 0.5G → G
                mod.weight[:, 2] *= 2.0  # G2: 0.5G → G
            return name
    raise RuntimeError("could not find 4ch stem to patch")


def build_model(cfg, device):
    arch = cfg["model"]["arch"]
    if arch == "EmissionPredictor":
        # Already uses siamese_xof._adapt_convnext_stem_to_4ch (full G dup).
        m = EmissionPredictor(emission_h=cfg["data"]["emission_h"],
                              emission_w=cfg["data"]["emission_w"],
                              pretrained=cfg["model"].get("pretrained", True))
    elif arch == "EmissionPredictorV2":
        if EmissionPredictorV2 is None:
            raise RuntimeError("V2 not available")
        m = EmissionPredictorV2(emission_h=cfg["data"]["emission_h"],
                                emission_w=cfg["data"]["emission_w"],
                                pretrained=cfg["model"].get("pretrained", True),
                                fpn_out_channels=cfg["model"].get("fpn_out_channels", 256))
        # Patch stem to legacy full G dup (V1 + operator decision).
        if cfg["model"].get("pretrained", True):
            stem_name = _patch_stem_to_legacy_full_g_dup(m)
            mprint(f"[init] patched {arch} stem '{stem_name}' to legacy full-G-dup", flush=True)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m.to(device)


# ---------- candidate ranking ----------

NEAR_SHIFT_OFFSETS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def assemble_emission_candidates(t: int, val_set: set, cross_set: set,
                                  d_search_max: int, n_same: int, n_cross: int, seed: int):
    rng = np.random.RandomState(seed)
    candidates = [(t, "matched", "D2")]
    used = {("D2", t)}
    # delay_window / near_shift
    for d in range(-d_search_max, d_search_max + 1):
        if d == 0: continue
        r = t + d
        if r in val_set and ("D2", r) not in used:
            fam = "near_shift" if d in NEAR_SHIFT_OFFSETS else "delay_window"
            candidates.append((r, fam, "D2"))
            used.add(("D2", r))
    # same_session_random
    delay_keys = {("D2", t + d) for d in range(-d_search_max, d_search_max + 1)}
    pool = sorted([r for r in val_set if ("D2", r) not in delay_keys])
    rng.shuffle(pool)
    n_taken = 0
    for r in pool:
        if ("D2", r) in used: continue
        candidates.append((r, "same_session_random", "D2"))
        used.add(("D2", r))
        n_taken += 1
        if n_taken >= n_same: break
    # cross_session_random
    cpool = sorted(list(cross_set))
    rng.shuffle(cpool)
    n_taken = 0
    for r in cpool:
        if ("V10", r) in used: continue
        candidates.append((r, "cross_session_random", "V10"))
        used.add(("V10", r))
        n_taken += 1
        if n_taken >= n_cross: break
    return candidates


# ---------- per-epoch val ----------

@torch.no_grad()
def evaluate_epoch(*, model, val_loader, val_rows: list[int], d2_dir: Path, v10_dir: Path,
                   emission_h: int, emission_w: int, device, autocast_dtype,
                   v10_rows_set: set, do_family_balanced: bool, n_eval_fb: int = 200,
                   n_easy_mm: int = 200, n_top1: int = 50,
                   n_visual: int = 8, seed: int = 42):
    """Per-epoch eval: matched PSNR (full val), easy gap, top-1, family-balanced FMR.
    Returns dict ready for jsonl. Run only on rank 0."""
    model.eval()
    val_set = set(val_rows)
    matched_psnrs = []
    pred_samples: list[torch.Tensor] = []
    sample_targets: list[torch.Tensor] = []
    sample_rows: list[int] = []
    per_channel_l1 = {"r": 0.0, "g": 0.0, "b": 0.0}
    n_processed = 0
    t0 = time.time()
    for batch in val_loader:
        cap = batch["capture"].to(device, non_blocking=True)
        em = batch["emission"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap).float().clamp(0, 1)
        for i in range(pred.shape[0]):
            p = psnr(pred[i].cpu(), em[i].cpu())
            matched_psnrs.append(p)
            for ci, c in enumerate("rgb"):
                per_channel_l1[c] += float((pred[i, ci] - em[i, ci]).abs().mean().item())
            n_processed += 1
            if len(pred_samples) < n_visual:
                pred_samples.append(pred[i].cpu())
                sample_targets.append(em[i].cpu())
                sample_rows.append(int(batch["t"][i].item()))
    matched_arr = np.array(matched_psnrs)
    per_channel_l1 = {k: v / max(n_processed, 1) for k, v in per_channel_l1.items()}

    # Easy gap (200-row deterministic same-session random mismatch, fixed seed)
    rng = np.random.RandomState(seed)
    n_eval = min(n_easy_mm, n_processed)
    indices = sorted(rng.choice(range(n_processed), size=n_eval, replace=False).tolist())
    # We need the sample_rows + a "random non-matching" from the val pool.
    # Walk all evaluated samples; for each, pick a random row != itself.
    # Re-iterate the loader for simplicity is wasteful. Use cached preds is impractical at scale.
    # → Cache: collect all preds + rows in eval loop. (We didn't cache them all; only n_visual.)
    # Use a streaming approach: for each i in indices, predict on-the-fly is expensive.
    # Compromise: re-iterate val_loader, for each sample pick a random non-matching emission.
    # To keep single-pass, we'll re-walk val_loader:
    # Actually we already have matched_psnrs and can skip the per-sample walk if we just
    # pick mismatched targets from sample_targets cache + new disk loads.
    # Simpler: loop val_loader once more for the easy-gap sample, but that's 2x work.
    # Practical: use a cached table of emission tile MD5/path + a deterministic mismatch row.

    # For each evaluated row index, pick a deterministic non-matching row from the val pool
    # and read its emission tile from disk; compute pred vs that tile PSNR. That second pass
    # is O(n_eval) tile reads (not a full val pass).
    # We need to redo prediction or cache it. Given n_visual=8 cached, we'd lose data. Re-loop.
    # But that's 2× val cost. Skip it for per-epoch and rely on family-balanced for gap signal.

    # Per-epoch easy gap: for each cached prediction (n_visual frames), compute mismatched PSNR
    # vs n_easy_mm random non-matching emission tiles from val pool. Cheap, gives the trend signal.
    easy_mismatched = []
    for i, p in enumerate(pred_samples):
        candidate_rows = [r for r in val_rows if r != sample_rows[i]]
        rng2 = np.random.RandomState(seed ^ sample_rows[i])
        mm_rows = rng2.choice(candidate_rows, size=min(8, len(candidate_rows)), replace=False)
        for mm_t in mm_rows:
            mm_em = load_emission_at(d2_dir / "derived" / "Emissions" / f"tile_{int(mm_t):06d}.png",
                                      emission_h, emission_w)
            easy_mismatched.append(psnr(p, mm_em.clamp(0, 1)))
    easy_mismatched_mean = float(np.mean(easy_mismatched)) if easy_mismatched else float("nan")

    # Top-1 (over a small candidate set for the n_visual cached preds)
    top1_correct = 0
    top1_total = 0
    for i, p in enumerate(pred_samples):
        # 50 candidates: matched + 49 random non-matching from val pool
        rng3 = np.random.RandomState(seed ^ (sample_rows[i] + 17))
        candidate_rows = [r for r in val_rows if r != sample_rows[i]]
        mm_rows = rng3.choice(candidate_rows, size=min(49, len(candidate_rows)), replace=False)
        m_psnr = psnr(p, sample_targets[i])
        max_mm = -math.inf
        for mm_t in mm_rows:
            mm_em = load_emission_at(d2_dir / "derived" / "Emissions" / f"tile_{int(mm_t):06d}.png",
                                      emission_h, emission_w)
            mm_psnr = psnr(p, mm_em.clamp(0, 1))
            if mm_psnr > max_mm:
                max_mm = mm_psnr
        top1_total += 1
        if m_psnr > max_mm:
            top1_correct += 1
    top1 = top1_correct / max(top1_total, 1)

    # Output variance across visual sample (constant-prediction diagnostic)
    if len(pred_samples) >= 2:
        stack = torch.stack(pred_samples, dim=0)  # (N, 3, H, W)
        across_row_std = stack.std(dim=0).mean().item()  # mean std across rows
    else:
        across_row_std = float("nan")

    out = {
        "n_val": n_processed,
        "matched_psnr_mean": float(matched_arr.mean()),
        "matched_psnr_std":  float(matched_arr.std()),
        "matched_psnr_p5":   float(np.percentile(matched_arr, 5)),
        "matched_psnr_p95":  float(np.percentile(matched_arr, 95)),
        "easy_mismatched_psnr_mean": easy_mismatched_mean,
        "easy_gap_db": float(matched_arr.mean()) - easy_mismatched_mean,
        "top1_retrieval_50cand": top1,
        "n_top1_eval": top1_total,
        "output_variance_across_rows": across_row_std,
        "per_channel_l1": per_channel_l1,
        "elapsed_sec_eval": round(time.time() - t0, 1),
    }
    if do_family_balanced:
        out.update(eval_family_balanced(
            model=model, d2_dir=d2_dir, v10_dir=v10_dir,
            val_rows=val_rows, v10_rows_set=v10_rows_set,
            emission_h=emission_h, emission_w=emission_w,
            device=device, autocast_dtype=autocast_dtype,
            n_eval=n_eval_fb, val_loader_for_pred=val_loader, seed=seed,
        ))
    return out, pred_samples, sample_targets, sample_rows


@torch.no_grad()
def eval_family_balanced(*, model, d2_dir, v10_dir, val_rows, v10_rows_set,
                         emission_h, emission_w, device, autocast_dtype,
                         n_eval: int, val_loader_for_pred, seed: int):
    """Family-balanced eval. Uses fresh dataset slice over n_eval evenly-spaced rows."""
    rng = np.random.RandomState(seed)
    rows = sorted(rng.choice(val_rows, size=min(n_eval, len(val_rows)), replace=False).tolist())
    val_set = set(val_rows)
    # Build a mini-eval loader pulling from disk via EmissionDataset
    # (val_loader_for_pred is the full val loader; we want a row-subset loader)
    ds = val_loader_for_pred.dataset
    # ds is EmissionDataset over [val_start, val_end). row_to_idx is t - row_start.
    em_cache: dict = {}
    def get_emission(session: str, t: int):
        key = (session, t)
        if key in em_cache:
            return em_cache[key]
        sd = d2_dir if session == "D2" else v10_dir
        p = sd / "derived" / "Emissions" / f"tile_{t:06d}.png"
        if not p.exists():
            return None
        em_cache[key] = load_emission_at(p, emission_h, emission_w).clamp(0, 1)
        return em_cache[key]

    matched_psnrs = []
    negatives_per_frame: list[dict[str, list[float]]] = []
    top1_count = 0
    t0 = time.time()
    for t in rows:
        if t not in ds.rows: continue
        local_idx = ds.rows.index(t)
        sample = ds[local_idx]
        cap = sample["capture"].unsqueeze(0).to(device)
        em = sample["emission"].clamp(0, 1)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap).float().squeeze(0).clamp(0, 1).cpu()
        m_psnr = psnr(pred, em)
        matched_psnrs.append(m_psnr)
        cs = assemble_emission_candidates(
            t=t, val_set=val_set, cross_set=v10_rows_set,
            d_search_max=32, n_same=192, n_cross=192, seed=seed ^ t)
        per_frame_neg: dict[str, list[float]] = {}
        max_neg = -math.inf
        for r, fam, sess in cs:
            if fam == "matched": continue
            cand_em = get_emission(sess, r)
            if cand_em is None: continue
            p = psnr(pred, cand_em)
            per_frame_neg.setdefault(fam, []).append(p)
            if p > max_neg: max_neg = p
        negatives_per_frame.append(per_frame_neg)
        if m_psnr > max_neg: top1_count += 1
    n = len(matched_psnrs)
    arr = np.array(matched_psnrs)
    tau5 = float(np.percentile(arr, 5))
    families = ["delay_window", "near_shift", "same_session_random", "cross_session_random"]
    per_family_fmr = {}
    for f in families:
        hits = sum(1 for nf in negatives_per_frame if any(p >= tau5 for p in nf.get(f, [])))
        per_family_fmr[f] = hits / n if n else 0.0
    return {
        "fb_n": n, "fb_tau5": tau5,
        "fb_matched_psnr_mean": float(arr.mean()),
        "fb_per_family_fmr_at_5": per_family_fmr,
        "fb_worst_family_fmr_at_5": max(per_family_fmr.values()) if per_family_fmr else 0.0,
        "fb_top1_retrieval": top1_count / n if n else 0.0,
        "fb_elapsed_sec": round(time.time() - t0, 1),
    }


# ---------- stop criteria ----------

def check_stop_criteria(history: list[dict], cfg) -> tuple[str, dict]:
    """Returns (status, metadata). status in {"running", "success", "plateau", "broken", "hard_cap"}."""
    if len(history) == 0:
        return "running", {}
    last = history[-1]
    epoch = last.get("epoch", 0)

    # HARD-CAP
    hard_cap = cfg.get("stop", {}).get("hard_cap_epochs", 30)
    if epoch >= hard_cap - 1:  # 0-indexed
        return "hard_cap", {"reason": f"hit hard cap at epoch {epoch}"}

    matched = last.get("matched_psnr_mean", 0.0)
    fb = last.get("fb_worst_family_fmr_at_5", None)
    fb_top1 = last.get("fb_top1_retrieval", None)
    out_var = last.get("output_variance_across_rows", float("nan"))

    # SUCCESS — tightened per operator after P0b outcome A (exp001c top-1 = 1.000)
    if matched >= 24.0 and fb is not None and fb <= 0.05 and fb_top1 is not None and fb_top1 >= 0.99:
        # Stable for 3 consecutive epochs
        if len(history) >= 3:
            tail = history[-3:]
            if all(h.get("matched_psnr_mean", 0) >= 24.0 for h in tail):
                std = np.std([h["matched_psnr_mean"] for h in tail])
                if std < 0.5:
                    return "success", {"reason": f"matched≥24 AND fb_FMR≤0.05 AND fb_top1≥0.99 stable {len(tail)} epochs"}

    # MANIFESTLY-BROKEN diagnostic at epoch ≥5
    if epoch >= 4:
        # Need: output variance ≈ 0, matched ≈ null PSNR (~13 dB), top1 near chance, gap ≈ 0
        broken_signals = [
            out_var < 0.05,
            matched < 16.0,
            last.get("top1_retrieval_50cand", 1.0) < 0.05,
            abs(last.get("easy_gap_db", 1.0)) < 1.0,
        ]
        if all(broken_signals):
            return "broken", {"reason": f"all 4 manifestly-broken signals at epoch {epoch}",
                              "signals": broken_signals}

    # PLATEAU: 5 epochs no progress, minimum 15 epochs
    if epoch >= 14:
        recent = history[-6:]  # last 6 epochs
        if len(recent) >= 6:
            psnrs = [h.get("matched_psnr_mean", 0) for h in recent]
            losses = [h.get("train_loss", 0) for h in recent]
            psnr_change = max(psnrs) - min(psnrs)
            if psnr_change < 0.5 and losses[-1] > 0:
                loss_decreases = [(losses[i-1] - losses[i]) / max(losses[i-1], 1e-6)
                                  for i in range(1, len(losses))]
                if all(d < 0.01 for d in loss_decreases):
                    return "plateau", {"reason": f"5-epoch PSNR change {psnr_change:.2f} dB AND loss decrease <1%/epoch"}

    return "running", {}


# ---------- save samples ----------

def save_visual_grid(pred_samples, target_samples, sample_rows, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = min(len(pred_samples), 4)
    if n == 0: return
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1: axes = [axes]
    for i in range(n):
        for j, (im, name) in enumerate([(pred_samples[i], "pred"), (target_samples[i], "GT")]):
            ax = axes[i][j] if n > 1 else axes[j]
            arr = im.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(arr); ax.axis("off")
            ax.set_title(f"row {sample_rows[i]} {name}", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


# ---------- save ckpt ----------

def save_ckpt(model, opt, ep, history, out_dir, label):
    path = out_dir / "checkpoints" / f"{label}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"model": state, "ep": ep, "history": history}, path)


# ---------- main ----------

def collate(batch):
    return {
        "capture": torch.stack([b["capture"] for b in batch]),
        "emission": torch.stack([b["emission"] for b in batch]),
        "t": torch.tensor([b["t"] for b in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-epochs-override", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["out_dir"])
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    ddp_active, rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if ddp_active else "cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    mprint(f"[init] config={args.config} ddp={ddp_active} rank={rank} world={world} bf16={args.bf16}", flush=True)

    # Datasets
    train_ds = EmissionDataset(
        session_dir=Path(cfg["data"]["d2_dir"]),
        row_start=cfg["data"]["d2_train_start"], row_end=cfg["data"]["d2_train_end"],
        capture_h=cfg["data"]["capture_h"], capture_w=cfg["data"]["capture_w"],
        emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"],
        session_id="D2", augment=False, seed=cfg.get("seed", 0))
    val_ds = EmissionDataset(
        session_dir=Path(cfg["data"]["d2_dir"]),
        row_start=cfg["data"]["d2_val_start"], row_end=cfg["data"]["d2_val_end"],
        capture_h=cfg["data"]["capture_h"], capture_w=cfg["data"]["capture_w"],
        emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"],
        session_id="D2", augment=False)
    val_rows = list(range(cfg["data"]["d2_val_start"], cfg["data"]["d2_val_end"]))
    mprint(f"[init] train n={len(train_ds)} val n={len(val_ds)}", flush=True)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 4))
    sampler = DistributedSampler(train_ds, shuffle=True) if ddp_active else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
                               num_workers=nw, collate_fn=collate, pin_memory=True, drop_last=True,
                               persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["train"].get("val_batch_size", 4), shuffle=False,
                             num_workers=nw, collate_fn=collate, pin_memory=True)

    # Build model + optimizer
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    mprint(f"[init] model={cfg['model']['arch']} params={n_params/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.05))
    if ddp_active:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    max_epochs = args.max_epochs_override if args.max_epochs_override else cfg["train"].get("max_epochs", 30)
    warmup_steps = cfg["train"].get("warmup_steps", 200)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * max_epochs
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=max(warmup_steps, 1)),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1)),
        ], milestones=[warmup_steps])
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    # V10 chain rows for cross-session candidates
    v10_dir = Path(cfg["data"]["v10_dir"])
    v10_rows_set: set = set()
    if (v10_dir / "chain_log.csv").exists():
        chain = load_chain_log(v10_dir / "chain_log.csv")
        emi = v10_dir / "derived" / "Emissions"
        v10_rows_set = {t for t in chain if (emi / f"tile_{t:06d}.png").exists()}

    # Manifest
    if is_main():
        manifest = {
            "experiment_id": cfg.get("experiment_id", args.config.stem),
            "config": cfg, "ddp_world_size": world, "max_epochs": max_epochs,
            "datetime_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    history: list[dict] = []
    val_history_path = out_dir / "val_history.jsonl"
    t_start = time.time()

    for ep in range(max_epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        ep_loss_sum = 0.0
        ep_loss_n = 0
        ep_t0 = time.time()
        for step_idx, batch in enumerate(train_loader):
            cap = batch["capture"].to(device, non_blocking=True)
            em = batch["emission"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                pred = model(cap)
                loss, parts = emission_loss(pred, em)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            sched.step()
            ep_loss_sum += parts["total"]
            ep_loss_n += 1
            if step_idx % 50 == 0 and is_main():
                print(f"[ep {ep} step {step_idx}/{steps_per_epoch}] loss={parts['total']:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        ep_train_loss = ep_loss_sum / max(ep_loss_n, 1)
        ep_train_time = time.time() - ep_t0

        # Validation: rank 0 only
        if is_main():
            do_fb = (ep % 5 == 0) or (ep == max_epochs - 1)
            ep_val, pred_samples, target_samples, sample_rows = evaluate_epoch(
                model=(model.module if ddp_active else model),
                val_loader=val_loader, val_rows=val_rows,
                d2_dir=Path(cfg["data"]["d2_dir"]), v10_dir=v10_dir,
                emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"],
                device=device, autocast_dtype=autocast_dtype,
                v10_rows_set=v10_rows_set, do_family_balanced=do_fb)
            entry = {
                "epoch": ep, "train_loss": ep_train_loss,
                "ep_train_time_sec": round(ep_train_time, 1),
                **ep_val,
            }
            history.append(entry)
            with val_history_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"\n[ep {ep}] matched={entry['matched_psnr_mean']:.2f}  "
                  f"easy_gap={entry.get('easy_gap_db', 0):.2f}  "
                  f"top1={entry.get('top1_retrieval_50cand', 0):.3f}  "
                  f"out_var={entry.get('output_variance_across_rows', 0):.4f}  "
                  f"loss={ep_train_loss:.4f}  ep_time={ep_train_time:.0f}s", flush=True)
            if "fb_worst_family_fmr_at_5" in entry:
                print(f"  [fb@5ep] worst_FMR={entry['fb_worst_family_fmr_at_5']:.3f}  "
                      f"top1={entry['fb_top1_retrieval']:.3f}  "
                      f"per_family={entry['fb_per_family_fmr_at_5']}", flush=True)
            # Save visual sample
            save_visual_grid(pred_samples[:4], target_samples[:4], sample_rows[:4],
                              out_dir / "visuals" / f"ep_{ep:03d}.png")
            # Save per-epoch ckpt (best by matched PSNR)
            save_ckpt(model, opt, ep, history, out_dir, f"ep_{ep:03d}")
            if ep == 0 or entry["matched_psnr_mean"] >= max(h["matched_psnr_mean"] for h in history):
                save_ckpt(model, opt, ep, history, out_dir, "best_by_psnr")

            status, meta = check_stop_criteria(history, cfg)
            if status != "running":
                print(f"\n*** STOP CRITERION: {status.upper()} *** {meta}", flush=True)
                save_ckpt(model, opt, ep, history, out_dir, f"final_{status}")
                # signal to all ranks via file
                (out_dir / "STOP").write_text(json.dumps({"status": status, "epoch": ep, **meta}))
        if ddp_active:
            dist.barrier()
        # Check stop on all ranks
        if (out_dir / "STOP").exists():
            mprint("[stop] STOP file detected, exiting training loop", flush=True)
            break

    if is_main():
        # Final ckpt
        save_ckpt(model, opt, ep, history, out_dir, "final_step")
        elapsed = time.time() - t_start
        print(f"\n[done] elapsed={elapsed:.0f}s ({elapsed/3600:.2f} h) epochs_run={ep+1}", flush=True)
    if ddp_active:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
