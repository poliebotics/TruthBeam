"""Phase D production trainer (audit-conformant).

Single entry point for all six experiments (exp001h_a0/a1/a2/a4/a6/a7).
Reads a merged config (via `utils.config_loader.load_and_validate`),
refuses to start if the offset diagnostic hasn't set `cfg.offset.value`,
writes a full audit-conformant manifest, then runs:

  - dataset construction with the strict A6 D2-only path or the pooled
    D2+V10 path
  - model construction (XOFDecoderV2 or EmissionPredictorV2)
  - STN training mechanics for A7 (freeze 0..249, unfreeze at 250 with
    separate optimizer param group at 0.1 × encoder lr, identity
    regularizer λ=0.01 every step)
  - SmoothL1 + per-octave weighted loss for XOF (or charbonnier+ms_l1
    for emission)
  - periodic validation with full Window-FMR@95 candidate ranking on the
    CALIBRATION half; checkpoints saved as best_by_loss,
    best_by_calibration_half_window_fmr, final_step
  - final report on the REPORT half (D2-val second half + V10-val second
    half for pooled, or D2-val second half + ALL V10 bins for A6)

Run: `python src/training/train_phase_d.py --config configs/exp001h_a1.yaml`
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler


# ---------- DDP helpers ----------

def init_distributed() -> tuple[bool, int, int, int]:
    """Initialize NCCL DDP if torchrun env vars are present.

    Returns (ddp_active, rank, world_size, local_rank).
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, local_rank


def is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def main_rank_print(*args, **kwargs) -> None:
    if is_main_rank():
        print(*args, **kwargs)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.packed_cfa_dataset import PackedCFADataset, xof_octaves_centered_from_hex  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from eval.candidate_ranking import (  # noqa: E402
    NEAR_SHIFT_OFFSETS,
    assemble_candidate_set,
    compute_tau95,
    per_family_window_fmr,
    score_candidate_xof,
    score_variants_for_experiment,
)
from eval.observability import ObservabilityLogger, compute_row_observability  # noqa: E402
from losses.huber_xof import OCTAVE_WEIGHTS, huber_xof_loss  # noqa: E402
from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
from models.stn import decompose_theta, identity_regularizer  # noqa: E402
from models.xof_decoder_v2 import XOFDecoderV2  # noqa: E402
from preprocessing.normalization import load_stats  # noqa: E402
from utils.config_loader import load_and_validate  # noqa: E402
from utils.run_manifest import compute_code_hash, compute_target_cache_hash, write_run_manifest  # noqa: E402

EMISSION_OCTAVE_TASK = "emission"
XOF_OCTAVE_TASK = "xof"


# -----------------------------------------------------------------------------
# Dataset builders


def _v10_dir(cfg) -> Path:
    return Path(cfg["data"].get("v10_dir", str(Path(cfg["data"]["d2_dir"]).parent / "v10")))


def _strict_d2_only(cfg) -> bool:
    return bool(cfg.get("normalization", {}).get("strict_d2_only", False))


def _build_xof_dataset(
    cfg, session: str, rows: list[int], stats: dict, with_emission: bool = False,
) -> PackedCFADataset:
    session_dir = Path(cfg["data"]["d2_dir"]) if session == "D2" else _v10_dir(cfg)
    cache_root = Path(cfg["data"]["cache_root"])
    return PackedCFADataset(
        session_dir=session_dir,
        rows=rows,
        offset=int(cfg["offset"]["value"]),
        normalization_stats=stats,
        cache_root=cache_root,
        session_id=session,
        black_level=int(cfg["data"].get("black_level", 0)),
        with_xof=not with_emission or "xof" in cfg["data"].get("targets", ()),
        with_emission=with_emission,
        emission_h=int(cfg["data"].get("emission_h", 1080)),
        emission_w=int(cfg["data"].get("emission_w", 1920)),
    )


def _rows_in(cfg, key_start: str, key_end: str) -> list[int]:
    return list(range(int(cfg["data"][key_start]), int(cfg["data"][key_end])))


def build_datasets(cfg: dict, stats_dir: Path) -> dict[str, Any]:
    """Returns a dict with all relevant datasets (some may be None for A6)."""
    is_a6 = _strict_d2_only(cfg)
    d2_stats = load_stats(stats_dir / "d2_train_stats.json")
    if is_a6:
        v10_stats = d2_stats   # A6: D2-train stats applied to V10 too
    else:
        v10_stats = load_stats(stats_dir / "v10_train_stats.json")

    is_emission = cfg["model"]["class"] == "EmissionPredictorV2"
    with_emission = is_emission

    out: dict[str, Any] = {}

    out["d2_train"] = _build_xof_dataset(
        cfg, "D2", _rows_in(cfg, "d2_train_start", "d2_train_end"), d2_stats, with_emission,
    )
    out["d2_val_calib"] = _build_xof_dataset(
        cfg, "D2", _rows_in(cfg, "d2_val_calib_start", "d2_val_calib_end"), d2_stats, with_emission,
    )
    out["d2_val_report"] = _build_xof_dataset(
        cfg, "D2", _rows_in(cfg, "d2_val_report_start", "d2_val_report_end"), d2_stats, with_emission,
    )

    if is_a6:
        # V10 enters as eval-only; v10_stats is D2-train per A6 spec.
        out["v10_train"] = None
        out["v10_val_calib"] = None
        out["v10_val_report"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_val_report_start", "v10_val_report_end"),
            v10_stats, with_emission,
        )
        out["v10_early"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_early_start", "v10_early_end"), v10_stats, with_emission,
        )
        out["v10_mid"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_mid_start", "v10_mid_end"), v10_stats, with_emission,
        )
        out["v10_late"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_late_start", "v10_late_end"), v10_stats, with_emission,
        )
    else:
        out["v10_train"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_train_start", "v10_train_end"), v10_stats, with_emission,
        )
        out["v10_val_calib"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_val_calib_start", "v10_val_calib_end"),
            v10_stats, with_emission,
        )
        out["v10_val_report"] = _build_xof_dataset(
            cfg, "V10", _rows_in(cfg, "v10_val_report_start", "v10_val_report_end"),
            v10_stats, with_emission,
        )
        out["v10_early"] = None  # diagnostics-only for pooled runs
        out["v10_mid"] = None
        out["v10_late"] = out["v10_val_report"]
    return out


# -----------------------------------------------------------------------------
# Model + optimizer builders


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    cls = cfg["model"]["class"]
    if cls == "XOFDecoderV2":
        return XOFDecoderV2(
            encoder_size=cfg["model"].get("encoder_size", "tiny"),
            pretrained=cfg["model"].get("pretrained", True),
            fpn_out_channels=int(cfg["model"].get("fpn_out_channels", 256)),
            head_hidden=int(cfg["model"].get("head_hidden", 128)),
            enabled_octaves=tuple(cfg["model"].get("enabled_octaves", (0, 1, 2, 3))),
            use_stn=bool(cfg["model"].get("use_stn", False)),
        ).to(device)
    if cls == "EmissionPredictorV2":
        return EmissionPredictorV2(
            emission_h=int(cfg["data"].get("emission_h", 1080)),
            emission_w=int(cfg["data"].get("emission_w", 1920)),
            encoder_size=cfg["model"].get("encoder_size", "tiny"),
            pretrained=cfg["model"].get("pretrained", True),
            fpn_out_channels=int(cfg["model"].get("fpn_out_channels", 256)),
        ).to(device)
    raise ValueError(f"unknown model.class={cls!r}")


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """STN params get a separate param group at 0.1× the encoder lr (A7)."""
    base_lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.05))
    has_stn = bool(getattr(model, "stn", None))
    stn_factor = float(cfg.get("stn", {}).get("param_group_lr_factor", 0.1))
    if has_stn:
        stn_params = list(model.stn.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith("stn.")]
        groups = [
            {"params": other_params, "lr": base_lr},
            {"params": stn_params, "lr": base_lr * stn_factor},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": base_lr}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def stn_freeze_set(model: torch.nn.Module, frozen: bool) -> None:
    if getattr(model, "stn", None) is None:
        return
    for p in model.stn.parameters():
        p.requires_grad = not frozen


# -----------------------------------------------------------------------------
# Loss


def compute_xof_loss(
    preds: list[torch.Tensor | None],
    batch: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = [batch[f"xof_oct{i}"] for i in range(4)]
    return huber_xof_loss(preds, targets)


def compute_emission_loss(pred: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Charbonnier + 0.1 * multi-scale L1 (full + 1/2 + 1/4)."""
    target = batch["emission"]
    eps = 1e-3
    charb = torch.sqrt((pred - target) ** 2 + eps ** 2).mean()
    ms = torch.zeros_like(charb)
    for s in (0.5, 0.25):
        h = max(1, int(pred.shape[-2] * s))
        w = max(1, int(pred.shape[-1] * s))
        p_s = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        t_s = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=False)
        ms = ms + F.l1_loss(p_s, t_s)
    total = charb + 0.1 * ms
    return total, {"total": total.item(), "charb": charb.item(), "ms_l1": ms.item()}


# -----------------------------------------------------------------------------
# Validation: candidate ranking + Window-FMR@95


def _chain_rows_with_emission(session_dir: Path) -> list[int]:
    chain = load_chain_log(session_dir / "chain_log.csv")
    emi_dir = session_dir / "derived" / "Emissions"
    out = []
    for t in chain:
        if (emi_dir / f"tile_{t:06d}.png").exists():
            out.append(t)
    return out


def _xof_for_chain_row(session_dir: Path, t: int, chain: dict[int, str]) -> list[torch.Tensor]:
    """Return centered XOF octaves for chain_row=t. None if t not in chain."""
    if t not in chain:
        return [None, None, None, None]
    octs = xof_octaves_centered_from_hex(chain[t])
    return list(octs)


@torch.no_grad()
def run_validation(
    *,
    model: torch.nn.Module,
    dataset: PackedCFADataset,
    cfg: dict,
    device: torch.device,
    other_session_chain_rows: list[int],
    other_session_id: str,
    same_session_chain: dict[int, str],
    other_session_chain: dict[int, str],
    score_variants: tuple[str, ...],
    n_eval_frames: int,
    autocast_dtype: torch.dtype,
    obs_logger: ObservabilityLogger | None = None,
    split_label: str = "val",
    v10_bin: str | None = None,
) -> dict:
    """Evaluate `n_eval_frames` from `dataset`. Returns:
        {
          "matched_scores_per_variant": {variant: [score per frame]},
          "negatives_per_variant": {variant: [{family: [scores]} per frame]},
          "n_frames": int,
          "candidate_log": list of {capture_row, target_chain_row, candidates: [(session, row, family)]},
        }
    """
    model.eval()
    n = min(n_eval_frames, len(dataset))
    indices = list(range(n))
    matched_scores: dict[str, list[float]] = {v: [] for v in score_variants}
    negatives: dict[str, list[dict[str, list[float]]]] = {v: [] for v in score_variants}
    candidate_log: list[dict] = []

    same_chain_rows = list(same_session_chain.keys())

    for idx in indices:
        sample = dataset[idx]
        cap = sample["capture_norm"].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            preds, info = model(cap)
        # Move predictions to fp32 cpu for scoring
        preds_cpu = [p.float().squeeze(0).cpu() if p is not None else None for p in preds]

        cs = assemble_candidate_set(
            capture_session=sample["session_id"],
            capture_row=int(sample["t"]),
            offset=int(cfg["offset"]["value"]),
            same_session_chain_rows=same_chain_rows,
            other_session_chain_rows=other_session_chain_rows,
            other_session_id=other_session_id,
            d_search_max=int(cfg["candidate_ranking"]["d_search_max"]),
            n_same_session_random=int(cfg["candidate_ranking"].get("n_same_session_random", 192)),
            n_cross_session_random=int(cfg["candidate_ranking"].get("n_cross_session_random", 192)),
            seed=int(cfg.get("seed", 42)) ^ int(sample["t"]),
        )

        cand_records = []
        per_variant_scores: dict[str, dict[str, list[float]]] = {v: {} for v in score_variants}
        matched_per_variant: dict[str, float] = {}
        for c in cs.candidates:
            cand_chain = same_session_chain if c.session == sample["session_id"] else other_session_chain
            t_octs = _xof_for_chain_row(Path(""), c.row, cand_chain)
            for v in score_variants:
                # Build a list aligned with predictions; missing octaves contribute 0.
                cand_octs_aligned: list[torch.Tensor | None] = []
                for i in range(4):
                    if preds_cpu[i] is None or t_octs[i] is None:
                        cand_octs_aligned.append(None)
                    else:
                        cand_octs_aligned.append(t_octs[i])
                s = score_candidate_xof(preds_cpu, cand_octs_aligned, v)
                if c.family == "matched":
                    matched_per_variant[v] = s
                else:
                    per_variant_scores[v].setdefault(c.family, []).append(s)
            cand_records.append({"session": c.session, "row": c.row, "family": c.family})
        for v in score_variants:
            matched_scores[v].append(matched_per_variant[v])
            negatives[v].append(per_variant_scores[v])
        candidate_log.append({
            "capture_row": int(sample["t"]),
            "target_chain_row": int(sample["target_chain_row"]),
            "candidates": cand_records,
            "seed": cs.seed,
        })

        if obs_logger is not None:
            cap_pre = sample["capture_pre_norm"]
            obs = compute_row_observability(cap_pre)
            obs_logger.log(
                session=sample["session_id"],  # type: ignore[arg-type]
                row=int(sample["t"]),
                split=split_label,  # type: ignore[arg-type]
                v10_bin=v10_bin,
                normalization_stats_source=cfg.get("normalization", {}).get(
                    f"{sample['session_id'].lower()}_stats_source", "unknown"
                ),
                offset_used=int(cfg["offset"]["value"]),
                observability=obs,
            )

    return {
        "matched_scores_per_variant": matched_scores,
        "negatives_per_variant": negatives,
        "n_frames": n,
        "candidate_log": candidate_log,
    }


def aggregate_window_fmr(val_result: dict, score_variants: tuple[str, ...], tau95_per_variant: dict[str, float]) -> dict:
    out = {}
    for v in score_variants:
        out[v] = per_family_window_fmr(
            matched_scores_by_frame=val_result["matched_scores_per_variant"][v],
            negative_scores_by_frame_by_family=val_result["negatives_per_variant"][v],
            tau95=tau95_per_variant[v],
        )
    return out


# -----------------------------------------------------------------------------
# Main


def _scheduler(opt, total_steps: int, warmup: int):
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=max(warmup, 1)),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps - warmup, 1)),
        ],
        milestones=[warmup],
    )


def collate(batch):
    cap_norm = torch.stack([b["capture_norm"] for b in batch])
    cap_pre = torch.stack([b["capture_pre_norm"] for b in batch])
    out = {
        "capture_norm": cap_norm,
        "capture_pre_norm": cap_pre,
        "t": torch.tensor([b["t"] for b in batch]),
        "target_chain_row": torch.tensor([b["target_chain_row"] for b in batch]),
        "session_id": [b["session_id"] for b in batch],
    }
    if "xof_oct0" in batch[0]:
        for i in range(4):
            out[f"xof_oct{i}"] = torch.stack([b[f"xof_oct{i}"] for b in batch])
    if "emission" in batch[0]:
        out["emission"] = torch.stack([b["emission"] for b in batch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stats-dir", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override config max_steps; useful for smoke tests.")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-frames-calib", type=int, default=64,
                    help="Number of frames in calibration eval per validation pass.")
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 autocast (A100). Default fp16 — A10 has no bf16 perf.")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    cfg = load_and_validate(args.config)
    if cfg["offset"]["value"] is None:
        sys.exit("Refusing to start: cfg.offset.value is null. "
                 "Run the offset diagnostic and update the config first.")
    if args.no_pretrained:
        cfg.setdefault("model", {})["pretrained"] = False

    ddp_active, rank, world_size, local_rank = init_distributed()
    if ddp_active:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    if is_main_rank():
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp_active:
        dist.barrier()

    # Manifest first thing.
    splits = {k: cfg["data"][k] for k in cfg["data"] if k.endswith(("_start", "_end"))}
    # Pass the whole cfg.offset block through to the manifest so all the
    # operator-mandated provenance (source, evidence, diagnostic outcomes,
    # rationale) is captured verbatim — see _phase_d_common.yaml.
    offset_convention = dict(cfg["offset"])
    offset_convention["winning_offset"] = cfg["offset"]["value"]
    offset_convention["definition"] = cfg["offset"]["convention"]
    if is_main_rank():
        write_run_manifest(
            out_dir,
            experiment_id=args.config.stem,
            config=cfg,
            offset_convention=offset_convention,
        xof_stored_representation=cfg["xof"]["stored_representation"],
        normalization_stats_source=str(cfg.get("normalization", {})),
        d_search_max=int(cfg["candidate_ranking"]["d_search_max"]),
        negatives_seed=int(cfg.get("seed", 42)),
        splits=splits,
        bayer_channel_order="RGGB",
        black_level_handling=str(cfg["data"].get("black_level_source", "no_measurement_found / no_subtraction")),
        target_cache_hash=compute_target_cache_hash(Path(cfg["data"]["cache_root"])),
        code_hash=compute_code_hash(ROOT / "src"),
        candidate_negative_families=["delay_window", "near_shift",
                                       "same_session_random", "cross_session_random"],
        fmr_calibration_split="D2-val first half" + ("" if _strict_d2_only(cfg) else " + V10-val first half"),
        fmr_report_split="D2-val second half" + ("" if _strict_d2_only(cfg) else " + V10-val second half"),
        stn_guardrails=cfg.get("stn") if cfg.get("model", {}).get("use_stn") else None,
            ddp_config={"world_size": world_size, "ddp_active": ddp_active,
                          "backend": "nccl" if ddp_active else "none"},
            camera_photometry_locked=cfg["provenance"]["camera_photometry_locked"],
            projector_pipeline_lock_status=cfg["provenance"]["projector_pipeline_lock_status"],
        )
    if ddp_active:
        dist.barrier()

    # Build datasets.
    datasets = build_datasets(cfg, args.stats_dir)

    # Pooled training set: D2-train + V10-train (or D2-only for A6).
    is_a6 = _strict_d2_only(cfg)
    if is_a6:
        train_ds = datasets["d2_train"]
    else:
        train_ds = torch.utils.data.ConcatDataset([datasets["d2_train"], datasets["v10_train"]])
    print(f"[init] train dataset size: {len(train_ds)}", flush=True)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 4))
    sampler = DistributedSampler(train_ds, shuffle=True) if ddp_active else None
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler, num_workers=nw,
        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=(nw > 0),
    )

    # Build model + optimizer.
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    main_rank_print(f"[init] model={cfg['model']['class']} ({cfg['model'].get('encoder_size','tiny')}) "
          f"params={n_params/1e6:.1f} M  use_stn={cfg['model'].get('use_stn', False)} "
          f"world_size={world_size}", flush=True)

    # STN: freeze for steps 0..249.
    # `has_stn` operates on the underlying model BEFORE we wrap with DDP.
    has_stn = bool(getattr(model, "stn", None))
    if has_stn:
        stn_freeze_set(model, frozen=True)
    optimizer = build_optimizer(model, cfg)
    # Wrap with DDP after optimizer is constructed (DDP buckets register on params).
    if ddp_active:
        # find_unused_parameters=True needed for A0: FPN constructs all 4 levels
        # (P2..P5) but A0's heads only consume P5 and P4, so params on lat0,
        # lat1, smooth0, smooth1 don't receive gradients. ~5% overhead but
        # avoids spurious DDP failures across the matrix.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
            broadcast_buffers=False,
        )

    def _unwrap(m):
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    # Scheduler.
    max_steps = args.max_steps or int(cfg["train"].get("max_steps", 5000))
    warmup = int(cfg["train"].get("warmup_steps", 250))
    sched = _scheduler(optimizer, max_steps, warmup)

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    obs_logger = (ObservabilityLogger(out_dir / "observability.jsonl", experiment_id=args.config.stem)
                  if is_main_rank() else None)

    # Chain logs for candidate scoring.
    d2_chain = load_chain_log(Path(cfg["data"]["d2_dir"]) / "chain_log.csv")
    v10_chain = load_chain_log(_v10_dir(cfg) / "chain_log.csv")
    d2_chain_rows = _chain_rows_with_emission(Path(cfg["data"]["d2_dir"]))
    v10_chain_rows = _chain_rows_with_emission(_v10_dir(cfg))

    is_emission_task = cfg["model"]["class"] == "EmissionPredictorV2"
    score_variants = tuple() if is_emission_task else score_variants_for_experiment(args.config.stem)
    stn_lambda = float(cfg.get("stn", {}).get("identity_reg_weight", 0.01))
    freeze_until = int(cfg.get("stn", {}).get("freeze_until_step", 250))

    log_every = int(cfg["train"].get("log_every", 10))

    # Training loop.
    best_loss = math.inf
    best_fmr = math.inf
    step = 0
    train_iter = iter(train_loader)
    t_start = time.time()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if has_stn and step == freeze_until:
            stn_freeze_set(_unwrap(model), frozen=False)
            main_rank_print(f"[train] step {step}: STN unfrozen", flush=True)

        cap = batch["capture_norm"].to(device, non_blocking=True)
        model.train()
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            if is_emission_task:
                pred = model(cap)
                loss, parts = compute_emission_loss(pred, {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                                            for k, v in batch.items()})
            else:
                preds, info = model(cap)
                # move XOF targets to device
                target_dict = {f"xof_oct{i}": batch[f"xof_oct{i}"].to(device) for i in range(4)}
                loss, parts = compute_xof_loss(preds, target_dict)
                if has_stn and info.get("stn") is not None:
                    reg = identity_regularizer(info["stn"]["theta"])
                    loss = loss + stn_lambda * reg
                    parts["stn_id_reg"] = reg.item()
                    parts["stn_oob_frac"] = info["stn"]["grid_oob_fraction"]

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        sched.step()

        if step % log_every == 0:
            main_rank_print(f"[step {step}] loss={parts['total']:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} {parts}", flush=True)

        # Validation — only rank 0 runs candidate ranking (it's CPU-bound + needs full chain log)
        if step > 0 and step % args.val_every == 0 and not is_emission_task and is_main_rank():
            val_calib = datasets["d2_val_calib"]
            val_result = run_validation(
                model=_unwrap(model), dataset=val_calib, cfg=cfg, device=device,
                other_session_chain_rows=v10_chain_rows, other_session_id="V10",
                same_session_chain=d2_chain, other_session_chain=v10_chain,
                score_variants=score_variants,
                n_eval_frames=args.val_frames_calib,
                autocast_dtype=autocast_dtype,
                obs_logger=obs_logger, split_label="val_calib",
            )
            tau95_per_variant = {v: compute_tau95(val_result["matched_scores_per_variant"][v])
                                 for v in score_variants}
            window_fmr = aggregate_window_fmr(val_result, score_variants, tau95_per_variant)
            print(f"[val step {step}] tau95={ {v: round(t, 4) for v, t in tau95_per_variant.items()} } "
                  f"worst_family[all_octaves]={window_fmr.get('all_octaves', {}).get('worst_family_value')}",
                  flush=True)
            (out_dir / "val_history.jsonl").open("a").write(json.dumps({
                "step": step, "tau95": tau95_per_variant, "window_fmr": window_fmr,
            }) + "\n")
            # Checkpoint by FMR (worst-family on all_octaves variant if available, else first variant).
            primary_variant = "all_octaves" if "all_octaves" in window_fmr else next(iter(window_fmr))
            primary_fmr = window_fmr[primary_variant]["worst_family_value"]
            if primary_fmr < best_fmr:
                best_fmr = primary_fmr
                _save_ckpt(_unwrap(model), out_dir / "checkpoints" / "best_by_calibration_half_window_fmr.pt",
                           step, {"tau95": tau95_per_variant, "window_fmr": window_fmr})

        # Loss-based checkpoint (rank 0 only)
        if parts["total"] < best_loss and is_main_rank():
            best_loss = parts["total"]
            _save_ckpt(_unwrap(model), out_dir / "checkpoints" / "best_by_loss.pt", step,
                       {"loss": parts["total"]})

        step += 1

    # Final checkpoint (rank 0)
    if is_main_rank():
        _save_ckpt(_unwrap(model), out_dir / "checkpoints" / "final_step.pt", step, {})

    # Final report on REPORT halves (rank 0 only)
    if not is_emission_task and is_main_rank():
        print(f"[final] running REPORT-half evaluation", flush=True)
        report = {}
        for split_name, ds_key, v10_bin in [
            ("d2_val_report", "d2_val_report", None),
            ("v10_val_report", "v10_val_report", None),
            ("v10_early", "v10_early", "early"),
            ("v10_mid", "v10_mid", "mid"),
            ("v10_late", "v10_late", "late"),
        ]:
            ds = datasets.get(ds_key)
            if ds is None or len(ds) == 0:
                continue
            res = run_validation(
                model=_unwrap(model), dataset=ds, cfg=cfg, device=device,
                other_session_chain_rows=v10_chain_rows if ds_key.startswith("d2") else d2_chain_rows,
                other_session_id="V10" if ds_key.startswith("d2") else "D2",
                same_session_chain=d2_chain if ds_key.startswith("d2") else v10_chain,
                other_session_chain=v10_chain if ds_key.startswith("d2") else d2_chain,
                score_variants=score_variants,
                n_eval_frames=min(len(ds), args.val_frames_calib * 4),
                autocast_dtype=autocast_dtype,
                obs_logger=obs_logger, split_label="val_report", v10_bin=v10_bin,
            )
            tau95_per_variant = {v: compute_tau95(res["matched_scores_per_variant"][v])
                                 for v in score_variants}
            window_fmr = aggregate_window_fmr(res, score_variants, tau95_per_variant)
            report[split_name] = {"tau95": tau95_per_variant, "window_fmr": window_fmr,
                                   "n_frames": res["n_frames"]}
            print(f"[final] {split_name}: tau95={ {v: round(t, 4) for v, t in tau95_per_variant.items()} }",
                  flush=True)
        (out_dir / "final_report.json").write_text(json.dumps(report, indent=2))

    if obs_logger is not None:
        obs_logger.close()
    if ddp_active:
        dist.barrier()
        dist.destroy_process_group()
    elapsed = time.time() - t_start
    main_rank_print(f"[done] elapsed={elapsed:.0f}s steps={step}", flush=True)


def _save_ckpt(model: torch.nn.Module, path: Path, step: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "step": step, "meta": meta}, tmp)
    tmp.replace(path)


if __name__ == "__main__":
    main()
