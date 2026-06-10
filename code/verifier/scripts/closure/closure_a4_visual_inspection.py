"""A4 visual inspection grids — verify the 13 dB matched ≈ cross-pair result.

For each sample frame:
  6 columns: capture (display RGB) | A4 prediction | GT@t (matched)
             | GT@t+100 (cross same-session) | GT@t+1000 (cross same-session)
             | GT@V10 (cross-session)
  Annotate row index, session, PSNR vs each candidate.

D2-val: 8 frames sampled uniformly from rows [4792, 5992)
V10 early/mid/late: 4 frames each, from configured ranges.

Saves PNG grids to <out-dir>/{d2_val_grid,v10_early_grid,v10_mid_grid,v10_late_grid}.png
plus provenance.md noting checkpoint + architecture used.

Run:
  python scripts/closure/closure_a4_visual_inspection.py \
    --ckpt experiments/exp001h_a4/checkpoints/final_step.pt \
    --config configs/exp001h_a4.yaml \
    --stats-dir cache/normalization_stats \
    --out-dir experiments/closure_package/a4_visual_inspection
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.packed_cfa_dataset import PackedCFADataset  # noqa: E402
from data.emission_dataset import load_emission_at  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
from preprocessing.normalization import load_stats  # noqa: E402
from utils.config_loader import load_and_validate  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def cfa_to_display_rgb(cfa: torch.Tensor) -> np.ndarray:
    """Pack 4-channel CFA (R, G1, G2, B) into a 3-channel RGB image for display.
    Input expected to be float in [0, 1] range; output uint8.
    """
    if cfa.dim() == 3:
        c, h, w = cfa.shape
    else:
        raise ValueError(f"unexpected CFA shape {cfa.shape}")
    r, g1, g2, b = cfa[0], cfa[1], cfa[2], cfa[3]
    g = 0.5 * (g1 + g2)
    rgb = torch.stack([r, g, b], dim=0)
    rgb = rgb.clamp(0, 1).cpu().numpy()
    return (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def emission_to_display(em: torch.Tensor) -> np.ndarray:
    arr = em.clamp(0, 1).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    return (arr * 255.0).astype(np.uint8)


def downsample_for_display(img: np.ndarray, target=256) -> np.ndarray:
    """Resize to ~target on the smaller side, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = target / min(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    img_t = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return img_t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()


def gamma_correct(img_uint8: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    img = img_uint8.astype(np.float32) / 255.0
    img = np.clip(img, 1e-6, 1.0) ** (1.0 / gamma)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def safe_load_emission(session_dir: Path, t: int, h: int, w: int) -> torch.Tensor | None:
    """Wrap load_emission_at; returns None on missing/error."""
    try:
        path = session_dir / "derived" / "Emissions" / f"tile_{t:06d}.png"
        if not path.exists():
            return None
        return load_emission_at(path, h, w)
    except Exception as exc:
        print(f"[WARN] safe_load_emission({session_dir.name}, t={t}): {exc}", flush=True)
        return None


def build_grid_for_split(*, model, session_dir: Path, session_id: str, rows: list[int],
                         offset: int, stats: dict, cache_root: Path,
                         alt_session_dir: Path, alt_session_id: str, alt_rows: list[int],
                         emission_h: int, emission_w: int,
                         device: torch.device, autocast_dtype: torch.dtype,
                         out_path: Path, title: str, n_samples: int = 8,
                         seed: int = 7):
    """Build one grid PNG for a split."""
    if len(rows) == 0:
        print(f"[skip] {title}: no rows", flush=True)
        return

    rng = np.random.RandomState(seed)
    sample_rows = sorted(rng.choice(rows, size=min(n_samples, len(rows)), replace=False).tolist())
    print(f"[{title}] sampling rows: {sample_rows}", flush=True)

    ds = PackedCFADataset(
        session_dir=session_dir,
        rows=sample_rows,
        offset=offset,
        normalization_stats=stats,
        cache_root=cache_root,
        session_id=session_id,
        with_xof=False,
        with_emission=True,
    )
    if len(ds) == 0:
        print(f"[skip] {title}: dataset empty", flush=True)
        return

    n = len(ds)
    cols = ["capture (input)", "A4 pred", "GT @ t (matched)",
            "GT @ t+100 (same)", "GT @ t+1000 (same)",
            f"GT @ {alt_session_id} (cross)"]
    n_cols = len(cols)

    # Pick alt-session sample rows once (one per row, independently)
    alt_picks = list(rng.choice(alt_rows, size=n, replace=False))

    fig, axes = plt.subplots(n, n_cols, figsize=(3.2 * n_cols, 3.2 * n + 0.6),
                              squeeze=False)
    fig.suptitle(title, fontsize=14)

    chain_session = load_chain_log(session_dir / "chain_log.csv")
    alt_chain = load_chain_log(alt_session_dir / "chain_log.csv")

    for i in range(n):
        sample = ds[i]
        t = int(sample["t"])
        # CAPTURE display
        cap_pre = sample["capture_pre_norm"].float()
        # capture_pre_norm is in [0, 1] after the dataset's normalize step? Check shape.
        # Actually it's pre-normalized — uint8/255. Range should be [0, 1].
        cap_disp = cfa_to_display_rgb(cap_pre)
        cap_disp = downsample_for_display(cap_disp, target=256)

        # PREDICTION
        cap_norm = sample["capture_norm"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap_norm).float()
        pred = pred.squeeze(0).clamp(0, 1).cpu()
        pred_disp = emission_to_display(pred)
        pred_disp = downsample_for_display(pred_disp, target=256)

        # GT matched (the emission for row t — already loaded by dataset)
        gt_matched = sample["emission"].clamp(0, 1).cpu()
        gt_matched_disp = emission_to_display(gt_matched)
        gt_matched_disp = downsample_for_display(gt_matched_disp, target=256)
        psnr_matched = psnr(pred, gt_matched)

        # GT @ t+100 same session
        gt_100 = safe_load_emission(session_dir, t + 100, emission_h, emission_w)
        if gt_100 is None:
            psnr_100 = float("nan")
            gt100_disp = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            gt100_clamped = gt_100.clamp(0, 1)
            psnr_100 = psnr(pred, gt100_clamped)
            gt100_disp = downsample_for_display(emission_to_display(gt100_clamped))

        # GT @ t+1000 same session
        gt_1000 = safe_load_emission(session_dir, t + 1000, emission_h, emission_w)
        if gt_1000 is None:
            psnr_1000 = float("nan")
            gt1000_disp = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            gt1000_clamped = gt_1000.clamp(0, 1)
            psnr_1000 = psnr(pred, gt1000_clamped)
            gt1000_disp = downsample_for_display(emission_to_display(gt1000_clamped))

        # GT @ alt_session (cross-session)
        alt_t = int(alt_picks[i])
        gt_alt = safe_load_emission(alt_session_dir, alt_t, emission_h, emission_w)
        if gt_alt is None:
            psnr_alt = float("nan")
            gt_alt_disp = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            gt_alt_clamped = gt_alt.clamp(0, 1)
            psnr_alt = psnr(pred, gt_alt_clamped)
            gt_alt_disp = downsample_for_display(emission_to_display(gt_alt_clamped))

        # Plot row
        for j, (im, sub) in enumerate(zip(
                [cap_disp, pred_disp, gt_matched_disp, gt100_disp, gt1000_disp, gt_alt_disp],
                ["", "", f"PSNR={psnr_matched:.2f}", f"PSNR={psnr_100:.2f}",
                 f"PSNR={psnr_1000:.2f}", f"PSNR={psnr_alt:.2f}"])):
            ax = axes[i, j]
            ax.imshow(im)
            ax.axis("off")
            label = cols[j] if i == 0 else ""
            row_id = f"{session_id} t={t}" if j == 0 else (
                     f"{alt_session_id} t={alt_t}" if j == 5 else
                     (f"{session_id} t={t+100}" if j == 3 else
                      (f"{session_id} t={t+1000}" if j == 4 else "")))
            title_text = f"{label}\n{row_id}\n{sub}".strip()
            ax.set_title(title_text, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stats-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-d2", type=int, default=8)
    ap.add_argument("--n-v10", type=int, default=4)
    args = ap.parse_args()

    cfg = load_and_validate(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Confirm arch + load
    model = EmissionPredictorV2(
        emission_h=int(cfg["data"].get("emission_h", 1080)),
        emission_w=int(cfg["data"].get("emission_w", 1920)),
        encoder_size=cfg["model"].get("encoder_size", "tiny"),
        pretrained=False,
        fpn_out_channels=int(cfg["model"].get("fpn_out_channels", 256)),
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] ckpt key mismatch missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    cache_root = Path(cfg["data"]["cache_root"])
    d2_dir = Path(cfg["data"]["d2_dir"])
    v10_dir = Path(cfg["data"].get("v10_dir", str(d2_dir.parent / "v10")))
    d2_stats = load_stats(args.stats_dir / "d2_train_stats.json")
    v10_stats = load_stats(args.stats_dir / "v10_train_stats.json")
    offset = int(cfg["offset"]["value"])
    em_h = int(cfg["data"].get("emission_h", 1080))
    em_w = int(cfg["data"].get("emission_w", 1920))

    # Provenance
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        "model_class": "EmissionPredictorV2",
        "encoder_size": cfg["model"].get("encoder_size", "tiny"),
        "input_format": "4-channel packed CFA (R, G1, G2, B), median+IQR/1.349 normalized",
        "output_format": f"3-channel RGB, sigmoid in [0, 1], shape (3, {em_h}, {em_w})",
        "offset": offset,
        "ckpt_step": ck.get("step"),
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
        "datetime_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    (args.out_dir / "provenance.md").write_text(
        f"# A4 visual inspection — provenance\n\n"
        f"- Checkpoint: `{prov['ckpt']}`\n"
        f"- Config: `{prov['config']}`\n"
        f"- Model class: `{prov['model_class']}` (ConvNeXt-Tiny + CoordAware FPN + RGB U-Net decoder)\n"
        f"- Input: {prov['input_format']}\n"
        f"- Output: {prov['output_format']}\n"
        f"- Offset (target_chain_row = capture_row + offset): {prov['offset']}\n"
        f"- Checkpoint step: {prov['ckpt_step']}\n"
        f"- Missing/unexpected keys at load: {prov['missing_keys']}/{prov['unexpected_keys']}\n"
        f"- Generated: {prov['datetime_utc']}\n\n"
        f"## Note on prior visualizations\n\n"
        f"If the operator previously saw row-specific emission predictions, it is worth checking\n"
        f"whether those came from this A4 checkpoint or from Phase B's `exp001c` (the strong\n"
        f"emission baseline at 26.20 dB val PSNR + 15.97 dB cross-pair gap). The Phase D A4\n"
        f"final_step.pt produced **matched mean PSNR ≈ 13.13 dB and cross-pair PSNR ≈ 13.13 dB**\n"
        f"(post-hoc eval over ~1200 D2-val frames + V10 splits) — i.e. the prediction is\n"
        f"essentially identical against any candidate target. The grids in this folder are the\n"
        f"visual side of that quantitative observation.\n"
    )

    # D2 grid
    d2_val_rows = list(range(int(cfg["data"]["d2_val_start"]), int(cfg["data"]["d2_val_end"])))
    v10_val_rows = list(range(int(cfg["data"]["v10_val_start"]), int(cfg["data"]["v10_val_end"])))
    build_grid_for_split(
        model=model, session_dir=d2_dir, session_id="D2", rows=d2_val_rows,
        offset=offset, stats=d2_stats, cache_root=cache_root,
        alt_session_dir=v10_dir, alt_session_id="V10", alt_rows=v10_val_rows,
        emission_h=em_h, emission_w=em_w,
        device=device, autocast_dtype=autocast_dtype,
        out_path=args.out_dir / "d2_val_grid.png",
        title=f"A4 D2-val visual inspection (n={args.n_d2})",
        n_samples=args.n_d2,
        seed=7,
    )

    for split_name, ka, kb, sd in [
        ("v10_early", "v10_early_start", "v10_early_end", v10_dir),
        ("v10_mid", "v10_mid_start", "v10_mid_end", v10_dir),
        ("v10_late", "v10_late_start", "v10_late_end", v10_dir),
    ]:
        rows = list(range(int(cfg["data"][ka]), int(cfg["data"][kb])))
        build_grid_for_split(
            model=model, session_dir=v10_dir, session_id="V10", rows=rows,
            offset=offset, stats=v10_stats, cache_root=cache_root,
            alt_session_dir=d2_dir, alt_session_id="D2", alt_rows=d2_val_rows,
            emission_h=em_h, emission_w=em_w,
            device=device, autocast_dtype=autocast_dtype,
            out_path=args.out_dir / f"{split_name}_grid.png",
            title=f"A4 {split_name} visual inspection (n={args.n_v10})",
            n_samples=args.n_v10,
            seed=11 + hash(split_name) % 100,
        )

    print(f"[done] grids in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
