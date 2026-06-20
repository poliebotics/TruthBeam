"""Scoring-function comparison experiment — operator directive 2026-05-05.

Compare 7 ways of summarizing Phase G's per-pixel ε-residual tensor and
characterize what each correlates with spatially.

Scoring functions (all computed from existing per-pixel residual fields):

  1. ε-MSE           — mean over CFA channels of residual²        (current Phase G score)
  2. ε-MAE           — mean over CFA channels of |residual|       (test if squaring matters)
  3. signed residual — mean over CFA channels of residual         (preserves sign)
  4. low-freq MSE    — MSE of Gaussian-blurred residual (σ=4)     (coarse-scale signal)
  5. high-freq MSE   — MSE of (residual − blurred)                (fine-scale signal)
  6. VGG conv4_2 d   — ‖VGG(residual_cond) − VGG(residual_real)‖  (perceptual features)
  7. B1-encoder d    — ‖B1.enc(residual_cond) − B1.enc(real)‖     (project-trained features)

Reference fields (computed once per frame, used to interpret what each
scoring function correlates with spatially):

  R1. rendered E magnitude     — E_correct mean over RGB
  R2. scene gradient           — Sobel(grayscale(C_real))
  R3. scene content residual   — |C_real − temporal_median(C, ±k=2)|

Conditions per frame (7):
  real_correct, fake_5k, fake_25k, fake_70k, fake_100k,
  shuffled_E, cross_session_E.

Frame subset: 30 D2 + 30 V10 from EVAL_BLOCKS (blocked, evenly-spaced
per block, no adjacent frames).

Subcommands:
  extract  : per-frame extraction (Phase G + F-A v1 + VGG + B1 forwards).
             Distributable across GPUs by --frames-json shard.
  analyze  : Pearson reference correlations + AUROC + Δscore + Spearman
             cross-scoring agreement + hierarchical bootstrap.
  render   : 7 × 6 visual grids (1 representative + 3 supplementary
             frames per session).

Pre-registered interpretation thresholds (operator-locked):
  Pearson |r| > 0.5 → strongly follows;
  0.3 < |r| ≤ 0.5 → moderately;
  0.1 < |r| ≤ 0.3 → weakly;
  |r| ≤ 0.1 → does not follow.
  Spearman ρ > 0.7 → scoring functions agree on per-frame ranking;
  0.4 < ρ ≤ 0.7 → moderate; ρ ≤ 0.4 → disagree.

Standing rules: Phase G inference-only. No held-out asset use beyond
F-A v1 outputs. No F-A v2 trainer touch. No Phase G modification. No
information feedback to Phase G design.
"""
from __future__ import annotations

import argparse
import csv
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
    _crop_and_resize_C, _load_packed_cfa_float01,
    _resize_E_to_target, EMISSION_NATIVE_H, EMISSION_NATIVE_W,
    EVAL_BLOCKS,
)
from data.emission_dataset import load_emission_at  # noqa: E402
from phase_g.fa_loader import load_fa_v1_checkpoint, render_C_fake  # noqa: E402


# ----------------------------- constants -----------------------------

T_DIFFUSION = 1000
T_STEPS = (50, 150, 300, 500, 750)
K_NOISE = 4
NOISE_SEED_BASE = 42

PHASE_G_INPUT_H = 768
PHASE_G_INPUT_W = 1024
COARSE_FACTOR = 8
COARSE_H = PHASE_G_INPUT_H // COARSE_FACTOR  # 96
COARSE_W = PHASE_G_INPUT_W // COARSE_FACTOR  # 128

FA_V1_CKPT_STEPS = (5000, 25000, 70000, 100000)
PERTURBED_CONDS = (
    [f"fake_{s//1000}k" for s in FA_V1_CKPT_STEPS]
    + ["shuffled_E", "cross_session_E"]
)
ALL_CONDS = ["real_correct"] + PERTURBED_CONDS

SCORING_FUNCTIONS = (
    "eps_mse", "eps_mae", "signed_residual",
    "low_freq_mse", "high_freq_mse",
    "vgg_distance", "b1_distance",
)
SIGNED_FUNCTIONS = {"signed_residual"}
NONNEG_FUNCTIONS = {"eps_mse", "eps_mae", "low_freq_mse", "high_freq_mse",
                     "vgg_distance", "b1_distance"}

GAUSSIAN_BLUR_SIGMA = 4.0   # operator default for low/high frequency split
TEMPORAL_MEDIAN_K = 2        # ±k frames for R3


# ----------------------------- frame subset -----------------------------

def build_frame_subset() -> list[dict]:
    """Deterministic blocked-sampled 30 D2 + 30 V10 frames.

    Each block uses its inset range `[a + 30, b - 30)` (matching Phase G
    eval guard). Within each block, evenly-spaced frames at floor offsets
    so spacing is non-zero and there are no adjacent frames.

    D2: 3 blocks of 400 frames → 10 frames each (spacing ≈ 34).
    V10: 2 blocks of 250 frames → 15 frames each (spacing ≈ 12).
    """
    out: list[dict] = []
    for sess, n_per_block in (("D2", 10), ("V10", 15)):
        blocks = EVAL_BLOCKS[sess]
        for bi, (a, b) in enumerate(blocks):
            lo, hi = a + 30, b - 30
            span = hi - lo
            for i in range(n_per_block):
                row = lo + int(round(i * span / n_per_block))
                out.append({"session": sess, "row": row, "block": bi})
    return out


# ----------------------------- model loaders -----------------------------

def load_phase_g(ckpt_path: Path, device: torch.device,
                 dtype: torch.dtype) -> DiffusionDiagnosticUNet:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    saved_args = ck.get("args", {}) if isinstance(ck, dict) else {}
    base_ch = saved_args.get("base_ch", 96)
    mults = tuple(saved_args.get("mults", [1, 2, 4, 4]))
    attn_at = saved_args.get("attn_at")
    if attn_at is None:
        attn_at = tuple(i == len(mults) - 1 for i in range(len(mults)))
    else:
        attn_at = tuple(bool(x) for x in attn_at)
    cond_drop_prob = saved_args.get("cond_drop_prob", 0.2)
    model = DiffusionDiagnosticUNet(
        in_ch=4, base_ch=base_ch, channel_mults=mults, attn_at=attn_at,
        cond_drop_prob=cond_drop_prob, hint_in_ch=11,
    ).to(device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_b1_encoder(ckpt_path: Path, device: torch.device,
                    dtype: torch.dtype):
    """Load FreshBinderB and return its `encoder` submodule (timm
    convnext_tiny features_only). Forward returns a 4-tuple of features;
    we use the deepest (f3) as the bottleneck.
    """
    from models.fresh import FRESH_BINDER_REGISTRY
    cls = FRESH_BINDER_REGISTRY["B"]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    cfg = ck.get("cfg", {}) or {}
    model_kwargs = cfg.get("model_kwargs", {}) or {}
    capture_h = cfg.get("capture_h", 1150)
    capture_w = cfg.get("capture_w", 1330)
    # pretrained=False: we'll overwrite all weights from the ckpt.
    binder = cls(pretrained=False, **model_kwargs)
    binder.load_state_dict(state, strict=True)
    binder = binder.to(device, dtype=dtype).eval()
    for p in binder.parameters():
        p.requires_grad = False
    return binder, capture_h, capture_w


def load_vgg16_features(device: torch.device, dtype: torch.dtype):
    """torchvision VGG-16 conv4_2 features. Cuts the network at conv4_2
    (the 22nd module of vgg16.features → index 22 inclusive). Returns
    the truncated feature module + ImageNet normalization constants.
    """
    from torchvision.models import vgg16, VGG16_Weights
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # vgg16.features layer indices (Sequential of 31 modules, Conv-ReLU
    # alternating with Pool every 2 conv pairs):
    #  conv1_1=0, ReLU=1, conv1_2=2, ReLU=3, pool1=4,
    #  conv2_1=5, ReLU=6, conv2_2=7, ReLU=8, pool2=9,
    #  conv3_1=10, ReLU=11, conv3_2=12, ReLU=13, conv3_3=14, ReLU=15, pool3=16,
    #  conv4_1=17, ReLU=18, conv4_2=19, ReLU=20, conv4_3=21, ...
    # We take through the ReLU-after-conv4_2 (post-ReLU activation per
    # LPIPS convention). children()[:21] yields modules at indices 0..20
    # inclusive, ending with the ReLU at index 20 — i.e. post-ReLU(conv4_2).
    feat = torch.nn.Sequential(*list(m.features.children())[:21])
    feat = feat.to(device, dtype=dtype).eval()
    for p in feat.parameters():
        p.requires_grad = False
    # ImageNet mean/std for VGG normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    return feat, mean, std


# ----------------------------- IO helpers -----------------------------

def load_phase_g_C(session_dir: Path, row: int) -> torch.Tensor:
    return _crop_and_resize_C(_load_packed_cfa_float01(
        session_dir / "Recordings" / f"frame_{row:06d}.raw"))


def load_phase_g_E(session_dir: Path, row: int) -> torch.Tensor:
    return _resize_E_to_target(load_emission_at(
        session_dir / "derived" / "Emissions" / f"tile_{row:06d}.png",
        EMISSION_NATIVE_H, EMISSION_NATIVE_W))


def load_chain_keys(session_dir: Path) -> list[int]:
    import csv as _csv
    out: list[int] = []
    with open(session_dir / "chain_log.csv") as f:
        r = _csv.reader(f)
        for row in r:
            if not row or row[0].startswith("#"):
                continue
            try:
                out.append(int(row[0]))
            except ValueError:
                continue
    return sorted(set(out))


def deterministic_shuffled_row(this_row: int, keys: list[int]) -> int:
    if this_row not in keys:
        raise ValueError(f"row {this_row} missing")
    n = len(keys)
    idx = keys.index(this_row)
    return keys[(idx + n // 2) % n]


def deterministic_cross_session_row(this_row: int, this_keys: list[int],
                                     other_keys: list[int]) -> int:
    idx_self = this_keys.index(this_row)
    n_self, n_other = len(this_keys), len(other_keys)
    pct = idx_self / max(n_self - 1, 1)
    return other_keys[min(n_other - 1, int(round(pct * (n_other - 1))))]


def block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    """2-D non-overlapping block-mean. arr shape must divide by factor."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D, got {arr.shape}")
    h, w = arr.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"shape {arr.shape} not divisible by {factor}")
    h2, w2 = h // factor, w // factor
    return arr.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


# ----------------------------- Gaussian blur -----------------------------

def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply 2-D Gaussian blur to a (..., H, W) tensor via separable
    convolution. Kernel size = 2 * ceil(3σ) + 1."""
    radius = int(np.ceil(3 * sigma))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # Separable: blur along W then along H. Use depth-wise conv with
    # groups equal to channel count. We need 4D input for F.conv2d.
    if x.dim() == 2:
        x_4d = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x_4d = x.unsqueeze(0)
    elif x.dim() == 4:
        x_4d = x
    else:
        raise ValueError(f"unsupported dim: {x.dim()}")
    C = x_4d.shape[1]
    kx = g.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
    ky = g.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
    out = F.conv2d(x_4d, kx, padding=(0, radius), groups=C)
    out = F.conv2d(out, ky, padding=(radius, 0), groups=C)
    if x.dim() == 2:
        return out.squeeze(0).squeeze(0)
    if x.dim() == 3:
        return out.squeeze(0)
    return out


# ----------------------------- per-condition extraction -----------------------------

@torch.no_grad()
def extract_per_condition(
    model: DiffusionDiagnosticUNet,
    C: torch.Tensor,            # (4, H, W) float32 [0, 1]
    E: torch.Tensor,            # (3, H, W) float32 [0, 1]
    dc: dict,
    device: torch.device,
    dtype: torch.dtype,
    noise: torch.Tensor,        # (n_t, K, 4, H, W) float32
    timesteps: tuple[int, ...] = T_STEPS,
) -> dict:
    """Forward Phase G across timesteps × K with paired noise; accumulate
    streaming statistics for scoring functions 1-5 and the (t,k)-mean
    4-channel residual.

    Returns dict with keys:
      score_eps_mse / score_eps_mae / score_signed_residual
      score_low_freq_mse / score_high_freq_mse        — all (H, W) float32
      residual_4ch_mean                               — (4, H, W) float32
    """
    H, W = C.shape[-2:]
    n_t = len(timesteps)
    K_ = noise.shape[1]
    accum_mse = torch.zeros(H, W, dtype=torch.float64, device=device)
    accum_mae = torch.zeros(H, W, dtype=torch.float64, device=device)
    accum_signed = torch.zeros(H, W, dtype=torch.float64, device=device)
    accum_low = torch.zeros(H, W, dtype=torch.float64, device=device)
    accum_high = torch.zeros(H, W, dtype=torch.float64, device=device)
    accum_residual_4ch = torch.zeros(4, H, W, dtype=torch.float64, device=device)
    n_samples = n_t * K_

    for ti, t_val in enumerate(timesteps):
        t_tensor = torch.full((K_,), t_val, device=device, dtype=torch.long)
        t_float = t_tensor.float()
        C_rep = C.float().unsqueeze(0).expand(K_, -1, -1, -1).contiguous()
        C_t = q_sample(C_rep, t_tensor, dc, noise[ti]).to(dtype)
        E_batch = E.unsqueeze(0).to(device=device, dtype=dtype).expand(
            K_, -1, -1, -1).contiguous()
        with torch.amp.autocast("cuda", dtype=dtype):
            eps_pred = model(C_t, E_batch, t_float, force_uncond=False)
        residual_4ch = (eps_pred.float() - noise[ti].float())  # (K, 4, H, W)
        # 1-channel residual (mean over CFA channels)
        residual_1ch = residual_4ch.mean(dim=1)                 # (K, H, W)
        # Accumulate over K
        accum_mse.add_(residual_1ch.pow(2).sum(dim=0).double())
        accum_mae.add_(residual_1ch.abs().sum(dim=0).double())
        accum_signed.add_(residual_1ch.sum(dim=0).double())
        # Per-(K) blur for low/high-frequency split
        blurred = gaussian_blur_2d(residual_1ch.unsqueeze(1),
                                    GAUSSIAN_BLUR_SIGMA).squeeze(1)
        accum_low.add_(blurred.pow(2).sum(dim=0).double())
        accum_high.add_((residual_1ch - blurred).pow(2).sum(dim=0).double())
        accum_residual_4ch.add_(residual_4ch.sum(dim=0).double())

    # Finalize in-place to avoid creating six float64 temporaries on the
    # GPU when memory headroom is tight.
    accum_mse.div_(n_samples)
    accum_mae.div_(n_samples)
    accum_signed.div_(n_samples)
    accum_low.div_(n_samples)
    accum_high.div_(n_samples)
    accum_residual_4ch.div_(n_samples)
    return {
        "score_eps_mse":         accum_mse.float().cpu().numpy(),
        "score_eps_mae":         accum_mae.float().cpu().numpy(),
        "score_signed_residual": accum_signed.float().cpu().numpy(),
        "score_low_freq_mse":    accum_low.float().cpu().numpy(),
        "score_high_freq_mse":   accum_high.float().cpu().numpy(),
        "residual_4ch_mean":     accum_residual_4ch.float().cpu().numpy(),
    }


# ----------------------------- VGG / B1 distances -----------------------------

def cfa4_to_rgb3(residual_4ch: torch.Tensor) -> torch.Tensor:
    """(4, H, W) packed-CFA residual → (3, H, W) pseudo-RGB by mapping
    (R, G1, G2, B) → (R, mean(G1, G2), B)."""
    if residual_4ch.shape[0] != 4:
        raise ValueError(f"expected 4 CFA channels, got {residual_4ch.shape}")
    return torch.stack([
        residual_4ch[0],
        0.5 * (residual_4ch[1] + residual_4ch[2]),
        residual_4ch[3],
    ], dim=0)


@torch.no_grad()
def vgg_features(residual_4ch: torch.Tensor, vgg_feat, vgg_mean, vgg_std,
                 device, dtype) -> torch.Tensor:
    """(4, 768, 1024) residual → VGG conv4_2 feature map (C, 96, 128).

    Operator's spec: CFA → RGB via (R, mean(G1, G2), B), ImageNet
    normalization, conv4_2 per LPIPS. Residual values can be in any
    range — for VGG we treat them as activations and normalize per the
    standard ImageNet mean/std (subtracts mean, divides by std).
    """
    rgb = cfa4_to_rgb3(residual_4ch).unsqueeze(0).to(device, dtype)  # (1, 3, H, W)
    rgb = (rgb - vgg_mean) / vgg_std
    feat = vgg_feat(rgb)
    return feat.float().squeeze(0).cpu()  # (C, h, w)


@torch.no_grad()
def b1_encoder_features(residual_4ch: torch.Tensor, b1_model,
                         capture_h: int, capture_w: int,
                         device, dtype) -> torch.Tensor:
    """(4, 768, 1024) residual → B1 encoder bottleneck (f3, ~36, ~42).

    Resample to B1's expected (4, 1150, 1330) via cv2 area, then take
    the 4-tuple feature output and return f3 (deepest = bottleneck).
    """
    import cv2
    arr = residual_4ch.cpu().numpy()  # (4, H, W) float32
    out_chs: list[np.ndarray] = []
    for c in range(4):
        out_chs.append(cv2.resize(arr[c], (capture_w, capture_h),
                                    interpolation=cv2.INTER_AREA))
    resampled = np.stack(out_chs, axis=0)  # (4, capture_h, capture_w)
    x = torch.from_numpy(resampled).unsqueeze(0).to(device, dtype)
    feats = b1_model.encoder(x)  # 4-tuple (f0, f1, f2, f3)
    return feats[3].float().squeeze(0).cpu()  # (C, h, w)


def feature_distance_field(feat_cond: torch.Tensor,
                            feat_baseline: torch.Tensor) -> np.ndarray:
    """Per-spatial-position L2 distance between two feature maps. Returns
    (h, w) float32 numpy array."""
    if feat_cond.shape != feat_baseline.shape:
        raise ValueError(f"feature shape mismatch: {feat_cond.shape} vs "
                          f"{feat_baseline.shape}")
    diff = (feat_cond - feat_baseline)
    # L2 across channel dim
    return diff.pow(2).sum(dim=0).sqrt().numpy()


def upsample_to(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Upsample (h, w) → target_hw via bilinear (cv2)."""
    import cv2
    H, W = target_hw
    return cv2.resize(field.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


# ----------------------------- reference fields -----------------------------

def reference_R1(E_correct: torch.Tensor) -> np.ndarray:
    """E_correct (3, H, W) → magnitude (H, W) via mean over RGB."""
    return E_correct.float().mean(dim=0).cpu().numpy()


def reference_R2(C_real: torch.Tensor) -> np.ndarray:
    """Sobel(C_real_grayscale). Convert (4, H, W) packed CFA to grayscale
    by averaging R, mean(G1, G2), B then averaging RGB."""
    import cv2
    C = C_real.float().cpu().numpy()
    rgb = np.stack([C[0], 0.5 * (C[1] + C[2]), C[3]], axis=-1)  # (H, W, 3)
    gray = rgb.mean(axis=-1).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def reference_R3(session_dir: Path, row: int, k: int = TEMPORAL_MEDIAN_K
                  ) -> np.ndarray | None:
    """|C_real − temporal_median(C, ±k)|. Returns None if any neighbor row
    is missing."""
    rows = list(range(row - k, row + k + 1))
    Cs: list[torch.Tensor] = []
    for r in rows:
        path = session_dir / "Recordings" / f"frame_{r:06d}.raw"
        if not path.exists():
            return None
        Cs.append(_crop_and_resize_C(_load_packed_cfa_float01(path)))
    stacked = torch.stack(Cs, dim=0).float().numpy()  # (5, 4, H, W)
    median_C = np.median(stacked, axis=0)              # (4, H, W)
    diff = np.abs(stacked[k] - median_C)               # (4, H, W) — k-th = current
    # Single-channel via mean over CFA channels
    return diff.mean(axis=0).astype(np.float32)        # (H, W)


# ----------------------------- per-frame driver -----------------------------

def process_frame(
    sess: str,
    row: int,
    block: int,
    model: DiffusionDiagnosticUNet,
    fa_v1_ckpts: dict[int, "object"],
    sess_dirs: dict[str, Path],
    chain_keys: dict[str, list[int]],
    dc: dict,
    device: torch.device,
    dtype: torch.dtype,
    vgg_feat, vgg_mean, vgg_std,
    b1_model, b1_capture_h: int, b1_capture_w: int,
    out_dir: Path,
    log,
) -> None:
    frame_dir = out_dir / f"{sess}_f{row:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs ---
    C_real = load_phase_g_C(sess_dirs[sess], row)
    E_correct = load_phase_g_E(sess_dirs[sess], row)
    shuffled_row = deterministic_shuffled_row(row, chain_keys[sess])
    E_shuffled = load_phase_g_E(sess_dirs[sess], shuffled_row)
    other_sess = "V10" if sess == "D2" else "D2"
    cross_row = deterministic_cross_session_row(
        row, chain_keys[sess], chain_keys[other_sess])
    E_cross = load_phase_g_E(sess_dirs[other_sess], cross_row)

    keys = chain_keys[sess]
    target_idx = keys.index(row)
    source_row = keys[(target_idx + len(keys) // 4) % len(keys)]

    # F-A v1 fakes
    C_fakes: dict[int, torch.Tensor] = {}
    for step, fa_model in fa_v1_ckpts.items():
        Cf = render_C_fake(
            fa_model, sess_dirs[sess],
            source_row=source_row, target_row=row,
            device=device, dtype=dtype,
        ).cpu().float()
        C_fakes[step] = Cf

    cond_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("real_correct", C_real, E_correct),
    ]
    for step in FA_V1_CKPT_STEPS:
        cond_pairs.append((f"fake_{step//1000}k", C_fakes[step], E_correct))
    cond_pairs.append(("shuffled_E", C_real, E_shuffled))
    cond_pairs.append(("cross_session_E", C_real, E_cross))

    # --- noise (paired across conditions per (frame, t, k)) ---
    H, W = PHASE_G_INPUT_H, PHASE_G_INPUT_W
    seed = NOISE_SEED_BASE + (row * 7919 + (1 if sess == "V10" else 0))
    torch.manual_seed(seed)
    noise = torch.randn(len(T_STEPS), K_NOISE, 4, H, W,
                         device=device, dtype=torch.float32)

    # --- per-condition extraction (Phase G + scoring functions 1-5 + residual mean) ---
    per_cond: dict[str, dict] = {}
    for cond, C, E in cond_pairs:
        t0 = time.time()
        per_cond[cond] = extract_per_condition(
            model, C.to(device=device, dtype=torch.float32),
            E.to(device=device, dtype=torch.float32),
            dc, device, dtype, noise,
        )
        log(f"  [{sess} f={row}] {cond:18s} {time.time()-t0:.1f}s")

    # --- VGG and B1 features per condition ---
    vgg_feats: dict[str, torch.Tensor] = {}
    b1_feats: dict[str, torch.Tensor] = {}
    for cond in ALL_CONDS:
        residual_4ch = torch.from_numpy(per_cond[cond]["residual_4ch_mean"])
        vgg_feats[cond] = vgg_features(residual_4ch, vgg_feat,
                                         vgg_mean, vgg_std, device, dtype)
        b1_feats[cond] = b1_encoder_features(residual_4ch, b1_model,
                                               b1_capture_h, b1_capture_w,
                                               device, dtype)

    # Save per-condition score fields + residual + feature distances
    score_fields_8x: dict[tuple[str, str], np.ndarray] = {}
    for cond in ALL_CONDS:
        for fn in ("score_eps_mse", "score_eps_mae", "score_signed_residual",
                    "score_low_freq_mse", "score_high_freq_mse"):
            field_native = per_cond[cond][fn]
            score_fields_8x[(cond, fn.replace("score_", ""))] = block_average(
                field_native, COARSE_FACTOR)
            np.save(frame_dir / f"{fn}_{cond}.npy",
                    field_native.astype(np.float32))
        np.save(frame_dir / f"residual_4ch_mean_{cond}.npy",
                per_cond[cond]["residual_4ch_mean"].astype(np.float32))

    # VGG / B1 distances: per perturbed condition vs real_correct baseline.
    # Save the distance field at the feature-map's native resolution AND
    # an upsampled-to-(96, 128) version for correlation at a common grid.
    vgg_baseline = vgg_feats["real_correct"]
    b1_baseline = b1_feats["real_correct"]
    for cond in PERTURBED_CONDS:
        vgg_d = feature_distance_field(vgg_feats[cond], vgg_baseline)  # (96, 128)
        b1_d = feature_distance_field(b1_feats[cond], b1_baseline)     # (~36, 42)
        np.save(frame_dir / f"vgg_distance_native_{cond}.npy", vgg_d.astype(np.float32))
        np.save(frame_dir / f"b1_distance_native_{cond}.npy", b1_d.astype(np.float32))
        # Resample to (COARSE_H, COARSE_W) for cross-function correlations
        score_fields_8x[(cond, "vgg_distance")] = upsample_to(
            vgg_d, (COARSE_H, COARSE_W))
        score_fields_8x[(cond, "b1_distance")] = upsample_to(
            b1_d, (COARSE_H, COARSE_W))
        np.save(frame_dir / f"vgg_distance_8x_{cond}.npy",
                score_fields_8x[(cond, "vgg_distance")].astype(np.float32))
        np.save(frame_dir / f"b1_distance_8x_{cond}.npy",
                score_fields_8x[(cond, "b1_distance")].astype(np.float32))
    # For real_correct, vgg/b1 distance is 0 by definition (vs itself).
    score_fields_8x[("real_correct", "vgg_distance")] = np.zeros(
        (COARSE_H, COARSE_W), dtype=np.float32)
    score_fields_8x[("real_correct", "b1_distance")] = np.zeros(
        (COARSE_H, COARSE_W), dtype=np.float32)

    # --- reference fields (per frame) ---
    R1_native = reference_R1(E_correct)
    R2_native = reference_R2(C_real)
    R3_native = reference_R3(sess_dirs[sess], row)
    np.save(frame_dir / "reference_R1_E_magnitude_native.npy",
            R1_native.astype(np.float32))
    np.save(frame_dir / "reference_R2_scene_gradient_native.npy",
            R2_native.astype(np.float32))
    if R3_native is not None:
        np.save(frame_dir / "reference_R3_temporal_diff_native.npy",
                R3_native.astype(np.float32))
    R1_8x = block_average(R1_native, COARSE_FACTOR)
    R2_8x = block_average(R2_native, COARSE_FACTOR)
    R3_8x = (block_average(R3_native, COARSE_FACTOR)
              if R3_native is not None else None)
    np.save(frame_dir / "reference_R1_8x.npy", R1_8x.astype(np.float32))
    np.save(frame_dir / "reference_R2_8x.npy", R2_8x.astype(np.float32))
    if R3_8x is not None:
        np.save(frame_dir / "reference_R3_8x.npy", R3_8x.astype(np.float32))

    # --- per-frame manifest fragment ---
    manifest = {
        "session": sess, "row": row, "block": block,
        "shuffled_row": shuffled_row,
        "cross_session_row": int(cross_row),
        "fa_donor_row": int(source_row),
        "noise_seed": int(seed),
        "scoring_functions": list(SCORING_FUNCTIONS),
        "conditions": list(ALL_CONDS),
        "perturbed_conditions": list(PERTURBED_CONDS),
        "has_R3": R3_native is not None,
        "scoring_function_resolutions": {
            "eps_mse": [PHASE_G_INPUT_H, PHASE_G_INPUT_W],
            "eps_mae": [PHASE_G_INPUT_H, PHASE_G_INPUT_W],
            "signed_residual": [PHASE_G_INPUT_H, PHASE_G_INPUT_W],
            "low_freq_mse": [PHASE_G_INPUT_H, PHASE_G_INPUT_W],
            "high_freq_mse": [PHASE_G_INPUT_H, PHASE_G_INPUT_W],
            "vgg_distance": list(vgg_feats["real_correct"].shape[-2:]),
            "b1_distance": list(b1_feats["real_correct"].shape[-2:]),
        },
        "common_correlation_resolution": [COARSE_H, COARSE_W],
        "gaussian_blur_sigma": GAUSSIAN_BLUR_SIGMA,
        "temporal_median_window": TEMPORAL_MEDIAN_K,
    }
    (frame_dir / "frame_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    log(f"  [{sess} f={row}] DONE")


# ----------------------------- analysis -----------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    if af.size != bf.size or af.std() == 0 or bf.std() == 0:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    if af.size != bf.size or af.std() == 0 or bf.std() == 0:
        return float("nan")
    return pearson_corr(_rank_average(af), _rank_average(bf))


def _rank_average(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr, kind="stable")
    sorted_arr = arr[order]
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(sorted_arr):
        j = i
        while j + 1 < len(sorted_arr) and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks_idx = order[i:j + 1]
        ranks[ranks_idx] = avg
        i = j + 1
    return ranks


def auroc_pooled(correct: np.ndarray, wrong: np.ndarray) -> float:
    correct = correct[np.isfinite(correct)]
    wrong = wrong[np.isfinite(wrong)]
    if correct.size == 0 or wrong.size == 0:
        return float("nan")
    s_correct = -correct
    s_wrong = -wrong
    n1, n2 = s_correct.size, s_wrong.size
    all_scores = np.concatenate([s_correct, s_wrong])
    ranks = _rank_average(all_scores)
    R1 = ranks[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2
    return float(U / (n1 * n2))


def hierarchical_bootstrap_ci(values_per_frame: list[float], n_boot: int = 1000,
                              alpha: float = 0.05, seed: int = 0,
                              ) -> tuple[float, float, float]:
    """Resample frames with replacement (cluster bootstrap by frame).
    Returns (mean, q_lower, q_upper) of the bootstrap distribution.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values_per_frame, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = arr.size
    boot_means = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = arr[idx].mean()
    return (float(arr.mean()),
             float(np.quantile(boot_means, alpha / 2)),
             float(np.quantile(boot_means, 1 - alpha / 2)))


def load_frame_field(frame_dir: Path, key: str) -> np.ndarray | None:
    p = frame_dir / f"{key}.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def get_score_field_8x(frame_dir: Path, scoring: str, cond: str) -> np.ndarray | None:
    """Load score field at 96×128 common resolution. For scoring 1-5,
    block-average the saved native field; for VGG/B1, use the saved
    `_8x` field."""
    if scoring in ("eps_mse", "eps_mae", "signed_residual",
                    "low_freq_mse", "high_freq_mse"):
        native = load_frame_field(frame_dir, f"score_{scoring}_{cond}")
        if native is None:
            return None
        return block_average(native.astype(np.float64), COARSE_FACTOR)
    if scoring == "vgg_distance":
        if cond == "real_correct":
            return np.zeros((COARSE_H, COARSE_W), dtype=np.float64)
        f = load_frame_field(frame_dir, f"vgg_distance_8x_{cond}")
        return None if f is None else f.astype(np.float64)
    if scoring == "b1_distance":
        if cond == "real_correct":
            return np.zeros((COARSE_H, COARSE_W), dtype=np.float64)
        f = load_frame_field(frame_dir, f"b1_distance_8x_{cond}")
        return None if f is None else f.astype(np.float64)
    raise ValueError(f"unknown scoring: {scoring}")


def analyze(out_dir: Path, log) -> None:
    frame_manifests = sorted(out_dir.glob("*/frame_manifest.json"))
    if not frame_manifests:
        log(f"[analyze] no frame manifests under {out_dir}")
        return
    log(f"[analyze] found {len(frame_manifests)} frames")

    by_session: dict[str, list[Path]] = {}
    for fm in frame_manifests:
        m = json.loads(fm.read_text())
        by_session.setdefault(m["session"], []).append(fm.parent)

    # 1) Reference correlations (Pearson, common 96×128 grid)
    rows: list[dict] = []
    for sess in ("D2", "V10"):
        frames = by_session.get(sess, [])
        for scoring in SCORING_FUNCTIONS:
            for cond in ALL_CONDS:
                # skip vgg/b1 for real_correct (zero field by definition,
                # uninformative correlation)
                if cond == "real_correct" and scoring in ("vgg_distance",
                                                            "b1_distance"):
                    continue
                for ref_label in ("R1", "R2", "R3"):
                    pearsons: list[float] = []
                    for fdir in frames:
                        field = get_score_field_8x(fdir, scoring, cond)
                        ref = load_frame_field(fdir, f"reference_{ref_label}_8x")
                        if field is None or ref is None:
                            continue
                        if ref_label == "R1":
                            ref_full = load_frame_field(
                                fdir, "reference_R1_8x")
                        else:
                            ref_full = ref
                        # Pearson on flattened
                        pearsons.append(pearson_corr(field, ref_full))
                    pearsons_f = [v for v in pearsons if np.isfinite(v)]
                    if pearsons_f:
                        med = float(np.median(pearsons_f))
                        q1, q3 = np.percentile(pearsons_f, [25, 75])
                        rows.append({
                            "scoring_function": scoring,
                            "condition": cond,
                            "reference": ref_label,
                            "session": sess,
                            "median_pearson": med,
                            "iqr_lo": float(q1),
                            "iqr_hi": float(q3),
                            "iqr_pearson": float(q3 - q1),
                            "n_frames": len(pearsons_f),
                        })

    # CSV
    csv_path = out_dir / "reference_correlations.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scoring_function", "condition", "reference", "session",
            "median_pearson", "iqr_lo", "iqr_hi", "iqr_pearson", "n_frames",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"[analyze] wrote {csv_path} ({len(rows)} rows)")

    # Markdown summary
    md = ["# Reference correlations — scoring function × condition × reference",
          "",
          "Per-frame Pearson correlation between flattened score field and "
          "flattened reference field on the common 96×128 grid.",
          "Aggregated as median / IQR across frames per session.",
          "",
          "Pre-registered thresholds: "
          "|r| > 0.5 strongly follows; "
          "0.3 < |r| ≤ 0.5 moderately; "
          "0.1 < |r| ≤ 0.3 weakly; "
          "|r| ≤ 0.1 does not follow.",
          ""]
    # 7 × 6 × 3 grid per session (skip vgg/b1 × real_correct)
    for sess in sorted({r["session"] for r in rows}):
        md.append(f"## Session {sess}")
        md.append("")
        for ref in ("R1", "R2", "R3"):
            md.append(f"### Reference {ref}")
            md.append("")
            md.append("| scoring_function | "
                      + " | ".join(ALL_CONDS) + " |")
            md.append("|" + "---|" * (1 + len(ALL_CONDS)))
            for scoring in SCORING_FUNCTIONS:
                cells = [scoring]
                for cond in ALL_CONDS:
                    matches = [
                        r for r in rows
                        if r["session"] == sess and r["reference"] == ref
                        and r["scoring_function"] == scoring
                        and r["condition"] == cond
                    ]
                    if not matches:
                        cells.append("—")
                    else:
                        m = matches[0]["median_pearson"]
                        flag = ("**" if abs(m) > 0.5 else
                                ("" if abs(m) > 0.3 else "_"))
                        cells.append(f"{flag}{m:+.3f}{flag}")
                md.append("| " + " | ".join(cells) + " |")
            md.append("")
    md.append("**Bold** = |r| > 0.5 (strongly follows). _Italic_ = |r| ≤ 0.3.")
    md.append("")
    (out_dir / "reference_correlations_summary.md").write_text("\n".join(md))
    log(f"[analyze] wrote {out_dir / 'reference_correlations_summary.md'}")

    # 2) Cross-scoring agreement (Spearman) at fake_100k
    fake_cond = "fake_100k"
    delta_per_frame_per_scoring: dict[str, list[float]] = {
        s: [] for s in SCORING_FUNCTIONS
    }
    frame_order = sorted(frame_manifests)
    for fm in frame_order:
        fdir = fm.parent
        for scoring in SCORING_FUNCTIONS:
            field_cond = get_score_field_8x(fdir, scoring, fake_cond)
            field_real = get_score_field_8x(fdir, scoring, "real_correct")
            if field_cond is None or field_real is None:
                delta_per_frame_per_scoring[scoring].append(float("nan"))
                continue
            # Δscore = mean of (perturbed - real_correct) field
            d = float(np.mean(field_cond - field_real))
            delta_per_frame_per_scoring[scoring].append(d)

    # Spearman matrix on per-frame Δscore vectors
    spearman_matrix = np.full(
        (len(SCORING_FUNCTIONS), len(SCORING_FUNCTIONS)),
        np.nan, dtype=np.float64)
    for i, si in enumerate(SCORING_FUNCTIONS):
        for j, sj in enumerate(SCORING_FUNCTIONS):
            a = np.array(delta_per_frame_per_scoring[si], dtype=np.float64)
            b = np.array(delta_per_frame_per_scoring[sj], dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                continue
            spearman_matrix[i, j] = spearman_corr(a[mask], b[mask])

    csv_path = out_dir / "cross_scoring_agreement.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(SCORING_FUNCTIONS))
        for i, si in enumerate(SCORING_FUNCTIONS):
            w.writerow([si] + [f"{spearman_matrix[i, j]:.4f}"
                                for j in range(len(SCORING_FUNCTIONS))])
    log(f"[analyze] wrote {csv_path}")

    # Heatmap visualization
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        im = ax.imshow(spearman_matrix, cmap="RdBu", vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(len(SCORING_FUNCTIONS)))
        ax.set_yticks(range(len(SCORING_FUNCTIONS)))
        ax.set_xticklabels(SCORING_FUNCTIONS, rotation=45, ha="right")
        ax.set_yticklabels(SCORING_FUNCTIONS)
        for i in range(len(SCORING_FUNCTIONS)):
            for j in range(len(SCORING_FUNCTIONS)):
                v = spearman_matrix[i, j]
                ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "—",
                         ha="center", va="center", fontsize=8)
        ax.set_title(f"Cross-scoring Spearman agreement on Δscore "
                     f"(per-frame, {fake_cond})")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "cross_scoring_agreement.png",
                     dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"[analyze] wrote {out_dir / 'cross_scoring_agreement.png'}")
    except Exception as e:  # noqa: BLE001
        log(f"[analyze] plot failed: {e!r}")

    # 3) Summary stats: AUROC + Δscore + bootstrap CIs per scoring × condition
    summary: dict = {}
    for scoring in SCORING_FUNCTIONS:
        summary[scoring] = {}
        for sess in ("D2", "V10"):
            summary[scoring][sess] = {}
            frames_sess = by_session.get(sess, [])
            for cond in PERTURBED_CONDS:
                deltas: list[float] = []
                real_scalars: list[float] = []
                cond_scalars: list[float] = []
                for fdir in frames_sess:
                    f_cond = get_score_field_8x(fdir, scoring, cond)
                    f_real = get_score_field_8x(fdir, scoring, "real_correct")
                    if f_cond is None or f_real is None:
                        continue
                    s_cond = float(f_cond.mean())
                    s_real = float(f_real.mean())
                    cond_scalars.append(s_cond)
                    real_scalars.append(s_real)
                    deltas.append(s_cond - s_real)
                if not deltas:
                    summary[scoring][sess][cond] = None
                    continue
                deltas_np = np.array(deltas, dtype=np.float64)
                med = float(np.median(deltas_np))
                q1, q3 = np.percentile(deltas_np, [25, 75])
                # AUROC: real vs perturbed by paired-frame scalars (pooled,
                # signed-MAGNITUDE convention: lower MSE = "more correct")
                auroc = auroc_pooled(np.array(real_scalars),
                                      np.array(cond_scalars))
                m, lo, hi = hierarchical_bootstrap_ci(deltas)
                summary[scoring][sess][cond] = {
                    "n_frames": len(deltas),
                    "delta_median": med,
                    "delta_iqr_lo": float(q1),
                    "delta_iqr_hi": float(q3),
                    "delta_mean": float(deltas_np.mean()),
                    "delta_bootstrap_ci_95": [lo, hi],
                    "auroc_real_vs_cond": auroc,
                }
    summary_path = out_dir / "summary_stats.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"[analyze] wrote {summary_path}")

    # Markdown summary
    md2 = ["# Summary stats — scoring function × condition × session",
           "",
           "Per-frame Δscore = scalar(perturbed) − scalar(real_correct), "
           "where scalar = spatial mean of the score field.",
           "",
           "AUROC: pooled real-vs-cond using -score (lower = more 'real-like').",
           "Bootstrap CI (95%): hierarchical, cluster by frame.",
           ""]
    for scoring in SCORING_FUNCTIONS:
        md2.append(f"## {scoring}")
        md2.append("")
        md2.append("| session | condition | n | Δ median | Δ IQR | "
                   "Δ bootstrap 95% CI | AUROC |")
        md2.append("|---|---|---|---|---|---|---|")
        for sess in ("D2", "V10"):
            for cond in PERTURBED_CONDS:
                s = summary[scoring][sess].get(cond)
                if s is None:
                    md2.append(f"| {sess} | {cond} | — | — | — | — | — |")
                    continue
                md2.append(
                    f"| {sess} | {cond} | {s['n_frames']} | "
                    f"{s['delta_median']:+.5f} | "
                    f"[{s['delta_iqr_lo']:+.5f}, {s['delta_iqr_hi']:+.5f}] | "
                    f"[{s['delta_bootstrap_ci_95'][0]:+.5f}, "
                    f"{s['delta_bootstrap_ci_95'][1]:+.5f}] | "
                    f"{s['auroc_real_vs_cond']:.4f} |")
        md2.append("")
    (out_dir / "summary_stats.md").write_text("\n".join(md2))
    log(f"[analyze] wrote {out_dir / 'summary_stats.md'}")


# ----------------------------- rendering -----------------------------

def render_grid(out_dir: Path, frame_dir: Path, sess: str, row: int,
                kind: str, log) -> Path:
    """7 (scoring functions) × 6 (perturbed conditions) grid."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        len(SCORING_FUNCTIONS), len(PERTURBED_CONDS),
        figsize=(2.5 * len(PERTURBED_CONDS), 2.0 * len(SCORING_FUNCTIONS)),
        dpi=110, squeeze=False,
    )
    for i, scoring in enumerate(SCORING_FUNCTIONS):
        # Per-row vmax = 95th percentile of |row's score values|
        row_fields: dict[str, np.ndarray | None] = {}
        for cond in PERTURBED_CONDS:
            f = get_score_field_8x(frame_dir, scoring, cond)
            row_fields[cond] = f
        # Decide signed vs nonneg coloring for this scoring function
        if scoring in SIGNED_FUNCTIONS:
            # Use |signed_residual| for vmax
            vals = np.concatenate([np.abs(f).flatten()
                                    for f in row_fields.values() if f is not None])
            cmap = "RdBu"
            vmax = float(np.percentile(vals, 95)) if vals.size else 1.0
            vmin = -vmax
        else:
            # Render the SIGNED CONTRAST against real_correct for non-signed
            # magnitude functions (eps_mse, eps_mae, low_freq, high_freq)
            # so the grid emphasizes condition deviation. VGG/B1 distances
            # are already "vs real" so render as magnitude.
            if scoring in ("vgg_distance", "b1_distance"):
                vals = np.concatenate([f.flatten()
                                        for f in row_fields.values() if f is not None])
                cmap = "viridis"
                vmax = float(np.percentile(vals, 95)) if vals.size else 1.0
                vmin = 0.0
            else:
                # Build signed-contrast versions
                f_real = get_score_field_8x(frame_dir, scoring, "real_correct")
                contrasts = {}
                for cond in PERTURBED_CONDS:
                    if row_fields[cond] is None or f_real is None:
                        contrasts[cond] = None
                    else:
                        contrasts[cond] = row_fields[cond] - f_real
                row_fields = contrasts
                vals = np.concatenate([np.abs(f).flatten()
                                        for f in contrasts.values() if f is not None])
                cmap = "RdBu"
                vmax = float(np.percentile(vals, 95)) if vals.size else 1.0
                vmin = -vmax
        if vmax <= 0:
            vmax = 1.0; vmin = -1.0 if cmap == "RdBu" else 0.0
        for j, cond in enumerate(PERTURBED_CONDS):
            ax = axes[i, j]
            f = row_fields[cond]
            if f is None:
                ax.axis("off")
                continue
            ax.imshow(f, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            if i == 0:
                ax.set_title(cond, fontsize=8)
            if j == 0:
                ax.set_ylabel(scoring, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Scoring-function comparison — {sess} f={row}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path = out_dir / f"grid_{sess}_f{row:06d}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"[render] wrote {out_path}")
    return out_path


def render(out_dir: Path, log) -> None:
    grids_dir = out_dir / "grids"
    supp_dir = grids_dir / "supplementary"
    grids_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    frame_manifests = sorted(out_dir.glob("*/frame_manifest.json"))
    by_sess: dict[str, list[Path]] = {}
    for fm in frame_manifests:
        m = json.loads(fm.read_text())
        by_sess.setdefault(m["session"], []).append(fm.parent)
    # 1 representative + 3 supplementary per session — pick by deterministic
    # spacing in the sorted-by-row list.
    for sess in sorted(by_sess.keys()):
        frames = sorted(by_sess[sess], key=lambda p: int(p.name.split("_f")[1]))
        # Representative: middle frame.
        if not frames:
            continue
        rep_idx = len(frames) // 2
        rep = frames[rep_idx]
        m = json.loads((rep / "frame_manifest.json").read_text())
        render_grid(grids_dir, rep, m["session"], m["row"], "headline", log)
        # Supplementary: 3 evenly-spaced others, excluding rep.
        n = len(frames)
        supp_indices = [int(round((i + 1) * n / 4)) for i in range(3)]
        seen = {rep_idx}
        for si in supp_indices:
            si = min(n - 1, si)
            if si in seen:
                continue
            seen.add(si)
            f = frames[si]
            mm = json.loads((f / "frame_manifest.json").read_text())
            render_grid(supp_dir, f, mm["session"], mm["row"], "supplementary", log)


# ----------------------------- CLI -----------------------------

def cmd_extract(args, log) -> int:
    frames_spec = json.loads(args.frames_json.read_text())
    if not isinstance(frames_spec, list) or not frames_spec:
        log("[extract] FATAL: frames-json must be non-empty list")
        return 2
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    log(f"[extract] device={device} dtype={dtype} "
        f"frames={len(frames_spec)} timesteps={T_STEPS} K={K_NOISE}")
    log(f"[extract] loading Phase G ckpt {args.phase_g_ckpt}")
    pg = load_phase_g(args.phase_g_ckpt, device, dtype)
    n_params = sum(p.numel() for p in pg.parameters())
    log(f"[extract] Phase G {n_params/1e6:.2f}M params")

    log(f"[extract] loading {len(FA_V1_CKPT_STEPS)} F-A v1 ckpts")
    fa_v1: dict[int, "object"] = {}
    for step in FA_V1_CKPT_STEPS:
        ckpt = args.fa_v1_ckpt_dir / f"step_{step:08d}.pt"
        if not ckpt.exists():
            log(f"[extract] FATAL: missing {ckpt}")
            return 2
        log(f"  loading step_{step:08d}…")
        fa_v1[step] = load_fa_v1_checkpoint(ckpt, device=device, dtype=dtype)

    log(f"[extract] loading B1 from {args.b1_ckpt}")
    b1, b1_h, b1_w = load_b1_encoder(args.b1_ckpt, device, dtype)

    log("[extract] loading VGG-16 features (conv4_2)")
    vgg_feat, vgg_mean, vgg_std = load_vgg16_features(device, dtype)

    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}
    chain_keys = {sess: load_chain_keys(sess_dirs[sess])
                   for sess in ("D2", "V10")}
    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    for spec in frames_spec:
        sess = spec["session"]; row = int(spec["row"])
        block = int(spec.get("block", 0))
        log(f"[extract] === {sess} f={row} (block {block}) ===")
        try:
            process_frame(sess, row, block, pg, fa_v1, sess_dirs, chain_keys,
                            dc, device, dtype,
                            vgg_feat, vgg_mean, vgg_std,
                            b1, b1_h, b1_w,
                            args.output_dir, log)
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"[extract] FRAME FAIL {sess} f={row}: {exc!r}")
            log(traceback.format_exc())
            return 3
    log(f"[extract] DONE — {len(frames_spec)} frames")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["extract", "analyze", "render",
                                          "build-frames"], required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frames-json", type=Path,
                    help="(extract) JSON list of {session, row, block}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phase-g-ckpt", type=Path)
    ap.add_argument("--d2-dir", type=Path)
    ap.add_argument("--v10-dir", type=Path)
    ap.add_argument("--fa-v1-ckpt-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/"
                                  "phase_f/f_a_full_v1/checkpoints"))
    ap.add_argument("--b1-ckpt", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/"
                                  "fa_v2/training/B1/model_final.pt"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / f"{args.mode}_log.txt"
    log_f = open(log_path, "a")
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    if args.mode == "build-frames":
        frames = build_frame_subset()
        out = args.output_dir / "frames_all.json"
        out.write_text(json.dumps(frames, indent=2))
        log(f"[build-frames] wrote {out}  ({len(frames)} frames)")
        # Per-GPU shards (round-robin across 7 GPUs: 0/1/2/3/5/6/7)
        gpus = [0, 1, 2, 3, 5, 6, 7]
        shards: dict[int, list] = {g: [] for g in gpus}
        for i, fr in enumerate(frames):
            shards[gpus[i % len(gpus)]].append(fr)
        shard_dir = args.output_dir / "frames"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for g, lst in shards.items():
            (shard_dir / f"g{g}.json").write_text(json.dumps(lst, indent=2))
            log(f"  g{g}.json: {len(lst)} frames")
        log_f.close()
        return 0

    if args.mode == "extract":
        for required in ("frames_json", "phase_g_ckpt", "d2_dir", "v10_dir",
                          "b1_ckpt"):
            if getattr(args, required) is None:
                log(f"[extract] FATAL: --{required.replace('_','-')} required")
                return 2
        rc = cmd_extract(args, log)
        log_f.close()
        return rc

    if args.mode == "analyze":
        analyze(args.output_dir, log)
        log_f.close()
        return 0

    if args.mode == "render":
        render(args.output_dir, log)
        log_f.close()
        return 0

    log(f"[main] unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
