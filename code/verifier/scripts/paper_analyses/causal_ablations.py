"""Causal ablations for Phase G discrimination.

5 ablations × 7 conditions × 120 frames = 4200 evaluations.

Ablations:
  E1 baseline             — no modification
  E2 body-only mask       — keep INSIDE body-box, replace outside with inside-box channel mean
  E3 off-body mask        — keep OUTSIDE body-box, replace inside with inside-box channel mean
  E4 wrong-center-E       — replace real_correct's E with E from frame_idx + 5 (perturbed
                            conditions unchanged from E1)
  E5 matched-pose impostor— replace real_correct's (C, E) with similarity-matched pair
                            from same session (perturbed conditions unchanged from E1)

Storage strategy: 5×7 cell grid per frame. Cells where E4/E5 don't change the
input from E1 are recorded by REUSING the E1 score and flagged
`unchanged_from_E1: true` in the manifest so the redundancy is explicit.

Conditions per ablation:
  real_correct, fake_5k, fake_25k, fake_70k, fake_100k,
  shuffled_E, cross_session_E.

Body-box construction (operator option (iii)):
  - Compute spatial variance of debayered grayscale C via 32×32 sliding window
  - Threshold at p75 of variance values
  - Take connected component with maximum vertical extent
  - Tight bounding box expanded by 10% in each direction
  - Per-frame box dimensions saved in manifest

E5 matched-pose features (operator-spec heuristic):
  - mean grayscale (1 dim)
  - spatial variance (1 dim)
  - temporal-bin (normalized session row index, 1 dim)
  Pairwise Euclidean distance; nearest neighbor with ±10 temporal exclusion.

Frame subset: 60 D2 + 60 V10 = 120 total. Combines the existing 30+30 from
scoring_function_comparison/ with the new 30+30 (half-step offset, min-gap≥5).

Subcommands:
  build-frames-combined : write 120-frame manifest by combining existing + new
  prep                   : per-frame body-box + per-session matched-pose pairs
  run                    : Phase G inference under all 5 ablations × 7 conditions
                           per frame. Distributable via --frames-json shard.
  analyze                : AUROC + Δscore + Mann-Whitney + hierarchical bootstrap.
  render-panels          : visual ablation panel for 1 representative frame per session.
  render-interp          : manuscript interpretation doc.

Standing rules: Phase G inference-only. No held-out asset use beyond F-A v1.
No F-A v2 trainer touch. No information feedback to Phase G design.
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

FA_V1_CKPT_STEPS = (5000, 25000, 70000, 100000)
PERTURBED_CONDS = (
    [f"fake_{s//1000}k" for s in FA_V1_CKPT_STEPS]
    + ["shuffled_E", "cross_session_E"]
)
ALL_CONDS = ["real_correct"] + PERTURBED_CONDS

ABLATIONS = ("E1", "E2", "E3", "E4", "E5")

VARIANCE_WINDOW = 32       # for body-box variance computation
BODY_BOX_VAR_PCT = 75.0    # p75 threshold per spec
BODY_BOX_EXPAND = 0.10     # 10% expansion per spec
E4_OFFSET = 5              # frame_idx + 5 per spec
E5_TEMPORAL_EXCLUSION = 10  # ±10 frames excluded per spec


# ----------------------------- frame subset -----------------------------

def existing_subset_d2() -> list[int]:
    """30 D2 frames already extracted in scoring_function_comparison."""
    out = []
    for block_a, block_b in [(1298, 1698), (2796, 3196), (4294, 4694)]:
        lo, hi = block_a + 30, block_b - 30
        span = hi - lo
        for i in range(10):
            out.append(lo + int(round(i * span / 10)))
    return sorted(set(out))


def existing_subset_v10() -> list[int]:
    """30 V10 frames already extracted."""
    out = []
    for block_a, block_b in [(1110, 1360), (2345, 2595)]:
        lo, hi = block_a + 30, block_b - 30
        span = hi - lo
        for i in range(15):
            out.append(lo + int(round(i * span / 15)))
    return sorted(set(out))


def new_subset_d2() -> list[int]:
    """30 new D2 frames (half-step offset)."""
    out = []
    for block_a, block_b in [(1298, 1698), (2796, 3196), (4294, 4694)]:
        lo, hi = block_a + 30, block_b - 30
        span = hi - lo
        for i in range(10):
            out.append(lo + int(round((i + 0.5) * span / 10)))
    return sorted(set(out))


def new_subset_v10() -> list[int]:
    """30 new V10 frames."""
    out = []
    for block_a, block_b in [(1110, 1360), (2345, 2595)]:
        lo, hi = block_a + 30, block_b - 30
        span = hi - lo
        for i in range(15):
            out.append(lo + int(round((i + 0.5) * span / 15)))
    return sorted(set(out))


def all_120_frames() -> list[dict]:
    out = []
    for r in sorted(set(existing_subset_d2() + new_subset_d2())):
        out.append({"session": "D2", "row": int(r)})
    for r in sorted(set(existing_subset_v10() + new_subset_v10())):
        out.append({"session": "V10", "row": int(r)})
    return out


# ----------------------------- IO -----------------------------

def load_phase_g_C(session_dir: Path, row: int) -> torch.Tensor:
    return _crop_and_resize_C(_load_packed_cfa_float01(
        session_dir / "Recordings" / f"frame_{row:06d}.raw"))


def load_phase_g_E(session_dir: Path, row: int) -> torch.Tensor:
    return _resize_E_to_target(load_emission_at(
        session_dir / "derived" / "Emissions" / f"tile_{row:06d}.png",
        EMISSION_NATIVE_H, EMISSION_NATIVE_W))


def load_chain_keys(session_dir: Path) -> list[int]:
    import csv as _csv
    out = []
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
    n = len(keys)
    idx = keys.index(this_row)
    return keys[(idx + n // 2) % n]


def deterministic_cross_session_row(this_row: int, this_keys: list[int],
                                     other_keys: list[int]) -> int:
    idx_self = this_keys.index(this_row)
    n_self, n_other = len(this_keys), len(other_keys)
    pct = idx_self / max(n_self - 1, 1)
    return other_keys[min(n_other - 1, int(round(pct * (n_other - 1))))]


# ----------------------------- body box -----------------------------

def C_to_grayscale_native(C_real: torch.Tensor) -> np.ndarray:
    """(4, H, W) packed-CFA float [0, 1] → (H, W) grayscale via mean of
    (R, mean(G1, G2), B)."""
    arr = C_real.cpu().numpy()
    return np.stack([arr[0], 0.5 * (arr[1] + arr[2]), arr[3]],
                     axis=0).mean(axis=0).astype(np.float32)


def spatial_variance(gray: np.ndarray, window: int = VARIANCE_WINDOW
                      ) -> np.ndarray:
    """Sliding-window variance of `gray` with a `window × window` mean
    filter. Returns a (H, W) array. Implemented as
    Var(X) = E[X²] − (E[X])² via box-filter convolution."""
    import cv2
    # OpenCV's boxFilter computes a uniform mean filter, with reflect padding.
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window, window),
                          normalize=True, borderType=cv2.BORDER_REFLECT)
    sqr_mean = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(window, window),
                              normalize=True, borderType=cv2.BORDER_REFLECT)
    var = sqr_mean - mean * mean
    return np.clip(var, 0.0, None)


def compute_body_box(C_real: torch.Tensor) -> tuple[int, int, int, int]:
    """Per spec (option iii): grayscale variance → p75 threshold → largest
    vertical-extent connected component → bounding box expanded 10%.

    Returns (y0, y1, x0, x1) in Phase G native (768, 1024) coords. Raises
    ValueError if the construction is degenerate (no connected component
    or single component covering full frame)."""
    import cv2
    gray = C_to_grayscale_native(C_real)
    var = spatial_variance(gray, VARIANCE_WINDOW)
    thr = float(np.percentile(var, BODY_BOX_VAR_PCT))
    binary = (var > thr).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if n_lab <= 1:
        raise ValueError("body-box: no connected components above threshold")
    # Pick the component with maximum vertical extent (height of bbox).
    # stats columns: cv2.CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA.
    best_label = -1
    best_h = -1
    for label_i in range(1, n_lab):  # skip background label 0
        h = int(stats[label_i, cv2.CC_STAT_HEIGHT])
        w = int(stats[label_i, cv2.CC_STAT_WIDTH])
        # Sanity: reject components that span the whole frame
        if h >= PHASE_G_INPUT_H * 0.95 and w >= PHASE_G_INPUT_W * 0.95:
            continue
        if h > best_h:
            best_h = h
            best_label = label_i
    if best_label < 0:
        raise ValueError("body-box: only full-frame components found")
    y0 = int(stats[best_label, cv2.CC_STAT_TOP])
    x0 = int(stats[best_label, cv2.CC_STAT_LEFT])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    y1 = y0 + h
    x1 = x0 + w
    # 10% expansion in each direction
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    new_h = h * (1.0 + 2 * BODY_BOX_EXPAND)
    new_w = w * (1.0 + 2 * BODY_BOX_EXPAND)
    y0e = max(0, int(round(cy - new_h / 2)))
    y1e = min(PHASE_G_INPUT_H, int(round(cy + new_h / 2)))
    x0e = max(0, int(round(cx - new_w / 2)))
    x1e = min(PHASE_G_INPUT_W, int(round(cx + new_w / 2)))
    if y1e <= y0e or x1e <= x0e:
        raise ValueError(
            f"body-box: degenerate after expansion: y={y0e}:{y1e} x={x0e}:{x1e}")
    return (y0e, y1e, x0e, x1e)


def inside_box_channel_mean(arr: torch.Tensor, box: tuple[int, int, int, int]
                             ) -> torch.Tensor:
    """Per-channel mean of values INSIDE the body box. Used as the
    replacement fill so masking introduces no new content beyond the
    inside-box statistics. arr shape (C, H, W); returns (C,) tensor."""
    y0, y1, x0, x1 = box
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"degenerate box: {box}")
    return arr[:, y0:y1, x0:x1].mean(dim=(1, 2))


def apply_body_only_mask(arr: torch.Tensor, box: tuple[int, int, int, int]
                          ) -> torch.Tensor:
    """Keep INSIDE box; replace OUTSIDE with per-channel inside-box mean.
    arr shape (C, H, W)."""
    y0, y1, x0, x1 = box
    fill = inside_box_channel_mean(arr, box)  # (C,)
    out = arr.clone()
    out[..., :y0, :] = fill.view(-1, 1, 1).expand(out.shape[0], y0, out.shape[2])
    out[..., y1:, :] = fill.view(-1, 1, 1).expand(out.shape[0],
                                                    out.shape[1] - y1,
                                                    out.shape[2])
    out[..., y0:y1, :x0] = fill.view(-1, 1, 1).expand(out.shape[0],
                                                       y1 - y0, x0)
    out[..., y0:y1, x1:] = fill.view(-1, 1, 1).expand(out.shape[0],
                                                       y1 - y0,
                                                       out.shape[2] - x1)
    return out


def apply_off_body_mask(arr: torch.Tensor, box: tuple[int, int, int, int]
                         ) -> torch.Tensor:
    """Keep OUTSIDE box; replace INSIDE with per-channel inside-box mean.
    arr shape (C, H, W)."""
    y0, y1, x0, x1 = box
    fill = inside_box_channel_mean(arr, box)
    out = arr.clone()
    out[..., y0:y1, x0:x1] = fill.view(-1, 1, 1).expand(out.shape[0],
                                                         y1 - y0, x1 - x0)
    return out


# ----------------------------- matched-pose -----------------------------

def session_features(session_dir: Path, rows: list[int]) -> np.ndarray:
    """Build a (n_rows, 3) feature matrix:
       [mean_gray, spatial_variance, temporal_bin]
    per spec heuristic."""
    feats = np.zeros((len(rows), 3), dtype=np.float64)
    n = len(rows)
    for i, r in enumerate(rows):
        gray = C_to_grayscale_native(load_phase_g_C(session_dir, r))
        feats[i, 0] = float(gray.mean())
        feats[i, 1] = float(gray.var())
        feats[i, 2] = float(i / max(n - 1, 1))   # temporal-bin in [0, 1]
    return feats


def find_matched_pose_in_subset(this_row: int,
                                  subset_rows: list[int],
                                  feats: np.ndarray,
                                  exclude_radius: int = E5_TEMPORAL_EXCLUSION
                                  ) -> int:
    """Nearest-neighbor matching within a SUBSET candidate pool.
    Operator decision 2026-05-05: candidate pool = the 120-frame test
    subset (instead of full session). This makes E5 a within-test-set
    similarity baseline; cheaper to compute and matches the recording
    'hard-negative pose-similar' intent.

    `this_row` is the test frame's session row.
    `subset_rows` is the list of subset rows for this session (sorted).
    `feats[i]` is the feature vector for `subset_rows[i]`.
    Exclusion: subset rows within ±exclude_radius of `this_row` in
    actual session frame distance (NOT subset-index distance, since
    subset is sparse).

    Sparse-pool fallback: if the entire subset is excluded, shrink
    exclude_radius until a candidate is admissible. Last resort
    (n=1 subset): return the index of this_row itself.

    Returns the index in `subset_rows` of the matched frame.
    """
    n = len(subset_rows)
    if n <= 1:
        return 0
    this_idx = subset_rows.index(this_row)
    base_dists = np.linalg.norm(feats - feats[this_idx], axis=1)
    radius = exclude_radius
    while radius >= 0:
        dists = base_dists.copy()
        for j, candidate_row in enumerate(subset_rows):
            if abs(candidate_row - this_row) <= radius:
                dists[j] = np.inf
        if np.isfinite(dists).any():
            return int(np.argmin(dists))
        radius -= 1
    return int(np.argmin(base_dists))


# ----------------------------- Phase G -----------------------------

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


@torch.no_grad()
def phase_g_score_scalar(
    model: DiffusionDiagnosticUNet,
    C: torch.Tensor,            # (4, H, W) float32 [0, 1]
    E: torch.Tensor,            # (3, H, W) float32 [0, 1]
    dc: dict,
    device: torch.device,
    dtype: torch.dtype,
    noise: torch.Tensor,        # (n_t, K, 4, H, W) float32, paired across conditions
    timesteps: tuple[int, ...] = T_STEPS,
) -> float:
    """Returns the SAME scalar Phase G uses for AUROC: mean of
    (eps_pred − noise)² over channel/H/W, averaged across (timestep × K).
    Matches eval_diffusion_diagnostic.py:362 then per-frame mean over
    (t, K)."""
    H, W = C.shape[-2:]
    K_ = noise.shape[1]
    n_t = len(timesteps)
    accum = 0.0
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
        diff = eps_pred.float() - noise[ti].float()
        # mean over (channel, H, W) per sample, summed across K
        mse_each = diff.pow(2).mean(dim=(1, 2, 3))   # (K,)
        accum += float(mse_each.sum().item())
    return accum / n_samples


# ----------------------------- per-frame ablation runner -----------------------------

def process_frame(
    sess: str,
    row: int,
    block: int,
    model: DiffusionDiagnosticUNet,
    fa_v1_ckpts: dict[int, "object"],
    sess_dirs: dict[str, Path],
    chain_keys: dict[str, list[int]],
    body_boxes: dict[tuple[str, int], tuple[int, int, int, int]],
    matched_pose_rows: dict[tuple[str, int], int],
    dc: dict,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
    log,
) -> dict:
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

    # E4: E from frame_idx + 5 within the same session.
    e4_idx = (target_idx + E4_OFFSET) % len(keys)
    e4_row = keys[e4_idx]
    E_e4 = load_phase_g_E(sess_dirs[sess], e4_row)

    # E5: matched-pose pair (C, E) from same session.
    matched_row = matched_pose_rows[(sess, row)]
    C_matched = load_phase_g_C(sess_dirs[sess], matched_row)
    E_matched = load_phase_g_E(sess_dirs[sess], matched_row)

    # Body box for E2/E3.
    box = body_boxes[(sess, row)]

    # F-A v1 fakes
    C_fakes: dict[int, torch.Tensor] = {}
    for step, fa_model in fa_v1_ckpts.items():
        Cf = render_C_fake(
            fa_model, sess_dirs[sess],
            source_row=source_row, target_row=row,
            device=device, dtype=dtype,
        ).cpu().float()
        C_fakes[step] = Cf

    # --- noise (paired across all 5 ablations × 7 conditions) ---
    H, W = PHASE_G_INPUT_H, PHASE_G_INPUT_W
    seed = NOISE_SEED_BASE + (row * 7919 + (1 if sess == "V10" else 0))
    torch.manual_seed(seed)
    noise = torch.randn(len(T_STEPS), K_NOISE, 4, H, W,
                         device=device, dtype=torch.float32)

    # --- build condition base inputs (per condition's E1 baseline (C, E) pair) ---
    cond_base: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "real_correct": (C_real, E_correct),
    }
    for step in FA_V1_CKPT_STEPS:
        cond_base[f"fake_{step//1000}k"] = (C_fakes[step], E_correct)
    cond_base["shuffled_E"] = (C_real, E_shuffled)
    cond_base["cross_session_E"] = (C_real, E_cross)

    # --- per-(ablation, condition) (C, E) construction ---
    def inputs_for(ablation: str, cond: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Returns (C_input, E_input, unchanged_from_E1).

        Per the design commitment in the module docstring:
          E1: no modification.
          E2: body-only mask on (C_base, E_base) for all conditions.
          E3: off-body mask on (C_base, E_base) for all conditions.
          E4: only modifies real_correct (E → E_e4); other conditions
              are unchanged from E1.
          E5: only modifies real_correct ((C, E) → (C_matched, E_matched));
              other conditions unchanged from E1.
        """
        C_b, E_b = cond_base[cond]
        if ablation == "E1":
            return C_b, E_b, False  # E1 by definition is the reference
        if ablation == "E2":
            return apply_body_only_mask(C_b, box), apply_body_only_mask(E_b, box), False
        if ablation == "E3":
            return apply_off_body_mask(C_b, box), apply_off_body_mask(E_b, box), False
        if ablation == "E4":
            if cond == "real_correct":
                return C_b, E_e4, False
            return C_b, E_b, True   # unchanged from E1
        if ablation == "E5":
            if cond == "real_correct":
                return C_matched, E_matched, False
            return C_b, E_b, True   # unchanged from E1
        raise ValueError(f"unknown ablation {ablation}")

    # --- per-(ablation, condition) Phase G score ---
    scalars: dict[tuple[str, str], float] = {}
    unchanged_flag: dict[tuple[str, str], bool] = {}
    e1_scalar_cache: dict[str, float] = {}

    # Compute E1 first so E4/E5 unchanged cells can reuse.
    for ablation in ABLATIONS:
        for cond in ALL_CONDS:
            C_in, E_in, unchanged = inputs_for(ablation, cond)
            unchanged_flag[(ablation, cond)] = unchanged
            if unchanged and cond in e1_scalar_cache:
                scalars[(ablation, cond)] = e1_scalar_cache[cond]
                continue
            t0 = time.time()
            score = phase_g_score_scalar(
                model, C_in.to(device=device, dtype=torch.float32),
                E_in.to(device=device, dtype=torch.float32),
                dc, device, dtype, noise,
            )
            scalars[(ablation, cond)] = score
            if ablation == "E1":
                e1_scalar_cache[cond] = score
            log(f"  [{sess} f={row}] {ablation}/{cond:18s} "
                f"score={score:.6f} {'(reused E1)' if unchanged else ''} "
                f"{time.time()-t0:.2f}s")

    # --- save per-frame manifest ---
    manifest = {
        "session": sess,
        "row": row,
        "block": block,
        "shuffled_row": shuffled_row,
        "cross_session_row": int(cross_row),
        "fa_donor_row": int(source_row),
        "e4_row": int(e4_row),
        "matched_pose_row": int(matched_row),
        "noise_seed": int(seed),
        "body_box": list(box),
        "scalars": {
            f"{ablation}|{cond}": scalars[(ablation, cond)]
            for ablation in ABLATIONS for cond in ALL_CONDS
        },
        "unchanged_from_E1": {
            f"{ablation}|{cond}": unchanged_flag[(ablation, cond)]
            for ablation in ABLATIONS for cond in ALL_CONDS
        },
    }
    (frame_dir / "ablations_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    log(f"  [{sess} f={row}] DONE")
    return manifest


# ----------------------------- analysis -----------------------------

def auroc_pooled(real_scalars: np.ndarray, perturbed_scalars: np.ndarray) -> float:
    """Mann-Whitney AUROC with score = -MSE (lower MSE = more 'real-like').
    Same convention as eval_diffusion_diagnostic.py:88."""
    real = real_scalars[np.isfinite(real_scalars)]
    pert = perturbed_scalars[np.isfinite(perturbed_scalars)]
    if real.size == 0 or pert.size == 0:
        return float("nan")
    s_real = -real
    s_pert = -pert
    n1, n2 = s_real.size, s_pert.size
    all_scores = np.concatenate([s_real, s_pert])
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


def hierarchical_bootstrap(values: list[float], n_boot: int = 1000,
                           alpha: float = 0.05, seed: int = 0
                           ) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
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


def mw_effect_size(real: np.ndarray, perturbed: np.ndarray) -> float:
    """Mann-Whitney effect size r = (2*AUROC − 1). Positive = perturbed
    higher MSE than real (= positive discrimination)."""
    auc = auroc_pooled(real, perturbed)
    if not np.isfinite(auc):
        return float("nan")
    return 2 * auc - 1


def regime(auroc: float, delta_pct: float) -> str:
    """Pre-registered thresholds."""
    if not np.isfinite(auroc):
        return "nan"
    if auroc < 0.85 or abs(delta_pct) > 50:
        return "strong_dependence"
    if auroc < 0.95 or 20 <= abs(delta_pct) <= 50:
        return "moderate_dependence"
    return "weak_dependence"


def analyze(out_dir: Path, log) -> None:
    frame_jsons = sorted(out_dir.glob("*/ablations_manifest.json"))
    if not frame_jsons:
        log("[analyze] no per-frame manifests")
        return
    log(f"[analyze] found {len(frame_jsons)} frames")

    by_session: dict[str, list[dict]] = {}
    for fj in frame_jsons:
        m = json.loads(fj.read_text())
        by_session.setdefault(m["session"], []).append(m)

    rows: list[dict] = []
    # E1 baseline real_correct scalars by session, used for delta_degradation
    e1_real_scalars: dict[str, np.ndarray] = {}
    e1_delta_per_cond: dict[tuple[str, str], np.ndarray] = {}
    for sess in by_session:
        e1_real = np.array([
            m["scalars"][f"E1|real_correct"] for m in by_session[sess]
        ], dtype=np.float64)
        e1_real_scalars[sess] = e1_real
        for cond in PERTURBED_CONDS:
            e1_pert = np.array([
                m["scalars"][f"E1|{cond}"] for m in by_session[sess]
            ], dtype=np.float64)
            e1_delta_per_cond[(sess, cond)] = e1_pert - e1_real

    for ablation in ABLATIONS:
        for cond in PERTURBED_CONDS:
            for sess in sorted(by_session.keys()):
                frames = by_session[sess]
                real_arr = np.array([
                    m["scalars"][f"{ablation}|real_correct"] for m in frames
                ], dtype=np.float64)
                pert_arr = np.array([
                    m["scalars"][f"{ablation}|{cond}"] for m in frames
                ], dtype=np.float64)
                deltas = (pert_arr - real_arr).tolist()
                auc = auroc_pooled(real_arr, pert_arr)
                ci_mean, ci_lo, ci_hi = hierarchical_bootstrap(deltas)
                med = float(np.median([d for d in deltas if np.isfinite(d)]))
                q1, q3 = np.percentile(
                    [d for d in deltas if np.isfinite(d)], [25, 75]) \
                    if any(np.isfinite(deltas)) else (float("nan"), float("nan"))
                effect = mw_effect_size(real_arr, pert_arr)
                # AUROC degradation vs E1 same (sess, cond)
                auc_e1 = None
                if ablation != "E1":
                    real_e1 = e1_real_scalars[sess]
                    pert_e1 = np.array([
                        m["scalars"][f"E1|{cond}"] for m in frames
                    ], dtype=np.float64)
                    auc_e1 = auroc_pooled(real_e1, pert_e1)
                auc_degradation = (
                    None if auc_e1 is None or not np.isfinite(auc_e1)
                    else 100.0 * (1.0 - auc / auc_e1) if auc_e1 > 0
                    else float("nan")
                )
                # Δscore degradation
                e1_med_delta = float(np.median(e1_delta_per_cond[(sess, cond)]))
                delta_degradation = (
                    None if ablation == "E1"
                    else 100.0 * (1.0 - med / e1_med_delta) if e1_med_delta != 0
                    else float("nan")
                )
                regime_label = regime(
                    auc,
                    delta_degradation if delta_degradation is not None else 0.0,
                )
                rows.append({
                    "ablation": ablation,
                    "condition": cond,
                    "session": sess,
                    "n_frames": int(real_arr.size),
                    "auroc": float(auc),
                    "auroc_ci_low": ci_lo if False else None,
                    "auroc_ci_high": ci_hi if False else None,
                    "median_delta_score": med,
                    "delta_iqr_low": float(q1),
                    "delta_iqr_high": float(q3),
                    "mw_effect_size": effect,
                    "delta_bootstrap_mean": ci_mean,
                    "delta_bootstrap_ci_low": ci_lo,
                    "delta_bootstrap_ci_high": ci_hi,
                    "auroc_degradation_pct": auc_degradation,
                    "delta_degradation_pct": delta_degradation,
                    "regime_classification": regime_label,
                })

    csv_path = out_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    log(f"[analyze] wrote {csv_path} ({len(rows)} rows)")

    # Markdown summary
    md = ["# Causal-ablation summary table",
          "",
          "Pre-registered thresholds:",
          "- AUROC > 0.95 = weak dependence",
          "- AUROC 0.85–0.95 = moderate",
          "- AUROC < 0.85 = strong dependence",
          "- |Δ| reduction > 50% = strong; 20–50% = moderate; <20% = weak.",
          "",
          "## AUROC × condition × ablation (per session)",
          ""]
    for sess in sorted(by_session):
        md.append(f"### {sess}")
        md.append("")
        md.append(
            "| ablation | "
            + " | ".join(PERTURBED_CONDS) + " |")
        md.append("|" + "---|" * (1 + len(PERTURBED_CONDS)))
        for ablation in ABLATIONS:
            cells = [ablation]
            for cond in PERTURBED_CONDS:
                m = next((r for r in rows
                           if r["session"] == sess
                           and r["ablation"] == ablation
                           and r["condition"] == cond), None)
                if m is None:
                    cells.append("—")
                else:
                    flag = m["regime_classification"]
                    code = ({"strong_dependence": "**strong**",
                              "moderate_dependence": "_moderate_",
                              "weak_dependence": "weak",
                              "nan": "—"}.get(flag, flag))
                    cells.append(f"{m['auroc']:.3f} {code}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    (out_dir / "ablation_summary.md").write_text("\n".join(md))
    log(f"[analyze] wrote {out_dir / 'ablation_summary.md'}")

    # Quick interpretation doc
    interp = ["# Manuscript interpretation — Phase G causal ablations",
              "",
              "## Pre-registered interpretation thresholds (frozen before analysis)",
              "",
              "AUROC degradation regimes:",
              "- `strong_dependence`: AUROC < 0.85 OR median |Δscore| reduction > 50%",
              "- `moderate_dependence`: AUROC 0.85–0.95 OR Δ reduction 20–50%",
              "- `weak_dependence`: AUROC > 0.95 AND Δ reduction < 20%",
              "",
              "Combined ablation interpretations (pre-registered):",
              "- E2 maintains > 0.95: body-only evidence sufficient. Chain-coupling on subject survives.",
              "- E2 drops: body region NOT where discrimination lives. Chain-coupling-on-subject claim weakens.",
              "- E3 maintains > 0.95: off-body evidence sufficient. Discrimination operates without body.",
              "- E3 drops: off-body region needed.",
              "- E4 drops: Phase G discriminates on chain-coupled (C, E) match.",
              "- E4 maintains: Phase G reads 'E plausible for session'.",
              "- E5 drops: discrimination depends on pose cue.",
              "- E5 maintains: discrimination beyond pose alone.",
              "",
              "## Per-ablation findings",
              "",
              "(See `ablation_summary.md` for the AUROC × condition table.)",
              "",
              "Per-(ablation, perturbed-condition, session) regime classifications drawn directly from the AUROC and Δscore degradation values in `ablation_table.csv`. The pre-registered combined-interpretation logic is to be applied to the table values without post-hoc adjustment.",
              "",
              "## Updated priors (recorded before analysis, not used to gate interpretation)",
              "",
              "Visual-grids review noted body-region anti-discriminative on F-A v1 across ε-MSE / VGG / B1 scoring functions. Pre-analysis priors:",
              "- E2 (body-only): expected to drop substantially. Prior: AUROC 0.5–0.8.",
              "- E3 (off-body): expected to retain near 1.0. Prior: AUROC > 0.95.",
              "- E4: expected to drop.",
              "- E5: unclear (body anti-discriminative may mean pose isn't dominant).",
              "",
              "## What this experiment establishes",
              "",
              "- Whether Phase G's AUROC = 1.000 baseline depends causally on body-region content (E2 vs E3 contrast).",
              "- Whether discrimination is specific to chain-coupled (C, E) match (E4) vs plausible-E-for-session.",
              "- Whether discrimination requires pose cues (E5) or operates beyond them.",
              "",
              "## What this experiment does NOT establish",
              "",
              "- Whether Phase G's mechanism would generalize to harder attackers (F-A v1 is the easy target; F-A v2 may produce different spatial signatures).",
              "- Whether the patterns generalize beyond same-rig D2+V10.",
              "- Whether the underlying optical-coupling primitive is chain-bound at the per-edge level (separate experiment).",
              "",
              "## Standing rules acknowledged",
              "",
              "Phase G inference-only. No held-out asset use beyond F-A v1. No F-A v2 trainer touch. No information from this experiment feeds back into Phase G design or F-A v2 training. Pre-registered thresholds frozen — no post-hoc adjustment.",
              ""]
    (out_dir / "manuscript_interpretation.md").write_text("\n".join(interp))
    log(f"[analyze] wrote {out_dir / 'manuscript_interpretation.md'}")


# ----------------------------- main / CLI -----------------------------

def cmd_prep(args, log) -> int:
    """Pre-compute body boxes per frame + matched-pose pairs per session.
    Outputs: <out>/body_boxes.json + <out>/matched_pose.json"""
    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}
    chain_keys = {sess: load_chain_keys(sess_dirs[sess])
                   for sess in ("D2", "V10")}
    frames = all_120_frames()
    log(f"[prep] {len(frames)} frames; computing body boxes…")
    boxes: dict[str, dict[str, list[int]]] = {"D2": {}, "V10": {}}
    for spec in frames:
        sess = spec["session"]
        row = int(spec["row"])
        try:
            C_real = load_phase_g_C(sess_dirs[sess], row)
            box = compute_body_box(C_real)
        except ValueError as e:
            log(f"[prep] FATAL: body-box failed for {sess} f={row}: {e}")
            return 3
        boxes[sess][str(row)] = list(box)
        log(f"  {sess} f={row}: box={box}")
    (args.output_dir / "body_boxes.json").write_text(
        json.dumps(boxes, indent=2))

    # Matched-pose: candidate pool = the 120-frame test subset
    # (operator decision 2026-05-05). Per-session subset features
    # computed once; ±10 temporal exclusion measured in actual session
    # frame distance, not subset-index distance. Matched row stored
    # directly in matched_pose.json (no index lookup required at run
    # time).
    matched: dict[str, dict[str, int]] = {"D2": {}, "V10": {}}
    for sess in ("D2", "V10"):
        subset_rows = sorted({int(s["row"]) for s in frames
                                if s["session"] == sess})
        log(f"[prep] {sess}: computing features over {len(subset_rows)} "
            f"subset rows (candidate pool = test subset)…")
        feats = session_features(sess_dirs[sess], subset_rows)
        for row in subset_rows:
            match_idx = find_matched_pose_in_subset(row, subset_rows, feats)
            matched_row = subset_rows[match_idx]
            matched[sess][str(row)] = int(matched_row)
            log(f"  {sess} f={row} → matched f={matched_row}")
    matched["_meta"] = {
        "candidate_pool": "120-frame test subset (per-session)",
        "exclusion_radius_session_frames": E5_TEMPORAL_EXCLUSION,
        "rationale": "Operator decision 2026-05-05: matched-pose impostor "
                      "is a hard-negative pose-similar baseline; the 120-"
                      "frame subset gives plenty of similarity options "
                      "without paying the full-session I/O cost. This "
                      "makes E5 a within-test-set similarity baseline.",
    }
    (args.output_dir / "matched_pose.json").write_text(
        json.dumps(matched, indent=2))
    log("[prep] DONE — body_boxes.json + matched_pose.json written")
    return 0


def cmd_run(args, log) -> int:
    frames_spec = json.loads(args.frames_json.read_text())
    if not isinstance(frames_spec, list) or not frames_spec:
        log("[run] FATAL: frames-json must be non-empty list")
        return 2
    boxes_raw = json.loads((args.prep_dir / "body_boxes.json").read_text())
    body_boxes = {
        (sess, int(row)): tuple(box)
        for sess, by_row in boxes_raw.items()
        for row, box in by_row.items()
    }
    matched_raw = json.loads((args.prep_dir / "matched_pose.json").read_text())
    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}
    chain_keys = {sess: load_chain_keys(sess_dirs[sess])
                   for sess in ("D2", "V10")}
    # Map (sess, row) → matched_row directly (the per-session subset
    # candidate-pool decision means we no longer need the chain-keys
    # index). Skip the _meta key written by cmd_prep.
    matched_pose_rows: dict[tuple[str, int], int] = {}
    for sess, by_row in matched_raw.items():
        if sess.startswith("_"):
            continue
        for row, matched_row in by_row.items():
            matched_pose_rows[(sess, int(row))] = int(matched_row)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    log(f"[run] device={device} dtype={dtype} frames={len(frames_spec)}")

    log(f"[run] loading Phase G ckpt {args.phase_g_ckpt}")
    pg = load_phase_g(args.phase_g_ckpt, device, dtype)

    log(f"[run] loading {len(FA_V1_CKPT_STEPS)} F-A v1 ckpts")
    fa_v1: dict[int, "object"] = {}
    for step in FA_V1_CKPT_STEPS:
        ckpt = args.fa_v1_ckpt_dir / f"step_{step:08d}.pt"
        if not ckpt.exists():
            log(f"[run] FATAL: missing {ckpt}")
            return 2
        log(f"  loading step_{step:08d}…")
        fa_v1[step] = load_fa_v1_checkpoint(ckpt, device=device, dtype=dtype)

    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    for spec in frames_spec:
        sess = spec["session"]; row = int(spec["row"])
        block = int(spec.get("block", 0))
        if (sess, row) not in body_boxes:
            log(f"[run] FATAL: body box missing for {sess} f={row}")
            return 3
        if (sess, row) not in matched_pose_rows:
            log(f"[run] FATAL: matched-pose missing for {sess} f={row}")
            return 3
        log(f"[run] === {sess} f={row} (block {block}) ===")
        try:
            process_frame(
                sess, row, block, pg, fa_v1, sess_dirs, chain_keys,
                body_boxes, matched_pose_rows, dc, device, dtype,
                args.output_dir, log,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"[run] FRAME FAIL {sess} f={row}: {exc!r}")
            log(traceback.format_exc())
            return 4
    log(f"[run] DONE — {len(frames_spec)} frames")
    return 0


def cmd_build_frames_combined(args, log) -> int:
    frames = all_120_frames()
    out = args.output_dir / "frames_120.json"
    out.write_text(json.dumps(frames, indent=2))
    log(f"[build-frames-combined] wrote {out} ({len(frames)} frames)")
    gpus = [0, 1, 2, 3, 5, 6, 7]
    shards = {g: [] for g in gpus}
    for i, fr in enumerate(frames):
        shards[gpus[i % len(gpus)]].append(fr)
    shard_dir = args.output_dir / "shards_run"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for g, lst in shards.items():
        (shard_dir / f"g{g}.json").write_text(json.dumps(lst, indent=2))
        log(f"  g{g}.json: {len(lst)} frames")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["build-frames-combined", "prep", "run",
                                          "analyze", "render-panels",
                                          "render-interp"], required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prep-dir", type=Path,
                    help="(run) dir containing body_boxes.json + matched_pose.json")
    ap.add_argument("--frames-json", type=Path, help="(run) per-GPU shard JSON")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phase-g-ckpt", type=Path)
    ap.add_argument("--d2-dir", type=Path)
    ap.add_argument("--v10-dir", type=Path)
    ap.add_argument("--fa-v1-ckpt-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/"
                                  "phase_f/f_a_full_v1/checkpoints"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.mode}_log.txt"
    log_f = open(log_path, "a")
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    if args.mode == "build-frames-combined":
        rc = cmd_build_frames_combined(args, log)
    elif args.mode == "prep":
        for required in ("d2_dir", "v10_dir"):
            if getattr(args, required) is None:
                log(f"[prep] FATAL: --{required.replace('_','-')} required")
                rc = 2; log_f.close(); return rc
        rc = cmd_prep(args, log)
    elif args.mode == "run":
        for required in ("frames_json", "phase_g_ckpt", "d2_dir", "v10_dir",
                          "prep_dir"):
            if getattr(args, required) is None:
                log(f"[run] FATAL: --{required.replace('_','-')} required")
                rc = 2; log_f.close(); return rc
        rc = cmd_run(args, log)
    elif args.mode == "analyze":
        analyze(args.output_dir, log)
        rc = 0
    elif args.mode in ("render-panels", "render-interp"):
        log(f"[{args.mode}] not yet implemented; analyze writes "
            f"manuscript_interpretation.md as an early draft.")
        rc = 0
    else:
        log(f"[main] unknown mode {args.mode}")
        rc = 2
    log_f.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
