"""Step 2 of the verifier-heatmap two-step.

All four answers confirmed: averaged (K=4 × 5-timestep), 8× only
    (96×128), both magnitude and signed contrast, same 4+4 stratified
    subset.

    ADDITION — projection-footprint quantification:
      footprint_fraction = sum(|MSE_wrong − MSE_correct| inside footprint) /
                           sum(|MSE_wrong − MSE_correct| over full field)
      footprint_mask = (E_rendered_8x > threshold)
      thresholds: percentile-50, p75, p90 of E_rendered_8x.

Conditions per frame (7 total):
    real_correct   — (C_real, E_correct)
    fake_5k        — (C_fake_step_5000,   E_correct)  ← E_target = E_correct
    fake_25k       — (C_fake_step_25000,  E_correct)
    fake_70k       — (C_fake_step_70000,  E_correct)
    fake_100k      — (C_fake_step_100000, E_correct)
    shuffled_E     — (C_real, E_shuffled)         deterministic far-row from same session
    cross_session_E— (C_real, E_cross)            deterministic same-percentile row from other session

Frame set: 4 D2 + 4 V10 stratified subset (1 low + 2 near-median + 1 high
gap-quantile per session — same as T2A / T2B comparison panels).

Each (frame, condition) pair scored at 5 timesteps × K=4 noise samples =
20 forwards. SAME paired noise across conditions for each (frame, t, k)
— makes signed contrast `MSE_perturbed − MSE_correct` a proper paired
comparison.

Outputs per frame at `<output_dir>/<sess>_f<row>/`:
    magnitude_<cond>.npy             # (96, 128) float32
    magnitude_<cond>.png             # viridis cmap
    signed_contrast_<cond>.npy       # (96, 128) float32 (non-real conditions only)
    signed_contrast_<cond>.png       # RdBu cmap, symmetric vmin/vmax
    e_rendered_8x.npy                # (96, 128) reference E for footprint mask
    footprint_mask_p50.npy / p75 / p90  # (96, 128) bool
    frame_manifest.json              # per-frame summary + footprint_fraction values

Aggregator (separate invocation, --mode aggregate):
    aggregate_manifest.csv           # one row per (frame, condition, threshold)
    aggregate_stats.json             # median + IQR of footprint_fraction
                                     #   per (session, condition, threshold)

Standing rules:
    Phase G is consumed inference-only via existing forward path. No
    Phase G code changes. No held-out asset use beyond F-A v1 ckpts the
    pilot already used.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    _crop_and_resize_C, _load_packed_cfa_float01,
    _resize_E_to_target, EMISSION_NATIVE_H, EMISSION_NATIVE_W,
)
from data.emission_dataset import load_emission_at  # noqa: E402
from phase_g.fa_loader import load_fa_v1_checkpoint, render_C_fake  # noqa: E402


# ----------------------------- constants -----------------------------

T_DIFFUSION = 1000
T_STEPS = (50, 150, 300, 500, 750)   # full-eval timesteps (operator-confirmed)
K_NOISE = 4                            # full-eval K (operator-confirmed)
NOISE_SEED_BASE = 42

PHASE_G_INPUT_H = 768
PHASE_G_INPUT_W = 1024
DOWNSAMPLE_FACTOR = 8
COARSE_H = PHASE_G_INPUT_H // DOWNSAMPLE_FACTOR   # 96
COARSE_W = PHASE_G_INPUT_W // DOWNSAMPLE_FACTOR   # 128

FA_V1_CKPT_STEPS = (5000, 25000, 70000, 100000)
PERTURBED_CONDS = (
    [f"fake_{s//1000}k" for s in FA_V1_CKPT_STEPS]
    + ["shuffled_E", "cross_session_E"]
)
ALL_CONDS = ["real_correct"] + PERTURBED_CONDS

FOOTPRINT_PERCENTILES = (50, 75, 90)


# ----------------------------- utilities -----------------------------

def block_average_8x(field: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """Average non-overlapping (factor × factor) blocks. arr is 2-D and
    each axis is divisible by factor for our pilot inputs (768/8=96,
    1024/8=128). Same convention as compare_resolution_stability."""
    if field.ndim != 2:
        raise ValueError(f"expected 2-D field, got {field.shape}")
    h, w = field.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"shape {field.shape} not divisible by {factor}")
    h2, w2 = h // factor, w // factor
    return field.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def load_chain_keys(session_dir: Path) -> list[int]:
    """Read chain_log.csv frame indices in sorted order. Reused for the
    deterministic shuffled / cross-session row pickers."""
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
    """Pick a row deterministically far from `this_row` in the same
    session. Convention: this_row's index in `keys`, plus N//2, mod N.
    Guarantees |delta| ≥ N/4 typically (large enough that the chosen E
    is uncorrelated with this_row's chain state)."""
    n = len(keys)
    if n == 0:
        raise ValueError("empty chain keys")
    if this_row not in keys:
        raise ValueError(f"row {this_row} missing from chain keys")
    idx = keys.index(this_row)
    return keys[(idx + n // 2) % n]


def deterministic_cross_session_row(this_row: int, this_keys: list[int],
                                     other_keys: list[int]) -> int:
    """Pick a same-percentile row from the other session. If this_row is
    at percentile p in this session, pick the row at percentile p in the
    other. Deterministic and reproducible per (session pair, this_row)."""
    if this_row not in this_keys:
        raise ValueError(f"row {this_row} missing from this_keys")
    n_self = len(this_keys)
    n_other = len(other_keys)
    if n_other == 0:
        raise ValueError("empty other_keys")
    idx_self = this_keys.index(this_row)
    pct = idx_self / max(n_self - 1, 1)
    idx_other = min(n_other - 1, int(round(pct * (n_other - 1))))
    return other_keys[idx_other]


# ----------------------------- model load -----------------------------

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


def load_phase_g_C(session_dir: Path, row: int) -> torch.Tensor:
    """(4, 768, 1024) Phase G C input."""
    return _crop_and_resize_C(_load_packed_cfa_float01(
        session_dir / "Recordings" / f"frame_{row:06d}.raw"))


def load_phase_g_E(session_dir: Path, row: int) -> torch.Tensor:
    """(3, 768, 1024) Phase G E input."""
    return _resize_E_to_target(load_emission_at(
        session_dir / "derived" / "Emissions" / f"tile_{row:06d}.png",
        EMISSION_NATIVE_H, EMISSION_NATIVE_W))


# ----------------------------- per-frame scoring -----------------------------

@torch.no_grad()
def per_pixel_mse_field(
    model: DiffusionDiagnosticUNet,
    C: torch.Tensor,            # (4, H, W) float32 [0, 1]
    E: torch.Tensor,            # (3, H, W) float32 [0, 1]  OR None for uncond
    dc: dict,
    device: torch.device,
    dtype: torch.dtype,
    noise: torch.Tensor,        # (n_t, K, 4, H, W) float32 — pre-generated
    timesteps: tuple[int, ...] = T_STEPS,
) -> np.ndarray:
    """Compute the (H, W) per-pixel mean ε-prediction MSE field for one
    (C, E) pair, averaged over (timestep × K) and the 4 CFA channels.

    Returns float32 numpy array of shape (H, W).
    """
    H, W = C.shape[-2:]
    n_t = len(timesteps)
    accum = torch.zeros(H, W, dtype=torch.float64, device=device)
    n_samples = n_t * noise.shape[1]
    for ti, t_val in enumerate(timesteps):
        K_ = noise.shape[1]
        t_tensor = torch.full((K_,), t_val, device=device, dtype=torch.long)
        t_float = t_tensor.float()
        C_rep = C.float().unsqueeze(0).expand(K_, -1, -1, -1).contiguous()
        C_t = q_sample(C_rep, t_tensor, dc, noise[ti]).to(dtype)
        if E is None:
            E_batch = torch.zeros(K_, 3, H, W, device=device, dtype=dtype)
            force_uncond = True
        else:
            E_batch = E.unsqueeze(0).to(device=device, dtype=dtype).expand(
                K_, -1, -1, -1).contiguous()
            force_uncond = False
        with torch.amp.autocast("cuda", dtype=dtype):
            eps_pred = model(C_t, E_batch, t_float, force_uncond=force_uncond)
        diff = eps_pred.float() - noise[ti].float()
        # Per-pixel scalar: mean over the 4 CFA channels, summed across K.
        per_pixel = diff.pow(2).mean(dim=1).sum(dim=0).double()  # (H, W)
        accum.add_(per_pixel)
    accum.div_(n_samples)   # mean across timestep × K
    return accum.float().cpu().numpy()


def render_magnitude_png(field96: np.ndarray, title: str, out_path: Path,
                          vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    if vmax is None:
        vmax = float(field96.max()) if field96.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(field96, cmap="viridis", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_signed_contrast_png(field96: np.ndarray, title: str,
                                out_path: Path, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    if vmax is None:
        vmax = float(np.max(np.abs(field96))) if field96.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(field96, cmap="RdBu", vmin=-vmax, vmax=+vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- per-frame driver -----------------------------

def process_frame(
    sess: str,
    row: int,
    quantile: str,
    model: DiffusionDiagnosticUNet,
    fa_v1_ckpts: dict[int, "object"],   # step → loaded EditorControlNet
    sess_dirs: dict[str, Path],
    chain_keys: dict[str, list[int]],
    dc: dict,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
    log,
) -> dict:
    """Process one frame. Returns the manifest dict for this frame."""
    frame_dir = out_dir / f"{sess}_f{row:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- C and E inputs ---
    C_real = load_phase_g_C(sess_dirs[sess], row)
    E_correct = load_phase_g_E(sess_dirs[sess], row)

    # Shuffled E: same session, deterministic far-row.
    shuffled_row = deterministic_shuffled_row(row, chain_keys[sess])
    E_shuffled = load_phase_g_E(sess_dirs[sess], shuffled_row)

    # Cross-session E: other session, same-percentile row.
    other_sess = "V10" if sess == "D2" else "D2"
    cross_row = deterministic_cross_session_row(
        row, chain_keys[sess], chain_keys[other_sess]
    )
    E_cross = load_phase_g_E(sess_dirs[other_sess], cross_row)

    # F-A v1 fakes (4 ckpts). Donor convention matches the spatial heatmap
    # pilot: source_row = chain[(target_idx + N // 4) % N].
    keys = chain_keys[sess]
    target_idx = keys.index(row)
    source_row = keys[(target_idx + len(keys) // 4) % len(keys)]
    log(f"  [{sess} f={row}] shuffled→{shuffled_row}  "
        f"cross→{other_sess}/{cross_row}  fa_donor→{source_row}")

    C_fakes: dict[int, torch.Tensor] = {}
    for step, fa_model in fa_v1_ckpts.items():
        Cf = render_C_fake(
            fa_model, sess_dirs[sess],
            source_row=source_row, target_row=row,
            device=device, dtype=dtype,
        ).cpu().float()
        C_fakes[step] = Cf

    # --- pre-generate paired noise (same noise across conditions per (t, k)) ---
    H, W = PHASE_G_INPUT_H, PHASE_G_INPUT_W
    seed = NOISE_SEED_BASE + (row * 7919 + (1 if sess == "V10" else 0))
    torch.manual_seed(seed)
    noise = torch.randn(len(T_STEPS), K_NOISE, 4, H, W,
                         device=device, dtype=torch.float32)

    # --- compute per-condition magnitude field at native (768, 1024) ---
    cond_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("real_correct", C_real, E_correct),
    ]
    for step in FA_V1_CKPT_STEPS:
        cond_pairs.append((f"fake_{step//1000}k", C_fakes[step], E_correct))
    cond_pairs.append(("shuffled_E", C_real, E_shuffled))
    cond_pairs.append(("cross_session_E", C_real, E_cross))

    magnitude_native: dict[str, np.ndarray] = {}    # (768, 1024)
    magnitude_8x: dict[str, np.ndarray] = {}        # (96, 128)
    for cond, C, E in cond_pairs:
        t0 = time.time()
        field = per_pixel_mse_field(
            model, C.to(device=device, dtype=torch.float32),
            E.to(device=device, dtype=torch.float32),
            dc, device, dtype, noise,
        )
        magnitude_native[cond] = field
        magnitude_8x[cond] = block_average_8x(field)
        log(f"    {cond:18s} mean={field.mean():.5f} max={field.max():.5f} "
            f"{time.time()-t0:.1f}s")

    # --- signed contrasts (perturbed - real_correct) at 8× ---
    signed_contrast_8x: dict[str, np.ndarray] = {}
    for cond in PERTURBED_CONDS:
        signed_contrast_8x[cond] = (
            magnitude_8x[cond] - magnitude_8x["real_correct"]
        )

    # --- E reference for footprint at 8× ---
    e_rendered_native = E_correct.float().mean(dim=0).numpy()  # (768, 1024)
    e_rendered_8x = block_average_8x(e_rendered_native)         # (96, 128)
    np.save(frame_dir / "e_rendered_8x.npy", e_rendered_8x.astype(np.float32))

    # --- footprint masks at p50/p75/p90 ---
    footprint_masks: dict[int, np.ndarray] = {}
    for pct in FOOTPRINT_PERCENTILES:
        thr = float(np.percentile(e_rendered_8x, pct))
        mask = e_rendered_8x > thr
        footprint_masks[pct] = mask
        np.save(frame_dir / f"footprint_mask_p{pct}.npy", mask)

    # --- save magnitude maps (per-frame normalized + native) ---
    # Operator's spec: per-frame norm = divide by frame's max-abs to put
    # values in scale-comparable form. Save BOTH the raw 8× field (for
    # cross-condition shared-vmax plots) and a per-frame-normalized PNG.
    for cond, m8x in magnitude_8x.items():
        np.save(frame_dir / f"magnitude_{cond}.npy", m8x.astype(np.float32))
    # Cross-condition shared vmax for the magnitude PNGs (uses max across
    # all conditions for this frame).
    mag_vmax = float(max(np.max(m) for m in magnitude_8x.values()))
    if mag_vmax <= 0:
        mag_vmax = 1.0
    for cond, m8x in magnitude_8x.items():
        render_magnitude_png(
            m8x, f"magnitude · {sess} f={row} · {cond}",
            frame_dir / f"magnitude_{cond}.png",
            vmax=mag_vmax,
        )

    # --- save signed contrast maps (cross-condition shared vmax) ---
    sc_vmax = float(max(
        np.max(np.abs(s)) for s in signed_contrast_8x.values()
    )) if signed_contrast_8x else 1.0
    if sc_vmax <= 0:
        sc_vmax = 1.0
    for cond, sc in signed_contrast_8x.items():
        np.save(frame_dir / f"signed_contrast_{cond}.npy", sc.astype(np.float32))
        render_signed_contrast_png(
            sc, f"signed_contrast · {sess} f={row} · {cond} − real_correct",
            frame_dir / f"signed_contrast_{cond}.png",
            vmax=sc_vmax,
        )

    # --- footprint fractions ---
    # For each (perturbed condition, threshold percentile):
    #   inside  = sum(|signed_contrast| over mask)
    #   outside = sum(|signed_contrast| over ~mask)
    #   fraction = inside / (inside + outside)  (== inside / total)
    rows: list[dict] = []
    for cond in PERTURBED_CONDS:
        sc = signed_contrast_8x[cond]
        abs_sc = np.abs(sc)
        total = float(abs_sc.sum())
        for pct in FOOTPRINT_PERCENTILES:
            mask = footprint_masks[pct]
            inside = float(abs_sc[mask].sum())
            outside = total - inside
            frac = (inside / total) if total > 0 else float("nan")
            rows.append({
                "session": sess, "frame": row, "quantile": quantile,
                "condition": cond, "threshold_pct": pct,
                "footprint_fraction": frac,
                "signal_inside": inside,
                "signal_outside": outside,
                "signal_total": total,
                "magnitude_npy": str(frame_dir / f"magnitude_{cond}.npy"),
                "signed_contrast_npy": str(frame_dir
                                            / f"signed_contrast_{cond}.npy"),
                "footprint_mask_npy": str(frame_dir
                                          / f"footprint_mask_p{pct}.npy"),
            })

    # Per-frame manifest fragment.
    frame_manifest = {
        "session": sess, "frame": row, "quantile": quantile,
        "shuffled_row": shuffled_row,
        "cross_session_row": int(cross_row),
        "fa_donor_row": int(source_row),
        "noise_seed": int(seed),
        "rows": rows,
    }
    (frame_dir / "frame_manifest.json").write_text(
        json.dumps(frame_manifest, indent=2)
    )
    log(f"  [{sess} f={row}] DONE — {len(rows)} manifest rows")
    return frame_manifest


# ----------------------------- aggregator -----------------------------

def aggregate(out_dir: Path, log) -> None:
    """Read all per-frame frame_manifest.json files and produce
    aggregate_manifest.csv + aggregate_stats.json."""
    rows: list[dict] = []
    for frame_json in sorted(out_dir.glob("*/frame_manifest.json")):
        m = json.loads(frame_json.read_text())
        rows.extend(m.get("rows", []))
    if not rows:
        log("[aggregate] no frame manifests found under "
            f"{out_dir}; nothing to aggregate")
        return

    # Master CSV
    csv_path = out_dir / "aggregate_manifest.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"[aggregate] wrote {csv_path}  ({len(rows)} rows)")

    # Aggregate stats: median + IQR of footprint_fraction grouped by
    # (session, condition, threshold_pct).
    by_group: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["session"], r["condition"], r["threshold_pct"])
        v = r["footprint_fraction"]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        by_group.setdefault(key, []).append(float(v))

    agg = {}
    for (sess, cond, pct), vals in sorted(by_group.items()):
        arr = np.array(vals, dtype=np.float64)
        q1, q3 = (np.percentile(arr, [25, 75])
                   if arr.size >= 2 else (float("nan"), float("nan")))
        agg.setdefault(sess, {}).setdefault(cond, {})[f"p{pct}"] = {
            "n_frames": int(arr.size),
            "median": float(np.median(arr)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    stats_path = out_dir / "aggregate_stats.json"
    stats_path.write_text(json.dumps(agg, indent=2))
    log(f"[aggregate] wrote {stats_path}")

    # Brief operator-facing print.
    log("\n[aggregate] median footprint_fraction "
        "(by session × condition × threshold):")
    for sess in sorted(agg.keys()):
        for cond in sorted(agg[sess].keys()):
            for pct_lbl in sorted(agg[sess][cond].keys()):
                m = agg[sess][cond][pct_lbl]
                log(f"  {sess:3s}  {cond:18s}  {pct_lbl}  "
                    f"median={m['median']:.4f}  "
                    f"IQR=[{m['q1']:.4f}, {m['q3']:.4f}]  "
                    f"n={m['n_frames']}")


# ----------------------------- CLI -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["render", "aggregate"], default="render")
    ap.add_argument("--frames-json", type=Path,
                    help="(render) JSON list of {session, row, quantile}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phase-g-ckpt", type=Path,
                    help="(render) Phase G ckpt path")
    ap.add_argument("--d2-dir", type=Path,
                    help="(render) D2 session dir")
    ap.add_argument("--v10-dir", type=Path,
                    help="(render) V10 session dir")
    ap.add_argument("--fa-v1-ckpt-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/"
                                  "phase_f/f_a_full_v1/checkpoints"),
                    help="(render) F-A v1 ckpt dir")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / (
        "render_log.txt" if args.mode == "render" else "aggregate_log.txt"
    )
    log_f = open(log_path, "a")
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    if args.mode == "aggregate":
        aggregate(args.output_dir, log)
        log_f.close()
        return 0

    # render mode
    for required in ("frames_json", "phase_g_ckpt", "d2_dir", "v10_dir"):
        if getattr(args, required) is None:
            log(f"[render] FATAL: --{required.replace('_','-')} required")
            return 2
    frames_spec = json.loads(args.frames_json.read_text())
    if not isinstance(frames_spec, list) or not frames_spec:
        log("[render] FATAL: frames-json must be a non-empty list")
        return 2

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    log(f"[render] device={device} dtype={dtype} "
        f"frames={len(frames_spec)} timesteps={T_STEPS} K={K_NOISE}")

    log(f"[render] loading Phase G ckpt {args.phase_g_ckpt}")
    model = load_phase_g(args.phase_g_ckpt, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[render] Phase G loaded — {n_params/1e6:.2f}M params")

    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}
    chain_keys = {sess: load_chain_keys(sess_dirs[sess]) for sess in ("D2", "V10")}
    log(f"[render] chain keys: D2={len(chain_keys['D2'])} "
        f"V10={len(chain_keys['V10'])}")

    log(f"[render] loading {len(FA_V1_CKPT_STEPS)} F-A v1 ckpts")
    fa_v1_ckpts: dict[int, "object"] = {}
    for step in FA_V1_CKPT_STEPS:
        ckpt = args.fa_v1_ckpt_dir / f"step_{step:08d}.pt"
        if not ckpt.exists():
            log(f"[render] FATAL: missing F-A v1 ckpt {ckpt}")
            return 2
        log(f"  loading step_{step:08d}…")
        fa_v1_ckpts[step] = load_fa_v1_checkpoint(ckpt, device=device, dtype=dtype)

    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    # Process each frame.
    for spec in frames_spec:
        sess = spec["session"]
        row = int(spec["row"])
        quantile = spec.get("quantile", "")
        log(f"[render] === {sess} f={row} ({quantile}) ===")
        try:
            process_frame(
                sess, row, quantile, model, fa_v1_ckpts, sess_dirs,
                chain_keys, dc, device, dtype, args.output_dir, log,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"[render] FRAME FAIL {sess} f={row}: {exc!r}")
            log(traceback.format_exc())
            return 3

    log(f"[render] DONE — {len(frames_spec)} frames processed")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
