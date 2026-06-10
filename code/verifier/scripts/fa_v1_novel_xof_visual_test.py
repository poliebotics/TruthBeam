"""Quick visual test — F-A v1 on novel-XOF E.

Generates a single PNG showing F-A v1 (step 100k checkpoint) producing C_fake
under two regimes:

    1. E_target drawn from a real session chain (baseline — what F-A v1 saw
       distributions of during training).
    2. E_target rendered from NOVEL chain seeds (random 32-byte values that no
       session ever produced) — chain-state distribution F-A has never seen.

Operator wants to visually assess whether F-A v1 already generalises to novel-
XOF E or whether outputs are obviously degraded / adversarial-looking. Result
informs whether F-A v2's novel-XOF training is essential (with realism
regularisers) or merely additive.

Layout: 6 rows × 8 columns
    rows: 6 (source_frame, session) picks (3 D2 + 3 V10)
    cols:
        0 — C_source                          (the source capture F-A had as input)
        1 — E_source                          (source emission)
        2 — E_target_real                     (real session target E)
        3 — F-A(E_target_real)                (baseline F-A output)
        4 — E_novel_1                         (novel-XOF target E)
        5 — F-A(E_novel_1)                    (test F-A output)
        6 — E_novel_2                         (different novel-XOF target E)
        7 — F-A(E_novel_2)                    (test F-A output)

Halt conditions:
    - F-A inference shape mismatch / dtype error
    - identity-render parity check fails on D2/V10
    - any C_fake has NaN / Inf / all-zero / saturated [<1e-3 std]

Compute: 24 F-A inferences ≈ 30 sec on 1 A100. Budget: GPU 7 (idle while Phase H
runs on 0-6). Falls back to whichever GPU is least loaded.

Output: visual_grids/fa_v1_novel_xof_visual_test.png plus diagnostic stdout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.fa_loader import (  # noqa: E402
    load_fa_v1_checkpoint, load_C_native, load_E_native,
    C_native_to_phase_g_input, FA_EM_H, FA_EM_W,
)
from phase_g.xof_perturb import (  # noqa: E402
    load_chain_log, expand_streams_from_s_t, render_streams_to_tile,
    verify_identity_render_parity,
)
from phase_g.diffusion_diagnostic_dataset import EVAL_BLOCKS  # noqa: E402


# -------- defaults / paths --------

DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/visual_grids")
SESSION_DIRS = {
    "D2":  Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"),
}
DEFAULT_CKPT = Path(
    "/path/to/poliebotics_phase_b/experiments/phase_f/"
    "f_a_full_v1/checkpoints/step_00100000.pt")

# Strict path-mode resolution (same pattern as visual_grids.py): if any of the
# Lambda roots is missing, try local; if mixed, abort. This prevents writing
# to the wrong tree when half-mounted.
# Probe the checkpoint FILE (not just its parent dir) — a stub directory with
# no .pt would otherwise put us in Lambda mode and fail late.
_PROBE = [SESSION_DIRS["D2"], SESSION_DIRS["V10"], DEFAULT_CKPT]
_AVAIL = sum(int(p.exists()) for p in _PROBE)
if _AVAIL == len(_PROBE):
    pass  # Lambda mode
elif _AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    SESSION_DIRS = {
        "D2":  LOCAL_ROOT / "data" / "d2",
        "V10": LOCAL_ROOT / "data" / "v10",
    }
    DEFAULT_CKPT = (LOCAL_ROOT / "experiments" / "phase_f"
                    / "f_a_full_v1" / "checkpoints" / "step_00100000.pt")
    DEFAULT_OUT = LOCAL_ROOT / "visual_grids"
else:
    raise SystemExit(
        f"[fa_v1_novel_xof] mixed root state: {_AVAIL}/{len(_PROBE)} Lambda "
        "paths exist. Refusing to run.")


# Six (session, target_row) picks — chosen from each session's Phase G held-out
# blocks so they are guaranteed to be real held-out frames F-A has never seen
# trained on. Source row = target_row - 2 (matches Stage 0 / fa_loader convention).
# Picks span the breadth of held-out blocks so the grid shows scene variety.
SOURCE_LAG = 2
DEFAULT_FRAME_PICKS = [
    ("D2",  1500),
    ("D2",  3000),
    ("D2",  4500),
    ("V10", 1300),
    ("V10", 1900),
    ("V10", 2500),
]


# -------- novel-seed generation --------

def generate_novel_s_t_hex(n: int, seed_bytes: bytes | None = None) -> list[str]:
    """Generate n novel S_t-equivalent 32-byte values as hex strings.

    A real S_t in the chain log is the post-emission BLAKE2b-256 hash of the
    chain state at frame t (32 bytes). The XOF expansion in
    expand_streams_from_s_t(s_t_hex) consumes any 32-byte value and produces
    valid (stream_R, stream_G, stream_B). Any random 32-byte value is therefore
    a syntactically valid novel chain state F-A has never observed.

    For determinism we accept an optional seed_bytes; otherwise os.urandom.
    """
    if seed_bytes is None:
        return [os.urandom(32).hex() for _ in range(n)]
    # Deterministic mode: seed_bytes drives a hash chain.
    from blake3 import blake3
    out = []
    for i in range(n):
        h = blake3(seed_bytes + i.to_bytes(8, "big")).digest(length=32)
        out.append(h.hex())
    return out


def render_novel_E_native(novel_s_t_hex: str) -> torch.Tensor:
    """Novel-XOF E at native (1080, 1920) — the F-A input resolution.

    Returns: (3, 1080, 1920) float32 in [0, 1].
    """
    streams = expand_streams_from_s_t(novel_s_t_hex)
    tile = render_streams_to_tile(streams, device="cpu")  # (3, 1080, 1920) uint8
    return tile.float() / 255.0


# -------- F-A inference at native resolution with explicit E --------

@torch.no_grad()
def fa_render_C_fake_explicit_E(model, C_source: torch.Tensor,
                                 E_source: torch.Tensor,
                                 E_target: torch.Tensor,
                                 device: torch.device,
                                 dtype: torch.dtype) -> torch.Tensor:
    """Run F-A on (C_source, E_source, E_target) where all 3 are native-res
    tensors already. Returns C_fake at Phase G resolution (4, 768, 1024) on CPU
    in float32.
    """
    model.eval()
    C_s = C_source.to(device, dtype=dtype).unsqueeze(0)
    E_s = E_source.to(device, dtype=dtype).unsqueeze(0)
    E_t = E_target.to(device, dtype=dtype).unsqueeze(0)
    if E_s.shape[-2:] != (FA_EM_H, FA_EM_W) or E_t.shape[-2:] != (FA_EM_H, FA_EM_W):
        raise RuntimeError(
            f"E shape mismatch: E_source={tuple(E_s.shape)} "
            f"E_target={tuple(E_t.shape)} (expected -2:-1 = {FA_EM_H}x{FA_EM_W})")
    C_pred = model(C_s, E_s, E_t)  # expected (1, 4, 2300, 2660)
    expected_shape = (1, 4, 2300, 2660)
    if tuple(C_pred.shape) != expected_shape:
        raise RuntimeError(
            f"F-A output shape {tuple(C_pred.shape)} != expected {expected_shape}; "
            "refusing to silently downstream a malformed tensor.")
    return C_native_to_phase_g_input(C_pred.squeeze(0).float().cpu())


# -------- visualisation helpers (consistent with scripts/visual_grids.py) --------

def _gamma(rgb01: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    return np.clip(rgb01, 0, 1) ** (1.0 / gamma)


def _packed_cfa_to_rgb(cfa: torch.Tensor | np.ndarray,
                       resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """(4, H, W) [0,1] → (H', W', 3) uint8."""
    if hasattr(cfa, "numpy"):
        cfa = cfa.detach().float().cpu().numpy()
    R, G1, G2, B = cfa[0], cfa[1], cfa[2], cfa[3]
    G = 0.5 * (G1 + G2)
    rgb = np.stack([R, G, B], axis=-1)
    rgb = _gamma(rgb)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if resize_to is not None:
        rgb = cv2.resize(rgb, (resize_to[1], resize_to[0]),
                          interpolation=cv2.INTER_AREA)
    return rgb


def _e_tensor_to_rgb(e: torch.Tensor | np.ndarray,
                     resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """(3, H, W) [0,1] → (H', W', 3) uint8."""
    if hasattr(e, "numpy"):
        e = e.detach().float().cpu().numpy()
    arr = e.transpose(1, 2, 0)
    arr = _gamma(arr)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if resize_to is not None:
        arr = cv2.resize(arr, (resize_to[1], resize_to[0]),
                          interpolation=cv2.INTER_AREA)
    return arr


def _tensor_stats(t: torch.Tensor) -> dict:
    arr = t.detach().float().cpu().numpy()
    return {
        "shape": tuple(arr.shape),
        "min":   float(np.nanmin(arr)) if arr.size else float("nan"),
        "max":   float(np.nanmax(arr)) if arr.size else float("nan"),
        "mean":  float(np.nanmean(arr)) if arr.size else float("nan"),
        "std":   float(np.nanstd(arr))  if arr.size else float("nan"),
        "n_nan": int(np.isnan(arr).sum()),
        "n_inf": int(np.isinf(arr).sum()),
    }


def _flag_anomalies(name: str, stats: dict) -> list[str]:
    flags = []
    if stats["n_nan"] > 0: flags.append(f"NaN×{stats['n_nan']}")
    if stats["n_inf"] > 0: flags.append(f"Inf×{stats['n_inf']}")
    if stats["std"] < 1e-3: flags.append(f"flat (std={stats['std']:.2e})")
    if stats["max"] - stats["min"] < 1e-3: flags.append("zero-range")
    if stats["max"] > 1.5 or stats["min"] < -0.5:
        flags.append(f"out-of-range [{stats['min']:.2f}, {stats['max']:.2f}]")
    return flags


# -------- main --------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_OUT / "fa_v1_novel_xof_visual_test.png")
    ap.add_argument("--n-novel-seeds", type=int, default=20,
                    help="# novel S_t hex values to generate (we use 2 per row "
                    "but sample from a wider pool for variety).")
    ap.add_argument("--seed", type=str, default=None,
                    help="Optional: hex-encoded seed bytes for deterministic "
                    "novel-S_t generation. Default: os.urandom (non-determinstic).")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda:N or cpu; default = least-loaded GPU.")
    ap.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp32"))
    args = ap.parse_args()

    # Validate --n-novel-seeds early (we use 2 distinct seeds per row).
    needed_seeds = 2 * len(DEFAULT_FRAME_PICKS)
    if args.n_novel_seeds < needed_seeds:
        raise SystemExit(
            f"--n-novel-seeds={args.n_novel_seeds} too small; need at least "
            f"{needed_seeds} (2 per row × {len(DEFAULT_FRAME_PICKS)} rows).")

    # ---- pick device ----
    # CRITICAL: Phase H trains DDP on GPUs 0-6 (rank 7 sometimes idle). Default
    # to cuda:7 to avoid contention. Codex audit 2026-05-04.
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 7:
            device = torch.device("cuda:7")
        else:
            # Smaller box — pick GPU with most free memory.
            free_per = []
            for i in range(n_gpu):
                free, _total = torch.cuda.mem_get_info(i)
                free_per.append((free, i))
            free_per.sort(reverse=True)
            device = torch.device(f"cuda:{free_per[0][1]}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    print(f"[fa_v1_novel_xof] device={device} dtype={dtype} ckpt={args.ckpt}")

    # ---- identity-render parity gate (HARD) ----
    print("[fa_v1_novel_xof] identity-render parity gate...")
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        if not chain:
            print(f"  WARN: {sess} chain log empty; skipping parity")
            continue
        samples = [f for f in (100, 1500, 3000) if f in chain][:3]
        if not samples:
            samples = [sorted(chain.keys())[len(chain) // 2]]
        ok, details = verify_identity_render_parity(sd, chain, samples, tol=1e-6)
        print(f"  {sess}: max_overall={details['max_overall']:.2e}  ok={ok}")
        if not ok:
            print(f"  HARD GATE FAIL — aborting (details={details})")
            return 1
    print("[fa_v1_novel_xof] parity OK")

    # ---- load F-A v1 ----
    if not args.ckpt.exists():
        print(f"[fa_v1_novel_xof] ckpt missing at {args.ckpt}")
        return 2
    t0 = time.time()
    model = load_fa_v1_checkpoint(args.ckpt, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fa_v1_novel_xof] F-A v1 loaded in {time.time()-t0:.1f}s, "
          f"params={n_params/1e6:.1f}M")

    # ---- generate novel S_t pool ----
    seed_bytes = bytes.fromhex(args.seed) if args.seed else None
    novel_seeds = generate_novel_s_t_hex(args.n_novel_seeds, seed_bytes=seed_bytes)
    print(f"[fa_v1_novel_xof] generated {len(novel_seeds)} novel S_t values; "
          f"first={novel_seeds[0][:16]}…")
    # Verify they are not in any session's chain log (sanity)
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        if not chain: continue
        existing = set(chain.values())
        collisions = sum(1 for s in novel_seeds if s in existing)
        if collisions > 0:
            print(f"  WARN: {collisions} novel seeds collide with {sess} chain "
                  f"(astronomically unlikely; abort).")
            return 3
    print("[fa_v1_novel_xof] no collisions with real chains (as expected).")

    # ---- pick which novel seeds to use per row (2 per row, distinct) ----
    # Use indices [0, 1] for row 0; [2, 3] for row 1; etc. — distinct seeds per row.
    n_rows = len(DEFAULT_FRAME_PICKS)
    if 2 * n_rows > len(novel_seeds):
        print(f"  ERROR: need 2×{n_rows} novel seeds, only {len(novel_seeds)}")
        return 4

    # ---- run inference + collect stats ----
    print("[fa_v1_novel_xof] running 24 F-A inferences "
          f"(6 frames × 4 conditions = 24)...")
    rows_data = []
    inf_total_t = 0.0

    for ri, (sess, target_row) in enumerate(DEFAULT_FRAME_PICKS):
        sd = SESSION_DIRS[sess]
        source_row = target_row - SOURCE_LAG
        # Verify rows are in chain log + on disk
        chain = load_chain_log(sd)
        for r in (target_row, source_row):
            if r not in chain:
                print(f"  ERROR: {sess} row {r} not in chain_log; aborting")
                return 5

        # Native loads
        t0 = time.time()
        C_source = load_C_native(sd, source_row)               # (4, 2300, 2660)
        E_source = load_E_native(sd, source_row)               # (3, 1080, 1920)
        E_target_real = load_E_native(sd, target_row)          # (3, 1080, 1920)
        load_t = time.time() - t0

        # Two distinct novel E for this row
        novel_a = novel_seeds[2 * ri + 0]
        novel_b = novel_seeds[2 * ri + 1]
        E_novel_1 = render_novel_E_native(novel_a)             # (3, 1080, 1920)
        E_novel_2 = render_novel_E_native(novel_b)             # (3, 1080, 1920)

        # F-A inferences
        ti = time.time()
        C_fake_real = fa_render_C_fake_explicit_E(
            model, C_source, E_source, E_target_real, device, dtype)
        C_fake_n1   = fa_render_C_fake_explicit_E(
            model, C_source, E_source, E_novel_1, device, dtype)
        C_fake_n2   = fa_render_C_fake_explicit_E(
            model, C_source, E_source, E_novel_2, device, dtype)
        inf_t = time.time() - ti
        inf_total_t += inf_t

        # Stats + anomaly check on each F-A output
        anomalies_any = []
        for nm, t in (("C_fake_real", C_fake_real),
                      ("C_fake_n1",   C_fake_n1),
                      ("C_fake_n2",   C_fake_n2)):
            st = _tensor_stats(t)
            flags = _flag_anomalies(nm, st)
            print(f"  [{sess} row={target_row}] {nm}: "
                  f"min={st['min']:.3f} max={st['max']:.3f} "
                  f"mean={st['mean']:.3f} std={st['std']:.3f}  "
                  f"flags={flags or 'none'}")
            if flags:
                anomalies_any.append((nm, flags))

        if anomalies_any:
            print(f"  HALT: anomalies on {sess} row {target_row}: {anomalies_any}")
            return 6

        rows_data.append({
            "session":      sess,
            "target_row":   target_row,
            "source_row":   source_row,
            "C_source":     C_source,
            "E_source":     E_source,
            "E_target_real": E_target_real,
            "E_novel_1":    E_novel_1,
            "E_novel_2":    E_novel_2,
            "C_fake_real":  C_fake_real,
            "C_fake_n1":    C_fake_n1,
            "C_fake_n2":    C_fake_n2,
            "novel_seed_a": novel_a,
            "novel_seed_b": novel_b,
            "load_t":       load_t,
            "inf_t":        inf_t,
        })
        print(f"  [{sess} row={target_row}] load={load_t:.2f}s "
              f"inf={inf_t:.2f}s ✓")

    print(f"[fa_v1_novel_xof] all 24 inferences done; "
          f"total inference time = {inf_total_t:.1f}s")

    # ---- compose grid ----
    print("[fa_v1_novel_xof] composing grid...")
    n_rows = len(rows_data)
    n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.6 * n_cols, 2.0 * n_rows))
    col_titles = [
        "C_source (capture)",
        "E_source (frame−2)",
        "E_target (real)",
        "F-A( E_target_real )",
        "E_novel_1 (novel-XOF)",
        "F-A( E_novel_1 )",
        "E_novel_2 (novel-XOF)",
        "F-A( E_novel_2 )",
    ]

    DISPLAY_HW = (480, 640)  # consistent display size for all cells

    for ri, row in enumerate(rows_data):
        cells = [
            _packed_cfa_to_rgb(row["C_source"],       resize_to=DISPLAY_HW),
            _e_tensor_to_rgb(  row["E_source"],       resize_to=DISPLAY_HW),
            _e_tensor_to_rgb(  row["E_target_real"],  resize_to=DISPLAY_HW),
            _packed_cfa_to_rgb(row["C_fake_real"],    resize_to=DISPLAY_HW),
            _e_tensor_to_rgb(  row["E_novel_1"],      resize_to=DISPLAY_HW),
            _packed_cfa_to_rgb(row["C_fake_n1"],      resize_to=DISPLAY_HW),
            _e_tensor_to_rgb(  row["E_novel_2"],      resize_to=DISPLAY_HW),
            _packed_cfa_to_rgb(row["C_fake_n2"],      resize_to=DISPLAY_HW),
        ]
        for ci, img in enumerate(cells):
            ax = axes[ri, ci] if n_rows > 1 else axes[ci]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{row['session']} row {row['target_row']}",
                              fontsize=9, rotation=0, labelpad=42, ha="right",
                              va="center")

    fig.suptitle(
        "F-A v1 visual test — real-session E_target vs novel-XOF E\n"
        f"checkpoint: {args.ckpt.name}   "
        f"novel-seed pool size: {len(novel_seeds)}   "
        f"frames: {n_rows} ({sum(1 for r in rows_data if r['session']=='D2')} D2 / "
        f"{sum(1 for r in rows_data if r['session']=='V10')} V10)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Defer parent-dir creation until ALL gates have passed (parity, ckpt,
    # seeds, inference). Halts above must not leave empty visual_grids/ dirs.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sz_kb = args.out.stat().st_size // 1024
    print(f"[fa_v1_novel_xof] wrote {args.out}  ({sz_kb} KB)")

    # ---- diagnostic sidecar ----
    log = args.out.with_suffix(".log.txt")
    with open(log, "w") as f:
        f.write(f"# fa_v1_novel_xof_visual_test diagnostics\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"device: {device}  dtype: {dtype}\n")
        f.write(f"n_novel_seeds_pool: {len(novel_seeds)}\n")
        f.write(f"first_5_novel_seeds: {[s[:16]+'..' for s in novel_seeds[:5]]}\n")
        f.write(f"total_inference_time_sec: {inf_total_t:.2f}\n")
        f.write(f"frames:\n")
        for r in rows_data:
            f.write(f"  {r['session']} row={r['target_row']} src={r['source_row']} "
                    f"novel_a={r['novel_seed_a'][:16]}.. "
                    f"novel_b={r['novel_seed_b'][:16]}.. "
                    f"load={r['load_t']:.2f}s inf={r['inf_t']:.2f}s\n")
        f.write("\n# Per-output stats\n")
        for r in rows_data:
            for nm in ("C_fake_real", "C_fake_n1", "C_fake_n2"):
                st = _tensor_stats(r[nm])
                f.write(f"  {r['session']} row={r['target_row']} {nm}: "
                        f"min={st['min']:.4f} max={st['max']:.4f} "
                        f"mean={st['mean']:.4f} std={st['std']:.4f}\n")
    print(f"[fa_v1_novel_xof] diagnostics → {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
