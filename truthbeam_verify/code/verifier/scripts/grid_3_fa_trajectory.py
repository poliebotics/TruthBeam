"""Grid 3 — F-A v1 trajectory across 4 checkpoints.

Visualises how F-A v1's C_fake outputs evolve across training checkpoints
{5k, 25k, 70k, 100k} on the same source frames. Single PNG showing rows =
sample frames, cols = (C_real reference, F-A@5k, F-A@25k, F-A@70k, F-A@100k).

Runs on CPU (g1a) since F-A ckpts and session data are mirrored locally.
No GPU contention with Phase H resume on Lambda.

Output: visual_grids/fa_v1_trajectory.png
"""
from __future__ import annotations

import argparse
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
    C_native_to_phase_g_input,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    _crop_and_resize_C, _load_packed_cfa_float01,
)


# Local g1a paths (where F-A ckpts + sessions are mirrored)
LOCAL_ROOT = Path("/path/to/poliebotics_phase_b")
SESSION_DIRS = {
    "D2":  LOCAL_ROOT / "data" / "d2",
    "V10": LOCAL_ROOT / "data" / "v10",
}
FA_CKPTS = [
    ("step_00005000",  LOCAL_ROOT / "experiments/phase_f/f_a_full_v1/checkpoints/step_00005000.pt"),
    ("step_00025000",  LOCAL_ROOT / "experiments/phase_f/f_a_full_v1/checkpoints/step_00025000.pt"),
    ("step_00070000",  LOCAL_ROOT / "experiments/phase_f/f_a_full_v1/checkpoints/step_00070000.pt"),
    ("step_00100000",  LOCAL_ROOT / "experiments/phase_f/f_a_full_v1/checkpoints/step_00100000.pt"),
]
DEFAULT_OUT = LOCAL_ROOT / "visual_grids" / "fa_v1_trajectory.png"

SOURCE_LAG = 2

# 4 representative frames (2 D2 + 2 V10) from held-out blocks
FRAME_PICKS = [
    ("D2",  1500),
    ("D2",  3000),
    ("V10", 1300),
    ("V10", 1900),
]


def _gamma(rgb01: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    return np.clip(rgb01, 0, 1) ** (1.0 / gamma)


def _packed_cfa_to_rgb(cfa: torch.Tensor | np.ndarray,
                       resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """(4, H, W) [0,1] → (H, W, 3) uint8."""
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


def _load_C_real_phase_g(session: str, row: int) -> torch.Tensor:
    """Load real C and crop+resize to Phase G resolution (4, 768, 1024)."""
    return _crop_and_resize_C(_load_packed_cfa_float01(
        SESSION_DIRS[session] / "Recordings" / f"frame_{row:06d}.raw"))


@torch.no_grad()
def render_C_fake_at_phase_g_resolution(model, session: str, source_row: int,
                                         target_row: int) -> torch.Tensor:
    """Run F-A on (C_source, E_source, E_target) → return (4, 768, 1024) C_fake."""
    sd = SESSION_DIRS[session]
    device = torch.device("cpu")
    dtype = torch.float32
    model.eval()
    C_s = load_C_native(sd, source_row).to(device, dtype=dtype).unsqueeze(0)
    E_s = load_E_native(sd, source_row).to(device, dtype=dtype).unsqueeze(0)
    E_t = load_E_native(sd, target_row).to(device, dtype=dtype).unsqueeze(0)
    C_pred = model(C_s, E_s, E_t)  # (1, 4, 2300, 2660)
    return C_native_to_phase_g_input(C_pred.squeeze(0).float()).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # Pre-flight
    for label, ck in FA_CKPTS:
        if not ck.exists():
            raise SystemExit(f"missing F-A ckpt: {ck}")
    for sess, sd in SESSION_DIRS.items():
        if not sd.exists():
            raise SystemExit(f"missing session dir: {sd}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[grid3] {len(FA_CKPTS)} F-A ckpts × {len(FRAME_PICKS)} frames "
          f"= {len(FA_CKPTS) * len(FRAME_PICKS)} CPU inferences")

    device = torch.device("cpu")
    dtype = torch.float32

    # Load each ckpt + run inferences on all frames before unloading
    rows_data: list[dict] = [{
        "session": s, "target_row": r, "source_row": r - SOURCE_LAG,
        "C_real": _load_C_real_phase_g(s, r),
        "C_fakes": {}
    } for s, r in FRAME_PICKS]

    overall_t0 = time.time()
    for label, ckpt_path in FA_CKPTS:
        print(f"[grid3] loading F-A {label}...")
        t0 = time.time()
        model = load_fa_v1_checkpoint(ckpt_path, device, dtype)
        print(f"  loaded in {time.time()-t0:.1f}s")
        for ri, row in enumerate(rows_data):
            t0 = time.time()
            cf = render_C_fake_at_phase_g_resolution(
                model, row["session"], row["source_row"], row["target_row"])
            row["C_fakes"][label] = cf
            print(f"  [{label}] frame {ri+1}/{len(rows_data)} "
                  f"({row['session']} row {row['target_row']}) inf={time.time()-t0:.1f}s")
        del model
    print(f"[grid3] all inferences done in {time.time()-overall_t0:.0f}s")

    # Compose grid: rows = frames, cols = (C_real | F-A@5k | F-A@25k | F-A@70k | F-A@100k)
    n_rows = len(rows_data)
    n_cols = 1 + len(FA_CKPTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows))
    col_titles = ["C_real (target)"] + [f"F-A {l}" for l, _ in FA_CKPTS]
    DISPLAY_HW = (480, 640)

    for ri, row in enumerate(rows_data):
        axes_row = axes[ri] if n_rows > 1 else axes
        # Col 0: C_real
        axes_row[0].imshow(_packed_cfa_to_rgb(row["C_real"], resize_to=DISPLAY_HW))
        # Cols 1..N: F-A outputs at each ckpt
        for ci, (label, _) in enumerate(FA_CKPTS, start=1):
            axes_row[ci].imshow(_packed_cfa_to_rgb(row["C_fakes"][label], resize_to=DISPLAY_HW))
        for ci in range(n_cols):
            axes_row[ci].set_xticks([]); axes_row[ci].set_yticks([])
            if ri == 0:
                axes_row[ci].set_title(col_titles[ci], fontsize=9)
            if ci == 0:
                axes_row[ci].set_ylabel(
                    f"{row['session']} row {row['target_row']}", fontsize=9,
                    rotation=0, labelpad=42, ha="right", va="center")

    fig.suptitle(
        "Grid 3 — F-A v1 trajectory across training checkpoints\n"
        "rows: held-out frames; cols: real C | F-A@{5k, 25k, 70k, 100k}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sz_kb = args.out.stat().st_size // 1024
    print(f"[grid3] wrote {args.out}  ({sz_kb} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
