"""Phase G — build a small set of visual exemplars showing exactly what the
three diffusion-diagnostic networks see during training.

Outputs (PNGs under results/diffusion_diagnostic/exemplars/):
  - data_exemplars_d2.png — 4 D2 frames: C-as-RGB ‖ E
  - data_exemplars_v10.png — 4 V10 frames: C-as-RGB ‖ E
  - shuffled_pairs.png — 4 examples of the shuffled control:
        row r → real C  ‖ real E  ‖  partner row r' → wrong E[r']
  - synthetic_positive.png — 2 frames showing C vs C+αM·blur(E)
        for α ∈ {0.05, 0.10, 0.20}
  - noised_diffusion.png — 1 frame at timesteps t ∈ {50, 150, 300, 500, 750}

CPU-only. Runs in ~3-5 min.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_dataset import (
    DiffusionDiagnosticDataset, _crop_and_resize_C,
    _load_packed_cfa_float01, _resize_E_to_target, EMISSION_NATIVE_H,
    EMISSION_NATIVE_W, build_shuffle_permutation, train_rows, held_out_rows,
    SESSION_SEED_OFFSET, EVAL_BLOCKS,
)
from phase_g.diffusion_diagnostic_model import build_diffusion_constants, q_sample
from data.emission_dataset import load_emission_at


OUT = ROOT / "results" / "diffusion_diagnostic" / "exemplars"
OUT.mkdir(parents=True, exist_ok=True)


def cfa_to_rgb_for_display(cfa: torch.Tensor) -> np.ndarray:
    """Packed CFA (4, H, W) in [0,1] → RGB uint8 (H, W, 3) for display.
    Channel order in packed CFA: R, G1, G2, B. We average the two greens
    and gamma-correct slightly so dim regions are visible."""
    R, G1, G2, B = cfa[0].numpy(), cfa[1].numpy(), cfa[2].numpy(), cfa[3].numpy()
    G = 0.5 * (G1 + G2)
    rgb = np.stack([R, G, B], axis=-1)
    # Light gamma for display only
    rgb = np.clip(rgb, 0, 1) ** (1 / 1.6)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def E_to_rgb_for_display(E: torch.Tensor) -> np.ndarray:
    """E is already RGB in [0,1]. Return uint8 (H, W, 3)."""
    arr = E.numpy().transpose(1, 2, 0)
    return (np.clip(arr, 0, 1) ** (1 / 1.6) * 255).clip(0, 255).astype(np.uint8)


def label_strip(text: str, w: int, h: int = 28, bg=(40, 40, 40),
                fg=(220, 220, 220)) -> np.ndarray:
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                fg, 1, cv2.LINE_AA)
    return img


def hstack_pad(images: list[np.ndarray], gap: int = 6,
               bg=(30, 30, 30)) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[0] < h:
            pad = np.full((h - im.shape[0], im.shape[1], 3), bg, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        if i > 0:
            parts.append(np.full((h, gap, 3), bg, dtype=np.uint8))
        parts.append(im)
    return np.concatenate(parts, axis=1)


def vstack_pad(images: list[np.ndarray], gap: int = 6,
               bg=(30, 30, 30)) -> np.ndarray:
    w = max(im.shape[1] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[1] < w:
            pad = np.full((im.shape[0], w - im.shape[1], 3), bg, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=1)
        if i > 0:
            parts.append(np.full((gap, w, 3), bg, dtype=np.uint8))
        parts.append(im)
    return np.concatenate(parts, axis=0)


def downsample_for_grid(img: np.ndarray, target_w: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    new_h = int(h * target_w / w)
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


# ----- (1) data exemplars per session -----

def session_data_exemplar(session: str, sess_dir: Path, frame_rows: list[int],
                          out_path: Path) -> None:
    rows = []
    for r in frame_rows:
        cfa = _crop_and_resize_C(_load_packed_cfa_float01(
            sess_dir / "Recordings" / f"frame_{r:06d}.raw"))
        E = _resize_E_to_target(load_emission_at(
            sess_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W))
        c_rgb = downsample_for_grid(cfa_to_rgb_for_display(cfa), 480)
        e_rgb = downsample_for_grid(E_to_rgb_for_display(E), 480)
        c_lbl = label_strip(f"{session}  row={r}  C (rendered)", c_rgb.shape[1])
        e_lbl = label_strip(f"E  ({EMISSION_NATIVE_H}x{EMISSION_NATIVE_W} → 768x1024)",
                            e_rgb.shape[1])
        row_img = hstack_pad([
            vstack_pad([c_lbl, c_rgb]),
            vstack_pad([e_lbl, e_rgb]),
        ])
        rows.append(row_img)
    grid = vstack_pad(rows, gap=10)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"[done] {out_path}  shape={grid.shape}")


# ----- (2) shuffled control visualization -----

def shuffled_pairs(session: str, sess_dir: Path, n: int, out_path: Path) -> None:
    tr = train_rows(session, stride=3)
    perm = build_shuffle_permutation(tr, min_gap=60,
                                     seed=12345 + SESSION_SEED_OFFSET[session])
    np.random.seed(0)
    sample_rows = np.random.choice(tr, size=n, replace=False)
    rows = []
    for r in sample_rows:
        partner = perm[int(r)]
        cfa = _crop_and_resize_C(_load_packed_cfa_float01(
            sess_dir / "Recordings" / f"frame_{r:06d}.raw"))
        E_real = _resize_E_to_target(load_emission_at(
            sess_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W))
        E_shuf = _resize_E_to_target(load_emission_at(
            sess_dir / "derived" / "Emissions" / f"tile_{partner:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W))

        c_rgb = downsample_for_grid(cfa_to_rgb_for_display(cfa), 360)
        e_real = downsample_for_grid(E_to_rgb_for_display(E_real), 360)
        e_shuf = downsample_for_grid(E_to_rgb_for_display(E_shuf), 360)

        row_img = hstack_pad([
            vstack_pad([label_strip(f"C  row {r}", c_rgb.shape[1]), c_rgb]),
            vstack_pad([label_strip(f"main: E[{r}]", e_real.shape[1]), e_real]),
            vstack_pad([label_strip(f"shuffled: E[{partner}]  (gap={abs(int(r)-partner)})",
                                    e_shuf.shape[1]), e_shuf]),
        ])
        rows.append(row_img)
    grid = vstack_pad(rows, gap=10)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"[done] {out_path}  shape={grid.shape}")


# ----- (3) synthetic_positive visualization -----

def synthetic_pairs(session: str, sess_dir: Path, frame_rows: list[int],
                    out_path: Path) -> None:
    ds = DiffusionDiagnosticDataset(
        session_dirs={session: sess_dir},
        mode="synthetic_positive", role="train", stride=1,
    )
    rows = []
    for r in frame_rows:
        cfa = _crop_and_resize_C(_load_packed_cfa_float01(
            sess_dir / "Recordings" / f"frame_{r:06d}.raw"))
        E = _resize_E_to_target(load_emission_at(
            sess_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W))
        c_real = cfa_to_rgb_for_display(cfa)
        # Apply each alpha manually (the dataset samples alpha randomly; we want
        # a deterministic sweep)
        import random
        cells = [vstack_pad([label_strip(f"{session}  row {r}: C real",
                                          c_real.shape[1]),
                              downsample_for_grid(c_real, 360)])]
        for alpha in (0.05, 0.10, 0.20):
            ds.synth_alphas = (alpha,)
            sample = ds._apply_synthetic_perturbation(cfa.clone(), E.clone(),
                                                     random.Random(0))
            c_pert = cfa_to_rgb_for_display(sample)
            cells.append(vstack_pad([
                label_strip(f"C + α={alpha} · blur(E)",
                            downsample_for_grid(c_pert, 360).shape[1]),
                downsample_for_grid(c_pert, 360),
            ]))
        # Also show E for context
        cells.append(vstack_pad([
            label_strip("E (input projection)",
                        downsample_for_grid(E_to_rgb_for_display(E), 360).shape[1]),
            downsample_for_grid(E_to_rgb_for_display(E), 360),
        ]))
        rows.append(hstack_pad(cells))
    grid = vstack_pad(rows, gap=10)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"[done] {out_path}  shape={grid.shape}")


# ----- (4) noise schedule visualization -----

def noise_schedule_demo(session: str, sess_dir: Path, row: int,
                        out_path: Path) -> None:
    cfa = _crop_and_resize_C(_load_packed_cfa_float01(
        sess_dir / "Recordings" / f"frame_{row:06d}.raw"))
    dc = build_diffusion_constants(1000, torch.device("cpu"), torch.float32)
    cells = []
    for t_val in (0, 50, 150, 300, 500, 750, 999):
        if t_val == 0:
            disp = cfa
        else:
            t = torch.tensor([t_val], dtype=torch.long)
            noise = torch.randn_like(cfa).unsqueeze(0)
            ct = q_sample(cfa.unsqueeze(0), t, dc, noise).squeeze(0)
            # Clip to [0,1] for display (q_sample can go outside this)
            disp = torch.clamp(ct, 0, 1)
        rgb = cfa_to_rgb_for_display(disp)
        cell = vstack_pad([
            label_strip(f"t={t_val}  (alpha_cum={dc['alphas_cum'][min(t_val, 999)].item():.3f})",
                        downsample_for_grid(rgb, 280).shape[1]),
            downsample_for_grid(rgb, 280),
        ])
        cells.append(cell)
    # Two-row layout
    top = hstack_pad(cells[:4])
    bot = hstack_pad(cells[4:])
    grid = vstack_pad([top, bot], gap=10)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"[done] {out_path}  shape={grid.shape}")


def main() -> None:
    d2_dir = ROOT / "data" / "d2"
    v10_dir = ROOT / "data" / "v10"

    # Pick frames spread across each session's training rows
    d2_train = train_rows("D2", stride=3)
    v10_train = train_rows("V10", stride=3)
    np.random.seed(7)
    d2_picks = sorted(np.random.choice(d2_train, size=4, replace=False).tolist())
    v10_picks = sorted(np.random.choice(v10_train, size=4, replace=False).tolist())

    # 1. Data exemplars per session
    session_data_exemplar("D2", d2_dir, d2_picks, OUT / "data_exemplars_d2.png")
    session_data_exemplar("V10", v10_dir, v10_picks, OUT / "data_exemplars_v10.png")

    # 2. Shuffled control visualization
    shuffled_pairs("D2", d2_dir, n=4, out_path=OUT / "shuffled_pairs_d2.png")

    # 3. Synthetic positive (D2)
    np.random.seed(11)
    synth_picks = sorted(np.random.choice(d2_train, size=2, replace=False).tolist())
    synthetic_pairs("D2", d2_dir, synth_picks, OUT / "synthetic_positive.png")

    # 4. Noise schedule demo on one frame
    held = held_out_rows("D2")
    held_pick = held[len(held) // 2]
    noise_schedule_demo("D2", d2_dir, held_pick,
                        OUT / "noise_schedule_demo.png")

    print(f"\n[all done] outputs in {OUT}")


if __name__ == "__main__":
    main()
