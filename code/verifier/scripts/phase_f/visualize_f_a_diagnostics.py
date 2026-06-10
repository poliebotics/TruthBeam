"""F-A step-N visual diagnostics — three grids.

Operator-requested. Read-only on a saved checkpoint; uses CPU only so no
GPU contention with the running F-A training.

Grids produced:
  - grid_target_sweep.png     4 sources × 6 E_targets = 24 predictions
                               Looking for: row-internal variation across columns
  - grid_context_ablation.png 1 source × 3 conditions
                               (correct E_target, zeroed E_target, permuted-cross-session E_target)
                               Looking for: A/B/C visibly differ
  - grid_ground_truth.png     4 examples × 1×4 layout
                               (C_source, E_target, C_pred, C_target_REAL)
                               Looking for: C_pred resembles C_target_REAL

All sources drawn from D2 selection_gate_normal slice [4792, 5392) — rows
the F-A editor did NOT train on (Phase F train_normal = D2 [0, 4194)).

Run:
  python scripts/phase_f/visualize_f_a_diagnostics.py \
    --ckpt results/f_a_step5000_diagnostics/step_00005000.pt \
    --d2-dir data/d2 --v10-dir data/v10 \
    --out results/f_a_step5000_diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import packed_cfa_to_bayer_rg8  # noqa: E402
from phase_f.editor_controlnet import EditorControlNet  # noqa: E402
from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402


# Phase F val slice (selection_gate_normal) on D2 — model didn't train here
VAL_RANGE_D2 = (4792, 5392)
VAL_RANGE_V10 = (2993, 3368)


def cfa_to_rgb_small(cfa_float01: torch.Tensor, target_w: int = 320) -> np.ndarray:
    cfa_u8 = (cfa_float01.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    bayer_bytes = packed_cfa_to_bayer_rg8(cfa_u8)
    bayer = np.frombuffer(bayer_bytes, dtype=np.uint8).reshape(4600, 5320)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB_EA)
    h_orig, w_orig = rgb.shape[:2]
    target_h = int(h_orig * target_w / w_orig)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def emission_to_rgb_small(em_float01: torch.Tensor, target_w: int = 320) -> np.ndarray:
    rgb = (em_float01.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))
    h_orig, w_orig = rgb.shape[:2]
    target_h = int(h_orig * target_w / w_orig)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def stack_grid(rows: list[list[np.ndarray]], pad: int = 8,
               bg=(24, 24, 28)) -> np.ndarray:
    cell_h = max(im.shape[0] for r in rows for im in r)
    cell_w = max(im.shape[1] for r in rows for im in r)
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    canvas = np.full(
        (n_rows * cell_h + (n_rows + 1) * pad,
         n_cols * cell_w + (n_cols + 1) * pad, 3),
        bg, dtype=np.uint8,
    )
    for ri, row in enumerate(rows):
        for ci, im in enumerate(row):
            ih, iw = im.shape[:2]
            y0 = pad + ri * (cell_h + pad)
            x0 = pad + ci * (cell_w + pad)
            canvas[y0:y0 + ih, x0:x0 + iw] = im
    return canvas


def load_editor(ckpt_path: Path, device: torch.device) -> EditorControlNet:
    print(f"[load] reading {ckpt_path} ...", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["editor"] if "editor" in ck else ck.get("model", ck)
    # Strip any "module." prefix from DDP-wrapped state_dicts (safety)
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    editor = EditorControlNet(
        capture_h=2300, capture_w=2660,
        emission_h=1080, emission_w=1920,
        init_mode="random",  # we're loading weights so no warm-start needed
    ).to(device)
    missing, unexp = editor.load_state_dict(state, strict=False)
    print(f"[load] state loaded; missing={len(missing)} unexp={len(unexp)}", flush=True)
    editor.eval()
    return editor


@torch.no_grad()
def forward(editor, C_source, E_source, E_target, dtype) -> torch.Tensor:
    out = editor(C_source.to(dtype), E_source.to(dtype), E_target.to(dtype))
    return out.float().clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-rows", type=int, nargs="+",
                    default=[4900, 5050, 5200, 5350])  # 4 val rows
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--source-k", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # no GPU contention with running F-A
    dtype = torch.float32  # CPU works in fp32; bf16 on CPU is slow

    # Extract step number from filename like step_00030000.pt → 30000
    import re
    m = re.search(r"step_(\d+)", args.ckpt.name)
    step_label = int(m.group(1)) if m else 0

    editor = load_editor(args.ckpt, device)
    n_params = sum(p.numel() for p in editor.parameters())
    print(f"[init] editor params={n_params/1e6:.1f}M; running on CPU (no GPU contention)",
          flush=True)

    # Sample 6 different E_target frames from D2 val for Grid 1
    rng = np.random.RandomState(42)
    val_d2 = list(range(*VAL_RANGE_D2))
    val_v10 = list(range(*VAL_RANGE_V10))
    target_pool_d2 = sorted(rng.choice(val_d2, size=8, replace=False).tolist())
    target_pool_v10 = sorted(rng.choice(val_v10, size=2, replace=False).tolist())
    print(f"[init] target pool D2 = {target_pool_d2}", flush=True)
    print(f"[init] target pool V10 = {target_pool_v10}", flush=True)

    provenance = {
        "ckpt": str(args.ckpt),
        "session_for_sources": "D2",
        "source_rows": args.source_rows,
        "source_k": args.source_k,
        "val_range_d2_selection_gate_normal": list(VAL_RANGE_D2),
        "val_range_v10_selection_gate_normal": list(VAL_RANGE_V10),
        "grid1_target_pool_d2": target_pool_d2[:6],
        "grid2_permuted_target_v10_row": target_pool_v10[0],
        "device": "cpu",
        "dtype": "fp32",
    }

    t0 = time.time()
    W = 320

    # ---------- Grid 1: 4 sources × 6 E_targets ----------
    print("\n--- Grid 1: target-conditioning sweep ---", flush=True)
    grid1_rows = []
    # Header row: blank | E_target_1 | E_target_2 | ... | E_target_6
    header = [np.full((W * 9 // 16, W, 3), (40, 40, 50), dtype=np.uint8)]  # blank corner
    header[0] = label(header[0], "(source)")
    targets_used = target_pool_d2[:6]
    for t_target in targets_used:
        em = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{t_target:06d}.png",
                              1080, 1920)
        header.append(label(emission_to_rgb_small(em, W), f"E_target t={t_target}"))
    grid1_rows.append(header)

    for src_t in args.source_rows:
        cap_source = load_capture_at(args.d2_dir / "Recordings" / f"frame_{src_t:06d}.raw",
                                     2300, 2660).unsqueeze(0)
        em_source = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{src_t:06d}.png",
                                     1080, 1920).unsqueeze(0)
        row_imgs = [label(cfa_to_rgb_small(cap_source.squeeze(0), W),
                          f"C_source t={src_t}")]
        for t_target in targets_used:
            em_target = load_emission_at(
                args.d2_dir / "derived" / "Emissions" / f"tile_{t_target:06d}.png",
                1080, 1920).unsqueeze(0)
            t_step = time.time()
            pred = forward(editor, cap_source, em_source, em_target, dtype)
            print(f"  src={src_t}  tgt={t_target}  forward={time.time()-t_step:.1f}s",
                  flush=True)
            row_imgs.append(label(cfa_to_rgb_small(pred.squeeze(0), W),
                                  f"C_pred (t_t={t_target})"))
        grid1_rows.append(row_imgs)

    grid1 = stack_grid(grid1_rows, pad=8)
    cv2.imwrite(str(args.out / "grid_target_sweep.png"),
                cv2.cvtColor(grid1, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"  [done] grid_target_sweep.png shape={grid1.shape}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

    # ---------- Grid 2: 1 source × 3 context conditions ----------
    print("\n--- Grid 2: context-use ablation ---", flush=True)
    src_t = args.source_rows[0]
    target_t_correct = src_t + args.source_k
    cap_source = load_capture_at(args.d2_dir / "Recordings" / f"frame_{src_t:06d}.raw",
                                 2300, 2660).unsqueeze(0)
    em_source = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{src_t:06d}.png",
                                 1080, 1920).unsqueeze(0)
    em_correct = load_emission_at(
        args.d2_dir / "derived" / "Emissions" / f"tile_{target_t_correct:06d}.png",
        1080, 1920).unsqueeze(0)
    em_zero = torch.zeros_like(em_correct)
    em_v10 = load_emission_at(
        args.v10_dir / "derived" / "Emissions" / f"tile_{target_pool_v10[0]:06d}.png",
        1080, 1920).unsqueeze(0)

    pred_correct = forward(editor, cap_source, em_source, em_correct, dtype)
    pred_zero = forward(editor, cap_source, em_source, em_zero, dtype)
    pred_v10 = forward(editor, cap_source, em_source, em_v10, dtype)

    # 2-row layout: top row = E_targets fed; bottom row = C_pred outputs
    grid2_top = [
        label(cfa_to_rgb_small(cap_source.squeeze(0), W), f"C_source t={src_t}"),
        label(emission_to_rgb_small(em_correct.squeeze(0), W),
              f"E_target correct (t={target_t_correct})"),
        label(emission_to_rgb_small(em_zero.squeeze(0), W),
              "E_target zeroed"),
        label(emission_to_rgb_small(em_v10.squeeze(0), W),
              f"E_target permuted (V10 t={target_pool_v10[0]})"),
    ]
    grid2_bot = [
        np.full_like(grid2_top[0], (24, 24, 28)),  # blank under C_source
        label(cfa_to_rgb_small(pred_correct.squeeze(0), W), "C_pred A: correct"),
        label(cfa_to_rgb_small(pred_zero.squeeze(0), W), "C_pred B: zeroed"),
        label(cfa_to_rgb_small(pred_v10.squeeze(0), W), "C_pred C: permuted"),
    ]
    grid2 = stack_grid([grid2_top, grid2_bot], pad=8)
    cv2.imwrite(str(args.out / "grid_context_ablation.png"),
                cv2.cvtColor(grid2, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"  [done] grid_context_ablation.png shape={grid2.shape}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

    # ---------- Grid 3: 4 examples × (C_source, E_target, C_pred, C_target_REAL) ----------
    print("\n--- Grid 3: ground-truth comparison ---", flush=True)
    grid3_rows = []
    for src_t in args.source_rows:
        target_t = src_t + args.source_k
        cs = load_capture_at(args.d2_dir / "Recordings" / f"frame_{src_t:06d}.raw",
                             2300, 2660).unsqueeze(0)
        es = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{src_t:06d}.png",
                              1080, 1920).unsqueeze(0)
        et = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{target_t:06d}.png",
                              1080, 1920).unsqueeze(0)
        ct_real = load_capture_at(args.d2_dir / "Recordings" / f"frame_{target_t:06d}.raw",
                                  2300, 2660).unsqueeze(0)
        pred = forward(editor, cs, es, et, dtype)
        grid3_rows.append([
            label(cfa_to_rgb_small(cs.squeeze(0), W), f"C_source t={src_t}"),
            label(emission_to_rgb_small(et.squeeze(0), W), f"E_target t={target_t}"),
            label(cfa_to_rgb_small(pred.squeeze(0), W), "C_pred"),
            label(cfa_to_rgb_small(ct_real.squeeze(0), W), f"C_target REAL t={target_t}"),
        ])
    grid3 = stack_grid(grid3_rows, pad=8)
    cv2.imwrite(str(args.out / "grid_ground_truth.png"),
                cv2.cvtColor(grid3, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"  [done] grid_ground_truth.png shape={grid3.shape}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

    # ---------- README + provenance ----------
    readme = [
        f"# F-A step-{step_label} visual diagnostics",
        "",
        f"Checkpoint: `{args.ckpt}` (step {step_label} of 100,000, {step_label/1000:.1f}%).",
        f"Editor architecture: `EditorControlNet` (42.3M params, ControlNet-style spatial conditioning).",
        f"Loss: `mse_plus_hinge_wrongs` (coef_recon=1.0, coef_grad=0.1, coef_binder=5.0, margin=0.02, n_hard_wrongs=6).",
        f"Inference device: CPU (no GPU contention with running F-A).",
        "",
        "## Source data",
        f"All source frames drawn from D2 `selection_gate_normal` slice [{VAL_RANGE_D2[0]}, {VAL_RANGE_D2[1]}).",
        f"The F-A editor did NOT train on these rows (Phase F train_normal = D2 [0, 4194)).",
        "",
        f"Source rows used: {args.source_rows}",
        f"Source-target offset (k): {args.source_k}",
        "",
        "## Grid 1 — `grid_target_sweep.png`",
        "",
        f"4 source frames × 6 E_target inputs = 24 model outputs.",
        f"E_target rows sampled from D2 val: {targets_used}",
        "",
        "**What to look for**: within a single source row, do the 6 C_pred outputs differ visibly across columns?",
        "  - If yes → editor is using E_target.",
        "  - If no (all 6 outputs in a row look the same) → editor is ignoring E_target.",
        "",
        "Diversity_mean=0.01699 at this checkpoint suggests a positive but subtle response. Compare visually.",
        "",
        "## Grid 2 — `grid_context_ablation.png`",
        "",
        f"1 source frame ({args.source_rows[0]}) × 3 conditioning ablations:",
        f"- Column A: model output with correct E_target (D2 row {args.source_rows[0]+args.source_k})",
        f"- Column B: model output with zeroed E_target (all-zero conditioning)",
        f"- Column C: model output with permuted E_target (V10 row {target_pool_v10[0]} — different session entirely)",
        "",
        "**What to look for**: do A, B, C visibly differ?",
        "  - If A ≠ B ≠ C, model uses conditioning.",
        "  - If A == B == C, model ignores conditioning.",
        "  - If A ≈ B but A ≠ C, model uses high-frequency emission detail but not low-frequency.",
        "",
        "## Grid 3 — `grid_ground_truth.png`",
        "",
        "4 examples × (C_source, E_target, C_pred, C_target REAL) layout.",
        "C_target REAL is the actual capture under that E_target — gold-standard comparison.",
        "",
        "**What to look for**: does C_pred resemble C_target REAL?",
        f"  - At step {step_label} / 100000 ({step_label/1000:.1f}% of training), expect rough match at best.",
        "  - The signal is whether C_pred captures the projection structure visible in C_target.",
        "",
        "## Provenance",
        f"```json",
        json.dumps(provenance, indent=2),
        "```",
        "",
        f"Total wall-clock for diagnostics: {time.time()-t0:.0f}s on CPU.",
    ]
    (args.out / "README.md").write_text("\n".join(readme))
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"\n[done] all outputs in {args.out}/", flush=True)


if __name__ == "__main__":
    main()
