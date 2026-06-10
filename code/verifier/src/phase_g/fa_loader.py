"""Stage 0 — F-A v1 checkpoint loader.

Loads an `EditorControlNet` from a Phase F-A v1 checkpoint and provides a
function to produce C_fake at Phase G's diffusion-verifier input resolution
(4 × 768 × 1024) given (target row, source row, session).

Pipeline:
    1. Load .raw → packed CFA (4, 2300, 2660) for both source and target rows.
    2. Load emission tile.png → (3, 1080, 1920) for both.
    3. Run EditorControlNet(C_source, E_source, E_target) → C_pred at 2300×2660.
    4. Crop y=[0:1704], x=[155:2433] → resize to 768×1024 via area interp.
       (Same crop spec as Phase G's diffusion model input.)

This is the C_fake input that Stage 0's diffusion-verifier eval uses.

The F-A v1 model defaults (verified against checkpoint args):
    capture_h=2300, capture_w=2660
    emission_h=1080, emission_w=1920
    hint_use_source=True (v1 default; v1.5 ablation has v1_control variants)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.editor_controlnet import EditorControlNet  # noqa: E402
from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa  # noqa: E402
from data.emission_dataset import load_emission_at  # noqa: E402


# Phase G diffusion model input shape
TARGET_C_H = 768
TARGET_C_W = 1024
TARGET_E_H = 768
TARGET_E_W = 1024

# Phase G crop spec (matches src/phase_g/diffusion_diagnostic_dataset.py)
CROP_Y0, CROP_Y1 = 0, 1704
CROP_X0, CROP_X1 = 155, 2433

# F-A v1 native shapes (defaults per checkpoint)
FA_CAP_H = 2300
FA_CAP_W = 2660
FA_EM_H = 1080
FA_EM_W = 1920


def load_fa_v1_checkpoint(ckpt_path: Path, device: torch.device,
                          dtype: torch.dtype = torch.bfloat16) -> EditorControlNet:
    """Load an EditorControlNet from a F-A v1 checkpoint.

    The checkpoint has keys {step, editor, optimizer, args}. The args dict
    may not have all fields (older trainer); we use safe defaults.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("editor", ck)
    args = ck.get("args", {}) if isinstance(ck, dict) else {}
    cap_h = args.get("capture_h") or FA_CAP_H
    cap_w = args.get("capture_w") or FA_CAP_W
    em_h = args.get("emission_h") or FA_EM_H
    em_w = args.get("emission_w") or FA_EM_W
    hint_mode = args.get("hint_mode") or "v1_5_treatment"
    hint_use_source = (hint_mode == "v1_5_treatment")
    init_mode = "exp001c-warm-start" if args.get("exp001c_ckpt_warm_start") else "scratch"

    model = EditorControlNet(
        capture_h=cap_h, capture_w=cap_w,
        emission_h=em_h, emission_w=em_w,
        init_mode=init_mode,
        hint_use_source=hint_use_source,
    )
    # The original training applied zero_init_source_channels() before the
    # first step, then the optimizer overrode those weights. Loading
    # state_dict directly doesn't need that step (loaded weights are post-
    # training, not init).
    model.load_state_dict(state)
    model = model.to(device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_C_native(session_dir: Path, row: int) -> torch.Tensor:
    """Load (4, 2300, 2660) packed CFA from row's .raw, float32 [0, 1]."""
    raw = (session_dir / "Recordings" / f"frame_{row:06d}.raw").read_bytes()
    cfa = bayer_rg8_to_packed_cfa(raw)
    return torch.from_numpy(cfa.astype(np.float32) / 255.0)


def load_E_native(session_dir: Path, row: int) -> torch.Tensor:
    """Load (3, 1080, 1920) emission tile from PNG, float32 [0, 1]."""
    em_path = session_dir / "derived" / "Emissions" / f"tile_{row:06d}.png"
    return load_emission_at(em_path, FA_EM_H, FA_EM_W)


def C_native_to_phase_g_input(C_native: torch.Tensor) -> torch.Tensor:
    """Transform (4, 2300, 2660) → (4, 768, 1024) via Phase G's crop+resize."""
    cropped = C_native[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]  # (4, 1704, 2278)
    out = F.interpolate(cropped.unsqueeze(0).float(),
                        size=(TARGET_C_H, TARGET_C_W),
                        mode="area").squeeze(0)
    return out


@torch.no_grad()
def render_C_fake(model: EditorControlNet,
                  session_dir: Path,
                  source_row: int, target_row: int,
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Run F-A on (source_row, target_row) → return (4, 768, 1024) C_fake
    at Phase G's diffusion verifier input resolution.

    F-A is given C_source, E_source, E_target — all at native resolution.
    """
    # Defensive: ensure eval mode even though load_fa_v1_checkpoint sets it.
    # Costs nothing if already in eval, prevents accidental drift if any
    # caller flips the model to train.
    model.eval()
    C_s = load_C_native(session_dir, source_row).to(device, dtype=dtype).unsqueeze(0)
    E_s = load_E_native(session_dir, source_row).to(device, dtype=dtype).unsqueeze(0)
    E_t = load_E_native(session_dir, target_row).to(device, dtype=dtype).unsqueeze(0)
    C_pred = model(C_s, E_s, E_t)  # (1, 4, 2300, 2660), still in [0, 1]
    return C_native_to_phase_g_input(C_pred.squeeze(0).float())


def self_test(ckpt_path: Path = None, session_dir: Path = None) -> None:
    """Round-trip sanity: load a F-A checkpoint, run on a held-out frame,
    verify shapes and value range."""
    if ckpt_path is None:
        ckpt_path = (Path("/path/to/poliebotics_phase_b/experiments/phase_f")
                     / "f_a_full_v1/checkpoints/step_00100000.pt")
    if session_dir is None:
        session_dir = Path("/path/to/poliebotics_phase_b/data/d2")

    if not ckpt_path.exists():
        print(f"[self_test] checkpoint missing at {ckpt_path}; skipping")
        return
    if not (session_dir / "Recordings").exists():
        print(f"[self_test] session_dir invalid {session_dir}; skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[self_test] loading {ckpt_path} on {device} dtype={dtype}...")
    model = load_fa_v1_checkpoint(ckpt_path, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[self_test] EditorControlNet loaded, params={n_params/1e6:.1f}M")

    # Use rows 1500 and 1502 (target=1502, source=1500, lag=2)
    target_row = 1502
    source_row = 1500
    print(f"[self_test] rendering C_fake for target={target_row} source={source_row}...")
    import time
    t0 = time.time()
    C_fake = render_C_fake(model, session_dir, source_row, target_row, device, dtype)
    elapsed = time.time() - t0
    print(f"[self_test] C_fake shape={tuple(C_fake.shape)} dtype={C_fake.dtype}")
    print(f"[self_test]   min={C_fake.min().item():.4f}  max={C_fake.max().item():.4f}  "
          f"mean={C_fake.mean().item():.4f}")
    print(f"[self_test]   render time: {elapsed:.2f}s")

    # Compare against C_real_target at Phase G input resolution
    C_real = load_C_native(session_dir, target_row)
    C_real_pg = C_native_to_phase_g_input(C_real)
    diff = (C_fake.float().cpu() - C_real_pg.float()).abs().mean().item()
    print(f"[self_test]   mean abs diff vs C_real_target: {diff:.4f}")


if __name__ == "__main__":
    self_test()
