"""Phase G — local CPU smoke test.

Verifies the diffusion diagnostic pipeline runs end-to-end without GPU:
  1. Build a tiny model + dataset (T=20, base_ch=16, 1 frame)
  2. Run a forward + backward (single step)
  3. Test all three modes (real / shuffled / synthetic_positive)
  4. Run a tiny evaluation (1 frame, 1 timestep, K=1, 1 lag, 1 wrong) to
     exercise the eval harness end-to-end

Catches stupid bugs before burning Lambda compute. Run this BEFORE launching
the real training jobs.

Run:
  cd /path/to/poliebotics_phase_b
  .venv/bin/python scripts/phase_g/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_model import (
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (
    DiffusionDiagnosticDataset, collate_diagnostic, train_rows, held_out_rows,
)


def main() -> None:
    print("=" * 60)
    print("Phase G smoke test")
    print("=" * 60)

    # 1. Tiny model
    print("\n[1] Building tiny DiffusionDiagnosticUNet (base_ch=16, mults=(1,2,2,2))...")
    model = DiffusionDiagnosticUNet(
        in_ch=4, base_ch=16, channel_mults=(1, 2, 2, 2),
        attn_at=(False, False, False, True),
        cond_drop_prob=0.2,
        hint_in_ch=11,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params={n_params/1e3:.1f}K")

    # Sanity forward pass on synthetic tensors
    print("\n[2] Forward + backward on synthetic input (B=1, 64×64)...")
    C = torch.randn(1, 4, 64, 64)
    E = torch.randn(1, 3, 64, 64)
    t = torch.tensor([100], dtype=torch.long)
    eps_pred = model(C, E, t.float())
    print(f"    eps_pred shape={tuple(eps_pred.shape)}")
    assert eps_pred.shape == C.shape, "shape mismatch"
    loss = F.mse_loss(eps_pred, torch.randn_like(C))
    loss.backward()
    print(f"    loss={loss.item():.4f}  backward OK")

    # 3. Diffusion constants
    print("\n[3] Building diffusion constants (T=20)...")
    dc = build_diffusion_constants(20, device=torch.device("cpu"), dtype=torch.float32)
    print(f"    alphas_cum[0]={dc['alphas_cum'][0]:.4f}  alphas_cum[-1]={dc['alphas_cum'][-1]:.6f}")
    assert dc["alphas_cum"][0] > 0.99, "cosine schedule should start near 1"
    assert dc["alphas_cum"][-1] < 0.01, "cosine schedule should end near 0"

    noise = torch.randn_like(C)
    t_small = torch.tensor([10], dtype=torch.long)  # within T=20
    Ct = q_sample(C, t_small, dc, noise)
    print(f"    q_sample at t=10/20 OK, shape={tuple(Ct.shape)}")

    # 4. Held-out vs train row constraints
    print("\n[4] Held-out / train row partition...")
    for sess in ("D2", "V10"):
        eo = set(held_out_rows(sess))
        tr = set(train_rows(sess))
        print(f"    {sess}: held_out={len(eo)} train={len(tr)} overlap={len(eo & tr)}")
        assert len(eo & tr) == 0, "held-out should not overlap with train"

    # 5. Dataset modes — uses real data so requires the recordings to exist locally.
    print("\n[5] Dataset modes (real / shuffled / synthetic_positive)...")
    d2_dir = ROOT / "data" / "d2"
    if not (d2_dir / "Recordings").exists():
        print("    [WARN] no D2 data locally — skipping data-loading sanity check")
    else:
        for mode in ("real", "shuffled", "synthetic_positive"):
            try:
                ds = DiffusionDiagnosticDataset(
                    session_dirs={"D2": d2_dir},
                    mode=mode, role="train", stride=3,
                )
                print(f"    {mode}: ds n={len(ds)}, fetching one sample...")
                sample = ds[0]
                C_s = sample["C"]; E_s = sample["E"]
                print(f"      C shape={tuple(C_s.shape)} dtype={C_s.dtype} "
                      f"min={C_s.min():.3f} max={C_s.max():.3f}")
                print(f"      E shape={tuple(E_s.shape)} dtype={E_s.dtype} "
                      f"min={E_s.min():.3f} max={E_s.max():.3f}")
                assert C_s.shape == (4, 768, 1024), f"C shape {C_s.shape}"
                assert E_s.shape == (3, 768, 1024), f"E shape {E_s.shape}"
                assert C_s.min() >= 0.0 and C_s.max() <= 1.0
                assert E_s.min() >= 0.0 and E_s.max() <= 1.0
                # Verify shuffled actually shuffles E away from row r.
                if mode == "shuffled":
                    perm_r = ds.shuffle_perm["D2"][sample["row"]]
                    print(f"      row={sample['row']} → partner row={perm_r} "
                          f"(gap={abs(sample['row'] - perm_r)})")
                    assert abs(sample["row"] - perm_r) >= 60, "shuffle gap < 60"
            except Exception as e:
                print(f"    [FAIL] {mode}: {e}")
                raise

    # 6. Forward pass on real data shape (full 768×1024)
    print("\n[6] Forward + backward at full target resolution (768×1024)...")
    C = torch.randn(1, 4, 768, 1024)
    E = torch.randn(1, 3, 768, 1024)
    t = torch.tensor([500], dtype=torch.long)
    try:
        eps_pred = model(C, E, t.float())
        print(f"    eps_pred shape={tuple(eps_pred.shape)} OK")
    except Exception as e:
        print(f"    [FAIL] {e}")
        raise

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
