"""Side-by-side capture↔emission pairing visual for a v9 session.

Takes a session_dir, samples matched + deliberately-mismatched rows,
debayers the captures, contrast-normalizes each panel for visibility,
stacks everything into a vertical panel, saves to
<session_dir>/correspondence_check/pairings.png.

Also copies itself into the session's correspondence_check/ dir so the
artifact is self-contained and reproducible from the bundle alone.

Usage:
    python3 tools/pairing_visual.py --session-dir <path>
    python3 tools/pairing_visual.py --session-dir <path> --n-matched 10 --n-mismatched 3
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW_H, RAW_W = 4600, 5320
BBY = (26, 3410)   # full projection-containing crop in raw Bayer
BBX = (300, 4970)
PANEL_HW = (256, 512)   # per-panel display resolution


def load_F_cropped(raw_dir: Path, t: int) -> np.ndarray:
    """Raw Bayer → direct per-channel extraction → bbox crop → resize.
    Returns (H, W, 3) float32 in [0, 1]."""
    raw = np.fromfile(raw_dir / f"frame_{t:06d}.raw", dtype=np.uint8)
    bayer = raw.reshape(RAW_H, RAW_W)
    crop = bayer[BBY[0]:BBY[1], BBX[0]:BBX[1]]
    R = crop[0::2, 0::2].astype(np.float32)
    G = (crop[0::2, 1::2].astype(np.float32) +
         crop[1::2, 0::2].astype(np.float32)) * 0.5
    B = crop[1::2, 1::2].astype(np.float32)
    R = cv2.resize(R, (PANEL_HW[1], PANEL_HW[0]), interpolation=cv2.INTER_AREA)
    G = cv2.resize(G, (PANEL_HW[1], PANEL_HW[0]), interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (PANEL_HW[1], PANEL_HW[0]), interpolation=cv2.INTER_AREA)
    return np.stack([R, G, B], axis=-1) / 255.0


def load_E(em_dir: Path, t: int) -> np.ndarray:
    img = cv2.imread(str(em_dir / f"tile_{t:06d}.png"), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ds = cv2.resize(rgb, (PANEL_HW[1], PANEL_HW[0]),
                    interpolation=cv2.INTER_AREA)
    return ds.astype(np.float32) / 255.0


def contrast_normalize(img: np.ndarray) -> np.ndarray:
    """Per-channel p1-p99 stretch for visibility. Does NOT affect any
    numeric correlation; this is a display-only adjustment."""
    out = np.zeros_like(img)
    for c in range(3):
        v = img[:, :, c]
        lo = np.percentile(v, 1)
        hi = np.percentile(v, 99)
        if hi - lo < 1e-6:
            out[:, :, c] = 0
        else:
            out[:, :, c] = (v - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def per_channel_pearson_mean(F: np.ndarray, E: np.ndarray) -> float:
    vals = []
    for c in range(3):
        f = F[:, :, c].flatten()
        e = E[:, :, c].flatten()
        if f.std() < 1e-6 or e.std() < 1e-6:
            continue
        vals.append(float(np.corrcoef(f, e)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def find_chain_log(session_dir: Path) -> Path:
    for cand in (session_dir / "chain_log.csv",
                 session_dir / "chain" / "chain_log.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no chain_log.csv under {session_dir}")


def count_chain_rows(chain_log: Path) -> int:
    with open(chain_log) as f:
        f.readline()  # comment
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True, type=Path)
    ap.add_argument("--n-matched", type=int, default=10,
                    help="Matched pairs (cap_t ↔ tile_t.png). Default 10.")
    ap.add_argument("--n-mismatched", type=int, default=3,
                    help="Deliberately-mismatched pairs (cap_t ↔ tile_t'.png "
                         "for t ≠ t'). Default 3.")
    ap.add_argument("--skip-first", type=int, default=15,
                    help="Skip the first N rows (startup dark period).")
    args = ap.parse_args()

    session_dir = args.session_dir
    raw_dir = session_dir / "Recordings"
    em_dir = session_dir / "derived" / "Emissions"
    out_dir = session_dir / "correspondence_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_log = find_chain_log(session_dir)
    n_rows = count_chain_rows(chain_log)
    print(f"session: {session_dir.name}")
    print(f"n_rows:  {n_rows}")

    # Sample matched rows — spread evenly across middle + end.
    start = max(args.skip_first, 0)
    if start >= n_rows:
        print(f"FAIL: session too short (n_rows={n_rows} <= skip_first={start})")
        return 2
    matched_rows = np.linspace(
        start, n_rows - 1, args.n_matched).astype(int).tolist()
    # De-dupe in case n_rows is small.
    matched_rows = sorted(set(matched_rows))

    # Mismatched pairs. Pair capture from rows in the middle with emission
    # from rows far away. Deterministic offsets for reproducibility.
    mismatched_pairs = []
    if args.n_mismatched > 0:
        mm_caps = np.linspace(
            start + 2, n_rows - 10, args.n_mismatched).astype(int).tolist()
        for t_cap in mm_caps:
            # Pair with a row at least n_rows//3 away.
            offset = max(n_rows // 3, 20)
            t_em = (t_cap + offset) % n_rows
            if t_em == t_cap:
                t_em = (t_em + 10) % n_rows
            mismatched_pairs.append((t_cap, t_em))

    print(f"matched rows:       {matched_rows}")
    print(f"mismatched pairs:   {mismatched_pairs}")

    # Build the panel.
    rows = [(t, t, "MATCHED") for t in matched_rows]
    rows += [(t_cap, t_em, "MISMATCHED") for (t_cap, t_em) in mismatched_pairs]

    fig, axes = plt.subplots(
        len(rows), 2, figsize=(10, 2.4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, 2)

    for i, (t_cap, t_em, label) in enumerate(rows):
        try:
            F = load_F_cropped(raw_dir, t_cap)
            E = load_E(em_dir, t_em)
            corr = per_channel_pearson_mean(F, E)
            F_disp = contrast_normalize(F)
            E_disp = contrast_normalize(E)

            axes[i, 0].imshow(F_disp)
            title_l = f"capture t={t_cap}  (debayered, bbox-cropped, p1-p99 stretched)"
            axes[i, 0].set_title(title_l, fontsize=8)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(E_disp)
            lbl_color = "red" if label == "MISMATCHED" else "green"
            title_r = (f"emission t={t_em}  [{label}]  "
                       f"Pearson mean={corr:+.3f}")
            axes[i, 1].set_title(title_r, fontsize=8, color=lbl_color)
            axes[i, 1].axis("off")
        except Exception as e:
            print(f"  row {i} t_cap={t_cap} t_em={t_em} FAILED: {e!r}")
            axes[i, 0].text(0.5, 0.5, f"error: {e}",
                            ha="center", va="center", fontsize=8)
            axes[i, 1].axis("off")

    fig.suptitle(
        f"v9 pairing visual — session {session_dir.name}\n"
        f"matched ({len(matched_rows)}): cap_t ↔ tile_t.png  |  "
        f"mismatched ({len(mismatched_pairs)}): labelled in red",
        fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "pairings.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # Self-copy into the session so the artifact is reproducible.
    self_copy_dest = out_dir / "pairing_visual.py"
    try:
        shutil.copy2(Path(__file__), self_copy_dest)
        print(f"copied script to: {self_copy_dest}")
    except Exception as e:
        print(f"WARN: script self-copy failed: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
