#!/usr/bin/env python3
"""Regenerate the XOF-perturbation per-condition AUROC table (paper Sec 7/8/11).

Reads the raw Phase-G eval residuals for the Item-1 extended-perturbation sweep
(eval_{d2,v10}_raw.npz: per-condition arrays of shape (n_frames, 5 timesteps, 4 K))
and computes, per condition, AUROC(real-correct vs perturbed) using a per-frame
score = mean residual over (timestep, K). Pure-numpy AUROC (no sklearn needed).

Usage: python3 regenerate_bitflip_auroc.py  ->  writes bitflip_auroc_table.csv
"""
import csv, numpy as np
from pathlib import Path

def auroc(neg, pos):
    s = np.r_[neg, pos]
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    npos, nneg = len(pos), len(neg)
    return (ranks[nneg:].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def per_frame(a): return a.reshape(a.shape[0], -1).mean(axis=1)

here = Path(__file__).resolve().parent
sess = {"D2": np.load(here / "eval_d2_raw.npz", allow_pickle=True),
        "V10": np.load(here / "eval_v10_raw.npz", allow_pickle=True)}
conds = [k for k in sess["D2"].keys() if k.startswith("cond_") and k != "cond_correct"]
rows = []
for c in conds:
    r = {"condition": c.replace("cond_", "")}
    for s, d in sess.items():
        r[f"{s}_auroc"] = round(float(auroc(per_frame(d["cond_correct"]), per_frame(d[c]))), 4)
    rows.append(r)
out = here / "bitflip_auroc_table.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "D2_auroc", "V10_auroc"])
    w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} conditions)")
for r in rows:
    if r["condition"].startswith("xof_"):
        print(f"  {r['condition']:26s} D2={r['D2_auroc']:.3f}  V10={r['V10_auroc']:.3f}")
