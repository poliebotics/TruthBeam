# XOF perturbation-sensitivity sweep (Type 1–6) — per-condition AUROC

Backing for the XOF bit-flip / perturbation AUROCs quoted in the paper (Sec 7/8/11): the claim that the verifier is
**insensitive to fine XOF bit-flips** (near chance) while remaining **perfectly sensitive to gross perturbations**.

## Files
- `eval_d2_raw.npz`, `eval_v10_raw.npz` — raw Phase-G eval residuals for the Item-1 **extended-perturbation** sweep
  (the `--extended-perturbations` run, +35 conditions). Each `cond_*` array is shape `(n_frames, 5 timesteps, 4 K)`;
  `n_frames` = 198 (D2) / 200 (V10), the same held-out blocks as the headline result.
- `regenerate_bitflip_auroc.py` — recomputes the table from those npz (pure numpy, no sklearn). Per-frame score =
  mean residual over (timestep, K); AUROC = real-correct vs the perturbed condition.
- `bitflip_auroc_table.csv` — the generated table (52 conditions × {D2, V10}).

## Provenance
These raw eval outputs were produced on the (now-retired) training host and preserved in the R2 evidence tree
(`r2:truthbeam/lambda/experiments/item_1/eval/`). The original `eval/summary.json` aggregated only the *standard*
conditions, so the extended per-condition table was regenerated here from the raw residuals. Regenerated values
reproduce the paper's quoted anchors exactly (e.g. Type-1 global D2 k=1 → **0.503**, k=4096 → **0.599**; V10 k=4096
→ **0.583**; Type-3 region k=256 → **0.505**; Type-4 swap oct0 → **1.000**, oct3 → **0.57**).

## Headline reading
| family | conditions | AUROC | reading |
|---|---|---|---|
| Type 1 global bit-flips | k=1…4096 | 0.503 → 0.599 (D2) | **near chance**, rising only at very large k |
| Type 2 per-octave flips | oct0…3, k=1…64 | 0.503 → 0.588 | sensitivity concentrated in **low octaves**; higher octaves ≈ chance |
| Type 3 localized flips | region k=16…256 | 0.503 → 0.505 | **near chance** |
| Type 4 octave swaps | oct0…3 | 1.000 / 1.000 / 0.96 / 0.57 | gross low-octave swaps **perfectly separable** |
| Type 5 channel swaps | R / G / B | 1.000 / 1.000 / 0.99 | **perfectly separable** |
| Type 6 replacement / calibration | — | 1.000 | **perfectly separable** |

Interpretation (per the paper): fine XOF bit-flips fall below the optical-relevance threshold of the 8-bit,
optically low-pass channel — they barely change the rendered emission `ΔE`, so the verifier (correctly) cannot
distinguish them; separability emerges only as the perturbation grows large enough to move `ΔE`.
