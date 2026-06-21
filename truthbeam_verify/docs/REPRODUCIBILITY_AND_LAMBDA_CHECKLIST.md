# TruthBeam — Reproducibility Checklist

Result of the code-grounded whitepaper verification. The paper's **headline results are all backed by eval-output
JSONs** and were verified directly against the code (not against the paper's prose). This checklist records exactly
which eval artifact backs each number and where it lives in this release.

**Hosting note.** The training box (referred to in older notes as "Lambda") has been **decommissioned**;
**Cloudflare R2 is the system of record** for the bulk artifacts (raw corpus, model weights, full eval trees), with
a Zenodo DOI to be minted (pending at the time of this snapshot) for the curated citable core. See `RESTORE.md` and `DATA_MODEL_PUBLISHING_PLAN.md`. The
red-team / segmentation / perturbation summaries below were recovered from the pre-decommission archive and are
**included in this repository** under `results/redteam_segmentation_evals/`.

## A. In `results/eval/` — headline numbers (paper-reproducible from this repo)

- Phase G main run: AUROC=1.000 correct-vs-wrong on D2 (n=198) and V10 (n=200), Δ_wrong +0.0040/+0.0041, lag curve.
- Shuffled-pairing control AUROC 0.5006, Δ_wrong ~7e-8.
- Synthetic-positive control AUROC 1.000.
- F-A v1 fake rejection AUROC=1.000 at 5k/25k/70k/100k.
- Cross-session leave-one-out verifiers (D2-only on V10, etc.), step 100000.
- Architecture/training config (39.77M params, ControlNet-not-FiLM, 25k steps, 500-step warm-up, 8-bit).

## B. In `results/redteam_segmentation_evals/` — red-team / segmentation / perturbation summaries (recovered + included)

1. **Off-body segmentation ablation** (§07, §08, §11): E2_seg AUROC 0.009–0.538, E3_seg AUROC 1.000 across all 12
   cells, 106/120 usable frames, the σ=4 low-freq recovery (~0.998), and the independent Codex audit.
   → `proper_segmentation/` (`phase4_summary.json` = 12-cell classifications, `SEGMENTATION_RESULTS.md`,
   `CODEX_AUDIT.md`).
2. **EXP 6 per-frame relative ranking** (§07, §11): top-1 = 100% across 120 frames, 51 candidates each (Phase-G
   relative chain-coupling ranking — distinct from the video's in-sample L1 ranking).
   → `causal_ablations/` (`ablation_table.csv`, `manuscript_interpretation.md`).
3. **EXP 7 synthetic counterfactual-E perturbation** (§07): monotone graded Δscore vs. achieved ΔE_rms across three
   preprocessing scales; characterizes verifier sensitivity to synthetic E perturbation (NOT XOF bit-flips).
   → `xof_sensitivity/summary.md` (the directory name is a legacy code label; the report states the experiment).
4. **Synthetic excess-red decline** 67.4%→52.4% across 5k→100k.
   → `excess_red/EXECUTIVE_README.md` (per-step source).
5. **Low-frequency body-region recovery (σ=4)**: `low_freq_mse` retains body-only AUROC ≈ 0.997/0.998 on E2_seg.
   → `low_freq_typicality/` (`interpretation_summary.md`, `precondition_table.csv` — recovered from R2
   `paper_analyses/typicality_layer/phase_0/`).
6. **XOF bit-flip / perturbation sweep (Type 1–6, §07/§08/§11)**: the per-condition AUROC table (fine flips near
   chance, gross perturbations → 1.000).
   → `xof_bitflip/` (`bitflip_auroc_table.csv` + `regenerate_bitflip_auroc.py` + the raw `eval_{d2,v10}_raw.npz`,
   recovered from R2 `item_1/eval/`). Regenerated values reproduce the paper's quoted anchors exactly.

## C. Reproducibility gaps

None outstanding for the quoted results. The XOF bit-flip per-condition table (previously not located as a single
artifact) has been **regenerated from the raw R2 eval residuals and ships in `xof_bitflip/`**; the σ=4 low-frequency
recovery numbers ship in `low_freq_typicality/`. No headline number is unbacked.

## D. One paper↔outreach wording note

The paper says the verifier "never samples or denoises" (precise: it never runs the generative loop); outreach
materials use the colloquial "denoises cleanly = low residual" framing. Both describe the same noise-prediction
residual and are not contradictory; no code/claim change required.
