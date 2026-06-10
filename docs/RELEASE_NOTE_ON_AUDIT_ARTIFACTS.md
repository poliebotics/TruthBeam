# Release note — how to read the audit artifacts

`docs/verification_ground_truth.json` and `docs/whitepaper_claim_check.json` are **point-in-time** records of the
code-grounded claim verification performed on **2026-06-01**. They are included verbatim as an honesty/audit trail —
they record what was checked, what was backed, and the explicit *do-not-claim* boundaries at that time. Read them
with the following updates in mind:

1. **The training host ("Lambda") has been DELETED.** Where these artifacts say an eval output or model weight
   "lives on Lambda", "is not in the local mirror", "needs Lambda verification", or "confirm retrievable from
   Lambda", read that as **historical**. The host no longer exists. **Cloudflare R2** is now the **system of
   record** for all bulk artifacts (raw capture corpus, model weights, full eval trees); see `RESTORE.md` and
   `docs/DATA_MODEL_PUBLISHING_PLAN.md`.

2. **The flagged red-team / segmentation / perturbation eval outputs were subsequently recovered and are INCLUDED
   in this release** under `results/redteam_segmentation_evals/`:
   - off-body segmentation ablation (`proper_segmentation/`: `phase4_summary.json` = 12-cell classifications,
     `SEGMENTATION_RESULTS.md`, `CODEX_AUDIT.md`),
   - EXP-6 per-frame ranking — raw per-frame artifact now under `results/eval/exp6_correct_e_rank/`
     (`rank_distribution.csv`, n=120 × 51 candidates; the causal-ablation summaries remain in
     `causal_ablations/`),
   - EXP-7 synthetic-E perturbation (`xof_sensitivity/`),
   - excess-red decline (`excess_red/`).
   So claim-check notes of the form "these per-cell numbers have no locatable backing / likely on Lambda" are
   **superseded**: the backing eval artifacts now ship in this repository.

3. **The previously-flagged gaps are now closed.** The merged Type-1–6 XOF bit-flip per-condition table was
   regenerated from the raw R2 eval residuals and ships in `results/redteam_segmentation_evals/xof_bitflip/`
   (with the `eval_{d2,v10}_raw.npz` and the regeneration script); the σ=4 low-frequency body-region recovery
   numbers ship in `results/redteam_segmentation_evals/low_freq_typicality/`. Regenerated AUROCs reproduce the
   paper's quoted anchors exactly. As of the 2026-06-10 fix round (see `AUDIT_FIXES_2026-06-10.md`), the
   frame-level metric table (`results/csv/visual_metrics_wide.csv`) and the raw EXP-6 ranking artifact also
   ship in-repo, so every headline number is backed by a shipped primary artifact (with one documented
   fourth-decimal aggregation difference on the frame-level shuffled contrast, reconciled in paper §6).

4. **Model weights** (Phase G `main`/`shuffled`/`synthetic_positive`, F-A v1 checkpoints, the 14 binders) are **not
   committed** to this code repository by design (size + deliberate checkpoint selection). They are released via R2 /
   Zenodo per the data plan. Where an audit note says "checkpoints/ is empty locally / weights lived on Lambda",
   that is expected: the weights are an R2/Zenodo artifact, not a git artifact.

In short: the audit artifacts are an accurate record of the **2026-06-01** state; this release **closed** the
"recover the flagged eval outputs" action (they are in `results/`) and **migrated** all bulk storage from the
now-deleted training host to **R2**.
