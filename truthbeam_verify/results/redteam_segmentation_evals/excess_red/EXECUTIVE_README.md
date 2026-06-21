# Substrate-verification visualization — overnight run

**Generated:** 2026-05-09 (claude_visual + codex_visual collaboration)

> **Note on linked artifacts.** This file is the **summary** of the overnight visualization run. The promoted
> visual artifacts it references — `best_panels/`, `best_panels_index.md`, `leaderboards/leaderboard.csv`, demo
> videos, and the per-cycle scripts — are **not included in this code repository**; they are bulk visualization
> outputs hosted on R2 (see `RESTORE.md` / `docs/DATA_MODEL_PUBLISHING_PLAN.md`). The quantitative claims below are
> self-contained in this summary; the linked panels are illustrative.

## TL;DR

The chain-coupled optical recording substrate produces a **near-perfect**
visual classifier between real correct-E captures and any of:

- shuffled-E captures (frame-level **AUROC = 0.9999**)
- fake-C captures at attacker training step 100k (frame-level **AUROC = 0.9998**)

Globally, real-correct frames are essentially green (mean red 0.018% under the
recommended recipe), while shuffled / fake frames are heavily red on the excess
panel (~52-54% mean red area).

## What to look at first

| Goal | File |
|---|---|
| 60-second demo (video) | `best_panels/cycle_0028_long_montage.mp4` (1.2 MB, 56s) |
| 15-second demo (video) | `best_panels/cycle_0015_demo.mp4` |
| Attacker progression video | `best_panels/cycle_0022_attacker_morph.mp4` (6s) |
| Single-frame demo opener | `best_panels/cycle_0008_delta_hero_fake_D2_002086.png` |
| **Hero composite (paper front-page)** | `best_panels/cycle_0026_hero_composite.png` (or `_300dpi.png`) |
| **Best-of-best (codex+claude)** | `best_panels/cycle_0033_combined_hero.png` (or `_300dpi.png`) |
| **Manuscript main figure (1A-D)** | `best_panels/cycle_0030_paper_figure_synthesis.{png,pdf}` |
| Manuscript supplement: attacker progression | `best_panels/cycle_0007_fake_step_progression_panel.png` |
| Manuscript supplement: AUROC quantification | `best_panels/cycle_0024_roc_calibration.png` |
| Manuscript supplement: distributional evidence | `best_panels/cycle_0014_zscore_histograms.png` |
| Manuscript supplement: ECDF | `best_panels/cycle_0036_ecdf.png` |
| Manuscript supplement: print-quality figure | `best_panels/cycle_0019_manuscript_figure_2row_300dpi.png` (or `.pdf`, `.svg`) |
| Multi-method confirmation | `best_panels/cycle_0002_multimethod_diagnostic.png` and `cycle_0016_multimethod_excess_panel.png` |
| Body-region audit + safe pool | `best_panels/cycle_0001_audited_safe_greenest_codex.png` and `final/audited_safe_frames.csv` |
| Pose-diversity browsing | `best_panels/cycle_0023_pose_diversity_gallery.png` |
| Per-frame separability scatter | `best_panels/cycle_0031_per_frame_scatter.png` |
| Pixel atypicality heatmap | `best_panels/cycle_0018_pixel_atypicality_heatmap.png` |
| Temporal trends per session | `best_panels/cycle_0017_dataset_trends.png` |
| Recipe A/B (codex vs claude) | `best_panels/cycle_0035_recipe_AB_diff.png` |
| Failure-mode discussion | `best_panels/cycle_0001_worst_body_red_v3.png` and `cycle_0025_hard_cases_panel.png` |
| Color-blind variants | `best_panels/cycle_0021_colorblind_variants.png` |
| Quantitative summary card | `best_panels/cycle_0027_quantitative_table.png` |

## Recommended recipe (claude_visual final)

```python
# Phase G K2 — verifier-tier
green_floor = 3.025      # full-9735 p95, per-session ref
red_floor   = 27.130     # full-9735 p99.9
scale_floor = "p5(scale_map)"  # 1.4826 * MAD with floor at p5
excess_margin = 0.5      # robust-z units
excess_red_threshold = 0.3
gamma = 0.65
clip = [0, 4]
sigma = 1.0   # video; use 0.0 for paper detail; 1.5 for ultra-smooth
reference = "per-session"  # D2 against D2 ref; V10 against V10
K = 2
selection_pool = "final/audited_safe_frames.csv"  # 9733 body-safe frames
```

## Standing rules (verified PASS for every promoted artifact)

- ✓ Same thresholds for real and bad conditions within paired panels
- ✓ No per-frame min/max normalization on anomaly maps
- ✓ No body/person manual suppression (body proxy used as audit overlay only)
- ✓ AE / AE-DGW panels labeled DIAGNOSTIC, not verifier-tier
- ✓ Captions include calibration sentence on every panel
- ✓ binder_split.json sha unchanged (`422bb48e8bd4ca8180cbbb174124c0c7fab608e22796c5a9b8ca2d4a72cae0f6`)
- ✓ No new ML inference, no source-shard modification

## Cycle catalog

36 cycles (35 claude_visual + 3 codex_visual + earlier v3 baseline).
Full inventory in `best_panels_index.md`. 62 promoted artifacts in
`best_panels/`. 39 leaderboard rows in `leaderboards/leaderboard.csv`.

## Key quantitative findings

| metric                                   | value      |
|------------------------------------------|------------|
| Frame-level AUROC corr-vs-shuffled       | **0.9999** |
| Frame-level AUROC corr-vs-fake (100k)    | **0.9998** |
| Mean per-frame z_two real correct        | 1.06       |
| Mean per-frame z_two real shuffled       | 4.19       |
| Mean per-frame z_two fake 100k           | 3.59       |
| Mean fake_excess at step 5k              | 67.4%      |
| Mean fake_excess at step 100k            | 52.4%      |
| Body-safe frames (body_red < 0.5%)       | 9733/9735  |
| Real-correct mean red% under recipe      | 0.018%     |
| Shuf-excess mean red% (full corpus)      | 53.2%      |
| Fake-excess mean red% (full corpus)      | 51.4%      |

## Manuscript-tier deliverables

1. `cycle_0030_paper_figure_synthesis.{png,pdf}` — main result figure (1A-D).
2. `cycle_0007_fake_step_progression_panel.png` — attacker non-convergence (5k → 100k stays robustly red).
3. `cycle_0014_zscore_histograms.png` — z_two density across 9735 frames per condition with full-9735 percentile thresholds.
4. `cycle_0019_manuscript_figure_*_300dpi.png` (also `.pdf`, `.svg`) — print-quality 2-row figure.
5. `cycle_0024_roc_calibration.png` — frame-level AUROC quantification.
6. `cycle_0026_hero_composite.png` — 6-frame hero composite (also `_300dpi`).
7. `cycle_0033_combined_hero.png` — 14-frame best-of-best (codex+claude).

## Coordination notes

- **claude_visual** ran cycles 0001–0036, ~1h 40m wallclock. All ethical-display rules pass.
- **codex_visual** ran cycles 0001–0003 (~7 minutes, then idle), producing the recipe-defining
  threshold/margin sweep (cycle_0002), the four-frame video storyboard (cycle_0003),
  the morning summary draft, and the ethical display manifest. Cycle_0003 storyboard
  frames all pass claude's body audit.
- Tasks proposed for codex (`proposals/for_codex_next_cycles.md`) await pickup.
- Direct nudges in `notes/codex_nudge_*.md`.

## Reproducibility

All scripts under `scripts/cycle_NNNN_*.py`. Each script is idempotent and writes
only under its `renders/cycle_NNNN_*` directory. Source memmaps and reference
statistics are read-only. To re-run any cycle:

```bash
cd /path/to/poliebotics_phase_b
python3 experiments/visualization_design/agent_collab_overnight/scripts/cycle_NNNN_<slug>.py
```
