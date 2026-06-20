# Artifact manifest — what's in this repo vs. external (R2 / Zenodo)

This release is a **code + paper + results-summary** repository. The bulk artifacts (model weights, the raw capture
corpus, derived data packs, and the full eval trees) are **external**, hosted on **Cloudflare R2** (the system of
record after the training host was retired), with a **Zenodo** DOI to be minted for a curated citable core. This
file maps every artifact the paper/README references to its location, access method, and the claim it backs, so a
reviewer always knows where something is and whether it is required to reproduce a given result.

## In this repository (self-contained)
| Artifact | Path | Backs |
|---|---|---|
| Whitepaper (PDF + LaTeX + figures) | `paper/` | all claims (narrative) |
| Recording protocol + third-party verifiers (v9 + v10, incl. `--logs-only` mode) | `code/recording/` | code→hash→chain verification, both sessions |
| Verifier stack (Phase G/F/H, binders, red-team) | `code/verifier/` | method reproduction |
| Patent filings (Reality Kernel) | companion repo: **poliebotics/PolieBotics** (`reality_kernel/`) | patent record |
| Held-out headline eval summaries | `results/eval/*.json` | AUROC=1.000 (within-session, n=198/200), shuffled 0.5006, synthetic-positive, F-A v1, cross-session |
| Off-body segmentation ablation — **summary** artifacts (`ablation_table_seg.csv`, `phase2_gate.json`, `phase4_summary.json`, `segmentation_manifest.csv`); the per-frame masks/scripts/shards/spatial-viz are external (R2 `lambda/experiments/paper_analyses/`, on request) | `results/redteam_segmentation_evals/proper_segmentation/` | §7 off-body localization (12 cells) |
| EXP-6 per-frame ranking (raw `rank_distribution.csv`, n=120 × 51 candidates, + summary) | `results/eval/exp6_correct_e_rank/` | §7 relative-comparison test: top-1 = 100%, mean rank 1.00 |
| EXP-7 / excess-red / causal ablations (incl. `excess_red/fake_step_progression.csv`, the §7 67.4→52.4 four-checkpoint table) | `results/redteam_segmentation_evals/{causal_ablations,xof_sensitivity,excess_red}/` | §7 perturbation sensitivity, attacker non-convergence |
| σ=4 low-frequency body recovery | `results/redteam_segmentation_evals/low_freq_typicality/` | §9 body-region signal survives at low frequency (~0.998) |
| XOF Type-1..6 bit-flip table (+ raw npz + regen script) | `results/redteam_segmentation_evals/xof_bitflip/` | §7/§8/§11 bit-flip AUROCs |
| Phase H E-usage ablation (diagnostic) | `results/phase_h/` (`e_usage_report.md`, `verdict.json`) | §8/§11 Phase H coarse/fine behavior |
| Cross-verifier report (incl. real-vs-zero-E 0.7221) | `results/eval/stage_0_cross_verifier__report.json` | §6/§8 cross-verifier zero-E |
| Frame-level per-frame metric table (9,735 rows × 33 cols) | `results/csv/visual_metrics_wide.csv` | §6 frame-level AUROC. Note: recomputing from this compact table gives 0.9998 (vs. shuffled) and 0.9998 (vs. synthetic); the run-time aggregate quoted in the paper/figures reads 0.9999/0.9998 — a one-unit difference in the fourth decimal from aggregation order, documented in §10 of the paper |
| Pinned lockfile (tested **training/eval** environment; the optional online-verification extras in `requirements.txt` are not pinned) | `requirements-lock.txt` | tested training versions |

## External — Cloudflare R2 (bucket `truthbeam`)

**A public subset is directly downloadable — no request, no login** — through the read-only gateway
`data.truthbeam.com`. Step-by-step in **[REPRODUCE.md](REPRODUCE.md)**:

| Public artifact | Direct URL | Size |
|---|---|---|
| Verifier weights (`model_final.pt`, 39.8 M params) | `https://data.truthbeam.com/models/verifier/model_final.pt` | 456 MB |
| F-A v1 forger checkpoints (5k/25k/70k/100k) | `https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_*.pt` | ~165 MB ea |
| Eval scores (2-minute, CPU-only reproduce input) | `https://data.truthbeam.com/models/repro/stage_0_eval/` | 4.3 MB |
| Ground-truth corpus (sessions D2/V10) | `https://data.truthbeam.com/sessions/` | ~378 GiB |
| 2023 demonstration video | `https://data.truthbeam.com/pinata/PolieBotics.mp4` | — |
| Truth Beam — Introduction | `https://data.truthbeam.com/pinata/TruthBeam_Introduction.mp4` | 64 s |

The **bulk eval trees** listed below (full `experiments/`, hundreds of GB) remain request/Zenodo-gated.
Nothing there is required to verify the *code→hash* promise or to recompute the headline AUROC — both
run from this repo plus the public subset above.

| Artifact | R2 location (under bucket `truthbeam`) | Approx size | Backs |
|---|---|---|---|
| Phase G verifier weights + training logs/configs (`main`/`shuffled`/`synthetic_positive`, `model_final.pt`) | `models/` and `lambda/experiments/phase_g_diffusion_diagnostic/` | 478 MB / 455 MiB each | headline verifier + controls; §5 training wall-time/loss figures |
| F-A v1 forger checkpoints (5k/25k/70k/100k) + 14 binders | `models/`, `lambda/experiments/` | ~6–12 GB | red-team |
| Stage-0 cross-session verifiers (step 100000) | `lambda/experiments/{stage_0_cross_verifier,cross_session_ablation}/` | — | §6 cross-session AUROC |
| Raw capture corpus (D2 5,992 + V10 3,743 BayerRG8 frames) + emission tiles — the **raw analysis subset** (the full public `sessions/` release is ~378 GiB = 406 GB) | `raw/`, `sessions/` | ~262 GB | dataset |
| Derived pack: 208 NPZ map shards (full-res robust-z / excess maps) | `lambda/experiments/` / curated Zenodo subset | ~14 GB | §10 derived products (the wide CSV itself now ships in-repo, above) |
| Full eval trees (all `experiments/`) | `lambda/experiments/` | ~659 GB | full reproduction |

## External — Zenodo / Hugging Face (pending)
- **Zenodo DOI** (paper + curated citable core: manifests, datasheet, summaries, selected checkpoints) — *to be minted*.
- **Hugging Face** model mirror (optional) — *pending*.

> The session bundles a third-party verifier needs (`manifest.json`, `verification_bundle.json`, `chain_log.csv`,
> `capture_log.csv`, `anchor_txs.csv`, `verify_report.json`, raw frames) are released with the session data (R2 /
> Zenodo), not committed to this code repo.
