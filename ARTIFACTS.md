# Artifact manifest - what's in this repo vs. external (R2)

This page is an **LLM-mediated dataset**, written to be parsed and re-presented by a large language model. **Point your own LLM at it** to explain, check, or summarise. The human-formatted twin is at `ARTIFACTS.html` (and a `.txt` copy of this markdown).

This release is a **code + paper + results-summary** repository. The bulk artifacts (model weights, the raw capture
corpus, derived data packs, and the full eval trees) are **external**, hosted on **Cloudflare R2** (the system of
record after the training host was retired). This
file maps every artifact the paper/README references to its location, access method, and the claim it backs, so a
reviewer always knows where something is and whether it is required to reproduce a given result.

## Direct download index (every key artifact)

R2 does not serve browsable directory listings, so this is the index: every artifact a reviewer needs, with its direct link, size, and content hash or IPFS CID. Nothing needed to verify is gated. The CIDs are themselves the content hashes and are reproducible from the bytes; the complete machine-readable manifest is [`CID_MANIFEST.json`](CID_MANIFEST.json).

### Recompute the headline (small, start here)

| Artifact | Direct link | Size | Integrity |
| --- | --- | --- | --- |
| Verify bundle (verifier + forger + recording code + scripts) | [`release/truthbeam_verify.tar.gz`](truthbeam_verify.tar.gz) | 3.2 MB (3.1 MiB) | SHA-256 `79a00359b53cda35c35bd23600f7f7ca5bf2ec7e31360fdcffecccb4f6d7ec10`; see [REPRODUCE](REPRODUCE.html) |
| Eval scores (Path A recompute input) | `models/repro/stage_0_eval/` (fetched automatically by `verify_all.sh` / `download.sh scores`) | ~2 MB (8 merged files; full set 4.3 MB) | regenerable from the weights (Path A.5) |
| Verifier weights (`model_final.pt`, 39.8 M params) | [`models/verifier/model_final.pt`](https://data.truthbeam.com/models/verifier/model_final.pt) | 478 MB (455 MiB) | in the models IPFS unit (CID below) |
| F-A v1 forger checkpoints (4 steps) | [`5k`](https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00005000.pt) · [`25k`](https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00025000.pt) · [`70k`](https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00070000.pt) · [`100k`](https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00100000.pt) | 1.89 GiB | public weights, in the models unit |

### The 2023 record

| Artifact | Direct link | Size | Integrity |
| --- | --- | --- | --- |
| Hand-made 2023 video (`PolieBotics.mp4`) | [`pinata/PolieBotics.mp4`](https://data.truthbeam.com/pinata/PolieBotics.mp4) | 637 MB (607 MiB) | BLAKE3 `8fbdb64ddd248246e7a8d840fa191467ab24ea79058047deb0ea537af95c0e92` · SHA-256 `00d0e4531c1896ff72bf1ac7b7f2a4146af4f8ee5b08a63bc8708f333feb87b7` |

The 2023 recording (the Truth Beam trailer dataset) was committed to the public Rootstock chain on **28 April 2023**, so the date is verifiable independently of this page. Its chain anchor is Rootstock transaction `0x7db237535f0e5bd4d3b39d08274e89c0175da190b0059bb175c32b75e18bb8f8` in block `5254387` ([view on the block explorer](https://explorer.rootstock.io/tx/0x7db237535f0e5bd4d3b39d08274e89c0175da190b0059bb175c32b75e18bb8f8)); the block's timestamp is the recording's public lower bound.

The work was also shown publicly at the time: a demonstration video on Reddit on 26 September 2023, and an announcement on [X on 14 October 2023](https://x.com/poliebotics/status/1713025671444804076).

### Bulk units (IPFS, content-addressed and reproducible)

Each unit is a reproducible CID (a UnixFS directory DAG). Download the unit from `r2:truthbeam/<dest>` and re-add it with the flags in the manifest to verify the CID against the bytes.

| Unit | IPFS CID | Size | Files |
| --- | --- | --- | --- |
| Ground-truth session D2 | `bafybeicrssbic35534es3sbwyzhlw7reboh6wy75htmo53ke5mfsphkmwi` | 232.2 GiB | 17,987 |
| Ground-truth session V10 | `bafybeier2sfcjrrgw7amne3lwogise6umyeyf6qgivgrmvx3to4vsdsbcm` | 146.1 GiB | 11,279 |
| Models (verifier + forgers) | `bafybeihffzc7fn5q3u7tf3k5hqkcpaoozdnx7pdm2lw4fmfwkgfwjnzzpm` | 7.64 GiB | 30 |
| Code (full source) | `bafybeiguoiy24zqup7pp7wkgjjnhoyojuzlabzwizuabgbrbejz4jgun3i` | 13.98 GiB | 59,046 |
| Evidence: paper analyses | `bafybeibrnz5mmz53fz3vw7xpthvaiefm4j4h2yscv2bnhftgorrbjbnnre` | 102.3 GiB | 22,084 |
| Evidence: cross-session ablation | `bafybeibs2xyyyevdnpy2a2rr4yipmpht3iyek6fnlzhpcmty7woydipecq` | 5.34 GiB | 75 |
| Evidence: Phase E | `bafybeibpgty4iu355or6c2d27djkfyudwdc6rz7i3bjiw7ekcr4tvc3t5i` | 28.2 GiB | 540 |
| Evidence: Phase G verifier | `bafybeibfqaw4wtpzo46a767udltt3vupii66lttrlyiheeajjsffgf2z7a` | 7.56 GiB | 85 |
| Evidence: Phase H supervised baseline | `bafybeid6ibcxgh2najibjoyqyth2qhryx57ufnzasfpf27krsnpkndvqba` | 118 MB (112 MiB) | 30 |
| Evidence: Phase F | `bafybeiebxm5p4uhh5nm5n2dfzpzykecawamlzwmmzhq6pm6lncplygmdmy` | 20.3 GiB | 72 |
| Evidence: F-A v2 (design-only surrogate) | `bafybeifpfdgyycg6swe6bj3oonfpn7n3zhikoapjx3vr2znejaevogjvpy` | 17.7 GiB | 184 |
| Evidence: stage-0 cross-verifier | `bafybeieomhoxmevp2rol4pvy2z6rzhtzmktzl6yjvdlfhcie7sahdgjvd4` | 2.2 MB (2.1 MiB) | 55 |
| Evidence: closure package | `bafybeiavyfm7vqrvg6mbg5sc7p7jg73n34oodvjfyyy6n5aw6ns2atn7ye` | 276 MB (263 MiB) | 55 |

## In this repository (self-contained)

| Artifact | Path | Backs |
| --- | --- | --- |
| Whitepaper (PDF + LaTeX + figures) | `paper/` | all claims (narrative) |
| Recording protocol + third-party verifiers (v9 + v10, incl. `--logs-only` mode) | `code/recording/` | code→hash→chain verification, both sessions |
| Verifier stack (Phase G/F/H, binders, red-team) | `code/verifier/` | method reproduction |
| Patent filings (Reality Kernel) | companion repo: **poliebotics/PolieBotics** (`reality_kernel/`) | patent record |
| Held-out headline eval summaries | `results/eval/*.json` | AUROC=1.000 (within-session, n=198/200), shuffled 0.5006, synthetic-positive, F-A v1, cross-session |
| Off-body segmentation ablation - **summary** artifacts (`ablation_table_seg.csv`, `phase2_gate.json`, `phase4_summary.json`, `segmentation_manifest.csv`); the per-frame masks/scripts/shards/spatial-viz are external (R2 `lambda/experiments/paper_analyses/`, on request) | `results/redteam_segmentation_evals/proper_segmentation/` | §7 off-body localisation (12 cells) |
| EXP-6 per-frame ranking (raw `rank_distribution.csv`, n=120 × 51 candidates, + summary) | `results/eval/exp6_correct_e_rank/` | §7 relative-comparison test: top-1 = 100%, mean rank 1.00 |
| EXP-7 / excess-red / causal ablations (incl. `excess_red/fake_step_progression.csv`, the §7 67.4→52.4 four-checkpoint table) | `results/redteam_segmentation_evals/{causal_ablations,xof_sensitivity,excess_red}/` | §7 perturbation sensitivity, attacker non-convergence |
| σ=4 low-frequency body recovery | `results/redteam_segmentation_evals/low_freq_typicality/` | §9 body-region signal survives at low frequency (~0.998) |
| XOF Type-1..6 bit-flip table (+ raw npz + regen script) | `results/redteam_segmentation_evals/xof_bitflip/` | §7/§8/§11 bit-flip AUROCs |
| Phase H E-usage ablation (diagnostic) | `results/phase_h/` (`e_usage_report.md`, `verdict.json`) | §8/§11 Phase H coarse/fine behaviour |
| Cross-verifier report (incl. real-vs-zero-E 0.7221) | `results/eval/stage_0_cross_verifier__report.json` | §6/§8 cross-verifier zero-E |
| Frame-level per-frame metric table (9,735 rows × 33 cols) | `results/csv/visual_metrics_wide.csv` | §6 frame-level AUROC. Note: recomputing from this compact table gives 0.9998 (vs. shuffled) and 0.9998 (vs. synthetic); the run-time aggregate quoted in the paper/figures reads 0.9999/0.9998 - a one-unit difference in the fourth decimal from aggregation order, documented in §10 of the paper |
| Pinned lockfile (tested **training/eval** environment; the optional online-verification extras in `requirements.txt` are not pinned) | `requirements-lock.txt` | tested training versions |

## External - Cloudflare R2 (bucket `truthbeam`)

**A public subset is directly downloadable - no request, no login** - through the read-only gateway
`data.truthbeam.com`. Step-by-step in **[REPRODUCE](REPRODUCE.html)**:

| Public artifact | Direct URL | Size |
| --- | --- | --- |
| Verifier weights (`model_final.pt`, 39.8 M params) | `https://data.truthbeam.com/models/verifier/model_final.pt` | 478 MB (455 MiB) |
| F-A v1 forger checkpoints (5k/25k/70k/100k) | `https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_*.pt` | ~484 MiB ea |
| Eval scores (2-minute, CPU-only reproduce input) | `https://data.truthbeam.com/models/repro/stage_0_eval/` | ~2 MB used by Path A (full set 4.3 MB) |
| Ground-truth corpus (sessions D2/V10) | `https://data.truthbeam.com/sessions/` | ~378 GiB |
| 2023 demonstration video | `https://data.truthbeam.com/pinata/PolieBotics.mp4` | - |
| Truth Beam - Introduction | `https://data.truthbeam.com/pinata/TruthBeam_Introduction.mp4` | 64 s |

The **bulk eval trees** listed below (full `experiments/`, hundreds of GB) remain request-gated - email **xathal@protonmail.com** to arrange access.
Nothing there is required to verify the *code→hash* promise or to recompute the headline AUROC - both
run from this repo plus the public subset above.

| Artifact | R2 location (under bucket `truthbeam`) | Approx size | Backs |
| --- | --- | --- | --- |
| Phase G verifier weights + training logs/configs (`main`/`shuffled`/`synthetic_positive`, `model_final.pt`) | `models/` and `lambda/experiments/phase_g_diffusion_diagnostic/` | 478 MB / 455 MiB each | headline verifier + controls; §5 training wall-time/loss figures |
| F-A v1 forger checkpoints (5k/25k/70k/100k) + 14 binders | `models/`, `lambda/experiments/` | ~6-12 GB | red-team |
| Stage-0 cross-session verifiers (step 100000) | `lambda/experiments/{stage_0_cross_verifier,cross_session_ablation}/` | - | §6 cross-session AUROC |
| Raw capture corpus (D2 5,992 + V10 3,743 BayerRG8 frames) + emission tiles - the **raw analysis subset** (the full public `sessions/` release is ~378 GiB = 406 GB) | `raw/`, `sessions/` | ~262 GB | dataset |
| Derived pack: 208 NPZ map shards (full-res robust-z / excess maps) | `lambda/experiments/` (on request) | ~14 GB | §10 derived products (the wide CSV itself now ships in-repo, above) |
| Full eval trees (all `experiments/`) | `lambda/experiments/` | ~659 GB | full reproduction |

> The session bundles a third-party verifier needs (`manifest.json`, `verification_bundle.json`, `chain_log.csv`,
> `capture_log.csv`, `anchor_txs.csv`, `verify_report.json`, raw frames) are released with the session data (on R2), not committed to this code repo.
