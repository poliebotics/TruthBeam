# Phase G E2/E3 Segmentation Causal Ablation Audit

Source note: `/path/to/poliebotics_phase_b` is not mounted in this audit environment. The output mirror is `/path/to/poliebotics_phase_b/...`. The requested `proper_segmentation/scripts/` directory is absent in the mirror; the result write-up identifies the executed sources as `/tmp/phase1_segment_120.py`, `/tmp/phase3_seg_ablations.py`, `/tmp/phase4_analyze.py`, `/tmp/phase5_spatial_viz.py`, and `/tmp/build_phase3_shards.py` (`SEGMENTATION_RESULTS.md:333-338`). Code findings below cite those `/tmp` scripts.

## Checklist Findings

| ID | Status | Evidence |
|---|---|---|
| A1 | PASS | Mask R-CNN weights path is fixed at `/path/to/poliebotics_phase_b/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` (`/tmp/phase1_segment_120.py:37`); architecture uses `weights=None`, `weights_backbone=None`, `num_classes=91`, then explicitly `torch.load` + `load_state_dict` (`/tmp/phase1_segment_120.py:121-127`). |
| A2 | PASS | Person-only filtering uses `person_idx = np.where(labels == 1)[0]` (`/tmp/phase1_segment_120.py:60-64`). |
| A3 | PASS | Highest-confidence person is selected via `np.argmax(scores[person_idx])` (`/tmp/phase1_segment_120.py:66-68`). |
| A4 | PASS | Mask is binarized with threshold `> 0.5` (`/tmp/phase1_segment_120.py:68`). |
| B1 | PASS | Connected components are computed with `cv2.connectedComponentsWithStats`; largest component is selected by `argmax(stats[1:, CC_STAT_AREA])` (`/tmp/phase1_segment_120.py:76-85`). |
| B2 | PASS | Cleanup is close then open using `cv2.MORPH_RECT` kernel `(3, 3)` (`/tmp/phase1_segment_120.py:40-42`, `/tmp/phase1_segment_120.py:86-90`). |
| B3 | PASS | Area bounds are `2.0` and `70.0`; classification returns too-small/too-large pathologies outside those bounds (`/tmp/phase1_segment_120.py:39-42`, `/tmp/phase1_segment_120.py:97-104`). |
| B4 | PASS | No predicted person sets `classification = pathological_no_mask` and saves an all-false mask (`/tmp/phase1_segment_120.py:141-147`). Empty post-cleanup masks also classify as `pathological_no_mask` (`/tmp/phase1_segment_120.py:97-100`). |
| B5 | PARTIAL | Pathological rows are dropped via `classification.startswith("pathological")`, and CSV records `dropped`, `orig_n_components`, and `post_cleanup_n_components` (`/tmp/phase1_segment_120.py:153-167`). Fragmented originals are included as `fragmented_resolved` when `orig_components > 1` (`/tmp/phase1_segment_120.py:105-106`). Concern: one included `clean` row has `post_cleanup_n_components=2` after cleanup (`segmentation_manifest.csv:2`), so post-cleanup fragmentation is recorded but not reclassified. |
| C1 | PASS | Phase 3 applies the same `mask_dev` to both C and E for E2/E3 (`/tmp/phase3_seg_ablations.py:136-149`). |
| C2 | PASS | Replacement fill is per-channel mean of inside-mask pixels: `inside = arr[:, mask]`, `fill = inside.mean(dim=1)` (`/tmp/phase3_seg_ablations.py:51-60`, `/tmp/phase3_seg_ablations.py:64-74`). This matches the reference inside-box per-channel mean convention (`causal_ablations.py:279-287`, `causal_ablations.py:290-318`). |
| C3 | PASS | E2 keeps inside and replaces outside: `torch.where(mask_3d, arr, fill_3d)` (`/tmp/phase3_seg_ablations.py:51-61`). |
| C4 | PASS | E3 keeps outside and replaces inside: `torch.where(~mask_3d, arr, fill_3d)` (`/tmp/phase3_seg_ablations.py:64-75`). Leakage check produced no flags (`phase4_summary.json:1-4`). |
| D1 | PASS | Phase 3 imports `phase_g_score_scalar` from `paper_analyses.causal_ablations` rather than reimplementing it (`/tmp/phase3_seg_ablations.py:37-46`); reference function is at `causal_ablations.py:402-435`. |
| D2 | PASS | Noise seed formula is identical to reference: `NOISE_SEED_BASE + (row * 7919 + (1 if sess == "V10" else 0))`, followed by `torch.manual_seed(seed)` before `torch.randn` (`/tmp/phase3_seg_ablations.py:120-125`; reference `causal_ablations.py:496-501`). |
| D3 | PASS | Phase 3 imports `T_STEPS`, `K_NOISE`, and `NOISE_SEED_BASE` from the reference (`/tmp/phase3_seg_ablations.py:39-46`); reference values are `T_STEPS=(50,150,300,500,750)` and `K_NOISE=4` (`causal_ablations.py:82-85`). |
| D4 | PASS | CUDA uses `torch.bfloat16`; diffusion constants are built with `build_diffusion_constants(T_DIFFUSION, device, torch.float32)` (`/tmp/phase3_seg_ablations.py:209-225`; reference `causal_ablations.py:938-955`). |
| D5 | PASS | Default Phase G checkpoint is `/path/to/poliebotics_phase_b/experiments/phase_g_diffusion_diagnostic/main/model_final.pt`, and logs confirm loading that path (`/tmp/phase3_seg_ablations.py:181-183`, `launch_g0.log:2`). |
| D6 | PASS | F-A v1 checkpoint dir is `/path/to/poliebotics_phase_b/experiments/phase_f/f_a_full_v1/checkpoints`; checkpoint filenames are `step_{step:08d}.pt`, with `FA_V1_CKPT_STEPS` imported from reference and `render_C_fake` using `source_row = keys[(target_idx + len(keys) // 4) % len(keys)]` (`/tmp/phase3_seg_ablations.py:106-118`, `/tmp/phase3_seg_ablations.py:184-185`, `/tmp/phase3_seg_ablations.py:218-223`; reference `causal_ablations.py:469-493`). |
| E1 | PASS | Paired bootstrap uses one `idx` per iteration for both seg and orig AUROCs (`/tmp/phase4_analyze.py:83-100`). |
| E2 | PASS | Frame-level paired comparison is built from the usable segmentation subset, same frame rows matched to original manifests (`/tmp/phase4_analyze.py:165-215`). The implementation resamples frames with replacement, not independent scalar pools. |
| E3 | PASS | AUROC uses Mann-Whitney with `score=-MSE`, stable sorting, and average ranks for ties (`/tmp/phase4_analyze.py:35-62`), matching the reference (`causal_ablations.py:597-623`). |
| E4 | PASS | Original rows are indexed by `(ablation, condition, session)` and compared through `COMPARE = {"E2_seg": "E2", "E3_seg": "E3"}` (`/tmp/phase4_analyze.py:29-33`, `/tmp/phase4_analyze.py:158-163`, `/tmp/phase4_analyze.py:187-215`). |
| E5 | PARTIAL | Thresholds `0.85/0.95` are hard-coded in `regime()` (`/tmp/phase4_analyze.py:103-107`) and the decision tree is hard-coded (`/tmp/phase4_analyze.py:327-368`). However Phase 4 ignores delta-score degradation thresholds from the original causal analysis regime function (`causal_ablations.py:653-661`). For this conclusion, AUROC alone is sufficient because all E2_seg rows are below 0.95 and all F-A E3_seg rows are 1.0 (`ablation_table_seg.csv:2-15`). |
| F1 | PASS | Scripts load Phase G for inference and do not save or train it (`/tmp/phase3_seg_ablations.py:215-225`; `/tmp/phase5_spatial_viz.py:157-164`). |
| F2 | PASS | Phase 3 uses F-A v1 fakes only (`/tmp/phase3_seg_ablations.py:18-19`, `/tmp/phase3_seg_ablations.py:110-118`, `/tmp/phase3_seg_ablations.py:218-223`). |
| F3 | PASS | Exact search of the five `/tmp` scripts found zero `fa_v2` or `f_a_v2` mentions. |
| F4 | PASS | Exact search of the five `/tmp` scripts found zero `binder_split.json` references. |
| F5 | PASS | Exact search of the five `/tmp` scripts found zero mentions of `D1`, `D_large`, `D2_a`, `B1`, `B2`, `A1`, `A2`, `C1`, `C2`, or `e2_fp16`. |

## Spot Checks

| Check | Result |
|---|---|
| `D2_f001328` mask area | Manifest `mask_area_pct=6.253306070963542` (`per_frame/D2_f001328/ablations_manifest.json:7-9`); loaded `masks/D2_f001328.npy`, computed `100 * mean = 6.253306070963542`; diff `0.0`. |
| `D2_f001328` noise seed | Manifest `noise_seed=10516474` (`per_frame/D2_f001328/ablations_manifest.json:7`); formula `42 + 1328 * 7919 = 10516474`. |
| `D2_f001328` scalar count | Manifest contains exactly 14 scalar entries, lines 10-23: 2 ablations x 7 conditions. |
| `V10_f001279` mask area | Manifest `mask_area_pct=8.102798461914062` (`per_frame/V10_f001279/ablations_manifest.json:7-9`); loaded `masks/V10_f001279.npy`, computed `100 * mean = 8.102798461914062`; diff `0.0`. |
| `V10_f001279` noise seed | Manifest `noise_seed=10128444` (`per_frame/V10_f001279/ablations_manifest.json:7`); formula `42 + 1279 * 7919 + 1 = 10128444`. |
| `V10_f001279` scalar count | Manifest contains exactly 14 scalar entries, lines 10-23: 2 ablations x 7 conditions. |
| Mask file audit | Loaded all 120 `.npy` masks: all shape `(768, 1024)`; all dtype `bool`; 110 have both `False/True`; 10 all-false masks correspond to no-mask pathologies. CSV area rounded values matched loaded masks within `5e-4` for all rows. |
| Comparison row recompute | Picked `E2 -> E2_seg`, `fake_5k`, `D2` (`comparison_table.csv:2`). From paired per-frame manifests: `n=50`, `AUROC_orig=0.9924`, `AUROC_seg=0.0088`, `delta=-0.9835999999999999`, exactly matching table. Bootstrap with `RandomState(0)`, same paired indices for orig/seg: mean `-0.9842124`, CI `[-0.9976, -0.9635900000000001]`, exactly matching table. |

## Bugs / Concerns

### Critical

None found in the mask application, noise pairing, AUROC orientation, or paired-bootstrap implementation that would flip the main E2_seg/E3_seg conclusion.

### Major

1. **Script provenance/location does not match the requested audit target.** The requested `/path/to/poliebotics_phase_b/proper_segmentation/scripts/` directory is absent in the mirror; the result markdown itself lists `/tmp/*.py` as sources (`SEGMENTATION_RESULTS.md:333-338`). The `/tmp` code is auditable now, but manuscript citation should preserve the exact executed scripts under the experiment directory or version control.

2. **The Phase 1/2 gate says `STOP`, but later phases proceeded.** `phase2_gate.json` reports `D2=50`, `V10=56`, and `"gate": "STOP"` (`phase2_gate.json:17-21`, `phase2_gate.json:41`). Phase 4 explicitly analyzes those 106 usable frames (`phase4_summary.json:1-3`; `/tmp/phase4_analyze.py:174-179`). This is not a math bug, but it is a protocol violation unless the operator explicitly re-authorized proceeding with reduced n.

### Minor

1. **One post-cleanup fragmentation edge case is included as `clean`.** `D2,1328` has `orig_n_components=1`, `post_cleanup_n_components=2`, `classification=clean`, `dropped=False` (`segmentation_manifest.csv:2`). This follows the current code because `fragmented_resolved` is keyed to original components, not post-cleanup components (`/tmp/phase1_segment_120.py:97-107`). It did not break the spot-checked mask math, but it should be documented or renamed.

2. **Phase 4 regime classification omits delta-degradation thresholds.** The original causal analysis regime includes AUROC and delta-degradation criteria (`causal_ablations.py:653-661`); Phase 4 uses AUROC only (`/tmp/phase4_analyze.py:103-107`). The headline body/off-body finding is still numerically clear, but the table regime labels are not a full reproduction of the original regime function.

## Overall Verdict

**PARTIAL.**

The implementation evidence supports the core technical conclusion: E2_seg body-only masking is not carrying the Phase G discrimination signal, while E3_seg off-body masking carries the F-A v1 fake discrimination. The key implementation risks named in the audit prompt pass: mask broadcasting is correct, per-channel mean axis is correct, noise is seeded before `torch.randn` with the reference formula, AUROC score orientation and tie handling match the reference, and paired bootstrap uses the same frame indices for orig/seg.

Before manuscript citation, fix the provenance/protocol issues:

1. Copy or archive the exact executed `/tmp` scripts under `proper_segmentation/scripts/` or another immutable experiment source directory.
2. Resolve the `phase2_gate.json` `STOP` condition in writing: either rerun with a pre-authorized reduced-n gate, change the gate policy before analysis and rerun Phase 4, or label the current result exploratory.
3. Decide whether post-cleanup `post_cleanup_n_components > 1` should be a separate classification or explicitly accepted as included.
4. If regime labels are manuscript-facing, make Phase 4’s regime function match the original AUROC-plus-delta rule or state that the segmentation comparison uses AUROC-only regimes.
