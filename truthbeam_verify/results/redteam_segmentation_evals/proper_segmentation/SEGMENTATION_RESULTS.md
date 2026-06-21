# SEGMENTATION_RESULTS.md — proper-segmentation E2/E3 ablations

> **Artifacts shipped in this repository (read this first).** This directory
> contains the **summary** artifacts that back the paper's §7 off-body
> numbers: `ablation_table_seg.csv` (the 12-cell E2_seg/E3_seg AUROCs),
> `phase2_gate.json`, `phase4_summary.json`, `segmentation_manifest.csv`, and
> this write-up. The bulkier provenance products this document refers to in
> places — per-frame Mask R-CNN `masks/`, the `per_frame/` ablation
> manifests, `spatial_viz/` overlays, the worker `shards/`, the generating
> `scripts/`, and `comparison_table.csv`/`comparison_summary.md` — are **not**
> in this code repository; they live in the external R2 evaluation tree
> (`lambda/experiments/paper_analyses/`) and are available on request. The
> shipped CSV/JSON summaries are sufficient to read off every number the
> paper cites; the larger products are for re-deriving the masks and figures.

**Generated**: 2026-05-05.
**Authorization**: operator directive 2026-05-05 (full E2/E3 re-run with
proper Mask R-CNN segmentation, post v2-visual-check pass).
**Standing rules**: Phase G inference-only; no held-out asset use beyond F-A
v1; binder_split.json LOCKED v3; no F-A v2 trainer touch; no D-family work.

---

## Headline

**The variance-box E2/E3 conclusion is OVERTURNED. Body region is NOT where
Phase G's discrimination on F-A v1 fakes lives — the off-body region is.**

All 12 (session, condition) cells classify as "MAJOR FINDING — body alone
insufficient" under the pre-registered interpretation logic.

| metric | variance-box (paired n=106) | proper-segmentation (n=106) | Δ |
|---|---|---|---|
| E2 AUROC range | 0.977–0.998 | 0.009–0.538 | **−0.46 to −0.98** |
| E3 AUROC range | 0.815–0.980 | 1.000 across all cells | **+0.02 to +0.19** |
| E2 regime (all cells) | weak_dependence | **strong_dependence** | inverted |
| E3 regime | mixed (moderate / strong / weak) | **weak_dependence** (all cells) | improved |

ΔAUROC 95% CIs exclude zero in every cell. The reversal is statistically
robust on hierarchical paired bootstrap.

**Interpretation**: variance-box "body" mask was conflating body with
high-variance off-body content (projection edges on background, scene
structure); variance-box "off-body" was the low-variance subset of true
off-body content. Proper segmentation cleanly separates the two and reveals
the signal lives off-body.

This is consistent with the original visual-grids interpretation (body
anti-discriminative across ε-MSE / VGG / B1 scoring). Variance-box
ablations had inverted that reading; proper segmentation re-confirms it.

---

## 1. Mask quality summary

Phase 1 ran Mask R-CNN ResNet50 FPN (COCO_V1 weights, stashed at
`/path/to/poliebotics_phase_b/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`)
on all 120 frames in the causal-ablations subset. Pre-registered drop policy:

- pathological (no mask / mask <2% / mask >70%): DROPPED
- fragmented (post-largest-CC + morph close+open): INCLUDED as clean
- clean: INCLUDED

### Final n per session

| session | total | dropped (pathological) | usable | clean | fragmented_resolved |
|---|---|---|---|---|---|
| **D2** | 60 | 10 (16.7%) | **50** | 45 | 5 |
| **V10** | 60 | 4 (6.7%) | **56** | 54 | 2 |
| **combined** | 120 | 14 (11.7%) | **106** | 99 | 7 |

### Pre-registered gate (operator-corrected)

| rule | result |
|---|---|
| Combined pathological rate < 10% → PROCEED clean | — |
| Combined pathological rate 10–25% → PROCEED with caveat | **applies (11.7%)** |
| Combined pathological rate > 25% → STOP | — |
| Per-session minimum ≥ 40 usable | satisfied (D2=50, V10=56) |

### Block-level asymmetry (caveat)

| block | dropped | rate |
|---|---|---|
| D2 block 1 (1298–1698) | 2 / 20 | 10.0% |
| **D2 block 2 (2796–3196)** | **6 / 20** | **30.0%** |
| D2 block 3 (4294–4694) | 2 / 20 | 10.0% |
| V10 block 1 (1110–1360) | 4 / 30 | 13.3% |
| V10 block 2 (2345–2595) | 0 / 30 | 0.0% |

Mask R-CNN reliably segments standing/walking subjects (V10 block 2 = 0%
pathological) and most yoga poses, but underperforms on the specific pose
subset in D2 block 2 — likely supine/horizontal/uncommon yoga orientations
less represented in COCO training data. **Manuscript caveat**: scope of
proper-segmentation analysis excludes the D2 block 2 supine-pose subset.

### Dropped frame IDs (logged)

D2 (10): 1396, 1430, 2860, 2877, 2894, 2945, 2962, 2996, 4341, 4409
V10 (4): 1140, 1153, 1191, 1197

Manifest: `proper_segmentation/segmentation_manifest.csv` (one row per
frame: classification, mask area %, confidence, post-cleanup component
count, dropped flag).

---

## 2. E2_seg / E3_seg AUROC tables

Computed over 50 D2 + 56 V10 frames; deterministic noise seeds matched to
causal_ablations E1 baseline (paired comparison enabled).

### D2

| ablation | fake_5k | fake_25k | fake_70k | fake_100k | shuffled_E | cross_session_E |
|---|---|---|---|---|---|---|
| E2_seg | 0.009 **strong** | 0.016 **strong** | 0.024 **strong** | 0.024 **strong** | 0.536 **strong** | 0.536 **strong** |
| E3_seg | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak |

### V10

| ablation | fake_5k | fake_25k | fake_70k | fake_100k | shuffled_E | cross_session_E |
|---|---|---|---|---|---|---|
| E2_seg | 0.148 **strong** | 0.196 **strong** | 0.261 **strong** | 0.251 **strong** | 0.538 **strong** | 0.537 **strong** |
| E3_seg | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak |

Full table with bootstrap CIs and Δscore distributions:
`proper_segmentation/ablation_table_seg.csv` (24 rows).

### Sub-finding: E2_seg AUROC < 0.5 on fakes

For F-A v1 fake conditions (5k/25k/70k/100k), **E2_seg AUROC drops below
0.5** — meaning under body-only masking, F-A v1 fakes appear MORE real-like
to Phase G than real frames. F-A v1 is good at producing low-MSE body
content; the discriminative signal Phase G normally uses lives off-body.
Without that off-body context, Phase G's MSE-based scoring is fooled.

For E-perturbation conditions (shuffled_E, cross_session_E), E2_seg AUROC
is ~0.54 — basically chance. Body-only does not let Phase G distinguish
correct E from shuffled E either; the chain-coupling signal is also
off-body.

---

## 3. Side-by-side comparison: variance-box vs proper segmentation

Paired on the 106 usable frames. Same noise seeds across orig/seg. ΔAUROC =
AUROC_seg − AUROC_orig.

### D2

| condition | E2 orig | E2_seg | ΔAUROC | 95% CI | E3 orig | E3_seg | ΔAUROC | 95% CI |
|---|---|---|---|---|---|---|---|---|
| fake_5k          | 0.992 | 0.009 | **−0.984** | [−0.998, −0.964] | 0.909 | 1.000 | +0.091 | [+0.042, +0.138] |
| fake_25k         | 0.990 | 0.016 | **−0.974** | [−0.995, −0.944] | 0.830 | 1.000 | +0.170 | [+0.106, +0.231] |
| fake_70k         | 0.985 | 0.024 | **−0.961** | [−0.990, −0.921] | 0.851 | 1.000 | +0.149 | [+0.090, +0.205] |
| fake_100k        | 0.977 | 0.024 | **−0.953** | [−0.986, −0.907] | 0.815 | 1.000 | +0.185 | [+0.121, +0.242] |
| shuffled_E       | 0.998 | 0.536 | −0.462 | [−0.468, −0.437] | 0.911 | 1.000 | +0.089 | [+0.044, +0.136] |
| cross_session_E  | 0.998 | 0.536 | −0.463 | [−0.468, −0.438] | 0.911 | 1.000 | +0.089 | [+0.045, +0.133] |

### V10

| condition | E2 orig | E2_seg | ΔAUROC | 95% CI | E3 orig | E3_seg | ΔAUROC | 95% CI |
|---|---|---|---|---|---|---|---|---|
| fake_5k          | 0.982 | 0.148 | **−0.835** | [−0.920, −0.734] | 0.980 | 1.000 | +0.020 | [+0.004, +0.042] |
| fake_25k         | 0.982 | 0.196 | **−0.786** | [−0.878, −0.679] | 0.931 | 1.000 | +0.069 | [+0.031, +0.117] |
| fake_70k         | 0.982 | 0.261 | **−0.722** | [−0.823, −0.615] | 0.952 | 1.000 | +0.048 | [+0.018, +0.086] |
| fake_100k        | 0.982 | 0.251 | **−0.731** | [−0.830, −0.625] | 0.928 | 1.000 | +0.072 | [+0.031, +0.121] |
| shuffled_E       | 0.995 | 0.538 | −0.458 | [−0.459, −0.437] | 0.948 | 1.000 | +0.052 | [+0.021, +0.087] |
| cross_session_E  | 0.995 | 0.537 | −0.458 | [−0.460, −0.435] | 0.942 | 1.000 | +0.058 | [+0.027, +0.092] |

**Every ΔAUROC CI excludes zero.** Differences are not bootstrap noise.

Full table: `proper_segmentation/comparison_table.csv`.

---

## 4. Pre-registered interpretation applied

Decision tree (frozen before analysis, taken verbatim from operator's spec):

| condition matched | classification |
|---|---|
| (E2_orig > 0.95 AND E2_seg > 0.95) AND (E3_orig < 0.85 OR E3_seg < 0.85) | body-region primacy survives |
| E2_orig > 0.95 AND E2_seg > 0.95 AND E3_seg > 0.95 | redundant paths |
| **E2_seg < 0.95** | **MAJOR FINDING — body alone insufficient** |
| E2_seg < 0.95 AND E3_seg < 0.95 | spatially-integrated discrimination |

### Per-(session, condition) result

| session | condition | classification |
|---|---|---|
| D2  | fake_5k          | MAJOR FINDING — body alone insufficient |
| D2  | fake_25k         | MAJOR FINDING — body alone insufficient |
| D2  | fake_70k         | MAJOR FINDING — body alone insufficient |
| D2  | fake_100k        | MAJOR FINDING — body alone insufficient |
| D2  | shuffled_E       | MAJOR FINDING — body alone insufficient |
| D2  | cross_session_E  | MAJOR FINDING — body alone insufficient |
| V10 | fake_5k          | MAJOR FINDING — body alone insufficient |
| V10 | fake_25k         | MAJOR FINDING — body alone insufficient |
| V10 | fake_70k         | MAJOR FINDING — body alone insufficient |
| V10 | fake_100k        | MAJOR FINDING — body alone insufficient |
| V10 | shuffled_E       | MAJOR FINDING — body alone insufficient |
| V10 | cross_session_E  | MAJOR FINDING — body alone insufficient |

**12/12 cells land in the MAJOR FINDING regime.** No mixed regimes.

### Manuscript implications

The current `manuscript_interpretation.md` (under
`experiments/paper_analyses/causal_ablations/per_frame/`) needs revision.
The body-region primacy claim drawn from variance-box E2 is an artifact of
the variance-based mask. The correct framing under proper segmentation:

1. **Phase G's discrimination on F-A v1 fakes lives off-body.** When
   off-body content is preserved (E3_seg), AUROC = 1.000 across all
   conditions and sessions. When off-body is replaced (E2_seg), AUROC
   drops to chance or below.
2. **F-A v1 succeeds at making body content low-MSE under Phase G** but
   fails to perturb off-body content (background scene structure,
   projection on static surfaces, sensor edges) in a chain-coupled way.
   The off-body region carries the chain-coupled signal Phase G uses.
3. **EXP 6's chain-coupling demonstration is unaffected.** EXP 6 tested
   chain-coupling at the per-frame level (top-1 = 100% across all 120
   frames); that finding is orthogonal to where in the frame the
   discrimination lives. EXP 6 remains the load-bearing chain-coupling
   demonstration, regardless of this re-run's outcome.
4. **Visual-grids original interpretation re-confirmed.** The visual-grids
   reading ("body anti-discriminative") was correct; the variance-box
   ablations had inverted that reading; proper segmentation now re-aligns
   with visual grids.

---

## 5. Leakage check (pre-registered)

Pre-registered: E2_seg AUROC > E1 AUROC + 0.1 → flag possible mask leakage
through Phase G's normalization or skip connections.

✅ **No cells exceed the leakage threshold.** E2_seg AUROC values are far
BELOW E1 (0.009 vs E1 ≈ 1.000), in the opposite direction. Mask
construction is not leaking through; it's actively suppressing the
discriminative signal as expected.

---

## 6. Spatial visualization

Four representative frames, picked by per-frame
|score_E2_seg(real_correct) − score_E3_seg(real_correct)| (max and min per
session):

| panel | session | row | label | E2_seg field max | E3_seg field max |
|---|---|---|---|---|---|
| `spatial_viz/D2_f001634_high_contrast.png`  | D2  | 1634 | high contrast | 1.234 | 0.875 |
| `spatial_viz/D2_f001413_low_contrast.png`   | D2  | 1413 | low contrast  | 1.060 | 0.456 |
| `spatial_viz/V10_f001292_high_contrast.png` | V10 | 1292 | high contrast | 1.755 | 1.084 |
| `spatial_viz/V10_f002540_low_contrast.png`  | V10 | 2540 | low contrast  | 1.050 | 0.768 |

Each panel is a 2×3 layout:

- Row 1: C debayered RGB / Mask R-CNN overlay (green) / C under E2_seg
  (body kept, outside replaced with per-channel inside-mask mean)
- Row 2: E2_seg residual field (per-pixel ε-MSE under body-only mask) /
  E3_seg residual field / signed contrast (E2_seg − E3_seg, red where E2
  has higher residual, blue where E3 has higher residual)

Score fields downsampled to 384×512 for display; inputs are at native
768×1024.

Visual reading: the signed-contrast panel shows red concentrated on the
body (E2_seg residual is high there because body content under body-only
mask is what Phase G is trying to denoise without context) and faint blue
elsewhere. The E3_seg field is uniformly low — Phase G's denoising under
off-body-only is well-behaved (low residual = real-like), and that's why
E3_seg AUROC = 1.000 (real and fake separate cleanly when only off-body
is shown).

---

## 7. Outputs / file pointers

- `proper_segmentation/segmentation_manifest.csv` — Phase 1 per-frame
  classification (120 rows; dropped flag for the 14 pathological)
- `proper_segmentation/masks/<sess>_f<row:06d>.npy` — binary masks at
  768×1024 (120 files; pathological frames have empty masks)
- `proper_segmentation/per_frame/<sess>_f<row:06d>/ablations_manifest.json`
  — Phase 3 scalars (106 files)
- `proper_segmentation/ablation_table_seg.csv` — Phase 4 AUROC + Δscore
  table (24 rows = 2 ablations × 6 conditions × 2 sessions)
- `proper_segmentation/comparison_table.csv` — Phase 4 paired comparison
  with variance-box (24 rows)
- `proper_segmentation/comparison_summary.md` — Phase 4 markdown summary
- `proper_segmentation/phase4_summary.json` — Phase 4 stats for downstream
  consumers
- `proper_segmentation/spatial_viz/<sess>_f<row:06d>_<label>.png` — 4
  Phase 5 spatial panels
- `proper_segmentation/phase2_gate.json` — Phase 2 gate result
- `proper_segmentation/shards/g{0..7}.json` — per-GPU shard manifests for
  Phase 3 reproducibility

---

## 8. Wall-clock summary

| phase | wall-clock | GPUs |
|---|---|---|
| Phase 1 (segmentation, 120 frames) | ~1 min | 1 |
| Phase 2 (gate) | inline | 0 |
| Phase 3 (E2_seg/E3_seg, 106 frames) | ~17 min | 8 (parallel) |
| Phase 4 (analysis) | ~10 sec | 0 |
| Phase 5 (spatial viz, 4 frames) | ~30 sec | 1 |
| **Total** | **~20 min** | up to 8 |

(Operator estimate was 50–70 min; actual ran faster because the model load
is a one-time cost amortized across many frames.)

---

## 9. Codex audit

**Status: PASS** (after the two procedural blockers were resolved — see §9.2 below).

Codex audit was run 2026-05-05 by `codex:codex-rescue` agent.
Findings file: `proper_segmentation/CODEX_AUDIT.md`.

### 9.1 Checklist results

35 / 35 functional checklist items: PASS or PASS-with-minor-note.
- Mask R-CNN integration (4/4 PASS): weights from PV cache, person-class
  filter, highest-confidence selection, 0.5 threshold
- Mask construction (5/5 PASS, 1 minor): largest-CC, morph close+open
  with 3×3, 2-70% area bounds, drop policy applied; minor edge case
  noted in §9.4
- Mask application to (C, E) (4/4 PASS): same mask both inputs,
  per-channel inside-mask mean fill, correct `torch.where` polarity for
  E2_seg / E3_seg; leakage check all-pass (E2_seg << E1)
- Phase G inference convention (6/6 PASS): same `phase_g_score_scalar`
  imported, same noise-seed formula, T_STEPS / K_NOISE / bf16 /
  build_diffusion_constants all match causal_ablations.py
- Comparison-table construction (5/5 PASS, 1 minor): paired bootstrap
  with same indices for orig and seg per iteration; AUROC orientation
  `-MSE`; tie-handling matches reference; minor regime-function note
  in §9.4
- Standing rules (5/5 PASS): no Phase G writes; F-A v1 only;
  zero `fa_v2` mentions; zero `binder_split.json` references; zero
  D-family or binder-pool references

### 9.2 Resolved blockers

#### Blocker 1: Script provenance — RESOLVED

Codex flagged that the audited scripts were at `/tmp/*.py` only, not under
`proper_segmentation/scripts/`. **This was a false positive caused by the
audit agent not having `/data` mounted.** The scripts had already
been copied to `proper_segmentation/scripts/` before the audit launch.
Verified post-audit:

```
proper_segmentation/scripts/
├── build_phase3_shards.py   1461 bytes
├── phase1_segment_120.py    8375 bytes
├── phase3_seg_ablations.py 10411 bytes
├── phase4_analyze.py       17813 bytes
└── phase5_spatial_viz.py    8898 bytes
```

#### Blocker 2: phase2_gate.json said STOP — RESOLVED

The original `phase2_gate.json` was generated inline by Phase 1 against the
pre-correction gate spec (`≥100 per session` — unreachable since the subset
is 60 per session by construction). The operator clarified the corrected
gate 2026-05-05 mid-run:

> Combined pathological rate < 10%: PROCEED clean
> Combined pathological rate 10–25%: PROCEED with caveat
> Combined pathological rate > 25%: STOP
> Per-session minimum: ≥40 usable frames per session

Regenerated with `regenerate_phase2_gate.py`; new gate result:
**PROCEED_WITH_CAVEAT** (combined 11.7%, per-session min 50). The
gate file now records `policy_version: operator_corrected_2026-05-05` and
the supersession note.

### 9.3 Spot-check results

| check | result |
|---|---|
| `D2_f001328` mask area % vs .npy | exact match (diff 0.0) |
| `D2_f001328` noise seed | matches formula `42 + 1328*7919 = 10516474` |
| `D2_f001328` scalars dict | exactly 14 entries (2 ablations × 7 conditions) |
| `V10_f001279` mask area % vs .npy | exact match (diff 0.0) |
| `V10_f001279` noise seed | matches formula `42 + 1279*7919 + 1 = 10128444` |
| `V10_f001279` scalars dict | exactly 14 entries |
| All 120 mask shapes/dtypes | all `(768, 1024)` `bool`; 110 with mask, 10 all-false (matches pathological_no_mask count) |
| Comparison row recompute | E2→E2_seg, fake_5k, D2: AUROC_orig=0.9924, AUROC_seg=0.0088, Δ=−0.9836; bootstrap mean −0.9842, CI [−0.9976, −0.9636] — exactly matches |

### 9.4 Minor concerns (not blocking)

#### Minor 1: post-cleanup fragmentation edge case

`D2_f001328` has `orig_n_components=1` and `post_cleanup_n_components=2`,
classified as `clean` (because `fragmented_resolved` keys on original
component count, not post-cleanup count). This means morph open with the
3×3 kernel disconnected a thin region in the original largest-CC mask
into 2 components.

This is not a math error — the mask still represents the body region with
6.25% area, and the AUROC computation uses the union of components. But
the classification label is misleading. If a future re-run wants the
strictest definition, change `classify()` to also fail when
`post_components > 1`.

**Manuscript impact**: zero. Single frame out of 106 usable; mask
quality verified by direct inspection; AUROC computation correct.

#### Minor 2: Phase 4 regime function uses AUROC-only

The original `causal_ablations.regime()` uses both AUROC and
delta_degradation_pct. Phase 4's `regime()` uses AUROC alone.
For this analysis both classify identically: all E2_seg AUROCs are
`< 0.95` (strong) and all E3_seg AUROCs are `1.000` (weak). The 12/12
MAJOR FINDING classification is unchanged.

**Manuscript impact**: zero for the headline finding; mention in
methods that the regime function used here is AUROC-only.

### 9.5 Verdict

Codex initial verdict: **PARTIAL** (procedural blockers).
After resolving §9.2 blockers: **PASS**.

The body-vs-off-body inversion is not invalidated by any implementation
bug. Mask broadcasting, per-channel mean axis, noise seeding, AUROC
orientation, and paired bootstrap are all correct per spot-check
recomputation. Result is manuscript-citable.

Source files (canonical, on persistent volume):
- `proper_segmentation/scripts/phase1_segment_120.py`
- `proper_segmentation/scripts/phase3_seg_ablations.py`
- `proper_segmentation/scripts/phase4_analyze.py`
- `proper_segmentation/scripts/phase5_spatial_viz.py`
- `proper_segmentation/scripts/build_phase3_shards.py`

---

## 10. Open questions for the operator

The 12/12 MAJOR FINDING result is consistent across all conditions and
sessions, with tight CIs. But some downstream questions:

1. **Which off-body region is load-bearing?** Static background?
   Projection-on-background? Sensor frame? An additional ablation
   (E3_seg minus background, E3_seg minus projection edges, etc.)
   could localize within off-body.
2. **Does the off-body primacy generalize beyond F-A v1?** F-A v1 is the
   easy attacker; if F-A v2 can perturb off-body content too, the
   off-body-primacy claim might not hold. Pre-register before F-A v2 final
   eval.
3. **Manuscript reframe scope**: rewrite `manuscript_interpretation.md`
   under causal_ablations? Update SESSION_HANDOFF.md? Update visual-grids
   memo? Pre-existing docs that cite "body-region primacy" need updating.

Awaiting operator direction.
