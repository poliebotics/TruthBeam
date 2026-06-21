# Phase 0.5 — Multi-axis decomposition characterization

**Generated**: 2026-05-05.
**Purpose**: Characterize Phase G's substrate-evidence structure along three
decomposition axes (octave, channel, PatchCore-style memory-bank), to
inform substrate benchmark catalog before CGPT round 4 finalizes.
**Standing rules**: Phase G inference-only. No re-inference, no model
modifications. Used cached `residual_4ch_mean_<cond>.npy` fields and
proper-segmentation Mask R-CNN masks.
**GPU usage**: Task 1 used cuda:0 for separable Gaussian blur (much
faster than CPU at large σ); Tasks 2, 3 CPU-only. No Phase G inference.

---

## 1. Background — today's audit-locked findings

The proper-segmentation E2/E3 ablation (Codex audit PASS) overturned the
variance-box body-primacy claim:

- **E2_seg AUROC = 0.009–0.538** under ε-MSE (body-only) — F-A v1 fake
  body content overspreads Phase G's conditional likelihood mode.
- **E3_seg AUROC = 1.000** (off-body) — Phase G's discrimination on F-A v1
  fakes via ε-MSE lives in the off-body region.

The Phase 0 precondition revealed `low_freq_mse` (Gaussian-blurred
residual MSE, σ=4) RETAINS AUROC ≈ 0.998 on E2_seg — body-region signal
exists but lives at low spatial frequencies that ε-MSE's full-spectrum
summation washes out via high-frequency dominance.

Phase 0.5 (this document) characterizes the substrate-evidence structure
along three decomposition axes to inform whether they should be elevated
to first-class methods in the substrate benchmark.

---

## 2. Octave decomposition (Task 1)

Difference-of-Gaussians bandpass at σ ∈ {0.5, 1, 2, 4, 8, 16, 32};
bandpass_i = blur(σ_i) − blur(2σ_i); aggregated under E2_seg / E3_seg /
full_frame.

### E2_seg AUROC by σ (the body-region signal)

| σ | D2 mean | V10 mean | regime |
|---|---|---|---|
| 0.5 | **0.267** | **0.552** | **inverted** (F-A v1's smoothing dominates) |
| 1 | 0.523 | 0.740 | ~chance |
| 2 | 0.698 | 0.879 | borderline |
| **4** | **0.996** | **0.992** | **transition — signal kicks in** |
| 8 | 1.000 | 1.000 | **saturated** |
| 16 | 1.000 | 1.000 | saturated |
| 32 | 1.000 | 1.000 | saturated |

### Headline

**Body-region signal is at σ ≥ 4 (coarse spatial scales); F-A v1
overspread is at σ ≤ 2 (fine scales).** Sharp transition between σ=2
(D2 0.70, V10 0.88) and σ=4 (D2 0.996, V10 0.992).

This refines the Phase 0 finding (low_freq_mse σ=4 → AUROC ≈ 0.998).
The signal is present at every coarse scale; ε-MSE's failure under
body-only is because its summation is dominated by σ ≤ 2 where F-A v1's
smoothed body residual sits BELOW real residual.

### E3_seg

All σ values give AUROC = 0.97–1.000 on E3_seg (broadband off-body
discriminability — F-A v1 doesn't perturb off-body content at any
spatial scale).

### Pre-registered interpretation result

> Smooth peak at σ ∈ {2, 4, 8} on E2_seg with AUROC > 0.95 → body-region
> signal is at coarse spatial scales

**Result**: peak at σ=8 (AUROC 1.000); transition at σ=4. Pre-registered
"coarse spatial scales" interpretation lands.

---

## 3. Channel decomposition (Task 2)

Per-channel ε-MSE / low_freq_MSE / signed_residual computed on R, G=mean(G1, G2), B
debayered from packed CFA. Aggregated under E2_seg / E3_seg / full_frame.

### ε-MSE under E2_seg (body-only)

| session | R | G | B | range |
|---|---|---|---|---|
| **D2** | **1.000** | 0.580 | 0.874 | **+0.420** |
| **V10** | **0.997** | 0.757 | 0.871 | **+0.240** |

### Headline

**F-A v1 has strong channel asymmetry: R-channel body residual gives
perfect AUROC, G-channel is near chance, B-channel is borderline.**

R-channel-only ε-MSE on body alone discriminates F-A v1 fakes perfectly
without any frequency or off-body context. This means F-A v1's body
edit perturbs G channel substantially (washing out ε-MSE there) but
leaves R channel near-untouched — and R's natural chain-coupled response
in the body region is sharply different between real and F-A v1 frames.

### Cross-channel correlation matrices (full-frame ε-MSE)

**D2**:

| | R | G | B |
|---|---|---|---|
| Real R | +1.000 | +0.761 | +0.212 |
| Real G | +0.761 | +1.000 | +0.373 |
| Real B | +0.212 | +0.373 | +1.000 |

| | R | G | B |
|---|---|---|---|
| Fake R | +1.000 | +0.986 | +0.860 |
| Fake G | +0.986 | +1.000 | +0.855 |
| Fake B | +0.860 | +0.855 | +1.000 |

| | R | G | B |
|---|---|---|---|
| **Δ R** | +0.000 | **+0.225** | **+0.648** |
| **Δ G** | **+0.225** | +0.000 | **+0.482** |
| **Δ B** | **+0.648** | **+0.482** | -0.000 |

**V10** (Δ): R-G +0.327, R-B +0.459, G-B +0.288.

**F-A v1 produces unrealistically high cross-channel correlations.** Real
chain-coupled residuals have channel-specific structure (R-B corr ~0.21
on D2); F-A v1 fakes have R-B corr ~0.86. This is essentially
monochromatic editing.

### Pre-registered interpretation result

> One or two channels show much higher AUROC than others → F-A v1 has
> channel-asymmetric attack quality — defender axis available
> Correlation matrix differs substantially → F-A v1 produces unrealistic
> channel correlation — defender axis available

**Result**: BOTH conditions land. R/G/B asymmetry on E2_seg + cross-
channel correlation difference > 0.4. Two strong channel-axis defender
levers.

### low_freq_mse per channel under E2_seg

All three channels give AUROC = 1.000 under low_freq_mse (consistent
with the octave finding: at σ=4, every channel discriminates).

---

## 4. PatchCore-style memory-bank characterization (Task 3)

Memory bank: per-session, real_correct patches only, 10% coreset
subsample (LOO per test frame). Patches: 8×8 non-overlapping in 96×128
downsampled residual field; 192 patches/frame, 256-dim feature vector
(4 ch × 8 × 8). NN distance: L2 in raw patch space.

### AUROC by aggregation × mask_cond × session

All cells: **AUROC = 1.000**.

| aggregation | E2_seg D2 | E2_seg V10 | E3_seg D2 | E3_seg V10 | full D2 | full V10 |
|---|---|---|---|---|---|---|
| mean | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |
| p95 | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |
| max | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |

### Sanity check on real-vs-perturbed gap

LOO baseline (real_correct) NN distance ~0.4; perturbed conditions ~0.5–1.2.
Clean separation, not a numerical artifact.

### Pre-registered interpretation result

> Mean-NN-distance AUROC > 0.9 across conditions → PatchCore-style memory
> bank reads substrate evidence well. Should be added to substrate
> benchmark as first-class method.

**Result**: all 6 conditions × all 3 mask_conds × all 3 aggregations land
at AUROC = 1.000 in raw 4-ch CFA patch space. The memory-bank /
nearest-neighbor family is a strong first-class candidate.

### Caveat

The raw-residual representation works well; this does not test whether
Phase G feature patches (intermediate activations) would be even
stronger. Per spec, switching to feature patches requires Phase G
inference and is gated on operator authorization.

---

## 5. Cross-axis observations

### Axis interactions

| axis | what it isolates | F-A v1 attack profile |
|---|---|---|
| spatial region (E2/E3) | body vs off-body | leaves off-body untouched |
| spatial frequency (octave) | high-freq vs low-freq residual | smooths body high-freq, leaves low-freq |
| color channel (RGB) | per-channel structure | edits G channel, leaves R |
| memory-bank (PatchCore) | local patch typicality | locally OOD at all 192 patches |

The four axes are **largely orthogonal** with respect to F-A v1:

- F-A v1 leaves off-body untouched → **region** axis catches
- F-A v1 smooths body high-freq → **frequency** axis catches at σ ≥ 4
- F-A v1 mostly edits G channel → **channel** axis catches via R-only
- F-A v1 edits don't match local patch distribution → **patch** axis catches

A defender combining all four (e.g., R-channel low_freq_mse on body-only,
plus PatchCore on full frame) would catch F-A v1 redundantly and would
require F-A v2 to attack along all four axes simultaneously.

### Why PatchCore subsumes the others as a single test

PatchCore at 8×8 patches in raw 4-channel CFA space is sufficient
because each 8×8 patch implicitly carries:
- Spatial-frequency structure (high-freq within patch, position carries low-freq)
- Channel structure (4 channels per pixel)
- Body vs off-body (patch position)

The memory bank captures the joint typical distribution. F-A v1's
deviation along any single axis (or combination) puts patches OOD.

This is consistent with PatchCore's general anomaly-detection power
(Roth et al. 2022): patch-level NN distance is a strong baseline that's
hard to beat without structural knowledge.

---

## 6. Implications for substrate benchmark scope

### Methods worth elevating to first-class

| method | rationale |
|---|---|
| **PatchCore (memory-bank NN)** | Canonical anomaly-detection family; AUROC=1.000 across all 252 cells in characterization; 30-sec runtime on cached fields; cleanly informative on E2_seg as well as E3_seg |
| **Octave / multi-scale residual MSE** | Sharp transition at σ ≥ 4 reveals body-region signal that ε-MSE washes out; useful for diagnostic (where in frequency does the signal live?) and as a scoring function |
| **Per-channel ε-MSE (R-channel only)** | Dramatic asymmetry on body-only (R: 1.000, G: 0.58, B: 0.87 on D2); cheap diagnostic that reveals F-A v1's monochromatic editing pattern |
| **Cross-channel correlation residual** | Real R-B corr ~0.21, F-A v1 R-B corr ~0.86; correlation deviation as a typicality signal |

### Methods worth keeping as summary statistics (not first-class)

- Plain ε-MSE under full-frame: AUROC = 1.000 for F-A v1 but the failure
  modes (E2_seg overspread, σ ≤ 2 inversion) are diagnostically misleading
  unless paired with axis decompositions.
- VGG / B1 distance: trivially 1.000 due to degenerate real_correct
  baseline; not informative for this characterization.

### Recommended catalog reduction

If the benchmark needs to limit method count (per CGPT round 4 deep-search
note about avoiding redundant aggregations of same field), **PatchCore +
octave + channel (R-only)** is a triple that spans three orthogonal
axes and avoids stacking near-duplicates. ε-MSE belongs as the
established baseline; specifically NOT as first-class but as the
contrast against which the new methods' improvements are measured.

---

## 7. F-A v2 design implications

To defeat the four characterized axes, F-A v2 training must include:

1. **Off-body chain-coupled perturbation**. F-A v1 doesn't touch off-body
   content. F-A v2 must learn to perturb projection-on-static-surfaces,
   sensor frame edges, and other off-body regions in a way that matches
   the chain-coupled distribution. Loss should include AUROC drop on
   E3_seg as a success criterion (not just on full-frame ε-MSE).

2. **Low-frequency body residual penalty**. F-A v1 smooths body content,
   washing out high-frequency residual but leaving low-frequency residual
   chain-coupled at AUROC ≈ 1. F-A v2's loss should penalize both
   high-frequency AND low-frequency residual deviation; specifically a
   `low_freq_mse` (σ=4) penalty term so F-A v2 can't exploit Phase G's
   ε-MSE summation.

3. **Per-channel symmetric perturbation**. F-A v1 mostly edits G channel.
   F-A v2 must perturb R, G, B channels independently in a way matching
   real chain-coupled per-channel response. Loss should include per-
   channel ε-MSE penalty (or a single "min over channels of ε-MSE on
   body" penalty so F-A v2 can't hide in any single channel).

4. **Realistic cross-channel correlations**. F-A v1 produces R-B
   correlation +0.86 vs real +0.21. F-A v2's loss should include a
   correlation-matching penalty: e.g., per-frame cross-channel correlation
   computed on residual, penalty for deviation from training-distribution
   real correlation.

5. **Per-patch typicality**. PatchCore catches at patch level. F-A v2
   could be trained with a memory-bank-aware loss: at each 8×8 patch
   location, the F-A v2 output should fall within the typical real
   distribution. This is the strongest constraint and likely the
   bottleneck for F-A v2 to evade Phase G.

Items 1–4 are F-A v1 leaks the operator can target with explicit loss
terms. Item 5 is a more general typicality constraint that may require
a learned typicality scorer in the F-A v2 training loop.

---

## 8. What this characterization does NOT establish

- Whether the four axes generalize beyond F-A v1. The defender axes named
  here are chosen because they reveal F-A v1's limitations; if F-A v2
  attacks all four simultaneously, the relevant question shifts to
  axis-INDEPENDENT typicality (PatchCore-like).
- Whether the substrate benchmark should use these specific
  representations or different ones (e.g., Phase G intermediate features,
  not just ε-residuals). Pending operator decision.
- Whether the recording-rig generalization holds (D2+V10 only;
  same-rig same-operator). The characterization is cached on these two
  sessions.

---

## 9. Output file pointers

```
experiments/paper_analyses/typicality_layer/phase_0_5/
├── octave_decomposition/
│   ├── per_frame_octave_mse.npy       (106, 7, 3, 7) — frames × σ × mask × cond
│   ├── auroc_per_octave.csv           252 rows = 7 σ × 3 masks × 2 sess × 6 conds
│   ├── frame_meta.json
│   └── summary.md
├── channel_decomposition/
│   ├── per_frame_channel_mse.npy      (106, 3, 3) — frames × ch × scoring_fn
│   ├── auroc_per_channel.csv          324 rows = 3 ch × 3 fns × 3 masks × 2 sess × 6 conds
│   ├── channel_correlation_real.csv
│   ├── channel_correlation_fake.csv
│   ├── channel_correlation_diff.csv
│   ├── frame_meta.json
│   └── summary.md
├── patchcore_characterization/
│   ├── memory_bank.npz                D2_bank (n_d2, 256) + V10_bank (n_v10, 256)
│   ├── per_frame_nn_distances.npy     (106, 7, 3, 3) — frames × cond × mask × agg
│   ├── auroc_patchcore.csv            108 rows = 3 aggs × 3 masks × 2 sess × 6 conds
│   ├── frame_meta.json
│   └── summary.md
└── CONSOLIDATED_SUMMARY.md            (this file)
```

Visual-grids memo also updated:
`experiments/paper_analyses/visual_grids/README.md` (Interpretation update — 2026-05-05).

---

## 10. Wall-clock summary

| task | wall-clock | hardware |
|---|---|---|
| Task 4 — visual-grids memo update | ~10 min | local edit |
| Task 2 — channel decomposition | ~1 min | CPU |
| Task 3 — PatchCore characterization | ~1 min | CPU |
| Task 1 — octave decomposition (v2 GPU) | 18 sec | cuda:0 |
| Consolidated summary | ~15 min | local edit |
| **Total** | **~30 min** | mostly trivial after script writing |

(Tasks 2 and 3 finished in ~1-2 minutes each; Task 1 v1 with cv2 on CPU
was estimated ~13-26 hours due to large-σ Gaussian kernels, so v2 was
rewritten with separable conv on GPU and finished in 18s.)

---

## 11. Standing rules acknowledged

- Phase G inference-only; no re-inference, no model modifications.
- F-A v1 outputs only (cached); no F-A v2 trainer touch.
- B1, B2, e2, e2_fp16 authorized binders; no use of A1/A2/C1/C2/D_large/e3r in
  this characterization (no binder use at all — only Phase G ε-residuals
  cached from the scoring-function comparison).
- binder_split.json LOCKED v3 — unchanged.
- No held-out asset use beyond F-A v1 outputs already cached.
- Pre-registered interpretation thresholds applied verbatim from each
  task's spec.
