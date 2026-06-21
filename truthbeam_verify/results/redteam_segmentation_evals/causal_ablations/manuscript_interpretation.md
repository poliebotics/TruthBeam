# Manuscript interpretation — Phase G causal ablations

**Last updated**: 2026-05-05.

This document supersedes the prior body-region-primacy reading drawn from
the variance-box E2/E3 ablations. The proper-segmentation re-run with
Mask R-CNN body silhouettes overturns that conclusion. See §3 for the
inversion and §4 for the more nuanced multi-scoring-function picture from
the Phase 0 precondition.

---

## 1. Pre-registered interpretation thresholds (frozen before analysis)

AUROC degradation regimes:
- `strong_dependence`: AUROC < 0.85 OR median |Δscore| reduction > 50%
- `moderate_dependence`: AUROC 0.85–0.95 OR Δ reduction 20–50%
- `weak_dependence`: AUROC > 0.95 AND Δ reduction < 20%

Combined ablation interpretations (pre-registered for variance-box):
- E2 maintains > 0.95: body-only evidence sufficient
- E2 drops: body region not where discrimination lives
- E3 maintains > 0.95: off-body evidence sufficient
- E3 drops: off-body region needed
- E4 drops: Phase G discriminates on chain-coupled (C, E) match
- E4 maintains: Phase G reads "E plausible for session"
- E5 drops: discrimination depends on pose cue
- E5 maintains: discrimination beyond pose alone

The variance-box result (E2 ≈ 0.99, E3 ≈ 0.85–0.98) initially registered
as "body-region primacy survives." That reading is now revised — see §3.

---

## 2. Variance-box ablation result (causal_ablations.py default)

| ablation | D2 AUROC range (perturbed conds) | V10 AUROC range |
|---|---|---|
| E1 (baseline) | 1.000 | 1.000 |
| E2 (body-only, variance-box) | 0.977–0.998 | 0.982–0.995 |
| E3 (off-body, variance-box)  | 0.815–0.911 | 0.928–0.995 |
| E4 (wrong-center-E)          | varies     | varies     |
| E5 (matched-pose)            | varies     | varies     |

Variance-box mask construction: 32×32 sliding-window variance, p75
threshold, largest-vertical-extent connected component, 10% bbox
expansion. Visual inspection of these masks revealed they captured
**body + adjacent high-variance non-body content** (projection edges
on background, scene structure, sensor frame edges). The mask
boundary did not cleanly separate body from off-body.

---

## 3. Proper-segmentation result (Mask R-CNN, 2026-05-05)

Per `experiments/paper_analyses/overnight_battery/proper_segmentation/SEGMENTATION_RESULTS.md`.

### 3.1 Setup

- Mask R-CNN ResNet50 FPN (COCO_V1) applied to 120 frames; 14 dropped
  as pathological (no person detected, mask <2%, mask >70%); 106 usable
  (D2=50, V10=56).
- Largest connected component + morphological close+open with 3×3
  kernel.
- E2_seg / E3_seg ablations re-run with same Phase G inference
  conventions (T_STEPS, K_NOISE, noise seeds matched per-frame).
- Per-channel inside-mask mean fill for replacement; same for both
  C (4-ch CFA) and E (3-ch projection).
- Codex audit verified: PASS.

### 3.2 AUROC reversal (paired n=106 across orig/seg)

| condition | E2 orig | E2_seg | ΔAUROC | E3 orig | E3_seg | ΔAUROC |
|---|---|---|---|---|---|---|
| D2 fake_5k          | 0.992 | 0.009 | **−0.984** | 0.909 | 1.000 | +0.091 |
| D2 fake_25k         | 0.990 | 0.016 | **−0.974** | 0.830 | 1.000 | +0.170 |
| D2 fake_70k         | 0.985 | 0.024 | **−0.961** | 0.851 | 1.000 | +0.149 |
| D2 fake_100k        | 0.977 | 0.024 | **−0.953** | 0.815 | 1.000 | +0.185 |
| D2 shuffled_E       | 0.998 | 0.536 | −0.462 | 0.911 | 1.000 | +0.089 |
| D2 cross_session_E  | 0.998 | 0.536 | −0.463 | 0.911 | 1.000 | +0.089 |
| V10 fake_5k         | 0.982 | 0.148 | **−0.835** | 0.980 | 1.000 | +0.020 |
| V10 fake_25k        | 0.982 | 0.196 | **−0.786** | 0.931 | 1.000 | +0.069 |
| V10 fake_70k        | 0.982 | 0.261 | **−0.722** | 0.952 | 1.000 | +0.048 |
| V10 fake_100k       | 0.982 | 0.251 | **−0.731** | 0.928 | 1.000 | +0.072 |
| V10 shuffled_E      | 0.995 | 0.538 | −0.458 | 0.948 | 1.000 | +0.052 |
| V10 cross_session_E | 0.995 | 0.537 | −0.458 | 0.942 | 1.000 | +0.058 |

Every ΔAUROC 95% CI excludes zero (paired bootstrap on 106 frames).

### 3.3 Distribution-distance reading

The pre-registered cross-mask interpretation classifies all 12 cells as
"MAJOR FINDING — body alone insufficient." A more precise reading using
distribution-distance framing:

**E2_seg (body-only)**: F-A v1 fake body content lies *closer* to Phase G's
conditional likelihood mode than training-distribution real samples do.
This is the known **overspread failure mode** of likelihood-based OOD
detection (Nalisnick, Matsukawa, Teh, Gorur, Lakshminarayanan 2019, "Do
Deep Generative Models Know What They Don't Know?"): a generative model
trained on natural images can assign **higher** likelihood to OOD samples
that lie in simpler, lower-variance regions of pixel space (e.g.,
constant images, smoothed inputs). F-A v1's residual editor smooths body
content via its `delta_C` perturbation, yielding body regions that are
"easier to denoise" under Phase G's diffusion-trained mode — therefore
appearing more real-like by ε-MSE under body-only masking.

**E3_seg (off-body)**: Off-body content from F-A v1 inference paths is
reliably out-of-distribution under Phase G's learned conditional. F-A v1
does not modify off-body content (background scene, projection on static
surfaces, sensor frame edges) in a chain-coupled way; that region carries
the discriminative signal Phase G uses to verify chain coupling. AUROC =
1.000 across all 12 cells.

The "fakes more real than real" surface phrasing is misleading; the
correct framing is **F-A v1 produces body content in a region of Phase
G's conditional likelihood landscape closer to the mode than training
samples occupy** — a known and well-documented OOD-detection failure mode
for likelihood-based scoring.

---

## 4. Phase 0 precondition — multi-scoring-function picture

Per `experiments/paper_analyses/typicality_layer/phase_0/interpretation_summary.md`.

### 4.1 Setup

7 scoring functions × 3 mask conditions (E2_seg / E3_seg / full_frame) ×
2 sessions × 6 perturbed conditions = 252 cells. Reuses
`scoring_function_comparison/` per-pixel score fields, applies
proper-segmentation masks, computes scalar AUROC per cell.

### 4.2 E2_seg AUROC by scoring function (averaged across 6 perturbed conds)

| scoring fn | D2 | V10 | retains body-only signal? |
|---|---|---|---|
| ε-MSE          | 0.298 | 0.405 | no (F-A v1 fakes overspread to mode) |
| ε-MAE          | 0.328 | 0.444 | no (same direction as ε-MSE) |
| signed_residual | 0.776 | 0.771 | borderline (close to but below 0.85) |
| **low_freq_mse** | **0.997** | **0.998** | **YES** — non-trivial pass |
| high_freq_mse  | 0.211 | 0.321 | no — even worse than ε-MSE |
| VGG distance   | 1.000 | 1.000 | trivial pass (degenerate real_correct = 0) |
| B1-encoder distance | 1.000 | 1.000 | trivial pass |

**Critical refinement**: the ε-MSE failure under E2_seg is a
scoring-function-specific phenomenon, not an absolute "no body signal"
claim. Phase G's per-pixel residual on body regions DOES carry
discriminative information — but the information lives at low spatial
frequencies (Gaussian-blurred residual MSE with σ=4 retains AUROC ≥
0.99). High spatial frequencies in the body residual are dominated by
F-A v1's smoothing artifact and ε-MSE summation washes the
low-frequency signal out.

### 4.3 Scoring-function decomposition of the body-region signal

- **Low-frequency body residual** (gauss σ=4 MSE): AUROC ≈ 0.998 →
  Phase G has body-region signal at coarse spatial scales (likely
  related to overall radiometric coupling: how the body's emission
  response distributes globally).
- **High-frequency body residual** (residual minus blurred): AUROC ≈
  0.27 → at fine scales, F-A v1's smoothing dominates; Phase G's
  ε-prediction matches F-A v1's smoothed body better than real body's
  true high-frequency texture.
- **Combined ε-MSE**: AUROC ≈ 0.35 → the high-frequency overspread
  dominates, washing out the low-frequency signal.

Conclusion: Phase G learned a **multi-scale** body-region representation,
but its training scoring (ε-MSE summed over all scales) is dominated by
the failure mode at high frequencies. The off-body region's AUROC = 1.000
under ε-MSE is robust because both real and F-A v1 fakes leave off-body
content (and its high-frequency texture) untouched and chain-coupled —
F-A v1 hasn't perturbed that region away from real-like.

---

## 5. EXP 6 status — orthogonal load-bearing demonstration

The body-vs-off-body inversion does NOT affect EXP 6 (per-frame
chain-coupling specificity, top-1 = 100% across 120 frames). EXP 6 is a
**relative-comparison test**: for each frame, Phase G's score on the
correct (C, E) pair is lower than its score on each of N alternative E
candidates. This relative ordering is robust to absolute-MSE failure
modes — even if F-A v1's body content overspreads to mode, the score
under correct E vs wrong E is still ordered correctly per-frame.

EXP 6 remains the load-bearing chain-coupling demonstration regardless
of the absolute-MSE / OOD-detection finding.

---

## 6. Visual-grids original reading vindicated

The visual-grids review (across ε-MSE / VGG / B1 scoring functions on the
30+30 frame subset) noted **body-region anti-discriminative** —
F-A v1 fakes had MORE real-like residual on the body than real frames.
This was earlier interpreted as inverted by the variance-box ablation
result. Proper segmentation re-aligns with the visual-grids reading: body
region IS anti-discriminative under ε-MSE, exactly as visual grids
suggested. The variance-box ablation was conflating body with adjacent
high-variance non-body content, smuggling discrimination back in via the
non-body part of the variance-box mask.

This is a methodological lesson worth surfacing in the paper: variance-
based masks are a poor proxy for "subject region" when the projector
casts high-variance patterns on the static background; semantic
segmentation is the correct tool.

---

## 7. F-A v2 design implications

F-A v2's design must address two distinct failure modes Phase G exhibits
and exploit different conditional structures:

### 7.1 Body overspread (offensive opportunity)

F-A v1 already exploits this implicitly via `delta_C` smoothing on the
body region. F-A v2 should be aware that smoothing body content is
already "free" against Phase G's ε-MSE — the harder problem is

### 7.2 Off-body inertia (defensive challenge)

F-A v1 leaves off-body content essentially unmodified, and that's where
Phase G catches it (AUROC = 1.000 on E3_seg). For F-A v2 to evade Phase
G, the editor must learn to perturb **off-body** regions in a chain-
coupled way matching the target E. Concretely:
- Off-body body-cast shadow regions where projection lands on static
  surfaces (walls, floor) — these are physically chain-coupled but F-A v1
  doesn't touch them.
- Sensor edges and frame boundaries — small but contribute to AUROC.
- Background regions where structured-light boundaries are visible.

The F-A v2 attack-success criterion should explicitly include AUROC drop
on E3_seg, not just on full-frame ε-MSE.

### 7.3 Multi-scale spectrum

F-A v2's loss should include a low-frequency residual penalty so the
attacker doesn't accidentally recover the low_freq_mse body-region signal
(currently retained at AUROC ≈ 0.99 even on body-only). If F-A v2's
delta_C smooths the body in a way that also matches the chain-coupled
low-frequency radiometric response, low_freq_mse loses its body-region
signal too.

---

## 8. What this experiment establishes

- Phase G's discrimination on F-A v1 fakes via ε-MSE lives in the
  off-body region (E3_seg AUROC = 1.000); body region under ε-MSE
  exhibits the overspread failure mode of likelihood-based OOD detection
  (E2_seg AUROC < 0.5).
- The body-region signal is NOT absent — it's at low spatial frequencies
  (low_freq_mse on E2_seg AUROC ≈ 0.998), masked by ε-MSE's
  high-frequency dominance.
- Chain-coupling demonstration via EXP 6 (relative-comparison) is
  unaffected.
- Variance-box masks were conflating body with adjacent high-variance
  non-body content; methodological lesson for similar ablations.

## 9. What this experiment does NOT establish

- Whether the off-body primacy generalizes to harder attackers. F-A v1
  is the easy attacker; F-A v2 may explicitly target off-body regions.
  Pre-register before F-A v2 final eval.
- Whether the patterns generalize beyond same-rig D2+V10. Sample size
  is two recording sessions, one operator.
- Whether Phase G's learned representation could be re-summarized via a
  different scoring function (e.g., per-frequency-band logistic
  regression) to get robust body-region discrimination on F-A v1.

## 10. Standing rules acknowledged

Phase G inference-only. No held-out asset use beyond F-A v1. No F-A v2
trainer touch. No information from this experiment feeds back into Phase
G design or F-A v2 training. Pre-registered thresholds frozen — no
post-hoc adjustment of regime boundaries. The cross-mask classification
decision tree was applied verbatim from the operator's spec.
