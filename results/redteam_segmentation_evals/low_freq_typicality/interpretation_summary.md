# Phase 0 precondition — scoring function ablation under proper-segmentation masks

## Setup

- Frames: 50 D2 + 56 V10 = 106 usable (post-segmentation gate)
- 7 scoring functions × 3 mask conditions (E2_seg / E3_seg / full_frame) × 2 sessions × 6 perturbed conditions = 252 cells
- Source: per-pixel score fields from `scoring_function_comparison/` infrastructure; masks from `proper_segmentation/masks/`
- VGG distance and B1 distance have implicit 0 baseline for real_correct (distance to self = 0); AUROC under those scoring functions is trivially 1.0 except where the perturbed scalar happens to be 0

## AUROC summary by (scoring_fn, mask_cond, session) — averaged across 6 perturbed conds

| scoring_fn | E2_seg D2 | E2_seg V10 | E3_seg D2 | E3_seg V10 | full_frame D2 | full_frame V10 |
|---|---|---|---|---|---|---|
| eps_mse | 0.298 · | 0.405 · | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |
| eps_mae | 0.328 · | 0.444 · | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |
| signed_residual | 0.776 · | 0.771 · | 0.805 · | 0.787 · | 0.932 ✓ | 0.957 ✓ |
| low_freq_mse | 0.997 ✓ | 0.998 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |
| high_freq_mse | 0.211 · | 0.321 · | 0.995 ✓ | 0.999 ✓ | 0.978 ✓ | 0.955 ✓ |
| vgg_distance | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |
| b1_distance | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |

(✓ = exceeds 0.85 precondition threshold; · = below)

## E2_seg precondition findings (which scoring functions retain AUROC > 0.85 on body-only)

- **eps_mse**: AUROC range [0.004, 0.862], mean 0.351 — FAILS body-only ablation
- **eps_mae**: AUROC range [0.007, 0.959], mean 0.386 — FAILS body-only ablation
- **signed_residual**: AUROC range [0.609, 0.996], mean 0.774 — FAILS body-only ablation
- **low_freq_mse**: AUROC range [0.986, 1.000], mean 0.997 — **RETAINS** discriminability on body-only
- **high_freq_mse**: AUROC range [0.001, 0.622], mean 0.266 — FAILS body-only ablation
- **vgg_distance**: AUROC range [1.000, 1.000], mean 1.000 — **RETAINS** discriminability on body-only
- **b1_distance**: AUROC range [1.000, 1.000], mean 1.000 — **RETAINS** discriminability on body-only

## Implications for typicality-layer investigation

- Scoring functions that **fail** E2_seg precondition (AUROC ≤ 0.85): ['eps_mse', 'eps_mae', 'signed_residual', 'high_freq_mse']
- Scoring functions that **trivially pass** (degenerate real_correct = 0; VGG/B1 distance): ['vgg_distance', 'b1_distance']
- Scoring functions that **non-trivially pass**: ['low_freq_mse']

If the non-trivial-pass list is empty, the E2_seg AUROC drop observed in Phase G's ε-MSE is **scoring-function-general** — no informative summary of Phase G's residual restricted to body pixels distinguishes real from F-A v1 fakes.

If the non-trivial-pass list is non-empty, those scoring functions reveal a body-region signal Phase G's ε-MSE framing misses.

## Per-cell AUROC table

| scoring_fn | mask_cond | sess | cond | n | real_mean | pert_mean | AUROC | >0.85? |
|---|---|---|---|---|---|---|---|---|
| eps_mse | E2_seg | D2 | fake_5k | 50 | 0.006997 | 0.005132 | 0.042 |  |
| eps_mse | E2_seg | D2 | fake_25k | 50 | 0.006997 | 0.004326 | 0.010 |  |
| eps_mse | E2_seg | D2 | fake_70k | 50 | 0.006997 | 0.004017 | 0.006 |  |
| eps_mse | E2_seg | D2 | fake_100k | 50 | 0.006997 | 0.003877 | 0.004 |  |
| eps_mse | E2_seg | D2 | shuffled_E | 50 | 0.006997 | 0.008229 | 0.860 | ✓ |
| eps_mse | E2_seg | D2 | cross_session_E | 50 | 0.006997 | 0.008235 | 0.862 | ✓ |
| eps_mse | E2_seg | V10 | fake_5k | 56 | 0.009249 | 0.007349 | 0.200 |  |
| eps_mse | E2_seg | V10 | fake_25k | 56 | 0.009249 | 0.007221 | 0.239 |  |
| eps_mse | E2_seg | V10 | fake_70k | 56 | 0.009249 | 0.007461 | 0.269 |  |
| eps_mse | E2_seg | V10 | fake_100k | 56 | 0.009249 | 0.007228 | 0.233 |  |
| eps_mse | E2_seg | V10 | shuffled_E | 56 | 0.009249 | 0.01048 | 0.742 |  |
| eps_mse | E2_seg | V10 | cross_session_E | 56 | 0.009249 | 0.01048 | 0.746 |  |
| eps_mse | E3_seg | D2 | fake_5k | 50 | 0.001591 | 0.004335 | 1.000 | ✓ |
| eps_mse | E3_seg | D2 | fake_25k | 50 | 0.001591 | 0.003489 | 1.000 | ✓ |
| eps_mse | E3_seg | D2 | fake_70k | 50 | 0.001591 | 0.003272 | 1.000 | ✓ |
| eps_mse | E3_seg | D2 | fake_100k | 50 | 0.001591 | 0.003017 | 1.000 | ✓ |
| eps_mse | E3_seg | D2 | shuffled_E | 50 | 0.001591 | 0.00321 | 1.000 | ✓ |
| eps_mse | E3_seg | D2 | cross_session_E | 50 | 0.001591 | 0.003207 | 1.000 | ✓ |
| eps_mse | E3_seg | V10 | fake_5k | 56 | 0.001659 | 0.004787 | 1.000 | ✓ |
| eps_mse | E3_seg | V10 | fake_25k | 56 | 0.001659 | 0.00412 | 1.000 | ✓ |
| eps_mse | E3_seg | V10 | fake_70k | 56 | 0.001659 | 0.004011 | 1.000 | ✓ |
| eps_mse | E3_seg | V10 | fake_100k | 56 | 0.001659 | 0.003768 | 1.000 | ✓ |
| eps_mse | E3_seg | V10 | shuffled_E | 56 | 0.001659 | 0.003329 | 1.000 | ✓ |
| eps_mse | E3_seg | V10 | cross_session_E | 56 | 0.001659 | 0.003326 | 1.000 | ✓ |
| eps_mse | full_frame | D2 | fake_5k | 50 | 0.001853 | 0.004379 | 1.000 | ✓ |
| eps_mse | full_frame | D2 | fake_25k | 50 | 0.001853 | 0.003533 | 1.000 | ✓ |
| eps_mse | full_frame | D2 | fake_70k | 50 | 0.001853 | 0.00331 | 1.000 | ✓ |
| eps_mse | full_frame | D2 | fake_100k | 50 | 0.001853 | 0.003061 | 1.000 | ✓ |
| eps_mse | full_frame | D2 | shuffled_E | 50 | 0.001853 | 0.003454 | 1.000 | ✓ |
| eps_mse | full_frame | D2 | cross_session_E | 50 | 0.001853 | 0.003452 | 1.000 | ✓ |
| eps_mse | full_frame | V10 | fake_5k | 56 | 0.002101 | 0.004936 | 1.000 | ✓ |
| eps_mse | full_frame | V10 | fake_25k | 56 | 0.002101 | 0.004298 | 1.000 | ✓ |
| eps_mse | full_frame | V10 | fake_70k | 56 | 0.002101 | 0.004206 | 1.000 | ✓ |
| eps_mse | full_frame | V10 | fake_100k | 56 | 0.002101 | 0.003962 | 1.000 | ✓ |
| eps_mse | full_frame | V10 | shuffled_E | 56 | 0.002101 | 0.003745 | 1.000 | ✓ |
| eps_mse | full_frame | V10 | cross_session_E | 56 | 0.002101 | 0.003742 | 1.000 | ✓ |
| eps_mae | E2_seg | D2 | fake_5k | 50 | 0.05482 | 0.04631 | 0.033 |  |
| eps_mae | E2_seg | D2 | fake_25k | 50 | 0.05482 | 0.04207 | 0.010 |  |
| eps_mae | E2_seg | D2 | fake_70k | 50 | 0.05482 | 0.04068 | 0.008 |  |
| eps_mae | E2_seg | D2 | fake_100k | 50 | 0.05482 | 0.0402 | 0.007 |  |
| eps_mae | E2_seg | D2 | shuffled_E | 50 | 0.05482 | 0.06269 | 0.959 | ✓ |
| eps_mae | E2_seg | D2 | cross_session_E | 50 | 0.05482 | 0.06273 | 0.952 | ✓ |
| eps_mae | E2_seg | V10 | fake_5k | 56 | 0.06347 | 0.0558 | 0.192 |  |
| eps_mae | E2_seg | V10 | fake_25k | 56 | 0.06347 | 0.0542 | 0.212 |  |
| eps_mae | E2_seg | V10 | fake_70k | 56 | 0.06347 | 0.05463 | 0.239 |  |
| eps_mae | E2_seg | V10 | fake_100k | 56 | 0.06347 | 0.05383 | 0.210 |  |
| eps_mae | E2_seg | V10 | shuffled_E | 56 | 0.06347 | 0.0712 | 0.907 | ✓ |
| eps_mae | E2_seg | V10 | cross_session_E | 56 | 0.06347 | 0.0713 | 0.904 | ✓ |
| eps_mae | E3_seg | D2 | fake_5k | 50 | 0.02264 | 0.03983 | 1.000 | ✓ |
| eps_mae | E3_seg | D2 | fake_25k | 50 | 0.02264 | 0.03534 | 1.000 | ✓ |
| eps_mae | E3_seg | D2 | fake_70k | 50 | 0.02264 | 0.03435 | 1.000 | ✓ |
| eps_mae | E3_seg | D2 | fake_100k | 50 | 0.02264 | 0.03323 | 1.000 | ✓ |
| eps_mae | E3_seg | D2 | shuffled_E | 50 | 0.02264 | 0.03581 | 1.000 | ✓ |
| eps_mae | E3_seg | D2 | cross_session_E | 50 | 0.02264 | 0.03582 | 1.000 | ✓ |
| eps_mae | E3_seg | V10 | fake_5k | 56 | 0.0233 | 0.04207 | 1.000 | ✓ |
| eps_mae | E3_seg | V10 | fake_25k | 56 | 0.0233 | 0.03836 | 1.000 | ✓ |
| eps_mae | E3_seg | V10 | fake_70k | 56 | 0.0233 | 0.0378 | 1.000 | ✓ |
| eps_mae | E3_seg | V10 | fake_100k | 56 | 0.0233 | 0.03684 | 1.000 | ✓ |
| eps_mae | E3_seg | V10 | shuffled_E | 56 | 0.0233 | 0.03675 | 1.000 | ✓ |
| eps_mae | E3_seg | V10 | cross_session_E | 56 | 0.0233 | 0.03676 | 1.000 | ✓ |
| eps_mae | full_frame | D2 | fake_5k | 50 | 0.02421 | 0.04016 | 1.000 | ✓ |
| eps_mae | full_frame | D2 | fake_25k | 50 | 0.02421 | 0.03568 | 1.000 | ✓ |
| eps_mae | full_frame | D2 | fake_70k | 50 | 0.02421 | 0.03466 | 1.000 | ✓ |
| eps_mae | full_frame | D2 | fake_100k | 50 | 0.02421 | 0.03357 | 1.000 | ✓ |
| eps_mae | full_frame | D2 | shuffled_E | 50 | 0.02421 | 0.03713 | 1.000 | ✓ |
| eps_mae | full_frame | D2 | cross_session_E | 50 | 0.02421 | 0.03713 | 1.000 | ✓ |
| eps_mae | full_frame | V10 | fake_5k | 56 | 0.02564 | 0.04286 | 1.000 | ✓ |
| eps_mae | full_frame | V10 | fake_25k | 56 | 0.02564 | 0.03927 | 1.000 | ✓ |
| eps_mae | full_frame | V10 | fake_70k | 56 | 0.02564 | 0.03876 | 1.000 | ✓ |
| eps_mae | full_frame | V10 | fake_100k | 56 | 0.02564 | 0.0378 | 1.000 | ✓ |
| eps_mae | full_frame | V10 | shuffled_E | 56 | 0.02564 | 0.03876 | 1.000 | ✓ |
| eps_mae | full_frame | V10 | cross_session_E | 56 | 0.02564 | 0.03877 | 1.000 | ✓ |
| signed_residual | E2_seg | D2 | fake_5k | 50 | 0.000705 | 0.002551 | 0.701 |  |
| signed_residual | E2_seg | D2 | fake_25k | 50 | 0.000705 | 0.00306 | 0.755 |  |
| signed_residual | E2_seg | D2 | fake_70k | 50 | 0.000705 | 0.002375 | 0.680 |  |
| signed_residual | E2_seg | D2 | fake_100k | 50 | 0.000705 | 0.002345 | 0.672 |  |
| signed_residual | E2_seg | D2 | shuffled_E | 50 | 0.000705 | 0.005577 | 0.913 | ✓ |
| signed_residual | E2_seg | D2 | cross_session_E | 50 | 0.000705 | 0.005813 | 0.938 | ✓ |
| signed_residual | E2_seg | V10 | fake_5k | 56 | 0.0005273 | 0.001639 | 0.719 |  |
| signed_residual | E2_seg | V10 | fake_25k | 56 | 0.0005273 | 0.002394 | 0.717 |  |
| signed_residual | E2_seg | V10 | fake_70k | 56 | 0.0005273 | 0.000666 | 0.609 |  |
| signed_residual | E2_seg | V10 | fake_100k | 56 | 0.0005273 | 0.0009237 | 0.611 |  |
| signed_residual | E2_seg | V10 | shuffled_E | 56 | 0.0005273 | 0.005902 | 0.974 | ✓ |
| signed_residual | E2_seg | V10 | cross_session_E | 56 | 0.0005273 | 0.006566 | 0.996 | ✓ |
| signed_residual | E3_seg | D2 | fake_5k | 50 | -0.0001962 | 0.0003205 | 0.975 | ✓ |
| signed_residual | E3_seg | D2 | fake_25k | 50 | -0.0001962 | 0.0002103 | 0.948 | ✓ |
| signed_residual | E3_seg | D2 | fake_70k | 50 | -0.0001962 | 0.0002109 | 0.959 | ✓ |
| signed_residual | E3_seg | D2 | fake_100k | 50 | -0.0001962 | 0.0001821 | 0.944 | ✓ |
| signed_residual | E3_seg | D2 | shuffled_E | 50 | -0.0001962 | -0.0001969 | 0.513 |  |
| signed_residual | E3_seg | D2 | cross_session_E | 50 | -0.0001962 | -0.0002069 | 0.492 |  |
| signed_residual | E3_seg | V10 | fake_5k | 56 | -0.0001222 | 0.0004398 | 0.962 | ✓ |
| signed_residual | E3_seg | V10 | fake_25k | 56 | -0.0001222 | 0.0003293 | 0.951 | ✓ |
| signed_residual | E3_seg | V10 | fake_70k | 56 | -0.0001222 | 0.0004163 | 0.977 | ✓ |
| signed_residual | E3_seg | V10 | fake_100k | 56 | -0.0001222 | 0.0003793 | 0.969 | ✓ |
| signed_residual | E3_seg | V10 | shuffled_E | 56 | -0.0001222 | -0.0001662 | 0.442 |  |
| signed_residual | E3_seg | V10 | cross_session_E | 56 | -0.0001222 | -0.0001899 | 0.423 |  |
| signed_residual | full_frame | D2 | fake_5k | 50 | -0.0001514 | 0.0004388 | 0.992 | ✓ |
| signed_residual | full_frame | D2 | fake_25k | 50 | -0.0001514 | 0.0003531 | 0.985 | ✓ |
| signed_residual | full_frame | D2 | fake_70k | 50 | -0.0001514 | 0.0003109 | 0.978 | ✓ |
| signed_residual | full_frame | D2 | fake_100k | 50 | -0.0001514 | 0.000286 | 0.970 | ✓ |
| signed_residual | full_frame | D2 | shuffled_E | 50 | -0.0001514 | 9.177e-05 | 0.836 |  |
| signed_residual | full_frame | D2 | cross_session_E | 50 | -0.0001514 | 8.827e-05 | 0.833 |  |
| signed_residual | full_frame | V10 | fake_5k | 56 | -8.365e-05 | 0.0005334 | 0.998 | ✓ |
| signed_residual | full_frame | V10 | fake_25k | 56 | -8.365e-05 | 0.0004623 | 0.996 | ✓ |
| signed_residual | full_frame | V10 | fake_70k | 56 | -8.365e-05 | 0.0004468 | 0.995 | ✓ |
| signed_residual | full_frame | V10 | fake_100k | 56 | -8.365e-05 | 0.0004277 | 0.993 | ✓ |
| signed_residual | full_frame | V10 | shuffled_E | 56 | -8.365e-05 | 0.0001775 | 0.871 | ✓ |
| signed_residual | full_frame | V10 | cross_session_E | 56 | -8.365e-05 | 0.0001976 | 0.890 | ✓ |
| low_freq_mse | E2_seg | D2 | fake_5k | 50 | 0.0007294 | 0.001402 | 1.000 | ✓ |
| low_freq_mse | E2_seg | D2 | fake_25k | 50 | 0.0007294 | 0.001159 | 1.000 | ✓ |
| low_freq_mse | E2_seg | D2 | fake_70k | 50 | 0.0007294 | 0.001133 | 0.986 | ✓ |
| low_freq_mse | E2_seg | D2 | fake_100k | 50 | 0.0007294 | 0.001161 | 0.994 | ✓ |
| low_freq_mse | E2_seg | D2 | shuffled_E | 50 | 0.0007294 | 0.001401 | 1.000 | ✓ |
| low_freq_mse | E2_seg | D2 | cross_session_E | 50 | 0.0007294 | 0.001399 | 1.000 | ✓ |
| low_freq_mse | E2_seg | V10 | fake_5k | 56 | 0.0007659 | 0.001332 | 1.000 | ✓ |
| low_freq_mse | E2_seg | V10 | fake_25k | 56 | 0.0007659 | 0.001193 | 0.996 | ✓ |
| low_freq_mse | E2_seg | V10 | fake_70k | 56 | 0.0007659 | 0.001212 | 0.992 | ✓ |
| low_freq_mse | E2_seg | V10 | fake_100k | 56 | 0.0007659 | 0.001205 | 0.999 | ✓ |
| low_freq_mse | E2_seg | V10 | shuffled_E | 56 | 0.0007659 | 0.001407 | 1.000 | ✓ |
| low_freq_mse | E2_seg | V10 | cross_session_E | 56 | 0.0007659 | 0.001427 | 1.000 | ✓ |
| low_freq_mse | E3_seg | D2 | fake_5k | 50 | 0.0002378 | 0.0011 | 1.000 | ✓ |
| low_freq_mse | E3_seg | D2 | fake_25k | 50 | 0.0002378 | 0.0009102 | 1.000 | ✓ |
| low_freq_mse | E3_seg | D2 | fake_70k | 50 | 0.0002378 | 0.0008974 | 1.000 | ✓ |
| low_freq_mse | E3_seg | D2 | fake_100k | 50 | 0.0002378 | 0.0008841 | 1.000 | ✓ |
| low_freq_mse | E3_seg | D2 | shuffled_E | 50 | 0.0002378 | 0.001369 | 1.000 | ✓ |
| low_freq_mse | E3_seg | D2 | cross_session_E | 50 | 0.0002378 | 0.001368 | 1.000 | ✓ |
| low_freq_mse | E3_seg | V10 | fake_5k | 56 | 0.0002435 | 0.001168 | 1.000 | ✓ |
| low_freq_mse | E3_seg | V10 | fake_25k | 56 | 0.0002435 | 0.001004 | 1.000 | ✓ |
| low_freq_mse | E3_seg | V10 | fake_70k | 56 | 0.0002435 | 0.0009965 | 1.000 | ✓ |
| low_freq_mse | E3_seg | V10 | fake_100k | 56 | 0.0002435 | 0.0009863 | 1.000 | ✓ |
| low_freq_mse | E3_seg | V10 | shuffled_E | 56 | 0.0002435 | 0.001413 | 1.000 | ✓ |
| low_freq_mse | E3_seg | V10 | cross_session_E | 56 | 0.0002435 | 0.00141 | 1.000 | ✓ |
| low_freq_mse | full_frame | D2 | fake_5k | 50 | 0.0002616 | 0.001113 | 1.000 | ✓ |
| low_freq_mse | full_frame | D2 | fake_25k | 50 | 0.0002616 | 0.0009208 | 1.000 | ✓ |
| low_freq_mse | full_frame | D2 | fake_70k | 50 | 0.0002616 | 0.0009074 | 1.000 | ✓ |
| low_freq_mse | full_frame | D2 | fake_100k | 50 | 0.0002616 | 0.0008962 | 1.000 | ✓ |
| low_freq_mse | full_frame | D2 | shuffled_E | 50 | 0.0002616 | 0.001371 | 1.000 | ✓ |
| low_freq_mse | full_frame | D2 | cross_session_E | 50 | 0.0002616 | 0.00137 | 1.000 | ✓ |
| low_freq_mse | full_frame | V10 | fake_5k | 56 | 0.0002738 | 0.001177 | 1.000 | ✓ |
| low_freq_mse | full_frame | V10 | fake_25k | 56 | 0.0002738 | 0.001014 | 1.000 | ✓ |
| low_freq_mse | full_frame | V10 | fake_70k | 56 | 0.0002738 | 0.001008 | 1.000 | ✓ |
| low_freq_mse | full_frame | V10 | fake_100k | 56 | 0.0002738 | 0.0009978 | 1.000 | ✓ |
| low_freq_mse | full_frame | V10 | shuffled_E | 56 | 0.0002738 | 0.001413 | 1.000 | ✓ |
| low_freq_mse | full_frame | V10 | cross_session_E | 56 | 0.0002738 | 0.00141 | 1.000 | ✓ |
| high_freq_mse | E2_seg | D2 | fake_5k | 50 | 0.005322 | 0.003072 | 0.018 |  |
| high_freq_mse | E2_seg | D2 | fake_25k | 50 | 0.005322 | 0.002557 | 0.006 |  |
| high_freq_mse | E2_seg | D2 | fake_70k | 50 | 0.005322 | 0.00226 | 0.002 |  |
| high_freq_mse | E2_seg | D2 | fake_100k | 50 | 0.005322 | 0.002085 | 0.001 |  |
| high_freq_mse | E2_seg | D2 | shuffled_E | 50 | 0.005322 | 0.005615 | 0.622 |  |
| high_freq_mse | E2_seg | D2 | cross_session_E | 50 | 0.005322 | 0.005617 | 0.618 |  |
| high_freq_mse | E2_seg | V10 | fake_5k | 56 | 0.007409 | 0.005079 | 0.143 |  |
| high_freq_mse | E2_seg | V10 | fake_25k | 56 | 0.007409 | 0.00514 | 0.208 |  |
| high_freq_mse | E2_seg | V10 | fake_70k | 56 | 0.007409 | 0.005319 | 0.230 |  |
| high_freq_mse | E2_seg | V10 | fake_100k | 56 | 0.007409 | 0.005094 | 0.195 |  |
| high_freq_mse | E2_seg | V10 | shuffled_E | 56 | 0.007409 | 0.007747 | 0.576 |  |
| high_freq_mse | E2_seg | V10 | cross_session_E | 56 | 0.007409 | 0.007735 | 0.577 |  |
| high_freq_mse | E3_seg | D2 | fake_5k | 50 | 0.001104 | 0.002712 | 1.000 | ✓ |
| high_freq_mse | E3_seg | D2 | fake_25k | 50 | 0.001104 | 0.00214 | 1.000 | ✓ |
| high_freq_mse | E3_seg | D2 | fake_70k | 50 | 0.001104 | 0.001942 | 1.000 | ✓ |
| high_freq_mse | E3_seg | D2 | fake_100k | 50 | 0.001104 | 0.001708 | 1.000 | ✓ |
| high_freq_mse | E3_seg | D2 | shuffled_E | 50 | 0.001104 | 0.001299 | 0.986 | ✓ |
| high_freq_mse | E3_seg | D2 | cross_session_E | 50 | 0.001104 | 0.001298 | 0.987 | ✓ |
| high_freq_mse | E3_seg | V10 | fake_5k | 56 | 0.001158 | 0.003022 | 1.000 | ✓ |
| high_freq_mse | E3_seg | V10 | fake_25k | 56 | 0.001158 | 0.002603 | 1.000 | ✓ |
| high_freq_mse | E3_seg | V10 | fake_70k | 56 | 0.001158 | 0.002499 | 1.000 | ✓ |
| high_freq_mse | E3_seg | V10 | fake_100k | 56 | 0.001158 | 0.002273 | 1.000 | ✓ |
| high_freq_mse | E3_seg | V10 | shuffled_E | 56 | 0.001158 | 0.00135 | 0.996 | ✓ |
| high_freq_mse | E3_seg | V10 | cross_session_E | 56 | 0.001158 | 0.001352 | 0.996 | ✓ |
| high_freq_mse | full_frame | D2 | fake_5k | 50 | 0.001308 | 0.002735 | 1.000 | ✓ |
| high_freq_mse | full_frame | D2 | fake_25k | 50 | 0.001308 | 0.002165 | 1.000 | ✓ |
| high_freq_mse | full_frame | D2 | fake_70k | 50 | 0.001308 | 0.00196 | 1.000 | ✓ |
| high_freq_mse | full_frame | D2 | fake_100k | 50 | 0.001308 | 0.00173 | 1.000 | ✓ |
| high_freq_mse | full_frame | D2 | shuffled_E | 50 | 0.001308 | 0.001508 | 0.935 | ✓ |
| high_freq_mse | full_frame | D2 | cross_session_E | 50 | 0.001308 | 0.001508 | 0.931 | ✓ |
| high_freq_mse | full_frame | V10 | fake_5k | 56 | 0.001523 | 0.003143 | 1.000 | ✓ |
| high_freq_mse | full_frame | V10 | fake_25k | 56 | 0.001523 | 0.00275 | 1.000 | ✓ |
| high_freq_mse | full_frame | V10 | fake_70k | 56 | 0.001523 | 0.00266 | 1.000 | ✓ |
| high_freq_mse | full_frame | V10 | fake_100k | 56 | 0.001523 | 0.002432 | 1.000 | ✓ |
| high_freq_mse | full_frame | V10 | shuffled_E | 56 | 0.001523 | 0.001723 | 0.863 | ✓ |
| high_freq_mse | full_frame | V10 | cross_session_E | 56 | 0.001523 | 0.001724 | 0.864 | ✓ |
| vgg_distance | E2_seg | D2 | fake_5k | 50 | 0 | 58.16 | 1.000 | ✓ |
| vgg_distance | E2_seg | D2 | fake_25k | 50 | 0 | 56.48 | 1.000 | ✓ |
| vgg_distance | E2_seg | D2 | fake_70k | 50 | 0 | 55.14 | 1.000 | ✓ |
| vgg_distance | E2_seg | D2 | fake_100k | 50 | 0 | 55.11 | 1.000 | ✓ |
| vgg_distance | E2_seg | D2 | shuffled_E | 50 | 0 | 38.47 | 1.000 | ✓ |
| vgg_distance | E2_seg | D2 | cross_session_E | 50 | 0 | 38.44 | 1.000 | ✓ |
| vgg_distance | E2_seg | V10 | fake_5k | 56 | 0 | 60.44 | 1.000 | ✓ |
| vgg_distance | E2_seg | V10 | fake_25k | 56 | 0 | 59.83 | 1.000 | ✓ |
| vgg_distance | E2_seg | V10 | fake_70k | 56 | 0 | 58.94 | 1.000 | ✓ |
| vgg_distance | E2_seg | V10 | fake_100k | 56 | 0 | 58.96 | 1.000 | ✓ |
| vgg_distance | E2_seg | V10 | shuffled_E | 56 | 0 | 36.28 | 1.000 | ✓ |
| vgg_distance | E2_seg | V10 | cross_session_E | 56 | 0 | 36.28 | 1.000 | ✓ |
| vgg_distance | E3_seg | D2 | fake_5k | 50 | 0 | 47.5 | 1.000 | ✓ |
| vgg_distance | E3_seg | D2 | fake_25k | 50 | 0 | 45.49 | 1.000 | ✓ |
| vgg_distance | E3_seg | D2 | fake_70k | 50 | 0 | 43.83 | 1.000 | ✓ |
| vgg_distance | E3_seg | D2 | fake_100k | 50 | 0 | 42.97 | 1.000 | ✓ |
| vgg_distance | E3_seg | D2 | shuffled_E | 50 | 0 | 26.54 | 1.000 | ✓ |
| vgg_distance | E3_seg | D2 | cross_session_E | 50 | 0 | 26.54 | 1.000 | ✓ |
| vgg_distance | E3_seg | V10 | fake_5k | 56 | 0 | 48.71 | 1.000 | ✓ |
| vgg_distance | E3_seg | V10 | fake_25k | 56 | 0 | 46.77 | 1.000 | ✓ |
| vgg_distance | E3_seg | V10 | fake_70k | 56 | 0 | 45.32 | 1.000 | ✓ |
| vgg_distance | E3_seg | V10 | fake_100k | 56 | 0 | 44.57 | 1.000 | ✓ |
| vgg_distance | E3_seg | V10 | shuffled_E | 56 | 0 | 26.58 | 1.000 | ✓ |
| vgg_distance | E3_seg | V10 | cross_session_E | 56 | 0 | 26.62 | 1.000 | ✓ |
| vgg_distance | full_frame | D2 | fake_5k | 50 | 0 | 48.01 | 1.000 | ✓ |
| vgg_distance | full_frame | D2 | fake_25k | 50 | 0 | 46.01 | 1.000 | ✓ |
| vgg_distance | full_frame | D2 | fake_70k | 50 | 0 | 44.37 | 1.000 | ✓ |
| vgg_distance | full_frame | D2 | fake_100k | 50 | 0 | 43.54 | 1.000 | ✓ |
| vgg_distance | full_frame | D2 | shuffled_E | 50 | 0 | 27.11 | 1.000 | ✓ |
| vgg_distance | full_frame | D2 | cross_session_E | 50 | 0 | 27.12 | 1.000 | ✓ |
| vgg_distance | full_frame | V10 | fake_5k | 56 | 0 | 49.39 | 1.000 | ✓ |
| vgg_distance | full_frame | V10 | fake_25k | 56 | 0 | 47.51 | 1.000 | ✓ |
| vgg_distance | full_frame | V10 | fake_70k | 56 | 0 | 46.09 | 1.000 | ✓ |
| vgg_distance | full_frame | V10 | fake_100k | 56 | 0 | 45.39 | 1.000 | ✓ |
| vgg_distance | full_frame | V10 | shuffled_E | 56 | 0 | 27.13 | 1.000 | ✓ |
| vgg_distance | full_frame | V10 | cross_session_E | 56 | 0 | 27.16 | 1.000 | ✓ |
| b1_distance | E2_seg | D2 | fake_5k | 50 | 0 | 34.37 | 1.000 | ✓ |
| b1_distance | E2_seg | D2 | fake_25k | 50 | 0 | 34.17 | 1.000 | ✓ |
| b1_distance | E2_seg | D2 | fake_70k | 50 | 0 | 31.29 | 1.000 | ✓ |
| b1_distance | E2_seg | D2 | fake_100k | 50 | 0 | 29.92 | 1.000 | ✓ |
| b1_distance | E2_seg | D2 | shuffled_E | 50 | 0 | 30.28 | 1.000 | ✓ |
| b1_distance | E2_seg | D2 | cross_session_E | 50 | 0 | 30.11 | 1.000 | ✓ |
| b1_distance | E2_seg | V10 | fake_5k | 56 | 0 | 31.31 | 1.000 | ✓ |
| b1_distance | E2_seg | V10 | fake_25k | 56 | 0 | 31.19 | 1.000 | ✓ |
| b1_distance | E2_seg | V10 | fake_70k | 56 | 0 | 30.02 | 1.000 | ✓ |
| b1_distance | E2_seg | V10 | fake_100k | 56 | 0 | 29.65 | 1.000 | ✓ |
| b1_distance | E2_seg | V10 | shuffled_E | 56 | 0 | 27.75 | 1.000 | ✓ |
| b1_distance | E2_seg | V10 | cross_session_E | 56 | 0 | 28.23 | 1.000 | ✓ |
| b1_distance | E3_seg | D2 | fake_5k | 50 | 0 | 41.33 | 1.000 | ✓ |
| b1_distance | E3_seg | D2 | fake_25k | 50 | 0 | 39.9 | 1.000 | ✓ |
| b1_distance | E3_seg | D2 | fake_70k | 50 | 0 | 37.65 | 1.000 | ✓ |
| b1_distance | E3_seg | D2 | fake_100k | 50 | 0 | 34.48 | 1.000 | ✓ |
| b1_distance | E3_seg | D2 | shuffled_E | 50 | 0 | 21.16 | 1.000 | ✓ |
| b1_distance | E3_seg | D2 | cross_session_E | 50 | 0 | 21.08 | 1.000 | ✓ |
| b1_distance | E3_seg | V10 | fake_5k | 56 | 0 | 41.79 | 1.000 | ✓ |
| b1_distance | E3_seg | V10 | fake_25k | 56 | 0 | 40.99 | 1.000 | ✓ |
| b1_distance | E3_seg | V10 | fake_70k | 56 | 0 | 39.68 | 1.000 | ✓ |
| b1_distance | E3_seg | V10 | fake_100k | 56 | 0 | 37.68 | 1.000 | ✓ |
| b1_distance | E3_seg | V10 | shuffled_E | 56 | 0 | 21.07 | 1.000 | ✓ |
| b1_distance | E3_seg | V10 | cross_session_E | 56 | 0 | 20.98 | 1.000 | ✓ |
| b1_distance | full_frame | D2 | fake_5k | 50 | 0 | 40.97 | 1.000 | ✓ |
| b1_distance | full_frame | D2 | fake_25k | 50 | 0 | 39.6 | 1.000 | ✓ |
| b1_distance | full_frame | D2 | fake_70k | 50 | 0 | 37.32 | 1.000 | ✓ |
| b1_distance | full_frame | D2 | fake_100k | 50 | 0 | 34.24 | 1.000 | ✓ |
| b1_distance | full_frame | D2 | shuffled_E | 50 | 0 | 21.59 | 1.000 | ✓ |
| b1_distance | full_frame | D2 | cross_session_E | 50 | 0 | 21.51 | 1.000 | ✓ |
| b1_distance | full_frame | V10 | fake_5k | 56 | 0 | 41.18 | 1.000 | ✓ |
| b1_distance | full_frame | V10 | fake_25k | 56 | 0 | 40.41 | 1.000 | ✓ |
| b1_distance | full_frame | V10 | fake_70k | 56 | 0 | 39.11 | 1.000 | ✓ |
| b1_distance | full_frame | V10 | fake_100k | 56 | 0 | 37.21 | 1.000 | ✓ |
| b1_distance | full_frame | V10 | shuffled_E | 56 | 0 | 21.44 | 1.000 | ✓ |
| b1_distance | full_frame | V10 | cross_session_E | 56 | 0 | 21.38 | 1.000 | ✓ |