# EXP 7 — Synthetic counterfactual-E perturbation sensitivity

**This experiment is NOT XOF bit-flip sensitivity, NOT optical washout floor**. It is synthetic Gaussian-noise perturbation of the rendered E field at three preprocessing scales (rendered, Phase G input, binder input).

3 modes × 3 σ × ~120 frames. Output dir name `xof_sensitivity` retained for code compatibility; report uses the synthetic-counterfactual-E label throughout.

## Δscore as function of achieved ΔE_rms

| session | mode | σ_target | n | median ΔE_rms | median Δscore | bootstrap CI |
|---|---|---|---|---|---|---|
| D2 | A_rendered | 0.1 | 60 | 0.0484 | +0.00006 | [+0.00006, +0.00006] |
| D2 | A_rendered | 0.3 | 60 | 0.1406 | +0.00040 | [+0.00039, +0.00041] |
| D2 | A_rendered | 0.5 | 60 | 0.2199 | +0.00092 | [+0.00091, +0.00093] |
| D2 | B_phase_g_input | 0.1 | 60 | 0.0978 | +0.00022 | [+0.00021, +0.00022] |
| D2 | B_phase_g_input | 0.3 | 60 | 0.2578 | +0.00127 | [+0.00125, +0.00128] |
| D2 | B_phase_g_input | 0.5 | 60 | 0.3515 | +0.00226 | [+0.00224, +0.00229] |
| D2 | C_binder_resolution | 0.1 | 60 | 0.0539 | +0.00007 | [+0.00007, +0.00007] |
| D2 | C_binder_resolution | 0.3 | 60 | 0.1555 | +0.00049 | [+0.00048, +0.00050] |
| D2 | C_binder_resolution | 0.5 | 60 | 0.2398 | +0.00109 | [+0.00109, +0.00111] |
| V10 | A_rendered | 0.1 | 60 | 0.0484 | +0.00006 | [+0.00006, +0.00006] |
| V10 | A_rendered | 0.3 | 60 | 0.1406 | +0.00042 | [+0.00041, +0.00042] |
| V10 | A_rendered | 0.5 | 60 | 0.2200 | +0.00096 | [+0.00095, +0.00097] |
| V10 | B_phase_g_input | 0.1 | 60 | 0.0978 | +0.00022 | [+0.00022, +0.00023] |
| V10 | B_phase_g_input | 0.3 | 60 | 0.2578 | +0.00132 | [+0.00130, +0.00132] |
| V10 | B_phase_g_input | 0.5 | 60 | 0.3515 | +0.00232 | [+0.00230, +0.00233] |
| V10 | C_binder_resolution | 0.1 | 60 | 0.0539 | +0.00007 | [+0.00007, +0.00008] |
| V10 | C_binder_resolution | 0.3 | 60 | 0.1555 | +0.00051 | [+0.00050, +0.00051] |
| V10 | C_binder_resolution | 0.5 | 60 | 0.2398 | +0.00113 | [+0.00112, +0.00114] |

## Monotonicity check (Δscore vs σ, per session × mode)

Pre-registered: monotonic non-decreasing Δscore as σ increases = sensitivity well-characterized. Non-monotonic = FLAG 🔧.

| session | mode | curve (σ → Δscore median) | monotonic? |
|---|---|---|---|
| D2 | A_rendered | σ=0.1 Δ=+0.0001 → σ=0.3 Δ=+0.0004 → σ=0.5 Δ=+0.0009 | ✓ |
| D2 | B_phase_g_input | σ=0.1 Δ=+0.0002 → σ=0.3 Δ=+0.0013 → σ=0.5 Δ=+0.0023 | ✓ |
| D2 | C_binder_resolution | σ=0.1 Δ=+0.0001 → σ=0.3 Δ=+0.0005 → σ=0.5 Δ=+0.0011 | ✓ |
| V10 | A_rendered | σ=0.1 Δ=+0.0001 → σ=0.3 Δ=+0.0004 → σ=0.5 Δ=+0.0010 | ✓ |
| V10 | B_phase_g_input | σ=0.1 Δ=+0.0002 → σ=0.3 Δ=+0.0013 → σ=0.5 Δ=+0.0023 | ✓ |
| V10 | C_binder_resolution | σ=0.1 Δ=+0.0001 → σ=0.3 Δ=+0.0005 → σ=0.5 Δ=+0.0011 | ✓ |

**All curves monotonic non-decreasing** — sensitivity well-characterized.