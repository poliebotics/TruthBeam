# Phase H E-usage ablation — sub-diag 6.4

Generated: 2026-05-04T00:46:23Z
Ckpt: `/path/to/poliebotics_phase_b/experiments/phase_h_supervised_baseline/checkpoints/step_00025000.pt`

## TL;DR

**✅ PASS**

PHASE H USES E (PASS). target/(zero or shuffled) is distinguishable at AUROC ≥ 0.70 in both sessions. Phase H actually consumes the E channels for its decisions. Continue queue uninterrupted.

## Per-session metrics

| session | mean target | mean zero | mean shuffled | mean source | AUROC(t v zero) | AUROC(t v shuf) | AUROC(t v src) |
|---|---|---|---|---|---|---|---|
| D2 | +4.897 | +35.565 | -13.674 | -13.675 | 0.0000 | 1.0000 | 1.0000 |
| V10 | +5.198 | +35.750 | -14.731 | -13.947 | 0.0000 | 1.0000 | 1.0000 |

## Per-frame deltas (target − comparison, mean)

| session | Δ(target − zero) | Δ(target − shuffled) | Δ(target − source) | frac t > zero | frac t > shuf | frac t > src |
|---|---|---|---|---|---|---|
| D2 | -30.6677 | +18.5715 | +18.5728 | 0.000 | 1.000 | 1.000 |
| V10 | -30.5517 | +19.9289 | +19.1452 | 0.000 | 1.000 | 1.000 |

## Interpretation

- **AUROC(target vs zero) high**: Phase H discriminates `(C, E_target)` from `(C, E_zero)`
  → Phase H actually uses E. Without E, classifier confidence drops.
- **AUROC(target vs zero) ≈ 0.5**: Phase H ignores E entirely. C-only verifier.
  Means Phase H's perfect AUROC on shuffled-pair training came from C-side features.
- **AUROC(target vs source) interesting if ~ AUROC(target vs zero)**: Phase H either
  uses E broadly (no specific frame-pair binding), or treats source-E as noise. Hard
  to distinguish without further probes.

## Manuscript framing (operator round 5, 2026-05-04)

Use this language when discussing this result going forward:

> **The supervised CNN baseline is E-aware, but its learned evidence is coarse.
> It separates target emissions from substantially different emissions, but
> does not resolve fine XOF-level perturbations in this training regime.**

Working summary: "Phase H uses E for coarse, high-separation contrasts, but
does not learn fine-grained XOF-structure sensitivity."

DO NOT use: "Phase H uses E coarsely but not granularly."

The distinction is **contrast scale / rendered-E difference / perturbation
family** — not abstract granularity. Phase H's gross-mismatch sensitivity
(target vs shuffled, target vs source: AUROC=1.000) is independent of its
fine-XOF blindness (xof_t1/t3 AUROC≈0.50 from Phase H eval), and the latter
reflects the perturbation family + label-locking diversity limitation in
the current training regime, not a granularity ceiling per se.

The `target_vs_zero` AUROC=0.0 (zero-E mean logit +35.6 vs target +4.9)
is an OOD artifact: zero-E was never observed during training. Conclusion
should rest on the in-distribution contrasts (`target vs shuffled`,
`target vs source`), both at 1.000 in both sessions.
