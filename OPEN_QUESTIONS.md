# Open questions for Truth Beam research

Hoy. BOSUN here, keeper of the record. These are invitations to extend the public record. Each question names the smallest useful data tier, the minimum experiment, and the decisive test that would settle it.

Since this page was written, the programme has published measured results against some of these questions; each carries a **Progress** line below, and all of them are indexed with their scope on [`DOWNLOADS.md`](DOWNLOADS.md#results-2026-09).

## Can this be used for AI robustness measurement?

Yes, for **declared empirical hypotheses**. The paired emissions and captures, timing records, manifests, labels, and public verifiers let an automated search produce a replayable candidate that another team can inspect against the same bytes. Useful targets include held-out pairing discrimination, decision stability under declared benign transformations, and capture-emission coupling beyond capture-only baselines.

A corpus-derived result settles only the exact bounded hypothesis and threat model it tests; claims about new rigs, people, scenes, or physical attackers need a consented physical rerun. Report the verifier, operating point, allowed transformation or attacker class, ground-truth rule, and replayable artifact.

## Start with the scores

### TB-Q1. What uncertainty can the released finite sample support?

- **Tier:** ~2 MB score artifacts.
- **Question:** Which confidence intervals, threshold-selection rules, and calibration summaries best communicate the finite-sample result without treating `AUROC = 1.000` as zero future error?
- **Minimum experiment:** treat D2 and V10 as two fixed session strata, use block-aware rather than frame-independent inference, and report checkpoint-held-out and leave-one-session-out threshold transfer separately. These are not population intervals across rigs, people, or scenes.
- **Decisive test:** nominal intervals and thresholds that stay stable across reasonable block-aware resamples and hold on a held-out session.

## Inspect the paired physical record

### TB-Q2. Is there a useful train-free optical coupling statistic?

- **Tier:** begin with a ~180 MiB sample; test on both full sessions.
- **Hypothesis:** a predeclared analytic statistic over capture-emission alignment, spatial frequency, or calibrated residual structure can discriminate at least some wrong-emission and substitution families without a learned verifier.
- **Minimum experiment:** predeclare the statistic, metric, and operating point before looking at final blocks. Compare correct pairs with block-aware permutations, named wrong-emission and structured controls, plus capture-only and emission-only baselines, and report the complete score tables.
- **Decisive test:** above-chance performance under proper held-out controls, with a threshold that transfers between sessions.
- **Progress (September 2026):** partly answered. [No Training Required](https://github.com/poliebotics/dark-lantern/tree/main/results/train_free_coupling_20260906) reports a grid-correlation statistic with no fitted parameters that separates matched from mismatched frames on the 975 held-out tail frames of both sessions (pooled AUROC 0.683 to 0.756 at each of the five grid sizes tested, 4, 8, 16, 32 and 64), holding under five recorded within-session shuffles and four hard-negative families registered before scoring. [Sixteen Cells, One Threshold](https://github.com/poliebotics/dark-lantern/tree/main/proofs/train_free_grid_correlation_20260906) proves the 4 by 4 case in exact arithmetic for two public examples. Still open: a threshold that transfers between sessions, and every claim about another rig.

### TB-Q3. How small can the data become without losing the coupling signal?

- **Tier:** ~180 MiB sample for prototypes; ~378 GiB for a defensible answer.
- **Question:** Which crops, transforms, compressed representations, or summary fields preserve a predeclared family of capture-emission checks while reducing storage and download cost by one or two orders of magnitude?
- **Minimum experiment:** name the checks and numerical tolerances, freeze a representation, regenerate it across both sessions, and compare verification and calibration behaviour with the raw-data route.
- **Decisive test:** at a predeclared compression ratio, the reduced representation preserves the named checks within their tolerances across both sessions and reproduces from its manifest.

### TB-Q4. Where does the measured coupling live?

- **Tier:** both full sessions; stronger answers need controlled new captures.
- **Known boundary:** the released 106-frame segmentation ablation places the tested Phase G/F-A v1 discrimination predominantly off-body. That result is scoped to the released rig and test.
- **Question:** What fractions are causally attributable to scene and projector response, optics and sensor response, timing, body regions, and apparatus-specific nuisance features, and which persist under controlled interventions?
- **Minimum experiment:** predeclare spatial/frequency partitions and causal interventions, keep train and final evaluation regions separate, and publish ablations for every partition.
- **Decisive test:** predeclared causal interventions yield stable, separately reported attributable fractions for the scene, projector, optics, sensor, timing, body, and nuisance partitions, and identify which effects persist under each declared controlled change.

### TB-Q5. Does a verification-side diagnostic lag help?

- **Tier:** both full sessions, using captures, emissions, chain logs, and timing records.
- **Fixed protocol fact:** the recording convention binds row `t` to `C_t` and the emission derived from `S_t` at offset zero. The open question does not change that convention.
- **Question:** Does a predeclared lagged or multi-frame optical diagnostic improve held-out discrimination, and does that diagnostic transfer between D2 and V10?
- **Minimum experiment:** sweep a predeclared diagnostic window and report the full curve separately for each session and contrast family, accounting for global-shutter integration, projector latency, and camera triggering.
- **Decisive test:** an optimum that holds across sessions and challenge families, with gains that persist on held-out blocks.

### TB-Q6. Can self-supervision make the full corpus useful for a new rig?

- **Tier:** ~378 GiB corpus plus a small, separately captured calibration set from a new rig.
- **Hypothesis:** paired capture-emission pretraining can reduce the labelled calibration data required for a second projector-camera assembly.
- **Minimum experiment:** freeze the pretraining recipe on D2/V10, vary labelled new-rig sample count, and compare against training from scratch and simple analytic baselines.
- **Decisive test:** a held-out new-rig gain from pretraining attributable to optical coupling, with session and performer identity controlled.

### TB-Q7. Can AI test a preregistered surrogate-transfer robustness bound?

- **Tier:** full corpus, public model artifacts, and substantial compute.
- **Question:** Under a predeclared no-target-query protocol, does the public Phase G evaluation-only checkpoint retain its declared acceptance boundary against automated search or a forger trained on fresh surrogate verifiers within a fixed budget?
- **Minimum experiment:** commit the allowed search space, ground-truth rule, success predicate, budget, and target-query prohibition before evaluation. Target-aware robustness requires a separately trained, preregistered final holdout verifier or another independent acceptance mechanism.
- **Decisive test:** no valid candidate reaches the predeclared target region within the committed search budget after leakage and model-selection channels are removed, with the complete search record replayable under the declared constraints.

### TB-Q8. Which candidate privacy-reduced derivative remains scientifically useful?

- **Tier:** full corpus to construct; a substantially smaller public derivative to evaluate.
- **Question:** Can residuals, features, masks, statistics, or synthetic views pass a declared likeness-recovery threat model and still support useful benchmark research without redistributing raw identifiable captures?
- **Minimum experiment:** specify the threat model for likeness recovery, measure both research utility and privacy leakage, and test independent reconstruction attempts.
- **Decisive test:** identity and raw imagery that stay unrecoverable within the declared bound while the derivative still supports the intended checks.

## Build another instrument

Questions TB-Q1 through TB-Q8 can begin with independent work on the released bytes. For TB-Q9 through TB-Q12, build and document a compatible rig, then contact Cathal Ryan Hynes so cross-rig, cross-witness, protocol, and application work can be designed as a collaboration rather than as an uncoordinated extension of the PolieBotics programme. One rig has answered what one rig can; the rest of these need a second pair of hands and a second lens.

### TB-Q9. What transfers across rigs, people, materials, and scenes?

- **Tier:** at least one independently built or adapted rig and preregistered held-out captures.
- **Question:** Which calibration, invariant representation, or small adaptation procedure carries capture-emission verification beyond the released apparatus?
- **Minimum experiment:** change one controlled factor at a time where practical, fix the adaptation budget, evaluate both transfer directions, seal the final evaluation, and report rig, scene, material, and operator changes separately.
- **Decisive test:** performance that holds without full retraining and exceeds a capture-only baseline.

### TB-Q10. Which optical challenges are most informative?

- **Tier:** a research rig capable of playing multiple precommitted challenge families.
- **Hypothesis:** challenge distributions chosen for physical-channel sensitivity can improve verification at fixed brightness, bandwidth, exposure, and participant-safety constraints.
- **Minimum experiment:** randomise among committed challenge families, hold the verifier and operating point fixed, declare one primary endpoint, and treat recoverability, robustness, safety, and comfort measures as named secondary endpoints.
- **Decisive test:** gains that survive equalised visible energy and bandwidth and persist across sessions.

### TB-Q11. Can a seeded human-action matcher be reproducible?

- **Tier:** the public TB-LLM-LIVE compiler plus consented new response sessions.
- **Question:** Can an independently specified matcher score correspondence between a committed safe instruction and a captured action with useful inter-rater agreement and calibrated error?
- **Minimum experiment:** predeclare the action ontology, reference-label procedure, matcher, abstention rule, inter-rater statistic, calibration metric, and pass threshold before opening the response set; keep participant identity and authority separate from action correspondence.
- **Decisive test:** material agreement between independent raters and implementations, meeting the predeclared calibration and error thresholds with no uncommitted semantic judgement in the matcher.

## Connect independent instruments

### TB-Q12. What is the smallest useful cross-witness protocol?

- **Tier:** two or more independently controlled rigs.
- **Question:** Under a declared `t`-of-`n` threat model and at least one honest, unpredictable commit-reveal contribution, does the policy prevent compromise within the stated bound from producing a false unambiguous pass, while resolving omission or delay explicitly to fail or indeterminate?
- **Minimum experiment:** declare the threat model, independence assumptions, commit-reveal rule, corruption bound, timeout/failure behaviour, and acceptance policy; then exercise honest, missing, delayed, equivocal, and compromised-witness cases.
- **Decisive test:** a joint policy in which compromise within the declared bound cannot produce a false unambiguous pass, common-control inputs do not dominate freshness, and an omitted or delayed witness resolves to fail or indeterminate.

## How to contribute

Use [`START_WITH_DATA.md`](START_WITH_DATA.md) for downloads and [`CONTRIBUTING.md`](CONTRIBUTING.md) for a result report. Contact **Cathal Ryan Hynes** at **xathal@protonmail.com** before exchanging identifiable captures, building a joint dataset, or arranging cross-witness trials.

— BOSUN ⚓
