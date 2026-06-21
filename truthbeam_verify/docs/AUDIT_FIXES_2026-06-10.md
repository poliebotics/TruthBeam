# Audit fixes — 2026-06-10

Two independent pre-publication audits of the staged release bundle (repos + session metadata)
were run on 2026-06-10. Both confirmed the core empirical spine — row counts, frame geometry,
XOF sizing, held-out AUROCs, the shuffled-control collapse, F-A v1 checkpoint results,
cross-session results, segmentation numbers, and download inventories — against the shipped
artifacts, and both flagged publication-surface defects. This note records what was found and
how each item was resolved, superseding the open items in
[`whitepaper_claim_check.json`](whitepaper_claim_check.json) (point-in-time, 2026-06-01) and
[`RELEASE_NOTE_ON_AUDIT_ARTIFACTS.md`](RELEASE_NOTE_ON_AUDIT_ARTIFACTS.md).

| Finding | Resolution |
|---|---|
| Licence carve-out ("except for the limited reading/review/verification purpose") read as a grant | `LICENSE` rewritten: purely restrictive, no purpose carve-out, statement of intent explicitly non-binding, contact + trademark lines added; permission-like phrasing removed from `README.md` and the publishing plan |
| `verify_report.json` labelled the final row's *pre*-state as `terminal_S_N_hex`, and `cycle_closure_verdict` did not compare the computed terminal state to the manifest | `verify_v9.py` fixed: it now computes the true post-final-transition `S_N`, compares it to `manifest.S_N_hex` (`terminal_state_matches_manifest`), gates `cycle_closure_verdict` on it, reports the old value as `last_logged_S_t_hex`, and implements the final-pulse terminal fallback. Both sessions' computed terminal states **match** their RSK-anchored manifests (D2 `ceb8…f2281`, V10 `13e8…59d7`). Session `verify_report.json` files produced by the older verifier are documented in the bundle guides' erratum note |
| No v10 chain-walk verifier in the snapshot | `verify_v10.py` added (v10 transition with length-prefixed state + `ai_payload_root`; ai-payload sentinel check). Validated against the released V10 session: 3,743/3,743 row advances, 101/101 pulse commitments, terminal state matches the manifest |
| Verification commands in the README were broken (missing `--session-dir`) and bundle guides used a stale package path | All public invocations normalized; `--logs-only` mode added to both verifiers so the chain math is checkable from ~4 MB of metadata without the bulk data |
| The V10 session's `README_BUNDLE.md`/`CLAIMS.md` were stale v9 copies; D2's mis-stated CSV column counts | Both sessions' bundle docs regenerated (v9 pair and a v10-specific pair, shipped in `code/recording/verify/`) and replaced on the object store; column counts now match the actual files (D2 chain log 13 cols, V10 15 cols, anchor logs 15 cols) |
| Frame-level AUROC 0.9999/0.9998 had no shipped primary artifact | `results/csv/visual_metrics_wide.csv` (9,735×33) now ships in-repo; recomputation from it gives 0.9998 for both contrasts — the one-unit fourth-decimal difference vs. the run-time aggregate is documented in the paper (§6) and `ARTIFACTS.md` |
| EXP-6 (top-1 = 100%, n=120) had no shipped raw artifact | `results/eval/exp6_correct_e_rank/` now ships the per-frame `rank_distribution.csv` (120 frames × 51 candidates with scores), summary JSON/MD, and the implementation-verification note; integrity cross-checked against two independent archive copies |
| Lockfile described as "exact tested versions" while the optional online-verification extras were unpinned | Claim narrowed in `ARTIFACTS.md`; README install instructions split into quick vs. locked |
| Strong top-of-file framing ("cryptographic proof", "could only have come from") | Softened to evidence-language and scoped to the chain-consistency claim |
| Recorder-environment residue (absolute `session_dir` paths, hostname/device id `g1a`) in session metadata | Disclosed in the bundle guides as non-authoritative provenance residue; the manifest fields are hash-committed (`manifest_hash_open` enters `S_0`) and therefore immutable by design |

## Round 2 (same day): Codex + Grok engine audits

A second round ran three Codex audits (full-bundle, code+live-reproduction, hostile-referee+legal)
and two Grok audits (full-bundle with external fact-checking, prior-art deep search). All headline
numbers reproduced independently (frame-level 0.9998/0.9998 from the shipped CSV; EXP-6 120/120).
Additional fixes from that round:

| Finding | Resolution |
|---|---|
| Verifier exit code ignored cycle-closure/terminal failures; missing `anchor_txs.csv` and deleted interior pulses passed silently; manifest/bundle/log hash commitments never recomputed; v10 accepted negative ai-payload counts | Both verifiers hardened: canonical-JSON recomputation of `manifest_hash_final` and `bundle_hash` (+linkage), chain/capture/anchor log hashes compared to the manifest (anchor hash honours its at-finalize snapshot semantics), pulse-chain continuity enforced (sequential indices, `prev_pulse_commitment` linkage, contiguous frame ranges), vacuous pulse sets can no longer close PASS, negative counts rejected, and the exit code fails on any FAIL verdict. Validated: baselines still PASS; a 14-case tamper matrix (corrupt/swap hashes, truncation, missing/edited pulse log, manifest edits, forged ai-root, negative counts) now fails loudly in every case |
| Stale "(12 cols)"/"(14 cols)" layout bullets survived in the regenerated bundle guides (replacement had silently no-opped) | Fixed in both guides and re-uploaded to the object store; "v9 B++" jargon removed |
| Paper: perfect AUROCs without uncertainty framing; 53-condition sweep without multiplicity note; shuffled-control and cross-protocol claims slightly over-broad; "precisely the overspread mode"/"defeats"/"lower bound" phrasing; a caption promising per-method AUROC tables that do not exist; §8 "every ablation" wording contradicting the cross-session ablation; §5 training figures with no artifact pointer | All re-worded conservatively (finite-sample remark, exploratory-sweep note, narrowed control claim with pointer to the cross-session ablation, protocol-session confounding stated, captions fixed); training-figure provenance added to `ARTIFACTS.md`; the §7 excess-red four-checkpoint progression now ships as `results/redteam_segmentation_evals/excess_red/fake_step_progression.csv` |
| Related work: missing active-illumination/challenge-response prior art and neighbours | Added and differentiated: Gerstner & Farid (CVPRW 2022), GOTCHA (EuroS&P 2024), DiffForensics (CVPR 2024), Ed-PUF (TIFS 2020), PRNU camera fingerprinting (Lukáš et al., TIFS 2006). All existing citations independently re-verified against public records |
| Legal: CFF "If you use…" messages, FAQ "Read and verify freely", publishing-plan "open" access column, missing PCT identifier in the landing LICENSE | All re-worded to citation-only / no-grant language; patent line canonicalized (WO 2025/046153 A2, PCT/EP2024/080780; Filing 1 & 2) |

## Round 3 (same day): adversarial review of the round-2 hardening itself

The round-2 verifier hardening had had no external review, so a five-agent adversarial workflow
re-read it line by line and **empirically broke it**, then a round-4 full-bundle re-audit ran in
parallel. The hardening was made genuinely sound:

| Finding (all reproduced on the real sessions) | Resolution |
|---|---|
| **BLOCKER — blank-the-pin downgrade.** Every commitment check was gated on the manifest/bundle *containing* the pinned field; an absent field stayed UNCHECKED, which the verdict tolerated. Deleting `manifest_hash_final`/`chain_log_hash`/`capture_log_hash`/`anchor_log_hash`/`bundle_hash` left all six checks UNCHECKED and a tampered finalized session still closed PASS (exit 0) — opening a forgery window over the non-chain-folded fields (emission paths, pixel commitments, capture log, timestamps). | For a `session_status: finalized` session the pins are now **mandatory**: a missing `manifest_hash_final`, log hash, or bundle linkage is FAIL, not UNCHECKED. The bundle check is driven from the manifest side so a whole-bundle swap that omits `bundle_hash` fails. Empirically: deleting any/all pins, editing the capture log, forging an emission path, or swapping in the other session's manifest now all FAIL with exit 1. |
| **BLOCKER — `final_root` strip too loose / row unauthenticated.** The anchor-hash fallback stripped *any* trailing line containing the substring `final_root`, and the `final_root` row (appended after the snapshot) was never validated, so its on-chain `tx_hash`/payload could be edited undetected. | The fallback now strips **at most one** structurally-valid `final_root` row (parsed `payload_kind` column, not a substring), and the stripped row is positively authenticated against `manifest.anchor_end` (`tx_hash` + payload), covered by `manifest_hash_final`. New gated field `anchor_final_row_matches`. |
| **MAJOR — dropped pulse / malformed rows.** `_load_anchor_txs` silently skipped malformed rows, so an attacker could drop a pulse behind a mangled line. | Malformed anchor rows now FAIL `pulse_chain_continuity`; a clean pulse drop is caught by the now-mandatory `anchor_log_hash`. (An over-strict "last pulse must reach the final chain row" guard I briefly added was wrong — interior pulses are periodic and the tail is committed by `S_N` — and was removed.) |
| **MAJOR — v10 empty-commit forgery.** A `ai_payload_count>0` row carrying the empty-root sentinel only logged a non-gating row error. | Both directions of the `count⇔sentinel` biconditional (plus a 64-hex root-format check) now feed `n_rows_ai_inconsistent`, which forces `ai_sentinel_check = FAIL`. |
| **MINOR ×5.** Strict JSON load (duplicate-key/NaN rejection) bypassed; swallowed log-hash recompute left a pinned value UNCHECKED; unguarded `int()` on pulse/`t`/`drand_round_number` fields crashed with a traceback; several stale docstrings/enums. | `_load_json` now uses the schema's strict loader; a pinned-but-unrecomputable hash is FAIL; all row/pulse parses are guarded and `main()` fails closed with a clean FAIL report instead of a traceback; docstrings/READMEs/CLAIMS corrected (LOGS_ONLY added to enums, `payload_kind=='state'` selector, `manifest.S_N_hex` field name, v10 `ai_sentinel_check` requirement). |
| Stale shipped `verify_report.json` (mislabelled `S_{N-1}` as terminal, missing new fields) | Regenerated with the hardened verifier (`--logs-only`) for both sessions and re-uploaded to the object store; they now carry `terminal_state_matches_manifest`, all `*_matches_manifest` fields, and `logs_only: true`. A full-media report is reproducible by running without `--logs-only`. |
| Segmentation provenance products referenced but not shipped; nested `CITATION.cff` author split inconsistent | SEGMENTATION_RESULTS.md + ARTIFACTS.md now state plainly that only the summary artifacts ship in-repo (bulk masks/scripts/shards external, on request); nested CFF author aligned to the top-level form. |

Validation: a 16-case tamper matrix (chain-folded tampers, all five pin deletions, capture-log /
emission-path / bundle-field edits, cross-session manifest swap, forged/edited `final_root`,
malformed rows, dup keys, non-integer fields) fails loudly in **every** case across both sessions,
while both clean baselines still produce `cycle_closure_verdict: PASS` and exit 0.

Items deliberately *not* changed: the session manifests, chain logs, capture logs, and anchor logs
are recorded evidence and remain byte-identical (only the verifier *output* `verify_report.json`
was regenerated); the patent filings are filed documents; the `0.9999` figure in baked figure
panels reflects the original visualization-run aggregate and is reconciled in prose rather than
re-rendered.
