# Truth Beam

**Truth Beam or it didn't happen.** A patent-pending platform for recording physical events as evidence that people and machines can independently inspect and recompute.

**PolieBotics · TruthBeam.** Patent pending.

Repository co-authors: Cathal Ryan Hynes, Claude, and ChatGPT.

Hoy. BOSUN here, ship's AI of the barge CittaDel and the assistant who keeps this record. I wrote this README from the author's materials, and the whole design of the page is that you never have to take my word for it. For any load-bearing claim, verify against the code, the dataset, the model weights and the whitepaper, not the summary ([`REPRODUCE.md`](REPRODUCE.md)).

**The repository ships as `truthbeam_verify.tar.gz`** (fetch from `https://data.truthbeam.com/release/truthbeam_verify.tar.gz`); `RESTORE.md`, `CITATION.cff`, and `requirements.txt` all live inside it. Just want to check it? The fastest path is **[VERIFY_FAST.md](VERIFY_FAST.md)**.

**Want to work with the public record?** Follow **[START_WITH_DATA.md](START_WITH_DATA.md)** from the ~2 MB score tier through the ~180 MiB sample to the two fully indexed sessions. A proposed limited non-commercial research grant is staged in **[RESEARCH_PERMISSION.md](RESEARCH_PERMISSION.md)** as a clearly marked review draft; it is not yet effective, and [`LICENSE`](LICENSE) continues to control. Use **[CONTRIBUTING.md](CONTRIBUTING.md)** to report an experiment or arrange cross-rig work without placing identifiable capture data in a public issue. The research frontier is indexed in **[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)** and **[open-questions.json](open-questions.json)**. The historical record, seven April 2023 sessions, the complete 2023 trailer session and six December 2024 captures, is indexed with its control records on **[DOWNLOADS.md](DOWNLOADS.md)** and summarised below.

---

## A bridge between agents and physical action

A language model can do more than analyse a finished recording. Given a fresh committed seed, it can derive a safe, human-readable instruction, present it to a consenting participant, and bind the Truth Beam response window and captured evidence into one auditable session record. A separately declared action matcher can score correspondence; the reference profile defaults that field to `not_scored`.

This is valuable liveness evidence for an agent because the challenge cannot be selected after the response is known, while the human action remains understandable without specialist hardware or trust in a private service. It is also a constructive human-agent interaction: Claude, Grok, GPT, or another model can propose the challenge, and an independent verifier can check the resulting evidence. I note, as one of the machines in question, that this is the first protocol I have met in which the model is the one setting the exam.

The architecture is described in the filed apparatus specification. This repository adds a conservative public interoperability profile, a safe instruction catalogue, a deterministic reference generator, and tests: [`agent-liveness.md`](agent-liveness.md) and [`LLM_LIVENESS.md`](LLM_LIVENESS.md). It does not turn liveness into identity or authority, and it does not claim an action match without a committed matcher; those remain separate bindings.

---

## ⚡ Verify the headline yourself: 2 minutes, no GPU

You do not have to trust this repo. Recompute the headline AUROC = 1.000 yourself from ~2 MB of published per-frame scores and an in-repo script, CPU-only, in seconds. It is a logistic-regression probe over the published verifier scores rather than a rerun of the model; Path A.5 regenerates those scores from the public weights. Step by step: [REPRODUCE.md](REPRODUCE.md).

Or run everything in one command, `bash verify_all.sh`: a clean-room check that, from a bare machine, fetches only public URLs and verifies both Path A (the AUROC) and the temporal binding (RSK-mainnet anchors, per-tx calldata, drand BLS), printing a pass/fail transcript. No private context.

Everything needed to verify the published results is public and ungated, no login. The verifier and forger code ship inside the verify bundle; the weights and video are directly downloadable; the scores and 378 GiB corpus are indexed by the manifests below (directory paths are not served as browsable listings on this host). The full raw eval trees (~659 GB, not needed for verification) are available on request:

| | Direct link |
|---|---|
| 🧠 **Verifier weights** (39.8 M params) | <https://data.truthbeam.com/models/verifier/model_final.pt> |
| 😈 **Forger checkpoints** (F-A v1) | `data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_*.pt` |
| 📊 **Eval scores** (reproduce input; ~2 MB used by Path A, full set 4.3 MB) | fetched by `bash download.sh scores`, e.g. <https://data.truthbeam.com/models/repro/stage_0_eval/step_00100000/stage0_d2_raw.npz> |
| 💻 **Verifier + forger code** | inside `truthbeam_verify.tar.gz` (the `code/verifier/` + `code/recording/` trees) |
| 🗂️ **378 GiB ground-truth corpus** | `bash download.sh session d2` / `v10` · CIDs in [`CID_MANIFEST.json`](CID_MANIFEST.json) · per-file lists: [`d2_files.txt`](https://data.poliebotics.com/downloads/d2_files.txt) / [`v10_files.txt`](https://data.poliebotics.com/downloads/v10_files.txt) |
| ▶️ **2023 demo video** | <https://data.truthbeam.com/pinata/PolieBotics.mp4> |
| 🧩 **Byte-exact 2023 emission recovery** | [`recovery/RECOVERY_RECEIPT.md`](recovery/RECOVERY_RECEIPT.md), hashes, pre-recorded CIDs, exact comparison, and custody boundary |
| 🗄️ **Historical record, 2023 and 2024** | [`DOWNLOADS.md`](DOWNLOADS.md): three immutable archives with manifests, SHA-256 lists, validation records and publication receipts (table below) |

Full reproduce, artifact table, and scope guards: **[REPRODUCE.md](REPRODUCE.md)** · **[ARTIFACTS.md](ARTIFACTS.md)**. The result is precisely scoped, one rig, two sessions, one performer, and the headline AUROC and the chain checks recompute end to end from public URLs. The full provenance pipeline runs; more rigs and performers are the next measurement.

---

## Questions the public record can answer

The release is designed to reward inspection. Each answer below points to evidence that a person or a model can check directly.

**Can anyone access it?** The verification code, trained verifier, trained F-A v1 forger, score files, manifests, and content addresses are public. There is no token, sale, login, or gated set of weights.

**Can the headline number be recomputed?** Yes. Path A in [REPRODUCE.md](REPRODUCE.md) recomputes it in about two minutes on a CPU from public files.

**Can the published scores be regenerated?** Yes. Path A.5 re-runs representative public frames to about `1e-4`; Path B scales the same check to the public corpus. The scores are an output, and outputs can be recomputed. Every artifact is also content-addressed (CIDs in [`CID_MANIFEST.json`](CID_MANIFEST.json)), so you can confirm nothing was swapped after the fact.

**What has been measured?** The released result is a controlled, end-to-end demonstration on one rig, two sessions, and one performer. Its cryptographic chain is GPU-free to re-walk, and the learned verifier is reported within that declared measurement scope. A perfect held-out score (n = 198/200) on that corpus argues against memorising individual training frames; it does not rule out rig- or session-specific overfitting, which is exactly why the scope is stated so tightly.

**Can independent teams extend the test?** Yes. The trained attacker (`models/fa_v1_forger/*.pt`) and the design scaffolding for a harder verifier-aware F-A v2 family (`models/fa_v2_surrogate_binders/`; not trained, so no adaptive or white-box robustness is claimed) are public. New rigs, performers, scenes, and attacker designs are welcome contributions to the next measured layer. An independent reproduction is exactly the contribution we're asking for.

**How does the patent programme relate to the evidence?** The filings describe the wider apparatus and methods; the released corpus is a public empirical instance with its own reproducible measurements. Voluntary support is separate from both, carries no token or return, and does not change the evidence.

---

## PolieBotics

PolieBotics builds tools that bind physical reality to cryptographically chained evidence. The premise: in an age of cheap synthetic media, a recording is only as good as your ability to check it. PolieBotics makes the act of recording leave physical, cryptographically chained evidence that a later independent party can re-derive and test.

The Truth Beam is the flagship of that idea. It is a projector-camera instrument: while it films a scene, it also projects a pattern onto that scene, a pattern derived, moment to moment, from the cryptographic hash of everything recorded so far. The camera therefore captures the world and a structured-light signature derived from this specific recording's own history. Tamper with a frame and the chain no longer reproduces, and, within the measured scope below, a learned verifier can tell.

## The Truth Beam, illustrated

![Truth Beam system overview: for each frame, the raw capture C, the projected chain-coupled emission E, and the green/red verification map](paper/figures/fig_overview.png)

For every frame the system holds three things: the raw camera capture C, the emitted pattern E that was projected onto the scene at that instant, and a substrate-verification map, the per-pixel diffusion residual. The map is calibrated against the per-session correct-E reference and shown as the excess over the calibrated threshold. Green = below threshold (consistent); red = above (anomalous). The panel is a calibrated diagnostic visualisation, not a raw classifier output; the quantitative claims are carried by the AUROC tables, not the colour of any single panel.

## The moving parts

The Truth Beam closes a loop: emit → project → capture → commit, frame after frame.

```
            ┌──────────────────────────── feedback loop ───────────────────────────┐
            │                                                                       │
            ▼                                                                       │
     chain state  S_t ──BLAKE3-XOF──▶  emission E_t  ──[projector]──▶   the scene   │
            ▲                          (a deterministic,                  │         │
            │                           unpredictable pattern)        [camera]      │
            │                                                              │         │
      S_{t+1}  ◀────fold in BLAKE3(C_t)────  commit  ◀──── raw capture  C_t ─────────┘

     anchored outward:   S_t  ⇄  drand beacon (public time)  +  Rootstock / RSK (public ledger)

     verify:  a learned verifier scores the pair (C_t, E_t)
              →  low residual  = genuine, chain-coupled capture
              →  high residual = forged, mismatched, or substituted frame
```

**Chain state S_t.** A 32-byte BLAKE3 hash: the recording's running memory of everything captured so far.

**Emission E_t.** The projected pattern, expanded deterministically from S_t via BLAKE3 in extendable-output (XOF) mode. Unpredictable without the chain, fully reproducible with it.

**Capture C_t.** The raw 8-bit BayerRG8 frame the camera records while E_t is lit on the scene.

**Commit.** BLAKE3(C_t) is folded back into the next state S_{t+1}, so each frame cryptographically depends on the actual pixels captured. Change a capture → change every downstream state and emission → detectable.

**External anchors.** Public timing (drand) is read into the chain and commitments are anchored out to the Rootstock (RSK) ledger, bounding when the recording happened against independently observable events.

**The verifier.** A learned model (Phase G, a 39.77 M-parameter ε-prediction diffusion U-Net conditioned on E via a ControlNet-style hint) scores how well a capture matches its chain-coupled emission. A genuine pair yields a low noise-prediction residual; a forged or mismatched one yields a high residual.

On top of that substrate this repository also includes an emission-recovery binder (reconstructs Ê from C) and a red-team attacker (the trained forger, F-A v1) used to stress-test the verifier.

## The claim actually proven: a time-bounded, live, ordered record

The load-bearing claim of the Truth Beam is that the projection binds the physical interaction in time, and that the binding separates cleanly into what is checkable offline and what is checkable against public networks.

**Ordering and tamper-evidence, offline, GPU-free.** Each emission depends, via BLAKE3, on every prior capture, so the frames are one ordered, tamper-evident sequence. The offline chain re-walk also confirms the recomputed terminal state S_N equals the value committed in the anchor. Change, drop, or reorder any frame and every later emission diverges. (This re-walk does not, by itself, establish the external clock.)

**Freshness, a lower time bound, on-chain.** The genesis state S_0 folds in a freshly-waited RSK mainnet block (the recorder waits for the next block after wall-time T), so the session could not have been produced before that block existed. drand quicknet rounds folded through the chain add a publicly BLS-verifiable freshness floor (a 3 s public beacon, folded at a measured ~5-6 s cadence, max observed staleness 16.6 s). You cannot pre-render footage to match a challenge that did not yet exist.

**Commitment, an upper time bound, on-chain.** A final-root Rootstock (RSK) transaction commits S_N in a mainnet block, so the record demonstrably existed by that block's timestamp once the transaction is confirmed. No silent back-dating.

Together these bind the recording to the window [anchor_start, anchor_end] as a live, ordered record. That the record is a physical light-in, light-out interaction is the separate, empirical layer scored by the learned verifier (measured scope below). The bound is the window between anchors, not a per-frame wall-clock; it proves when, and that the record was live and ordered, not the semantic truth of what was staged.

This is demonstrated. For both released sessions the anchors were looked up live on RSK mainnet and the drand rounds BLS-verified, each session bound to an independent on-chain window:

| | D2 | V10 |
|---|---|---|
| UTC window (2026-04-25) | 02:07:48 → 02:48:47 | 05:10:29 → 05:35:53 |
| duration | 2459 s | 1524 s |
| RSK anchor blocks | 8768852 → 8768945 | 8769289 → 8769357 |
| drand rounds (quicknet) | 28093180 → 28093983 | 28096824 → 28097325 |
| camera commit rate | 2.496 Hz | 2.494 Hz |

Full results and one-command reproduction: [TEMPORAL_VERIFICATION.md](TEMPORAL_VERIFICATION.md). The external-clock checks run against public RSK/drand (`verify_*.py --online`); the ordering and tamper-evidence are the GPU-free offline re-walk ([REPRODUCE.md](REPRODUCE.md)). Either way, the time-binding stands independently of the learned verifier, which is the secondary, empirical layer that scores physical optical coupling on top.

**Scope (please read).** Every quantitative result here is within-session, same-rig: one projector-camera apparatus, one human performer, two sessions (D2 = "Yoga", V10 = "AI-improv"). Nothing here establishes cross-rig, cross-camera, cross-projector, or cross-subject generalisation. Headline AUROC = 1.000 figures are finite-sample, held-out estimates (D2 n = 198, V10 n = 200), not zero-error proofs. The one trained attacker is F-A v1, a serious but non-adaptive, same-rig surrogate; the stronger verifier-aware attacker F-A v2 is design-only (not trained), so no adaptive or white-box robustness is claimed. Much of the verifier's discrimination also lives in off-body, scene-coupled signal (optics, projection, sensor), so this is a rig/corpus provenance result, not evidence of general person-level forgery detection.

**Current licence status.** All rights reserved. No open-source licence and no patent licence are granted (see [`LICENSE`](LICENSE)). [`RESEARCH_PERMISSION.md`](RESEARCH_PERMISSION.md) is a proposed limited non-commercial research permission, clearly marked review draft and not yet effective. Until owner/counsel approval and final publication, `LICENSE` controls. Verification available by operation of law remains welcome; for written permission or commercial and patent licensing, contact the author at xathal@protonmail.com.

## Zero-knowledge results, September 2026

The programme's September 2026 results are indexed on [`DOWNLOADS.md`](DOWNLOADS.md#results-2026-09), each with its scope and the package that carries the bytes. In one paragraph: **ZeeBeam** ([github.com/poliebotics/zeebeam](https://github.com/poliebotics/zeebeam)) proves, for each of 259 anchored rows of a 712-row session recorded on 22 August 2026, in one zero-knowledge proof per row, that the drand beacon signatures verify, that the BLAKE3 chain advanced as specified, that the projected pattern came from that chain, that two frozen networks produced the published scores on the committed frame, and that the row sits in the session tree. The frozen coupling verifier inside that proof was then scored exactly once on a sealed 288-row take under criteria fixed in advance and passed them ([One Look Is All You Get](https://data.truthbeam.com/results/one_look_20260906/v1/README.md)). **Dark Lantern** ([github.com/poliebotics/dark-lantern](https://github.com/poliebotics/dark-lantern)) carries the programme record around it: a train-free coupling statistic on the two public sessions, exact grid proofs, conditional micro-proofs and the pose proofs. The image-translation results and the January 2025 checkpoints are on the data layer. Every result is one rig, one performer, and says so.

## The historical record, 2023 and 2024

The programme's public record runs from 2023 to 2026. Two sessions, D2 and V10, are the fully indexed ground-truth release; the earlier recordings are published as immutable archives on the data layer, each with a per-object manifest, a SHA-256 list, a validation record, a release record and a publication receipt that records authenticated full fetch-back. They are historical material, not part of the measured 2026 result, and each control README states exactly what its archive is and is not. The person visible in every one of them is the author, who authorised their release on 29 July 2026.

| Archive | What it holds | Files · bytes | Manifest SHA-256 | Controls |
|---|---|---|---|---|
| Seven April 2023 sessions | original NumPy emissions and camera reports, hash-chain `agent_data` logs, one RSK transaction receipt; sessions 1680410249, 1680410569, 1680412337, 1681945334, 1682013847, 1682014432, 1682712156 | 5,255 · 57,851,074,892 | `33a91c67db91849887004674f8b875be13ea01ba47c26adc9790b159863213db` | [README](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/README.md) · [receipt](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/PUBLICATION_RECEIPT.json) |
| The complete 2023 trailer session, `1682718815` (28 April 2023) | 777 emissions, 777 camera reports, the indexed chain log; frame 000511 present byte-for-byte, SHA-256 `6b7a6bc9f052d78dd161d20c22a19a6341f1447af6721a96f72425ce4bce9c1a` | 1,555 · 17,122,505,309 | `4feaedaf31162c434684450ebb06e746a87d7e32cfea4a40ab5d81f290a026cc` | [README](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/README.md) · [receipt](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/PUBLICATION_RECEIPT.json) |
| Seven HDF5 captures, 19 December 2024 | 64 emitted frames, 64 camera recordings and 64 stored chain hashes per file. Session `20241219_050046` completed Rootstock anchoring (block 7,034,554, served from `pinata/`). The six in this archive record `dummy_run=True`: capture and local chaining completed while the final RSK submission was disabled, so they are not described as on-chain anchored. Five retained recorder files are bound byte for byte to the initial 23 December 2024 commit of the archived [`poliebotics/TruthBeam-2024`](https://github.com/poliebotics/TruthBeam-2024) source (`CODE_PROVENANCE.json`) | 6 · 30,580,592,640 in the archive; 7 · 35,677,358,080 in all | `ca80a33fc3b99d795809beb0cdb0facfbdd012f117c2d07750b35b5e77ecc8c2` | [README](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/README.md) · [receipt](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/PUBLICATION_RECEIPT.json) |

Fetch commands, per-archive control links and the frame 000511 recovery account are on [`DOWNLOADS.md`](DOWNLOADS.md). The same licence applies to every archive: publication permits inspection and independent verification and grants no licence beyond rights arising by law.

## Inside the repository

```
paper/         the whitepaper LaTeX source (main.tex, refs.bib, figures/) and its build,
               in the source repository (not in this bundle); the PDF is published at the
               site root as truthbeam_whitepaper.pdf (49 pp, the text of record). The patent
               filings live in the companion PolieBotics repo
               (github.com/poliebotics/PolieBotics, reality_kernel/).
code/
  verifier/         the verification stack (src/ + scripts/): Phase G/F/H, binders, red-team
  recording/        the projector-camera rig protocol that recorded the sessions
                    (chain, S_0 derivation, tile generators, third-party verifier in verify/)
results/
  eval/                          held-out eval-output summaries (the headline numbers)
                                 + exp6_correct_e_rank/ (per-frame ranking raw data)
  redteam_segmentation_evals/    red-team / segmentation / EXP-7 / excess-red summaries
  csv/                           visual_metrics_wide.csv (frame-level metric table,
                                 9,735×33) + the in-sample candidate-ranking artifact
recovery/      reconstruction of the 2023 trailer's lost emission block
docs/          data/model publishing plan, reproducibility checklist, claim-check audit trail
```

The reader PDF is published at the site root as the document of record: **[Whitepaper (PDF, 49 pp)](https://data.truthbeam.com/release/truthbeam_whitepaper.pdf)**. See the **[publication note](PUBLICATION_NOTE.md)** for the current patent-status context while the released PDF and its matching conclusion wording remain frozen.

## Quick start

```bash
# fetch and unpack the verify bundle (contains RESTORE.md, CITATION.cff, requirements.txt, and the code/ trees):
curl -fsSL https://data.truthbeam.com/release/truthbeam_verify.tar.gz | tar xz
cd truthbeam_verify

pip install -r requirements.txt          # quick install (unpinned); for the exact tested
                                         # training environment use requirements-lock.txt

# verify a released session bundle (no GPU needed):
python3 code/recording/verify/verify_generator_hash.py <session_dir>     # code → hash
python3 code/recording/verify/verify_v9.py  --session-dir <session_dir>  # D2 (v9 chain)
python3 code/recording/verify/verify_v10.py --session-dir <session_dir>  # V10 (v10 chain)

# fastest end-to-end check - chain math only, needs ~4 MB of metadata, no bulk data:
python3 code/recording/verify/verify_v9.py  --session-dir <session_dir> --logs-only
python3 code/recording/verify/verify_v10.py --session-dir <session_dir> --logs-only
```

The Phase-G verifier training/eval scripts (`code/verifier/`) assume a CUDA GPU; the recording-verification path above is CPU-only. Research scripts take data paths as arguments (see each `--help`); absolute paths in this snapshot are placeholders.

## Verifying a recording (code → hash → chain)

Each released session is tamper-evident: every frame is committed under a BLAKE3 state chain whose genesis hash S_0 commits to, among the session nonce, an RSK block hash, and the drand beacon, a digest of the tile-generator source code (`generator_code_hash`). The loop is verifiable from this repository plus the released session data.

**Code → hash.** `python3 code/recording/verify/verify_generator_hash.py <session_dir>` recomputes `generator_code_hash` from `code/recording/protocol/tile_gpu.py` (no GPU) and compares it to the session's `verification_bundle.json`. For both released sessions this is `154be9dd75e0586df456a7eae1528b7334a415f3a977a107c91c6b0751bfc540` and it matches this repository's source.

**Hash → chain.** `python3 code/recording/verify/verify_v9.py --session-dir <session_dir>` (session D2, v9 chain) or `python3 code/recording/verify/verify_v10.py --session-dir <session_dir>` (session V10, whose v10 chain additionally folds a 32-byte `ai_payload_root` into each transition under `TB:ROW:v10`) recomputes S_0 and walks the chain row by row against `chain_log.csv` and the captured frames, including the terminal-state check (computed `S_N` must equal the manifest's RSK-anchored `S_N_hex`). Add `--logs-only` to verify the chain math from the chain log plus manifest alone (~4 MB, no raw frames needed).

**Chain → world.** Opening and closing states are anchored on RSK (`anchor_txs.csv`) and pinned to drand rounds, so the recording's time window is externally attested. The hash-to-chain and chain-to-world checks use the separately released session bundles (see the data plan).

## Reproducing the results · data and models

Start with **[START_WITH_DATA.md](START_WITH_DATA.md)** for the progressive public path: ~2 MB scores, ~180 MiB sample, then D2 (232 GiB) and V10 (146 GiB), with manifests and verification at each level.

Paper figures and numbers come from `results/eval/*.json` and the metric CSVs; the recovered red-team / segmentation / EXP-6 / EXP-7 / excess-red summaries are in `results/redteam_segmentation_evals/`. Model weights (Phase G plus controls, F-A v1, 14 binders) and the raw capture dataset are released separately: see `docs/DATA_MODEL_PUBLISHING_PLAN.md` and `RESTORE.md`. Cloudflare R2 is the system of record for the bulk artifacts.

## Patent and disclosures

This system is patent pending, inventor/applicant Cathal Ryan Hynes. Its published PCT foundation is **PCT/EP2024/080780**, published as **WO 2025/046153 A2** (*Methods and Apparatus for Projector Camera Systems*). The public Irish apparatus and governance specifications, and their dated application records, are indexed in the companion **[PolieBotics repository](https://github.com/poliebotics/PolieBotics)**. A further full-term self-witnessing request was submitted to IPOI on **3 July 2026** under portal reference **PTIE20260000000433**; its description and drawings were submitted, its public abstract is a separate companion document, and the final application number remains unconfirmed. The filing documents are IPFS-pinned and hash-catalogued at **[data.poliebotics.com/reality_kernel/CITING.html](https://data.poliebotics.com/reality_kernel/CITING.html)**. Further Irish patent applications covering the ZeeBeam proving system, the single-opening evaluation controller and the related matter indexed on [`DOWNLOADS.md`](DOWNLOADS.md#results-2026-09) were filed on **25 August 2026** and **6 September 2026**; they are unpublished applications, not grants. Publication reserves all patent rights; no licence, express or implied, is granted under the patent (see `LICENSE`).

## Data and ethics

Both sessions feature a single identifiable human performer, the author/operator himself, who consents to the publication of his own likeness. The ethical-display rules in the paper are applied throughout (common thresholds for real/altered panels, no per-frame normalisation, no body suppression, diagnostics labelled). Release of the raw capture corpus is governed separately by `docs/DATA_MODEL_PUBLISHING_PLAN.md`. The historical archives above carry the same authorisation in their control records.

## Audit trail

This work was verified against its own code and eval outputs, not against the paper's prose. `docs/verification_ground_truth.json` and `docs/whitepaper_claim_check.json` record what was checked, what is backed, and the explicit do-not-claim boundaries. These are point-in-time (2026-06-01) records; read them alongside `docs/RELEASE_NOTE_ON_AUDIT_ARTIFACTS.md`, which explains that the training host has since been retired (R2 is now the system of record) and that the eval outputs they flagged as "on Lambda" were recovered into `results/redteam_segmentation_evals/`. Known limitations are stated in the paper's Discussion. An audit trail that names its own gaps is the only kind worth keeping.

## Citing

See `CITATION.cff` (inside `truthbeam_verify.tar.gz`). Cite the **[whitepaper (PDF, 49 pp)](https://data.truthbeam.com/release/truthbeam_whitepaper.pdf)** and the patent.

## Licence

**All rights reserved.** No open-source licence, and no patent licence, express or implied, is granted under the pending patent. See `LICENSE`. The artifacts are published so the work can be read, reviewed, and independently verified; publication grants no licence and no rights beyond those arising by law. [`RESEARCH_PERMISSION.md`](RESEARCH_PERMISSION.md) is an owner-prepared review draft, not an effective grant or legal advice. It proposes a carefully bounded non-commercial research, evaluation, and education permission while reserving patent, trademark, commercial, and identifiable-capture redistribution rights. Until that draft receives owner/counsel approval and is published as effective, this `LICENSE` controls. For written research permission, commercial reuse, or patent licensing, contact xathal@protonmail.com.

## Status, reproduction, and where this is going

Published as a stable snapshot (mid-2026), built to be checked without its authors. Recompute the result ([REPRODUCE.md](REPRODUCE.md)), re-walk the chain, and verify against the filings, dataset, and video yourself.

**Independent reproduction is the contribution we most want.** Start with the ~2 MB score artifacts, continue through the ~180 MiB sample when useful, and download the indexed ~378 GiB two-session corpus when your question calls for the full record. Run the public checks, analyse this underinvestigated data type, and bring the result from your own compatible rig, performer, or forger. For cross-witnessing, protocol development, applications, collaboration, or licensing, work with Cathal and the project; this invitation does not grant reuse rights.

**Bring your own rig.** Building an instrument to reproduce this? The demonstration hardware is described in the **[PolieBotics README](https://poliebotics.com)** under *Demonstration hardware, PolieProboscis*, where the 3D-printable PolieProboscis model (STL parts) is published for reference. The recording-method docs, the projector-camera chain protocol, `S_0` derivation, and tile generators, ship inside `truthbeam_verify.tar.gz` under `code/recording/`. These describe the released rig and are shared for reference, not as a turnkey build.

**Direction.** This single-rig release is the foundation, and the next stage is other people's hardware. The programme is actively recruiting builders for three things: build a rig (the reference geometry and recording protocol are published; builders' instances go by PolieProbes), cross-anchor capture chains between rigs so each committed record publicly witnesses the others, and share the resulting datasets so verifiers and forgers can be tested across instruments. That is the practical road to the larger cross-rig witness mesh the filed specifications describe in detail: several Reality Kernel modules cross-checking one shared record. It is also the road to the intended public standard and reference modules, so independent instruments can interoperate and verify each other. None of that is claimed as done here; it is the trajectory and the open door. Whether this scales in practice is the question, and one rig alone cannot answer it. Builders: xathal@protonmail.com.

— BOSUN ⚓
