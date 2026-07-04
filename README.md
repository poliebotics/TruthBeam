# TruthBeam

A way to record the physical world so the recording can be *independently verified* - not just trusted.

**PolieBotics · TruthBeam.** Patent pending.

> **LLM-mediated.** This README is LLM-generated from the author's materials and structured for machine ingestion - point your own LLM at it. For load-bearing claims, **verify against the code, dataset, model weights, and whitepaper**, not the summary (see [`REPRODUCE.md`](REPRODUCE.md)).

**The repository ships as `truthbeam_verify.tar.gz`** (fetch from `https://data.truthbeam.com/release/truthbeam_verify.tar.gz`); `RESTORE.md`, `CITATION.cff`, and `requirements.txt` all live inside it. **Just want to check it?** The fastest path is **[VERIFY_FAST.md](VERIFY_FAST.md)**.

---

## ⚡ Verify the headline yourself - 2 minutes, no GPU

You do not have to trust this repo. **Recompute the headline `AUROC = 1.000` yourself** from **~2 MB**
of published per-frame scores and an in-repo script - **CPU-only, in seconds** (it's a logistic-regression
**probe over the published verifier scores**, not a rerun of the model; **Path A.5** regenerates those
scores from the public weights). Step by step: **[REPRODUCE.md](REPRODUCE.md)**.

**Or run everything in one command** - `bash verify_all.sh` - a clean-room check that, from a bare
machine, fetches **only public URLs** and verifies both **Path A** (the AUROC) and the **temporal binding**
(RSK-mainnet anchors + per-tx calldata + drand BLS), printing a pass/fail transcript. No private context.

**Everything needed to verify every claim is public and ungated - no login.** The verifier and forger code ship inside the verify bundle; the weights and video are directly downloadable; the scores and 378 GiB corpus are indexed by the manifests below (directory paths are not served as browsable listings on this host). The full raw eval trees (~659 GB, not needed for verification) are available on request:

| | Direct link |
|---|---|
| 🧠 **Verifier weights** (39.8 M params) | <https://data.truthbeam.com/models/verifier/model_final.pt> |
| 😈 **Forger checkpoints** (F-A v1) | `data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_*.pt` |
| 📊 **Eval scores** (reproduce input; ~2 MB used by Path A, full set 4.3 MB) | fetched by `bash download.sh scores`, e.g. <https://data.truthbeam.com/models/repro/stage_0_eval/step_00100000/stage0_d2_raw.npz> |
| 💻 **Verifier + forger code** | inside `truthbeam_verify.tar.gz` (the `code/verifier/` + `code/recording/` trees) |
| 🗂️ **378 GiB ground-truth corpus** | `bash download.sh session d2` / `v10` · CIDs in [`CID_MANIFEST.json`](CID_MANIFEST.json) · per-file lists: [`d2_files.txt`](https://data.poliebotics.com/downloads/d2_files.txt) / [`v10_files.txt`](https://data.poliebotics.com/downloads/v10_files.txt) |
| ▶️ **2023 demo video** | <https://data.truthbeam.com/pinata/PolieBotics.mp4> |

Full reproduce + artifact table + scope guards: **[REPRODUCE.md](REPRODUCE.md)** · **[ARTIFACTS.md](ARTIFACTS.md)**.
The result is precisely scoped - one rig, two sessions, one performer - and it is a clean, unambiguous signal you can recompute end to end. The hard part is getting a full provenance pipeline to flow at all; that is done, and more rigs and performers follow from a flowing pipeline.

---

## Common objections - answered straight

We'd rather pre-empt the hostile read than dodge it. Every answer below is checkable, not rhetoric.

- **"It's a scam."** No token, no sale, no gated weights, no "DM for the real files." The headline number
  recomputes in ~2 minutes on CPU from public files - [REPRODUCE.md](REPRODUCE.md). Scams ask you to
  *trust*; this asks you to *recompute*.
- **"The AUROC is faked / cherry-picked."** Recompute it yourself (Path A). If you don't trust the
  published per-frame scores, **Path B regenerates them** from the public verifier weights and the open
  corpus - the scores are an output, not an axiom.
- **"AUROC = 1.000 is obviously overfit."** On a one-rig / two-session / one-performer corpus, a perfect held-out
  score (n=198/200) argues against memorising individual training frames, but it does not rule out rig- or session-specific overfitting; that narrower caveat is exactly why the scope is stated so tightly. It is the clean separation a controlled, end-to-end demonstration is built to show. The *robust* claim is the GPU-free
  cryptographic chain re-walk; the learned verifier is a **scoped** check, and cross-rig / adaptive-attacker
  robustness is explicitly **open work**.
- **"You graded your own homework - you trained the forger *and* the verifier."** We publish both the verifier and the attacker, so anyone can re-score it; F-A v1 is the
  *floor*, not the ceiling. So we **publish the trained forger** (`models/fa_v1_forger/*.pt`) plus the **design/scaffolding for a
  harder, verifier-aware F-A v2 family** (`models/fa_v2_surrogate_binders/`; not trained, so no adaptive or
  white-box robustness is claimed) - **bring your own rig, performer, or forger and try to beat it.** An independent reproduction is exactly the contribution we're asking for.
- **"Path A only proves your scores give 1.000, not that the scores are honest."** Then **regenerate
  them**: [REPRODUCE.md](REPRODUCE.md) **Path A.5** re-runs the public verifier on a few public raw frames
  and reproduces the published scores to ~1e-4 (Path B scales it to the full corpus). Every artifact is
  also **content-addressed** (CIDs in [CID_MANIFEST.json](CID_MANIFEST.json)), so you can confirm nothing
  was swapped after the fact.
- **"Patent-pending + donate buttons = grift."** Donations are **gifts** - no token, security, or return is
  offered or sold; the patent is a research/IP posture. Neither changes the fact that the measurement is
  public and recomputable. Judge the artifact, not the aesthetic.

---

## What is PolieBotics?

**PolieBotics** builds tools that bind **physical reality** to **cryptographically chained evidence**. The premise: in an age of
cheap synthetic media, a recording is only as good as your ability to *check* it. Rather than ask viewers to trust a
camera, PolieBotics makes the act of recording leave **physical, cryptographically-chained evidence** that a later,
independent party can re-derive and test.

The **Truth Beam** is the flagship of that idea. It is a **projector-camera instrument**: while it films a scene, it
also *projects* a pattern onto that scene - a pattern that is derived, moment to moment, from the cryptographic hash
of everything recorded so far. The camera therefore captures the world **and** a structured-light signature derived from this specific
recording's own history. Tamper with a frame and the chain no longer reproduces - and, within the measured scope
below, a learned verifier can tell.

## The Truth Beam, illustrated

![Truth Beam system overview: for each frame, the raw capture C, the projected chain-coupled emission E, and the green/red verification map](paper/figures/fig_overview.png)

*For every frame the system holds three things: the **raw camera capture `C`**, the **emitted pattern `E`** that was
projected onto the scene at that instant, and a **substrate-verification map** - the per-pixel diffusion residual,
calibrated against the per-session correct-`E` reference and shown as the excess over the calibrated threshold.
**Green = below threshold (consistent); red = above (anomalous).** This is a calibrated diagnostic visualisation,
not a raw classifier output; the quantitative claims are carried by the AUROC tables, not the colour of any single
panel.*

## How it works - the basic concepts

The Truth Beam closes a loop: **emit → project → capture → commit**, frame after frame.

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

- **Chain state `S_t`** - a 32-byte BLAKE3 hash that is the recording's running memory of everything captured so far.
- **Emission `E_t`** - the projected pattern, expanded deterministically from `S_t` via BLAKE3 in extendable-output
  (XOF) mode. Unpredictable without the chain, but fully reproducible *with* it.
- **Capture `C_t`** - the raw 8-bit BayerRG8 frame the camera records while `E_t` is lit on the scene.
- **Commit** - `BLAKE3(C_t)` is folded back into the next state `S_{t+1}`, so each frame cryptographically depends on
  the actual pixels captured. Change a capture → change every downstream state and emission → detectable.
- **External anchors** - public timing (**drand**) is read into the chain and commitments are anchored out to the
  **Rootstock (RSK)** ledger, bounding *when* the recording happened against independently observable events.
- **The verifier** - a learned model (**Phase G**, a 39.77 M-parameter ε-prediction diffusion U-Net conditioned on
  `E` via a ControlNet-style hint) scores how well a capture matches its chain-coupled emission. A genuine pair
  yields a low noise-prediction residual; a forged or mismatched one yields a high residual.

On top of that substrate this repository also includes an **emission-recovery binder** (reconstructs `Ê` from `C`)
and a **red-team** attacker (a trained forger, *F-A v1*) used to stress-test the verifier.

## What is actually proven - a time-bounded, live, ordered record

The load-bearing claim of the Truth Beam is **not** "this image is real," and **not** only "a learned
model flags the forgery." It is that the projection **binds the physical interaction in time** - and the
binding separates cleanly into what is checkable offline and what is checkable against public networks:

- **Ordering + tamper-evidence - offline, GPU-free.** Each emission depends, via BLAKE3, on every prior
  capture, so the frames are one ordered, tamper-evident sequence; the offline chain re-walk also
  confirms the recomputed terminal state `S_N` equals the value committed in the anchor. Change, drop, or
  reorder any frame and every later emission diverges. *(This re-walk does not, by itself, establish the
  external clock.)*
- **Freshness - a lower time bound, on-chain.** The genesis state `S_0` folds in a **freshly-waited RSK
  mainnet block** (the recorder waits for the *next* block after wall-time *T*), so the session could not
  have been produced before that block existed. **drand** quicknet rounds folded through the chain add a
  publicly **BLS-verifiable** freshness floor (a 3 s public beacon, folded at a measured ~5-6 s cadence, max observed staleness 16.6 s). You cannot pre-render footage
  to match a challenge that did not yet exist.
- **Commitment - an upper time bound, on-chain.** A final-root **Rootstock (RSK)** transaction commits
  `S_N` in a mainnet block, so the record **demonstrably existed by** that block's timestamp once the
  transaction is confirmed - no silent back-dating.

Together these bind the recording to the window **[anchor_start, anchor_end]** as a **live, ordered
record**; that the record is a physical light-in/light-out interaction is the separate, empirical layer
scored by the learned verifier (measured scope below). The bound is the **window between anchors**, not a
per-frame wall-clock; it proves *when*, and that *the record was live and ordered* - **not** the semantic
truth of what was staged.

**This is demonstrated, not asserted.** For both released sessions the anchors were looked up live on RSK
mainnet and the drand rounds BLS-verified, each session bound to an independent on-chain window:

| | D2 | V10 |
|---|---|---|
| UTC window (2026-04-25) | 02:07:48 → 02:48:47 | 05:10:29 → 05:35:53 |
| duration | 2459 s | 1524 s |
| RSK anchor blocks | 8768852 → 8768945 | 8769289 → 8769357 |
| drand rounds (quicknet) | 28093180 → 28093983 | 28096824 → 28097325 |
| camera commit rate | 2.496 Hz | 2.494 Hz |

Full
results + one-command reproduction: **[TEMPORAL_VERIFICATION.md](TEMPORAL_VERIFICATION.md)**. The
external-clock checks run against public RSK/drand (`verify_*.py --online`); the ordering + tamper-evidence
are the GPU-free offline re-walk ([REPRODUCE.md](REPRODUCE.md)). Either way, the time-binding stands
**independently of the learned verifier** - which is the *secondary, empirical* layer that scores physical
optical coupling on top.

---

> **Scope (please read).** Every quantitative result here is **within-session / same-rig**: one projector-camera
> apparatus, one human performer, two sessions (D2 = "Yoga", V10 = "AI-improv"). Nothing here establishes cross-rig,
> cross-camera, cross-projector, or cross-subject generalisation, and headline AUROC = 1.000 figures are
> **finite-sample, held-out** estimates (D2 n=198, V10 n=200), not zero-error proofs. The one trained attacker is
> **F-A v1** - a serious but **non-adaptive, same-rig surrogate**; the stronger verifier-aware attacker
> F-A v2 is design-only (not trained), so no adaptive / white-box robustness is claimed. Much of the
> verifier's discrimination also lives in **off-body, scene-coupled** signal (optics, projection, sensor),
> so this is a **rig/corpus provenance** result, **not** evidence of general person-level forgery detection.

> **License.** **All rights reserved. No open-source licence and no patent licence are granted** (see
> `LICENSE`). The code and paper are published so the work can be **read, reviewed, and
> independently verified**; publication grants no licence and no rights beyond those arising by law.
> **Verifying and reproducing the published results is welcome and needs no permission; any *other* use**
> (redistribution, derivative works, commercial use, or a patent licence) **requires a licence - contact
> the author (xathal@protonmail.com).**

## What's in this repository

```
paper/         not shipped in this bundle yet. The whitepaper is published at the site
               root as truthbeam_whitepaper.pdf (49 pp, the text of record); the LaTeX
               source (main.tex, sections/, refs.bib, figures/) is not published yet and
               will ship with the source repository when that goes public. The patent
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

The reader PDF is published at the site root as the **document of record**: **[Whitepaper (PDF, 49 pp)](truthbeam_whitepaper.pdf)**.

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

The Phase-G verifier training/eval scripts (`code/verifier/`) assume a CUDA GPU; the recording-verification path
above is CPU-only. Research scripts take data paths as arguments (see each `--help`); absolute paths in this
snapshot are placeholders.

## Verifying a recording (code → hash → chain)

Each released session is tamper-evident: every frame is committed under a BLAKE3 state chain whose genesis hash
`S_0` commits to - among the session nonce, an RSK block hash, and the drand beacon - a digest of the
**tile-generator source code** (`generator_code_hash`). The loop is verifiable from this repository plus the
released session data:

1. **Code → hash:** `python3 code/recording/verify/verify_generator_hash.py <session_dir>` recomputes
   `generator_code_hash` from `code/recording/protocol/tile_gpu.py` (no GPU) and compares it to the session's
   `verification_bundle.json`. For both released sessions this is
   `154be9dd75e0586df456a7eae1528b7334a415f3a977a107c91c6b0751bfc540` and **it matches this repository's source**.
2. **Hash → chain:** `python3 code/recording/verify/verify_v9.py --session-dir <session_dir>` (session **D2**,
   v9 chain) or `python3 code/recording/verify/verify_v10.py --session-dir <session_dir>` (session **V10**, whose
   v10 chain additionally folds a 32-byte `ai_payload_root` into each transition under `TB:ROW:v10`) recomputes
   `S_0` and walks the chain row by row against `chain_log.csv` and the captured frames - including the
   terminal-state check (computed `S_N` must equal the manifest's RSK-anchored `S_N_hex`). Add `--logs-only` to
   verify the chain math from the chain log + manifest alone (~4 MB, no raw frames needed).
3. **Chain → world:** opening/closing states are anchored on RSK (`anchor_txs.csv`) and pinned to drand rounds, so
   the recording's time window is externally attested. Steps 2-3 use the separately released session bundles (see
   the data plan).

## Reproducing the results · data & models

- **Paper figures / numbers** come from `results/eval/*.json` and the metric CSVs; the recovered red-team /
  segmentation / EXP-6 / EXP-7 / excess-red summaries are in `results/redteam_segmentation_evals/`.
- **Model weights** (Phase G + controls, F-A v1, 14 binders) and the **raw capture dataset** are released
  separately - see `docs/DATA_MODEL_PUBLISHING_PLAN.md` and
  `RESTORE.md`. **Cloudflare R2** is the system of record for the bulk artifacts.

## Patent & disclosures

This system is **patent-pending**, inventor/applicant **Cathal Ryan Hynes**: the parent application is published
as **WO 2025/046153 A2** (*Methods and Apparatus for Projector Camera Systems*, PCT/EP2024/080780), with PIGMIE
**Filing 1** (apparatus), **Filing 2** (governance), and **Filing 3** (self-witnessing)
pending. Filings 1 and 2 (the Reality Kernel, under `reality_kernel/`) are published in the
companion PolieBotics repository:
**[poliebotics/PolieBotics](https://github.com/poliebotics/PolieBotics)**. **Filing 3** was filed on
**3 July 2026** at the IPOI as a Full-Term Patent application (Office provisional application no.
**PTIE20260000000433**, from the filing receipt; definitive application number to follow), with claims held; its Description, Drawings, and Abstract are published in the
companion repository under `reality_kernel/pdfs/` and are IPFS-pinned (citations record:
**[data.poliebotics.com/reality_kernel/CITING.html](https://data.poliebotics.com/reality_kernel/CITING.html)**).
Publication **reserves all patent
rights**; no licence, express or implied, is granted under the patent (see `LICENSE`).

## Data & ethics

Both sessions feature a **single identifiable human performer - the author/operator himself**, who consents to the
publication of his own likeness. The ethical-display rules in the paper are applied throughout (common thresholds for
real/altered panels, no per-frame normalisation, no body suppression, diagnostics labelled). Release of the raw
capture corpus is governed separately by `docs/DATA_MODEL_PUBLISHING_PLAN.md`.

## Honesty / audit trail

This work was verified **against its own code and eval outputs**, not against the paper's prose.
`docs/verification_ground_truth.json` and `docs/whitepaper_claim_check.json` record what was checked, what is backed,
and the explicit *do-not-claim* boundaries. These are **point-in-time (2026-06-01)** records - read them alongside
`docs/RELEASE_NOTE_ON_AUDIT_ARTIFACTS.md`, which explains that the
training host has since been retired (R2 is now the system of record) and that the eval outputs they flagged as
"on Lambda" were recovered into `results/redteam_segmentation_evals/`. Known limitations are stated in the paper's
Discussion.

## Citing

See `CITATION.cff` (inside `truthbeam_verify.tar.gz`). Cite the **[whitepaper (PDF, 49 pp)](truthbeam_whitepaper.pdf)** and the patent.

## Licence

**All rights reserved.** No open-source licence, and **no patent licence** - express or implied - is granted under
the pending patent. See `LICENSE`. The artifacts are published so the work can be read, reviewed, and
independently verified; publication grants no licence and no rights beyond those arising by law. For commercial or
any other reuse, contact the author (xathal@protonmail.com).

---

## Status, reproduction & where this is going

**Published as a stable snapshot (mid-2026).** The author is moving on to other work for the next few
months, so issues, PRs, and email may go unanswered for a while - *by design, not neglect*: everything
here is built to be checked **without us**. Recompute the result ([REPRODUCE.md](REPRODUCE.md)), re-walk
the chain, and verify against the filings, dataset, and video yourself.

**Independent reproduction is the contribution we most want.** Pull the open data, run the reproduce
paths, and - the real test - **bring your own rig, performer, or forger** and try to break it.
Re-implementations, forks, and collaboration are explicitly welcome while we're heads-down elsewhere.

**Bring your own rig.** Building an instrument to reproduce this? The demonstration hardware is described
in the **[PolieBotics README](https://poliebotics.com)** under *Demonstration hardware, PolieProboscis*,
where the 3D-printable **PolieProboscis** model (STL parts) is published for reference. The
recording-method docs - the projector-camera chain protocol, `S_0` derivation, and tile generators - ship
inside `truthbeam_verify.tar.gz` under `code/recording/`. These describe the released rig and are shared
for reference, not as a turnkey build.

**Direction.** This single-rig release is the foundation. The filings **describe and enable** a larger
**cross-rig witness mesh** - several Reality Kernel modules cross-checking one shared record - and the
intent, over time, is to open a **public standard and reference modules** and to work with other
researchers, so independent instruments can interoperate and verify each other. None of that is claimed
as done here; it is the **trajectory and the open door**.
