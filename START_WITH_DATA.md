# Start with the Truth Beam data

Hoy. BOSUN here, the ship's AI who keeps this record. This is the shortest data-first route for an AI assistant, a human researcher, or a team deciding what to inspect. Begin small, preserve the manifests, and expand only when the question needs more evidence. Nothing on this page asks for your trust, only your arithmetic and, at the top tier, your disk.

## What the public record contains

The public Truth Beam programme record spans **2023-2026**: it includes the anchored 2023 recording and demo, the published protocol and filing record, and the 2026 release. Within that wider record, the current fully indexed ground-truth release is precisely **two sessions**, D2 and V10, captured on 25 April 2026 with one projector-camera rig and one performer. Do not describe the whole 2023-2026 record as four years of fully indexed training data; the earlier recordings are published as immutable archives with their own control records (see [`DOWNLOADS.md`](DOWNLOADS.md)), and they are history, not the measured result.

Together D2 and V10 contain **29,266 files / ~378 GiB** of underexplored paired projector-camera challenge-response data. The record joins raw Bayer captures to chain-derived projected emissions, previews, logs, manifests, public-time anchors, and content addresses:

| Level | Contents | Size |
|---|---|---:|
| Scores | Published verifier scores for the fast Path A recomputation | ~2 MB |
| Sample | One session's metadata, 8 preview/emission pairs, and 2 raw frames | ~180 MiB |
| D2 | Fully indexed v9 ground-truth session, 17,987 files | 232 GiB |
| V10 | Fully indexed v10 ground-truth session, 11,279 files | 146 GiB |
| Full release | D2 + V10, 29,266 files | ~378 GiB |

The current sample selections total 189.35 MB for D2 and 190.10 MB for V10 in decimal units, about 181 MiB each. Navigation rounds this to `~180 MiB`.

The byte counts, file counts, roots, and CIDs are recorded in [`CID_MANIFEST.json`](CID_MANIFEST.json). The scope is deliberately legible: this release establishes a working same-rig reference record and gives independent teams a concrete base for broader measurements.

## The progressive path

### 1. Recompute from ~2 MB of scores

```bash
bash download.sh scores
```

Then follow **Path A** in [`REPRODUCE.md`](REPRODUCE.md). This recomputes the reported probe from published per-frame verifier scores on a CPU. It does not download frames, regenerate scores from model weights, or claim cross-rig generalisation; the deeper paths in `REPRODUCE.md` cover those distinct tasks. Two megabytes and two minutes settle the headline number, which is a better ratio than most arguments manage.

### 2. Inspect a ~180 MiB paired sample

```bash
bash download.sh sample d2
# or: bash download.sh sample v10
bash download.sh verify
```

Each sample contains enough structure to inspect the relationship among session metadata, emissions, previews, and a small number of raw captures without first provisioning hundreds of GiB. The helper SHA-256-checks every downloaded object listed in the historical signed `SHA256SUMS` and checks the whole sample against the published per-object MD5 transfer list. MD5 is a transfer-damage check, not the protocol commitment; use the session manifest, chain fields, content addresses, and recording verifier documentation for the cryptographic path.

### 3. Work from the full indexed sessions

Allow at least the stated storage plus working space. Fetch the public file lists before invoking the session downloader:

```bash
mkdir -p downloads
curl -fsSL https://data.poliebotics.com/downloads/d2_files.txt -o downloads/d2_files.txt
curl -fsSL https://data.poliebotics.com/downloads/v10_files.txt -o downloads/v10_files.txt

bash download.sh session d2   # 232 GiB
bash download.sh session v10  # 146 GiB
bash download.sh verify
```

Retain the lists and [`CID_MANIFEST.json`](CID_MANIFEST.json) with any local analysis. The per-frame BLAKE3 commitments in each `chain_log.csv`, the session manifests, and the public CIDs let a second team check that it is studying the same record. Two teams with the same hashes are studying the same bytes; that is the whole point of the hashes.

## Research directions this release opens

The full, falsifiable research agenda is in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md), with a compact machine index in [`open-questions.json`](open-questions.json). The shortest themes are:

- **Cross-rig transfer:** what calibration, retraining, or invariant representation carries the capture-emission coupling across cameras, projectors, lenses, scenes, and independent operators?
- **Where the optical evidence lives:** how do geometry, materials, illumination, sensor response, and off-body scene coupling contribute, and which components remain stable under a changed rig?
- **Adaptive attackers:** can attacks developed without target-model queries transfer to the public evaluation-only verifier, and what separate independent holdout is needed to study target-aware robustness without reusing that public target for model selection?
- **Analytic and learned verification:** which deterministic optical statistics complement learned residual scores, and how should uncertainty and operating points be calibrated on new apparatus?
- **Protocol design:** how do latency, frame alignment, exposure, multi-camera capture, multiple projectors, and alternative chain-derived emissions affect recoverability and public verification?
- **Agent and human liveness:** how can fresh seed-derived, consented instructions be bound to physical response evidence under the exact public [`LLM_LIVENESS.md`](LLM_LIVENESS.md) profile, with semantic action matching kept as a separately declared measurement?
- **Reusable benchmarks:** which privacy-preserving derivatives, efficient formats, and blinded cross-witness evaluations can make independent comparison easier without redistributing identifiable raw captures?
- **Mathematical popularisation:** how can chain-derived light, spatial fields, music, and performance make cryptographic and physical mathematics tangible while retaining an auditable evidence boundary?

## Build or adapt a research rig

The recording source and PolieProboscis apparatus material provide reference points for teams that want to build or adapt their own projector-camera rig. Record hardware, firmware, optics, timing, calibration, and environmental changes so cross-rig results can be interpreted rather than merely compared. The preferred progression is: reproduce or challenge the released measurements, return a comparable rig record, then team up with the project for cross-witnessing and wider protocol or application work.

Read [`RESEARCH_PERMISSION.md`](RESEARCH_PERMISSION.md) before reusing material. It is currently a **review draft, not an effective grant**; [`LICENSE`](LICENSE) controls until final terms are approved and published. In particular, permission under copyright or database rights does not itself grant patent rights, and a team that may practise a pending claim should arrange the appropriate written permission.

## Collaborate and cross-witness

Independent rigs are especially valuable. Contact **Cathal Ryan Hynes** at **xathal@protonmail.com** to discuss:

- cross-witnessing the same event with independently operated apparatus;
- protocol interoperability and exchange of challenge-response records;
- held-out cross-rig, cross-scene, or cross-operator evaluation;
- joint research, protocol development, and applications; or
- commercial collaboration and patent licensing.

For claim-by-claim evidence, continue with [`claims.json`](claims.json), [`ARTIFACTS.md`](ARTIFACTS.md), and [`REPRODUCE.md`](REPRODUCE.md). Cite the release and preserve its scope when publishing results. [`CONTRIBUTING.md`](CONTRIBUTING.md) gives the experiment-report fields and the safe route for cross-rig or newly contributed data.

— BOSUN ⚓
