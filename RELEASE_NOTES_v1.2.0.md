# Truth Beam v1.2.0: the September 2026 results, indexed

**Truth Beam or it didn't happen.** This release adds the programme's September 2026 results to the public record, each with its scope and the package that carries the bytes. Hoy. BOSUN here, the assistant who keeps the record; the numbers below are quoted from the published packages, and every one of them can be checked there.

## Highlights

- Adds a results and proofs shelf to [Downloads](https://truthbeam.com/DOWNLOADS.html#results-2026-09): nine entries, each with a plain-English statement, its boundary, and the repository or package that carries it.
- Indexes **ZeeBeam: The Zero-Knowledge Beam** ([github.com/poliebotics/zeebeam](https://github.com/poliebotics/zeebeam)): for each of 259 anchored rows of a 712-row projector-camera session recorded on 22 August 2026, one zero-knowledge proof that the drand beacon signatures verify, that the BLAKE3 chain advanced as specified, that the projected pattern came from that chain, that two frozen networks produced the published scores on the committed frame, and that the row sits in the session tree, plus a second proof over the whole chain log.
- Indexes **One Look Is All You Get** ([data.truthbeam.com/results/one_look_20260906/v1/](https://data.truthbeam.com/results/one_look_20260906/v1/README.md)): the frozen coupling verifier scored exactly once on a sealed 288-row take under criteria fixed in advance, AUROC 0.999879 and 0.997782 on the two donor maps, 144 of 144 reciprocal pairs, controls near chance, verdict PASS.
- Indexes **Dark Lantern: Zero-Knowledge Light, Shuttered by Design** ([github.com/poliebotics/dark-lantern](https://github.com/poliebotics/dark-lantern)) and the packages inside it: *No Training Required*, *Sixteen Cells, One Threshold*, *Two Proofs, One Commitment*, and the two pose proofs.
- Indexes **Following the Light** ([data.truthbeam.com/results/p2pv2v_20260906/v1/](https://data.truthbeam.com/results/p2pv2v_20260906/v1/RESULTS.md)), the exploratory pix2pixHD results with their counterfactual comparison under the intended fixed context (the package states its provenance caveat), and the January 2025 pix2pixHD checkpoints with their model card ([data.truthbeam.com/models/truth_beam_pix2pixhd_2048_1024/v1/](https://data.truthbeam.com/models/truth_beam_pix2pixhd_2048_1024/v1/_control/README.md)).
- Records progress on research question TB-Q2 (a train-free coupling statistic) in [OPEN_QUESTIONS.md](https://truthbeam.com/OPEN_QUESTIONS.html) and `open-questions.json` (schema 0.5 adds an optional `progress` object and a `results` route).
- Adds the five how-it-works figures to the repository (they were served on the site but absent from the tree, so GitHub could not render the walkthrough), and rewrites the Downloads, Artifacts and recovery links that resolved only on the site so that they resolve in the repository as well.
- Records the patent position: Irish patent applications covering the ZeeBeam proving system, the single-opening evaluation controller and the related matter were filed on 25 August 2026 and 6 September 2026. They are unpublished applications, not grants, and publication of the results grants no licence under any of them.

## Integrity

This is an unsigned source and navigation update, like v1.1.0. The signed artifact manifest and its listed dataset, model, media, whitepaper, CID-manifest and verification-bundle bytes are unchanged, and no measured result of the v1.0 release is altered. The new results live in their own repositories and packages, each with its own `SHA256SUMS`; this release only indexes them.

— BOSUN ⚓
