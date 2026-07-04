# How Truth Beam works

[← Truth Beam](index.html)

*A walkthrough built from the actual evaluation artifacts, not screenshots. The figures referenced are real captures, real emissions, and real measured scores from the released dataset.*

A photograph or a video is, cryptographically, just a bitstring. Hashing it proves that this exact bitstring existed by the time the hash was published, and nothing more. It cannot tell you whether the scene in front of the camera was real, whether the lighting was added afterward, or whether a generative model produced the pixels. As synthetic media became cheap, that gap became the central problem of provenance: a recording's authenticity is no longer self-evident from the recording.

Truth Beam attacks the problem from the other end. Instead of certifying a finished file, it makes the *act of capture* verifiable, by turning the scene's illumination into a cryptographic challenge that the camera must answer in the moment.

## The rig as a physical challenge-response coupling

*Figure (`fig_rig.jpg`): the apparatus, before the projection lights up - a projector and a camera on tripods face the scene (a ship's bridge), with a laptop running the chain. Truth Beam treats this projector-scene-camera system as one physical challenge-response coupling.*

*Figure (`fig_cap.jpg`): the same rig, mid-recording - a real, unedited capture (session V10, frame 2530). Everything on the performer and the walls is the projected challenge for that single frame.*

Treat the projector, the scene, and the camera as one physical challenge-response system. Given a projected pattern, the emission `E_t`, the camera's capture `C_t` of the scene under it is shaped by the exact optics, geometry, surfaces, and timing of this one rig. The coupling behaves like a physical unclonable function: easy to measure here, hard to reproduce without this rig. We say "behaves like" deliberately. Physical unclonability is treated here as an empirical property, not something we prove; an adversary who wants to forge a capture has to reproduce that mapping without the rig, and do it for a challenge that was not knowable in advance.

## The loop

*Figure (`fig_emi.jpg`): the emission `E_t` the projector threw for that frame - a BLAKE3-XOF expansion of the chain state, rendered to light.*

The challenge is not arbitrary. At each step the system holds a 32-byte chain state `S_t`, a BLAKE3 hash chain that folds in the previous captures and fresh public randomness from *drand* (the League of Entropy beacon). The opening state also commits a fresh *Rootstock* block, and later state commitments and the final root are anchored to Rootstock, a Bitcoin sidechain. The two play different roles: drand supplies unpredictable public randomness, while Rootstock supplies a public, timestamped commitment. A BLAKE3 extendable-output expansion of the state produces the emission; the raw capture is hashed back into the next state, closing the loop.

Two properties follow. First, because the emission depends on the unpredictable drand beacon round, a public value that does not exist until it is published, the emission for a given moment cannot be fixed in advance: under the measured commit timing (the projection ran at about 2.5 Hz, roughly 0.4 s per frame, so the window between a challenge appearing and its capture being committed is short), no one can pre-render a forged video against a challenge that has not yet been drawn. Second, because each capture is folded into the next state and the chain is anchored to a public ledger, the record is ordered and tamper-evident. Altering or reordering a frame breaks the chain from that point onward.

## Two kinds of guarantee

It helps to separate two claims, because they are not the same kind of thing. The chain gives a *cryptographic* guarantee: the captures are ordered, tamper-evident, and anchored to public time, and that holds by construction. Whether a given capture actually *answers* its challenge is a *learned, empirical* judgment, made by a neural verifier. Its accuracy is measured, not proven, and every number below is a finite-sample estimate on a single rig.

## Verification: did the capture answer the challenge?

*Figure (`fig_verifier_plot.png`): diffusion-residual scores for 120 evaluation frames (60 from D2, 60 from V10). For each frame the verifier scores the correct emission against the best of fifty wrong ones. The bands never cross.*

The primary check is a diffusion verifier: a 39.8-million-parameter ε-prediction U-Net that, conditioned on a candidate emission, scores how consistent a capture is with having been taken under that emission. A low residual means a good optical match. The plot shows that residual for 120 frames, 60 from each session: the correct emission (green) and the best-scoring of fifty wrong emissions (red) form two bands that never touch.

> **Why the numbers here are careful.** The ideas have a long history, and the loop itself has run before: Pix2PixHD-class models and earlier rigs closed this same projector-scene-camera loop, and the 2023 recordings from that work are content-addressed on IPFS and anchored on-chain. This release is the deliberate tidy version of that line of work, and its new discipline is in the packaging of the evidence: the signal is characterised, its signal-to-noise ratio measured rather than assumed, and every claim is scoped to what the measurement supports. That is why the numbers below are deliberately modest and recomputable.

### How the evaluation works

There are two separate evaluation designs here, and it matters not to collapse them.

- **The ranking test.** For 120 frames (60 from D2, 60 from V10) each capture is scored against 51 candidate emissions: the true one plus 50 random same-session emissions, with the ±10 temporal neighbours excluded and no curation. The capture, the noise seeds, and the timestep are held fixed, so only the emission varies. The correct emission ranks first of 51 in every one of the 120 frames (top-1 = 100%); the scatter is those same frames.
- **The held-out AUROC.** A separate design scores each held-out frame (n = 198 for D2, n = 200 for V10) against five temporal-offset negatives: the emissions at -2, +2, -15, +15, and +30 frames. The verifier separates the correct emission from these at *AUROC = 1.000*, per offset and averaged. These offsets deliberately include the near neighbours the ranking test excludes, so the two are distinct designs and not the same numbers.
- **Against a forger.** A separate question is whether a real capture can be told from a forged one. F-A v1 is the single trained forger in this release, and its weights are public, so you can try to beat the verifier yourself. A Path-A probe (a logistic regression on the verifier's scores) separates real from F-A v1 captures at AUROC = 1.000. This is a measured, budget-indexed floor against that one attacker, not a formal guarantee; a stronger adaptive attacker (F-A v2) exists only as a design.

**What F-A v1 is.** F-A v1 is a trained forger whose task is to produce captures that pass as genuine. It is *non-adaptive*: it does not see the verifier or train against it, so this is a fixed-attacker test, not a white-box one. It is a *same-rig surrogate*, trained on this rig's genuine captures, and its checkpoints (from 5,000 to 100,000 steps) and weights are public, so anyone can load it and attack the verifier directly. It is deliberately the v1 *floor*: a serious but non-adaptive attacker. The adaptive, verifier-aware attacker F-A v2 exists only as a design and is not trained, so no adaptive or white-box robustness is claimed. The exact architecture and inputs are in the published forger code.

**On the perfect scores.** AUROC = 1.000 means no errors were observed on the held-out set, not that the error rate is exactly zero. With zero errors on n of order 200, the rule of three places the upper 95% bound on the error rate at about 3/n, near 1.5%. The two AUROCs also measure different things and should not be collapsed into one number: the emission-discrimination AUROC checks that the optical coupling carries the committed challenge, an easier sanity test against near-orthogonal random emissions; the forger AUROC is the harder security test, real captures against F-A v1.

## Recovery: the coupling really carries the challenge

*Figure (`fig_recovery_quad.jpg`): input capture, true emission, the binder's reconstruction `Ê`, and the difference. The reconstruction recovers the emission at about 26.2 dB PSNR (Pearson 0.975).*

A natural worry is that the verifier might be exploiting some incidental correlation rather than the optical coupling itself. A second model addresses this directly: a feed-forward binder that tries to reconstruct the emission from the capture alone, with no access to the true pattern. That the reconstruction lands within 26.2 dB PSNR of the truth is consistent, as an in-sample illustration, with the chain-derived challenge being present in and recoverable from the captured light.

One caveat, stated plainly: this recovery is *in-sample*. The binder was trained on the frame range it is reconstructing here, so it illustrates that the coupling carries the challenge but is not a held-out benchmark. No held-out recovery metric is claimed; the held-out results are the emission-discrimination and forger AUROCs above.

## A live demonstration: AI-directed improv

To show the chain can seal more than frames, the V10 session ran a live improvisation directed by four large language models: Claude, Grok, and two OpenAI models (labelled `claude`, `grok`, `pro`, and `thinking` on-chain). Each directive was committed through the session's `ai_payload_root`, sealed into the same hash chain as the captures. You can [read every directive](https://data.truthbeam.com/sessions/v10/ai_payloads/index.html), including the moment Grok asked the performer to mouth ten unguessable words, "hate moaning improve dinghy opposite gecko unmixable swerve obvious tropics": a small liveness beat, fresh words nobody had chosen in advance.

## Scope, stated plainly

Every quantitative result here is same-rig, single-performer, and drawn from two sessions (D2 and V10). What each result is, precisely:

- The emission-discrimination AUROC is a finite-sample held-out estimate (n = 198 for D2, n = 200 for V10).
- The forger-probe AUROC is a separate Path-A held-out estimate on its own test split.
- The only trained forger is F-A v1, so no claim is made about an adaptive attacker.
- The recovery demonstration is in-sample.

The same scope as a scannable grid - each row is exactly what the prose above and below states, no more:

| Condition | Status |
|---|---|
| Ordered, tamper-evident hash chain | **Demonstrated - recomputable** |
| Public time anchoring (drand + Rootstock, both edges) | **Demonstrated - recomputable** |
| Same rig, held-out frames, two sessions (D2, V10) | **Demonstrated - recomputable** |
| Emission discrimination (correct vs wrong emission) | **Demonstrated - recomputable** |
| Trained forger F-A v1 (non-adaptive) | **Demonstrated - recomputable floor** |
| Adaptive / white-box forger (F-A v2) | Not claimed - design only, never trained |
| Cross-rig, new camera, new performer | Not claimed - enabled in the filings, not demonstrated |
| General deepfake detection | Not claimed |
| Semantic truth of the scene | Out of scope by design - provenance, not content |

The honest-rig assumption is about the initial data collection - building the trusted genuine corpus to train on - not a free pass in the held-out evaluation, which the verifier still has to pass on unseen frames. (A compromised operator or projector faking a genuine capture is a separate threat, outside what is evaluated here.) That assumption is a property of how the released data was collected, not a fundamental ceiling. The design intent, set out in the filings, is that a verifier trained across many rigs can in principle evaluate an untrusted rig, which is the whole point; that broader capability is enabled in the filings, not demonstrated by the released results. For the released sessions the system demonstrates an ordered, tamper-evident, time-anchored capture record. It does not establish the semantic truth of the staged scene, nor that the method generalises to other rigs, cameras, or performers. It is a clean, measured, end-to-end result - a strong, unambiguous signal on the released sessions - not a claim of general deepfake detection. By design it is the *reference signal*: the clean, recomputable baseline - the optical-provenance counterpart of the canonical Reality Kernel loop - that cross-session, cross-rig, and networked datasets are measured against. More sessions, cross-session verification, and partner rigs follow.

## Verify it yourself

```bash
curl -fsSL https://data.truthbeam.com/release/truthbeam_verify.tar.gz | tar xz
cd truthbeam_verify
bash verify_all.sh
```

Recomputes the forger AUROC (Path A, real vs F-A v1); the emission-discrimination figures are in the evaluation box above. Re-checks the on-chain anchors (Rootstock) and the drand rounds, and re-derives random frame hashes. Public URLs, no login.

## Related approaches

Truth Beam sits among several ways to make media trustworthy, and it helps to say how it differs.

**Content provenance (C2PA / Content Credentials).** These cryptographically sign a file and its edit history, so a viewer can check who signed it and how it was changed. That is valuable, but it certifies the bitstream and its handling, not that the scene in front of the camera was physically real. Truth Beam attests the physical light-in and light-out interaction itself, which signing alone does not.

**Watermark illumination (Noise-Coded Illumination, VeriLight).** These embed a coded light signal at the scene and later correlate the video against it, close in spirit to Truth Beam. Noise-Coded Illumination uses fixed secret light codes, which leaves it more exposed to replay. VeriLight's signature is dynamic and content-bound - ground the Truth Beam has occupied since its 2023 recordings, whose emissions derive from the recording's own hash-chained history. Truth Beam additionally binds each moment to a public, unpredictable value, a drand beacon round, together with public time, so the light for a given moment could not have been produced before that moment's public value was drawn. VeriLight's distinctive contribution is imperceptibility: the coded light is modulated to stay invisible to the eye and unobtrusive in ordinary footage while remaining decodable, and making that survive real-world compression is genuine work - credit where it is due.

On timing: Truth Beam's earliest recordings are anchored on the public Rootstock chain in 2023, and the patent family was first filed that year, around two years before VeriLight's 2025 publication; the pins and anchors are third-party-verifiable. The two lines of work were developed independently. Undetectability was never this programme's focus - the released system is built for strong, characterisable signal, its signal-to-noise ratio measured and every claim recomputable. The filings describe imperceptible operation in their own idiom: co-emission companion streams declared below a perceptual-visibility threshold (imperceptibility stated relative to a declared bleed-through envelope, never as an absolute), and the infrared probing channel - strong signal in a band the scene never sees, rather than a weak signal hidden inside the picture. Enabled in the filings, not demonstrated here.

**Optical physical unclonable functions.** The hardness reading of Truth Beam is in this lineage: a physical interaction that is hard to reproduce. The known caution from that field is modelling attacks, which is exactly why the hardness here is stated as a measured property against a declared attacker and budget, to be re-measured as attackers improve, rather than as an absolute.

## Infrared probing: early experiments

Early experiments with infrared projectors and infrared-sensitive cameras are promising, and results will be forthcoming. The coupling architecture behind these experiments is set out with the wider Reality Kernel at [rk-embodiments](https://data.poliebotics.com/rk-embodiments.html#infrared) - enabled in the filings, not demonstrated here.

## The wider programme

The formalism behind this is the [Reality Kernel](https://data.poliebotics.com/reality-kernel.html), the Markov kernel of which this projector-camera loop is **one demonstrated instance**; the formalism is documented at [PolieBotics](https://poliebotics.com), and its named conceptual layers (the proposed Filing-2 governance) at [PoliePals](https://data.poliepals.com/named-layers.html), alongside a deliberately imaginal story layer (the PolieBot Park and CittaDel deployments).

## Read the full account

This page is a walkthrough. The document of record, with the full method, evaluation, and scope, is the [Whitepaper (PDF, 49 pp)](truthbeam_whitepaper.pdf).

---

*This is the machine-readable twin of `how-it-works.html` (same content, plus a `.txt` copy). It is an LLM-mediated walkthrough of the demonstrated digital Truth Beam; verify load-bearing claims against `claims.json`, the whitepaper, and the verify bundle rather than this prose.*
