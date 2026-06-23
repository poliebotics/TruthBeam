# Reproduce it yourself

**Everything you need to check the headline claim is public and linked here.** No login, no permission,
no trust in us. Two paths: a **2-minute, CPU-only** recomputation of the headline AUROC, and a **full**
re-scoring path for the thorough.

---

## ⚡ Path A — recompute `AUROC = 1.000` in ~2 minutes, no GPU, no 378 GiB

The headline number is computed by an in-repo script from **published per-frame scores** (the 8 merged score files, ~2 MB).
You do **not** need the model, a GPU, or the full corpus to check it.

```bash
# 0. deps
pip install numpy scikit-learn

# 1. this repo
cd code/verifier/scripts

# 2. download the eval scores (8 merged npz, ~2 MB) (the real + forged per-frame verifier outputs)
for ck in 00005000 00025000 00070000 00100000; do
  for s in d2 v10; do
    mkdir -p stage_0_eval/step_$ck
    curl -sL -o stage_0_eval/step_$ck/stage0_${s}_raw.npz \
      https://data.truthbeam.com/models/repro/stage_0_eval/step_$ck/stage0_${s}_raw.npz
  done
done

# 3. recompute the headline AUROC (fixed seed, CPU-only, seconds)
python3 decomposition_part_1.py --stage-0-root stage_0_eval --out out --seed 0
```

**Expected output** (every probe, every held-out forger checkpoint, both sessions):

| probe | AUROC combined | AUROC D2 | AUROC V10 |
|---|---|---|---|
| Raw | **1.0000** | 1.0000 | 1.0000 |
| Coupling | **1.0000** | 1.0000 | 1.0000 |
| All | **1.0000** | 1.0000 | 1.0000 |

That number is exactly as honest as its scope: **one rig, two sessions, one performer, against the
F-A v1 forger.** A perfect score on a narrow same-rig corpus is what a narrow demo yields — it is **not**
a cross-rig or adaptive-attacker claim. Security here is **empirical, attacker- and budget-indexed
hardness**, never formal or unconditional. (See [`README.md`](README.md) → scope guards.)

*(Precisely what Path A computes: a **diffusion-feature probe** — logistic regression on the published
per-frame verifier scores, trained on the {5k, 25k, 70k} forger checkpoints and tested on the held-out
100k checkpoint. It reproduces the same perfect real-vs-fake separation the headline reports; the
verifier's own scores are the input, and those are regenerable from the public model via Path A.5.)*

---

## 🔁 Path A.5 — *regenerate* the scores from the public model (small GPU, a few frames)

Path A recomputes the AUROC from the published per-frame scores. If you don't want to trust that those
scores were honestly produced, **regenerate a few of them yourself** by running the public verifier on a
handful of public raw frames — and watch them reproduce the published `.npz`. A few frames, one modest
GPU; no 378 GiB, no days of compute.

```bash
cd code/verifier/scripts/stage_0

# public weights
curl -sLO https://data.truthbeam.com/models/verifier/model_final.pt
curl -sLO https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00100000.pt

# a few public raw frames + emissions (rows 1336/1337 and their F-A source rows 1334/1335)
mkdir -p d2/Recordings d2/derived/Emissions
for r in 001334 001335 001336 001337; do
  curl -sL -o d2/Recordings/frame_$r.raw         https://data.truthbeam.com/sessions/d2/Recordings/frame_$r.raw
  curl -sL -o d2/derived/Emissions/tile_$r.png   https://data.truthbeam.com/sessions/d2/derived/Emissions/tile_$r.png
done

# the published scores to diff against
curl -sL -o stage0_d2_raw.npz https://data.truthbeam.com/models/repro/stage_0_eval/step_00100000/stage0_d2_raw.npz

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 verify_published_scores.py \
  --diffusion-ckpt model_final.pt --fa-ckpt f_a_v1_step_00100000.pt \
  --d2-dir d2 --published-npz stage0_d2_raw.npz --n-frames 2 --bs 1
```

**Expected output** — the regenerated MSEs reproduce the published ones (worst cell `|Δ| ≈ 2e-4`,
well inside the `1e-3` tolerance; GPU float order accounts for the rest), and the real-vs-fake
separation falls out:

```
  row      condition      max|Δ|   published  regenerated  result
  1336   real_correct    1.38e-04     0.00363      0.00360  OK
  1336   fake_correct    2.05e-04     0.00729      0.00725  OK
  1337   real_correct    8.71e-05     0.00379      0.00377  OK
  1337   fake_correct    2.24e-04     0.00740      0.00735  OK
RESULT: PASS — regenerated scores reproduce the published npz.
```

The deterministic frame-selection seed and per-frame noise seed are the *same ones that produced the
release* (the fixture reuses `eval.py`'s scorer verbatim) — so there is nothing to fabricate: the
published scores are an honest output of the public model on the public frames. Path B below scales the
same idea to the full corpus.

---

## All artifacts — direct links, spelled out

| Artifact | Where | Size |
|---|---|---|
| **Verifier weights** (ε-prediction U-Net, 39,769,828 params) | <https://data.truthbeam.com/models/verifier/model_final.pt> | 456 MB |
| **Forger checkpoints** (F-A v1, 42.3 M params) | `.../models/fa_v1_forger/f_a_v1_step_{00005000,00025000,00070000,00100000}.pt` | ~165 MB ea |
| **Eval scores** (Path A input) | <https://data.truthbeam.com/models/repro/stage_0_eval/> | ~2 MB (the 8 merged files Path A uses; full score set 4.3 MB) |
| **Verifier code** | [`code/verifier/src/phase_g/diffusion_diagnostic_model.py`](code/verifier/src/phase_g/diffusion_diagnostic_model.py) | in repo |
| **Forger code** | [`code/verifier/src/phase_f/editor_controlnet.py`](code/verifier/src/phase_f/editor_controlnet.py), [`scripts/phase_f/train_phase_f_a_full.py`](code/verifier/scripts/phase_f/train_phase_f_a_full.py) | in repo |
| **AUROC script** | [`code/verifier/scripts/decomposition_part_1.py`](code/verifier/scripts/decomposition_part_1.py) | in repo |
| **Ground-truth corpus** (raw + tiles, sessions D2/V10) | <https://data.truthbeam.com/sessions/> · CIDs in [`CID_MANIFEST.json`](CID_MANIFEST.json) | ~378 GiB |
| **2023 demonstration video** | <https://data.truthbeam.com/pinata/PolieBotics.mp4> · on IPFS | — |
| **Truth Beam — Introduction** | <https://data.truthbeam.com/pinata/TruthBeam_Introduction.mp4> | 64 s |
| **Launch video** | <https://data.truthbeam.com/pinata/FULL_128_TEST_v108.mp4> | 128 s |

Forger checkpoint URLs in full:
- <https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00005000.pt>
- <https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00025000.pt>
- <https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00070000.pt>
- <https://data.truthbeam.com/models/fa_v1_forger/f_a_v1_step_00100000.pt>

**About the F-A v1 forger (and dual use).** It is a *real* learned adversary (an EditorControlNet
trained on same-rig D2/V10 temporal pairs), not a strawman — but it is a **same-rig surrogate**, trained
*without* white-box access to the verifier's gradients, so beating it is the **floor**, not a
robust-adaptive guarantee. The weights are published for **reproducibility and red-teaming of this
rig/corpus** — they are scoped to this setup, not a general face-swap tool. Use them to *check the
claim* (or to build a stronger attacker and break it), not to forge.

---

## Path B — full re-scoring (GPU, the heavy path)

To regenerate the per-frame scores Path A consumes (instead of trusting the published ones):

1. Download the **verifier weights** (`models/verifier/model_final.pt`) and the **forger checkpoints**
   (`models/fa_v1_forger/*.pt`).
2. Download the **ground-truth corpus** from `data.truthbeam.com/sessions/` (verify against
   [`CID_MANIFEST.json`](CID_MANIFEST.json)).
3. Run the verifier over real captures and over the forger's outputs to emit the `stage0_*_raw.npz`
   score files, then feed them to `decomposition_part_1.py` as in Path A. Model/loader entry points:
   `code/verifier/src/phase_g/diffusion_diagnostic_model.py`, `.../fa_loader.py`.

---

## Re-walk the cryptographic chain (no model at all)

Independent of any learned verifier, the **hash-chain + anchors are reproducible math**: re-walk the
committed chain, check each frame's emission against its committed BLAKE3 hash, verify the
tile-generator source hash against the genesis commit, and check the drand + Rootstock anchors. *(The
shipped verifier hash-checks the committed emissions and the anchors — it does not re-render each `E_t`
pixel-for-pixel.)* GPU-free verifier entry points are under
[`code/recording/verify/`](code/recording/verify/); see [`RESTORE.md`](RESTORE.md).

---

*Patent pending (WO 2025/046153 A2). All rights reserved; no licence granted by publication. The
authoritative artifacts are the whitepaper, the open dataset, and the 2023 video — verify against those
yourself; that is the only real authority.*
