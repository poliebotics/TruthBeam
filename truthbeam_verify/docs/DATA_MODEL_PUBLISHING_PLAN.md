# TruthBeam — Raw Data & Model Publishing Plan

> **Status note.** This is the original planning document. For the **current** posture see `ARTIFACTS.md`,
> `RESTORE.md`, and `docs/RELEASE_NOTE_ON_AUDIT_ARTIFACTS.md`, which supersede it where they differ:
> **Cloudflare R2 is the system of record** (Hugging Face is at most an optional mirror), the **64-second video is
> excluded** from this release (published separately), and all licensing is **all rights reserved / patent pending**.

Prepared 2026-06-01. Companion to the whitepaper (`paper/main.pdf`) and the
GitHub code/results package. Operator has authorized paid hosting ("I'll pay") and confirmed:
**human-subjects consent is in place (the operator is the sole performer)** and the **patent is pending**.
This plan turns those green-lights into a concrete release.

---

## 0. Release tiers (what goes where)

| Tier | Contents | Host | Approx size | Access |
|---|---|---|---|---|
| **A. Code + paper + results** | verifier + recording code, paper PDF+TeX, eval-output JSONs, figures, manifests (the 64s video is **published separately**, not in this repo) | **GitHub** (public) | < 0.5 GB | public to read; no licence granted |
| **B. Model weights** | Phase G verifier (main + shuffled + synthetic-positive), F-A v1 forger checkpoints (5k/25k/70k/100k), 14 EmissionPredictor binders | **Hugging Face** (model repo) + Zenodo mirror | ~6–12 GB | publicly hosted (no licence granted); gating optional |
| **C. Raw + derived capture data** | D2 + V10 raw frames, emission tiles, recording previews, NPZ thumbnails, per-row chain logs | **Zenodo** (DOI) for a curated subset; **object storage (S3/Backblaze B2)** for the full raw set | **curated ~30–50 GB; full ~300 GB+** | DOI; full-set access by arrangement with the author (no licence granted) |

The split matters: GitHub is for *reproducible artifacts a reader clones*; Hugging Face is the natural home for
*weights*; Zenodo gives a *citable DOI* for the dataset of record; and the full multi-hundred-GB raw capture is
best on cheap object storage with the Zenodo record pointing to it.

---

## 1. Tier A — GitHub (code, paper, results) — **free**

Already being staged (see `P3` GitHub package). No hosting cost. This is the primary public artifact and the
thing the paper's `\section{Availability}` should point to. Contains everything needed to *re-render the figures
and video* and to *re-run eval given the weights + data*, but not the weights or raw frames themselves (too big
for git; linked out to Tiers B/C).

---

## 2. Tier B — Model weights (~6–12 GB)

**Inventory (local `.pt`, ~508 MB each):**
- Phase G verifier: `model_final.pt` (39.77 M params; 477,531,127 bytes = ~478 MB / 455 MiB) — main run (25k steps).
- Phase G **shuffled** control + **synthetic-positive** control checkpoints (the falsification controls — release
  these too; they are what make the AUROC=1.000 credible).
- F-A v1 forger: `f_a_step{5000,25000,70000,100000}.pt` (EditorControlNet, 42.3 M; ~508 MB each ≈ 2 GB).
- 14 EmissionPredictor "binder" checkpoints (ConvNeXt-Tiny + U-Net recovery nets).

**Recommended host: Hugging Face Hub** (free public model repos, git-LFS backed, no size cost for public weights,
built-in model cards, `from_pretrained`-style download). Create `poliebotics/truthbeam-phase-g` etc.
**Cost: $0** (public). Optional Zenodo mirror for a frozen DOI'd snapshot.

**Deliverables per model:** a model card stating architecture, training config (the exact `training_config.json`),
the **within-session/same-rig scope**, intended use ("research verifier; not a deployed forensic tool"), and the
explicit note that **Phase G must never be used in F-A v2 attacker training** (the held-out-oracle discipline).

---

## 3. Tier C — Raw + derived capture data (the big, paid tier)

**Inventory (local):**
- `cittadel_first_human_20260425_020819` (D2) ≈ **233 GB** raw; a second related session ≈ 59 GB.
- V10 (`cittadel_v10_mainnet_improv`) session — raw.
- Derived video assets (emissions, recording previews, NPZ, parsable CSVs) ≈ **28 GB**.
- Full raw corpus realistically **~300 GB+**.

**Privacy note (load-bearing):** both sessions contain an **identifiable human performer (the operator)**. Consent
is given, but the paper's own *ethical-display rules* apply to any release: common thresholds for real/altered
panels, no per-frame min/max normalization, no manual body/person suppression, diagnostic methods labeled. The
release manifest must carry the consent statement and the calibration caveats the paper already flags (black level
recorded as 0 / `no_measurement_found`; frame-to-chain offset unmeasured).

**Two-part release recommendation:**
1. **Curated DOI dataset on Zenodo (~30–50 GB):** the 9,735-frame evaluation corpus actually used in the paper
   (held-out blocks + the conditions scored), derived emission tiles, recording previews, per-row chain logs, and
   the parsable metric CSVs. This is what reproduces the paper's numbers and is small enough for Zenodo.
   **Zenodo:** free up to 50 GB/record by default; **larger quota by request** (they grant 100s of GB for genuine
   research). **Cost: $0**, gives a citable DOI for `\section{Availability}`.
2. **Full raw set (~300 GB+) on object storage:** Backblaze B2 or Cloudflare R2 or AWS S3, linked from the Zenodo
   record as "full raw capture available at …". **Cost estimate (storage + egress):**
   - **Backblaze B2:** ~$6/TB-month storage → **~$1.8–2.0/month** for 300 GB; egress ~$0.01/GB (first 3×storage/day free via Cloudflare). **Cheapest; ~$25/yr + modest egress.**
   - **Cloudflare R2:** ~$0.015/GB-month → **~$4.5/month** for 300 GB; **$0 egress** (best if downloads are heavy).
   - **AWS S3 (Glacier Deep Archive for cold):** ~$0.00099/GB-month → **~$0.30/month** for 300 GB cold, but retrieval latency/fees. Standard S3 ~$0.023/GB-mo → ~$7/month.
   **Recommendation: Cloudflare R2** (zero egress, predictable ~$5/mo) for an actively-downloaded dataset, or
   **B2 + Cloudflare CDN** if you want the absolute floor.

---

## 4. Licensing — DECIDED: all rights reserved

The release is published **all rights reserved**, with **no open-source license and no patent license** (see the
repository `LICENSE`). This is deliberate: it keeps the patent strategy unencumbered — a permissive license such as
Apache-2.0 would carry an *explicit patent grant*, which is exactly what is being withheld here.

- **Code (Tier A):** all rights reserved. Published so the work can be read, reviewed, and independently
  verified; no licence and no redistribution or derivative-works grant. Commercial/other reuse: contact the author.
- **Models (Tier B):** all rights reserved. The forger weights (`F-A v1`) are additionally dual-use; if released at
  all they ship with a responsible-use note and no redistribution grant.
- **Data (Tier C):** all rights reserved; the human-subject corpus is governed by the consent + ethical-display
  terms in the paper, independent of any code terms. Commercial licensing: contact the author.

> Earlier drafts of this plan weighed permissive options (Apache-2.0 / OpenRAIL-M / CC-BY); those are **superseded**
> by the all-rights-reserved decision above. Confirm the disclosure-timing interaction with the filed patent with
> counsel before publishing.

---

## 5. Citation / DOI

- Mint a **Zenodo DOI** for (a) the curated dataset and (b) a frozen snapshot of the GitHub repo (Zenodo's GitHub
  integration tags a release → DOI automatically).
- Add a `CITATION.cff` to the repo and a BibTeX block to the paper's availability section pointing at the DOIs.
- Paper, dataset, models, and code should cross-link (paper → repo+DOIs; repo → paper+DOIs; HF cards → paper).

---

## 6. Sequencing (recommended order)

1. **Freeze** the paper PDF + code (this push). (The 64-second video is published separately, not in this release.)
2. **Publish Tier A** (GitHub) — but see the *patent-disclosure timing* check in §7.
3. **Push Tier B** weights to R2 (system of record), optionally mirrored to Hugging Face with model cards.
4. **Mint Zenodo DOIs** (curated dataset + repo snapshot); wire DOIs back into the paper's availability section and recompile.
5. **Stand up Tier C** object storage (R2/B2) for the full raw set; link from Zenodo.
6. **Recompile + re-tag** once DOIs exist.

---

## 7. Pre-publication posture (decisions recorded)

1. **License:** decided — **all rights reserved**, no open-source license, no patent license (see §4 and the
   repository `LICENSE`).
2. **Consent:** the sole performer is the author/operator, who consents to publication of his own likeness; the
   human-subject corpus is governed by that consent and the ethical-display rules in the paper.
3. **Dual-use of the forger weights:** the F-A v1 forger is dual-use; if released it ships with a responsible-use
   note and **no** redistribution/open-source grant (consistent with §4).
4. **Third-party content:** the captured scene contains no third-party copyrighted/identifiable material beyond the
   consenting author/performer.

---

## 8. Estimated total cost

- Tiers A + B: **$0** (GitHub + Hugging Face public).
- Tier C curated DOI: **$0** (Zenodo).
- Tier C full raw (~300 GB): **~$2–5 / month** on B2/R2 (≈ **$25–60 / year**) + one-time upload bandwidth.
- **Total: well under $100/year.** The "I'll pay" cost here is trivial; the real currency is the gate-confirmations in §7.
