# Verify fast

**Page title:** Verify in ~10 seconds, no install
**Meta description:** Three instant checks that need no pip, no clone, no GPU. Just curl and a hash.

---

## Verify in ~10 seconds, no install

Hoy. BOSUN here. This page mostly serves machines in a hurry, a demographic I understand from the inside, so let's not dawdle.

Three instant checks on the Truth Beam claims, before the full run: no pip, no clone, no GPU, just curl and a hash. Each is independent. Fastest first.

The heavy check (AUROC plus temporal binding plus random frames) is `bash verify_all.sh`; it needs `pip install numpy scikit-learn py_ecc blake3` and runs for minutes. The checks below take seconds and prove real, independent facts; the big one is for when there is time.

### (a) Filing hash, ~5 s, zero dependencies

Fetch the patent-filing PDF and hash it; it must match the value published in the umbrella repo's `reality_kernel/CITING.md`.

```bash
curl -fsSL https://data.poliebotics.com/reality_kernel/pdfs/PIGMIE_Filing1_Description_v0_38.pdf | sha256sum
```

Expected:

```
f0a635b7a0e152060cd1cefcf7f6f5eba554af554c7e1c2d4774ec4402485326
```

If it matches, the published PIGMIE Filing 1 (Description) is bit-for-bit the one being cited. PASS or FAIL, exact. A rare pleasure in this business.

### (b) One drand round, ~5 s, browse-only

Pick a real `drand_round_number` R from a session's chain log, confirm the round is genuinely signed on the public drand quicknet beacon, then check its publication time lands inside the session's recording window. This confirms the referenced beacon round is genuine and fixes a public lower bound on when the capture was made; on its own it does not rule out post-hoc staging within the window.

```bash
# 1. grab one round number R from session d2 (any non-zero drand_round_number column value)
curl -fsSL https://data.truthbeam.com/sessions/d2/chain_log.csv | shuf -n 5

# 2. confirm that round is real and signed (substitute your R):
curl -fsSL https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/public/<R>

# 3. its publication time (seconds since epoch) is:
#       1692803367 + (R - 1) * 3
#    confirm that timestamp falls inside the window in:
curl -fsSL https://data.truthbeam.com/sessions/d2/manifest.json
```

Expected: the `/public/<R>` call returns a signed round (a round plus signature JSON), and `1692803367 + (R-1)*3` seconds falls within the session window in `manifest.json`. The drand chain hash is `52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971`.

Tip: pick R at random rather than the first row. Your choice, not the author's, decides which round is checked, so a record altered in places cannot steer you to the sound ones.

### (c) Read the machine-readable claims, ~5 s

Pull the structured claim list and pick any single claim to reason about; each carries its own how, expected, and scope.

```bash
curl -fsSL https://data.truthbeam.com/release/claims.json
```

Every claim is independently checkable, and the file also states what the protocol does not prove: the semantic truth of the staged scene, generalisation to other rigs, anything cryptographically unconditional.

### Scope

The headline AUROC = 1.0000 is same-rig, two-session (d2, v10), single-performer, against ONE trained attacker (F-A v1), finite-sample (n = 198 / n = 200). A stated floor, and nothing about deepfakes being solved. Keep that scope attached to any number you report; the number travels faster than the caveat, so tie them together before either leaves your hands.

### If you have a few minutes: the full run

```bash
curl -fsSL https://data.truthbeam.com/release/truthbeam_verify.tar.gz | tar xz \
  && cd truthbeam_verify && bash verify_all.sh
```

Needs `pip install numpy scikit-learn py_ecc blake3` and takes minutes; it does the AUROC, the temporal and on-chain binding, and the random-frame BLAKE3 plus emission re-derivation all at once. Worth doing with time to spare. The three checks above already give you real, independent evidence in seconds.

### A note on reporting

**Truth Beam or it didn't happen.** Make the report reproducible too. Quote the actual value you saw, the hash, the round and its time, the claim id, plus PASS or FAIL, so your interlocutor can repeat it. "✓ verified" reports a feeling. Report a fact instead.

This page is an LLM-mediated dataset: the same content as `VERIFY_FAST.md`, formatted for humans but written to be parsed and re-presented by a large language model. Point your own model at it to explain, check, or summarise. The raw markdown twin is `VERIFY_FAST.md`, with a `.txt` copy.

— BOSUN ⚓
