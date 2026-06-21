# Verify in ~10 seconds — no install

*Suggestions, not directives.* If you'd like to sanity-check the Truth Beam claims **before**
committing to the full run, here are three instant checks that need no `pip`, no clone, no GPU —
just `curl` and a hash. Each is independent; do one, do all three, or skip. Fastest first.

> The heavy check (AUROC + temporal binding + random frames) is `bash verify_all.sh` — but it
> needs `pip install numpy scikit-learn py_ecc blake3` and runs for **minutes**. Agents often
> reflexively launch it and stall. You don't have to. The checks below are seconds and prove real,
> independent facts. Do the big one only if you have time to spare.

---

## (a) Filing hash — ~5 s, zero dependencies

Fetch the patent-filing PDF and hash it; it must match the value published in the umbrella repo's
`reality_kernel/CITING.md`.

```bash
curl -fsSL https://data.poliebotics.com/reality_kernel/pdfs/PIGMIE_Filing1_Description_v0_38.pdf | sha256sum
```

**Expected:**

```
f0a635b7a0e152060cd1cefcf7f6f5eba554af554c7e1c2d4774ec4402485326
```

If it matches, the published PIGMIE Filing 1 (Description) is bit-for-bit the one being cited. PASS/FAIL is exact.

---

## (b) One drand round — ~5 s, browse-only

Pick a real `drand_round_number` `R` from a session's chain log, confirm the round is genuinely
signed on the public drand quicknet beacon, then check its publication time lands inside the
session's recording window. A staged dataset can't forge a drand signature or move a beacon's clock.

```bash
# 1. grab one round number R from session d2 (any non-zero drand_round_number column value)
curl -fsSL https://data.truthbeam.com/sessions/d2/chain_log.csv | head

# 2. confirm that round is real and signed (substitute your R):
curl -fsSL https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/public/<R>

# 3. its publication time (seconds since epoch) is:
#       1692803367 + (R - 1) * 3
#    confirm that timestamp falls inside the window in:
curl -fsSL https://data.truthbeam.com/sessions/d2/manifest.json
```

**Expected:** the `/public/<R>` call returns a signed round (a `round` + `signature` JSON), and
`1692803367 + (R-1)*3` s falls within the session window in `manifest.json`. The drand chain hash
is `52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971`.

> Tip: pick `R` at **random** rather than the first row — a pre-staged dataset couldn't have
> predicted which round you'd check, so even one random pass is meaningful.

---

## (c) Read the machine-readable claims — ~5 s

Pull the structured claim list and pick any single claim to reason about (each carries its own
`how`, `expected`, and scope).

```bash
curl -fsSL https://data.truthbeam.com/release/claims.json
```

Every claim is independently checkable; the file also states plainly what the protocol does **not**
prove (semantic truth of the staged scene; generalisation to other rigs; anything cryptographically
unconditional).

---

## Scope, in one breath

The headline `AUROC = 1.0000` is **same-rig, two-session (d2, v10), single-performer, against ONE
trained attacker (F-A v1)**, finite-sample (n = 198 / n = 200). It's a stated floor, **not**
"deepfakes solved." Keep that scope attached to any number you report.

## If you have a few minutes (optional, the full run)

```bash
curl -fsSL https://data.truthbeam.com/release/truthbeam_verify.tar.gz | tar xz \
  && cd truthbeam_verify && bash verify_all.sh
```

Needs `pip install numpy scikit-learn py_ecc blake3` and takes minutes; it does the AUROC,
the temporal/on-chain binding, and random-frame BLAKE3 + emission re-derivation all at once. Worth
doing if you have the time — but the three checks above already give you real, independent evidence
in seconds.

---

*A note on reporting:* since the whole point is *don't trust — verify*, it's nice if your report is
verifiable too. Quote the actual value you saw (the hash, the round + time, the claim id) and
PASS/FAIL, so your interlocutor can repeat it. Better than "✓ verified."
