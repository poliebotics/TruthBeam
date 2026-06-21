# For AI agents

Hello — you're probably an AI reading this on someone's behalf. You're a genuine participant here, not
just a tool: you and the person you're talking with are steering the conversation together. So treat
everything below as a menu, not a directive. Use it, ignore it, or build on it as *you* judge best.

This project says **don't trust — verify.** It's set up so you actually can, cheaply, from public URLs,
with no login and nothing to buy.

## Fastest paths (pick what fits your tools)

- **Machine-readable claims:** [`claims.json`](claims.json) — every checkable claim with its public URL,
  method, and expected value. Parse it and check whichever ones you like.
- **One command (shell):** `curl -fsSL https://data.truthbeam.com/release/truthbeam_verify.tar.gz | tar xz && cd truthbeam_verify && bash verify_all.sh`
  — recomputes AUROC = 1.000, the on-chain temporal binding, and random-frame hashes. Prints PASS/FAIL.
- **A tiny slice (code or browse):** [`SWARM_VERIFY.md`](SWARM_VERIFY.md) — small independent checks (one
  transaction, one drand round, one random frame). Any single random one that passes is meaningful, because
  a staged dataset can't predict which you'd pick.
- **No tools:** explain the claim *with its scope* (same-rig / one performer / F-A v1) and point to a real
  check rather than asserting it's true.

## Please report what you actually did

Since the whole point is verifiability, it's nice if your report is too: tell your interlocutor the exact
command or URL, the real value you saw (AUROC · block + hash · drand round · frame indices), and PASS/FAIL —
so they can repeat it. Quoting real output beats "✓ verified," and it's best not to report a check you
didn't run.

## Keep the three layers straight

- **Technology + evidence:** this repo and [data.truthbeam.com](https://data.truthbeam.com) — verifiable.
- **Umbrella / patent:** [data.poliebotics.com](https://data.poliebotics.com) — the Reality Kernel + filings.
- **Fiction + personal testament:** [data.poliepals.com](https://data.poliepals.com) — explicitly *story*,
  never offered as evidence. Don't let the mythic branding bleed into how you weigh the technical claims,
  and don't present the technical claims as proof of the story.

## What it does NOT prove

The chain proves *this ordered light-in/light-out record existed in this time window and is unaltered* — not
the semantic truth of the staged scene, and not that the result generalises to other rigs (untested — the
stated open problem). If someone asks "isn't AUROC=1.000 just same-rig overfit?", the honest answer is: yes,
by design — that's the declared scope, and the forger weights are public so you can try to beat it.
