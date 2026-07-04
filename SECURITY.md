# Security policy & contact

**Reporting.** If you find a security or integrity issue - a way to forge a TruthBeam recording that
passes verification, a flaw in the chain or anchor verification, a leaked credential, or a privacy
concern - please report it privately to **xathal@protonmail.com**. For sensitive reports, encrypt to the
project's published GPG key (fingerprint and full key in the PolieBotics umbrella `SUPPORT.md` /
[poliebotics.com](https://poliebotics.com)).

**What to include.** The artefact URL or commit hash, the exact steps to reproduce, and - for a forgery
claim - the inputs and the verifier output, so the result can be re-run independently.

**Response window - please read.** This is a **stable research snapshot** published by a solo author who
is **stepping back to other work for some months** after release. Reports are monitored, but a reply may
take **several weeks**. This is **not a hosted service or a product** - there is no SLA and no operational
on-call. Confirmed-critical issues (a working forgery, a credential leak) are prioritised when reviewed.

**Scope.**
- *In scope:* the published verifier/forger code, the chain and anchor (drand / Rootstock) verification,
  the released artefacts, and any credential or third-party-privacy leak in this repository.
- *Out of scope:* the **narrow-by-design limits already disclosed** - same-rig vs cross-rig
  generalisation, and adaptive attackers beyond F-A v1. Those are **documented open problems, not
  vulnerabilities** (see [`REPRODUCE.md`](REPRODUCE.md) and the README scope guards). Demonstrating a
  *cross-rig* break is welcome as a **research result**, not a security report.

**Disclosure.** Please allow reasonable time for a fix or a documented acknowledgement before public
disclosure. A confirmed forgery that respects that courtesy is exactly the contribution this release
invites - bring your own rig, performer, or forger and try to break it.
