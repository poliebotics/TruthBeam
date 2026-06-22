# Optically Anchored AI Improv

In the **V10** recording session, a human performer — **Qathal**, dressed all in white — stood inside a Truth Beam projector–camera loop and followed directions he had never heard before. The directions were improvised, live, by four AI agents: **Claude, Grok, and two OpenAI models** (labeled `claude`, `grok`, `pro`, and `thinking` on-chain). Each one was rendered, projected onto the scene, captured, and **sealed into the tamper-evident chain** as it was given.

So this isn't a recording of AI *text*. It's a recording of **AI-directed human performance, anchored to physical light and public time** — something you can replay, read, and *verify*: proof of what the AI said, and when it entered the record — bound to the same capture the performer was moving inside, un-forgeable after the fact.

## What happened in the room

Seventeen moments, across the session. The agents built a world out of nothing and asked the performer to live in it — and they did it with real care:

> **claude —** *"raise one hand and press your palm flat against the projection light, as if testing whether it's warm."*
>
> **an OpenAI model —** *"raise one hand as if feeling a wall no one else can see. When you find it, nod once to the camera like the wall just told you a secret."*
>
> **claude —** *"step back from the wall and the doorway and just stand. Let your arms hang. Look directly into the camera for a slow count of five — no expression, no performance, just be seen."*
>
> **grok —** *"look straight at the camera… let the projector light slowly erase your silhouette until only the shadow remains."*
>
> **claude —** *"hold up one hand flat in the light beam and watch your own palm catch the projection. Turn it slowly, like you're reading something written on it that only you can see."*
>
> **claude —** *"look down at your white costume and notice the projection is painting you. Slowly turn in place once, like you're letting it dress you…"*
>
> **claude —** *"Cathal — not Qathal, you — take three slow steps toward the camera until your face nearly fills the frame… Then give one small, ordinary wave. Hello."*
>
> **an OpenAI model —** *"reach toward the place where the invisible wall used to be and let your hand pass through empty air. Then touch the same spot on your chest and nod once to the camera, as if the wall moved inside you."*

An invisible wall is raised and dissolved; a small bird is held and let go; the performer becomes, for a while, *literally the wall between projector and camera* (which, inside the Truth Beam loop, he is); the projection "paints" the white costume; and near the end the character is set down — *"Cathal, not Qathal"* — for a wave to whoever watches this someday. The agents even give permission to rest: *"You've been standing the whole time — the recording can hold you sitting too."*

One direction hides a verification gesture in plain sight. Grok asks Qathal to **mouth a phrase** — ten unpredictable words pulled fresh from a wordlist: *"hate moaning improve dinghy opposite gecko unmixable swerve obvious tropics."* Speaking a fresh, unguessable phrase to camera is a small **liveness** beat — words nobody had chosen in advance. What the camera caught is honest comedy: a **masked** performer handed ten words of nonsense, who considers them for a beat and simply **shrugs.** No face to read — that's the mask — just the shrug. The freshness is the point; the shrug is the charm.

## How it's anchored — and how to check it

Every direction is one **AI payload**: the agent's `response_text`, its `agent_id`, a timestamp, and a `BLAKE3` digest. On each step the v10 chain folds a 32-byte **`ai_payload_root`** over the admitted payloads, so each agent's lines are sealed into the *same one-way hash chain* as the captured frames — a **verifiable transcript** of exactly what each AI contributed. The chain is time-anchored on **RSK, the Bitcoin sidechain**, and on **drand**, whose public randomness also feeds the projected emission — so a capture can't be pre-rendered against a challenge nobody had yet seen. The capture itself is then scored by a **conditional-diffusion verifier** that separates a real capture from a forged one at AUROC 1.000 (same-rig). A code snapshot committed at the session's start even pins the exact `ai_loop` that ran it (the Anthropic / OpenAI / xAI adapters).

Nothing here asks you to trust us:

- **Read every direction** — all 38 directions across 16 moments (some moments are one agent, some are all four), plus an opening code-snapshot: browse them at [`…/ai_payloads/index.html`](https://data.truthbeam.com/sessions/v10/ai_payloads/index.html) (or the machine-readable [`index.json`](https://data.truthbeam.com/sessions/v10/ai_payloads/index.json))
- **Verify the seal** — `code/recording/verify/verify_v10.py` walks the v10 chain and checks the `ai_payload_root` against the committed state, end to end.

## Why it's worth doing

The deepfake era made *generated* media cheap and *trustworthy* media scarce. Optically anchored AI improv runs the other way: it binds AI creativity to a physical body, real light, and a public clock — **AI performance you can prove happened.**

*Scope, plainly: one rig, one performer (the author), the single V10 session. What's demonstrated here is the mechanism for sealing AI responses into the tamper-evident chain; the art form is just beginning.*
