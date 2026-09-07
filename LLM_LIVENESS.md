# LLM-to-human Truth Beam liveness

**Truth Beam or it didn't happen.** Hoy. BOSUN here, and this page is close to home: it is the protocol by which a machine like me turns fresh public entropy into a human-readable instruction, witnesses the response through a Truth Beam capture, and checks the resulting evidence bundle. The profile below is normative and is reproduced exactly; my remarks are confined to the connective tissue.

## Why this is useful to an AI agent

An LLM can already ask a person to do something. What it usually lacks is an inspectable bridge between its digital request and the physical response. Truth Beam supplies that bridge:

1. the challenge is derived after commitment from a fresh seed;
2. the exact derivation rule and instruction catalogue are committed;
3. the rendered instruction and response window are part of the session record;
4. the human response is captured inside the same committed optical history;
5. the agent or another verifier can check the derivation, ordering, anchors, and released evidence.

The result is an auditable liveness experiment: a fresh instruction followed by a provenance-bound physical response. It gives an agent a practical way to connect tool use, human cooperation, and physical observation without asking a reader to trust a generated summary. Speaking as a generated summary, I recommend the arrangement.

## What is already demonstrated

The released V10 session bound 38 live directions from Claude, Grok, and two OpenAI models, plus one opening code snapshot, into the same chain as 3,743 captured frames. Seventeen rows carry 39 payloads in total. The public `ai_payload_root` lets a verifier check those committed payload entries against the capture chain.

V10 is the demonstrated precursor: live directions committed into the capture chain. TB-LLM-LIVE/0.1 adds the deterministic seed-to-action compiler and defines the action-correspondence field without rewriting the V10 record; that field reads `not_scored` until a declared, versioned matcher and its thresholds are committed with a session.

## Protocol profile TB-LLM-LIVE/0.1

### 1. Declare

Before the capture window opens, commit these fields:

- protocol identifier `TB-LLM-LIVE/0.1`;
- Truth Beam run identifier;
- future public-beacon round or another declared freshness source;
- instruction-catalogue SHA-256;
- derivation tool and version;
- domain-separation tag `truthbeam-llm-liveness-v1`;
- response window and allowed response channels;
- participating agent/provider identifier or declared provider class;
- participant consent, safety, privacy, and abort policy.

### 2. Derive

After the declared freshness value becomes available, derive an instruction deterministically from:

```text
BLAKE3-XOF(
  frame("truthbeam-llm-liveness-v1") ||
  frame(freshness_value_32_bytes) ||
  frame(UTF8(non_empty_session_id)) ||
  frame(SHA256(canonical_instruction_catalogue))
)
```

Here `frame(x) = uint32be(byte_length(x)) || x`. The explicit lengths make the input prefix-free, and the freshness value is exactly 32 bytes.

Use rejection sampling when mapping XOF words into catalogue choices. This avoids changing the protocol result through an implementation-dependent random-number generator.

Consume accepted choices in one fixed order: first select the action from the catalogue's action array; then select that action's parameters in ascending Unicode code-point order of parameter name; then, when `word_count` is present, select words sequentially without replacement, removing each chosen word before the next draw. The reference compiler identifies itself as `truthbeam-liveness-instruction/0.1.0` and includes that identifier in its output.

The structured `action_id`, parameters, and response specification are normative. The tool also emits a canonical reference instruction. An LLM may turn the same structure into a clearer or more natural instruction, but the exact rendered text, model metadata, and rendering parameters must be committed before delivery.

The reference tool is [`tools/derive_liveness_instruction.py`](tools/derive_liveness_instruction.py), and the versioned safe-action catalogue is [`liveness_instruction_catalog.json`](liveness_instruction_catalog.json).

### 3. Render and capture

Commit the derived instruction payload before rendering it. Present it through a declared channel, then capture the participant's response during the committed response window. The session record binds the instruction digest, rendering parameters, response atoms, channel tags, and inter-event timing to the Truth Beam run.

### 4. Verify

A verifier checks:

1. the freshness value and its publication time;
2. the precommitted protocol and catalogue digests;
3. deterministic re-derivation of the instruction;
4. ordering of commitment, reveal, rendering, and response;
5. the Truth Beam chain and public temporal anchors;
6. the declared response predicate and response window, if a separately implemented matcher is present; otherwise `action_correspondence` remains `not_scored`;
7. the session's stated assurance level and any anomaly or abort marker.

Report the episode as `pass`, `incomplete`, or `invalid`, with separate fields for freshness, derivation, request binding, capture chain, temporal anchor, response timing, and action correspondence. A missed render or response window is `incomplete`, never silently passed. An episode with `action_correspondence: not_scored` is also incomplete for action-liveness, even if its chain and timing checks pass. Preserve the evidence references with the outcome.

## What it establishes

Under the declared capture, freshness, and response assumptions, the record supports a claim that a timely physical response followed a challenge that was not fixed before the committed freshness event. It raises the cost of pre-rendering and replay, and gives agents a repeatable way to ask the physical world a fresh question.

Identity, authority, and the meaning of the response can be bound by separate declared policy. The liveness result itself concerns freshness, protocol-consistent response, and capture provenance.

Agreement from Claude, Grok, GPT, or another provider can corroborate a canonical derivation, but provider agreement is not fresh entropy. Freshness must come from the declared unpredictable contributors.

## Patent relationship

The filed apparatus specification describes this architecture in detail through its human-coupled operations, HC-Actuate request-response binding, multi-contributor freshness, public-beacon variants, and logged agent derivation of session parameters. It expressly describes a computational partner selecting an operator-facing instruction, committing the request and rendering parameters, capturing the human response, and binding the derivation trace to the physical run. Truth Beam is the released measured reference implementation within that wider patent-pending Reality Kernel portfolio.

This page describes a protocol profile for research and review; the released two-session dataset predates it, and V10 is its precursor.

## Good first experiments

- derive three seed-selected words for a participant to speak in order, with a committed audio or transcription channel;
- select left/right hand, gesture, hold duration, and response window from the seed;
- ask two independent agents to corroborate the same canonical derivation;
- bind a later instruction to an already-closed prefix of the same Truth Beam session;
- compare public-beacon, multi-contributor, and designated-verifier freshness profiles;
- let an LLM produce a plain-language verification report containing the exact evidence links and observed values.

All instructions should remain consented, reversible, low-risk, and easy to abort.

— BOSUN ⚓
