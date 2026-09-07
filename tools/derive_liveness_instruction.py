#!/usr/bin/env python3
"""Derive a canonical safe human instruction for TB-LLM-LIVE/0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from blake3 import blake3


PROTOCOL = "TB-LLM-LIVE/0.1"
DOMAIN = b"truthbeam-llm-liveness-v1"
COMPILER_ID = "truthbeam-liveness-instruction"
COMPILER_VERSION = "0.1.0"
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "liveness_instruction_catalog.json"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def length_prefix(parts: tuple[bytes, ...]) -> bytes:
    framed = bytearray()
    for part in parts:
        if len(part) >= 1 << 32:
            raise ValueError("derivation field is too long")
        framed.extend(len(part).to_bytes(4, "big"))
        framed.extend(part)
    return bytes(framed)


def derivation_material(seed: bytes, session_id: str, catalog_sha256: str) -> bytes:
    if len(seed) != 32:
        raise ValueError("freshness seed must be exactly 32 bytes")
    if not session_id:
        raise ValueError("session_id must be non-empty")
    return length_prefix((
        DOMAIN,
        seed,
        session_id.encode("utf-8"),
        bytes.fromhex(catalog_sha256),
    ))


class Choices:
    def __init__(self, material: bytes):
        self.stream = blake3(material).digest(length=4096)
        self.offset = 0

    def index(self, size: int) -> int:
        if size < 1:
            raise ValueError("choice set must be non-empty")
        limit = (1 << 32) - ((1 << 32) % size)
        while self.offset + 4 <= len(self.stream):
            value = int.from_bytes(self.stream[self.offset:self.offset + 4], "little")
            self.offset += 4
            if value < limit:
                return value % size
        raise RuntimeError("BLAKE3-XOF choice stream exhausted")

    def pick(self, values: list[object]) -> object:
        return values[self.index(len(values))]


def derive(seed: bytes, session_id: str, catalog: dict[str, object]) -> dict[str, object]:
    catalog_blob = canonical_bytes(catalog)
    catalog_sha256 = hashlib.sha256(catalog_blob).hexdigest()
    material = derivation_material(seed, session_id, catalog_sha256)
    choices = Choices(material)

    actions = catalog["actions"]
    if not isinstance(actions, list):
        raise ValueError("catalog actions must be a list")
    action = actions[choices.index(len(actions))]
    if not isinstance(action, dict):
        raise ValueError("catalog action must be an object")

    values: dict[str, object] = {}
    parameters = action.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("action parameters must be an object")
    for key in sorted(parameters):
        options = parameters[key]
        if not isinstance(options, list):
            raise ValueError(f"parameter {key} must be a list")
        values[key] = choices.pick(options)

    word_count = int(action.get("word_count", 0))
    if word_count:
        wordlist = catalog.get("wordlist")
        if not isinstance(wordlist, list) or len(wordlist) < word_count:
            raise ValueError("catalog wordlist is too short")
        pool = list(wordlist)
        words = []
        for _ in range(word_count):
            words.append(str(pool.pop(choices.index(len(pool)))))
        values["words"] = " ".join(words)

    template = action.get("template")
    if not isinstance(template, str):
        raise ValueError("action template must be text")
    response = action.get("response")
    if not isinstance(response, dict):
        raise ValueError("catalog action must declare a response object")

    return {
        "protocol": PROTOCOL,
        "compiler": {"id": COMPILER_ID, "version": COMPILER_VERSION},
        "domain_separation_tag": DOMAIN.decode("ascii"),
        "session_id": session_id,
        "freshness_value_hex": seed.hex(),
        "catalog_id": catalog.get("catalog_id"),
        "catalog_sha256": catalog_sha256,
        "action_id": action.get("id"),
        "parameters": values,
        "response_spec": response,
        "instruction": template.format(**values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"{COMPILER_ID} {COMPILER_VERSION}")
    parser.add_argument("--seed-hex", required=True, help="public freshness value as hexadecimal bytes")
    parser.add_argument("--session-id", required=True, help="committed Truth Beam session identifier")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()

    try:
        seed = bytes.fromhex(args.seed_hex)
    except ValueError as exc:
        raise SystemExit(f"invalid --seed-hex: {exc}") from exc
    if len(seed) != 32:
        raise SystemExit("--seed-hex must contain exactly 32 bytes")
    if not args.session_id:
        raise SystemExit("--session-id must be non-empty")

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    result = derive(seed, args.session_id, catalog)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
