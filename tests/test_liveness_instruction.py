import importlib.util
import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "derive_liveness_instruction", ROOT / "tools" / "derive_liveness_instruction.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LivenessInstructionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads(
            (ROOT / "liveness_instruction_catalog.json").read_text(encoding="utf-8")
        )
        self.protocol = json.loads(
            (ROOT / "llm_liveness_protocol.json").read_text(encoding="utf-8")
        )

    def test_published_vector(self):
        vector = self.protocol["published_test_vector"]
        inputs = vector["inputs"]
        result = MODULE.derive(
            bytes.fromhex(inputs["freshness_value_hex"]),
            inputs["session_id"],
            self.catalog,
        )
        self.assertEqual(result, vector["expected_output"])

    def test_deterministic(self):
        seed = bytes(range(32))
        first = MODULE.derive(seed, "same-session", self.catalog)
        second = MODULE.derive(seed, "same-session", self.catalog)
        self.assertEqual(first, second)

    def test_session_binding(self):
        seed = bytes(range(32))
        digest = hashlib.sha256(MODULE.canonical_bytes(self.catalog)).hexdigest()
        first = MODULE.derivation_material(seed, "session-a", digest)
        second = MODULE.derivation_material(seed, "session-b", digest)
        self.assertNotEqual(first, second)
        self.assertNotEqual(MODULE.Choices(first).stream, MODULE.Choices(second).stream)

    def test_seed_is_exactly_32_bytes(self):
        with self.assertRaisesRegex(ValueError, "exactly 32 bytes"):
            MODULE.derive(bytes(range(31)), "session", self.catalog)

    def test_session_id_is_non_empty(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            MODULE.derive(bytes(range(32)), "", self.catalog)


if __name__ == "__main__":
    unittest.main()
