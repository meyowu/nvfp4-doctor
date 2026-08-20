import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class E004ReplayEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "replay-single-projection.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest-replay.json").read_text(encoding="utf-8")
        )

    def test_case_is_real_weight_with_explicitly_synthetic_activation(self) -> None:
        self.assertEqual(self.results["status"], "pass")
        self.assertEqual(self.results["decision"], "continue")
        self.assertEqual(self.results["case"]["layer"], 0)
        self.assertEqual(self.results["case"]["projection"], "o_proj")
        self.assertEqual(
            self.results["case"]["activation_provenance"],
            "synthetic_deterministic",
        )
        self.assertEqual(len(self.results["source_tensors"]), 4)
        self.assertIn(
            "does not establish real Qwen activation",
            self.results["claim_boundary"],
        )

    def test_transform_preserves_weight_and_matches_scale_oracle(self) -> None:
        transforms = {item["name"]: item for item in self.results["transforms"]}
        weight = transforms["packed_weight_materialization"]
        scale = transforms["weight_scale_swizzle"]
        self.assertEqual(weight["source_sha256"], weight["destination_sha256"])
        self.assertEqual(weight["padding_bytes"], 0)
        self.assertEqual(scale["source_layout"], "linear_row_major")
        self.assertEqual(scale["destination_layout"], "cutlass_128x4")
        self.assertTrue(scale["vllm_candidate_byte_exact"])
        self.assertNotEqual(scale["source_sha256"], scale["destination_sha256"])

    def test_runtime_metadata_and_repetitions_are_complete(self) -> None:
        tensors = {item["name"]: item for item in self.results["runtime_tensors"]}
        self.assertEqual(tensors["weight"]["logical_shape"], [4096, 4096])
        self.assertEqual(tensors["weight"]["physical_shape"], [4096, 2048])
        self.assertEqual(tensors["activation_scale"]["logical_shape"], [16, 256])
        self.assertEqual(tensors["activation_scale"]["physical_shape"], [128, 256])
        replay = self.results["replay"]
        self.assertEqual(replay["repetitions"], 3)
        self.assertTrue(replay["all_finite"])
        self.assertTrue(replay["output_hash_stable"])
        self.assertEqual(len(set(replay["output_sha256s"])), 1)

    def test_backend_fields_and_range_scoped_kernel_evidence_are_separate(self) -> None:
        backend = self.results["backend"]
        self.assertEqual(backend["requested_backend"], "cutlass")
        self.assertEqual(
            backend["selected_vllm_kernel"],
            "FlashInferCutlassNvFp4LinearKernel",
        )
        self.assertIsNone(backend["reported_backend"])
        self.assertTrue(backend["expected_sm120_cutlass_signature_present"])
        self.assertEqual(backend["fallback_status"], "not_detected")
        self.assertTrue(backend["target_kernels"])
        self.assertFalse(
            any("cublas" in name.lower() for name in backend["target_kernels"])
        )

    def test_manifest_pins_clean_provenance_and_tracked_result_hash(self) -> None:
        self.assertFalse(self.manifest["git"]["dirty"])
        self.assertTrue(GIT_COMMIT.fullmatch(self.manifest["git"]["commit"]))
        self.assertTrue(SHA256.fullmatch(self.manifest["source_bundle_sha256"]))
        artifact = next(
            item
            for item in self.manifest["artifacts"]
            if item["kind"] == "normalized-replay-result"
        )
        path = ROOT / artifact["path"]
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            artifact["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
