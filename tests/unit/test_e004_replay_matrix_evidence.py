import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class E004ReplayMatrixEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "replay-matrix.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest-replay-matrix.json").read_text(encoding="utf-8")
        )

    def test_matrix_covers_the_frozen_cases_once(self) -> None:
        cases = self.results["cases"]
        coverage = {(item["layer"], item["projection"]) for item in cases}
        expected = {
            (layer, projection)
            for layer in (0, 18, 35)
            for projection in (
                "q_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        }
        self.assertEqual(coverage, expected)
        self.assertEqual(len(cases), 15)
        self.assertEqual(self.results["matrix"]["case_count"], 15)

    def test_all_case_invariants_pass_without_overstating_fused_cases(self) -> None:
        matrix = self.results["matrix"]
        self.assertTrue(matrix["all_cases_finite"])
        self.assertTrue(matrix["all_case_hashes_stable"])
        self.assertTrue(matrix["all_weights_byte_preserved"])
        self.assertTrue(matrix["all_weight_padding_zero"])
        self.assertTrue(matrix["all_scale_swizzles_candidate_exact"])
        self.assertEqual(matrix["distinct_output_sha256_count"], 15)

        scopes = [item["adapter_scope"] for item in self.results["cases"]]
        self.assertEqual(scopes.count("production_aligned_unfused"), 6)
        self.assertEqual(scopes.count("individual_fused_family_preflight"), 9)
        for item in self.results["cases"]:
            with self.subTest(layer=item["layer"], projection=item["projection"]):
                self.assertTrue(item["all_finite"])
                self.assertTrue(item["output_hash_stable"])
                self.assertTrue(item["weight_bytes_preserved"])
                self.assertEqual(item["weight_padding_bytes"], 0)
                self.assertTrue(item["scale_swizzle_candidate_byte_exact"])
                self.assertLessEqual(
                    item["activation_max_abs"],
                    item["checkpoint_calibrated_max_abs"],
                )
                self.assertTrue(SHA256.fullmatch(item["output_sha256"]))

    def test_backend_anchor_preserves_requested_selected_reported_observed_split(
        self,
    ) -> None:
        backend = self.results["backend_identity_anchor"]
        self.assertEqual(backend["requested_backend"], "cutlass")
        self.assertEqual(
            backend["selected_vllm_kernel"],
            "FlashInferCutlassNvFp4LinearKernel",
        )
        self.assertIsNone(backend["reported_backend"])
        self.assertTrue(backend["expected_sm120_cutlass_signature_present"])
        self.assertEqual(backend["fallback_status"], "not_detected")
        self.assertTrue(SHA256.fullmatch(backend["profiler_sha256"]))

    def test_manifest_pins_clean_commit_and_tracked_artifacts(self) -> None:
        self.assertFalse(self.manifest["git"]["dirty"])
        self.assertTrue(GIT_COMMIT.fullmatch(self.manifest["git"]["commit"]))
        self.assertTrue(SHA256.fullmatch(self.manifest["source_bundle_sha256"]))
        for artifact in self.manifest["artifacts"]:
            if artifact["kind"] == "raw-matrix-case":
                continue
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    artifact["sha256"],
                )

    def test_claim_boundary_keeps_real_activation_out_of_scope(self) -> None:
        self.assertIn(
            "does not establish real Qwen activation capture",
            self.results["claim_boundary"],
        )


if __name__ == "__main__":
    unittest.main()
