import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class E004RealActivationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result_path = EXPERIMENT / "real-activation-replay.json"
        self.manifest_path = EXPERIMENT / "manifest-real-activation-replay.json"
        self.result_text = self.result_path.read_text(encoding="utf-8")
        self.results = json.loads(self.result_text)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def test_result_is_one_scoped_real_activation_observation(self) -> None:
        self.assertEqual(self.results["status"], "pass")
        self.assertEqual(self.results["decision"], "continue")
        self.assertEqual(self.results["slice"], "real_activation_capture_replay_v1")
        case = self.results["capture"]["case"]
        self.assertEqual(case["layer"], 0)
        self.assertEqual(case["projection"], "o_proj")
        self.assertEqual(case["adapter_scope"], "production_aligned_unfused")
        self.assertEqual(case["activation_provenance"], "real_qwen_prefill")
        self.assertEqual(case["event_count"], 1)
        self.assertIn(
            "does not establish NVFP4 numerical correctness",
            self.results["claim_boundary"],
        )

    def test_input_identity_is_hashed_without_prompt_payload(self) -> None:
        identity = self.results["input_identity"]
        self.assertFalse(identity["token_ids_committed_in_result"])
        self.assertEqual(identity["token_count"], 9)
        self.assertTrue(SHA256.fullmatch(identity["token_ids_sha256"]))
        self.assertNotIn("token_ids", identity)
        self.assertNotIn("prompt_text", identity)
        self.assertNotIn("NVFP4 smoke test", self.result_text)

    def test_capture_transfer_metadata_is_explicit_and_preserved(self) -> None:
        capture = self.results["capture"]
        expected_fields = [
            "shape",
            "dtype",
            "stride",
            "storage_offset",
            "byte_length",
            "sha256",
        ]
        self.assertEqual(capture["metadata_preserved_fields"], expected_fields)
        self.assertTrue(capture["device_transfer_recorded"])
        for key in ("input_artifact", "captured_module_output_artifact"):
            artifact = capture[key]
            self.assertTrue(artifact["ignored"])
            self.assertEqual(artifact["preserved_fields"], expected_fields)
            self.assertEqual(
                artifact["device_transfer"],
                {"source": "cuda:0", "destination": "cpu"},
            )
            for field in expected_fields:
                self.assertEqual(
                    artifact["source_metadata"][field], artifact["tensor"][field]
                )
        activation = capture["input_artifact"]["tensor"]
        self.assertEqual(activation["shape"], [9, 4096])
        self.assertEqual(activation["dtype"], "bfloat16")
        self.assertEqual(activation["stride"], [4096, 1])

    def test_replay_is_stable_and_byte_exact_to_captured_module_output(self) -> None:
        replay = self.results["replay"]
        self.assertEqual(replay["repetitions"], 3)
        self.assertTrue(replay["synchronized"])
        self.assertTrue(replay["all_finite"])
        self.assertTrue(replay["output_hash_stable"])
        self.assertEqual(len(replay["output_sha256s"]), 3)
        self.assertEqual(len(set(replay["output_sha256s"])), 1)
        self.assertEqual(
            replay["bitwise_captured_module_output_matches"], [True, True, True]
        )
        self.assertEqual(replay["max_abs_error"], 0.0)
        self.assertEqual(replay["mean_abs_error"], 0.0)
        self.assertEqual(
            replay["captured_module_output_sha256"], replay["output_sha256s"][0]
        )

    def test_backend_identity_is_range_scoped_and_has_no_known_fallback(self) -> None:
        backend = self.results["backend"]
        self.assertEqual(backend["requested_backend"], "auto")
        self.assertEqual(
            backend["selected_vllm_kernel"],
            "FlashInferCutlassNvFp4LinearKernel",
        )
        self.assertIsNone(backend["reported_backend"])
        self.assertEqual(
            backend["target_nvtx_range"],
            "e004:real_activation:layer_00:o_proj:nvfp4_gemm",
        )
        self.assertTrue(backend["expected_sm120_cutlass_signature_present"])
        self.assertEqual(backend["fallback_status"], "not_detected")
        self.assertTrue(backend["target_kernels"])
        self.assertTrue(
            any(
                "MainloopSm120TmaWarpSpecializedBlockScaled" in name
                and "cutlass::float_e2m1_t" in name
                and "SM120_16x8x64_TN_VS" in name
                for name in backend["target_kernels"]
            )
        )
        self.assertFalse(
            any("cublas" in name.lower() for name in backend["target_kernels"])
        )

    def test_manifest_pins_clean_provenance_and_dependencies(self) -> None:
        self.assertFalse(self.manifest["git"]["dirty"])
        self.assertTrue(GIT_COMMIT.fullmatch(self.manifest["git"]["commit"]))
        self.assertTrue(SHA256.fullmatch(self.manifest["source_bundle_sha256"]))
        self.assertTrue(self.manifest["model"]["complete_snapshot_acquired"])
        dependencies = self.manifest["dependencies"]
        self.assertEqual(len(dependencies), 2)
        for dependency in dependencies:
            path = ROOT / dependency["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                dependency["sha256"],
            )

    def test_manifest_hashes_tracked_and_available_local_artifacts(self) -> None:
        for artifact in self.manifest["artifacts"]:
            self.assertTrue(SHA256.fullmatch(artifact["sha256"]))
            path = ROOT / artifact["path"]
            if artifact["ignored"]:
                if path.is_file():
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        artifact["sha256"],
                    )
                continue
            self.assertTrue(path.is_file())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
