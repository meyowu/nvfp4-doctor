import hashlib
import json
import re
import unittest
from pathlib import Path

from nvfp4_doctor.capture import E004_UNFUSED_REAL_ACTIVATION_CASES
from scripts.finalize_e004_real_activation_matrix import (
    DEPENDENCY_PATHS,
    SOURCE_PATHS,
)

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
RESULT = EXPERIMENT / "real-activation-unfused-matrix.json"
MANIFEST = EXPERIMENT / "manifest-real-activation-unfused-matrix.json"
PRIOR_RESULT = EXPERIMENT / "real-activation-replay.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class E004RealActivationMatrixEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RESULT.is_file() or not MANIFEST.is_file():
            state = (ROOT / "research-state.md").read_text(encoding="utf-8")
            if "representative_unfused_real_activation_matrix_pass" in state:
                raise AssertionError(
                    "research state declares matrix evidence that is missing"
                )
            raise unittest.SkipTest("profiled matrix evidence has not been generated")
        cls.result_text = RESULT.read_text(encoding="utf-8")
        cls.result = json.loads(cls.result_text)
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_result_has_exact_six_case_scope(self) -> None:
        self.assertEqual(self.result["status"], "pass")
        self.assertEqual(self.result["decision"], "continue")
        self.assertEqual(
            self.result["slice"],
            "representative_unfused_real_activation_replay_matrix_v1",
        )
        self.assertEqual(
            [case["case_id"] for case in self.result["cases"]],
            [case.case_id for case in E004_UNFUSED_REAL_ACTIVATION_CASES],
        )
        self.assertEqual(self.result["matrix"]["evaluated_range_count"], 6)
        self.assertEqual(self.result["matrix"]["observed_target_range_count"], 6)
        self.assertTrue(self.result["matrix"]["layer_00_o_proj_regression_match"])

    def test_input_identity_is_hashed_without_prompt_payload(self) -> None:
        identity = self.result["input_identity"]
        self.assertFalse(identity["token_ids_committed_in_result"])
        self.assertEqual(identity["token_count"], 9)
        self.assertTrue(SHA256.fullmatch(identity["token_ids_sha256"]))
        self.assertNotIn("token_ids", identity)
        self.assertNotIn("prompt_text", identity)
        self.assertNotIn("NVFP4 smoke test", self.result_text)

    def test_every_capture_and_replay_preserves_the_frozen_contract(self) -> None:
        input_hashes: set[str] = set()
        output_hashes: set[str] = set()
        artifact_paths: set[str] = set()
        for spec, case in zip(
            E004_UNFUSED_REAL_ACTIVATION_CASES,
            self.result["cases"],
            strict=True,
        ):
            capture = case["capture"]
            input_tensor = capture["input_artifact"]["tensor"]
            module_output = capture["captured_module_output_artifact"]["tensor"]
            self.assertEqual(input_tensor["shape"], list(spec.input_shape(9)))
            self.assertEqual(input_tensor["stride"], [spec.input_width, 1])
            self.assertEqual(module_output["shape"], list(spec.output_shape(9)))
            self.assertEqual(module_output["stride"], [4096, 1])
            input_hashes.add(input_tensor["sha256"])
            output_hashes.add(module_output["sha256"])
            replay = case["replay"]
            self.assertTrue(replay["all_finite"])
            self.assertTrue(replay["output_hash_stable"])
            self.assertTrue(replay["logical_byte_exact_captured_input_match"])
            self.assertEqual(
                replay["reconstructed_activation_metadata"],
                capture["input_artifact"]["source_metadata"],
            )
            self.assertEqual(
                replay["logical_byte_exact_captured_module_output_matches"],
                [True, True, True],
            )
            self.assertEqual(replay["max_abs_error"], 0.0)
            self.assertEqual(replay["mean_abs_error"], 0.0)
            for artifact in (
                capture["input_artifact"],
                capture["captured_module_output_artifact"],
                replay["replay_output_artifact"],
            ):
                self.assertTrue(artifact["ignored"])
                self.assertNotIn(artifact["path"], artifact_paths)
                artifact_paths.add(artifact["path"])
                for field in artifact["preserved_fields"]:
                    self.assertEqual(
                        artifact["source_metadata"][field], artifact["tensor"][field]
                    )
        self.assertEqual(len(input_hashes), 6)
        self.assertEqual(len(output_hashes), 6)
        self.assertEqual(len(artifact_paths), 18)

    def test_each_backend_range_has_quantization_cutlass_and_no_known_fallback(
        self,
    ) -> None:
        catalog = {
            entry["kernel_id"]: entry["name"]
            for entry in self.result["backend"]["kernel_catalog"]
        }
        for kernel_id, name in catalog.items():
            self.assertEqual(
                kernel_id, hashlib.sha256(name.encode("utf-8")).hexdigest()
            )
        for case in self.result["cases"]:
            evidence = case["backend_range"]
            names = [catalog[kernel_id] for kernel_id in evidence["target_kernel_ids"]]
            set_digest = hashlib.sha256()
            for name in sorted(names):
                set_digest.update(name.encode("utf-8"))
                set_digest.update(b"\0")
            self.assertEqual(
                set_digest.hexdigest(), evidence["target_kernel_set_sha256"]
            )
            self.assertTrue(evidence["expected_sm120_cutlass_signature_present"])
            self.assertTrue(evidence["activation_quantization_signature_present"])
            self.assertEqual(evidence["fallback_status"], "not_detected")
            self.assertTrue(any("MainloopSm120" in name for name in names))
            self.assertTrue(any("cvt_fp16_to_fp4" in name for name in names))
            self.assertFalse(any("cublas" in name.lower() for name in names))

    def test_layer_zero_overlap_matches_the_prior_real_case(self) -> None:
        prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
        first = self.result["cases"][0]
        self.assertEqual(
            first["capture"]["input_artifact"]["tensor"]["sha256"],
            prior["capture"]["input_artifact"]["tensor"]["sha256"],
        )
        self.assertEqual(
            first["replay"]["captured_module_output_sha256"],
            prior["replay"]["captured_module_output_sha256"],
        )

    def test_manifest_pins_clean_provenance_dependencies_and_artifacts(self) -> None:
        self.assertFalse(self.manifest["git"]["dirty"])
        self.assertTrue(GIT_COMMIT.fullmatch(self.manifest["git"]["commit"]))
        self.assertTrue(SHA256.fullmatch(self.manifest["source_bundle_sha256"]))
        source_digest = hashlib.sha256()
        for path in SOURCE_PATHS:
            source_digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update(path.read_bytes())
            source_digest.update(b"\0")
        self.assertEqual(
            source_digest.hexdigest(), self.manifest["source_bundle_sha256"]
        )
        self.assertEqual(len(self.manifest["dependencies"]), 6)
        self.assertEqual(
            [
                (dependency["kind"], dependency["path"])
                for dependency in self.manifest["dependencies"]
            ],
            [
                (kind, path.relative_to(ROOT).as_posix())
                for kind, path in DEPENDENCY_PATHS
            ],
        )
        for dependency in self.manifest["dependencies"]:
            path = ROOT / dependency["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), dependency["sha256"]
            )
        self.assertEqual(len(self.manifest["artifacts"]), 21)
        self.assertEqual(
            sum(not artifact["ignored"] for artifact in self.manifest["artifacts"]),
            1,
        )
        for artifact in self.manifest["artifacts"]:
            self.assertTrue(SHA256.fullmatch(artifact["sha256"]))
            path = ROOT / artifact["path"]
            if artifact["ignored"] and not path.is_file():
                continue
            self.assertTrue(path.is_file())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
            )

    def test_claim_boundary_keeps_gate_two_open(self) -> None:
        claim = self.result["claim_boundary"]
        for limitation in (
            "NVFP4 numerical correctness",
            "prompt diversity",
            "fused qkv_proj or gate_up_proj coverage",
            "final-logit or model quality",
            "completion of Gate 2",
        ):
            self.assertIn(limitation, claim)


if __name__ == "__main__":
    unittest.main()
