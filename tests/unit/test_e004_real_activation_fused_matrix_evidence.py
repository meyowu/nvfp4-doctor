import hashlib
import json
import re
import unittest
from pathlib import Path

from nvfp4_doctor.capture.e004_fused import E004_FUSED_REAL_ACTIVATION_CASES
from scripts.finalize_e004_real_activation_fused_matrix import (
    DEPENDENCY_PATHS,
    SOURCE_PATHS,
)

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
RESULT = EXPERIMENT / "real-activation-fused-matrix.json"
MANIFEST = EXPERIMENT / "manifest-real-activation-fused-matrix.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class E004RealActivationFusedMatrixEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RESULT.is_file() or not MANIFEST.is_file():
            state = (ROOT / "research-state.md").read_text(encoding="utf-8")
            if "representative_fused_real_activation_matrix_pass" in state:
                raise AssertionError(
                    "research state declares fused matrix evidence that is missing"
                )
            raise unittest.SkipTest("profiled fused matrix evidence is not generated")
        cls.result_text = RESULT.read_text(encoding="utf-8")
        cls.result = json.loads(cls.result_text)
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_result_is_the_exact_gate_two_fused_matrix(self) -> None:
        self.assertEqual(self.result["status"], "pass")
        self.assertEqual(self.result["decision"], "go")
        self.assertEqual(
            self.result["slice"],
            "representative_fused_real_activation_replay_matrix_v1",
        )
        self.assertEqual(
            [case["case_id"] for case in self.result["cases"]],
            [case.case_id for case in E004_FUSED_REAL_ACTIVATION_CASES],
        )
        matrix = self.result["matrix"]
        self.assertEqual(matrix["case_count"], 6)
        self.assertEqual(matrix["evaluated_range_count"], 6)
        self.assertEqual(matrix["observed_target_range_count"], 6)
        self.assertTrue(matrix["request_identity_regression_match"])
        self.assertEqual(matrix["source_overlap_regression_count"], 9)
        self.assertTrue(matrix["source_overlap_regression_match"])

    def test_request_identity_is_hashed_and_matches_the_unfused_dependency(
        self,
    ) -> None:
        identity = self.result["input_identity"]
        self.assertFalse(identity["token_ids_committed_in_result"])
        self.assertEqual(identity["token_count"], 9)
        self.assertTrue(SHA256.fullmatch(identity["token_ids_sha256"]))
        self.assertNotIn("token_ids", identity)
        self.assertNotIn("prompt_text", identity)
        unfused = json.loads(
            (EXPERIMENT / "real-activation-unfused-matrix.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(identity, unfused["input_identity"])

    def test_component_fusion_capture_and_replay_contracts_pass(self) -> None:
        input_hashes: set[str] = set()
        output_hashes: set[str] = set()
        artifact_paths: set[str] = set()
        overlap_count = 0
        for spec, case in zip(
            E004_FUSED_REAL_ACTIVATION_CASES,
            self.result["cases"],
            strict=True,
        ):
            self.assertEqual(case["module_class"], spec.module_class)
            source = case["source_construction"]
            self.assertEqual(
                source["component_order"], list(spec.component_projections)
            )
            self.assertEqual(
                source["component_row_boundaries"],
                list(spec.component_row_boundaries),
            )
            self.assertTrue(source["source_snapshot_verified"])
            for boundary, component in zip(
                spec.component_boundaries, source["components"], strict=True
            ):
                self.assertEqual(component["projection"], boundary.projection)
                self.assertEqual(
                    component["row_range"], [boundary.row_start, boundary.row_end]
                )
                self.assertTrue(
                    SHA256.fullmatch(component["prepared_packed_weight_sha256"])
                )
                self.assertTrue(
                    SHA256.fullmatch(component["prepared_runtime_weight_scale_sha256"])
                )
                regression = component["replay_matrix_regression_match"]
                if boundary.projection in {"q_proj", "gate_proj", "up_proj"}:
                    self.assertTrue(regression)
                    overlap_count += 1
                else:
                    self.assertIsNone(regression)
                self.assertEqual(
                    set(component["source_tensors"]),
                    {"input_scale", "weight", "weight_scale", "weight_scale_2"},
                )
                for tensor in component["source_tensors"].values():
                    self.assertTrue(SHA256.fullmatch(tensor["sha256"]))
                    self.assertTrue(SHA256.fullmatch(tensor["shard_sha256"]))
            self.assertEqual(
                len({component["input_scale"] for component in source["components"]}),
                1,
            )
            self.assertEqual(
                len(
                    {component["weight_scale_2"] for component in source["components"]}
                ),
                1,
            )

            runtime = case["runtime_projection"]
            self.assertEqual(runtime["tp_size"], 1)
            self.assertEqual(runtime["tp_rank"], 0)
            self.assertFalse(runtime["gather_output"])
            self.assertEqual(
                runtime["logical_widths"], list(spec.component_output_widths)
            )
            self.assertTrue(runtime["global_scale_reduction_matches"])
            self.assertEqual(runtime["weights_padding_cols"], 0)
            self.assertEqual(
                len(runtime["component_bindings"]), len(spec.component_boundaries)
            )
            for binding in runtime["component_bindings"]:
                self.assertTrue(binding["checkpoint_weight_match"])
                self.assertTrue(binding["independent_scale_swizzle_match"])

            capture = case["capture"]
            input_tensor = capture["input_artifact"]["tensor"]
            output_tensor = capture["captured_module_output_artifact"]["tensor"]
            self.assertEqual(input_tensor["shape"], list(spec.input_shape(9)))
            self.assertEqual(output_tensor["shape"], list(spec.output_shape(9)))
            input_hashes.add(input_tensor["sha256"])
            output_hashes.add(output_tensor["sha256"])
            replay = case["replay"]
            self.assertTrue(replay["all_finite"])
            self.assertTrue(replay["output_hash_stable"])
            self.assertEqual(
                replay["logical_byte_exact_captured_module_output_matches"],
                [True, True, True],
            )
            self.assertEqual(replay["max_abs_error"], 0.0)
            self.assertEqual(replay["mean_abs_error"], 0.0)
            self.assertEqual(
                len(replay["component_output_slices"]),
                len(spec.component_boundaries),
            )
            for component_slice in replay["component_output_slices"]:
                self.assertEqual(component_slice["logical_matches"], [True] * 3)
                self.assertEqual(len(set(component_slice["replay_sha256s"])), 1)
                self.assertEqual(
                    component_slice["captured_sha256"],
                    component_slice["replay_sha256s"][0],
                )
            for artifact in (
                capture["input_artifact"],
                capture["captured_module_output_artifact"],
                replay["replay_output_artifact"],
            ):
                self.assertTrue(artifact["ignored"])
                self.assertNotIn(artifact["path"], artifact_paths)
                artifact_paths.add(artifact["path"])
        self.assertEqual(overlap_count, 9)
        self.assertEqual(len(input_hashes), 6)
        self.assertEqual(len(output_hashes), 6)
        self.assertEqual(len(artifact_paths), 18)

    def test_every_exact_range_has_quantization_cutlass_and_no_known_fallback(
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
            digest = hashlib.sha256()
            for name in sorted(names):
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
            self.assertEqual(digest.hexdigest(), evidence["target_kernel_set_sha256"])
            self.assertTrue(evidence["expected_sm120_cutlass_signature_present"])
            self.assertTrue(evidence["activation_quantization_signature_present"])
            self.assertEqual(evidence["fallback_status"], "not_detected")
            self.assertTrue(any("MainloopSm120" in name for name in names))
            self.assertTrue(any("cvt_fp16_to_fp4" in name for name in names))
            self.assertFalse(any("cublas" in name.lower() for name in names))

    def test_manifest_binds_clean_provenance_dependencies_and_artifacts(self) -> None:
        self.assertFalse(self.manifest["git"]["dirty"])
        self.assertTrue(GIT_COMMIT.fullmatch(self.manifest["git"]["commit"]))
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
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
            )

    def test_claim_boundary_completes_only_the_bounded_gate(self) -> None:
        claim = self.result["claim_boundary"]
        self.assertIn("bounded Gate 2", claim)
        for limitation in (
            "NVFP4 numerical correctness",
            "separately executed q_proj",
            "prompt diversity",
            "final-logit or model quality",
            "performance",
            "cross-backend agreement",
            "high-precision checkpoint",
            "generalization beyond the pinned cases",
        ):
            self.assertIn(limitation, claim)


if __name__ == "__main__":
    unittest.main()
