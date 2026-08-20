import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E003-synthetic-faults"


class E003EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "results.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest.json").read_text(encoding="utf-8")
        )
        self.execution_results = json.loads(
            (EXPERIMENT / "results-execution.json").read_text(encoding="utf-8")
        )
        self.execution_manifest = json.loads(
            (EXPERIMENT / "manifest-execution.json").read_text(encoding="utf-8")
        )
        self.heldout_results = json.loads(
            (EXPERIMENT / "results-heldout.json").read_text(encoding="utf-8")
        )
        self.heldout_manifest = json.loads(
            (EXPERIMENT / "manifest-heldout.json").read_text(encoding="utf-8")
        )

    def test_format_fault_slice_has_no_observed_classification_failures(self) -> None:
        self.assertEqual(self.results["slice"], "format_faults_v1")
        self.assertEqual(self.results["clean_artifacts_checked"], 3)
        self.assertEqual(self.results["clean_contract_evaluations"], 18)
        self.assertEqual(self.results["clean_false_rejects"], 0)
        self.assertEqual(self.results["faults_injected"], 6)
        self.assertEqual(self.results["faults_detected"], 6)
        self.assertEqual(self.results["false_accepts"], 0)
        self.assertEqual(self.results["localization_failures"], 0)
        self.assertEqual(self.results["reversibility_failures"], 0)
        self.assertEqual(self.results["slice_status"], "pass")
        self.assertEqual(self.results["decision"], "continue")

    def test_each_fault_is_synthetic_reversible_and_exactly_localized(self) -> None:
        expected_kinds = {
            "nibble_swap",
            "scale_index_shift",
            "block_scale_reversal",
            "global_scale_multiplier",
            "scale_layout_mislabel",
            "padding_corruption",
        }
        cases = self.results["fault_cases"]
        self.assertEqual({case["fault_kind"] for case in cases}, expected_kinds)
        for case in cases:
            with self.subTest(kind=case["fault_kind"]):
                self.assertEqual(case["label"], "synthetic")
                self.assertTrue(case["detected"])
                self.assertTrue(case["exact_localization"])
                self.assertTrue(case["reversible"])
                self.assertEqual(
                    case["expected_failed_contracts"],
                    case["observed_failed_contracts"],
                )

    def test_padding_control_is_detected_but_reconstruction_remains_exact(self) -> None:
        padding = next(
            case
            for case in self.results["fault_cases"]
            if case["fault_kind"] == "padding_corruption"
        )
        outcomes = {
            outcome["contract_id"]: outcome for outcome in padding["contract_outcomes"]
        }
        self.assertFalse(outcomes["scale_padding"]["passed"])
        self.assertEqual(outcomes["scale_padding"]["mismatch_count"], 1)
        self.assertTrue(outcomes["reconstruction"]["passed"])

    def test_manifest_separates_backend_fields_and_hashes_results(self) -> None:
        backend = self.manifest["backend"]
        self.assertEqual(backend["requested"], "cpu_oracle")
        self.assertEqual(backend["reported"], "cpu_oracle")
        self.assertIsNone(backend["observed_kernel"])
        for artifact in self.manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )

    def test_execution_evidence_slice_has_no_observed_classification_failures(
        self,
    ) -> None:
        results = self.execution_results
        self.assertEqual(results["slice"], "execution_evidence_faults_v1")
        self.assertEqual(results["clean_contract_evaluations"], 6)
        self.assertEqual(results["clean_false_rejects"], 0)
        self.assertEqual(results["faults_injected"], 5)
        self.assertEqual(results["faults_detected"], 5)
        self.assertEqual(results["false_accepts"], 0)
        self.assertEqual(results["localization_failures"], 0)
        self.assertEqual(results["reversibility_failures"], 0)
        self.assertEqual(results["slice_status"], "pass")
        self.assertEqual(results["decision"], "continue")

    def test_execution_faults_preserve_evidence_field_separation(self) -> None:
        cases = self.execution_results["fault_cases"]
        expected_kinds = {
            "stride_axis_permutation",
            "stride_gap",
            "requested_backend_mismatch",
            "reported_backend_mismatch",
            "observed_fallback_kernel",
        }
        self.assertEqual({case["fault_kind"] for case in cases}, expected_kinds)
        for case in cases:
            with self.subTest(kind=case["fault_kind"]):
                self.assertEqual(case["label"], "synthetic")
                self.assertTrue(case["detected"])
                self.assertTrue(case["exact_localization"])
                self.assertTrue(case["reversible"])
                self.assertEqual(
                    case["expected_failed_contracts"],
                    case["observed_failed_contracts"],
                )

        reported = next(
            case for case in cases if case["fault_kind"] == "reported_backend_mismatch"
        )
        self.assertEqual(
            reported["backend"]["clean"]["requested"],
            reported["backend"]["faulted"]["requested"],
        )
        self.assertEqual(
            reported["backend"]["clean"]["observed_kernels"],
            reported["backend"]["faulted"]["observed_kernels"],
        )

    def test_execution_manifest_hashes_results_and_marks_observations_synthetic(
        self,
    ) -> None:
        baseline = self.execution_manifest["synthetic_backend_baseline"]
        self.assertEqual(baseline["requested"], "cutlass")
        self.assertEqual(baseline["reported"], "cutlass")
        self.assertEqual(baseline["fallback_status"], "not_detected")
        execution = self.execution_manifest["execution_backend"]
        self.assertEqual(execution["requested"], "cpu_oracle")
        self.assertEqual(execution["reported"], "cpu_oracle")
        self.assertIsNone(execution["observed_kernel"])
        for artifact in self.execution_manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )

    def test_heldout_matrix_uses_frozen_thresholds_and_has_no_failures(self) -> None:
        results = self.heldout_results
        self.assertEqual(results["slice"], "packed_permutation_heldout_v1")
        self.assertEqual(results["evaluation_role"], "held_out")
        self.assertTrue(
            results["threshold_policy"]["frozen_before_matrix_construction"]
        )
        self.assertFalse(results["threshold_policy"]["tuned_on_held_out_cases"])
        self.assertEqual(results["clean_artifacts_checked"], 3)
        self.assertEqual(results["clean_contract_evaluations"], 18)
        self.assertEqual(results["clean_false_rejects"], 0)
        self.assertEqual(results["faults_injected"], 9)
        self.assertEqual(results["faults_detected"], 9)
        self.assertEqual(results["false_accepts"], 0)
        self.assertEqual(results["localization_failures"], 0)
        self.assertEqual(results["reversibility_failures"], 0)
        self.assertEqual(results["slice_status"], "pass")
        self.assertEqual(results["e003_status"], "complete")
        self.assertEqual(results["decision"], "continue")

    def test_heldout_permutations_are_synthetic_and_exactly_localized(self) -> None:
        cases = self.heldout_results["fault_cases"]
        self.assertEqual(
            {case["fault_kind"] for case in cases},
            {
                "packed_block_permutation",
                "packed_row_permutation",
                "packed_column_permutation",
            },
        )
        self.assertEqual({case["evaluation_role"] for case in cases}, {"held_out"})
        for case in cases:
            with self.subTest(artifact=case["artifact_id"], kind=case["fault_kind"]):
                self.assertEqual(case["label"], "synthetic")
                self.assertTrue(case["detected"])
                self.assertTrue(case["exact_localization"])
                self.assertTrue(case["reversible"])
                self.assertEqual(
                    case["expected_failed_contracts"],
                    ["packed_values", "reconstruction"],
                )
                self.assertEqual(
                    case["expected_failed_contracts"],
                    case["observed_failed_contracts"],
                )

    def test_heldout_manifest_hashes_results(self) -> None:
        self.assertEqual(
            self.heldout_manifest["slice"], "packed_permutation_heldout_v1"
        )
        for artifact in self.heldout_manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
