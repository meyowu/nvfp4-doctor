import unittest

from nvfp4_doctor.contracts import (
    EVIDENCE_CONTRACT_SPECS,
    EvidenceContractId,
    ExecutionEvidence,
    contiguous_stride,
    evaluate_execution_evidence,
    is_row_major_contiguous,
)
from nvfp4_doctor.env import BackendEvidence, FallbackStatus, TensorMetadata
from nvfp4_doctor.faults import (
    EvidenceFaultKind,
    inject_observed_fallback_kernel,
    inject_reported_backend_mismatch,
    inject_requested_backend_mismatch,
    inject_stride_axis_permutation,
    inject_stride_gap,
    revert_evidence_fault,
)

CLEAN_KERNEL = (
    "e003:nvfp4_gemm/cutlass::device_kernel<"
    "MainloopSm120TmaWarpSpecializedBlockScaled,"
    "cutlass::float_e2m1_t,SM120_16x8x64_TN_VS>"
)
FALLBACK_KERNEL = "e003:nvfp4_gemm/cublasGemmEx"


def _clean_evidence() -> ExecutionEvidence:
    return ExecutionEvidence(
        tensor=TensorMetadata(
            name="packed_input",
            logical_shape=(4, 16),
            physical_shape=(4, 8),
            dtype="uint8",
            stride=(8, 1),
            device="cuda:0",
        ),
        backend=BackendEvidence(
            requested_format="nvfp4",
            requested_backend="cutlass",
            reported_backend="cutlass",
            observed_kernels=(CLEAN_KERNEL,),
            fallback_status=FallbackStatus.NOT_DETECTED,
            profiler_artifact_sha256=None,
        ),
    )


class ExecutionEvidenceFaultTests(unittest.TestCase):
    def test_every_evidence_contract_has_an_auditable_exact_specification(self) -> None:
        self.assertEqual(len(EVIDENCE_CONTRACT_SPECS), 6)
        for spec in EVIDENCE_CONTRACT_SPECS:
            with self.subTest(contract=spec.contract_id):
                self.assertTrue(spec.domain)
                self.assertTrue(spec.preconditions)
                self.assertTrue(spec.invariant)
                self.assertTrue(spec.metric)
                self.assertEqual(spec.threshold, 0)
                self.assertTrue(spec.limitations)

    def test_clean_evidence_passes_without_conflating_identity_fields(self) -> None:
        clean = _clean_evidence()
        outcomes = evaluate_execution_evidence(clean, clean)
        self.assertTrue(all(outcome.passed for outcome in outcomes))
        self.assertEqual(contiguous_stride((4, 8)), (8, 1))
        self.assertTrue(is_row_major_contiguous(clean.tensor))

    def test_all_positive_controls_are_detected_localized_and_reversible(self) -> None:
        clean = _clean_evidence()
        cases = (
            (
                inject_stride_axis_permutation(clean),
                {
                    EvidenceContractId.TENSOR_STRIDE,
                    EvidenceContractId.TENSOR_CONTIGUITY,
                },
            ),
            (
                inject_stride_gap(clean, 8),
                {
                    EvidenceContractId.TENSOR_STRIDE,
                    EvidenceContractId.TENSOR_CONTIGUITY,
                },
            ),
            (
                inject_requested_backend_mismatch(clean, "cublas"),
                {EvidenceContractId.REQUESTED_BACKEND},
            ),
            (
                inject_reported_backend_mismatch(clean, "cublas"),
                {EvidenceContractId.REPORTED_BACKEND},
            ),
            (
                inject_observed_fallback_kernel(clean, FALLBACK_KERNEL),
                {
                    EvidenceContractId.OBSERVED_KERNELS,
                    EvidenceContractId.FALLBACK_STATUS,
                },
            ),
        )
        for injection, expected_failures in cases:
            with self.subTest(kind=injection.kind):
                outcomes = evaluate_execution_evidence(
                    injection.clean, injection.faulted
                )
                observed_failures = {
                    outcome.spec.contract_id
                    for outcome in outcomes
                    if not outcome.passed
                }
                self.assertEqual(observed_failures, expected_failures)
                self.assertEqual(revert_evidence_fault(injection), clean)
                self.assertEqual(injection.label, "synthetic")

    def test_reported_mismatch_does_not_rewrite_request_or_observation(self) -> None:
        injection = inject_reported_backend_mismatch(_clean_evidence(), "cublas")
        self.assertEqual(
            injection.clean.backend.requested_backend,
            injection.faulted.backend.requested_backend,
        )
        self.assertEqual(
            injection.clean.backend.observed_kernels,
            injection.faulted.backend.observed_kernels,
        )
        outcomes = {
            outcome.spec.contract_id: outcome
            for outcome in evaluate_execution_evidence(
                injection.clean, injection.faulted
            )
        }
        self.assertFalse(outcomes[EvidenceContractId.REPORTED_BACKEND].passed)
        self.assertTrue(outcomes[EvidenceContractId.REQUESTED_BACKEND].passed)
        self.assertTrue(outcomes[EvidenceContractId.OBSERVED_KERNELS].passed)

    def test_fault_preconditions_reject_noop_or_invalid_parameters(self) -> None:
        clean = _clean_evidence()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            inject_stride_gap(clean, 0)
        with self.assertRaisesRegex(ValueError, "differ from the request"):
            inject_requested_backend_mismatch(clean, "cutlass")
        with self.assertRaisesRegex(ValueError, "differ from the report"):
            inject_reported_backend_mismatch(clean, "cutlass")
        with self.assertRaisesRegex(ValueError, "differ from clean"):
            inject_observed_fallback_kernel(clean, CLEAN_KERNEL)

    def test_fault_kind_values_are_stable_labels(self) -> None:
        self.assertEqual(
            {kind.value for kind in EvidenceFaultKind},
            {
                "stride_axis_permutation",
                "stride_gap",
                "requested_backend_mismatch",
                "reported_backend_mismatch",
                "observed_fallback_kernel",
            },
        )


if __name__ == "__main__":
    unittest.main()
