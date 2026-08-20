import hashlib
import tempfile
import unittest
from pathlib import Path

from nvfp4_doctor.backends import (
    E001_GEMM_RANGE,
    NsightEvidenceError,
    NsightKernelEvidence,
    assess_e001_fallback,
    attach_kernel_evidence,
    extract_kernel_evidence,
    parse_cuda_gpu_kernel_summary,
)
from nvfp4_doctor.env import EnvironmentManifest, FallbackStatus

CSV = """Time (%),Total Time (ns),Instances,Name
60.0,600,1,"void cutlass::device_kernel<sm120>()"
30.0,300,2,"void tensorrt_llm::kernels::quantize_with_block_size()"
10.0,100,1,"void cutlass::device_kernel<sm120>()"
"""

EXPECTED_CUTLASS = (
    f"{E001_GEMM_RANGE}/void cutlass::device_kernel<"
    "MainloopSm120TmaWarpSpecializedBlockScaled, cutlass::float_e2m1_t, "
    "SM120_16x8x64_TN_VS>()"
)


class NsightEvidenceTests(unittest.TestCase):
    def test_parses_unique_kernel_names_in_report_order(self) -> None:
        self.assertEqual(
            parse_cuda_gpu_kernel_summary(CSV),
            (
                "void cutlass::device_kernel<sm120>()",
                "void tensorrt_llm::kernels::quantize_with_block_size()",
            ),
        )

    def test_ignores_nsys_status_preamble(self) -> None:
        payload = "Using existing SQLite export\nProcessing report\n" + CSV
        self.assertEqual(len(parse_cuda_gpu_kernel_summary(payload)), 2)

    def test_rejects_missing_name_column(self) -> None:
        with self.assertRaisesRegex(NsightEvidenceError, "Name column"):
            parse_cuda_gpu_kernel_summary("Time (%),Instances\n100,1\n")

    def test_rejects_empty_kernel_summary(self) -> None:
        with self.assertRaisesRegex(NsightEvidenceError, "no CUDA kernels"):
            parse_cuda_gpu_kernel_summary("Name\n")

    def test_hashes_report_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "capture.nsys-rep"
            report.write_bytes(b"profiler evidence")
            evidence = extract_kernel_evidence(report, CSV)
        self.assertEqual(
            evidence.report_sha256,
            hashlib.sha256(b"profiler evidence").hexdigest(),
        )

    def test_attach_preserves_unknown_backend_conclusions(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "e001_manifest_v1.json"
        manifest = EnvironmentManifest.from_path(fixture)
        evidence = NsightKernelEvidence("a" * 64, ("kernel",))
        updated = attach_kernel_evidence(manifest, evidence, Path("capture.nsys-rep"))
        self.assertIsNone(updated.backend.reported_backend)
        self.assertEqual(
            updated.backend.fallback_status, manifest.backend.fallback_status
        )
        self.assertEqual(updated.backend.observed_kernels, ("kernel",))
        self.assertEqual(updated.artifacts[-1].sha256, "a" * 64)

    def test_expected_range_scoped_cutlass_signature_detects_no_fallback(self) -> None:
        observed = (
            EXPECTED_CUTLASS,
            "e001:unquantized_reference/void cublasGemmEx()",
        )
        self.assertEqual(assess_e001_fallback(observed), FallbackStatus.NOT_DETECTED)

    def test_known_fallback_inside_target_range_is_detected(self) -> None:
        observed = (f"{E001_GEMM_RANGE}/void cublasGemmEx()",)
        self.assertEqual(assess_e001_fallback(observed), FallbackStatus.DETECTED)

    def test_expected_signature_outside_target_range_remains_unknown(self) -> None:
        observed = (EXPECTED_CUTLASS.removeprefix(f"{E001_GEMM_RANGE}/"),)
        self.assertEqual(assess_e001_fallback(observed), FallbackStatus.UNKNOWN)

    def test_unrecognized_target_kernel_remains_unknown(self) -> None:
        observed = (f"{E001_GEMM_RANGE}/void future_nvfp4_kernel()",)
        self.assertEqual(assess_e001_fallback(observed), FallbackStatus.UNKNOWN)

    def test_attach_records_range_scoped_assessment_without_reported_backend(
        self,
    ) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "e001_manifest_v1.json"
        manifest = EnvironmentManifest.from_path(fixture)
        evidence = NsightKernelEvidence("b" * 64, (EXPECTED_CUTLASS,))
        updated = attach_kernel_evidence(manifest, evidence, Path("capture.nsys-rep"))
        self.assertIsNone(updated.backend.reported_backend)
        self.assertEqual(updated.backend.fallback_status, FallbackStatus.NOT_DETECTED)


if __name__ == "__main__":
    unittest.main()
