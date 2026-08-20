"""Backend-observation adapters that do not import GPU frameworks."""

from .nsys import (
    E001_GEMM_RANGE,
    NsightEvidenceError,
    NsightKernelEvidence,
    assess_e001_fallback,
    assess_range_fallback,
    attach_kernel_evidence,
    expected_sm120_cutlass_present,
    extract_kernel_evidence,
    kernels_in_nvtx_range,
    parse_cuda_gpu_kernel_summary,
    sha256_file,
)

__all__ = [
    "E001_GEMM_RANGE",
    "NsightEvidenceError",
    "NsightKernelEvidence",
    "assess_e001_fallback",
    "assess_range_fallback",
    "attach_kernel_evidence",
    "expected_sm120_cutlass_present",
    "extract_kernel_evidence",
    "kernels_in_nvtx_range",
    "parse_cuda_gpu_kernel_summary",
    "sha256_file",
]
