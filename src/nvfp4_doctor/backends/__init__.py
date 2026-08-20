"""Backend-observation adapters that do not import GPU frameworks."""

from .nsys import (
    NsightEvidenceError,
    NsightKernelEvidence,
    attach_kernel_evidence,
    extract_kernel_evidence,
    parse_cuda_gpu_kernel_summary,
    sha256_file,
)

__all__ = [
    "NsightEvidenceError",
    "NsightKernelEvidence",
    "attach_kernel_evidence",
    "extract_kernel_evidence",
    "parse_cuda_gpu_kernel_summary",
    "sha256_file",
]
