"""Synthetic faults for tensor metadata and backend identity evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from nvfp4_doctor.contracts import ExecutionEvidence
from nvfp4_doctor.env import BackendEvidence, FallbackStatus, TensorMetadata

EvidenceFaultParameter = tuple[str, int | str]


class EvidenceFaultKind(StrEnum):
    STRIDE_AXIS_PERMUTATION = "stride_axis_permutation"
    STRIDE_GAP = "stride_gap"
    REQUESTED_BACKEND_MISMATCH = "requested_backend_mismatch"
    REPORTED_BACKEND_MISMATCH = "reported_backend_mismatch"
    OBSERVED_FALLBACK_KERNEL = "observed_fallback_kernel"


@dataclass(frozen=True, slots=True)
class EvidenceFaultInjection:
    kind: EvidenceFaultKind
    clean: ExecutionEvidence
    faulted: ExecutionEvidence
    parameters: tuple[EvidenceFaultParameter, ...]
    label: str = "synthetic"

    def __post_init__(self) -> None:
        if self.clean == self.faulted:
            raise ValueError("a positive-control fault must change the evidence")


def _replace_tensor(
    evidence: ExecutionEvidence, tensor: TensorMetadata
) -> ExecutionEvidence:
    return replace(evidence, tensor=tensor)


def _replace_backend(
    evidence: ExecutionEvidence, backend: BackendEvidence
) -> ExecutionEvidence:
    return replace(evidence, backend=backend)


def inject_stride_axis_permutation(
    evidence: ExecutionEvidence,
) -> EvidenceFaultInjection:
    if len(evidence.tensor.stride) < 2:
        raise ValueError("stride-axis permutation requires rank of at least two")
    faulted_tensor = replace(evidence.tensor, stride=evidence.tensor.stride[::-1])
    return EvidenceFaultInjection(
        EvidenceFaultKind.STRIDE_AXIS_PERMUTATION,
        evidence,
        _replace_tensor(evidence, faulted_tensor),
        (("rank", len(evidence.tensor.stride)),),
    )


def inject_stride_gap(evidence: ExecutionEvidence, gap: int) -> EvidenceFaultInjection:
    if isinstance(gap, bool) or not isinstance(gap, int) or gap <= 0:
        raise ValueError("gap must be a positive integer")
    stride = (evidence.tensor.stride[0] + gap, *evidence.tensor.stride[1:])
    faulted_tensor = replace(evidence.tensor, stride=stride)
    return EvidenceFaultInjection(
        EvidenceFaultKind.STRIDE_GAP,
        evidence,
        _replace_tensor(evidence, faulted_tensor),
        (("gap", gap),),
    )


def inject_requested_backend_mismatch(
    evidence: ExecutionEvidence, backend: str
) -> EvidenceFaultInjection:
    if not backend.strip() or backend == evidence.backend.requested_backend:
        raise ValueError("backend must be non-empty and differ from the request")
    faulted_backend = replace(evidence.backend, requested_backend=backend)
    return EvidenceFaultInjection(
        EvidenceFaultKind.REQUESTED_BACKEND_MISMATCH,
        evidence,
        _replace_backend(evidence, faulted_backend),
        (("backend", backend),),
    )


def inject_reported_backend_mismatch(
    evidence: ExecutionEvidence, backend: str
) -> EvidenceFaultInjection:
    if not backend.strip() or backend == evidence.backend.reported_backend:
        raise ValueError("backend must be non-empty and differ from the report")
    faulted_backend = replace(evidence.backend, reported_backend=backend)
    return EvidenceFaultInjection(
        EvidenceFaultKind.REPORTED_BACKEND_MISMATCH,
        evidence,
        _replace_backend(evidence, faulted_backend),
        (("backend", backend),),
    )


def inject_observed_fallback_kernel(
    evidence: ExecutionEvidence, kernel: str
) -> EvidenceFaultInjection:
    if not kernel.strip() or kernel in evidence.backend.observed_kernels:
        raise ValueError("kernel must be non-empty and differ from clean observations")
    faulted_backend = replace(
        evidence.backend,
        observed_kernels=(kernel,),
        fallback_status=FallbackStatus.DETECTED,
    )
    return EvidenceFaultInjection(
        EvidenceFaultKind.OBSERVED_FALLBACK_KERNEL,
        evidence,
        _replace_backend(evidence, faulted_backend),
        (("kernel", kernel),),
    )


def revert_evidence_fault(injection: EvidenceFaultInjection) -> ExecutionEvidence:
    """Restore the retained immutable evidence snapshot exactly."""
    return injection.clean
