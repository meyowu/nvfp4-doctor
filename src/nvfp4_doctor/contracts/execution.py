"""Exact contracts for tensor-layout metadata and backend identity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nvfp4_doctor.env import BackendEvidence, TensorMetadata


class EvidenceContractId(StrEnum):
    TENSOR_STRIDE = "tensor_stride"
    TENSOR_CONTIGUITY = "tensor_contiguity"
    REQUESTED_BACKEND = "requested_backend"
    REPORTED_BACKEND = "reported_backend"
    OBSERVED_KERNELS = "observed_kernels"
    FALLBACK_STATUS = "fallback_status"


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Tensor metadata and distinct requested, reported, and observed facts."""

    tensor: TensorMetadata
    backend: BackendEvidence


@dataclass(frozen=True, slots=True)
class EvidenceContractSpec:
    contract_id: EvidenceContractId
    domain: str
    preconditions: str
    invariant: str
    metric: str
    threshold: int
    limitations: str


@dataclass(frozen=True, slots=True)
class EvidenceContractOutcome:
    spec: EvidenceContractSpec
    mismatch_count: int
    details: str

    @property
    def passed(self) -> bool:
        return self.mismatch_count <= self.spec.threshold


EVIDENCE_CONTRACT_SPECS = (
    EvidenceContractSpec(
        EvidenceContractId.TENSOR_STRIDE,
        "Two metadata records for the same physical tensor storage.",
        "Name, shapes, dtype, device, and stride rank describe the same tensor.",
        "Every recorded stride component is identical.",
        "Number of unequal or missing stride components.",
        0,
        "Checks recorded metadata; it does not inspect a framework tensor.",
    ),
    EvidenceContractSpec(
        EvidenceContractId.TENSOR_CONTIGUITY,
        "Row-major contiguity implied by physical shape and recorded stride.",
        "Shapes are positive and stride rank equals physical-shape rank.",
        "Clean and candidate records have the same contiguity classification.",
        "Zero for equal classification and one for disagreement.",
        0,
        "Only canonical dense row-major strides are classified as contiguous.",
    ),
    EvidenceContractSpec(
        EvidenceContractId.REQUESTED_BACKEND,
        "The format and backend requested by the caller.",
        "Requested fields are non-empty strings.",
        "Requested format and backend exactly match the clean record.",
        "Number of unequal requested fields.",
        0,
        "A request is intent, not proof of execution.",
    ),
    EvidenceContractSpec(
        EvidenceContractId.REPORTED_BACKEND,
        "The backend identity reported by the adapter.",
        "Reported identity may be unknown but must never be inferred.",
        "The reported value exactly matches the clean record.",
        "Zero for equality and one for inequality.",
        0,
        "Adapter reporting is not independent kernel evidence.",
    ),
    EvidenceContractSpec(
        EvidenceContractId.OBSERVED_KERNELS,
        "Profiler-observed kernel names retained as an ordered tuple.",
        "Kernel names are explicit observations and contain no blanks.",
        "Every observed kernel string exactly matches the clean record.",
        "Number of unequal or missing observed kernel entries.",
        0,
        "Does not infer a backend name from an unfamiliar kernel.",
    ),
    EvidenceContractSpec(
        EvidenceContractId.FALLBACK_STATUS,
        "The bounded fallback classification attached to kernel evidence.",
        "Status is one of unknown, detected, or not_detected.",
        "The status exactly matches the clean record.",
        "Zero for equality and one for inequality.",
        0,
        "The classifier recognizes only its documented kernel signatures.",
    ),
)

_SPECS = {spec.contract_id: spec for spec in EVIDENCE_CONTRACT_SPECS}


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return canonical dense row-major strides for a positive shape."""
    stride: list[int] = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= extent
    return tuple(reversed(stride))


def is_row_major_contiguous(tensor: TensorMetadata) -> bool:
    return tensor.stride == contiguous_stride(tensor.physical_shape)


def _sequence_mismatches(left: tuple[object, ...], right: tuple[object, ...]) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def evaluate_execution_evidence(
    clean: ExecutionEvidence, candidate: ExecutionEvidence
) -> tuple[EvidenceContractOutcome, ...]:
    """Compare evidence fields without conflating request, report, or observation."""
    stride_mismatches = _sequence_mismatches(
        tuple(clean.tensor.stride), tuple(candidate.tensor.stride)
    )
    contiguity_mismatches = int(
        is_row_major_contiguous(clean.tensor)
        != is_row_major_contiguous(candidate.tensor)
    )
    requested_mismatches = sum(
        expected != observed
        for expected, observed in (
            (
                clean.backend.requested_format,
                candidate.backend.requested_format,
            ),
            (
                clean.backend.requested_backend,
                candidate.backend.requested_backend,
            ),
        )
    )
    reported_mismatches = int(
        clean.backend.reported_backend != candidate.backend.reported_backend
    )
    kernel_mismatches = _sequence_mismatches(
        tuple(clean.backend.observed_kernels),
        tuple(candidate.backend.observed_kernels),
    )
    fallback_mismatches = int(
        clean.backend.fallback_status != candidate.backend.fallback_status
    )
    return (
        EvidenceContractOutcome(
            _SPECS[EvidenceContractId.TENSOR_STRIDE],
            stride_mismatches,
            "exact recorded-stride comparison",
        ),
        EvidenceContractOutcome(
            _SPECS[EvidenceContractId.TENSOR_CONTIGUITY],
            contiguity_mismatches,
            (
                f"clean={is_row_major_contiguous(clean.tensor)}, "
                f"candidate={is_row_major_contiguous(candidate.tensor)}"
            ),
        ),
        EvidenceContractOutcome(
            _SPECS[EvidenceContractId.REQUESTED_BACKEND],
            requested_mismatches,
            "requested format/backend comparison",
        ),
        EvidenceContractOutcome(
            _SPECS[EvidenceContractId.REPORTED_BACKEND],
            reported_mismatches,
            "reported adapter identity comparison",
        ),
        EvidenceContractOutcome(
            _SPECS[EvidenceContractId.OBSERVED_KERNELS],
            kernel_mismatches,
            "observed kernel tuple comparison without backend inference",
        ),
        EvidenceContractOutcome(
            _SPECS[EvidenceContractId.FALLBACK_STATUS],
            fallback_mismatches,
            "bounded fallback-status comparison",
        ),
    )
