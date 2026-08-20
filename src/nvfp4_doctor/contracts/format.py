"""Exact structural contracts for independently interpreted NVFP4 tensors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    cutlass_128x4_offset,
    unswizzle_scales_128x4,
)
from nvfp4_doctor.oracle import NVFP4Tensor, reconstruct_nvfp4


class FormatContractId(StrEnum):
    """Stable identifiers for the Gate 1 structural contract layers."""

    METADATA = "metadata"
    PACKED_VALUES = "packed_values"
    LOGICAL_SCALES = "logical_scales"
    SCALE_PADDING = "scale_padding"
    GLOBAL_SCALE = "global_scale"
    RECONSTRUCTION = "reconstruction"


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Auditable definition of one exact format contract."""

    contract_id: FormatContractId
    domain: str
    preconditions: str
    invariant: str
    metric: str
    threshold: int
    limitations: str


@dataclass(frozen=True, slots=True)
class ContractOutcome:
    """Observed mismatch count for one contract specification."""

    spec: ContractSpec
    mismatch_count: int
    details: str

    @property
    def passed(self) -> bool:
        return self.mismatch_count <= self.spec.threshold


FORMAT_CONTRACT_SPECS = (
    ContractSpec(
        FormatContractId.METADATA,
        "Two NVFP4Tensor descriptions of the same logical operand.",
        "Both tensors pass NVFP4Tensor construction validation.",
        "Rows, columns, block size, and declared scale layout are identical.",
        "Number of unequal metadata fields.",
        0,
        "Does not represent runtime strides or backend identity.",
    ),
    ContractSpec(
        FormatContractId.PACKED_VALUES,
        "Packed E2M1 storage for the same logical operand.",
        "The logical shapes and packed storage lengths are equal.",
        "Every packed byte is identical.",
        "Number of unequal or missing packed bytes.",
        0,
        "Localizes storage disagreement, not its numerical cause.",
    ),
    ContractSpec(
        FormatContractId.LOGICAL_SCALES,
        "UE4M3 scales after explicit layout normalization.",
        "The logical scale shapes are equal and layouts are supported.",
        "Every logical block-scale code is identical.",
        "Number of unequal or missing logical scale codes.",
        0,
        "Does not inspect physical padding bytes.",
    ),
    ContractSpec(
        FormatContractId.SCALE_PADDING,
        "Physical padding in CUTLASS 128x4 scale storage.",
        "The reference uses CUTLASS 128x4 and physical lengths are equal.",
        "Every non-logical physical byte matches the clean reference.",
        "Number of unequal padding bytes.",
        0,
        "Not applicable to linear storage or layout-mismatched candidates.",
    ),
    ContractSpec(
        FormatContractId.GLOBAL_SCALE,
        "The scalar FP32 dequantization multiplier.",
        "Both values are finite non-negative scalars.",
        "The scalar values are exactly equal.",
        "Zero for equality and one for inequality.",
        0,
        "Exact equality is for synthetic controls, not learned-scale tolerance.",
    ),
    ContractSpec(
        FormatContractId.RECONSTRUCTION,
        "Explicitly reconstructed logical NVFP4 values.",
        "Both tensors have the same logical shape and decodable scale codes.",
        "Every reconstructed value is exactly equal.",
        "Number of unequal logical values.",
        0,
        "Does not test GEMM accumulation or arbitrary-input quantization error.",
    ),
)

_SPECS = {spec.contract_id: spec for spec in FORMAT_CONTRACT_SPECS}


def _mismatch_count(left: bytes, right: bytes) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def _linear_scales(tensor: NVFP4Tensor) -> bytes:
    if tensor.scale_layout == ScaleFactorLayout.LINEAR:
        return tensor.scale_codes
    return unswizzle_scales_128x4(
        tensor.scale_codes,
        tensor.rows,
        tensor.columns // tensor.block_size,
    )


def _padding_offsets(tensor: NVFP4Tensor) -> tuple[int, ...]:
    if tensor.scale_layout != ScaleFactorLayout.CUTLASS_128X4:
        return ()
    scale_columns = tensor.columns // tensor.block_size
    logical_offsets = {
        cutlass_128x4_offset(row, column, tensor.rows, scale_columns)
        for row in range(tensor.rows)
        for column in range(scale_columns)
    }
    return tuple(
        offset
        for offset in range(len(tensor.scale_codes))
        if offset not in logical_offsets
    )


def evaluate_format_contracts(
    clean: NVFP4Tensor, candidate: NVFP4Tensor
) -> tuple[ContractOutcome, ...]:
    """Compare a candidate with a clean reference using exact Gate 1 contracts."""
    metadata_fields = (
        ("rows", clean.rows, candidate.rows),
        ("columns", clean.columns, candidate.columns),
        ("block_size", clean.block_size, candidate.block_size),
        ("scale_layout", clean.scale_layout, candidate.scale_layout),
    )
    metadata_mismatches = tuple(
        name for name, expected, observed in metadata_fields if expected != observed
    )

    logical_shape_matches = (
        clean.rows == candidate.rows
        and clean.columns == candidate.columns
        and clean.block_size == candidate.block_size
    )
    packed_mismatches = (
        _mismatch_count(clean.packed_values, candidate.packed_values)
        if logical_shape_matches
        else max(len(clean.packed_values), len(candidate.packed_values), 1)
    )

    if logical_shape_matches:
        clean_scales = _linear_scales(clean)
        candidate_scales = _linear_scales(candidate)
        logical_scale_mismatches = _mismatch_count(clean_scales, candidate_scales)
    else:
        logical_scale_mismatches = max(
            len(clean.scale_codes), len(candidate.scale_codes), 1
        )

    padding_mismatches = 0
    padding_details = "not applicable"
    if (
        clean.scale_layout == ScaleFactorLayout.CUTLASS_128X4
        and candidate.scale_layout == clean.scale_layout
        and len(clean.scale_codes) == len(candidate.scale_codes)
    ):
        padding_offsets = _padding_offsets(clean)
        padding_mismatches = sum(
            clean.scale_codes[offset] != candidate.scale_codes[offset]
            for offset in padding_offsets
        )
        padding_details = f"checked {len(padding_offsets)} physical padding bytes"

    global_scale_mismatches = int(clean.global_scale != candidate.global_scale)

    reconstruction_mismatches = 0
    reconstruction_details = "exact logical value comparison"
    if logical_shape_matches:
        clean_rows = reconstruct_nvfp4(clean)
        candidate_rows = reconstruct_nvfp4(candidate)
        reconstruction_mismatches = sum(
            expected != observed
            for clean_row, candidate_row in zip(clean_rows, candidate_rows)
            for expected, observed in zip(clean_row, candidate_row)
        )
    else:
        reconstruction_mismatches = max(clean.rows * clean.columns, 1)
        reconstruction_details = "logical shapes differ"

    return (
        ContractOutcome(
            _SPECS[FormatContractId.METADATA],
            len(metadata_mismatches),
            "mismatched fields: " + ", ".join(metadata_mismatches)
            if metadata_mismatches
            else "all declared fields match",
        ),
        ContractOutcome(
            _SPECS[FormatContractId.PACKED_VALUES],
            packed_mismatches,
            "exact packed-byte comparison",
        ),
        ContractOutcome(
            _SPECS[FormatContractId.LOGICAL_SCALES],
            logical_scale_mismatches,
            "layout-normalized logical scale comparison",
        ),
        ContractOutcome(
            _SPECS[FormatContractId.SCALE_PADDING],
            padding_mismatches,
            padding_details,
        ),
        ContractOutcome(
            _SPECS[FormatContractId.GLOBAL_SCALE],
            global_scale_mismatches,
            "exact scalar comparison",
        ),
        ContractOutcome(
            _SPECS[FormatContractId.RECONSTRUCTION],
            reconstruction_mismatches,
            reconstruction_details,
        ),
    )
