"""Deterministic, reversible, non-mutating NVFP4 positive controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    cutlass_128x4_offset,
    pack_e2m1,
    scale_storage_size,
    unpack_e2m1,
    unswizzle_scales_128x4,
)
from nvfp4_doctor.oracle import NVFP4Tensor

FaultParameter = tuple[str, int | float | str]


class NVFP4FaultKind(StrEnum):
    NIBBLE_SWAP = "nibble_swap"
    SCALE_INDEX_SHIFT = "scale_index_shift"
    BLOCK_SCALE_REVERSAL = "block_scale_reversal"
    GLOBAL_SCALE_MULTIPLIER = "global_scale_multiplier"
    SCALE_LAYOUT_MISLABEL = "scale_layout_mislabel"
    PADDING_CORRUPTION = "padding_corruption"
    PACKED_BLOCK_PERMUTATION = "packed_block_permutation"
    PACKED_ROW_PERMUTATION = "packed_row_permutation"
    PACKED_COLUMN_PERMUTATION = "packed_column_permutation"


@dataclass(frozen=True, slots=True)
class FaultInjection:
    """One labeled synthetic positive control and its immutable clean source."""

    kind: NVFP4FaultKind
    clean: NVFP4Tensor
    faulted: NVFP4Tensor
    parameters: tuple[FaultParameter, ...]
    label: str = "synthetic"

    def __post_init__(self) -> None:
        if self.clean == self.faulted:
            raise ValueError("a positive-control fault must change the artifact")


def _linear_scales(tensor: NVFP4Tensor) -> bytes:
    if tensor.scale_layout == ScaleFactorLayout.LINEAR:
        return tensor.scale_codes
    return unswizzle_scales_128x4(
        tensor.scale_codes,
        tensor.rows,
        tensor.columns // tensor.block_size,
    )


def _replace_linear_scales(tensor: NVFP4Tensor, linear: bytes) -> NVFP4Tensor:
    scale_columns = tensor.columns // tensor.block_size
    if len(linear) != tensor.rows * scale_columns:
        raise ValueError("linear scale storage length does not match logical shape")
    if tensor.scale_layout == ScaleFactorLayout.LINEAR:
        return replace(tensor, scale_codes=linear)

    storage = bytearray(tensor.scale_codes)
    for row in range(tensor.rows):
        for column in range(scale_columns):
            storage[cutlass_128x4_offset(row, column, tensor.rows, scale_columns)] = (
                linear[row * scale_columns + column]
            )
    return replace(tensor, scale_codes=bytes(storage))


def swap_packed_nibbles(tensor: NVFP4Tensor) -> NVFP4Tensor:
    """Swap the two logical E2M1 payloads in every packed byte."""
    swapped = bytes(
        ((value & 0xF) << 4) | (value >> 4) for value in tensor.packed_values
    )
    return replace(tensor, packed_values=swapped)


def _cyclic_permutation(items: tuple[int, ...], offset: int) -> tuple[int, ...]:
    normalized_offset = offset % len(items)
    return tuple(
        items[(index - normalized_offset) % len(items)] for index in range(len(items))
    )


def _permutation_offset(offset: int, extent: int, dimension: str) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an integer")
    normalized_offset = offset % extent
    if extent < 2 or normalized_offset == 0:
        raise ValueError(f"{dimension} permutation must move at least one item")
    return normalized_offset


def permute_packed_blocks(tensor: NVFP4Tensor, offset: int) -> NVFP4Tensor:
    """Cyclically permute complete E2M1 quantization blocks within every row."""
    block_columns = tensor.columns // tensor.block_size
    normalized_offset = _permutation_offset(offset, block_columns, "block")
    codes = unpack_e2m1(tensor.packed_values)
    permuted: list[int] = []
    for row in range(tensor.rows):
        row_start = row * tensor.columns
        blocks = tuple(
            codes[
                row_start + block * tensor.block_size : row_start
                + (block + 1) * tensor.block_size
            ]
            for block in range(block_columns)
        )
        for block in _cyclic_permutation(
            tuple(range(block_columns)), normalized_offset
        ):
            permuted.extend(blocks[block])
    return replace(tensor, packed_values=pack_e2m1(permuted))


def permute_packed_rows(tensor: NVFP4Tensor, offset: int) -> NVFP4Tensor:
    """Cyclically permute complete packed E2M1 rows."""
    normalized_offset = _permutation_offset(offset, tensor.rows, "row")
    codes = unpack_e2m1(tensor.packed_values)
    row_order = _cyclic_permutation(tuple(range(tensor.rows)), normalized_offset)
    permuted = (
        code
        for row in row_order
        for code in codes[row * tensor.columns : (row + 1) * tensor.columns]
    )
    return replace(tensor, packed_values=pack_e2m1(permuted))


def permute_packed_columns(tensor: NVFP4Tensor, offset: int) -> NVFP4Tensor:
    """Cyclically permute logical E2M1 columns within every row."""
    normalized_offset = _permutation_offset(offset, tensor.columns, "column")
    codes = unpack_e2m1(tensor.packed_values)
    permuted: list[int] = []
    for row in range(tensor.rows):
        row_start = row * tensor.columns
        row_codes = codes[row_start : row_start + tensor.columns]
        permuted.extend(_cyclic_permutation(row_codes, normalized_offset))
    return replace(tensor, packed_values=pack_e2m1(permuted))


def shift_scale_indices(tensor: NVFP4Tensor, offset: int) -> NVFP4Tensor:
    """Cyclically shift logical scale columns within every row."""
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an integer")
    scale_columns = tensor.columns // tensor.block_size
    normalized_offset = offset % scale_columns
    if scale_columns < 2 or normalized_offset == 0:
        raise ValueError("scale shift must move at least one logical scale column")
    linear = _linear_scales(tensor)
    shifted = bytearray(len(linear))
    for row in range(tensor.rows):
        row_start = row * scale_columns
        for column in range(scale_columns):
            shifted[row_start + column] = linear[
                row_start + (column - normalized_offset) % scale_columns
            ]
    return _replace_linear_scales(tensor, bytes(shifted))


def reverse_scale_blocks(tensor: NVFP4Tensor) -> NVFP4Tensor:
    """Reverse logical scale columns within every row."""
    scale_columns = tensor.columns // tensor.block_size
    if scale_columns < 2:
        raise ValueError("scale reversal requires at least two scale columns")
    linear = _linear_scales(tensor)
    reversed_codes = b"".join(
        linear[row * scale_columns : (row + 1) * scale_columns][::-1]
        for row in range(tensor.rows)
    )
    return _replace_linear_scales(tensor, reversed_codes)


def multiply_global_scale(tensor: NVFP4Tensor, factor: float) -> NVFP4Tensor:
    """Multiply the scalar dequantization scale by a finite positive factor."""
    if not math.isfinite(factor) or factor <= 0.0 or factor == 1.0:
        raise ValueError("factor must be finite, positive, and different from one")
    value = tensor.global_scale * factor
    if not math.isfinite(value):
        raise ValueError("the resulting global scale must be finite")
    return replace(tensor, global_scale=value)


def relabel_scale_layout(tensor: NVFP4Tensor, target: ScaleFactorLayout) -> NVFP4Tensor:
    """Change only layout metadata when both layouts have the same byte length."""
    if target == tensor.scale_layout:
        raise ValueError("target layout must differ from the clean layout")
    scale_columns = tensor.columns // tensor.block_size
    expected_length = scale_storage_size(tensor.rows, scale_columns, target)
    if expected_length != len(tensor.scale_codes):
        raise ValueError("target layout is not length-compatible with this storage")
    return replace(tensor, scale_layout=target)


def toggle_padding_byte(tensor: NVFP4Tensor, offset: int) -> NVFP4Tensor:
    """Toggle one physical padding byte without changing a logical scale."""
    if tensor.scale_layout != ScaleFactorLayout.CUTLASS_128X4:
        raise ValueError("padding corruption requires CUTLASS 128x4 storage")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an integer")
    if not 0 <= offset < len(tensor.scale_codes):
        raise ValueError("offset is outside scale storage")
    scale_columns = tensor.columns // tensor.block_size
    logical_offsets = {
        cutlass_128x4_offset(row, column, tensor.rows, scale_columns)
        for row in range(tensor.rows)
        for column in range(scale_columns)
    }
    if offset in logical_offsets:
        raise ValueError("offset identifies a logical scale, not padding")
    storage = bytearray(tensor.scale_codes)
    storage[offset] ^= 0x01
    return replace(tensor, scale_codes=bytes(storage))


def first_padding_offset(tensor: NVFP4Tensor) -> int:
    """Return the first physical padding offset in a CUTLASS scale tensor."""
    if tensor.scale_layout != ScaleFactorLayout.CUTLASS_128X4:
        raise ValueError("padding lookup requires CUTLASS 128x4 storage")
    scale_columns = tensor.columns // tensor.block_size
    logical_offsets = {
        cutlass_128x4_offset(row, column, tensor.rows, scale_columns)
        for row in range(tensor.rows)
        for column in range(scale_columns)
    }
    for offset in range(len(tensor.scale_codes)):
        if offset not in logical_offsets:
            return offset
    raise ValueError("the scale tensor has no physical padding")


def inject_nibble_swap(tensor: NVFP4Tensor) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.NIBBLE_SWAP,
        tensor,
        swap_packed_nibbles(tensor),
        (),
    )


def inject_scale_index_shift(tensor: NVFP4Tensor, offset: int) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.SCALE_INDEX_SHIFT,
        tensor,
        shift_scale_indices(tensor, offset),
        (("offset", offset),),
    )


def inject_block_scale_reversal(tensor: NVFP4Tensor) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.BLOCK_SCALE_REVERSAL,
        tensor,
        reverse_scale_blocks(tensor),
        (),
    )


def inject_global_scale_multiplier(
    tensor: NVFP4Tensor, factor: float
) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.GLOBAL_SCALE_MULTIPLIER,
        tensor,
        multiply_global_scale(tensor, factor),
        (("factor", factor),),
    )


def inject_scale_layout_mislabel(
    tensor: NVFP4Tensor, target: ScaleFactorLayout
) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.SCALE_LAYOUT_MISLABEL,
        tensor,
        relabel_scale_layout(tensor, target),
        (("target", target.value),),
    )


def inject_padding_corruption(
    tensor: NVFP4Tensor, offset: int | None = None
) -> FaultInjection:
    selected_offset = first_padding_offset(tensor) if offset is None else offset
    return FaultInjection(
        NVFP4FaultKind.PADDING_CORRUPTION,
        tensor,
        toggle_padding_byte(tensor, selected_offset),
        (("offset", selected_offset),),
    )


def inject_packed_block_permutation(tensor: NVFP4Tensor, offset: int) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.PACKED_BLOCK_PERMUTATION,
        tensor,
        permute_packed_blocks(tensor, offset),
        (("offset", offset),),
    )


def inject_packed_row_permutation(tensor: NVFP4Tensor, offset: int) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.PACKED_ROW_PERMUTATION,
        tensor,
        permute_packed_rows(tensor, offset),
        (("offset", offset),),
    )


def inject_packed_column_permutation(
    tensor: NVFP4Tensor, offset: int
) -> FaultInjection:
    return FaultInjection(
        NVFP4FaultKind.PACKED_COLUMN_PERMUTATION,
        tensor,
        permute_packed_columns(tensor, offset),
        (("offset", offset),),
    )


def revert_fault(injection: FaultInjection) -> NVFP4Tensor:
    """Apply the mathematical inverse and return the restored clean artifact."""
    parameters = dict(injection.parameters)
    if injection.kind == NVFP4FaultKind.NIBBLE_SWAP:
        return swap_packed_nibbles(injection.faulted)
    if injection.kind == NVFP4FaultKind.SCALE_INDEX_SHIFT:
        return shift_scale_indices(injection.faulted, -int(parameters["offset"]))
    if injection.kind == NVFP4FaultKind.BLOCK_SCALE_REVERSAL:
        return reverse_scale_blocks(injection.faulted)
    if injection.kind == NVFP4FaultKind.GLOBAL_SCALE_MULTIPLIER:
        return multiply_global_scale(
            injection.faulted, 1.0 / float(parameters["factor"])
        )
    if injection.kind == NVFP4FaultKind.SCALE_LAYOUT_MISLABEL:
        return relabel_scale_layout(injection.faulted, injection.clean.scale_layout)
    if injection.kind == NVFP4FaultKind.PADDING_CORRUPTION:
        return toggle_padding_byte(injection.faulted, int(parameters["offset"]))
    if injection.kind == NVFP4FaultKind.PACKED_BLOCK_PERMUTATION:
        return permute_packed_blocks(injection.faulted, -int(parameters["offset"]))
    if injection.kind == NVFP4FaultKind.PACKED_ROW_PERMUTATION:
        return permute_packed_rows(injection.faulted, -int(parameters["offset"]))
    if injection.kind == NVFP4FaultKind.PACKED_COLUMN_PERMUTATION:
        return permute_packed_columns(injection.faulted, -int(parameters["offset"]))
    raise ValueError(f"unsupported fault kind: {injection.kind!r}")
