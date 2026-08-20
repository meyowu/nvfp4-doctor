"""Explicit CPU reconstruction of row-wise NVFP4 tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nvfp4_doctor.formats import (
    NVFP4_BLOCK_SIZE,
    ScaleFactorLayout,
    decode_e2m1,
    decode_ue4m3,
    unpack_e2m1,
    unswizzle_scales_128x4,
)


@dataclass(frozen=True, slots=True)
class NVFP4Tensor:
    """All format metadata required for unambiguous NVFP4 reconstruction."""

    rows: int
    columns: int
    packed_values: bytes
    scale_codes: bytes
    scale_layout: ScaleFactorLayout
    global_scale: float
    block_size: int = NVFP4_BLOCK_SIZE

    def __post_init__(self) -> None:
        if (
            isinstance(self.rows, bool)
            or not isinstance(self.rows, int)
            or self.rows <= 0
        ):
            raise ValueError("rows must be a positive integer")
        if (
            isinstance(self.columns, bool)
            or not isinstance(self.columns, int)
            or self.columns <= 0
        ):
            raise ValueError("columns must be a positive integer")
        if self.block_size != NVFP4_BLOCK_SIZE:
            raise ValueError("dense NVFP4 requires a block size of 16")
        if self.columns % self.block_size:
            raise ValueError("columns must be divisible by the NVFP4 block size")
        if self.columns % 2:
            raise ValueError("columns must be even for packed E2M1 storage")
        if len(self.packed_values) != self.rows * self.columns // 2:
            raise ValueError("packed value storage length does not match logical shape")
        if not math.isfinite(self.global_scale) or self.global_scale < 0.0:
            raise ValueError("global_scale must be a non-negative finite scalar")

        scale_columns = self.columns // self.block_size
        if self.scale_layout == ScaleFactorLayout.LINEAR:
            expected_scale_bytes = self.rows * scale_columns
        elif self.scale_layout == ScaleFactorLayout.CUTLASS_128X4:
            from nvfp4_doctor.formats import scale_storage_size

            expected_scale_bytes = scale_storage_size(
                self.rows, scale_columns, self.scale_layout
            )
        else:
            raise ValueError(f"unsupported scale-factor layout: {self.scale_layout!r}")
        if len(self.scale_codes) != expected_scale_bytes:
            raise ValueError("scale storage length does not match its declared layout")


def reconstruct_nvfp4(tensor: NVFP4Tensor) -> tuple[tuple[float, ...], ...]:
    """Return ``E2M1 * UE4M3 block scale * FP32 global scale`` explicitly."""
    scale_columns = tensor.columns // tensor.block_size
    if tensor.scale_layout == ScaleFactorLayout.LINEAR:
        linear_scales = tensor.scale_codes
    else:
        linear_scales = unswizzle_scales_128x4(
            tensor.scale_codes, tensor.rows, scale_columns
        )
    decoded_scales = tuple(decode_ue4m3(code) for code in linear_scales)
    value_codes = unpack_e2m1(tensor.packed_values)

    rows: list[tuple[float, ...]] = []
    for row in range(tensor.rows):
        values: list[float] = []
        for column in range(tensor.columns):
            value_code = value_codes[row * tensor.columns + column]
            scale_column = column // tensor.block_size
            block_scale = decoded_scales[row * scale_columns + scale_column]
            values.append(decode_e2m1(value_code) * block_scale * tensor.global_scale)
        rows.append(tuple(values))
    return tuple(rows)
