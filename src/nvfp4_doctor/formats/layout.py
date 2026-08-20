"""Logical-to-physical mappings for Blackwell NVFP4 scale factors."""

from __future__ import annotations

from enum import StrEnum

_ATOM_ROWS = 128
_ATOM_SCALE_COLUMNS = 4
_ATOM_BYTES = _ATOM_ROWS * _ATOM_SCALE_COLUMNS


class ScaleFactorLayout(StrEnum):
    LINEAR = "linear"
    CUTLASS_128X4 = "cutlass_128x4"


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_shape(rows: int, scale_columns: int) -> None:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if (
        isinstance(scale_columns, bool)
        or not isinstance(scale_columns, int)
        or scale_columns <= 0
    ):
        raise ValueError("scale_columns must be a positive integer")


def scale_physical_shape(
    rows: int, scale_columns: int, layout: ScaleFactorLayout
) -> tuple[int, int]:
    _require_shape(rows, scale_columns)
    if layout == ScaleFactorLayout.LINEAR:
        return rows, scale_columns
    if layout == ScaleFactorLayout.CUTLASS_128X4:
        return (
            _ceil_div(rows, _ATOM_ROWS) * _ATOM_ROWS,
            _ceil_div(scale_columns, _ATOM_SCALE_COLUMNS) * _ATOM_SCALE_COLUMNS,
        )
    raise ValueError(f"unsupported scale-factor layout: {layout!r}")


def scale_storage_size(rows: int, scale_columns: int, layout: ScaleFactorLayout) -> int:
    physical_rows, physical_columns = scale_physical_shape(rows, scale_columns, layout)
    return physical_rows * physical_columns


def cutlass_128x4_offset(
    row: int, scale_column: int, rows: int, scale_columns: int
) -> int:
    """Return the byte offset for a logical scale in CUTLASS's 512-byte atom."""
    _require_shape(rows, scale_columns)
    if not 0 <= row < rows:
        raise IndexError("row is outside the logical scale matrix")
    if not 0 <= scale_column < scale_columns:
        raise IndexError("scale_column is outside the logical scale matrix")

    scale_atoms = _ceil_div(scale_columns, _ATOM_SCALE_COLUMNS)
    row_atom, row_in_atom = divmod(row, _ATOM_ROWS)
    scale_atom, scale_in_atom = divmod(scale_column, _ATOM_SCALE_COLUMNS)
    row_group, row_in_group = divmod(row_in_atom, 32)
    atom_offset = row_in_group * 16 + row_group * 4 + scale_in_atom
    return (row_atom * scale_atoms + scale_atom) * _ATOM_BYTES + atom_offset


def swizzle_scales_128x4(
    linear: bytes | bytearray | memoryview,
    rows: int,
    scale_columns: int,
    *,
    padding_code: int = 0,
) -> bytes:
    """Map logical row-major scale bytes into the CUTLASS 128x4 layout."""
    _require_shape(rows, scale_columns)
    if not 0 <= padding_code <= 0x7E:
        raise ValueError("padding_code must be a finite UE4M3 code")
    source = bytes(linear)
    if len(source) != rows * scale_columns:
        raise ValueError("linear scale storage length does not match logical shape")
    output = bytearray(
        [padding_code]
        * scale_storage_size(rows, scale_columns, ScaleFactorLayout.CUTLASS_128X4)
    )
    for row in range(rows):
        for scale_column in range(scale_columns):
            output[cutlass_128x4_offset(row, scale_column, rows, scale_columns)] = (
                source[row * scale_columns + scale_column]
            )
    return bytes(output)


def unswizzle_scales_128x4(
    storage: bytes | bytearray | memoryview,
    rows: int,
    scale_columns: int,
) -> bytes:
    """Recover logical row-major scale bytes while discarding physical padding."""
    _require_shape(rows, scale_columns)
    source = bytes(storage)
    expected = scale_storage_size(rows, scale_columns, ScaleFactorLayout.CUTLASS_128X4)
    if len(source) != expected:
        raise ValueError("swizzled scale storage length does not match physical shape")
    output = bytearray(rows * scale_columns)
    for row in range(rows):
        for scale_column in range(scale_columns):
            output[row * scale_columns + scale_column] = source[
                cutlass_128x4_offset(row, scale_column, rows, scale_columns)
            ]
    return bytes(output)
