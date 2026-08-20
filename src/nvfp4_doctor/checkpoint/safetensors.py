"""Dependency-free safetensors header validation."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class SafetensorsHeaderError(ValueError):
    """Raised when a safetensors header violates exact storage boundaries."""


@dataclass(frozen=True, slots=True)
class SafetensorsTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def data_bytes(self) -> int:
        return self.data_end - self.data_start


@dataclass(frozen=True, slots=True)
class SafetensorsHeader:
    header_length: int
    file_size: int
    payload_start: int
    payload_bytes: int
    tensors: tuple[SafetensorsTensor, ...]


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SafetensorsHeaderError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SafetensorsHeaderError(f"{field} must be a non-negative integer")
    return value


def _tensor(name: str, value: object, payload_bytes: int) -> SafetensorsTensor:
    if not isinstance(value, dict):
        raise SafetensorsHeaderError(f"tensor {name} metadata must be an object")
    if set(value) != {"dtype", "shape", "data_offsets"}:
        raise SafetensorsHeaderError(f"tensor {name} has unsupported metadata fields")

    dtype = value["dtype"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise SafetensorsHeaderError(f"tensor {name} has an unsupported dtype")
    shape_value = value["shape"]
    if not isinstance(shape_value, list):
        raise SafetensorsHeaderError(f"tensor {name} shape must be a list")
    shape = tuple(
        _non_negative_integer(dimension, f"tensor {name} shape")
        for dimension in shape_value
    )
    offsets = value["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise SafetensorsHeaderError(f"tensor {name} data_offsets must have length two")
    start = _non_negative_integer(offsets[0], f"tensor {name} data start")
    end = _non_negative_integer(offsets[1], f"tensor {name} data end")
    if start > end or end > payload_bytes:
        raise SafetensorsHeaderError(f"tensor {name} data offsets exceed payload")

    expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
    if end - start != expected_bytes:
        raise SafetensorsHeaderError(
            f"tensor {name} byte length does not match dtype/shape"
        )
    return SafetensorsTensor(name, dtype, shape, start, end)


def parse_safetensors_header(
    prefix: bytes,
    header_bytes: bytes,
    file_size: int,
) -> SafetensorsHeader:
    """Parse a prefix plus JSON header and verify exact contiguous payload bounds."""
    file_size = _non_negative_integer(file_size, "file_size")
    if len(prefix) != 8:
        raise SafetensorsHeaderError("safetensors prefix must contain exactly 8 bytes")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length != len(header_bytes):
        raise SafetensorsHeaderError(
            "declared header length does not match fetched bytes"
        )
    payload_start = 8 + header_length
    if payload_start > file_size:
        raise SafetensorsHeaderError("header extends beyond the declared file size")
    payload_bytes = file_size - payload_start

    try:
        document = json.loads(header_bytes, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SafetensorsHeaderError("header is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise SafetensorsHeaderError("header JSON must be an object")
    metadata = document.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise SafetensorsHeaderError("__metadata__ must be an object when present")
    if not document:
        raise SafetensorsHeaderError("header must describe at least one tensor")

    tensors = tuple(
        sorted(
            (_tensor(name, value, payload_bytes) for name, value in document.items()),
            key=lambda tensor: tensor.data_start,
        )
    )
    expected_start = 0
    for tensor in tensors:
        if tensor.data_start != expected_start:
            raise SafetensorsHeaderError("tensor payload intervals are not contiguous")
        expected_start = tensor.data_end
    if expected_start != payload_bytes:
        raise SafetensorsHeaderError("tensor payload intervals do not cover the file")

    return SafetensorsHeader(
        header_length=header_length,
        file_size=file_size,
        payload_start=payload_start,
        payload_bytes=payload_bytes,
        tensors=tensors,
    )
