"""Strict loading and layout preparation for acquired ModelOpt NVFP4 tensors."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nvfp4_doctor.formats import swizzle_scales_128x4

_DTYPE_BYTES = {"F32": 4, "F8_E4M3": 1, "U8": 1}
_REQUIRED_SUFFIXES = (
    "input_scale",
    "weight",
    "weight_scale",
    "weight_scale_2",
)


class ProjectionPayloadError(ValueError):
    """Raised when acquired projection bytes violate their recorded contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectionPayloadError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionPayloadError(f"{field} must be a non-negative integer")
    return value


def _shape(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ProjectionPayloadError(f"{field} must be a list")
    shape = tuple(_integer(dimension, field) for dimension in value)
    if any(dimension == 0 for dimension in shape):
        raise ProjectionPayloadError(f"{field} dimensions must be positive")
    return shape


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


@dataclass(frozen=True, slots=True)
class StoredTensorPayload:
    """One immutable tensor artifact with its declared storage metadata."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    local_path: str
    recorded_sha256: str
    data: bytes

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_BYTES:
            raise ProjectionPayloadError(f"unsupported stored dtype: {self.dtype}")
        expected_bytes = _product(self.shape) * _DTYPE_BYTES[self.dtype]
        if len(self.data) != expected_bytes:
            raise ProjectionPayloadError(
                f"stored byte length disagrees with dtype/shape: {self.name}"
            )
        if _sha256(self.data) != self.recorded_sha256:
            raise ProjectionPayloadError(f"stored SHA-256 mismatch: {self.name}")

    @property
    def byte_length(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class CutlassProjectionPayload:
    """Explicit CUTLASS-ready view of one aligned ModelOpt projection."""

    layer: int
    projection: str
    rows: int
    columns: int
    packed_weight: bytes
    weight_scale_128x4: bytes
    input_scale: float
    weight_scale_2: float
    source_weight_sha256: str
    runtime_weight_sha256: str
    source_weight_scale_sha256: str
    runtime_weight_scale_sha256: str
    weight_padding_bytes: int

    @property
    def alpha(self) -> float:
        return self.input_scale * self.weight_scale_2

    @property
    def input_global_scale_inv(self) -> float:
        return 1.0 / self.input_scale


@dataclass(frozen=True, slots=True)
class ModelOptProjectionPayload:
    """Four ModelOpt tensors for one unfused or independently replayed projection."""

    layer: int
    projection: str
    input_scale: StoredTensorPayload
    weight: StoredTensorPayload
    weight_scale: StoredTensorPayload
    weight_scale_2: StoredTensorPayload

    def __post_init__(self) -> None:
        expected = {
            "input_scale": (self.input_scale, "F32", ()),
            "weight": (self.weight, "U8", None),
            "weight_scale": (self.weight_scale, "F8_E4M3", None),
            "weight_scale_2": (self.weight_scale_2, "F32", ()),
        }
        for suffix, (tensor, dtype, shape) in expected.items():
            if not tensor.name.endswith(f".{suffix}"):
                raise ProjectionPayloadError(f"incorrect tensor suffix for {suffix}")
            if tensor.dtype != dtype:
                raise ProjectionPayloadError(f"incorrect stored dtype for {suffix}")
            if shape is not None and tensor.shape != shape:
                raise ProjectionPayloadError(f"incorrect stored shape for {suffix}")

        if len(self.weight.shape) != 2 or len(self.weight_scale.shape) != 2:
            raise ProjectionPayloadError("weight and weight_scale must be rank two")
        rows, packed_columns = self.weight.shape
        columns = packed_columns * 2
        if columns % 16:
            raise ProjectionPayloadError("logical input width must be divisible by 16")
        if self.weight_scale.shape != (rows, columns // 16):
            raise ProjectionPayloadError(
                "weight_scale shape disagrees with packed weight shape"
            )
        if any(code > 0x7E for code in self.weight_scale.data):
            raise ProjectionPayloadError(
                "weight_scale contains a negative or non-finite E4M3 code"
            )
        for name, value in (
            ("input_scale", self.input_scale_value),
            ("weight_scale_2", self.weight_scale_2_value),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ProjectionPayloadError(f"{name} must be positive and finite")

    @property
    def rows(self) -> int:
        return self.weight.shape[0]

    @property
    def columns(self) -> int:
        return self.weight.shape[1] * 2

    @property
    def input_scale_value(self) -> float:
        return struct.unpack("<f", self.input_scale.data)[0]

    @property
    def weight_scale_2_value(self) -> float:
        return struct.unpack("<f", self.weight_scale_2.data)[0]

    def prepare_cutlass_128x4(self) -> CutlassProjectionPayload:
        """Swizzle only block scales; reject dimensions that would require padding."""
        if self.rows % 32 or self.columns % 32:
            raise ProjectionPayloadError(
                "projection requires explicit CUTLASS weight padding"
            )
        swizzled = swizzle_scales_128x4(
            self.weight_scale.data,
            self.rows,
            self.columns // 16,
        )
        source_weight_hash = _sha256(self.weight.data)
        return CutlassProjectionPayload(
            layer=self.layer,
            projection=self.projection,
            rows=self.rows,
            columns=self.columns,
            packed_weight=self.weight.data,
            weight_scale_128x4=swizzled,
            input_scale=self.input_scale_value,
            weight_scale_2=self.weight_scale_2_value,
            source_weight_sha256=source_weight_hash,
            runtime_weight_sha256=source_weight_hash,
            source_weight_scale_sha256=_sha256(self.weight_scale.data),
            runtime_weight_scale_sha256=_sha256(swizzled),
            weight_padding_bytes=0,
        )


def _load_record(record: Mapping[str, object], root: Path) -> StoredTensorPayload:
    name = _string(record.get("tensor_name"), "tensor_name")
    dtype = _string(record.get("dtype"), "dtype")
    shape = _shape(record.get("shape"), "shape")
    local_path = _string(record.get("local_path"), "local_path")
    recorded_sha256 = _string(record.get("sha256"), "sha256")
    recorded_length = _integer(record.get("byte_length"), "byte_length")

    resolved_root = root.resolve()
    path = (resolved_root / local_path).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ProjectionPayloadError("local_path escapes the artifact root") from error
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ProjectionPayloadError(
            f"could not read tensor artifact: {path}"
        ) from error
    if len(data) != recorded_length:
        raise ProjectionPayloadError(f"recorded byte length mismatch: {name}")
    return StoredTensorPayload(
        name=name,
        dtype=dtype,
        shape=shape,
        local_path=local_path,
        recorded_sha256=recorded_sha256,
        data=data,
    )


def load_modelopt_projection(
    evidence_path: Path,
    artifact_root: Path,
    *,
    layer: int,
    projection: str,
) -> ModelOptProjectionPayload:
    """Load exactly four hash-verified tensors without reshape, transpose, or cast."""
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ProjectionPayloadError("layer must be a non-negative integer")
    if not projection:
        raise ProjectionPayloadError("projection must be non-empty")
    try:
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectionPayloadError("could not read payload evidence") from error
    if not isinstance(document, Mapping):
        raise ProjectionPayloadError("payload evidence must be an object")
    records = document.get("payloads")
    if not isinstance(records, list):
        raise ProjectionPayloadError("payloads must be a list")

    marker = f"model.layers.{layer}."
    projection_marker = f".{projection}."
    selected: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ProjectionPayloadError("payload records must be objects")
        name = record.get("tensor_name")
        if (
            isinstance(name, str)
            and name.startswith(marker)
            and projection_marker in name
        ):
            suffix = name.rsplit(".", 1)[-1]
            if suffix in selected:
                raise ProjectionPayloadError(f"duplicate projection tensor: {suffix}")
            selected[suffix] = record

    if set(selected) != set(_REQUIRED_SUFFIXES):
        raise ProjectionPayloadError("projection does not have exactly four tensors")
    loaded = {
        suffix: _load_record(selected[suffix], artifact_root)
        for suffix in _REQUIRED_SUFFIXES
    }
    return ModelOptProjectionPayload(
        layer=layer,
        projection=projection,
        input_scale=loaded["input_scale"],
        weight=loaded["weight"],
        weight_scale=loaded["weight_scale"],
        weight_scale_2=loaded["weight_scale_2"],
    )
