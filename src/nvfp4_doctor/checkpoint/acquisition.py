"""Metadata-only tensor byte-range acquisition planning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from .safetensors import SafetensorsHeader, SafetensorsTensor


class AcquisitionPlanError(ValueError):
    """Raised when an exact tensor acquisition plan cannot be constructed."""


@dataclass(frozen=True, slots=True)
class TensorByteRange:
    tensor_name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    file_start: int
    file_end: int

    @property
    def byte_length(self) -> int:
        return self.file_end - self.file_start

    @property
    def http_range(self) -> str:
        return f"bytes={self.file_start}-{self.file_end - 1}"


def _tensor_index(
    headers: Mapping[str, SafetensorsHeader],
) -> dict[str, tuple[str, SafetensorsHeader, SafetensorsTensor]]:
    index: dict[str, tuple[str, SafetensorsHeader, SafetensorsTensor]] = {}
    for shard, header in headers.items():
        if not shard:
            raise AcquisitionPlanError("shard names must be non-empty")
        for tensor in header.tensors:
            if tensor.name in index:
                raise AcquisitionPlanError(
                    f"tensor occurs in more than one shard: {tensor.name}"
                )
            index[tensor.name] = (shard, header, tensor)
    return index


def plan_tensor_byte_ranges(
    headers: Mapping[str, SafetensorsHeader],
    tensor_names: Iterable[str],
) -> tuple[TensorByteRange, ...]:
    """Map unique tensor names to exact, shard-local HTTP byte ranges."""
    names = tuple(tensor_names)
    if not names:
        raise AcquisitionPlanError("at least one tensor must be requested")
    if len(set(names)) != len(names):
        raise AcquisitionPlanError("requested tensor names must be unique")

    index = _tensor_index(headers)
    missing = sorted(set(names) - set(index))
    if missing:
        raise AcquisitionPlanError(f"requested tensors are missing: {missing}")

    ranges: list[TensorByteRange] = []
    for name in names:
        shard, header, tensor = index[name]
        file_start = header.payload_start + tensor.data_start
        file_end = header.payload_start + tensor.data_end
        if file_start >= file_end:
            raise AcquisitionPlanError(f"tensor has no downloadable bytes: {name}")
        if file_end > header.file_size:
            raise AcquisitionPlanError(f"tensor range exceeds shard boundary: {name}")
        ranges.append(
            TensorByteRange(
                tensor_name=name,
                shard=shard,
                dtype=tensor.dtype,
                shape=tensor.shape,
                file_start=file_start,
                file_end=file_end,
            )
        )

    ranges.sort(key=lambda item: (item.shard, item.file_start, item.tensor_name))
    for previous, current in pairwise(ranges):
        if previous.shard == current.shard and previous.file_end > current.file_start:
            raise AcquisitionPlanError("planned tensor byte ranges overlap")
    return tuple(ranges)
