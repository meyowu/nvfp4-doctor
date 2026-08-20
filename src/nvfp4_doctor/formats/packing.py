"""CUTLASS-compatible packed E2M1 storage helpers."""

from __future__ import annotations

from collections.abc import Iterable


def _codes(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ValueError("E2M1 payloads must be integers")
    if any(not 0 <= value <= 0xF for value in result):
        raise ValueError("E2M1 payloads must be in [0, 15]")
    return result


def pack_e2m1(values: Iterable[int]) -> bytes:
    """Pack logical element 0 in the low nibble and element 1 in the high nibble."""
    codes = _codes(values)
    if len(codes) % 2:
        raise ValueError("packed E2M1 storage requires an even element count")
    return bytes(low | (high << 4) for low, high in zip(codes[::2], codes[1::2]))


def unpack_e2m1(storage: bytes | bytearray | memoryview) -> tuple[int, ...]:
    """Unpack bytes into logical low-nibble-first E2M1 payloads."""
    result: list[int] = []
    for item in bytes(storage):
        result.extend((item & 0xF, item >> 4))
    return tuple(result)
