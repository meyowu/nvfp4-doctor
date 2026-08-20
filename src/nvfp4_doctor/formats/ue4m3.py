"""Exact, dependency-free unsigned E4M3 scale semantics."""

from __future__ import annotations

import math

UE4M3_MAX = 448.0
UE4M3_NAN = 0x7F


def _require_code(code: int) -> None:
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 0xFF:
        raise ValueError("UE4M3 storage code must be an integer in [0, 255]")
    if code & 0x80:
        raise ValueError(
            "UE4M3 storage requires its padded most-significant bit to be zero"
        )
    if code == UE4M3_NAN:
        raise ValueError("UE4M3 code 0x7f is NaN and is not a valid scale")


def decode_ue4m3(code: int) -> float:
    """Decode one finite UE4M3 scale byte."""
    _require_code(code)
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return math.ldexp(float(mantissa), -9)
    return math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def encode_ue4m3(value: float) -> int:
    """Encode a non-negative finite scale with saturation and nearest-even ties."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("UE4M3 encoder requires a non-negative finite value")
    target = min(value, UE4M3_MAX)
    distances = tuple(abs(target - decode_ue4m3(code)) for code in range(0x7F))
    best_distance = min(distances)
    candidates = tuple(
        code for code, distance in enumerate(distances) if distance == best_distance
    )
    return min(candidates, key=lambda code: code & 1)
