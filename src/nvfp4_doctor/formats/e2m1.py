"""Exact, dependency-free E2M1 value semantics."""

from __future__ import annotations

import math

E2M1_MAX = 6.0
_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _require_code(code: int) -> None:
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 0xF:
        raise ValueError("E2M1 code must be an integer in [0, 15]")


def decode_e2m1(code: int) -> float:
    """Decode one four-bit E2M1 payload, preserving signed zero."""
    _require_code(code)
    magnitude = _MAGNITUDES[code & 0x7]
    return math.copysign(magnitude, -1.0 if code & 0x8 else 1.0)


def encode_e2m1(value: float) -> int:
    """Encode a finite value using saturation and round-to-nearest-even."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("E2M1 encoder requires a finite value")
    negative = math.copysign(1.0, value) < 0.0
    magnitude = min(abs(value), E2M1_MAX)
    distances = tuple(abs(magnitude - candidate) for candidate in _MAGNITUDES)
    best_distance = min(distances)
    candidates = tuple(
        code for code, distance in enumerate(distances) if distance == best_distance
    )
    magnitude_code = min(candidates, key=lambda code: code & 1)
    return magnitude_code | (0x8 if negative else 0)
