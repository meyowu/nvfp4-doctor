"""NVFP4 hierarchical scale formulas from public NVIDIA semantics."""

from __future__ import annotations

import math

from .e2m1 import E2M1_MAX
from .ue4m3 import UE4M3_MAX

NVFP4_BLOCK_SIZE = 16
NVFP4_COMBINED_MAX = E2M1_MAX * UE4M3_MAX


def _require_amax(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value")
    return result


def compute_global_scale(global_amax: float) -> float:
    """Return the tensor-wide multiplicative dequantization scale."""
    return _require_amax(global_amax, "global_amax") / NVFP4_COMBINED_MAX


def compute_raw_block_scale(block_amax: float, global_scale: float) -> float:
    """Return the pre-UE4M3 local scale from the hierarchical recipe."""
    block_amax = _require_amax(block_amax, "block_amax")
    global_scale = _require_amax(global_scale, "global_scale")
    if global_scale == 0.0:
        if block_amax == 0.0:
            return 0.0
        raise ValueError("a non-zero block cannot use a zero global scale")
    return (block_amax / E2M1_MAX) / global_scale
