"""Declared tensor metadata for the checked-in E001 synthetic workload."""

from __future__ import annotations

from .manifest import TensorMetadata


def e001_smoke_tensors() -> tuple[TensorMetadata, ...]:
    """Describe logical values separately from packed or padded storage."""
    return (
        TensorMetadata("a", (16, 256), (16, 256), "bfloat16", (256, 1), "cuda:0"),
        TensorMetadata("b", (128, 256), (128, 256), "bfloat16", (256, 1), "cuda:0"),
        TensorMetadata("a_fp4", (16, 256), (16, 128), "uint8", (128, 1), "cuda:0"),
        TensorMetadata("b_fp4", (128, 256), (128, 128), "uint8", (128, 1), "cuda:0"),
        TensorMetadata("a_scale", (16, 16), (128, 16), "uint8", (16, 1), "cuda:0"),
        TensorMetadata("b_scale", (128, 16), (128, 16), "uint8", (16, 1), "cuda:0"),
        TensorMetadata("output", (16, 128), (16, 128), "bfloat16", (128, 1), "cuda:0"),
    )
