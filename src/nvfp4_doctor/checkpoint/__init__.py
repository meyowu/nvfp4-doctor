"""Checkpoint metadata and model-adapter boundaries."""

from .acquisition import (
    AcquisitionPlanError,
    TensorByteRange,
    plan_tensor_byte_ranges,
)
from .modelopt import (
    REQUIRED_QUANT_TENSORS,
    TARGET_PROJECTIONS,
    CheckpointMetadataError,
    ModelOptCheckpointInspection,
    ProjectionInventory,
    inspect_modelopt_checkpoint,
)
from .safetensors import (
    SafetensorsHeader,
    SafetensorsHeaderError,
    SafetensorsTensor,
    parse_safetensors_header,
)

__all__ = [
    "REQUIRED_QUANT_TENSORS",
    "TARGET_PROJECTIONS",
    "AcquisitionPlanError",
    "CheckpointMetadataError",
    "ModelOptCheckpointInspection",
    "ProjectionInventory",
    "SafetensorsHeader",
    "SafetensorsHeaderError",
    "SafetensorsTensor",
    "TensorByteRange",
    "inspect_modelopt_checkpoint",
    "parse_safetensors_header",
    "plan_tensor_byte_ranges",
]
