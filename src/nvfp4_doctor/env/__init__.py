"""Immutable environment fingerprinting boundaries."""

from .assembly import assemble_e001_manifest, collect_git
from .collectors import (
    CollectionError,
    CommandResult,
    collect_gpu,
    collect_software,
)
from .e001 import e001_smoke_tensors
from .manifest import (
    SCHEMA_VERSION,
    ArtifactEvidence,
    BackendEvidence,
    CommandEvidence,
    EnvironmentManifest,
    FallbackStatus,
    GitFingerprint,
    GPUFingerprint,
    ManifestValidationError,
    SoftwareFingerprint,
    TensorMetadata,
)

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactEvidence",
    "BackendEvidence",
    "CollectionError",
    "CommandEvidence",
    "CommandResult",
    "EnvironmentManifest",
    "FallbackStatus",
    "GPUFingerprint",
    "GitFingerprint",
    "ManifestValidationError",
    "SoftwareFingerprint",
    "TensorMetadata",
    "assemble_e001_manifest",
    "collect_git",
    "collect_gpu",
    "collect_software",
    "e001_smoke_tensors",
]
