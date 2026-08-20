"""Versioned, CPU-only data model for E001 evidence manifests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ManifestValidationError(ValueError):
    """Raised when manifest data violates the versioned schema."""


class FallbackStatus(StrEnum):
    UNKNOWN = "unknown"
    NOT_DETECTED = "not_detected"
    DETECTED = "detected"


def _strict_keys(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ManifestValidationError(
            f"{context} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _non_empty(value: str, field: str) -> None:
    if not value.strip():
        raise ManifestValidationError(f"{field} must not be empty")


def _positive_tuple(values: tuple[int, ...], field: str) -> None:
    if not values or any(not isinstance(value, int) or value <= 0 for value in values):
        raise ManifestValidationError(f"{field} must contain positive integers")


@dataclass(frozen=True, slots=True)
class GPUFingerprint:
    name: str
    compute_capability: str
    memory_mib: int
    driver_version: str

    def __post_init__(self) -> None:
        for field in ("name", "compute_capability", "driver_version"):
            _non_empty(getattr(self, field), f"gpu.{field}")
        if self.memory_mib <= 0:
            raise ManifestValidationError("gpu.memory_mib must be positive")


@dataclass(frozen=True, slots=True)
class SoftwareFingerprint:
    os: str
    wsl_version: str
    kernel: str
    python: str
    torch: str
    cuda_runtime: str
    cuda_toolkit: str | None
    vllm: str
    flashinfer: str

    def __post_init__(self) -> None:
        for field in (
            "os",
            "wsl_version",
            "kernel",
            "python",
            "torch",
            "cuda_runtime",
            "vllm",
            "flashinfer",
        ):
            _non_empty(getattr(self, field), f"software.{field}")
        if self.cuda_toolkit is not None:
            _non_empty(self.cuda_toolkit, "software.cuda_toolkit")


@dataclass(frozen=True, slots=True)
class GitFingerprint:
    commit: str
    dirty: bool

    def __post_init__(self) -> None:
        if not _GIT_COMMIT.fullmatch(self.commit):
            raise ManifestValidationError(
                "git.commit must be a lowercase 40-digit hash"
            )


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    device: str

    def __post_init__(self) -> None:
        _non_empty(self.name, "tensor.name")
        _positive_tuple(self.logical_shape, f"tensor[{self.name}].logical_shape")
        _positive_tuple(self.physical_shape, f"tensor[{self.name}].physical_shape")
        if len(self.physical_shape) != len(self.stride):
            raise ManifestValidationError(
                f"tensor[{self.name}].stride rank must match physical_shape"
            )
        if any(not isinstance(value, int) or value < 0 for value in self.stride):
            raise ManifestValidationError(
                f"tensor[{self.name}].stride must contain non-negative integers"
            )
        _non_empty(self.dtype, f"tensor[{self.name}].dtype")
        _non_empty(self.device, f"tensor[{self.name}].device")


@dataclass(frozen=True, slots=True)
class BackendEvidence:
    requested_format: str
    requested_backend: str
    reported_backend: str | None
    observed_kernels: tuple[str, ...]
    fallback_status: FallbackStatus
    profiler_artifact_sha256: str | None

    def __post_init__(self) -> None:
        _non_empty(self.requested_format, "backend.requested_format")
        _non_empty(self.requested_backend, "backend.requested_backend")
        if self.reported_backend is not None:
            _non_empty(self.reported_backend, "backend.reported_backend")
        if any(not kernel.strip() for kernel in self.observed_kernels):
            raise ManifestValidationError(
                "backend.observed_kernels cannot contain blanks"
            )
        if self.profiler_artifact_sha256 is not None and not _SHA256.fullmatch(
            self.profiler_artifact_sha256
        ):
            raise ManifestValidationError(
                "backend.profiler_artifact_sha256 must be a lowercase SHA-256 hash"
            )


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    argv: tuple[str, ...]
    cwd: str
    seed: int

    def __post_init__(self) -> None:
        if not self.argv or any(not value for value in self.argv):
            raise ManifestValidationError(
                "command.argv must contain non-empty arguments"
            )
        _non_empty(self.cwd, "command.cwd")
        if self.seed < 0:
            raise ManifestValidationError("command.seed must be non-negative")


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    kind: str
    sha256: str
    local_path: str

    def __post_init__(self) -> None:
        _non_empty(self.kind, "artifact.kind")
        _non_empty(self.local_path, "artifact.local_path")
        if not _SHA256.fullmatch(self.sha256):
            raise ManifestValidationError(
                "artifact.sha256 must be a lowercase SHA-256 hash"
            )


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    schema_version: int
    experiment_id: str
    captured_at_utc: str
    gpu: GPUFingerprint
    software: SoftwareFingerprint
    git: GitFingerprint
    backend: BackendEvidence
    tensors: tuple[TensorMetadata, ...]
    command: CommandEvidence
    artifacts: tuple[ArtifactEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestValidationError(
                f"unsupported schema_version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.experiment_id != "E001-kernel-identity":
            raise ManifestValidationError(
                "experiment_id must be E001-kernel-identity for schema version 1"
            )
        try:
            timestamp = datetime.fromisoformat(
                self.captured_at_utc.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ManifestValidationError("captured_at_utc must be ISO 8601") from error
        offset = timestamp.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ManifestValidationError("captured_at_utc must include a UTC offset")
        if not self.tensors:
            raise ManifestValidationError("tensors must not be empty")
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ManifestValidationError("tensor names must be unique")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["backend"]["fallback_status"] = self.backend.fallback_status.value
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> Self:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ManifestValidationError("manifest is not valid JSON") from error
        if not isinstance(data, dict):
            raise ManifestValidationError("manifest root must be an object")
        return cls.from_dict(data)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        root_keys = {
            "schema_version",
            "experiment_id",
            "captured_at_utc",
            "gpu",
            "software",
            "git",
            "backend",
            "tensors",
            "command",
            "artifacts",
        }
        _strict_keys(data, root_keys, "manifest")
        nested = {
            "gpu": (
                GPUFingerprint,
                {"name", "compute_capability", "memory_mib", "driver_version"},
            ),
            "software": (
                SoftwareFingerprint,
                {
                    "os",
                    "wsl_version",
                    "kernel",
                    "python",
                    "torch",
                    "cuda_runtime",
                    "cuda_toolkit",
                    "vllm",
                    "flashinfer",
                },
            ),
            "git": (GitFingerprint, {"commit", "dirty"}),
            "backend": (
                BackendEvidence,
                {
                    "requested_format",
                    "requested_backend",
                    "reported_backend",
                    "observed_kernels",
                    "fallback_status",
                    "profiler_artifact_sha256",
                },
            ),
            "command": (CommandEvidence, {"argv", "cwd", "seed"}),
        }
        values: dict[str, Any] = {}
        for name, (model, keys) in nested.items():
            item = data[name]
            if not isinstance(item, dict):
                raise ManifestValidationError(f"{name} must be an object")
            _strict_keys(item, keys, name)
            values[name] = model(**item)

        backend_data = data["backend"]
        try:
            values["backend"] = BackendEvidence(
                **{
                    **backend_data,
                    "observed_kernels": tuple(backend_data["observed_kernels"]),
                    "fallback_status": FallbackStatus(backend_data["fallback_status"]),
                }
            )
        except (TypeError, ValueError) as error:
            raise ManifestValidationError("backend contains invalid values") from error

        tensor_keys = {
            "name",
            "logical_shape",
            "physical_shape",
            "dtype",
            "stride",
            "device",
        }
        tensors = []
        for index, item in enumerate(data["tensors"]):
            if not isinstance(item, dict):
                raise ManifestValidationError(f"tensors[{index}] must be an object")
            _strict_keys(item, tensor_keys, f"tensors[{index}]")
            tensors.append(
                TensorMetadata(
                    **{
                        **item,
                        "logical_shape": tuple(item["logical_shape"]),
                        "physical_shape": tuple(item["physical_shape"]),
                        "stride": tuple(item["stride"]),
                    }
                )
            )

        artifact_keys = {"kind", "sha256", "local_path"}
        artifacts = []
        for index, item in enumerate(data["artifacts"]):
            if not isinstance(item, dict):
                raise ManifestValidationError(f"artifacts[{index}] must be an object")
            _strict_keys(item, artifact_keys, f"artifacts[{index}]")
            artifacts.append(ArtifactEvidence(**item))

        try:
            return cls(
                schema_version=data["schema_version"],
                experiment_id=data["experiment_id"],
                captured_at_utc=data["captured_at_utc"],
                gpu=values["gpu"],
                software=values["software"],
                git=values["git"],
                backend=values["backend"],
                tensors=tuple(tensors),
                command=CommandEvidence(
                    **{**data["command"], "argv": tuple(data["command"]["argv"])}
                ),
                artifacts=tuple(artifacts),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ManifestValidationError):
                raise
            raise ManifestValidationError("manifest contains invalid values") from error
