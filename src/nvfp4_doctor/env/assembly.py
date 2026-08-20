"""Assemble E001 manifests while preserving unknown execution evidence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from .collectors import (
    CollectionError,
    CommandRunner,
    PackageVersionProvider,
    collect_gpu,
    collect_software,
    installed_version,
    run_command,
)
from .manifest import (
    SCHEMA_VERSION,
    BackendEvidence,
    CommandEvidence,
    EnvironmentManifest,
    FallbackStatus,
    GitFingerprint,
    TensorMetadata,
)


def collect_git(runner: CommandRunner = run_command) -> GitFingerprint:
    commit_result = runner(("git", "rev-parse", "HEAD"))
    if commit_result.returncode != 0 or not commit_result.stdout.strip():
        raise CollectionError(
            "git rev-parse failed; "
            f"returncode={commit_result.returncode}; stderr={commit_result.stderr!r}"
        )
    commit = commit_result.stdout.strip()
    status = runner(("git", "status", "--porcelain", "--untracked-files=normal"))
    if status.returncode != 0:
        raise CollectionError(
            f"git status failed with returncode={status.returncode}: {status.stderr!r}"
        )
    return GitFingerprint(commit=commit, dirty=bool(status.stdout.strip()))


def utc_now() -> datetime:
    return datetime.now(UTC)


def assemble_e001_manifest(
    *,
    tensors: tuple[TensorMetadata, ...],
    command: CommandEvidence,
    requested_format: str = "nvfp4",
    requested_backend: str = "cutlass",
    runner: CommandRunner = run_command,
    versions: PackageVersionProvider = installed_version,
    clock: Callable[[], datetime] = utc_now,
    python_version: str | None = None,
    nvcc_executable: str | None = None,
) -> EnvironmentManifest:
    """Collect known evidence without inferring profiler-derived fields."""
    captured_at = clock()
    if captured_at.utcoffset() is None or captured_at.utcoffset() != UTC.utcoffset(
        None
    ):
        raise ValueError("clock must return a UTC-aware datetime")
    return EnvironmentManifest(
        schema_version=SCHEMA_VERSION,
        experiment_id="E001-kernel-identity",
        captured_at_utc=captured_at.isoformat().replace("+00:00", "Z"),
        gpu=collect_gpu(runner),
        software=collect_software(
            runner,
            versions,
            python_version=python_version,
            nvcc_executable=nvcc_executable,
        ),
        git=collect_git(runner),
        backend=BackendEvidence(
            requested_format=requested_format,
            requested_backend=requested_backend,
            reported_backend=None,
            observed_kernels=(),
            fallback_status=FallbackStatus.UNKNOWN,
            profiler_artifact_sha256=None,
        ),
        tensors=tensors,
        command=command,
        artifacts=(),
    )
