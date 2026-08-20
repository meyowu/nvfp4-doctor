"""CPU-only extraction of kernel evidence from Nsight Systems CSV output."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, replace
from pathlib import Path

from nvfp4_doctor.env import (
    ArtifactEvidence,
    BackendEvidence,
    EnvironmentManifest,
    FallbackStatus,
)

E001_GEMM_RANGE = "e001:nvfp4_gemm"
_E001_CUTLASS_SIGNATURE = (
    "cutlass::device_kernel",
    "MainloopSm120TmaWarpSpecializedBlockScaled",
    "cutlass::float_e2m1_t",
    "SM120_16x8x64_TN_VS",
)
_KNOWN_FALLBACK_SIGNATURES = ("cublas",)


class NsightEvidenceError(ValueError):
    """Raised when profiler evidence is missing or malformed."""


@dataclass(frozen=True, slots=True)
class NsightKernelEvidence:
    report_sha256: str
    observed_kernels: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise NsightEvidenceError(f"could not read profiler report: {path}") from error
    return digest.hexdigest()


def parse_cuda_gpu_kernel_summary(payload: str) -> tuple[str, ...]:
    """Return unique kernel names from ``cuda_gpu_kern_sum`` CSV output."""
    lines = payload.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Name" in next(csv.reader((line,)), [])
        ),
        None,
    )
    if header_index is None:
        raise NsightEvidenceError("Nsight CSV must contain a Name column")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    if reader.fieldnames is None or "Name" not in reader.fieldnames:
        raise NsightEvidenceError("Nsight CSV must contain a Name column")

    names: list[str] = []
    seen: set[str] = set()
    for row in reader:
        name = (row.get("Name") or "").strip()
        if not name:
            raise NsightEvidenceError("Nsight CSV contains a blank kernel name")
        if name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        raise NsightEvidenceError("Nsight CSV contains no CUDA kernels")
    return tuple(names)


def extract_kernel_evidence(report_path: Path, stats_csv: str) -> NsightKernelEvidence:
    return NsightKernelEvidence(
        report_sha256=sha256_file(report_path),
        observed_kernels=parse_cuda_gpu_kernel_summary(stats_csv),
    )


def kernels_in_nvtx_range(
    observed_kernels: tuple[str, ...], range_name: str
) -> tuple[str, ...]:
    prefix = f"{range_name}/"
    return tuple(
        name.removeprefix(prefix)
        for name in observed_kernels
        if name.startswith(prefix)
    )


def assess_e001_fallback(observed_kernels: tuple[str, ...]) -> FallbackStatus:
    """Classify only kernels launched inside the E001 NVFP4 GEMM range.

    ``NOT_DETECTED`` means the expected SM120 block-scaled CUTLASS signature was
    observed and no known fallback signature was observed in that same range.
    Missing or unfamiliar evidence remains ``UNKNOWN``.
    """
    target_kernels = kernels_in_nvtx_range(observed_kernels, E001_GEMM_RANGE)
    if not target_kernels:
        return FallbackStatus.UNKNOWN
    if any(
        signature in kernel.lower()
        for kernel in target_kernels
        for signature in _KNOWN_FALLBACK_SIGNATURES
    ):
        return FallbackStatus.DETECTED
    if any(
        all(fragment in kernel for fragment in _E001_CUTLASS_SIGNATURE)
        for kernel in target_kernels
    ):
        return FallbackStatus.NOT_DETECTED
    return FallbackStatus.UNKNOWN


def attach_kernel_evidence(
    manifest: EnvironmentManifest,
    evidence: NsightKernelEvidence,
    report_path: Path,
) -> EnvironmentManifest:
    """Attach observations and a range-scoped fallback assessment."""
    backend = BackendEvidence(
        requested_format=manifest.backend.requested_format,
        requested_backend=manifest.backend.requested_backend,
        reported_backend=manifest.backend.reported_backend,
        observed_kernels=evidence.observed_kernels,
        fallback_status=assess_e001_fallback(evidence.observed_kernels),
        profiler_artifact_sha256=evidence.report_sha256,
    )
    artifact = ArtifactEvidence(
        kind="nsight-systems-report",
        sha256=evidence.report_sha256,
        local_path=str(report_path),
    )
    artifacts = tuple(item for item in manifest.artifacts if item.kind != artifact.kind)
    return replace(manifest, backend=backend, artifacts=(*artifacts, artifact))
