"""Machine-readable repeatability summary for E001 manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nvfp4_doctor.backends import E001_GEMM_RANGE, kernels_in_nvtx_range
from nvfp4_doctor.env import EnvironmentManifest, FallbackStatus


class E001SummaryError(ValueError):
    """Raised when manifests cannot support a repeatability summary."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _controlled_fingerprint(manifest: EnvironmentManifest) -> tuple[Any, ...]:
    return (
        manifest.gpu,
        manifest.software,
        manifest.git,
        manifest.backend.requested_format,
        manifest.backend.requested_backend,
        manifest.tensors,
        manifest.command,
    )


def summarize_e001_manifests(paths: tuple[Path, ...]) -> dict[str, Any]:
    if len(paths) < 3:
        raise E001SummaryError("E001 repeatability requires at least three manifests")

    manifests: list[EnvironmentManifest] = []
    payloads: list[bytes] = []
    for path in paths:
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise E001SummaryError(f"could not read E001 manifest: {path}") from error
        payloads.append(payload)
        manifests.append(EnvironmentManifest.from_json(payload.decode("utf-8")))

    fingerprints = {_controlled_fingerprint(manifest) for manifest in manifests}
    target_hashes: list[str] = []
    runs: list[dict[str, Any]] = []
    for path, payload, manifest in zip(paths, payloads, manifests, strict=True):
        target_kernels = tuple(
            sorted(
                kernels_in_nvtx_range(
                    manifest.backend.observed_kernels, E001_GEMM_RANGE
                )
            )
        )
        target_hash = _sha256_bytes("\n".join(target_kernels).encode("utf-8"))
        target_hashes.append(target_hash)
        runs.append(
            {
                "run_id": path.stem,
                "captured_at_utc": manifest.captured_at_utc,
                "manifest_sha256": _sha256_bytes(payload),
                "profiler_sha256": manifest.backend.profiler_artifact_sha256,
                "observed_kernel_count": len(manifest.backend.observed_kernels),
                "target_kernel_count": len(target_kernels),
                "target_kernel_set_sha256": target_hash,
                "fallback_status": manifest.backend.fallback_status.value,
            }
        )

    environment_stable = len(fingerprints) == 1
    target_kernel_set_stable = len(set(target_hashes)) == 1
    profiler_evidence_complete = all(
        manifest.backend.profiler_artifact_sha256 is not None for manifest in manifests
    )
    no_known_fallback_detected = all(
        manifest.backend.fallback_status == FallbackStatus.NOT_DETECTED
        for manifest in manifests
    )
    gate0_repeatability = all(
        (
            environment_stable,
            target_kernel_set_stable,
            profiler_evidence_complete,
            no_known_fallback_detected,
        )
    )

    first = manifests[0]
    return {
        "schema_version": 1,
        "experiment_id": "E001-kernel-identity",
        "repetitions": len(manifests),
        "requested_format": first.backend.requested_format,
        "requested_backend": first.backend.requested_backend,
        "reported_backend": first.backend.reported_backend,
        "target_nvtx_range": E001_GEMM_RANGE,
        "environment_stable": environment_stable,
        "target_kernel_set_stable": target_kernel_set_stable,
        "profiler_evidence_complete": profiler_evidence_complete,
        "no_known_fallback_detected": no_known_fallback_detected,
        "gate0_repeatability": "pass" if gate0_repeatability else "fail",
        "decision": "go" if gate0_repeatability else "repeat",
        "claim_boundary": (
            "No known fallback signature was detected inside the target NVTX "
            "range; this does not prove that all silent fallback modes are impossible."
        ),
        "runs": runs,
    }
