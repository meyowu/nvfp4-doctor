#!/usr/bin/env python3
"""Run the CPU-only E003 tensor-metadata and backend-evidence fault matrix."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nvfp4_doctor.contracts import (
    EvidenceContractId,
    ExecutionEvidence,
    evaluate_execution_evidence,
)
from nvfp4_doctor.env import BackendEvidence, FallbackStatus, TensorMetadata
from nvfp4_doctor.faults import (
    EvidenceFaultInjection,
    inject_observed_fallback_kernel,
    inject_reported_backend_mismatch,
    inject_requested_backend_mismatch,
    inject_stride_axis_permutation,
    inject_stride_gap,
    revert_evidence_fault,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E003-synthetic-faults"
RESULTS = EXPERIMENT / "results-execution.json"
MANIFEST = EXPERIMENT / "manifest-execution.json"
SEED = 0
CLEAN_KERNEL = (
    "e003:nvfp4_gemm/cutlass::device_kernel<"
    "MainloopSm120TmaWarpSpecializedBlockScaled,"
    "cutlass::float_e2m1_t,SM120_16x8x64_TN_VS>"
)
FALLBACK_KERNEL = "e003:nvfp4_gemm/cublasGemmEx"
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "contracts" / "execution.py",
    ROOT / "src" / "nvfp4_doctor" / "faults" / "evidence.py",
    ROOT / "tests" / "unit" / "test_execution_evidence_faults.py",
    Path(__file__).resolve(),
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _command(*argv: str) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.replace("\x00", "").strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _clean_evidence() -> ExecutionEvidence:
    return ExecutionEvidence(
        tensor=TensorMetadata(
            name="packed_input",
            logical_shape=(4, 16),
            physical_shape=(4, 8),
            dtype="uint8",
            stride=(8, 1),
            device="cuda:0",
        ),
        backend=BackendEvidence(
            requested_format="nvfp4",
            requested_backend="cutlass",
            reported_backend="cutlass",
            observed_kernels=(CLEAN_KERNEL,),
            fallback_status=FallbackStatus.NOT_DETECTED,
            profiler_artifact_sha256=None,
        ),
    )


def _fault_matrix() -> tuple[
    tuple[EvidenceFaultInjection, set[EvidenceContractId]], ...
]:
    clean = _clean_evidence()
    return (
        (
            inject_stride_axis_permutation(clean),
            {
                EvidenceContractId.TENSOR_STRIDE,
                EvidenceContractId.TENSOR_CONTIGUITY,
            },
        ),
        (
            inject_stride_gap(clean, 8),
            {
                EvidenceContractId.TENSOR_STRIDE,
                EvidenceContractId.TENSOR_CONTIGUITY,
            },
        ),
        (
            inject_requested_backend_mismatch(clean, "cublas"),
            {EvidenceContractId.REQUESTED_BACKEND},
        ),
        (
            inject_reported_backend_mismatch(clean, "cublas"),
            {EvidenceContractId.REPORTED_BACKEND},
        ),
        (
            inject_observed_fallback_kernel(clean, FALLBACK_KERNEL),
            {
                EvidenceContractId.OBSERVED_KERNELS,
                EvidenceContractId.FALLBACK_STATUS,
            },
        ),
    )


def _case_result(
    injection: EvidenceFaultInjection,
    expected_failures: set[EvidenceContractId],
) -> dict[str, Any]:
    outcomes = evaluate_execution_evidence(injection.clean, injection.faulted)
    failed = {outcome.spec.contract_id for outcome in outcomes if not outcome.passed}
    return {
        "fault_kind": injection.kind.value,
        "label": injection.label,
        "parameters": dict(injection.parameters),
        "expected_failed_contracts": sorted(item.value for item in expected_failures),
        "observed_failed_contracts": sorted(item.value for item in failed),
        "exact_localization": failed == expected_failures,
        "detected": bool(failed),
        "reversible": revert_evidence_fault(injection) == injection.clean,
        "fault_changes_evidence": injection.clean != injection.faulted,
        "tensor": {
            "clean_stride": list(injection.clean.tensor.stride),
            "faulted_stride": list(injection.faulted.tensor.stride),
        },
        "backend": {
            "clean": {
                "requested": injection.clean.backend.requested_backend,
                "reported": injection.clean.backend.reported_backend,
                "observed_kernels": list(injection.clean.backend.observed_kernels),
                "fallback_status": injection.clean.backend.fallback_status.value,
            },
            "faulted": {
                "requested": injection.faulted.backend.requested_backend,
                "reported": injection.faulted.backend.reported_backend,
                "observed_kernels": list(injection.faulted.backend.observed_kernels),
                "fallback_status": injection.faulted.backend.fallback_status.value,
            },
        },
        "contract_outcomes": [
            {
                "contract_id": outcome.spec.contract_id.value,
                "passed": outcome.passed,
                "mismatch_count": outcome.mismatch_count,
                "threshold": outcome.spec.threshold,
                "details": outcome.details,
            }
            for outcome in outcomes
        ],
    }


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    clean = _clean_evidence()
    clean_outcomes = evaluate_execution_evidence(clean, clean)
    clean_false_rejects = sum(not outcome.passed for outcome in clean_outcomes)
    cases = tuple(
        _case_result(injection, expected) for injection, expected in _fault_matrix()
    )
    false_accepts = sum(not case["detected"] for case in cases)
    localization_failures = sum(not case["exact_localization"] for case in cases)
    reversibility_failures = sum(not case["reversible"] for case in cases)
    passed = (
        clean_false_rejects == 0
        and false_accepts == 0
        and localization_failures == 0
        and reversibility_failures == 0
    )
    results = {
        "schema_version": 1,
        "experiment_id": "E003-synthetic-faults",
        "slice": "execution_evidence_faults_v1",
        "execution_device": "cpu",
        "seed": SEED,
        "clean_contract_evaluations": len(clean_outcomes),
        "clean_false_rejects": clean_false_rejects,
        "fault_cases": list(cases),
        "faults_injected": len(cases),
        "faults_detected": sum(case["detected"] for case in cases),
        "false_accepts": false_accepts,
        "localization_failures": localization_failures,
        "reversibility_failures": reversibility_failures,
        "slice_status": "pass" if passed else "fail",
        "decision": "continue" if passed else "repeat",
        "claim_boundary": (
            "Detection and exact localization apply only to five synthetic CPU "
            "metadata/evidence controls. Observed kernel strings are explicit "
            "synthetic inputs, not profiler observations from this run."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E003-synthetic-faults",
        "slice": "execution_evidence_faults_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e003_execution_faults.py"],
            "cwd": str(ROOT),
            "seed": SEED,
        },
        "git": {
            "branch": _command("git", "branch", "--show-current"),
            "commit": _command("git", "rev-parse", "HEAD"),
            "dirty": bool(git_status),
        },
        "platform": {
            "os": platform.system(),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "software": {
            "torch": _package_version("torch"),
            "vllm": _package_version("vllm"),
            "flashinfer": _package_version("flashinfer-python"),
        },
        "model": {"status": "not_applicable_synthetic_execution_evidence"},
        "execution_backend": {
            "requested": "cpu_oracle",
            "reported": "cpu_oracle",
            "observed_kernel": None,
        },
        "synthetic_backend_baseline": {
            "requested": clean.backend.requested_backend,
            "reported": clean.backend.reported_backend,
            "observed_kernels": list(clean.backend.observed_kernels),
            "fallback_status": clean.backend.fallback_status.value,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "execution-evidence-fault-results",
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(RESULTS),
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"results={RESULTS}")
    print(f"manifest={MANIFEST}")
    print(f"clean_false_rejects={clean_false_rejects}")
    print(f"false_accepts={false_accepts}")
    print(f"localization_failures={localization_failures}")
    print(f"reversibility_failures={reversibility_failures}")
    print(f"slice_status={results['slice_status']}")
    print(f"decision={results['decision']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
