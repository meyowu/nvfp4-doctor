#!/usr/bin/env python3
"""Run the bounded CPU-only E003 format-fault positive-control matrix."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nvfp4_doctor.contracts import FormatContractId, evaluate_format_contracts
from nvfp4_doctor.faults import (
    FaultInjection,
    inject_block_scale_reversal,
    inject_global_scale_multiplier,
    inject_nibble_swap,
    inject_padding_corruption,
    inject_scale_index_shift,
    inject_scale_layout_mislabel,
    revert_fault,
)
from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    pack_e2m1,
    swizzle_scales_128x4,
)
from nvfp4_doctor.oracle import NVFP4Tensor

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E003-synthetic-faults"
RESULTS = EXPERIMENT / "results.json"
MANIFEST = EXPERIMENT / "manifest.json"
SEED = 0
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "contracts" / "format.py",
    ROOT / "src" / "nvfp4_doctor" / "faults" / "nvfp4.py",
    ROOT / "tests" / "unit" / "test_format_faults.py",
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


def _packed_values(rows: int, columns: int) -> bytes:
    pattern = tuple(range(16))
    return pack_e2m1(
        pattern[(row * columns + column) % len(pattern)]
        for row in range(rows)
        for column in range(columns)
    )


def _linear_tensor() -> NVFP4Tensor:
    return NVFP4Tensor(
        rows=2,
        columns=32,
        packed_values=_packed_values(2, 32),
        scale_codes=bytes((0x38, 0x40, 0x48, 0x50)),
        scale_layout=ScaleFactorLayout.LINEAR,
        global_scale=0.5,
    )


def _cutlass_tensor(rows: int, columns: int) -> NVFP4Tensor:
    scale_columns = columns // 16
    linear_scales = bytes(
        0x38 + ((row * scale_columns + column) % 16)
        for row in range(rows)
        for column in range(scale_columns)
    )
    return NVFP4Tensor(
        rows=rows,
        columns=columns,
        packed_values=_packed_values(rows, columns),
        scale_codes=swizzle_scales_128x4(
            linear_scales, rows, scale_columns, padding_code=0
        ),
        scale_layout=ScaleFactorLayout.CUTLASS_128X4,
        global_scale=0.5,
    )


def _fault_matrix() -> tuple[tuple[FaultInjection, set[FormatContractId]], ...]:
    linear = _linear_tensor()
    layout = _cutlass_tensor(128, 64)
    padded = _cutlass_tensor(129, 80)
    return (
        (
            inject_nibble_swap(linear),
            {FormatContractId.PACKED_VALUES, FormatContractId.RECONSTRUCTION},
        ),
        (
            inject_scale_index_shift(linear, 1),
            {FormatContractId.LOGICAL_SCALES, FormatContractId.RECONSTRUCTION},
        ),
        (
            inject_block_scale_reversal(linear),
            {FormatContractId.LOGICAL_SCALES, FormatContractId.RECONSTRUCTION},
        ),
        (
            inject_global_scale_multiplier(linear, 2.0),
            {FormatContractId.GLOBAL_SCALE, FormatContractId.RECONSTRUCTION},
        ),
        (
            inject_scale_layout_mislabel(layout, ScaleFactorLayout.LINEAR),
            {
                FormatContractId.METADATA,
                FormatContractId.LOGICAL_SCALES,
                FormatContractId.RECONSTRUCTION,
            },
        ),
        (
            inject_padding_corruption(padded),
            {FormatContractId.SCALE_PADDING},
        ),
    )


def _case_result(
    injection: FaultInjection, expected_failures: set[FormatContractId]
) -> dict[str, Any]:
    outcomes = evaluate_format_contracts(injection.clean, injection.faulted)
    failed = {outcome.spec.contract_id for outcome in outcomes if not outcome.passed}
    return {
        "fault_kind": injection.kind.value,
        "label": injection.label,
        "parameters": dict(injection.parameters),
        "clean_shape": [injection.clean.rows, injection.clean.columns],
        "clean_layout": injection.clean.scale_layout.value,
        "faulted_layout": injection.faulted.scale_layout.value,
        "expected_failed_contracts": sorted(item.value for item in expected_failures),
        "observed_failed_contracts": sorted(item.value for item in failed),
        "exact_localization": failed == expected_failures,
        "detected": bool(failed),
        "reversible": revert_fault(injection) == injection.clean,
        "fault_changes_artifact": injection.clean != injection.faulted,
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
    clean_artifacts = (
        _linear_tensor(),
        _cutlass_tensor(128, 64),
        _cutlass_tensor(129, 80),
    )
    clean_outcomes = [
        evaluate_format_contracts(clean, clean) for clean in clean_artifacts
    ]
    clean_false_rejects = sum(
        not outcome.passed for outcomes in clean_outcomes for outcome in outcomes
    )

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
        "slice": "format_faults_v1",
        "execution_device": "cpu",
        "seed": SEED,
        "clean_artifacts_checked": len(clean_artifacts),
        "clean_contract_evaluations": sum(len(item) for item in clean_outcomes),
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
            "Detection and exact localization apply only to the six deterministic "
            "CPU format controls in this matrix. Stride, non-contiguous storage, "
            "backend identity, dispatch, GEMM, and model propagation are not tested."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E003-synthetic-faults",
        "slice": "format_faults_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e003_format_faults.py"],
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
        "model": {"status": "not_applicable_synthetic_format_controls"},
        "backend": {
            "requested": "cpu_oracle",
            "reported": "cpu_oracle",
            "observed_kernel": None,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "format-fault-results",
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
