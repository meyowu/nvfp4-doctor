#!/usr/bin/env python3
"""Evaluate packed-value permutation controls on a held-out CPU matrix."""

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
    inject_packed_block_permutation,
    inject_packed_column_permutation,
    inject_packed_row_permutation,
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
RESULTS = EXPERIMENT / "results-heldout.json"
MANIFEST = EXPERIMENT / "manifest-heldout.json"
SEED = 20260820
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "contracts" / "format.py",
    ROOT / "src" / "nvfp4_doctor" / "faults" / "nvfp4.py",
    ROOT / "tests" / "unit" / "test_packed_permutation_faults.py",
    ROOT / "tests" / "unit" / "test_e003_evidence.py",
    Path(__file__).resolve(),
)
EXPECTED_FAILURES = {
    FormatContractId.PACKED_VALUES,
    FormatContractId.RECONSTRUCTION,
}


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


def _heldout_tensor(
    rows: int,
    columns: int,
    layout: ScaleFactorLayout,
    salt: int,
) -> NVFP4Tensor:
    scale_columns = columns // 16
    packed_values = pack_e2m1(
        (salt + row * 5 + column * 3 + (row * column) % 7 + (column // 16) * 2) % 16
        for row in range(rows)
        for column in range(columns)
    )
    linear_scales = bytes(
        0x38 + ((salt + row * 3 + block * 5) % 16)
        for row in range(rows)
        for block in range(scale_columns)
    )
    scale_codes = (
        linear_scales
        if layout == ScaleFactorLayout.LINEAR
        else swizzle_scales_128x4(
            linear_scales,
            rows,
            scale_columns,
            padding_code=0,
        )
    )
    return NVFP4Tensor(
        rows=rows,
        columns=columns,
        packed_values=packed_values,
        scale_codes=scale_codes,
        scale_layout=layout,
        global_scale=0.5,
    )


def _heldout_clean_matrix() -> tuple[tuple[str, NVFP4Tensor], ...]:
    return (
        (
            "linear_3x48_salt11",
            _heldout_tensor(3, 48, ScaleFactorLayout.LINEAR, 11),
        ),
        (
            "linear_5x64_salt23",
            _heldout_tensor(5, 64, ScaleFactorLayout.LINEAR, 23),
        ),
        (
            "cutlass_131x80_salt37",
            _heldout_tensor(131, 80, ScaleFactorLayout.CUTLASS_128X4, 37),
        ),
    )


def _fault_matrix() -> tuple[tuple[str, FaultInjection], ...]:
    cases: list[tuple[str, FaultInjection]] = []
    for case_index, (artifact_id, clean) in enumerate(_heldout_clean_matrix()):
        cases.extend(
            (
                (
                    artifact_id,
                    inject_packed_block_permutation(clean, 1 + case_index),
                ),
                (
                    artifact_id,
                    inject_packed_row_permutation(clean, 1 + case_index),
                ),
                (
                    artifact_id,
                    inject_packed_column_permutation(clean, 3 + 2 * case_index),
                ),
            )
        )
    return tuple(cases)


def _case_result(artifact_id: str, injection: FaultInjection) -> dict[str, Any]:
    outcomes = evaluate_format_contracts(injection.clean, injection.faulted)
    failed = {outcome.spec.contract_id for outcome in outcomes if not outcome.passed}
    return {
        "artifact_id": artifact_id,
        "evaluation_role": "held_out",
        "fault_kind": injection.kind.value,
        "label": injection.label,
        "parameters": dict(injection.parameters),
        "clean_shape": [injection.clean.rows, injection.clean.columns],
        "clean_layout": injection.clean.scale_layout.value,
        "expected_failed_contracts": sorted(item.value for item in EXPECTED_FAILURES),
        "observed_failed_contracts": sorted(item.value for item in failed),
        "exact_localization": failed == EXPECTED_FAILURES,
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
    clean_matrix = _heldout_clean_matrix()
    clean_outcomes = [
        evaluate_format_contracts(clean, clean) for _, clean in clean_matrix
    ]
    clean_false_rejects = sum(
        not outcome.passed for outcomes in clean_outcomes for outcome in outcomes
    )

    cases = tuple(
        _case_result(artifact_id, injection)
        for artifact_id, injection in _fault_matrix()
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
        "slice": "packed_permutation_heldout_v1",
        "execution_device": "cpu",
        "seed": SEED,
        "evaluation_role": "held_out",
        "threshold_policy": {
            "source": "Gate 1 exact zero-mismatch contract thresholds",
            "frozen_before_matrix_construction": True,
            "tuned_on_held_out_cases": False,
        },
        "clean_artifact_ids": [artifact_id for artifact_id, _ in clean_matrix],
        "clean_artifacts_checked": len(clean_matrix),
        "clean_contract_evaluations": sum(len(item) for item in clean_outcomes),
        "clean_false_rejects": clean_false_rejects,
        "fault_cases": list(cases),
        "faults_injected": len(cases),
        "faults_detected": sum(case["detected"] for case in cases),
        "false_accepts": false_accepts,
        "localization_failures": localization_failures,
        "reversibility_failures": reversibility_failures,
        "slice_status": "pass" if passed else "fail",
        "e003_status": "complete" if passed else "in_progress",
        "decision": "continue" if passed else "repeat",
        "claim_boundary": (
            "Detection and exact localization apply only to three deterministic "
            "packed-value permutation families over this held-out CPU matrix. "
            "Runtime storage, dispatch, GEMM, and model propagation are not tested."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E003-synthetic-faults",
        "slice": "packed_permutation_heldout_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e003_heldout_permutations.py"],
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
                "kind": "heldout-packed-permutation-results",
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
    print(f"e003_status={results['e003_status']}")
    print(f"decision={results['decision']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
