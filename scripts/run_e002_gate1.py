#!/usr/bin/env python3
"""Run exact E002 fixtures and a FlashInfer format differential on CUDA."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    cutlass_128x4_offset,
    decode_e2m1,
    decode_ue4m3,
    encode_e2m1,
    encode_ue4m3,
    pack_e2m1,
    scale_physical_shape,
    swizzle_scales_128x4,
    unpack_e2m1,
)
from nvfp4_doctor.oracle import NVFP4Tensor, reconstruct_nvfp4

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "E002-format-oracle"
FIXTURE = ROOT / "tests" / "fixtures" / "e002_format_oracle_v1.json"
SOURCES = EXPERIMENT_DIR / "sources.json"
RESULTS = EXPERIMENT_DIR / "results.json"
MANIFEST = EXPERIMENT_DIR / "manifest.json"
SEED = 0
VALUE_PATTERN = (0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15)
ORACLE_SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "formats" / "e2m1.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "ue4m3.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "packing.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "layout.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "scales.py",
    ROOT / "src" / "nvfp4_doctor" / "oracle" / "nvfp4.py",
    ROOT / "docs" / "nvfp4-contract.md",
    Path(__file__).resolve(),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256(path.read_bytes())


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in (*ORACLE_SOURCE_PATHS, FIXTURE, SOURCES):
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


def _validate_fixtures() -> dict[str, Any]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    e2m1_cases = fixture["e2m1_decode"]
    if len(e2m1_cases) != 16:
        raise AssertionError("the E2M1 fixture must cover all sixteen payloads")
    for case in e2m1_cases:
        observed = decode_e2m1(case["code"])
        if observed != case["value"] or math.copysign(1.0, observed) != math.copysign(
            1.0, case["value"]
        ):
            raise AssertionError(f"E2M1 fixture mismatch for code {case['code']}")
        if encode_e2m1(observed) != case["code"]:
            raise AssertionError(f"E2M1 round trip failed for code {case['code']}")

    ue4m3_cases = fixture["ue4m3_decode"]
    for case in ue4m3_cases:
        if decode_ue4m3(case["code"]) != case["value"]:
            raise AssertionError(f"UE4M3 fixture mismatch for code {case['code']}")
    finite_scales = tuple(decode_ue4m3(code) for code in range(0x7F))
    if any(left >= right for left, right in pairwise(finite_scales)):
        raise AssertionError("finite UE4M3 codes must be strictly ordered")
    if any(encode_ue4m3(value) != code for code, value in enumerate(finite_scales)):
        raise AssertionError("finite UE4M3 codes must round trip")

    packing = fixture["packing"]
    packed = pack_e2m1(packing["logical_codes"])
    if packed != bytes(packing["packed_bytes"]):
        raise AssertionError("packed E2M1 fixture mismatch")
    if unpack_e2m1(packed) != tuple(packing["logical_codes"]):
        raise AssertionError("packed E2M1 fixture did not round trip")

    for case in fixture["layout_offsets"]:
        observed = cutlass_128x4_offset(
            case["row"],
            case["scale_column"],
            case["rows"],
            case["scale_columns"],
        )
        if observed != case["offset"]:
            raise AssertionError(f"scale-layout fixture mismatch: {case}")

    reconstruction = fixture["reconstruction"]
    tensor = NVFP4Tensor(
        rows=reconstruction["rows"],
        columns=reconstruction["columns"],
        packed_values=bytes(reconstruction["packed_bytes"]),
        scale_codes=bytes(reconstruction["scale_codes"]),
        scale_layout=ScaleFactorLayout.LINEAR,
        global_scale=reconstruction["global_scale"],
        block_size=reconstruction["block_size"],
    )
    observed_values = reconstruct_nvfp4(tensor)[0]
    if observed_values != tuple(reconstruction["values"]):
        raise AssertionError("hierarchical reconstruction fixture mismatch")

    return {
        "e2m1_payloads_checked": len(e2m1_cases),
        "finite_ue4m3_codes_checked": len(finite_scales),
        "golden_layout_offsets_checked": len(fixture["layout_offsets"]),
        "golden_packed_bytes_checked": len(packing["packed_bytes"]),
        "hierarchical_reconstruction": "exact",
        "status": "pass",
    }


def _raw_bytes(tensor: Any) -> bytes:
    import torch

    return bytes(tensor.view(torch.uint8).cpu().reshape(-1).tolist())


def _build_exact_case(
    rows: int, columns: int, global_scale: float
) -> tuple[list[list[float]], bytes, bytes]:
    scale_columns = columns // 16
    values: list[list[float]] = []
    packed = bytearray()
    linear_scale_codes = bytearray()
    for row in range(rows):
        row_values: list[float] = []
        row_codes: list[int] = []
        for block in range(scale_columns):
            scale_code = (row * scale_columns + block) % 0x7F
            scale = decode_ue4m3(scale_code)
            codes = (0,) * 16 if scale_code == 0 else VALUE_PATTERN
            linear_scale_codes.append(scale_code)
            row_codes.extend(codes)
            row_values.extend(
                decode_e2m1(code) * scale * global_scale for code in codes
            )
        packed.extend(pack_e2m1(row_codes))
        values.append(row_values)
    return values, bytes(packed), bytes(linear_scale_codes)


def _run_flashinfer_case(
    rows: int, columns: int, global_scale: float
) -> dict[str, Any]:
    import torch
    from flashinfer import SfLayout, nvfp4_quantize

    values, expected_packed, expected_linear_scales = _build_exact_case(
        rows, columns, global_scale
    )
    source = torch.tensor(values, device="cuda", dtype=torch.bfloat16)
    inverse_global_scale = torch.tensor(
        [1.0 / global_scale], device="cuda", dtype=torch.float32
    )
    packed_linear, scales_linear = nvfp4_quantize(
        source,
        inverse_global_scale,
        sfLayout=SfLayout.layout_linear,
        do_shuffle=False,
        backend="cuda",
    )
    packed_swizzled, scales_swizzled = nvfp4_quantize(
        source,
        inverse_global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    torch.cuda.synchronize()

    actual_packed_linear = _raw_bytes(packed_linear)
    actual_packed_swizzled = _raw_bytes(packed_swizzled)
    actual_linear_scales = _raw_bytes(scales_linear)
    actual_swizzled_scales = _raw_bytes(scales_swizzled)
    scale_columns = columns // 16
    expected_swizzled_scales = swizzle_scales_128x4(
        expected_linear_scales, rows, scale_columns
    )
    reconstructed = reconstruct_nvfp4(
        NVFP4Tensor(
            rows=rows,
            columns=columns,
            packed_values=actual_packed_swizzled,
            scale_codes=actual_swizzled_scales,
            scale_layout=ScaleFactorLayout.CUTLASS_128X4,
            global_scale=global_scale,
        )
    )
    source_values = tuple(tuple(float(value) for value in row) for row in source.cpu())
    max_abs_error = max(
        abs(observed - expected)
        for observed_row, expected_row in zip(reconstructed, source_values)
        for observed, expected in zip(observed_row, expected_row)
    )
    checks = {
        "linear_packed_values_exact": actual_packed_linear == expected_packed,
        "swizzled_packed_values_exact": actual_packed_swizzled == expected_packed,
        "linear_scales_exact": actual_linear_scales == expected_linear_scales,
        "cutlass_128x4_scales_exact": (
            actual_swizzled_scales == expected_swizzled_scales
        ),
        "hierarchical_reconstruction_exact": max_abs_error == 0.0,
    }
    if not all(checks.values()):
        raise AssertionError(f"FlashInfer differential failed: {checks}")
    return {
        "checks": checks,
        "columns": columns,
        "global_scale": global_scale,
        "inverse_global_scale_passed_to_flashinfer": 1.0 / global_scale,
        "global_scale_metadata": {
            "dtype": "float32",
            "logical_shape": [],
            "value": global_scale,
        },
        "logical_scale_shape": [rows, scale_columns],
        "max_abs_reconstruction_error": max_abs_error,
        "packed_sha256": _sha256(actual_packed_swizzled),
        "physical_scale_shape": list(
            scale_physical_shape(rows, scale_columns, ScaleFactorLayout.CUTLASS_128X4)
        ),
        "rows": rows,
        "scale_codes_covered": len(set(expected_linear_scales)),
        "swizzled_scale_sha256": _sha256(actual_swizzled_scales),
        "tensors": {
            "input": {
                "dtype": str(source.dtype),
                "shape": list(source.shape),
                "stride": list(source.stride()),
            },
            "packed": {
                "dtype": str(packed_swizzled.dtype),
                "physical_shape": list(packed_swizzled.shape),
                "stride": list(packed_swizzled.stride()),
            },
            "scale_linear": {
                "dtype": str(scales_linear.dtype),
                "physical_shape": list(scales_linear.shape),
                "stride": list(scales_linear.stride()),
            },
            "scale_swizzled": {
                "dtype": str(scales_swizzled.dtype),
                "physical_shape": list(scales_swizzled.shape),
                "stride": list(scales_swizzled.stride()),
            },
        },
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E002 FlashInfer differential requires CUDA")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("E002 is pinned to the RTX 5080 sm_120 matrix")
    torch.manual_seed(SEED)

    fixture_checks = _validate_fixtures()
    cases = (
        _run_flashinfer_case(128, 64, 1.0),
        _run_flashinfer_case(17, 80, 0.5),
        _run_flashinfer_case(129, 80, 2.0),
    )
    results = {
        "adapter_differential": "pass",
        "claim_boundary": (
            "Exact agreement applies only to the constructed, exactly representable "
            "row-wise block-16 cases and the pinned FlashInfer layouts; it does not "
            "establish GEMM correctness or a numerical-error envelope."
        ),
        "decision": "go",
        "experiment_id": "E002-format-oracle",
        "fixture_checks": fixture_checks,
        "gpu_cases": list(cases),
        "oracle_independent_of_candidate": True,
        "schema_version": 1,
    }
    result_payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    RESULTS.write_text(result_payload, encoding="utf-8")

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    manifest = {
        "artifacts": [
            {
                "kind": "gate1-results",
                "path": str(RESULTS.relative_to(ROOT)),
                "sha256": _sha256(result_payload.encode("utf-8")),
            },
            {
                "kind": "golden-fixture",
                "path": str(FIXTURE.relative_to(ROOT)),
                "sha256": _sha256_path(FIXTURE),
            },
            {
                "kind": "public-source-record",
                "path": str(SOURCES.relative_to(ROOT)),
                "sha256": _sha256_path(SOURCES),
            },
        ],
        "candidate_adapter": {
            "flashinfer": importlib.metadata.version("flashinfer-python"),
            "observed_kernel": None,
            "reported_backend": None,
            "requested_backend": "cuda",
            "requested_layouts": ["linear", "cutlass_128x4"],
            "torch": importlib.metadata.version("torch"),
        },
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e002_gate1.py"],
            "cwd": str(ROOT),
            "seed": SEED,
        },
        "experiment_id": "E002-format-oracle",
        "git": {
            "branch": _command("git", "branch", "--show-current"),
            "commit": _command("git", "rev-parse", "HEAD"),
            "dirty": bool(_command("git", "status", "--porcelain")),
        },
        "gpu": {
            "compute_capability": "12.0",
            "name": torch.cuda.get_device_name(0),
            "nvidia_smi": _command(
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ),
        },
        "model": {
            "revision": None,
            "status": "not_applicable_synthetic_format_fixture",
        },
        "oracle": {
            "public_sources": [source["id"] for source in sources["sources"]],
            "source_bundle_sha256": _source_bundle_sha256(),
        },
        "platform": {
            "kernel": platform.release(),
            "os": platform.system(),
            "python": platform.python_version(),
            "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
            "wsl_version": _command("wsl.exe", "--version").splitlines()[0],
        },
        "software": {
            "cuda_runtime": importlib.metadata.version("nvidia-cuda-runtime"),
            "cuda_toolkit": _command(
                str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"), "--version"
            ),
            "vllm": importlib.metadata.version("vllm"),
            "uv": _command("uv", "--version"),
        },
        "schema_version": 1,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"results={RESULTS}")
    print(f"manifest={MANIFEST}")
    print(f"fixture_checks={fixture_checks['status']}")
    print("adapter_differential=pass")
    print("decision=go")


if __name__ == "__main__":
    main()
