#!/usr/bin/env python3
"""Finalize the profiled E004 production-fused real-activation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from nvfp4_doctor.backends import (
    assess_range_fallback,
    expected_sm120_cutlass_present,
    extract_kernel_evidence,
    kernels_in_nvtx_range,
)
from nvfp4_doctor.capture.e004_fused import E004_FUSED_REAL_ACTIVATION_CASES
from nvfp4_doctor.env import collect_git, collect_gpu, collect_software
from nvfp4_doctor.formats import swizzle_scales_128x4
from scripts.finalize_e004_real_activation import (
    EXPECTED_INPUT_IDENTITY_SHA256,
    EXPECTED_KERNEL,
    PRESERVED_TRANSFER_FIELDS,
    REVISION,
    ROOT,
    SHA256,
    RealActivationFinalizationError,
    _artifact,
    _command,
    _json,
    _mapping,
    _sha256_path,
    _validate_tensor_artifact,
)
from scripts.run_e004_real_activation_capture import _tensor_metadata

EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
DEFAULT_RUN = (
    ROOT
    / "artifacts"
    / "E004-qwen3-layer-capture"
    / "real-activation-fused-matrix"
    / "fused-matrix.json"
)
DEFAULT_REPORT = (
    ROOT / ".local" / "profiles" / "e004-real-activation-fused-matrix.nsys-rep"
)
DEFAULT_RESULTS = EXPERIMENT / "real-activation-fused-matrix.json"
DEFAULT_MANIFEST = EXPERIMENT / "manifest-real-activation-fused-matrix.json"
FULL_MODEL_RESULT = EXPERIMENT / "full-model-acquisition.json"
FULL_MODEL_MANIFEST = EXPERIMENT / "manifest-full-model-acquisition.json"
REPLAY_MATRIX_RESULT = EXPERIMENT / "replay-matrix.json"
REPLAY_MATRIX_MANIFEST = EXPERIMENT / "manifest-replay-matrix.json"
UNFUSED_MATRIX_RESULT = EXPERIMENT / "real-activation-unfused-matrix.json"
UNFUSED_MATRIX_MANIFEST = EXPERIMENT / "manifest-real-activation-unfused-matrix.json"

SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "backends" / "__init__.py",
    ROOT / "src" / "nvfp4_doctor" / "backends" / "nsys.py",
    ROOT / "src" / "nvfp4_doctor" / "capture" / "__init__.py",
    ROOT / "src" / "nvfp4_doctor" / "capture" / "e004.py",
    ROOT / "src" / "nvfp4_doctor" / "capture" / "e004_fused.py",
    ROOT / "src" / "nvfp4_doctor" / "env" / "__init__.py",
    ROOT / "src" / "nvfp4_doctor" / "env" / "assembly.py",
    ROOT / "src" / "nvfp4_doctor" / "env" / "collectors.py",
    ROOT / "src" / "nvfp4_doctor" / "env" / "e001.py",
    ROOT / "src" / "nvfp4_doctor" / "env" / "manifest.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "__init__.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "e2m1.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "layout.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "packing.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "scales.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "ue4m3.py",
    ROOT / "scripts" / "run_e004_real_activation_capture.py",
    ROOT / "scripts" / "run_e004_real_activation_matrix.py",
    ROOT / "scripts" / "finalize_e004_real_activation.py",
    ROOT / "scripts" / "run_e004_real_activation_fused_matrix.py",
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_e004_real_activation_fused_matrix_profile.sh",
    ROOT / "tests" / "unit" / "test_e004_real_activation_fused_matrix_cases.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_fused_matrix_capture.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_fused_matrix_finalization.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_fused_matrix_evidence.py",
    ROOT / "tests" / "unit" / "test_nsys_evidence.py",
)
DEPENDENCY_PATHS = (
    ("full-model-acquisition-result", FULL_MODEL_RESULT),
    ("full-model-acquisition-manifest", FULL_MODEL_MANIFEST),
    ("representative-replay-matrix-result", REPLAY_MATRIX_RESULT),
    ("representative-replay-matrix-manifest", REPLAY_MATRIX_MANIFEST),
    ("representative-unfused-real-activation-result", UNFUSED_MATRIX_RESULT),
    ("representative-unfused-real-activation-manifest", UNFUSED_MATRIX_MANIFEST),
)
DEPENDENCY_PAIRS = (
    (FULL_MODEL_RESULT, FULL_MODEL_MANIFEST, "full_model_snapshot_acquisition_v1"),
    (
        REPLAY_MATRIX_RESULT,
        REPLAY_MATRIX_MANIFEST,
        "representative_projection_replay_matrix_v1",
    ),
    (
        UNFUSED_MATRIX_RESULT,
        UNFUSED_MATRIX_MANIFEST,
        "representative_unfused_real_activation_replay_matrix_v1",
    ),
)

RealActivationFusedMatrixFinalizationError = RealActivationFinalizationError

TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "slice",
    "captured_at_utc",
    "status",
    "decision",
    "repository",
    "model_load",
    "input_identity",
    "matrix",
    "identity_dependencies",
    "backend",
    "cases",
    "gpu",
    "command",
    "claim_boundary",
}
TENSOR_METADATA_KEYS = {
    "shape",
    "dtype",
    "stride",
    "storage_offset",
    "device",
    "contiguous",
    "numel",
    "byte_length",
    "sha256",
    "sha256_encoding",
}
TENSOR_ARTIFACT_KEYS = {
    "path",
    "ignored",
    "encoding",
    "file_bytes",
    "file_sha256",
    "tensor",
    "source_metadata",
    "preserved_fields",
    "device_transfer",
}
SOURCE_TENSOR_KEYS = {
    "tensor_name",
    "shard_path",
    "shard_sha256",
    "data_offsets",
    "dtype",
    "shape",
    "byte_length",
    "sha256",
}
COMPONENT_SOURCE_KEYS = {
    "projection",
    "row_range",
    "source_tensors",
    "prepared_packed_weight_sha256",
    "prepared_runtime_weight_scale_sha256",
    "input_scale",
    "weight_scale_2",
    "replay_matrix_regression_match",
}
SOURCE_CONSTRUCTION_KEYS = {
    "component_order",
    "component_row_boundaries",
    "components",
    "fused_packed_weight_sha256",
    "fused_runtime_weight_scale_sha256",
    "expected_runtime_scalars",
    "source_snapshot_verified",
}
RUNTIME_SCALAR_KEYS = {
    "reduction_rule",
    "input_global_scale",
    "weight_global_scale",
    "alpha",
    "input_global_scale_inv",
}
RUNTIME_PROJECTION_KEYS = {
    "module_path",
    "module_class",
    "quant_method_class",
    "selected_kernel",
    "tp_size",
    "tp_rank",
    "gather_output",
    "logical_widths",
    "packed_weight",
    "runtime_weight_scale",
    "weights_padding_cols",
    "input_global_scale",
    "weight_global_scale",
    "alpha",
    "input_global_scale_inv",
    "component_bindings",
    "global_scale_reduction_matches",
}
COMPONENT_BINDING_KEYS = {
    "projection",
    "row_range",
    "packed_weight_slice_sha256",
    "runtime_weight_scale_slice_sha256",
    "checkpoint_weight_match",
    "independent_scale_swizzle_match",
}
COMPONENT_OUTPUT_SLICE_KEYS = {
    "projection",
    "feature_range",
    "captured_sha256",
    "replay_sha256s",
    "logical_matches",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-evidence", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RealActivationFusedMatrixFinalizationError(
            f"{label} fields changed; missing={missing}, extra={extra}"
        )


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        if not path.is_file():
            raise RealActivationFusedMatrixFinalizationError(
                f"source-bundle path is missing: {path}"
            )
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_items(items: list[str]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items):
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _full_snapshot_inventory() -> tuple[Path, dict[str, dict[str, Any]]]:
    result = _json(FULL_MODEL_RESULT)
    snapshot = _mapping(result.get("snapshot"), "full-model snapshot")
    local_root = snapshot.get("local_root")
    files = snapshot.get("files")
    expected_local_root = (
        Path("models") / "nvidia--Qwen3-8B-NVFP4" / REVISION
    ).as_posix()
    if local_root != expected_local_root or not isinstance(files, list):
        raise RealActivationFusedMatrixFinalizationError(
            "full-model snapshot inventory changed"
        )
    linked_model_dir = ROOT / expected_local_root
    if not linked_model_dir.is_dir():
        raise RealActivationFusedMatrixFinalizationError(
            "full-model snapshot directory is missing"
        )
    # The project-local models path may be a deliberate link to the canonical
    # WSL snapshot cache. Each resolved shard is still constrained to this
    # resolved directory and independently checked against its pinned hash.
    model_dir = linked_model_dir.resolve()
    inventory: dict[str, dict[str, Any]] = {}
    for value in files:
        entry = _mapping(value, "full-model inventory entry")
        path = entry.get("path")
        if not isinstance(path, str) or path in inventory:
            raise RealActivationFusedMatrixFinalizationError(
                "full-model snapshot inventory has invalid paths"
            )
        inventory[path] = entry
    return model_dir, inventory


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors prefix")
            header_length = struct.unpack("<Q", prefix)[0]
            header_bytes = stream.read(header_length)
        header = json.loads(header_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RealActivationFusedMatrixFinalizationError(
            f"could not parse safetensors header: {path}"
        ) from error
    if not isinstance(header, dict):
        raise RealActivationFusedMatrixFinalizationError(
            f"safetensors header is not an object: {path}"
        )
    return 8 + header_length, header


def _verify_snapshot_shard(
    shard: Path,
    *,
    shard_path: str,
    inventory_entry: dict[str, Any],
    shard_hash_cache: dict[str, str],
) -> str:
    """Bind a used local shard to the acquired full-snapshot inventory once."""
    expected_sha256 = inventory_entry.get("sha256")
    expected_size = inventory_entry.get("size_bytes")
    if (
        inventory_entry.get("role") != "weight_shard"
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not SHA256.fullmatch(str(expected_sha256 or ""))
        or inventory_entry.get("lfs_sha256") != expected_sha256
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"full-model shard inventory changed: {shard_path}"
        )
    if shard_path not in shard_hash_cache:
        if not shard.is_file() or shard.stat().st_size != expected_size:
            raise RealActivationFusedMatrixFinalizationError(
                f"local snapshot shard size changed: {shard_path}"
            )
        shard_hash_cache[shard_path] = _sha256_path(shard)
    observed_sha256 = shard_hash_cache[shard_path]
    if observed_sha256 != expected_sha256:
        raise RealActivationFusedMatrixFinalizationError(
            f"local snapshot shard hash changed: {shard_path}"
        )
    return observed_sha256


def _read_snapshot_tensor(
    value: object,
    *,
    expected_name: str,
    expected_dtype: str,
    expected_shape: list[int],
    model_dir: Path,
    inventory: dict[str, dict[str, Any]],
    header_cache: dict[str, tuple[int, dict[str, Any]]],
    shard_hash_cache: dict[str, str],
) -> bytes:
    record = _mapping(value, f"source tensor {expected_name}")
    _require_exact_keys(
        record, SOURCE_TENSOR_KEYS, label=f"source tensor {expected_name}"
    )
    shard_path = record.get("shard_path")
    if not isinstance(shard_path, str) or shard_path not in inventory:
        raise RealActivationFusedMatrixFinalizationError(
            f"source tensor shard changed: {expected_name}"
        )
    shard_inventory = inventory[shard_path]
    if (
        record.get("shard_sha256") != shard_inventory.get("sha256")
        or record.get("tensor_name") != expected_name
        or record.get("dtype") != expected_dtype
        or record.get("shape") != expected_shape
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"source tensor identity changed: {expected_name}"
        )
    shard = (model_dir / shard_path).resolve()
    try:
        shard.relative_to(model_dir)
    except ValueError as error:
        raise RealActivationFusedMatrixFinalizationError(
            f"source tensor shard escapes the snapshot: {expected_name}"
        ) from error
    _verify_snapshot_shard(
        shard,
        shard_path=shard_path,
        inventory_entry=shard_inventory,
        shard_hash_cache=shard_hash_cache,
    )
    if shard_path not in header_cache:
        header_cache[shard_path] = _safetensors_header(shard)
    payload_start, header = header_cache[shard_path]
    header_entry = _mapping(header.get(expected_name), f"header tensor {expected_name}")
    if set(header_entry) != {"dtype", "shape", "data_offsets"}:
        raise RealActivationFusedMatrixFinalizationError(
            f"header tensor fields changed: {expected_name}"
        )
    offsets = header_entry.get("data_offsets")
    if (
        header_entry.get("dtype") != expected_dtype
        or header_entry.get("shape") != expected_shape
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
        or offsets[0] < 0
        or offsets[1] <= offsets[0]
        or record.get("data_offsets") != offsets
        or record.get("byte_length") != offsets[1] - offsets[0]
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"header binding changed: {expected_name}"
        )
    try:
        with shard.open("rb") as stream:
            stream.seek(payload_start + offsets[0])
            payload = stream.read(offsets[1] - offsets[0])
    except OSError as error:
        raise RealActivationFusedMatrixFinalizationError(
            f"could not read source tensor: {expected_name}"
        ) from error
    if len(payload) != record.get("byte_length") or _sha256_bytes(
        payload
    ) != record.get("sha256"):
        raise RealActivationFusedMatrixFinalizationError(
            f"source tensor bytes changed: {expected_name}"
        )
    return payload


def _replay_matrix_cases() -> dict[tuple[int, str], dict[str, Any]]:
    result = _json(REPLAY_MATRIX_RESULT)
    values = result.get("cases")
    if not isinstance(values, list):
        raise RealActivationFusedMatrixFinalizationError(
            "representative replay matrix has no cases"
        )
    cases: dict[tuple[int, str], dict[str, Any]] = {}
    for value in values:
        entry = _mapping(value, "representative replay case")
        layer = entry.get("layer")
        projection = entry.get("projection")
        if not isinstance(layer, int) or not isinstance(projection, str):
            raise RealActivationFusedMatrixFinalizationError(
                "representative replay case identity changed"
            )
        key = (layer, projection)
        if key in cases:
            raise RealActivationFusedMatrixFinalizationError(
                "representative replay case identity changed"
            )
        cases[key] = entry
    return cases


def _validate_source_construction(
    value: object,
    *,
    case_index: int,
    model_dir: Path,
    inventory: dict[str, dict[str, Any]],
    header_cache: dict[str, tuple[int, dict[str, Any]]],
    shard_hash_cache: dict[str, str],
    replay_cases: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = E004_FUSED_REAL_ACTIVATION_CASES[case_index]
    source = _mapping(value, f"{case.case_id} source construction")
    _require_exact_keys(
        source, SOURCE_CONSTRUCTION_KEYS, label=f"{case.case_id} source construction"
    )
    components = source.get("components")
    if not isinstance(components, list) or len(components) != len(
        case.component_boundaries
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{case.case_id} source components changed"
        )
    packed_parts: list[bytes] = []
    linear_scale_parts: list[bytes] = []
    input_scales: list[float] = []
    weight_scales_2: list[float] = []
    for component_value, boundary in zip(
        components, case.component_boundaries, strict=True
    ):
        component = _mapping(
            component_value, f"{case.case_id} {boundary.projection} source component"
        )
        _require_exact_keys(
            component,
            COMPONENT_SOURCE_KEYS,
            label=f"{case.case_id} {boundary.projection} source component",
        )
        tensors = _mapping(
            component.get("source_tensors"),
            f"{case.case_id} {boundary.projection} source tensors",
        )
        _require_exact_keys(
            tensors,
            {"input_scale", "weight", "weight_scale", "weight_scale_2"},
            label=f"{case.case_id} {boundary.projection} source tensors",
        )
        expected_prefix = f"{case.checkpoint_parent_path}.{boundary.projection}"
        input_bytes = _read_snapshot_tensor(
            tensors["input_scale"],
            expected_name=f"{expected_prefix}.input_scale",
            expected_dtype="F32",
            expected_shape=[],
            model_dir=model_dir,
            inventory=inventory,
            header_cache=header_cache,
            shard_hash_cache=shard_hash_cache,
        )
        weight_bytes = _read_snapshot_tensor(
            tensors["weight"],
            expected_name=f"{expected_prefix}.weight",
            expected_dtype="U8",
            expected_shape=list(boundary.packed_weight_shape),
            model_dir=model_dir,
            inventory=inventory,
            header_cache=header_cache,
            shard_hash_cache=shard_hash_cache,
        )
        scale_bytes = _read_snapshot_tensor(
            tensors["weight_scale"],
            expected_name=f"{expected_prefix}.weight_scale",
            expected_dtype="F8_E4M3",
            expected_shape=list(boundary.weight_scale_shape),
            model_dir=model_dir,
            inventory=inventory,
            header_cache=header_cache,
            shard_hash_cache=shard_hash_cache,
        )
        scale_2_bytes = _read_snapshot_tensor(
            tensors["weight_scale_2"],
            expected_name=f"{expected_prefix}.weight_scale_2",
            expected_dtype="F32",
            expected_shape=[],
            model_dir=model_dir,
            inventory=inventory,
            header_cache=header_cache,
            shard_hash_cache=shard_hash_cache,
        )
        input_scale = struct.unpack("<f", input_bytes)[0]
        weight_scale_2 = struct.unpack("<f", scale_2_bytes)[0]
        prepared_scale = swizzle_scales_128x4(
            scale_bytes,
            boundary.output_width,
            boundary.input_width // 16,
        )
        overlap = replay_cases.get((case.layer, boundary.projection))
        expected_regression: bool | None = None
        if overlap is not None:
            source_hashes = _mapping(
                overlap.get("source_tensor_sha256"),
                f"{case.case_id} replay overlap source hashes",
            )
            expected_regression = bool(
                source_hashes
                == {
                    suffix: _mapping(
                        tensors[suffix], f"{boundary.projection} {suffix}"
                    )["sha256"]
                    for suffix in (
                        "input_scale",
                        "weight",
                        "weight_scale",
                        "weight_scale_2",
                    )
                }
                and overlap.get("runtime_weight_scale_sha256")
                == _sha256_bytes(prepared_scale)
                and overlap.get("input_scale") == input_scale
                and overlap.get("weight_scale_2") == weight_scale_2
            )
        if (
            component.get("projection") != boundary.projection
            or component.get("row_range") != [boundary.row_start, boundary.row_end]
            or component.get("prepared_packed_weight_sha256")
            != _sha256_bytes(weight_bytes)
            or component.get("prepared_runtime_weight_scale_sha256")
            != _sha256_bytes(prepared_scale)
            or component.get("input_scale") != input_scale
            or component.get("weight_scale_2") != weight_scale_2
            or component.get("replay_matrix_regression_match")
            is not expected_regression
            or (overlap is not None and expected_regression is not True)
        ):
            raise RealActivationFusedMatrixFinalizationError(
                f"{case.case_id} {boundary.projection} source construction changed"
            )
        packed_parts.append(weight_bytes)
        linear_scale_parts.append(scale_bytes)
        input_scales.append(input_scale)
        weight_scales_2.append(weight_scale_2)

    fused_weight = b"".join(packed_parts)
    fused_linear_scale = b"".join(linear_scale_parts)
    fused_runtime_scale = swizzle_scales_128x4(
        fused_linear_scale,
        case.output_width,
        case.input_width // 16,
    )
    if len(set(input_scales)) != 1 or len(set(weight_scales_2)) != 1:
        raise RealActivationFusedMatrixFinalizationError(
            f"{case.case_id} component scalar identities disagree"
        )
    input_global_scale = _f32(max(input_scales))
    weight_global_scale = _f32(max(weight_scales_2))
    expected_scalars = {
        "reduction_rule": "float32_max_over_ordered_components",
        "input_global_scale": input_global_scale,
        "weight_global_scale": weight_global_scale,
        "alpha": _f32(input_global_scale * weight_global_scale),
        "input_global_scale_inv": _f32(1.0 / input_global_scale),
    }
    if (
        source.get("component_order") != list(case.component_projections)
        or source.get("component_row_boundaries") != list(case.component_row_boundaries)
        or source.get("fused_packed_weight_sha256") != _sha256_bytes(fused_weight)
        or source.get("fused_runtime_weight_scale_sha256")
        != _sha256_bytes(fused_runtime_scale)
        or source.get("expected_runtime_scalars") != expected_scalars
        or source.get("source_snapshot_verified") is not True
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{case.case_id} fused source construction changed"
        )
    return source, expected_scalars


def _runtime_tensor_metadata(
    *, shape: tuple[int, int], dtype: str, sha256: str
) -> dict[str, object]:
    numel = shape[0] * shape[1]
    return {
        "shape": list(shape),
        "dtype": dtype,
        "stride": [shape[1], 1],
        "storage_offset": 0,
        "device": "cuda:0",
        "contiguous": True,
        "numel": numel,
        "byte_length": numel,
        "sha256": sha256,
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }


def _validate_case(
    value: object,
    *,
    case_index: int,
    model_dir: Path,
    inventory: dict[str, dict[str, Any]],
    header_cache: dict[str, tuple[int, dict[str, Any]]],
    shard_hash_cache: dict[str, str],
    replay_cases: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    spec = E004_FUSED_REAL_ACTIVATION_CASES[case_index]
    result = _mapping(value, f"fused matrix case {case_index}")
    _require_exact_keys(
        result,
        {
            "case_id",
            "layer",
            "role",
            "projection",
            "adapter_scope",
            "module_path",
            "module_class",
            "tensor_role",
            "phase",
            "event_count",
            "activation_provenance",
            "source_construction",
            "capture",
            "runtime_projection",
            "replay",
            "backend_range",
        },
        label=f"fused matrix case {case_index}",
    )
    observed_header = (
        result.get("case_id"),
        result.get("layer"),
        result.get("role"),
        result.get("projection"),
        result.get("adapter_scope"),
        result.get("module_path"),
        result.get("module_class"),
        result.get("tensor_role"),
        result.get("phase"),
        result.get("event_count"),
        result.get("activation_provenance"),
    )
    expected_header = (
        spec.case_id,
        spec.layer,
        spec.role,
        spec.projection,
        spec.adapter_scope,
        spec.module_path,
        spec.module_class,
        "module_input",
        "prefill",
        1,
        "real_qwen_prefill",
    )
    if observed_header != expected_header:
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} case identity changed"
        )
    source, expected_scalars = _validate_source_construction(
        result.get("source_construction"),
        case_index=case_index,
        model_dir=model_dir,
        inventory=inventory,
        header_cache=header_cache,
        shard_hash_cache=shard_hash_cache,
        replay_cases=replay_cases,
    )

    capture = _mapping(result.get("capture"), f"{spec.case_id} capture")
    _require_exact_keys(
        capture,
        {
            "input_artifact",
            "captured_module_output_artifact",
            "metadata_preserved_fields",
            "device_transfer_recorded",
        },
        label=f"{spec.case_id} capture",
    )
    if (
        capture.get("metadata_preserved_fields") != PRESERVED_TRANSFER_FIELDS
        or capture.get("device_transfer_recorded") is not True
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} capture transfer contract changed"
        )
    artifact_prefix = (
        "artifacts/E004-qwen3-layer-capture/real-activation-fused-matrix/"
        f"{spec.artifact_slug}"
    )
    input_artifact = _validate_tensor_artifact(
        capture.get("input_artifact"),
        label=f"{spec.case_id} input artifact",
        expected_path=f"{artifact_prefix}-input.pt",
    )
    output_artifact = _validate_tensor_artifact(
        capture.get("captured_module_output_artifact"),
        label=f"{spec.case_id} captured module-output artifact",
        expected_path=f"{artifact_prefix}-captured-module-output.pt",
    )
    _validate_tensor_artifact_schema(
        input_artifact, label=f"{spec.case_id} input artifact"
    )
    _validate_tensor_artifact_schema(
        output_artifact, label=f"{spec.case_id} captured module-output artifact"
    )
    input_tensor = _mapping(
        input_artifact.get("tensor"), f"{spec.case_id} input tensor"
    )
    output_tensor = _mapping(
        output_artifact.get("tensor"), f"{spec.case_id} captured output tensor"
    )
    if (
        input_tensor.get("shape") != list(spec.input_shape(9))
        or input_tensor.get("dtype") != "bfloat16"
        or input_tensor.get("stride") != [spec.input_width, 1]
        or input_tensor.get("storage_offset") != 0
        or output_tensor.get("shape") != list(spec.output_shape(9))
        or output_tensor.get("dtype") != "bfloat16"
        or output_tensor.get("stride") != [spec.output_width, 1]
        or output_tensor.get("storage_offset") != 0
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} captured tensor metadata changed"
        )

    runtime = _mapping(
        result.get("runtime_projection"), f"{spec.case_id} runtime projection"
    )
    _require_exact_keys(
        runtime, RUNTIME_PROJECTION_KEYS, label=f"{spec.case_id} runtime projection"
    )
    expected_weight_sha = str(source["fused_packed_weight_sha256"])
    expected_scale_sha = str(source["fused_runtime_weight_scale_sha256"])
    expected_bindings = [
        {
            "projection": boundary.projection,
            "row_range": [boundary.row_start, boundary.row_end],
            "packed_weight_slice_sha256": component["prepared_packed_weight_sha256"],
            "runtime_weight_scale_slice_sha256": component[
                "prepared_runtime_weight_scale_sha256"
            ],
            "checkpoint_weight_match": True,
            "independent_scale_swizzle_match": True,
        }
        for boundary, component in zip(
            spec.component_boundaries, source["components"], strict=True
        )
    ]
    bindings = runtime.get("component_bindings")
    if not isinstance(bindings, list):
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} runtime component bindings are missing"
        )
    for index, binding in enumerate(bindings):
        _require_exact_keys(
            _mapping(binding, f"{spec.case_id} runtime binding {index}"),
            COMPONENT_BINDING_KEYS,
            label=f"{spec.case_id} runtime binding {index}",
        )
    if (
        runtime.get("module_path") != spec.module_path
        or runtime.get("module_class") != spec.module_class
        or runtime.get("quant_method_class") != "ModelOptNvFp4LinearMethod"
        or runtime.get("selected_kernel") != EXPECTED_KERNEL
        or runtime.get("tp_size") != 1
        or runtime.get("tp_rank") != 0
        or runtime.get("gather_output") is not False
        or runtime.get("logical_widths") != list(spec.component_output_widths)
        or runtime.get("packed_weight")
        != _runtime_tensor_metadata(
            shape=spec.packed_weight_shape,
            dtype="uint8",
            sha256=expected_weight_sha,
        )
        or runtime.get("runtime_weight_scale")
        != _runtime_tensor_metadata(
            shape=spec.weight_scale_shape,
            dtype="float8_e4m3fn",
            sha256=expected_scale_sha,
        )
        or runtime.get("weights_padding_cols") != 0
        or runtime.get("input_global_scale") != expected_scalars["input_global_scale"]
        or runtime.get("weight_global_scale") != expected_scalars["weight_global_scale"]
        or runtime.get("alpha") != expected_scalars["alpha"]
        or runtime.get("input_global_scale_inv")
        != expected_scalars["input_global_scale_inv"]
        or bindings != expected_bindings
        or runtime.get("global_scale_reduction_matches") is not True
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} runtime fused projection changed"
        )

    replay = _mapping(result.get("replay"), f"{spec.case_id} replay")
    _require_exact_keys(
        replay,
        {
            "warmup_runs",
            "repetitions",
            "synchronized",
            "all_finite",
            "output_shape",
            "output_dtype",
            "output_sha256s",
            "output_hash_stable",
            "captured_module_output_sha256",
            "logical_byte_exact_captured_module_output_matches",
            "reconstructed_activation_metadata",
            "logical_byte_exact_captured_input_match",
            "max_abs_error",
            "mean_abs_error",
            "input_sha256",
            "replay_output_artifact",
            "component_output_slices",
        },
        label=f"{spec.case_id} replay",
    )
    replay_artifact = _validate_tensor_artifact(
        replay.get("replay_output_artifact"),
        label=f"{spec.case_id} replay output artifact",
        expected_path=f"{artifact_prefix}-replay-output.pt",
    )
    _validate_tensor_artifact_schema(
        replay_artifact, label=f"{spec.case_id} replay output artifact"
    )
    replay_tensor = _mapping(
        replay_artifact.get("tensor"), f"{spec.case_id} replay artifact tensor"
    )
    output_hashes = replay.get("output_sha256s")
    logical_matches = replay.get("logical_byte_exact_captured_module_output_matches")
    reconstructed = _mapping(
        replay.get("reconstructed_activation_metadata"),
        f"{spec.case_id} reconstructed activation",
    )
    component_slices = replay.get("component_output_slices")
    if (
        not isinstance(output_hashes, list)
        or not isinstance(logical_matches, list)
        or not isinstance(component_slices, list)
        or len(component_slices) != len(spec.component_boundaries)
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} replay hashes are missing"
        )
    for index, (slice_value, boundary) in enumerate(
        zip(component_slices, spec.component_boundaries, strict=True)
    ):
        component_slice = _mapping(
            slice_value, f"{spec.case_id} component output slice {index}"
        )
        _require_exact_keys(
            component_slice,
            COMPONENT_OUTPUT_SLICE_KEYS,
            label=f"{spec.case_id} component output slice {index}",
        )
        slice_hashes = component_slice.get("replay_sha256s")
        slice_matches = component_slice.get("logical_matches")
        if (
            component_slice.get("projection") != boundary.projection
            or component_slice.get("feature_range")
            != [boundary.row_start, boundary.row_end]
            or not SHA256.fullmatch(str(component_slice.get("captured_sha256", "")))
            or not isinstance(slice_hashes, list)
            or len(slice_hashes) != 3
            or len(set(slice_hashes)) != 1
            or slice_hashes[0] != component_slice.get("captured_sha256")
            or slice_matches != [True, True, True]
        ):
            raise RealActivationFusedMatrixFinalizationError(
                f"{spec.case_id} component output replay changed"
            )
    input_source = _mapping(
        input_artifact.get("source_metadata"), f"{spec.case_id} input source"
    )
    if not all(
        (
            replay.get("warmup_runs") == 1,
            replay.get("repetitions") == 3,
            replay.get("synchronized") is True,
            replay.get("all_finite") is True,
            replay.get("output_shape") == list(spec.output_shape(9)),
            replay.get("output_dtype") == "bfloat16",
            len(output_hashes) == 3,
            len(set(output_hashes)) == 1,
            replay.get("output_hash_stable") is True,
            logical_matches == [True, True, True],
            replay.get("max_abs_error") == 0.0,
            replay.get("mean_abs_error") == 0.0,
            replay.get("logical_byte_exact_captured_input_match") is True,
            reconstructed == input_source,
            replay.get("input_sha256") == input_tensor.get("sha256"),
            replay.get("input_sha256") == reconstructed.get("sha256"),
            replay.get("captured_module_output_sha256") == output_tensor.get("sha256"),
            output_hashes[0] == output_tensor.get("sha256"),
            replay_tensor == output_tensor,
        )
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} replay invariants changed"
        )

    pending_range = _mapping(
        result.get("backend_range"), f"{spec.case_id} backend range"
    )
    if pending_range != {
        "target_nvtx_range": spec.target_nvtx_range,
        "target_kernel_ids": [],
        "target_kernel_set_sha256": None,
        "expected_sm120_cutlass_signature_present": False,
        "activation_quantization_signature_present": False,
        "fallback_status": "pending_profiler",
    }:
        raise RealActivationFusedMatrixFinalizationError(
            f"{spec.case_id} pending backend range changed"
        )
    return result, (input_artifact, output_artifact, replay_artifact)


def _validate_run(
    run: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    _validate_model_input_and_command(run)
    _validate_matrix_header(run)
    values = run.get("cases")
    if not isinstance(values, list) or len(values) != 6:
        raise RealActivationFusedMatrixFinalizationError(
            "fused matrix must contain exactly six cases"
        )
    model_dir, inventory = _full_snapshot_inventory()
    header_cache: dict[str, tuple[int, dict[str, Any]]] = {}
    shard_hash_cache: dict[str, str] = {}
    replay_cases = _replay_matrix_cases()
    cases: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        case, case_artifacts = _validate_case(
            value,
            case_index=index,
            model_dir=model_dir,
            inventory=inventory,
            header_cache=header_cache,
            shard_hash_cache=shard_hash_cache,
            replay_cases=replay_cases,
        )
        cases.append(case)
        artifacts.extend(case_artifacts)
    paths = [str(artifact["path"]) for artifact in artifacts]
    if len(artifacts) != 18 or len(set(paths)) != 18:
        raise RealActivationFusedMatrixFinalizationError(
            "fused tensor artifact paths must be unique"
        )
    input_hashes = [
        str(
            _mapping(
                _mapping(case["capture"], "capture")["input_artifact"],
                "input artifact",
            )["tensor"]["sha256"]
        )
        for case in cases
    ]
    output_hashes = [
        str(_mapping(case["replay"], "replay")["captured_module_output_sha256"])
        for case in cases
    ]
    if len(set(input_hashes)) != 6 or len(set(output_hashes)) != 6:
        raise RealActivationFusedMatrixFinalizationError(
            "fused cases do not have six distinct input and output identities"
        )
    return cases, tuple(artifacts)


def _load_local_tensor_artifact(
    artifact: dict[str, Any], *, root: Path
) -> torch.Tensor:
    path = root / str(artifact["path"])
    if (
        not path.is_file()
        or path.stat().st_size != artifact["file_bytes"]
        or _sha256_path(path) != artifact["file_sha256"]
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"local artifact identity changed: {artifact['path']}"
        )
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise RealActivationFusedMatrixFinalizationError(
            f"could not load local tensor artifact: {artifact['path']}"
        ) from error
    if not isinstance(loaded, dict) or set(loaded) != {
        "tensor",
        "source_metadata",
        "destination_metadata",
    }:
        raise RealActivationFusedMatrixFinalizationError(
            f"local tensor artifact schema changed: {artifact['path']}"
        )
    tensor = loaded.get("tensor")
    if not isinstance(tensor, torch.Tensor):
        raise RealActivationFusedMatrixFinalizationError(
            f"local tensor artifact has no tensor: {artifact['path']}"
        )
    if (
        loaded.get("source_metadata") != artifact["source_metadata"]
        or loaded.get("destination_metadata") != artifact["tensor"]
        or _tensor_metadata(tensor) != artifact["tensor"]
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"local tensor bytes or metadata changed: {artifact['path']}"
        )
    return tensor


def _validate_local_tensor_artifacts(
    cases: list[dict[str, Any]],
    artifacts: tuple[dict[str, Any], ...],
    *,
    root: Path = ROOT,
) -> None:
    loaded = [
        _load_local_tensor_artifact(artifact, root=root) for artifact in artifacts
    ]
    for case_index, (case, spec) in enumerate(
        zip(cases, E004_FUSED_REAL_ACTIVATION_CASES, strict=True)
    ):
        captured_output = loaded[case_index * 3 + 1]
        replay_output = loaded[case_index * 3 + 2]
        slices = _mapping(case.get("replay"), f"{spec.case_id} replay").get(
            "component_output_slices"
        )
        if not isinstance(slices, list):
            raise RealActivationFusedMatrixFinalizationError(
                f"{spec.case_id} component slices are missing"
            )
        for slice_value, boundary in zip(
            slices, spec.component_boundaries, strict=True
        ):
            evidence = _mapping(
                slice_value, f"{spec.case_id} {boundary.projection} output slice"
            )
            captured_slice = captured_output[:, boundary.row_start : boundary.row_end]
            replay_slice = replay_output[:, boundary.row_start : boundary.row_end]
            captured_sha = str(_tensor_metadata(captured_slice)["sha256"])
            replay_sha = str(_tensor_metadata(replay_slice)["sha256"])
            if (
                captured_sha != evidence.get("captured_sha256")
                or replay_sha != captured_sha
                or evidence.get("replay_sha256s") != [replay_sha] * 3
                or not torch.equal(captured_slice, replay_slice)
            ):
                raise RealActivationFusedMatrixFinalizationError(
                    f"{spec.case_id} {boundary.projection} local output slice changed"
                )


def _temporary_sibling(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return Path(temporary_name)


def _restore_published_path(target: Path, previous: bytes | None) -> None:
    if previous is None:
        target.unlink(missing_ok=True)
        return
    rollback = _temporary_sibling(target, previous)
    try:
        os.replace(rollback, target)
    finally:
        rollback.unlink(missing_ok=True)


def _publish_result_and_manifest(
    *,
    results_path: Path,
    results_text: str,
    manifest_path: Path,
    manifest_text: str,
) -> None:
    """Publish the pair late and restore both prior files on replace failure."""
    previous_results = results_path.read_bytes() if results_path.is_file() else None
    previous_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
    results_temporary = _temporary_sibling(results_path, results_text.encode("utf-8"))
    manifest_temporary = _temporary_sibling(
        manifest_path, manifest_text.encode("utf-8")
    )
    results_published = False
    manifest_published = False
    try:
        os.replace(results_temporary, results_path)
        results_published = True
        os.replace(manifest_temporary, manifest_path)
        manifest_published = True
    except BaseException:
        if results_published:
            _restore_published_path(results_path, previous_results)
        if manifest_published:
            _restore_published_path(manifest_path, previous_manifest)
        raise
    finally:
        results_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    run = _json(args.run_evidence)
    cases, tensor_artifacts = _validate_run(run)
    _validate_local_tensor_artifacts(cases, tensor_artifacts)
    if not args.report.is_file():
        raise RealActivationFusedMatrixFinalizationError("Nsight report is missing")
    dependencies = _dependency_artifacts()

    git = collect_git()
    if git.dirty:
        raise RealActivationFusedMatrixFinalizationError(
            "finalization requires a clean implementation commit"
        )
    stats_argv = (
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        "cuda_gpu_kern_sum:nvtx-name",
        "--format",
        "csv",
        str(args.report),
    )
    stats_csv = _command(*stats_argv)
    evidence = extract_kernel_evidence(args.report, stats_csv)
    backend, backend_ranges, passed = _profile_backend(
        evidence.observed_kernels, evidence.report_sha256
    )
    finalized_cases = [
        {
            **case,
            "backend_range": backend_ranges[str(case["case_id"])],
        }
        for case in cases
    ]
    observed_range_count = sum(
        bool(value["target_kernel_ids"]) for value in backend_ranges.values()
    )
    if passed:
        claim_boundary = (
            "For one fixed nine-token request in the pinned environment, this "
            "result establishes metadata-preserving capture and deterministic "
            "same-module replay at six production-fused qkv_proj and gate_up_proj "
            "boundaries spanning layers 0, 18, and 35, with exact checkpoint-"
            "component-to-runtime fusion bindings and range-scoped SM120 CUTLASS "
            "NVFP4 identity. Combined with the required passing representative "
            "unfused dependency, this satisfies the bounded Gate 2 capture, replay, "
            "and backend-identity criterion. It does not establish NVFP4 numerical "
            "correctness, equivalence to separately executed q_proj, k_proj, v_proj, "
            "gate_proj, or up_proj modules, prompt diversity, final-logit or model "
            "quality, performance, cross-backend agreement, equivalence to a high-"
            "precision checkpoint, or generalization beyond the pinned cases."
        )
    else:
        claim_boundary = (
            "The runtime produced six metadata-preserving production-fused captures "
            "and deterministic same-module replays for the fixed request, but the "
            "six-range backend criterion was not met. This result is inconclusive, "
            "does not complete Gate 2, and establishes no NVFP4 numerical correctness, "
            "model quality, performance, or cross-backend claim."
        )
    results = {
        **run,
        "slice": "representative_fused_real_activation_replay_matrix_v1",
        "status": "pass" if passed else "inconclusive",
        "decision": "go" if passed else "repeat",
        "matrix": {
            **_mapping(run["matrix"], "matrix"),
            "request_identity_regression_match": True,
            "source_overlap_regression_count": 9,
            "source_overlap_regression_match": True,
            "evaluated_range_count": len(backend_ranges),
            "observed_target_range_count": observed_range_count,
        },
        "backend": backend,
        "cases": finalized_cases,
        "claim_boundary": claim_boundary,
    }

    gpu = asdict(collect_gpu())
    software = asdict(collect_software())
    software["nsight_systems"] = _command("nsys", "--version", timeout=10)
    branch = _command("git", "branch", "--show-current", timeout=10)
    profile_argv = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--flush-on-cudaprofilerstop=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--wait=primary",
        "--output=.local/profiles/e004-real-activation-fused-matrix",
        "--force-overwrite=true",
        "/home/meyowu/projects/nvfp4-doctor/.venv/bin/python",
        "-m",
        "scripts.run_e004_real_activation_fused_matrix",
        "--model-dir",
        f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}",
        "--artifact-root",
        "artifacts/E004-qwen3-layer-capture/real-activation-fused-matrix",
        "--output",
        (
            "artifacts/E004-qwen3-layer-capture/real-activation-fused-matrix/"
            "fused-matrix.json"
        ),
        "--profile-capture",
    ]
    results_text = json.dumps(results, indent=2) + "\n"
    result_artifact = {
        "kind": "normalized-real-activation-fused-matrix-result",
        "path": args.results.absolute().relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(results_text.encode("utf-8")).hexdigest(),
        "ignored": False,
    }
    local_tensor_entries: list[dict[str, object]] = []
    for spec, artifacts in zip(
        E004_FUSED_REAL_ACTIVATION_CASES,
        (tensor_artifacts[index : index + 3] for index in range(0, 18, 3)),
        strict=True,
    ):
        for suffix, artifact in zip(
            ("captured-activation", "captured-module-output", "replay-output"),
            artifacts,
            strict=True,
        ):
            local_tensor_entries.append(
                _artifact(
                    kind=f"{spec.case_id}-{suffix}",
                    path=ROOT / str(artifact["path"]),
                    ignored=True,
                    sha256=str(artifact["file_sha256"]),
                )
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_fused_real_activation_replay_matrix_v1",
        "captured_at_utc": run["captured_at_utc"],
        "git": {"commit": git.commit, "branch": branch, "dirty": git.dirty},
        "gpu": gpu,
        "software": software,
        "model": {
            "repository": "nvidia/Qwen3-8B-NVFP4",
            "revision": REVISION,
            "local_snapshot_path": run["model_load"]["local_snapshot_path"],
            "complete_snapshot_acquired": True,
            "layers": [0, 18, 35],
            "fused_projections": ["qkv_proj", "gate_up_proj"],
            "component_projections": [
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
            ],
            "adapter_scope": "production_aligned_fused",
            "activation_provenance": "real_qwen_prefill",
        },
        "dependencies": dependencies,
        "commands": {
            "workflow": [
                "bash",
                "scripts/run_e004_real_activation_fused_matrix_profile.sh",
            ],
            "profile": profile_argv,
            "stats": list(stats_argv),
            "runtime": run["command"],
            "finalize": [sys.executable, *sys.argv],
        },
        "backend": backend,
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            result_artifact,
            _artifact(
                kind="raw-real-activation-fused-matrix-observation",
                path=args.run_evidence,
                ignored=True,
            ),
            *local_tensor_entries,
            _artifact(
                kind="nsight-systems-report",
                path=args.report,
                ignored=True,
                sha256=evidence.report_sha256,
            ),
        ],
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"

    _publish_result_and_manifest(
        results_path=args.results,
        results_text=results_text,
        manifest_path=args.manifest,
        manifest_text=manifest_text,
    )
    print(f"case_count={len(finalized_cases)}")
    print(f"evaluated_range_count={len(backend_ranges)}")
    print(f"observed_target_range_count={observed_range_count}")
    print(f"profiler_sha256={evidence.report_sha256}")
    print(f"status={results['status']}")
    print(f"decision={results['decision']}")
    return 0 if passed else 1


def _validate_tensor_artifact_schema(artifact: dict[str, Any], *, label: str) -> None:
    _require_exact_keys(artifact, TENSOR_ARTIFACT_KEYS, label=label)
    source = _mapping(artifact.get("source_metadata"), f"{label} source metadata")
    tensor = _mapping(artifact.get("tensor"), f"{label} tensor metadata")
    _require_exact_keys(source, TENSOR_METADATA_KEYS, label=f"{label} source metadata")
    _require_exact_keys(tensor, TENSOR_METADATA_KEYS, label=f"{label} tensor metadata")
    if (
        not isinstance(artifact.get("file_bytes"), int)
        or artifact["file_bytes"] <= 0
        or not SHA256.fullmatch(str(artifact.get("file_sha256", "")))
    ):
        raise RealActivationFusedMatrixFinalizationError(
            f"{label} file metadata changed"
        )


def _dependency_artifacts() -> list[dict[str, str]]:
    dependencies: list[dict[str, str]] = []
    for kind, path in DEPENDENCY_PATHS:
        if not path.is_file():
            raise RealActivationFusedMatrixFinalizationError(
                f"required dependency is missing: {path}"
            )
        dependencies.append(
            {
                "kind": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(path),
            }
        )
    for result_path, manifest_path, expected_slice in DEPENDENCY_PAIRS:
        result = _json(result_path)
        manifest = _json(manifest_path)
        git = _mapping(manifest.get("git"), "dependency manifest git")
        if (
            result.get("schema_version") != 1
            or result.get("experiment_id") != "E004-qwen3-layer-capture"
            or result.get("slice") != expected_slice
            or result.get("status") != "pass"
            or result.get("decision") not in {"continue", "go"}
            or manifest.get("schema_version") != 1
            or manifest.get("experiment_id") != "E004-qwen3-layer-capture"
            or manifest.get("slice") != expected_slice
            or git.get("dirty") is not False
        ):
            raise RealActivationFusedMatrixFinalizationError(
                f"dependency semantic identity changed: {result_path}"
            )
        artifacts = manifest.get("artifacts")
        expected_path = result_path.relative_to(ROOT).as_posix()
        expected_sha256 = _sha256_path(result_path)
        matches = (
            [
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict)
                and artifact.get("path") == expected_path
                and artifact.get("sha256") == expected_sha256
            ]
            if isinstance(artifacts, list)
            else []
        )
        if len(matches) != 1:
            raise RealActivationFusedMatrixFinalizationError(
                f"dependency manifest does not bind its result: {manifest_path}"
            )
    return dependencies


def _profile_backend(
    observed_kernels: tuple[str, ...], report_sha256: str
) -> tuple[dict[str, object], dict[str, dict[str, object]], bool]:
    target_by_case = {
        case.case_id: list(
            kernels_in_nvtx_range(observed_kernels, case.target_nvtx_range)
        )
        for case in E004_FUSED_REAL_ACTIVATION_CASES
    }
    catalog_names = list(
        dict.fromkeys(
            kernel
            for case in E004_FUSED_REAL_ACTIVATION_CASES
            for kernel in target_by_case[case.case_id]
        )
    )
    catalog = [
        {"kernel_id": hashlib.sha256(name.encode("utf-8")).hexdigest(), "name": name}
        for name in catalog_names
    ]
    kernel_id_by_name = {
        str(entry["name"]): str(entry["kernel_id"]) for entry in catalog
    }
    ranges: dict[str, dict[str, object]] = {}
    passed = True
    for case in E004_FUSED_REAL_ACTIVATION_CASES:
        names = target_by_case[case.case_id]
        signature_present = expected_sm120_cutlass_present(
            observed_kernels, case.target_nvtx_range
        )
        quantization_present = any("vllm::cvt_fp16_to_fp4" in name for name in names)
        fallback_status = assess_range_fallback(
            observed_kernels, case.target_nvtx_range
        ).value
        case_passed = bool(
            names
            and signature_present
            and quantization_present
            and fallback_status == "not_detected"
        )
        passed = passed and case_passed
        ranges[case.case_id] = {
            "target_nvtx_range": case.target_nvtx_range,
            "target_kernel_ids": [kernel_id_by_name[name] for name in names],
            "target_kernel_set_sha256": _sha256_items(names),
            "expected_sm120_cutlass_signature_present": signature_present,
            "activation_quantization_signature_present": quantization_present,
            "fallback_status": fallback_status,
        }
    backend: dict[str, object] = {
        "requested_format": "nvfp4",
        "requested_backend": "auto",
        "expected_selected_vllm_kernel": EXPECTED_KERNEL,
        "reported_backend": None,
        "profiler_sha256": report_sha256,
        "kernel_catalog": catalog,
    }
    return backend, ranges, passed


def _expected_raw_dependencies() -> list[dict[str, str]]:
    return [
        {
            "kind": "full-model-acquisition",
            "path": FULL_MODEL_RESULT.relative_to(ROOT).as_posix(),
            "sha256": _sha256_path(FULL_MODEL_RESULT),
            "slice": "full_model_snapshot_acquisition_v1",
        },
        {
            "kind": "representative-replay-matrix",
            "path": REPLAY_MATRIX_RESULT.relative_to(ROOT).as_posix(),
            "sha256": _sha256_path(REPLAY_MATRIX_RESULT),
            "slice": "representative_projection_replay_matrix_v1",
        },
        {
            "kind": "representative-unfused-real-activation",
            "path": UNFUSED_MATRIX_RESULT.relative_to(ROOT).as_posix(),
            "sha256": _sha256_path(UNFUSED_MATRIX_RESULT),
            "slice": "representative_unfused_real_activation_replay_matrix_v1",
        },
    ]


def _validate_model_input_and_command(run: dict[str, Any]) -> None:
    _require_exact_keys(run, TOP_LEVEL_KEYS, label="runtime observation")
    if (
        run.get("schema_version") != 1
        or run.get("experiment_id") != "E004-qwen3-layer-capture"
        or run.get("slice") != "representative_fused_real_activation_observation_v1"
        or run.get("status") != "pass"
        or run.get("decision") != "pending_profiler"
    ):
        raise RealActivationFusedMatrixFinalizationError(
            "fused runtime observation did not pass"
        )
    repository = _mapping(run.get("repository"), "repository")
    _require_exact_keys(repository, {"id", "revision"}, label="repository")
    if repository != {
        "id": "nvidia/Qwen3-8B-NVFP4",
        "revision": REVISION,
    }:
        raise RealActivationFusedMatrixFinalizationError("repository identity changed")

    model_load = _mapping(run.get("model_load"), "model load")
    _require_exact_keys(
        model_load,
        {
            "local_snapshot_path",
            "frozen_environment",
            "requested_args",
            "observed_model_class",
            "model_load_count",
            "request_count",
            "request_completed",
            "free_memory_before_bytes",
            "free_memory_after_load_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        },
        label="model load",
    )
    expected_requested = {
        "runner": "generate",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "quantization": "modelopt_fp4",
        "load_format": "safetensors",
        "trust_remote_code": False,
        "skip_tokenizer_init": True,
        "max_model_len": 64,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 64,
        "gpu_memory_utilization": 0.80,
        "cpu_offload_gb": 0,
        "kv_cache_dtype": "auto",
        "kv_cache_memory_bytes": 256 * 1024**2,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "compilation_config": 0,
        "linear_backend": "auto",
        "seed": 0,
    }
    expected_environment = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    memory_values = tuple(
        model_load.get(field)
        for field in (
            "free_memory_before_bytes",
            "free_memory_after_load_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
    )
    if (
        model_load.get("local_snapshot_path")
        != f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}"
        or model_load.get("observed_model_class") != "Qwen3ForCausalLM"
        or model_load.get("model_load_count") != 1
        or model_load.get("request_count") != 1
        or model_load.get("request_completed") is not True
        or _mapping(model_load.get("requested_args"), "requested model args")
        != expected_requested
        or _mapping(model_load.get("frozen_environment"), "frozen environment")
        != expected_environment
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in memory_values
        )
    ):
        raise RealActivationFusedMatrixFinalizationError("model-load contract changed")

    input_identity = _mapping(run.get("input_identity"), "input identity")
    unfused_identity = _mapping(
        _json(UNFUSED_MATRIX_RESULT).get("input_identity"),
        "unfused input identity",
    )
    if (
        input_identity != unfused_identity
        or input_identity.get("token_ids_sha256") != EXPECTED_INPUT_IDENTITY_SHA256
        or input_identity.get("token_ids_committed_in_result") is not False
        or "token_ids" in input_identity
        or "prompt_text" in input_identity
    ):
        raise RealActivationFusedMatrixFinalizationError(
            "hashed request regression changed"
        )

    dependencies = run.get("identity_dependencies")
    if dependencies != _expected_raw_dependencies():
        raise RealActivationFusedMatrixFinalizationError(
            "runtime identity dependencies changed"
        )

    gpu = _mapping(run.get("gpu"), "gpu")
    _require_exact_keys(
        gpu, {"name", "compute_capability", "total_memory_bytes"}, label="gpu"
    )
    if (
        gpu.get("name") != "NVIDIA GeForce RTX 5080"
        or gpu.get("compute_capability") != [12, 0]
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or gpu["total_memory_bytes"] <= 0
    ):
        raise RealActivationFusedMatrixFinalizationError("GPU identity changed")

    command = _mapping(run.get("command"), "command")
    _require_exact_keys(command, {"argv", "cwd"}, label="command")
    expected_argv = [
        "/home/meyowu/projects/nvfp4-doctor/.venv/bin/python",
        str(ROOT / "scripts" / "run_e004_real_activation_fused_matrix.py"),
        "--model-dir",
        str(ROOT / "models" / "nvidia--Qwen3-8B-NVFP4" / REVISION),
        "--artifact-root",
        str(
            ROOT
            / "artifacts"
            / "E004-qwen3-layer-capture"
            / "real-activation-fused-matrix"
        ),
        "--output",
        str(DEFAULT_RUN),
        "--profile-capture",
    ]
    if command.get("argv") != expected_argv or command.get("cwd") != str(ROOT):
        raise RealActivationFusedMatrixFinalizationError("runtime command changed")


def _validate_matrix_header(run: dict[str, Any]) -> None:
    matrix = _mapping(run.get("matrix"), "matrix")
    _require_exact_keys(
        matrix,
        {
            "layers",
            "layer_roles",
            "projections",
            "component_projections",
            "case_ids",
            "case_count",
            "repetitions_per_case",
            "hook_count",
            "hook_event_order",
            "distinct_input_sha256_count",
            "distinct_module_output_sha256_count",
        },
        label="matrix",
    )
    expected_ids = [case.case_id for case in E004_FUSED_REAL_ACTIVATION_CASES]
    expected_events = [
        f"{case.case_id}:{role}"
        for case in E004_FUSED_REAL_ACTIVATION_CASES
        for role in ("input", "module_output")
    ]
    if matrix != {
        "layers": [0, 18, 35],
        "layer_roles": ["early", "middle", "late"],
        "projections": ["qkv_proj", "gate_up_proj"],
        "component_projections": [
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
        ],
        "case_ids": expected_ids,
        "case_count": 6,
        "repetitions_per_case": 3,
        "hook_count": 12,
        "hook_event_order": expected_events,
        "distinct_input_sha256_count": 6,
        "distinct_module_output_sha256_count": 6,
    }:
        raise RealActivationFusedMatrixFinalizationError("matrix coverage changed")

    backend = _mapping(run.get("backend"), "backend")
    _require_exact_keys(
        backend,
        {
            "requested_format",
            "requested_backend",
            "expected_selected_vllm_kernel",
            "reported_backend",
            "profiler_sha256",
            "kernel_catalog",
        },
        label="backend",
    )
    if backend != {
        "requested_format": "nvfp4",
        "requested_backend": "auto",
        "expected_selected_vllm_kernel": EXPECTED_KERNEL,
        "reported_backend": None,
        "profiler_sha256": None,
        "kernel_catalog": [],
    }:
        raise RealActivationFusedMatrixFinalizationError(
            "pending backend identity changed"
        )


if __name__ == "__main__":
    raise SystemExit(main())
