#!/usr/bin/env python3
"""Capture and replay the six production-fused Qwen3 NVFP4 modules."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from nvfp4_doctor.capture.e004_fused import (
    E004_FUSED_REAL_ACTIVATION_CASES,
    E004FusedRealActivationCase,
)
from nvfp4_doctor.formats import swizzle_scales_128x4
from scripts.run_e004_real_activation_capture import (
    EXPECTED_KERNEL,
    FROZEN_VLLM_ENVIRONMENT,
    MODEL_RELATIVE,
    PRESERVED_TRANSFER_FIELDS,
    PROMPT_TOKEN_IDS,
    REPO_ID,
    REVISION,
    ROOT,
    _bitwise_tensor_match,
    _canonical_token_bytes,
    _copy_to_cpu_preserving_stride,
    _cuda_status_code,
    _linear_output,
    _project_relative,
    _save_tensor_artifact,
    _sha256,
    _sha256_path,
    _tensor_bytes,
    _tensor_metadata,
)

EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
FULL_MODEL_RESULT = EXPERIMENT / "full-model-acquisition.json"
REPLAY_MATRIX_RESULT = EXPERIMENT / "replay-matrix.json"
UNFUSED_MATRIX_RESULT = EXPERIMENT / "real-activation-unfused-matrix.json"
DEFAULT_MODEL_DIR = ROOT / MODEL_RELATIVE
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "E004-qwen3-layer-capture" / "real-activation-fused-matrix"
)
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "fused-matrix.json"
REPETITIONS = 3
TOKENIZER_JSON_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)

_CAPTURES: dict[str, dict[str, Any]] = {}
_HOOK_HANDLES: list[Any] = []
_CAPTURE_EVENT_ORDER: list[str] = []


class RealActivationFusedMatrixError(RuntimeError):
    """Raised when fused capture or replay violates the frozen contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--profile-capture", action="store_true")
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealActivationFusedMatrixError(f"{label} must be an object")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RealActivationFusedMatrixError(f"could not read {path}") from error
    return _mapping(value, str(path))


def _snapshot_inventory(model_dir: Path) -> dict[str, dict[str, Any]]:
    result = _json(FULL_MODEL_RESULT)
    snapshot = _mapping(result.get("snapshot"), "full-model snapshot")
    if (
        result.get("slice") != "full_model_snapshot_acquisition_v1"
        or result.get("status") != "pass"
        or snapshot.get("local_root") != model_dir.relative_to(ROOT).as_posix()
    ):
        raise RealActivationFusedMatrixError("full-model snapshot identity changed")
    values = snapshot.get("files")
    if not isinstance(values, list):
        raise RealActivationFusedMatrixError("full-model inventory is missing")
    inventory: dict[str, dict[str, Any]] = {}
    for value in values:
        entry = _mapping(value, "full-model inventory entry")
        path = entry.get("path")
        if not isinstance(path, str) or path in inventory:
            raise RealActivationFusedMatrixError("invalid full-model inventory path")
        inventory[path] = entry
    return inventory


def _verify_used_shards(
    *,
    model_dir: Path,
    weight_map: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    shard_paths: set[str] = set()
    for case in E004_FUSED_REAL_ACTIVATION_CASES:
        for boundary in case.component_boundaries:
            prefix = f"{case.checkpoint_parent_path}.{boundary.projection}"
            for suffix in (
                "input_scale",
                "weight",
                "weight_scale",
                "weight_scale_2",
            ):
                tensor_name = f"{prefix}.{suffix}"
                shard_path = weight_map.get(tensor_name)
                if not isinstance(shard_path, str):
                    raise RealActivationFusedMatrixError(
                        f"checkpoint index is missing {tensor_name}"
                    )
                shard_paths.add(shard_path)
    for shard_path in sorted(shard_paths):
        entry = _mapping(inventory.get(shard_path), f"inventory {shard_path}")
        snapshot_root = model_dir.resolve()
        shard = (snapshot_root / shard_path).resolve()
        try:
            shard.relative_to(snapshot_root)
        except ValueError as error:
            raise RealActivationFusedMatrixError(
                f"checkpoint shard escapes the snapshot: {shard_path}"
            ) from error
        if entry.get("role") != "weight_shard" or _sha256_path(shard) != entry.get(
            "sha256"
        ):
            raise RealActivationFusedMatrixError(
                f"used checkpoint shard hash changed: {shard_path}"
            )
    return tuple(sorted(shard_paths))


def _replay_dependency_cases() -> dict[tuple[int, str], dict[str, Any]]:
    result = _json(REPLAY_MATRIX_RESULT)
    values = result.get("cases")
    if (
        result.get("slice") != "representative_projection_replay_matrix_v1"
        or result.get("status") != "pass"
        or not isinstance(values, list)
    ):
        raise RealActivationFusedMatrixError("replay-matrix dependency changed")
    cases: dict[tuple[int, str], dict[str, Any]] = {}
    for value in values:
        entry = _mapping(value, "replay-matrix case")
        layer = entry.get("layer")
        projection = entry.get("projection")
        if not isinstance(layer, int) or not isinstance(projection, str):
            raise RealActivationFusedMatrixError("invalid replay-matrix case identity")
        key = (layer, projection)
        if key in cases:
            raise RealActivationFusedMatrixError("duplicate replay-matrix case")
        cases[key] = entry
    return cases


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors header length")
            header_length = struct.unpack("<Q", prefix)[0]
            header = json.loads(stream.read(header_length))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RealActivationFusedMatrixError(
            f"could not parse safetensors header: {path}"
        ) from error
    return 8 + header_length, _mapping(header, f"safetensors header {path}")


def _read_source_tensor(
    *,
    tensor_name: str,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    model_dir: Path,
    weight_map: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    header_cache: dict[str, tuple[int, dict[str, Any]]],
) -> tuple[bytes, dict[str, object]]:
    shard_path = weight_map.get(tensor_name)
    if not isinstance(shard_path, str) or shard_path not in inventory:
        raise RealActivationFusedMatrixError(
            f"checkpoint index is missing {tensor_name}"
        )
    inventory_entry = inventory[shard_path]
    shard_sha256 = inventory_entry.get("sha256")
    if inventory_entry.get("role") != "weight_shard" or not isinstance(
        shard_sha256, str
    ):
        raise RealActivationFusedMatrixError(
            f"checkpoint shard identity changed for {tensor_name}"
        )
    snapshot_root = model_dir.resolve()
    shard = (snapshot_root / shard_path).resolve()
    try:
        shard.relative_to(snapshot_root)
    except ValueError as error:
        raise RealActivationFusedMatrixError(
            f"checkpoint shard escapes the snapshot: {shard_path}"
        ) from error
    if shard_path not in header_cache:
        header_cache[shard_path] = _safetensors_header(shard)
    payload_start, header = header_cache[shard_path]
    entry = _mapping(header.get(tensor_name), f"safetensors tensor {tensor_name}")
    offsets = entry.get("data_offsets")
    if (
        entry.get("dtype") != expected_dtype
        or entry.get("shape") != list(expected_shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
        or offsets[0] < 0
        or offsets[1] <= offsets[0]
    ):
        raise RealActivationFusedMatrixError(
            f"checkpoint tensor metadata changed for {tensor_name}"
        )
    byte_length = offsets[1] - offsets[0]
    element_count = math.prod(expected_shape) if expected_shape else 1
    element_bytes = 4 if expected_dtype == "F32" else 1
    if byte_length != element_count * element_bytes:
        raise RealActivationFusedMatrixError(
            f"checkpoint tensor byte length changed for {tensor_name}"
        )
    try:
        with shard.open("rb") as stream:
            stream.seek(payload_start + offsets[0])
            payload = stream.read(byte_length)
    except OSError as error:
        raise RealActivationFusedMatrixError(
            f"could not read checkpoint tensor {tensor_name}"
        ) from error
    if len(payload) != byte_length:
        raise RealActivationFusedMatrixError(
            f"checkpoint tensor is truncated: {tensor_name}"
        )
    return payload, {
        "tensor_name": tensor_name,
        "shard_path": shard_path,
        "shard_sha256": shard_sha256,
        "data_offsets": offsets,
        "dtype": expected_dtype,
        "shape": list(expected_shape),
        "byte_length": byte_length,
        "sha256": _sha256_bytes(payload),
    }


def _assemble_fused_payloads(
    case: E004FusedRealActivationCase,
    packed_parts: list[bytes],
    linear_scale_parts: list[bytes],
) -> tuple[bytes, bytes, tuple[bytes, ...]]:
    if len(packed_parts) != len(case.component_boundaries) or len(
        linear_scale_parts
    ) != len(case.component_boundaries):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} source component count changed"
        )
    prepared_component_scales: list[bytes] = []
    for packed, linear_scale, boundary in zip(
        packed_parts, linear_scale_parts, case.component_boundaries, strict=True
    ):
        if len(packed) != math.prod(boundary.packed_weight_shape) or len(
            linear_scale
        ) != math.prod(boundary.weight_scale_shape):
            raise RealActivationFusedMatrixError(
                f"{case.case_id} {boundary.projection} source shape changed"
            )
        prepared_component_scales.append(
            swizzle_scales_128x4(
                linear_scale,
                boundary.output_width,
                boundary.input_width // 16,
            )
        )
    fused_weight = b"".join(packed_parts)
    fused_runtime_scale = swizzle_scales_128x4(
        b"".join(linear_scale_parts),
        case.output_width,
        case.input_width // 16,
    )
    if len(fused_weight) != math.prod(case.packed_weight_shape) or len(
        fused_runtime_scale
    ) != math.prod(case.weight_scale_shape):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} fused source storage length changed"
        )
    return fused_weight, fused_runtime_scale, tuple(prepared_component_scales)


def _inspect_source_construction(
    case: E004FusedRealActivationCase,
    *,
    model_dir: Path,
    weight_map: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    header_cache: dict[str, tuple[int, dict[str, Any]]],
    replay_cases: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, object]:
    components: list[dict[str, object]] = []
    packed_parts: list[bytes] = []
    linear_scale_parts: list[bytes] = []
    input_scales: list[float] = []
    weight_scales_2: list[float] = []
    for boundary in case.component_boundaries:
        prefix = f"{case.checkpoint_parent_path}.{boundary.projection}"
        tensors: dict[str, dict[str, object]] = {}
        payloads: dict[str, bytes] = {}
        for suffix, dtype, shape in (
            ("input_scale", "F32", ()),
            ("weight", "U8", boundary.packed_weight_shape),
            ("weight_scale", "F8_E4M3", boundary.weight_scale_shape),
            ("weight_scale_2", "F32", ()),
        ):
            payload, record = _read_source_tensor(
                tensor_name=f"{prefix}.{suffix}",
                expected_dtype=dtype,
                expected_shape=shape,
                model_dir=model_dir,
                weight_map=weight_map,
                inventory=inventory,
                header_cache=header_cache,
            )
            payloads[suffix] = payload
            tensors[suffix] = record
        input_scale = struct.unpack("<f", payloads["input_scale"])[0]
        weight_scale_2 = struct.unpack("<f", payloads["weight_scale_2"])[0]
        if not (
            math.isfinite(input_scale)
            and input_scale > 0
            and math.isfinite(weight_scale_2)
            and weight_scale_2 > 0
        ):
            raise RealActivationFusedMatrixError(
                f"{case.case_id} {boundary.projection} has invalid global scales"
            )
        packed_parts.append(payloads["weight"])
        linear_scale_parts.append(payloads["weight_scale"])
        input_scales.append(input_scale)
        weight_scales_2.append(weight_scale_2)
        components.append(
            {
                "projection": boundary.projection,
                "row_range": [boundary.row_start, boundary.row_end],
                "source_tensors": tensors,
                "prepared_packed_weight_sha256": _sha256_bytes(payloads["weight"]),
                "prepared_runtime_weight_scale_sha256": None,
                "input_scale": input_scale,
                "weight_scale_2": weight_scale_2,
                "replay_matrix_regression_match": None,
            }
        )
    fused_weight, fused_runtime_scale, prepared_scales = _assemble_fused_payloads(
        case, packed_parts, linear_scale_parts
    )
    if len(set(input_scales)) != 1 or len(set(weight_scales_2)) != 1:
        raise RealActivationFusedMatrixError(
            f"{case.case_id} component global scales are not exactly shared"
        )
    for component, prepared_scale, boundary in zip(
        components, prepared_scales, case.component_boundaries, strict=True
    ):
        component["prepared_runtime_weight_scale_sha256"] = _sha256_bytes(
            prepared_scale
        )
        overlap = replay_cases.get((case.layer, boundary.projection))
        if overlap is not None:
            source_hashes = _mapping(
                overlap.get("source_tensor_sha256"), "replay source hashes"
            )
            tensors = _mapping(component["source_tensors"], "component tensors")
            component["replay_matrix_regression_match"] = bool(
                source_hashes
                == {
                    suffix: _mapping(tensors[suffix], suffix)["sha256"]
                    for suffix in (
                        "input_scale",
                        "weight",
                        "weight_scale",
                        "weight_scale_2",
                    )
                }
                and overlap.get("runtime_weight_scale_sha256")
                == component["prepared_runtime_weight_scale_sha256"]
                and overlap.get("input_scale") == component["input_scale"]
                and overlap.get("weight_scale_2") == component["weight_scale_2"]
            )
            if component["replay_matrix_regression_match"] is not True:
                raise RealActivationFusedMatrixError(
                    f"{case.case_id} {boundary.projection} replay regression changed"
                )
    input_global_scale = _f32(max(input_scales))
    weight_global_scale = _f32(max(weight_scales_2))
    return {
        "component_order": list(case.component_projections),
        "component_row_boundaries": list(case.component_row_boundaries),
        "components": components,
        "fused_packed_weight_sha256": _sha256_bytes(fused_weight),
        "fused_runtime_weight_scale_sha256": _sha256_bytes(fused_runtime_scale),
        "expected_runtime_scalars": {
            "reduction_rule": "float32_max_over_ordered_components",
            "input_global_scale": input_global_scale,
            "weight_global_scale": weight_global_scale,
            "alpha": _f32(input_global_scale * weight_global_scale),
            "input_global_scale_inv": _f32(1.0 / input_global_scale),
        },
        "source_snapshot_verified": True,
    }


def _target_module(
    model: torch.nn.Module, case: E004FusedRealActivationCase
) -> tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    target = modules.get(case.module_path)
    if target is None:
        raise RealActivationFusedMatrixError(
            f"expected exact raw-model module path {case.module_path}"
        )
    return case.module_path, target


def _runtime_projection(
    model: torch.nn.Module,
    case: E004FusedRealActivationCase,
    source: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, object]]:
    module_path, target = _target_module(model, case)
    quant_method = getattr(target, "quant_method", None)
    kernel = getattr(quant_method, "kernel", None)
    weight = getattr(target, "weight", None)
    weight_scale = getattr(target, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(
        weight_scale, torch.Tensor
    ):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} NVFP4 runtime tensors are missing"
        )
    selected_kernel = type(kernel).__name__ if kernel is not None else None
    if (
        type(target).__name__ != case.module_class
        or type(quant_method).__name__ != "ModelOptNvFp4LinearMethod"
        or selected_kernel != EXPECTED_KERNEL
    ):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} fused runtime class or kernel changed"
        )
    expected_widths = list(case.component_output_widths)
    logical_widths = getattr(target, "logical_widths", None)
    tp_contract = {
        "tp_size": 1,
        "tp_rank": 0,
        "input_size_per_partition": case.input_width,
        "output_size_per_partition": case.output_width,
        "output_sizes": expected_widths,
        "output_partition_sizes": expected_widths,
        "gather_output": False,
    }
    if (
        any(getattr(target, name, None) != value for name, value in tp_contract.items())
        or logical_widths != expected_widths
    ):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} tensor-parallel metadata changed"
        )
    weight_metadata = _tensor_metadata(weight)
    scale_metadata = _tensor_metadata(weight_scale)
    if (
        weight_metadata["shape"] != list(case.packed_weight_shape)
        or weight_metadata["sha256"] != source["fused_packed_weight_sha256"]
        or scale_metadata["shape"] != list(case.weight_scale_shape)
        or scale_metadata["sha256"] != source["fused_runtime_weight_scale_sha256"]
        or getattr(target, "weights_padding_cols", None) != 0
    ):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} fused runtime tensor identity changed"
        )
    expected_scalars = _mapping(
        source.get("expected_runtime_scalars"), "expected runtime scalars"
    )
    runtime_scalars = {
        "input_global_scale": float(target.input_global_scale.item()),
        "weight_global_scale": float(target.weight_global_scale.item()),
        "alpha": float(target.alpha.item()),
        "input_global_scale_inv": float(target.input_global_scale_inv.item()),
    }
    if any(runtime_scalars[name] != expected_scalars[name] for name in runtime_scalars):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} fused runtime scalar reduction changed"
        )
    source_components = source.get("components")
    if not isinstance(source_components, list):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} source components are missing"
        )
    component_bindings: list[dict[str, object]] = []
    for boundary, component_value in zip(
        case.component_boundaries, source_components, strict=True
    ):
        component = _mapping(component_value, "source component")
        packed_hash = _sha256(
            _tensor_bytes(weight[boundary.row_start : boundary.row_end])
        )
        scale_hash = _sha256(
            _tensor_bytes(weight_scale[boundary.row_start : boundary.row_end])
        )
        expected_packed_hash = component["prepared_packed_weight_sha256"]
        expected_scale_hash = component["prepared_runtime_weight_scale_sha256"]
        packed_match = packed_hash == expected_packed_hash
        scale_match = scale_hash == expected_scale_hash
        if not packed_match or not scale_match:
            raise RealActivationFusedMatrixError(
                f"{case.case_id} {boundary.projection} runtime slice changed"
            )
        component_bindings.append(
            {
                "projection": boundary.projection,
                "row_range": [boundary.row_start, boundary.row_end],
                "packed_weight_slice_sha256": packed_hash,
                "runtime_weight_scale_slice_sha256": scale_hash,
                "checkpoint_weight_match": packed_match,
                "independent_scale_swizzle_match": scale_match,
            }
        )
    return target, {
        "module_path": module_path,
        "module_class": type(target).__name__,
        "quant_method_class": type(quant_method).__name__,
        "selected_kernel": selected_kernel,
        "logical_widths": list(logical_widths),
        "packed_weight": weight_metadata,
        "runtime_weight_scale": scale_metadata,
        "weights_padding_cols": getattr(target, "weights_padding_cols", None),
        "tp_size": target.tp_size,
        "tp_rank": target.tp_rank,
        "gather_output": target.gather_output,
        **runtime_scalars,
        "component_bindings": component_bindings,
        "global_scale_reduction_matches": True,
    }


def _capture_tensor(case_id: str, role: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise RealActivationFusedMatrixError(
            f"{case_id} expected BF16 {role}, got {tensor.dtype}"
        )
    entry = _CAPTURES.setdefault(case_id, {})
    if role in entry:
        raise RealActivationFusedMatrixError(
            f"{case_id} {role} hook fired more than once"
        )
    entry[role] = {
        "tensor": _copy_to_cpu_preserving_stride(tensor),
        "source_metadata": _tensor_metadata(tensor),
        "event_count": 1,
    }
    _CAPTURE_EVENT_ORDER.append(f"{case_id}:{role}")


def _capture_input(
    case_id: str, _module: torch.nn.Module, args: tuple[object, ...]
) -> None:
    if len(args) != 1 or not isinstance(args[0], torch.Tensor):
        raise RealActivationFusedMatrixError(
            f"{case_id} pre-hook did not receive one tensor"
        )
    _capture_tensor(case_id, "input", args[0])


def _capture_output(
    case_id: str,
    _module: torch.nn.Module,
    _args: tuple[object, ...],
    output: object,
) -> None:
    _capture_tensor(case_id, "module_output", _linear_output(output))


def _install_capture_hooks(
    model: torch.nn.Module,
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, object]:
    _CAPTURES.clear()
    _HOOK_HANDLES.clear()
    _CAPTURE_EVENT_ORDER.clear()
    targets: list[dict[str, object]] = []
    for case in E004_FUSED_REAL_ACTIVATION_CASES:
        target, runtime = _runtime_projection(model, case, source_by_id[case.case_id])
        _HOOK_HANDLES.extend(
            (
                target.register_forward_pre_hook(
                    functools.partial(_capture_input, case.case_id)
                ),
                target.register_forward_hook(
                    functools.partial(_capture_output, case.case_id)
                ),
            )
        )
        targets.append({"case_id": case.case_id, **runtime})
    return {
        "observed_model_class": type(model).__name__,
        "targets": targets,
        "installed_hook_count": len(_HOOK_HANDLES),
    }


def _remove_capture_hooks(_model: torch.nn.Module) -> int:
    count = len(_HOOK_HANDLES)
    for handle in _HOOK_HANDLES:
        handle.remove()
    _HOOK_HANDLES.clear()
    return count


def _capture_record(case_id: str, role: str) -> dict[str, Any]:
    entry = _CAPTURES.get(case_id)
    if not isinstance(entry, dict):
        raise RealActivationFusedMatrixError(f"missing {case_id} capture")
    value = entry.get(role)
    if not isinstance(value, dict) or value.get("event_count") != 1:
        raise RealActivationFusedMatrixError(f"invalid {case_id} {role} capture")
    if not isinstance(value.get("tensor"), torch.Tensor) or not isinstance(
        value.get("source_metadata"), dict
    ):
        raise RealActivationFusedMatrixError(f"invalid {case_id} {role} tensor")
    return value


def _expected_hook_event_order() -> list[str]:
    return [
        f"{case.case_id}:{role}"
        for case in E004_FUSED_REAL_ACTIVATION_CASES
        for role in ("input", "module_output")
    ]


def _component_output_slices(
    case: E004FusedRealActivationCase,
    captured: torch.Tensor,
    outputs: list[torch.Tensor],
) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for boundary in case.component_boundaries:
        captured_slice = captured[:, boundary.row_start : boundary.row_end]
        replay_slices = [
            output[:, boundary.row_start : boundary.row_end] for output in outputs
        ]
        captured_hash = _sha256(_tensor_bytes(captured_slice))
        replay_hashes = [_sha256(_tensor_bytes(value)) for value in replay_slices]
        matches = [
            _bitwise_tensor_match(value, captured_slice) for value in replay_slices
        ]
        if (
            replay_hashes != [captured_hash] * REPETITIONS
            or matches != [True] * REPETITIONS
        ):
            raise RealActivationFusedMatrixError(
                f"{case.case_id} {boundary.projection} replay slice changed"
            )
        values.append(
            {
                "projection": boundary.projection,
                "feature_range": [boundary.row_start, boundary.row_end],
                "captured_sha256": captured_hash,
                "replay_sha256s": replay_hashes,
                "logical_matches": matches,
            }
        )
    return values


def _replay_case(
    model: torch.nn.Module, case: E004FusedRealActivationCase
) -> dict[str, object]:
    _path, target = _target_module(model, case)
    input_capture = _capture_record(case.case_id, "input")
    output_capture = _capture_record(case.case_id, "module_output")
    input_cpu = input_capture["tensor"]
    captured_output_cpu = output_capture["tensor"]
    activation = torch.empty_strided(
        tuple(input_cpu.shape),
        tuple(input_cpu.stride()),
        dtype=input_cpu.dtype,
        device="cuda",
    )
    activation.copy_(input_cpu)
    torch.cuda.synchronize()
    activation_metadata = _tensor_metadata(activation)
    if activation_metadata != input_capture["source_metadata"]:
        raise RealActivationFusedMatrixError(
            f"{case.case_id} reconstructed activation identity changed"
        )
    outputs_gpu: list[torch.Tensor] = []
    with torch.inference_mode():
        _linear_output(target(activation))
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(case.target_nvtx_range):
            for _ in range(REPETITIONS):
                outputs_gpu.append(_linear_output(target(activation)))
                torch.cuda.synchronize()
    output_source_metadata = _tensor_metadata(outputs_gpu[0])
    finite_flags = [bool(torch.isfinite(output).all()) for output in outputs_gpu]
    torch.cuda.synchronize()
    outputs = [_copy_to_cpu_preserving_stride(output) for output in outputs_gpu]
    output_hashes = [_sha256(_tensor_bytes(output)) for output in outputs]
    captured_output_hash = _sha256(_tensor_bytes(captured_output_cpu))
    logical_matches = [
        _bitwise_tensor_match(output, captured_output_cpu)
        and output_hash == captured_output_hash
        for output, output_hash in zip(outputs, output_hashes, strict=True)
    ]
    if (
        not all(finite_flags)
        or output_hashes != [captured_output_hash] * REPETITIONS
        or not all(logical_matches)
    ):
        raise RealActivationFusedMatrixError(
            f"{case.case_id} replay is not finite, stable, and byte exact"
        )
    return {
        "warmup_runs": 1,
        "repetitions": REPETITIONS,
        "synchronized": True,
        "all_finite": True,
        "output_shape": list(outputs[0].shape),
        "output_dtype": str(outputs[0].dtype).removeprefix("torch."),
        "output_sha256s": output_hashes,
        "output_hash_stable": True,
        "captured_module_output_sha256": captured_output_hash,
        "logical_byte_exact_captured_module_output_matches": logical_matches,
        "reconstructed_activation_metadata": activation_metadata,
        "logical_byte_exact_captured_input_match": True,
        "max_abs_error": max(
            float((output.float() - captured_output_cpu.float()).abs().max())
            for output in outputs
        ),
        "mean_abs_error": max(
            float((output.float() - captured_output_cpu.float()).abs().mean())
            for output in outputs
        ),
        "component_output_slices": _component_output_slices(
            case, captured_output_cpu, outputs
        ),
        "output_tensor": outputs[0],
        "output_source_metadata": output_source_metadata,
    }


def _replay_matrix(model: torch.nn.Module) -> list[dict[str, object]]:
    return [
        {"case_id": case.case_id, **_replay_case(model, case)}
        for case in E004_FUSED_REAL_ACTIVATION_CASES
    ]


def _require_cuda_success(operation: str, result: object) -> None:
    status = _cuda_status_code(result)
    if status != 0:
        raise RealActivationFusedMatrixError(
            f"{operation} failed with CUDA runtime status {status}"
        )


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.max_model_len < len(PROMPT_TOKEN_IDS) + 1:
        raise ValueError("max-model-len cannot hold the frozen request")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    if args.kv_cache_memory_bytes <= 0:
        raise ValueError("kv-cache-memory-bytes must be positive")
    _project_relative(args.artifact_root, label="artifact root")
    _project_relative(args.output, label="output")
    model_dir = args.model_dir.absolute()
    try:
        model_relative = model_dir.relative_to(ROOT)
    except ValueError as error:
        raise RealActivationFusedMatrixError(
            "model directory must remain under the repository root"
        ) from error
    if not model_relative.parts or model_relative.parts[0] != "models":
        raise RealActivationFusedMatrixError(
            "model directory must remain under models/"
        )
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"pinned model snapshot is missing: {model_dir}")
    return model_dir, model_relative


def _identity_dependencies() -> list[dict[str, str]]:
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


def _pending_backend_range(case: E004FusedRealActivationCase) -> dict[str, object]:
    return {
        "target_nvtx_range": case.target_nvtx_range,
        "target_kernel_ids": [],
        "target_kernel_set_sha256": None,
        "expected_sm120_cutlass_signature_present": False,
        "activation_quantization_signature_present": False,
        "fallback_status": "pending_profiler",
    }


def main() -> int:
    args = parse_args()
    model_dir, model_relative = _validate_args(args)
    index = _json(model_dir / "model.safetensors.index.json")
    weight_map = _mapping(index.get("weight_map"), "checkpoint weight map")
    inventory = _snapshot_inventory(model_dir)
    _verify_used_shards(
        model_dir=model_dir,
        weight_map=weight_map,
        inventory=inventory,
    )
    replay_cases = _replay_dependency_cases()
    header_cache: dict[str, tuple[int, dict[str, Any]]] = {}
    source_by_id = {
        case.case_id: _inspect_source_construction(
            case,
            model_dir=model_dir,
            weight_map=weight_map,
            inventory=inventory,
            header_cache=header_cache,
            replay_cases=replay_cases,
        )
        for case in E004_FUSED_REAL_ACTIVATION_CASES
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("E004 is pinned to the RTX 5080 sm_120 environment")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.utils.flashinfer import has_flashinfer

    if not has_flashinfer():
        raise RealActivationFusedMatrixError(
            "FlashInfer is unavailable; source activate-nvfp4-lab.sh before running"
        )
    requested_args: dict[str, object] = {
        "runner": "generate",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "quantization": "modelopt_fp4",
        "load_format": "safetensors",
        "trust_remote_code": False,
        "skip_tokenizer_init": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": 0,
        "kv_cache_dtype": "auto",
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "compilation_config": 0,
        "linear_backend": "auto",
        "seed": 0,
    }
    llm = LLM(model=str(model_dir), tokenizer=str(model_dir), **requested_args)
    free_after_load, _ = torch.cuda.mem_get_info()
    install_results = llm.apply_model(
        functools.partial(_install_capture_hooks, source_by_id=source_by_id)
    )
    if len(install_results) != 1 or not isinstance(install_results[0], dict):
        raise RealActivationFusedMatrixError("hook installation returned no metadata")
    installation = install_results[0]
    if installation.get("installed_hook_count") != 12:
        raise RealActivationFusedMatrixError(
            "the fused matrix did not install exactly 12 hooks"
        )
    try:
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=list(PROMPT_TOKEN_IDS))],
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                detokenize=False,
                seed=0,
            ),
            use_tqdm=False,
        )
    finally:
        removed = llm.apply_model(_remove_capture_hooks)
    if removed != [12]:
        raise RealActivationFusedMatrixError(
            f"unexpected removed-hook count: {removed}"
        )
    generated_ids = list(outputs[0].outputs[0].token_ids)
    if len(generated_ids) != 1:
        raise RealActivationFusedMatrixError(
            f"expected one generated token, observed {len(generated_ids)}"
        )
    for case in E004_FUSED_REAL_ACTIVATION_CASES:
        _capture_record(case.case_id, "input")
        _capture_record(case.case_id, "module_output")
    if _CAPTURE_EVENT_ORDER != _expected_hook_event_order():
        raise RealActivationFusedMatrixError(
            f"unexpected capture event order: {_CAPTURE_EVENT_ORDER}"
        )

    profiler_started = False
    if args.profile_capture:
        _require_cuda_success(
            "cudaProfilerStart", torch.cuda.cudart().cudaProfilerStart()
        )
        profiler_started = True
    try:
        replay_results = llm.apply_model(_replay_matrix)
    finally:
        if profiler_started:
            _require_cuda_success(
                "cudaProfilerStop", torch.cuda.cudart().cudaProfilerStop()
            )
    if len(replay_results) != 1 or not isinstance(replay_results[0], list):
        raise RealActivationFusedMatrixError("fused replay returned no evidence")
    replay_by_id = {str(entry["case_id"]): entry for entry in replay_results[0]}
    expected_case_ids = {case.case_id for case in E004_FUSED_REAL_ACTIVATION_CASES}
    if set(replay_by_id) != expected_case_ids:
        raise RealActivationFusedMatrixError("fused replay returned the wrong cases")
    targets = installation.get("targets")
    if not isinstance(targets, list):
        raise RealActivationFusedMatrixError("runtime target metadata is missing")
    target_by_id = {str(entry["case_id"]): entry for entry in targets}
    if set(target_by_id) != expected_case_ids:
        raise RealActivationFusedMatrixError(
            "runtime target metadata has the wrong cases"
        )

    case_results: list[dict[str, object]] = []
    for case in E004_FUSED_REAL_ACTIVATION_CASES:
        input_capture = _capture_record(case.case_id, "input")
        output_capture = _capture_record(case.case_id, "module_output")
        input_tensor = input_capture["tensor"]
        output_tensor = output_capture["tensor"]
        if tuple(input_tensor.shape) != case.input_shape(len(PROMPT_TOKEN_IDS)):
            raise RealActivationFusedMatrixError(
                f"unexpected {case.case_id} input shape: {tuple(input_tensor.shape)}"
            )
        if tuple(output_tensor.shape) != case.output_shape(len(PROMPT_TOKEN_IDS)):
            raise RealActivationFusedMatrixError(
                f"unexpected {case.case_id} output shape: {tuple(output_tensor.shape)}"
            )
        replay = replay_by_id[case.case_id]
        replay_tensor = replay.pop("output_tensor", None)
        replay_source_metadata = replay.pop("output_source_metadata", None)
        replay.pop("case_id", None)
        if not isinstance(replay_tensor, torch.Tensor) or not isinstance(
            replay_source_metadata, dict
        ):
            raise RealActivationFusedMatrixError(
                f"{case.case_id} replay output metadata is missing"
            )
        input_artifact = _save_tensor_artifact(
            args.artifact_root / f"{case.artifact_slug}-input.pt",
            input_tensor,
            input_capture["source_metadata"],
        )
        output_artifact = _save_tensor_artifact(
            args.artifact_root / f"{case.artifact_slug}-captured-module-output.pt",
            output_tensor,
            output_capture["source_metadata"],
        )
        replay_artifact = _save_tensor_artifact(
            args.artifact_root / f"{case.artifact_slug}-replay-output.pt",
            replay_tensor,
            replay_source_metadata,
        )
        runtime = dict(target_by_id[case.case_id])
        runtime.pop("case_id", None)
        case_results.append(
            {
                "case_id": case.case_id,
                "layer": case.layer,
                "role": case.role,
                "projection": case.projection,
                "adapter_scope": case.adapter_scope,
                "module_path": case.module_path,
                "module_class": case.module_class,
                "tensor_role": "module_input",
                "phase": "prefill",
                "event_count": 1,
                "activation_provenance": "real_qwen_prefill",
                "source_construction": source_by_id[case.case_id],
                "capture": {
                    "input_artifact": input_artifact,
                    "captured_module_output_artifact": output_artifact,
                    "metadata_preserved_fields": list(PRESERVED_TRANSFER_FIELDS),
                    "device_transfer_recorded": True,
                },
                "runtime_projection": runtime,
                "replay": {
                    **replay,
                    "input_sha256": replay["reconstructed_activation_metadata"][
                        "sha256"
                    ],
                    "replay_output_artifact": replay_artifact,
                },
                "backend_range": _pending_backend_range(case),
            }
        )
        print(f"saved_case={case.case_id}", flush=True)

    input_hashes = [
        str(case["capture"]["input_artifact"]["tensor"]["sha256"])
        for case in case_results
    ]
    output_hashes = [
        str(case["replay"]["captured_module_output_sha256"]) for case in case_results
    ]
    if len(set(input_hashes)) != 6 or len(set(output_hashes)) != 6:
        raise RealActivationFusedMatrixError(
            "captured cases do not have six distinct input and output identities"
        )
    input_identity = {
        "provenance": "fixed_public_token_sequence",
        "token_ids_committed_in_result": False,
        "token_ids_encoding": "little_endian_signed_int32",
        "token_count": len(PROMPT_TOKEN_IDS),
        "token_ids_sha256": _sha256(_canonical_token_bytes(PROMPT_TOKEN_IDS)),
        "generated_token_count": len(generated_ids),
        "generated_token_ids_sha256": _sha256(_canonical_token_bytes(generated_ids)),
        "tokenizer_initialized": False,
        "tokenizer_revision": REVISION,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "sampling": {
            "temperature": 0.0,
            "max_tokens": 1,
            "detokenize": False,
            "seed": 0,
        },
    }
    expected_input_identity = _mapping(
        _json(UNFUSED_MATRIX_RESULT).get("input_identity"),
        "unfused input identity",
    )
    if input_identity != expected_input_identity:
        raise RealActivationFusedMatrixError("request identity regression changed")

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    result = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_fused_real_activation_observation_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "decision": "pending_profiler",
        "repository": {"id": REPO_ID, "revision": REVISION},
        "model_load": {
            "local_snapshot_path": model_relative.as_posix(),
            "frozen_environment": FROZEN_VLLM_ENVIRONMENT,
            "requested_args": requested_args,
            "observed_model_class": installation["observed_model_class"],
            "model_load_count": 1,
            "request_count": 1,
            "request_completed": True,
            "free_memory_before_bytes": free_before,
            "free_memory_after_load_bytes": free_after_load,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "input_identity": input_identity,
        "matrix": {
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
            "case_ids": [case.case_id for case in E004_FUSED_REAL_ACTIVATION_CASES],
            "case_count": 6,
            "repetitions_per_case": REPETITIONS,
            "hook_count": 12,
            "hook_event_order": list(_CAPTURE_EVENT_ORDER),
            "distinct_input_sha256_count": len(set(input_hashes)),
            "distinct_module_output_sha256_count": len(set(output_hashes)),
        },
        "identity_dependencies": _identity_dependencies(),
        "backend": {
            "requested_format": "nvfp4",
            "requested_backend": "auto",
            "expected_selected_vllm_kernel": EXPECTED_KERNEL,
            "reported_backend": None,
            "profiler_sha256": None,
            "kernel_catalog": [],
        },
        "cases": case_results,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": total_memory,
        },
        "command": {"argv": [sys.executable, *sys.argv], "cwd": str(ROOT)},
        "claim_boundary": (
            "For one fixed nine-token request, this raw observation records six "
            "production-fused Qwen prefill activations, exact checkpoint-component "
            "bindings, and deterministic same-module replays for qkv_proj and "
            "gate_up_proj at layers 0, 18, and 35. Profiler identity remains "
            "pending. It does not establish NVFP4 numerical correctness, equivalence "
            "to separately executed component projections, prompt diversity, final-"
            "logit or model quality, performance, cross-backend agreement, high-"
            "precision equivalence, or completion of Gate 2."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"case_count={len(case_results)}")
    print(f"artifact_count={len(case_results) * 3}")
    print(f"replay_count={len(case_results) * REPETITIONS}")
    print(f"peak_allocated_bytes={peak_allocated}")
    print(f"selected_vllm_kernel={EXPECTED_KERNEL}")
    print("status=pass")
    print("decision=pending_profiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
