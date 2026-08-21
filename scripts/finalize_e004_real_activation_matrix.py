#!/usr/bin/env python3
"""Finalize the profiled E004 unfused real-activation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from nvfp4_doctor.capture import E004_UNFUSED_REAL_ACTIVATION_CASES
from nvfp4_doctor.env import collect_git, collect_gpu, collect_software
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
    / "real-activation-matrix"
    / "unfused-matrix.json"
)
DEFAULT_REPORT = (
    ROOT / ".local" / "profiles" / "e004-real-activation-unfused-matrix.nsys-rep"
)
DEFAULT_RESULTS = EXPERIMENT / "real-activation-unfused-matrix.json"
DEFAULT_MANIFEST = EXPERIMENT / "manifest-real-activation-unfused-matrix.json"
FULL_MODEL_RESULT = EXPERIMENT / "full-model-acquisition.json"
FULL_MODEL_MANIFEST = EXPERIMENT / "manifest-full-model-acquisition.json"
REPLAY_MATRIX_RESULT = EXPERIMENT / "replay-matrix.json"
REPLAY_MATRIX_MANIFEST = EXPERIMENT / "manifest-replay-matrix.json"
SINGLE_REAL_RESULT = EXPERIMENT / "real-activation-replay.json"
SINGLE_REAL_MANIFEST = EXPERIMENT / "manifest-real-activation-replay.json"
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "backends" / "nsys.py",
    ROOT / "src" / "nvfp4_doctor" / "backends" / "__init__.py",
    ROOT / "src" / "nvfp4_doctor" / "capture" / "__init__.py",
    ROOT / "src" / "nvfp4_doctor" / "capture" / "e004.py",
    ROOT / "scripts" / "run_e004_real_activation_capture.py",
    ROOT / "scripts" / "finalize_e004_real_activation.py",
    ROOT / "scripts" / "run_e004_real_activation_matrix.py",
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_e004_real_activation_matrix_profile.sh",
    ROOT / "tests" / "unit" / "test_e004_real_activation_matrix_cases.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_matrix_capture.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_matrix_finalization.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_matrix_evidence.py",
    ROOT / "tests" / "unit" / "test_nsys_evidence.py",
)
DEPENDENCY_PATHS = (
    ("full-model-acquisition-result", FULL_MODEL_RESULT),
    ("full-model-acquisition-manifest", FULL_MODEL_MANIFEST),
    ("representative-replay-matrix-result", REPLAY_MATRIX_RESULT),
    ("representative-replay-matrix-manifest", REPLAY_MATRIX_MANIFEST),
    ("single-real-activation-result", SINGLE_REAL_RESULT),
    ("single-real-activation-manifest", SINGLE_REAL_MANIFEST),
)
DEPENDENCY_PAIRS = (
    (FULL_MODEL_RESULT, FULL_MODEL_MANIFEST),
    (REPLAY_MATRIX_RESULT, REPLAY_MATRIX_MANIFEST),
    (SINGLE_REAL_RESULT, SINGLE_REAL_MANIFEST),
)

RealActivationMatrixFinalizationError = RealActivationFinalizationError

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
    "identity_dependency",
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
        raise RealActivationMatrixFinalizationError(
            f"{label} fields changed; missing={missing}, extra={extra}"
        )


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
        raise RealActivationMatrixFinalizationError(f"{label} file metadata changed")


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
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


def _expected_identity_by_id() -> dict[str, dict[str, Any]]:
    replay_matrix = _json(REPLAY_MATRIX_RESULT)
    if (
        replay_matrix.get("slice") != "representative_projection_replay_matrix_v1"
        or replay_matrix.get("status") != "pass"
    ):
        raise RealActivationMatrixFinalizationError(
            "representative replay dependency did not pass"
        )
    values = replay_matrix.get("cases")
    if not isinstance(values, list):
        raise RealActivationMatrixFinalizationError(
            "representative replay dependency has no cases"
        )
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for value in values:
        entry = _mapping(value, "representative replay case")
        layer = entry.get("layer")
        projection = entry.get("projection")
        if not isinstance(layer, int) or not isinstance(projection, str):
            raise RealActivationMatrixFinalizationError(
                "representative replay case identity is malformed"
            )
        key = (layer, projection)
        by_key.setdefault(key, []).append(entry)
    identities: dict[str, dict[str, Any]] = {}
    for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
        matches = by_key.get((case.layer, case.projection), [])
        if len(matches) != 1:
            raise RealActivationMatrixFinalizationError(
                f"expected one dependency identity for {case.case_id}"
            )
        entry = matches[0]
        source_hashes = _mapping(
            entry.get("source_tensor_sha256"),
            f"{case.case_id} source tensor hashes",
        )
        _require_exact_keys(
            source_hashes,
            {"input_scale", "weight", "weight_scale", "weight_scale_2"},
            label=f"{case.case_id} source tensor hashes",
        )
        scalar_values = (
            entry.get("input_scale"),
            entry.get("weight_scale_2"),
            entry.get("alpha_runtime_f32"),
        )
        if (
            not all(SHA256.fullmatch(str(value)) for value in source_hashes.values())
            or not SHA256.fullmatch(str(entry.get("runtime_weight_scale_sha256", "")))
            or not all(isinstance(value, (int, float)) for value in scalar_values)
        ):
            raise RealActivationMatrixFinalizationError(
                f"{case.case_id} dependency identity is malformed"
            )
        identities[case.case_id] = {
            "checkpoint_source_tensor_sha256": source_hashes,
            "runtime_packed_weight_sha256": source_hashes.get("weight"),
            "runtime_weight_scale_sha256": entry.get("runtime_weight_scale_sha256"),
            "packed_weight_shape": list(case.packed_weight_shape),
            "weight_scale_shape": list(case.weight_scale_shape),
            "expected_runtime_scalars": {
                "input_global_scale": entry.get("input_scale"),
                "weight_global_scale": entry.get("weight_scale_2"),
                "alpha": entry.get("alpha_runtime_f32"),
            },
        }
    return identities


def _validate_model_and_input(run: dict[str, Any]) -> None:
    _require_exact_keys(run, TOP_LEVEL_KEYS, label="runtime observation")
    if (
        run.get("schema_version") != 1
        or run.get("experiment_id") != "E004-qwen3-layer-capture"
        or run.get("slice") != "representative_unfused_real_activation_observation_v1"
        or run.get("status") != "pass"
        or run.get("decision") != "pending_profiler"
    ):
        raise RealActivationMatrixFinalizationError(
            "matrix runtime observation did not pass"
        )
    repository = _mapping(run.get("repository"), "repository")
    _require_exact_keys(repository, {"id", "revision"}, label="repository")
    if repository != {
        "id": "nvidia/Qwen3-8B-NVFP4",
        "revision": REVISION,
    }:
        raise RealActivationMatrixFinalizationError("repository identity changed")
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
    requested = _mapping(model_load.get("requested_args"), "requested model args")
    frozen_environment = _mapping(
        model_load.get("frozen_environment"), "frozen environment"
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
    memory_fields = (
        model_load.get("free_memory_before_bytes"),
        model_load.get("free_memory_after_load_bytes"),
        model_load.get("peak_allocated_bytes"),
        model_load.get("peak_reserved_bytes"),
    )
    if (
        model_load.get("local_snapshot_path")
        != f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}"
        or model_load.get("observed_model_class") != "Qwen3ForCausalLM"
        or model_load.get("model_load_count") != 1
        or model_load.get("request_count") != 1
        or model_load.get("request_completed") is not True
        or requested != expected_requested
        or frozen_environment != expected_environment
        or not all(isinstance(value, int) and value > 0 for value in memory_fields)
    ):
        raise RealActivationMatrixFinalizationError("model-load contract changed")
    input_identity = _mapping(run.get("input_identity"), "input identity")
    _require_exact_keys(
        input_identity,
        {
            "provenance",
            "token_ids_committed_in_result",
            "token_ids_encoding",
            "token_count",
            "token_ids_sha256",
            "generated_token_count",
            "generated_token_ids_sha256",
            "tokenizer_initialized",
            "tokenizer_revision",
            "tokenizer_json_sha256",
            "sampling",
        },
        label="input identity",
    )
    prior_identity = _mapping(
        _json(SINGLE_REAL_RESULT).get("input_identity"), "prior input identity"
    )
    if (
        input_identity.get("provenance") != "fixed_public_token_sequence"
        or input_identity.get("token_ids_committed_in_result") is not False
        or input_identity.get("token_ids_encoding") != "little_endian_signed_int32"
        or input_identity.get("token_count") != 9
        or input_identity.get("token_ids_sha256") != EXPECTED_INPUT_IDENTITY_SHA256
        or input_identity.get("generated_token_count") != 1
        or input_identity.get("generated_token_ids_sha256")
        != prior_identity.get("generated_token_ids_sha256")
        or input_identity.get("tokenizer_initialized") is not False
        or input_identity.get("tokenizer_revision") != REVISION
        or input_identity.get("tokenizer_json_sha256")
        != "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
        or input_identity.get("sampling")
        != {
            "temperature": 0.0,
            "max_tokens": 1,
            "detokenize": False,
            "seed": 0,
        }
        or "token_ids" in input_identity
        or "prompt_text" in input_identity
    ):
        raise RealActivationMatrixFinalizationError("hashed input identity changed")
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
        raise RealActivationMatrixFinalizationError("GPU identity changed")
    command = _mapping(run.get("command"), "command")
    _require_exact_keys(command, {"argv", "cwd"}, label="command")
    argv = command.get("argv")
    expected_argv = [
        "/home/meyowu/projects/nvfp4-doctor/.venv/bin/python",
        str(ROOT / "scripts" / "run_e004_real_activation_matrix.py"),
        "--model-dir",
        str(ROOT / "models" / "nvidia--Qwen3-8B-NVFP4" / REVISION),
        "--artifact-root",
        str(ROOT / "artifacts" / "E004-qwen3-layer-capture" / "real-activation-matrix"),
        "--output",
        str(DEFAULT_RUN),
        "--profile-capture",
    ]
    if argv != expected_argv or command.get("cwd") != str(ROOT):
        raise RealActivationMatrixFinalizationError("runtime command changed")


def _validate_matrix_header(run: dict[str, Any]) -> None:
    matrix = _mapping(run.get("matrix"), "matrix")
    _require_exact_keys(
        matrix,
        {
            "layers",
            "layer_roles",
            "projections",
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
    expected_ids = [case.case_id for case in E004_UNFUSED_REAL_ACTIVATION_CASES]
    expected_event_order = [
        f"{case.case_id}:{role}"
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES
        for role in ("input", "module_output")
    ]
    if (
        matrix.get("layers") != [0, 18, 35]
        or matrix.get("layer_roles") != ["early", "middle", "late"]
        or matrix.get("projections") != ["o_proj", "down_proj"]
        or matrix.get("case_ids") != expected_ids
        or matrix.get("case_count") != 6
        or matrix.get("repetitions_per_case") != 3
        or matrix.get("hook_count") != 12
        or matrix.get("hook_event_order") != expected_event_order
        or matrix.get("distinct_input_sha256_count") != 6
        or matrix.get("distinct_module_output_sha256_count") != 6
    ):
        raise RealActivationMatrixFinalizationError("matrix coverage changed")
    dependency = _mapping(run.get("identity_dependency"), "identity dependency")
    _require_exact_keys(
        dependency, {"path", "sha256", "slice"}, label="identity dependency"
    )
    if dependency != {
        "path": REPLAY_MATRIX_RESULT.relative_to(ROOT).as_posix(),
        "sha256": _sha256_path(REPLAY_MATRIX_RESULT),
        "slice": "representative_projection_replay_matrix_v1",
    }:
        raise RealActivationMatrixFinalizationError(
            "runtime identity dependency changed"
        )
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
        raise RealActivationMatrixFinalizationError("pending backend identity changed")


def _validate_case(
    value: object,
    *,
    expected_identity: dict[str, Any],
    case_index: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    case = E004_UNFUSED_REAL_ACTIVATION_CASES[case_index]
    result = _mapping(value, f"matrix case {case_index}")
    _require_exact_keys(
        result,
        {
            "case_id",
            "layer",
            "role",
            "projection",
            "adapter_scope",
            "module_path",
            "tensor_role",
            "phase",
            "event_count",
            "activation_provenance",
            "checkpoint_identity",
            "capture",
            "runtime_projection",
            "replay",
            "backend_range",
        },
        label=f"matrix case {case_index}",
    )
    expected_header = (
        case.case_id,
        case.layer,
        case.role,
        case.projection,
        "production_aligned_unfused",
        case.module_path,
        "module_input",
        "prefill",
        1,
        "real_qwen_prefill",
    )
    observed_header = (
        result.get("case_id"),
        result.get("layer"),
        result.get("role"),
        result.get("projection"),
        result.get("adapter_scope"),
        result.get("module_path"),
        result.get("tensor_role"),
        result.get("phase"),
        result.get("event_count"),
        result.get("activation_provenance"),
    )
    if observed_header != expected_header:
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} case identity changed"
        )
    if result.get("checkpoint_identity") != expected_identity:
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} checkpoint identity changed"
        )
    capture = _mapping(result.get("capture"), f"{case.case_id} capture")
    _require_exact_keys(
        capture,
        {
            "input_artifact",
            "captured_module_output_artifact",
            "metadata_preserved_fields",
            "device_transfer_recorded",
        },
        label=f"{case.case_id} capture",
    )
    if (
        capture.get("metadata_preserved_fields") != PRESERVED_TRANSFER_FIELDS
        or capture.get("device_transfer_recorded") is not True
    ):
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} capture transfer contract changed"
        )
    artifact_prefix = (
        "artifacts/E004-qwen3-layer-capture/real-activation-matrix/"
        f"{case.artifact_slug}"
    )
    input_artifact = _validate_tensor_artifact(
        capture.get("input_artifact"),
        label=f"{case.case_id} input artifact",
        expected_path=f"{artifact_prefix}-input.pt",
    )
    module_output_artifact = _validate_tensor_artifact(
        capture.get("captured_module_output_artifact"),
        label=f"{case.case_id} captured module-output artifact",
        expected_path=f"{artifact_prefix}-captured-module-output.pt",
    )
    _validate_tensor_artifact_schema(
        input_artifact, label=f"{case.case_id} input artifact"
    )
    _validate_tensor_artifact_schema(
        module_output_artifact,
        label=f"{case.case_id} captured module-output artifact",
    )
    input_tensor = _mapping(
        input_artifact.get("tensor"), f"{case.case_id} input tensor"
    )
    module_output_tensor = _mapping(
        module_output_artifact.get("tensor"), f"{case.case_id} module output tensor"
    )
    if (
        input_tensor.get("shape") != list(case.input_shape(9))
        or input_tensor.get("dtype") != "bfloat16"
        or input_tensor.get("stride") != [case.input_width, 1]
        or input_tensor.get("storage_offset") != 0
        or module_output_tensor.get("shape") != list(case.output_shape(9))
        or module_output_tensor.get("dtype") != "bfloat16"
        or module_output_tensor.get("stride") != [case.output_width, 1]
        or module_output_tensor.get("storage_offset") != 0
    ):
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} captured tensor metadata changed"
        )
    runtime = _mapping(
        result.get("runtime_projection"), f"{case.case_id} runtime projection"
    )
    _require_exact_keys(
        runtime,
        {
            "module_path",
            "module_class",
            "quant_method_class",
            "selected_kernel",
            "packed_weight",
            "runtime_weight_scale",
            "weights_padding_cols",
            "input_global_scale",
            "weight_global_scale",
            "alpha",
        },
        label=f"{case.case_id} runtime projection",
    )
    packed_weight = _mapping(
        runtime.get("packed_weight"), f"{case.case_id} packed weight"
    )
    runtime_scale = _mapping(
        runtime.get("runtime_weight_scale"), f"{case.case_id} runtime weight scale"
    )
    packed_shape = list(case.packed_weight_shape)
    scale_shape = list(case.weight_scale_shape)
    expected_packed_metadata = {
        "shape": packed_shape,
        "dtype": "uint8",
        "stride": [packed_shape[1], 1],
        "storage_offset": 0,
        "device": "cuda:0",
        "contiguous": True,
        "numel": packed_shape[0] * packed_shape[1],
        "byte_length": packed_shape[0] * packed_shape[1],
        "sha256": expected_identity["runtime_packed_weight_sha256"],
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }
    expected_scale_metadata = {
        "shape": scale_shape,
        "dtype": "float8_e4m3fn",
        "stride": [scale_shape[1], 1],
        "storage_offset": 0,
        "device": "cuda:0",
        "contiguous": True,
        "numel": scale_shape[0] * scale_shape[1],
        "byte_length": scale_shape[0] * scale_shape[1],
        "sha256": expected_identity["runtime_weight_scale_sha256"],
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }
    expected_scalars = _mapping(
        expected_identity.get("expected_runtime_scalars"),
        f"{case.case_id} expected runtime scalars",
    )
    if (
        runtime.get("module_path") != case.module_path
        or runtime.get("module_class") != "RowParallelLinear"
        or runtime.get("quant_method_class") != "ModelOptNvFp4LinearMethod"
        or runtime.get("selected_kernel") != EXPECTED_KERNEL
        or runtime.get("weights_padding_cols") != 0
        or packed_weight != expected_packed_metadata
        or runtime_scale != expected_scale_metadata
        or runtime.get("input_global_scale")
        != expected_scalars.get("input_global_scale")
        or runtime.get("weight_global_scale")
        != expected_scalars.get("weight_global_scale")
        or runtime.get("alpha") != expected_scalars.get("alpha")
    ):
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} runtime projection changed"
        )
    replay = _mapping(result.get("replay"), f"{case.case_id} replay")
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
        },
        label=f"{case.case_id} replay",
    )
    replay_artifact = _validate_tensor_artifact(
        replay.get("replay_output_artifact"),
        label=f"{case.case_id} replay output artifact",
        expected_path=f"{artifact_prefix}-replay-output.pt",
    )
    _validate_tensor_artifact_schema(
        replay_artifact, label=f"{case.case_id} replay output artifact"
    )
    replay_artifact_tensor = _mapping(
        replay_artifact.get("tensor"), f"{case.case_id} replay artifact tensor"
    )
    reconstructed_activation = _mapping(
        replay.get("reconstructed_activation_metadata"),
        f"{case.case_id} reconstructed activation",
    )
    input_source_metadata = _mapping(
        input_artifact.get("source_metadata"), f"{case.case_id} input source metadata"
    )
    output_hashes = replay.get("output_sha256s")
    logical_matches = replay.get("logical_byte_exact_captured_module_output_matches")
    if not isinstance(output_hashes, list) or not isinstance(logical_matches, list):
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} replay hashes are missing"
        )
    if not all(
        (
            replay.get("warmup_runs") == 1,
            replay.get("repetitions") == 3,
            replay.get("synchronized") is True,
            replay.get("all_finite") is True,
            replay.get("output_hash_stable") is True,
            replay.get("output_shape") == list(case.output_shape(9)),
            replay.get("output_dtype") == "bfloat16",
            len(output_hashes) == 3,
            len(set(output_hashes)) == 1,
            logical_matches == [True, True, True],
            replay.get("max_abs_error") == 0.0,
            replay.get("mean_abs_error") == 0.0,
            replay.get("logical_byte_exact_captured_input_match") is True,
            reconstructed_activation == input_source_metadata,
            replay.get("input_sha256") == reconstructed_activation.get("sha256"),
            replay.get("input_sha256") == input_tensor.get("sha256"),
            replay.get("captured_module_output_sha256")
            == module_output_tensor.get("sha256"),
            output_hashes[0] == module_output_tensor.get("sha256"),
            replay_artifact_tensor == module_output_tensor,
            replay_artifact_tensor.get("sha256") == output_hashes[0],
        )
    ):
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} replay invariants changed"
        )
    backend_range = _mapping(
        result.get("backend_range"), f"{case.case_id} backend range"
    )
    if backend_range != {
        "target_nvtx_range": case.target_nvtx_range,
        "target_kernel_ids": [],
        "target_kernel_set_sha256": None,
        "expected_sm120_cutlass_signature_present": False,
        "activation_quantization_signature_present": False,
        "fallback_status": "pending_profiler",
    }:
        raise RealActivationMatrixFinalizationError(
            f"{case.case_id} pending backend range changed"
        )
    return result, (input_artifact, module_output_artifact, replay_artifact)


def _validate_prior_regression(cases: list[dict[str, Any]]) -> None:
    prior = _json(SINGLE_REAL_RESULT)
    prior_input = _mapping(
        _mapping(prior.get("capture"), "prior capture").get("input_artifact"),
        "prior input artifact",
    )
    prior_output_sha = _mapping(prior.get("replay"), "prior replay").get(
        "captured_module_output_sha256"
    )
    first = cases[0]
    first_capture = _mapping(first.get("capture"), "layer-0 o_proj capture")
    first_input = _mapping(first_capture.get("input_artifact"), "matrix input")
    first_replay = _mapping(first.get("replay"), "layer-0 o_proj replay")
    if (
        _mapping(first_input.get("tensor"), "matrix input tensor").get("sha256")
        != _mapping(prior_input.get("tensor"), "prior input tensor").get("sha256")
        or first_replay.get("captured_module_output_sha256") != prior_output_sha
    ):
        raise RealActivationMatrixFinalizationError(
            "layer-0 o_proj does not reproduce the prior real-activation result"
        )


def _validate_run(
    run: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    _validate_model_and_input(run)
    _validate_matrix_header(run)
    values = run.get("cases")
    if not isinstance(values, list) or len(values) != 6:
        raise RealActivationMatrixFinalizationError(
            "matrix must contain exactly six cases"
        )
    expected_identities = _expected_identity_by_id()
    cases: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        case_spec = E004_UNFUSED_REAL_ACTIVATION_CASES[index]
        case, case_artifacts = _validate_case(
            value,
            expected_identity=expected_identities[case_spec.case_id],
            case_index=index,
        )
        cases.append(case)
        artifacts.extend(case_artifacts)
    artifact_paths = [str(artifact["path"]) for artifact in artifacts]
    if len(set(artifact_paths)) != 18:
        raise RealActivationMatrixFinalizationError(
            "matrix tensor artifact paths must be unique"
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
        raise RealActivationMatrixFinalizationError(
            "matrix cases do not have distinct input and output identities"
        )
    _validate_prior_regression(cases)
    return cases, tuple(artifacts)


def _validate_local_tensor_artifacts(
    artifacts: tuple[dict[str, Any], ...], *, root: Path = ROOT
) -> None:
    for artifact in artifacts:
        path = root / str(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != artifact["file_bytes"]
            or _sha256_path(path) != artifact["file_sha256"]
        ):
            raise RealActivationMatrixFinalizationError(
                f"local artifact identity changed: {artifact['path']}"
            )
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise RealActivationMatrixFinalizationError(
                f"could not load local tensor artifact: {artifact['path']}"
            ) from error
        if not isinstance(loaded, dict) or set(loaded) != {
            "tensor",
            "source_metadata",
            "destination_metadata",
        }:
            raise RealActivationMatrixFinalizationError(
                f"local tensor artifact schema changed: {artifact['path']}"
            )
        tensor = loaded.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            raise RealActivationMatrixFinalizationError(
                f"local tensor artifact has no tensor: {artifact['path']}"
            )
        if (
            loaded.get("source_metadata") != artifact["source_metadata"]
            or loaded.get("destination_metadata") != artifact["tensor"]
            or _tensor_metadata(tensor) != artifact["tensor"]
        ):
            raise RealActivationMatrixFinalizationError(
                f"local tensor bytes or metadata changed: {artifact['path']}"
            )


def _profile_backend(
    observed_kernels: tuple[str, ...], report_sha256: str
) -> tuple[dict[str, object], dict[str, dict[str, object]], bool]:
    target_kernels_by_case = {
        case.case_id: list(
            kernels_in_nvtx_range(observed_kernels, case.target_nvtx_range)
        )
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES
    }
    catalog_names = list(
        dict.fromkeys(
            kernel
            for case in E004_UNFUSED_REAL_ACTIVATION_CASES
            for kernel in target_kernels_by_case[case.case_id]
        )
    )
    kernel_catalog = [
        {"kernel_id": hashlib.sha256(name.encode("utf-8")).hexdigest(), "name": name}
        for name in catalog_names
    ]
    kernel_id_by_name = {
        str(entry["name"]): str(entry["kernel_id"]) for entry in kernel_catalog
    }
    ranges: dict[str, dict[str, object]] = {}
    passed = True
    for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
        target_kernels = target_kernels_by_case[case.case_id]
        signature_present = expected_sm120_cutlass_present(
            observed_kernels, case.target_nvtx_range
        )
        quantization_present = any(
            "vllm::cvt_fp16_to_fp4" in kernel for kernel in target_kernels
        )
        fallback_status = assess_range_fallback(
            observed_kernels, case.target_nvtx_range
        ).value
        case_passed = bool(
            target_kernels
            and signature_present
            and quantization_present
            and fallback_status == "not_detected"
        )
        passed = passed and case_passed
        ranges[case.case_id] = {
            "target_nvtx_range": case.target_nvtx_range,
            "target_kernel_ids": [kernel_id_by_name[name] for name in target_kernels],
            "target_kernel_set_sha256": _sha256_items(target_kernels),
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
        "kernel_catalog": kernel_catalog,
    }
    return backend, ranges, passed


def _dependency_artifacts() -> list[dict[str, str]]:
    dependencies: list[dict[str, str]] = []
    for kind, path in DEPENDENCY_PATHS:
        if not path.is_file():
            raise RealActivationMatrixFinalizationError(
                f"required dependency is missing: {path}"
            )
        dependencies.append(
            {
                "kind": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(path),
            }
        )
    expected_slices = {
        FULL_MODEL_RESULT: "full_model_snapshot_acquisition_v1",
        REPLAY_MATRIX_RESULT: "representative_projection_replay_matrix_v1",
        SINGLE_REAL_RESULT: "real_activation_capture_replay_v1",
    }
    for result_path, manifest_path in DEPENDENCY_PAIRS:
        result = _json(result_path)
        manifest = _json(manifest_path)
        expected_slice = expected_slices[result_path]
        manifest_git = _mapping(manifest.get("git"), "dependency manifest git")
        if (
            result.get("schema_version") != 1
            or result.get("experiment_id") != "E004-qwen3-layer-capture"
            or result.get("slice") != expected_slice
            or result.get("status") != "pass"
            or result.get("decision") != "continue"
            or manifest.get("schema_version") != 1
            or manifest.get("experiment_id") != "E004-qwen3-layer-capture"
            or manifest.get("slice") != expected_slice
            or manifest_git.get("dirty") is not False
        ):
            raise RealActivationMatrixFinalizationError(
                f"dependency semantic identity changed: {result_path}"
            )
        if result_path == FULL_MODEL_RESULT:
            repository = _mapping(result.get("repository"), "full-model repository")
            if (
                repository.get("id") != "nvidia/Qwen3-8B-NVFP4"
                or repository.get("requested_revision") != REVISION
                or repository.get("resolved_sha") != REVISION
            ):
                raise RealActivationMatrixFinalizationError(
                    "full-model dependency repository changed"
                )
        elif result_path == REPLAY_MATRIX_RESULT:
            model = _mapping(manifest.get("model"), "replay-matrix model")
            if (
                model.get("repository") != "nvidia/Qwen3-8B-NVFP4"
                or model.get("revision") != REVISION
                or model.get("replayed_case_count") != 15
            ):
                raise RealActivationMatrixFinalizationError(
                    "replay-matrix dependency model changed"
                )
        else:
            repository = _mapping(result.get("repository"), "single-real repository")
            if repository != {
                "id": "nvidia/Qwen3-8B-NVFP4",
                "revision": REVISION,
            }:
                raise RealActivationMatrixFinalizationError(
                    "single-real dependency repository changed"
                )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise RealActivationMatrixFinalizationError(
                f"dependency manifest has no artifacts: {manifest_path}"
            )
        expected_path = result_path.relative_to(ROOT).as_posix()
        expected_sha256 = _sha256_path(result_path)
        matching = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
            and artifact.get("path") == expected_path
            and artifact.get("sha256") == expected_sha256
        ]
        if len(matching) != 1:
            raise RealActivationMatrixFinalizationError(
                f"dependency manifest does not bind its result: {manifest_path}"
            )
    return dependencies


def main() -> int:
    args = parse_args()
    run = _json(args.run_evidence)
    cases, tensor_artifacts = _validate_run(run)
    _validate_local_tensor_artifacts(tensor_artifacts)
    if not args.report.is_file():
        raise RealActivationMatrixFinalizationError("Nsight report is missing")
    dependencies = _dependency_artifacts()

    git = collect_git()
    if git.dirty:
        raise RealActivationMatrixFinalizationError(
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
    finalized_cases: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        finalized_cases.append({**case, "backend_range": backend_ranges[case_id]})
    observed_target_range_count = sum(
        bool(evidence["target_kernel_ids"]) for evidence in backend_ranges.values()
    )
    if passed:
        claim_boundary = (
            "For one fixed token-ID request in the pinned environment, this "
            "result establishes metadata-preserving capture and deterministic "
            "same-module replay for six unfused o_proj and down_proj cases "
            "spanning layers 0, 18, and 35, with range-scoped SM120 CUTLASS "
            "NVFP4 identity. It does not establish NVFP4 numerical correctness, "
            "prompt diversity, fused qkv_proj or gate_up_proj coverage, "
            "final-logit or model quality, cross-backend agreement, equivalence "
            "to a high-precision checkpoint, or completion of Gate 2."
        )
    else:
        claim_boundary = (
            "The runtime produced six metadata-preserving captures and stable "
            "same-module replays for the fixed request, but the six-case "
            "range-scoped backend criterion was not met. This result is "
            "inconclusive and establishes no complete matrix backend identity, "
            "NVFP4 numerical correctness, model quality, or Gate 2 completion."
        )
    results = {
        **run,
        "slice": "representative_unfused_real_activation_replay_matrix_v1",
        "status": "pass" if passed else "inconclusive",
        "decision": "continue" if passed else "repeat",
        "matrix": {
            **_mapping(run["matrix"], "matrix"),
            "layer_00_o_proj_regression_match": True,
            "evaluated_range_count": len(backend_ranges),
            "observed_target_range_count": observed_target_range_count,
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
        "--output=.local/profiles/e004-real-activation-unfused-matrix",
        "--force-overwrite=true",
        "/home/meyowu/projects/nvfp4-doctor/.venv/bin/python",
        "-m",
        "scripts.run_e004_real_activation_matrix",
        "--model-dir",
        f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}",
        "--artifact-root",
        "artifacts/E004-qwen3-layer-capture/real-activation-matrix",
        "--output",
        (
            "artifacts/E004-qwen3-layer-capture/real-activation-matrix/"
            "unfused-matrix.json"
        ),
        "--profile-capture",
    ]
    results_text = json.dumps(results, indent=2) + "\n"
    result_artifact = {
        "kind": "normalized-real-activation-unfused-matrix-result",
        "path": args.results.absolute().relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(results_text.encode("utf-8")).hexdigest(),
        "ignored": False,
    }
    local_tensor_entries: list[dict[str, object]] = []
    for case_spec, artifacts in zip(
        E004_UNFUSED_REAL_ACTIVATION_CASES,
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
                    kind=f"{case_spec.case_id}-{suffix}",
                    path=ROOT / str(artifact["path"]),
                    ignored=True,
                    sha256=str(artifact["file_sha256"]),
                )
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_unfused_real_activation_replay_matrix_v1",
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
            "projections": ["o_proj", "down_proj"],
            "adapter_scope": "production_aligned_unfused",
            "activation_provenance": "real_qwen_prefill",
        },
        "dependencies": dependencies,
        "commands": {
            "workflow": [
                "bash",
                "scripts/run_e004_real_activation_matrix_profile.sh",
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
                kind="raw-real-activation-unfused-matrix-observation",
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
    results_temporary = args.results.with_name(f"{args.results.name}.tmp")
    manifest_temporary = args.manifest.with_name(f"{args.manifest.name}.tmp")
    try:
        results_temporary.write_text(results_text, encoding="utf-8")
        manifest_temporary.write_text(manifest_text, encoding="utf-8")
        results_temporary.replace(args.results)
        manifest_temporary.replace(args.manifest)
    finally:
        results_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    print(f"case_count={len(finalized_cases)}")
    print(f"evaluated_range_count={len(backend_ranges)}")
    print(f"observed_target_range_count={observed_target_range_count}")
    print(f"profiler_sha256={evidence.report_sha256}")
    print(f"status={results['status']}")
    print(f"decision={results['decision']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
