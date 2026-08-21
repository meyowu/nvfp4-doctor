#!/usr/bin/env python3
"""Capture and replay six production-aligned Qwen3 NVFP4 activations."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from nvfp4_doctor.capture import (
    E004_UNFUSED_REAL_ACTIVATION_CASES,
    E004RealActivationCase,
)
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
REPLAY_MATRIX_RESULT = EXPERIMENT / "replay-matrix.json"
DEFAULT_MODEL_DIR = ROOT / MODEL_RELATIVE
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "E004-qwen3-layer-capture" / "real-activation-matrix"
)
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "unfused-matrix.json"
REPETITIONS = 3
TOKENIZER_JSON_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)

_CAPTURES: dict[str, dict[str, Any]] = {}
_HOOK_HANDLES: list[Any] = []
_CAPTURE_EVENT_ORDER: list[str] = []


class RealActivationMatrixError(RuntimeError):
    """Raised when matrix capture or replay violates the frozen contract."""


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
        raise RealActivationMatrixError(f"{label} must be an object")
    return value


def _load_expected_runtime_identities() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(REPLAY_MATRIX_RESULT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RealActivationMatrixError(
            "could not read the representative replay dependency"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("slice") != "representative_projection_replay_matrix_v1"
        or payload.get("status") != "pass"
    ):
        raise RealActivationMatrixError("representative replay dependency did not pass")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise RealActivationMatrixError("representative replay dependency has no cases")
    expected_keys = {
        (case.layer, case.projection): case
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES
    }
    identities: dict[str, dict[str, Any]] = {}
    seen: set[tuple[int, str]] = set()
    for value in cases:
        entry = _mapping(value, "representative replay case")
        key = (entry.get("layer"), entry.get("projection"))
        if key not in expected_keys:
            continue
        if key in seen:
            raise RealActivationMatrixError(
                f"duplicate representative replay identity: {key}"
            )
        seen.add(key)
        case = expected_keys[key]
        source_hashes = _mapping(
            entry.get("source_tensor_sha256"),
            f"{case.case_id} source tensor hashes",
        )
        if (
            entry.get("adapter_scope") != "production_aligned_unfused"
            or entry.get("packed_weight_shape") != list(case.packed_weight_shape)
            or entry.get("weight_scale_shape") != list(case.weight_scale_shape)
            or entry.get("weight_padding_bytes") != 0
            or entry.get("selected_vllm_kernel") != EXPECTED_KERNEL
            or not isinstance(source_hashes.get("weight"), str)
            or not isinstance(entry.get("runtime_weight_scale_sha256"), str)
            or not isinstance(entry.get("input_scale"), (int, float))
            or not isinstance(entry.get("weight_scale_2"), (int, float))
            or not isinstance(entry.get("alpha_runtime_f32"), (int, float))
        ):
            raise RealActivationMatrixError(
                f"{case.case_id} representative replay identity changed"
            )
        identities[case.case_id] = {
            "checkpoint_source_tensor_sha256": source_hashes,
            "runtime_packed_weight_sha256": source_hashes["weight"],
            "runtime_weight_scale_sha256": entry["runtime_weight_scale_sha256"],
            "packed_weight_shape": list(case.packed_weight_shape),
            "weight_scale_shape": list(case.weight_scale_shape),
            "expected_runtime_scalars": {
                "input_global_scale": entry["input_scale"],
                "weight_global_scale": entry["weight_scale_2"],
                "alpha": entry["alpha_runtime_f32"],
            },
        }
    if seen != set(expected_keys):
        missing = sorted(set(expected_keys) - seen)
        raise RealActivationMatrixError(
            f"representative replay dependency is missing cases: {missing}"
        )
    return identities


def _target_module(
    model: torch.nn.Module, case: E004RealActivationCase
) -> tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    if case.module_path not in modules:
        raise RealActivationMatrixError(
            f"expected exact raw-model module path {case.module_path}"
        )
    return case.module_path, modules[case.module_path]


def _capture_tensor(case_id: str, role: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise RealActivationMatrixError(
            f"{case_id} expected BF16 {role}, got {tensor.dtype}"
        )
    entry = _CAPTURES.setdefault(case_id, {})
    if role in entry:
        raise RealActivationMatrixError(f"{case_id} {role} hook fired more than once")
    source_metadata = _tensor_metadata(tensor)
    entry[role] = {
        "tensor": _copy_to_cpu_preserving_stride(tensor),
        "source_metadata": source_metadata,
        "event_count": 1,
    }
    _CAPTURE_EVENT_ORDER.append(f"{case_id}:{role}")


def _capture_input(
    case_id: str,
    _module: torch.nn.Module,
    args: tuple[object, ...],
) -> None:
    if len(args) != 1 or not isinstance(args[0], torch.Tensor):
        raise RealActivationMatrixError(
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


def _runtime_projection(
    model: torch.nn.Module,
    case: E004RealActivationCase,
    expected: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, object]]:
    module_path, target = _target_module(model, case)
    quant_method = getattr(target, "quant_method", None)
    kernel = getattr(quant_method, "kernel", None)
    selected_kernel = type(kernel).__name__ if kernel is not None else None
    weight = getattr(target, "weight", None)
    weight_scale = getattr(target, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(
        weight_scale, torch.Tensor
    ):
        raise RealActivationMatrixError(f"{case.case_id} NVFP4 tensors are missing")
    weight_metadata = _tensor_metadata(weight)
    scale_metadata = _tensor_metadata(weight_scale)
    if (
        weight_metadata["shape"] != expected["packed_weight_shape"]
        or weight_metadata["sha256"] != expected["runtime_packed_weight_sha256"]
        or scale_metadata["shape"] != expected["weight_scale_shape"]
        or scale_metadata["sha256"] != expected["runtime_weight_scale_sha256"]
    ):
        raise RealActivationMatrixError(
            f"{case.case_id} runtime tensor identity changed"
        )
    if selected_kernel != EXPECTED_KERNEL:
        raise RealActivationMatrixError(
            f"{case.case_id} selected unexpected kernel: {selected_kernel}"
        )
    if type(target).__name__ != "RowParallelLinear":
        raise RealActivationMatrixError(
            f"{case.case_id} is not a RowParallelLinear module"
        )
    if type(quant_method).__name__ != "ModelOptNvFp4LinearMethod":
        raise RealActivationMatrixError(
            f"{case.case_id} has an unexpected quantization method"
        )
    if getattr(target, "weights_padding_cols", None) != 0:
        raise RealActivationMatrixError(
            f"{case.case_id} unexpectedly requires weight padding"
        )
    runtime_scalars = {
        "input_global_scale": float(target.input_global_scale.item()),
        "weight_global_scale": float(target.weight_global_scale.item()),
        "alpha": float(target.alpha.item()),
    }
    if runtime_scalars != expected["expected_runtime_scalars"]:
        raise RealActivationMatrixError(f"{case.case_id} runtime scale scalars changed")
    return target, {
        "module_path": module_path,
        "module_class": type(target).__name__,
        "quant_method_class": type(quant_method).__name__,
        "selected_kernel": selected_kernel,
        "packed_weight": weight_metadata,
        "runtime_weight_scale": scale_metadata,
        "weights_padding_cols": getattr(target, "weights_padding_cols", None),
        **runtime_scalars,
    }


def _install_capture_hooks(model: torch.nn.Module) -> dict[str, object]:
    _CAPTURES.clear()
    _HOOK_HANDLES.clear()
    _CAPTURE_EVENT_ORDER.clear()
    expected_identities = _load_expected_runtime_identities()
    targets: list[dict[str, object]] = []
    for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
        target, metadata = _runtime_projection(
            model, case, expected_identities[case.case_id]
        )
        _HOOK_HANDLES.extend(
            [
                target.register_forward_pre_hook(
                    functools.partial(_capture_input, case.case_id)
                ),
                target.register_forward_hook(
                    functools.partial(_capture_output, case.case_id)
                ),
            ]
        )
        targets.append(
            {
                "case_id": case.case_id,
                "checkpoint_identity": expected_identities[case.case_id],
                **metadata,
            }
        )
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
        raise RealActivationMatrixError(f"missing {case_id} capture")
    value = entry.get(role)
    if not isinstance(value, dict) or value.get("event_count") != 1:
        raise RealActivationMatrixError(f"invalid {case_id} {role} capture")
    tensor = value.get("tensor")
    metadata = value.get("source_metadata")
    if not isinstance(tensor, torch.Tensor) or not isinstance(metadata, dict):
        raise RealActivationMatrixError(f"invalid {case_id} {role} tensor")
    return value


def _replay_case(
    model: torch.nn.Module, case: E004RealActivationCase
) -> dict[str, object]:
    _module_path, target = _target_module(model, case)
    input_capture = _capture_record(case.case_id, "input")
    output_capture = _capture_record(case.case_id, "module_output")
    input_cpu = input_capture["tensor"]
    captured_output_cpu = output_capture["tensor"]
    if not isinstance(input_cpu, torch.Tensor) or not isinstance(
        captured_output_cpu, torch.Tensor
    ):
        raise RealActivationMatrixError(f"{case.case_id} captured tensors are missing")
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
        raise RealActivationMatrixError(
            f"{case.case_id} reconstructed CUDA activation changed metadata or bytes"
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
    if not all(finite_flags) or len(set(output_hashes)) != 1:
        raise RealActivationMatrixError(
            f"{case.case_id} replay is not finite and stable"
        )
    if not all(logical_matches):
        raise RealActivationMatrixError(
            f"{case.case_id} replay differs from its captured module output"
        )
    max_abs_error = max(
        float((output.float() - captured_output_cpu.float()).abs().max())
        for output in outputs
    )
    mean_abs_error = max(
        float((output.float() - captured_output_cpu.float()).abs().mean())
        for output in outputs
    )
    return {
        "warmup_runs": 1,
        "repetitions": REPETITIONS,
        "synchronized": True,
        "all_finite": all(finite_flags),
        "output_shape": list(outputs[0].shape),
        "output_dtype": str(outputs[0].dtype).removeprefix("torch."),
        "output_sha256s": output_hashes,
        "output_hash_stable": len(set(output_hashes)) == 1,
        "captured_module_output_sha256": captured_output_hash,
        "logical_byte_exact_captured_module_output_matches": logical_matches,
        "reconstructed_activation_metadata": activation_metadata,
        "logical_byte_exact_captured_input_match": True,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "output_tensor": outputs[0],
        "output_source_metadata": output_source_metadata,
    }


def _replay_matrix(model: torch.nn.Module) -> list[dict[str, object]]:
    return [
        {"case_id": case.case_id, **_replay_case(model, case)}
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES
    ]


def _require_cuda_success(operation: str, result: object) -> None:
    status = _cuda_status_code(result)
    if status != 0:
        raise RealActivationMatrixError(
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
        raise RealActivationMatrixError(
            "model directory must remain under the repository root"
        ) from error
    if not model_relative.parts or model_relative.parts[0] != "models":
        raise RealActivationMatrixError("model directory must remain under models/")
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"pinned model snapshot is missing: {model_dir}")
    return model_dir, model_relative


def main() -> int:
    args = parse_args()
    model_dir, model_relative = _validate_args(args)
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
        raise RealActivationMatrixError(
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
    install_results = llm.apply_model(_install_capture_hooks)
    if len(install_results) != 1 or not isinstance(install_results[0], dict):
        raise RealActivationMatrixError("hook installation returned no metadata")
    installation = install_results[0]
    if installation.get("installed_hook_count") != 12:
        raise RealActivationMatrixError("the matrix did not install exactly 12 hooks")

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
        raise RealActivationMatrixError(f"unexpected removed-hook count: {removed}")
    generated_ids = list(outputs[0].outputs[0].token_ids)
    if len(generated_ids) != 1:
        raise RealActivationMatrixError(
            f"expected one generated token, observed {len(generated_ids)}"
        )
    for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
        _capture_record(case.case_id, "input")
        _capture_record(case.case_id, "module_output")
    expected_event_order = [
        f"{case.case_id}:{role}"
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES
        for role in ("input", "module_output")
    ]
    if _CAPTURE_EVENT_ORDER != expected_event_order:
        raise RealActivationMatrixError(
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
        raise RealActivationMatrixError("matrix replay returned no evidence")
    replay_by_id = {str(entry["case_id"]): entry for entry in replay_results[0]}
    if set(replay_by_id) != {
        case.case_id for case in E004_UNFUSED_REAL_ACTIVATION_CASES
    }:
        raise RealActivationMatrixError("matrix replay returned the wrong case set")
    targets = installation.get("targets")
    if not isinstance(targets, list):
        raise RealActivationMatrixError("runtime target metadata is missing")
    target_by_id = {str(entry["case_id"]): entry for entry in targets}
    if set(target_by_id) != set(replay_by_id):
        raise RealActivationMatrixError("runtime target metadata has the wrong cases")

    case_results: list[dict[str, object]] = []
    for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
        input_capture = _capture_record(case.case_id, "input")
        module_output_capture = _capture_record(case.case_id, "module_output")
        input_tensor = input_capture["tensor"]
        module_output_tensor = module_output_capture["tensor"]
        if not isinstance(input_tensor, torch.Tensor) or not isinstance(
            module_output_tensor, torch.Tensor
        ):
            raise RealActivationMatrixError(
                f"{case.case_id} captured artifacts are not tensors"
            )
        if tuple(input_tensor.shape) != case.input_shape(len(PROMPT_TOKEN_IDS)):
            raise RealActivationMatrixError(
                f"unexpected {case.case_id} input shape: {tuple(input_tensor.shape)}"
            )
        if tuple(module_output_tensor.shape) != case.output_shape(
            len(PROMPT_TOKEN_IDS)
        ):
            raise RealActivationMatrixError(
                f"unexpected {case.case_id} output shape: "
                f"{tuple(module_output_tensor.shape)}"
            )
        replay = replay_by_id[case.case_id]
        replay_tensor = replay.pop("output_tensor", None)
        replay_source_metadata = replay.pop("output_source_metadata", None)
        replay.pop("case_id", None)
        if not isinstance(replay_tensor, torch.Tensor) or not isinstance(
            replay_source_metadata, dict
        ):
            raise RealActivationMatrixError(
                f"{case.case_id} replay output metadata is missing"
            )
        slug = case.artifact_slug
        input_artifact = _save_tensor_artifact(
            args.artifact_root / f"{slug}-input.pt",
            input_tensor,
            input_capture["source_metadata"],
        )
        module_output_artifact = _save_tensor_artifact(
            args.artifact_root / f"{slug}-captured-module-output.pt",
            module_output_tensor,
            module_output_capture["source_metadata"],
        )
        replay_output_artifact = _save_tensor_artifact(
            args.artifact_root / f"{slug}-replay-output.pt",
            replay_tensor,
            replay_source_metadata,
        )
        runtime = dict(target_by_id[case.case_id])
        runtime.pop("case_id", None)
        checkpoint_identity = runtime.pop("checkpoint_identity")
        case_results.append(
            {
                "case_id": case.case_id,
                "layer": case.layer,
                "role": case.role,
                "projection": case.projection,
                "adapter_scope": "production_aligned_unfused",
                "module_path": case.module_path,
                "tensor_role": "module_input",
                "phase": "prefill",
                "event_count": 1,
                "activation_provenance": "real_qwen_prefill",
                "checkpoint_identity": checkpoint_identity,
                "capture": {
                    "input_artifact": input_artifact,
                    "captured_module_output_artifact": module_output_artifact,
                    "metadata_preserved_fields": list(PRESERVED_TRANSFER_FIELDS),
                    "device_transfer_recorded": True,
                },
                "runtime_projection": runtime,
                "replay": {
                    **replay,
                    "input_sha256": replay["reconstructed_activation_metadata"][
                        "sha256"
                    ],
                    "replay_output_artifact": replay_output_artifact,
                },
                "backend_range": {
                    "target_nvtx_range": case.target_nvtx_range,
                    "target_kernel_ids": [],
                    "target_kernel_set_sha256": None,
                    "expected_sm120_cutlass_signature_present": False,
                    "activation_quantization_signature_present": False,
                    "fallback_status": "pending_profiler",
                },
            }
        )

    input_hashes = [
        str(case["capture"]["input_artifact"]["tensor"]["sha256"])
        for case in case_results
    ]
    output_hashes = [
        str(case["replay"]["captured_module_output_sha256"]) for case in case_results
    ]
    if len(set(input_hashes)) != 6 or len(set(output_hashes)) != 6:
        raise RealActivationMatrixError(
            "captured cases do not have six distinct input and output identities"
        )

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    result = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_unfused_real_activation_observation_v1",
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
        "input_identity": {
            "provenance": "fixed_public_token_sequence",
            "token_ids_committed_in_result": False,
            "token_ids_encoding": "little_endian_signed_int32",
            "token_count": len(PROMPT_TOKEN_IDS),
            "token_ids_sha256": _sha256(_canonical_token_bytes(PROMPT_TOKEN_IDS)),
            "generated_token_count": len(generated_ids),
            "generated_token_ids_sha256": _sha256(
                _canonical_token_bytes(generated_ids)
            ),
            "tokenizer_initialized": False,
            "tokenizer_revision": REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "sampling": {
                "temperature": 0.0,
                "max_tokens": 1,
                "detokenize": False,
                "seed": 0,
            },
        },
        "matrix": {
            "layers": [0, 18, 35],
            "layer_roles": ["early", "middle", "late"],
            "projections": ["o_proj", "down_proj"],
            "case_ids": [case.case_id for case in E004_UNFUSED_REAL_ACTIVATION_CASES],
            "case_count": 6,
            "repetitions_per_case": REPETITIONS,
            "hook_count": 12,
            "hook_event_order": list(_CAPTURE_EVENT_ORDER),
            "distinct_input_sha256_count": len(set(input_hashes)),
            "distinct_module_output_sha256_count": len(set(output_hashes)),
        },
        "identity_dependency": {
            "path": REPLAY_MATRIX_RESULT.relative_to(ROOT).as_posix(),
            "sha256": _sha256_path(REPLAY_MATRIX_RESULT),
            "slice": "representative_projection_replay_matrix_v1",
        },
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
            "For one fixed token-ID request, this observation records six real "
            "Qwen prefill activations and deterministic same-module replays for "
            "the unfused o_proj and down_proj modules at layers 0, 18, and 35. "
            "Profiler identity remains pending. It does not establish NVFP4 "
            "numerical correctness, prompt diversity, fused qkv_proj or "
            "gate_up_proj coverage, final-logit or model quality, cross-backend "
            "agreement, high-precision equivalence, or completion of Gate 2."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"case_count={len(case_results)}")
    print(f"replay_count={len(case_results) * REPETITIONS}")
    print(f"peak_allocated_bytes={peak_allocated}")
    print(f"selected_vllm_kernel={EXPECTED_KERNEL}")
    print("status=pass")
    print("decision=pending_profiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
