#!/usr/bin/env python3
"""Capture and replay one real Qwen3 layer-0 o_proj prefill activation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
REPO_ID = "nvidia/Qwen3-8B-NVFP4"
MODEL_RELATIVE = Path("models") / "nvidia--Qwen3-8B-NVFP4" / REVISION
DEFAULT_MODEL_DIR = ROOT / MODEL_RELATIVE
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "E004-qwen3-layer-capture" / "real-activation"
)
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "layer-00-o-proj.json"
TARGET_MODULE_PATH = "model.layers.0.self_attn.o_proj"
TARGET_RANGE = "e004:real_activation:layer_00:o_proj:nvfp4_gemm"
EXPECTED_KERNEL = "FlashInferCutlassNvFp4LinearKernel"
EXPECTED_WEIGHT_SHA256 = (
    "1db669cf9be8e913653ff5aea1e30d4db2da2e86a2a635c952f8fa8346056f8a"
)
EXPECTED_SCALE_SHA256 = (
    "4e6992cbfa93bd7136816762fbf212861a103944f1d6054cd6d13eac15347be2"
)
PROMPT_TOKEN_IDS = (36326, 11698, 19, 16205, 1273, 25, 470, 10402, 13)
REPETITIONS = 3
FROZEN_VLLM_ENVIRONMENT = {
    name: os.environ[name]
    for name in (
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_WSL2_ENABLE_PIN_MEMORY",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    )
}
PRESERVED_TRANSFER_FIELDS = (
    "shape",
    "dtype",
    "stride",
    "storage_offset",
    "byte_length",
    "sha256",
)

_CAPTURE: dict[str, Any] = {}
_HOOK_HANDLES: list[Any] = []


class RealActivationCaptureError(RuntimeError):
    """Raised when capture or replay evidence violates the frozen contract."""


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_token_bytes(token_ids: tuple[int, ...] | list[int]) -> bytes:
    return b"".join(struct.pack("<i", token_id) for token_id in token_ids)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "numel": tensor.numel(),
        "byte_length": tensor.numel() * tensor.element_size(),
        "sha256": _sha256(_tensor_bytes(tensor)),
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }


def _bitwise_tensor_match(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and _sha256(_tensor_bytes(left)) == _sha256(_tensor_bytes(right))
    )


def _copy_to_cpu_preserving_stride(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.layout != torch.strided:
        raise RealActivationCaptureError(
            "only strided activation tensors are supported"
        )
    if tensor.storage_offset() != 0:
        raise RealActivationCaptureError(
            "non-zero storage offsets require an explicit capture transform"
        )
    torch.cuda.synchronize(tensor.device)
    cpu = torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device="cpu",
    )
    cpu.copy_(tensor.detach())
    torch.cuda.synchronize(tensor.device)
    if (
        cpu.dtype != tensor.dtype
        or tuple(cpu.shape) != tuple(tensor.shape)
        or tuple(cpu.stride()) != tuple(tensor.stride())
        or cpu.storage_offset() != tensor.storage_offset()
    ):
        raise RealActivationCaptureError("CPU capture changed tensor metadata")
    return cpu


def _linear_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RealActivationCaptureError("target module returned an unexpected value")


def _target_module(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    if TARGET_MODULE_PATH in modules:
        return TARGET_MODULE_PATH, modules[TARGET_MODULE_PATH]
    candidates = [
        (name, module)
        for name, module in modules.items()
        if name.endswith(f".{TARGET_MODULE_PATH}")
    ]
    if len(candidates) != 1:
        raise RealActivationCaptureError(
            f"expected one {TARGET_MODULE_PATH} module, found {len(candidates)}"
        )
    return candidates[0]


def _capture_input(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
    if len(args) != 1 or not isinstance(args[0], torch.Tensor):
        raise RealActivationCaptureError("o_proj pre-hook did not receive one tensor")
    if "input" in _CAPTURE:
        raise RealActivationCaptureError("o_proj input hook fired more than once")
    tensor = args[0]
    if tensor.dtype != torch.bfloat16:
        raise RealActivationCaptureError(
            f"expected BF16 activation, got {tensor.dtype}"
        )
    _CAPTURE["input"] = {
        "tensor": _copy_to_cpu_preserving_stride(tensor),
        "source_metadata": _tensor_metadata(tensor),
    }


def _capture_output(
    _module: torch.nn.Module,
    _args: tuple[object, ...],
    output: object,
) -> None:
    if "module_output" in _CAPTURE:
        raise RealActivationCaptureError(
            "o_proj module-output hook fired more than once"
        )
    tensor = _linear_output(output)
    if tensor.dtype != torch.bfloat16:
        raise RealActivationCaptureError(
            f"expected BF16 module output, got {tensor.dtype}"
        )
    _CAPTURE["module_output"] = {
        "tensor": _copy_to_cpu_preserving_stride(tensor),
        "source_metadata": _tensor_metadata(tensor),
    }


def _install_capture_hooks(model: torch.nn.Module) -> dict[str, object]:
    _CAPTURE.clear()
    _HOOK_HANDLES.clear()
    module_path, target = _target_module(model)
    quant_method = getattr(target, "quant_method", None)
    kernel = getattr(quant_method, "kernel", None)
    selected_kernel = type(kernel).__name__ if kernel is not None else None
    weight = getattr(target, "weight", None)
    weight_scale = getattr(target, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(
        weight_scale, torch.Tensor
    ):
        raise RealActivationCaptureError("target NVFP4 tensors are missing")
    weight_sha256 = _sha256(_tensor_bytes(weight))
    scale_sha256 = _sha256(_tensor_bytes(weight_scale))
    if weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise RealActivationCaptureError("full-model packed weight hash changed")
    if scale_sha256 != EXPECTED_SCALE_SHA256:
        raise RealActivationCaptureError("full-model scale swizzle hash changed")
    if selected_kernel != EXPECTED_KERNEL:
        raise RealActivationCaptureError(f"unexpected NVFP4 kernel: {selected_kernel}")
    if getattr(target, "weights_padding_cols", None) != 0:
        raise RealActivationCaptureError("layer-0 o_proj unexpectedly requires padding")
    _HOOK_HANDLES.extend(
        [
            target.register_forward_pre_hook(_capture_input),
            target.register_forward_hook(_capture_output),
        ]
    )
    return {
        "observed_model_class": type(model).__name__,
        "module_path": module_path,
        "module_class": type(target).__name__,
        "quant_method_class": type(quant_method).__name__,
        "selected_kernel": selected_kernel,
        "weight": _tensor_metadata(weight),
        "weight_scale": _tensor_metadata(weight_scale),
        "weights_padding_cols": getattr(target, "weights_padding_cols", None),
        "input_global_scale": float(target.input_global_scale.item()),
        "weight_global_scale": float(target.weight_global_scale.item()),
        "alpha": float(target.alpha.item()),
    }


def _remove_capture_hooks(_model: torch.nn.Module) -> int:
    count = len(_HOOK_HANDLES)
    for handle in _HOOK_HANDLES:
        handle.remove()
    _HOOK_HANDLES.clear()
    return count


def _replay_captured_activation(model: torch.nn.Module) -> dict[str, object]:
    module_path, target = _target_module(model)
    captured_input = _CAPTURE.get("input")
    captured_module_output = _CAPTURE.get("module_output")
    if not isinstance(captured_input, dict) or not isinstance(
        captured_module_output, dict
    ):
        raise RealActivationCaptureError("capture hooks did not produce both tensors")
    input_cpu = captured_input.get("tensor")
    captured_module_output_cpu = captured_module_output.get("tensor")
    if not isinstance(input_cpu, torch.Tensor) or not isinstance(
        captured_module_output_cpu, torch.Tensor
    ):
        raise RealActivationCaptureError("captured tensors are missing")
    activation = torch.empty_strided(
        tuple(input_cpu.shape),
        tuple(input_cpu.stride()),
        dtype=input_cpu.dtype,
        device="cuda",
    )
    activation.copy_(input_cpu)
    torch.cuda.synchronize()

    outputs_gpu: list[torch.Tensor] = []
    with torch.inference_mode():
        _linear_output(target(activation))
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(TARGET_RANGE):
            for _ in range(REPETITIONS):
                output = _linear_output(target(activation))
                torch.cuda.synchronize()
                outputs_gpu.append(output)

    replay_source_metadata = _tensor_metadata(outputs_gpu[0])
    outputs = [_copy_to_cpu_preserving_stride(output) for output in outputs_gpu]
    output_hashes = [_sha256(_tensor_bytes(output)) for output in outputs]
    all_finite = all(bool(torch.isfinite(output).all()) for output in outputs_gpu)

    captured_module_output_hash = _sha256(_tensor_bytes(captured_module_output_cpu))
    bitwise_matches = [
        _bitwise_tensor_match(output, captured_module_output_cpu)
        and output_hash == captured_module_output_hash
        for output, output_hash in zip(outputs, output_hashes, strict=True)
    ]
    if not all_finite or len(set(output_hashes)) != 1:
        raise RealActivationCaptureError("standalone replay is not finite and stable")
    if not all(bitwise_matches):
        raise RealActivationCaptureError(
            "standalone replay differs from the captured module output"
        )
    max_abs_error = max(
        float((output.float() - captured_module_output_cpu.float()).abs().max())
        for output in outputs
    )
    mean_abs_error = max(
        float((output.float() - captured_module_output_cpu.float()).abs().mean())
        for output in outputs
    )
    return {
        "module_path": module_path,
        "warmup_runs": 1,
        "repetitions": REPETITIONS,
        "synchronized": True,
        "all_finite": all_finite,
        "output_shape": list(outputs[0].shape),
        "output_dtype": str(outputs[0].dtype).removeprefix("torch."),
        "output_sha256s": output_hashes,
        "output_hash_stable": len(set(output_hashes)) == 1,
        "captured_module_output_sha256": captured_module_output_hash,
        "bitwise_captured_module_output_matches": bitwise_matches,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "output_tensor": outputs[0],
        "output_source_metadata": replay_source_metadata,
    }


def _save_tensor_artifact(
    path: Path,
    tensor: torch.Tensor,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination_metadata = _tensor_metadata(tensor)
    for field in PRESERVED_TRANSFER_FIELDS:
        if source_metadata.get(field) != destination_metadata.get(field):
            raise RealActivationCaptureError(
                f"{field} changed during device transfer for {path}"
            )
    torch.save(
        {
            "tensor": tensor,
            "source_metadata": source_metadata,
            "destination_metadata": destination_metadata,
        },
        path,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    loaded_tensor = loaded.get("tensor") if isinstance(loaded, dict) else None
    if not isinstance(loaded_tensor, torch.Tensor):
        raise RealActivationCaptureError(f"could not reload {path}")
    if _tensor_metadata(loaded_tensor) != destination_metadata:
        raise RealActivationCaptureError(f"reloaded metadata changed for {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "ignored": True,
        "encoding": "torch_save_cpu_tensor_v1",
        "file_bytes": path.stat().st_size,
        "file_sha256": _sha256_path(path),
        "tensor": destination_metadata,
        "source_metadata": source_metadata,
        "preserved_fields": list(PRESERVED_TRANSFER_FIELDS),
        "device_transfer": {
            "source": source_metadata.get("device"),
            "destination": destination_metadata.get("device"),
        },
    }


def _project_relative(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(ROOT)
    except ValueError as error:
        raise RealActivationCaptureError(
            f"{label} must remain under the repository root"
        ) from error
    if not relative.parts or relative.parts[0] not in {"artifacts", ".local"}:
        raise RealActivationCaptureError(
            f"{label} must remain under artifacts/ or .local/"
        )
    return relative


def _cuda_status_code(result: object) -> int:
    if result is None:
        return 0
    if isinstance(result, tuple):
        if not result:
            return 0
        result = result[0]
    value = getattr(result, "value", result)
    return int(value)


def _require_cuda_success(operation: str, result: object) -> None:
    status = _cuda_status_code(result)
    if status != 0:
        raise RealActivationCaptureError(
            f"{operation} failed with CUDA runtime status {status}"
        )


def _capture_record(name: str) -> dict[str, Any]:
    value = _CAPTURE.get(name)
    if not isinstance(value, dict):
        raise RealActivationCaptureError(f"missing {name} capture")
    tensor = value.get("tensor")
    metadata = value.get("source_metadata")
    if not isinstance(tensor, torch.Tensor) or not isinstance(metadata, dict):
        raise RealActivationCaptureError(f"invalid {name} capture")
    return value


def main() -> int:
    args = parse_args()
    if args.max_model_len < len(PROMPT_TOKEN_IDS) + 1:
        raise ValueError("max-model-len cannot hold the frozen request")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    if args.kv_cache_memory_bytes <= 0:
        raise ValueError("kv-cache-memory-bytes must be positive")
    _project_relative(args.artifact_root, label="artifact root")
    _project_relative(args.output, label="output")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("E004 is pinned to the RTX 5080 sm_120 environment")

    model_dir = args.model_dir.absolute()
    try:
        model_relative = model_dir.relative_to(ROOT)
    except ValueError as error:
        raise RealActivationCaptureError(
            "model directory must remain under the repository root"
        ) from error
    if not model_relative.parts or model_relative.parts[0] != "models":
        raise RealActivationCaptureError("model directory must remain under models/")
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"pinned model snapshot is missing: {model_dir}")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.utils.flashinfer import has_flashinfer

    if not has_flashinfer():
        raise RealActivationCaptureError(
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
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        **requested_args,
    )
    free_after_load, _ = torch.cuda.mem_get_info()
    install_results = llm.apply_model(_install_capture_hooks)
    if len(install_results) != 1 or not isinstance(install_results[0], dict):
        raise RealActivationCaptureError("hook installation returned no metadata")
    target = install_results[0]

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
    generated_ids = list(outputs[0].outputs[0].token_ids)
    if len(generated_ids) != 1:
        raise RealActivationCaptureError(
            f"expected one generated token, observed {len(generated_ids)}"
        )
    input_capture = _capture_record("input")
    module_output_capture = _capture_record("module_output")
    removed = llm.apply_model(_remove_capture_hooks)
    if removed != [2]:
        raise RealActivationCaptureError(f"unexpected removed-hook count: {removed}")

    profiler_started = False
    if args.profile_capture:
        _require_cuda_success(
            "cudaProfilerStart", torch.cuda.cudart().cudaProfilerStart()
        )
        profiler_started = True
    try:
        replay_results = llm.apply_model(_replay_captured_activation)
    finally:
        if profiler_started:
            _require_cuda_success(
                "cudaProfilerStop", torch.cuda.cudart().cudaProfilerStop()
            )
    if len(replay_results) != 1 or not isinstance(replay_results[0], dict):
        raise RealActivationCaptureError("standalone replay returned no evidence")
    replay = replay_results[0]
    replay_tensor = replay.pop("output_tensor", None)
    replay_output_source_metadata = replay.pop("output_source_metadata", None)
    if not isinstance(replay_tensor, torch.Tensor):
        raise RealActivationCaptureError("standalone replay output is missing")
    if not isinstance(replay_output_source_metadata, dict):
        raise RealActivationCaptureError(
            "standalone replay output source metadata is missing"
        )

    input_tensor = input_capture["tensor"]
    captured_module_output_tensor = module_output_capture["tensor"]
    if not isinstance(input_tensor, torch.Tensor) or not isinstance(
        captured_module_output_tensor, torch.Tensor
    ):
        raise RealActivationCaptureError("captured artifacts are not tensors")
    if input_tensor.shape != (len(PROMPT_TOKEN_IDS), 4096):
        raise RealActivationCaptureError(
            f"unexpected captured activation shape: {tuple(input_tensor.shape)}"
        )
    input_artifact = _save_tensor_artifact(
        args.artifact_root / "layer-00-o-proj-input.pt",
        input_tensor,
        input_capture["source_metadata"],
    )
    captured_module_output_artifact = _save_tensor_artifact(
        args.artifact_root / "layer-00-o-proj-captured-module-output.pt",
        captured_module_output_tensor,
        module_output_capture["source_metadata"],
    )
    replay_output_artifact = _save_tensor_artifact(
        args.artifact_root / "layer-00-o-proj-replay-output.pt",
        replay_tensor,
        replay_output_source_metadata,
    )

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    captured_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    result = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "real_activation_capture_replay_observation_v1",
        "captured_at_utc": captured_at,
        "status": "pass",
        "decision": "pending_profiler",
        "repository": {"id": REPO_ID, "revision": REVISION},
        "model_load": {
            "local_snapshot_path": model_relative.as_posix(),
            "frozen_environment": FROZEN_VLLM_ENVIRONMENT,
            "requested_args": requested_args,
            "observed_model_class": target["observed_model_class"],
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
            "tokenizer_json_sha256": (
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
            ),
            "sampling": {
                "temperature": 0.0,
                "max_tokens": 1,
                "detokenize": False,
                "seed": 0,
            },
        },
        "capture": {
            "case": {
                "layer": 0,
                "projection": "o_proj",
                "adapter_scope": "production_aligned_unfused",
                "module_path": target["module_path"],
                "module_class": target["module_class"],
                "tensor_role": "module_input",
                "phase": "prefill",
                "event_count": 1,
                "activation_provenance": "real_qwen_prefill",
            },
            "input_artifact": input_artifact,
            "captured_module_output_artifact": captured_module_output_artifact,
            "metadata_preserved_fields": list(PRESERVED_TRANSFER_FIELDS),
            "device_transfer_recorded": True,
        },
        "runtime_projection": target,
        "replay": {
            **replay,
            "input_sha256": input_artifact["tensor"]["sha256"],
            "replay_output_artifact": replay_output_artifact,
        },
        "backend": {
            "requested_format": "nvfp4",
            "requested_backend": "auto",
            "selected_vllm_kernel": target["selected_kernel"],
            "reported_backend": None,
            "target_nvtx_range": TARGET_RANGE,
            "observed_kernels": [],
            "fallback_status": "pending_profiler",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": total_memory,
        },
        "command": {"argv": [sys.executable, *sys.argv], "cwd": str(ROOT)},
        "claim_boundary": (
            "For one fixed token-ID request, this observation establishes capture "
            "of one real Qwen prefill activation with shape, dtype, stride, storage "
            "offset, and canonical logical bytes preserved across the recorded "
            "CUDA-to-CPU transfer, plus deterministic standalone replay of the "
            "same unfused layer-0 o_proj. It does not establish NVFP4 numerical "
            "correctness, final-logit or model quality, other prompts, layers, or "
            "projections, cross-backend agreement, or equivalence to a "
            "high-precision checkpoint."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"captured_activation_sha256={input_artifact['tensor']['sha256']}")
    print(f"output_sha256={replay['output_sha256s'][0]}")
    print(f"peak_allocated_bytes={peak_allocated}")
    print(f"selected_vllm_kernel={target['selected_kernel']}")
    print("status=pass")
    print("decision=pending_profiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
