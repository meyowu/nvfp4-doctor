#!/usr/bin/env python3
"""Replay one acquired ModelOpt NVFP4 projection through vLLM's selected kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from nvfp4_doctor.checkpoint import load_modelopt_projection

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
PAYLOADS = EXPERIMENT / "payloads.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "E004-qwen3-layer-capture" / "replay" / "layer-00-o-proj.json"
)
REPO_ID = "nvidia/Qwen3-8B-NVFP4"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
EXPECTED_KERNEL = "FlashInferCutlassNvFp4LinearKernel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--projection", default="o_proj")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes(tensor) -> bytes:
    return (
        tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    )


def _runtime_tensor(
    name: str,
    tensor,
    *,
    logical_shape: tuple[int, ...],
    layout: str,
) -> dict[str, object]:
    return {
        "name": name,
        "logical_shape": list(logical_shape),
        "physical_shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
        "layout": layout,
        "sha256": _sha256(_tensor_bytes(tensor)),
    }


def _source_tensor(suffix: str, tensor) -> dict[str, object]:
    return {
        "suffix": suffix,
        "tensor_name": tensor.name,
        "local_path": tensor.local_path,
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "byte_length": tensor.byte_length,
        "sha256": tensor.recorded_sha256,
    }


def main() -> int:
    args = parse_args()
    if args.layer < 0 or args.rows <= 0 or args.seed < 0 or args.repetitions < 3:
        raise ValueError(
            "layer/rows must be positive and at least three runs are required"
        )

    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("E004 is pinned to the RTX 5080 sm_120 environment")

    projection = load_modelopt_projection(
        PAYLOADS,
        ROOT,
        layer=args.layer,
        projection=args.projection,
    )
    cutlass = projection.prepare_cutlass_128x4()

    weight = (
        torch.frombuffer(bytearray(projection.weight.data), dtype=torch.uint8)
        .reshape(projection.weight.shape)
        .to("cuda")
    )
    linear_scale = (
        torch.frombuffer(bytearray(projection.weight_scale.data), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(projection.weight_scale.shape)
        .to("cuda")
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(linear_scale, requires_grad=False)
    layer.output_size_per_partition = projection.rows
    layer.alpha = torch.nn.Parameter(
        torch.tensor(cutlass.alpha, dtype=torch.float32, device="cuda"),
        requires_grad=False,
    )
    layer.input_global_scale_inv = torch.nn.Parameter(
        torch.tensor(
            cutlass.input_global_scale_inv,
            dtype=torch.float32,
            device="cuda",
        ),
        requires_grad=False,
    )

    kernel = init_nvfp4_linear_kernel()
    selected_kernel = type(kernel).__name__
    if selected_kernel != EXPECTED_KERNEL:
        raise RuntimeError(f"unexpected vLLM NVFP4 kernel: {selected_kernel}")
    kernel.process_weights_after_loading(layer)

    runtime_weight = _tensor_bytes(layer.weight)
    runtime_scale = _tensor_bytes(layer.weight_scale)
    if _sha256(runtime_weight) != cutlass.runtime_weight_sha256:
        raise RuntimeError("vLLM changed packed weight bytes unexpectedly")
    if runtime_scale != cutlass.weight_scale_128x4:
        raise RuntimeError("vLLM scale swizzle disagrees with the independent oracle")
    if getattr(layer, "weights_padding_cols", None) != 0:
        raise RuntimeError("representative projection unexpectedly required padding")

    indices = torch.arange(
        args.rows * projection.columns,
        dtype=torch.int32,
        device="cuda",
    ).reshape(args.rows, projection.columns)
    activation = (
        ((indices + args.seed * 17).remainder(257) - 128).to(torch.bfloat16) / 512
    ).contiguous()
    activation_max_abs = float(activation.abs().max())
    calibrated_max_abs = cutlass.input_scale * 6 * 448
    if activation_max_abs > calibrated_max_abs:
        raise RuntimeError(
            "synthetic activation exceeds the checkpoint calibration bound"
        )

    activation_fp4, activation_scale = scaled_fp4_quant(
        activation,
        layer.input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )
    quantized_activation = QuantizedActivation(
        activation_fp4,
        activation_scale,
        activation.dtype,
        activation.shape,
        kernel.input_quant_key(),
    )
    range_name = f"e004:layer_{args.layer:02d}:{args.projection}:nvfp4_gemm"

    output_hashes: list[str] = []
    all_finite = True
    with torch.inference_mode():
        kernel.apply_weights(layer, quantized_activation)
        torch.cuda.synchronize()
        for _ in range(args.repetitions):
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range(range_name):
                output = kernel.apply_weights(layer, quantized_activation)
                torch.cuda.synchronize()
            all_finite = all_finite and bool(torch.isfinite(output).all())
            output_hashes.append(_sha256(_tensor_bytes(output)))

    expected_output_shape = (args.rows, projection.rows)
    if tuple(output.shape) != expected_output_shape or output.dtype != torch.bfloat16:
        raise RuntimeError("projection output metadata is incorrect")
    if not all_finite or len(set(output_hashes)) != 1:
        raise RuntimeError("projection replay is non-finite or non-deterministic")

    runtime_tensors = [
        _runtime_tensor(
            "activation_bf16",
            activation,
            logical_shape=tuple(activation.shape),
            layout="row_major",
        ),
        _runtime_tensor(
            "activation_fp4",
            activation_fp4,
            logical_shape=tuple(activation.shape),
            layout="packed_low_nibble_first",
        ),
        _runtime_tensor(
            "activation_scale",
            activation_scale,
            logical_shape=(args.rows, projection.columns // 16),
            layout="cutlass_128x4",
        ),
        _runtime_tensor(
            "weight",
            layer.weight,
            logical_shape=(projection.rows, projection.columns),
            layout="packed_low_nibble_first_row_major",
        ),
        _runtime_tensor(
            "weight_scale",
            layer.weight_scale,
            logical_shape=projection.weight_scale.shape,
            layout="cutlass_128x4",
        ),
        _runtime_tensor(
            "output",
            output,
            logical_shape=expected_output_shape,
            layout="row_major",
        ),
    ]
    result = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "single_projection_runtime_observation_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "decision": "pending_profiler",
        "repository": {"id": REPO_ID, "revision": REVISION},
        "case": {
            "layer": args.layer,
            "projection": args.projection,
            "selection_rationale": "unfused vLLM projection with all four acquired tensors",
            "activation_provenance": "synthetic_deterministic",
            "activation_recipe": "bf16((((arange + seed*17) % 257) - 128) / 512)",
            "seed": args.seed,
            "logical_mnk": [args.rows, projection.rows, projection.columns],
        },
        "source_tensors": [
            _source_tensor("input_scale", projection.input_scale),
            _source_tensor("weight", projection.weight),
            _source_tensor("weight_scale", projection.weight_scale),
            _source_tensor("weight_scale_2", projection.weight_scale_2),
        ],
        "runtime_tensors": runtime_tensors,
        "transforms": [
            {
                "name": "packed_weight_materialization",
                "operation": "frombuffer + recorded reshape + CUDA copy",
                "source_layout": "safetensors row-major U8, low nibble first",
                "destination_layout": "CUDA row-major U8, low nibble first",
                "source_sha256": cutlass.source_weight_sha256,
                "destination_sha256": cutlass.runtime_weight_sha256,
                "padding_bytes": cutlass.weight_padding_bytes,
            },
            {
                "name": "weight_scale_swizzle",
                "operation": "independent linear-to-CUTLASS 128x4 byte permutation",
                "source_layout": "linear_row_major",
                "destination_layout": "cutlass_128x4",
                "source_shape": list(projection.weight_scale.shape),
                "destination_shape": list(layer.weight_scale.shape),
                "source_sha256": cutlass.source_weight_scale_sha256,
                "destination_sha256": cutlass.runtime_weight_scale_sha256,
                "vllm_candidate_byte_exact": True,
            },
            {
                "name": "gemm_b_metadata_views",
                "operation": "vLLM adapter passes weight.t() and weight_scale.t()",
                "weight_source_shape": list(layer.weight.shape),
                "weight_source_stride": list(layer.weight.stride()),
                "weight_view_shape": list(layer.weight.t().shape),
                "weight_view_stride": list(layer.weight.t().stride()),
                "scale_source_shape": list(layer.weight_scale.shape),
                "scale_source_stride": list(layer.weight_scale.stride()),
                "scale_view_shape": list(layer.weight_scale.t().shape),
                "scale_view_stride": list(layer.weight_scale.t().stride()),
                "storage_bytes_changed": False,
            },
        ],
        "scaling": {
            "formula": "dequant = E2M1 * E4M3 * weight_scale_2; alpha = input_scale * weight_scale_2",
            "input_scale": projection.input_scale_value,
            "weight_scale_2": projection.weight_scale_2_value,
            "input_global_scale_inv_math": cutlass.input_global_scale_inv,
            "input_global_scale_inv_runtime_f32": float(
                layer.input_global_scale_inv.item()
            ),
            "alpha_math": cutlass.alpha,
            "alpha_runtime_f32": float(layer.alpha.item()),
            "activation_max_abs": activation_max_abs,
            "checkpoint_calibrated_max_abs": calibrated_max_abs,
        },
        "replay": {
            "warmup_runs": 1,
            "repetitions": args.repetitions,
            "synchronized": True,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype).removeprefix("torch."),
            "all_finite": all_finite,
            "output_sha256s": output_hashes,
            "output_hash_stable": len(set(output_hashes)) == 1,
        },
        "backend": {
            "requested_format": "nvfp4",
            "requested_backend": "cutlass",
            "selected_vllm_kernel": selected_kernel,
            "reported_backend": None,
            "target_nvtx_range": range_name,
            "observed_kernels": [],
            "fallback_status": "pending_profiler",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "command": {
            "argv": [sys.executable, *sys.argv],
            "cwd": str(ROOT),
        },
        "claim_boundary": (
            "This is real checkpoint weight replay with a deterministic synthetic "
            "activation. It does not establish real Qwen activation capture, numerical "
            "correctness, model-level accuracy, or arbitrary-backend support."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"selected_vllm_kernel={selected_kernel}")
    print(f"target_nvtx_range={range_name}")
    print(f"output_sha256={output_hashes[0]}")
    print("status=pass")
    print("decision=pending_profiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
