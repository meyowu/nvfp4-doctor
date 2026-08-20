import copy
import unittest

from scripts.finalize_e004_real_activation import (
    EXPECTED_INPUT_IDENTITY_SHA256,
    REVISION,
    TARGET_RANGE,
    RealActivationFinalizationError,
    _validate_run,
)

PRESERVED_FIELDS = [
    "shape",
    "dtype",
    "stride",
    "storage_offset",
    "byte_length",
    "sha256",
]


def _artifact(path: str, tensor_sha256: str) -> dict[str, object]:
    source = {
        "shape": [9, 4096],
        "dtype": "bfloat16",
        "stride": [4096, 1],
        "storage_offset": 0,
        "device": "cuda:0",
        "contiguous": True,
        "numel": 36864,
        "byte_length": 73728,
        "sha256": tensor_sha256,
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }
    destination = {**source, "device": "cpu"}
    return {
        "path": path,
        "ignored": True,
        "encoding": "torch_save_cpu_tensor_v1",
        "file_bytes": 76000,
        "file_sha256": "a" * 64,
        "tensor": destination,
        "source_metadata": source,
        "preserved_fields": PRESERVED_FIELDS,
        "device_transfer": {"source": "cuda:0", "destination": "cpu"},
    }


def _run() -> dict[str, object]:
    input_sha256 = "1" * 64
    output_sha256 = "2" * 64
    input_artifact = _artifact(
        ("artifacts/E004-qwen3-layer-capture/real-activation/layer-00-o-proj-input.pt"),
        input_sha256,
    )
    module_output_artifact = _artifact(
        (
            "artifacts/E004-qwen3-layer-capture/real-activation/"
            "layer-00-o-proj-captured-module-output.pt"
        ),
        output_sha256,
    )
    replay_artifact = _artifact(
        (
            "artifacts/E004-qwen3-layer-capture/real-activation/"
            "layer-00-o-proj-replay-output.pt"
        ),
        output_sha256,
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "decision": "pending_profiler",
        "repository": {
            "id": "nvidia/Qwen3-8B-NVFP4",
            "revision": REVISION,
        },
        "model_load": {
            "local_snapshot_path": (f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}"),
            "frozen_environment": {
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
            "requested_args": {
                "quantization": "modelopt_fp4",
                "tensor_parallel_size": 1,
                "cpu_offload_gb": 0,
                "enforce_eager": True,
                "linear_backend": "auto",
            },
            "observed_model_class": "Qwen3ForCausalLM",
            "request_completed": True,
        },
        "input_identity": {
            "token_ids_committed_in_result": False,
            "token_count": 9,
            "token_ids_sha256": EXPECTED_INPUT_IDENTITY_SHA256,
        },
        "capture": {
            "case": {
                "layer": 0,
                "projection": "o_proj",
                "adapter_scope": "production_aligned_unfused",
                "tensor_role": "module_input",
                "phase": "prefill",
                "event_count": 1,
                "activation_provenance": "real_qwen_prefill",
            },
            "input_artifact": input_artifact,
            "captured_module_output_artifact": module_output_artifact,
            "metadata_preserved_fields": PRESERVED_FIELDS,
            "device_transfer_recorded": True,
        },
        "runtime_projection": {
            "module_path": "model.layers.0.self_attn.o_proj",
            "selected_kernel": "FlashInferCutlassNvFp4LinearKernel",
            "weights_padding_cols": 0,
        },
        "replay": {
            "warmup_runs": 1,
            "repetitions": 3,
            "synchronized": True,
            "all_finite": True,
            "output_hash_stable": True,
            "output_shape": [9, 4096],
            "output_dtype": "bfloat16",
            "output_sha256s": [output_sha256] * 3,
            "captured_module_output_sha256": output_sha256,
            "bitwise_captured_module_output_matches": [True, True, True],
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "input_sha256": input_sha256,
            "replay_output_artifact": replay_artifact,
        },
        "backend": {
            "requested_format": "nvfp4",
            "requested_backend": "auto",
            "selected_vllm_kernel": "FlashInferCutlassNvFp4LinearKernel",
            "reported_backend": None,
            "target_nvtx_range": TARGET_RANGE,
            "fallback_status": "pending_profiler",
        },
    }


class E004RealActivationFinalizationTests(unittest.TestCase):
    def test_accepts_the_frozen_runtime_observation(self) -> None:
        range_name, artifacts = _validate_run(_run())
        self.assertEqual(range_name, TARGET_RANGE)
        self.assertEqual(len(artifacts), 3)

    def test_rejects_committed_token_ids(self) -> None:
        run = _run()
        run["input_identity"]["token_ids"] = [1, 2, 3]
        with self.assertRaisesRegex(RealActivationFinalizationError, "input identity"):
            _validate_run(run)

    def test_rejects_non_bitwise_module_output_match(self) -> None:
        run = _run()
        run["replay"]["bitwise_captured_module_output_matches"] = [
            True,
            False,
            True,
        ]
        with self.assertRaisesRegex(RealActivationFinalizationError, "invariants"):
            _validate_run(run)

    def test_rejects_changed_transfer_metadata(self) -> None:
        run = _run()
        changed = copy.deepcopy(run)
        changed["capture"]["input_artifact"]["tensor"]["stride"] = [1, 9]
        with self.assertRaisesRegex(RealActivationFinalizationError, "preserve"):
            _validate_run(changed)

    def test_rejects_changed_backend_identity(self) -> None:
        run = _run()
        run["backend"]["requested_backend"] = "cutlass"
        with self.assertRaisesRegex(RealActivationFinalizationError, "backend"):
            _validate_run(run)


if __name__ == "__main__":
    unittest.main()
