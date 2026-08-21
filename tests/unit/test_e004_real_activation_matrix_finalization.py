import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from nvfp4_doctor.capture import E004_UNFUSED_REAL_ACTIVATION_CASES
from scripts.finalize_e004_real_activation import (
    EXPECTED_INPUT_IDENTITY_SHA256,
    EXPECTED_KERNEL,
    PRESERVED_TRANSFER_FIELDS,
    REVISION,
    ROOT,
    _sha256_path,
)
from scripts.finalize_e004_real_activation_matrix import (
    REPLAY_MATRIX_RESULT,
    SINGLE_REAL_RESULT,
    RealActivationMatrixFinalizationError,
    _dependency_artifacts,
    _expected_identity_by_id,
    _json,
    _profile_backend,
    _validate_local_tensor_artifacts,
    _validate_run,
)
from scripts.run_e004_real_activation_capture import _tensor_metadata


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(path: str, shape: list[int], tensor_sha256: str) -> dict[str, object]:
    stride = [shape[1], 1]
    numel = shape[0] * shape[1]
    source = {
        "shape": shape,
        "dtype": "bfloat16",
        "stride": stride,
        "storage_offset": 0,
        "device": "cuda:0",
        "contiguous": True,
        "numel": numel,
        "byte_length": numel * 2,
        "sha256": tensor_sha256,
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }
    return {
        "path": path,
        "ignored": True,
        "encoding": "torch_save_cpu_tensor_v1",
        "file_bytes": numel * 2 + 1024,
        "file_sha256": _digest(f"file:{path}"),
        "tensor": {**source, "device": "cpu"},
        "source_metadata": source,
        "preserved_fields": PRESERVED_TRANSFER_FIELDS,
        "device_transfer": {"source": "cuda:0", "destination": "cpu"},
    }


def _runtime_tensor_metadata(
    shape: list[int], dtype: str, tensor_sha256: str
) -> dict[str, object]:
    numel = shape[0] * shape[1]
    return {
        "shape": shape,
        "dtype": dtype,
        "stride": [shape[1], 1],
        "storage_offset": 0,
        "device": "cuda:0",
        "contiguous": True,
        "numel": numel,
        "byte_length": numel,
        "sha256": tensor_sha256,
        "sha256_encoding": "canonical_contiguous_logical_bytes",
    }


def _run() -> dict[str, object]:
    identities = _expected_identity_by_id()
    prior = _json(SINGLE_REAL_RESULT)
    prior_input_sha = prior["capture"]["input_artifact"]["tensor"]["sha256"]
    prior_output_sha = prior["replay"]["captured_module_output_sha256"]
    prior_generated_sha = prior["input_identity"]["generated_token_ids_sha256"]
    cases: list[dict[str, object]] = []
    for index, spec in enumerate(E004_UNFUSED_REAL_ACTIVATION_CASES):
        input_sha = prior_input_sha if index == 0 else _digest(f"input:{index}")
        output_sha = prior_output_sha if index == 0 else _digest(f"output:{index}")
        prefix = (
            "artifacts/E004-qwen3-layer-capture/real-activation-matrix/"
            f"{spec.artifact_slug}"
        )
        input_artifact = _artifact(
            f"{prefix}-input.pt", list(spec.input_shape(9)), input_sha
        )
        module_output_artifact = _artifact(
            f"{prefix}-captured-module-output.pt",
            list(spec.output_shape(9)),
            output_sha,
        )
        replay_artifact = _artifact(
            f"{prefix}-replay-output.pt",
            list(spec.output_shape(9)),
            output_sha,
        )
        identity = identities[spec.case_id]
        cases.append(
            {
                "case_id": spec.case_id,
                "layer": spec.layer,
                "role": spec.role,
                "projection": spec.projection,
                "adapter_scope": "production_aligned_unfused",
                "module_path": spec.module_path,
                "tensor_role": "module_input",
                "phase": "prefill",
                "event_count": 1,
                "activation_provenance": "real_qwen_prefill",
                "checkpoint_identity": identity,
                "capture": {
                    "input_artifact": input_artifact,
                    "captured_module_output_artifact": module_output_artifact,
                    "metadata_preserved_fields": PRESERVED_TRANSFER_FIELDS,
                    "device_transfer_recorded": True,
                },
                "runtime_projection": {
                    "module_path": spec.module_path,
                    "module_class": "RowParallelLinear",
                    "quant_method_class": "ModelOptNvFp4LinearMethod",
                    "selected_kernel": EXPECTED_KERNEL,
                    "packed_weight": _runtime_tensor_metadata(
                        identity["packed_weight_shape"],
                        "uint8",
                        identity["runtime_packed_weight_sha256"],
                    ),
                    "runtime_weight_scale": _runtime_tensor_metadata(
                        identity["weight_scale_shape"],
                        "float8_e4m3fn",
                        identity["runtime_weight_scale_sha256"],
                    ),
                    "weights_padding_cols": 0,
                    **identity["expected_runtime_scalars"],
                },
                "replay": {
                    "warmup_runs": 1,
                    "repetitions": 3,
                    "synchronized": True,
                    "all_finite": True,
                    "output_shape": list(spec.output_shape(9)),
                    "output_dtype": "bfloat16",
                    "output_sha256s": [output_sha] * 3,
                    "output_hash_stable": True,
                    "captured_module_output_sha256": output_sha,
                    "logical_byte_exact_captured_module_output_matches": [
                        True,
                        True,
                        True,
                    ],
                    "reconstructed_activation_metadata": copy.deepcopy(
                        input_artifact["source_metadata"]
                    ),
                    "logical_byte_exact_captured_input_match": True,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "input_sha256": input_sha,
                    "replay_output_artifact": replay_artifact,
                },
                "backend_range": {
                    "target_nvtx_range": spec.target_nvtx_range,
                    "target_kernel_ids": [],
                    "target_kernel_set_sha256": None,
                    "expected_sm120_cutlass_signature_present": False,
                    "activation_quantization_signature_present": False,
                    "fallback_status": "pending_profiler",
                },
            }
        )
    event_order = [
        f"{case.case_id}:{role}"
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES
        for role in ("input", "module_output")
    ]
    return {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_unfused_real_activation_observation_v1",
        "captured_at_utc": "2026-08-20T00:00:00Z",
        "status": "pass",
        "decision": "pending_profiler",
        "repository": {
            "id": "nvidia/Qwen3-8B-NVFP4",
            "revision": REVISION,
        },
        "model_load": {
            "local_snapshot_path": f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}",
            "frozen_environment": {
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "requested_args": {
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
            },
            "observed_model_class": "Qwen3ForCausalLM",
            "model_load_count": 1,
            "request_count": 1,
            "request_completed": True,
            "free_memory_before_bytes": 10_000_000_000,
            "free_memory_after_load_bytes": 8_000_000_000,
            "peak_allocated_bytes": 7_000_000_000,
            "peak_reserved_bytes": 7_500_000_000,
        },
        "input_identity": {
            "provenance": "fixed_public_token_sequence",
            "token_ids_committed_in_result": False,
            "token_ids_encoding": "little_endian_signed_int32",
            "token_count": 9,
            "token_ids_sha256": EXPECTED_INPUT_IDENTITY_SHA256,
            "generated_token_count": 1,
            "generated_token_ids_sha256": prior_generated_sha,
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
        "matrix": {
            "layers": [0, 18, 35],
            "layer_roles": ["early", "middle", "late"],
            "projections": ["o_proj", "down_proj"],
            "case_ids": [case.case_id for case in E004_UNFUSED_REAL_ACTIVATION_CASES],
            "case_count": 6,
            "repetitions_per_case": 3,
            "hook_count": 12,
            "hook_event_order": event_order,
            "distinct_input_sha256_count": 6,
            "distinct_module_output_sha256_count": 6,
        },
        "identity_dependency": {
            "path": REPLAY_MATRIX_RESULT.relative_to(
                REPLAY_MATRIX_RESULT.parents[2]
            ).as_posix(),
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
        "cases": cases,
        "gpu": {
            "name": "NVIDIA GeForce RTX 5080",
            "compute_capability": [12, 0],
            "total_memory_bytes": 17_095_000_000,
        },
        "command": {
            "argv": [
                "/home/meyowu/projects/nvfp4-doctor/.venv/bin/python",
                str(ROOT / "scripts" / "run_e004_real_activation_matrix.py"),
                "--model-dir",
                str(ROOT / "models" / "nvidia--Qwen3-8B-NVFP4" / REVISION),
                "--artifact-root",
                str(
                    ROOT
                    / "artifacts"
                    / "E004-qwen3-layer-capture"
                    / "real-activation-matrix"
                ),
                "--output",
                str(
                    ROOT
                    / "artifacts"
                    / "E004-qwen3-layer-capture"
                    / "real-activation-matrix"
                    / "unfused-matrix.json"
                ),
                "--profile-capture",
            ],
            "cwd": str(ROOT),
        },
        "claim_boundary": "bounded raw observation",
    }


def _observed_kernels() -> tuple[str, ...]:
    names: list[str] = []
    for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
        names.extend(
            [
                (
                    f"{case.target_nvtx_range}/void cutlass::device_kernel<"
                    "MainloopSm120TmaWarpSpecializedBlockScaled, "
                    "cutlass::float_e2m1_t, SM120_16x8x64_TN_VS>()"
                ),
                (
                    f"{case.target_nvtx_range}/"
                    "void vllm::cvt_fp16_to_fp4<__nv_bfloat16>()"
                ),
            ]
        )
    return tuple(names)


class E004RealActivationMatrixFinalizationTests(unittest.TestCase):
    def test_accepts_the_exact_six_case_observation(self) -> None:
        cases, artifacts = _validate_run(_run())
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(artifacts), 18)

    def test_rejects_missing_duplicate_and_extra_cases(self) -> None:
        missing = _run()
        missing["cases"].pop()
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "exactly six"
        ):
            _validate_run(missing)
        duplicate = _run()
        duplicate["cases"][-1] = copy.deepcopy(duplicate["cases"][0])
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "case identity"
        ):
            _validate_run(duplicate)
        extra = _run()
        extra["cases"].append(copy.deepcopy(extra["cases"][0]))
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "exactly six"
        ):
            _validate_run(extra)

    def test_rejects_runtime_hash_and_transfer_stride_changes(self) -> None:
        wrong_hash = _run()
        wrong_hash["cases"][2]["runtime_projection"]["packed_weight"]["sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "runtime projection"
        ):
            _validate_run(wrong_hash)
        wrong_stride = _run()
        wrong_stride["cases"][3]["capture"]["input_artifact"]["tensor"]["stride"] = [
            1,
            9,
        ]
        with self.assertRaisesRegex(RealActivationMatrixFinalizationError, "preserve"):
            _validate_run(wrong_stride)

    def test_rejects_a_replay_artifact_not_bound_to_the_replay_hash(self) -> None:
        changed = _run()
        replay_artifact = changed["cases"][4]["replay"]["replay_output_artifact"]
        replay_artifact["tensor"]["sha256"] = "f" * 64
        replay_artifact["source_metadata"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "replay invariants"
        ):
            _validate_run(changed)

    def test_rejects_a_reconstructed_activation_hash_change(self) -> None:
        changed = _run()
        changed["cases"][1]["replay"]["reconstructed_activation_metadata"]["sha256"] = (
            "e" * 64
        )
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "replay invariants"
        ):
            _validate_run(changed)

    def test_rejects_prompt_payload(self) -> None:
        run = _run()
        run["input_identity"]["token_ids"] = [1, 2, 3]
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "input identity"
        ):
            _validate_run(run)

    def test_rejects_unknown_top_case_and_replay_fields(self) -> None:
        top_level = _run()
        top_level["unsupported_claim"] = True
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "fields changed"
        ):
            _validate_run(top_level)
        case_level = _run()
        case_level["cases"][0]["unexpected"] = "value"
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "fields changed"
        ):
            _validate_run(case_level)
        replay_level = _run()
        replay_level["cases"][0]["replay"]["prompt_text"] = "not allowed"
        with self.assertRaisesRegex(
            RealActivationMatrixFinalizationError, "fields changed"
        ):
            _validate_run(replay_level)

    def test_dependency_pairs_are_semantically_bound(self) -> None:
        dependencies = _dependency_artifacts()
        self.assertEqual(len(dependencies), 6)

    def test_local_tensor_artifact_is_reloaded_and_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tensor.pt"
            tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
            metadata = _tensor_metadata(tensor)
            torch.save(
                {
                    "tensor": tensor,
                    "source_metadata": metadata,
                    "destination_metadata": metadata,
                },
                path,
            )
            artifact = {
                "path": "tensor.pt",
                "file_bytes": path.stat().st_size,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tensor": metadata,
                "source_metadata": metadata,
            }
            _validate_local_tensor_artifacts((artifact,), root=root)
            changed = copy.deepcopy(artifact)
            changed["tensor"]["sha256"] = "d" * 64
            with self.assertRaisesRegex(
                RealActivationMatrixFinalizationError, "bytes or metadata"
            ):
                _validate_local_tensor_artifacts((changed,), root=root)

    def test_profile_requires_cutlass_quantization_and_no_fallback_per_range(
        self,
    ) -> None:
        backend, ranges, passed = _profile_backend(_observed_kernels(), "a" * 64)
        self.assertTrue(passed)
        self.assertEqual(len(ranges), 6)
        self.assertEqual(len(backend["kernel_catalog"]), 2)
        quantization_ids = {
            evidence["target_kernel_ids"][1] for evidence in ranges.values()
        }
        self.assertEqual(len(quantization_ids), 1)
        for evidence in ranges.values():
            self.assertTrue(evidence["expected_sm120_cutlass_signature_present"])
            self.assertTrue(evidence["activation_quantization_signature_present"])
            self.assertEqual(evidence["fallback_status"], "not_detected")

    def test_profile_fails_one_missing_or_fallback_range(self) -> None:
        kernels = list(_observed_kernels())
        missing_quantization = tuple(
            name
            for name in kernels
            if not (
                E004_UNFUSED_REAL_ACTIVATION_CASES[2].target_nvtx_range in name
                and "cvt_fp16_to_fp4" in name
            )
        )
        _backend, ranges, passed = _profile_backend(missing_quantization, "b" * 64)
        self.assertFalse(passed)
        self.assertFalse(
            ranges["layer-18-o-proj"]["activation_quantization_signature_present"]
        )
        fallback = (
            *kernels,
            (
                f"{E004_UNFUSED_REAL_ACTIVATION_CASES[-1].target_nvtx_range}/"
                "void cublasGemmEx()"
            ),
        )
        _backend, ranges, passed = _profile_backend(fallback, "c" * 64)
        self.assertFalse(passed)
        self.assertEqual(ranges["layer-35-down-proj"]["fallback_status"], "detected")


if __name__ == "__main__":
    unittest.main()
