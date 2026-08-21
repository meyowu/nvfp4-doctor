import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nvfp4_doctor.capture.e004_fused import E004_FUSED_REAL_ACTIVATION_CASES
from scripts import finalize_e004_real_activation_fused_matrix as finalizer
from scripts.finalize_e004_real_activation import (
    EXPECTED_KERNEL,
    PRESERVED_TRANSFER_FIELDS,
    REVISION,
    ROOT,
)
from scripts.run_e004_real_activation_capture import _tensor_metadata


def _digest(label: str | bytes) -> str:
    payload = label if isinstance(label, bytes) else label.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _source_record(
    *,
    name: str,
    payload: bytes,
    dtype: str,
    shape: list[int],
    shard_path: str,
) -> dict[str, object]:
    return {
        "tensor_name": name,
        "shard_path": shard_path,
        "shard_sha256": _digest(f"shard:{shard_path}"),
        "data_offsets": [0, len(payload)],
        "dtype": dtype,
        "shape": shape,
        "byte_length": len(payload),
        "sha256": _digest(payload),
    }


def _source_fixtures() -> tuple[
    dict[str, dict[str, object]],
    dict[str, bytes],
    dict[tuple[int, str], dict[str, object]],
]:
    sources: dict[str, dict[str, object]] = {}
    payloads: dict[str, bytes] = {}
    replay_cases: dict[tuple[int, str], dict[str, object]] = {}
    component_scale = {
        "q_proj": (1.0, 2.0),
        "k_proj": (1.0, 2.0),
        "v_proj": (1.0, 2.0),
        "gate_proj": (1.0, 2.0),
        "up_proj": (1.0, 2.0),
    }
    for spec in E004_FUSED_REAL_ACTIVATION_CASES:
        components: list[dict[str, object]] = []
        packed_parts: list[bytes] = []
        scale_parts: list[bytes] = []
        input_scales: list[float] = []
        weight_scales: list[float] = []
        shard_path = f"model-layer-{spec.layer:02d}.safetensors"
        for boundary in spec.component_boundaries:
            prefix = f"{spec.checkpoint_parent_path}.{boundary.projection}"
            input_scale, weight_scale_2 = component_scale[boundary.projection]
            values = {
                "input_scale": struct.pack("<f", input_scale),
                "weight": hashlib.sha256(f"{prefix}.weight".encode()).digest(),
                "weight_scale": hashlib.sha256(
                    f"{prefix}.weight_scale".encode()
                ).digest(),
                "weight_scale_2": struct.pack("<f", weight_scale_2),
            }
            dtypes = {
                "input_scale": "F32",
                "weight": "U8",
                "weight_scale": "F8_E4M3",
                "weight_scale_2": "F32",
            }
            shapes = {
                "input_scale": [],
                "weight": list(boundary.packed_weight_shape),
                "weight_scale": list(boundary.weight_scale_shape),
                "weight_scale_2": [],
            }
            tensors: dict[str, dict[str, object]] = {}
            for suffix, payload in values.items():
                name = f"{prefix}.{suffix}"
                payloads[name] = payload
                tensors[suffix] = _source_record(
                    name=name,
                    payload=payload,
                    dtype=dtypes[suffix],
                    shape=shapes[suffix],
                    shard_path=shard_path,
                )
            regression = (
                True
                if boundary.projection in {"q_proj", "gate_proj", "up_proj"}
                else None
            )
            component = {
                "projection": boundary.projection,
                "row_range": [boundary.row_start, boundary.row_end],
                "source_tensors": tensors,
                "prepared_packed_weight_sha256": _digest(values["weight"]),
                "prepared_runtime_weight_scale_sha256": _digest(values["weight_scale"]),
                "input_scale": input_scale,
                "weight_scale_2": weight_scale_2,
                "replay_matrix_regression_match": regression,
            }
            components.append(component)
            packed_parts.append(values["weight"])
            scale_parts.append(values["weight_scale"])
            input_scales.append(input_scale)
            weight_scales.append(weight_scale_2)
            if regression:
                replay_cases[(spec.layer, boundary.projection)] = {
                    "source_tensor_sha256": {
                        suffix: tensor["sha256"] for suffix, tensor in tensors.items()
                    },
                    "runtime_weight_scale_sha256": _digest(values["weight_scale"]),
                    "input_scale": input_scale,
                    "weight_scale_2": weight_scale_2,
                }
        input_global_scale = finalizer._f32(max(input_scales))
        weight_global_scale = finalizer._f32(max(weight_scales))
        sources[spec.case_id] = {
            "component_order": list(spec.component_projections),
            "component_row_boundaries": list(spec.component_row_boundaries),
            "components": components,
            "fused_packed_weight_sha256": _digest(b"".join(packed_parts)),
            "fused_runtime_weight_scale_sha256": _digest(b"".join(scale_parts)),
            "expected_runtime_scalars": {
                "reduction_rule": "float32_max_over_ordered_components",
                "input_global_scale": input_global_scale,
                "weight_global_scale": weight_global_scale,
                "alpha": finalizer._f32(input_global_scale * weight_global_scale),
                "input_global_scale_inv": finalizer._f32(1.0 / input_global_scale),
            },
            "source_snapshot_verified": True,
        }
    return sources, payloads, replay_cases


SOURCES, SOURCE_PAYLOADS, REPLAY_CASES = _source_fixtures()


def _read_source_tensor(
    _value: object,
    *,
    expected_name: str,
    **_kwargs: object,
) -> bytes:
    return SOURCE_PAYLOADS[expected_name]


def _run() -> dict[str, object]:
    unfused = json.loads(finalizer.UNFUSED_MATRIX_RESULT.read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    for index, spec in enumerate(E004_FUSED_REAL_ACTIVATION_CASES):
        source = copy.deepcopy(SOURCES[spec.case_id])
        input_sha = _digest(f"input:{index}")
        output_sha = _digest(f"output:{index}")
        prefix = (
            "artifacts/E004-qwen3-layer-capture/real-activation-fused-matrix/"
            f"{spec.artifact_slug}"
        )
        input_artifact = _artifact(
            f"{prefix}-input.pt", list(spec.input_shape(9)), input_sha
        )
        output_artifact = _artifact(
            f"{prefix}-captured-module-output.pt",
            list(spec.output_shape(9)),
            output_sha,
        )
        replay_artifact = _artifact(
            f"{prefix}-replay-output.pt",
            list(spec.output_shape(9)),
            output_sha,
        )
        bindings = [
            {
                "projection": boundary.projection,
                "row_range": [boundary.row_start, boundary.row_end],
                "packed_weight_slice_sha256": component[
                    "prepared_packed_weight_sha256"
                ],
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
        slice_records = []
        for boundary in spec.component_boundaries:
            slice_sha = _digest(f"slice:{spec.case_id}:{boundary.projection}")
            slice_records.append(
                {
                    "projection": boundary.projection,
                    "feature_range": [boundary.row_start, boundary.row_end],
                    "captured_sha256": slice_sha,
                    "replay_sha256s": [slice_sha] * 3,
                    "logical_matches": [True] * 3,
                }
            )
        scalars = source["expected_runtime_scalars"]
        cases.append(
            {
                "case_id": spec.case_id,
                "layer": spec.layer,
                "role": spec.role,
                "projection": spec.projection,
                "adapter_scope": spec.adapter_scope,
                "module_path": spec.module_path,
                "module_class": spec.module_class,
                "tensor_role": "module_input",
                "phase": "prefill",
                "event_count": 1,
                "activation_provenance": "real_qwen_prefill",
                "source_construction": source,
                "capture": {
                    "input_artifact": input_artifact,
                    "captured_module_output_artifact": output_artifact,
                    "metadata_preserved_fields": PRESERVED_TRANSFER_FIELDS,
                    "device_transfer_recorded": True,
                },
                "runtime_projection": {
                    "module_path": spec.module_path,
                    "module_class": spec.module_class,
                    "quant_method_class": "ModelOptNvFp4LinearMethod",
                    "selected_kernel": EXPECTED_KERNEL,
                    "tp_size": 1,
                    "tp_rank": 0,
                    "gather_output": False,
                    "logical_widths": list(spec.component_output_widths),
                    "packed_weight": finalizer._runtime_tensor_metadata(
                        shape=spec.packed_weight_shape,
                        dtype="uint8",
                        sha256=source["fused_packed_weight_sha256"],
                    ),
                    "runtime_weight_scale": finalizer._runtime_tensor_metadata(
                        shape=spec.weight_scale_shape,
                        dtype="float8_e4m3fn",
                        sha256=source["fused_runtime_weight_scale_sha256"],
                    ),
                    "weights_padding_cols": 0,
                    "input_global_scale": scalars["input_global_scale"],
                    "weight_global_scale": scalars["weight_global_scale"],
                    "alpha": scalars["alpha"],
                    "input_global_scale_inv": scalars["input_global_scale_inv"],
                    "component_bindings": bindings,
                    "global_scale_reduction_matches": True,
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
                    "logical_byte_exact_captured_module_output_matches": [True] * 3,
                    "reconstructed_activation_metadata": copy.deepcopy(
                        input_artifact["source_metadata"]
                    ),
                    "logical_byte_exact_captured_input_match": True,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "input_sha256": input_sha,
                    "replay_output_artifact": replay_artifact,
                    "component_output_slices": slice_records,
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
    return {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_fused_real_activation_observation_v1",
        "captured_at_utc": "2026-08-20T00:00:00Z",
        "status": "pass",
        "decision": "pending_profiler",
        "repository": {"id": "nvidia/Qwen3-8B-NVFP4", "revision": REVISION},
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
        "input_identity": copy.deepcopy(unfused["input_identity"]),
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
            "repetitions_per_case": 3,
            "hook_count": 12,
            "hook_event_order": [
                f"{case.case_id}:{role}"
                for case in E004_FUSED_REAL_ACTIVATION_CASES
                for role in ("input", "module_output")
            ],
            "distinct_input_sha256_count": 6,
            "distinct_module_output_sha256_count": 6,
        },
        "identity_dependencies": finalizer._expected_raw_dependencies(),
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
                str(finalizer.DEFAULT_RUN),
                "--profile-capture",
            ],
            "cwd": str(ROOT),
        },
        "claim_boundary": "bounded raw observation pending profiler",
    }


def _observed_kernels() -> tuple[str, ...]:
    names: list[str] = []
    for case in E004_FUSED_REAL_ACTIVATION_CASES:
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


class E004RealActivationFusedMatrixFinalizationTests(unittest.TestCase):
    def test_accepts_the_exact_project_local_snapshot_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            repository_root = temporary_root / "repository"
            canonical_snapshot = temporary_root / "canonical-snapshot"
            canonical_snapshot.mkdir()
            relative_snapshot = Path("models") / "nvidia--Qwen3-8B-NVFP4" / REVISION
            linked_snapshot = repository_root / relative_snapshot
            linked_snapshot.parent.mkdir(parents=True)
            linked_snapshot.symlink_to(canonical_snapshot, target_is_directory=True)
            snapshot_result = {
                "snapshot": {
                    "local_root": relative_snapshot.as_posix(),
                    "files": [],
                }
            }
            with (
                patch.object(finalizer, "ROOT", repository_root),
                patch.object(finalizer, "_json", return_value=snapshot_result),
            ):
                model_dir, inventory = finalizer._full_snapshot_inventory()

        self.assertEqual(model_dir, canonical_snapshot.resolve())
        self.assertEqual(inventory, {})

    def _validate(self, run: dict[str, object]) -> tuple[list[dict], tuple[dict, ...]]:
        with (
            patch.object(
                finalizer,
                "_full_snapshot_inventory",
                return_value=(Path("/synthetic-model"), {}),
            ),
            patch.object(finalizer, "_replay_matrix_cases", return_value=REPLAY_CASES),
            patch.object(
                finalizer, "_read_snapshot_tensor", side_effect=_read_source_tensor
            ),
            patch.object(
                finalizer,
                "swizzle_scales_128x4",
                side_effect=lambda payload, *_: payload,
            ),
        ):
            return finalizer._validate_run(run)

    def test_accepts_the_exact_six_case_fused_observation(self) -> None:
        cases, artifacts = self._validate(_run())
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(artifacts), 18)

    def test_rejects_component_reorder_boundary_and_scalar_mutations(self) -> None:
        reordered = _run()
        reordered["cases"][0]["source_construction"]["components"].reverse()
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "source construction",
        ):
            self._validate(reordered)

        boundary = _run()
        boundary["cases"][1]["source_construction"]["component_row_boundaries"][1] += (
            128
        )
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "fused source construction",
        ):
            self._validate(boundary)

        scalar = _run()
        scalar["cases"][2]["runtime_projection"]["input_global_scale"] += 1.0
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "runtime fused projection",
        ):
            self._validate(scalar)

        unequal = _run()
        spec = E004_FUSED_REAL_ACTIVATION_CASES[0]
        tensor_name = f"{spec.checkpoint_parent_path}.k_proj.input_scale"
        original_payload = SOURCE_PAYLOADS[tensor_name]
        try:
            changed_payload = struct.pack("<f", 3.0)
            SOURCE_PAYLOADS[tensor_name] = changed_payload
            component = unequal["cases"][0]["source_construction"]["components"][1]
            component["input_scale"] = 3.0
            component["source_tensors"]["input_scale"]["sha256"] = _digest(
                changed_payload
            )
            with self.assertRaisesRegex(
                finalizer.RealActivationFusedMatrixFinalizationError,
                "component scalar identities disagree",
            ):
                self._validate(unequal)
        finally:
            SOURCE_PAYLOADS[tensor_name] = original_payload

    def test_rejects_runtime_slice_tp_and_replay_slice_mutations(self) -> None:
        runtime_slice = _run()
        runtime_slice["cases"][3]["runtime_projection"]["component_bindings"][0][
            "packed_weight_slice_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "runtime fused projection",
        ):
            self._validate(runtime_slice)

        tensor_parallel = _run()
        tensor_parallel["cases"][4]["runtime_projection"]["tp_size"] = 2
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "runtime fused projection",
        ):
            self._validate(tensor_parallel)

        output_slice = _run()
        output_slice["cases"][5]["replay"]["component_output_slices"][0][
            "logical_matches"
        ] = [True, False, True]
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "component output replay",
        ):
            self._validate(output_slice)

    def test_rejects_unknown_fields_and_nonexact_case_coverage(self) -> None:
        extra = _run()
        extra["cases"][0]["source_construction"]["components"][0]["unsupported"] = True
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "fields changed",
        ):
            self._validate(extra)

        missing = _run()
        missing["cases"].pop()
        with self.assertRaisesRegex(
            finalizer.RealActivationFusedMatrixFinalizationError,
            "exactly six",
        ):
            self._validate(missing)

    def test_used_shard_is_stream_hashed_once_and_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            shard = model_dir / "tiny.safetensors"
            payload = bytes(range(16))
            tensor_name = "model.layers.0.self_attn.q_proj.weight"
            header = json.dumps(
                {
                    tensor_name: {
                        "dtype": "U8",
                        "shape": [2, 8],
                        "data_offsets": [0, len(payload)],
                    }
                },
                separators=(",", ":"),
            ).encode("utf-8")
            shard.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            shard_sha = _digest(shard.read_bytes())
            inventory = {
                "tiny.safetensors": {
                    "path": "tiny.safetensors",
                    "role": "weight_shard",
                    "size_bytes": shard.stat().st_size,
                    "sha256": shard_sha,
                    "lfs_sha256": shard_sha,
                }
            }
            record = {
                "tensor_name": tensor_name,
                "shard_path": "tiny.safetensors",
                "shard_sha256": shard_sha,
                "data_offsets": [0, len(payload)],
                "dtype": "U8",
                "shape": [2, 8],
                "byte_length": len(payload),
                "sha256": _digest(payload),
            }
            header_cache: dict = {}
            shard_hash_cache: dict[str, str] = {}
            first = finalizer._read_snapshot_tensor(
                record,
                expected_name=tensor_name,
                expected_dtype="U8",
                expected_shape=[2, 8],
                model_dir=model_dir,
                inventory=inventory,
                header_cache=header_cache,
                shard_hash_cache=shard_hash_cache,
            )
            self.assertEqual(first, payload)
            self.assertEqual(shard_hash_cache, {"tiny.safetensors": shard_sha})

            shard_hash_cache.clear()
            changed = bytearray(shard.read_bytes())
            changed[-1] ^= 1
            shard.write_bytes(changed)
            with self.assertRaisesRegex(
                finalizer.RealActivationFusedMatrixFinalizationError,
                "shard hash changed",
            ):
                finalizer._read_snapshot_tensor(
                    record,
                    expected_name=tensor_name,
                    expected_dtype="U8",
                    expected_shape=[2, 8],
                    model_dir=model_dir,
                    inventory=inventory,
                    header_cache={},
                    shard_hash_cache=shard_hash_cache,
                )

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
                "file_sha256": _digest(path.read_bytes()),
                "tensor": metadata,
                "source_metadata": metadata,
            }
            loaded = finalizer._load_local_tensor_artifact(artifact, root=root)
            self.assertTrue(torch.equal(loaded, tensor))
            artifact["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                finalizer.RealActivationFusedMatrixFinalizationError,
                "local artifact identity changed",
            ):
                finalizer._load_local_tensor_artifact(artifact, root=root)

    def test_profile_requires_all_six_exact_ranges(self) -> None:
        backend, ranges, passed = finalizer._profile_backend(
            _observed_kernels(), "a" * 64
        )
        self.assertTrue(passed)
        self.assertEqual(len(ranges), 6)
        self.assertEqual(len(backend["kernel_catalog"]), 2)
        for evidence in ranges.values():
            self.assertTrue(evidence["expected_sm120_cutlass_signature_present"])
            self.assertTrue(evidence["activation_quantization_signature_present"])
            self.assertEqual(evidence["fallback_status"], "not_detected")

        target = E004_FUSED_REAL_ACTIVATION_CASES[2].target_nvtx_range
        missing = tuple(
            name
            for name in _observed_kernels()
            if not (target in name and "cvt_fp16_to_fp4" in name)
        )
        _backend, failed_ranges, passed = finalizer._profile_backend(missing, "b" * 64)
        self.assertFalse(passed)
        self.assertFalse(
            failed_ranges[E004_FUSED_REAL_ACTIVATION_CASES[2].case_id][
                "activation_quantization_signature_present"
            ]
        )

    def test_dependency_pairs_are_semantically_and_hash_bound(self) -> None:
        dependencies = finalizer._dependency_artifacts()
        self.assertEqual(len(dependencies), 6)

    def test_pair_publication_restores_prior_files_on_second_replace_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "result.json"
            manifest = root / "manifest.json"
            results.write_text("old result", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            real_replace = os.replace
            calls = 0

            def fail_second(source: str | Path, target: str | Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated manifest replace failure")
                real_replace(source, target)

            with patch.object(finalizer.os, "replace", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "simulated"):
                    finalizer._publish_result_and_manifest(
                        results_path=results,
                        results_text="new result",
                        manifest_path=manifest,
                        manifest_text="new manifest",
                    )
            self.assertEqual(results.read_text(encoding="utf-8"), "old result")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
