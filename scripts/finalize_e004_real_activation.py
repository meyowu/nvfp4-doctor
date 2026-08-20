#!/usr/bin/env python3
"""Finalize one profiled E004 real-activation capture and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nvfp4_doctor.backends import (
    assess_range_fallback,
    expected_sm120_cutlass_present,
    extract_kernel_evidence,
    kernels_in_nvtx_range,
)
from nvfp4_doctor.env import collect_git, collect_gpu, collect_software

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
TARGET_RANGE = "e004:real_activation:layer_00:o_proj:nvfp4_gemm"
EXPECTED_KERNEL = "FlashInferCutlassNvFp4LinearKernel"
EXPECTED_INPUT_IDENTITY_SHA256 = (
    "154c66e5fa3bad6f385105bb54d93b6f2ab1e3bc9e3b1452bffcbfa6fd97413e"
)
DEFAULT_RUN = (
    ROOT
    / "artifacts"
    / "E004-qwen3-layer-capture"
    / "real-activation"
    / "layer-00-o-proj.json"
)
DEFAULT_REPORT = (
    ROOT / ".local" / "profiles" / "e004-real-activation-layer-00-o-proj.nsys-rep"
)
DEFAULT_RESULTS = EXPERIMENT / "real-activation-replay.json"
DEFAULT_MANIFEST = EXPERIMENT / "manifest-real-activation-replay.json"
FULL_MODEL_RESULT = EXPERIMENT / "full-model-acquisition.json"
FULL_MODEL_MANIFEST = EXPERIMENT / "manifest-full-model-acquisition.json"
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "backends" / "nsys.py",
    ROOT / "scripts" / "run_e004_real_activation_capture.py",
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_e004_real_activation_profile.sh",
    ROOT / "tests" / "unit" / "test_e004_real_activation_capture.py",
    ROOT / "tests" / "unit" / "test_e004_real_activation_finalization.py",
    ROOT / "tests" / "unit" / "test_nsys_evidence.py",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
PRESERVED_TRANSFER_FIELDS = [
    "shape",
    "dtype",
    "stride",
    "storage_offset",
    "byte_length",
    "sha256",
]


class RealActivationFinalizationError(RuntimeError):
    """Raised when runtime or profiler evidence violates the frozen contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-evidence", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RealActivationFinalizationError(f"could not read {path}") from error
    return digest.hexdigest()


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RealActivationFinalizationError(f"could not read {path}") from error
    if not isinstance(value, dict):
        raise RealActivationFinalizationError(f"{path} must contain an object")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealActivationFinalizationError(f"{label} must be an object")
    return value


def _command(*argv: str, timeout: int = 180) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.replace("\x00", "").strip()


def _validate_tensor_artifact(
    value: object,
    *,
    label: str,
    expected_path: str,
) -> dict[str, Any]:
    artifact = _mapping(value, label)
    if (
        artifact.get("path") != expected_path
        or artifact.get("ignored") is not True
        or artifact.get("encoding") != "torch_save_cpu_tensor_v1"
        or not SHA256.fullmatch(str(artifact.get("file_sha256", "")))
    ):
        raise RealActivationFinalizationError(f"{label} identity changed")
    source = _mapping(artifact.get("source_metadata"), f"{label} source metadata")
    destination = _mapping(artifact.get("tensor"), f"{label} tensor metadata")
    if artifact.get("preserved_fields") != PRESERVED_TRANSFER_FIELDS:
        raise RealActivationFinalizationError(f"{label} preservation fields changed")
    if artifact.get("device_transfer") != {
        "source": "cuda:0",
        "destination": "cpu",
    }:
        raise RealActivationFinalizationError(f"{label} device transfer changed")
    if source.get("device") != "cuda:0" or destination.get("device") != "cpu":
        raise RealActivationFinalizationError(f"{label} devices changed")
    for field in PRESERVED_TRANSFER_FIELDS:
        if source.get(field) != destination.get(field):
            raise RealActivationFinalizationError(f"{label} did not preserve {field}")
    if (
        source.get("sha256_encoding") != "canonical_contiguous_logical_bytes"
        or destination.get("sha256_encoding") != "canonical_contiguous_logical_bytes"
    ):
        raise RealActivationFinalizationError(f"{label} hash encoding changed")
    return artifact


def _validate_run(run: dict[str, Any]) -> tuple[str, tuple[dict[str, Any], ...]]:
    if (
        run.get("schema_version") != 1
        or run.get("status") != "pass"
        or run.get("decision") != "pending_profiler"
    ):
        raise RealActivationFinalizationError("runtime observation did not pass")
    repository = _mapping(run.get("repository"), "repository")
    if repository != {"id": "nvidia/Qwen3-8B-NVFP4", "revision": REVISION}:
        raise RealActivationFinalizationError("repository identity changed")

    model_load = _mapping(run.get("model_load"), "model load")
    requested = _mapping(model_load.get("requested_args"), "requested model args")
    frozen_environment = _mapping(
        model_load.get("frozen_environment"), "frozen environment"
    )
    if (
        model_load.get("local_snapshot_path")
        != f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}"
        or model_load.get("observed_model_class") != "Qwen3ForCausalLM"
        or model_load.get("request_completed") is not True
        or requested.get("quantization") != "modelopt_fp4"
        or requested.get("tensor_parallel_size") != 1
        or requested.get("cpu_offload_gb") != 0
        or requested.get("enforce_eager") is not True
        or requested.get("linear_backend") != "auto"
        or frozen_environment.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0"
        or frozen_environment.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1"
        or frozen_environment.get("VLLM_USE_FLASHINFER_SAMPLER") != "0"
    ):
        raise RealActivationFinalizationError("model-load contract changed")

    input_identity = _mapping(run.get("input_identity"), "input identity")
    if (
        input_identity.get("token_ids_committed_in_result") is not False
        or input_identity.get("token_count") != 9
        or input_identity.get("token_ids_sha256") != EXPECTED_INPUT_IDENTITY_SHA256
        or "token_ids" in input_identity
        or "prompt_text" in input_identity
    ):
        raise RealActivationFinalizationError("hashed input identity changed")

    capture = _mapping(run.get("capture"), "capture")
    case = _mapping(capture.get("case"), "capture case")
    if (
        case.get("layer"),
        case.get("projection"),
        case.get("adapter_scope"),
        case.get("tensor_role"),
        case.get("phase"),
        case.get("event_count"),
        case.get("activation_provenance"),
    ) != (
        0,
        "o_proj",
        "production_aligned_unfused",
        "module_input",
        "prefill",
        1,
        "real_qwen_prefill",
    ):
        raise RealActivationFinalizationError("capture case changed")
    if (
        capture.get("metadata_preserved_fields") != PRESERVED_TRANSFER_FIELDS
        or capture.get("device_transfer_recorded") is not True
    ):
        raise RealActivationFinalizationError("capture transfer contract changed")

    input_artifact = _validate_tensor_artifact(
        capture.get("input_artifact"),
        label="input artifact",
        expected_path=(
            "artifacts/E004-qwen3-layer-capture/real-activation/"
            "layer-00-o-proj-input.pt"
        ),
    )
    module_output_artifact = _validate_tensor_artifact(
        capture.get("captured_module_output_artifact"),
        label="captured module-output artifact",
        expected_path=(
            "artifacts/E004-qwen3-layer-capture/real-activation/"
            "layer-00-o-proj-captured-module-output.pt"
        ),
    )
    input_tensor = _mapping(input_artifact.get("tensor"), "input tensor")
    if (
        input_tensor.get("shape") != [9, 4096]
        or input_tensor.get("dtype") != "bfloat16"
        or input_tensor.get("stride") != [4096, 1]
        or input_tensor.get("storage_offset") != 0
    ):
        raise RealActivationFinalizationError("captured activation metadata changed")

    runtime = _mapping(run.get("runtime_projection"), "runtime projection")
    if (
        runtime.get("module_path") != "model.layers.0.self_attn.o_proj"
        or runtime.get("selected_kernel") != EXPECTED_KERNEL
        or runtime.get("weights_padding_cols") != 0
    ):
        raise RealActivationFinalizationError("runtime projection changed")

    replay = _mapping(run.get("replay"), "replay")
    replay_artifact = _validate_tensor_artifact(
        replay.get("replay_output_artifact"),
        label="replay output artifact",
        expected_path=(
            "artifacts/E004-qwen3-layer-capture/real-activation/"
            "layer-00-o-proj-replay-output.pt"
        ),
    )
    output_hashes = replay.get("output_sha256s")
    bitwise_matches = replay.get("bitwise_captured_module_output_matches")
    module_output_tensor = _mapping(
        module_output_artifact.get("tensor"), "captured module output tensor"
    )
    if not isinstance(output_hashes, list) or not isinstance(bitwise_matches, list):
        raise RealActivationFinalizationError("replay hashes are missing")
    if not all(
        (
            replay.get("warmup_runs") == 1,
            replay.get("repetitions") == 3,
            replay.get("synchronized") is True,
            replay.get("all_finite") is True,
            replay.get("output_hash_stable") is True,
            replay.get("output_shape") == [9, 4096],
            replay.get("output_dtype") == "bfloat16",
            len(output_hashes) == 3,
            len(set(output_hashes)) == 1,
            bitwise_matches == [True, True, True],
            replay.get("max_abs_error") == 0.0,
            replay.get("mean_abs_error") == 0.0,
            replay.get("input_sha256") == input_tensor.get("sha256"),
            replay.get("captured_module_output_sha256")
            == module_output_tensor.get("sha256"),
            output_hashes[0] == module_output_tensor.get("sha256"),
        )
    ):
        raise RealActivationFinalizationError("replay invariants changed")

    backend = _mapping(run.get("backend"), "backend")
    if (
        backend.get("requested_format") != "nvfp4"
        or backend.get("requested_backend") != "auto"
        or backend.get("selected_vllm_kernel") != EXPECTED_KERNEL
        or backend.get("reported_backend") is not None
        or backend.get("target_nvtx_range") != TARGET_RANGE
        or backend.get("fallback_status") != "pending_profiler"
    ):
        raise RealActivationFinalizationError("runtime backend identity changed")
    return TARGET_RANGE, (input_artifact, module_output_artifact, replay_artifact)


def _validate_local_artifacts(artifacts: tuple[dict[str, Any], ...]) -> None:
    for artifact in artifacts:
        path = ROOT / str(artifact["path"])
        if not path.is_file() or _sha256_path(path) != artifact["file_sha256"]:
            raise RealActivationFinalizationError(
                f"local artifact hash changed: {artifact['path']}"
            )


def _artifact(
    *,
    kind: str,
    path: Path,
    ignored: bool,
    sha256: str | None = None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256 or _sha256_path(path),
        "ignored": ignored,
    }


def main() -> int:
    args = parse_args()
    run = _json(args.run_evidence)
    range_name, tensor_artifacts = _validate_run(run)
    _validate_local_artifacts(tensor_artifacts)
    if not args.report.is_file():
        raise RealActivationFinalizationError("Nsight report is missing")
    if not FULL_MODEL_RESULT.is_file() or not FULL_MODEL_MANIFEST.is_file():
        raise RealActivationFinalizationError(
            "full-model acquisition dependencies are missing"
        )

    git = collect_git()
    if git.dirty:
        raise RealActivationFinalizationError(
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
    target_kernels = kernels_in_nvtx_range(evidence.observed_kernels, range_name)
    expected_signature = expected_sm120_cutlass_present(
        evidence.observed_kernels, range_name
    )
    fallback_status = assess_range_fallback(evidence.observed_kernels, range_name).value
    passed = bool(
        target_kernels and expected_signature and fallback_status == "not_detected"
    )
    backend = {
        "requested_format": "nvfp4",
        "requested_backend": "auto",
        "selected_vllm_kernel": EXPECTED_KERNEL,
        "reported_backend": None,
        "target_nvtx_range": range_name,
        "observed_kernels": list(evidence.observed_kernels),
        "target_kernels": list(target_kernels),
        "expected_sm120_cutlass_signature_present": expected_signature,
        "fallback_status": fallback_status,
        "profiler_sha256": evidence.report_sha256,
    }
    results = {
        **run,
        "slice": "real_activation_capture_replay_v1",
        "status": "pass" if passed else "inconclusive",
        "decision": "continue" if passed else "repeat",
        "backend": backend,
        "claim_boundary": (
            "For one fixed token-ID request, this result establishes capture of "
            "one real Qwen prefill activation with shape, dtype, stride, storage "
            "offset, and canonical logical bytes preserved across the recorded "
            "CUDA-to-CPU transfer, plus deterministic replay of the same unfused "
            "layer-0 o_proj through the range-attributed SM120 CUTLASS NVFP4 path. "
            "It does not establish NVFP4 numerical correctness, final-logit or "
            "model quality, other prompts, layers, or projections, cross-backend "
            "agreement, or equivalence to a high-precision checkpoint."
        ),
    }
    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

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
        "--output=.local/profiles/e004-real-activation-layer-00-o-proj",
        "--force-overwrite=true",
        "/home/meyowu/projects/nvfp4-doctor/.venv/bin/python",
        "scripts/run_e004_real_activation_capture.py",
        "--model-dir",
        f"models/nvidia--Qwen3-8B-NVFP4/{REVISION}",
        "--artifact-root",
        "artifacts/E004-qwen3-layer-capture/real-activation",
        "--output",
        ("artifacts/E004-qwen3-layer-capture/real-activation/layer-00-o-proj.json"),
        "--profile-capture",
    ]
    result_artifact = _artifact(
        kind="normalized-real-activation-result",
        path=args.results,
        ignored=False,
    )
    local_tensor_entries = [
        _artifact(
            kind=kind,
            path=ROOT / str(artifact["path"]),
            ignored=True,
            sha256=str(artifact["file_sha256"]),
        )
        for kind, artifact in zip(
            (
                "captured-activation",
                "captured-module-output",
                "replay-output",
            ),
            tensor_artifacts,
            strict=True,
        )
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "real_activation_capture_replay_v1",
        "captured_at_utc": run["captured_at_utc"],
        "git": {"commit": git.commit, "branch": branch, "dirty": git.dirty},
        "gpu": gpu,
        "software": software,
        "model": {
            "repository": "nvidia/Qwen3-8B-NVFP4",
            "revision": REVISION,
            "local_snapshot_path": run["model_load"]["local_snapshot_path"],
            "complete_snapshot_acquired": True,
            "replayed_layer": 0,
            "replayed_projection": "o_proj",
            "activation_provenance": "real_qwen_prefill",
        },
        "dependencies": [
            {
                "kind": "full-model-acquisition-result",
                "path": FULL_MODEL_RESULT.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(FULL_MODEL_RESULT),
            },
            {
                "kind": "full-model-acquisition-manifest",
                "path": FULL_MODEL_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(FULL_MODEL_MANIFEST),
            },
        ],
        "commands": {
            "workflow": ["bash", "scripts/run_e004_real_activation_profile.sh"],
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
                kind="raw-real-activation-observation",
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
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"target_kernel_count={len(target_kernels)}")
    print(f"expected_signature_present={expected_signature}")
    print(f"fallback_status={fallback_status}")
    print(f"profiler_sha256={evidence.report_sha256}")
    print(f"status={results['status']}")
    print(f"decision={results['decision']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
