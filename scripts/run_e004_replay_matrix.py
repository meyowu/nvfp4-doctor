#!/usr/bin/env python3
"""Run the frozen E004 early/middle/late projection replay matrix."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from nvfp4_doctor.env import collect_git, collect_gpu, collect_software

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
RESULTS = EXPERIMENT / "replay-matrix.json"
MANIFEST = EXPERIMENT / "manifest-replay-matrix.json"
LOCAL_ROOT = ROOT / "artifacts" / "E004-qwen3-layer-capture" / "replay-matrix"
SINGLE_REPLAY = EXPERIMENT / "replay-single-projection.json"
LAYERS = ((0, "early"), (18, "middle"), (35, "late"))
PROJECTIONS = (
    ("q_proj", "individual_fused_family_preflight"),
    ("o_proj", "production_aligned_unfused"),
    ("gate_proj", "individual_fused_family_preflight"),
    ("up_proj", "individual_fused_family_preflight"),
    ("down_proj", "production_aligned_unfused"),
)
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "checkpoint" / "payload.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "layout.py",
    ROOT / "scripts" / "run_e004_projection_replay.py",
    Path(__file__).resolve(),
)


class ReplayMatrixError(RuntimeError):
    """Raised when a frozen matrix case fails its replay contract."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayMatrixError(f"could not read {path}") from error
    if not isinstance(value, dict):
        raise ReplayMatrixError(f"{path} must contain an object")
    return value


def _command(*argv: str) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.replace("\x00", "").strip()


def _validate_case(
    run: dict[str, object],
    *,
    layer: int,
    projection: str,
) -> None:
    case = run.get("case")
    replay = run.get("replay")
    backend = run.get("backend")
    transforms = run.get("transforms")
    if run.get("status") != "pass" or run.get("decision") != "pending_profiler":
        raise ReplayMatrixError("case runtime status is not pass")
    if not isinstance(case, dict) or (
        case.get("layer"),
        case.get("projection"),
        case.get("seed"),
    ) != (layer, projection, 0):
        raise ReplayMatrixError("case identity changed")
    if not isinstance(replay, dict) or not all(
        (
            replay.get("repetitions") == 3,
            replay.get("all_finite") is True,
            replay.get("output_hash_stable") is True,
        )
    ):
        raise ReplayMatrixError("case replay invariants failed")
    output_hashes = replay.get("output_sha256s")
    if (
        not isinstance(output_hashes, list)
        or len(output_hashes) != 3
        or len(set(output_hashes)) != 1
    ):
        raise ReplayMatrixError("case output hashes are incomplete")
    if not isinstance(backend, dict) or (
        backend.get("requested_backend"),
        backend.get("selected_vllm_kernel"),
        backend.get("reported_backend"),
    ) != ("cutlass", "FlashInferCutlassNvFp4LinearKernel", None):
        raise ReplayMatrixError("case backend fields changed")
    if not isinstance(transforms, list):
        raise ReplayMatrixError("case transforms are missing")
    indexed = {item.get("name"): item for item in transforms if isinstance(item, dict)}
    weight = indexed.get("packed_weight_materialization")
    scale = indexed.get("weight_scale_swizzle")
    if (
        not isinstance(weight, dict)
        or weight.get("source_sha256") != weight.get("destination_sha256")
        or weight.get("padding_bytes") != 0
    ):
        raise ReplayMatrixError("case weight transform changed bytes or added padding")
    if (
        not isinstance(scale, dict)
        or scale.get("vllm_candidate_byte_exact") is not True
    ):
        raise ReplayMatrixError("case scale transform disagrees with vLLM")


def _run_case(layer: int, projection: str, output: Path) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    argv = (
        sys.executable,
        str(ROOT / "scripts" / "run_e004_projection_replay.py"),
        "--layer",
        str(layer),
        "--projection",
        projection,
        "--rows",
        "16",
        "--seed",
        "0",
        "--repetitions",
        "3",
        "--output",
        str(output),
    )
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise ReplayMatrixError(
            f"case failed: layer={layer} projection={projection}\n{completed.stderr}"
        )
    run = _json(output)
    _validate_case(run, layer=layer, projection=projection)
    return run


def _case_summary(
    run: dict[str, object],
    *,
    role: str,
    adapter_scope: str,
    local_path: Path,
) -> dict[str, object]:
    replay = run["replay"]
    backend = run["backend"]
    transforms = {
        item["name"]: item for item in run["transforms"] if isinstance(item, dict)
    }
    runtime = {
        item["name"]: item for item in run["runtime_tensors"] if isinstance(item, dict)
    }
    return {
        "layer": run["case"]["layer"],
        "role": role,
        "projection": run["case"]["projection"],
        "adapter_scope": adapter_scope,
        "activation_provenance": run["case"]["activation_provenance"],
        "logical_mnk": run["case"]["logical_mnk"],
        "source_tensor_sha256": {
            item["suffix"]: item["sha256"] for item in run["source_tensors"]
        },
        "weight_shape": runtime["weight"]["logical_shape"],
        "packed_weight_shape": runtime["weight"]["physical_shape"],
        "weight_scale_shape": runtime["weight_scale"]["logical_shape"],
        "output_shape": replay["output_shape"],
        "output_dtype": replay["output_dtype"],
        "output_sha256": replay["output_sha256s"][0],
        "output_hash_stable": replay["output_hash_stable"],
        "all_finite": replay["all_finite"],
        "weight_bytes_preserved": (
            transforms["packed_weight_materialization"]["source_sha256"]
            == transforms["packed_weight_materialization"]["destination_sha256"]
        ),
        "weight_padding_bytes": transforms["packed_weight_materialization"][
            "padding_bytes"
        ],
        "scale_swizzle_candidate_byte_exact": transforms["weight_scale_swizzle"][
            "vllm_candidate_byte_exact"
        ],
        "runtime_weight_scale_sha256": transforms["weight_scale_swizzle"][
            "destination_sha256"
        ],
        "input_scale": run["scaling"]["input_scale"],
        "weight_scale_2": run["scaling"]["weight_scale_2"],
        "alpha_runtime_f32": run["scaling"]["alpha_runtime_f32"],
        "activation_max_abs": run["scaling"]["activation_max_abs"],
        "checkpoint_calibrated_max_abs": run["scaling"][
            "checkpoint_calibrated_max_abs"
        ],
        "requested_backend": backend["requested_backend"],
        "selected_vllm_kernel": backend["selected_vllm_kernel"],
        "reported_backend": backend["reported_backend"],
        "raw_evidence_path": local_path.relative_to(ROOT).as_posix(),
        "raw_evidence_sha256": _sha256_path(local_path),
    }


def main() -> int:
    git = collect_git()
    if git.dirty:
        raise ReplayMatrixError("matrix requires a clean implementation commit")
    profile_anchor = _json(SINGLE_REPLAY)
    anchor_backend = profile_anchor.get("backend")
    if not isinstance(anchor_backend, dict) or (
        anchor_backend.get("fallback_status") != "not_detected"
        or anchor_backend.get("expected_sm120_cutlass_signature_present") is not True
    ):
        raise ReplayMatrixError("single-projection profiler anchor is incomplete")

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    raw_paths: list[Path] = []
    total = len(LAYERS) * len(PROJECTIONS)
    index = 0
    for layer, role in LAYERS:
        for projection, adapter_scope in PROJECTIONS:
            index += 1
            output = LOCAL_ROOT / f"layer-{layer:02d}-{projection}.json"
            print(
                f"[{index:02d}/{total}] layer={layer} projection={projection}",
                flush=True,
            )
            run = _run_case(layer, projection, output)
            cases.append(
                _case_summary(
                    run,
                    role=role,
                    adapter_scope=adapter_scope,
                    local_path=output,
                )
            )
            raw_paths.append(output)

    expected_coverage = {
        (layer, projection)
        for layer, _role in LAYERS
        for projection, _scope in PROJECTIONS
    }
    coverage = {(case["layer"], case["projection"]) for case in cases}
    if coverage != expected_coverage:
        raise ReplayMatrixError("matrix coverage is incomplete")
    output_hashes = [str(case["output_sha256"]) for case in cases]
    results = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_projection_replay_matrix_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "decision": "continue",
        "matrix": {
            "layers": [layer for layer, _role in LAYERS],
            "projections": [projection for projection, _scope in PROJECTIONS],
            "case_count": len(cases),
            "repetitions_per_case": 3,
            "all_cases_finite": all(bool(case["all_finite"]) for case in cases),
            "all_case_hashes_stable": all(
                bool(case["output_hash_stable"]) for case in cases
            ),
            "all_weights_byte_preserved": all(
                bool(case["weight_bytes_preserved"]) for case in cases
            ),
            "all_weight_padding_zero": all(
                case["weight_padding_bytes"] == 0 for case in cases
            ),
            "all_scale_swizzles_candidate_exact": all(
                bool(case["scale_swizzle_candidate_byte_exact"]) for case in cases
            ),
            "distinct_output_sha256_count": len(set(output_hashes)),
        },
        "cases": cases,
        "backend_identity_anchor": {
            "case": {"layer": 0, "projection": "o_proj"},
            "requested_backend": anchor_backend["requested_backend"],
            "selected_vllm_kernel": anchor_backend["selected_vllm_kernel"],
            "reported_backend": anchor_backend["reported_backend"],
            "target_nvtx_range": anchor_backend["target_nvtx_range"],
            "expected_sm120_cutlass_signature_present": anchor_backend[
                "expected_sm120_cutlass_signature_present"
            ],
            "fallback_status": anchor_backend["fallback_status"],
            "profiler_sha256": anchor_backend["profiler_sha256"],
        },
        "claim_boundary": (
            "The matrix replays real acquired weights with deterministic synthetic "
            "activations. o_proj and down_proj are production-aligned unfused cases; "
            "q_proj, gate_proj, and up_proj are individual fused-family kernel "
            "preflights. The matrix does not establish real Qwen activation capture, "
            "numerical correctness, or model-level accuracy."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_projection_replay_matrix_v1",
        "captured_at_utc": results["captured_at_utc"],
        "git": {
            "commit": git.commit,
            "branch": _command("git", "branch", "--show-current"),
            "dirty": git.dirty,
        },
        "host": {
            "os": _command("lsb_release", "-ds").strip('"'),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "gpu": asdict(collect_gpu()),
        "software": asdict(collect_software()),
        "model": {
            "repository": "nvidia/Qwen3-8B-NVFP4",
            "revision": "ccd10a893cbca613259517c3efe08e151ddf2b8e",
            "weight_files_downloaded": False,
            "acquired_tensor_count": 60,
            "replayed_case_count": len(cases),
        },
        "command": {
            "argv": [sys.executable, *sys.argv],
            "cwd": str(ROOT),
            "subprocess": "scripts/run_e004_projection_replay.py per matrix case",
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "backend_identity_anchor": results["backend_identity_anchor"],
        "artifacts": [
            {
                "kind": "normalized-replay-matrix",
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(RESULTS),
            },
            {
                "kind": "profiled-single-replay",
                "path": SINGLE_REPLAY.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(SINGLE_REPLAY),
            },
            *(
                {
                    "kind": "raw-matrix-case",
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": _sha256_path(path),
                }
                for path in raw_paths
            ),
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"case_count={len(cases)}")
    print(f"distinct_output_sha256_count={len(set(output_hashes))}")
    print("all_weights_byte_preserved=true")
    print("all_scale_swizzles_candidate_exact=true")
    print("status=pass")
    print("decision=continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
