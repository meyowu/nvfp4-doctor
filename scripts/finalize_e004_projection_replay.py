#!/usr/bin/env python3
"""Attach Nsight evidence and finalize the E004 single-projection replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from nvfp4_doctor.backends import (
    assess_range_fallback,
    expected_sm120_cutlass_present,
    extract_kernel_evidence,
    kernels_in_nvtx_range,
)
from nvfp4_doctor.env import collect_git, collect_gpu, collect_software

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
DEFAULT_RUN = (
    ROOT / "artifacts" / "E004-qwen3-layer-capture" / "replay" / "layer-00-o-proj.json"
)
DEFAULT_REPORT = ROOT / ".local" / "profiles" / "e004-layer-00-o-proj.nsys-rep"
DEFAULT_RESULTS = EXPERIMENT / "replay-single-projection.json"
DEFAULT_MANIFEST = EXPERIMENT / "manifest-replay.json"
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "checkpoint" / "payload.py",
    ROOT / "src" / "nvfp4_doctor" / "formats" / "layout.py",
    ROOT / "src" / "nvfp4_doctor" / "backends" / "nsys.py",
    ROOT / "scripts" / "run_e004_projection_replay.py",
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_e004_projection_profile.sh",
    ROOT / "tests" / "unit" / "test_projection_payload.py",
    ROOT / "tests" / "unit" / "test_nsys_evidence.py",
    ROOT / "tests" / "unit" / "test_e004_replay_finalization.py",
)


class ReplayFinalizationError(RuntimeError):
    """Raised when runtime or profiler evidence is incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-evidence", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


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
        raise ReplayFinalizationError(f"could not read {path}") from error
    if not isinstance(value, dict):
        raise ReplayFinalizationError(f"{path} must contain an object")
    return value


def _command(*argv: str, timeout: int = 120) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.replace("\x00", "").strip()


def _validate_run(run: dict[str, object]) -> tuple[str, dict[str, object]]:
    if run.get("status") != "pass" or run.get("decision") != "pending_profiler":
        raise ReplayFinalizationError("runtime replay did not pass")
    case = run.get("case")
    replay = run.get("replay")
    backend = run.get("backend")
    if not isinstance(case, dict) or (
        case.get("layer"),
        case.get("projection"),
        case.get("activation_provenance"),
    ) != (0, "o_proj", "synthetic_deterministic"):
        raise ReplayFinalizationError("runtime case is not the frozen first replay")
    if not isinstance(replay, dict) or not all(
        (
            replay.get("repetitions") == 3,
            replay.get("all_finite") is True,
            replay.get("output_hash_stable") is True,
            replay.get("output_shape") == [16, 4096],
            replay.get("output_dtype") == "bfloat16",
        )
    ):
        raise ReplayFinalizationError("runtime replay invariants are incomplete")
    if not isinstance(backend, dict):
        raise ReplayFinalizationError("runtime backend evidence is missing")
    if (
        backend.get("requested_backend") != "cutlass"
        or backend.get("selected_vllm_kernel") != "FlashInferCutlassNvFp4LinearKernel"
        or backend.get("reported_backend") is not None
    ):
        raise ReplayFinalizationError("runtime backend identity fields changed")
    range_name = backend.get("target_nvtx_range")
    if not isinstance(range_name, str) or not range_name:
        raise ReplayFinalizationError("target NVTX range is missing")
    return range_name, backend


def main() -> int:
    args = parse_args()
    run = _json(args.run_evidence)
    range_name, _runtime_backend = _validate_run(run)
    if not args.report.is_file():
        raise ReplayFinalizationError("Nsight report is missing")

    git = collect_git()
    if git.dirty:
        raise ReplayFinalizationError(
            "finalization requires a clean implementation commit"
        )
    stats_argv = (
        "nsys",
        "stats",
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
        "requested_backend": "cutlass",
        "selected_vllm_kernel": "FlashInferCutlassNvFp4LinearKernel",
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
        "slice": "single_projection_replay_v1",
        "status": "pass" if passed else "inconclusive",
        "decision": "continue" if passed else "repeat",
        "backend": backend,
        "claim_boundary": (
            "The observed result establishes deterministic replay of one real layer-0 "
            "o_proj weight with a synthetic activation through the selected vLLM "
            "FlashInfer-CUTLASS path. It does not establish real Qwen activation "
            "capture, numerical correctness, model accuracy, or general backend support."
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
        "--output=.local/profiles/e004-layer-00-o-proj",
        "--force-overwrite=true",
        ".venv/bin/python",
        "scripts/run_e004_projection_replay.py",
        "--layer",
        "0",
        "--projection",
        "o_proj",
        "--rows",
        "16",
        "--seed",
        "0",
        "--repetitions",
        "3",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "single_projection_replay_v1",
        "captured_at_utc": run["captured_at_utc"],
        "git": {"commit": git.commit, "branch": branch, "dirty": git.dirty},
        "gpu": gpu,
        "software": software,
        "model": {
            "repository": "nvidia/Qwen3-8B-NVFP4",
            "revision": "ccd10a893cbca613259517c3efe08e151ddf2b8e",
            "weight_files_downloaded": False,
            "replayed_layer": 0,
            "replayed_projection": "o_proj",
            "activation_provenance": "synthetic_deterministic",
        },
        "commands": {
            "profile": profile_argv,
            "stats": list(stats_argv),
            "runtime": run["command"],
        },
        "backend": backend,
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "normalized-replay-result",
                "path": args.results.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(args.results),
            },
            {
                "kind": "raw-replay-observation",
                "path": args.run_evidence.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(args.run_evidence),
            },
            {
                "kind": "nsight-systems-report",
                "path": args.report.relative_to(ROOT).as_posix(),
                "sha256": evidence.report_sha256,
            },
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
