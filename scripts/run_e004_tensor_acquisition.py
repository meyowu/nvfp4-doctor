#!/usr/bin/env python3
"""Acquire the authorized E004 tensor ranges into ignored local storage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
PLAN = EXPERIMENT / "acquisition-plan.json"
RESULTS = EXPERIMENT / "payloads.json"
MANIFEST = EXPERIMENT / "manifest-payloads.json"
LOCAL_ROOT = ROOT / "artifacts" / "E004-qwen3-layer-capture" / "tensor-payloads"
LOCAL_PROGRESS = LOCAL_ROOT / "acquisition-progress.json"
REPO_ID = "nvidia/Qwen3-8B-NVFP4"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
EXPECTED_TENSOR_COUNT = 60
EXPECTED_TOTAL_BYTES = 311_427_192
EXPECTED_LAYERS = {0, 18, 35}
EXPECTED_PROJECTIONS = {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
EXPECTED_SUFFIXES = {"input_scale", "weight", "weight_scale", "weight_scale_2"}
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "tests" / "unit" / "test_tensor_acquisition.py",
)


class TensorAcquisitionError(RuntimeError):
    """Raised when a remote range or local artifact violates the pinned plan."""


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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TensorAcquisitionError(f"{path.name} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorAcquisitionError(f"plan field {field} must be an integer")
    return value


def _string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise TensorAcquisitionError(f"plan field {field} must be a non-empty string")
    return value


def _validate_plan(plan: Mapping[str, object]) -> list[dict[str, object]]:
    repository = plan.get("repository")
    if repository != {"id": REPO_ID, "revision": REVISION}:
        raise TensorAcquisitionError(
            "plan does not match the pinned repository revision"
        )
    policy = plan.get("request_policy")
    if not isinstance(policy, dict):
        raise TensorAcquisitionError("plan request_policy must be an object")
    if policy.get("planned_payload_bytes") != EXPECTED_TOTAL_BYTES:
        raise TensorAcquisitionError("plan payload byte total changed")
    ranges = plan.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != EXPECTED_TENSOR_COUNT:
        raise TensorAcquisitionError("plan must contain exactly 60 ranges")

    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    coverage: set[tuple[int, str, str]] = set()
    total_bytes = 0
    for value in ranges:
        if not isinstance(value, dict):
            raise TensorAcquisitionError("each planned range must be an object")
        name = _string(value, "tensor_name")
        layer = _integer(value, "layer")
        projection = _string(value, "projection")
        suffix = _string(value, "suffix")
        start = _integer(value, "file_start")
        end = _integer(value, "file_end_exclusive")
        length = _integer(value, "byte_length")
        source_url = _string(value, "source_url")
        if name in names:
            raise TensorAcquisitionError(f"duplicate planned tensor: {name}")
        names.add(name)
        if layer not in EXPECTED_LAYERS:
            raise TensorAcquisitionError(f"unexpected planned layer: {layer}")
        if projection not in EXPECTED_PROJECTIONS or suffix not in EXPECTED_SUFFIXES:
            raise TensorAcquisitionError("planned projection or suffix is outside E004")
        if coverage & {(layer, projection, suffix)}:
            raise TensorAcquisitionError("duplicate layer/projection/suffix coverage")
        coverage.add((layer, projection, suffix))
        if start < 0 or end <= start or end - start != length:
            raise TensorAcquisitionError(f"invalid planned byte interval: {name}")
        if value.get("http_range") != f"bytes={start}-{end - 1}":
            raise TensorAcquisitionError(f"planned HTTP range is inconsistent: {name}")
        expected_prefix = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/"
        if not source_url.startswith(expected_prefix):
            raise TensorAcquisitionError(f"unexpected source URL: {name}")
        total_bytes += length
        normalized.append(dict(value))

    expected_coverage = {
        (layer, projection, suffix)
        for layer in EXPECTED_LAYERS
        for projection in EXPECTED_PROJECTIONS
        for suffix in EXPECTED_SUFFIXES
    }
    if coverage != expected_coverage or total_bytes != EXPECTED_TOTAL_BYTES:
        raise TensorAcquisitionError("planned coverage or byte total is incomplete")
    return normalized


def _artifact_path(record: Mapping[str, object]) -> Path:
    layer = _integer(record, "layer")
    projection = _string(record, "projection")
    suffix = _string(record, "suffix")
    return LOCAL_ROOT / f"layer-{layer:02d}" / projection / f"{suffix}.bin"


def _download_range(
    client: httpx.Client,
    record: Mapping[str, object],
    destination: Path,
) -> dict[str, object]:
    start = _integer(record, "file_start")
    end = _integer(record, "file_end_exclusive")
    expected_length = _integer(record, "byte_length")
    source_url = _string(record, "source_url")
    shard_size = _integer(record, "shard_file_size")
    requested_range = f"bytes={start}-{end - 1}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)

    digest = hashlib.sha256()
    observed_length = 0
    try:
        with client.stream(
            "GET",
            source_url,
            headers={"Range": requested_range, "Accept-Encoding": "identity"},
        ) as response:
            if response.status_code != 206:
                raise TensorAcquisitionError(
                    f"range request returned {response.status_code}; refusing body"
                )
            content_encoding = response.headers.get("content-encoding")
            if content_encoding not in (None, "identity"):
                raise TensorAcquisitionError("range response is unexpectedly encoded")
            content_range = response.headers.get("content-range")
            match = _CONTENT_RANGE.fullmatch(content_range or "")
            if match is None or tuple(map(int, match.groups())) != (
                start,
                end - 1,
                shard_size,
            ):
                raise TensorAcquisitionError("range response boundaries are not exact")
            content_length = response.headers.get("content-length")
            if content_length is None or int(content_length) != expected_length:
                raise TensorAcquisitionError(
                    "range response Content-Length is not exact"
                )

            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    observed_length += len(chunk)
                    if observed_length > expected_length:
                        raise TensorAcquisitionError(
                            "range response exceeded planned length"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if observed_length != expected_length:
            raise TensorAcquisitionError("range response body length is not exact")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "status_code": 206,
        "content_range": f"bytes {start}-{end - 1}/{shard_size}",
        "content_length": observed_length,
        "sha256": digest.hexdigest(),
    }


def _progress_entry_matches(
    record: Mapping[str, object],
    progress: Mapping[str, object],
    path: Path,
) -> bool:
    if not path.is_file():
        return False
    expected = {
        "tensor_name": record.get("tensor_name"),
        "source_url": record.get("source_url"),
        "shard": record.get("shard"),
        "shard_lfs_sha256": record.get("shard_lfs_sha256"),
        "http_range": record.get("http_range"),
        "byte_length": record.get("byte_length"),
        "dtype": record.get("dtype"),
        "shape": record.get("shape"),
    }
    if any(progress.get(key) != value for key, value in expected.items()):
        return False
    byte_length = _integer(record, "byte_length")
    sha256 = progress.get("sha256")
    return (
        path.stat().st_size == byte_length
        and isinstance(sha256, str)
        and _sha256_path(path) == sha256
    )


def main() -> int:
    plan = _json(PLAN)
    records = _validate_plan(plan)
    header_shards = _json(EXPERIMENT / "headers.json").get("shards")
    if not isinstance(header_shards, list):
        raise TensorAcquisitionError("header evidence shards must be a list")
    shard_sizes = {
        str(shard["path"]): int(shard["file_size"])
        for shard in header_shards
        if isinstance(shard, dict)
    }
    for record in records:
        shard = _string(record, "shard")
        if shard not in shard_sizes:
            raise TensorAcquisitionError(f"plan references an unknown shard: {shard}")
        record["shard_file_size"] = shard_sizes[shard]

    if LOCAL_PROGRESS.is_file():
        local = _json(LOCAL_PROGRESS)
    else:
        local = {
            "schema_version": 1,
            "repository": {"id": REPO_ID, "revision": REVISION},
            "artifacts": {},
        }
    if local.get("repository") != {"id": REPO_ID, "revision": REVISION}:
        raise TensorAcquisitionError("local progress belongs to another revision")
    progress_artifacts = local.get("artifacts")
    if not isinstance(progress_artifacts, dict):
        raise TensorAcquisitionError("local progress artifacts must be an object")

    downloaded_this_run = 0
    reused_this_run = 0
    timeout = httpx.Timeout(120.0, connect=30.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for index, record in enumerate(records, start=1):
            name = _string(record, "tensor_name")
            destination = _artifact_path(record)
            prior = progress_artifacts.get(name)
            if isinstance(prior, dict) and _progress_entry_matches(
                record, prior, destination
            ):
                reused_this_run += 1
                print(f"[{index:02d}/{len(records)}] verified {name}")
                continue

            response = _download_range(client, record, destination)
            byte_length = _integer(record, "byte_length")
            downloaded_this_run += byte_length
            entry = {
                "tensor_name": name,
                "source_url": record["source_url"],
                "shard": record["shard"],
                "shard_lfs_sha256": record["shard_lfs_sha256"],
                "http_range": record["http_range"],
                "byte_length": byte_length,
                "dtype": record["dtype"],
                "shape": record["shape"],
                "local_path": destination.relative_to(ROOT).as_posix(),
                "status_code": response["status_code"],
                "content_range": response["content_range"],
                "content_length": response["content_length"],
                "sha256": response["sha256"],
                "acquired_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            progress_artifacts[name] = entry
            _write_json_atomic(LOCAL_PROGRESS, local)
            print(f"[{index:02d}/{len(records)}] acquired {name}")

    selected = {
        name: progress_artifacts[name]
        for name in sorted(_string(record, "tensor_name") for record in records)
    }
    for name, entry in selected.items():
        if not isinstance(entry, dict):
            raise TensorAcquisitionError(f"invalid local progress entry: {name}")
        path_value = _string(entry, "local_path")
        if not _progress_entry_matches(
            next(record for record in records if record["tensor_name"] == name),
            entry,
            ROOT / path_value,
        ):
            raise TensorAcquisitionError(f"local artifact verification failed: {name}")

    payloads = [
        {
            key: entry[key]
            for key in (
                "tensor_name",
                "shard",
                "shard_lfs_sha256",
                "http_range",
                "byte_length",
                "dtype",
                "shape",
                "local_path",
                "status_code",
                "content_range",
                "content_length",
                "sha256",
            )
        }
        for entry in selected.values()
        if isinstance(entry, dict)
    ]
    total_bytes = sum(int(entry["byte_length"]) for entry in payloads)
    if len(payloads) != EXPECTED_TENSOR_COUNT or total_bytes != EXPECTED_TOTAL_BYTES:
        raise TensorAcquisitionError("verified local payload set is incomplete")

    results = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_tensor_acquisition_v1",
        "status": "pass",
        "decision": "continue",
        "repository": {"id": REPO_ID, "revision": REVISION},
        "request_policy": {
            "exact_http_ranges_only": True,
            "accept_encoding": "identity",
            "reject_non_partial_response_before_body_read": True,
        },
        "payload_request_count": len(payloads),
        "payload_bytes_downloaded": total_bytes,
        "weight_files_downloaded": False,
        "local_artifact_count": len(payloads),
        "all_lengths_exact": True,
        "all_local_hashes_recorded": True,
        "payloads": payloads,
        "observations": [
            "all 60 planned tensor ranges returned exact HTTP 206 boundaries",
            "all response and local file lengths matched the acquisition plan",
            "each ignored local tensor artifact has a recorded SHA-256",
            "only selected tensor ranges were acquired; neither complete shard was downloaded",
        ],
        "claim_boundary": (
            "Successful byte acquisition establishes reproducible stored payloads only. "
            "It does not establish logical NVFP4 layout, runtime strides, scale mapping, "
            "activation capture, backend support, kernel identity, or numerical correctness."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_tensor_acquisition_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e004_tensor_acquisition.py"],
            "cwd": str(ROOT),
        },
        "git": {
            "branch": _command("git", "branch", "--show-current"),
            "commit": _command("git", "rev-parse", "HEAD"),
            "dirty": bool(git_status),
        },
        "platform": {
            "os": platform.system(),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "software": {
            "httpx": _package_version("httpx"),
            "torch": _package_version("torch"),
            "vllm": _package_version("vllm"),
            "flashinfer": _package_version("flashinfer-python"),
        },
        "model": {
            "repository": REPO_ID,
            "revision": REVISION,
            "weight_files_downloaded": False,
            "tensor_payload_count": len(payloads),
            "payload_bytes_downloaded": total_bytes,
        },
        "backend": {
            "requested": "http_range_tensor_acquisition",
            "reported": "http_206_partial_content",
            "observed_kernel": None,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "tensor-payload-inventory",
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(RESULTS),
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"payload_request_count={len(payloads)}")
    print(f"payload_bytes_downloaded={total_bytes}")
    print(f"downloaded_this_run={downloaded_this_run}")
    print(f"reused_this_run={reused_this_run}")
    print("status=pass")
    print("decision=continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
