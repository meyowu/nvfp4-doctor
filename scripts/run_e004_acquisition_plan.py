#!/usr/bin/env python3
"""Build a metadata-only byte-range plan for representative Qwen3 layers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from nvfp4_doctor.checkpoint import (
    SafetensorsHeader,
    parse_safetensors_header,
    plan_tensor_byte_ranges,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
METADATA_RESULTS = EXPERIMENT / "metadata.json"
RESULTS = EXPERIMENT / "acquisition-plan.json"
MANIFEST = EXPERIMENT / "manifest-acquisition-plan.json"
LOCAL_INDEX = (
    ROOT
    / "artifacts"
    / "E004-checkpoint-metadata"
    / "source"
    / "model.safetensors.index.json"
)
REPO_ID = "nvidia/Qwen3-8B-NVFP4"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
INDEX_SHA256 = "d7cddcad23c80b41201f23e3fccb22ef200ed551da067c61afb0b09f67864f47"
LAYERS = ((0, "early"), (18, "middle"), (35, "late"))
PROJECTIONS = (
    ("q_proj", "self_attn"),
    ("o_proj", "self_attn"),
    ("gate_proj", "mlp"),
    ("up_proj", "mlp"),
    ("down_proj", "mlp"),
)
SUFFIXES = ("input_scale", "weight", "weight_scale", "weight_scale_2")
MAX_HEADER_BYTES = 16 * 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "checkpoint" / "acquisition.py",
    ROOT / "tests" / "unit" / "test_acquisition_plan.py",
    Path(__file__).resolve(),
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _fetch_range(
    client: httpx.Client,
    url: str,
    start: int,
    end: int,
    expected_total: int,
) -> bytes:
    expected_length = end - start + 1
    with client.stream(
        "GET",
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
    ) as response:
        if response.status_code != 206:
            raise ValueError(
                f"range request returned {response.status_code}; refusing response body"
            )
        content_range = response.headers.get("content-range")
        match = _CONTENT_RANGE.fullmatch(content_range or "")
        if match is None or tuple(map(int, match.groups())) != (
            start,
            end,
            expected_total,
        ):
            raise ValueError("range response boundaries do not match the request")
        if response.headers.get("content-encoding") not in (None, "identity"):
            raise ValueError("range response uses an unexpected content encoding")
        content_length = response.headers.get("content-length")
        if content_length is None or int(content_length) != expected_length:
            raise ValueError("range response Content-Length is not exact")
        body = response.read()
    if len(body) != expected_length:
        raise ValueError("range response body length is not exact")
    return body


def _target_name(layer: int, projection: str, parent: str, suffix: str) -> str:
    return f"model.layers.{layer}.{parent}.{projection}.{suffix}"


def _selection() -> list[dict[str, object]]:
    return [
        {
            "layer": layer,
            "role": role,
            "rationale": (
                "boundary layer" if role != "middle" else "first layer of second half"
            ),
        }
        for layer, role in LAYERS
    ]


def main() -> int:
    if not LOCAL_INDEX.is_file():
        raise FileNotFoundError(
            "run scripts/run_e004_checkpoint_metadata.py first to acquire the pinned index"
        )
    if _sha256_path(LOCAL_INDEX) != INDEX_SHA256:
        raise ValueError(
            "local pinned safetensors index hash does not match E004 evidence"
        )

    prior = _json(METADATA_RESULTS)
    repository = prior.get("repository")
    if not isinstance(repository, dict) or repository.get("revision") != REVISION:
        raise ValueError("metadata result does not match the pinned revision")
    shard_records = prior.get("weight_shards")
    if not isinstance(shard_records, list) or len(shard_records) != 2:
        raise ValueError("metadata result must declare exactly two weight shards")

    headers: dict[str, SafetensorsHeader] = {}
    shard_sources: dict[str, dict[str, object]] = {}
    header_bytes_downloaded = 0
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for record in shard_records:
            if not isinstance(record, dict):
                raise ValueError("weight shard record must be an object")
            filename = str(record["path"])
            file_size = int(record["size_bytes"])
            url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{filename}"
            prefix = _fetch_range(client, url, 0, 7, file_size)
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 0 or header_length > MAX_HEADER_BYTES:
                raise ValueError(
                    "declared safetensors header length is outside the bound"
                )
            header = _fetch_range(client, url, 8, 7 + header_length, file_size)
            headers[filename] = parse_safetensors_header(prefix, header, file_size)
            header_bytes_downloaded += len(prefix) + len(header)
            shard_sources[filename] = {
                "source_url": url,
                "file_size": file_size,
                "lfs_sha256": record["lfs_sha256"],
            }

    requested = [
        _target_name(layer, projection, parent, suffix)
        for layer, _role in LAYERS
        for projection, parent in PROJECTIONS
        for suffix in SUFFIXES
    ]
    plan = plan_tensor_byte_ranges(headers, requested)
    if len(plan) != len(LAYERS) * len(PROJECTIONS) * len(SUFFIXES):
        raise ValueError("acquisition plan does not have complete target coverage")

    records: list[dict[str, object]] = []
    for item in plan:
        parts = item.tensor_name.split(".")
        projection = parts[-2]
        records.append(
            {
                "tensor_name": item.tensor_name,
                "layer": int(parts[2]),
                "projection": projection,
                "suffix": parts[-1],
                "shard": item.shard,
                "source_url": shard_sources[item.shard]["source_url"],
                "shard_lfs_sha256": shard_sources[item.shard]["lfs_sha256"],
                "dtype": item.dtype,
                "shape": list(item.shape),
                "file_start": item.file_start,
                "file_end_exclusive": item.file_end,
                "http_range": item.http_range,
                "byte_length": item.byte_length,
            }
        )

    planned_payload_bytes = sum(item.byte_length for item in plan)
    per_layer = {
        str(layer): sum(
            item.byte_length
            for item in plan
            if int(item.tensor_name.split(".")[2]) == layer
        )
        for layer, _role in LAYERS
    }
    results = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_acquisition_plan_v1",
        "status": "pass",
        "decision": "continue",
        "repository": {"id": REPO_ID, "revision": REVISION},
        "selection": _selection(),
        "coverage": {
            "layer_count": len(LAYERS),
            "projection_count": len(PROJECTIONS),
            "tensor_kinds_per_projection": len(SUFFIXES),
            "planned_tensor_count": len(plan),
        },
        "request_policy": {
            "metadata_only": True,
            "header_range_requests_executed": len(headers) * 2,
            "header_bytes_downloaded": header_bytes_downloaded,
            "payload_requests_executed": 0,
            "payload_bytes_downloaded": 0,
            "planned_payload_request_count": len(plan),
            "planned_payload_bytes": planned_payload_bytes,
        },
        "planned_bytes_by_layer": per_layer,
        "ranges": records,
        "observations": [
            "all selected tensor names occur exactly once in the pinned shard headers",
            "all planned HTTP ranges are non-empty, non-overlapping, and within shard boundaries",
            "the plan covers early, middle, and late layers for all five projection families",
            "no tensor payload request was executed while constructing the plan",
        ],
        "claim_boundary": (
            "This metadata-only plan identifies exact stored byte intervals. It does not "
            "validate payload hashes, decode checkpoint values, establish runtime layout or "
            "strides, capture activations, execute a backend, or demonstrate correctness."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "representative_acquisition_plan_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e004_acquisition_plan.py"],
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
            "payload_bytes_downloaded": 0,
            "planned_payload_bytes": planned_payload_bytes,
        },
        "backend": {
            "requested": "metadata_only_acquisition_planning",
            "reported": "http_206_header_ranges_only",
            "observed_kernel": None,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "tensor-acquisition-plan",
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(RESULTS),
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"selected_layers={','.join(str(layer) for layer, _role in LAYERS)}")
    print(f"planned_tensor_count={len(plan)}")
    print(f"planned_payload_bytes={planned_payload_bytes}")
    print(f"header_bytes_downloaded={header_bytes_downloaded}")
    print("payload_bytes_downloaded=0")
    print("status=pass")
    print("decision=continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
