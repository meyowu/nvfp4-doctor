#!/usr/bin/env python3
"""Inspect pinned safetensors headers with bounded HTTP range requests."""

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

from nvfp4_doctor.checkpoint import parse_safetensors_header

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
METADATA_RESULTS = EXPERIMENT / "metadata.json"
RESULTS = EXPERIMENT / "headers.json"
MANIFEST = EXPERIMENT / "manifest-headers.json"
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
TARGET_PROJECTIONS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TARGET_SUFFIXES = ("input_scale", "weight", "weight_scale", "weight_scale_2")
MAX_HEADER_BYTES = 16 * 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "checkpoint" / "safetensors.py",
    ROOT / "tests" / "unit" / "test_safetensors_header.py",
    ROOT / "tests" / "unit" / "test_e004_header_evidence.py",
    Path(__file__).resolve(),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
) -> tuple[bytes, dict[str, object]]:
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
        content_encoding = response.headers.get("content-encoding")
        if content_encoding not in (None, "identity"):
            raise ValueError("range response uses an unexpected content encoding")
        content_range = response.headers.get("content-range")
        match = _CONTENT_RANGE.fullmatch(content_range or "")
        if match is None:
            raise ValueError("range response has an invalid Content-Range")
        observed_start, observed_end, observed_total = map(int, match.groups())
        if (observed_start, observed_end, observed_total) != (
            start,
            end,
            expected_total,
        ):
            raise ValueError("range response boundaries do not match the request")
        content_length = response.headers.get("content-length")
        if content_length is None or int(content_length) != expected_length:
            raise ValueError("range response Content-Length is not exact")
        body = response.read()
    if len(body) != expected_length:
        raise ValueError("range response body length is not exact")
    return body, {
        "status_code": 206,
        "requested_range": f"bytes={start}-{end}",
        "content_range": content_range,
        "content_length": expected_length,
    }


def _projection_inventory(
    tensors: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    inventories: list[dict[str, object]] = []
    for projection in TARGET_PROJECTIONS:
        projection_tensors = {
            name: value for name, value in tensors.items() if f".{projection}." in name
        }
        layers = {
            int(name.split(".")[2])
            for name in projection_tensors
            if name.startswith("model.layers.")
        }
        if layers != set(range(36)):
            raise ValueError(f"{projection} does not cover all 36 layers")

        suffix_records: list[dict[str, object]] = []
        for suffix in TARGET_SUFFIXES:
            selected = {
                name: value
                for name, value in projection_tensors.items()
                if name.endswith(f".{suffix}")
            }
            if len(selected) != 36:
                raise ValueError(f"{projection}.{suffix} does not have 36 tensors")
            descriptors = {
                (str(value["dtype"]), tuple(value["shape"]))
                for value in selected.values()
            }
            if len(descriptors) != 1:
                raise ValueError(f"{projection}.{suffix} shape/dtype is not uniform")
            dtype, shape = descriptors.pop()
            suffix_records.append(
                {
                    "suffix": suffix,
                    "dtype": dtype,
                    "shape": list(shape),
                    "tensor_count": len(selected),
                    "shards": sorted(
                        {str(value["shard"]) for value in selected.values()}
                    ),
                }
            )
        inventories.append(
            {
                "projection": projection,
                "layer_count": len(layers),
                "tensor_count": len(projection_tensors),
                "tensors": suffix_records,
            }
        )
    return inventories


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
    repository = prior["repository"]
    if not isinstance(repository, dict) or repository.get("revision") != REVISION:
        raise ValueError("metadata result does not match the pinned revision")
    shard_records = prior["weight_shards"]
    if not isinstance(shard_records, list) or len(shard_records) != 2:
        raise ValueError("metadata result must declare exactly two weight shards")
    index = _json(LOCAL_INDEX)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("pinned index weight_map must be an object")

    shard_evidence: list[dict[str, object]] = []
    all_tensors: dict[str, dict[str, object]] = {}
    range_bytes_downloaded = 0
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for record in shard_records:
            if not isinstance(record, dict):
                raise ValueError("weight shard record must be an object")
            filename = str(record["path"])
            file_size = int(record["size_bytes"])
            url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{filename}"

            prefix, prefix_range = _fetch_range(client, url, 0, 7, file_size)
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 0 or header_length > MAX_HEADER_BYTES:
                raise ValueError(
                    "declared safetensors header length is outside the bound"
                )
            header_end = 7 + header_length
            header_bytes, header_range = _fetch_range(
                client,
                url,
                8,
                header_end,
                file_size,
            )
            parsed = parse_safetensors_header(prefix, header_bytes, file_size)
            names = {tensor.name for tensor in parsed.tensors}
            indexed_names = {
                name for name, shard in weight_map.items() if shard == filename
            }
            if names != indexed_names:
                raise ValueError(
                    f"{filename} header names do not match the pinned index"
                )

            for tensor in parsed.tensors:
                all_tensors[tensor.name] = {
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "data_start": tensor.data_start,
                    "data_end": tensor.data_end,
                    "shard": filename,
                }
            downloaded = len(prefix) + len(header_bytes)
            range_bytes_downloaded += downloaded
            shard_evidence.append(
                {
                    "path": filename,
                    "source_url": url,
                    "file_size": file_size,
                    "lfs_sha256": record["lfs_sha256"],
                    "prefix_range": prefix_range,
                    "header_range": header_range,
                    "header_length": parsed.header_length,
                    "header_region_bytes": downloaded,
                    "header_region_sha256": _sha256_bytes(prefix + header_bytes),
                    "payload_start": parsed.payload_start,
                    "payload_bytes": parsed.payload_bytes,
                    "tensor_count": len(parsed.tensors),
                    "index_names_match": True,
                    "payload_boundaries_exact": True,
                }
            )

    if set(all_tensors) != set(weight_map):
        raise ValueError("combined headers do not match the full pinned index")
    projections = _projection_inventory(all_tensors)
    results = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "safetensors_headers_v1",
        "status": "pass",
        "decision": "continue",
        "repository": {"id": REPO_ID, "revision": REVISION},
        "request_policy": {
            "http_range_only": True,
            "maximum_header_bytes_per_shard": MAX_HEADER_BYTES,
            "reject_non_partial_response_before_body_read": True,
            "accept_encoding": "identity",
        },
        "shards": shard_evidence,
        "range_request_count": len(shard_evidence) * 2,
        "range_bytes_downloaded": range_bytes_downloaded,
        "payload_bytes_downloaded": 0,
        "weight_files_downloaded": False,
        "combined_tensor_count": len(all_tensors),
        "combined_index_names_match": True,
        "capture_target_tensor_count": sum(
            int(projection["tensor_count"]) for projection in projections
        ),
        "capture_target_projections": projections,
        "observations": [
            "both shard URLs honored exact prefix and JSON-header byte ranges",
            "header tensor names exactly match the pinned safetensors index",
            "all tensor intervals are contiguous and exactly cover each shard payload",
            "the five capture targets have uniform stored dtypes and shapes across 36 layers",
        ],
        "claim_boundary": (
            "Stored U8 weight shapes describe packed checkpoint bytes, not logical "
            "matrix shapes or runtime strides. Header inspection does not read payloads, "
            "load tensors, establish scale layout semantics, execute a backend, or "
            "demonstrate numerical correctness."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "safetensors_headers_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e004_safetensors_headers.py"],
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
        },
        "backend": {
            "requested": "http_range_header_only",
            "reported": "http_206_partial_content",
            "observed_kernel": None,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "safetensors-header-inspection",
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(RESULTS),
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"range_request_count={results['range_request_count']}")
    print(f"range_bytes_downloaded={range_bytes_downloaded}")
    print(f"payload_bytes_downloaded={results['payload_bytes_downloaded']}")
    print(f"combined_tensor_count={len(all_tensors)}")
    print(f"capture_target_tensor_count={results['capture_target_tensor_count']}")
    print(f"status={results['status']}")
    print(f"decision={results['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
