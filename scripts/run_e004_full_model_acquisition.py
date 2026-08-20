#!/usr/bin/env python3
"""Validate and record the complete pinned Qwen3-8B-NVFP4 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from nvfp4_doctor.env import collect_git

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
REPO_ID = "nvidia/Qwen3-8B-NVFP4"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
MODEL_RELATIVE = Path("models") / "nvidia--Qwen3-8B-NVFP4" / REVISION
DEFAULT_MODEL_DIR = ROOT / MODEL_RELATIVE
DEFAULT_RESULTS = EXPERIMENT / "full-model-acquisition.json"
DEFAULT_MANIFEST = EXPERIMENT / "manifest-full-model-acquisition.json"
EXPECTED_FILE_COUNT = 15
EXPECTED_TOTAL_BYTES = 6_413_063_143
EXPECTED_WEIGHT_BYTES = 6_397_066_384
EXPECTED_LFS_SHA256 = {
    "model-00001-of-00002.safetensors": (
        "6c13ef7322f4e5460858782e32da7e34b6c6fa8148cbeb70abcd2b44455d43f0"
    ),
    "model-00002-of-00002.safetensors": (
        "cf084e6b0e9f4bed9d15b6a454c34c0a1e8c4b74668db62b4063defc5a601c96"
    ),
    "tokenizer.json": (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    ),
}
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "tests" / "unit" / "test_e004_full_model_acquisition.py",
)


class FullModelAcquisitionError(RuntimeError):
    """Raised when the local pinned snapshot is incomplete or inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_id(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullModelAcquisitionError(f"could not read {path}") from error
    if not isinstance(value, dict):
        raise FullModelAcquisitionError(f"{path} must contain a JSON object")
    return value


def _role(path: str) -> str:
    if path.endswith(".safetensors"):
        return "weight_shard"
    if path in {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }:
        return "tokenizer"
    return "model_metadata"


def _inspect_snapshot(
    model_dir: Path, tree: dict[str, object]
) -> tuple[list[dict[str, object]], int]:
    files_value = tree.get("files")
    if not isinstance(files_value, dict) or not files_value:
        raise FullModelAcquisitionError("HF tree metadata has no file inventory")

    expected = set(files_value)
    local_files = {path.name for path in model_dir.iterdir() if path.is_file()}
    unexpected_directories = {
        path.name
        for path in model_dir.iterdir()
        if path.is_dir() and path.name != ".cache"
    }
    if local_files != expected or unexpected_directories:
        raise FullModelAcquisitionError(
            "top-level snapshot inventory differs from the pinned HF tree"
        )

    records: list[dict[str, object]] = []
    for relative_path in sorted(expected):
        metadata = files_value[relative_path]
        if not isinstance(metadata, dict):
            raise FullModelAcquisitionError(f"invalid metadata for {relative_path}")
        path = model_dir / relative_path
        size = path.stat().st_size
        if size != metadata.get("size"):
            raise FullModelAcquisitionError(f"size mismatch for {relative_path}")
        sha256 = _sha256_path(path)
        lfs_sha256 = metadata.get("lfs_sha256")
        if lfs_sha256 is not None:
            if not isinstance(lfs_sha256, str) or sha256 != lfs_sha256:
                raise FullModelAcquisitionError(
                    f"LFS SHA-256 mismatch for {relative_path}"
                )
        else:
            blob_id = metadata.get("blob_id")
            if (
                not isinstance(blob_id, str)
                or _git_blob_id(path.read_bytes()) != blob_id
            ):
                raise FullModelAcquisitionError(
                    f"Git blob mismatch for {relative_path}"
                )
        records.append(
            {
                "path": relative_path,
                "role": _role(relative_path),
                "size_bytes": size,
                "sha256": sha256,
                "git_blob_id": metadata.get("blob_id"),
                "lfs_sha256": lfs_sha256,
            }
        )

    cache_root = model_dir / ".cache"
    cache_files = sum(1 for path in cache_root.rglob("*") if path.is_file())
    return records, cache_files


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _record_size(record: dict[str, object]) -> int:
    size = record.get("size_bytes")
    if not isinstance(size, int):
        raise FullModelAcquisitionError("snapshot record has no integer size")
    return size


def _git_branch() -> str:
    completed = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    branch = completed.stdout.strip()
    if not branch:
        raise FullModelAcquisitionError("could not determine the Git branch")
    return branch


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.absolute()
    try:
        local_root = model_dir.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise FullModelAcquisitionError(
            "model snapshot must remain under the repository root"
        ) from error
    if not local_root.startswith("models/"):
        raise FullModelAcquisitionError("model snapshot must remain under models/")

    tree_path = model_dir / ".cache" / "huggingface" / "trees" / f"{REVISION}.json"
    tree = _json_object(tree_path)
    if tree.get("format_version") != 1:
        raise FullModelAcquisitionError("unsupported HF tree metadata version")
    records, cache_files = _inspect_snapshot(model_dir, tree)
    total_bytes = sum(_record_size(record) for record in records)
    weight_bytes = sum(
        _record_size(record) for record in records if record["role"] == "weight_shard"
    )
    if (
        len(records) != EXPECTED_FILE_COUNT
        or total_bytes != EXPECTED_TOTAL_BYTES
        or weight_bytes != EXPECTED_WEIGHT_BYTES
    ):
        raise FullModelAcquisitionError("pinned snapshot totals changed")
    observed_lfs = {
        str(record["path"]): str(record["sha256"])
        for record in records
        if record["path"] in EXPECTED_LFS_SHA256
    }
    if observed_lfs != EXPECTED_LFS_SHA256:
        raise FullModelAcquisitionError("pinned LFS identities changed")

    captured_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    result = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "full_model_snapshot_acquisition_v1",
        "captured_at_utc": captured_at,
        "status": "pass",
        "decision": "continue",
        "repository": {
            "id": REPO_ID,
            "requested_revision": REVISION,
            "resolved_sha": REVISION,
        },
        "snapshot": {
            "local_root": local_root,
            "ignored": True,
            "inventory_exclusions": [".cache/"],
            "download_cache_metadata_files": cache_files,
            "file_count": len(records),
            "total_bytes": total_bytes,
            "weight_shard_count": sum(
                record["role"] == "weight_shard" for record in records
            ),
            "weight_bytes": weight_bytes,
            "tokenizer_and_small_file_bytes": total_bytes - weight_bytes,
            "files": records,
        },
        "verification": {
            "hf_cache_verify_files_checked": EXPECTED_FILE_COUNT,
            "hf_cache_verify_checksums_matched": True,
            "hf_cache_extra_files_are_download_metadata": True,
            "repository_files_complete": True,
            "top_level_inventory_exact": True,
            "all_sizes_exact": True,
            "all_local_sha256_recorded": True,
            "all_lfs_sha256_matched": True,
            "ordinary_git_blob_ids_matched": True,
        },
        "observations": [
            "All 15 files from the immutable repository revision are present.",
            "Ordinary files match their Git blob IDs and all three LFS payloads match SHA-256.",
            "Hugging Face download metadata is retained under the ignored .cache directory.",
        ],
        "claim_boundary": (
            "This result establishes byte-complete acquisition and local integrity of "
            "the pinned repository snapshot only. It does not establish model loading, "
            "RTX 5080 execution, real activation capture, NVFP4 kernel identity, or "
            "numerical correctness."
        ),
    }
    git = collect_git()
    branch = _git_branch()
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "full_model_snapshot_acquisition_v1",
        "captured_at_utc": captured_at,
        "command": {"argv": [sys.executable, *sys.argv], "cwd": str(ROOT)},
        "git": {"branch": branch, "commit": git.commit, "dirty": git.dirty},
        "platform": {
            "os": platform.system(),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "software": {
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        },
        "model": {
            "repository": REPO_ID,
            "revision": REVISION,
            "local_snapshot_path": local_root,
            "repository_file_count": len(records),
            "repository_bytes": total_bytes,
            "weight_files_downloaded": True,
        },
        "backend": {
            "requested": "huggingface_hub_download",
            "reported": "hf_cache_verify_and_local_rehash",
            "observed_kernel": None,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "full-model-snapshot-acquisition",
                "path": args.results.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(args.results),
            }
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"files={len(records)}")
    print(f"bytes={total_bytes}")
    print(f"weight_bytes={weight_bytes}")
    print("status=pass")
    print("decision=continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
