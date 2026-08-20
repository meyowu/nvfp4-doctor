#!/usr/bin/env python3
"""Collect and inspect pinned Qwen3-8B-NVFP4 metadata without model weights."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from nvfp4_doctor.checkpoint import inspect_modelopt_checkpoint

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
RESULTS = EXPERIMENT / "metadata.json"
MANIFEST = EXPERIMENT / "manifest-metadata.json"
LOCAL_SOURCE = ROOT / "artifacts" / "E004-checkpoint-metadata" / "source"
REPO_ID = "nvidia/Qwen3-8B-NVFP4"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
METADATA_FILES = (
    "README.md",
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
)
SOURCE_PATHS = (
    ROOT / "src" / "nvfp4_doctor" / "checkpoint" / "modelopt.py",
    ROOT / "tests" / "unit" / "test_checkpoint_metadata.py",
    ROOT / "tests" / "unit" / "test_e004_evidence.py",
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


def _isoformat(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return None


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sibling_record(sibling: Any) -> dict[str, object]:
    lfs = sibling.lfs if isinstance(sibling.lfs, dict) else {}
    return {
        "path": sibling.rfilename,
        "size_bytes": sibling.size,
        "git_blob_id": sibling.blob_id,
        "lfs_sha256": lfs.get("sha256"),
        "lfs_pointer_size": lfs.get("pointer_size"),
    }


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    LOCAL_SOURCE.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Hub resolved revision does not match the immutable pin")
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    missing = set(METADATA_FILES) - siblings.keys()
    if missing:
        raise ValueError(
            f"pinned repository is missing metadata files: {sorted(missing)}"
        )

    local_files: dict[str, Path] = {}
    for filename in METADATA_FILES:
        resolved = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            local_dir=LOCAL_SOURCE,
        )
        local_files[filename] = Path(resolved)

    config = _json(local_files["config.json"])
    hf_quant_config = _json(local_files["hf_quant_config.json"])
    safetensors_index = _json(local_files["model.safetensors.index.json"])
    inspection = inspect_modelopt_checkpoint(
        config,
        hf_quant_config,
        safetensors_index,
    )

    metadata_sources = [
        {
            **_sibling_record(siblings[filename]),
            "sha256": _sha256_path(local_files[filename]),
            "source_url": (
                f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{filename}"
            ),
        }
        for filename in METADATA_FILES
    ]
    weight_shards = [
        _sibling_record(sibling)
        for sibling in info.siblings
        if sibling.rfilename.startswith("model-")
        and sibling.rfilename.endswith(".safetensors")
    ]
    weight_file_bytes = sum(int(record["size_bytes"] or 0) for record in weight_shards)
    header_overhead_bytes = weight_file_bytes - inspection.tensor_payload_bytes
    if header_overhead_bytes < 0:
        raise ValueError("tensor payload bytes exceed repository weight file sizes")

    results = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "checkpoint_metadata_v1",
        "status": "pass",
        "decision": "continue",
        "repository": {
            "id": REPO_ID,
            "revision": REVISION,
            "resolved_sha": info.sha,
            "private": info.private,
            "gated": info.gated,
            "library_name": info.library_name,
            "pipeline_tag": info.pipeline_tag,
            "created_at": _isoformat(info.created_at),
            "last_modified": _isoformat(info.last_modified),
        },
        "metadata_sources": metadata_sources,
        "metadata_download_bytes": sum(
            int(record["size_bytes"] or 0) for record in metadata_sources
        ),
        "weight_shards": weight_shards,
        "weight_file_bytes": weight_file_bytes,
        "tensor_payload_bytes": inspection.tensor_payload_bytes,
        "safetensors_header_overhead_bytes": header_overhead_bytes,
        "weight_files_downloaded": False,
        "inspection": asdict(inspection),
        "capture_targets": [
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "observations": [
            "config.json and hf_quant_config.json agree on NVFP4 group size 16",
            "weights and input activations are declared as static 4-bit float",
            "lm_head is excluded and the KV-cache scheme is separately declared FP8",
            "all five capture-target projections have four quantization tensors in every layer",
        ],
        "claim_boundary": (
            "This metadata-only inspection does not read safetensors headers or "
            "payloads, load the model, establish runtime tensor shapes or layouts, "
            "identify an executed backend, or demonstrate numerical correctness."
        ),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    git_status = _command("git", "status", "--porcelain=v1")
    manifest = {
        "schema_version": 1,
        "experiment_id": "E004-qwen3-layer-capture",
        "slice": "checkpoint_metadata_v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "argv": ["python", "scripts/run_e004_checkpoint_metadata.py"],
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
            "huggingface_hub": _package_version("huggingface-hub"),
            "torch": _package_version("torch"),
            "vllm": _package_version("vllm"),
            "flashinfer": _package_version("flashinfer-python"),
        },
        "model": {
            "repository": REPO_ID,
            "revision": REVISION,
            "weight_files_downloaded": False,
        },
        "backend": {
            "requested": "metadata_only",
            "reported": "huggingface_hub_api",
            "observed_kernel": None,
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "artifacts": [
            {
                "kind": "checkpoint-metadata-inspection",
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": _sha256_path(RESULTS),
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"repository={REPO_ID}")
    print(f"revision={REVISION}")
    print(f"metadata_download_bytes={results['metadata_download_bytes']}")
    print(f"weight_files_downloaded={results['weight_files_downloaded']}")
    print(f"tensor_count={inspection.tensor_count}")
    print(f"target_projection_count={len(inspection.target_projections)}")
    print(f"status={results['status']}")
    print(f"decision={results['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
