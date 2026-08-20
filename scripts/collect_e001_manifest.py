#!/usr/bin/env python3
"""Collect the pre-profiler E001 manifest for the checked-in smoke test."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from nvfp4_doctor.env import (
    CommandEvidence,
    TensorMetadata,
    assemble_e001_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".local" / "artifacts" / "E001-kernel-identity" / "manifest.json"


def smoke_tensors() -> tuple[TensorMetadata, ...]:
    return (
        TensorMetadata("a", (16, 256), (16, 256), "bfloat16", (256, 1), "cuda:0"),
        TensorMetadata("b", (128, 256), (128, 256), "bfloat16", (256, 1), "cuda:0"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--requested-format", default="nvfp4")
    parser.add_argument("--requested-backend", default="cutlass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = assemble_e001_manifest(
        tensors=smoke_tensors(),
        command=CommandEvidence(
            argv=("python", "smoke_nvfp4.py"),
            cwd=str(ROOT),
            seed=0,
        ),
        requested_format=args.requested_format,
        requested_backend=args.requested_backend,
    )
    payload = manifest.to_json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"manifest={args.output}")
    print(f"sha256={hashlib.sha256(payload.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
