#!/usr/bin/env python3
"""Attach independent Nsight Systems kernel observations to an E001 manifest."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nvfp4_doctor.backends import attach_kernel_evidence, extract_kernel_evidence
from nvfp4_doctor.env import EnvironmentManifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / ".local" / "artifacts" / "E001-kernel-identity" / "manifest.json"
)
DEFAULT_REPORT = ROOT / ".local" / "profiles" / "e001-smoke.nsys-rep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = subprocess.run(
        (
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum:nvtx-name",
            "--format",
            "csv",
            str(args.report),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    evidence = extract_kernel_evidence(args.report, completed.stdout)
    manifest = attach_kernel_evidence(
        EnvironmentManifest.from_path(args.manifest), evidence, args.report
    )
    args.manifest.write_text(manifest.to_json(), encoding="utf-8")
    print(f"manifest={args.manifest}")
    print(f"profiler_sha256={evidence.report_sha256}")
    print(f"observed_kernel_count={len(evidence.observed_kernels)}")
    print("reported_backend=unknown")
    print(f"fallback_status={manifest.backend.fallback_status.value}")


if __name__ == "__main__":
    main()
