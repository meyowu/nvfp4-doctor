#!/usr/bin/env python3
"""Report resumable nvfp4-doctor state without mutating the environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> dict[str, Any]:
    """Run a bounded read-only command and preserve failure evidence."""
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(command), "error": str(error)}

    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    state_path = ROOT / "research-state.md"
    report = {
        "schema_version": 1,
        "repository_root": str(ROOT),
        "state_file": str(state_path),
        "state_file_present": state_path.is_file(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        },
        "packages": {
            name: package_version(name)
            for name in ("torch", "vllm", "flashinfer-python")
        },
        "git": {
            "head": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "status": run("git", "status", "--short", "--branch"),
        },
        "gpu": run(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
