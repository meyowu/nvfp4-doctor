"""CPU-only collectors for E001 host and software fingerprints."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .manifest import GPUFingerprint, SoftwareFingerprint


class CollectionError(RuntimeError):
    """Raised when an observation cannot be collected or parsed."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> CommandResult: ...


class PackageVersionProvider(Protocol):
    def __call__(self, distribution: str) -> str: ...


def run_command(argv: Sequence[str]) -> CommandResult:
    """Execute a bounded read-only observation command."""
    command = tuple(argv)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CollectionError(f"failed to execute {command!r}: {error}") from error
    return CommandResult(
        argv=command,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise CollectionError(
            f"required distribution is not installed: {distribution}"
        ) from error


def _successful(runner: CommandRunner, argv: Sequence[str]) -> str:
    result = runner(argv)
    if result.returncode != 0:
        raise CollectionError(
            f"command failed: {result.argv!r}; returncode={result.returncode}; "
            f"stderr={result.stderr!r}"
        )
    if not result.stdout.strip():
        raise CollectionError(f"command returned no output: {result.argv!r}")
    return result.stdout.strip()


def collect_gpu(runner: CommandRunner = run_command) -> GPUFingerprint:
    output = _successful(
        runner,
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ),
    )
    rows = [row for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise CollectionError(f"expected exactly one GPU row, observed {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise CollectionError(
            f"expected four nvidia-smi fields, observed {len(fields)}"
        )
    name, driver_version, memory_mib, compute_capability = fields
    try:
        memory = int(memory_mib)
    except ValueError as error:
        raise CollectionError(f"invalid GPU memory value: {memory_mib!r}") from error
    return GPUFingerprint(
        name=name,
        compute_capability=compute_capability,
        memory_mib=memory,
        driver_version=driver_version,
    )


def _parse_wsl_version(output: str) -> str:
    normalized = output.replace("\x00", "")
    match = re.search(r"WSL version:\s*([^\s]+)", normalized, re.IGNORECASE)
    if match is None:
        raise CollectionError("could not parse WSL version")
    return match.group(1)


def _parse_cuda_toolkit(output: str) -> str:
    match = re.search(r"V(\d+\.\d+\.\d+)", output)
    if match is None:
        raise CollectionError("could not parse CUDA toolkit build from nvcc output")
    return match.group(1)


def collect_software(
    runner: CommandRunner = run_command,
    versions: PackageVersionProvider = installed_version,
    python_version: str | None = None,
    nvcc_executable: str | None = None,
) -> SoftwareFingerprint:
    os_name = _successful(runner, ("lsb_release", "-ds")).strip('"')
    kernel = _successful(runner, ("uname", "-r"))
    wsl_version = _parse_wsl_version(_successful(runner, ("wsl.exe", "--version")))
    if nvcc_executable is None:
        cuda_home = os.environ.get("CUDA_HOME")
        nvcc_executable = str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else "nvcc"
    toolkit = _parse_cuda_toolkit(_successful(runner, (nvcc_executable, "--version")))
    cuda_runtime = versions("nvidia-cuda-runtime")
    return SoftwareFingerprint(
        os=os_name,
        wsl_version=wsl_version,
        kernel=kernel,
        python=python_version or platform.python_version(),
        torch=versions("torch"),
        cuda_runtime=cuda_runtime,
        cuda_toolkit=toolkit,
        vllm=versions("vllm"),
        flashinfer=versions("flashinfer-python"),
    )
