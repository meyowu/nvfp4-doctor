import json
import unittest
from collections.abc import Sequence
from datetime import datetime, timezone

from nvfp4_doctor.env import (
    CommandEvidence,
    CommandResult,
    FallbackStatus,
    TensorMetadata,
    assemble_e001_manifest,
    collect_git,
)


class FakeRunner:
    def __init__(self, outputs: dict[tuple[str, ...], tuple[int, str, str]]) -> None:
        self.outputs = outputs

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        command = tuple(argv)
        return CommandResult(command, *self.outputs[command])


def observations() -> dict[tuple[str, ...], tuple[int, str, str]]:
    return {
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ): (0, "NVIDIA GeForce RTX 5080, 595.95, 16303, 12.0", ""),
        ("lsb_release", "-ds"): (0, '"Ubuntu 24.04.4 LTS"', ""),
        ("uname", "-r"): (0, "6.18.33.2-microsoft-standard-WSL2", ""),
        ("wsl.exe", "--version"): (0, "WSL version: 2.7.12.0", ""),
        ("nvcc", "--version"): (0, "release 13.2, V13.2.86", ""),
        ("git", "rev-parse", "HEAD"): (
            0,
            "62d672774bd1061556551a96d31c15490155e885",
            "",
        ),
        ("git", "status", "--porcelain", "--untracked-files=normal"): (
            0,
            " M research-state.md",
            "",
        ),
    }


class EnvironmentAssemblyTests(unittest.TestCase):
    def test_git_dirty_state_is_observed_not_assumed(self) -> None:
        fingerprint = collect_git(FakeRunner(observations()))

        self.assertTrue(fingerprint.dirty)
        self.assertEqual(
            fingerprint.commit, "62d672774bd1061556551a96d31c15490155e885"
        )

    def test_manifest_assembly_preserves_unknown_execution_evidence(self) -> None:
        versions = {
            "torch": "2.13.0+cu132",
            "nvidia-cuda-runtime": "13.2.75",
            "vllm": "0.27.1",
            "flashinfer-python": "0.6.16.post3",
        }
        tensors = (
            TensorMetadata("a", (16, 256), (16, 256), "bfloat16", (256, 1), "cuda:0"),
            TensorMetadata("b", (128, 256), (128, 256), "bfloat16", (256, 1), "cuda:0"),
        )

        manifest = assemble_e001_manifest(
            tensors=tensors,
            command=CommandEvidence(
                ("python", "smoke_nvfp4.py"),
                "/home/meyowu/projects/nvfp4-doctor",
                0,
            ),
            runner=FakeRunner(observations()),
            versions=versions.__getitem__,
            clock=lambda: datetime(2026, 8, 20, 6, 30, tzinfo=timezone.utc),
            python_version="3.12.3",
            nvcc_executable="nvcc",
        )

        reloaded = type(manifest).from_json(manifest.to_json())
        self.assertEqual(reloaded, manifest)
        self.assertEqual(manifest.backend.requested_backend, "cutlass")
        self.assertEqual(manifest.backend.observed_kernels, ())
        self.assertEqual(manifest.backend.fallback_status, FallbackStatus.UNKNOWN)
        self.assertIsNone(manifest.backend.reported_backend)
        self.assertTrue(manifest.git.dirty)
        self.assertEqual(json.loads(manifest.to_json())["schema_version"], 1)

    def test_non_utc_clock_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "UTC-aware"):
            assemble_e001_manifest(
                tensors=(
                    TensorMetadata("a", (1,), (1,), "bfloat16", (1,), "cuda:0"),
                ),
                command=CommandEvidence(("python", "smoke_nvfp4.py"), "/tmp", 0),
                runner=FakeRunner(observations()),
                versions=lambda _: "1",
                clock=lambda: datetime(2026, 8, 20, 6, 30),
                nvcc_executable="nvcc",
            )


if __name__ == "__main__":
    unittest.main()
