import unittest
from collections.abc import Sequence

from nvfp4_doctor.env import (
    CollectionError,
    CommandResult,
    GPUFingerprint,
    SoftwareFingerprint,
    collect_gpu,
    collect_software,
)


class FakeRunner:
    def __init__(self, observations: dict[tuple[str, ...], CommandResult]) -> None:
        self.observations = observations
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        command = tuple(argv)
        self.calls.append(command)
        return self.observations[command]


class EnvironmentCollectorTests(unittest.TestCase):
    def test_gpu_collection_uses_only_injected_command_evidence(self) -> None:
        command = (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
        runner = FakeRunner(
            {
                command: CommandResult(
                    command,
                    0,
                    "NVIDIA GeForce RTX 5080, 595.95, 16303, 12.0\n",
                    "",
                )
            }
        )

        observed = collect_gpu(runner)

        self.assertEqual(
            observed,
            GPUFingerprint("NVIDIA GeForce RTX 5080", "12.0", 16303, "595.95"),
        )
        self.assertEqual(runner.calls, [command])

    def test_software_collection_uses_injected_commands_and_metadata(self) -> None:
        observations = {
            ("lsb_release", "-ds"): CommandResult(
                ("lsb_release", "-ds"), 0, '"Ubuntu 24.04.4 LTS"\n', ""
            ),
            ("uname", "-r"): CommandResult(
                ("uname", "-r"), 0, "6.18.33.2-microsoft-standard-WSL2\n", ""
            ),
            ("wsl.exe", "--version"): CommandResult(
                ("wsl.exe", "--version"),
                0,
                "W\x00S\x00L\x00 \x00v\x00e\x00r\x00s\x00i\x00o\x00n\x00:\x00 \x002\x00.\x007\x00.\x001\x002\x00.\x000\x00\n",
                "",
            ),
            ("nvcc", "--version"): CommandResult(
                ("nvcc", "--version"),
                0,
                "Cuda compilation tools, release 13.2, V13.2.86\n",
                "",
            ),
        }
        versions = {
            "torch": "2.13.0+cu132",
            "nvidia-cuda-runtime": "13.2.86",
            "vllm": "0.27.1",
            "flashinfer-python": "0.6.16.post3",
        }

        observed = collect_software(
            FakeRunner(observations),
            versions.__getitem__,
            python_version="3.12.3",
            nvcc_executable="nvcc",
        )

        self.assertEqual(
            observed,
            SoftwareFingerprint(
                os="Ubuntu 24.04.4 LTS",
                wsl_version="2.7.12.0",
                kernel="6.18.33.2-microsoft-standard-WSL2",
                python="3.12.3",
                torch="2.13.0+cu132",
                cuda_runtime="13.2.86",
                cuda_toolkit="13.2.86",
                vllm="0.27.1",
                flashinfer="0.6.16.post3",
            ),
        )

    def test_failed_command_is_not_silently_replaced(self) -> None:
        command = (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
        runner = FakeRunner(
            {command: CommandResult(command, 9, "", "GPU observation failed")}
        )

        with self.assertRaisesRegex(CollectionError, "returncode=9"):
            collect_gpu(runner)

    def test_multiple_gpu_rows_are_rejected_for_single_gpu_scope(self) -> None:
        command = (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
        runner = FakeRunner(
            {
                command: CommandResult(
                    command,
                    0,
                    "NVIDIA RTX 5080, 595.95, 16303, 12.0\n"
                    "NVIDIA RTX 5080, 595.95, 16303, 12.0\n",
                    "",
                )
            }
        )

        with self.assertRaisesRegex(CollectionError, "exactly one GPU row"):
            collect_gpu(runner)

    def test_missing_distribution_is_reported_by_name(self) -> None:
        observations = {
            ("lsb_release", "-ds"): CommandResult(
                ("lsb_release", "-ds"), 0, "Ubuntu 24.04.4 LTS", ""
            ),
            ("uname", "-r"): CommandResult(("uname", "-r"), 0, "kernel", ""),
            ("wsl.exe", "--version"): CommandResult(
                ("wsl.exe", "--version"), 0, "WSL version: 2.7.12.0", ""
            ),
            ("nvcc", "--version"): CommandResult(
                ("nvcc", "--version"), 0, "V13.2.86", ""
            ),
        }

        def missing(distribution: str) -> str:
            raise CollectionError(
                f"required distribution is not installed: {distribution}"
            )

        with self.assertRaisesRegex(CollectionError, "nvidia-cuda-runtime"):
            collect_software(FakeRunner(observations), missing, nvcc_executable="nvcc")


if __name__ == "__main__":
    unittest.main()
