import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from nvfp4_doctor.backends import E001_GEMM_RANGE
from nvfp4_doctor.env import BackendEvidence, EnvironmentManifest, FallbackStatus
from nvfp4_doctor.report import E001SummaryError, summarize_e001_manifests

EXPECTED_CUTLASS = (
    f"{E001_GEMM_RANGE}/void cutlass::device_kernel<"
    "MainloopSm120TmaWarpSpecializedBlockScaled, cutlass::float_e2m1_t, "
    "SM120_16x8x64_TN_VS>()"
)


def manifest_for_run(base: EnvironmentManifest, index: int) -> EnvironmentManifest:
    return replace(
        base,
        captured_at_utc=f"2026-08-20T04:00:0{index}Z",
        backend=BackendEvidence(
            requested_format="nvfp4",
            requested_backend="cutlass",
            reported_backend=None,
            observed_kernels=(EXPECTED_CUTLASS,),
            fallback_status=FallbackStatus.NOT_DETECTED,
            profiler_artifact_sha256=str(index) * 64,
        ),
    )


class E001SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "e001_manifest_v1.json"
        self.base = EnvironmentManifest.from_path(fixture)

    def write_runs(self, directory: str) -> tuple[Path, ...]:
        paths = []
        for index in range(1, 4):
            path = Path(directory) / f"run-{index:02d}.json"
            path.write_text(
                manifest_for_run(self.base, index).to_json(), encoding="utf-8"
            )
            paths.append(path)
        return tuple(paths)

    def test_three_stable_runs_pass_gate0_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = summarize_e001_manifests(self.write_runs(directory))
        self.assertEqual(summary["repetitions"], 3)
        self.assertEqual(summary["gate0_repeatability"], "pass")
        self.assertEqual(summary["decision"], "go")
        self.assertTrue(summary["target_kernel_set_stable"])

    def test_environment_change_fails_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_runs(directory)
            third = EnvironmentManifest.from_path(paths[2])
            paths[2].write_text(
                replace(
                    third, gpu=replace(third.gpu, driver_version="different")
                ).to_json(),
                encoding="utf-8",
            )
            summary = summarize_e001_manifests(paths)
        self.assertFalse(summary["environment_stable"])
        self.assertEqual(summary["decision"], "repeat")

    def test_missing_target_signature_fails_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_runs(directory)
            third = json.loads(paths[2].read_text(encoding="utf-8"))
            third["backend"]["fallback_status"] = "unknown"
            third["backend"]["observed_kernels"] = []
            paths[2].write_text(json.dumps(third), encoding="utf-8")
            summary = summarize_e001_manifests(paths)
        self.assertFalse(summary["no_known_fallback_detected"])
        self.assertFalse(summary["target_kernel_set_stable"])

    def test_fewer_than_three_runs_is_rejected(self) -> None:
        with self.assertRaisesRegex(E001SummaryError, "at least three"):
            summarize_e001_manifests((Path("one.json"), Path("two.json")))


if __name__ == "__main__":
    unittest.main()
