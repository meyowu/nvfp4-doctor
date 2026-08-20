import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from nvfp4_doctor.env import EnvironmentManifest, ManifestValidationError


FIXTURE = Path(__file__).parents[1] / "fixtures" / "e001_manifest_v1.json"


class EnvironmentManifestTests(unittest.TestCase):
    def test_golden_manifest_round_trips_exactly(self) -> None:
        payload = FIXTURE.read_text(encoding="utf-8")

        manifest = EnvironmentManifest.from_json(payload)

        self.assertEqual(manifest.to_json(), payload)
        self.assertEqual(manifest.backend.requested_backend, "cutlass")
        self.assertEqual(manifest.backend.observed_kernels, ())

    def test_manifest_is_immutable(self) -> None:
        manifest = EnvironmentManifest.from_path(FIXTURE)

        with self.assertRaises(FrozenInstanceError):
            manifest.schema_version = 2  # type: ignore[misc]

    def test_unknown_root_field_is_rejected(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["unreviewed"] = True

        with self.assertRaisesRegex(ManifestValidationError, "unknown=.*unreviewed"):
            EnvironmentManifest.from_dict(data)

    def test_unknown_nested_field_is_rejected(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["backend"]["inferred_backend"] = "cutlass"

        with self.assertRaisesRegex(ManifestValidationError, "inferred_backend"):
            EnvironmentManifest.from_dict(data)

    def test_future_schema_version_is_rejected(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["schema_version"] = 2

        with self.assertRaisesRegex(ManifestValidationError, "unsupported schema_version"):
            EnvironmentManifest.from_dict(data)

    def test_missing_backend_evidence_is_rejected(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        del data["backend"]

        with self.assertRaisesRegex(ManifestValidationError, "missing=.*backend"):
            EnvironmentManifest.from_dict(data)

    def test_stride_rank_mismatch_is_rejected(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["tensors"][0]["stride"] = [1]

        with self.assertRaisesRegex(ManifestValidationError, "stride rank"):
            EnvironmentManifest.from_dict(data)


if __name__ == "__main__":
    unittest.main()
