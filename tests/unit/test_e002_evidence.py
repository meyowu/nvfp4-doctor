import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E002-format-oracle"


class E002EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "results.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest.json").read_text(encoding="utf-8")
        )
        self.sources = json.loads(
            (EXPERIMENT / "sources.json").read_text(encoding="utf-8")
        )

    def test_results_cover_gate1_exact_domains(self) -> None:
        self.assertEqual(self.results["experiment_id"], "E002-format-oracle")
        self.assertEqual(self.results["fixture_checks"]["e2m1_payloads_checked"], 16)
        self.assertEqual(
            self.results["fixture_checks"]["finite_ue4m3_codes_checked"], 127
        )
        self.assertEqual(self.results["adapter_differential"], "pass")
        self.assertEqual(self.results["decision"], "go")
        self.assertTrue(self.results["oracle_independent_of_candidate"])
        self.assertEqual(len(self.results["gpu_cases"]), 3)
        for case in self.results["gpu_cases"]:
            with self.subTest(shape=(case["rows"], case["columns"])):
                self.assertTrue(all(case["checks"].values()))
                self.assertEqual(case["max_abs_reconstruction_error"], 0.0)

    def test_manifest_preserves_requested_and_observed_backend_separation(self) -> None:
        adapter = self.manifest["candidate_adapter"]
        self.assertEqual(adapter["requested_backend"], "cuda")
        self.assertIsNone(adapter["reported_backend"])
        self.assertIsNone(adapter["observed_kernel"])

    def test_manifest_artifact_hashes_match_tracked_files(self) -> None:
        for artifact in self.manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )

    def test_public_source_record_is_unique_and_pinned(self) -> None:
        ids = [source["id"] for source in self.sources["sources"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 6)
        self.assertEqual(
            self.sources["cutlass_revision"]["commit"],
            "e6233cbac5d7c7a865c19c91cd684ceece19513c",
        )


if __name__ == "__main__":
    unittest.main()
