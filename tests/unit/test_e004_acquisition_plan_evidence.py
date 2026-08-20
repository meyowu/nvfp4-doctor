import hashlib
import json
import unittest
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"


class E004AcquisitionPlanEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "acquisition-plan.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest-acquisition-plan.json").read_text(encoding="utf-8")
        )

    def test_selection_covers_early_middle_and_late_layers(self) -> None:
        self.assertEqual(
            self.results["selection"],
            [
                {"layer": 0, "role": "early", "rationale": "boundary layer"},
                {
                    "layer": 18,
                    "role": "middle",
                    "rationale": "first layer of second half",
                },
                {"layer": 35, "role": "late", "rationale": "boundary layer"},
            ],
        )
        self.assertEqual(
            self.results["coverage"],
            {
                "layer_count": 3,
                "projection_count": 5,
                "tensor_kinds_per_projection": 4,
                "planned_tensor_count": 60,
            },
        )

    def test_ranges_are_complete_unique_and_non_overlapping(self) -> None:
        ranges = self.results["ranges"]
        self.assertEqual(len(ranges), 60)
        self.assertEqual(len({record["tensor_name"] for record in ranges}), 60)
        self.assertEqual({record["layer"] for record in ranges}, {0, 18, 35})
        self.assertEqual(
            {record["projection"] for record in ranges},
            {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        )
        self.assertEqual(
            {record["suffix"] for record in ranges},
            {"input_scale", "weight", "weight_scale", "weight_scale_2"},
        )

        for shard, records in groupby(ranges, key=lambda record: record["shard"]):
            previous_end = None
            for record in records:
                with self.subTest(shard=shard, tensor=record["tensor_name"]):
                    self.assertGreater(
                        record["file_end_exclusive"], record["file_start"]
                    )
                    self.assertEqual(
                        record["byte_length"],
                        record["file_end_exclusive"] - record["file_start"],
                    )
                    self.assertEqual(
                        record["http_range"],
                        f"bytes={record['file_start']}-{record['file_end_exclusive'] - 1}",
                    )
                    if previous_end is not None:
                        self.assertLessEqual(previous_end, record["file_start"])
                    previous_end = record["file_end_exclusive"]

    def test_plan_records_cost_without_downloading_payloads(self) -> None:
        policy = self.results["request_policy"]
        self.assertTrue(policy["metadata_only"])
        self.assertEqual(policy["header_range_requests_executed"], 4)
        self.assertEqual(policy["header_bytes_downloaded"], 134032)
        self.assertEqual(policy["payload_requests_executed"], 0)
        self.assertEqual(policy["payload_bytes_downloaded"], 0)
        self.assertEqual(policy["planned_payload_request_count"], 60)
        self.assertEqual(policy["planned_payload_bytes"], 311427192)
        self.assertEqual(
            self.results["planned_bytes_by_layer"],
            {"0": 103809064, "18": 103809064, "35": 103809064},
        )

    def test_manifest_hashes_results_and_preserves_claim_boundary(self) -> None:
        self.assertEqual(
            self.manifest["backend"]["requested"],
            "metadata_only_acquisition_planning",
        )
        self.assertEqual(
            self.manifest["backend"]["reported"],
            "http_206_header_ranges_only",
        )
        self.assertIsNone(self.manifest["backend"]["observed_kernel"])
        self.assertEqual(self.manifest["model"]["payload_bytes_downloaded"], 0)
        self.assertFalse(self.manifest["model"]["weight_files_downloaded"])
        for artifact in self.manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
