import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
SHA256 = re.compile(r"[0-9a-f]{64}")


class E004PayloadEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "payloads.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest-payloads.json").read_text(encoding="utf-8")
        )

    def test_acquisition_is_complete_and_does_not_claim_full_shards(self) -> None:
        self.assertEqual(self.results["payload_request_count"], 60)
        self.assertEqual(self.results["payload_bytes_downloaded"], 311427192)
        self.assertEqual(self.results["local_artifact_count"], 60)
        self.assertFalse(self.results["weight_files_downloaded"])
        self.assertTrue(self.results["all_lengths_exact"])
        self.assertTrue(self.results["all_local_hashes_recorded"])

    def test_payload_records_cover_the_frozen_representative_matrix(self) -> None:
        payloads = self.results["payloads"]
        self.assertEqual(len(payloads), 60)
        self.assertEqual(len({item["tensor_name"] for item in payloads}), 60)
        coverage = {
            (
                int(item["tensor_name"].split(".")[2]),
                item["tensor_name"].split(".")[-2],
                item["tensor_name"].split(".")[-1],
            )
            for item in payloads
        }
        expected = {
            (layer, projection, suffix)
            for layer in (0, 18, 35)
            for projection in (
                "q_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
            for suffix in (
                "input_scale",
                "weight",
                "weight_scale",
                "weight_scale_2",
            )
        }
        self.assertEqual(coverage, expected)

    def test_every_record_has_exact_response_and_local_hash_evidence(self) -> None:
        for item in self.results["payloads"]:
            with self.subTest(tensor=item["tensor_name"]):
                self.assertEqual(item["status_code"], 206)
                self.assertEqual(item["content_length"], item["byte_length"])
                self.assertTrue(item["content_range"].startswith("bytes "))
                self.assertTrue(SHA256.fullmatch(item["sha256"]))
                self.assertTrue(SHA256.fullmatch(item["shard_lfs_sha256"]))
                self.assertTrue(
                    item["local_path"].startswith(
                        "artifacts/E004-qwen3-layer-capture/tensor-payloads/"
                    )
                )

    def test_manifest_hashes_inventory_and_preserves_backend_boundary(self) -> None:
        self.assertEqual(
            self.manifest["backend"]["requested"],
            "http_range_tensor_acquisition",
        )
        self.assertEqual(
            self.manifest["backend"]["reported"],
            "http_206_partial_content",
        )
        self.assertIsNone(self.manifest["backend"]["observed_kernel"])
        self.assertFalse(self.manifest["model"]["weight_files_downloaded"])
        self.assertEqual(self.manifest["model"]["tensor_payload_count"], 60)
        self.assertEqual(self.manifest["model"]["payload_bytes_downloaded"], 311427192)
        for artifact in self.manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
