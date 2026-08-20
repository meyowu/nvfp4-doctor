import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"


class E004HeaderEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = json.loads(
            (EXPERIMENT / "headers.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest-headers.json").read_text(encoding="utf-8")
        )

    def test_only_exact_header_ranges_were_downloaded(self) -> None:
        self.assertTrue(self.results["request_policy"]["http_range_only"])
        self.assertTrue(
            self.results["request_policy"][
                "reject_non_partial_response_before_body_read"
            ]
        )
        self.assertEqual(self.results["range_request_count"], 4)
        self.assertEqual(self.results["range_bytes_downloaded"], 134032)
        self.assertEqual(self.results["payload_bytes_downloaded"], 0)
        self.assertFalse(self.results["weight_files_downloaded"])

    def test_shard_headers_have_exact_index_and_payload_boundaries(self) -> None:
        shards = {shard["path"]: shard for shard in self.results["shards"]}
        expected = {
            "model-00001-of-00002.safetensors": {
                "file_size": 4987209424,
                "header_length": 129048,
                "payload_start": 129056,
                "payload_bytes": 4987080368,
                "tensor_count": 1181,
            },
            "model-00002-of-00002.safetensors": {
                "file_size": 1409856960,
                "header_length": 4968,
                "payload_start": 4976,
                "payload_bytes": 1409851984,
                "tensor_count": 46,
            },
        }
        for name, values in expected.items():
            with self.subTest(shard=name):
                shard = shards[name]
                for field, value in values.items():
                    self.assertEqual(shard[field], value)
                self.assertTrue(shard["index_names_match"])
                self.assertTrue(shard["payload_boundaries_exact"])
        self.assertEqual(self.results["combined_tensor_count"], 1227)
        self.assertTrue(self.results["combined_index_names_match"])

    def test_capture_target_shapes_and_dtypes_are_uniform_across_layers(self) -> None:
        projections = {
            item["projection"]: item
            for item in self.results["capture_target_projections"]
        }
        expected_shapes = {
            "q_proj": ([4096, 2048], [4096, 256]),
            "o_proj": ([4096, 2048], [4096, 256]),
            "gate_proj": ([12288, 2048], [12288, 256]),
            "up_proj": ([12288, 2048], [12288, 256]),
            "down_proj": ([4096, 6144], [4096, 768]),
        }
        self.assertEqual(set(projections), set(expected_shapes))
        self.assertEqual(self.results["capture_target_tensor_count"], 720)
        for name, (weight_shape, scale_shape) in expected_shapes.items():
            with self.subTest(projection=name):
                projection = projections[name]
                self.assertEqual(projection["layer_count"], 36)
                self.assertEqual(projection["tensor_count"], 144)
                tensors = {item["suffix"]: item for item in projection["tensors"]}
                self.assertEqual(
                    (tensors["weight"]["dtype"], tensors["weight"]["shape"]),
                    ("U8", weight_shape),
                )
                self.assertEqual(
                    (
                        tensors["weight_scale"]["dtype"],
                        tensors["weight_scale"]["shape"],
                    ),
                    ("F8_E4M3", scale_shape),
                )
                for scalar in ("input_scale", "weight_scale_2"):
                    self.assertEqual(
                        (tensors[scalar]["dtype"], tensors[scalar]["shape"]),
                        ("F32", []),
                    )

    def test_manifest_hashes_results_and_preserves_backend_boundary(self) -> None:
        self.assertEqual(
            self.manifest["backend"]["requested"], "http_range_header_only"
        )
        self.assertEqual(
            self.manifest["backend"]["reported"], "http_206_partial_content"
        )
        self.assertIsNone(self.manifest["backend"]["observed_kernel"])
        self.assertEqual(self.manifest["model"]["payload_bytes_downloaded"], 0)
        for artifact in self.manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
