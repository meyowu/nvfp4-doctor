import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"


class E004EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = json.loads(
            (EXPERIMENT / "metadata.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (EXPERIMENT / "manifest-metadata.json").read_text(encoding="utf-8")
        )

    def test_checkpoint_revision_and_download_boundary_are_explicit(self) -> None:
        repository = self.metadata["repository"]
        self.assertEqual(repository["id"], "nvidia/Qwen3-8B-NVFP4")
        self.assertEqual(
            repository["revision"], "ccd10a893cbca613259517c3efe08e151ddf2b8e"
        )
        self.assertEqual(repository["revision"], repository["resolved_sha"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(self.metadata["weight_files_downloaded"])
        self.assertEqual(self.metadata["metadata_download_bytes"], 112965)

    def test_quantization_declarations_and_index_inventory_are_preserved(self) -> None:
        inspection = self.metadata["inspection"]
        self.assertEqual(inspection["architecture"], "Qwen3ForCausalLM")
        self.assertEqual(inspection["model_type"], "qwen3")
        self.assertEqual(inspection["num_hidden_layers"], 36)
        self.assertEqual(inspection["hidden_size"], 4096)
        self.assertEqual(inspection["intermediate_size"], 12288)
        self.assertEqual(inspection["quant_algo"], "NVFP4")
        self.assertEqual(inspection["group_size"], 16)
        self.assertEqual(inspection["weight_num_bits"], 4)
        self.assertEqual(inspection["input_num_bits"], 4)
        self.assertEqual(inspection["kv_cache_quant_algo"], "FP8")
        self.assertEqual(inspection["excluded_modules"], ["lm_head"])
        self.assertEqual(inspection["producer_name"], "modelopt")
        self.assertEqual(inspection["producer_version"], "0.35.0")
        self.assertEqual(inspection["tensor_count"], 1227)
        self.assertEqual(inspection["tensor_payload_bytes"], 6396932352)
        self.assertEqual(len(inspection["shards"]), 2)

    def test_every_capture_target_has_complete_layer_metadata(self) -> None:
        projections = self.metadata["inspection"]["target_projections"]
        self.assertEqual(
            {projection["projection"] for projection in projections},
            {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        )
        for projection in projections:
            with self.subTest(projection=projection["projection"]):
                self.assertEqual(projection["layer_count"], 36)
                self.assertEqual(projection["tensor_count"], 144)
                self.assertEqual(
                    projection["tensor_suffixes"],
                    ["input_scale", "weight", "weight_scale", "weight_scale_2"],
                )

    def test_weight_shards_are_recorded_but_not_downloaded(self) -> None:
        shards = {shard["path"]: shard for shard in self.metadata["weight_shards"]}
        self.assertEqual(
            {name: shard["size_bytes"] for name, shard in shards.items()},
            {
                "model-00001-of-00002.safetensors": 4987209424,
                "model-00002-of-00002.safetensors": 1409856960,
            },
        )
        self.assertEqual(
            shards["model-00001-of-00002.safetensors"]["lfs_sha256"],
            "6c13ef7322f4e5460858782e32da7e34b6c6fa8148cbeb70abcd2b44455d43f0",
        )
        self.assertEqual(
            shards["model-00002-of-00002.safetensors"]["lfs_sha256"],
            "cf084e6b0e9f4bed9d15b6a454c34c0a1e8c4b74668db62b4063defc5a601c96",
        )
        self.assertEqual(self.metadata["weight_file_bytes"], 6397066384)
        self.assertEqual(self.metadata["safetensors_header_overhead_bytes"], 134032)

    def test_manifest_hashes_the_normalized_metadata_result(self) -> None:
        self.assertEqual(self.manifest["backend"]["requested"], "metadata_only")
        self.assertEqual(self.manifest["backend"]["reported"], "huggingface_hub_api")
        self.assertIsNone(self.manifest["backend"]["observed_kernel"])
        self.assertFalse(self.manifest["model"]["weight_files_downloaded"])
        for artifact in self.manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
