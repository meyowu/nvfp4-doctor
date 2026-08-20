import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "E004-qwen3-layer-capture"
RESULTS = EXPERIMENT / "full-model-acquisition.json"
MANIFEST = EXPERIMENT / "manifest-full-model-acquisition.json"
REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
EXPECTED_LFS = {
    "model-00001-of-00002.safetensors": (
        "6c13ef7322f4e5460858782e32da7e34b6c6fa8148cbeb70abcd2b44455d43f0"
    ),
    "model-00002-of-00002.safetensors": (
        "cf084e6b0e9f4bed9d15b6a454c34c0a1e8c4b74668db62b4063defc5a601c96"
    ),
    "tokenizer.json": (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    ),
}
SOURCE_PATHS = (
    ROOT / "scripts" / "run_e004_full_model_acquisition.py",
    ROOT / "tests" / "unit" / "test_e004_full_model_acquisition.py",
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class FullModelAcquisitionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _json(RESULTS)
        cls.manifest = _json(MANIFEST)

    def test_result_freezes_complete_snapshot_identity(self) -> None:
        self.assertEqual(self.result["schema_version"], 1)
        self.assertEqual(self.result["slice"], "full_model_snapshot_acquisition_v1")
        self.assertEqual(
            (self.result["status"], self.result["decision"]), ("pass", "continue")
        )
        repository = self.result["repository"]
        self.assertEqual(repository["id"], "nvidia/Qwen3-8B-NVFP4")
        self.assertEqual(repository["requested_revision"], REVISION)
        self.assertEqual(repository["resolved_sha"], REVISION)

    def test_inventory_has_exact_totals_and_lfs_hashes(self) -> None:
        snapshot = self.result["snapshot"]
        self.assertEqual(snapshot["file_count"], 15)
        self.assertEqual(snapshot["total_bytes"], 6_413_063_143)
        self.assertEqual(snapshot["weight_shard_count"], 2)
        self.assertEqual(snapshot["weight_bytes"], 6_397_066_384)
        self.assertEqual(snapshot["tokenizer_and_small_file_bytes"], 15_996_759)
        self.assertTrue(snapshot["ignored"])
        self.assertTrue(snapshot["local_root"].startswith("models/"))
        self.assertEqual(snapshot["inventory_exclusions"], [".cache/"])
        files = snapshot["files"]
        self.assertEqual(len(files), 15)
        self.assertEqual(len({record["path"] for record in files}), 15)
        self.assertEqual(sum(record["size_bytes"] for record in files), 6_413_063_143)
        observed_lfs = {
            record["path"]: record["sha256"]
            for record in files
            if record["path"] in EXPECTED_LFS
        }
        self.assertEqual(observed_lfs, EXPECTED_LFS)
        self.assertTrue(all(len(record["sha256"]) == 64 for record in files))

    def test_verification_and_claim_boundary_are_scoped(self) -> None:
        verification = self.result["verification"]
        self.assertEqual(verification["hf_cache_verify_files_checked"], 15)
        self.assertTrue(
            all(
                value is True
                for key, value in verification.items()
                if key != "hf_cache_verify_files_checked"
            )
        )
        claim = self.result["claim_boundary"]
        self.assertIn("does not establish model loading", claim)
        self.assertIn("numerical correctness", claim)

    def test_manifest_records_clean_implementation_and_result_hash(self) -> None:
        git = self.manifest["git"]
        self.assertEqual(git["commit"], "6dfa5abe03e72f8c852d3e686a474994304a86ef")
        self.assertEqual(git["branch"], "exp/e004-real-activation")
        self.assertFalse(git["dirty"])
        self.assertEqual(self.manifest["source_bundle_sha256"], _source_bundle_sha256())
        artifact = self.manifest["artifacts"][0]
        self.assertEqual(
            artifact["path"],
            "experiments/E004-qwen3-layer-capture/full-model-acquisition.json",
        )
        self.assertEqual(artifact["sha256"], _sha256(RESULTS))
        self.assertTrue(self.manifest["model"]["weight_files_downloaded"])


if __name__ == "__main__":
    unittest.main()
