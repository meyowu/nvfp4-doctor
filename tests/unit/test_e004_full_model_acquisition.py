import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_e004_full_model_acquisition import (
    FullModelAcquisitionError,
    _git_blob_id,
    _inspect_snapshot,
)


class FullModelAcquisitionTests(unittest.TestCase):
    def test_git_blob_id_matches_known_fixture(self) -> None:
        self.assertEqual(
            _git_blob_id(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    def test_inspection_checks_git_and_lfs_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = b"hello\n"
            lfs = b"payload"
            (root / "README.md").write_bytes(ordinary)
            (root / "model.safetensors").write_bytes(lfs)
            (root / ".cache").mkdir()
            (root / ".cache" / "metadata").write_text("ignored", encoding="utf-8")
            tree = {
                "files": {
                    "README.md": {
                        "size": len(ordinary),
                        "blob_id": _git_blob_id(ordinary),
                    },
                    "model.safetensors": {
                        "size": len(lfs),
                        "blob_id": "pointer-blob",
                        "lfs_sha256": hashlib.sha256(lfs).hexdigest(),
                    },
                }
            }

            records, cache_files = _inspect_snapshot(root, tree)

            self.assertEqual(
                [record["path"] for record in records],
                ["README.md", "model.safetensors"],
            )
            self.assertEqual(cache_files, 1)

    def test_inspection_rejects_extra_top_level_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("expected", encoding="utf-8")
            (root / "extra.txt").write_text("extra", encoding="utf-8")
            tree = {
                "files": {
                    "README.md": {
                        "size": 8,
                        "blob_id": _git_blob_id(b"expected"),
                    }
                }
            }
            with self.assertRaisesRegex(FullModelAcquisitionError, "inventory"):
                _inspect_snapshot(root, tree)

    def test_pinned_tree_metadata_is_json(self) -> None:
        value = json.loads('{"format_version": 1, "files": {}}')
        self.assertEqual(value["format_version"], 1)


if __name__ == "__main__":
    unittest.main()
