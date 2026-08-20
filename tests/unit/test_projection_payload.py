import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from nvfp4_doctor.checkpoint import (
    ProjectionPayloadError,
    load_modelopt_projection,
)
from nvfp4_doctor.formats import unswizzle_scales_128x4


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ProjectionPayloadTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        tensors = {
            "input_scale": ("F32", (), struct.pack("<f", 0.25)),
            "weight": ("U8", (128, 32), bytes(range(256)) * 16),
            "weight_scale": (
                "F8_E4M3",
                (128, 4),
                bytes(range(127)) * 4 + bytes((0, 1, 2, 3)),
            ),
            "weight_scale_2": ("F32", (), struct.pack("<f", 0.5)),
        }
        records = []
        for suffix, (dtype, shape, payload) in tensors.items():
            local_path = f"artifacts/layer-00/o_proj/{suffix}.bin"
            path = root / local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            records.append(
                {
                    "tensor_name": f"model.layers.0.self_attn.o_proj.{suffix}",
                    "dtype": dtype,
                    "shape": list(shape),
                    "local_path": local_path,
                    "byte_length": len(payload),
                    "sha256": _sha256(payload),
                }
            )
        evidence = root / "payloads.json"
        evidence.write_text(json.dumps({"payloads": records}), encoding="utf-8")
        return evidence

    def test_loads_exact_bytes_and_prepares_only_explicit_scale_swizzle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection = load_modelopt_projection(
                self._fixture(root), root, layer=0, projection="o_proj"
            )
            cutlass = projection.prepare_cutlass_128x4()

        self.assertEqual(projection.rows, 128)
        self.assertEqual(projection.columns, 64)
        self.assertEqual(projection.input_scale_value, 0.25)
        self.assertEqual(projection.weight_scale_2_value, 0.5)
        self.assertIs(cutlass.packed_weight, projection.weight.data)
        self.assertEqual(cutlass.weight_padding_bytes, 0)
        self.assertEqual(cutlass.source_weight_sha256, cutlass.runtime_weight_sha256)
        self.assertNotEqual(
            cutlass.source_weight_scale_sha256,
            cutlass.runtime_weight_scale_sha256,
        )
        self.assertEqual(
            unswizzle_scales_128x4(cutlass.weight_scale_128x4, 128, 4),
            projection.weight_scale.data,
        )
        self.assertEqual(cutlass.input_global_scale_inv, 4.0)
        self.assertEqual(cutlass.alpha, 0.125)

    def test_rejects_artifact_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._fixture(root)
            (root / "artifacts/layer-00/o_proj/weight.bin").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ProjectionPayloadError, "byte length mismatch"):
                load_modelopt_projection(evidence, root, layer=0, projection="o_proj")

    def test_rejects_negative_or_nonfinite_scale_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._fixture(root)
            document = json.loads(evidence.read_text(encoding="utf-8"))
            record = next(
                item
                for item in document["payloads"]
                if item["tensor_name"].endswith("weight_scale")
            )
            path = root / record["local_path"]
            payload = bytes((0xFF,)) + path.read_bytes()[1:]
            path.write_bytes(payload)
            record["sha256"] = _sha256(payload)
            evidence.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ProjectionPayloadError, "non-finite E4M3"):
                load_modelopt_projection(evidence, root, layer=0, projection="o_proj")

    def test_rejects_projection_that_would_need_implicit_weight_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._fixture(root)
            document = json.loads(evidence.read_text(encoding="utf-8"))
            for record in document["payloads"]:
                suffix = record["tensor_name"].rsplit(".", 1)[-1]
                if suffix == "weight":
                    payload = bytes(48 * 8)
                    record["shape"] = [48, 8]
                elif suffix == "weight_scale":
                    payload = bytes(48)
                    record["shape"] = [48, 1]
                else:
                    continue
                path = root / record["local_path"]
                path.write_bytes(payload)
                record["byte_length"] = len(payload)
                record["sha256"] = _sha256(payload)
            evidence.write_text(json.dumps(document), encoding="utf-8")
            projection = load_modelopt_projection(
                evidence, root, layer=0, projection="o_proj"
            )

            with self.assertRaisesRegex(ProjectionPayloadError, "padding"):
                projection.prepare_cutlass_128x4()


if __name__ == "__main__":
    unittest.main()
