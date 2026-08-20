import json
import struct
import unittest

from nvfp4_doctor.checkpoint import (
    AcquisitionPlanError,
    parse_safetensors_header,
    plan_tensor_byte_ranges,
)


def _header(document: dict[str, object]):
    payload_bytes = max(
        int(value["data_offsets"][1])
        for value in document.values()
        if isinstance(value, dict)
    )
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    return parse_safetensors_header(
        struct.pack("<Q", len(encoded)), encoded, 8 + len(encoded) + payload_bytes
    )


class AcquisitionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _header(
            {
                "layer.0.weight": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 4],
                },
                "layer.0.scale": {
                    "dtype": "F32",
                    "shape": [],
                    "data_offsets": [4, 8],
                },
            }
        )
        self.second = _header(
            {
                "layer.1.weight": {
                    "dtype": "U8",
                    "shape": [2],
                    "data_offsets": [0, 2],
                }
            }
        )

    def test_maps_payload_offsets_to_exact_file_ranges(self) -> None:
        plan = plan_tensor_byte_ranges(
            {"b.safetensors": self.second, "a.safetensors": self.first},
            ["layer.1.weight", "layer.0.scale", "layer.0.weight"],
        )

        self.assertEqual(
            [item.tensor_name for item in plan],
            ["layer.0.weight", "layer.0.scale", "layer.1.weight"],
        )
        self.assertEqual(plan[0].file_start, self.first.payload_start)
        self.assertEqual(plan[0].file_end, self.first.payload_start + 4)
        self.assertEqual(
            plan[0].http_range,
            f"bytes={self.first.payload_start}-{self.first.payload_start + 3}",
        )
        self.assertEqual(sum(item.byte_length for item in plan), 10)

    def test_rejects_duplicate_or_missing_requests(self) -> None:
        with self.assertRaisesRegex(AcquisitionPlanError, "unique"):
            plan_tensor_byte_ranges(
                {"a.safetensors": self.first},
                ["layer.0.weight", "layer.0.weight"],
            )
        with self.assertRaisesRegex(AcquisitionPlanError, "missing"):
            plan_tensor_byte_ranges({"a.safetensors": self.first}, ["layer.2.weight"])

    def test_rejects_tensor_names_repeated_across_shards(self) -> None:
        with self.assertRaisesRegex(AcquisitionPlanError, "more than one shard"):
            plan_tensor_byte_ranges(
                {"a.safetensors": self.first, "copy.safetensors": self.first},
                ["layer.0.weight"],
            )


if __name__ == "__main__":
    unittest.main()
