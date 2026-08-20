import json
import struct
import unittest

from nvfp4_doctor.checkpoint import (
    SafetensorsHeaderError,
    parse_safetensors_header,
)


def _header_bytes(document: dict[str, object]) -> bytes:
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


class SafetensorsHeaderTests(unittest.TestCase):
    def test_valid_header_preserves_shapes_dtypes_and_exact_boundaries(self) -> None:
        header_bytes = _header_bytes(
            {
                "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
                "packed": {
                    "dtype": "U8",
                    "shape": [2, 2],
                    "data_offsets": [4, 8],
                },
            }
        )
        prefix = struct.pack("<Q", len(header_bytes))
        parsed = parse_safetensors_header(
            prefix, header_bytes, 8 + len(header_bytes) + 8
        )

        self.assertEqual(parsed.header_length, len(header_bytes))
        self.assertEqual(parsed.payload_start, 8 + len(header_bytes))
        self.assertEqual(parsed.payload_bytes, 8)
        self.assertEqual(
            [(tensor.name, tensor.dtype, tensor.shape) for tensor in parsed.tensors],
            [("scalar", "F32", ()), ("packed", "U8", (2, 2))],
        )

    def test_declared_header_length_must_match_fetched_bytes(self) -> None:
        header_bytes = _header_bytes(
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
        )
        with self.assertRaisesRegex(SafetensorsHeaderError, "header length"):
            parse_safetensors_header(
                struct.pack("<Q", len(header_bytes) + 1),
                header_bytes,
                8 + len(header_bytes) + 1,
            )

    def test_payload_intervals_must_be_contiguous_and_cover_file(self) -> None:
        header_bytes = _header_bytes(
            {
                "x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
            }
        )
        with self.assertRaisesRegex(SafetensorsHeaderError, "contiguous"):
            parse_safetensors_header(
                struct.pack("<Q", len(header_bytes)),
                header_bytes,
                8 + len(header_bytes) + 2,
            )

    def test_tensor_byte_length_must_match_dtype_and_shape(self) -> None:
        header_bytes = _header_bytes(
            {
                "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
            }
        )
        with self.assertRaisesRegex(SafetensorsHeaderError, "dtype/shape"):
            parse_safetensors_header(
                struct.pack("<Q", len(header_bytes)),
                header_bytes,
                8 + len(header_bytes) + 4,
            )


if __name__ == "__main__":
    unittest.main()
