import json
import unittest
from pathlib import Path

from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    cutlass_128x4_offset,
    pack_e2m1,
    scale_physical_shape,
    scale_storage_size,
    swizzle_scales_128x4,
    unpack_e2m1,
    unswizzle_scales_128x4,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "e002_format_oracle_v1.json"


class PackingAndLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_low_nibble_first_packing_matches_golden_bytes(self) -> None:
        case = self.fixture["packing"]
        packed = pack_e2m1(case["logical_codes"])
        self.assertEqual(packed, bytes(case["packed_bytes"]))
        self.assertEqual(unpack_e2m1(packed), tuple(case["logical_codes"]))

    def test_hand_authored_cutlass_offsets(self) -> None:
        for case in self.fixture["layout_offsets"]:
            with self.subTest(row=case["row"], column=case["scale_column"]):
                self.assertEqual(
                    cutlass_128x4_offset(
                        case["row"],
                        case["scale_column"],
                        case["rows"],
                        case["scale_columns"],
                    ),
                    case["offset"],
                )

    def test_padding_is_explicit_for_boundary_shape(self) -> None:
        self.assertEqual(
            scale_physical_shape(129, 5, ScaleFactorLayout.CUTLASS_128X4),
            (256, 8),
        )
        self.assertEqual(
            scale_storage_size(129, 5, ScaleFactorLayout.CUTLASS_128X4),
            2048,
        )

    def test_swizzle_places_values_at_golden_offsets(self) -> None:
        linear = bytes(range(16))
        swizzled = swizzle_scales_128x4(linear, 4, 4, padding_code=0x7E)
        for row in range(4):
            for column in range(4):
                offset = cutlass_128x4_offset(row, column, 4, 4)
                self.assertEqual(swizzled[offset], linear[row * 4 + column])
        self.assertEqual(swizzled[511], 0x7E)
        self.assertEqual(unswizzle_scales_128x4(swizzled, 4, 4), linear)

    def test_invalid_packing_and_layout_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            pack_e2m1((0,))
        with self.assertRaisesRegex(ValueError, "length"):
            swizzle_scales_128x4(b"\x00", 2, 2)
        with self.assertRaises(IndexError):
            cutlass_128x4_offset(128, 0, 128, 4)


if __name__ == "__main__":
    unittest.main()
