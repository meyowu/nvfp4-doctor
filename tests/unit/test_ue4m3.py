import json
import math
import unittest
from itertools import pairwise
from pathlib import Path

from nvfp4_doctor.formats import decode_ue4m3, encode_ue4m3

FIXTURE = Path(__file__).parents[1] / "fixtures" / "e002_format_oracle_v1.json"


class UE4M3Tests(unittest.TestCase):
    def test_selected_codes_match_hand_authored_boundary_values(self) -> None:
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["ue4m3_decode"]
        for case in cases:
            with self.subTest(code=case["code"]):
                self.assertEqual(decode_ue4m3(case["code"]), case["value"])

    def test_all_finite_codes_are_ordered_and_round_trip(self) -> None:
        values = tuple(decode_ue4m3(code) for code in range(0x7F))
        self.assertEqual(len(values), 127)
        self.assertTrue(all(left < right for left, right in pairwise(values)))
        for code, value in enumerate(values):
            with self.subTest(code=code):
                self.assertEqual(encode_ue4m3(value), code)

    def test_saturation_and_nearest_even(self) -> None:
        self.assertEqual(encode_ue4m3(1000.0), 0x7E)
        self.assertEqual(encode_ue4m3(1.0625), 0x38)
        self.assertEqual(encode_ue4m3(1.1875), 0x3A)

    def test_nan_padded_msb_negative_and_nonfinite_are_rejected(self) -> None:
        for code in (0x7F, 0x80, 0xFF, -1, 256, True):
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    decode_ue4m3(code)
        for value in (-1.0, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    encode_ue4m3(value)


if __name__ == "__main__":
    unittest.main()
