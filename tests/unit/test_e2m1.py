import json
import math
import unittest
from pathlib import Path

from nvfp4_doctor.formats import decode_e2m1, encode_e2m1

FIXTURE = Path(__file__).parents[1] / "fixtures" / "e002_format_oracle_v1.json"


class E2M1Tests(unittest.TestCase):
    def test_all_sixteen_codes_match_hand_authored_values(self) -> None:
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["e2m1_decode"]
        self.assertEqual(len(cases), 16)
        for case in cases:
            with self.subTest(code=case["code"]):
                observed = decode_e2m1(case["code"])
                self.assertEqual(observed, case["value"])
                self.assertEqual(
                    math.copysign(1.0, observed), math.copysign(1.0, case["value"])
                )
                self.assertEqual(encode_e2m1(observed), case["code"])

    def test_round_to_nearest_even_at_every_positive_midpoint(self) -> None:
        expected = {
            0.25: 0,
            0.75: 2,
            1.25: 2,
            1.75: 4,
            2.5: 4,
            3.5: 6,
            5.0: 6,
        }
        for value, code in expected.items():
            with self.subTest(value=value):
                self.assertEqual(encode_e2m1(value), code)
                self.assertEqual(encode_e2m1(-value), code | 0x8)

    def test_finite_out_of_range_values_saturate(self) -> None:
        self.assertEqual(encode_e2m1(100.0), 0x7)
        self.assertEqual(encode_e2m1(-100.0), 0xF)

    def test_invalid_values_and_codes_are_rejected(self) -> None:
        for code in (-1, 16, True):
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    decode_e2m1(code)
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    encode_e2m1(value)


if __name__ == "__main__":
    unittest.main()
