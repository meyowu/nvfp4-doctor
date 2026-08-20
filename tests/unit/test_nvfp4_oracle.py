import json
import math
import unittest
from pathlib import Path

from nvfp4_doctor.formats import (
    NVFP4_COMBINED_MAX,
    ScaleFactorLayout,
    compute_global_scale,
    compute_raw_block_scale,
    swizzle_scales_128x4,
)
from nvfp4_doctor.oracle import NVFP4Tensor, reconstruct_nvfp4

FIXTURE = Path(__file__).parents[1] / "fixtures" / "e002_format_oracle_v1.json"


class NVFP4OracleTests(unittest.TestCase):
    def test_hand_computable_hierarchical_reconstruction(self) -> None:
        case = json.loads(FIXTURE.read_text(encoding="utf-8"))["reconstruction"]
        tensor = NVFP4Tensor(
            rows=case["rows"],
            columns=case["columns"],
            packed_values=bytes(case["packed_bytes"]),
            scale_codes=bytes(case["scale_codes"]),
            scale_layout=ScaleFactorLayout.LINEAR,
            global_scale=case["global_scale"],
            block_size=case["block_size"],
        )

        observed = reconstruct_nvfp4(tensor)[0]
        self.assertEqual(observed, tuple(case["values"]))
        self.assertEqual(math.copysign(1.0, observed[8]), -1.0)

    def test_linear_and_cutlass_scale_layouts_reconstruct_identically(self) -> None:
        packed = bytes([0x22] * 32)
        linear_scales = bytes((0x38, 0x40, 0x48, 0x50))
        common = {
            "rows": 2,
            "columns": 32,
            "packed_values": packed,
            "global_scale": 0.5,
        }
        linear = NVFP4Tensor(
            **common,
            scale_codes=linear_scales,
            scale_layout=ScaleFactorLayout.LINEAR,
        )
        swizzled = NVFP4Tensor(
            **common,
            scale_codes=swizzle_scales_128x4(linear_scales, 2, 2),
            scale_layout=ScaleFactorLayout.CUTLASS_128X4,
        )
        self.assertEqual(reconstruct_nvfp4(linear), reconstruct_nvfp4(swizzled))

    def test_public_hierarchical_scale_formulas(self) -> None:
        self.assertEqual(NVFP4_COMBINED_MAX, 2688.0)
        self.assertEqual(compute_global_scale(2688.0), 1.0)
        self.assertEqual(compute_raw_block_scale(12.0, 0.5), 4.0)
        self.assertEqual(compute_raw_block_scale(0.0, 0.0), 0.0)

    def test_scalar_global_scale_and_storage_lengths_are_explicit(self) -> None:
        tensor = NVFP4Tensor(
            rows=1,
            columns=16,
            packed_values=bytes(8),
            scale_codes=bytes(1),
            scale_layout=ScaleFactorLayout.LINEAR,
            global_scale=0.0,
        )
        self.assertIsInstance(tensor.global_scale, float)
        self.assertEqual(reconstruct_nvfp4(tensor), (tuple(0.0 for _ in range(16)),))
        with self.assertRaisesRegex(ValueError, "scale storage length"):
            NVFP4Tensor(
                rows=1,
                columns=16,
                packed_values=bytes(8),
                scale_codes=b"",
                scale_layout=ScaleFactorLayout.LINEAR,
                global_scale=1.0,
            )


if __name__ == "__main__":
    unittest.main()
