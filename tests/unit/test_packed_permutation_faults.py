import unittest

from nvfp4_doctor.contracts import FormatContractId, evaluate_format_contracts
from nvfp4_doctor.faults import (
    inject_packed_block_permutation,
    inject_packed_column_permutation,
    inject_packed_row_permutation,
    permute_packed_blocks,
    permute_packed_columns,
    permute_packed_rows,
    revert_fault,
)
from nvfp4_doctor.formats import ScaleFactorLayout, pack_e2m1, unpack_e2m1
from nvfp4_doctor.oracle import NVFP4Tensor


def _tensor(rows: int = 3, columns: int = 48) -> NVFP4Tensor:
    codes = tuple(
        (row * 5 + column * 3 + (row * column) % 7) % 16
        for row in range(rows)
        for column in range(columns)
    )
    return NVFP4Tensor(
        rows=rows,
        columns=columns,
        packed_values=pack_e2m1(codes),
        scale_codes=bytes(
            0x38 + ((row * (columns // 16) + block) % 8)
            for row in range(rows)
            for block in range(columns // 16)
        ),
        scale_layout=ScaleFactorLayout.LINEAR,
        global_scale=0.5,
    )


class PackedPermutationFaultTests(unittest.TestCase):
    def test_permutations_are_detected_exactly_and_reversible(self) -> None:
        clean = _tensor()
        cases = (
            inject_packed_block_permutation(clean, 1),
            inject_packed_row_permutation(clean, 1),
            inject_packed_column_permutation(clean, 3),
        )
        expected = {
            FormatContractId.PACKED_VALUES,
            FormatContractId.RECONSTRUCTION,
        }

        for injection in cases:
            with self.subTest(kind=injection.kind):
                failed = {
                    outcome.spec.contract_id
                    for outcome in evaluate_format_contracts(
                        injection.clean, injection.faulted
                    )
                    if not outcome.passed
                }
                self.assertEqual(failed, expected)
                self.assertEqual(revert_fault(injection), clean)
                self.assertEqual(injection.label, "synthetic")

    def test_transforms_move_the_declared_logical_axis(self) -> None:
        block_rows = (
            (0,) * 16 + (1,) * 16,
            (2,) * 16 + (3,) * 16,
        )
        clean = NVFP4Tensor(
            rows=2,
            columns=32,
            packed_values=pack_e2m1(code for row in block_rows for code in row),
            scale_codes=bytes((0x38, 0x39, 0x3A, 0x3B)),
            scale_layout=ScaleFactorLayout.LINEAR,
            global_scale=1.0,
        )

        block_codes = unpack_e2m1(permute_packed_blocks(clean, 1).packed_values)
        self.assertEqual(block_codes[:32], (1,) * 16 + (0,) * 16)
        self.assertEqual(block_codes[32:], (3,) * 16 + (2,) * 16)

        row_codes = unpack_e2m1(permute_packed_rows(clean, 1).packed_values)
        self.assertEqual(row_codes[:32], block_rows[1])
        self.assertEqual(row_codes[32:], block_rows[0])

        column_codes = unpack_e2m1(
            permute_packed_columns(_tensor(rows=2, columns=32), 1).packed_values
        )
        original = unpack_e2m1(_tensor(rows=2, columns=32).packed_values)
        self.assertEqual(column_codes[:32], original[31:32] + original[:31])
        self.assertEqual(column_codes[32:], original[63:64] + original[32:63])

    def test_invalid_or_identity_offsets_are_rejected(self) -> None:
        clean = _tensor()
        with self.assertRaisesRegex(ValueError, "integer"):
            permute_packed_rows(clean, True)
        with self.assertRaisesRegex(ValueError, "must move"):
            permute_packed_blocks(clean, clean.columns // clean.block_size)
        with self.assertRaisesRegex(ValueError, "must move"):
            permute_packed_rows(clean, clean.rows)
        with self.assertRaisesRegex(ValueError, "must move"):
            permute_packed_columns(clean, clean.columns)


if __name__ == "__main__":
    unittest.main()
