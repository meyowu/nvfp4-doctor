import unittest

from nvfp4_doctor.contracts import (
    FORMAT_CONTRACT_SPECS,
    FormatContractId,
    evaluate_format_contracts,
)
from nvfp4_doctor.faults import (
    NVFP4FaultKind,
    inject_block_scale_reversal,
    inject_global_scale_multiplier,
    inject_nibble_swap,
    inject_padding_corruption,
    inject_scale_index_shift,
    inject_scale_layout_mislabel,
    revert_fault,
    shift_scale_indices,
    toggle_padding_byte,
)
from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    pack_e2m1,
    swizzle_scales_128x4,
)
from nvfp4_doctor.oracle import NVFP4Tensor


def _packed_values(rows: int, columns: int) -> bytes:
    pattern = tuple(range(16))
    codes = tuple(
        pattern[(row * columns + column) % len(pattern)]
        for row in range(rows)
        for column in range(columns)
    )
    return pack_e2m1(codes)


def _linear_tensor() -> NVFP4Tensor:
    return NVFP4Tensor(
        rows=2,
        columns=32,
        packed_values=_packed_values(2, 32),
        scale_codes=bytes((0x38, 0x40, 0x48, 0x50)),
        scale_layout=ScaleFactorLayout.LINEAR,
        global_scale=0.5,
    )


def _cutlass_tensor(rows: int, columns: int) -> NVFP4Tensor:
    scale_columns = columns // 16
    linear_scales = bytes(
        0x38 + ((row * scale_columns + column) % 16)
        for row in range(rows)
        for column in range(scale_columns)
    )
    return NVFP4Tensor(
        rows=rows,
        columns=columns,
        packed_values=_packed_values(rows, columns),
        scale_codes=swizzle_scales_128x4(
            linear_scales, rows, scale_columns, padding_code=0
        ),
        scale_layout=ScaleFactorLayout.CUTLASS_128X4,
        global_scale=0.5,
    )


class FormatFaultTests(unittest.TestCase):
    def test_every_contract_has_an_auditable_exact_specification(self) -> None:
        self.assertEqual(len(FORMAT_CONTRACT_SPECS), 6)
        for spec in FORMAT_CONTRACT_SPECS:
            with self.subTest(contract=spec.contract_id):
                self.assertTrue(spec.domain)
                self.assertTrue(spec.preconditions)
                self.assertTrue(spec.invariant)
                self.assertTrue(spec.metric)
                self.assertEqual(spec.threshold, 0)
                self.assertTrue(spec.limitations)

    def test_clean_artifacts_pass_every_exact_contract(self) -> None:
        for clean in (_linear_tensor(), _cutlass_tensor(129, 80)):
            with self.subTest(shape=(clean.rows, clean.columns)):
                outcomes = evaluate_format_contracts(clean, clean)
                self.assertTrue(all(outcome.passed for outcome in outcomes))

    def test_positive_controls_are_detected_and_reversible(self) -> None:
        linear = _linear_tensor()
        layout = _cutlass_tensor(128, 64)
        padded = _cutlass_tensor(129, 80)
        cases = (
            (
                inject_nibble_swap(linear),
                {FormatContractId.PACKED_VALUES, FormatContractId.RECONSTRUCTION},
            ),
            (
                inject_scale_index_shift(linear, 1),
                {FormatContractId.LOGICAL_SCALES, FormatContractId.RECONSTRUCTION},
            ),
            (
                inject_block_scale_reversal(linear),
                {FormatContractId.LOGICAL_SCALES, FormatContractId.RECONSTRUCTION},
            ),
            (
                inject_global_scale_multiplier(linear, 2.0),
                {FormatContractId.GLOBAL_SCALE, FormatContractId.RECONSTRUCTION},
            ),
            (
                inject_scale_layout_mislabel(layout, ScaleFactorLayout.LINEAR),
                {
                    FormatContractId.METADATA,
                    FormatContractId.LOGICAL_SCALES,
                    FormatContractId.RECONSTRUCTION,
                },
            ),
            (
                inject_padding_corruption(padded),
                {FormatContractId.SCALE_PADDING},
            ),
        )

        for injection, required_failures in cases:
            with self.subTest(kind=injection.kind):
                outcomes = evaluate_format_contracts(injection.clean, injection.faulted)
                failures = {
                    outcome.spec.contract_id
                    for outcome in outcomes
                    if not outcome.passed
                }
                self.assertTrue(required_failures <= failures)
                self.assertEqual(revert_fault(injection), injection.clean)
                self.assertEqual(injection.label, "synthetic")

    def test_padding_fault_is_structural_but_numerically_silent(self) -> None:
        injection = inject_padding_corruption(_cutlass_tensor(129, 80))
        outcomes = {
            outcome.spec.contract_id: outcome
            for outcome in evaluate_format_contracts(injection.clean, injection.faulted)
        }
        self.assertFalse(outcomes[FormatContractId.SCALE_PADDING].passed)
        self.assertTrue(outcomes[FormatContractId.RECONSTRUCTION].passed)

    def test_transform_preconditions_reject_invalid_controls(self) -> None:
        linear = _linear_tensor()
        with self.assertRaisesRegex(ValueError, "must move"):
            shift_scale_indices(linear, 2)
        with self.assertRaisesRegex(ValueError, "CUTLASS"):
            toggle_padding_byte(linear, 0)
        with self.assertRaisesRegex(ValueError, "positive-control"):
            inject_nibble_swap(
                NVFP4Tensor(
                    rows=1,
                    columns=16,
                    packed_values=bytes((0x11,) * 8),
                    scale_codes=bytes((0x38,)),
                    scale_layout=ScaleFactorLayout.LINEAR,
                    global_scale=1.0,
                )
            )

    def test_fault_kind_values_are_stable_labels(self) -> None:
        self.assertEqual(
            {kind.value for kind in NVFP4FaultKind},
            {
                "nibble_swap",
                "scale_index_shift",
                "block_scale_reversal",
                "global_scale_multiplier",
                "scale_layout_mislabel",
                "padding_corruption",
            },
        )


if __name__ == "__main__":
    unittest.main()
