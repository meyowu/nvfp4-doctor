import unittest
from itertools import pairwise

from nvfp4_doctor.capture.e004_fused import (
    E004_FUSED_REAL_ACTIVATION_CASES,
    MODEL_OPT_TENSOR_SUFFIXES,
)


class E004FusedRealActivationCaseTests(unittest.TestCase):
    def test_case_set_is_the_frozen_fused_matrix(self) -> None:
        self.assertEqual(
            [
                (case.layer, case.role, case.projection)
                for case in E004_FUSED_REAL_ACTIVATION_CASES
            ],
            [
                (0, "early", "qkv_proj"),
                (0, "early", "gate_up_proj"),
                (18, "middle", "qkv_proj"),
                (18, "middle", "gate_up_proj"),
                (35, "late", "qkv_proj"),
                (35, "late", "gate_up_proj"),
            ],
        )
        self.assertTrue(
            all(
                case.adapter_scope == "production_aligned_fused"
                for case in E004_FUSED_REAL_ACTIVATION_CASES
            )
        )

    def test_qkv_metadata_preserves_grouped_query_attention_widths(self) -> None:
        case = E004_FUSED_REAL_ACTIVATION_CASES[0]
        self.assertEqual(case.module_path, "model.layers.0.self_attn.qkv_proj")
        self.assertEqual(case.module_class, "QKVParallelLinear")
        self.assertEqual(case.component_projections, ("q_proj", "k_proj", "v_proj"))
        self.assertEqual(case.component_output_widths, (4096, 1024, 1024))
        self.assertEqual(case.component_row_boundaries, (0, 4096, 5120, 6144))
        self.assertEqual(case.input_shape(9), (9, 4096))
        self.assertEqual(case.output_shape(9), (9, 6144))
        self.assertEqual(case.packed_weight_shape, (6144, 2048))
        self.assertEqual(case.weight_scale_shape, (6144, 256))
        self.assertEqual(
            [boundary.packed_weight_shape for boundary in case.component_boundaries],
            [(4096, 2048), (1024, 2048), (1024, 2048)],
        )
        self.assertEqual(
            [boundary.weight_scale_shape for boundary in case.component_boundaries],
            [(4096, 256), (1024, 256), (1024, 256)],
        )

    def test_gate_up_metadata_preserves_component_order_and_widths(self) -> None:
        case = E004_FUSED_REAL_ACTIVATION_CASES[1]
        self.assertEqual(case.module_path, "model.layers.0.mlp.gate_up_proj")
        self.assertEqual(case.module_class, "MergedColumnParallelLinear")
        self.assertEqual(case.component_projections, ("gate_proj", "up_proj"))
        self.assertEqual(case.component_output_widths, (12288, 12288))
        self.assertEqual(case.component_row_boundaries, (0, 12288, 24576))
        self.assertEqual(case.input_shape(9), (9, 4096))
        self.assertEqual(case.output_shape(9), (9, 24576))
        self.assertEqual(case.packed_weight_shape, (24576, 2048))
        self.assertEqual(case.weight_scale_shape, (24576, 256))

    def test_component_boundaries_are_exact_and_cutlass_aligned(self) -> None:
        for case in E004_FUSED_REAL_ACTIVATION_CASES:
            boundaries = case.component_boundaries
            self.assertEqual(boundaries[0].row_start, 0)
            self.assertEqual(boundaries[-1].row_end, case.output_width)
            self.assertTrue(
                all(boundary.cutlass_row_aligned for boundary in boundaries)
            )
            for previous, current in pairwise(boundaries):
                self.assertEqual(previous.row_end, current.row_start)
            for boundary in boundaries:
                self.assertEqual(
                    case.component_boundary(boundary.projection),
                    boundary,
                )
        with self.assertRaisesRegex(KeyError, "unknown fused component"):
            E004_FUSED_REAL_ACTIVATION_CASES[0].component_boundary("o_proj")

    def test_checkpoint_tensor_names_keep_component_order(self) -> None:
        qkv = E004_FUSED_REAL_ACTIVATION_CASES[2]
        self.assertEqual(
            qkv.component_tensor_names("weight"),
            (
                "model.layers.18.self_attn.q_proj.weight",
                "model.layers.18.self_attn.k_proj.weight",
                "model.layers.18.self_attn.v_proj.weight",
            ),
        )
        gate_up = E004_FUSED_REAL_ACTIVATION_CASES[-1]
        self.assertEqual(
            gate_up.component_tensor_names("weight_scale_2"),
            (
                "model.layers.35.mlp.gate_proj.weight_scale_2",
                "model.layers.35.mlp.up_proj.weight_scale_2",
            ),
        )
        self.assertEqual(
            MODEL_OPT_TENSOR_SUFFIXES,
            ("input_scale", "weight", "weight_scale", "weight_scale_2"),
        )
        with self.assertRaisesRegex(ValueError, "unsupported ModelOpt"):
            qkv.component_tensor_names("bias")

    def test_paths_ranges_and_artifact_slugs_are_unique_and_frozen(self) -> None:
        for attribute in (
            "case_id",
            "artifact_slug",
            "module_path",
            "target_nvtx_range",
        ):
            values = {
                getattr(case, attribute) for case in E004_FUSED_REAL_ACTIVATION_CASES
            }
            self.assertEqual(len(values), 6)
        self.assertEqual(
            E004_FUSED_REAL_ACTIVATION_CASES[0].target_nvtx_range,
            "e004:real_activation_fused_matrix:layer_00:qkv_proj:nvfp4_gemm",
        )
        self.assertEqual(
            E004_FUSED_REAL_ACTIVATION_CASES[-1].target_nvtx_range,
            "e004:real_activation_fused_matrix:layer_35:gate_up_proj:nvfp4_gemm",
        )

    def test_shape_helpers_reject_empty_token_batches(self) -> None:
        case = E004_FUSED_REAL_ACTIVATION_CASES[0]
        with self.assertRaisesRegex(ValueError, "token count"):
            case.input_shape(0)
        with self.assertRaisesRegex(ValueError, "token count"):
            case.output_shape(-1)


if __name__ == "__main__":
    unittest.main()
