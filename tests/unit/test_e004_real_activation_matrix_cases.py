import unittest

from nvfp4_doctor.capture import E004_UNFUSED_REAL_ACTIVATION_CASES


class E004RealActivationMatrixCaseTests(unittest.TestCase):
    def test_case_set_is_the_frozen_unfused_matrix(self) -> None:
        self.assertEqual(
            [
                (case.layer, case.projection)
                for case in E004_UNFUSED_REAL_ACTIVATION_CASES
            ],
            [
                (0, "o_proj"),
                (0, "down_proj"),
                (18, "o_proj"),
                (18, "down_proj"),
                (35, "o_proj"),
                (35, "down_proj"),
            ],
        )
        self.assertEqual(
            [case.role for case in E004_UNFUSED_REAL_ACTIVATION_CASES],
            ["early", "early", "middle", "middle", "late", "late"],
        )

    def test_paths_ranges_and_artifact_slugs_are_unique(self) -> None:
        for attribute in ("module_path", "target_nvtx_range", "artifact_slug"):
            values = {
                getattr(case, attribute) for case in E004_UNFUSED_REAL_ACTIVATION_CASES
            }
            self.assertEqual(len(values), 6)

    def test_module_boundaries_and_shapes_are_explicit(self) -> None:
        for case in E004_UNFUSED_REAL_ACTIVATION_CASES:
            self.assertEqual(case.output_shape(9), (9, 4096))
            if case.projection == "o_proj":
                self.assertEqual(
                    case.module_path,
                    f"model.layers.{case.layer}.self_attn.o_proj",
                )
                self.assertEqual(case.input_shape(9), (9, 4096))
                self.assertEqual(case.packed_weight_shape, (4096, 2048))
                self.assertEqual(case.weight_scale_shape, (4096, 256))
            else:
                self.assertEqual(
                    case.module_path,
                    f"model.layers.{case.layer}.mlp.down_proj",
                )
                self.assertEqual(case.input_shape(9), (9, 12288))
                self.assertEqual(case.packed_weight_shape, (4096, 6144))
                self.assertEqual(case.weight_scale_shape, (4096, 768))


if __name__ == "__main__":
    unittest.main()
