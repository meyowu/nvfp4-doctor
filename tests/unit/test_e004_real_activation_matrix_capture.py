import functools
import unittest
from unittest.mock import patch

import torch

from scripts.run_e004_real_activation_matrix import (
    _CAPTURE_EVENT_ORDER,
    _CAPTURES,
    RealActivationMatrixError,
    _capture_input,
    _capture_output,
    _load_expected_runtime_identities,
)


class E004RealActivationMatrixCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        _CAPTURES.clear()
        _CAPTURE_EVENT_ORDER.clear()

    def test_runtime_identity_dependency_covers_six_unfused_cases(self) -> None:
        identities = _load_expected_runtime_identities()
        self.assertEqual(
            list(identities),
            [
                "layer-00-o-proj",
                "layer-00-down-proj",
                "layer-18-o-proj",
                "layer-18-down-proj",
                "layer-35-o-proj",
                "layer-35-down-proj",
            ],
        )
        for identity in identities.values():
            self.assertEqual(len(identity["runtime_packed_weight_sha256"]), 64)
            self.assertEqual(len(identity["runtime_weight_scale_sha256"]), 64)

    def test_bound_callbacks_keep_case_state_separate(self) -> None:
        first = torch.zeros((9, 4096), dtype=torch.bfloat16)
        second = torch.ones((9, 12288), dtype=torch.bfloat16)
        first_hook = functools.partial(_capture_input, "layer-00-o-proj")
        second_hook = functools.partial(_capture_input, "layer-00-down-proj")
        with patch(
            "scripts.run_e004_real_activation_matrix._copy_to_cpu_preserving_stride",
            side_effect=lambda tensor: tensor.clone(),
        ):
            first_hook(torch.nn.Identity(), (first,))
            second_hook(torch.nn.Identity(), (second,))
        self.assertEqual(set(_CAPTURES), {"layer-00-o-proj", "layer-00-down-proj"})
        self.assertEqual(
            _CAPTURE_EVENT_ORDER,
            ["layer-00-o-proj:input", "layer-00-down-proj:input"],
        )
        self.assertEqual(
            _CAPTURES["layer-00-o-proj"]["input"]["tensor"].shape,
            (9, 4096),
        )
        self.assertEqual(
            _CAPTURES["layer-00-down-proj"]["input"]["tensor"].shape,
            (9, 12288),
        )

    def test_duplicate_hook_event_is_rejected(self) -> None:
        tensor = torch.zeros((9, 4096), dtype=torch.bfloat16)
        with patch(
            "scripts.run_e004_real_activation_matrix._copy_to_cpu_preserving_stride",
            side_effect=lambda value: value.clone(),
        ):
            _capture_input("layer-00-o-proj", torch.nn.Identity(), (tensor,))
            with self.assertRaisesRegex(RealActivationMatrixError, "more than once"):
                _capture_input("layer-00-o-proj", torch.nn.Identity(), (tensor,))

    def test_output_hook_accepts_the_vllm_linear_tuple(self) -> None:
        tensor = torch.zeros((9, 4096), dtype=torch.bfloat16)
        with patch(
            "scripts.run_e004_real_activation_matrix._copy_to_cpu_preserving_stride",
            side_effect=lambda value: value.clone(),
        ):
            _capture_output(
                "layer-00-o-proj",
                torch.nn.Identity(),
                (tensor,),
                (tensor, None),
            )
        self.assertIn("module_output", _CAPTURES["layer-00-o-proj"])


if __name__ == "__main__":
    unittest.main()
