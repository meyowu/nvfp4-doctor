import functools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nvfp4_doctor.capture.e004_fused import (
    E004_FUSED_REAL_ACTIVATION_CASES,
    E004FusedRealActivationCase,
)
from nvfp4_doctor.formats import swizzle_scales_128x4
from scripts.run_e004_real_activation_fused_matrix import (
    _CAPTURE_EVENT_ORDER,
    _CAPTURES,
    RealActivationFusedMatrixError,
    _assemble_fused_payloads,
    _capture_input,
    _capture_output,
    _component_output_slices,
    _expected_hook_event_order,
    _sha256_bytes,
    _verify_used_shards,
)


def _small_fused_case() -> E004FusedRealActivationCase:
    return E004FusedRealActivationCase(
        layer=0,
        role="test",
        projection="pair_proj",
        module_path="model.layers.0.test.pair_proj",
        module_class="TestParallelLinear",
        checkpoint_parent_path="model.layers.0.test",
        component_projections=("left_proj", "right_proj"),
        component_output_widths=(128, 128),
        input_width=64,
    )


class E004RealActivationFusedMatrixCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        _CAPTURES.clear()
        _CAPTURE_EVENT_ORDER.clear()

    def test_hook_order_covers_six_fused_modules_and_twelve_events(self) -> None:
        events = _expected_hook_event_order()
        self.assertEqual(len(events), 12)
        self.assertEqual(
            events[:4],
            [
                "layer-00-qkv-proj:input",
                "layer-00-qkv-proj:module_output",
                "layer-00-gate-up-proj:input",
                "layer-00-gate-up-proj:module_output",
            ],
        )

    def test_bound_callbacks_keep_fused_case_state_separate(self) -> None:
        qkv = torch.zeros((9, 4096), dtype=torch.bfloat16)
        gate_up = torch.ones((9, 4096), dtype=torch.bfloat16)
        qkv_hook = functools.partial(_capture_input, "layer-00-qkv-proj")
        gate_up_hook = functools.partial(_capture_input, "layer-00-gate-up-proj")
        with patch(
            "scripts.run_e004_real_activation_fused_matrix."
            "_copy_to_cpu_preserving_stride",
            side_effect=lambda tensor: tensor.clone(),
        ):
            qkv_hook(torch.nn.Identity(), (qkv,))
            gate_up_hook(torch.nn.Identity(), (gate_up,))
        self.assertEqual(set(_CAPTURES), {"layer-00-qkv-proj", "layer-00-gate-up-proj"})
        self.assertEqual(
            _CAPTURE_EVENT_ORDER,
            ["layer-00-qkv-proj:input", "layer-00-gate-up-proj:input"],
        )

    def test_duplicate_hook_event_is_rejected(self) -> None:
        tensor = torch.zeros((9, 4096), dtype=torch.bfloat16)
        with patch(
            "scripts.run_e004_real_activation_fused_matrix."
            "_copy_to_cpu_preserving_stride",
            side_effect=lambda value: value.clone(),
        ):
            _capture_input("layer-00-qkv-proj", torch.nn.Identity(), (tensor,))
            with self.assertRaisesRegex(
                RealActivationFusedMatrixError, "more than once"
            ):
                _capture_input("layer-00-qkv-proj", torch.nn.Identity(), (tensor,))

    def test_output_hook_accepts_vllm_linear_tuple(self) -> None:
        tensor = torch.zeros((9, 6144), dtype=torch.bfloat16)
        with patch(
            "scripts.run_e004_real_activation_fused_matrix."
            "_copy_to_cpu_preserving_stride",
            side_effect=lambda value: value.clone(),
        ):
            _capture_output(
                "layer-00-qkv-proj",
                torch.nn.Identity(),
                (tensor,),
                (tensor, None),
            )
        self.assertIn("module_output", _CAPTURES["layer-00-qkv-proj"])

    def test_source_assembly_concatenates_rows_before_full_scale_swizzle(self) -> None:
        case = _small_fused_case()
        packed_parts = [bytes([0x12]) * 4096, bytes([0x34]) * 4096]
        scale_parts = [bytes(range(128)) * 4, bytes(reversed(range(128))) * 4]
        fused_weight, fused_scale, component_scales = _assemble_fused_payloads(
            case, packed_parts, scale_parts
        )
        self.assertEqual(fused_weight, b"".join(packed_parts))
        self.assertEqual(
            fused_scale,
            swizzle_scales_128x4(b"".join(scale_parts), 256, 4),
        )
        self.assertEqual(fused_scale, b"".join(component_scales))

    def test_component_output_slices_are_independently_byte_exact(self) -> None:
        case = _small_fused_case()
        captured = (
            torch.arange(512, dtype=torch.float32).reshape(2, 256).to(torch.bfloat16)
        )
        evidence = _component_output_slices(
            case, captured, [captured.clone() for _ in range(3)]
        )
        self.assertEqual(
            [entry["feature_range"] for entry in evidence],
            [[0, 128], [128, 256]],
        )
        for entry in evidence:
            self.assertEqual(entry["replay_sha256s"], [entry["captured_sha256"]] * 3)
            self.assertEqual(entry["logical_matches"], [True, True, True])

    def test_used_checkpoint_shard_is_stream_hashed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory).resolve()
            shard = model_dir / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"sealed shard")
            weight_map = {
                tensor_name: shard.name
                for case in E004_FUSED_REAL_ACTIVATION_CASES
                for boundary in case.component_boundaries
                for tensor_name in (
                    f"{case.checkpoint_parent_path}.{boundary.projection}.input_scale",
                    f"{case.checkpoint_parent_path}.{boundary.projection}.weight",
                    f"{case.checkpoint_parent_path}.{boundary.projection}.weight_scale",
                    f"{case.checkpoint_parent_path}.{boundary.projection}.weight_scale_2",
                )
            }
            inventory = {
                shard.name: {
                    "role": "weight_shard",
                    "sha256": _sha256_bytes(b"sealed shard"),
                }
            }
            with patch(
                "scripts.run_e004_real_activation_fused_matrix._sha256_path",
                wraps=lambda path: _sha256_bytes(path.read_bytes()),
            ) as stream_hash:
                verified = _verify_used_shards(
                    model_dir=model_dir,
                    weight_map=weight_map,
                    inventory=inventory,
                )
            self.assertEqual(verified, (shard.name,))
            stream_hash.assert_called_once_with(shard)


if __name__ == "__main__":
    unittest.main()
