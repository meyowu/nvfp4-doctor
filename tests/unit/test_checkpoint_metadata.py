import copy
import unittest

from nvfp4_doctor.checkpoint import (
    CheckpointMetadataError,
    inspect_modelopt_checkpoint,
)


def _metadata_fixture() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    config: dict[str, object] = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 16,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head"],
            "quant_algo": "NVFP4",
            "kv_cache_scheme": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float",
            },
            "producer": {"name": "modelopt", "version": "0.35.0"},
        },
    }
    hf_quant_config: dict[str, object] = {
        "producer": {"name": "modelopt", "version": "0.35.0"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": ["lm_head"],
        },
    }
    weight_map = {
        f"model.layers.0.{scope}.{projection}.{suffix}": "model-00001.safetensors"
        for scope, projection in (
            ("self_attn", "q_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_proj"),
            ("mlp", "up_proj"),
            ("mlp", "down_proj"),
        )
        for suffix in ("input_scale", "weight", "weight_scale", "weight_scale_2")
    }
    index: dict[str, object] = {
        "metadata": {"total_parameters": 123, "total_size": 456},
        "weight_map": weight_map,
    }
    return config, hf_quant_config, index


class CheckpointMetadataTests(unittest.TestCase):
    def test_inspection_preserves_quantization_and_projection_facts(self) -> None:
        inspection = inspect_modelopt_checkpoint(*_metadata_fixture())

        self.assertEqual(inspection.architecture, "Qwen3ForCausalLM")
        self.assertEqual(inspection.quant_algo, "NVFP4")
        self.assertEqual(inspection.group_size, 16)
        self.assertEqual(inspection.weight_num_bits, 4)
        self.assertEqual(inspection.input_num_bits, 4)
        self.assertEqual(inspection.kv_cache_quant_algo, "FP8")
        self.assertEqual(inspection.excluded_modules, ("lm_head",))
        self.assertEqual(inspection.producer_name, "modelopt")
        self.assertEqual(inspection.producer_version, "0.35.0")
        self.assertEqual(inspection.tensor_count, 20)
        self.assertEqual(inspection.total_parameters, 123)
        self.assertEqual(inspection.tensor_payload_bytes, 456)
        self.assertEqual(len(inspection.target_projections), 5)
        for projection in inspection.target_projections:
            with self.subTest(projection=projection.projection):
                self.assertEqual(projection.layer_count, 1)
                self.assertEqual(projection.tensor_count, 4)
                self.assertEqual(
                    projection.tensor_suffixes,
                    ("input_scale", "weight", "weight_scale", "weight_scale_2"),
                )

    def test_conflicting_quantization_declarations_are_rejected(self) -> None:
        config, hf_quant_config, index = _metadata_fixture()
        conflicting = copy.deepcopy(hf_quant_config)
        conflicting["quantization"]["group_size"] = 32  # type: ignore[index]

        with self.assertRaisesRegex(CheckpointMetadataError, "group_size"):
            inspect_modelopt_checkpoint(config, conflicting, index)

    def test_incomplete_projection_metadata_is_rejected(self) -> None:
        config, hf_quant_config, index = _metadata_fixture()
        incomplete = copy.deepcopy(index)
        weight_map = incomplete["weight_map"]
        assert isinstance(weight_map, dict)
        del weight_map["model.layers.0.self_attn.q_proj.weight_scale_2"]

        with self.assertRaisesRegex(CheckpointMetadataError, "q_proj"):
            inspect_modelopt_checkpoint(config, hf_quant_config, incomplete)


if __name__ == "__main__":
    unittest.main()
