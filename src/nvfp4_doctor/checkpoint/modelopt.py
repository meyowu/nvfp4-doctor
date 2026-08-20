"""CPU-only inspection of pinned ModelOpt NVFP4 checkpoint metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TARGET_PROJECTIONS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
REQUIRED_QUANT_TENSORS = (
    "input_scale",
    "weight",
    "weight_scale",
    "weight_scale_2",
)


class CheckpointMetadataError(ValueError):
    """Raised when pinned checkpoint metadata violates the declared contract."""


@dataclass(frozen=True, slots=True)
class ProjectionInventory:
    projection: str
    layer_count: int
    tensor_count: int
    tensor_suffixes: tuple[str, ...]
    shards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelOptCheckpointInspection:
    architecture: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    torch_dtype: str
    quant_algo: str
    group_size: int
    weight_num_bits: int
    input_num_bits: int
    kv_cache_quant_algo: str
    excluded_modules: tuple[str, ...]
    producer_name: str
    producer_version: str
    tensor_count: int
    total_parameters: int
    tensor_payload_bytes: int
    shards: tuple[str, ...]
    target_projections: tuple[ProjectionInventory, ...]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointMetadataError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointMetadataError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointMetadataError(f"{field} must be a non-negative integer")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CheckpointMetadataError(f"{field} must be a list of strings")
    return tuple(value)


def _require_equal(field: str, *values: object) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise CheckpointMetadataError(f"{field} declarations disagree")


def _quant_scheme(value: object, field: str) -> Mapping[str, object]:
    scheme = _mapping(value, field)
    if scheme.get("dynamic") is not False:
        raise CheckpointMetadataError(f"{field}.dynamic must be false")
    if scheme.get("type") != "float":
        raise CheckpointMetadataError(f"{field}.type must be float")
    return scheme


def _projection_inventory(
    weight_map: Mapping[str, object],
    projection: str,
    num_hidden_layers: int,
) -> ProjectionInventory:
    marker = f".{projection}."
    selected = {
        key: value
        for key, value in weight_map.items()
        if isinstance(key, str) and key.startswith("model.layers.") and marker in key
    }
    if not selected:
        raise CheckpointMetadataError(f"no tensors found for {projection}")
    if not all(isinstance(value, str) for value in selected.values()):
        raise CheckpointMetadataError(f"{projection} shard names must be strings")

    layers: set[int] = set()
    suffixes: set[str] = set()
    for key in selected:
        parts = key.split(".")
        try:
            layers.add(int(parts[2]))
        except (IndexError, ValueError) as error:
            raise CheckpointMetadataError(
                f"invalid layer-qualified tensor name: {key}"
            ) from error
        suffixes.add(parts[-1])

    expected_layers = set(range(num_hidden_layers))
    if layers != expected_layers:
        raise CheckpointMetadataError(f"{projection} layer coverage is incomplete")
    if suffixes != set(REQUIRED_QUANT_TENSORS):
        raise CheckpointMetadataError(
            f"{projection} quantization tensor suffixes are incomplete"
        )
    expected_tensor_count = num_hidden_layers * len(REQUIRED_QUANT_TENSORS)
    if len(selected) != expected_tensor_count:
        raise CheckpointMetadataError(
            f"{projection} tensor count does not match layer coverage"
        )

    return ProjectionInventory(
        projection=projection,
        layer_count=len(layers),
        tensor_count=len(selected),
        tensor_suffixes=tuple(sorted(suffixes)),
        shards=tuple(sorted({str(value) for value in selected.values()})),
    )


def inspect_modelopt_checkpoint(
    config: Mapping[str, object],
    hf_quant_config: Mapping[str, object],
    safetensors_index: Mapping[str, object],
) -> ModelOptCheckpointInspection:
    """Validate and summarize a pinned ModelOpt NVFP4 metadata snapshot."""
    architectures = _strings(config.get("architectures"), "architectures")
    if len(architectures) != 1:
        raise CheckpointMetadataError("exactly one architecture is required")

    quantization_config = _mapping(
        config.get("quantization_config"), "quantization_config"
    )
    standalone_quantization = _mapping(
        hf_quant_config.get("quantization"), "hf_quant_config.quantization"
    )
    config_groups = _mapping(
        quantization_config.get("config_groups"), "quantization_config.config_groups"
    )
    group = _mapping(config_groups.get("group_0"), "config_groups.group_0")
    weights = _quant_scheme(group.get("weights"), "group_0.weights")
    inputs = _quant_scheme(group.get("input_activations"), "group_0.input_activations")
    targets = _strings(group.get("targets"), "group_0.targets")
    if targets != ("Linear",):
        raise CheckpointMetadataError("group_0 must target only Linear modules")

    quant_algo = _string(quantization_config.get("quant_algo"), "quant_algo")
    standalone_algo = _string(
        standalone_quantization.get("quant_algo"), "standalone quant_algo"
    )
    _require_equal("quant_algo", quant_algo, standalone_algo, "NVFP4")

    group_size = _integer(weights.get("group_size"), "weights.group_size")
    _require_equal(
        "group_size",
        group_size,
        _integer(inputs.get("group_size"), "input_activations.group_size"),
        _integer(standalone_quantization.get("group_size"), "standalone group_size"),
    )
    weight_num_bits = _integer(weights.get("num_bits"), "weights.num_bits")
    input_num_bits = _integer(inputs.get("num_bits"), "input_activations.num_bits")
    _require_equal("NVFP4 bit width", weight_num_bits, input_num_bits, 4)

    ignored = tuple(sorted(_strings(quantization_config.get("ignore"), "ignore")))
    excluded = tuple(
        sorted(
            _strings(standalone_quantization.get("exclude_modules"), "exclude_modules")
        )
    )
    _require_equal("excluded modules", ignored, excluded)

    kv_cache = _quant_scheme(
        quantization_config.get("kv_cache_scheme"), "kv_cache_scheme"
    )
    if _integer(kv_cache.get("num_bits"), "kv_cache_scheme.num_bits") != 8:
        raise CheckpointMetadataError("kv_cache_scheme.num_bits must be 8")
    kv_cache_quant_algo = _string(
        standalone_quantization.get("kv_cache_quant_algo"), "kv_cache_quant_algo"
    )
    if kv_cache_quant_algo != "FP8":
        raise CheckpointMetadataError("kv_cache_quant_algo must be FP8")

    producer = _mapping(quantization_config.get("producer"), "producer")
    standalone_producer = _mapping(hf_quant_config.get("producer"), "hf producer")
    producer_name = _string(producer.get("name"), "producer.name")
    producer_version = _string(producer.get("version"), "producer.version")
    _require_equal(
        "producer name",
        producer_name,
        _string(standalone_producer.get("name"), "hf producer.name"),
    )
    _require_equal(
        "producer version",
        producer_version,
        _string(standalone_producer.get("version"), "hf producer.version"),
    )

    num_hidden_layers = _integer(config.get("num_hidden_layers"), "num_hidden_layers")
    weight_map = _mapping(safetensors_index.get("weight_map"), "weight_map")
    metadata = _mapping(safetensors_index.get("metadata"), "index metadata")
    if not all(isinstance(key, str) for key in weight_map):
        raise CheckpointMetadataError("weight_map keys must be strings")
    if not all(isinstance(value, str) for value in weight_map.values()):
        raise CheckpointMetadataError("weight_map shard names must be strings")

    projections = tuple(
        _projection_inventory(weight_map, projection, num_hidden_layers)
        for projection in TARGET_PROJECTIONS
    )
    return ModelOptCheckpointInspection(
        architecture=architectures[0],
        model_type=_string(config.get("model_type"), "model_type"),
        hidden_size=_integer(config.get("hidden_size"), "hidden_size"),
        intermediate_size=_integer(
            config.get("intermediate_size"), "intermediate_size"
        ),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=_integer(
            config.get("num_attention_heads"), "num_attention_heads"
        ),
        num_key_value_heads=_integer(
            config.get("num_key_value_heads"), "num_key_value_heads"
        ),
        vocab_size=_integer(config.get("vocab_size"), "vocab_size"),
        torch_dtype=_string(config.get("torch_dtype"), "torch_dtype"),
        quant_algo=quant_algo,
        group_size=group_size,
        weight_num_bits=weight_num_bits,
        input_num_bits=input_num_bits,
        kv_cache_quant_algo=kv_cache_quant_algo,
        excluded_modules=ignored,
        producer_name=producer_name,
        producer_version=producer_version,
        tensor_count=len(weight_map),
        total_parameters=_integer(
            metadata.get("total_parameters"), "metadata.total_parameters"
        ),
        tensor_payload_bytes=_integer(
            metadata.get("total_size"), "metadata.total_size"
        ),
        shards=tuple(sorted({str(value) for value in weight_map.values()})),
        target_projections=projections,
    )
