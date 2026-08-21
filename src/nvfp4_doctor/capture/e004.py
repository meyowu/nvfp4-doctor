"""CPU-only case definitions for the E004 real-activation matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class E004RealActivationCase:
    """One production-aligned unfused module capture boundary."""

    layer: int
    role: str
    projection: str
    module_path: str
    input_width: int
    output_width: int

    @property
    def case_id(self) -> str:
        return f"layer-{self.layer:02d}-{self.projection.replace('_', '-')}"

    @property
    def artifact_slug(self) -> str:
        return self.case_id

    @property
    def target_nvtx_range(self) -> str:
        return (
            "e004:real_activation_matrix:"
            f"layer_{self.layer:02d}:{self.projection}:nvfp4_gemm"
        )

    @property
    def packed_weight_shape(self) -> tuple[int, int]:
        return (self.output_width, self.input_width // 2)

    @property
    def weight_scale_shape(self) -> tuple[int, int]:
        return (self.output_width, self.input_width // 16)

    def input_shape(self, token_count: int) -> tuple[int, int]:
        return (token_count, self.input_width)

    def output_shape(self, token_count: int) -> tuple[int, int]:
        return (token_count, self.output_width)


def _case(layer: int, role: str, projection: str) -> E004RealActivationCase:
    if projection == "o_proj":
        module_path = f"model.layers.{layer}.self_attn.o_proj"
        input_width = 4096
    elif projection == "down_proj":
        module_path = f"model.layers.{layer}.mlp.down_proj"
        input_width = 12288
    else:  # pragma: no cover - construction is frozen below
        raise ValueError(f"unsupported unfused projection: {projection}")
    return E004RealActivationCase(
        layer=layer,
        role=role,
        projection=projection,
        module_path=module_path,
        input_width=input_width,
        output_width=4096,
    )


E004_UNFUSED_REAL_ACTIVATION_CASES = tuple(
    _case(layer, role, projection)
    for layer, role in ((0, "early"), (18, "middle"), (35, "late"))
    for projection in ("o_proj", "down_proj")
)

if len({case.case_id for case in E004_UNFUSED_REAL_ACTIVATION_CASES}) != 6:
    raise RuntimeError("E004 real-activation case identifiers must be unique")
if len({case.target_nvtx_range for case in E004_UNFUSED_REAL_ACTIVATION_CASES}) != 6:
    raise RuntimeError("E004 real-activation NVTX ranges must be unique")
