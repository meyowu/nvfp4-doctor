"""CPU-only case definitions for E004 fused real-activation replay."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_HIDDEN_SIZE = 4096
NVFP4_GROUP_SIZE = 16
CUTLASS_SCALE_ATOM_ROWS = 128
MODEL_OPT_TENSOR_SUFFIXES = (
    "input_scale",
    "weight",
    "weight_scale",
    "weight_scale_2",
)


@dataclass(frozen=True, slots=True)
class E004FusedComponentBoundary:
    """One checkpoint projection's exact row interval in a fused module."""

    projection: str
    row_start: int
    row_end: int
    input_width: int

    def __post_init__(self) -> None:
        if not self.projection:
            raise ValueError("component projection must be non-empty")
        if self.row_start < 0 or self.row_end <= self.row_start:
            raise ValueError("component row interval must be non-empty")
        if self.input_width <= 0 or self.input_width % NVFP4_GROUP_SIZE:
            raise ValueError("component input width must be NVFP4-group aligned")

    @property
    def output_width(self) -> int:
        return self.row_end - self.row_start

    @property
    def packed_weight_shape(self) -> tuple[int, int]:
        return (self.output_width, self.input_width // 2)

    @property
    def weight_scale_shape(self) -> tuple[int, int]:
        return (self.output_width, self.input_width // NVFP4_GROUP_SIZE)

    @property
    def cutlass_row_aligned(self) -> bool:
        return (
            self.row_start % CUTLASS_SCALE_ATOM_ROWS == 0
            and self.row_end % CUTLASS_SCALE_ATOM_ROWS == 0
        )


@dataclass(frozen=True, slots=True)
class E004FusedRealActivationCase:
    """One production fused module and its ordered checkpoint components."""

    layer: int
    role: str
    projection: str
    module_path: str
    module_class: str
    checkpoint_parent_path: str
    component_projections: tuple[str, ...]
    component_output_widths: tuple[int, ...]
    input_width: int = MODEL_HIDDEN_SIZE

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if not all((self.role, self.projection, self.module_path, self.module_class)):
            raise ValueError("case identity fields must be non-empty")
        if len(self.component_projections) < 2 or len(
            self.component_projections
        ) != len(self.component_output_widths):
            raise ValueError("fused cases require matching component metadata")
        if len(set(self.component_projections)) != len(self.component_projections):
            raise ValueError("component projections must be unique")
        if self.input_width <= 0 or self.input_width % NVFP4_GROUP_SIZE:
            raise ValueError("input width must be NVFP4-group aligned")
        if any(
            width <= 0 or width % CUTLASS_SCALE_ATOM_ROWS
            for width in self.component_output_widths
        ):
            raise ValueError("component output widths must be CUTLASS-row aligned")
        if self.module_path.rsplit(".", 1)[0] != self.checkpoint_parent_path:
            raise ValueError("checkpoint parent must match the fused module parent")

    @property
    def case_id(self) -> str:
        return f"layer-{self.layer:02d}-{self.projection.replace('_', '-')}"

    @property
    def artifact_slug(self) -> str:
        return self.case_id

    @property
    def adapter_scope(self) -> str:
        return "production_aligned_fused"

    @property
    def target_nvtx_range(self) -> str:
        return (
            "e004:real_activation_fused_matrix:"
            f"layer_{self.layer:02d}:{self.projection}:nvfp4_gemm"
        )

    @property
    def output_width(self) -> int:
        return sum(self.component_output_widths)

    @property
    def packed_weight_shape(self) -> tuple[int, int]:
        return (self.output_width, self.input_width // 2)

    @property
    def weight_scale_shape(self) -> tuple[int, int]:
        return (self.output_width, self.input_width // NVFP4_GROUP_SIZE)

    @property
    def component_boundaries(self) -> tuple[E004FusedComponentBoundary, ...]:
        boundaries: list[E004FusedComponentBoundary] = []
        row_start = 0
        for projection, output_width in zip(
            self.component_projections,
            self.component_output_widths,
            strict=True,
        ):
            row_end = row_start + output_width
            boundaries.append(
                E004FusedComponentBoundary(
                    projection=projection,
                    row_start=row_start,
                    row_end=row_end,
                    input_width=self.input_width,
                )
            )
            row_start = row_end
        return tuple(boundaries)

    @property
    def component_row_boundaries(self) -> tuple[int, ...]:
        return (0, *(boundary.row_end for boundary in self.component_boundaries))

    def component_boundary(self, projection: str) -> E004FusedComponentBoundary:
        matches = tuple(
            boundary
            for boundary in self.component_boundaries
            if boundary.projection == projection
        )
        if len(matches) != 1:
            raise KeyError(f"unknown fused component: {projection}")
        return matches[0]

    def component_tensor_names(self, suffix: str) -> tuple[str, ...]:
        if suffix not in MODEL_OPT_TENSOR_SUFFIXES:
            raise ValueError(f"unsupported ModelOpt tensor suffix: {suffix}")
        return tuple(
            f"{self.checkpoint_parent_path}.{projection}.{suffix}"
            for projection in self.component_projections
        )

    def input_shape(self, token_count: int) -> tuple[int, int]:
        if token_count <= 0:
            raise ValueError("token count must be positive")
        return (token_count, self.input_width)

    def output_shape(self, token_count: int) -> tuple[int, int]:
        if token_count <= 0:
            raise ValueError("token count must be positive")
        return (token_count, self.output_width)


def _case(layer: int, role: str, projection: str) -> E004FusedRealActivationCase:
    if projection == "qkv_proj":
        checkpoint_parent = f"model.layers.{layer}.self_attn"
        module_class = "QKVParallelLinear"
        component_projections = ("q_proj", "k_proj", "v_proj")
        component_output_widths = (4096, 1024, 1024)
    elif projection == "gate_up_proj":
        checkpoint_parent = f"model.layers.{layer}.mlp"
        module_class = "MergedColumnParallelLinear"
        component_projections = ("gate_proj", "up_proj")
        component_output_widths = (12288, 12288)
    else:  # pragma: no cover - construction is frozen below
        raise ValueError(f"unsupported fused projection: {projection}")
    return E004FusedRealActivationCase(
        layer=layer,
        role=role,
        projection=projection,
        module_path=f"{checkpoint_parent}.{projection}",
        module_class=module_class,
        checkpoint_parent_path=checkpoint_parent,
        component_projections=component_projections,
        component_output_widths=component_output_widths,
    )


E004_FUSED_REAL_ACTIVATION_CASES = tuple(
    _case(layer, role, projection)
    for layer, role in ((0, "early"), (18, "middle"), (35, "late"))
    for projection in ("qkv_proj", "gate_up_proj")
)

if len({case.case_id for case in E004_FUSED_REAL_ACTIVATION_CASES}) != 6:
    raise RuntimeError("E004 fused case identifiers must be unique")
if len({case.module_path for case in E004_FUSED_REAL_ACTIVATION_CASES}) != 6:
    raise RuntimeError("E004 fused module paths must be unique")
if len({case.target_nvtx_range for case in E004_FUSED_REAL_ACTIVATION_CASES}) != 6:
    raise RuntimeError("E004 fused NVTX ranges must be unique")
