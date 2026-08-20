from hypothesis import given, settings
from hypothesis import strategies as st

from nvfp4_doctor.formats import (
    ScaleFactorLayout,
    cutlass_128x4_offset,
    pack_e2m1,
    scale_storage_size,
    swizzle_scales_128x4,
    unpack_e2m1,
    unswizzle_scales_128x4,
)


@st.composite
def even_e2m1_codes(draw: st.DrawFn) -> tuple[int, ...]:
    pairs = draw(st.integers(min_value=0, max_value=128))
    return tuple(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=15),
                min_size=pairs * 2,
                max_size=pairs * 2,
            )
        )
    )


@st.composite
def scale_matrices(draw: st.DrawFn) -> tuple[int, int, bytes]:
    rows = draw(st.integers(min_value=1, max_value=260))
    columns = draw(st.integers(min_value=1, max_value=12))
    payload = draw(st.binary(min_size=rows * columns, max_size=rows * columns))
    return rows, columns, payload


@given(even_e2m1_codes())
def test_e2m1_packing_round_trips(codes: tuple[int, ...]) -> None:
    assert unpack_e2m1(pack_e2m1(codes)) == codes


@settings(max_examples=50, deadline=None)
@given(scale_matrices())
def test_cutlass_scale_layout_round_trips(
    case: tuple[int, int, bytes],
) -> None:
    rows, columns, linear = case
    swizzled = swizzle_scales_128x4(linear, rows, columns)

    assert len(swizzled) == scale_storage_size(
        rows, columns, ScaleFactorLayout.CUTLASS_128X4
    )
    assert unswizzle_scales_128x4(swizzled, rows, columns) == linear


@settings(max_examples=50, deadline=None)
@given(
    rows=st.integers(min_value=1, max_value=300),
    columns=st.integers(min_value=1, max_value=16),
)
def test_cutlass_scale_offsets_are_unique_and_in_bounds(
    rows: int, columns: int
) -> None:
    offsets = {
        cutlass_128x4_offset(row, column, rows, columns)
        for row in range(rows)
        for column in range(columns)
    }
    assert len(offsets) == rows * columns
    assert max(offsets) < scale_storage_size(
        rows, columns, ScaleFactorLayout.CUTLASS_128X4
    )
