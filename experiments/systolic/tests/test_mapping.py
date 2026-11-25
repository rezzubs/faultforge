"""Tests for hand-built `Mapping`/`Pass` construction, as an alternative to
`Mapping.auto_for`. Exercises the raw `systolic._rust` bindings directly,
since nothing above this layer builds a `Mapping` any other way yet.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from systolic._rust import (
    ArrayConfig,
    Fault,
    Index2,
    LiftedFault,
    Mapping,
    Pass,
    PeRegisterKind,
    StuckAtKind,
    fault_radix,
    simulated_matmul,
)


def test_pass_defaults_offsets_to_zero() -> None:
    p = Pass(activation_rows=(0, 2), output_rows=(0, 3))
    assert p.array_row_start == 0
    assert p.array_col_start == 0
    assert p.activation_rows == (0, 2)
    assert p.output_rows == (0, 3)


def test_mapping_new_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Mapping.new([])


def test_mapping_new_single_pass_matches_auto_for() -> None:
    """A hand-built single-pass mapping covering the whole array should be
    interchangeable with what `auto_for` derives for weights that fit in one
    pass: same matmul output, same fault lift."""
    rng = np.random.default_rng(0)
    out_features, in_features, batch = 3, 4, 2
    array_nrows, array_ncols = in_features, out_features

    weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
    activations = rng.standard_normal((in_features, batch)).astype(np.float32)

    auto = Mapping.auto_for(weights, array_nrows, array_ncols)
    custom = Mapping.new(
        [Pass(activation_rows=(0, in_features), output_rows=(0, out_features))]
    )
    custom.validate()

    expected = simulated_matmul(
        auto, weights, activations, array_nrows, array_ncols, None
    )
    actual = simulated_matmul(
        custom, weights, activations, array_nrows, array_ncols, None
    )
    np.testing.assert_array_equal(actual, expected)

    fault = Fault.Register(
        target=Index2(x=0, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=5,
    )
    assert type(custom.lift(fault)) is type(auto.lift(fault))


@given(
    in_features=st.integers(min_value=1, max_value=12),
    out_features=st.integers(min_value=1, max_value=12),
    array_nrows=st.integers(min_value=1, max_value=5),
    array_ncols=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
def test_mapping_auto_matches_auto_for(
    in_features: int,
    out_features: int,
    array_nrows: int,
    array_ncols: int,
    data: st.DataObject,
) -> None:
    from_shape = Mapping.auto(in_features, out_features, array_nrows, array_ncols)
    from_weights = Mapping.auto_for(
        np.zeros((out_features, in_features), dtype=np.float32),
        array_nrows,
        array_ncols,
    )
    from_shape.validate()

    # Passes can't be compared directly from Python, but two mappings which
    # lift every fault identically are the same for our purposes.
    array = ArrayConfig(array_nrows, array_ncols, 32)
    registers = {
        PeRegisterKind.Activation,
        PeRegisterKind.Weight,
        PeRegisterKind.Accumulator,
    }
    fault_id = data.draw(
        st.integers(min_value=0, max_value=fault_radix(array, registers) - 1)
    )
    fault = Fault.from_id(fault_id, array, registers)
    assert from_shape.lift(fault) == from_weights.lift(fault)


@pytest.mark.parametrize(
    "shape",
    [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)],
)
def test_mapping_auto_rejects_zero(shape: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="empty"):
        Mapping.auto(*shape)


def test_mapping_auto_for_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="empty"):
        Mapping.auto_for(np.zeros((0, 3), dtype=np.float32), 2, 2)


def test_mapping_validate_detects_missing_connections() -> None:
    """Two passes whose combined activation/output ranges imply a bigger
    universe than either pass alone connects, leaving the cross-connections
    between them unaccounted for."""
    mapping = Mapping.new(
        [
            Pass(activation_rows=(0, 2), output_rows=(0, 2)),
            Pass(activation_rows=(2, 4), output_rows=(2, 4)),
        ]
    )
    with pytest.raises(ValueError):
        mapping.validate()


def test_mapping_validate_detects_duplicate_connections() -> None:
    mapping = Mapping.new(
        [
            Pass(activation_rows=(0, 2), output_rows=(0, 2)),
            Pass(activation_rows=(0, 2), output_rows=(0, 2)),
        ]
    )
    with pytest.raises(ValueError):
        mapping.validate()


def test_weight_lift_has_a_part_per_pass() -> None:
    """Passes that only use rows above the fault have no part."""
    fault = Fault.Register(
        target=Index2(x=1, y=1),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=0,
    )
    lifted = Mapping.auto(7, 3, 3, 2).lift(fault)

    assert isinstance(lifted, LiftedFault.Weight)
    assert [
        (part.affected_output_row, part.for_activations) for part in lifted.parts
    ] == [
        (1, (1, 3)),
        (1, (4, 6)),
    ]
