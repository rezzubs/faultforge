"""Tests for `Array` and `ArrayMatmul`."""

import pytest
import torch
from systolic import (
    Array,
    ArrayMatmul,
    Fault,
    Index2,
    LiftedFault,
    PeRegisterKind,
    StuckAtKind,
)


def _matmul(out_features: int, in_features: int, batch: int = 2) -> ArrayMatmul:
    return ArrayMatmul(
        torch.randn(out_features, in_features), torch.randn(in_features, batch)
    )


def _fault(x: int = 0, y: int = 0, bit_index: int = 0) -> Fault:
    return Fault.Register(
        target=Index2(x=x, y=y),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=bit_index,
    )


@pytest.mark.parametrize(
    ("weights", "activations", "message"),
    [
        (torch.ones(2, 3).double(), torch.ones(3, 4), "weights must be float32"),
        (torch.ones(2, 3), torch.ones(3, 4).half(), "activations must be float32"),
        (torch.ones(3), torch.ones(3, 4), "weights must have 2 or 3 dim"),
        (torch.ones(2, 3), torch.ones(3), "activations must have 2 or 3 dim"),
        (torch.ones(0, 3), torch.ones(3, 4), "weights must not be empty"),
        (torch.ones(2, 3), torch.ones(3, 0), "activations must not be empty"),
        (torch.ones(2, 3), torch.ones(4, 3), "cannot multiply"),
        (torch.ones(5, 2, 3), torch.ones(3, 4), "cannot multiply"),
        (torch.ones(5, 2, 3), torch.ones(6, 3, 4), "cannot multiply"),
        (torch.ones(1, 2, 3), torch.ones(6, 3, 4), "cannot multiply"),
        (torch.ones(2, 3, device="meta"), torch.ones(3, 4), "same device"),
    ],
)
def test_matmul_rejects_invalid_operands(
    weights: torch.Tensor, activations: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ArrayMatmul(weights, activations)


def test_matmul_features() -> None:
    matmul = ArrayMatmul(torch.ones(5, 2, 3), torch.ones(5, 3, 4))
    assert matmul.out_features == 2
    assert matmul.in_features == 3


def test_unstack_of_a_single_matmul_is_itself() -> None:
    matmul = _matmul(2, 3)
    assert list(matmul.unstack()) == [matmul]


def test_unstack_follows_the_stack_order() -> None:
    weights = torch.randn(5, 2, 3)
    activations = torch.randn(5, 3, 4)
    matmuls = list(ArrayMatmul(weights, activations).unstack())

    assert len(matmuls) == 5
    for index, matmul in enumerate(matmuls):
        assert torch.equal(matmul.weights, weights[index])
        assert torch.equal(matmul.activations, activations[index])


@pytest.mark.parametrize(("nrows", "ncols"), [(0, 1), (1, 0), (-1, 1)])
def test_array_rejects_invalid_size(nrows: int, ncols: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Array(nrows, ncols)


def test_mapping_is_reused_per_shape() -> None:
    array = Array(4, 4)
    mapping = array.mapping(_matmul(3, 4))

    assert array.mapping(_matmul(3, 4, batch=7)) is mapping
    assert array.mapping(_matmul(4, 3)) is not mapping


def test_mapping_ignores_the_stack() -> None:
    array = Array(4, 4)
    stacked = ArrayMatmul(torch.ones(5, 3, 4), torch.ones(5, 4, 2))
    assert array.mapping(stacked) is array.mapping(_matmul(3, 4))


def test_lift_is_reused_per_shape() -> None:
    array = Array(4, 4)
    lifted = array.lift(_matmul(3, 4), _fault())

    assert array.lift(_matmul(3, 4), _fault()) is lifted
    assert array.lift(_matmul(4, 3), _fault()) is not lifted


def test_lift_follows_the_fault() -> None:
    array = Array(4, 4)
    matmul = _matmul(3, 4)

    for x in (0, 1, 0):
        lifted = array.lift(matmul, _fault(x=x, y=3))
        assert isinstance(lifted, LiftedFault.Weight)
        [part] = lifted.parts
        assert part.affected_output_row == x
        assert part.for_activations == (3, 4)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (_fault(x=3), "outside the 2x3 array"),
        (_fault(y=2), "outside the 2x3 array"),
        (_fault(bit_index=32), "32-bit register"),
    ],
)
def test_lift_rejects_faults_outside_the_array(fault: Fault, message: str) -> None:
    array = Array(2, 3)
    # Rejected every time, not only on the first use of the fault.
    for _ in range(2):
        with pytest.raises(ValueError, match=message):
            array.lift(_matmul(3, 4), fault)
