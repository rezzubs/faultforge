"""Tests for `OnArray`."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from systolic import (
    Array,
    ArrayConfig,
    Fault,
    Index2,
    LiftedEvaluator,
    OnArray,
    PeRegisterKind,
    RegisterFaults,
    SimulatedEvaluator,
    StuckAtKind,
)
from torch import Tensor, nn

_ARRAY = Array(2, 3)
_ARRAY_CONFIG = ArrayConfig(nrows=2, ncols=3, dtype_bits=32)
_FAULTS = st.integers(
    min_value=0, max_value=RegisterFaults().radix(_ARRAY_CONFIG) - 1
).map(lambda fault_id: RegisterFaults().fault_from_id(fault_id, _ARRAY_CONFIG))

# Bit 30 is the top bit of the exponent, so the fault changes any value it
# touches by a lot.
_WEIGHT_FAULT = Fault.Register(
    target=Index2(x=0, y=0),
    register=PeRegisterKind.Weight,
    stuck_at=StuckAtKind.One,
    bit_index=30,
)


def _integers(*shape: int, seed: int = 0) -> Tensor:
    """Small non-negative integers as float32.

    Their sums are exact, so results don't depend on the summation order.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 5, shape, generator=generator).float()


def _integer_linear(in_features: int, out_features: int, *, bias: bool) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        linear.weight.copy_(_integers(out_features, in_features, seed=in_features))
        if linear.bias is not None:
            linear.bias.copy_(_integers(out_features, seed=out_features))
    return linear


def _mlp() -> nn.Module:
    """Uses both `addmm` and `mm`."""
    return nn.Sequential(
        _integer_linear(5, 7, bias=True),
        nn.ReLU(),
        _integer_linear(7, 4, bias=False),
    )


def _set_precision(
    *, everywhere: str = "none", cuda: str = "none", onednn: str = "none"
) -> None:
    # Torch adds some of these attributes at runtime, so ty can't see them.
    torch.backends.fp32_precision = everywhere  # ty: ignore[unresolved-attribute]
    torch.backends.cuda.matmul.fp32_precision = cuda
    torch.backends.mkldnn.matmul.fp32_precision = onednn  # ty: ignore[unresolved-attribute]


@pytest.fixture
def restore_precision() -> Iterator[None]:
    yield
    _set_precision()


@pytest.mark.parametrize("input_shape", [(3, 5), (2, 3, 5)])
def test_without_a_fault_matches_torch(input_shape: tuple[int, ...]) -> None:
    model = _mlp()
    inputs = _integers(*input_shape)

    with torch.no_grad():
        expected = model(inputs)
        with OnArray(LiftedEvaluator(_ARRAY), None):
            output = model(inputs)

    assert torch.equal(output, expected)
    assert output.is_contiguous()


def test_a_fault_changes_the_output() -> None:
    model = _mlp()
    inputs = _integers(3, 5)

    with torch.no_grad():
        clean = model(inputs)
        with OnArray(LiftedEvaluator(_ARRAY), _WEIGHT_FAULT):
            faulty = model(inputs)

    assert not torch.equal(faulty, clean)


@pytest.mark.parametrize(
    "evaluator",
    [LiftedEvaluator(Array(3, 3)), SimulatedEvaluator(Array(3, 3))],
    ids=["lifted", "simulated"],
)
def test_weights_are_stationary(
    evaluator: LiftedEvaluator | SimulatedEvaluator,
) -> None:
    # The weights fit in the array, so the fault hits exactly one of them.
    linear = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    inputs = _integers(4, 3) + 1
    # Clearing the lowest bit of the exponent halves 1.0.
    fault = Fault.Register(
        target=Index2(x=1, y=2),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.Zero,
        bit_index=23,
    )

    with torch.no_grad():
        clean = linear(inputs)
        with OnArray(evaluator, fault):
            faulty = linear(inputs)

    # A faulty weight changes one output feature, for every input. If the
    # inputs were stationary instead, one input would change instead.
    changed = faulty != clean
    changed_features = changed.any(dim=0).nonzero().flatten().tolist()
    assert len(changed_features) == 1
    assert changed[:, changed_features[0]].all()


@settings(deadline=None)
@given(fault=_FAULTS)
def test_lifted_matches_simulated(fault: Fault) -> None:
    model = _mlp()
    inputs = _integers(3, 5)

    with torch.no_grad():
        with OnArray(LiftedEvaluator(_ARRAY), fault):
            lifted = model(inputs)
        with OnArray(SimulatedEvaluator(_ARRAY), fault):
            simulated = model(inputs)

    torch.testing.assert_close(lifted, simulated, rtol=1e-5, atol=1e-3, equal_nan=True)


def _batched(left: Tensor, right: Tensor, bias: Tensor) -> Tensor:
    """Uses `bmm` through a 4D `matmul` and `baddbmm`."""
    product = left.unsqueeze(0).expand(2, -1, -1, -1) @ right
    return torch.baddbmm(bias, left, right) + product.sum(dim=0)


def _batched_operands() -> tuple[Tensor, Tensor, Tensor]:
    return _integers(3, 4, 5), _integers(3, 5, 2, seed=1), _integers(3, 4, 2, seed=2)


def test_batched_without_a_fault_matches_torch() -> None:
    operands = _batched_operands()

    with torch.no_grad():
        expected = _batched(*operands)
        with OnArray(LiftedEvaluator(_ARRAY), None):
            output = _batched(*operands)

    assert torch.equal(output, expected)
    assert output.is_contiguous()


@settings(deadline=None)
@given(fault=_FAULTS)
def test_batched_lifted_matches_simulated(fault: Fault) -> None:
    operands = _batched_operands()

    with torch.no_grad():
        with OnArray(LiftedEvaluator(_ARRAY), fault):
            lifted = _batched(*operands)
        with OnArray(SimulatedEvaluator(_ARRAY), fault):
            simulated = _batched(*operands)

    torch.testing.assert_close(lifted, simulated, rtol=1e-5, atol=1e-3, equal_nan=True)


@given(fault=_FAULTS)
def test_batch_matches_individual_matmuls(fault: Fault) -> None:
    left, right, _ = _batched_operands()

    with torch.no_grad(), OnArray(LiftedEvaluator(_ARRAY), fault):
        batched = torch.bmm(left, right)
        individual = torch.stack(
            [
                torch.mm(single_left, single_right)
                for single_left, single_right in zip(left, right, strict=True)
            ]
        )

    torch.testing.assert_close(
        batched, individual, rtol=1e-5, atol=1e-3, equal_nan=True
    )


def _integer_conv2d(
    in_channels: int, out_channels: int, kernel_size: int, **options: Any
) -> nn.Conv2d:
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, **options)
    with torch.no_grad():
        conv2d.weight.copy_(_integers(*conv2d.weight.shape, seed=out_channels))
        if conv2d.bias is not None:
            conv2d.bias.copy_(_integers(out_channels, seed=in_channels))
    return conv2d


def _cnn() -> nn.Module:
    """Uses `convolution` and `addmm`."""
    return nn.Sequential(
        _integer_conv2d(2, 3, 3, padding=1),
        nn.ReLU(),
        _integer_conv2d(3, 2, 2, stride=2, bias=False),
        nn.Flatten(),
        _integer_linear(18, 4, bias=True),
    )


def test_cnn_without_a_fault_matches_torch() -> None:
    model = _cnn()
    inputs = _integers(2, 2, 6, 6)

    with torch.no_grad():
        expected = model(inputs)
        with OnArray(LiftedEvaluator(_ARRAY), None):
            output = model(inputs)

    assert torch.equal(output, expected)


@settings(deadline=None, max_examples=50)
@given(
    in_channels=st.integers(min_value=1, max_value=3),
    out_channels=st.integers(min_value=1, max_value=3),
    kernel_size=st.integers(min_value=1, max_value=3),
    stride=st.integers(min_value=1, max_value=2),
    padding=st.integers(min_value=0, max_value=2),
    dilation=st.integers(min_value=1, max_value=2),
    batch_size=st.integers(min_value=1, max_value=3),
    input_size=st.integers(min_value=5, max_value=8),
)
def test_convolution_without_a_fault_matches_torch(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    batch_size: int,
    input_size: int,
) -> None:
    conv2d = _integer_conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    inputs = _integers(batch_size, in_channels, input_size, input_size)

    with torch.no_grad():
        expected = conv2d(inputs)
        with OnArray(LiftedEvaluator(_ARRAY), None):
            output = conv2d(inputs)

    assert torch.equal(output, expected)
    assert output.is_contiguous()


@settings(deadline=None)
@given(fault=_FAULTS)
def test_cnn_lifted_matches_simulated(fault: Fault) -> None:
    model = _cnn()
    inputs = _integers(2, 2, 6, 6)

    with torch.no_grad():
        with OnArray(LiftedEvaluator(_ARRAY), fault):
            lifted = model(inputs)
        with OnArray(SimulatedEvaluator(_ARRAY), fault):
            simulated = model(inputs)

    torch.testing.assert_close(lifted, simulated, rtol=1e-5, atol=1e-3, equal_nan=True)


def test_bias_is_added_after_the_array() -> None:
    inputs = _integers(3, 5)
    weights = _integers(5, 4, seed=1)
    bias = _integers(4, seed=2)

    with torch.no_grad(), OnArray(LiftedEvaluator(_ARRAY), _WEIGHT_FAULT):
        with_bias = torch.addmm(bias, inputs, weights)
        without_bias = inputs @ weights

    # The fault makes some of the values infinite or NaN.
    torch.testing.assert_close(
        with_bias, without_bias + bias, rtol=0, atol=0, equal_nan=True
    )


def test_batched_bias_is_added_after_the_array() -> None:
    left, right, bias = _batched_operands()

    with torch.no_grad(), OnArray(LiftedEvaluator(_ARRAY), _WEIGHT_FAULT):
        with_bias = torch.baddbmm(bias, left, right)
        without_bias = torch.bmm(left, right)

    # The fault makes some of the values infinite or NaN.
    torch.testing.assert_close(
        with_bias, without_bias + bias, rtol=0, atol=0, equal_nan=True
    )


def test_convolution_bias_is_added_after_the_array() -> None:
    inputs = _integers(2, 2, 5, 5)
    weight = _integers(3, 2, 3, 3, seed=1)
    bias = _integers(3, seed=2)

    with torch.no_grad(), OnArray(LiftedEvaluator(_ARRAY), _WEIGHT_FAULT):
        with_bias = F.conv2d(inputs, weight, bias)
        without_bias = F.conv2d(inputs, weight)

    # The fault makes some of the values infinite or NaN.
    torch.testing.assert_close(
        with_bias, without_bias + bias.view(1, -1, 1, 1), rtol=0, atol=0, equal_nan=True
    )


def test_other_operations_run_as_usual() -> None:
    values = torch.tensor([-1.0, 2.0])

    with torch.no_grad(), OnArray(LiftedEvaluator(_ARRAY), _WEIGHT_FAULT):
        output = F.relu(values) + 1

    assert torch.equal(output, torch.tensor([1.0, 3.0]))


def test_rejects_a_fault_outside_the_array() -> None:
    fault = Fault.Register(
        target=Index2(x=3, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=0,
    )
    with pytest.raises(ValueError, match="outside"):
        OnArray(LiftedEvaluator(_ARRAY), fault)


@pytest.mark.parametrize(
    "unsupported",
    [
        lambda: torch.mv(torch.ones(2, 3), torch.ones(3)),
        lambda: torch.dot(torch.ones(3), torch.ones(3)),
        lambda: torch.addmm(torch.ones(2), torch.ones(2, 3), torch.ones(3, 2), beta=2),
        lambda: torch.mm(torch.ones(2, 3), torch.ones(3, 2), out=torch.empty(2, 2)),
        lambda: torch.baddbmm(
            torch.ones(1, 2, 2), torch.ones(1, 2, 3), torch.ones(1, 3, 2), alpha=2
        ),
        lambda: torch.bmm(
            torch.ones(1, 2, 3), torch.ones(1, 3, 2), out=torch.empty(1, 2, 2)
        ),
        lambda: F.scaled_dot_product_attention(
            torch.ones(1, 2, 3), torch.ones(1, 2, 3), torch.ones(1, 2, 3)
        ),
        lambda: F.conv2d(torch.ones(1, 2, 3, 3), torch.ones(2, 1, 2, 2), groups=2),
        lambda: F.conv_transpose2d(torch.ones(1, 1, 3, 3), torch.ones(1, 1, 2, 2)),
        lambda: F.conv1d(torch.ones(1, 1, 3), torch.ones(1, 1, 2)),
        lambda: F.conv3d(torch.ones(1, 1, 3, 3, 3), torch.ones(1, 1, 2, 2, 2)),
    ],
    ids=[
        "mv",
        "dot",
        "scaled addmm",
        "mm out",
        "scaled baddbmm",
        "bmm out",
        "attention",
        "grouped conv2d",
        "conv_transpose2d",
        "conv1d",
        "conv3d",
    ],
)
def test_rejects_unsupported_matmuls(unsupported: Callable[[], Tensor]) -> None:
    with (
        torch.no_grad(),
        OnArray(LiftedEvaluator(_ARRAY), None),
        pytest.raises(NotImplementedError),
    ):
        unsupported()


def test_rejects_inference_mode() -> None:
    with (
        torch.inference_mode(),
        OnArray(LiftedEvaluator(_ARRAY), None),
        pytest.raises(NotImplementedError, match="no_grad"),
    ):
        _mlp()(_integers(3, 5))


def test_rejects_gradients() -> None:
    with (
        OnArray(LiftedEvaluator(_ARRAY), None),
        pytest.raises(RuntimeError, match="no_grad"),
    ):
        _mlp()(_integers(3, 5))


def test_rejects_float64() -> None:
    with (
        torch.no_grad(),
        OnArray(LiftedEvaluator(_ARRAY), None),
        pytest.raises(ValueError, match="float32"),
    ):
        torch.ones(2, 3, dtype=torch.float64) @ torch.ones(3, 2, dtype=torch.float64)


@pytest.mark.usefixtures("restore_precision")
@pytest.mark.parametrize(
    "precision",
    [{"everywhere": "tf32"}, {"cuda": "tf32"}, {"onednn": "bf16"}],
    ids=["everywhere", "cuda", "onednn"],
)
def test_rejects_reduced_precision(precision: dict[str, str]) -> None:
    _set_precision(**precision)
    with (
        torch.no_grad(),
        OnArray(LiftedEvaluator(_ARRAY), None),
        pytest.raises(RuntimeError, match="precision"),
    ):
        torch.ones(2, 3) @ torch.ones(3, 2)


@pytest.mark.usefixtures("restore_precision")
def test_full_precision_is_accepted() -> None:
    _set_precision(everywhere="tf32", cuda="ieee", onednn="ieee")
    with torch.no_grad(), OnArray(LiftedEvaluator(_ARRAY), None):
        output = torch.ones(2, 3) @ torch.ones(3, 2)

    assert torch.equal(output, torch.full((2, 2), 3.0))
