"""Tests for `OnArray` on a small language model."""

import pytest
import torch
from systolic import (
    Array,
    Fault,
    Index2,
    LiftedEvaluator,
    OnArray,
    PeRegisterKind,
    SimulatedEvaluator,
    StuckAtKind,
)
from torch import Tensor
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedModel

_ARRAY = Array(4, 4)


def _gpt2(attention: str) -> PreTrainedModel:
    """A GPT2 with random weights, small enough to simulate."""
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_embd=16,
        n_head=2,
        vocab_size=50,
        n_positions=16,
        # The defaults are outside the small vocabulary.
        bos_token_id=0,
        eos_token_id=0,
    )
    return AutoModelForCausalLM.from_config(
        config, attn_implementation=attention, dtype=torch.float32
    ).eval()


def _tokens() -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 50, (2, 8), generator=generator)


def _logits(model: PreTrainedModel, tokens: Tensor) -> Tensor:
    return model(tokens).logits


def test_without_a_fault_matches_torch() -> None:
    model = _gpt2("eager")
    tokens = _tokens()

    with torch.no_grad():
        expected = _logits(model, tokens)
        with OnArray(LiftedEvaluator(_ARRAY), None):
            output = _logits(model, tokens)

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "register",
    [PeRegisterKind.Weight, PeRegisterKind.Activation, PeRegisterKind.Accumulator],
    ids=["weight", "activation", "accumulator"],
)
def test_lifted_matches_simulated(register: PeRegisterKind) -> None:
    model = _gpt2("eager")
    tokens = _tokens()
    # A low bit of the exponent changes values by a factor of two, enough to
    # see in the logits without turning them infinite.
    fault = Fault.Register(
        target=Index2(x=1, y=2),
        register=register,
        stuck_at=StuckAtKind.One,
        bit_index=23,
    )

    with torch.no_grad():
        clean = _logits(model, tokens)
        with OnArray(LiftedEvaluator(_ARRAY), fault):
            lifted = _logits(model, tokens)
        with OnArray(SimulatedEvaluator(_ARRAY), fault):
            simulated = _logits(model, tokens)

    assert not torch.allclose(lifted, clean)
    # The two sum in different orders, and the differences add up over the
    # layers.
    torch.testing.assert_close(lifted, simulated, rtol=1e-4, atol=1e-4)


def test_rejects_fused_attention() -> None:
    model = _gpt2("sdpa")

    with (
        torch.no_grad(),
        OnArray(LiftedEvaluator(_ARRAY), None),
        pytest.raises(NotImplementedError, match="attention"),
    ):
        _logits(model, _tokens())
