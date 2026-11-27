"""Tests for `SystolicFaultInjection` on a small language model."""

import math
from typing import override

import torch
from systolic import (
    PeRegisterKind,
    RegisterFaults,
    SystolicFaultInjection,
)
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedModel

from faultforge import Fingerprint
from faultforge.dataset import BatchedDataset, DeviceLike
from faultforge.loading import ModelBundle
from faultforge.metric import Perplexity
from faultforge.progress import Progress

_VOCABULARY_SIZE = 50
_SEQUENCE_LENGTH = 8
_SEQUENCES = 4


class _Logits(nn.Module):
    """Returns only the logits of a HuggingFace causal language model."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        return self.model(input_ids).logits


class _TinyGpt2Bundle(ModelBundle):
    """A GPT2 with random weights and random tokens, small enough to simulate."""

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        progress: Progress | None = None,
    ) -> nn.Module:
        _ = progress
        torch.manual_seed(0)
        config = GPT2Config(
            n_layer=2,
            n_embd=16,
            n_head=2,
            vocab_size=_VOCABULARY_SIZE,
            n_positions=_SEQUENCE_LENGTH,
            # The defaults are outside the small vocabulary.
            bos_token_id=0,
            eos_token_id=0,
        )
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager", dtype=torch.float32
        )
        return _Logits(model).eval().to(device=device)

    @override
    def load_dataset(
        self,
        batch_size: int,
        device: DeviceLike,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        progress: Progress | None = None,
    ) -> BatchedDataset:
        _ = progress
        generator = torch.Generator().manual_seed(1)
        tokens = torch.randint(
            0,
            _VOCABULARY_SIZE,
            (_SEQUENCES, _SEQUENCE_LENGTH + 1),
            generator=generator,
        )
        # Each position predicts the token after it.
        return BatchedDataset.from_dataset(
            TensorDataset(tokens[:, :-1], tokens[:, 1:]),
            batch_size,
            device,
            shuffle=shuffle,
            seed=seed,
        )

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="tiny_gpt2_bundle")


def _make_experiment() -> SystolicFaultInjection:
    return SystolicFaultInjection(
        _TinyGpt2Bundle(),
        (4, 4),
        Perplexity(),
        fault=RegisterFaults(registers=frozenset({PeRegisterKind.Weight})),
        batch_size=2,
    )


def test_campaign_round_trips() -> None:
    experiment = _make_experiment()
    golden_score = experiment.golden_score()
    experiment.run()
    experiment.run()

    reloaded = _make_experiment()
    reloaded.deserialize(experiment.serialize())

    # Random weights predict about as well as a uniform guess.
    assert 1.0 < golden_score < 10 * _VOCABULARY_SIZE
    assert reloaded.golden_score() == golden_score
    assert len(reloaded.scores()) == 2
    for reloaded_score, score in zip(
        reloaded.scores(), experiment.scores(), strict=True
    ):
        assert reloaded_score == score or (
            math.isnan(reloaded_score) and math.isnan(score)
        )
