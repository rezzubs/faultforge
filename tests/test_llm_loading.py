"""Tests for the LLM dataset loading (`faultforge._internal.loading.wikitext`)."""

from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock, patch

import torch
from transformers import PreTrainedTokenizerBase

from faultforge._internal.loading import wikitext

_PAD_ID = 0


def _make_formatter(num_tokens: int) -> wikitext.SlidingWindowDataset:
    """Build the formatter on the token ids `1..num_tokens`.

    Ids start at 1 so they never collide with the pad id. Uses windows of 4
    tokens moving 2 at a time.
    """
    return wikitext.SlidingWindowDataset(
        torch.arange(1, num_tokens + 1),
        _PAD_ID,
        sequence_length=4,
        stride=2,
    )


def test_fetches_wikitext_with_config_and_split() -> None:
    with patch.object(
        wikitext.datasets, "load_dataset", return_value={"text": ["abc"]}
    ) as load_dataset:
        wikitext.load_wikitext("config", wikitext.WikiTextSplit.Validation)

    load_dataset.assert_called_once_with(
        "Salesforce/wikitext", "config", split="validation"
    )


def test_stride_larger_than_sequence_length_is_rejected() -> None:
    # Tokens between the windows would never be scored.
    try:
        wikitext.SlidingWindowDataset(torch.arange(1, 4), _PAD_ID, 4, stride=5)
    except ValueError:
        pass
    else:
        assert False, "expected a ValueError"


def test_text_is_joined_like_huggingface() -> None:
    # HuggingFace's perplexity recipe joins with "\n\n" and keeps empty lines.
    with patch.object(
        wikitext.datasets,
        "load_dataset",
        return_value={"text": ["first", "", "second"]},
    ):
        text = wikitext.load_wikitext("config", wikitext.WikiTextSplit.Test)

    assert text == "first\n\n\n\nsecond"


def test_inputs_are_overlapping_windows_padded_at_the_end() -> None:
    formatter = _make_formatter(7)

    assert formatter._inputs.tolist() == [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [5, 6, 7, _PAD_ID],
    ]


def test_inputs_have_sequence_length_for_any_text_length() -> None:
    short = _make_formatter(3)
    long = _make_formatter(9)

    assert short._inputs.shape == (1, 4)
    assert long._inputs.shape == (4, 4)


def test_first_window_starts_at_the_first_token() -> None:
    formatter = _make_formatter(7)

    assert formatter._inputs[0, 0] == 1


def test_windows_move_by_stride() -> None:
    formatter = _make_formatter(7)

    assert formatter._inputs[:, 0].tolist() == [1, 3, 5]


def test_consecutive_windows_overlap() -> None:
    formatter = _make_formatter(7)

    first, second = formatter._inputs[0], formatter._inputs[1]

    assert first[2:].tolist() == second[:2].tolist()


def test_last_token_is_in_the_last_window() -> None:
    formatter = _make_formatter(7)

    assert 7 in formatter._inputs[-1]


def test_padding_is_only_at_the_end() -> None:
    formatter = _make_formatter(7)

    assert formatter._inputs[-1, -1] == _PAD_ID
    assert _PAD_ID not in formatter._inputs[:-1]


def test_text_of_exactly_one_window_is_not_padded() -> None:
    formatter = _make_formatter(4)

    assert formatter._inputs.tolist() == [[1, 2, 3, 4]]


def test_targets_are_inputs_shifted_by_one() -> None:
    formatter = _make_formatter(7)

    inputs, targets = formatter[0]

    assert targets[:-1].tolist() == inputs[1:].tolist()


def test_targets_mask_overlap_last_position_and_padding() -> None:
    # -100 marks positions that aren't scored: the overlap already scored by
    # the previous window, the last position (scored by the next window) and
    # the padding.
    formatter = _make_formatter(7)

    assert formatter._targets.tolist() == [
        [2, 3, 4, -100],
        [-100, 5, 6, -100],
        [-100, 7, -100, -100],
    ]


def test_first_token_is_never_a_target() -> None:
    # Nothing comes before the first token, so it can't be predicted.
    formatter = _make_formatter(7)

    assert 1 not in formatter._targets


def test_last_position_of_every_window_is_masked() -> None:
    # Its next token is outside the window.
    formatter = _make_formatter(7)

    assert formatter._targets[:, -1].tolist() == [-100, -100, -100]


def test_overlap_is_masked_in_later_windows() -> None:
    # The overlap was already scored by the previous window.
    formatter = _make_formatter(7)

    assert formatter._targets[1:, 0].tolist() == [-100, -100]


def test_first_window_has_no_overlap_mask() -> None:
    formatter = _make_formatter(7)

    assert formatter._targets[0, :-1].tolist() == [2, 3, 4]


def test_every_token_is_scored_exactly_once() -> None:
    # Scoring a token twice or never would bias the perplexity.
    formatter = _make_formatter(7)

    scored = formatter._targets[formatter._targets != -100]

    assert sorted(scored.tolist()) == [2, 3, 4, 5, 6, 7]


def test_sequence_length_is_read_from_model_config() -> None:
    config = SimpleNamespace(max_position_embeddings=1024)

    with patch.object(wikitext.AutoConfig, "from_pretrained", return_value=config):
        assert wikitext.model_max_seq_len("model") == 1024


def test_sequence_length_falls_back_to_512() -> None:
    config = SimpleNamespace()

    with patch.object(wikitext.AutoConfig, "from_pretrained", return_value=config):
        assert wikitext.model_max_seq_len("model") == 512


def _make_bundle(sequence_length: int | None = None) -> wikitext.WikiTextBundle:
    """Build the bundle without touching the network.

    The model config reports a max sequence length of 1024.
    """
    config = SimpleNamespace(max_position_embeddings=1024)
    with (
        patch.object(wikitext.AutoConfig, "from_pretrained", return_value=config),
        patch.object(wikitext, "CausalTokenizer"),
    ):
        return wikitext.WikiTextBundle(
            model_id="model", sequence_length=sequence_length
        )


def test_explicit_sequence_length_wins_over_model_config() -> None:
    bundle = _make_bundle(sequence_length=8)

    assert bundle._sequence_length == 8


def test_stride_defaults_to_half_the_sequence_length() -> None:
    bundle = _make_bundle(sequence_length=8)

    assert bundle._stride == 4


def test_missing_pad_token_falls_back_to_eos() -> None:
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = None

    with patch.object(
        wikitext.AutoTokenizer, "from_pretrained", return_value=tokenizer
    ):
        causal = wikitext.CausalTokenizer("model")

    assert causal.padding_token_id == 2


def test_fingerprint_is_filled_in() -> None:
    bundle = _make_bundle(sequence_length=8)

    fingerprint = bundle.fingerprint()

    assert fingerprint.kind == "wikitext"
    assert fingerprint.scalars["model"] == "model"
    assert fingerprint.scalars["split"] == "test"
    assert fingerprint.scalars["sequence_len"] == 8


def test_wrapper_returns_only_the_logits() -> None:
    logits = torch.zeros(1, 3, 5)

    class FakeModel(torch.nn.Module):
        @override
        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=logits, other_metadata="ignored")

    wrapper = wikitext.HuggingFaceLMWrapper(FakeModel())

    assert wrapper(torch.tensor([[1, 2, 3]])) is logits
