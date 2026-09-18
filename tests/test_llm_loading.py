"""Tests for the LLM dataset loading (`faultforge._internal.loading.wikitext`)."""

from types import SimpleNamespace
from typing import cast, override
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerBase

from faultforge._internal.loading import wikitext

_PAD_ID = 0


class _FakeTokenizer:
    """Turns a text of `n` characters into the token ids `1..n`.

    Ids start at 1 so they never collide with the pad id. The last text it
    received is kept in `text`.
    """

    pad_token = "<pad>"
    pad_token_id = _PAD_ID

    def __init__(self) -> None:
        self.text = ""

    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        self.text = text
        ids = torch.arange(1, len(text) + 1).unsqueeze(0)
        return SimpleNamespace(input_ids=ids)


def _make_formatter(
    lines: list[str], tokenizer: _FakeTokenizer | None = None
) -> wikitext._WikiTextFormatter:
    """Build the formatter on `lines` instead of downloading WikiText.

    Uses windows of 4 tokens moving 2 at a time.
    """
    if tokenizer is None:
        tokenizer = _FakeTokenizer()
    with patch.object(wikitext, "load_dataset", return_value={"text": lines}):
        return wikitext._WikiTextFormatter(
            cast(PreTrainedTokenizerBase, tokenizer),
            sequence_length=4,
            config="config",
            split="split",
            stride=2,
        )


def test_fetches_wikitext_with_config_and_split() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, _FakeTokenizer())

    with patch.object(
        wikitext, "load_dataset", return_value={"text": ["abc"]}
    ) as load_dataset:
        wikitext._WikiTextFormatter(tokenizer, 4, "config", "split", stride=2)

    load_dataset.assert_called_once_with("Salesforce/wikitext", "config", split="split")


def test_stride_larger_than_sequence_length_is_rejected() -> None:
    # Tokens between the windows would never be scored.
    tokenizer = cast(PreTrainedTokenizerBase, _FakeTokenizer())

    with patch.object(wikitext, "load_dataset", return_value={"text": ["abc"]}):
        try:
            wikitext._WikiTextFormatter(tokenizer, 4, "config", "split", stride=5)
        except ValueError:
            pass
        else:
            assert False, "expected a ValueError"


def test_text_is_joined_like_huggingface() -> None:
    # HuggingFace's perplexity recipe joins with "\n\n" and keeps empty lines.
    tokenizer = _FakeTokenizer()

    _make_formatter(["first", "", "second"], tokenizer)

    assert tokenizer.text == "first\n\n\n\nsecond"


def test_inputs_are_overlapping_windows_padded_at_the_end() -> None:
    formatter = _make_formatter(["a" * 7])

    assert formatter._inputs.tolist() == [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [5, 6, 7, _PAD_ID],
    ]


def test_inputs_have_sequence_length_for_any_text_length() -> None:
    short = _make_formatter(["a" * 3])
    long = _make_formatter(["a" * 9])

    assert short._inputs.shape == (1, 4)
    assert long._inputs.shape == (4, 4)


def test_first_window_starts_at_the_first_token() -> None:
    formatter = _make_formatter(["a" * 7])

    assert formatter._inputs[0, 0] == 1


def test_windows_move_by_stride() -> None:
    formatter = _make_formatter(["a" * 7])

    assert formatter._inputs[:, 0].tolist() == [1, 3, 5]


def test_consecutive_windows_overlap() -> None:
    formatter = _make_formatter(["a" * 7])

    first, second = formatter._inputs[0], formatter._inputs[1]

    assert first[2:].tolist() == second[:2].tolist()


def test_last_token_is_in_the_last_window() -> None:
    formatter = _make_formatter(["a" * 7])

    assert 7 in formatter._inputs[-1]


def test_padding_is_only_at_the_end() -> None:
    formatter = _make_formatter(["a" * 7])

    assert formatter._inputs[-1, -1] == _PAD_ID
    assert _PAD_ID not in formatter._inputs[:-1]


def test_text_of_exactly_one_window_is_not_padded() -> None:
    formatter = _make_formatter(["a" * 4])

    assert formatter._inputs.tolist() == [[1, 2, 3, 4]]


def test_targets_are_inputs_shifted_by_one() -> None:
    formatter = _make_formatter(["a" * 7])

    inputs, targets = formatter[0]

    assert targets[:-1].tolist() == inputs[1:].tolist()


def test_targets_mask_overlap_last_position_and_padding() -> None:
    # -100 marks positions that aren't scored: the overlap already scored by
    # the previous window, the last position (scored by the next window) and
    # the padding.
    formatter = _make_formatter(["a" * 7])

    assert formatter._targets.tolist() == [
        [2, 3, 4, -100],
        [-100, 5, 6, -100],
        [-100, 7, -100, -100],
    ]


def test_first_token_is_never_a_target() -> None:
    # Nothing comes before the first token, so it can't be predicted.
    formatter = _make_formatter(["a" * 7])

    assert 1 not in formatter._targets


def test_last_position_of_every_window_is_masked() -> None:
    # Its next token is outside the window.
    formatter = _make_formatter(["a" * 7])

    assert formatter._targets[:, -1].tolist() == [-100, -100, -100]


def test_overlap_is_masked_in_later_windows() -> None:
    # The overlap was already scored by the previous window.
    formatter = _make_formatter(["a" * 7])

    assert formatter._targets[1:, 0].tolist() == [-100, -100]


def test_first_window_has_no_overlap_mask() -> None:
    formatter = _make_formatter(["a" * 7])

    assert formatter._targets[0, :-1].tolist() == [2, 3, 4]


def test_every_token_is_scored_exactly_once() -> None:
    # Scoring a token twice or never would bias the perplexity.
    formatter = _make_formatter(["a" * 7])

    scored = formatter._targets[formatter._targets != -100]

    assert sorted(scored.tolist()) == [2, 3, 4, 5, 6, 7]


def test_sequence_length_is_read_from_model_config() -> None:
    config = SimpleNamespace(max_position_embeddings=1024)

    with patch.object(wikitext.AutoConfig, "from_pretrained", return_value=config):
        assert wikitext._model_max_seq_len("model") == 1024


def test_sequence_length_falls_back_to_512() -> None:
    config = SimpleNamespace()

    with patch.object(wikitext.AutoConfig, "from_pretrained", return_value=config):
        assert wikitext._model_max_seq_len("model") == 512


def test_explicit_sequence_length_wins_over_model_config() -> None:
    config = SimpleNamespace(max_position_embeddings=1024)
    bundle = wikitext.WikiTextBundle(model_id="model", sequence_length=8)

    with patch.object(wikitext.AutoConfig, "from_pretrained", return_value=config):
        assert bundle._resolved_seq_len() == 8


def test_stride_defaults_to_half_the_sequence_length() -> None:
    bundle = wikitext.WikiTextBundle(model_id="model", sequence_length=8)

    assert bundle._resolved_stride() == 4


def test_missing_pad_token_is_set_to_eos() -> None:
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
    bundle = wikitext.WikiTextBundle(model_id="model")

    with patch.object(
        wikitext.AutoTokenizer, "from_pretrained", return_value=tokenizer
    ):
        assert bundle._cached_tokenizer().pad_token == "<eos>"


def test_fingerprint_is_filled_in() -> None:
    bundle = wikitext.WikiTextBundle(model_id="model", sequence_length=8)

    fingerprint = bundle.fingerprint()

    assert fingerprint.kind == "huggingface_causal_lm"
    assert fingerprint.scalars["model"] == "model"


def test_wrapper_returns_only_the_logits() -> None:
    logits = torch.zeros(1, 3, 5)

    class FakeModel(torch.nn.Module):
        @override
        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=logits, other_metadata="ignored")

    wrapper = wikitext._HuggingFaceLMWrapper(FakeModel())

    assert wrapper(torch.tensor([[1, 2, 3]])) is logits
