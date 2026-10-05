"""Parsing and loading Wikitext dataset with and HuggingFace's decoder models"""

import enum
from typing import Final, final, override

import datasets
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from faultforge._internal.dataset import BatchedDataset, DeviceLike
from faultforge._internal.dtype import dtype_name
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.loading.abc import ModelBundle
from faultforge._internal.progress import Progress, stage

CROSS_ENTROPY_IGNORE = -100
"""A sentinel value that pytorch uses to ignore cross-entropy loss for certain positions."""


@final
class WikiTextSplit(enum.StrEnum):
    # TODO: doc comment
    Test = "test"
    Train = "train"
    Validation = "validation"


@final
class WikiTextConfig(enum.StrEnum):
    # TODO: doc comment
    Tokenized2 = "wikitext-2-v1"
    Tokenized103 = "wikitext-103-v1"
    Raw2 = "wikitext-2-raw-v1"
    Raw103 = "wikitext-103-raw-v1"


@final
class WikiTextBundle(ModelBundle):
    """Loads a HuggingFace causal LM and the matching WikiText dataset for next-token evaluation.

    - `model_id`: the HuggingFace model id to load.
    - `dataset_config`: which WikiText variant to use.
    - `split`: which dataset split to load, defaults to `test`.
    - `dtype`: parameter dtype the model is cast to, defaults to `float32`.
    - `sequence_length`: the max sequence length to use, defaults to `None`
      (model's config).
    - `stride`: how many tokens the sliding window moves each step, defaults to
      `None` (half the sequence length). Keep it below `sequence_length`,
      otherwise some tokens are never scored. Larger than `sequence_length`
      raises a `ValueError`.

    The tokenizer is resolved and cached automatically (its pad token defaults
    to `eos_token` if the model doesn't define one), as is the max sequence
    length (from the model's config, falling back to 512 if undetermined).
    """

    def __init__(
        self,
        model_id: str,
        sequence_length: int | None = None,
        stride: int | None = None,
        dataset_config: str = WikiTextConfig.Raw2,
        split: WikiTextSplit = WikiTextSplit.Test,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self._model_id = model_id
        self._sequence_length = (
            sequence_length
            if sequence_length is not None
            else model_max_seq_len(self._model_id)
        )
        self._stride = stride if stride is not None else self._sequence_length // 2

        self._dataset_config = dataset_config
        self._split = split
        self._dtype = dtype

        self._tokenizer = CausalTokenizer(self._model_id)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="wikitext",
            scalars={
                "model": self._model_id,
                "dataset_config": self._dataset_config,
                "split": self._split.value,
                "sequence_len": self._sequence_length,
                "stride": self._stride,
                "dtype": dtype_name(self._dtype),
            },
        )

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        progress: Progress | None = None,
    ) -> nn.Module:
        with stage(progress, f"Loading model {self._model_id}"):
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id, torch_dtype=self._dtype
            )
        return HuggingFaceLMWrapper(model).to(device=device, dtype=self._dtype)

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
        with stage(progress, f"Loading dataset {self._dataset_config}"):
            text = load_wikitext(self._dataset_config, self._split)
            dataset = SlidingWindowDataset(
                token_ids=self._tokenizer.encode(text),
                padding_token_id=self._tokenizer.padding_token_id,
                sequence_length=self._sequence_length,
                stride=self._stride,
            )
        # batch_size = dataset._inputs.shape[0]
        return BatchedDataset.from_dataset(
            dataset, batch_size, device, shuffle=shuffle, seed=seed
        )


class HuggingFaceLMWrapper(nn.Module):
    """Unwraps a HuggingFace causal LM's output down to a plain logits Tensor.

    HuggingFace causal LMs return a lot more than just logits. It returns an
    object with different metadata. This wrapper class extracts just the logits
    for use in the model bundle.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        return self._model(input_ids).logits


@final
class CausalTokenizer:
    """A HuggingFace tokenizer with a guaranteed `eos_token_id` and `pad_token_id`.

    If the tokenizer defines no pad token, `pad_token_id` falls back to
    `eos_token_id`. Raises `ValueError` if there's no eos token.
    """

    def __init__(self, model_id: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise RuntimeError(
                f"Expected a PreTrainedTokenizerBase for {model_id}, got {type(tokenizer)}"
            )

        end_of_sequence_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(end_of_sequence_token_id, int):
            raise ValueError(f"Tokenizer for {model_id} defines no eos token")

        padding_token_id = getattr(tokenizer, "pad_token_id", None)
        if padding_token_id is None:
            padding_token_id = end_of_sequence_token_id
        if not isinstance(padding_token_id, int):
            raise ValueError(
                f"Tokenizer for {model_id} has an invalid pad token: {padding_token_id!r}"
            )

        self._tokenizer = tokenizer
        self.end_of_sequence_token_id: Final[int] = end_of_sequence_token_id
        self.padding_token_id: Final[int] = padding_token_id

    def encode(self, text: str) -> Tensor:
        """Tokenizes `text` into a flat 1D tensor of token ids."""
        input_ids = self._tokenizer(text, return_tensors="pt")["input_ids"]
        if not isinstance(input_ids, Tensor):
            raise RuntimeError(
                f"Expected tokenizer to return a Tensor, got {type(input_ids)}"
            )
        # `return_tensors="pt"` always returns a batch, here of one: `(1, n)`.
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(
                f"Expected tokenizer to return shape (1, n), got {tuple(input_ids.shape)}"
            )
        return input_ids[0]


class SlidingWindowDataset(Dataset):
    """Formats a token stream into `(input, target)` token ID pairs for next-token evaluation.

    Given one flat sequence of token ids (e.g. a tokenized WikiText split), this
    class slices it into windows of inputs according to the model's context
    length.

    - inputs: a sequence of token id windows, used for model input. The last
      window may be padded with `pad_id` tokens so every window ends up with the
      same length.
    - targets: the same window shifted by 1 position, with [`CROSS_ENTROPY_IGNORE`]
      marking positions that shouldn't be scored for evaluation.

    We use windows because of the model's maximum sequence length - the full
    token stream can't fit into a single forward pass. The sliding window
    produces `(input slice, next-token target)` pairs by moving across the token
    stream `stride` tokens at a time. If `stride` is not given, it defaults to
    half the sequence length. Overlapping windows (stride < sequence_length)
    ensures most predictions have real preceding context to draw from.

    This format is specially designed for next-token prediction evaluation (see
    [`Perplexity`]).

    Mirrors HuggingFace's own [sliding-window perplexity recipe].

    [`Perplexity`]: faultforge.metric.Perplexity
    [sliding-window perplexity recipe]: https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        token_ids: Tensor,
        padding_token_id: int,
        sequence_length: int,
        stride: int | None = None,
    ) -> None:
        if stride is None:
            stride = sequence_length // 2
        if stride > sequence_length:
            raise ValueError(
                f"stride ({stride}) must not be larger than sequence_length "
                f"({sequence_length}), otherwise tokens between windows are skipped"
            )
        if stride < 1:
            raise ValueError("expected a stride of at least 1")
        if sequence_length < 2:
            raise ValueError(
                f"expected a sequence_length of at least 2, got {sequence_length}"
            )
        if len(token_ids.shape) != 1:
            raise ValueError("expected a 1D tensor of token IDs")

        token_count = token_ids.numel()

        inputs: list[Tensor] = []
        targets: list[Tensor] = []

        previous_end_index = 0

        for window_start_index in range(0, token_count, stride):
            window_end_index = min(window_start_index + sequence_length, token_count)

            input_window = token_ids[window_start_index:window_end_index]

            # Every position is ignored by default and "valid" indices are
            # filled in below.
            target_window = torch.full_like(input_window, CROSS_ENTROPY_IGNORE)

            target_window[:-1] = input_window[1:]

            # On the last window target length may be shorter than sequence_length
            target_length = window_end_index - previous_end_index

            # Each window overlaps with the previous one, so we mask out the overlap
            window_overlap_mask = input_window.numel() - target_length - 1
            if window_overlap_mask > 0:
                target_window[:window_overlap_mask] = CROSS_ENTROPY_IGNORE

            if input_window.numel() < sequence_length:
                required_padding = sequence_length - input_window.numel()

                input_window = torch.cat(
                    [
                        input_window,
                        torch.full((required_padding,), padding_token_id),
                    ]
                )
                target_window = torch.cat(
                    [
                        target_window,
                        torch.full((required_padding,), CROSS_ENTROPY_IGNORE),
                    ]
                )

            previous_end_index = window_end_index

            inputs.append(input_window)
            targets.append(target_window)
            if window_end_index == token_count:
                break

        self._inputs = torch.stack(inputs)
        self._targets = torch.stack(targets)

    def __len__(self) -> int:
        return self._inputs.shape[0]

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._inputs[index], self._targets[index]


def load_wikitext(config: str, split: WikiTextSplit) -> str:
    """Loads a WikiText split as one string.

    Lines are joined exactly like HuggingFace's perplexity recipe (empty lines
    included) so results are comparable to published numbers.
    """
    raw = datasets.load_dataset("Salesforce/wikitext", config, split=split.value)
    return "\n\n".join(raw["text"])


def model_max_seq_len(model_id: str) -> int:
    """The model's own maximum context length, or 512 if it can't be determined."""
    config = AutoConfig.from_pretrained(model_id)
    # max_position_embeddings, n_positions, n_ctx are common names for sequence
    # length in HuggingFace models
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return value
    return 512
