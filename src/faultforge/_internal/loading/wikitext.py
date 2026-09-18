"""Parsing and loading Wikitext dataset with and HuggingFace's decoder models"""

from dataclasses import dataclass, field
from typing import override

import torch
from datasets import load_dataset
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


class _HuggingFaceLMWrapper(nn.Module):
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


def _model_max_seq_len(model_id: str) -> int:
    """The model's own maximum context length, or 512 if it can't be determined."""
    config = AutoConfig.from_pretrained(model_id)
    # max_position_embeddings, n_positions, n_ctx are common names for sequence
    # length in HuggingFace models
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return value
    return 512


class _WikiTextFormatter(Dataset):
    """Formats WikiText into `(input, target)` token ID pairs for next-token evaluation.

    Given a tokenizer and a text split (both chosen by the caller or
    defaulted), this class loads the split's raw text, tokenizes it into one
    flat sequence of ids, and slices that sequence into sliding windows of
    inputs according to the model's context length.

    - inputs: a window of token ids, meant to be fed to the model directly. The
      final window may include trailing `pad_id` tokens so every window has the
      same length.
    - targets: the same window shifted by 1 position, with `-100` marking
      positions that shouldn't be scored for evaluation.

    Windowing exists because of the model's maximum sequence length - the full
    token stream can't fit into a single forward pass. The sliding window
    produces `(input, next-token target)` pairs by moving across the token
    stream `stride` tokens at a time. If `stride` isn't given, it defaults to
    half the sequence length. Overlapping windows this way (rather than
    non-overlapping chunks) ensures most predictions have real preceding
    context to draw from.

    This formatting is specially designed for next-token prediction evaluation
    (see [`Perplexity`]).

    Mirrors HuggingFace's own [sliding-window perplexity recipe].

    [`Perplexity`]: faultforge.metric.Perplexity
    [sliding-window perplexity recipe]: https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        config: str,
        split: str,
        stride: int | None = None,
    ) -> None:
        if stride is None:
            stride = sequence_length // 2
        if stride > sequence_length:
            raise ValueError(
                f"stride ({stride}) must not be larger than sequence_length "
                f"({sequence_length}), otherwise tokens between windows are skipped"
            )

        raw = load_dataset("Salesforce/wikitext", config, split=split)
        # Joined exactly like HuggingFace's perplexity recipe (empty lines
        # included) so results are comparable to published numbers.
        text = "\n\n".join(raw["text"])
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        pad_id = tokenizer.pad_token_id
        assert pad_id is not None

        total = ids.numel()

        inputs: list[Tensor] = []
        targets: list[Tensor] = []

        previous_end = 0

        # `stride` should be kept below `sequence_length`: at
        # `stride == sequence_length` the windows stop overlapping, so the
        # first token of each window has no context and is never scored.
        # Larger strides would skip whole gaps and are rejected above.
        for begin in range(0, total, stride):
            end = min(begin + sequence_length, total)
            target_length = end - previous_end

            input_window = ids[begin:end]

            target_window = torch.full_like(input_window, -100)

            # In case the final window is 1 token long.
            if end - begin > 1:
                target_window[:-1] = ids[begin + 1 : end]

            # HF masks the overlap before shifting, and the shift drops the
            # window's first position from the shifted array entirely - so
            # the equivalent mask on already-shifted `window_target` is
            # one shorter than the unshifted `mask_len`.
            mask_len = (end - begin) - target_length
            shifted_mask_len = max(mask_len - 1, 0)
            if shifted_mask_len > 0:
                target_window[:shifted_mask_len] = -100

            if input_window.numel() < sequence_length:
                pad_len = sequence_length - input_window.numel()
                input_window = torch.cat(
                    [input_window, input_window.new_full((pad_len,), pad_id)]
                )
                target_window = torch.cat(
                    [target_window, target_window.new_full((pad_len,), -100)]
                )

            inputs.append(input_window)
            targets.append(target_window)

            previous_end = end
            if end == total:
                break

        self._inputs = torch.stack(inputs)
        self._targets = torch.stack(targets)

    def __len__(self) -> int:
        return self._inputs.shape[0]

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._inputs[index], self._targets[index]


@dataclass(slots=True)
class WikiTextBundle(ModelBundle):
    """Loads a HuggingFace causal LM and the matching WikiText dataset for next-token evaluation.

    - `model_id`: the HuggingFace model id to load.
    - `dataset_config`: which WikiText variant to use (e.g. `wikitext-2-raw-v1`
      vs `wikitext-103-raw-v1`).
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

    model_id: str
    sequence_length: int | None = None
    stride: int | None = None
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    dtype: torch.dtype = torch.float32
    _tokenizer: PreTrainedTokenizerBase | None = field(default=None, init=False)
    _seq_len_cache: int | None = field(default=None, init=False)

    def _cached_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            assert tokenizer is not None
            self._tokenizer = tokenizer
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _resolved_seq_len(self) -> int:
        if self._seq_len_cache is None:
            self._seq_len_cache = (
                self.sequence_length
                if self.sequence_length is not None
                else _model_max_seq_len(self.model_id)
            )
        return self._seq_len_cache

    def _resolved_stride(self) -> int:
        return self.stride if self.stride is not None else self._resolved_seq_len() // 2

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="huggingface_causal_lm",
            scalars={
                "model": self.model_id,
                "dataset_config": self.dataset_config,
                "split": self.split,
                "seq_len": self._resolved_seq_len(),
                "stride": self._resolved_stride(),
                "dtype": dtype_name(self.dtype),
            },
        )

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        progress: Progress | None = None,
    ) -> nn.Module:
        with stage(progress, f"Loading model {self.model_id}"):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            )
        return _HuggingFaceLMWrapper(model).to(device=device, dtype=self.dtype)

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
        with stage(progress, f"Loading dataset {self.dataset_config}"):
            dataset = _WikiTextFormatter(
                self._cached_tokenizer(),
                self._resolved_seq_len(),
                self.dataset_config,
                self.split,
                stride=self._resolved_stride(),
            )
        return BatchedDataset.from_dataset(
            dataset, batch_size, device, shuffle=shuffle, seed=seed
        )
