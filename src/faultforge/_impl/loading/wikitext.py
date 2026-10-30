"""The WikiText dataset with HuggingFace causal LMs."""

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final, override

import datasets
import torch
from huggingface_hub import (
    CacheNotFound,
    HfApi,
    is_offline_mode,
    scan_cache_dir,
)
from torch import Tensor, nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from faultforge._impl.dataset import BatchedDataset, DeviceLike
from faultforge._impl.dtype import dtype_name
from faultforge._impl.fingerprint import Fingerprint
from faultforge._impl.loading.abc import ModelBundle
from faultforge._impl.metric import Perplexity
from faultforge._impl.progress import Progress, stage

_WIKITEXT_REPOSITORY = "Salesforce/wikitext"
"""The HuggingFace dataset repository of WikiText."""

_WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
"""The pinned commit of [`_WIKITEXT_REPOSITORY`] that is loaded."""

_COMMIT_HASH_PATTERN = re.compile(r"[0-9a-f]{40}")
"""A full git commit hash as used by the HuggingFace Hub: 40 lowercase hex characters."""


@final
class WikiTextSplit(enum.StrEnum):
    """A split of the WikiText dataset."""

    Test = "test"
    Train = "train"
    Validation = "validation"


@final
class WikiTextConfig(enum.StrEnum):
    """A variant of the WikiText dataset.

    - `2` and `103` name the size of the dataset in millions of tokens.
    - The tokenized variants replace rare words with `<unk>`.
    - The raw variants keep the original text.
    """

    Tokenized2 = "wikitext-2-v1"
    Tokenized103 = "wikitext-103-v1"
    Raw2 = "wikitext-2-raw-v1"
    Raw103 = "wikitext-103-raw-v1"


@final
@dataclass(slots=True, frozen=True)
class WikiTextBundle(ModelBundle):
    """A HuggingFace causal LM and the WikiText dataset for next-token evaluation.

    Only models on the HuggingFace Hub are supported, local directories are not.

    Raises `ValueError` if:

    - `revision` isn't a full commit hash.
    - `model_id` is a local directory.
    - `sequence_length` or `stride` are invalid.
    """

    model_id: str
    """The HuggingFace Hub model id to load, e.g. `openai-community/gpt2`."""
    revision: str
    """The commit hash of the model to load.

    Branch names like `main` are rejected because they change over time. Use
    [`resolve_huggingface_revision`] to look up a branch's current commit.

    [`resolve_huggingface_revision`]: faultforge.loading.resolve_huggingface_revision
    """
    sequence_length: int
    """The number of tokens in each window. Must be at least 2.

    It shouldn't exceed the model's maximum context length. Memory use grows
    with it.
    """
    stride: int | None = None
    """How many tokens the window moves each step.

    Must be at least 1 and smaller than `sequence_length`. `None` uses half the
    sequence length.
    """
    dataset_config: WikiTextConfig = WikiTextConfig.Raw2
    """Which WikiText variant to use."""
    split: WikiTextSplit = WikiTextSplit.Test
    """Which dataset split to load."""
    dtype: torch.dtype = torch.float32
    """Parameter dtype the model is loaded with."""

    def __post_init__(self) -> None:
        if _COMMIT_HASH_PATTERN.fullmatch(self.revision) is None:
            raise ValueError(
                f"expected revision to be a full commit hash, got {self.revision!r}"
            )
        _ensure_not_local_model(self.model_id)
        _validate_window(self.sequence_length, self._resolved_stride())

    def _resolved_stride(self) -> int:
        """The stride with the default filled in."""
        if self.stride is None:
            return self.sequence_length // 2
        return self.stride

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="wikitext",
            scalars={
                "model": self.model_id,
                "model_revision": self.revision,
                "dataset_revision": _WIKITEXT_REVISION,
                "dataset_config": self.dataset_config.value,
                "split": self.split.value,
                "sequence_length": self.sequence_length,
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
        _ensure_not_local_model(self.model_id)
        with stage(progress, f"Loading model {self.model_id}"):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, dtype=self.dtype
            )
        return _HuggingFaceLmWrapper(model).to(device=device)

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
        _ensure_not_local_model(self.model_id)
        with stage(progress, f"Loading dataset {self.dataset_config}"):
            tokenizer = _CausalTokenizer(self.model_id, self.revision)
            text = _load_wikitext(self.dataset_config, self.split)
            dataset = _SlidingWindowDataset(
                token_ids=tokenizer.encode(text),
                padding_token_id=tokenizer.pad_token_id,
                sequence_length=self.sequence_length,
                stride=self._resolved_stride(),
            )
        return BatchedDataset.from_dataset(
            dataset, batch_size, device, shuffle=shuffle, seed=seed
        )


@final
class _HuggingFaceLmWrapper(nn.Module):
    """Unwraps a HuggingFace causal LM's output down to a plain logits Tensor.

    HuggingFace causal LMs return an object with the logits and other metadata.
    This wrapper returns only the logits.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    # torch types `forward` as a mutable `Callable[..., Any]`; narrowing it is intended.
    @override
    def forward(self, input_ids: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self._model(input_ids).logits


@final
class _CausalTokenizer:
    """A HuggingFace tokenizer with a guaranteed `pad_token_id`.

    If the tokenizer defines no pad token, `pad_token_id` falls back to the
    eos token. Raises `TypeError` if there's no eos token.
    """

    def __init__(self, model_id: str, revision: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(
                f"expected a PreTrainedTokenizerBase for {model_id}, got {type(tokenizer)}"
            )

        end_of_sequence_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(end_of_sequence_token_id, int):
            raise TypeError(f"tokenizer for {model_id} defines no eos token")

        padding_token_id = getattr(tokenizer, "pad_token_id", None)
        if padding_token_id is None:
            padding_token_id = end_of_sequence_token_id
        if not isinstance(padding_token_id, int):
            raise TypeError(
                f"tokenizer for {model_id} has an invalid pad token: {padding_token_id!r}"
            )

        self._tokenizer = tokenizer
        self.pad_token_id: Final[int] = padding_token_id

    def encode(self, text: str) -> Tensor:
        """Tokenizes `text` into a flat 1D tensor of token ids."""
        input_ids = self._tokenizer(text, return_tensors="pt")["input_ids"]
        if not isinstance(input_ids, Tensor):
            raise TypeError(
                f"expected tokenizer to return a Tensor, got {type(input_ids)}"
            )
        # `return_tensors="pt"` always returns a batch, here of one: `(1, n)`.
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(
                f"expected tokenizer to return shape (1, n), got {tuple(input_ids.shape)}"
            )
        return input_ids[0]


@final
class _SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Formats a token stream into `(input, target)` token id pairs for next-token evaluation.

    Given one flat sequence of token ids (e.g. a tokenized WikiText split), this
    class slices it into windows that fit the model's context length.

    - inputs: windows of `sequence_length` token ids. The last window may be
      padded with `padding_token_id` so every window has the same length.
    - targets: the same window shifted by 1 position. Positions that shouldn't
      be scored are set to [`Perplexity.IGNORED_TARGET`].

    The window moves across the token stream `stride` tokens at a time.
    Overlapping windows give most predictions real preceding context. Every
    token except the first is scored exactly once.

    Raises `ValueError` if:

    - `sequence_length` is smaller than 2.
    - `stride` is smaller than 1 or not smaller than `sequence_length`.
    - `token_ids` isn't 1D or has fewer than 2 tokens.

    This format is designed for next-token prediction evaluation with
    [`Perplexity`]. It mirrors HuggingFace's own [sliding-window perplexity
    recipe].

    [`Perplexity`]: faultforge.metric.Perplexity
    [`Perplexity.IGNORED_TARGET`]: faultforge.metric.Perplexity.IGNORED_TARGET
    [sliding-window perplexity recipe]: https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        token_ids: Tensor,
        padding_token_id: int,
        sequence_length: int,
        stride: int,
    ) -> None:
        _validate_window(sequence_length, stride)
        if token_ids.ndim != 1:
            raise ValueError(
                f"expected a 1D tensor of token ids, got shape {tuple(token_ids.shape)}"
            )

        token_count = token_ids.numel()
        # The first token has no context, so a single token has nothing to score.
        if token_count < 2:
            raise ValueError(f"expected at least 2 tokens, got {token_count}")

        inputs: list[Tensor] = []
        targets: list[Tensor] = []

        previous_end_index = 0

        for window_start_index in range(0, token_count, stride):
            window_end_index = min(window_start_index + sequence_length, token_count)

            input_window = token_ids[window_start_index:window_end_index]

            # Every position is ignored by default and "valid" indices are
            # filled in below.
            target_window = torch.full_like(input_window, Perplexity.IGNORED_TARGET)

            # The last position is left ignored because its target is outside
            # the window. The next window scores it instead, which is why
            # `stride` has to be smaller than `sequence_length`.
            target_window[:-1] = input_window[1:]

            # The number of tokens this window scores. On the last window it
            # may be shorter than `stride`.
            scored_token_count = window_end_index - previous_end_index

            # Mask out the overlap with the previous window, if any.
            overlap_length = input_window.numel() - scored_token_count - 1
            if overlap_length > 0:
                target_window[:overlap_length] = Perplexity.IGNORED_TARGET

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
                        torch.full((required_padding,), Perplexity.IGNORED_TARGET),
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


def _validate_window(sequence_length: int, stride: int) -> None:
    """Checks the sliding window parameters of [`_SlidingWindowDataset`]."""
    # The last position of a window is never scored because the token it
    # predicts is outside the window. A window of 1 token would score nothing.
    if sequence_length < 2:
        raise ValueError(
            f"expected a sequence_length of at least 2, got {sequence_length}"
        )
    # The window has to move forward. A stride of 0 would never advance and a
    # negative one would produce no windows at all.
    if stride < 1:
        raise ValueError(f"expected a stride of at least 1, got {stride}")
    # A window starting at `start` scores the tokens up to `start +
    # sequence_length - 1`, its own last token. The token after that is
    # predicted from the window's unscored last position, so the next window
    # has to score it. That window starts at `start + stride` and can only score
    # from `start + stride + 1` on, because its first token has no context.
    # Covering the gap needs `stride + 1 <= sequence_length`. With `stride ==
    # sequence_length`, one token per window boundary would never be scored.
    # HuggingFace's perplexity recipe has the same gap.
    if stride >= sequence_length:
        raise ValueError(
            f"stride ({stride}) must be smaller than sequence_length "
            f"({sequence_length}), otherwise tokens between windows are skipped"
        )


def _load_wikitext(config: WikiTextConfig, split: WikiTextSplit) -> str:
    """Loads a WikiText split at [`_WIKITEXT_REVISION`] as one string.

    Lines are joined exactly like HuggingFace's perplexity recipe (empty lines
    included) so results are comparable to published numbers.
    """
    split_dataset = datasets.load_dataset(
        _WIKITEXT_REPOSITORY,
        config.value,
        split=split.value,
        revision=_WIKITEXT_REVISION,
    )
    return "\n\n".join(split_dataset["text"])


def _ensure_not_local_model(model_id: str) -> None:
    """Raises `ValueError` if `model_id` refers to a local directory.

    HuggingFace loads a local directory instead of the Hub model when one
    exists at `model_id`.
    """
    # Local models have no commit hash to fingerprint, so they're rejected
    # instead of being silently loaded in place of the Hub model.
    if Path(model_id).is_dir():
        raise ValueError(
            f"{model_id!r} is a local directory, only HuggingFace Hub models are supported"
        )


def resolve_huggingface_revision(model_id: str, revision: str = "main") -> str:
    """Resolves a revision of a HuggingFace Hub model to a full commit hash.

    - A full commit hash is returned unchanged.
    - A branch or tag is looked up on the Hub.
    - In offline mode (`HF_HUB_OFFLINE=1`), a branch or tag is looked up in the
      local HuggingFace cache instead. This gives the commit it pointed to
      when the model was last downloaded.

    Raises:

    - `ValueError` if the revision can't be found in offline mode.
    - `huggingface_hub.errors.RepositoryNotFoundError` or
      `huggingface_hub.errors.RevisionNotFoundError` if the Hub doesn't know
      the model or revision.
    """
    if _COMMIT_HASH_PATTERN.fullmatch(revision) is not None:
        return revision

    if is_offline_mode():
        return _cached_huggingface_revision(model_id, revision)

    commit_hash = HfApi().model_info(model_id, revision=revision).sha
    if commit_hash is None:
        raise RuntimeError(f"the Hub returned no commit hash for {model_id}@{revision}")
    return commit_hash


def _cached_huggingface_revision(model_id: str, revision: str) -> str:
    """Looks up the commit hash of a branch or tag in the local HuggingFace cache."""
    try:
        cache = scan_cache_dir()
    except CacheNotFound as error:
        raise ValueError(
            f"can't resolve {model_id}@{revision} offline: no HuggingFace cache found"
        ) from error

    for repository in cache.repos:
        if repository.repo_type != "model" or repository.repo_id != model_id:
            continue
        cached_revision = repository.refs.get(revision)
        if cached_revision is not None:
            return cached_revision.commit_hash

    raise ValueError(
        f"can't resolve {model_id}@{revision} offline: it isn't in the HuggingFace cache"
    )
