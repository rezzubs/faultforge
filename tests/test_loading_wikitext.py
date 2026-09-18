"""Tests for WikiText loading (`faultforge._impl.loading.wikitext`)."""

from pathlib import Path
from types import SimpleNamespace
from typing import override
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from transformers import PreTrainedTokenizerBase

from faultforge._impl.loading import wikitext
from faultforge.metric import Perplexity

PADDING_TOKEN_ID = 0
IGNORED_TARGET = Perplexity.IGNORED_TARGET
REVISION = "0123456789abcdef0123456789abcdef01234567"


def make_windows(
    token_count: int, sequence_length: int = 4, stride: int = 2
) -> wikitext.SlidingWindowDataset:
    """Build the dataset on the token ids `1..token_count`.

    Ids start at 1 so they never collide with the padding token id.
    """
    return wikitext.SlidingWindowDataset(
        torch.arange(1, token_count + 1),
        PADDING_TOKEN_ID,
        sequence_length=sequence_length,
        stride=stride,
    )


def all_inputs(dataset: wikitext.SlidingWindowDataset) -> torch.Tensor:
    return torch.stack([dataset[index][0] for index in range(len(dataset))])


def all_targets(dataset: wikitext.SlidingWindowDataset) -> torch.Tensor:
    return torch.stack([dataset[index][1] for index in range(len(dataset))])


def test_fetches_pinned_wikitext_with_config_and_split() -> None:
    with patch.object(
        wikitext.datasets, "load_dataset", return_value={"text": ["abc"]}
    ) as load_dataset:
        wikitext.load_wikitext(
            wikitext.WikiTextConfig.Raw2, wikitext.WikiTextSplit.Validation
        )

    load_dataset.assert_called_once_with(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        revision=wikitext.WIKITEXT_REVISION,
    )


def test_text_is_joined_like_huggingface() -> None:
    # HuggingFace's perplexity recipe joins with "\n\n" and keeps empty lines.
    with patch.object(
        wikitext.datasets,
        "load_dataset",
        return_value={"text": ["first", "", "second"]},
    ):
        text = wikitext.load_wikitext(
            wikitext.WikiTextConfig.Raw2, wikitext.WikiTextSplit.Test
        )

    assert text == "first\n\n\n\nsecond"


@pytest.mark.parametrize("stride", [4, 5])
def test_stride_not_smaller_than_sequence_length_is_rejected(stride: int) -> None:
    # The last position of each window is never scored, so the next window has
    # to overlap the previous one to score it.
    with pytest.raises(ValueError, match="stride"):
        make_windows(8, sequence_length=4, stride=stride)


def test_stride_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="stride"):
        make_windows(8, sequence_length=4, stride=0)


def test_sequence_length_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        make_windows(8, sequence_length=1, stride=1)


@pytest.mark.parametrize("token_count", [0, 1])
def test_fewer_than_two_tokens_are_rejected(token_count: int) -> None:
    # The first token can't be scored, so there would be nothing to evaluate.
    with pytest.raises(ValueError, match="at least 2 tokens"):
        make_windows(token_count)


def test_multidimensional_token_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="1D"):
        wikitext.SlidingWindowDataset(
            torch.ones(2, 4, dtype=torch.int64), PADDING_TOKEN_ID, 4, 2
        )


def test_inputs_are_overlapping_windows_padded_at_the_end() -> None:
    dataset = make_windows(7)

    assert all_inputs(dataset).tolist() == [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [5, 6, 7, PADDING_TOKEN_ID],
    ]


def test_inputs_have_sequence_length_for_any_text_length() -> None:
    short = make_windows(3)
    long = make_windows(9)

    assert all_inputs(short).shape == (1, 4)
    assert all_inputs(long).shape == (4, 4)


def test_text_of_exactly_one_window_is_not_padded() -> None:
    dataset = make_windows(4)

    assert all_inputs(dataset).tolist() == [[1, 2, 3, 4]]


def test_targets_mask_overlap_last_position_and_padding() -> None:
    # Positions that aren't scored: the overlap already scored by the previous
    # window, the last position (scored by the next window) and the padding.
    dataset = make_windows(7)

    assert all_targets(dataset).tolist() == [
        [2, 3, 4, IGNORED_TARGET],
        [IGNORED_TARGET, 5, 6, IGNORED_TARGET],
        [IGNORED_TARGET, 7, IGNORED_TARGET, IGNORED_TARGET],
    ]


@given(
    token_count=st.integers(min_value=2, max_value=64),
    sequence_length=st.integers(min_value=2, max_value=16),
    data=st.data(),
)
def test_every_token_but_the_first_is_scored_exactly_once(
    token_count: int, sequence_length: int, data: st.DataObject
) -> None:
    # Scoring a token twice or never would bias the perplexity.
    stride = data.draw(st.integers(min_value=1, max_value=sequence_length - 1))
    dataset = make_windows(token_count, sequence_length, stride)

    targets = all_targets(dataset)
    scored = targets[targets != IGNORED_TARGET]

    assert sorted(scored.tolist()) == list(range(2, token_count + 1))


@given(
    token_count=st.integers(min_value=2, max_value=64),
    sequence_length=st.integers(min_value=2, max_value=16),
    data=st.data(),
)
def test_scored_targets_are_the_next_input_token(
    token_count: int, sequence_length: int, data: st.DataObject
) -> None:
    stride = data.draw(st.integers(min_value=1, max_value=sequence_length - 1))
    dataset = make_windows(token_count, sequence_length, stride)

    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        scored = targets[:-1] != IGNORED_TARGET
        assert targets[:-1][scored].tolist() == inputs[1:][scored].tolist()
        assert targets[-1] == IGNORED_TARGET


@given(
    token_count=st.integers(min_value=2, max_value=64),
    sequence_length=st.integers(min_value=2, max_value=16),
    data=st.data(),
)
def test_padding_is_only_at_the_end_of_the_last_window(
    token_count: int, sequence_length: int, data: st.DataObject
) -> None:
    stride = data.draw(st.integers(min_value=1, max_value=sequence_length - 1))
    dataset = make_windows(token_count, sequence_length, stride)

    inputs = all_inputs(dataset)
    real_token_count = int((inputs[-1] != PADDING_TOKEN_ID).sum())

    assert PADDING_TOKEN_ID not in inputs[:-1]
    assert (inputs[-1, real_token_count:] == PADDING_TOKEN_ID).all()
    assert inputs[-1, real_token_count - 1] == token_count


def test_bundle_construction_loads_nothing() -> None:
    with (
        patch.object(wikitext.AutoTokenizer, "from_pretrained") as tokenizer_loader,
        patch.object(wikitext.AutoModelForCausalLM, "from_pretrained") as model_loader,
    ):
        wikitext.WikiTextBundle(model_id="model", revision=REVISION, sequence_length=8)

    tokenizer_loader.assert_not_called()
    model_loader.assert_not_called()


@pytest.mark.parametrize(("sequence_length", "stride"), [(8, 8), (8, 0), (1, None)])
def test_bundle_rejects_invalid_window(
    sequence_length: int, stride: int | None
) -> None:
    with pytest.raises(ValueError):
        wikitext.WikiTextBundle(
            model_id="model",
            revision=REVISION,
            sequence_length=sequence_length,
            stride=stride,
        )


@pytest.mark.parametrize(
    "revision",
    [
        "main",
        "v1.0",
        # Abbreviated hash.
        REVISION[:7],
        REVISION.upper(),
    ],
)
def test_bundle_rejects_revisions_other_than_commit_hashes(revision: str) -> None:
    with pytest.raises(ValueError, match="commit hash"):
        wikitext.WikiTextBundle(model_id="model", revision=revision, sequence_length=8)


def test_bundle_rejects_local_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A directory shadows the Hub model of the same name.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model").mkdir()

    with pytest.raises(ValueError, match="local directory"):
        wikitext.WikiTextBundle(model_id="model", revision=REVISION, sequence_length=8)


def test_loading_rejects_local_models_created_after_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle = wikitext.WikiTextBundle(
        model_id="model", revision=REVISION, sequence_length=8
    )
    (tmp_path / "model").mkdir()

    with pytest.raises(ValueError, match="local directory"):
        bundle.load_model("cpu")
    with pytest.raises(ValueError, match="local directory"):
        bundle.load_dataset(1, "cpu")


def test_stride_defaults_to_half_the_sequence_length() -> None:
    bundle = wikitext.WikiTextBundle(
        model_id="model", revision=REVISION, sequence_length=8
    )

    assert bundle.fingerprint().scalars["stride"] == 4


def test_fingerprint_is_filled_in() -> None:
    bundle = wikitext.WikiTextBundle(
        model_id="model", revision=REVISION, sequence_length=8
    )

    fingerprint = bundle.fingerprint()

    assert fingerprint.kind == "wikitext"
    assert fingerprint.scalars == {
        "model": "model",
        "model_revision": REVISION,
        "dataset_revision": wikitext.WIKITEXT_REVISION,
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "sequence_length": 8,
        "stride": 4,
        "dtype": "float32",
    }


def test_resolving_a_commit_hash_returns_it_unchanged() -> None:
    with patch.object(wikitext.HfApi, "model_info") as model_info:
        assert wikitext.resolve_huggingface_revision("model", REVISION) == REVISION

    model_info.assert_not_called()


def test_resolving_a_branch_looks_it_up_on_the_hub() -> None:
    with (
        patch.object(wikitext, "is_offline_mode", return_value=False),
        patch.object(
            wikitext.HfApi, "model_info", return_value=SimpleNamespace(sha=REVISION)
        ) as model_info,
    ):
        assert wikitext.resolve_huggingface_revision("model", "dev") == REVISION

    model_info.assert_called_once_with("model", revision="dev")


def fake_cache(*repositories: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(repos=list(repositories))


def fake_cached_repository(
    repository_id: str, repository_type: str, refs: dict[str, str]
) -> SimpleNamespace:
    return SimpleNamespace(
        repo_id=repository_id,
        repo_type=repository_type,
        refs={
            ref: SimpleNamespace(commit_hash=commit_hash)
            for ref, commit_hash in refs.items()
        },
    )


def test_resolving_a_branch_offline_uses_the_cache() -> None:
    other_revision = "f" * 40
    cache = fake_cache(
        # Same id but a dataset, which must not match.
        fake_cached_repository("model", "dataset", {"main": other_revision}),
        fake_cached_repository("model", "model", {"main": REVISION}),
    )

    with (
        patch.object(wikitext, "is_offline_mode", return_value=True),
        patch.object(wikitext, "scan_cache_dir", return_value=cache),
        patch.object(wikitext.HfApi, "model_info") as model_info,
    ):
        assert wikitext.resolve_huggingface_revision("model", "main") == REVISION

    model_info.assert_not_called()


def test_resolving_an_uncached_branch_offline_is_rejected() -> None:
    cache = fake_cache(fake_cached_repository("model", "model", {"main": REVISION}))

    with (
        patch.object(wikitext, "is_offline_mode", return_value=True),
        patch.object(wikitext, "scan_cache_dir", return_value=cache),
        pytest.raises(ValueError, match="isn't in the HuggingFace cache"),
    ):
        wikitext.resolve_huggingface_revision("model", "dev")


def test_resolving_offline_without_a_cache_is_rejected() -> None:
    with (
        patch.object(wikitext, "is_offline_mode", return_value=True),
        patch.object(
            wikitext, "scan_cache_dir", side_effect=wikitext.CacheNotFound("", "")
        ),
        pytest.raises(ValueError, match="no HuggingFace cache"),
    ):
        wikitext.resolve_huggingface_revision("model", "main")


def test_missing_pad_token_falls_back_to_eos() -> None:
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = None

    with patch.object(
        wikitext.AutoTokenizer, "from_pretrained", return_value=tokenizer
    ):
        causal_tokenizer = wikitext.CausalTokenizer("model", REVISION)

    assert causal_tokenizer.pad_token_id == 2


def test_wrapper_returns_only_the_logits() -> None:
    logits = torch.zeros(1, 3, 5)

    class FakeModel(torch.nn.Module):
        @override
        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:  # ty: ignore[invalid-mutable-override]
            _ = input_ids
            return SimpleNamespace(logits=logits, other_metadata="ignored")

    wrapper = wikitext.HuggingFaceLmWrapper(FakeModel())

    assert wrapper(torch.tensor([[1, 2, 3]])) is logits
