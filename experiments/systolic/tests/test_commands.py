"""Tests for the `systolic` CLI's argument handling."""

import click
import pytest
import typer
from systolic import commands
from systolic.commands import (
    DatasetChoice,
    MetricChoice,
    _init_model_bundle,
    _resolve_metric,
    app,
)
from typer.testing import CliRunner

from faultforge.loading import (
    AttentionImplementation,
    Cifar,
    WikiTextBundle,
)

_REVISION = "0123456789abcdef0123456789abcdef01234567"

_IMAGE_DATASETS = [
    DatasetChoice.Cifar10,
    DatasetChoice.Cifar100,
    DatasetChoice.ImageNet,
]
_CLASSIFICATION_METRICS = [
    metric for metric in MetricChoice if metric != MetricChoice.Perplexity
]


@pytest.fixture
def resolved_revisions(monkeypatch) -> list[tuple[str, str]]:
    """Replace the Hub lookup, recording each `(model_id, revision)` it's asked for."""
    requests: list[tuple[str, str]] = []

    def fake_resolve(model_id: str, revision: str = "main") -> str:
        requests.append((model_id, revision))
        return _REVISION

    monkeypatch.setattr(commands, "resolve_huggingface_revision", fake_resolve)
    return requests


def test_wikitext_defaults_to_perplexity() -> None:
    assert _resolve_metric(DatasetChoice.WikiText, None) == MetricChoice.Perplexity


@pytest.mark.parametrize("dataset", _IMAGE_DATASETS)
def test_image_datasets_default_to_accuracy(dataset: DatasetChoice) -> None:
    assert _resolve_metric(dataset, None) == MetricChoice.Accuracy


@pytest.mark.parametrize("dataset", _IMAGE_DATASETS)
def test_image_datasets_reject_perplexity(dataset: DatasetChoice) -> None:
    with pytest.raises(typer.BadParameter, match="perplexity"):
        _ = _resolve_metric(dataset, MetricChoice.Perplexity)


@pytest.mark.parametrize("metric", _CLASSIFICATION_METRICS)
def test_wikitext_rejects_classification_metrics(metric: MetricChoice) -> None:
    with pytest.raises(typer.BadParameter, match="wikitext"):
        _ = _resolve_metric(DatasetChoice.WikiText, metric)


def test_wikitext_bundle_uses_eager_attention(resolved_revisions) -> None:
    bundle = _init_model_bundle(
        DatasetChoice.WikiText, "gpt2", None, sequence_length=8, stride=4
    )

    assert resolved_revisions == [("gpt2", "main")]
    assert bundle == WikiTextBundle(
        model_id="gpt2",
        revision=_REVISION,
        sequence_length=8,
        stride=4,
        attention=AttentionImplementation.Eager,
    )


def test_wikitext_bundle_resolves_given_revision(resolved_revisions) -> None:
    bundle = _init_model_bundle(
        DatasetChoice.WikiText,
        "gpt2",
        None,
        model_revision="v1.0",
        sequence_length=8,
    )

    assert resolved_revisions == [("gpt2", "v1.0")]
    assert isinstance(bundle, WikiTextBundle)
    assert bundle.revision == _REVISION


def test_wikitext_bundle_requires_sequence_length(resolved_revisions) -> None:
    with pytest.raises(typer.BadParameter, match="--sequence-length"):
        _ = _init_model_bundle(DatasetChoice.WikiText, "gpt2", None)
    assert resolved_revisions == []


@pytest.mark.parametrize(
    ("option", "arguments"),
    [
        ("--model-revision", {"model_revision": "main"}),
        ("--sequence-length", {"sequence_length": 8}),
        ("--stride", {"stride": 4}),
    ],
)
def test_wikitext_options_are_rejected_for_other_datasets(
    option: str, arguments: dict
) -> None:
    with pytest.raises(typer.BadParameter, match=option):
        _ = _init_model_bundle(DatasetChoice.Cifar10, "resnet20", None, **arguments)


def test_other_datasets_accept_missing_wikitext_options() -> None:
    bundle = _init_model_bundle(DatasetChoice.Cifar10, "resnet20", None)
    assert isinstance(bundle, Cifar)


def test_list_models_for_wikitext() -> None:
    result = CliRunner().invoke(app, ["list-models", "--dataset", "wikitext"])

    assert result.exit_code == 0
    assert "HuggingFace Hub" in result.output


def test_profile_rejects_wikitext(resolved_revisions) -> None:
    result = CliRunner().invoke(
        app,
        [
            "profile",
            "--model",
            "gpt2",
            "--dataset",
            "wikitext",
            "--array-rows",
            "4",
            "--array-cols",
            "4",
            "--capacity",
            "1",
            "--subsample-size",
            "1",
            "--output",
            "unused",
        ],
    )

    assert result.exit_code != 0
    # Typer forces styled output when `GITHUB_ACTIONS` is set, and the styling
    # splits option names apart, so match against the unstyled text.
    assert "--dataset" in click.unstyle(result.output)
    assert resolved_revisions == []
