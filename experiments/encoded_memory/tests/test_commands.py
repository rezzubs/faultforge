"""Tests for the `encoded-memory` CLI's argument handling."""

import click
import pytest
import torch
import typer
from encoded_memory import commands
from encoded_memory.commands import (
    DatasetChoice,
    EstimateChoice,
    MetricChoice,
    _init_model_bundle,
    _resolve_metric,
    _resolve_stability,
    app,
)
from typer.testing import CliRunner

from faultforge.experiment import (
    AbsoluteThreshold,
    FailureRate,
    Mean,
    RelativeThreshold,
    Stability,
)
from faultforge.loading import Cifar, WikiTextBundle
from faultforge.metric import (
    Accuracy,
    Metric,
    Perplexity,
    ScoreDirection,
    Sdc,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"

IMAGE_DATASETS = [DatasetChoice.Cifar10, DatasetChoice.Cifar100, DatasetChoice.ImageNet]
CLASSIFICATION_METRICS = [
    metric for metric in MetricChoice if metric != MetricChoice.Perplexity
]


@pytest.fixture
def resolved_revisions(monkeypatch) -> list[tuple[str, str]]:
    """Replace the Hub lookup, recording each `(model_id, revision)` it's asked for."""
    requests: list[tuple[str, str]] = []

    def fake_resolve(model_id: str, revision: str = "main") -> str:
        requests.append((model_id, revision))
        return REVISION

    monkeypatch.setattr(commands, "resolve_huggingface_revision", fake_resolve)
    return requests


def init_bundle(
    dataset: DatasetChoice,
    *,
    model: str = "gpt2",
    model_revision: str | None = None,
    sequence_length: int | None = None,
    stride: int | None = None,
    dtype: torch.dtype = torch.float32,
):
    return _init_model_bundle(
        dataset,
        model,
        None,
        model_revision,
        sequence_length,
        stride,
        dtype,
    )


def test_wikitext_defaults_to_perplexity():
    assert _resolve_metric(DatasetChoice.WikiText, None) == MetricChoice.Perplexity


@pytest.mark.parametrize("dataset", IMAGE_DATASETS)
def test_image_datasets_default_to_accuracy(dataset: DatasetChoice):
    assert _resolve_metric(dataset, None) == MetricChoice.Accuracy


@pytest.mark.parametrize("dataset", IMAGE_DATASETS)
@pytest.mark.parametrize("metric", CLASSIFICATION_METRICS)
def test_image_datasets_accept_classification_metrics(
    dataset: DatasetChoice, metric: MetricChoice
):
    assert _resolve_metric(dataset, metric) == metric


@pytest.mark.parametrize("dataset", IMAGE_DATASETS)
def test_image_datasets_reject_perplexity(dataset: DatasetChoice):
    with pytest.raises(typer.BadParameter, match="perplexity"):
        _ = _resolve_metric(dataset, MetricChoice.Perplexity)


@pytest.mark.parametrize("metric", CLASSIFICATION_METRICS)
def test_wikitext_rejects_classification_metrics(metric: MetricChoice):
    with pytest.raises(typer.BadParameter, match="wikitext"):
        _ = _resolve_metric(DatasetChoice.WikiText, metric)


def test_wikitext_bundle_resolves_main_by_default(resolved_revisions):
    bundle = init_bundle(
        DatasetChoice.WikiText, sequence_length=8, stride=4, dtype=torch.float16
    )

    assert resolved_revisions == [("gpt2", "main")]
    assert bundle == WikiTextBundle(
        model_id="gpt2",
        revision=REVISION,
        sequence_length=8,
        stride=4,
        dtype=torch.float16,
    )


def test_wikitext_bundle_resolves_given_revision(resolved_revisions):
    bundle = init_bundle(
        DatasetChoice.WikiText, model_revision="v1.0", sequence_length=8
    )

    assert resolved_revisions == [("gpt2", "v1.0")]
    assert isinstance(bundle, WikiTextBundle)
    assert bundle.revision == REVISION


def test_wikitext_bundle_requires_sequence_length(resolved_revisions):
    with pytest.raises(typer.BadParameter, match="--sequence-length"):
        _ = init_bundle(DatasetChoice.WikiText)
    assert resolved_revisions == []


def test_wikitext_bundle_reports_invalid_window(resolved_revisions):
    _ = resolved_revisions
    with pytest.raises(typer.BadParameter, match="stride"):
        _ = init_bundle(DatasetChoice.WikiText, sequence_length=8, stride=8)


@pytest.mark.parametrize(
    ("option", "arguments"),
    [
        ("--model-revision", {"model_revision": "main"}),
        ("--sequence-length", {"sequence_length": 8}),
        ("--stride", {"stride": 4}),
    ],
)
def test_wikitext_options_are_rejected_for_other_datasets(option: str, arguments: dict):
    with pytest.raises(typer.BadParameter, match=option):
        _ = init_bundle(DatasetChoice.Cifar10, model="resnet20", **arguments)


def test_other_datasets_accept_missing_wikitext_options():
    bundle = init_bundle(DatasetChoice.Cifar10, model="resnet20")
    assert isinstance(bundle, Cifar)


def test_list_models_for_wikitext():
    result = CliRunner().invoke(app, ["list-models", "--dataset", "wikitext"])

    assert result.exit_code == 0
    assert "HuggingFace Hub" in result.output


def invoke_record(*arguments: str):
    return CliRunner().invoke(app, ["record", "--model", "gpt2", *arguments])


def test_record_requires_batch_size():
    result = invoke_record("--dataset", "wikitext", "--sequence-length", "8")

    assert result.exit_code != 0
    # Typer forces styled output when `GITHUB_ACTIONS` is set, and the styling
    # splits option names apart, so match against the unstyled text.
    assert "--batch-size" in click.unstyle(result.output)


def test_record_rejects_estimate_without_margin_limit(resolved_revisions):
    result = invoke_record(
        "--dataset",
        "wikitext",
        "--batch-size",
        "1",
        "--sequence-length",
        "8",
        "--estimate",
        "failure-rate",
    )

    assert result.exit_code != 0
    assert "--estimate" in click.unstyle(result.output)
    # Rejected before anything is loaded.
    assert resolved_revisions == []


def resolve_stability(
    metric: Metric | None = None,
    *,
    estimate: EstimateChoice = EstimateChoice.Mean,
    relative_failure_threshold: float | None = None,
    absolute_failure_threshold: float | None = None,
    max_relative_margin: float | None = None,
    max_absolute_margin: float | None = 1.0,
    min_runs: int | None = None,
):
    return _resolve_stability(
        metric if metric is not None else Perplexity(),
        estimate=estimate,
        relative_failure_threshold=relative_failure_threshold,
        absolute_failure_threshold=absolute_failure_threshold,
        max_relative_margin=max_relative_margin,
        max_absolute_margin=max_absolute_margin,
        min_runs=min_runs,
    )


def test_no_stability_without_margin_limits():
    assert resolve_stability(max_absolute_margin=None) is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"estimate": EstimateChoice.FailureRate},
        {"relative_failure_threshold": 2.0},
        {"absolute_failure_threshold": 2.0},
    ],
)
def test_estimate_options_require_a_margin_limit(arguments: dict):
    with pytest.raises(typer.BadParameter, match="margin"):
        _ = resolve_stability(max_absolute_margin=None, **arguments)


def test_stability_defaults_to_mean():
    stability = resolve_stability(max_relative_margin=1.0, min_runs=5)

    assert stability == Stability(
        estimator=Mean(),
        min_samples=5,
        max_absolute_margin=1.0,
        max_relative_margin=1.0,
    )


@pytest.mark.parametrize(
    "arguments",
    [{"relative_failure_threshold": 2.0}, {"absolute_failure_threshold": 2.0}],
)
def test_mean_rejects_failure_thresholds(arguments: dict):
    with pytest.raises(typer.BadParameter, match="failure-rate"):
        _ = resolve_stability(estimate=EstimateChoice.Mean, **arguments)


@pytest.mark.parametrize(
    "arguments",
    [{}, {"relative_failure_threshold": 2.0, "absolute_failure_threshold": 2.0}],
)
def test_failure_rate_requires_exactly_one_threshold(arguments: dict):
    with pytest.raises(typer.BadParameter, match="exactly one"):
        _ = resolve_stability(estimate=EstimateChoice.FailureRate, **arguments)


@pytest.mark.parametrize(
    ("metric", "direction"),
    [
        (Perplexity(), ScoreDirection.LowerIsBetter),
        (Accuracy(), ScoreDirection.HigherIsBetter),
    ],
)
def test_failure_rate_direction_comes_from_metric(
    metric: Metric, direction: ScoreDirection
):
    stability = resolve_stability(
        metric, estimate=EstimateChoice.FailureRate, relative_failure_threshold=2.0
    )

    assert stability is not None
    assert stability.estimator == FailureRate(RelativeThreshold(2.0), direction)


def test_failure_rate_absolute_threshold():
    stability = resolve_stability(
        Sdc(), estimate=EstimateChoice.FailureRate, absolute_failure_threshold=5.0
    )

    assert stability is not None
    assert stability.estimator == FailureRate(
        AbsoluteThreshold(5.0), ScoreDirection.LowerIsBetter
    )


def test_failure_rate_rejects_invalid_threshold():
    with pytest.raises(typer.BadParameter, match="factor"):
        _ = resolve_stability(
            estimate=EstimateChoice.FailureRate, relative_failure_threshold=0.0
        )
