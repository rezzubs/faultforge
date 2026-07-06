"""Tests for `faultforge_cli.encoded_memory.plots` (figure building)."""

import pytest
from faultforge import Fingerprint
from faultforge.experiments.encoded_memory import ReliabilityMetric
from faultforge_cli.encoded_memory.plots import (
    GroupBy,
    build_compare_figure,
    build_heatmap_figure,
    group_key,
)
from faultforge_cli.encoded_memory.results import load_results
from matplotlib.figure import Figure

# SECTION group_key


def test_group_key_ungrouped_is_none():
    fingerprint = Fingerprint(kind="k", scalars={"dtype": "f32"})
    assert group_key(GroupBy.Ungrouped, fingerprint) is None


def test_group_key_dtype_and_metric():
    fingerprint = Fingerprint(
        kind="k", scalars={"dtype": "f16", "reliability_metric": "sdc"}
    )
    assert group_key(GroupBy.Dtype, fingerprint) == "f16"
    assert group_key(GroupBy.Metric, fingerprint) == "sdc"


def test_group_key_model_and_dataset():
    fingerprint = Fingerprint(
        kind="k",
        children={
            "bundle": [
                Fingerprint(
                    kind="cifar", scalars={"model": "resnet20", "dataset": "cifar10"}
                )
            ]
        },
    )
    assert group_key(GroupBy.Model, fingerprint) == "resnet20"
    assert group_key(GroupBy.Dataset, fingerprint) == "cifar10"


def test_group_key_dataset_raises_when_absent():
    fingerprint = Fingerprint(
        kind="k",
        children={"bundle": [Fingerprint(kind="imagenet", scalars={"model": "vit"})]},
    )
    with pytest.raises(ValueError, match="dataset"):
        group_key(GroupBy.Dataset, fingerprint)


# SECTION build_compare_figure


def test_build_compare_figure_returns_figure(tmp_path, make_configuration):
    config_a = make_configuration(tmp_path / "a", label="A")
    config_b = make_configuration(tmp_path / "b", label="B")
    fig = build_compare_figure([config_a, config_b])
    assert isinstance(fig, Figure)


def test_build_compare_figure_grid_by_model(tmp_path, make_configuration):
    config_a = make_configuration(tmp_path / "a", label="A", model="resnet18")
    config_b = make_configuration(tmp_path / "b", label="B", model="resnet50")
    fig = build_compare_figure([config_a, config_b], col_by=GroupBy.Model)
    assert isinstance(fig, Figure)


def test_build_compare_figure_raises_on_empty():
    with pytest.raises(ValueError, match="no configurations"):
        build_compare_figure([])


def test_build_compare_figure_raises_on_mismatched_metric(tmp_path, make_configuration):
    sdc = make_configuration(
        tmp_path / "sdc", label="sdc", metric=ReliabilityMetric.Sdc
    )
    accuracy = make_configuration(tmp_path / "acc", label="accuracy")

    with pytest.raises(ValueError, match="reliability metric"):
        build_compare_figure([sdc, accuracy])


# SECTION build_heatmap_figure


def test_build_heatmap_figure_returns_figure(tmp_path, make_configuration):
    configuration = make_configuration(tmp_path / "a", label="a")
    results = [result for _, result in configuration.results]
    fig = build_heatmap_figure(results)
    assert isinstance(fig, Figure)


def test_build_heatmap_figure_raises_on_empty():
    with pytest.raises(ValueError, match="no results"):
        build_heatmap_figure([])


def test_build_heatmap_figure_raises_without_compare_bitwise(tmp_path, save_result):
    save_result(tmp_path / "simple.json", faults=0.05, compare_bitwise=False)
    loaded = load_results([tmp_path])
    results = [result for _, result in loaded]

    with pytest.raises(ValueError, match="compare-bitwise"):
        build_heatmap_figure(results)
