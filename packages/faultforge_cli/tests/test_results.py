"""Tests for `faultforge_cli.encoded_memory.results` (the compare/heatmap workflow)."""

from faultforge.experiments.encoded_memory import ReliabilityMetric
from faultforge_cli.encoded_memory.results import (
    bit_position_histogram,
    build_configurations,
    configuration_points,
    discover_result_files,
    load_results,
)

# SECTION discover_result_files


def test_discover_result_files_single_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{}")
    assert discover_result_files(path) == [path]


def test_discover_result_files_recurses_directories(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.json").write_text("{}")
    (tmp_path / "b" / "nested").mkdir(parents=True)
    (tmp_path / "b" / "nested" / "two.json").write_text("{}")

    found = discover_result_files(tmp_path)
    assert found == sorted(found)
    assert {p.name for p in found} == {"one.json", "two.json"}


# SECTION load_results


def test_load_results_flattens_across_paths(tmp_path, save_result):
    save_result(tmp_path / "a" / "result.json", faults=1)
    save_result(tmp_path / "b" / "result.json", faults=2)

    loaded = load_results([tmp_path / "a", tmp_path / "b"])
    assert len(loaded) == 2


def test_load_results_empty_for_nothing_found(tmp_path):
    assert load_results([tmp_path]) == []


def test_load_results_skips_invalid_files_with_a_warning(tmp_path, save_result, caplog):
    save_result(tmp_path / "good.json", faults=1)
    (tmp_path / "not_a_result.json").write_text("this is not a result file")

    with caplog.at_level("WARNING"):
        loaded = load_results([tmp_path])

    assert len(loaded) == 1
    assert loaded[0][0].name == "good.json"
    assert any("skipping" in message for message in caplog.messages)


# SECTION build_configurations


def test_build_configurations_merges_matching_fingerprints_across_paths(
    tmp_path, save_result
):
    for faults, directory in [(1, "a"), (2, "a"), (3, "b")]:
        save_result(tmp_path / directory / f"faults_{faults}.json", faults=faults)

    loaded = load_results([tmp_path])
    configurations = build_configurations(loaded, label_overrides={})

    # All three share every fingerprint scalar except `faults`, so they merge
    # into one configuration regardless of which directory they came from.
    assert len(configurations) == 1
    assert len(configurations[0].results) == 3


def test_build_configurations_separates_differing_configurations(tmp_path, save_result):
    save_result(tmp_path / "a.json", faults=1)
    save_result(tmp_path / "b.json", faults=2, metric=ReliabilityMetric.Sdc)

    loaded = load_results([tmp_path])
    configurations = build_configurations(loaded, label_overrides={})

    assert len(configurations) == 2


def test_build_configurations_label_override_beats_filename_default(
    tmp_path, save_result
):
    path = save_result(tmp_path / "ber_0.05.json", faults=0.05)

    loaded = load_results([tmp_path])
    (default,) = build_configurations(loaded, label_overrides={})
    assert default.label == "ber_0.05"

    (overridden,) = build_configurations(loaded, label_overrides={path: "Identity"})
    assert overridden.label == "Identity"


# SECTION configuration_points


def test_configuration_points_pools_and_sorts_by_rate(tmp_path, make_configuration):
    configuration = make_configuration(tmp_path / "a", label="a", rates=(0.05, 0.01))
    points = configuration_points(configuration, percentile=None)
    rates = [rate for rate, _ in points]
    assert rates == sorted(rates)
    assert len(points) == 2


# SECTION bit_position_histogram


def test_bit_position_histogram_decomposes_positions():
    # bit 0 set once, bit 1 set twice (once alone, once alongside bit 0)
    histogram = bit_position_histogram([0b01, 0b11, 0b10])
    assert histogram == {0: 2, 1: 2}


def test_bit_position_histogram_skip_multi_bit():
    histogram = bit_position_histogram([0b01, 0b11, 0b10], skip_multi_bit=True)
    assert histogram == {0: 1, 1: 1}


def test_bit_position_histogram_empty_for_no_bitmask():
    assert bit_position_histogram([]) == {}
