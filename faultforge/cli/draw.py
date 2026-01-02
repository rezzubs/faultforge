import enum
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib import patches
from matplotlib.colors import LogNorm

from faultforge.stats import Stats

logger = logging.getLogger(__name__)

app = typer.Typer()


def load_path_sequence_stats(paths: Iterable[Path]) -> list[Stats]:
    stats: list[Stats] = []

    for path in paths:
        path = path.expanduser()

        if path.is_dir():
            stats.extend(load_path_sequence_stats(path.iterdir()))
            continue

        try:
            s = Stats.load(path)
        except Exception as e:
            logger.warning(
                f"Failed to load stats from path `{path}` - skipping\n-> {e}"
            )
            continue

        stats.append(s)

    return stats


@app.command()
def scatter(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="All the paths that contain stats. These paths will be searched recursively."
        ),
    ],
    max_accuracy: Annotated[
        float,
        typer.Option(
            max=100,
            min=0,
            help="Filter the values above this accuracy threshold.",
        ),
    ] = 100,
    min_accuracy: Annotated[
        float,
        typer.Option(
            max=100,
            min=0,
            help="Filter the values above this accuracy threshold.",
        ),
    ] = 0,
    use_3d: Annotated[
        bool,
        typer.Option(
            "--3d",
            help="Draw with a 3d projection",
        ),
    ] = False,
    max_total_faults: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Skip the entries where the total number of faults exceeds this value.",
        ),
    ] = None,
    skip_multi_bit_faults: Annotated[
        bool,
        typer.Option(
            help="Skip the cases where there were multiple faults per parameter.",
        ),
    ] = False,
    inverse_drawing_order: Annotated[
        bool,
        typer.Option(
            help="Draw the cases with the lowest number of faults per bit last",
        ),
    ] = False,
    bins: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Instead of drawing bits hit for an individual runs, aggregate the results over all runs. \
This parameter defines the x-axis resolution",
        ),
    ] = None,
):
    """Draw a scatter plot for all the entries in the given stats.

    The z-axis corresponds to the number of faults in each bit for a given accuracy.
    """
    stats = load_path_sequence_stats(paths)
    entries = [e for d in stats for e in d.entries]

    xs: list[float] = []
    ys: list[int] = []
    zs: list[int] = []

    total_xs: list[float] = []
    total_ys: list[int] = []

    for i, entry in enumerate(entries):
        if entry.accuracy > max_accuracy or entry.accuracy < min_accuracy:
            continue

        fault_map = entry.faults_per_bit_index(skip_multi_bit_faults)
        if fault_map is None:
            logger.debug(f"Faulty parameters not recorded for entry {i}")
            continue

        total_faults = sum(fault_map.values())

        if max_total_faults is not None and total_faults > max_total_faults:
            continue

        total_xs.append(entry.accuracy)
        total_ys.append(total_faults)

        for bit_index, faults_count in fault_map.items():
            if faults_count == 0:
                continue
            xs.append(entry.accuracy)
            ys.append(bit_index)
            zs.append(faults_count)

    xs_ = np.array(xs)
    ys_ = np.array(ys)
    zs_ = np.array(zs)

    if inverse_drawing_order:
        order = np.argsort(-zs_)
    else:
        order = np.argsort(zs_)

    total_xs_ = np.array(total_xs)
    total_ys_ = np.array(total_ys)

    total_order = np.argsort(total_ys_)

    plt.style.use("dark_background")

    fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]

    if use_3d:
        ax = fig.subplots(subplot_kw=dict(projection="3d"))

        mappable = ax.scatter(  # pyright: ignore[reportUnknownMemberType]
            xs_[order], ys_[order], zs_[order], c=zs_[order], cmap="plasma"
        )
        _ = ax.set_xlabel("Accuracy [%]")  # pyright: ignore[reportUnknownMemberType]
        _ = ax.set_ylabel("Bit index")  # pyright: ignore[reportUnknownMemberType]
        ax.set_zlabel("Number of faults")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        _ = plt.colorbar(mappable, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    else:
        ax = fig.subplot_mosaic(
            [["top", "cbar"], ["bot", "cbar"]],
            width_ratios=[0.95, 0.05],
            height_ratios=[1 / 3, 2 / 3],
        )
        ax["top"].sharex(ax["bot"])

        if bins:
            y_min = int(ys_.min())  # pyright: ignore[reportAny]
            y_max = int(ys_.max())  # pyright: ignore[reportAny]
            x_min = int(xs_.min())  # pyright: ignore[reportAny]
            x_max = int(xs_.max())  # pyright: ignore[reportAny]

            bit_index_range = y_max - y_min + 1

            mappable = ax["bot"].hist2d(  # pyright: ignore[reportUnknownMemberType]
                xs_[order],
                ys_[order],
                (bins, bit_index_range),
                range=[[x_min, x_max], [y_min - 0.5, y_max + 0.5]],
                cmap="plasma",
                norm=LogNorm(),
            )[3]
        else:
            mappable = ax["bot"].scatter(  # pyright: ignore[reportUnknownMemberType]
                xs_[order], ys_[order], c=zs_[order], cmap="plasma"
            )
        _ = ax["bot"].set_xlabel("Accuracy [%]")  # pyright: ignore[reportUnknownMemberType]
        _ = ax["bot"].set_ylabel("Bit index")  # pyright: ignore[reportUnknownMemberType]

        _ = ax["top"].scatter(total_xs_[total_order], total_ys_[total_order])  # pyright: ignore[reportUnknownMemberType]
        _ = ax["top"].set_ylabel("Total faults")  # pyright: ignore[reportUnknownMemberType]
        ax["top"].set_yscale("log")  # pyright: ignore[reportUnknownMemberType]

        _ = fig.colorbar(mappable, ax["cbar"])  # pyright: ignore[reportUnknownMemberType]
        _ = ax["cbar"].set_ylabel("Number of faults per bit index")  # pyright: ignore[reportUnknownMemberType]

    fig.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]


@app.command()
def mean(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="Paths which store the files where stats are recorded.",
        ),
    ],
    stability_threshold: Annotated[
        float,
        typer.Option(
            min=0,
            max=100,
            help="The percentage of mean drift that's considered stable",
        ),
    ] = 1,
    stable_within: Annotated[
        int,
        typer.Option(
            min=1,
        ),
    ] = 100,
):
    """Plot the progression of the mean accuracy value over the number of runs."""

    stats = load_path_sequence_stats(paths)

    for s in stats:
        fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]

        _ = ax.set_title(  # pyright: ignore[reportUnknownMemberType]
            "\n".join(
                [f"{k}={v}" for k, v in s.metadata.items()]
                + [f"BER={s.faults_count / s.bits_count:.2e}"]
            )
        )

        means = s.means()

        _ = ax.plot(means)  # pyright: ignore[reportUnknownMemberType]

        result = s.mean_drift(stable_within)
        if result is None:
            plt.show()  # pyright: ignore[reportUnknownMemberType]
            return

        drift_min, drift_max = result

        entries_count = len(s.entries)

        box_start_x = entries_count - stable_within

        drift = drift_max - drift_min

        edge_color = "green" if drift <= stability_threshold else "red"

        rect = patches.Rectangle(
            (box_start_x, drift_min),
            stable_within,
            drift,
            facecolor="none",
            edgecolor=edge_color,
        )

        _ = ax.add_patch(rect)
        _ = ax.text(box_start_x, drift_max, f"drift {drift:.2}%")  # pyright: ignore[reportUnknownMemberType]

        _ = ax.set_xlabel("number of runs")  # pyright: ignore[reportUnknownMemberType]
        _ = ax.set_ylabel("accuracy [%]")  # pyright: ignore[reportUnknownMemberType]

        fig.tight_layout()
        plt.show()  # pyright: ignore[reportUnknownMemberType]


class CompareMode(enum.StrEnum):
    MeanAccuracy = "mean-accuracy"
    MedianAccuracy = "median-accuracy"
    Overhead = "overhead"


MARKERS = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "P",
    "*",
    "h",
    "+",
    "X",
    "D",
]


@app.command()
def configurations(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="Paths which store the files where stats are recorded.",
        ),
    ],
    percentile: Annotated[
        float | None,
        typer.Option(
            help="Draw a nth percentile instead of the mean.",
        ),
    ] = None,
    ignore_key: Annotated[
        list[str] | None,
        typer.Option(
            min=0,
            max=100,
            help="Keys to ignore during printing",
        ),
    ] = None,
    default_label: Annotated[
        str | None,
        typer.Option(
            help="The label to use when all the metadata keys have been ignored.",
        ),
    ] = None,
    logx: Annotated[
        bool,
        typer.Option(
            help="Display the x axis in a log scale.",
        ),
    ] = False,
    custom_legend: Annotated[
        list[str] | None,
        typer.Option(
            help='Use a custom label for a given entry. The format is "existing_label;new_label"',
        ),
    ] = None,
    print_legend: Annotated[
        bool,
        typer.Option(
            help="Print each legend entry. Exists to help with the usage of --custom-legend."
        ),
    ] = False,
):
    """Compare different configurations for recorded stats.

    Entries with the same metadata and bit error rate will be merged.
    """
    stats = load_path_sequence_stats(paths)

    by_metadata: dict[
        tuple[tuple[str, str], ...], tuple[dict[float, list[Stats.Entry]], float]
    ] = dict()

    for s in stats:
        overhead_str = s.metadata.get("memory_overhead", None)
        if overhead_str is not None:
            overhead_float_part = overhead_str.split("%")[0]
            overhead_parsed = float(overhead_float_part)
        else:
            overhead_parsed = 0

        metadata = list(s.metadata.items())
        metadata.sort(key=lambda x: x[0])
        metadata = tuple(metadata)

        by_bit_error_rate, _ = by_metadata.get(metadata, (dict(), 0))

        entries = by_bit_error_rate.get(s.bit_error_rate(), [])
        entries.extend(s.entries)

        by_bit_error_rate[s.bit_error_rate()] = entries
        by_metadata[metadata] = (by_bit_error_rate, overhead_parsed)

    fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]
    fig.set_layout_engine("constrained")  # pyright: ignore[reportUnknownMemberType]
    ax = fig.add_subplot()

    marker_i = 0

    by_metadata_list = list(by_metadata.items())
    # sorting by the overhead
    by_metadata_list.sort(key=lambda x: x[1][1])

    custom_label_map: dict[str, str] = dict()
    for mapping in custom_legend or []:
        try:
            [k, v] = mapping.split(";")
        except Exception as e:
            print(
                f"--custom-legend should be a string split by a single `;`, got `{mapping}`\n-> {e}"
            )
            raise typer.Abort()
        custom_label_map[k] = v

    def map_label(metadata: tuple[tuple[str, str], ...]):
        if ignore_key is not None:
            label_parts = [
                f"{key}={val}" for key, val in metadata if key not in ignore_key
            ]
        else:
            label_parts = [f"{key}={val}" for key, val in metadata]

        label = " ".join(label_parts)
        if label == "" and default_label is not None:
            label = default_label

        if label in custom_label_map:
            label = custom_label_map[label]

        if print_legend:
            print(label)

        return label

    for metadata, (by_bit_error_rate, _) in by_metadata_list:
        label = map_label(metadata)

        sorted = list(by_bit_error_rate.items())
        sorted.sort(key=lambda x: x[0])

        xs = [x for x, _ in sorted]
        if percentile is not None:
            _ = ax.set_ylabel(f"{percentile}th Percentile")  # pyright: ignore[reportUnknownMemberType]
            ys = [
                np.percentile([e.accuracy for e in entries], percentile)
                for _, entries in sorted
            ]
            _ = ax.set_xlabel("Bit Error Rate")  # pyright: ignore[reportUnknownMemberType]
        else:
            _ = ax.set_ylabel("Mean Acuraccy")  # pyright: ignore[reportUnknownMemberType]
            ys = [np.mean([e.accuracy for e in entries]) for _, entries in sorted]

        _ = ax.plot(  # pyright: ignore[reportUnknownMemberType]
            xs,
            ys,
            label=label,
            marker=MARKERS[marker_i],
        )

        if (marker_i := marker_i + 1) == len(MARKERS):
            marker_i = 0

    if logx:
        ax.set_xscale("log")  # pyright: ignore[reportUnknownMemberType]

    _ = fig.legend(  # pyright: ignore[reportUnknownMemberType]
        loc="outside upper left",
        frameon=False,
        ncols=2,
    )

    _ = ax.grid()  # pyright: ignore[reportUnknownMemberType]

    plt.show()  # pyright: ignore[reportUnknownMemberType]
