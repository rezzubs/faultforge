from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from faultforge.cli.draw import load_path_sequence_stats


def partition(path: str) -> dict[str, list[tuple[float, float]]]:
    output: dict[str, dict[float, float]] = {
        "ep+ecc": {},
        "ep": {},
        "mset+ecc": {},
        "mset": {},
        "ecc": {},
        "unprotected": {},
    }
    for stats in load_path_sequence_stats([Path(path)]):
        md = stats.metadata
        if "embedded_parity" in md and "chunk_size" in md:
            target = "ep+ecc"
        elif "embedded_parity" in md:
            target = "ep"
        elif "msb_duplicated" in md and "chunk_size" in md:
            target = "mset+ecc"
        elif "msb_duplicated" in md:
            target = "mset"
        elif "chunk_size" in md:
            target = "ecc"
        else:
            target = "unprotected"

        ber = stats.bit_error_rate()
        print(ber)
        output[target][ber] = float(np.mean([e.accuracy for e in stats.entries]))

    return {category: sorted(rates.items()) for category, rates in output.items()}


models = [
    ("ViT", "/Users/rezzubs/Documents/work/transformers/2026-01-09_2/vit_base-f32"),
    ("DeiT", "/Users/rezzubs/Documents/work/transformers/2026-01-09_2/deit-base-f32"),
    (
        "Swin (tiny)",
        "/Users/rezzubs/Documents/work/transformers/2026-01-09_2/swin-tiny-f32",
    ),
    (
        "ResNet152",
        "/Users/rezzubs/Documents/work/transformers/2026-01-09_2/resnet152-f32",
    ),
    (
        "MobileNet",
        "/Users/rezzubs/Documents/work/transformers/2026-01-09_2/mobilenet-f32",
    ),
    (
        "Inception",
        "/Users/rezzubs/Documents/work/transformers/2026-01-09_2/inception-f32",
    ),
]

cmap = plt.get_cmap("tab10")
labels = {
    "unprotected": ("Unprotected", cmap(5), "<"),
    "ecc": ("ECC", cmap(4), ">"),
    "mset+ecc": ("MSET+ECC", cmap(2), "^"),
    "ep+ecc": ("EP+ECC", cmap(0), "o"),
    "mset": ("MSET", cmap(3), "s"),
    "ep": ("EP", cmap(1), "v"),
}


fig, axes = plt.subplots(
    2,
    3,
    sharey=True,
    sharex=True,
)

for i, ((model_name, path), ax) in enumerate(zip(models, axes.flatten())):
    assert isinstance(ax, Axes)
    configs = partition(path)

    for config_name, error_rates in configs.items():
        _, color, marker = labels[config_name]
        ber = list(x[0] for x in error_rates)
        accuracy = list(x[1] for x in error_rates)

        ax.plot(ber, accuracy, color=color, marker=marker)

    ax.set_title(model_name)
    ax.set_xscale("log")

    if i in [0, 3]:
        ax.set_ylabel("Accuracy [%]")

    if i in [3, 4, 5]:
        ax.set_xlabel("BER")

fig.subplots_adjust(top=0.82, right=0.98, left=0.10, bottom=0.10)
fig.legend(
    handles=[
        plt.Line2D([0], [0], color=c, marker=m, lw=2) for _, c, m in labels.values()
    ],
    labels=[l[0] for l in labels.values()],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.0),
)


plt.show()
