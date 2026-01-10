from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from faultforge.stats import Stats

f32d3 = Stats.load(
    Path(
        "/Users/rezzubs/Documents/work/transformers/2026-01-08/vit_base-f32/dataset-ImageNet_dtype-Float32_embedded_parity-true_memory_overhead-0.0%_model-VitBase_protected-true_ber-3.16e-05.json"
    )
)
f32d7 = Stats.load(
    Path(
        "/Users/rezzubs/Documents/work/transformers/2026-01-08/vit_base-f32-ext/d7p1.json"
    )
)
f32d15 = Stats.load(
    Path(
        "/Users/rezzubs/Documents/work/transformers/2026-01-08/vit_base-f32-ext/d15p1.json"
    )
)

f16d3 = Stats.load(
    Path(
        "/Users/rezzubs/Documents/work/transformers/2026-01-08/vit_base-f16/dataset-ImageNet_dtype-Float16_embedded_parity-true_memory_overhead-0.0%_model-VitBase_protected-true_ber-3.16e-05.json"
    )
)
f16d7 = Stats.load(
    Path(
        "/Users/rezzubs/Documents/work/transformers/2026-01-08/vit_base-f16-ext/d7p1.json"
    )
)
f16d15 = Stats.load(
    Path(
        "/Users/rezzubs/Documents/work/transformers/2026-01-08/vit_base-f16-ext/d15p1.json"
    )
)

f32 = [
    np.mean([e.accuracy for e in entries])
    for entries in [f32d3.entries, f32d7.entries, f32d15.entries]
]

f16 = [
    np.mean([e.accuracy for e in entries])
    for entries in [f16d3.entries, f16d7.entries, f16d15.entries]
]

groups = ["FP32", "FP16"]
accuracy_vs_config = {
    "3 bit": [f32[0], f16[0]],
    "7 bit": [f32[1], f16[1]],
    "15 bit": [f32[2], f16[2]],
}

x = np.arange(len(groups))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

figure_width = 6
aspect_ratio = 2.5
figure_height = figure_width / aspect_ratio

fig, ax = plt.subplots(layout="constrained", figsize=(figure_width, figure_height))

for attribute, accuracy in accuracy_vs_config.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, accuracy, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel("Accuracy [%]")
ax.set_xticks(x + width, groups)
ax.legend(ncols=3)
ax.set_ylim(0, 100)

plt.show()
