"""The [`Metric`] ABC and various metric implementations.

A **metric** is something that can evaluate a DNN model's outptput over many
batches and resolve it to a final score. At the core of the metric API is the
[`Metric`] abstract base class which defines the contract that all metrics
follow. See [`Metric`] for details about how to use metrics in general and how
to implement new ones.

Here's the full list of built-in metrics that this module provides:
- [`AccuracyDegradation`]
- [`Accuracy`]
- [`Sdc`]
- [`Top1Sdc`]

In addition to metrics, this module provides a [`GoldenCache`] type. This cache
is meant to simplify working with metrics which rely on outputs from a golden
(non-faulty) model.
"""

from faultforge._internal.metric import (
    Accuracy,
    AccuracyDegradation,
    AccuracyDegradationResult,
    AccuracyResult,
    GoldenCache,
    Metric,
    Sdc,
    SdcResult,
    Top1Sdc,
)

__all__ = [
    "Accuracy",
    "AccuracyDegradation",
    "AccuracyDegradationResult",
    "AccuracyResult",
    "GoldenCache",
    "Metric",
    "Sdc",
    "SdcResult",
    "Top1Sdc",
]
