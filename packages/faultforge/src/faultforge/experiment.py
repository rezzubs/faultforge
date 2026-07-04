"""Classes for running experiments.

See `Experiment` for the full overview.
"""

from faultforge._internal.experiment import (
    Experiment,
    ExperimentDisplay,
    RunLimit,
    SaveConfig,
    Stability,
    StopCondition,
    relative_margin_of_error,
)

__all__ = [
    "Experiment",
    "ExperimentDisplay",
    "RunLimit",
    "SaveConfig",
    "Stability",
    "StopCondition",
    "relative_margin_of_error",
]
