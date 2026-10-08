"""The `Experiment` base class and related types.

See `Experiment` for the full overview.
"""

from faultforge._internal.experiment.abc import Experiment
from faultforge._internal.experiment.config import (
    ExperimentDisplay,
    SaveConfig,
)
from faultforge._internal.experiment.helper import relative_margin_of_error
from faultforge._internal.experiment.stop import (
    AdditionalRuns,
    MaxRuns,
    Stability,
    StopCondition,
)

__all__ = [
    "AdditionalRuns",
    "Experiment",
    "ExperimentDisplay",
    "MaxRuns",
    "SaveConfig",
    "Stability",
    "StopCondition",
    "relative_margin_of_error",
]
