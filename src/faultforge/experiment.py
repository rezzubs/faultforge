"""The `Experiment` base class and related types.

See `Experiment` for the full overview.
"""

from faultforge._impl.experiment.abc import Experiment
from faultforge._impl.experiment.config import (
    ExperimentDisplay,
    SaveConfig,
    ScoreDisplay,
)
from faultforge._impl.experiment.helper import relative_margin_of_error
from faultforge._impl.experiment.stop import (
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
    "ScoreDisplay",
    "Stability",
    "StopCondition",
    "relative_margin_of_error",
]
