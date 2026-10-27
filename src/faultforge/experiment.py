"""The `Experiment` base class and related types.

See `Experiment` for the full overview.
"""

from faultforge._impl.experiment.abc import Experiment
from faultforge._impl.experiment.config import (
    ExperimentDisplay,
    SaveConfig,
    ScoreDisplay,
)
from faultforge._impl.experiment.estimate import (
    AbsoluteThreshold,
    Estimate,
    Estimator,
    FailureRate,
    Mean,
    RelativeThreshold,
)
from faultforge._impl.experiment.helper import relative_margin_of_error
from faultforge._impl.experiment.stop import (
    AdditionalRuns,
    MaxRuns,
    Stability,
    StopCondition,
)

__all__ = [
    "AbsoluteThreshold",
    "AdditionalRuns",
    "Estimate",
    "Estimator",
    "Experiment",
    "ExperimentDisplay",
    "FailureRate",
    "MaxRuns",
    "Mean",
    "RelativeThreshold",
    "SaveConfig",
    "ScoreDisplay",
    "Stability",
    "StopCondition",
    "relative_margin_of_error",
]
