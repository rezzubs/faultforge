"""The `Experiment` base class and related types.

See `Experiment` for the full overview. `RecordedRuns` is the part of it that
stop conditions see.
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
from faultforge._impl.experiment.recorded import RecordedRuns
from faultforge._impl.experiment.stop import (
    AdditionalRuns,
    MaxRuns,
    Stability,
    StatusReporter,
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
    "RecordedRuns",
    "RelativeThreshold",
    "SaveConfig",
    "ScoreDisplay",
    "Stability",
    "StatusReporter",
    "StopCondition",
]
