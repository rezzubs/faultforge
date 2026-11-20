"""Margin of error limits that don't depend on the metric."""

lazy from dataclasses import (
    dataclass,
    field,
)

lazy from faultforge.experiment import (
    AbsoluteThreshold,
    Estimator,
    FailureRate,
    Mean,
    NonFiniteRate,
    RelativeThreshold,
    Stability,
)
lazy from faultforge.metric import Metric


@dataclass(frozen=True, kw_only=True)
class StabilityLimits:
    """Limits on an estimate's margin of error, for any metric.

    Like a `Stability` condition, but without a fixed `Estimator`. Use
    `stability` to build the condition for a metric.

    At least one margin is required. Margins must be finite and not negative.
    """

    estimate: Mean | NonFiniteRate | RelativeThreshold | AbsoluteThreshold = field(
        default_factory=Mean
    )
    """What the limits apply to. A threshold estimates the failure rate against
    it, with the metric's direction."""
    min_runs: int = 0
    """Minimum number of runs before checking the limits."""
    max_absolute_margin: float | None = None
    """See `Stability.max_absolute_margin`."""
    max_relative_margin: float | None = None
    """See `Stability.max_relative_margin`."""

    def __post_init__(self) -> None:
        # NOTE: `Stability` validates the margins on construction and its
        # estimator doesn't affect that.
        _ = self._stability(Mean())

    def _estimator(self, metric: Metric) -> Estimator:
        """The `Estimator` to use for scores of `metric`."""
        match self.estimate:
            case Mean() | NonFiniteRate():
                return self.estimate
            case RelativeThreshold() | AbsoluteThreshold():
                return FailureRate(self.estimate, metric.direction())

    def stability(self, metric: Metric) -> Stability:
        """The `Stability` condition for scores of `metric`."""
        return self._stability(self._estimator(metric))

    def _stability(self, estimator: Estimator) -> Stability:
        return Stability(
            estimator=estimator,
            min_samples=self.min_runs,
            max_absolute_margin=self.max_absolute_margin,
            max_relative_margin=self.max_relative_margin,
        )
