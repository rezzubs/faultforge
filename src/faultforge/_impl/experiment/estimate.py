import abc
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Final,
    final,
    override,
)

import scipy.stats

from faultforge._impl.experiment.config import ScoreDisplay
from faultforge._impl.metric.base import ScoreDirection

CONFIDENCE_LEVEL: Final = 0.95
"""The confidence level of every [`Estimate`]'s bounds (95%)."""


@final
@dataclass(frozen=True, slots=True)
class Estimate:
    """A value summarizing many scores, with a 95% confidence interval.

    Produced by an [`Estimator`]. The bounds don't need to be symmetric around
    the value, e.g. a failure rate near 0% can't extend below 0%.

    All fields must be finite and `lower <= value <= upper`, otherwise
    construction raises a `ValueError`.
    """

    value: float
    """The estimated value."""
    lower: float
    """The lower bound of the confidence interval."""
    upper: float
    """The upper bound of the confidence interval."""

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(field) for field in (self.value, self.lower, self.upper)
        ):
            raise ValueError(f"every field of an Estimate must be finite, got {self}")
        if not self.lower <= self.value <= self.upper:
            raise ValueError(
                f"an Estimate must satisfy lower <= value <= upper, got {self}"
            )

    def margin(self) -> float:
        """The margin of error: the larger distance from `value` to a bound.

        In the same units as `value`.
        """
        return max(self.value - self.lower, self.upper - self.value)

    def relative_margin(self) -> float:
        """[`margin`] as a percentage of `|value|`, e.g. `1.0` means 1%.

        If `value` is 0 the relative margin is 0% if the margin is also 0,
        otherwise it's infinite.

        [`margin`]: Estimate.margin
        """
        margin = self.margin()
        if self.value == 0:
            return 0.0 if margin == 0 else math.inf
        return margin / abs(self.value) * 100


class Estimator(abc.ABC):
    """Summarizes the scores of many runs as an [`Estimate`].

    Estimators only see plain scores and optionally a golden score, so the
    same estimate can be computed both while running an experiment and later
    from saved results.
    """

    @abc.abstractmethod
    def estimate(
        self, scores: Sequence[float], golden_score: float | None
    ) -> Estimate | None:
        """Summarize `scores`, or `None` if there aren't enough of them yet.

        `golden_score` is the score of a fault-free reference run. The caller
        only needs to provide it if [`requires_golden`] returns true, otherwise
        it may be `None`.

        [`requires_golden`]: Estimator.requires_golden
        """

    @abc.abstractmethod
    def requires_golden(self) -> bool:
        """Whether [`estimate`] needs a golden score.

        The return value of this function is assumed to be constant for any
        specific `Estimator`.

        [`estimate`]: Estimator.estimate
        """

    @abc.abstractmethod
    def name(self) -> str:
        """What is being estimated, for display purposes."""

    def format_value(self, value: float, score_display: ScoreDisplay) -> str:
        """Format a value in the units of this estimator's [`Estimate`]s.

        Used for the value, the bounds and the margin. `score_display`
        describes how the underlying scores are presented, for estimators whose
        estimates share the units of the scores. The default implementation uses
        [`ScoreDisplay.format_score`] directly.
        """
        return score_display.format_score(value)


@final
@dataclass(frozen=True, slots=True)
class Mean(Estimator):
    """The mean of the scores, with a Student's t confidence interval.

    Needs at least 2 scores. Raises a `ValueError` for non-finite scores: the
    mean of scores that can diverge isn't meaningful, use [`FailureRate`]
    instead.

    The interval assumes the mean is approximately normally distributed. This
    holds for many runs, but heavily skewed scores need more runs before the
    interval can be trusted.
    """

    @override
    def estimate(
        self, scores: Sequence[float], golden_score: float | None
    ) -> Estimate | None:
        _ = golden_score

        for run_index, score in enumerate(scores):
            if not math.isfinite(score):
                raise ValueError(
                    f"Mean can't summarize non-finite scores, got {score} for run "
                    f"{run_index + 1}. Use FailureRate for scores that can diverge."
                )

        count = len(scores)
        if count < 2:
            return None

        mean = statistics.fmean(scores)
        standard_error = statistics.stdev(scores, xbar=mean) / math.sqrt(count)
        t_quantile = scipy.stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=count - 1)
        # NOTE: scipy isn't typed. numpy's float64 is a subclass of float.
        assert isinstance(t_quantile, float)

        margin = t_quantile * standard_error
        return Estimate(value=mean, lower=mean - margin, upper=mean + margin)

    @override
    def requires_golden(self) -> bool:
        return False

    @override
    def name(self) -> str:
        return "mean"


@final
@dataclass(frozen=True, slots=True)
class RelativeThreshold:
    """A [`FailureRate`] threshold relative to the golden score.

    The threshold is `factor` times the golden score. The golden score must
    be finite and positive.
    """

    factor: float
    """Multiplied with the golden score to get the threshold.

    Must be finite and positive. For example `2.0` with
    [`ScoreDirection.LowerIsBetter`] means a run fails if its score is more
    than twice the golden score. `0.5` with [`ScoreDirection.HigherIsBetter`]
    means a run fails if its score is less than half of the golden score.

    [`ScoreDirection.LowerIsBetter`]: faultforge.metric.ScoreDirection.LowerIsBetter
    [`ScoreDirection.HigherIsBetter`]: faultforge.metric.ScoreDirection.HigherIsBetter
    """

    def __post_init__(self) -> None:
        if not (math.isfinite(self.factor) and self.factor > 0):
            raise ValueError(f"factor must be finite and positive, got {self.factor}")


@final
@dataclass(frozen=True, slots=True)
class AbsoluteThreshold:
    """A [`FailureRate`] threshold that's a fixed score.

    Doesn't need a golden score.
    """

    score: float
    """The threshold score, in the units of the scores. Must be finite."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")


@final
@dataclass(frozen=True, slots=True)
class FailureRate(Estimator):
    """The percentage of runs that failed, with a Wilson score interval.

    A run fails if its score is worse than the threshold, as defined by
    `direction`. A score equal to the threshold isn't a failure. Non-finite
    scores (`NaN`, `inf` and `-inf`) always count as failures.

    The value and bounds are percentages, so the margin is in percentage
    points. The estimate is available from the first score onward. The
    largest possible margin shrinks with every run, so any margin limit in
    percentage points is eventually reached.

    With a [`RelativeThreshold`] the golden score is required and must be
    finite and positive, otherwise [`estimate`] raises a `ValueError`. Use an
    [`AbsoluteThreshold`] for scores whose golden score is 0, such as SDC.

    [`estimate`]: FailureRate.estimate
    """

    threshold: RelativeThreshold | AbsoluteThreshold
    """Which scores count as failures."""
    direction: ScoreDirection
    """Whether higher or lower scores are better. Usually the
    [`Metric.direction`] of the metric that produced the scores.

    [`Metric.direction`]: faultforge.metric.Metric.direction
    """

    @override
    def estimate(
        self, scores: Sequence[float], golden_score: float | None
    ) -> Estimate | None:
        # NOTE: resolved before checking for scores so an unusable golden score
        # is reported before any runs happen.
        limit = self._limit(golden_score)

        count = len(scores)
        if count == 0:
            return None

        failures = sum(1 for score in scores if self._is_failure(score, limit))

        interval = scipy.stats.binomtest(failures, count).proportion_ci(
            confidence_level=CONFIDENCE_LEVEL, method="wilson"
        )
        lower = interval.low
        upper = interval.high
        # NOTE: scipy isn't typed. numpy's float64 is a subclass of float.
        assert isinstance(lower, float)
        assert isinstance(upper, float)

        value = failures / count * 100
        # Floating point error can put `value` slightly outside its bounds at
        # 0% and 100%.
        return Estimate(
            value=value,
            lower=min(lower * 100, value),
            upper=max(upper * 100, value),
        )

    def _limit(self, golden_score: float | None) -> float:
        match self.threshold:
            case AbsoluteThreshold(score=score):
                return score
            case RelativeThreshold(factor=factor):
                if golden_score is None:
                    raise ValueError(
                        "FailureRate with a RelativeThreshold requires a golden score"
                    )
                if not (math.isfinite(golden_score) and golden_score > 0):
                    raise ValueError(
                        "FailureRate with a RelativeThreshold requires a finite, "
                        f"positive golden score, got {golden_score}. Use an "
                        "AbsoluteThreshold instead."
                    )
                return factor * golden_score

    def _is_failure(self, score: float, limit: float) -> bool:
        if not math.isfinite(score):
            return True
        match self.direction:
            case ScoreDirection.LowerIsBetter:
                return score > limit
            case ScoreDirection.HigherIsBetter:
                return score < limit

    @override
    def requires_golden(self) -> bool:
        return isinstance(self.threshold, RelativeThreshold)

    @override
    def name(self) -> str:
        match self.direction:
            case ScoreDirection.LowerIsBetter:
                comparison = ">"
            case ScoreDirection.HigherIsBetter:
                comparison = "<"

        match self.threshold:
            case RelativeThreshold(factor=factor):
                return f"failure rate ({comparison}{factor:g}x golden)"
            case AbsoluteThreshold(score=score):
                return f"failure rate ({comparison}{score:g})"

    @override
    def format_value(self, value: float, score_display: ScoreDisplay) -> str:
        _ = score_display
        return f"{value:.2f}%"
