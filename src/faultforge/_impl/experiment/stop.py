import abc
import math
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from typing import override

from faultforge._impl.experiment.config import ScoreDisplay
from faultforge._impl.experiment.estimate import (
    CONFIDENCE_LEVEL,
    Estimator,
)
from faultforge._impl.experiment.recorded import RecordedRuns

type StopCondition = Callable[[RecordedRuns], str | None]
"""Decides whether a series of runs should stop.

`Experiment.run_loop` checks it each iteration, before `run`.

Returns a human-readable reason to stop, or `None` to keep going.

A condition can also subclass [`StatusReporter`] to add to the status line.
"""


class StatusReporter(abc.ABC):
    """A base for [`StopCondition`]s which add a fragment to the status line.

    `Experiment.run_loop` shows the fragment after every run.
    """

    @abc.abstractmethod
    def status(self, runs: RecordedRuns, score_display: ScoreDisplay) -> str | None:
        """The fragment to show, or `None` to show nothing.

        `score_display` is the [`ScoreDisplay`] of `runs`.
        """


@dataclass(slots=True, kw_only=True)
class Stability(StatusReporter):
    """A `StopCondition`: stop once an estimate's margin of error is small enough.

    The margin of error is [`Estimate.margin`] of `estimator`'s estimate over
    the recorded scores. It can be limited in absolute terms, relative to
    the estimated value, or both, in which case whichever limit is reached
    first stops the experiment. At least one limit is required. Limits must be
    finite and not negative.

    Nothing stops the experiment while `estimator` has too few scores for an
    estimate. Some estimates never reach a limit, so it can be useful to
    combine `Stability` with [`MaxRuns`]:
    - A relative limit is never reached while the estimated value is 0, e.g.
      a [`FailureRate`] without failures. See [`Estimate.relative_margin`].
    - [`Mean`] raises a `ValueError` when it gets a non-finite score.

    Also shows the estimate and the margins that are checked in the status
    line.

    [`FailureRate`]: faultforge.experiment.FailureRate
    [`Mean`]: faultforge.experiment.Mean
    """

    estimator: Estimator
    """Produces the estimate whose margin is checked."""
    min_samples: int = 0
    """Minimum number of runs before checking the limits."""
    max_absolute_margin: float | None = None
    """Stop when the margin of error is at most this, in the estimate's units."""
    max_relative_margin: float | None = None
    """Stop when the margin of error is at most this percentage of the
    estimated value, e.g. `1.0` means 1%."""

    def __post_init__(self) -> None:
        if self.max_absolute_margin is None and self.max_relative_margin is None:
            raise ValueError(
                "Stability requires max_absolute_margin, max_relative_margin or both"
            )
        for name, limit in (
            ("max_absolute_margin", self.max_absolute_margin),
            ("max_relative_margin", self.max_relative_margin),
        ):
            if limit is not None and not (math.isfinite(limit) and limit >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {limit}")

    def __call__(self, runs: RecordedRuns) -> str | None:
        # NOTE: estimated before checking `min_samples` so that problems such
        # as an unusable golden score are reported before the first run.
        estimate = runs.estimate(self.estimator)
        if runs.run_count() < self.min_samples or estimate is None:
            return None

        margin = estimate.margin()
        if self.max_absolute_margin is not None and margin <= self.max_absolute_margin:
            score_display = runs.display().score_display()
            limit = self._format_value(self.max_absolute_margin, score_display)
            return (
                f"Reached absolute margin of error {limit} "
                f"({self._format_value(margin, score_display)})"
            )

        relative = estimate.relative_margin()
        if (
            self.max_relative_margin is not None
            and relative <= self.max_relative_margin
        ):
            return (
                f"Reached relative margin of error {self.max_relative_margin:.2f}% "
                f"({relative:.2f}%)"
            )

        return None

    @override
    def status(self, runs: RecordedRuns, score_display: ScoreDisplay) -> str | None:
        """Show the estimate and the margins checked against the limits.

        For example `mean 1.00 [0.90, 1.10] (95% CI) | margin 0.10 (limit
        0.05)`. Nothing is shown while there's no estimate.
        """
        estimate = runs.estimate(self.estimator)
        if estimate is None:
            return None

        def format_value(value: float) -> str:
            return self._format_value(value, score_display)

        margins: list[str] = []
        if self.max_absolute_margin is not None:
            margins.append(
                f"{format_value(estimate.margin())} "
                f"(limit {format_value(self.max_absolute_margin)})"
            )
        if self.max_relative_margin is not None:
            margins.append(
                f"{estimate.relative_margin():.2f}% of value "
                f"(limit {self.max_relative_margin:.2f}%)"
            )

        confidence_percent = round(CONFIDENCE_LEVEL * 100)
        return (
            f"{self.estimator.name()} {format_value(estimate.value)} "
            f"[{format_value(estimate.lower)}, {format_value(estimate.upper)}] "
            f"({confidence_percent}% CI) | margin {', '.join(margins)}"
        )

    def _format_value(self, value: float, score_display: ScoreDisplay) -> str:
        # NOTE: score formats may be padded for alignment, which looks odd
        # inside brackets and parentheses.
        return self.estimator.format_value(value, score_display).strip()


@dataclass(slots=True)
class AdditionalRuns:
    """A `StopCondition`: stop after `count` more runs, on top of however many
    already existed when this instance first got checked (typically the start
    of `run_loop`).

    Use `MaxRuns` instead if you want to cap the *total* run count regardless
    of how many results already exist (e.g. loaded from a save file).
    """

    count: int
    _baseline: int | None = field(default=None, init=False, repr=False)

    def __call__(self, runs: RecordedRuns) -> str | None:
        if self._baseline is None:
            self._baseline = runs.run_count()
        if runs.run_count() - self._baseline >= self.count:
            return f"Reached requested additional run count (+{self.count})"
        return None


@dataclass(slots=True)
class MaxRuns:
    """A `StopCondition`: stop once the total run count reaches `total`,
    including any results that already existed before this instance was ever
    checked (e.g. loaded from a save file).

    Use `AdditionalRuns` instead if you want to run a fixed number more
    regardless of how many results already exist.
    """

    total: int

    def __call__(self, runs: RecordedRuns) -> str | None:
        if runs.run_count() >= self.total:
            return f"Reached max run count ({self.total})"
        return None
