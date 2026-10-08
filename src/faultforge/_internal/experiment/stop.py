from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from typing import TYPE_CHECKING

from faultforge._internal.experiment.helper import relative_margin_of_error

if TYPE_CHECKING:
    from faultforge._internal.experiment.abc import Experiment

type StopCondition = Callable[[Experiment], str | None]
"""A check run by `Experiment.run_loop` each iteration, before `run`.

Returns a human-readable reason to stop, or `None` to keep going.
"""


@dataclass(slots=True)
class Stability:
    """A `StopCondition`: stop once the mean score's margin of error is small
    relative to the mean."""

    min_samples: int
    """Minimum number of runs before checking the stopping criterion."""
    threshold: float
    """Stop when the relative margin of error (95% margin of error as a percentage of the mean) falls below this value, e.g. 1.0 = 1%."""

    def __call__(self, experiment: Experiment) -> str | None:
        if experiment.run_count() < self.min_samples:
            return None
        relative = relative_margin_of_error(
            experiment.mean_score(), experiment.margin_of_error()
        )
        if relative is not None and relative <= self.threshold:
            return (
                f"Reached stability threshold {self.threshold:.2f}% ({relative:.2f}%)"
            )
        return None


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

    def __call__(self, experiment: Experiment) -> str | None:
        if self._baseline is None:
            self._baseline = experiment.run_count()
        if experiment.run_count() - self._baseline >= self.count:
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

    def __call__(self, experiment: Experiment) -> str | None:
        if experiment.run_count() >= self.total:
            return f"Reached max run count ({self.total})"
        return None
