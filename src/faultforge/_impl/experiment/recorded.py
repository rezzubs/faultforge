import abc
from collections.abc import Sequence

from faultforge._impl.experiment.config import ExperimentDisplay
from faultforge._impl.experiment.estimate import (
    Estimate,
    Estimator,
)


class RecordedRuns(abc.ABC):
    """A series of scored runs, as seen by stop conditions.

    Subclasses only have to report their scores. This is enough to check
    whether a series needs more runs, e.g. on results loaded from a file.
    [`Experiment`] builds on it to also perform the runs.

    - `scores`: every recorded score, in run order, as plain floats.
    - `golden_score` (optional): the score of a fault-free reference run, for
      estimators that compare runs against it (e.g. `FailureRate` with a
      `RelativeThreshold`).
    - `display` (optional): describe how to format the scores, via
      `ExperimentDisplay`.

    `estimate` summarizes the scores with an `Estimator`, such as `Mean` or
    `FailureRate`.

    [`Experiment`]: faultforge.experiment.Experiment
    """

    @abc.abstractmethod
    def scores(self) -> Sequence[float]:
        """Every score recorded so far, in the order the runs happened."""

    def golden_score(self) -> float | None:
        """The score of a fault-free reference run, or `None` if there is none.

        Only requested when an `Estimator` needs it (see
        `Estimator.requires_golden`), so it may be computed lazily on the first
        call. The default implementation provides no golden score.
        """
        return None

    def estimate(self, estimator: Estimator) -> Estimate | None:
        """Summarize the current scores with `estimator`.

        Requests `golden_score` only if `estimator` requires it. Raises a
        `ValueError` if it's required but there is none.
        """
        golden_score = None
        if estimator.requires_golden():
            golden_score = self.golden_score()
            if golden_score is None:
                raise ValueError(
                    f"{estimator.name()} requires a golden score but "
                    f"{type(self).__name__} doesn't provide one"
                )
        return estimator.estimate(self.scores(), golden_score)

    def display(self) -> ExperimentDisplay:
        """Describes how to render the scores."""
        return ExperimentDisplay()

    def run_count(self) -> int:
        """Return the number of recorded runs."""
        return len(self.scores())
