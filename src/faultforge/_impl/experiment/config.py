from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from faultforge._impl.io import AnyPath


@dataclass(slots=True)
class SaveConfig:
    """Where and how often `Experiment.run_loop` persists progress."""

    path: AnyPath
    """Where to save the experiment's results, via `Experiment.save_atomic`."""
    interval_seconds: float | None
    """How many seconds between saves. None means save only at the end."""
    compressed: bool = False
    """Whether to save through zstd compression; passed straight through to
    `Experiment.save_atomic`."""


class ExperimentDisplay:
    """Formats an `Experiment`'s status line for `run_loop`.

    Returned by `RecordedRuns.display`; nothing here is stored on the experiment,
    it's computed on demand. Override any piece to customize; the default
    renders `[Run n]: name = score | condition status | ...`, where each
    condition status comes from a stop condition subclassing
    `StatusReporter`, such as `Stability`.
    """

    def score_display(self) -> ScoreDisplay:
        """Return how scores should be presented, if configured.

        Subclasses can narrow the return type, e.g. to a specific [`Metric`].

        [`Metric`]: faultforge.metric.Metric
        """
        return _DefaultScoreDisplay()

    def progress_label(self, run_count: int) -> str:
        """The leading `[...]` progress marker.

        The base case only knows the run count. An experiment that also
        knows a total (e.g. an exhaustive search over a known number of
        cases) should override this using a value it tracks itself, rather
        than have `Experiment` prescribe a "total" concept every subclass
        must carry.
        """
        return f"[Run {run_count}]"

    def extra(self) -> str | None:
        """Add a custom string to the end of `format`.

        Include any leading separator/spacing yourself.

        Enables to add add extra content after the format line without
        overriding the default `format` implementation.
        """
        return None

    def format(
        self,
        *,
        run_count: int,
        score: float,
        condition_status: Sequence[str] = (),
    ) -> str:
        """Compose the full status line from the pieces above.

        `condition_status` holds the fragments of the stop conditions
        currently configured on `run_loop`, in order.
        """
        parts: list[str] = [self.progress_label(run_count), ": "]

        score_display = self.score_display()

        score_name = score_display.display_name()
        if score_name is not None:
            parts.append(score_name)
            parts.append(" = ")

        parts.append(score_display.format_score(score))

        for status in condition_status:
            parts.append(" | ")
            parts.append(status)

        if extra := self.extra():
            parts.append(extra)

        return "".join(parts)


class ScoreDisplay(Protocol):
    """How to present a scalar score.

    All [`Metric`]s implement these methods.

    Every method has a default, so subclasses only need to override what they
    want to customize. Classes that match this protocol structurally, without
    inheriting from it, must implement every method themselves.

    [`Metric`]: faultforge.metric.Metric
    """

    def display_name(self) -> str | None:
        """The name of the score used for display purposes. Override to set"""
        return None

    def display_unit(self) -> str | None:
        """The unit of the score used for display purposes. Override to set"""
        return None

    def format_score(self, score: float) -> str:
        """Format a single score value."""
        return default_score_format(score, self.display_unit())


class _DefaultScoreDisplay(ScoreDisplay):
    """A [`ScoreDisplay`] implementation which uses the protcol defaults for all methods.

    This exists because a Protocol class cannot be instantiated directly.
    """


def default_score_format(score: float, unit: str | None = None) -> str:
    # subjective reasoning: values larger than 1e4 are significantly harder
    # to read. Using powers of 10 improves readablity.
    if score < 1e4:
        # 6 = 4 wide integer part + 2 wide decimal part. (9999.99)
        score_str = f"{score:6.2f}"
    else:
        # match the width and precision of the other case.
        score_str = f"{score:6.2e}"

    return score_str + (unit or "")
