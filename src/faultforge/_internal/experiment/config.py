from collections.abc import Sequence
from dataclasses import dataclass

from faultforge._internal.experiment.helper import relative_margin_of_error
from faultforge._internal.experiment.stop import (
    Stability,
    StopCondition,
)
from faultforge._internal.io import AnyPath


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

    Returned by `Experiment.display`; nothing here is stored on the experiment,
    it's computed on demand. Override any piece to customize; the default
    renders `[Run n]: name = score unit | mean ± moe (95% CI) | Relative MoE: x%
    of mean`, with the `Relative MoE` fragment only shown when a `Stability`
    condition is among the ones currently configured on `run_loop` - it's the
    exact quantity `Stability` checks against its threshold, so it has nothing
    to say if there's no threshold to preview.
    """

    def score_name(self) -> str | None:
        """The name given to the result score, or None to omit it."""
        return None

    def score_unit(self) -> str | None:
        """The unit printed after a score, or None to omit it."""
        return None

    def format_score(self, score: float) -> str:
        """Format a single score value (the latest score, mean, or margin of error)."""
        return f"{score:6.2f}"

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
        """A string to append as-is to the end of the status message, or None
        to omit it. Include any leading separator/spacing yourself."""
        return None

    def format(
        self,
        *,
        run_count: int,
        score: float,
        mean: float | None,
        margin_of_error: float | None,
        stop_conditions: Sequence[StopCondition] = (),
    ) -> str:
        """Compose the full status line from the pieces above.

        `stop_conditions` is whatever's currently configured on `run_loop`
        (both intrinsic and caller-supplied), passed through so a subclass can
        shape its output around what's actually being checked - the default
        implementation uses it only to decide whether to show `Relative MoE`.
        """
        parts: list[str] = [self.progress_label(run_count), ": "]

        def build() -> None:
            score_name = self.score_name()
            if score_name is not None:
                parts.append(score_name)
                parts.append(" = ")

            score_unit = self.score_unit()

            parts.append(self.format_score(score))
            if score_unit is not None:
                parts.append(score_unit)

            if mean is None:
                return
            parts.append(" | ")
            parts.append(f"mean {self.format_score(mean)}")
            if score_unit is not None:
                parts.append(score_unit)

            if margin_of_error is None:
                return
            parts.append(f" ±{self.format_score(margin_of_error)} (95% CI)")

            has_stability = any(
                isinstance(condition, Stability) for condition in stop_conditions
            )
            if not has_stability:
                return

            relative = relative_margin_of_error(mean, margin_of_error)
            if relative is None:
                return
            parts.append(f" | Relative MoE: {relative:.2f}% of mean")

        build()
        if extra := self.extra():
            parts.append(extra)

        return "".join(parts)
