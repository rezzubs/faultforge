"""Classes for running experiments.

See `faultforge.experiment` for a general overview.
"""

import abc
import logging
import os
import signal
import tempfile
import time
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import scipy.stats

from faultforge._internal.common import AnyPath

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StabilityConfig:
    """When `Experiment.run_loop` should stop based on the stability of the mean score."""

    min_samples: int
    """Minimum number of runs before checking the stopping criterion."""
    threshold: float
    """Stop when the relative margin of error (the 95% margin of error as a percentage of the mean) falls below this value, e.g. 1.0 = 1%."""


@dataclass(slots=True)
class SaveConfig:
    """Where and how often `Experiment.run_loop` persists progress."""

    path: AnyPath
    """Where to save the experiment's results, via `Experiment.save_atomic`."""
    interval_seconds: float | None
    """How many seconds between saves. None means save only at the end."""


@dataclass(slots=True)
class DisplayConfig:
    """How `Experiment.format_status` renders a score.

    Returned by `Experiment.display`, purely a description of what this
    experiment's score means, computed on demand the same way `fingerprint`
    describes its identity on demand.
    """

    score_name: str = "Score"
    """The name given to the result score."""
    score_unit: str | None = None
    """The unit printed after the result score."""
    score_fmt: str = "6.2f"
    """The format string used for printing result scores."""


class Experiment(abc.ABC):
    """The behavior of a repeatable, scoreable unit of work.

    Subclasses own the shape of their own results (a list, a dict keyed
    however makes sense, ...) and how that state is persisted; this class only
    prescribes what's needed to drive and monitor a series of runs:

    - `run`: perform one iteration and record it internally, any way you like.
    - `scores`: report every recorded score so far, in run order, as plain
      floats. This is the only view the generic machinery below needs, so it
      never has to know your result type.
    - `display`: describe how to format your score for `format_status`.
    - `serialize` / `deserialize`: turn your own state into a string and back.
      That's the only shape-specific part of persistence; `save`/`save_file`/
      `save_atomic`/`load_from`/`load_from_file` handle the file mechanics
      (atomic writes, path expansion, reading a path or an already-open file)
      generically on top of these two. Fingerprinting isn't part of this
      contract either: if you want to verify a loaded file against your
      current configuration, include your own `Fingerprint` in `serialize`'s
      output and check it in `deserialize` via `Fingerprint.raise_if_differs`,
      which raises `FingerprintError` on a mismatch.

    `run_loop` drives the experiment: it calls `run` repeatedly, prints
    progress via `format_status`, and stops once a `StabilityConfig`
    threshold is met, `max_runs` is exhausted, or the user interrupts with
    Ctrl+C. `StabilityConfig`/`SaveConfig` are passed to `run_loop` directly
    rather than stored on the experiment, since they're session-level
    concerns decided by the caller, not part of the experiment's identity.

    See `faultforge.experiments.encoded_memory` for a complete example.
    """

    @abc.abstractmethod
    def run(self) -> None:
        """Run a single iteration and record its result internally."""

    @abc.abstractmethod
    def scores(self) -> Sequence[float]:
        """Every score recorded so far, in the order the runs happened."""

    def display(self) -> DisplayConfig:
        """Describes how `format_status` should render this experiment's score."""
        return DisplayConfig()

    @abc.abstractmethod
    def serialize(self) -> str:
        """Serialize current results to a string, for `save`/`save_atomic`.

        If you want to guard against loading a file saved under a different
        configuration, include your own `Fingerprint` in the output here so
        `deserialize` can check it, e.g. via `Fingerprint.raise_if_differs`.
        """

    @abc.abstractmethod
    def deserialize(self, content: str) -> None:
        """Restore results from a string previously produced by `serialize`."""

    def save(self, path: AnyPath) -> None:
        """Save current results to `path`."""
        with open(Path(path).expanduser(), "w") as f:
            self.save_file(f)

    def save_file(self, file: IO[str]) -> None:
        """Save current results to `file`.

        See `save` for a version that takes a path.
        """
        file.write(self.serialize())

    def save_atomic(self, path: AnyPath) -> None:
        """Save current results to `path`, atomically.

        Will not corrupt existing data if the write fails partway.
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as temp:
            temp.write(self.serialize())
        os.replace(temp.name, Path(path).expanduser())

    def load_from(self, path: AnyPath) -> None:
        """Restore results from `path`, previously written by `save`/`save_atomic`.

        See `load_from_file` for a version that takes a file-like object.
        """
        with open(Path(path).expanduser(), "r") as f:
            self.load_from_file(f)

    def load_from_file(self, file: IO[str]) -> None:
        """Restore results from `file`, previously written by `save_file`.

        See `load_from` for a version that takes a path.
        """
        self.deserialize(file.read())

    def max_runs(self) -> int | None:
        """The number of possible different runs."""
        return None

    def run_count(self) -> int:
        """Return the number of recorded runs."""
        return len(self.scores())

    def run_loop(
        self,
        *,
        stability_config: StabilityConfig | None = None,
        save_config: SaveConfig | None = None,
    ) -> None:
        """Keep running until interrupted, reached stability, or exhausted all possible cases."""

        stop_requested = False
        original_handler = signal.getsignal(signal.SIGINT)

        def _handle_first_interrupt(sig: int, frame: types.FrameType | None) -> None:
            _ = sig, frame
            nonlocal stop_requested
            stop_requested = True
            logger.info(
                "Received Ctrl+C. Finishing current run before stopping. Use Ctrl+C again to force quit."
            )
            _ = signal.signal(signal.SIGINT, original_handler)

        _ = signal.signal(signal.SIGINT, _handle_first_interrupt)

        dirty = False
        passed_seconds = 0.0
        start = time.monotonic()

        try:
            while not stop_requested:
                max_runs = self.max_runs()
                if max_runs is not None and self.run_count() >= max_runs:
                    logger.info("Exhausted all possible cases.")
                    break

                mean, margin_of_error = self._stability_metrics(stability_config)
                relative_margin_of_error = self._relative_margin_of_error(
                    mean, margin_of_error
                )

                if (
                    stability_config is not None
                    and relative_margin_of_error is not None
                    and relative_margin_of_error <= stability_config.threshold
                ):
                    logger.info(
                        f"Reached stability threshold {stability_config.threshold:.2f}% ({relative_margin_of_error:.2f}%), stopping."
                    )
                    break

                self.run()
                # Recomputed rather than reusing `mean`/`margin_of_error` above:
                # the existing stability metrics were computed before
                # `self.run()` and would otherwise be stale by one run.
                print(self.format_status(*self._stability_metrics(stability_config)))
                dirty = True

                if save_config is not None and save_config.interval_seconds is not None:
                    now = time.monotonic()
                    passed_seconds += now - start
                    start = now

                    if save_config.interval_seconds < passed_seconds:
                        logger.debug(
                            f"Passed {save_config.interval_seconds}s ({passed_seconds}) since last save"
                        )
                        self.save_atomic(save_config.path)
                        passed_seconds = 0.0

        finally:
            _ = signal.signal(signal.SIGINT, original_handler)

        if dirty and save_config is not None:
            self.save_atomic(save_config.path)

    def margin_of_error(self) -> float | None:
        """Return the margin of error (half-width of the 95% confidence interval)
        for the mean of the current set of scores.

        None if there are less than 2 results.
        """
        scores = self.scores()
        n = len(scores)
        if n < 2:
            return None
        t = scipy.stats.t.ppf(0.975, df=n - 1)
        return float(t * scipy.stats.sem(scores))

    def mean_score(self) -> float | None:
        """Return the mean of the current set of scores.

        None if there are no results yet.
        """
        scores = self.scores()
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    def _stability_metrics(
        self, stability_config: StabilityConfig | None
    ) -> tuple[float | None, float | None]:
        """Return `(mean_score(), margin_of_error())` once `stability_config.min_samples`
        has been reached; `(None, None)` if stability checking isn't configured
        or there aren't enough runs yet.
        """
        if stability_config is None or self.run_count() < stability_config.min_samples:
            return None, None
        return self.mean_score(), self.margin_of_error()

    @staticmethod
    def _relative_margin_of_error(
        mean: float | None, margin_of_error: float | None
    ) -> float | None:
        """The 95% margin of error as a percentage of the mean.

        `None` if either input is `None`. A mean of exactly `0` would otherwise
        raise `ZeroDivisionError` (a legitimate outcome for e.g. a 0% SDC
        score); that case is treated as 0% relative error when there is no
        error either, and as an undefined (infinite) relative error
        otherwise.
        """
        if mean is None or margin_of_error is None:
            return None
        if mean == 0:
            return 0.0 if margin_of_error == 0 else float("inf")
        return margin_of_error / mean * 100

    def format_status(
        self, mean: float | None, margin_of_error: float | None
    ) -> str | None:
        """Formats the current status of the experiment as a str.

        None if there are no results yet.
        """
        scores = self.scores()
        run_count = len(scores)
        if run_count == 0:
            return None

        if mean is None:
            mean = self.mean_score()
        if margin_of_error is None:
            margin_of_error = self.margin_of_error()

        display = self.display()
        parts: list[str] = []

        max_runs = self.max_runs()
        if max_runs is not None:
            width = len(str(max_runs))
            completion_percentage = (run_count / max_runs) * 100
            progress = (
                f"[{run_count:>{width}} / {max_runs} | {completion_percentage:6.2f}%]: "
            )
        else:
            progress = f"[Run {run_count}]: "

        parts.append(progress)

        if display.score_name is not None:
            parts.append(display.score_name)
            parts.append(" = ")

        parts.append(f"{scores[-1]:{display.score_fmt}}")

        if display.score_unit is not None:
            parts.append(display.score_unit)

        parts.append(" | ")

        if mean is not None:
            parts.append(f"mean {mean:{display.score_fmt}}")

            if display.score_unit is not None:
                parts.append(display.score_unit)

            if margin_of_error is not None:
                parts.append(f" ±{margin_of_error:{display.score_fmt}} (95% CI)")

                relative_margin_of_error = self._relative_margin_of_error(
                    mean, margin_of_error
                )
                if relative_margin_of_error is not None:
                    parts.append(
                        f" | Relative MoE: {relative_margin_of_error:.2f}% of mean"
                    )

        return "".join(parts)
