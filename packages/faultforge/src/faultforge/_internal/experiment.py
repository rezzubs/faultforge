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
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import scipy.stats
from pydantic import BaseModel

from faultforge._internal.common import AnyPath
from faultforge._internal.fingerprint import Fingerprint, format_differences

logger = logging.getLogger(__name__)


class Data[R = float, C = None](BaseModel):
    """The data collected during an experiment.

    This is the part of an experiment that gets saved to disk.
    """

    fingerprint: Fingerprint
    """A structural identity used to verify a loaded file against the current configuration."""
    context: C
    """Additional context separate from results."""
    results: dict[int, R]
    """The results computed thus far.

    Each key should be a unique identifier for that run. For non-deterministic
    experiments the key can be set to the run number.
    """


@dataclass(slots=True)
class StabilityConfig:
    """When `Experiment.run_loop` should stop based on the stability of the mean score."""

    min_samples: int
    """Minimum number of runs before checking the stopping criterion."""
    threshold: float
    """Stop when the relative margin of error (the 95% margin of error as a percentage of the mean) falls below this value, e.g. 1.0 = 1%."""


@dataclass(slots=True)
class SaveConfig:
    """Where and how often `Experiment.run_loop` persists its data."""

    path: AnyPath
    """Where to save the experiment data."""
    interval_seconds: float | None
    """How many seconds between saves. None means save only at the end."""


@dataclass(slots=True)
class DisplayConfig:
    """Controls how `Experiment.format_status` renders progress."""

    score_name: str | None = None
    """The name that is given to the result score."""
    score_unit: str | None = None
    """The unit that is printed after the result score."""
    score_fmt: str = "6.2f"
    """The format string used for printing result scores."""


class Experiment[R = float, C = None](abc.ABC):
    """Record a series of experiments.

    Each run produces a result which is scored by `self.result_score`. The
    `data` attribute will be saved to disk. Other attributes are used for
    configuration.

    The experiment can also have arbitrary context (`C`) associated with it
    which is saved next to the results.

    A concrete experiment is constructed directly from its live components
    (models, encoders, ...). The identity of those components is captured as a
    `Fingerprint` in `data.fingerprint` so that a previously saved file can be
    verified against the current configuration when loaded.

    All fields of `data` need to be (de)serializable by pydantic.
    """

    data: Data[R, C]
    """The data that is saved to disk."""
    display: DisplayConfig
    """Configuration for displaying experiment results."""
    stability_config: StabilityConfig | None
    """Configuration for stability checking."""
    save_config: SaveConfig | None
    """Configuration for saving experiment results."""

    def __init__(
        self,
        data: Data[R, C],
        display: DisplayConfig | None = None,
        stability_config: StabilityConfig | None = None,
        save_config: SaveConfig | None = None,
    ) -> None:
        self.data = data
        self.display = display or DisplayConfig()
        self.stability_config = stability_config
        self.save_config = save_config

    def result_score(self, result: R) -> float:
        """Convert a result to a float score.

        A default implementation is provided for `float` and `int` results. For
        other result types this should be overridden.
        """
        if isinstance(result, (float, int)):
            return float(result)
        raise NotImplementedError(
            f"{type(self)} did not override `score` but has a custom result type {type(result)}"
        )

    @abc.abstractmethod
    def run(self) -> None:
        """Run a single iteration of the experiment."""

    @abc.abstractmethod
    def latest_result(self) -> int | None:
        """Return the id of the latest result."""
        ...

    def max_runs(self) -> int | None:
        """The number of possible different runs."""
        return None

    def run_loop(
        self,
    ) -> None:
        """Keep running the experiment until interupted, reached stability, or exhausted all possible cases."""

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

                mean, margin_of_error = self._stability_metrics()
                relative_margin_of_error = (
                    margin_of_error / mean * 100
                    if mean is not None and margin_of_error is not None
                    else None
                )

                if (
                    self.stability_config is not None
                    and relative_margin_of_error is not None
                    and relative_margin_of_error <= self.stability_config.threshold
                ):
                    logger.info(
                        f"Reached stability threshold {self.stability_config.threshold:.2f}% ({relative_margin_of_error:.2f}%), stopping."
                    )
                    break

                self.run()
                print(self.format_status(mean, margin_of_error))
                dirty = True

                if (
                    self.save_config is not None
                    and self.save_config.interval_seconds is not None
                ):
                    now = time.monotonic()
                    passed_seconds += now - start
                    start = now

                    if self.save_config.interval_seconds < passed_seconds:
                        logger.debug(
                            f"Passed {self.save_config.interval_seconds}s ({passed_seconds}) since last save"
                        )

                        self.save_atomic(self.save_config.path)
                        passed_seconds = 0.0

        finally:
            _ = signal.signal(signal.SIGINT, original_handler)

        if dirty and self.save_config is not None:
            self.save(self.save_config.path)

    def load_from(self, path: AnyPath) -> None:
        """Overwrite current data from a file.

        The saved fingerprint is verified against the current configuration. A
        mismatch raises `ValueError` describing exactly which parameters differ.

        See `load_into_file` for a version that uses a file-like object.
        """

        with open(Path(path).expanduser(), "r") as f:
            self.load_from_file(f)

    def load_from_file(self, file: IO[str]) -> None:
        """Overwrite current data from a file.

        The saved fingerprint is verified against the current configuration. A
        mismatch raises `ValueError` describing exactly which parameters differ.

        See `load_into` for a version that uses a path-like object.
        """
        json = file.read()
        data = type(self.data).model_validate_json(json)
        differences = self.data.fingerprint.diff(data.fingerprint)
        if differences:
            raise ValueError(
                "Saved experiment does not match the current configuration:\n"
                + format_differences(differences)
            )
        self.data = data

    def save(self, path: AnyPath) -> None:
        """Save the experiment data to disk."""
        with open(path, "w") as f:
            self._save_file_helper(f, None)

    def save_file(self, file: IO[str]) -> None:
        """Save the experiment data to disk using an IO object."""
        self._save_file_helper(file, None)

    def save_atomic(self, path: AnyPath) -> None:
        """Save the experiment data to disk atomically.

        Will not corrupt existing data if the write fails partway.
        """

        logger.info(f"Saving experiment data to {path}")

        logger.debug("Running atomic save")

        with tempfile.NamedTemporaryFile("w", delete=False) as temp:
            self._save_file_helper(temp, temp.name)

        logger.debug(f"Moving saved data from {temp.name} to {path}")

        os.replace(temp.name, Path(path).expanduser())

        logger.debug("Move successful")

    def run_count(self) -> int:
        """Return the number recorded runs."""
        return len(self.data.results)

    def margin_of_error(self) -> float | None:
        """Return the margin of error (half-width of the 95% confidence interval)
        for the mean of the current set of result scores.

        None if there are less than 2 results.
        """
        scores = [self.result_score(r) for r in self.data.results.values()]
        n = len(scores)
        if n < 2:
            return None
        t = scipy.stats.t.ppf(0.975, df=n - 1)
        return float(t * scipy.stats.sem(scores))

    def mean_score(self) -> float | None:
        """Return the mean score of the current set of result scores.

        None if there are no results yet.
        """
        scores = [self.result_score(r) for r in self.data.results.values()]
        if not scores:
            return None
        return float(sum(scores) / self.run_count())

    def _stability_metrics(self) -> tuple[float | None, float | None]:
        """Return `(mean_score(), margin_of_error())` once `stability_config.min_samples`
        has been reached; `(None, None)` if stability checking isn't configured
        or there aren't enough runs yet.
        """
        if (
            self.stability_config is None
            or self.run_count() < self.stability_config.min_samples
        ):
            return None, None
        return self.mean_score(), self.margin_of_error()

    def format_status(
        self, mean: float | None, margin_of_error: float | None
    ) -> str | None:
        """Formats the current status of the experiment as a str.

        None if there are no results yet.
        """
        run_count = self.run_count()
        if run_count == 0:
            return None

        if mean is None:
            mean = self.mean_score()
        if margin_of_error is None:
            margin_of_error = self.margin_of_error()

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

        if self.display.score_name is not None:
            parts.append(self.display.score_name)
            parts.append(" = ")

        latest_key = self.latest_result()
        if latest_key is None:
            return None

        last = self.result_score(self.data.results[latest_key])
        parts.append(f"{last:{self.display.score_fmt}}")

        if self.display.score_unit is not None:
            parts.append(self.display.score_unit)

        parts.append(" | ")

        mean = self.mean_score()

        if mean is not None:
            parts.append(f"mean {mean:{self.display.score_fmt}}")

            if self.display.score_unit is not None:
                parts.append(self.display.score_unit)

            if margin_of_error is not None:
                parts.append(
                    f" ±{margin_of_error:{self.display.score_fmt}} (95% CI) | "
                )

                relative_margin_of_error = margin_of_error / mean * 100
                parts.append(f"Relative MoE: {relative_margin_of_error:.2f}% of mean")

        return "".join(parts)

    def _save_file_helper(self, to: IO[str], temp_name: str | None) -> None:
        """Save the experiment data to disk."""
        if temp_name is not None:
            logger.debug(f"Saving to tempfile {temp_name}")
        else:
            logger.info(f"Saving experiment data to {to}")

        json = self.data.model_dump_json()

        written = to.write(json)
        assert written == len(json)

        logger.debug("Save successful")
