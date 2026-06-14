"""Classes for running experiments."""

import abc
import dataclasses
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
    min_samples: int
    """Minimum number of runs before checking the stopping criterion."""
    threshold: float
    """Stop when the 95% CI half-width falls below this value (percentage points, e.g. 1.0 = ±1%)."""


@dataclass(slots=True)
class SaveConfig:
    path: AnyPath
    interval_seconds: float | None
    """How many seconds between saves. None means save only at the end."""


@dataclass(slots=True)
class DisplayConfig:
    score_name: str | None = None
    """The name that is given to the result score."""
    score_unit: str | None = None
    """The unit that is printed after the result score."""
    score_fmt: str = "6.2f"
    """The format string used for printing result scores."""


@dataclass
class Experiment[R = float, C = None](abc.ABC):
    """Recoding a series of experiments until a statistically significant result is reached.

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
    display: DisplayConfig = dataclasses.field(default_factory=lambda: DisplayConfig())
    """Configuration for displaying experiment results."""
    stability_config: StabilityConfig | None = None
    """Configuration for stability checking."""
    save_config: SaveConfig | None = None
    """Configuration for saving experiment results."""

    def result_score(self, result: R) -> float:
        """Convert a result to a float score.

        A default implementation is provided for `float` results. For other
        result types this should be overridden.
        """
        if isinstance(result, float):
            return result
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

                ci_half_width = None

                if (
                    self.stability_config is not None
                    and self.run_count() >= self.stability_config.min_samples
                ):
                    ci_half_width = self.ci_half_width()
                    if (
                        ci_half_width is not None
                        and ci_half_width < self.stability_config.threshold
                    ):
                        logger.info("Reached stability threshold, stopping.")
                        break

                self.run()
                print(self.format_status(ci_half_width))
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

    def load_into(self, path: AnyPath) -> None:
        """Overwrite current data from a file.

        The saved fingerprint is verified against the current configuration. A
        mismatch raises `ValueError` describing exactly which parameters differ.

        See `load_into_file` for a version that uses a file-like object.
        """

        with open(Path(path).expanduser(), "r") as f:
            self.load_into_file(f)

    def load_into_file(self, file: IO[str]) -> None:
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

    def ci_half_width(self) -> float | None:
        """Return the confidence interval at 95% for the current set of result scores.

        None if there are less than 2 results.
        """
        scores = [self.result_score(r) for r in self.data.results.values()]
        n = len(scores)
        if n < 2:
            return None
        t = scipy.stats.t.ppf(0.975, df=n - 1)
        return float(t * scipy.stats.sem(scores))

    def format_status(self, ci: float | None) -> str | None:
        """Formats the current status of the experiment as a str.

        None if there are no results yet.
        """
        run_count = self.run_count()
        if run_count == 0:
            return None

        if ci is None:
            ci = self.ci_half_width()

        parts: list[str] = []

        max_runs = self.max_runs()
        if max_runs is not None:
            width = len(str(max_runs))
            completion_percentage = (run_count / max_runs) * 100
            progress = (
                f"[{run_count:>{width}} / {max_runs} | {completion_percentage:6.2f}%]:"
            )
        else:
            progress = f"[Run {run_count}]:"

        parts.append(progress)

        if self.display.score_name is not None:
            parts.append(self.display.score_name)
            parts.append("=")

        latest_key = self.latest_result()
        if latest_key is None:
            return

        last = self.result_score(self.data.results[latest_key])
        parts.append(f"{last:{self.display.score_fmt}}")
        parts.append("-")

        mean = (
            sum(self.result_score(r) for r in self.data.results.values())
            / self.run_count()
        )
        parts.append(f"mean {mean:{self.display.score_fmt}}")

        if ci is not None:
            parts.append(f"±{ci:{self.display.score_fmt}}")

        if self.display.score_unit is not None:
            parts.append(self.display.score_unit)

        return " ".join(parts)

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
