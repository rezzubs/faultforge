"""Classes for running experiments.

`Experiment[R, C]` is the base class for a repeatable operation whose results
accumulate until they stabilize. `R` is the type of a single run's result and
`C` is arbitrary context stored alongside it; both must be (de)serializable by
pydantic. To define a new experiment, subclass `Experiment` and implement
`run` (perform one iteration and record its result under `self.data.results`)
and `latest_result` (return the key of the run just recorded). Override
`result_score` if `R` isn't already `float`/`int`.

`run_loop` drives the experiment: it calls `run` repeatedly, prints progress
via `format_status`, and stops once a `StabilityConfig` threshold is met,
`max_runs` is exhausted, or the user interrupts with Ctrl+C.

`Data` is the part of an experiment that gets saved to disk: a
`faultforge.Fingerprint` identifying the configuration that
produced it, the `context`, and the accumulated `results`. Saving is
controlled by `SaveConfig` (periodic autosave during `run_loop`, or manual
`Experiment.save`/`save_atomic`); loading a file back in verifies its
fingerprint against the current configuration and raises if they disagree, so
a mismatched result file is caught rather than silently misused.

`StabilityConfig` decides when `run_loop` should stop based on the relative
margin of error (the margin of error as a percentage of the mean) of the mean
result score. `DisplayConfig` controls how that progress is printed.

See `faultforge.experiments.encoded_memory` for a complete example experiment.
"""

from faultforge._internal.experiment import (
    Data,
    DisplayConfig,
    Experiment,
    SaveConfig,
    StabilityConfig,
)

__all__ = [
    "Data",
    "DisplayConfig",
    "Experiment",
    "SaveConfig",
    "StabilityConfig",
]
