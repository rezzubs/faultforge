"""Check whether saved results still need more runs."""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

import faultforge.metric as m
from encoded_memory.experiment import (
    EncodedFaultInjectionDisplay,
    SavedResult,
)
from encoded_memory.stability import StabilityLimits
from faultforge import Fingerprint
from faultforge.experiment import (
    Estimate,
    ExperimentDisplay,
    MaxRuns,
    RecordedRuns,
)
from faultforge.io import AnyPath
from faultforge.metric import Metric


class CheckStatus(enum.StrEnum):
    """Whether saved results need more runs, and why."""

    Stable = "stable"
    """The margin limits are reached."""
    MaxRuns = "max-runs"
    """The run count limit is reached."""
    Incomplete = "incomplete"
    """No limit is reached yet."""
    Missing = "missing"
    """There is no result file."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of `check_saved` for one result file."""

    path: Path
    """The checked file."""
    status: CheckStatus
    """Whether the file needs more runs."""
    reason: str | None
    """Why the file got its status, if there's more to say than the status."""
    run_count: int
    """How many runs the file holds."""
    estimate: Estimate | None
    """The estimate the margin limits apply to, if there are any and the runs
    allow one."""

    def should_run(self) -> bool:
        """Whether `record` would add to this file.

        That's usually more runs. A file without the golden score its limits
        need may only get the golden score.
        """
        return self.status in (CheckStatus.Incomplete, CheckStatus.Missing)

    def to_dict(self) -> dict[str, object]:
        """A JSON-compatible view of this result.

        - `path` is a string.
        - `status` is the `CheckStatus` value, e.g. `"max-runs"`.
        - `estimate` has `value`, `lower` and `upper`, or is `None`.
        """
        estimate = self.estimate
        return {
            "path": str(self.path),
            "status": self.status.value,
            "reason": self.reason,
            "run_count": self.run_count,
            "estimate": (
                None
                if estimate is None
                else {
                    "value": float(estimate.value),
                    "lower": float(estimate.lower),
                    "upper": float(estimate.upper),
                }
            ),
        }


def check_saved(
    path: AnyPath,
    *,
    stability: StabilityLimits | None = None,
    max_runs: int | None = None,
) -> CheckResult:
    """Check whether the results at `path` need more runs to reach the limits.

    Answers the question `record` would with the same limits, without loading
    the model or dataset. At least one of `stability` or `max_runs` is
    required.

    - If both limits are reached, the status is `CheckStatus.Stable`.
    - If `stability` needs a golden score but the file has none, the margins
      can't be checked. `record` would compute the golden score first, so the
      file counts as `CheckStatus.Incomplete` unless `max_runs` is reached.

    Raises if the file can't be read, or if the estimate can't be computed,
    e.g. a mean over a non-finite score.
    """
    if stability is None and max_runs is None:
        raise ValueError("check_saved requires stability, max_runs or both")

    path = Path(path).expanduser()
    if not path.exists():
        return CheckResult(
            path=path,
            status=CheckStatus.Missing,
            reason=None,
            run_count=0,
            estimate=None,
        )

    saved = SavedResult.load(path)
    runs = _SavedRuns(saved, _metric_from_fingerprint(saved.fingerprint))
    estimate: Estimate | None = None

    def result(status: CheckStatus, reason: str | None) -> CheckResult:
        return CheckResult(
            path=path,
            status=status,
            reason=reason,
            run_count=runs.run_count(),
            estimate=estimate,
        )

    missing_golden = False
    # NOTE: stability is checked first, so a file that reaches both limits
    # reports `Stable`.
    if stability is not None:
        condition = stability.stability(runs.metric)
        if condition.estimator.requires_golden() and saved.golden_score is None:
            missing_golden = True
        else:
            estimate = runs.estimate(condition.estimator)
            reason = condition(runs)
            if reason is not None:
                return result(CheckStatus.Stable, reason)

    if max_runs is not None and (reason := MaxRuns(max_runs)(runs)) is not None:
        return result(CheckStatus.MaxRuns, reason)

    return result(
        CheckStatus.Incomplete,
        "no golden score recorded" if missing_golden else None,
    )


_KNOWN_METRICS: Sequence[Metric] = (
    m.Accuracy(),
    m.AccuracyDegradation(),
    m.Perplexity(),
    m.Sdc(),
    m.Top1Sdc(),
)


def _metric_from_fingerprint(fingerprint: Fingerprint) -> Metric:
    """The metric that `EncodedFaultInjection` recorded `fingerprint` with."""
    # NOTE: every known metric is fully identified by its fingerprint, so
    # matching it is enough to rebuild the metric.
    match fingerprint.children.get("reliability_metric"):
        case [recorded]:
            pass
        case _:
            raise ValueError("The fingerprint doesn't record a reliability metric")
    for metric in _KNOWN_METRICS:
        if metric.fingerprint() == recorded:
            return metric
    raise ValueError(f"Unknown reliability metric {recorded.kind!r}")


class _SavedRuns(RecordedRuns):
    """The runs of a `SavedResult`, which `metric` scored."""

    def __init__(self, saved: SavedResult, metric: Metric) -> None:
        self.saved = saved
        self.metric = metric

    @override
    def scores(self) -> Sequence[float]:
        return self.saved.scores()

    @override
    def golden_score(self) -> float | None:
        return self.saved.golden_score

    @override
    def display(self) -> ExperimentDisplay:
        return EncodedFaultInjectionDisplay(self.metric)
