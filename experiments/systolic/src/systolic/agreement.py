"""Matmul-level agreement tooling: run the same fault through every evaluator
on one matrix multiplication and report how their outputs compare.

Not a unit test. Equality is only expected for integer arithmetic, so this
reports similarity, not pass/fail; see the roadmap's oracle/workhorse ladder
and `docs/fault-lifting.md`. Doubles as a tolerance-based regression check
and is cheap enough to run per fault/shape, unlike a full model-level
campaign comparison.
"""

lazy from collections import defaultdict
lazy from collections.abc import Sequence
lazy from dataclasses import dataclass

lazy from torch import Tensor

lazy from systolic._rust import Fault
lazy from systolic.array import Array, ArrayMatmul
lazy from systolic.lifted_evaluator import LiftedEvaluator
lazy from systolic.simulated_evaluator import SimulatedEvaluator


@dataclass(slots=True, frozen=True)
class Evaluation:
    """One way's output for one matrix multiplication."""

    name: str
    output: Tensor


@dataclass(slots=True, frozen=True)
class Agreement:
    """How `other` compares against `baseline` (the oracle) on the same matmul."""

    baseline: str
    other: str
    max_abs_error: float
    max_relative_error: float
    top1_flip_fraction: float
    """Fraction of batch items whose argmax over the out_features axis
    differs between `baseline` and `other`."""


def _compare(baseline: Evaluation, other: Evaluation) -> Agreement:
    diff = (baseline.output - other.output).abs()

    # A fault can saturate an output element to +-inf (or a bit pattern that's
    # literally NaN); when both compute the exact same inf/NaN there
    # they agree, but `inf - inf` and `nan - nan` both evaluate to NaN, which
    # would otherwise read as a divergence (and poison any later reduction,
    # since NaN propagates through `.max()`). Force those elements to zero
    # difference; a real divergence (e.g. one saturates and the other
    # doesn't) still produces a large or infinite value here, which is the
    # signal this comparison exists to catch.
    both_nan = (baseline.output == other.output) | (
        baseline.output.isnan() & other.output.isnan()
    )
    diff = diff.masked_fill(both_nan, 0.0)

    # `diff` is already zeroed at `both_nan` elements, but dividing by
    # `baseline.output`'s magnitude there can still be 0/inf or 0/NaN (NaN in
    # either case) - mask the ratio too rather than just the diff.
    denominator = baseline.output.abs().clamp_min(1e-30)
    relative_error = (diff / denominator).masked_fill(both_nan, 0.0)

    # The output is `([stack,] out_features, batch)`.
    baseline_top1 = baseline.output.argmax(dim=-2)
    other_top1 = other.output.argmax(dim=-2)

    return Agreement(
        baseline=baseline.name,
        other=other.name,
        max_abs_error=diff.max().item(),
        max_relative_error=relative_error.max().item(),
        top1_flip_fraction=(baseline_top1 != other_top1).float().mean().item(),
    )


def compare_matmul(
    array: Array, matmul: ArrayMatmul, fault: Fault | None
) -> list[Agreement]:
    """Run `matmul` fault-free in torch and through every evaluator with
    `fault` in `array`, and report how each one under test (`torch`,
    `lifted`) agrees with `simulated` (the oracle).
    """
    results = [
        Evaluation(name="torch", output=matmul.weights @ matmul.activations),
        Evaluation(
            name="simulated",
            output=SimulatedEvaluator(array).evaluate(matmul, fault),
        ),
        Evaluation(
            name="lifted", output=LiftedEvaluator(array).evaluate(matmul, fault)
        ),
    ]

    oracle = next(result for result in results if result.name == "simulated")
    return [_compare(oracle, result) for result in results if result is not oracle]


@dataclass(slots=True, frozen=True)
class AgreementSummary:
    """The aggregated agreement of one way under test with the oracle.

    Computed across several sampled faults on one weight shape.
    """

    baseline: str
    other: str
    samples: int
    mean_max_abs_error: float
    max_max_abs_error: float
    mean_max_relative_error: float
    max_max_relative_error: float
    mean_top1_flip_fraction: float
    max_top1_flip_fraction: float


def summarize(samples: Sequence[list[Agreement]]) -> list[AgreementSummary]:
    """Aggregate `compare_matmul` agreements from many faults.

    Returns one `AgreementSummary` per way under test.
    """
    by_other: dict[str, list[Agreement]] = defaultdict(list)

    for agreements in samples:
        for agreement in agreements:
            by_other[agreement.other].append(agreement)

    summaries: list[AgreementSummary] = []
    for other, agreements in by_other.items():
        abs_errors = [agreement.max_abs_error for agreement in agreements]
        relative_errors = [agreement.max_relative_error for agreement in agreements]
        flip_fractions = [agreement.top1_flip_fraction for agreement in agreements]

        summaries.append(
            AgreementSummary(
                baseline=agreements[0].baseline,
                other=other,
                samples=len(agreements),
                mean_max_abs_error=sum(abs_errors) / len(abs_errors),
                max_max_abs_error=max(abs_errors),
                mean_max_relative_error=sum(relative_errors) / len(relative_errors),
                max_max_relative_error=max(relative_errors),
                mean_top1_flip_fraction=sum(flip_fractions) / len(flip_fractions),
                max_top1_flip_fraction=max(flip_fractions),
            )
        )

    return summaries
