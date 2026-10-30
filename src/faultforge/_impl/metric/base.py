"""See [`faultforge.metric`]"""

import abc
import enum

from torch import Tensor

from faultforge._impl.experiment.config import ScoreDisplay
from faultforge._impl.fingerprint import Fingerprint


class ScoreDirection(enum.Enum):
    """Which direction of a score is better."""

    HigherIsBetter = enum.auto()
    """Higher scores are better, e.g. accuracy."""
    LowerIsBetter = enum.auto()
    """Lower scores are better, e.g. perplexity or SDC."""


class Metric[R](ScoreDisplay, abc.ABC):
    """A metric for evaluating the output of a DNN.

    In the end all metrics need to produce a numeric [`score`] but can use any
    intermediate representation (the generic type `R`) which is later turned
    into a scalar float by [`score`].

    Metric data (the result type `R`) is created by [`evaluate_batch`] which
    is expected to be called on every iteration over the [batches] of the
    [dataset]. [`evaluate_batch`] has three parameters:
    - `model_output` - The output of a fault-injected [`Module`] for a single
      batch of data.
    - `golden` - The output of a non-faulty [`Module`] for the same batch
      of data, passed through [`preprocess_golden`]. See the [Golden
      outputs](#golden-outputs) section below.
    - `targets` - the `targets` field from the current [`DataBatch`]. The
      expected output of the model as defined by the [dataset].

    [`evaluate_batch`] processes a result for just one batch. The final metric
    is produced by repeated runs of [`accumulate`] which takes two instances of
    the result (`R`) as arguments. The caller is expected to feed the combined
    result as the `existing` argument of [`accumulate`] and the new result as
    the [`new`] argument.

    # Golden outputs

    Computing `golden` may not be required for all metrics so the
    [`requies_golden`] method can be used to signal whether golden outputs
    should be computed. Metrics which don't need golden outputs can ignore the
    `golden` parameter on [`evaluate_batch`].

    In some cases you may want to run a processing step on the golden outputs
    which is the same for every run. In this case the metric can override
    [`preprocess_golden`] to define that processing step. This enables the
    caller to cache the processed golden values so they don't have to be
    recomputed every time [`evaluate_batch`] is called. The caller is always
    expected to pass the golden outputs through [`preprocess_golden`], the
    default implementation just returns the values without altering them.

    The [`GoldenCache`] type exists to simplify common workflows with golden
    outputs.

    [`accumulate`]: Metric.accumulate
    [`evaluate_batch`]: Metric.evaluate_batch
    [`preprocess_golden`]: Metric.preprocess_golden
    [`requires_golden`]: Metric.requires_golden
    [`score`]: Metric.score

    [`DataBatch`]: faultforge.dataset.DataBatch
    [`Module`]: torch.nn.Module
    [batches]: faultforge.dataset.DataBatch
    [dataset]: faultforge.dataset.BatchedDataset
    """

    @abc.abstractmethod
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> R:
        """Create result data for this metric from model output for one dataset batch.

        See the docstring on [`Metric`] for details on the parameters.
        """

    def preprocess_golden(self, golden: Tensor) -> Tensor:
        """Process the golden outputs of the model before they are passed to [`evaluate_batch`].

        See [Metric#golden-outputs] for details.

        [`evaluate_batch`]: Metric.evaluate_batch
        """
        return golden

    @abc.abstractmethod
    def requires_golden(self) -> bool:
        """Whether `evaluate_batch` requires golden inputs.

        The return value of this function assumed to be constant for any
        specific `Metric`.

        This is a signal for the user of the metric of whether to compute golden
        outputs. If this function returns false then the caller may pass an
        empty tensor to [`evaluate_batch`] because it's implied that it will be
        ignored anyway.

        [`evaluate_batch`]: Metric.evaluate_batch
        """

    @abc.abstractmethod
    def accumulate(self, existing: R, new: R) -> R:
        """Combine the result of the current iteration with existing ones.

        - `existing` is the combined result of the already completed iterations
        - `new` is the result of the current iterations
        """

    @abc.abstractmethod
    def score(self, result: R) -> float:
        """Give a numeric "score" for the result."""

    @abc.abstractmethod
    def direction(self) -> ScoreDirection:
        """Whether higher or lower scores of [`score`] are better.

        The return value of this function is assumed to be constant for any
        specific `Metric`.

        [`score`]: Metric.score
        """

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for the metric"""

    def init_or_accumulate(self, existing: R | None, new: R) -> R:
        """Helper for `accumulate` given an initial non-existing result."""
        if existing is None:
            return new
        else:
            return self.accumulate(existing, new)

    def evaluate_and_accumulate(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
        existing_result: R | None,
    ) -> R:
        """Helper for running `evaluate_batch` and `init_or_accumulate` in sequence."""
        batch_result = self.evaluate_batch(
            batch_model_output, batch_golden, batch_targets
        )

        return self.init_or_accumulate(existing_result, batch_result)
