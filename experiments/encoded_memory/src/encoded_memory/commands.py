"""The `encoded-memory` CLI commands."""

import enum
import logging
from pathlib import Path
from typing import Annotated

import torch
import typer

import faultforge.metric as m
from encoded_memory import (
    EncodedFaultInjection,
    discard_bitmasks_in_file,
)
from encoded_memory.stability import StabilityLimits
from faultforge.encoding import (
    CepEncoder,
    CepScheme,
    Encoder,
    EncoderSequence,
    IdentityEncoder,
    MsetEncoder,
    SecdedEncoder,
)
from faultforge.experiment import (
    AbsoluteThreshold,
    AdditionalRuns,
    MaxRuns,
    RelativeThreshold,
    SaveConfig,
    StopCondition,
)
from faultforge.fingerprint import FingerprintError
from faultforge.io import is_compressed
from faultforge.loading import (
    Cifar,
    CifarDataset,
    CifarModel,
    ImageNet,
    ImageNetModel,
    ModelBundle,
    WikiTextBundle,
    resolve_huggingface_revision,
)
from faultforge.metric import Metric
from faultforge.progress import Progress

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
logger = logging.getLogger(__name__)


class DatasetChoice(enum.StrEnum):
    Cifar10 = "cifar10"
    Cifar100 = "cifar100"
    ImageNet = "imagenet"
    WikiText = "wikitext"


class MetricChoice(enum.StrEnum):
    Accuracy = "accuracy"
    AccuracyDegradation = "accuracy-degradation"
    Sdc = "sdc"
    Top1Sdc = "top1-sdc"
    Perplexity = "perplexity"

    def into_metric(self) -> Metric:
        match self:
            case MetricChoice.Accuracy:
                return m.Accuracy()
            case MetricChoice.AccuracyDegradation:
                return m.AccuracyDegradation()
            case MetricChoice.Sdc:
                return m.Sdc()
            case MetricChoice.Top1Sdc:
                return m.Top1Sdc()
            case MetricChoice.Perplexity:
                return m.Perplexity()


def _resolve_metric(
    dataset: DatasetChoice, metric: MetricChoice | None
) -> MetricChoice:
    """Pick the dataset's default metric if none was given and reject unfitting pairs.

    Perplexity needs a language model's per-token logits and next-token
    targets. The other metrics compare top-1 predictions or whole outputs
    against class labels. On WikiText they would count padding and window
    overlap as wrong predictions.
    """
    is_language_dataset = dataset == DatasetChoice.WikiText
    if metric is None:
        return MetricChoice.Perplexity if is_language_dataset else MetricChoice.Accuracy

    is_language_metric = metric == MetricChoice.Perplexity
    if is_language_dataset != is_language_metric:
        raise typer.BadParameter(
            f"metric {metric.value} doesn't apply to dataset {dataset.value}",
            param_hint="--reliability-metric",
        )
    return metric


class EstimateChoice(enum.StrEnum):
    Mean = "mean"
    FailureRate = "failure-rate"


_MaxRunsOption = Annotated[
    int | None,
    typer.Option(
        help="Stop once the results contain N runs in total, including any recorded earlier. Can be combined with the margin limits.",
        rich_help_panel="Stop Conditions",
    ),
]

_MinRunsOption = Annotated[
    int | None,
    typer.Option(
        help="Make sure the results contain at least N runs before the margin limits are checked.",
        rich_help_panel="Stop Conditions",
    ),
]

_MaxRelativeMarginOption = Annotated[
    float | None,
    typer.Option(
        min=0.0,
        help="Run until the estimate's margin of error at 95% confidence is at most N% of the estimated value. "
        "Never reached while the value is 0, e.g. a failure rate without failures. "
        "Can be combined with --max-absolute-margin, whichever is reached first stops.",
        rich_help_panel="Stop Conditions",
    ),
]

_MaxAbsoluteMarginOption = Annotated[
    float | None,
    typer.Option(
        min=0.0,
        help="Run until the estimate's margin of error at 95% confidence is at most N, in the "
        "estimate's units (percentage points for failure-rate). "
        "Can be combined with --max-relative-margin, whichever is reached first stops.",
        rich_help_panel="Stop Conditions",
    ),
]

_EstimateOption = Annotated[
    EstimateChoice,
    typer.Option(
        help="What the margin limits apply to. `mean` is the mean score; it "
        "stops with an error at the first non-finite score, e.g. an infinite perplexity. "
        "`failure-rate` is the percentage of runs whose score is worse than a failure "
        "threshold; non-finite scores always count as failures.",
        rich_help_panel="Stop Conditions",
    ),
]

_RelativeFailureThresholdOption = Annotated[
    float | None,
    typer.Option(
        help="With --estimate failure-rate: a run fails if its score is worse than N times the "
        "golden model's score, e.g. 2.0 for perplexity or 0.5 for accuracy. The golden score "
        "is computed once and saved with the results. Requires a finite, positive golden score.",
        rich_help_panel="Stop Conditions",
    ),
]

_AbsoluteFailureThresholdOption = Annotated[
    float | None,
    typer.Option(
        help="With --estimate failure-rate: a run fails if its score is worse than N, in the "
        "metric's units, e.g. 5.0 for 5% SDC.",
        rich_help_panel="Stop Conditions",
    ),
]


def _resolve_stability_limits(
    *,
    estimate: EstimateChoice,
    relative_failure_threshold: float | None,
    absolute_failure_threshold: float | None,
    max_relative_margin: float | None,
    max_absolute_margin: float | None,
    min_runs: int | None,
) -> StabilityLimits | None:
    """Build the `StabilityLimits` for the margin options, if any are given."""
    if max_relative_margin is None and max_absolute_margin is None:
        for is_set, option in [
            (estimate != EstimateChoice.Mean, "--estimate"),
            (relative_failure_threshold is not None, "--relative-failure-threshold"),
            (absolute_failure_threshold is not None, "--absolute-failure-threshold"),
        ]:
            if is_set:
                raise typer.BadParameter(
                    f"{option} only applies with --max-relative-margin or "
                    "--max-absolute-margin",
                    param_hint=option,
                )
        return None

    failure_threshold: RelativeThreshold | AbsoluteThreshold | None
    match estimate:
        case EstimateChoice.Mean:
            for value, option in [
                (relative_failure_threshold, "--relative-failure-threshold"),
                (absolute_failure_threshold, "--absolute-failure-threshold"),
            ]:
                if value is not None:
                    raise typer.BadParameter(
                        f"{option} only applies to --estimate failure-rate",
                        param_hint=option,
                    )
            failure_threshold = None
        case EstimateChoice.FailureRate:
            try:
                match (relative_failure_threshold, absolute_failure_threshold):
                    case (None, None) | (float(), float()):
                        raise typer.BadParameter(
                            "--estimate failure-rate requires exactly one of "
                            "--relative-failure-threshold or "
                            "--absolute-failure-threshold",
                            param_hint="--estimate",
                        )
                    case (float(), None):
                        failure_threshold = RelativeThreshold(
                            relative_failure_threshold
                        )
                    case (None, float()):
                        failure_threshold = AbsoluteThreshold(
                            absolute_failure_threshold
                        )
            except ValueError as error:
                raise typer.BadParameter(str(error)) from error

    try:
        return StabilityLimits(
            failure_threshold=failure_threshold,
            min_runs=min_runs if min_runs is not None else 0,
            max_absolute_margin=max_absolute_margin,
            max_relative_margin=max_relative_margin,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _init_model_bundle(
    dataset: DatasetChoice,
    model: str | None,
    imagenet_root: str | None,
    model_revision: str | None,
    sequence_length: int | None,
    stride: int | None,
    dtype: torch.dtype,
) -> ModelBundle:
    """Build the `ModelBundle` for the given CLI choices and load the model/dataset from it."""
    if model is None:
        raise typer.BadParameter("A --model must be specified.", param_hint="--model")

    if dataset != DatasetChoice.WikiText:
        for value, option in [
            (model_revision, "--model-revision"),
            (sequence_length, "--sequence-length"),
            (stride, "--stride"),
        ]:
            if value is not None:
                raise typer.BadParameter(
                    f"{option} only applies to --dataset wikitext", param_hint=option
                )

    bundle: ModelBundle
    match dataset:
        case DatasetChoice.Cifar10 | DatasetChoice.Cifar100:
            try:
                cifar_model = CifarModel(model)
            except ValueError as error:
                choices = ", ".join(m.value for m in CifarModel)
                raise typer.BadParameter(
                    f"Unknown model {model!r} for dataset {dataset.value}. Choices: {choices}",
                    param_hint="--model",
                ) from error
            bundle = Cifar(
                model=cifar_model, dataset=CifarDataset(dataset.value), dtype=dtype
            )
        case DatasetChoice.ImageNet:
            if imagenet_root is None:
                raise typer.BadParameter(
                    "--imagenet-root is required when --dataset imagenet.",
                    param_hint="--imagenet-root",
                )
            try:
                imagenet_model = ImageNetModel(model)
            except ValueError as error:
                choices = ", ".join(m.value for m in ImageNetModel)
                raise typer.BadParameter(
                    f"Unknown model {model!r} for dataset imagenet. Choices: {choices}",
                    param_hint="--model",
                ) from error
            bundle = ImageNet(kind=imagenet_model, root=imagenet_root, dtype=dtype)
        case DatasetChoice.WikiText:
            if sequence_length is None:
                raise typer.BadParameter(
                    "--sequence-length is required when --dataset wikitext.",
                    param_hint="--sequence-length",
                )
            requested_revision = model_revision or "main"
            try:
                revision = resolve_huggingface_revision(model, requested_revision)
                if revision != requested_revision:
                    logger.info(
                        f"Resolved revision {requested_revision} of {model} to {revision}"
                    )
                bundle = WikiTextBundle(
                    model_id=model,
                    revision=revision,
                    sequence_length=sequence_length,
                    stride=stride,
                    dtype=dtype,
                )
            except ValueError as error:
                raise typer.BadParameter(str(error)) from error

    return bundle


def _resolve_encoder(
    *, mset: bool, cep: bool, cep_scheme: CepScheme, secded: int | None
) -> Encoder:
    match (mset, cep):
        case (True, False):
            head = MsetEncoder()
        case (False, True):
            head = CepEncoder(cep_scheme)
        case (True, True):
            raise typer.BadParameter("Using MSET and CEP together is not allowed.")
        case (False, False):
            head = None

    match (head, secded):
        case (None, None):
            return IdentityEncoder()
        case (_, None):
            # Asserts to help out the type checker.
            assert head is not None
            return head
        case (None, _):
            assert secded is not None
            return SecdedEncoder(secded)
        case (_, _):
            assert secded is not None
            assert head is not None
            return EncoderSequence([head], SecdedEncoder(secded))


@app.command()
def list_models(
    dataset: Annotated[
        DatasetChoice, typer.Option(help="Which dataset to use")
    ] = DatasetChoice.ImageNet,
) -> None:
    """List all available models for the given dataset."""
    match dataset:
        case DatasetChoice.Cifar10 | DatasetChoice.Cifar100:
            models = [model.value for model in CifarModel]
        case DatasetChoice.ImageNet:
            models = [model.value for model in ImageNetModel]
        case DatasetChoice.WikiText:
            typer.echo(
                "Any causal language model on the HuggingFace Hub, e.g. openai-community/gpt2"
            )
            return

    for model in models:
        typer.echo(model)


@app.command(
    no_args_is_help=True,
)
def record(
    model: Annotated[
        str,
        typer.Option(
            help="Which model to use. Choices depend on the dataset. The the list-models command can be used to see available models.",
            rich_help_panel="Model Setup",
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="The number of samples per batch. Larger batches use more memory "
            "but only run faster until the device is fully utilized.",
            rich_help_panel="Model Setup",
        ),
    ],
    dataset: Annotated[
        DatasetChoice,
        typer.Option(
            help="Which dataset to use",
            rich_help_panel="Model Setup",
        ),
    ] = DatasetChoice.ImageNet,
    imagenet_root: Annotated[
        str | None,
        typer.Option(
            help="Path to a local directory containing ILSVRC2012_devkit_t12.tar.gz "
            "and ILSVRC2012_img_val.tar. Required when --dataset is imagenet.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    model_revision: Annotated[
        str | None,
        typer.Option(
            help="A branch, tag or commit hash of the model on the HuggingFace Hub. "
            "Resolved to a commit hash, which is what gets recorded. "
            "Defaults to main. Only for --dataset wikitext.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    sequence_length: Annotated[
        int | None,
        typer.Option(
            help="The number of tokens per evaluation window. Shouldn't exceed the "
            "model's maximum context length. Memory use grows with it. "
            "Required for --dataset wikitext.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    stride: Annotated[
        int | None,
        typer.Option(
            help="How many tokens each evaluation window moves forward. Must be "
            "smaller than --sequence-length. Defaults to half of it. "
            "Only for --dataset wikitext.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    preload_batches: Annotated[
        bool,
        typer.Option(
            help="Preload all batches into memory before starting the experiment. Otherwise, batches are loaded from disk on-demand. This should be set if there is enough memory as it is much faster.",
            rich_help_panel="Model Setup",
        ),
    ] = True,
    batch_limit: Annotated[
        int | None,
        typer.Option(
            help="Use only the first N batches of the dataset",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    f16: Annotated[
        bool,
        typer.Option(
            help="Load and run the model in float16 (half precision) instead of "
            "float32. Halves the encoded bit-width per parameter, changing what "
            "--faults/--bit-error-rate mean numerically. Some PyTorch CPU "
            "kernels have incomplete float16 support for certain architectures; "
            "if you hit a 'not implemented for Half' error, try a CUDA "
            "--device instead.",
            rich_help_panel="Model Setup",
        ),
    ] = False,
    reliability_metric: Annotated[
        MetricChoice | None,
        typer.Option(
            help="Which metric to use for reliability measurements. Defaults to "
            "perplexity for wikitext and accuracy for the other datasets.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    bit_error_rate: Annotated[
        float | None,
        typer.Option(
            "--bit-error-rate",
            "--ber",
            min=0.0,
            max=1.0,
            help="The bit error rate to use. Mutually exclusive with --faults.",
            rich_help_panel="Fault Injection",
        ),
    ] = None,
    faults: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="The number of faults to inject. Mutually exclusive with --bit-error-rate.",
            rich_help_panel="Fault Injection",
        ),
    ] = None,
    compare_bitwise: Annotated[
        bool,
        typer.Option(
            help="Additionally record a bitwise comparison of each run's faulty "
            "parameters against the golden model's, keeping the nonzero xor "
            "results. Increases the size of recorded results.",
            rich_help_panel="Fault Injection",
        ),
    ] = False,
    fault_summary: Annotated[
        bool,
        typer.Option(
            help="Print a per-run fault-injection summary (bits flipped, BER, and "
            "- if --compare-bitwise is also set - affected/masked counts and a "
            "faulty-bit histogram) after each run.",
            rich_help_panel="Fault Injection",
        ),
    ] = False,
    secded: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Use a SECDED encoding with n data bits. Multiples of 8 perform better.",
            rich_help_panel="Encoding Settings",
        ),
    ] = None,
    mset: Annotated[
        bool,
        typer.Option(
            help="Use an MSET encoding. Chunking is determined by the data type.",
            rich_help_panel="Encoding Settings",
        ),
    ] = False,
    cep: Annotated[
        bool,
        typer.Option(
            help="Use a CEP encoding.",
            rich_help_panel="Encoding Settings",
        ),
    ] = False,
    cep_scheme: Annotated[
        CepScheme,
        typer.Option(
            help="Determines the chunking for CEP encoding. Maps to the number of data bits per parity bit in a chunk.",
            rich_help_panel="Encoding Settings",
        ),
    ] = CepScheme.D3P1,
    golden_is_encoded: Annotated[
        bool,
        typer.Option(
            help="Use the non-faulty encoded model as the golden model. By default the unencoded model is used.",
            rich_help_panel="Encoding Settings",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            help="A file path to save results to.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    autosave: Annotated[
        float | None,
        typer.Option(
            help="Save after N seconds have passed. Ignored if --output is not set. Only saves at the end of the experiment or when interrupted by default.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    compress: Annotated[
        bool,
        typer.Option(
            help="Save --output zstd-compressed. Only controls the format of a "
            "newly created file; if --output already exists, its existing "
            "on-disk format (compressed or not) is kept regardless of this flag.",
            rich_help_panel="Recording Settings",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="If --output already exists but was recorded with a different configuration, discard it and start fresh instead of aborting.",
            rich_help_panel="Recording Settings",
        ),
    ] = False,
    runs: Annotated[
        int | None,
        typer.Option(
            help="Run the experiment N additional times on top of any existing results (e.g. loaded via --output). Incompatible with --min-runs, --max-relative-margin and --max-absolute-margin. Can be combined with --max-runs.",
            rich_help_panel="Stop Conditions",
        ),
    ] = None,
    max_runs: _MaxRunsOption = None,
    min_runs: _MinRunsOption = None,
    max_relative_margin: _MaxRelativeMarginOption = None,
    max_absolute_margin: _MaxAbsoluteMarginOption = None,
    estimate: _EstimateOption = EstimateChoice.Mean,
    relative_failure_threshold: _RelativeFailureThresholdOption = None,
    absolute_failure_threshold: _AbsoluteFailureThresholdOption = None,
    device: Annotated[
        str,
        typer.Option(
            help="Which device to use. PyTorch device string.",
            rich_help_panel="Misc Settings",
        ),
    ] = "cpu",
) -> None:
    """Run an encoded memory fault injection experiment and record the results."""
    metric = _resolve_metric(dataset, reliability_metric).into_metric()
    stability_limits = _resolve_stability_limits(
        estimate=estimate,
        relative_failure_threshold=relative_failure_threshold,
        absolute_failure_threshold=absolute_failure_threshold,
        max_relative_margin=max_relative_margin,
        max_absolute_margin=max_absolute_margin,
        min_runs=min_runs,
    )

    dtype = torch.float16 if f16 else torch.float32
    bundle = _init_model_bundle(
        dataset,
        model,
        imagenet_root,
        model_revision,
        sequence_length,
        stride,
        dtype,
    )

    encoder = _resolve_encoder(
        mset=mset,
        cep=cep,
        cep_scheme=cep_scheme,
        secded=secded,
    )

    if stability_limits is not None and runs is not None:
        raise typer.BadParameter(
            "Cannot specify both --runs and --max-relative-margin/--max-absolute-margin"
        )
    if runs is not None and min_runs is not None:
        raise typer.BadParameter("Cannot specify both --runs and --min-runs")

    match (faults, bit_error_rate):
        case (None, None):
            faults_ = 0
        case (None, _):
            assert bit_error_rate is not None
            # Need to make sure it's a float as only floats are treated as BER.
            faults_ = float(bit_error_rate)
        case (_, None):
            assert faults is not None
            faults_ = faults
        case _:
            raise typer.BadParameter(
                "Only one of --faults or --bit-error-rate can be specified"
            )

    experiment = EncodedFaultInjection(
        bundle,
        encoder,
        reliability_metric=metric,
        golden_is_encoded=golden_is_encoded,
        faults=faults_,
        compare_bitwise=compare_bitwise,
        fault_summary=fault_summary,
        preload_dataset=preload_batches,
        dataset_batch_limit=batch_limit,
        batch_size=batch_size,
        device=device,
        progress=Progress(),
    )
    stop_conditions: list[StopCondition] = []

    save_config: SaveConfig | None = None
    output_exists = False
    if output is not None:
        output = Path(output).expanduser()
        output_exists = output.exists()
        if not output.parent.exists():
            logger.info(f"Creating output parent directory {output.parent}")
            output.parent.mkdir(parents=True)
        else:
            logger.debug(f"Output parent directory {output.parent} already exists")

        effective_compressed = is_compressed(output) if output_exists else compress
        save_config = SaveConfig(
            path=output, interval_seconds=autosave, compressed=effective_compressed
        )

    if stability_limits is not None:
        stop_conditions.append(stability_limits.stability(metric))

    if runs is not None:
        stop_conditions.append(AdditionalRuns(runs))

    if max_runs is not None:
        stop_conditions.append(MaxRuns(max_runs))

    if output is not None and output_exists:
        try:
            experiment.load_from(output)
        except FingerprintError as error:
            if not overwrite:
                logger.error(str(error))
                raise typer.Exit(1) from None
            logger.warning(
                f"{output} was recorded with a different configuration and will be overwritten:\n{error}"
            )

    experiment.run_loop(stop_conditions=stop_conditions, save_config=save_config)


@app.command()
def discard_bitmasks(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to a result file previously recorded with --compare-bitwise."
        ),
    ],
) -> None:
    """Discard any recorded bitmasks from a saved result file, shrinking it.

    Operates directly on the saved file; unlike `record`, this doesn't need to
    reload the model or dataset that produced it.
    """
    discard_bitmasks_in_file(path)
