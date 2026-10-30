"""The `encoded-memory` CLI commands."""

import enum
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch

import faultforge.metric as m
from encoded_memory import (
    EncodedFaultInjection,
    StabilityLimits,
    check_saved,
    discard_bitmasks_in_file,
)
from encoded_memory.args import (
    App,
    Group,
    Option,
    Positional,
    UsageError,
)
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

app = App(prog="encoded-memory")
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
        raise UsageError(
            f"metric {metric.value} doesn't apply to dataset {dataset.value}",
            option="--reliability-metric",
        )
    return metric


class EstimateChoice(enum.StrEnum):
    Mean = "mean"
    FailureRate = "failure-rate"


@dataclass(frozen=True, kw_only=True)
class StopConditionOptions:
    """The stop conditions shared by `record` and `check`."""

    max_runs: Annotated[
        int | None,
        Option(
            help="Stop once the results contain N runs in total, including any recorded earlier. Can be combined with the margin limits.",
            metavar="N",
        ),
    ] = None
    min_runs: Annotated[
        int | None,
        Option(
            help="Make sure the results contain at least N runs before the margin limits are checked.",
            metavar="N",
        ),
    ] = None
    max_relative_margin: Annotated[
        float | None,
        Option(
            help="Run until the estimate's margin of error at 95% confidence is at most N% of the estimated value. "
            "Never reached while the value is 0, e.g. a failure rate without failures. "
            "Can be combined with --max-absolute-margin, whichever is reached first stops.",
            metavar="N",
            minimum=0.0,
        ),
    ] = None
    max_absolute_margin: Annotated[
        float | None,
        Option(
            help="Run until the estimate's margin of error at 95% confidence is at most N, in the "
            "estimate's units (percentage points for failure-rate). "
            "Can be combined with --max-relative-margin, whichever is reached first stops.",
            metavar="N",
            minimum=0.0,
        ),
    ] = None
    estimate: Annotated[
        EstimateChoice,
        Option(
            help="What the margin limits apply to. `mean` is the mean score; it "
            "stops with an error at the first non-finite score, e.g. an infinite perplexity. "
            "`failure-rate` is the percentage of runs whose score is worse than a failure "
            "threshold; non-finite scores always count as failures.",
        ),
    ] = EstimateChoice.Mean
    relative_failure_threshold: Annotated[
        float | None,
        Option(
            help="With --estimate failure-rate: a run fails if its score is worse than N times the "
            "golden model's score, e.g. 2.0 for perplexity or 0.5 for accuracy. The golden score "
            "is computed once and saved with the results. Requires a finite, positive golden score.",
            metavar="N",
        ),
    ] = None
    absolute_failure_threshold: Annotated[
        float | None,
        Option(
            help="With --estimate failure-rate: a run fails if its score is worse than N, in the "
            "metric's units, e.g. 5.0 for 5% SDC.",
            metavar="N",
        ),
    ] = None


def _resolve_stability_limits(options: StopConditionOptions) -> StabilityLimits | None:
    """Build the `StabilityLimits` for the margin options, if any are given."""
    if options.max_relative_margin is None and options.max_absolute_margin is None:
        for is_set, option in [
            (options.estimate != EstimateChoice.Mean, "--estimate"),
            (
                options.relative_failure_threshold is not None,
                "--relative-failure-threshold",
            ),
            (
                options.absolute_failure_threshold is not None,
                "--absolute-failure-threshold",
            ),
        ]:
            if is_set:
                raise UsageError(
                    "only applies with --max-relative-margin or --max-absolute-margin",
                    option=option,
                )
        return None

    failure_threshold: RelativeThreshold | AbsoluteThreshold | None
    match options.estimate:
        case EstimateChoice.Mean:
            for value, option in [
                (options.relative_failure_threshold, "--relative-failure-threshold"),
                (options.absolute_failure_threshold, "--absolute-failure-threshold"),
            ]:
                if value is not None:
                    raise UsageError(
                        "only applies to --estimate failure-rate", option=option
                    )
            failure_threshold = None
        case EstimateChoice.FailureRate:
            try:
                match (
                    options.relative_failure_threshold,
                    options.absolute_failure_threshold,
                ):
                    case (None, None) | (float(), float()):
                        raise UsageError(
                            "failure-rate requires exactly one of "
                            "--relative-failure-threshold or "
                            "--absolute-failure-threshold",
                            option="--estimate",
                        )
                    case (float() as threshold, None):
                        failure_threshold = RelativeThreshold(threshold)
                    case (None, float() as threshold):
                        failure_threshold = AbsoluteThreshold(threshold)
            except ValueError as error:
                raise UsageError(str(error)) from error

    try:
        return StabilityLimits(
            failure_threshold=failure_threshold,
            min_runs=options.min_runs if options.min_runs is not None else 0,
            max_absolute_margin=options.max_absolute_margin,
            max_relative_margin=options.max_relative_margin,
        )
    except ValueError as error:
        raise UsageError(str(error)) from error


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
        raise UsageError("required", option="--model")

    if dataset != DatasetChoice.WikiText:
        for value, option in [
            (model_revision, "--model-revision"),
            (sequence_length, "--sequence-length"),
            (stride, "--stride"),
        ]:
            if value is not None:
                raise UsageError("only applies to --dataset wikitext", option=option)

    bundle: ModelBundle
    match dataset:
        case DatasetChoice.Cifar10 | DatasetChoice.Cifar100:
            try:
                cifar_model = CifarModel(model)
            except ValueError as error:
                choices = ", ".join(m.value for m in CifarModel)
                raise UsageError(
                    f"Unknown model {model!r} for dataset {dataset.value}. Choices: {choices}",
                    option="--model",
                ) from error
            bundle = Cifar(
                model=cifar_model, dataset=CifarDataset(dataset.value), dtype=dtype
            )
        case DatasetChoice.ImageNet:
            if imagenet_root is None:
                raise UsageError(
                    "required when --dataset imagenet",
                    option="--imagenet-root",
                )
            try:
                imagenet_model = ImageNetModel(model)
            except ValueError as error:
                choices = ", ".join(m.value for m in ImageNetModel)
                raise UsageError(
                    f"Unknown model {model!r} for dataset imagenet. Choices: {choices}",
                    option="--model",
                ) from error
            bundle = ImageNet(kind=imagenet_model, root=imagenet_root, dtype=dtype)
        case DatasetChoice.WikiText:
            if sequence_length is None:
                raise UsageError(
                    "required when --dataset wikitext",
                    option="--sequence-length",
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
                raise UsageError(str(error)) from error

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
            raise UsageError("Using MSET and CEP together is not allowed.")
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


@dataclass(frozen=True, kw_only=True)
class ListModelsOptions:
    dataset: Annotated[DatasetChoice, Option(help="Which dataset to use.")] = (
        DatasetChoice.ImageNet
    )


@app.command
def list_models(options: ListModelsOptions) -> None:
    """List all available models for the given dataset."""
    match options.dataset:
        case DatasetChoice.Cifar10 | DatasetChoice.Cifar100:
            models = [model.value for model in CifarModel]
        case DatasetChoice.ImageNet:
            models = [model.value for model in ImageNetModel]
        case DatasetChoice.WikiText:
            print(
                "Any causal language model on the HuggingFace Hub, e.g. openai-community/gpt2"
            )
            return

    for model in models:
        print(model)


@dataclass(frozen=True, kw_only=True)
class ModelSetupOptions:
    model: Annotated[
        str,
        Option(
            help="Which model to use. Choices depend on the dataset. The the list-models command can be used to see available models.",
        ),
    ]
    batch_size: Annotated[
        int,
        Option(
            help="The number of samples per batch. Larger batches use more memory "
            "but only run faster until the device is fully utilized.",
            minimum=1,
        ),
    ]
    dataset: Annotated[DatasetChoice, Option(help="Which dataset to use.")] = (
        DatasetChoice.ImageNet
    )
    imagenet_root: Annotated[
        str | None,
        Option(
            help="Path to a local directory containing ILSVRC2012_devkit_t12.tar.gz "
            "and ILSVRC2012_img_val.tar. Required when --dataset is imagenet.",
        ),
    ] = None
    model_revision: Annotated[
        str | None,
        Option(
            help="A branch, tag or commit hash of the model on the HuggingFace Hub. "
            "Resolved to a commit hash, which is what gets recorded. "
            "Defaults to main. Only for --dataset wikitext.",
        ),
    ] = None
    sequence_length: Annotated[
        int | None,
        Option(
            help="The number of tokens per evaluation window. Shouldn't exceed the "
            "model's maximum context length. Memory use grows with it. "
            "Required for --dataset wikitext.",
        ),
    ] = None
    stride: Annotated[
        int | None,
        Option(
            help="How many tokens each evaluation window moves forward. Must be "
            "smaller than --sequence-length. Defaults to half of it. "
            "Only for --dataset wikitext.",
        ),
    ] = None
    preload_batches: Annotated[
        bool,
        Option(
            help="Preload all batches into memory before starting the experiment. Otherwise, batches are loaded from disk on-demand. This should be set if there is enough memory as it is much faster.",
        ),
    ] = True
    batch_limit: Annotated[
        int | None,
        Option(help="Use only the first N batches of the dataset.", metavar="N"),
    ] = None
    f16: Annotated[
        bool,
        Option(
            help="Load and run the model in float16 (half precision) instead of "
            "float32. Halves the encoded bit-width per parameter, changing what "
            "--faults/--bit-error-rate mean numerically. Some PyTorch CPU "
            "kernels have incomplete float16 support for certain architectures; "
            "if you hit a 'not implemented for Half' error, try a CUDA "
            "--device instead.",
        ),
    ] = False
    reliability_metric: Annotated[
        MetricChoice | None,
        Option(
            help="Which metric to use for reliability measurements. Defaults to "
            "perplexity for wikitext and accuracy for the other datasets.",
        ),
    ] = None


@dataclass(frozen=True, kw_only=True)
class FaultInjectionOptions:
    bit_error_rate: Annotated[
        float | None,
        Option(
            help="The bit error rate to use. Mutually exclusive with --faults.",
            aliases=("--ber",),
            metavar="RATE",
            minimum=0.0,
            maximum=1.0,
        ),
    ] = None
    faults: Annotated[
        int | None,
        Option(
            help="The number of faults to inject. Mutually exclusive with --bit-error-rate.",
            metavar="N",
            minimum=0,
        ),
    ] = None
    compare_bitwise: Annotated[
        bool,
        Option(
            help="Additionally record a bitwise comparison of each run's faulty "
            "parameters against the golden model's, keeping the nonzero xor "
            "results. Increases the size of recorded results.",
        ),
    ] = False
    fault_summary: Annotated[
        bool,
        Option(
            help="Print a per-run fault-injection summary (bits flipped, BER, and "
            "- if --compare-bitwise is also set - affected/masked counts and a "
            "faulty-bit histogram) after each run.",
        ),
    ] = False


@dataclass(frozen=True, kw_only=True)
class EncodingOptions:
    secded: Annotated[
        int | None,
        Option(
            help="Use a SECDED encoding with N data bits. Multiples of 8 perform better.",
            metavar="N",
            minimum=1,
        ),
    ] = None
    mset: Annotated[
        bool,
        Option(help="Use an MSET encoding. Chunking is determined by the data type."),
    ] = False
    cep: Annotated[bool, Option(help="Use a CEP encoding.")] = False
    cep_scheme: Annotated[
        CepScheme,
        Option(
            help="Determines the chunking for CEP encoding. Maps to the number of data bits per parity bit in a chunk.",
        ),
    ] = CepScheme.D3P1
    golden_is_encoded: Annotated[
        bool,
        Option(
            help="Use the non-faulty encoded model as the golden model. By default the unencoded model is used.",
        ),
    ] = False


@dataclass(frozen=True, kw_only=True)
class RecordingOptions:
    output: Annotated[
        Path | None, Option(help="A file path to save results to.", metavar="PATH")
    ] = None
    autosave: Annotated[
        float | None,
        Option(
            help="Save after N seconds have passed. Ignored if --output is not set. Only saves at the end of the experiment or when interrupted by default.",
            metavar="N",
        ),
    ] = None
    compress: Annotated[
        bool,
        Option(
            help="Save --output zstd-compressed. Only controls the format of a "
            "newly created file; if --output already exists, its existing "
            "on-disk format (compressed or not) is kept regardless of this flag.",
        ),
    ] = False
    overwrite: Annotated[
        bool,
        Option(
            help="If --output already exists but was recorded with a different configuration, discard it and start fresh instead of aborting.",
        ),
    ] = False


@dataclass(frozen=True, kw_only=True)
class RecordStopOptions:
    runs: Annotated[
        int | None,
        Option(
            help="Run the experiment N additional times on top of any existing results (e.g. loaded via --output). Incompatible with --min-runs, --max-relative-margin and --max-absolute-margin. Can be combined with --max-runs.",
            metavar="N",
        ),
    ] = None
    limits: StopConditionOptions


@dataclass(frozen=True, kw_only=True)
class RecordOptions:
    model_setup: Annotated[ModelSetupOptions, Group("Model Setup")]
    fault_injection: Annotated[FaultInjectionOptions, Group("Fault Injection")]
    encoding: Annotated[EncodingOptions, Group("Encoding Settings")]
    recording: Annotated[RecordingOptions, Group("Recording Settings")]
    stop: Annotated[RecordStopOptions, Group("Stop Conditions")]
    device: Annotated[
        str, Option(help="Which device to use. PyTorch device string.")
    ] = "cpu"


@app.command
def record(options: RecordOptions) -> int | None:
    """Run an encoded memory fault injection experiment and record the results."""
    setup = options.model_setup
    injection = options.fault_injection
    encoding = options.encoding
    recording = options.recording
    stop = options.stop

    metric = _resolve_metric(setup.dataset, setup.reliability_metric).into_metric()
    stability_limits = _resolve_stability_limits(stop.limits)

    dtype = torch.float16 if setup.f16 else torch.float32
    bundle = _init_model_bundle(
        setup.dataset,
        setup.model,
        setup.imagenet_root,
        setup.model_revision,
        setup.sequence_length,
        setup.stride,
        dtype,
    )

    encoder = _resolve_encoder(
        mset=encoding.mset,
        cep=encoding.cep,
        cep_scheme=encoding.cep_scheme,
        secded=encoding.secded,
    )

    if stability_limits is not None and stop.runs is not None:
        raise UsageError(
            "Cannot specify both --runs and --max-relative-margin/--max-absolute-margin"
        )
    if stop.runs is not None and stop.limits.min_runs is not None:
        raise UsageError("Cannot specify both --runs and --min-runs")

    match (injection.faults, injection.bit_error_rate):
        case (None, None):
            faults_ = 0
        case (None, bit_error_rate):
            # Need to make sure it's a float as only floats are treated as BER.
            faults_ = float(bit_error_rate)
        case (faults, None):
            faults_ = faults
        case _:
            raise UsageError(
                "Only one of --faults or --bit-error-rate can be specified"
            )

    experiment = EncodedFaultInjection(
        bundle,
        encoder,
        reliability_metric=metric,
        golden_is_encoded=encoding.golden_is_encoded,
        faults=faults_,
        compare_bitwise=injection.compare_bitwise,
        fault_summary=injection.fault_summary,
        preload_dataset=setup.preload_batches,
        dataset_batch_limit=setup.batch_limit,
        batch_size=setup.batch_size,
        device=options.device,
        progress=Progress(),
    )
    stop_conditions: list[StopCondition] = []

    output = recording.output
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

        effective_compressed = (
            is_compressed(output) if output_exists else recording.compress
        )
        save_config = SaveConfig(
            path=output,
            interval_seconds=recording.autosave,
            compressed=effective_compressed,
        )

    if stability_limits is not None:
        stop_conditions.append(stability_limits.stability(metric))

    if stop.runs is not None:
        stop_conditions.append(AdditionalRuns(stop.runs))

    if stop.limits.max_runs is not None:
        stop_conditions.append(MaxRuns(stop.limits.max_runs))

    if output is not None and output_exists:
        try:
            experiment.load_from(output)
        except FingerprintError as error:
            if not recording.overwrite:
                logger.error(str(error))
                return 1
            logger.warning(
                f"{output} was recorded with a different configuration and will be overwritten:\n{error}"
            )

    experiment.run_loop(stop_conditions=stop_conditions, save_config=save_config)
    return None


_RESULT_SUFFIXES = (".json", ".json.zst", ".json.zstd")
"""File name endings of the results `record --output` writes."""


def _expand_directories(paths: list[Path]) -> list[Path]:
    """Replace each directory in `paths` with the result files below it, sorted."""
    expanded: list[Path] = []
    for path in paths:
        if not path.expanduser().is_dir():
            expanded.append(path)
            continue
        # NOTE: `rglob` doesn't follow symlinked directories, so a link cycle
        # can't make this loop forever.
        expanded.extend(
            sorted(
                found
                for found in path.expanduser().rglob("*")
                if found.name.endswith(_RESULT_SUFFIXES) and found.is_file()
            )
        )
    return expanded


class _CheckExitCode(enum.IntEnum):
    """The exit codes of `check`, from best to worst."""

    Done = 0
    ShouldRun = 1
    # NOTE: 2 is argparse's exit code for bad arguments.
    Error = 3


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    paths: Annotated[
        list[Path],
        Positional(
            help="Result files to check, as written by record --output. "
            "A directory includes every .json, .json.zst and .json.zstd file below it. "
            "A file given directly that doesn't exist is reported as missing.",
            metavar="PATH",
        ),
    ]
    stop: Annotated[StopConditionOptions, Group("Stop Conditions")]


@app.command
def check(options: CheckOptions) -> int:
    """Check whether saved result files need more runs to reach the limits.

    Takes the same stop conditions as record. Doesn't load the model or
    dataset. A directory is searched recursively for result files.

    Prints a JSON list to stdout, with one object per path. The status is one
    of stable, max-runs, incomplete, missing or error.

    Exit codes:

    - 0: no file needs more runs.
    - 1: a file is incomplete or missing.
    - 3: a file couldn't be checked.

    The worst one wins.
    """
    stability_limits = _resolve_stability_limits(options.stop)
    max_runs = options.stop.max_runs
    if stability_limits is None and max_runs is None:
        raise UsageError(
            "check requires --max-relative-margin, --max-absolute-margin or --max-runs"
        )

    entries: list[dict[str, object]] = []
    exit_code = _CheckExitCode.Done
    for path in _expand_directories(options.paths):
        try:
            result = check_saved(path, stability=stability_limits, max_runs=max_runs)
        # NOTE: one bad file shouldn't hide the status of the others. The
        # error is reported in its entry, in the log and through the exit
        # code, so nothing is swallowed.
        except Exception as error:
            logger.exception(f"Failed to check {path}")
            entries.append({"path": str(path), "status": "error", "error": str(error)})
            exit_code = max(exit_code, _CheckExitCode.Error)
            continue

        entries.append(result.to_dict())
        if result.should_run():
            exit_code = max(exit_code, _CheckExitCode.ShouldRun)

    print(json.dumps(entries, indent=2))
    return exit_code


@dataclass(frozen=True, kw_only=True)
class DiscardBitmasksOptions:
    path: Annotated[
        Path,
        Positional(
            help="Path to a result file previously recorded with --compare-bitwise."
        ),
    ]


@app.command
def discard_bitmasks(options: DiscardBitmasksOptions) -> None:
    """Discard any recorded bitmasks from a saved result file, shrinking it.

    Operates directly on the saved file; unlike `record`, this doesn't need to
    reload the model or dataset that produced it.
    """
    discard_bitmasks_in_file(options.path)
