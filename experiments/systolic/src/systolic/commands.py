"""The `systolic` CLI commands (recording register-fault campaigns and reporting backend agreement)."""

lazy import enum
lazy import logging
lazy from collections.abc import Sequence
lazy from pathlib import Path
lazy from typing import Annotated

lazy import matplotlib.pyplot as plt
lazy import numpy as np
lazy import torch
lazy import typer
lazy from matplotlib.backends.registry import BackendFilter, backend_registry
lazy from matplotlib.figure import Figure

lazy import faultforge.metric as m
lazy from faultforge import Picker
lazy from faultforge.dataset import DEFAULT_BATCH_SIZE
lazy from faultforge.experiment import (
    AbsoluteThreshold,
    AdditionalRuns,
    MaxRuns,
    Mean,
    NonFiniteRate,
    RelativeThreshold,
    SaveConfig,
    StopCondition,
)
lazy from faultforge.fingerprint import FingerprintError
lazy from faultforge.io import is_compressed
lazy from faultforge.loading import (
    Cifar,
    CifarDataset,
    CifarModel,
    ImageNet,
    ImageNetModel,
    ModelBundle,
)
lazy from faultforge.metric import Metric
lazy from faultforge.progress import Progress
lazy from systolic._rust import ArrayConfig, PeRegisterKind
lazy from systolic.agreement import compare_matmul, summarize
lazy from systolic.experiment import SystolicFaultInjection
lazy from systolic.fault import RegisterFaults
lazy from systolic.profiling import load_profiling_artifact, save_profiling_artifact
lazy from systolic.profiling_driver import run_profiling
lazy from systolic.profiling_plots import (
    build_ecdf_spread_figure,
    build_gap_heatmap_figure,
    build_pe_ecdf_figure,
)
lazy from systolic.profiling_similarity import Regime, gap_grids
lazy from systolic.stability import StabilityLimits

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
logger = logging.getLogger(__name__)


class DatasetChoice(enum.StrEnum):
    Cifar10 = "cifar10"
    Cifar100 = "cifar100"
    ImageNet = "imagenet"


class MetricChoice(enum.StrEnum):
    Accuracy = "accuracy"
    AccuracyDegradation = "accuracy-degradation"
    Sdc = "sdc"
    Top1Sdc = "top1-sdc"

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


class BackendChoice(enum.StrEnum):
    Simulated = "simulated"
    Lifted = "lifted"


class RegisterChoice(enum.StrEnum):
    Activation = "activation"
    Weight = "weight"
    Accumulator = "accumulator"


_REGISTER_MAP = {
    RegisterChoice.Activation: PeRegisterKind.Activation,
    RegisterChoice.Weight: PeRegisterKind.Weight,
    RegisterChoice.Accumulator: PeRegisterKind.Accumulator,
}


def _resolve_registers(values: Sequence[RegisterChoice] | None) -> RegisterFaults:
    if values is None:
        return RegisterFaults()
    return RegisterFaults(registers=frozenset(_REGISTER_MAP[v] for v in values))


class EstimateChoice(enum.StrEnum):
    Mean = "mean"
    FailureRate = "failure-rate"
    NonFiniteRate = "non-finite-rate"


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
        "estimate's units (percentage points for failure-rate and non-finite-rate). "
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
        "threshold; non-finite scores always count as failures. "
        "`non-finite-rate` is the percentage of runs with a non-finite score.",
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

    stability_estimate: Mean | NonFiniteRate | RelativeThreshold | AbsoluteThreshold
    match estimate:
        case EstimateChoice.Mean | EstimateChoice.NonFiniteRate:
            for value, option in [
                (relative_failure_threshold, "--relative-failure-threshold"),
                (absolute_failure_threshold, "--absolute-failure-threshold"),
            ]:
                if value is not None:
                    raise typer.BadParameter(
                        f"{option} only applies to --estimate failure-rate",
                        param_hint=option,
                    )
            stability_estimate = (
                Mean() if estimate == EstimateChoice.Mean else NonFiniteRate()
            )
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
                        stability_estimate = RelativeThreshold(
                            relative_failure_threshold
                        )
                    case (None, float()):
                        stability_estimate = AbsoluteThreshold(
                            absolute_failure_threshold
                        )
            except ValueError as error:
                raise typer.BadParameter(str(error)) from error

    try:
        return StabilityLimits(
            estimate=stability_estimate,
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
) -> ModelBundle:
    """Build the `ModelBundle` for the given CLI choices."""
    if model is None:
        raise typer.BadParameter("A --model must be specified.", param_hint="--model")

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
            bundle = Cifar(model=cifar_model, dataset=CifarDataset(dataset.value))
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
            bundle = ImageNet(kind=imagenet_model, root=imagenet_root)

    return bundle


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

    for model in models:
        typer.echo(model)


@app.command(no_args_is_help=True)
def record(
    model: Annotated[
        str,
        typer.Option(
            help="Which model to use. Choices depend on the dataset. The list-models command can be used to see available models. Only models built from groups=1 convolutions are supported.",
            rich_help_panel="Model Setup",
        ),
    ],
    array_rows: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of rows in the systolic array.",
            rich_help_panel="Systolic Array",
        ),
    ],
    array_cols: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of columns in the systolic array.",
            rich_help_panel="Systolic Array",
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
            help="Path to a local directory containing ILSVRC2012_devkit_t12.tar.gz  and ILSVRC2012_img_val.tar. Required when --dataset is imagenet.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size of the dataset",
            rich_help_panel="Model Setup",
        ),
    ] = DEFAULT_BATCH_SIZE,
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
    backend: Annotated[
        BackendChoice,
        typer.Option(
            help="simulated is the cycle-accurate oracle; lifted is the fast torch-side workhorse.",
            rich_help_panel="Systolic Array",
        ),
    ] = BackendChoice.Lifted,
    registers: Annotated[
        list[RegisterChoice] | None,
        typer.Option(
            "--registers",
            help="Restrict faults to these registers. Repeat to select several. Defaults to all three.",
            rich_help_panel="Systolic Array",
        ),
    ] = None,
    reliability_metric: Annotated[
        MetricChoice,
        typer.Option(
            help="Which metric to use for reliability measurements",
            rich_help_panel="Reliability",
        ),
    ] = MetricChoice.Accuracy,
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
            help="Save --output zstd-compressed. Only controls the format of a newly created file; if --output already exists, its existing on-disk format (compressed or not) is kept regardless of this flag.",
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
    """Run a systolic register-fault injection experiment and record the results."""
    metric = reliability_metric.into_metric()
    stability_limits = _resolve_stability_limits(
        estimate=estimate,
        relative_failure_threshold=relative_failure_threshold,
        absolute_failure_threshold=absolute_failure_threshold,
        max_relative_margin=max_relative_margin,
        max_absolute_margin=max_absolute_margin,
        min_runs=min_runs,
    )

    bundle = _init_model_bundle(dataset, model, imagenet_root)
    fault = _resolve_registers(registers)

    if stability_limits is not None and runs is not None:
        raise typer.BadParameter(
            "Cannot specify both --runs and --max-relative-margin/--max-absolute-margin"
        )
    if runs is not None and min_runs is not None:
        raise typer.BadParameter("Cannot specify both --runs and --min-runs")

    experiment = SystolicFaultInjection(
        bundle,
        (array_rows, array_cols),
        metric,
        fault=fault,
        backend=backend.value,
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


@app.command(no_args_is_help=True)
def agreement(
    array_rows: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of rows in the systolic array.",
            rich_help_panel="Systolic Array",
        ),
    ],
    array_cols: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of columns in the systolic array.",
            rich_help_panel="Systolic Array",
        ),
    ],
    registers: Annotated[
        list[RegisterChoice] | None,
        typer.Option(
            "--registers",
            help="Restrict sampled faults to these registers. Repeat to select several. Defaults to all three.",
            rich_help_panel="Systolic Array",
        ),
    ] = None,
    out_features: Annotated[
        int,
        typer.Option(min=1, rich_help_panel="Matmul Shape"),
    ] = 16,
    in_features: Annotated[
        int,
        typer.Option(min=1, rich_help_panel="Matmul Shape"),
    ] = 16,
    batch: Annotated[
        int,
        typer.Option(min=1, rich_help_panel="Matmul Shape"),
    ] = 8,
    samples: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of randomly sampled faults to compare.",
            rich_help_panel="Sampling",
        ),
    ] = 50,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Seed for the random weights and activations.",
            rich_help_panel="Sampling",
        ),
    ] = None,
) -> None:
    """Sample register faults and report how each backend agrees with the simulated oracle."""
    array = ArrayConfig(array_rows, array_cols, 32)
    fault_config = _resolve_registers(registers)
    radix = fault_config.radix(array)
    sample_count = min(samples, radix)

    rng = np.random.default_rng(seed)

    weights = torch.from_numpy(
        rng.standard_normal((out_features, in_features)).astype(np.float32)
    )
    activations = torch.from_numpy(
        rng.standard_normal((in_features, batch)).astype(np.float32)
    )

    picker = Picker(radix)
    all_agreements = [
        compare_matmul(
            (array_rows, array_cols),
            weights,
            activations,
            fault_config.fault_from_id(fault_id, array),
        )
        for fault_id in (next(picker) for _ in range(sample_count))
    ]

    for report in summarize(all_agreements):
        print(
            f"{report.other} vs {report.baseline} ({report.samples} faults): "
            f"max_abs_error mean={report.mean_max_abs_error:.3e} max={report.max_max_abs_error:.3e}, "
            f"max_relative_error mean={report.mean_max_relative_error:.3e} max={report.max_max_relative_error:.3e}, "
            f"top1_flip_fraction mean={report.mean_top1_flip_fraction:.4f} max={report.max_top1_flip_fraction:.4f}"
        )


@app.command(no_args_is_help=True)
def profile(
    model: Annotated[
        str,
        typer.Option(
            help="Which model to use. Choices depend on the dataset. The list-models command can be used to see available models. Only models built from groups=1 convolutions are supported.",
            rich_help_panel="Model Setup",
        ),
    ],
    array_rows: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of rows in the systolic array.",
            rich_help_panel="Systolic Array",
        ),
    ],
    array_cols: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of columns in the systolic array.",
            rich_help_panel="Systolic Array",
        ),
    ],
    capacity: Annotated[
        int,
        typer.Option(
            min=1,
            help="Per-PE, per-regime reservoir capacity.",
            rich_help_panel="Profiling",
        ),
    ],
    subsample_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of dataset images to sample uniformly at random.",
            rich_help_panel="Profiling",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            help="Where to save the profiling artifact.",
            rich_help_panel="Profiling",
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
            help="Path to a local directory containing ILSVRC2012_devkit_t12.tar.gz  and ILSVRC2012_img_val.tar. Required when --dataset is imagenet.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed for the dataset subsample and the profiler's reservoir sampling.",
            rich_help_panel="Profiling",
        ),
    ] = 0,
    device: Annotated[
        str,
        typer.Option(
            help="Which device to use. PyTorch device string.",
            rich_help_panel="Misc Settings",
        ),
    ] = "cpu",
) -> None:
    """Profile a model's per-PE logic inputs over a dataset and save the artifact."""
    bundle = _init_model_bundle(dataset, model, imagenet_root)

    artifact, metadata = run_profiling(
        bundle,
        (array_rows, array_cols),
        capacity,
        subsample_size=subsample_size,
        seed=seed,
        device=device,
        progress=Progress(),
    )

    output = Path(output).expanduser()
    if not output.parent.exists():
        logger.info(f"Creating output parent directory {output.parent}")
        output.parent.mkdir(parents=True)

    save_profiling_artifact(output, artifact, metadata)


def _show_or_save(fig: Figure, output: Path | None) -> None:
    if output is not None:
        fig.savefig(output)
        return

    # `plt.show()` silently does nothing on a non-interactive backend (e.g.
    # the "agg" fallback matplotlib picks when no GUI toolkit like PyQt or
    # tkinter is importable) - check first so a missing --output doesn't look
    # like the command did nothing.
    backend = plt.get_backend()
    interactive_backends = {
        name.lower()
        for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    if backend.lower() not in interactive_backends:
        logger.error(
            f"no --output given and matplotlib has no interactive backend "
            f"available (using {backend!r}). Install a GUI toolkit matplotlib "
            "can use (e.g. PyQt6, or a working tkinter), or pass --output to "
            "save the figure to a file instead."
        )
        raise typer.Exit(1)

    # `fig` was built via the `Figure` OO API directly (see `profiling_plots`),
    # so pyplot never tracked it the way a `plt.figure()`-created figure would
    # be - `plt.show()` only displays figures pyplot is tracking, so without
    # this it would silently do nothing. Passing an untracked `Figure` as
    # `num` makes `plt.figure()` adopt it instead of creating a new one.
    plt.figure(fig)
    plt.show()


@app.command(no_args_is_help=True)
def plot_gap_heatmap(
    artifact: Annotated[
        Path,
        typer.Argument(help="A profiling artifact produced by `systolic profile`."),
    ],
    regime: Annotated[
        Regime,
        typer.Option(help="Which profiled input regime to compare PEs within."),
    ] = Regime.Active,
    sample_size: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "How many observations to compare per PE. Every PE needs at "
                "least twice this many recorded samples to take part; PEs "
                "with fewer are excluded and shown in grey."
            ),
        ),
    ] = 200,
    seed: Annotated[
        int,
        typer.Option(help="Seed for subsampling each PE's observations."),
    ] = 0,
    output: Annotated[
        Path | None,
        typer.Option(help="Save the figure here instead of opening a window."),
    ] = None,
) -> None:
    """Show, per PE, how different its profiled inputs are from the array as a whole.

    For each variable the regime records (activation, weight, partial sum),
    draws an `array_rows x array_cols` heatmap of one number per PE: how much
    that PE's sample of values differs from a sample pooled over the entire
    array. A uniformly dark heatmap says the array can be treated as one
    pool; visible row or column stripes say row- or column-specific
    modeling would capture something real.
    """
    arrays, _metadata = load_profiling_artifact(artifact)
    grids = gap_grids(arrays, regime, sample_size=sample_size, seed=seed)

    noise_floors = ", ".join(
        f"{variable.value}={grid.noise_floor():.3f}" for variable, grid in grids.items()
    )
    print(f"Noise floor (gap of a PE against itself): {noise_floors}")
    print(
        "A PE's own gap against the pooled array is only meaningful once it clearly exceeds this."
    )

    fig = build_gap_heatmap_figure(grids, regime)
    _show_or_save(fig, output)


@app.command(no_args_is_help=True)
def plot_pe_ecdf(
    artifact: Annotated[
        Path,
        typer.Argument(help="A profiling artifact produced by `systolic profile`."),
    ],
    row: Annotated[int, typer.Option(help="Which PE row to inspect.")],
    col: Annotated[int, typer.Option(help="Which PE column to inspect.")],
    regime: Annotated[
        Regime,
        typer.Option(help="Which profiled input regime to compare the PE within."),
    ] = Regime.Active,
    output: Annotated[
        Path | None,
        typer.Option(help="Save the figure here instead of opening a window."),
    ] = None,
) -> None:
    """Show one PE's ECDF against the pooled array, for a closer look after `plot-gap-heatmap`.

    For each variable the regime records, overlays the chosen PE's step
    function against the whole array's and marks the largest vertical gap
    between them - the same comparison `plot-gap-heatmap`'s numbers
    summarize, made visible.
    """
    arrays, _metadata = load_profiling_artifact(artifact)
    fig = build_pe_ecdf_figure(arrays, regime, row=row, col=col)
    _show_or_save(fig, output)


@app.command(no_args_is_help=True)
def plot_ecdf_spread(
    artifact: Annotated[
        Path,
        typer.Argument(help="A profiling artifact produced by `systolic profile`."),
    ],
    regime: Annotated[
        Regime,
        typer.Option(help="Which profiled input regime to compare PEs within."),
    ] = Regime.Active,
    output: Annotated[
        Path | None,
        typer.Option(help="Save the figure here instead of opening a window."),
    ] = None,
) -> None:
    """Show every PE's ECDF against the pooled array, for each variable.

    Complements `plot-gap-heatmap`: the heatmap says how much each PE
    differs from the pooled array, this shows how - a shifted distribution,
    a heavier tail, a different spread - by drawing every PE's step
    function together with the pooled one.
    """
    arrays, _metadata = load_profiling_artifact(artifact)
    fig = build_ecdf_spread_figure(arrays, regime)
    _show_or_save(fig, output)
