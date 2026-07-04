"""Encoded memory experiments."""

import enum
import logging
from pathlib import Path
from typing import Annotated

import typer
from faultforge import DEFAULT_BATCH_SIZE
from faultforge.encoding import (
    CepEncoder,
    CepScheme,
    Encoder,
    EncoderSequence,
    IdentityEncoder,
    MsetEncoder,
    SecdedEncoder,
)
from faultforge.experiment import SaveConfig, StabilityConfig
from faultforge.experiments.encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from faultforge.fingerprint import FingerprintError
from faultforge.loading import (
    Cifar,
    CifarDataset,
    CifarModel,
    ImageNet,
    ImageNetModel,
    ModelBundle,
)
from faultforge.progress import Progress

app = typer.Typer()
logger = logging.getLogger(__name__)


class DatasetChoice(enum.StrEnum):
    Cifar10 = "cifar10"
    Cifar100 = "cifar100"
    ImageNet = "imagenet"


def _init_model_bundle(
    dataset: DatasetChoice,
    model: str | None,
    imagenet_root: str | None,
    batch_size: int,
    preload_batches: bool,
    device: str,
) -> ModelBundle:
    """Build the `ModelBundle` for the given CLI choices and load the model/dataset from it."""
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
    pass


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
    reliability_metric: Annotated[
        ReliabilityMetric,
        typer.Option(
            help="Which metric to use for reliability measurements",
            rich_help_panel="Model Setup",
        ),
    ] = ReliabilityMetric.Accuracy,
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
            help="Run the experiment an exact number of times. Incompatible with --min-runs and --stability-threshold.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    min_runs: Annotated[
        int | None,
        typer.Option(
            help="Make sure the results contain at least N runs.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    stability_threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=100.0,
            help="Run until the mean has a margin of error smaller or equal to N% of the mean value at 95% confidence.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            help="Which device to use. PyTorch device string.",
            rich_help_panel="Misc Settings",
        ),
    ] = "cpu",
) -> None:
    """Run an encoded memory fault injection experiment and record the results."""
    bundle = _init_model_bundle(
        dataset, model, imagenet_root, batch_size, preload_batches, device
    )

    encoder = _resolve_encoder(
        mset=mset,
        cep=cep,
        cep_scheme=cep_scheme,
        secded=secded,
    )

    if stability_threshold is not None and runs is not None:
        raise typer.BadParameter("Cannot specify both --stability-threshold and --runs")
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
        reliability_metric,
        golden_is_encoded=golden_is_encoded,
        faults=faults_,
        preload_dataset=preload_batches,
        dataset_batch_limit=batch_limit,
        batch_size=batch_size,
        device=device,
        progress=Progress(),
    )
    save_config: SaveConfig | None = None
    if output is not None:
        output = Path(output).expanduser()
        if not output.parent.exists():
            logger.info(f"Creating output parent directory {output.parent}")
            output.parent.mkdir(parents=True)
        else:
            logger.debug(f"Output parent directory {output.parent} already exists")

        save_config = SaveConfig(path=output, interval_seconds=autosave)

    stability_config: StabilityConfig | None = None
    if stability_threshold is not None:
        if min_runs is None:
            min_samples = 0
        else:
            min_samples = min_runs
        stability_config = StabilityConfig(
            min_samples=min_samples, threshold=stability_threshold
        )

    if output is not None and output.exists():
        try:
            experiment.load_from(output)
        except FingerprintError as error:
            if not overwrite:
                logger.error(str(error))
                raise typer.Exit(1) from None
            logger.warning(
                f"{output} was recorded with a different configuration and will be overwritten:\n{error}"
            )

    if runs is not None:
        for _ in range(runs):
            experiment.run()
        if output is not None:
            experiment.save_atomic(output)
            return

    experiment.run_loop(stability_config=stability_config, save_config=save_config)
