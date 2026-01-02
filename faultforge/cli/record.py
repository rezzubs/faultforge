from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Annotated,
    Any,
    cast,
)

import torch
import typer

from faultforge.cifar.dataset import Cifar
from faultforge.cifar.model import CachedModel as CifarModel
from faultforge.cifar.system import System as CifarSystem
from faultforge.cli.utils import setup_logging
from faultforge.data import (
    Autosave,
    Data,
)
from faultforge.dtype import DnnDtype
from faultforge.encoding.bit_pattern import BitPattern, BitPatternEncoder
from faultforge.encoding.embedded_parity import EmbeddedParityEncoder, EpScheme
from faultforge.encoding.msb import MsbEncoder
from faultforge.encoding.secded import SecdedEncoder
from faultforge.encoding.sequence import EncoderSequence, TensorEncoder
from faultforge.encoding.system import EncodedSystem
from faultforge.imagenet.dataset import ImageNet
from faultforge.imagenet.model import Model as ImagenetModel
from faultforge.imagenet.system import System as ImagenetSystem
from faultforge.system import BaseSystem

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def record(
    imagenet_model: Annotated[
        ImagenetModel | None,
        typer.Option(
            help="The model to run ImageNet on. \
See --cifar-kind to specify the exact dataset. \
Incompatible with --cifar-model.",
            rich_help_panel="Model setup",
        ),
    ] = None,
    cifar_model: Annotated[
        CifarModel | None,
        typer.Option(
            help="Choose an an model to run CIFAR on. \
See --cifar-kind to specify the exact dataset. \
",
            rich_help_panel="Model setup",
        ),
    ] = None,
    imagenet_path: Annotated[
        Path | None,
        typer.Option(
            help="The path that stores the ImageNet data.",
            rich_help_panel="Model setup",
        ),
    ] = None,
    imagenet_limit: Annotated[
        int | None,
        typer.Option(
            help="Only use the first n images from --imagenet-path.",
            rich_help_panel="Model setup",
        ),
    ] = None,
    cifar_kind: Annotated[
        Cifar.Kind,
        typer.Option(
            "--dataset",
            "-d",
            help="The dataset to use for evaluating a --cifar-model.",
            rich_help_panel="Model setup",
        ),
    ] = Cifar.Kind.CIFAR10,
    cifar_cache: Annotated[
        Path | None,
        typer.Option(
            help="The path to use for storing the automatically downloaded CIFAR datasets. \
`./cifar-cache` by default.",
            rich_help_panel="Model setup",
        ),
    ] = None,
    dtype: Annotated[
        DnnDtype,
        typer.Option(
            help="The data type to use for the model.",
            rich_help_panel="Model setup",
        ),
    ] = DnnDtype.Float32,
    batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size to use for the data loader.",
            rich_help_panel="Model setup",
        ),
    ] = 1000,
    bit_error_rate: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="The bit error rate to use for fault injection. Incompatible with --faults-count",
            rich_help_panel="Fault injection settings",
        ),
    ] = None,
    faults_count: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="How many faults to inject into the model. Incompatible with --bit_error_rate",
            rich_help_panel="Fault injection settings",
        ),
    ] = None,
    secded: Annotated[
        bool,
        typer.Option(
            "--secded",
            help="Use SECDED encoding based on hamming codes.",
            rich_help_panel="Encoding settings",
        ),
    ] = False,
    bit_pattern: Annotated[
        BitPattern | None,
        typer.Option(
            "--secded-bit-pattern",
            "--secded-bits",
            parser=BitPattern.parse,
            help="""Encode the specified repeating bit-pattern with hamming codes. Syntax: `_`
separated bit indices or ranges of indices. For example, 0-4_14_18-19. The bit
indices cannot exceed the number of bits in the data type.""",
            rich_help_panel="Encoding settings",
        ),
    ] = None,
    bits_per_chunk: Annotated[
        int,
        typer.Option(
            "--secded-chunk-size",
            help="How many bits to encode as a chunk. \
For example if we want to have 1 ecc per 2 float32 values (with all bits protected), \
a chunk size of 64 should be used.",
            rich_help_panel="Encoding settings",
        ),
    ] = 64,
    duplicate_msb: Annotated[
        bool,
        typer.Option(
            help="Duplicate the most significant bit inside two lowest ones.",
            rich_help_panel="Encoding settings",
        ),
    ] = False,
    embedded_parity: Annotated[
        bool,
        typer.Option(
            help="Use embedded parity encoding",
            rich_help_panel="Encoding settings",
        ),
    ] = False,
    embedded_parity_scheme: Annotated[
        EpScheme,
        typer.Option(
            help="""\
Which scheme to use for embedded parity. The number after D determines the
number of data bits per parity bits (P). The default is most likely optimal.
            """,
            rich_help_panel="Encoding settings",
        ),
    ] = EpScheme.D3P1,
    device: Annotated[  # pyright: ignore[reportRedeclaration]
        str,
        typer.Option(
            help="A pytorch device string, for example `cuda:0`.",
        ),
    ] = "cpu",
    runs: Annotated[
        int | None,
        typer.Option(
            min=2,
            help="How many runs to perform.",
            rich_help_panel="Recording settings. 1 run is done by default.",
        ),
    ] = None,
    until_stable: Annotated[
        int | None,
        typer.Option(
            min=2,
            help="run until the accuracy mean within this many runs has not fluctuated over --stability-threshold %. \
--runs can be used to signal the minimum number of runs",
            rich_help_panel="Recording settings",
        ),
    ] = None,
    stability_threshold: Annotated[
        float,
        typer.Option(
            min=0,
            max=100,
            help="run until the accuracy mean within this many runs has not fluctuated over --stability-threshold %",
            rich_help_panel="Recording settings",
        ),
    ] = 1,
    output_path: Annotated[
        Path | None,
        typer.Option(
            help="The path of the file to save the results into. \
If the file doesn't exist then it will be created. \
If the path is a directory (existing) then the file will be called data.json",
            rich_help_panel="Recording settings",
        ),
    ] = None,
    metadata_name: Annotated[
        bool,
        typer.Option(
            help="Generate a file name based on the configuration metadata. Must be used together with --output-path.",
            rich_help_panel="Recording settings",
        ),
    ] = False,
    autosave: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="How often to save the data while recording, interval - number of runs. \
The default is to only save at the very end",
            rich_help_panel="Recording settings",
        ),
    ] = None,
    summary: Annotated[
        bool,
        typer.Option(
            help="Print a summary at the end of each evaluation",
            rich_help_panel="Recording settings",
        ),
    ] = False,
    skip_comparison: Annotated[
        bool,
        typer.Option(
            help="Skip the bitwise comparison of tensors after fault injection. \
This can speed up recording when only accuracy is needed. \
This also greatly reduces the output file size for large numbers of faults.",
            rich_help_panel="Recording settings",
        ),
    ] = False,
):
    """Record data entries for a model and dataset."""
    device: torch.device = torch.device(device)

    if cifar_cache is None:
        cifar_cache = Path("./cifar-cache")
    cifar_cache = cifar_cache.expanduser()

    match (cifar_model, imagenet_model):
        case (None, None):
            print("No model provided")
            raise typer.Abort()
        case (_, None):
            system = CifarSystem(
                dataset=Cifar(cifar_kind, cifar_cache),
                model=cifar_model,
                dtype=dtype,
                device=device,
                batch_size=batch_size,
                dataset_cache=cifar_cache,
            )
        case (None, _):
            if imagenet_path is None:
                print("ImageNet path must be provided")
                raise typer.Abort()

            system = ImagenetSystem(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                model=imagenet_model,
                dataset=ImageNet(
                    imagenet_path.expanduser(),
                    transform=imagenet_model.get_transform(),
                    limit=imagenet_limit,
                ),
            )
        case (_, _):
            print("Only one model can be used at once")
            raise typer.Exit()

    system = cast(BaseSystem[Any], system)

    match (secded, bit_pattern):
        case (_, BitPattern()):
            logger.debug("Using BitPatternEncoder")
            encoder = BitPatternEncoder(
                pattern=bit_pattern,
                pattern_length=dtype.bits_count(),
                bits_per_chunk=bits_per_chunk,
            )
        case (True, None):
            logger.debug("Using FullEncoder")
            encoder = SecdedEncoder(bits_per_chunk)
        case _:
            encoder = None

    head_encoders: list[TensorEncoder] = []

    if duplicate_msb:
        if encoder is None:
            logger.debug("Using MsbEncoder")
            encoder = MsbEncoder()
        else:
            head_encoders.append(MsbEncoder())

    if embedded_parity:
        if encoder is None:
            logger.debug(f"Using EmbeddedParityEncoder with {embedded_parity_scheme}")
            encoder = EmbeddedParityEncoder(embedded_parity_scheme)
        else:
            head_encoders.append(EmbeddedParityEncoder(embedded_parity_scheme))

    if len(head_encoders) > 0:
        assert encoder is not None
        logger.debug(f"Wrapping previous encoder with {head_encoders}")
        encoder = EncoderSequence(head_encoders, encoder)

    if encoder is not None:
        logger.debug("Preparing the system to be encoded")
        system = EncodedSystem(system, encoder)

    bits_count_total = system.system_total_bits_count()

    match (faults_count, bit_error_rate):
        case (None, None):
            faults_count = 0
        case (faults_count, None):
            faults_count = faults_count
        case (None, bit_error_rate):
            faults_count = int(round(bits_count_total * bit_error_rate))
        case _:
            print("Choose one of --bit_error_rate and --faults_count")
            raise typer.Exit()

    if output_path is not None:
        output_path = output_path.expanduser()

    data = Data.load_or_create(
        output_path,
        faults_count=faults_count,
        bits_count=bits_count_total,
        metadata=system.system_metadata(),
        metadata_name=metadata_name,
    )

    logger.debug(f"Proceeding with metadata: {data.metadata}")

    match (runs, until_stable):
        case (None, None):
            _ = data.record_entry(
                cast(BaseSystem[Any], system),
                summary=summary,
                skip_comparison=skip_comparison,
            )
        case (_, None):
            if autosave is not None and output_path is not None:
                save_config = Autosave(autosave, output_path, metadata_name)
            else:
                save_config = None

            data.record_entries(
                cast(BaseSystem[Any], system),
                runs,
                summary=summary,
                skip_comparison=skip_comparison,
                autosave=save_config,
            )
        case _:
            if autosave is not None and output_path is not None:
                save_config = Autosave(autosave, output_path, metadata_name)
            else:
                save_config = None

            _ = data.record_until_stable(
                cast(BaseSystem[Any], system),
                threshold=stability_threshold,
                stable_within=until_stable,
                min_runs=runs,
                skip_comparison=skip_comparison,
                autosave=save_config,
                summary=summary,
            )

    if output_path:
        data.save(output_path, metadata_name)


if __name__ == "__main__":
    setup_logging()
    app()
