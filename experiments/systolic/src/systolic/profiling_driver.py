"""Walk a model's mapped layers over a dataset subsample, producing a profiling artifact."""

lazy import torch

lazy from faultforge import tensor_list_dtype
lazy from faultforge.dataset import DEFAULT_DEVICE, DeviceLike
lazy from faultforge.loading import ModelBundle
lazy from faultforge.progress import Progress, stage
lazy from systolic._rust import ProfilingArtifact
lazy from systolic.model import BackendModel
lazy from systolic.profiling import ProfilingMetadata
lazy from systolic.profiling_backend import ProfilingBackend


def run_profiling(
    bundle: ModelBundle,
    array: tuple[int, int],
    capacity: int,
    *,
    subsample_size: int,
    seed: int = 0,
    device: DeviceLike = DEFAULT_DEVICE,
    progress: Progress | None = None,
) -> tuple[ProfilingArtifact, ProfilingMetadata]:
    """Profile a model's per-PE logic inputs over a random dataset subsample.

    Walks every mapped `nn.Linear`/`nn.Conv2d` layer in one forward pass over
    `subsample_size` images. The subsample is sampled uniformly at random
    (via `bundle.load_dataset(shuffle=True)`), not stratified, so the profile
    reflects the real class mixture a deployed model sees. `seed` reproduces
    both that sampling and the `Profiler`'s own reservoir sampling.

    Raises:
        ValueError: If the model loaded by `bundle` isn't float32.
    """
    nrows, ncols = array
    model = bundle.load_model(device, progress=progress)
    dtype = tensor_list_dtype(list(model.parameters()))
    if dtype != torch.float32:
        raise ValueError(f"run_profiling only supports float32, got {dtype}")
    backend = ProfilingBackend(nrows, ncols, capacity, seed)
    profiling_model = BackendModel(model, backend, progress=progress)

    # One batch of exactly subsample_size.
    dataset = bundle.load_dataset(
        subsample_size, device, shuffle=True, seed=seed, progress=progress
    )

    with stage(progress, "Profiling"), torch.no_grad():
        batch = next(iter(dataset))
        profiling_model.forward(batch.inputs)

    metadata = ProfilingMetadata(
        array_rows=nrows, array_cols=ncols, capacity=capacity, seed=seed
    )
    return backend.artifact(), metadata
