"""The evaluator that records the inputs of the array's processing elements."""

lazy from typing import final, override

lazy import torch
lazy from torch import Tensor

lazy from systolic._rust import Fault, Profiler, ProfilingArtifact
lazy from systolic.array import Array, ArrayMatmul
lazy from systolic.evaluator import Evaluator


@final
class ProfilingEvaluator(Evaluator):
    """Simulates a fault-free array and records what its logic sees.

    Pools the inputs of every evaluated multiplication into one artifact.
    Always runs on the CPU.

    `capacity` bounds how many samples are kept per processing element and
    regime. `seed` determines which ones.
    """

    def __init__(self, array: Array, capacity: int, seed: int) -> None:
        super().__init__(array)
        self._profiler = Profiler(array.nrows, array.ncols, capacity, seed)

    @override
    def evaluate(self, matmul: ArrayMatmul, fault: Fault | None) -> Tensor:
        """Compute and record `matmul` on the array.

        Raises:
            ValueError: If `fault` isn't `None`. Only a fault-free array can be
                profiled.
        """
        if fault is not None:
            raise ValueError("cannot profile an array with a fault")

        mapping = self.array.mapping(matmul)
        outputs = [
            torch.from_numpy(
                self._profiler.run(
                    mapping,
                    single.weights.numpy(force=True),
                    single.activations.numpy(force=True),
                )
            )
            for single in matmul.unstack()
        ]
        output = outputs[0] if matmul.weights.dim() == 2 else torch.stack(outputs)
        return output.to(matmul.weights.device)

    def artifact(self) -> ProfilingArtifact:
        """Snapshot everything recorded so far.

        Non-destructive. Evaluation can continue afterwards.
        """
        return self._profiler.artifact()
