"""The evaluator that simulates the array."""

lazy from typing import final, override

lazy import torch
lazy from torch import Tensor

lazy from systolic._rust import Fault, simulated_matmul
lazy from systolic.array import ArrayMatmul
lazy from systolic.evaluator import Evaluator


@final
class SimulatedEvaluator(Evaluator):
    """Simulates the array cycle by cycle.

    Much slower than `LiftedEvaluator` and always runs on the CPU. Meant for
    checking other evaluators.
    """

    @override
    def evaluate(self, matmul: ArrayMatmul, fault: Fault | None) -> Tensor:
        # The simulator ignores a fault in a processing element that doesn't
        # exist.
        if fault is not None:
            self.array.check_fault(fault)

        mapping = self.array.mapping(matmul)
        outputs = [
            torch.from_numpy(
                simulated_matmul(
                    mapping,
                    single.weights.numpy(force=True),
                    single.activations.numpy(force=True),
                    self.array.nrows,
                    self.array.ncols,
                    fault,
                )
            )
            for single in matmul.unstack()
        ]
        output = outputs[0] if matmul.weights.dim() == 2 else torch.stack(outputs)
        return output.to(matmul.weights.device)
