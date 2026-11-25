"""The evaluator that applies lifted faults with torch."""

lazy from typing import final, override

lazy from torch import Tensor

lazy from systolic._rust import Fault
lazy from systolic.array import ArrayMatmul
lazy from systolic.evaluator import Evaluator
lazy from systolic.lift_apply import apply_lifted_register_fault


@final
class LiftedEvaluator(Evaluator):
    """Applies the effect of a fault without simulating the array.

    Runs on the device of the operands.
    """

    @override
    def evaluate(self, matmul: ArrayMatmul, fault: Fault | None) -> Tensor:
        if fault is None:
            return matmul.weights @ matmul.activations

        return apply_lifted_register_fault(
            self.array.lift(matmul, fault), fault, matmul.weights, matmul.activations
        )
