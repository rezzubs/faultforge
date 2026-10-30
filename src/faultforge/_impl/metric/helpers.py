"""See [`faultforge.metric`]"""

from torch import Tensor


def ratio_to_percent(ratio: float) -> float:
    return ratio * 100


def top1_logit_from_output(batch_model_output: Tensor) -> Tensor:
    """Extract the top-1 logit from the model output (batched).

    Input n-d, output (n-1)-d
    """
    if len(batch_model_output.shape) < 2:
        raise ValueError(
            f"Logit extraction expects >=2 dimensional tensors, got shape {batch_model_output.shape}"
        )

    return batch_model_output.argmax(dim=1)
