"""Common helper functions used across experiment submodules"""


def relative_margin_of_error(
    mean: float | None, margin_of_error: float | None
) -> float | None:
    """The 95% margin of error as a percentage of the mean.

    `None` if either input is `None`. A mean of exactly `0` would otherwise
    raise `ZeroDivisionError` (a legitimate outcome for e.g. a 0% SDC score);
    that case is treated as 0% relative error when there is no error either,
    and as an undefined (infinite) relative error otherwise.
    """
    if mean is None or margin_of_error is None:
        return None
    if mean == 0:
        return 0.0 if margin_of_error == 0 else float("inf")
    return margin_of_error / mean * 100
