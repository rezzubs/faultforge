"""Tests for ExperimentDisplay.extra."""

from typing import override

from faultforge.experiment import ExperimentDisplay

# `_` prefixed to not interpret it as a Test class.


class _DisplayWithExtra(ExperimentDisplay):
    def __init__(self, value: str | None) -> None:
        self._value = value

    @override
    def extra(self) -> str | None:
        return self._value


def test_extra_appended_as_is():
    # `extra` is appended verbatim, with no separator added by `format` - the
    # override is responsible for its own leading spacing/delimiter.
    display = _DisplayWithExtra(" | note")
    status = display.format(
        run_count=1,
        score=1.0,
        condition_status=["mean 1.50"],
    )
    assert status.endswith(" | note")


def test_extra_omitted_when_none():
    display = _DisplayWithExtra(None)
    status = display.format(
        run_count=1,
        score=1.0,
        condition_status=["mean 1.50"],
    )
    assert "note" not in status


def test_extra_shown_without_stability_condition():
    # `extra` applies regardless of whether any condition status ends up in
    # the line, since a `Stability` condition need not be configured.
    display = _DisplayWithExtra(" | note")
    status = display.format(
        run_count=1,
        score=1.0,
        condition_status=[],
    )
    assert status.endswith(" | note")


def test_extra_shown_with_single_score():
    # `extra` applies even when no condition status is passed.
    display = _DisplayWithExtra(" | note")
    status = display.format(
        run_count=1,
        score=1.0,
    )
    assert status.endswith(" | note")
