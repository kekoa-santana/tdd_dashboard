"""The Home ticker label must describe the slate, not always say LIVE."""
from __future__ import annotations

import pytest

from views.home import _slate_state


def _sched(*statuses: str) -> list[dict]:
    return [{"status": s} for s in statuses]


@pytest.mark.parametrize(
    "statuses,expected",
    [
        # Nothing has started: the normal state for most of the day.
        (("Scheduled", "Scheduled"), "UPCOMING"),
        (("Warmup", "Scheduled"), "FIRST PITCH"),
        (("Pre-Game", "Scheduled"), "FIRST PITCH"),
        # Any game in progress wins, wherever it sits in the slate.
        (("In Progress",), "LIVE"),
        (("Final", "In Progress", "Scheduled"), "LIVE"),
        (("Scheduled", "Scheduled", "Live"), "LIVE"),
        # Part way through the day.
        (("Final", "Scheduled"), "TODAY"),
        # Everything played.
        (("Final", "Final", "Game Over"), "FINAL"),
        # A postponed game will not start, so it does not hold the slate open.
        (("Final", "Postponed"), "FINAL"),
        (("Final", "Cancelled"), "FINAL"),
        (("Postponed", "Scheduled"), "UPCOMING"),
    ],
)
def test_slate_label(statuses: tuple[str, ...], expected: str) -> None:
    assert _slate_state(_sched(*statuses))[0] == expected


def test_only_a_live_slate_pulses() -> None:
    assert _slate_state(_sched("In Progress"))[2] is True
    for statuses in (("Scheduled",), ("Final",), ("Final", "Scheduled"), ("Warmup",)):
        assert _slate_state(_sched(*statuses))[2] is False


def test_missing_and_empty_statuses_do_not_raise() -> None:
    assert _slate_state([])[0] == "TODAY"
    assert _slate_state([{}])[0] == "UPCOMING"
    assert _slate_state([{"status": None}])[0] == "UPCOMING"


def test_state_matches_label_for_styling() -> None:
    """The CSS hook must stay in step with the label."""
    assert _slate_state(_sched("In Progress"))[1] == "live"
    assert _slate_state(_sched("Final"))[1] == "final"
    assert _slate_state(_sched("Scheduled"))[1] == "upcoming"
