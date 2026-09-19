"""The Home ticker label must describe the slate, not always say LIVE."""
from __future__ import annotations

import pytest

from views.home import _inning_tag, _slate_state, _ticker_game


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


# ---------------------------------------------------------------------------
# Scores in the ticker
# ---------------------------------------------------------------------------

def _game(status: str = "Scheduled", **kw) -> dict:
    base = {"game_pk": 1, "away": "CHC", "home": "CIN",
            "time": "6:40 PM ET", "status": status}
    base.update(kw)
    return base


def _live(away: int | None, home: int | None, status: str, **kw) -> dict:
    out = {"away_score": away, "home_score": home, "status": status,
           "inning": kw.get("inning"), "inning_state": kw.get("inning_state", "")}
    return out


def test_final_game_shows_score_and_marks_the_winner() -> None:
    html = _ticker_game(_game("Final"), _live(1, 3, "Final"))
    assert ">1<" in html and ">3<" in html
    assert "FINAL" in html
    # The home side won, so only it carries the winner class.
    home_part = html.split('class="home-ticker-at"')[1]
    away_part = html.split('class="home-ticker-at"')[0]
    assert "is-win" in home_part
    assert "is-win" not in away_part


def test_tie_marks_no_winner() -> None:
    html = _ticker_game(_game("Final"), _live(2, 2, "Final"))
    assert "is-win" not in html


def test_live_game_shows_score_and_half_inning() -> None:
    html = _ticker_game(
        _game("In Progress"),
        _live(6, 11, "In Progress", inning=5, inning_state="Bottom"),
    )
    assert ">6<" in html and ">11<" in html
    assert "B5" in html
    assert "is-live" in html
    # A game still being played has no winner yet.
    assert "is-win" not in html


def test_scheduled_game_shows_start_time_not_a_score() -> None:
    html = _ticker_game(_game("Scheduled"), None)
    assert "6:40 PM ET" in html
    assert "home-ticker-runs" not in html


def test_falls_back_to_status_when_the_score_fetch_fails() -> None:
    """An empty live mapping must still render, just without numbers."""
    final_html = _ticker_game(_game("Final"), None)
    assert "FINAL" in final_html
    assert "home-ticker-runs" not in final_html

    live_html = _ticker_game(_game("In Progress"), None)
    assert "home-ticker-runs" not in live_html
    assert "LIVE" in live_html


def test_live_status_overrides_a_stale_parquet_status() -> None:
    """The cached parquet can lag; the API status wins."""
    html = _ticker_game(
        _game("Scheduled"),
        _live(1, 0, "In Progress", inning=2, inning_state="Top"),
    )
    assert "T2" in html
    assert "6:40 PM ET" not in html

    state = _slate_state(
        [_game("Scheduled")],
        {1: _live(1, 0, "In Progress", inning=2, inning_state="Top")},
    )
    assert state[0] == "LIVE" and state[2] is True


@pytest.mark.parametrize(
    "inning,state,expected",
    [
        (7, "Top", "T7"),
        (9, "Bottom", "B9"),
        (3, "Middle", "T3"),
        (4, "End", "B4"),
        (None, "Top", "LIVE"),
        (6, "", "IN 6"),
    ],
)
def test_inning_tag(inning, state, expected) -> None:
    assert _inning_tag({"inning": inning, "inning_state": state}) == expected
