"""Home-page prop edge compatibility tests."""
from __future__ import annotations

import pandas as pd

from views.home import _top_prop_edges


def test_top_prop_edges_supports_current_game_props_schema():
    props = pd.DataFrame([
        {
            "player_name": "Over Player",
            "player_type": "pitcher",
            "team": "NYM",
            "stat": "K",
            "expected": 6.2,
            "line": 5.5,
        },
        {
            "player_name": "Under Player",
            "player_type": "pitcher",
            "team": "PHI",
            "stat": "Outs",
            "expected": 16.1,
            "line": 17.5,
        },
        {
            "player_name": "Incomplete Player",
            "expected": None,
            "line": 2.5,
        },
    ])

    edges = _top_prop_edges(props)

    assert [edge["name"] for edge in edges] == ["Under Player", "Over Player"]
    assert edges[0]["edge"] == -1.4
    assert edges[0]["direction"] == "under"
    assert edges[1]["edge"] == 0.7
    assert edges[1]["direction"] == "over"


def test_top_prop_edges_supports_legacy_schema():
    props = pd.DataFrame([{
        "player_name": "Legacy Player",
        "player_type": "batter",
        "team": "LAD",
        "stat": "H",
        "expected": 1.35,
        "vegas_line": 0.5,
        "model_edge": 0.85,
        "vegas_odds": -120,
    }])

    edges = _top_prop_edges(props)

    assert edges == [{
        "name": "Legacy Player",
        "player_type": "batter",
        "team": "LAD",
        "stat": "H",
        "expected": 1.35,
        "line": 0.5,
        "edge": 0.85,
        "odds": -120,
        "direction": "over",
    }]


def test_top_prop_edges_degrades_safely_for_partial_schema():
    props = pd.DataFrame([{"player_name": "No Projection", "line": 1.5}])

    assert _top_prop_edges(props) == []
    assert _top_prop_edges(pd.DataFrame()) == []
