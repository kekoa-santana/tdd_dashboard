"""Reusable league/division filter for team-based pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st


# Division display names → (league, division column value)
DIVISION_MAP: dict[str, dict[str, str | None]] = {
    "Overall": {"league": None, "division": None},
    "American League": {"league": "AL", "division": None},
    "National League": {"league": "NL", "division": None},
    "AL East": {"league": "AL", "division": "American League East"},
    "AL Central": {"league": "AL", "division": "American League Central"},
    "AL West": {"league": "AL", "division": "American League West"},
    "NL East": {"league": "NL", "division": "National League East"},
    "NL Central": {"league": "NL", "division": "National League Central"},
    "NL West": {"league": "NL", "division": "National League West"},
}

FILTER_OPTIONS = list(DIVISION_MAP.keys())


def division_selectbox(key: str = "div_filter") -> str:
    """Render a league/division selectbox and return the selected label."""
    return st.selectbox(
        "League / Division", FILTER_OPTIONS,
        key=key, label_visibility="collapsed",
    )


def apply_division_filter(
    df: pd.DataFrame,
    selected: str,
    league_col: str = "league",
    division_col: str = "division",
) -> pd.DataFrame:
    """Filter a DataFrame by the selected league/division label."""
    spec = DIVISION_MAP.get(selected)
    if spec is None:
        return df
    filtered = df.copy()
    if spec["league"] is not None and league_col in filtered.columns:
        filtered = filtered[filtered[league_col] == spec["league"]]
    if spec["division"] is not None and division_col in filtered.columns:
        filtered = filtered[filtered[division_col] == spec["division"]]
    return filtered
