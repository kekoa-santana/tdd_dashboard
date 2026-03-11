"""Utility helper functions for the TDD Dashboard."""
from __future__ import annotations

import logging
import unicodedata

import pandas as pd
import streamlit as st

from config import DASHBOARD_DIR
from services.data_loader import load_player_teams, load_preseason_injuries
from services.manifest import validate_manifest

logger = logging.getLogger(__name__)


def strip_accents(s: str) -> str:
    """Remove diacritical marks (é→e, ñ→n, ú→u, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def get_team_lookup() -> dict[int, str]:
    """Return {player_id: team_abbr} dict from cached player_teams."""
    teams_df = load_player_teams()
    if teams_df.empty:
        return {}
    return dict(zip(teams_df["player_id"].astype(int), teams_df["team_abbr"]))


def get_injury_lookup() -> dict[int, dict]:
    """Return {player_id: {injury, status, severity, est_return, missed_games}} dict."""
    inj = load_preseason_injuries()
    if inj.empty:
        return {}
    lookup = {}
    for _, row in inj.iterrows():
        pid = row.get("player_id")
        if pd.notna(pid):
            lookup[int(pid)] = {
                "injury": row["injury"],
                "status": row["status"],
                "severity": row["severity"],
                "est_return": row["est_return_date"],
                "missed_games": int(row["est_missed_games"]),
            }
    return lookup


def check_data_exists() -> bool:
    """Check if pre-computed data exists.

    Also runs lenient manifest validation when a manifest is present,
    storing any warnings in ``st.session_state["manifest_warnings"]``
    for the Data Health page to display.
    """
    required = [
        DASHBOARD_DIR / "hitter_projections.parquet",
        DASHBOARD_DIR / "pitcher_projections.parquet",
    ]
    if not all(p.exists() for p in required):
        return False

    # Lenient manifest validation (non-blocking)
    try:
        status = validate_manifest(DASHBOARD_DIR, strict=False)
        if status.warnings:
            st.session_state["manifest_warnings"] = status.warnings
            for w in status.warnings:
                logger.warning("Manifest: %s", w)
        else:
            st.session_state["manifest_warnings"] = []
    except Exception as exc:
        logger.warning("Manifest validation failed: %s", exc)
        st.session_state["manifest_warnings"] = [
            f"Manifest validation error: {exc}"
        ]

    return True
