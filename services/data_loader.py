"""Cached data loaders for the TDD Dashboard."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from config import DASHBOARD_DIR, AVAILABLE_SEASONS, PROJECTION_LABEL


@st.cache_data
def load_projections(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_k_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data
def load_bf_priors() -> pd.DataFrame:
    path = DASHBOARD_DIR / "bf_priors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_arsenal() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_vulnerability(career: bool = False) -> pd.DataFrame:
    if career:
        path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    path = DASHBOARD_DIR / "hitter_vuln.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_strength(career: bool = False) -> pd.DataFrame:
    if career:
        path = DASHBOARD_DIR / "hitter_str_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    path = DASHBOARD_DIR / "hitter_str.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_counting(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_counting.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_game_info() -> pd.DataFrame:
    path = DASHBOARD_DIR / "game_info.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_player_teams() -> pd.DataFrame:
    path = DASHBOARD_DIR / "player_teams.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_location_grid() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_location_grid.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_zone_grid(career: bool = False) -> pd.DataFrame:
    if career:
        path = DASHBOARD_DIR / "hitter_zone_grid_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    path = DASHBOARD_DIR / "hitter_zone_grid.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_todays_games() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_games.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_todays_sims() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_sims.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_todays_lineups() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_lineups.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_traditional_stats(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_traditional.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_aggressiveness() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_aggressiveness.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_efficiency() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_efficiency.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_traditional_stats_all(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_traditional_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_aggressiveness_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_aggressiveness_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_efficiency_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_efficiency_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_arsenal_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_arsenal_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_vulnerability_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_vuln_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_strength_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_str_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_full_stats(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_full_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_location_grid_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_location_grid_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_zone_grid_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_zone_grid_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_preseason_injuries() -> pd.DataFrame:
    path = DASHBOARD_DIR / "preseason_injuries.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitcher_offerings() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_offerings.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_vuln_arch() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_vuln_arch.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_hitter_vuln_arch_career() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_vuln_arch_career.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_cluster_metadata() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_cluster_metadata.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_baselines_arch() -> pd.DataFrame:
    path = DASHBOARD_DIR / "baselines_arch.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_update_metadata() -> dict:
    path = DASHBOARD_DIR / "update_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_backtest(name: str) -> pd.DataFrame:
    """Load a backtest results parquet (e.g. 'pitcher_k_backtest')."""
    path = DASHBOARD_DIR / f"backtest_{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_weekly_snapshots(player_type: str) -> dict[str, pd.DataFrame]:
    """Load all weekly snapshots for a player type.

    Returns {date_str: DataFrame} sorted by date.
    """
    weekly_dir = DASHBOARD_DIR / "snapshots" / "weekly"
    if not weekly_dir.exists():
        return {}
    prefix = f"{player_type}_projections_"
    result = {}
    for f in sorted(weekly_dir.glob(f"{prefix}*.parquet")):
        date_str = f.stem.replace(prefix, "")
        result[date_str] = pd.read_parquet(f)
    return result


@st.cache_data
def load_latest_weekly_snapshot(player_type: str) -> tuple[str, pd.DataFrame] | None:
    """Load the most recent weekly snapshot. Returns (date_str, df) or None."""
    snapshots = load_weekly_snapshots(player_type)
    if not snapshots:
        return None
    latest_date = max(snapshots.keys())
    return latest_date, snapshots[latest_date]


def season_selector(key_prefix: str, include_career: bool = True) -> str:
    """Render a season selector and return the choice."""
    options = (
        [PROJECTION_LABEL]
        + (["Career"] if include_career else [])
        + [str(s) for s in AVAILABLE_SEASONS]
    )
    return st.selectbox("Season", options, key=f"{key_prefix}_season")
