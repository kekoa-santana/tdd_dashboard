"""Cached data loaders for the TDD Dashboard."""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from config import DASHBOARD_DIR, AVAILABLE_SEASONS, PROJECTION_LABEL
from utils.archetype_names import get_pitch_archetype_name

# TTL for cached parquet data — ensures dashboard picks up fresh precompute
# output within 5 minutes without manual cache clearing / restart.
_DATA_TTL = timedelta(minutes=5)


@st.cache_data(ttl=_DATA_TTL)
def load_projections(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_k_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=_DATA_TTL)
def load_bb_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "pitcher_bb_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=_DATA_TTL)
def load_hr_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "pitcher_hr_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=_DATA_TTL)
def load_bf_priors() -> pd.DataFrame:
    path = DASHBOARD_DIR / "bf_priors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_arsenal() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vulnerability(career: bool = False) -> pd.DataFrame:
    if career:
        path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    path = DASHBOARD_DIR / "hitter_vuln.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
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
    # Prefer sim-based counting stats (correlated joint distributions)
    sim_path = DASHBOARD_DIR / f"{player_type}_counting_sim.parquet"
    if sim_path.exists():
        return pd.read_parquet(sim_path)
    # Fallback to old rate x BF counting stats
    path = DASHBOARD_DIR / f"{player_type}_counting.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_counting_sim(player_type: str) -> pd.DataFrame:
    """Load sim-based counting stat projections with confidence tiers.

    Parameters
    ----------
    player_type : str
        ``"pitcher"`` or ``"hitter"``.

    Returns
    -------
    pd.DataFrame
        Sim projections including ``confidence_tier`` and
        ``confidence_score`` columns (if available).
    """
    path = DASHBOARD_DIR / f"{player_type}_counting_sim.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_confidence_tiers(player_type: str) -> pd.DataFrame:
    """Extract confidence tier columns from the sim parquet.

    Returns a DataFrame with the player id column, ``confidence_score``,
    and ``confidence_tier``.  Returns empty if the parquet or columns
    are missing.

    Parameters
    ----------
    player_type : str
        ``"pitcher"`` or ``"hitter"``.

    Returns
    -------
    pd.DataFrame
        Columns: player id, confidence_score, confidence_tier.
    """
    df = load_counting_sim(player_type)
    if df.empty:
        return pd.DataFrame()

    id_col = "pitcher_id" if player_type == "pitcher" else "batter_id"
    name_col = "pitcher_name" if player_type == "pitcher" else "batter_name"
    keep = [id_col]
    if name_col in df.columns:
        keep.append(name_col)
    if "confidence_score" in df.columns:
        keep.append("confidence_score")
    if "confidence_tier" in df.columns:
        keep.append("confidence_tier")

    # Return empty if no confidence columns present
    if "confidence_score" not in df.columns and "confidence_tier" not in df.columns:
        return pd.DataFrame()

    return df[keep].copy()


@st.cache_data(ttl=_DATA_TTL)
def load_game_info() -> pd.DataFrame:
    path = DASHBOARD_DIR / "game_info.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_player_teams() -> pd.DataFrame:
    """Load player-to-team mapping.

    Prefers roster.parquet (has lineup positions + starter flags).
    Falls back to player_teams.parquet for backwards compatibility.
    """
    roster_path = DASHBOARD_DIR / "roster.parquet"
    if roster_path.exists():
        return pd.read_parquet(roster_path)
    path = DASHBOARD_DIR / "player_teams.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_roster() -> pd.DataFrame:
    """Load active MLB roster from pre-computed parquet.

    Falls back to player_teams.parquet if roster parquet is missing.
    """
    path = DASHBOARD_DIR / "roster.parquet"
    if not path.exists():
        return load_player_teams()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_probable_starters() -> pd.DataFrame:
    path = DASHBOARD_DIR / "probable_starters.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_probable_starters_by_hand() -> pd.DataFrame:
    path = DASHBOARD_DIR / "probable_starters_by_hand.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_lineup_priors() -> pd.DataFrame:
    path = DASHBOARD_DIR / "lineup_priors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_location_grid() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_location_grid.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_pitch_locations() -> pd.DataFrame:
    """Raw pitch coordinates (plate_x, plate_z) for KDE density charts."""
    path = DASHBOARD_DIR / "pitcher_pitch_locations.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_zone_grid(career: bool = False) -> pd.DataFrame:
    if career:
        path = DASHBOARD_DIR / "hitter_zone_grid_career.parquet"
        if path.exists():
            return pd.read_parquet(path)
    path = DASHBOARD_DIR / "hitter_zone_grid.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_games() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_games.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_sims() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_sims.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_lineups() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_lineups.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_batter_sims() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_batter_sims.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_game_props() -> pd.DataFrame:
    path = DASHBOARD_DIR / "game_props.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_sim_log() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_sim_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_batter_sim_log() -> pd.DataFrame:
    path = DASHBOARD_DIR / "batter_sim_log.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_traditional_stats(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_traditional.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_aggressiveness() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_aggressiveness.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_efficiency() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_efficiency.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_traditional_stats_all(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_traditional_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_aggressiveness_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_aggressiveness_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_efficiency_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_efficiency_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_arsenal_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_arsenal_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vulnerability_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_vuln_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_strength_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_str_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_full_stats(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_full_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_location_grid_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_location_grid_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_zone_grid_all() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_zone_grid_all.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_preseason_injuries() -> pd.DataFrame:
    path = DASHBOARD_DIR / "preseason_injuries.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_offerings() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_offerings.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vuln_arch() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_vuln_arch.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vuln_arch_career() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_vuln_arch_career.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_cluster_metadata() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_cluster_metadata.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "pitch_archetype" in df.columns:
        df["archetype_name"] = df["pitch_archetype"].apply(get_pitch_archetype_name)
    return df


@st.cache_data(ttl=_DATA_TTL)
def load_baselines_arch() -> pd.DataFrame:
    path = DASHBOARD_DIR / "baselines_arch.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_archetypes() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_archetypes.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_archetypes() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_archetypes.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_archetype_metadata() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_archetype_metadata.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_archetype_metadata() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_archetype_metadata.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_archetype_matchup_matrix() -> pd.DataFrame:
    path = DASHBOARD_DIR / "archetype_matchup_matrix.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_update_metadata() -> dict:
    path = DASHBOARD_DIR / "update_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=_DATA_TTL)
def load_backtest(name: str) -> pd.DataFrame:
    """Load a backtest results parquet (e.g. 'pitcher_k_backtest')."""
    path = DASHBOARD_DIR / f"backtest_{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
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


@st.cache_data(ttl=_DATA_TTL)
def load_latest_weekly_snapshot(player_type: str) -> tuple[str, pd.DataFrame] | None:
    """Load the most recent weekly snapshot. Returns (date_str, df) or None."""
    snapshots = load_weekly_snapshots(player_type)
    if not snapshots:
        return None
    latest_date = max(snapshots.keys())
    return latest_date, snapshots[latest_date]


@st.cache_data(ttl=600)  # 10-minute TTL for live schedule data
def fetch_live_schedule(game_date: str | None = None) -> pd.DataFrame:
    """Fetch live schedule from MLB Stats API with short TTL cache."""
    from lib.schedule import fetch_todays_schedule
    return fetch_todays_schedule(game_date=game_date)


@st.cache_data(ttl=600)  # 10-minute TTL for live lineup data
def fetch_live_lineups(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch live lineups from MLB Stats API with short TTL cache."""
    from lib.schedule import fetch_all_lineups
    if schedule_df.empty:
        return pd.DataFrame()
    # Convert to hashable form for caching
    return fetch_all_lineups(schedule_df)


@st.cache_data(ttl=_DATA_TTL)
def load_milb_translated(player_type: str) -> pd.DataFrame:
    """Load MiLB translated stats (batters or pitchers)."""
    path = DASHBOARD_DIR / f"milb_translated_{player_type}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_milb_factors(player_type: str) -> pd.DataFrame:
    """Load MiLB translation factors (batters or pitchers)."""
    path = DASHBOARD_DIR / f"milb_{player_type}_factors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_rankings(player_type: str) -> pd.DataFrame:
    """Load precomputed rankings (hitters, pitchers, or prospects)."""
    filename = f"{player_type}_rankings.parquet"
    path = DASHBOARD_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_position_eligibility() -> pd.DataFrame:
    """Load hitter multi-position eligibility table."""
    path = DASHBOARD_DIR / "hitter_position_eligibility.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_prospect_readiness() -> pd.DataFrame:
    """Load prospect readiness scores with rankings."""
    path = DASHBOARD_DIR / "prospect_readiness.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_breakout_candidates() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitter_breakout_candidates.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_breakout_candidates() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_breakout_candidates.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def season_selector(
    key_prefix: str,
    include_career: bool = True,
    label_visibility: str = "visible",
) -> str:
    """Render a season selector and return the choice."""
    options = (
        [PROJECTION_LABEL]
        + (["Career"] if include_career else [])
        + [str(s) for s in AVAILABLE_SEASONS]
    )
    return st.selectbox(
        "Season", options, key=f"{key_prefix}_season",
        label_visibility=label_visibility,
    )


# -----------------------------------------------------------------------
# Game Simulator Data (Layer 3 v2)
# -----------------------------------------------------------------------

@st.cache_data(ttl=_DATA_TTL)
def load_hitter_k_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "hitter_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_bb_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "hitter_bb_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_hr_samples() -> dict[str, np.ndarray]:
    path = DASHBOARD_DIR / "hitter_hr_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=_DATA_TTL)
def load_exit_model():
    """Load the trained pitcher exit model.

    Returns
    -------
    ExitModel or None
        Trained model, or None if file doesn't exist.
    """
    path = DASHBOARD_DIR / "exit_model.pkl"
    if not path.exists():
        return None
    try:
        from lib.game_sim.exit_model import ExitModel
    except ImportError:
        return None
    model = ExitModel()
    model.load(path)
    return model


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_pitch_count_features() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_pitch_count_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_batter_pitch_count_features() -> pd.DataFrame:
    path = DASHBOARD_DIR / "batter_pitch_count_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_tto_profiles() -> pd.DataFrame:
    path = DASHBOARD_DIR / "tto_profiles.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_bullpen_rates() -> pd.DataFrame:
    path = DASHBOARD_DIR / "team_bullpen_rates.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_exit_tendencies() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_exit_tendencies.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Team intelligence data (ELO, profiles, rankings)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_DATA_TTL)
def load_team_elo(preseason: bool = False) -> pd.DataFrame:
    """Load team ELO ratings (end-of-season or pre-season regressed)."""
    fname = "team_elo_preseason.parquet" if preseason else "team_elo.parquet"
    path = DASHBOARD_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_elo_history() -> pd.DataFrame:
    path = DASHBOARD_DIR / "team_elo_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_profiles() -> pd.DataFrame:
    path = DASHBOARD_DIR / "team_profiles.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_rankings() -> pd.DataFrame:
    path = DASHBOARD_DIR / "team_rankings.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
