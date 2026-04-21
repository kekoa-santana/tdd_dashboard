"""Cached data loaders for the TDD Dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from config import DASHBOARD_DIR, AVAILABLE_SEASONS, PROJECTION_LABEL
from utils.archetype_names import get_pitch_archetype_name

# TTL for cached parquet data — ensures dashboard picks up fresh precompute
# output within 5 minutes without manual cache clearing / restart.
_DATA_TTL = timedelta(minutes=5)


class LazyNpzDict:
    """Dict-like wrapper that loads NPZ arrays on demand.

    Avoids materializing all posterior sample arrays into memory at once.
    The underlying ``NpzFile`` decompresses individual arrays only when
    accessed, so peak memory is proportional to the number of players
    actually viewed rather than the full file.
    """

    def __init__(self, path: Path):
        self._path = path
        self._npz: np.lib.npyio.NpzFile | None = None
        self._keys: list[str] | None = None

    def _open(self) -> None:
        if self._npz is None:
            self._npz = np.load(self._path)
            self._keys = list(self._npz.files)

    def __contains__(self, key: object) -> bool:
        self._open()
        return str(key) in self._keys  # type: ignore[arg-type]

    def __getitem__(self, key: str) -> np.ndarray:
        self._open()
        return self._npz[str(key)]  # type: ignore[index]

    def get(self, key: str, default: np.ndarray | None = None) -> np.ndarray | None:
        if str(key) in self:
            return self[key]
        return default

    def keys(self) -> list[str]:
        self._open()
        return self._keys  # type: ignore[return-value]

    def __len__(self) -> int:
        self._open()
        return len(self._keys)  # type: ignore[arg-type]

    def __bool__(self) -> bool:
        self._open()
        return len(self._keys) > 0  # type: ignore[arg-type]

    def __iter__(self):
        self._open()
        return iter(self._keys)  # type: ignore[arg-type]


@st.cache_data(ttl=timedelta(hours=1))
def load_standings(season: int | None = None) -> dict[str, tuple[int, int]]:
    """Fetch current W-L records from MLB API (cached 1 hour)."""
    from lib.schedule import fetch_standings
    return fetch_standings(season)


@st.cache_data(ttl=_DATA_TTL)
def load_projections(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(ttl=_DATA_TTL)
def load_k_samples() -> LazyNpzDict | dict:
    path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    return LazyNpzDict(path)


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


@st.cache_data(ttl=_DATA_TTL)
def load_batter_platoon_splits() -> pd.DataFrame:
    path = DASHBOARD_DIR / "batter_platoon_splits.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_glicko() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_glicko.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_batter_glicko() -> pd.DataFrame:
    path = DASHBOARD_DIR / "batter_glicko.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_gb_pct() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitcher_gb_pct.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_matchup_baselines() -> dict:
    path = DASHBOARD_DIR / "matchup_baselines.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=_DATA_TTL)
def load_stat_tier_thresholds() -> dict:
    """Per-(player_type, stat) confidence thresholds for Lock/Strong/Lean tiers.

    Built by scripts/build_prop_calibration.py from the prop backtest log.
    Structure:
        {
          "tier_targets": {"Lock": 0.65, "Strong": 0.58, "Lean": 0.53},
          "min_n_per_bucket": int,
          "thresholds": {
            "batter_H": {"n_total": int, "Lock": None, "Strong": 0.62, "Lean": 0.50},
            ...
          }
        }
    Stats whose tier value is None did not clear that tier in backtest.
    """
    path = DASHBOARD_DIR / "stat_tier_thresholds.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
def load_game_predictions() -> pd.DataFrame:
    path = DASHBOARD_DIR / "todays_game_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_dk_props() -> pd.DataFrame:
    path = DASHBOARD_DIR / "dk_props.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pp_props() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pp_props.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def dedupe_pp_against_dk(
    dk: pd.DataFrame, pp: pd.DataFrame,
) -> pd.DataFrame:
    """Drop PP rows whose line values duplicate a DK market line.

    When DraftKings offers a market line at the same value as a PrizePicks
    standard / goblin / demon, the PP row carries no additional information
    (DK has vig-bearing odds, edge, etc.) so we keep only the DK row. PP
    rows with line values that DK does not offer are retained unchanged.

    Matching is done on ``player_id + stat + line`` and, when ``game_date``
    is available on both sides, also ``game_date``.

    Parameters
    ----------
    dk, pp : pd.DataFrame
        Book lines. ``pp`` may be returned empty if all values match DK.
    """
    if pp.empty or dk.empty:
        return pp

    use_date = "game_date" in dk.columns and "game_date" in pp.columns
    key_cols = ["player_id", "stat", "line"]
    if use_date:
        key_cols.append("game_date")

    pp_work = pp.copy()
    dk_work = dk[key_cols].copy()
    # Normalize numeric line values for equality comparison
    dk_work["line"] = pd.to_numeric(dk_work["line"], errors="coerce")
    pp_work["line"] = pd.to_numeric(pp_work["line"], errors="coerce")
    # Mark PP rows that match a DK key
    marker = dk_work.drop_duplicates(subset=key_cols).assign(_dk_match=True)
    merged = pp_work.merge(marker, on=key_cols, how="left")
    kept = merged[merged["_dk_match"].isna()].drop(columns=["_dk_match"])
    return kept


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
def load_full_stats(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_full_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_advanced_stats(player_type: str) -> pd.DataFrame:
    path = DASHBOARD_DIR / f"{player_type}_advanced.parquet"
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


@st.cache_data(ttl=600)
def backfill_missing_lineups(
    schedule_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch lineups from MLB API for games missing from the cached parquet.

    Parameters
    ----------
    schedule_df : pd.DataFrame
        Today's schedule (must have ``game_pk`` column).
    lineups_df : pd.DataFrame
        Cached lineup data (may be empty or missing some game_pks).

    Returns
    -------
    pd.DataFrame
        Original lineups with any newly-fetched lineups appended.
    """
    if schedule_df.empty:
        return lineups_df

    from lib.schedule import fetch_game_lineups

    # Build set of (game_pk, team_abbr) pairs that should have lineups
    expected: set[tuple[int, str]] = set()
    for _, row in schedule_df.iterrows():
        gpk = int(row["game_pk"])
        expected.add((gpk, row["away_abbr"]))
        expected.add((gpk, row["home_abbr"]))

    # Determine which teams are missing from cached lineups
    if not lineups_df.empty and "team_abbr" in lineups_df.columns:
        cached = {(int(r["game_pk"]), r["team_abbr"]) for _, r in lineups_df.iterrows()}
    else:
        cached = set()

    missing_pairs = expected - cached
    if not missing_pairs:
        return lineups_df

    # Fetch only the games that have incomplete lineups
    missing_gpks = {gpk for gpk, _ in missing_pairs}
    fetched: list[pd.DataFrame] = []
    for gpk in missing_gpks:
        lu = fetch_game_lineups(gpk)
        if not lu.empty:
            # Only keep rows for teams that were actually missing
            missing_teams = {t for g, t in missing_pairs if g == gpk}
            lu = lu[lu["team_abbr"].isin(missing_teams)]
            if not lu.empty:
                fetched.append(lu)

    if not fetched:
        return lineups_df

    backfilled = pd.concat(fetched, ignore_index=True)
    if lineups_df.empty:
        return backfilled
    return pd.concat([lineups_df, backfilled], ignore_index=True)


@st.cache_data(ttl=600)  # 10-minute TTL for live game stats
def fetch_live_boxscores(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch live player stats for in-progress/final games."""
    from lib.schedule import fetch_live_boxscores as _fetch
    if schedule_df.empty:
        return pd.DataFrame()
    return _fetch(schedule_df)


@st.cache_data(ttl=_DATA_TTL)
def load_milb_factors(player_type: str) -> pd.DataFrame:
    """Load MiLB translation factors (batters or pitchers)."""
    path = DASHBOARD_DIR / f"milb_{player_type}_factors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_rankings(player_type: str) -> pd.DataFrame:
    """Load live/internal MLB rankings or prospect tables from parquet.

    For hitters and pitchers this reads ``{player_type}_rankings.parquet``,
    which the projection repo refreshes on an in-season schedule. Use
    :func:`load_core_rankings` for the stable Player Rankings page lists.
    """
    filename = f"{player_type}_rankings.parquet"
    path = DASHBOARD_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitters_daily_standouts() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitters_daily_standouts.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitchers_daily_standouts() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitchers_daily_standouts.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitters_weekly_form() -> pd.DataFrame:
    path = DASHBOARD_DIR / "hitters_weekly_form.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitchers_weekly_form() -> pd.DataFrame:
    path = DASHBOARD_DIR / "pitchers_weekly_form.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


_CORE_RANKING_FILES = {
    "hitters": "hitters_core_rankings.parquet",
    "pitchers": "pitchers_core_rankings.parquet",
}


@st.cache_data(ttl=_DATA_TTL)
def load_core_rankings(player_type: str) -> pd.DataFrame:
    """Load frozen core MLB rankings (preseason contract, not weekly-refreshed).

    Written by ``player_profiles`` precompute as ``hitters_core_rankings.parquet``
    and ``pitchers_core_rankings.parquet``. Same composite columns as live
    rankings at build time, plus ``rank_type``, ``core_anchor_season``,
    ``core_projection_season``.

    Parameters
    ----------
    player_type
        ``'hitters'`` or ``'pitchers'``.
    """
    fname = _CORE_RANKING_FILES.get(player_type)
    if not fname:
        return pd.DataFrame()
    path = DASHBOARD_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


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

class GameSimSampleStore:
    """Lazy loader for precomputed game sim sample arrays.

    The NPZ file stores keys like ``{game_pk}_{player_id}_{stat}``
    (e.g. ``745123_543210_k``).  This wrapper provides a dict-like
    ``get(game_pk, player_id)`` that returns
    ``{"k": array, "bb": array, ...}`` or ``None``.
    """

    def __init__(self, path: Path, stat_names: list[str]):
        self._path = path
        self._stat_names = stat_names
        self._npz: np.lib.npyio.NpzFile | None = None
        self._keys: set[str] | None = None

    def _open(self) -> None:
        if self._npz is None:
            self._npz = np.load(self._path)
            self._keys = set(self._npz.files)

    def get(self, game_pk: int, player_id: int) -> dict[str, np.ndarray] | None:
        self._open()
        prefix = f"{game_pk}_{player_id}"
        first_key = f"{prefix}_{self._stat_names[0]}"
        if first_key not in self._keys:  # type: ignore[operator]
            return None
        return {
            stat: self._npz[f"{prefix}_{stat}"]  # type: ignore[index]
            for stat in self._stat_names
            if f"{prefix}_{stat}" in self._keys  # type: ignore[operator]
        }

    def __bool__(self) -> bool:
        self._open()
        return len(self._keys) > 0  # type: ignore[arg-type]


@st.cache_resource(ttl=_DATA_TTL)
def load_pitcher_game_sim_samples() -> GameSimSampleStore | None:
    path = DASHBOARD_DIR / "pitcher_game_sim_samples.npz"
    if not path.exists():
        return None
    return GameSimSampleStore(path, ["k", "bb", "h", "hr", "outs"])


@st.cache_resource(ttl=_DATA_TTL)
def load_batter_game_sim_samples() -> GameSimSampleStore | None:
    path = DASHBOARD_DIR / "batter_game_sim_samples.npz"
    if not path.exists():
        return None
    return GameSimSampleStore(path, ["k", "bb", "h", "hr"])


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


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_grade_ci() -> pd.DataFrame:
    """Load hitter scouting grade confidence intervals.

    Returns
    -------
    pd.DataFrame
        Per-player grade CIs with ``player_id``, ``grade_*_lo``,
        ``grade_*_hi``, and ``diamond_rating_lo/hi`` columns.
    """
    path = DASHBOARD_DIR / "hitter_grade_ci.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_grade_ci() -> pd.DataFrame:
    """Load pitcher scouting grade confidence intervals.

    Returns
    -------
    pd.DataFrame
        Per-player grade CIs with ``player_id``, ``grade_*_lo``,
        ``grade_*_hi``, and ``diamond_rating_lo/hi`` columns.
    """
    path = DASHBOARD_DIR / "pitcher_grade_ci.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
