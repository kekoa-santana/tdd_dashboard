"""Cached data loaders for the TDD Dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from config import DASHBOARD_DIR, AVAILABLE_SEASONS, PROJECTION_LABEL, CURRENT_SEASON
from services.artifacts import artifact_path, list_artifacts
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
    path = artifact_path(f"{player_type}_projections.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(ttl=_DATA_TTL)
def load_k_samples() -> LazyNpzDict | dict:
    path = artifact_path("pitcher_k_samples.npz")
    if not path.exists():
        return {}
    return LazyNpzDict(path)


@st.cache_data(ttl=_DATA_TTL)
def load_bf_priors() -> pd.DataFrame:
    path = artifact_path("bf_priors.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_arsenal() -> pd.DataFrame:
    path = artifact_path("pitcher_arsenal.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_arsenal_by_stand() -> pd.DataFrame:
    """Pitcher arsenal splits by batter hand (L/R usage, whiff, xwOBA)."""
    path = artifact_path("pitcher_arsenal_by_stand.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_putaway() -> pd.DataFrame:
    """Pitcher 2-strike putaway pitch selection and location by batter hand."""
    path = artifact_path("pitcher_putaway.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_game_logs() -> pd.DataFrame:
    """Pitcher game-by-game stats (K, BB, IP, pitches, etc.)."""
    path = artifact_path("pitcher_game_logs.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_advanced_stats() -> pd.DataFrame:
    """Pitcher advanced Statcast metrics (zone%, chase%, whiff%, etc.)."""
    path = artifact_path("pitcher_advanced.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_platoon_bb() -> pd.DataFrame:
    """Pitcher BB rate split by batter hand (LHB/RHB)."""
    path = artifact_path("pitcher_platoon_bb.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vulnerability(career: bool = False) -> pd.DataFrame:
    if career:
        path = artifact_path("hitter_vuln_career.parquet")
        if path.exists():
            return pd.read_parquet(path)
    path = artifact_path("hitter_vuln.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_strength(career: bool = False) -> pd.DataFrame:
    if career:
        path = artifact_path("hitter_str_career.parquet")
        if path.exists():
            return pd.read_parquet(path)
    path = artifact_path("hitter_str.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_batter_platoon_splits() -> pd.DataFrame:
    path = artifact_path("batter_platoon_splits.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_gb_pct() -> pd.DataFrame:
    path = artifact_path("pitcher_gb_pct.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


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
    path = artifact_path("stat_tier_thresholds.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_counting(player_type: str) -> pd.DataFrame:
    # Prefer sim-based counting stats (correlated joint distributions)
    sim_path = artifact_path(f"{player_type}_counting_sim.parquet")
    if sim_path.exists():
        return pd.read_parquet(sim_path)
    # Fallback to old rate x BF counting stats
    path = artifact_path(f"{player_type}_counting.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_preseason_projections(player_type: str) -> pd.DataFrame:
    """Frozen preseason rate projections (K%, BB%, HR rate)."""
    path = artifact_path(
        f"snapshots/{player_type}_projections_{CURRENT_SEASON}_preseason.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_preseason_breakout_candidates(player_type: str) -> pd.DataFrame:
    """Frozen preseason breakout candidate list, as called before Opening Day."""
    path = artifact_path(
        f"snapshots/{player_type}_breakout_candidates_{CURRENT_SEASON}_preseason.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_breakout_calibration() -> dict:
    """Platt scaling that maps raw breakout probabilities onto observed rates.

    Written by scripts/calibrate_breakouts.py. Empty dict when absent, in
    which case callers should show the raw score rather than a probability.
    """
    path = artifact_path("breakout_calibration.json")
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def calibrated_breakout_prob(probs, player_type: str, calibration: dict | None = None):
    """Apply the fitted calibration to raw breakout probabilities.

    Returns the input unchanged when no calibration has been fitted, so the
    page degrades to the model's own numbers instead of failing.
    """
    calibration = calibration if calibration is not None else load_breakout_calibration()
    params = calibration.get(player_type, {}) if calibration else {}
    a, b = params.get("platt_a"), params.get("platt_b")
    values = pd.to_numeric(pd.Series(probs), errors="coerce")
    if a is None or b is None:
        return values
    clipped = values.clip(1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))


@st.cache_data(ttl=_DATA_TTL)
def load_preseason_counting(player_type: str) -> pd.DataFrame:
    """Frozen preseason counting projections (rate x volume, SIERA-based ERA)."""
    path = artifact_path(
        f"snapshots/{player_type}_counting_{CURRENT_SEASON}_preseason.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_preseason_counting_sim(player_type: str) -> pd.DataFrame:
    """Load the frozen preseason season-long sim projections.

    These were generated before Opening Day from 2018-2025 data only, so
    they never see how a player is performing in the current season. The
    live ``*_counting_sim.parquet`` files are conjugate-updated with
    in-season results and are not interchangeable with these.
    """
    path = artifact_path(f"snapshots/{player_type}_counting_sim_{CURRENT_SEASON}_preseason.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_player_teams() -> pd.DataFrame:
    """Load player-to-team mapping.

    Prefers roster.parquet (has lineup positions + starter flags).
    Falls back to player_teams.parquet for backwards compatibility.
    """
    roster_path = artifact_path("roster.parquet")
    if roster_path.exists():
        return pd.read_parquet(roster_path)
    path = artifact_path("player_teams.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_roster() -> pd.DataFrame:
    """Load active MLB roster from pre-computed parquet.

    Falls back to player_teams.parquet if roster parquet is missing.
    """
    path = artifact_path("roster.parquet")
    if not path.exists():
        return load_player_teams()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_probable_starters() -> pd.DataFrame:
    path = artifact_path("probable_starters.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_location_grid() -> pd.DataFrame:
    path = artifact_path("pitcher_location_grid.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_pitch_locations() -> pd.DataFrame:
    """Raw pitch coordinates (plate_x, plate_z) for KDE density charts."""
    path = artifact_path("pitcher_pitch_locations.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_zone_grid(career: bool = False) -> pd.DataFrame:
    if career:
        path = artifact_path("hitter_zone_grid_career.parquet")
        if path.exists():
            return pd.read_parquet(path)
    path = artifact_path("hitter_zone_grid.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_games() -> pd.DataFrame:
    path = artifact_path("todays_games.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_lineups() -> pd.DataFrame:
    path = artifact_path("todays_lineups.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_todays_batter_sims() -> pd.DataFrame:
    path = artifact_path("todays_batter_sims.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_game_props() -> pd.DataFrame:
    path = artifact_path("game_props.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_prop_attribution() -> pd.DataFrame:
    """Per-driver projection attribution (game_prop_attribution.parquet).

    Keyed on (game_pk, player_id, stat); columns include baseline, driver_self,
    driver_opp, tto, umpire, catcher, park, weather, residual, expected, volume.
    Empty if the parquet is absent (older precompute).
    """
    path = artifact_path("game_prop_attribution.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_game_predictions() -> pd.DataFrame:
    path = artifact_path("todays_game_predictions.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_sim_log() -> pd.DataFrame:
    path = artifact_path("pitcher_sim_log.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_batter_sim_log() -> pd.DataFrame:
    path = artifact_path("batter_sim_log.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_traditional_stats(player_type: str) -> pd.DataFrame:
    path = artifact_path(f"{player_type}_traditional.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_aggressiveness() -> pd.DataFrame:
    path = artifact_path("hitter_aggressiveness.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_efficiency() -> pd.DataFrame:
    path = artifact_path("pitcher_efficiency.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_traditional_stats_all(player_type: str) -> pd.DataFrame:
    path = artifact_path(f"{player_type}_traditional_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_aggressiveness_all() -> pd.DataFrame:
    path = artifact_path("hitter_aggressiveness_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_efficiency_all() -> pd.DataFrame:
    path = artifact_path("pitcher_efficiency_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_arsenal_all() -> pd.DataFrame:
    path = artifact_path("pitcher_arsenal_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vulnerability_all() -> pd.DataFrame:
    path = artifact_path("hitter_vuln_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_full_stats(player_type: str) -> pd.DataFrame:
    path = artifact_path(f"{player_type}_full_stats.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_advanced_stats(player_type: str) -> pd.DataFrame:
    path = artifact_path(f"{player_type}_advanced.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_location_grid_all() -> pd.DataFrame:
    path = artifact_path("pitcher_location_grid_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_zone_grid_all() -> pd.DataFrame:
    path = artifact_path("hitter_zone_grid_all.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_preseason_injuries() -> pd.DataFrame:
    path = artifact_path("preseason_injuries.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_offerings() -> pd.DataFrame:
    path = artifact_path("pitcher_offerings.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vuln_arch() -> pd.DataFrame:
    path = artifact_path("hitter_vuln_arch.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_vuln_arch_career() -> pd.DataFrame:
    path = artifact_path("hitter_vuln_arch_career.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_cluster_metadata() -> pd.DataFrame:
    path = artifact_path("pitcher_cluster_metadata.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "pitch_archetype" in df.columns:
        df["archetype_name"] = df["pitch_archetype"].apply(get_pitch_archetype_name)
    return df


@st.cache_data(ttl=_DATA_TTL)
def load_baselines_arch() -> pd.DataFrame:
    path = artifact_path("baselines_arch.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_archetypes() -> pd.DataFrame:
    path = artifact_path("hitter_archetypes.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_archetypes() -> pd.DataFrame:
    path = artifact_path("pitcher_archetypes.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_archetype_matchup_matrix() -> pd.DataFrame:
    path = artifact_path("archetype_matchup_matrix.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_update_metadata() -> dict:
    path = artifact_path("update_metadata.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=_DATA_TTL)
def load_backtest(name: str) -> pd.DataFrame:
    """Load a backtest results parquet (e.g. 'pitcher_k_backtest')."""
    path = artifact_path(f"backtest_{name}.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_weekly_snapshots(player_type: str) -> dict[str, pd.DataFrame]:
    """Load all weekly snapshots for a player type.

    Returns {date_str: DataFrame} sorted by date.
    """
    prefix = f"{player_type}_projections_"
    result = {}
    # Object storage cannot glob, so enumerate via the published index; this
    # falls back to walking the local directory during development.
    for name in list_artifacts("snapshots/weekly"):
        stem = name.rsplit("/", 1)[-1]
        if not (stem.startswith(prefix) and stem.endswith(".parquet")):
            continue
        path = artifact_path(name)
        if not path.exists():
            continue
        result[stem[len(prefix):-len(".parquet")]] = pd.read_parquet(path)
    return dict(sorted(result.items()))


@st.cache_data(ttl=600)  # 10-minute TTL for live schedule data
def fetch_live_schedule(
    game_date: str | None = None,
    include_weather: bool = True,
) -> pd.DataFrame:
    """Fetch live schedule from MLB Stats API with short TTL cache."""
    from lib.schedule import fetch_todays_schedule
    return fetch_todays_schedule(
        game_date=game_date, include_weather=include_weather,
    )


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

    # Fetch only the games that have incomplete lineups (parallel)
    missing_gpks = {gpk for gpk, _ in missing_pairs}
    fetched: list[pd.DataFrame] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_game_lineups, gpk): gpk for gpk in missing_gpks}
        for future in as_completed(futures):
            gpk = futures[future]
            try:
                lu = future.result()
            except Exception:
                continue
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
    path = artifact_path(f"milb_{player_type}_factors.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_prospect_comps_batters() -> pd.DataFrame:
    """Load prospect-to-MLB batter comparables."""
    path = artifact_path("prospect_comps_batters.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_prospect_comps_pitchers() -> pd.DataFrame:
    """Load prospect-to-MLB pitcher comparables."""
    path = artifact_path("prospect_comps_pitchers.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_milb_priors() -> pd.DataFrame:
    """Load MiLB Bayesian model priors (distributional, logit scale)."""
    path = artifact_path("milb_priors.parquet")
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
    path = artifact_path(filename)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitters_daily_standouts() -> pd.DataFrame:
    path = artifact_path("hitters_daily_standouts.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitchers_daily_standouts() -> pd.DataFrame:
    path = artifact_path("pitchers_daily_standouts.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitters_weekly_form() -> pd.DataFrame:
    path = artifact_path("hitters_weekly_form.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitchers_weekly_form() -> pd.DataFrame:
    path = artifact_path("pitchers_weekly_form.parquet")
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
    path = artifact_path(fname)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_prospect_readiness() -> pd.DataFrame:
    """Load prospect readiness scores with rankings."""
    path = artifact_path("prospect_readiness.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hitter_breakout_candidates() -> pd.DataFrame:
    path = artifact_path("hitter_breakout_candidates.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_pitcher_breakout_candidates() -> pd.DataFrame:
    path = artifact_path("pitcher_breakout_candidates.parquet")
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


# ---------------------------------------------------------------------------
# Team intelligence data (ELO, profiles, rankings)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_DATA_TTL)
def load_team_elo(preseason: bool = False) -> pd.DataFrame:
    """Load team ELO ratings (end-of-season or pre-season regressed)."""
    fname = "team_elo_preseason.parquet" if preseason else "team_elo.parquet"
    path = artifact_path(fname)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_elo_history() -> pd.DataFrame:
    path = artifact_path("team_elo_history.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_profiles() -> pd.DataFrame:
    path = artifact_path("team_profiles.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_rankings() -> pd.DataFrame:
    path = artifact_path("team_rankings.parquet")
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
    path = artifact_path("hitter_grade_ci.parquet")
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
    path = artifact_path("pitcher_grade_ci.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Park / Umpire / Weather / Bullpen loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_DATA_TTL)
def load_park_factors() -> pd.DataFrame:
    path = artifact_path("park_factors.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_hr_park_factors() -> pd.DataFrame:
    path = artifact_path("hr_park_factors.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_umpire_tendencies() -> pd.DataFrame:
    path = artifact_path("umpire_tendencies.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_weather_effects() -> pd.DataFrame:
    path = artifact_path("weather_effects.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_team_bullpen_profiles() -> pd.DataFrame:
    path = artifact_path("team_bullpen_profiles.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_reliever_rankings() -> pd.DataFrame:
    path = artifact_path("reliever_rankings.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=_DATA_TTL)
def load_news_feed() -> pd.DataFrame:
    path = artifact_path("news_feed.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
