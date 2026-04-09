#!/usr/bin/env python
"""
Dashboard post-update script.

Runs the projection engine's in-season update (conjugate updating,
schedule fetch, game simulations), then handles dashboard-specific
bookkeeping: weekly snapshots, update metadata, and artifact manifest.

All model work (DB queries, conjugate updating, K samples, matchup
simulation) lives in the player_profiles repo. This script delegates
to it via subprocess.

Usage
-----
    python scripts/update_in_season.py                    # today's date (full update)
    python scripts/update_in_season.py --date 2026-04-15  # specific date
    python scripts/update_in_season.py --skip-schedule    # skip API calls
    python scripts/update_in_season.py --skip-engine      # skip projection engine, just do bookkeeping
    python scripts/update_in_season.py --snapshot          # force a weekly snapshot
    python scripts/update_in_season.py --schedule-only    # refresh schedule/lineups/sims only (hourly mode)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CURRENT_SEASON  # noqa: E402

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
PLAYER_PROFILES_DIR = PROJECT_ROOT.parent / "player_profiles"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASON = CURRENT_SEASON

# Major transaction keywords that trigger a precompute refresh
_MAJOR_MOVE_KEYWORDS = [
    "Injured List",
    "Disabled List",
    "Trade",
    "Traded",
    "Designated for Assignment",
    "Released",
    "Recalled From",
    "Selected to Roster",
    "Placed on Waivers",
]

_PRECOMPUTE_GROUPS = ["team", "rankings", "game_data", "health"]


# ---------------------------------------------------------------------------
# Weather / umpire parsing helpers
# ---------------------------------------------------------------------------

def _parse_temp_bucket(temp_str: object) -> str:
    """Convert temperature string to weather bucket."""
    if not temp_str:
        return "warm"
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        return "warm"
    if temp < 55:
        return "cold"
    if temp < 70:
        return "cool"
    if temp < 85:
        return "warm"
    return "hot"


def _parse_wind_category(wind_str: object) -> str:
    """Convert wind string to category."""
    if not wind_str:
        return "none"
    w = str(wind_str).lower()
    if "calm" in w or w.strip() == "":
        return "none"
    if "out" in w:
        return "out"
    if "in from" in w or "in," in w:
        return "in"
    if "l to r" in w or "r to l" in w:
        return "cross"
    return "none"


# ---------------------------------------------------------------------------
# Roster export
# ---------------------------------------------------------------------------

def export_roster() -> bool:
    """Export production.dim_roster joined with dim_team to a dashboard parquet.

    Returns True on success, False on failure.
    """
    try:
        from lib.db import read_sql
        import pandas as pd

        df = read_sql("""
            SELECT dr.player_id, dr.player_name, dr.org_id, dr.roster_status,
                   dr.primary_position, dr.is_starter,
                   dt.abbreviation AS team_abbr,
                   dt.league, dt.division, dt.team_name
            FROM production.dim_roster dr
            JOIN production.dim_team dt ON dr.org_id = dt.team_id
            WHERE dr.level = 'MLB'
              AND dr.roster_status NOT IN ('released', 'restricted', 'minors')
        """)
        df.to_parquet(DASHBOARD_DIR / "roster.parquet", index=False)
        logger.info("Exported roster: %d players", len(df))
        return True
    except Exception as e:
        logger.warning("Roster export failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Roster move detection → precompute trigger
# ---------------------------------------------------------------------------

def _trigger_precompute(groups: list[str]) -> bool:
    """Run precompute_dashboard_data.py for the given groups.

    Returns True on success.
    """
    engine_python = PLAYER_PROFILES_DIR / "myenv" / "Scripts" / "python.exe"
    if not engine_python.exists():
        engine_python = PLAYER_PROFILES_DIR / "myenv" / "bin" / "python"
    if not engine_python.exists():
        logger.error("player_profiles virtualenv not found")
        return False

    script = PLAYER_PROFILES_DIR / "scripts" / "precompute_dashboard_data.py"
    if not script.exists():
        logger.error("precompute_dashboard_data.py not found at %s", script)
        return False

    groups_str = ",".join(groups)
    cmd = [str(engine_python), str(script), "--include", groups_str]
    logger.info("Triggering precompute: %s", groups_str)
    result = subprocess.run(cmd, cwd=str(PLAYER_PROFILES_DIR))

    if result.returncode != 0:
        logger.error("Precompute exited with code %d", result.returncode)
        return False

    logger.info("Precompute refresh completed: %s", groups_str)
    return True


def check_roster_moves(game_date: str) -> bool:
    """Check for major roster moves and trigger precompute if needed.

    Fetches today's transactions from the MLB API, compares against
    previously seen transaction IDs, and triggers a precompute refresh
    of team, rankings, game_data, and health if new major moves found.

    Returns True if a precompute was triggered.
    """
    from lib.schedule import fetch_recent_transactions

    state_path = DASHBOARD_DIR / "roster_move_state.json"

    # Load state; reset if new day
    known_ids: set[int] = set()
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        if state.get("game_date") == game_date:
            known_ids = {int(x) for x in state.get("known_transaction_ids", [])}
        else:
            logger.info("New game date %s — resetting roster move state", game_date)

    # Fetch today's transactions
    txns = fetch_recent_transactions(game_date)
    if txns.empty:
        logger.info("No transactions for %s", game_date)
        return False

    # Filter for major moves
    major = txns[
        txns["type_desc"].str.contains(
            "|".join(_MAJOR_MOVE_KEYWORDS), case=False, na=False,
        )
    ]

    # Collect all valid IDs to persist
    all_ids = set(txns["transaction_id"].dropna().astype(int).tolist())

    if major.empty:
        logger.info("No major roster moves for %s", game_date)
        _save_roster_state(state_path, game_date, known_ids | all_ids)
        return False

    # Check for NEW major moves
    major_ids = set(major["transaction_id"].dropna().astype(int).tolist())
    new_ids = major_ids - known_ids

    if not new_ids:
        logger.info("No new major moves (all %d already processed)", len(major_ids))
        _save_roster_state(state_path, game_date, known_ids | all_ids)
        return False

    new_major = major[major["transaction_id"].isin(new_ids)]
    for _, move in new_major.iterrows():
        logger.info(
            "NEW ROSTER MOVE: %s — %s",
            move.get("player_name", "Unknown"),
            move.get("description", move.get("type_desc", "")),
        )

    logger.info(
        "Triggering precompute for %d new major moves...", len(new_major),
    )
    _trigger_precompute(_PRECOMPUTE_GROUPS)

    _save_roster_state(state_path, game_date, known_ids | all_ids)
    return True


def _save_roster_state(
    state_path: Path, game_date: str, known_ids: set[int],
) -> None:
    """Persist known transaction IDs for dedup across 30-min cycles."""
    with open(state_path, "w") as f:
        json.dump(
            {
                "game_date": game_date,
                "last_check": datetime.now().isoformat(),
                "known_transaction_ids": sorted(known_ids),
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Schedule-only refresh (hourly mode)
# ---------------------------------------------------------------------------

def _lineups_changed(new_lineups: "pd.DataFrame", schedule: "pd.DataFrame") -> bool:
    """Compare freshly fetched lineups/starters against what is on disk.

    Returns True if any pitcher or lineup batter changed, meaning sims
    need to be re-run.
    """
    import pandas as pd

    old_lu_path = DASHBOARD_DIR / "todays_lineups.parquet"
    old_sched_path = DASHBOARD_DIR / "todays_games.parquet"

    # --- Check pitcher changes via schedule ---
    if old_sched_path.exists() and not schedule.empty:
        old_sched = pd.read_parquet(old_sched_path)
        pitcher_cols = [c for c in ["away_pitcher_id", "home_pitcher_id"] if c in schedule.columns and c in old_sched.columns]
        if pitcher_cols and "game_pk" in old_sched.columns:
            old_pitchers = old_sched.set_index("game_pk")[pitcher_cols].sort_index()
            new_pitchers = schedule.set_index("game_pk")[pitcher_cols].sort_index()
            # Only compare games present in both
            common = old_pitchers.index.intersection(new_pitchers.index)
            if len(common) > 0:
                old_cmp = old_pitchers.loc[common].fillna(0)
                new_cmp = new_pitchers.loc[common].fillna(0)
                if not old_cmp.equals(new_cmp):
                    logger.info("Pitcher change detected")
                    return True
            # New games added
            if len(schedule) != len(old_sched):
                logger.info("Game count changed (%d -> %d)", len(old_sched), len(schedule))
                return True

    # --- Check lineup batter changes ---
    if new_lineups.empty:
        return False
    if not old_lu_path.exists():
        logger.info("No previous lineups on disk")
        return True

    old_lu = pd.read_parquet(old_lu_path)
    if old_lu.empty:
        return True

    id_col = "batter_id" if "batter_id" in new_lineups.columns else "player_id"
    old_id_col = "batter_id" if "batter_id" in old_lu.columns else "player_id"
    if id_col not in new_lineups.columns or old_id_col not in old_lu.columns:
        return True

    old_by_game = old_lu.groupby("game_pk")[old_id_col].apply(lambda s: frozenset(s.dropna().astype(int)))
    new_by_game = new_lineups.groupby("game_pk")[id_col].apply(lambda s: frozenset(s.dropna().astype(int)))

    for gpk in new_by_game.index:
        if gpk not in old_by_game.index:
            logger.info("New lineup for game %s", gpk)
            return True
        if new_by_game[gpk] != old_by_game[gpk]:
            logger.info("Lineup change detected for game %s", gpk)
            return True

    return False


def run_schedule_refresh(game_date: str) -> bool:
    """Fetch schedule/lineups and re-run sims using existing projections.

    This is the lightweight 30-min mode: no DB queries, no conjugate
    updates -- just MLB API calls and Monte Carlo sims with whatever
    projections and K samples are already on disk.

    Uses the PA-by-PA game simulator (Layer 3 v2) for multi-stat
    projections: K, BB, H, HR, IP, pitches, fantasy points.

    Returns True if sims were re-run (lineup/pitcher change detected),
    False if skipped (no changes).
    """
    import numpy as np
    import pandas as pd
    from lib.schedule import fetch_todays_schedule, fetch_all_lineups
    from lib.game_sim.simulator import simulate_game
    from lib.game_sim.exit_model import ExitModel
    from lib.game_sim.tto_model import build_all_tto_lifts
    from lib.game_sim.pitch_count_model import build_pitch_count_features
    from lib.game_sim.fantasy_scoring import compute_pitcher_fantasy
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    # Fetch schedule
    logger.info("Fetching schedule for %s...", game_date)
    schedule = fetch_todays_schedule(game_date)
    if schedule.empty:
        logger.info("No games scheduled for %s", game_date)
        return False

    # Fetch lineups
    lineups = fetch_all_lineups(schedule)

    # Check for changes before committing to full sim run
    changed = _lineups_changed(lineups, schedule)

    # Always save fresh schedule and lineups
    schedule.to_parquet(DASHBOARD_DIR / "todays_games.parquet", index=False)
    logger.info("Saved schedule: %d games", len(schedule))

    # Tag API lineups with source
    if not lineups.empty:
        lineups["lineup_source"] = "api"

    # --- Backfill roster batters for games missing API lineups ---
    roster_path = DASHBOARD_DIR / "roster.parquet"
    roster_df = pd.read_parquet(roster_path) if roster_path.exists() else pd.DataFrame()
    if not roster_df.empty and not schedule.empty:
        api_game_teams: set[tuple[int, str]] = set()
        if not lineups.empty:
            for _, _lr in lineups.iterrows():
                api_game_teams.add((int(_lr["game_pk"]), str(_lr["team_abbr"])))

        roster_rows: list[pd.DataFrame] = []
        _pitcher_pos = {"SP", "RP", "P"}

        # Detect two-way players (pitchers who also have hitter projections)
        _h_proj_path = DASHBOARD_DIR / "hitter_projections.parquet"
        _two_way_ids: set[int] = set()
        if _h_proj_path.exists():
            _h_proj = pd.read_parquet(_h_proj_path, columns=["batter_id"])
            _hitter_ids = set(_h_proj["batter_id"].dropna().astype(int))
            _pitcher_roster = roster_df[
                roster_df["primary_position"].isin(_pitcher_pos)
            ]
            for _, _pr in _pitcher_roster.iterrows():
                if int(_pr["player_id"]) in _hitter_ids:
                    _two_way_ids.add(int(_pr["player_id"]))
            if _two_way_ids:
                logger.info("Two-way players detected: %s", _two_way_ids)

        for _, _g in schedule.iterrows():
            _gpk = int(_g["game_pk"])
            for _side in ["away", "home"]:
                _team_abbr = _g.get(f"{_side}_abbr", "")
                _team_id = _g.get(f"{_side}_team_id")
                if (_gpk, _team_abbr) in api_game_teams:
                    continue
                # Active position players who have played recently
                # Include two-way players even though their primary_position is a pitcher role
                _team_roster = roster_df[
                    (roster_df["team_abbr"] == _team_abbr)
                    & (roster_df["roster_status"] == "active")
                    & (
                        (~roster_df["primary_position"].isin(_pitcher_pos))
                        | (roster_df["player_id"].isin(_two_way_ids))
                    )
                ].copy()
                if _team_roster.empty:
                    continue
                # Sort by recency, take top 14
                if "last_game_date" in _team_roster.columns:
                    _team_roster = _team_roster.sort_values(
                        "last_game_date", ascending=False, na_position="last",
                    )
                _team_roster = _team_roster.head(14)
                _r_lu = pd.DataFrame({
                    "batter_id": _team_roster["player_id"].values,
                    "batter_name": _team_roster["player_name"].values,
                    "batting_order": range(1, len(_team_roster) + 1),
                    "game_pk": _gpk,
                    "team_id": _team_id,
                    "team_abbr": _team_abbr,
                    "lineup_source": "roster",
                })
                roster_rows.append(_r_lu)

        if roster_rows:
            roster_combined = pd.concat(roster_rows, ignore_index=True)
            lineups = (
                pd.concat([lineups, roster_combined], ignore_index=True)
                if not lineups.empty else roster_combined
            )
            logger.info("Backfilled roster batters: %d players across %d team-games",
                        len(roster_combined), len(roster_rows))

    if not lineups.empty:
        lineups.to_parquet(DASHBOARD_DIR / "todays_lineups.parquet", index=False)
        n_api = (lineups["lineup_source"] == "api").sum()
        n_roster = (lineups["lineup_source"] == "roster").sum()
        logger.info("Saved lineups: %d batters (%d API, %d roster) across %d games",
                     len(lineups), n_api, n_roster, lineups["game_pk"].nunique())
    else:
        logger.info("No lineups available yet")

    if not changed:
        logger.info("No lineup or pitcher changes detected -- skipping sims")
        return False

    # --- Load existing projections and posterior samples ---
    p_path = DASHBOARD_DIR / "pitcher_projections.parquet"
    k_path = DASHBOARD_DIR / "pitcher_k_samples.npz"

    if not p_path.exists() or not k_path.exists():
        logger.warning("Missing projections or K samples — cannot simulate. "
                        "Run a full update first.")
        return

    pitcher_proj = pd.read_parquet(p_path)

    k_data = np.load(k_path)
    k_samples = {k: k_data[k] for k in k_data.files}

    bb_path = DASHBOARD_DIR / "pitcher_bb_samples.npz"
    bb_samples: dict[str, np.ndarray] = {}
    if bb_path.exists():
        _bb = np.load(bb_path)
        bb_samples = {k: _bb[k] for k in _bb.files}

    hr_path = DASHBOARD_DIR / "pitcher_hr_samples.npz"
    hr_samples: dict[str, np.ndarray] = {}
    if hr_path.exists():
        _hr = np.load(hr_path)
        hr_samples = {k: _hr[k] for k in _hr.files}

    arsenal_path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    vuln_path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
    arsenal_df = pd.read_parquet(arsenal_path) if arsenal_path.exists() else pd.DataFrame()
    vuln_df = pd.read_parquet(vuln_path) if vuln_path.exists() else pd.DataFrame()

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    # --- Load game simulator component data ---
    exit_model = ExitModel()
    exit_pkl = DASHBOARD_DIR / "exit_model.pkl"
    if exit_pkl.exists():
        exit_model.load(exit_pkl)
        logger.info("Loaded exit model")
    else:
        logger.warning("No exit model — using fallback sigmoid")

    pitcher_pc_path = DASHBOARD_DIR / "pitcher_pitch_count_features.parquet"
    batter_pc_path = DASHBOARD_DIR / "batter_pitch_count_features.parquet"
    pitcher_pc = pd.read_parquet(pitcher_pc_path) if pitcher_pc_path.exists() else pd.DataFrame()
    batter_pc = pd.read_parquet(batter_pc_path) if batter_pc_path.exists() else pd.DataFrame()

    tto_path = DASHBOARD_DIR / "tto_profiles.parquet"
    tto_profiles = pd.read_parquet(tto_path) if tto_path.exists() else pd.DataFrame()

    tend_path = DASHBOARD_DIR / "pitcher_exit_tendencies.parquet"
    exit_tendencies = pd.read_parquet(tend_path) if tend_path.exists() else pd.DataFrame()

    # --- Load depth chart fallback for pre-lineup sims ---
    dc_path = DASHBOARD_DIR / "probable_starters_by_hand.parquet"
    depth_chart = pd.read_parquet(dc_path) if dc_path.exists() else pd.DataFrame()
    if not depth_chart.empty:
        logger.info("Loaded depth charts: %d entries", len(depth_chart))

    # Build pitch_hand lookup from pitcher projections
    pitch_hand_lookup: dict[int, str] = {}
    if not pitcher_proj.empty and "pitch_hand" in pitcher_proj.columns:
        for _, pr in pitcher_proj[["pitcher_id", "pitch_hand"]].drop_duplicates().iterrows():
            pitch_hand_lookup[int(pr["pitcher_id"])] = pr["pitch_hand"]

    # --- Load umpire tendencies and weather effects ---
    ump_path = DASHBOARD_DIR / "umpire_tendencies.parquet"
    wx_path = DASHBOARD_DIR / "weather_effects.parquet"
    ump_lookup: dict[str, float] = {}
    if ump_path.exists():
        ump_df = pd.read_parquet(ump_path)
        for _, ur in ump_df.iterrows():
            ump_lookup[ur["hp_umpire_name"]] = float(ur["k_logit_lift"])
        logger.info("Loaded %d umpire tendencies", len(ump_lookup))

    wx_lookup: dict[tuple[str, str], dict] = {}
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        for _, wr in wx_df.iterrows():
            wx_lookup[(wr["temp_bucket"], wr["wind_category"])] = {
                "k_multiplier": float(wr["k_multiplier"]),
                "overall_k_rate": float(wr["overall_k_rate"]),
            }
        logger.info("Loaded %d weather effect combos", len(wx_lookup))

    # --- Helper: get pitcher avg exit pitches ---
    def _pitcher_avg_pitches(pid: int) -> float:
        if exit_tendencies.empty:
            return 88.0
        row = exit_tendencies[
            (exit_tendencies["pitcher_id"] == pid)
            & (exit_tendencies["season"] == SEASON - 1)
        ]
        if not row.empty:
            return float(row.iloc[0]["avg_pitches"])
        return 88.0

    # --- Helper: generate fallback rate samples ---
    _rng_fallback = np.random.default_rng(99)

    def _fallback_samples(rate: float, n: int = 4000) -> np.ndarray:
        """Beta(a, b) centered on rate with moderate spread."""
        r = np.clip(rate, 0.01, 0.99)
        kappa = 200  # concentration
        return _rng_fallback.beta(r * kappa, (1 - r) * kappa, size=n).astype(np.float32)

    # --- Simulate each starter ---
    logger.info("Simulating game props for today's starters...")
    results = []
    sim_sample_arrays: dict[str, np.ndarray] = {}
    for _, game in schedule.iterrows():
        gpk = game["game_pk"]

        # Per-game umpire lift
        hp_ump_name = game.get("hp_umpire_name", "")
        ump_k_lift = ump_lookup.get(hp_ump_name, 0.0) if hp_ump_name else 0.0

        # Per-game weather lift
        wx_k_lift = 0.0
        temp_bucket = _parse_temp_bucket(game.get("weather_temp"))
        wind_cat = _parse_wind_category(game.get("weather_wind"))
        wx_info = wx_lookup.get((temp_bucket, wind_cat))
        if wx_info:
            from scipy.special import logit as _logit
            k_mult = wx_info["k_multiplier"]
            overall_k = wx_info["overall_k_rate"]
            adj_k = np.clip(overall_k * k_mult, 1e-6, 1 - 1e-6)
            wx_k_lift = float(_logit(adj_k) - _logit(np.clip(overall_k, 1e-6, 1 - 1e-6)))

        for side in ("away", "home"):
            pid = game.get(f"{side}_pitcher_id")
            pname = game.get(f"{side}_pitcher_name", "")

            if pd.isna(pid):
                continue
            pid = int(pid)
            pid_str = str(pid)

            if pid_str not in k_samples:
                logger.debug("No K samples for pitcher %s (%s) — skipping", pname, pid)
                continue

            k_samp = k_samples[pid_str]

            # BB / HR samples (fallback to projection-based Beta if missing)
            p_row = pitcher_proj[pitcher_proj["pitcher_id"] == pid]
            proj_k_rate = float(p_row.iloc[0]["projected_k_rate"]) if not p_row.empty else float(np.mean(k_samp))
            proj_bb = float(p_row.iloc[0].get("projected_bb_rate", 0.08)) if not p_row.empty else 0.08
            proj_hr = float(p_row.iloc[0].get("projected_hr_per_bf", 0.03)) if not p_row.empty else 0.03
            composite = float(p_row.iloc[0].get("composite_score", 0)) if not p_row.empty else 0.0

            bb_samp = bb_samples.get(pid_str) if bb_samples else None
            if bb_samp is None:
                bb_samp = _fallback_samples(proj_bb)
            hr_samp = hr_samples.get(pid_str) if hr_samples else None
            if hr_samp is None:
                hr_samp = _fallback_samples(proj_hr)

            # Lineup matchup lifts
            opp_side = "home" if side == "away" else "away"
            opp_team_id = game.get(f"{opp_side}_team_id")
            opp_abbr = game.get(f"{opp_side}_abbr", "")
            lineup_matchup_lifts: dict[str, np.ndarray] = {}
            lineup_source = "none"  # "api", "depth_chart", or "none"
            lineup_batter_ids: list[int] = []

            # --- Try real lineup from API first ---
            game_lu = pd.DataFrame()
            if not lineups.empty:
                game_lu = lineups[
                    (lineups["game_pk"] == gpk) &
                    (lineups["team_id"] == opp_team_id)
                ].sort_values("batting_order")
                if len(game_lu) >= 9:
                    lineup_source = "api"

            # --- Fall back to depth chart by pitcher hand ---
            if lineup_source == "none" and not depth_chart.empty and opp_abbr:
                p_hand = pitch_hand_lookup.get(pid, "R")
                dc_lu = depth_chart[
                    (depth_chart["team_abbr"] == opp_abbr) &
                    (depth_chart["vs_hand"] == p_hand)
                ].sort_values("batting_order")
                if len(dc_lu) >= 9:
                    # Reshape to match API lineup columns
                    game_lu = dc_lu.head(9).rename(columns={
                        "player_id": "batter_id",
                        "player_name": "batter_name",
                    })[["batter_id", "batter_name", "batting_order"]]
                    lineup_source = "depth_chart"
                    logger.debug("Using depth chart for %s vs %s (%sHP)",
                                 opp_abbr, pname, p_hand)

            # --- Score matchup lifts if we have any lineup ---
            if lineup_source != "none" and not arsenal_df.empty and not vuln_df.empty:
                k_lifts, bb_lifts, hr_lifts = [], [], []
                for _, brow in game_lu.head(9).iterrows():
                    bid = int(brow["batter_id"])
                    lineup_batter_ids.append(bid)

                    k_m = score_matchup(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    kl = k_m.get("matchup_k_logit_lift", 0.0)
                    k_lifts.append(0.0 if np.isnan(kl) else kl)

                    bb_m = score_matchup_bb(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    bl = bb_m.get("matchup_bb_logit_lift", 0.0)
                    bb_lifts.append(0.0 if np.isnan(bl) else bl)

                    hr_m = score_matchup_hr(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    hl = hr_m.get("matchup_hr_logit_lift", 0.0)
                    hr_lifts.append(0.0 if np.isnan(hl) else hl)

                lineup_matchup_lifts = {
                    "k": np.array(k_lifts),
                    "bb": np.array(bb_lifts),
                    "hr": np.array(hr_lifts),
                }

            # TTO lifts
            tto_lifts = build_all_tto_lifts(
                tto_profiles if not tto_profiles.empty else None,
                pid, SEASON - 1,
            )

            # Pitch count features
            pitcher_ppa_adj = 0.0
            batter_ppa_adjs = np.zeros(9)
            if not pitcher_pc.empty and not batter_pc.empty and lineup_batter_ids:
                pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
                    pitcher_pc, batter_pc, pid, lineup_batter_ids, SEASON - 1,
                )

            avg_pitches = _pitcher_avg_pitches(pid)

            # Run PA-by-PA simulation
            result = simulate_game(
                pitcher_k_rate_samples=k_samp,
                pitcher_bb_rate_samples=bb_samp,
                pitcher_hr_rate_samples=hr_samp,
                lineup_matchup_lifts=lineup_matchup_lifts,
                tto_lifts=tto_lifts,
                pitcher_ppa_adj=pitcher_ppa_adj,
                batter_ppa_adjs=batter_ppa_adjs,
                exit_model=exit_model,
                pitcher_avg_pitches=avg_pitches,
                umpire_k_lift=ump_k_lift,
                weather_k_lift=wx_k_lift,
                n_sims=10_000,
                random_seed=42 + gpk + (0 if side == "away" else 1),
            )

            # Fantasy points
            fantasy = compute_pitcher_fantasy(result)
            dk = fantasy.dk_summary()
            espn = fantasy.espn_summary()

            # K prop lines
            k_over = result.over_probs("k", lines=[4.5, 5.5, 6.5, 7.5])
            p_over_dict = {}
            for _, kr in k_over.iterrows():
                col = f"p_over_{kr['line']:.1f}".replace(".", "_")
                p_over_dict[col] = kr["p_over"]

            # BB and HR prop lines
            bb_over = result.over_probs("bb", lines=[1.5, 2.5, 3.5])
            for _, br in bb_over.iterrows():
                col = f"p_over_bb_{br['line']:.1f}".replace(".", "_")
                p_over_dict[col] = br["p_over"]

            hr_over = result.over_probs("hr", lines=[0.5, 1.5])
            for _, hr_r in hr_over.iterrows():
                col = f"p_over_hr_{hr_r['line']:.1f}".replace(".", "_")
                p_over_dict[col] = hr_r["p_over"]

            # H prop lines
            h_over = result.over_probs("h", lines=[4.5, 5.5, 6.5, 7.5])
            for _, hr_row in h_over.iterrows():
                col = f"p_over_h_{hr_row['line']:.1f}".replace(".", "_")
                p_over_dict[col] = hr_row["p_over"]

            has_lineup = lineup_source != "none"
            avg_matchup = float(np.mean(lineup_matchup_lifts["k"])) if has_lineup else 0.0

            # Stash raw sample arrays for dashboard (avoids re-sim at render)
            _sim_key = f"{gpk}_{pid}"
            sim_sample_arrays[f"{_sim_key}_k"] = result.k_samples.astype(np.float32)
            sim_sample_arrays[f"{_sim_key}_bb"] = result.bb_samples.astype(np.float32)
            sim_sample_arrays[f"{_sim_key}_h"] = result.h_samples.astype(np.float32)
            sim_sample_arrays[f"{_sim_key}_hr"] = result.hr_samples.astype(np.float32)
            sim_sample_arrays[f"{_sim_key}_outs"] = result.outs_samples.astype(np.float32)
            sim_sample_arrays[f"{_sim_key}_runs"] = result.runs_samples.astype(np.float32)

            results.append({
                "game_pk": gpk,
                "side": side,
                "pitcher_id": pid,
                "pitcher_name": pname,
                "team_abbr": game.get(f"{side}_abbr", ""),
                "opp_abbr": game.get(f"{opp_side}_abbr", ""),
                "projected_k_rate": proj_k_rate,
                "composite_score": composite,
                # K
                "expected_k": float(np.mean(result.k_samples)),
                "k_std": float(np.std(result.k_samples)),
                "median_k": float(np.median(result.k_samples)),
                # BB
                "expected_bb": float(np.mean(result.bb_samples)),
                "bb_std": float(np.std(result.bb_samples)),
                # H
                "expected_h": float(np.mean(result.h_samples)),
                "h_std": float(np.std(result.h_samples)),
                # HR
                "expected_hr": float(np.mean(result.hr_samples)),
                "hr_std": float(np.std(result.hr_samples)),
                # IP / Pitches
                "expected_ip": float(np.mean(result.ip_samples())),
                "expected_pitches": float(np.mean(result.pitch_count_samples)),
                "expected_bf": float(np.mean(result.bf_samples)),
                # Fantasy
                "dk_mean": dk["mean"],
                "dk_median": dk["median"],
                "dk_q10": dk["q10"],
                "dk_q90": dk["q90"],
                "espn_mean": espn["mean"],
                "espn_median": espn["median"],
                # Context
                "has_lineup": has_lineup,
                "lineup_source": lineup_source,
                "avg_matchup_lift": avg_matchup,
                "umpire_k_logit_lift": ump_k_lift,
                "weather_k_logit_lift": wx_k_lift,
                "bf_mu": float(np.mean(result.bf_samples)),
                "bf_sigma": float(np.std(result.bf_samples)),
                **p_over_dict,
            })

    if results:
        sim_df = pd.DataFrame(results)
        sim_df.to_parquet(DASHBOARD_DIR / "todays_sims.parquet", index=False)
        logger.info("Saved game simulations for %d pitcher appearances", len(sim_df))

        # Save raw sample arrays so the dashboard can render distributions
        # without re-running Monte Carlo at render time.
        if sim_sample_arrays:
            np.savez_compressed(
                DASHBOARD_DIR / "pitcher_game_sim_samples.npz",
                **sim_sample_arrays,
            )
            logger.info("Saved pitcher game sim sample arrays (%d keys)", len(sim_sample_arrays))

        n_api = (sim_df["lineup_source"] == "api").sum()
        n_dc = (sim_df["lineup_source"] == "depth_chart").sum()
        n_none = (sim_df["lineup_source"] == "none").sum()
        logger.info("  Lineup sources: %d API, %d depth chart, %d none",
                     n_api, n_dc, n_none)

        # --- Game-level predictions (moneyline / spread / over-under) ---
        _build_game_predictions(sim_df, sim_sample_arrays)
    else:
        logger.warning("No pitchers could be simulated (missing K samples?)")

    return True


def _build_game_predictions(
    sim_df: "pd.DataFrame", sample_arrays: dict[str, "np.ndarray"],
) -> None:
    """Compute game-level predictions from pitcher sim runs and save."""
    import pandas as pd
    from lib.game_predictions import build_game_predictions_from_sims

    game_preds = build_game_predictions_from_sims(sim_df, sample_arrays)
    if game_preds.empty:
        logger.warning("No game-level predictions produced")
        return

    game_preds.to_parquet(
        DASHBOARD_DIR / "todays_game_predictions.parquet", index=False,
    )
    logger.info("Saved game predictions for %d games", len(game_preds))


# ---------------------------------------------------------------------------
# Game odds collection
# ---------------------------------------------------------------------------

def _run_game_accuracy_report() -> None:
    """Run game prediction accuracy report if enough data exists."""
    log_path = DASHBOARD_DIR / "game_prediction_log.parquet"
    if not log_path.exists():
        logger.info("No game prediction log yet. Skipping accuracy report.")
        return
    try:
        from scripts.validate_game_accuracy import print_accuracy_report
        import pandas as pd

        df = pd.read_parquet(log_path)
        if len(df) < 5:
            logger.info("Only %d games in log. Need ≥5 for report.", len(df))
            return
        print_accuracy_report(df)
    except Exception as e:
        logger.warning("Game accuracy report failed: %s", e)


def collect_game_odds_snapshot(game_date: str) -> None:
    """Fetch game-level odds from DK/Bovada and append to history parquet."""
    try:
        from scripts.collect_game_odds import collect_odds, append_to_history

        odds = collect_odds(game_date)
        if not odds.empty:
            append_to_history(odds)
        else:
            logger.info("No game odds collected for %s", game_date)
    except Exception as e:
        logger.warning("Game odds collection failed: %s", e)


# ---------------------------------------------------------------------------
# Projection engine delegation
# ---------------------------------------------------------------------------

def run_projection_engine(game_date: str, skip_schedule: bool = False) -> bool:
    """Run the player_profiles in-season update script.

    Returns True on success, False on failure.
    """
    engine_script = PLAYER_PROFILES_DIR / "scripts" / "update_in_season.py"
    if not engine_script.exists():
        logger.error(
            "Projection engine script not found at %s. "
            "Ensure player_profiles repo is at %s",
            engine_script, PLAYER_PROFILES_DIR,
        )
        return False

    # Use player_profiles' own virtualenv, not the dashboard's
    engine_python = PLAYER_PROFILES_DIR / "myenv" / "Scripts" / "python.exe"
    if not engine_python.exists():
        # Fall back to Unix layout
        engine_python = PLAYER_PROFILES_DIR / "myenv" / "bin" / "python"
    if not engine_python.exists():
        logger.error("player_profiles virtualenv not found at %s", engine_python.parent)
        return False

    cmd = [str(engine_python), str(engine_script), "--date", game_date]
    if skip_schedule:
        cmd.append("--skip-schedule")

    logger.info("Running projection engine: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PLAYER_PROFILES_DIR))

    if result.returncode != 0:
        logger.error("Projection engine exited with code %d", result.returncode)
        return False

    logger.info("Projection engine completed successfully")
    return True


# ---------------------------------------------------------------------------
# Weekly snapshots
# ---------------------------------------------------------------------------

def _is_snapshot_day(game_date: str) -> bool:
    """Return True if game_date falls on a Monday (weekly snapshot day)."""
    d = date.fromisoformat(game_date)
    return d.weekday() == 0  # Monday


def save_weekly_snapshot(game_date: str) -> None:
    """Copy current projections and rankings to a dated weekly snapshot.

    Idempotent: skips if a snapshot already exists for this date.
    """
    weekly_dir = DASHBOARD_DIR / "snapshots" / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    # Projections
    for ptype in ("hitter", "pitcher"):
        src = DASHBOARD_DIR / f"{ptype}_projections.parquet"
        dst = weekly_dir / f"{ptype}_projections_{game_date}.parquet"
        if dst.exists():
            logger.info("Weekly snapshot already exists: %s, skipping", dst.name)
            continue
        if not src.exists():
            logger.warning("No %s projections to snapshot", ptype)
            continue
        shutil.copy2(src, dst)
        logger.info("Saved weekly snapshot: %s", dst.name)

    # Rankings
    for rtype in ("hitters", "pitchers"):
        src = DASHBOARD_DIR / f"{rtype}_rankings.parquet"
        dst = weekly_dir / f"{rtype}_rankings_{game_date}.parquet"
        if dst.exists():
            continue
        if not src.exists():
            logger.warning("No %s rankings to snapshot", rtype)
            continue
        shutil.copy2(src, dst)
        logger.info("Saved weekly snapshot: %s", dst.name)


# ---------------------------------------------------------------------------
# Batter game simulations
# ---------------------------------------------------------------------------

def run_batter_sims(game_date: str) -> None:
    """Run batter-level game sims for today's lineups and save to parquet.

    Requires todays_sims.parquet (pitcher context) and todays_lineups.parquet
    to already exist from run_schedule_refresh().
    """
    import numpy as np
    import pandas as pd
    from lib.game_sim.batter_simulator import simulate_batter_game
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    sims_path = DASHBOARD_DIR / "todays_sims.parquet"
    lu_path = DASHBOARD_DIR / "todays_lineups.parquet"
    if not sims_path.exists() or not lu_path.exists():
        logger.warning("Missing sims or lineups parquet, cannot run batter sims")
        return

    pitcher_sims = pd.read_parquet(sims_path)
    lineups = pd.read_parquet(lu_path)
    if lineups.empty or pitcher_sims.empty:
        logger.info("No lineups or pitcher sims available for batter sims")
        return

    # Load hitter posterior samples
    h_k_path = DASHBOARD_DIR / "hitter_k_samples.npz"
    h_bb_path = DASHBOARD_DIR / "hitter_bb_samples.npz"
    h_hr_path = DASHBOARD_DIR / "hitter_hr_samples.npz"

    h_k_samples: dict[str, np.ndarray] = {}
    h_bb_samples: dict[str, np.ndarray] = {}
    h_hr_samples: dict[str, np.ndarray] = {}

    if h_k_path.exists():
        _d = np.load(h_k_path)
        h_k_samples = {k: _d[k] for k in _d.files}
    if h_bb_path.exists():
        _d = np.load(h_bb_path)
        h_bb_samples = {k: _d[k] for k in _d.files}
    if h_hr_path.exists():
        _d = np.load(h_hr_path)
        h_hr_samples = {k: _d[k] for k in _d.files}

    if not h_k_samples:
        logger.warning("No hitter K samples found, cannot run batter sims")
        return

    # Hitter projections for fallback rates
    h_proj_path = DASHBOARD_DIR / "hitter_projections.parquet"
    h_proj = pd.read_parquet(h_proj_path) if h_proj_path.exists() else pd.DataFrame()

    # Arsenal/vuln for matchup lifts
    arsenal_path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    vuln_path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
    arsenal_df = pd.read_parquet(arsenal_path) if arsenal_path.exists() else pd.DataFrame()
    vuln_df = pd.read_parquet(vuln_path) if vuln_path.exists() else pd.DataFrame()

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    # Bullpen rates
    bp_path = DASHBOARD_DIR / "team_bullpen_rates.parquet"
    bp_lookup: dict[str, dict] = {}
    if bp_path.exists():
        bp_df = pd.read_parquet(bp_path)
        for _, r in bp_df.iterrows():
            bp_lookup[r.get("team_abbr", "")] = {
                "k_rate": float(r.get("bullpen_k_rate", 0.253)),
                "bb_rate": float(r.get("bullpen_bb_rate", 0.084)),
                "hr_rate": float(r.get("bullpen_hr_rate", 0.024)),
            }

    # Fallback sample generator
    _rng = np.random.default_rng(77)

    def _fallback(rate: float, n: int = 4000) -> np.ndarray:
        r = np.clip(rate, 0.01, 0.99)
        return _rng.beta(r * 200, (1 - r) * 200, size=n).astype(np.float32)

    # Build pitcher context lookup from sims
    pitcher_ctx: dict[int, dict] = {}
    for _, ps in pitcher_sims.iterrows():
        pid = int(ps["pitcher_id"])
        # Derive BF from available columns
        _exp_bf = float(ps.get("expected_bf", 0))
        if _exp_bf < 1:
            _exp_bf = (
                float(ps.get("expected_outs", 16))
                + float(ps.get("expected_h", 5))
                + float(ps.get("expected_bb", 2))
                + float(ps.get("expected_hr", 0.5))
            )
        pitcher_ctx[pid] = {
            "k_rate": float(ps.get("projected_k_rate", 0.22)),
            "bb_rate": float(ps.get("expected_bb", 2)) / max(_exp_bf, 1),
            "hr_rate": float(ps.get("expected_hr", 0.5)) / max(_exp_bf, 1),
            "bf_mu": float(ps.get("bf_mu", _exp_bf)),
            "bf_sigma": float(ps.get("bf_sigma", 3)),
            "team_abbr": ps.get("team_abbr", ""),
            "opp_abbr": ps.get("opp_abbr", ""),
            "game_pk": int(ps["game_pk"]),
            "has_lineup": bool(ps.get("has_lineup", False)),
        }

    logger.info("Running batter sims for %d lineup batters...", len(lineups))
    results = []
    batter_sample_arrays: dict[str, np.ndarray] = {}
    skipped = 0

    for _, brow in lineups.iterrows():
        bid = int(brow["batter_id"])
        bid_str = str(bid)
        gpk = int(brow["game_pk"])

        if bid_str not in h_k_samples:
            skipped += 1
            continue

        # Find the opposing pitcher for this batter
        opp_pid = None
        for pid, ctx in pitcher_ctx.items():
            if ctx["game_pk"] == gpk and ctx["opp_abbr"] == brow.get("team_abbr", ""):
                opp_pid = pid
                break
        if opp_pid is None:
            skipped += 1
            continue

        pctx = pitcher_ctx[opp_pid]
        bp_rates = bp_lookup.get(pctx["team_abbr"], {})

        # Matchup lifts
        k_lift = bb_lift = hr_lift = 0.0
        if not arsenal_df.empty and not vuln_df.empty:
            k_m = score_matchup(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            k_lift = k_m.get("matchup_k_logit_lift", 0.0)
            k_lift = 0.0 if np.isnan(k_lift) else k_lift
            bb_m = score_matchup_bb(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            bb_lift = bb_m.get("matchup_bb_logit_lift", 0.0)
            bb_lift = 0.0 if np.isnan(bb_lift) else bb_lift
            hr_m = score_matchup_hr(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            hr_lift = hr_m.get("matchup_hr_logit_lift", 0.0)
            hr_lift = 0.0 if np.isnan(hr_lift) else hr_lift

        # Hitter posterior samples
        bk = h_k_samples[bid_str]
        bbb = h_bb_samples.get(bid_str)
        if bbb is None:
            hp = h_proj[h_proj["batter_id"] == bid] if not h_proj.empty else pd.DataFrame()
            rate = float(hp.iloc[0].get("projected_bb_rate", 0.08)) if not hp.empty else 0.08
            bbb = _fallback(rate)
        bhr = h_hr_samples.get(bid_str)
        if bhr is None:
            hp = h_proj[h_proj["batter_id"] == bid] if not h_proj.empty else pd.DataFrame()
            rate = float(hp.iloc[0].get("projected_hr_per_pa", 0.03)) if not hp.empty else 0.03
            bhr = _fallback(rate)

        batting_order = int(brow.get("batting_order", 5))

        sim_result = simulate_batter_game(
            batter_k_rate_samples=bk,
            batter_bb_rate_samples=bbb,
            batter_hr_rate_samples=bhr,
            batting_order=batting_order,
            starter_k_rate=pctx["k_rate"],
            starter_bb_rate=pctx["bb_rate"],
            starter_hr_rate=pctx["hr_rate"],
            starter_bf_mu=pctx["bf_mu"],
            starter_bf_sigma=pctx["bf_sigma"],
            matchup_k_lift=k_lift,
            matchup_bb_lift=bb_lift,
            matchup_hr_lift=hr_lift,
            bullpen_k_rate=bp_rates.get("k_rate", 0.253),
            bullpen_bb_rate=bp_rates.get("bb_rate", 0.084),
            bullpen_hr_rate=bp_rates.get("hr_rate", 0.024),
            n_sims=10_000,
            random_seed=42 + gpk + bid,
        )

        # Stash raw sample arrays for dashboard render
        _bsim_key = f"{gpk}_{bid}"
        batter_sample_arrays[f"{_bsim_key}_k"] = sim_result.k_samples.astype(np.float32)
        batter_sample_arrays[f"{_bsim_key}_bb"] = sim_result.bb_samples.astype(np.float32)
        batter_sample_arrays[f"{_bsim_key}_h"] = sim_result.h_samples.astype(np.float32)
        batter_sample_arrays[f"{_bsim_key}_hr"] = sim_result.hr_samples.astype(np.float32)

        summary = sim_result.summary()

        # Prop lines
        k_props = sim_result.over_probs("k", lines=[0.5, 1.5])
        h_props = sim_result.over_probs("h", lines=[0.5, 1.5])
        hr_props = sim_result.over_probs("hr", lines=[0.5])

        prop_dict = {}
        for _, kr in k_props.iterrows():
            col = f"p_k_over_{kr['line']:.1f}".replace(".", "_")
            prop_dict[col] = kr["p_over"]
        for _, hr in h_props.iterrows():
            col = f"p_h_over_{hr['line']:.1f}".replace(".", "_")
            prop_dict[col] = hr["p_over"]
        for _, hr in hr_props.iterrows():
            col = f"p_hr_over_{hr['line']:.1f}".replace(".", "_")
            prop_dict[col] = hr["p_over"]

        results.append({
            "game_pk": gpk,
            "batter_id": bid,
            "batter_name": brow.get("batter_name", ""),
            "team_abbr": brow.get("team_abbr", ""),
            "opp_abbr": pctx["opp_abbr"] if pctx["opp_abbr"] != brow.get("team_abbr") else pctx["team_abbr"],
            "batting_order": batting_order,
            "opp_starter_id": opp_pid,
            "lineup_source": brow.get("lineup_source", "api"),
            "matchup_k_lift": k_lift,
            "matchup_bb_lift": bb_lift,
            "matchup_hr_lift": hr_lift,
            "expected_k": summary["k"]["mean"],
            "std_k": summary["k"]["std"],
            "expected_bb": summary["bb"]["mean"],
            "expected_h": summary["h"]["mean"],
            "expected_hr": summary["hr"]["mean"],
            "expected_tb": summary["tb"]["mean"],
            "expected_pa": summary["pa"]["mean"],
            "has_lineup": pctx["has_lineup"],
            **prop_dict,
        })

    if results:
        df = pd.DataFrame(results)
        df.to_parquet(DASHBOARD_DIR / "todays_batter_sims.parquet", index=False)
        logger.info("Saved batter sims: %d batters (%d skipped, no samples)",
                     len(df), skipped)

        if batter_sample_arrays:
            np.savez_compressed(
                DASHBOARD_DIR / "batter_game_sim_samples.npz",
                **batter_sample_arrays,
            )
            logger.info("Saved batter game sim sample arrays (%d keys)", len(batter_sample_arrays))
    else:
        logger.warning("No batter sims produced (%d skipped)", skipped)


# ---------------------------------------------------------------------------
# Prediction archiving
# ---------------------------------------------------------------------------

def archive_yesterdays_predictions(game_date: str) -> None:
    """Archive yesterday's predictions joined with actuals to cumulative logs.

    Called at the start of main() before anything overwrites yesterday's files.
    """
    import pandas as pd
    from datetime import timedelta

    yesterday = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()

    # Check if we have yesterday's predictions
    sims_path = DASHBOARD_DIR / "todays_sims.parquet"
    games_path = DASHBOARD_DIR / "todays_games.parquet"
    batter_sims_path = DASHBOARD_DIR / "todays_batter_sims.parquet"

    if not sims_path.exists() or not games_path.exists():
        logger.info("No predictions to archive (files missing)")
        return

    # Verify the predictions are actually for yesterday
    games = pd.read_parquet(games_path)
    if games.empty:
        logger.info("No games in todays_games.parquet, nothing to archive")
        return

    pred_date = str(games.iloc[0].get("game_date", ""))[:10]
    if pred_date and pred_date != yesterday:
        logger.info("Predictions are for %s, not yesterday (%s). Skipping archive.",
                     pred_date, yesterday)
        return

    # Check for duplicate archiving
    pitcher_log_path = DASHBOARD_DIR / "pitcher_sim_log.parquet"
    if pitcher_log_path.exists():
        existing = pd.read_parquet(pitcher_log_path)
        if "prediction_date" in existing.columns:
            if yesterday in existing["prediction_date"].astype(str).values:
                logger.info("Already archived predictions for %s. Skipping.", yesterday)
                return

    # Query actuals from DB
    try:
        from lib.db import read_sql

        pitcher_actuals = read_sql("""
            SELECT player_id AS pitcher_id, game_pk,
                   pit_k AS actual_k, pit_bb AS actual_bb,
                   pit_h AS actual_h, pit_hr AS actual_hr,
                   pit_bf AS actual_bf, pit_pitches AS actual_pitches,
                   pit_ip AS actual_ip
            FROM production.fact_player_game_mlb
            WHERE player_role = 'pitcher'
              AND game_date = :gd
              AND season = :season
        """, params={"gd": yesterday, "season": SEASON})

        batter_actuals = read_sql("""
            SELECT player_id AS batter_id, game_pk,
                   bat_k AS actual_k, bat_bb AS actual_bb,
                   bat_h AS actual_h, bat_hr AS actual_hr,
                   bat_tb AS actual_tb, bat_pa AS actual_pa
            FROM production.fact_player_game_mlb
            WHERE player_role = 'batter'
              AND game_date = :gd
              AND season = :season
        """, params={"gd": yesterday, "season": SEASON})

    except Exception as e:
        logger.warning("Could not query actuals from DB: %s. Skipping archive.", e)
        return

    if pitcher_actuals.empty:
        logger.info("No pitcher actuals found for %s. Skipping archive.", yesterday)
        return

    # --- Archive pitcher predictions ---
    pitcher_sims = pd.read_parquet(sims_path)
    pitcher_sims["prediction_date"] = yesterday

    pitcher_merged = pitcher_sims.merge(
        pitcher_actuals, on=["game_pk", "pitcher_id"], how="inner",
    )

    if not pitcher_merged.empty:
        # Select columns for the log
        log_cols = [
            "game_pk", "prediction_date", "pitcher_id", "pitcher_name",
            "team_abbr", "opp_abbr",
            "expected_k", "k_std", "expected_bb", "expected_h", "expected_hr",
            "expected_bf", "expected_ip", "expected_pitches", "has_lineup",
        ]
        # Add prop columns if present
        for c in pitcher_merged.columns:
            if c.startswith("p_over_"):
                log_cols.append(c)
        # Add actuals
        log_cols += ["actual_k", "actual_bb", "actual_h", "actual_hr",
                     "actual_bf", "actual_ip", "actual_pitches"]

        log_cols = [c for c in log_cols if c in pitcher_merged.columns]
        new_rows = pitcher_merged[log_cols]

        if pitcher_log_path.exists():
            existing = pd.read_parquet(pitcher_log_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined.to_parquet(pitcher_log_path, index=False)
        logger.info("Archived %d pitcher predictions for %s (total: %d)",
                     len(new_rows), yesterday, len(combined))

    # --- Archive batter predictions ---
    if batter_sims_path.exists() and not batter_actuals.empty:
        batter_sims = pd.read_parquet(batter_sims_path)
        batter_sims["prediction_date"] = yesterday

        batter_merged = batter_sims.merge(
            batter_actuals, on=["game_pk", "batter_id"], how="inner",
        )

        if not batter_merged.empty:
            log_cols = [
                "game_pk", "prediction_date", "batter_id", "batter_name",
                "team_abbr", "batting_order", "opp_starter_id",
                "expected_k", "std_k", "expected_bb", "expected_h",
                "expected_hr", "expected_tb", "expected_pa", "has_lineup",
            ]
            for c in batter_merged.columns:
                if c.startswith("p_k_over_") or c.startswith("p_h_over_") or c.startswith("p_hr_over_"):
                    log_cols.append(c)
            log_cols += ["actual_k", "actual_bb", "actual_h", "actual_hr",
                         "actual_tb", "actual_pa"]
            log_cols = [c for c in log_cols if c in batter_merged.columns]
            new_rows = batter_merged[log_cols]

            batter_log_path = DASHBOARD_DIR / "batter_sim_log.parquet"
            if batter_log_path.exists():
                existing = pd.read_parquet(batter_log_path)
                combined = pd.concat([existing, new_rows], ignore_index=True)
            else:
                combined = new_rows

            combined.to_parquet(batter_log_path, index=False)
            logger.info("Archived %d batter predictions for %s (total: %d)",
                         len(new_rows), yesterday, len(combined))


def archive_yesterdays_game_predictions(game_date: str) -> None:
    """Archive yesterday's game-level predictions (ML/spread/O-U) with actuals.

    Reads todays_game_predictions.parquet (which contains yesterday's preds
    until overwritten), queries game scores from the DB, computes accuracy
    flags and CRPS, and appends to game_prediction_log.parquet.
    """
    import numpy as np
    import pandas as pd
    from datetime import timedelta

    yesterday = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()

    preds_path = DASHBOARD_DIR / "todays_game_predictions.parquet"
    games_path = DASHBOARD_DIR / "todays_games.parquet"

    if not preds_path.exists() or not games_path.exists():
        logger.info("No game predictions to archive (files missing)")
        return

    # Verify predictions are for yesterday
    games = pd.read_parquet(games_path)
    if games.empty:
        return
    pred_date = str(games.iloc[0].get("game_date", ""))[:10]
    if pred_date and pred_date != yesterday:
        logger.info("Game predictions are for %s, not yesterday (%s). Skipping.",
                     pred_date, yesterday)
        return

    # Check for duplicate archiving
    log_path = DASHBOARD_DIR / "game_prediction_log.parquet"
    if log_path.exists():
        existing = pd.read_parquet(log_path)
        if "prediction_date" in existing.columns:
            if yesterday in existing["prediction_date"].astype(str).values:
                logger.info("Already archived game predictions for %s. Skipping.",
                             yesterday)
                return

    # Query game scores from DB
    # Aggregate from fact_player_game_mlb (confirmed table) since
    # fact_game_mlb may not exist in all environments.
    try:
        from lib.db import read_sql

        scores = read_sql("""
            SELECT game_pk,
                   SUM(CASE WHEN team_id = away_team_id THEN runs_scored ELSE 0 END) AS away_score,
                   SUM(CASE WHEN team_id = home_team_id THEN runs_scored ELSE 0 END) AS home_score
            FROM (
                SELECT fpg.game_pk, fpg.team_id,
                       SUM(COALESCE(fpg.bat_r, 0)) AS runs_scored,
                       dg.home_team_id, dg.away_team_id
                FROM production.fact_player_game_mlb fpg
                JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
                WHERE fpg.game_date = :gd
                  AND fpg.season = :season
                  AND fpg.player_role = 'batter'
                GROUP BY fpg.game_pk, fpg.team_id, dg.home_team_id, dg.away_team_id
            ) t
            GROUP BY game_pk
            HAVING COUNT(DISTINCT team_id) = 2
        """, params={"gd": yesterday, "season": SEASON})
    except Exception as e:
        logger.warning("Could not query game scores from DB: %s. Skipping.", e)
        return

    if scores.empty:
        logger.info("No final game scores for %s. Skipping game archive.", yesterday)
        return

    preds = pd.read_parquet(preds_path)
    preds["prediction_date"] = yesterday

    # Load sample arrays for CRPS computation
    samples_path = DASHBOARD_DIR / "pitcher_game_sim_samples.npz"
    sample_arrays = None
    if samples_path.exists():
        try:
            _d = np.load(samples_path)
            sample_arrays = {k: _d[k] for k in _d.files}
        except Exception:
            pass

    from lib.game_predictions import evaluate_game_predictions

    evaluated = evaluate_game_predictions(preds, scores, sample_arrays)
    if evaluated.empty:
        logger.info("No game predictions matched to scores for %s", yesterday)
        return

    if log_path.exists():
        existing = pd.read_parquet(log_path)
        combined = pd.concat([existing, evaluated], ignore_index=True)
    else:
        combined = evaluated

    combined.to_parquet(log_path, index=False)
    logger.info("Archived %d game predictions for %s (total: %d)",
                 len(evaluated), yesterday, len(combined))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard in-season update")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--skip-schedule", action="store_true",
                        help="Skip fetching schedule/lineups from MLB API.")
    parser.add_argument("--skip-engine", action="store_true",
                        help="Skip projection engine; only run dashboard bookkeeping.")
    parser.add_argument("--snapshot", action="store_true",
                        help="Force saving a weekly projection snapshot.")
    parser.add_argument("--schedule-only", action="store_true",
                        help="Refresh schedule/lineups/sims only (no projection updates).")
    parser.add_argument("--batter-sims-only", action="store_true",
                        help="Re-run batter game sims only (skip projections and schedule fetch).")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("=" * 60)
    logger.info("Dashboard update for %s (season %d)", game_date, SEASON)
    logger.info("=" * 60)

    # Schedule-only mode: lightweight refresh, then exit
    if args.schedule_only:
        logger.info("Mode: schedule-only (30-min refresh)")

        # Check for major roster moves -> precompute if needed
        check_roster_moves(game_date)

        changed = run_schedule_refresh(game_date)

        if changed:
            run_batter_sims(game_date)
        else:
            logger.info("Skipping batter sims (no changes)")

        # Collect game odds snapshot
        collect_game_odds_snapshot(game_date)

        # Update metadata timestamp
        meta_path = DASHBOARD_DIR / "update_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["last_schedule_refresh"] = datetime.now().isoformat()
        metadata["game_date"] = game_date
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("Done! (schedule-only, changes=%s)", changed)
        return

    # Batter-sims-only mode: re-run batter sims without touching projections or schedule
    if args.batter_sims_only:
        logger.info("Mode: batter-sims-only (re-run batter game sims)")
        run_batter_sims(game_date)

        # Update metadata timestamp
        meta_path = DASHBOARD_DIR / "update_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["last_batter_sims_refresh"] = datetime.now().isoformat()
        metadata["game_date"] = game_date
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("Done! (batter-sims-only)")
        return

    # Step 0: Archive yesterday's predictions before anything overwrites them
    logger.info("Step 0: Archiving yesterday's predictions...")
    archive_yesterdays_predictions(game_date)
    archive_yesterdays_game_predictions(game_date)

    # Step 1: Run projection engine (model work lives in player_profiles)
    if not args.skip_engine:
        success = run_projection_engine(game_date, skip_schedule=args.skip_schedule)
        if not success:
            logger.warning(
                "Projection engine failed or not found. "
                "Continuing with dashboard bookkeeping using existing parquets."
            )
    else:
        logger.info("Step 1: Skipped (--skip-engine)")

    # Step 1b: Run batter sims (uses pitcher sims from Step 1)
    logger.info("Step 1b: Running batter game sims...")
    run_batter_sims(game_date)

    # Step 1c: Collect game odds snapshot
    logger.info("Step 1c: Collecting game odds...")
    collect_game_odds_snapshot(game_date)

    # Step 2: Export roster from DB to parquet
    logger.info("Step 2: Exporting roster...")
    export_roster()

    # Step 3: Weekly snapshot
    if args.snapshot or _is_snapshot_day(game_date):
        logger.info("Saving weekly projection snapshot...")
        save_weekly_snapshot(game_date)

    # Step 4: Save update metadata
    import pandas as pd
    h_path = DASHBOARD_DIR / "hitter_projections.parquet"
    p_path = DASHBOARD_DIR / "pitcher_projections.parquet"
    k_path = DASHBOARD_DIR / "pitcher_k_samples.npz"

    metadata = {
        "last_updated": datetime.now().isoformat(),
        "game_date": game_date,
        "season": SEASON,
        "hitters_updated": len(pd.read_parquet(h_path)) if h_path.exists() else 0,
        "pitchers_updated": len(pd.read_parquet(p_path)) if p_path.exists() else 0,
        "k_samples_count": len(dict(__import__("numpy").load(k_path))) if k_path.exists() else 0,
    }
    meta_path = DASHBOARD_DIR / "update_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved update metadata to %s", meta_path)

    # Step 5: Run game prediction accuracy report (if enough data)
    logger.info("Step 5: Game prediction accuracy check...")
    _run_game_accuracy_report()

    # Step 6: Generate artifact manifest
    from services.manifest import generate_manifest
    manifest = generate_manifest(DASHBOARD_DIR)
    manifest_path = DASHBOARD_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest with %d artifacts to %s",
                len(manifest.get("artifacts", [])), manifest_path)

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
