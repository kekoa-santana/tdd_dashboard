#!/usr/bin/env python
"""
Dashboard post-update script.

Handles dashboard-specific bookkeeping: weekly snapshots, update
metadata, artifact manifest, prediction archiving, roster export,
and game odds collection.

All model work (DB queries, conjugate updating, K samples, matchup
simulation, game sims) lives in the player_profiles repo.  This script
delegates projection updates via subprocess and consumes pre-computed
parquets for everything else.

Usage
-----
    python scripts/update_in_season.py                    # today's date (full update)
    python scripts/update_in_season.py --date 2026-04-15  # specific date
    python scripts/update_in_season.py --skip-schedule    # skip API calls
    python scripts/update_in_season.py --skip-engine      # skip projection engine, just do bookkeeping
    python scripts/update_in_season.py --snapshot          # force a weekly snapshot
    python scripts/update_in_season.py --schedule-only    # roster moves + odds only (10-min mode)
    python scripts/update_in_season.py --post-sims        # post-sim bookkeeping (game preds reshape + odds + metadata)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
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
# Roster export
# ---------------------------------------------------------------------------

def export_roster() -> bool:
    """Export production.dim_roster joined with dim_team to a dashboard parquet.

    Returns True on success, False on failure.
    """
    try:
        from lib.db import read_sql

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
# Roster move detection -> precompute trigger
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
    try:
        from lib.schedule import fetch_recent_transactions
    except ImportError as e:
        logger.warning("Roster move check unavailable (%s); skipping", e)
        return False

    state_path = DASHBOARD_DIR / "roster_move_state.json"

    # Load state; reset if new day
    known_ids: set[int] = set()
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Corrupt roster_move_state.json (%s); resetting", e)
            state = {}
        if state.get("game_date") == game_date:
            known_ids = {int(x) for x in state.get("known_transaction_ids", [])}
        else:
            logger.info("New game date %s, resetting roster move state", game_date)

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
            "NEW ROSTER MOVE: %s -- %s",
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
# Game predictions (reads confident_picks output)
# ---------------------------------------------------------------------------

def _build_game_predictions(game_date: str) -> None:
    """Copy game-level predictions from confident_picks output.

    Reads game_predictions.parquet (produced by the confident_picks
    precompute in player_profiles), filters to today's games, remaps
    columns to the dashboard schema, and saves as
    todays_game_predictions.parquet.

    Also appends to the running archive parquet so we accumulate a
    sim-vs-market history over the season for calibration and bias
    learning (see ``run_market_comparison.py``).
    """
    import pandas as pd

    src_path = DASHBOARD_DIR / "game_predictions.parquet"
    if not src_path.exists():
        logger.warning("game_predictions.parquet not found; skipping game preds")
        return

    full_df = pd.read_parquet(src_path)
    if full_df.empty:
        logger.warning("game_predictions.parquet is empty")
        return

    # Filter to today's games
    game_preds = full_df[full_df["game_date"].astype(str).str[:10] == game_date].copy()
    if game_preds.empty:
        logger.warning("No game predictions for %s in game_predictions.parquet", game_date)
        return

    # Remap columns: confident_picks uses home-centric naming,
    # dashboard archive/evaluation expects away-centric naming.
    game_preds["p_away_win"] = game_preds["away_win_prob"]
    game_preds["p_home_win"] = game_preds["home_win_prob"]
    game_preds["total_mean"] = game_preds["total_runs_mean"]
    game_preds["total_std"] = game_preds["total_runs_std"]
    game_preds["margin_mean"] = game_preds["home_margin_mean"]

    # Remap spread: home_cover -> away_cover (flip perspective)
    spread_map = {
        "p_home_cover_-2.5": "p_away_cover_p2_5",
        "p_home_cover_-1.5": "p_away_cover_p1_5",
        "p_home_cover_-0.5": "p_away_cover_p0_5",
        "p_home_cover_+0.5": "p_away_cover_m0_5",
        "p_home_cover_+1.5": "p_away_cover_m1_5",
        "p_home_cover_+2.5": "p_away_cover_m2_5",
    }
    for src_col, dst_col in spread_map.items():
        if src_col in game_preds.columns:
            game_preds[dst_col] = 1.0 - game_preds[src_col]

    # Remap O/U: "p_over_7.5" -> "p_over_7_5" + "p_under_7_5"
    for col in list(game_preds.columns):
        if col.startswith("p_over_") and "." in col:
            line_str = col.replace("p_over_", "")
            safe = line_str.replace(".", "_")
            game_preds[f"p_over_{safe}"] = game_preds[col]
            game_preds[f"p_under_{safe}"] = 1.0 - game_preds[col]

    game_preds.to_parquet(
        DASHBOARD_DIR / "todays_game_predictions.parquet", index=False,
    )
    logger.info("Saved game predictions for %d games (from confident_picks)", len(game_preds))

    # Archive append -- one row per (game_date, game_pk), re-running the
    # pipeline for the same date overwrites existing rows.
    try:
        archive_path = DASHBOARD_DIR / "sim_predictions_archive.parquet"
        tagged = game_preds.copy()
        tagged["updated_at"] = pd.Timestamp.now("UTC").isoformat()

        if archive_path.exists():
            existing = pd.read_parquet(archive_path)
            keys = ["game_date", "game_pk"]
            dedup_mask = ~existing.set_index(keys).index.isin(
                tagged.set_index(keys).index
            )
            combined = pd.concat([existing[dedup_mask], tagged], ignore_index=True)
        else:
            combined = tagged

        combined.to_parquet(archive_path, index=False)
        logger.info(
            "Archive updated: +%d rows (total %d)", len(tagged), len(combined),
        )
    except Exception as e:
        logger.warning("Failed to append to sim_predictions_archive: %s", e)


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
            logger.info("Only %d games in log. Need >= 5 for report.", len(df))
            return
        print_accuracy_report(df)
    except Exception as e:
        logger.warning("Game accuracy report failed: %s", e)


def collect_game_odds_snapshot(game_date: str) -> None:
    """Fetch game-level odds (ML, spread, total) from DK/Bovada; append history + daily wide."""
    try:
        from scripts.collect_game_odds import persist_game_odds

        odds = persist_game_odds(game_date)
        if odds.empty:
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
# Prediction archiving
# ---------------------------------------------------------------------------

def archive_yesterdays_predictions(game_date: str) -> None:
    """Archive yesterday's predictions joined with actuals to cumulative logs.

    Called at the start of main() before anything overwrites yesterday's files.
    """
    import pandas as pd

    yesterday = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()

    # Check if we have yesterday's predictions
    gp_path = DASHBOARD_DIR / "game_props.parquet"
    games_path = DASHBOARD_DIR / "todays_games.parquet"
    batter_sims_path = DASHBOARD_DIR / "todays_batter_sims.parquet"

    if not gp_path.exists() or not games_path.exists():
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
    gp = pd.read_parquet(gp_path)
    from lib.fantasy_report import _build_sims_from_game_props
    pitcher_sims = _build_sims_from_game_props(gp, game_date=yesterday)
    if pitcher_sims.empty:
        logger.info("No pitcher predictions for %s in game_props. Skipping archive.", yesterday)
        return
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
# Metadata helpers
# ---------------------------------------------------------------------------

def _save_metadata(game_date: str, extra: dict | None = None) -> None:
    """Write update_metadata.json with timestamps and counts."""
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
    if extra:
        metadata.update(extra)

    meta_path = DASHBOARD_DIR / "update_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved update metadata to %s", meta_path)


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
                        help="Roster moves + odds only (10-min mode). "
                             "Sims are handled by confident_picks.")
    parser.add_argument("--post-sims", action="store_true",
                        help="Post-sim bookkeeping: game predictions reshape, "
                             "odds collection, metadata update.")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("=" * 60)
    logger.info("Dashboard update for %s (season %d)", game_date, SEASON)
    logger.info("=" * 60)

    # --schedule-only: lightweight 10-min mode (roster moves + odds)
    # Sims are now handled by confident_picks in daily_update.bat Step 3.
    if args.schedule_only:
        logger.info("Mode: schedule-only (roster moves + odds)")
        check_roster_moves(game_date)
        collect_game_odds_snapshot(game_date)

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
        logger.info("Done! (schedule-only)")
        return

    # --post-sims: runs after confident_picks to do dashboard bookkeeping
    if args.post_sims:
        logger.info("Mode: post-sims (game predictions + odds + metadata)")
        _build_game_predictions(game_date)
        collect_game_odds_snapshot(game_date)
        _save_metadata(game_date, {"last_sims_refresh": datetime.now().isoformat()})

        # Generate artifact manifest
        from services.manifest import generate_manifest
        manifest = generate_manifest(DASHBOARD_DIR)
        manifest_path = DASHBOARD_DIR / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved manifest with %d artifacts to %s",
                    len(manifest.get("artifacts", [])), manifest_path)

        logger.info("=" * 60)
        logger.info("Done! (post-sims)")
        return

    # --- Full update mode ---

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

    # Step 2: Export roster from DB to parquet
    logger.info("Step 2: Exporting roster...")
    export_roster()

    # Step 3: Weekly snapshot
    if args.snapshot or _is_snapshot_day(game_date):
        logger.info("Saving weekly projection snapshot...")
        save_weekly_snapshot(game_date)

    # Step 4: Save update metadata
    _save_metadata(game_date)

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
