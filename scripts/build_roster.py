#!/usr/bin/env python
"""
Build production.dim_roster from historical game and transaction data.

Creates a source-of-truth roster table containing every player in each
MLB organization (MLB + MiLB levels), with primary and secondary positions
derived from career game data.

Initial population uses boxscore/lineup data for the base roster, then
overlays post-season transactions to reflect current org assignments.

After initial build, use update_roster.py for daily maintenance.

Usage:
    python scripts/build_roster.py              # build from prior season
    python scripts/build_roster.py --rebuild    # drop & recreate table
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PRIOR_SEASON  # noqa: E402
from lib.db import get_engine, read_sql  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL = """\
CREATE TABLE IF NOT EXISTS production.dim_roster (
    player_id           BIGINT PRIMARY KEY,
    player_name         TEXT NOT NULL,
    org_id              INTEGER NOT NULL,
    roster_status       VARCHAR(20) NOT NULL DEFAULT 'active',
    level               VARCHAR(5),
    primary_position    VARCHAR(4) NOT NULL,
    secondary_positions TEXT[] DEFAULT '{}',
    is_starter          BOOLEAN DEFAULT FALSE,
    team_id             INTEGER,
    team_name           TEXT,
    last_game_date      DATE,
    status_date         DATE NOT NULL DEFAULT CURRENT_DATE,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roster_org
    ON production.dim_roster (org_id);
CREATE INDEX IF NOT EXISTS idx_roster_status_org
    ON production.dim_roster (roster_status, org_id);
CREATE INDEX IF NOT EXISTS idx_roster_level_org
    ON production.dim_roster (level, org_id);
"""

STATUS_MAP = {
    "active": "active",
    "option_minors": "minors",
    "release": "released",
    "IL-7": "il_7",
    "IL-10": "il_10",
    "IL-15": "il_15",
    "IL-60": "il_60",
    "designate": "restricted",
    "trade": "active",
}

# Transaction types that change org assignment
ORG_CHANGE_TYPES = {"TR", "SFA", "SGN", "CLW", "SE", "PUR"}


def _normalize_il_type(il_type: str | None) -> str:
    """Convert il_type like '60-day' or 'IL-10' to 'il_60' / 'il_10'."""
    if not il_type:
        return "il_10"
    import re
    m = re.search(r"(\d+)", il_type)
    return f"il_{m.group(1)}" if m else "il_10"


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def create_table(rebuild: bool = False) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        if rebuild:
            conn.execute(text("DROP TABLE IF EXISTS production.dim_roster"))
            logger.info("Dropped existing dim_roster table")
        conn.execute(text(DDL))
    logger.info("Table production.dim_roster ready")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_mlb_appearances(season: int) -> pd.DataFrame:
    """Most recent MLB game per player in the given season."""
    return read_sql("""
        SELECT DISTINCT ON (player_id)
            player_id, team_id, game_date AS last_game_date, player_role
        FROM production.fact_player_game_mlb
        WHERE season = :season
        ORDER BY player_id, game_date DESC
    """, {"season": season})


def fetch_milb_appearances(season: int) -> pd.DataFrame:
    """Most recent MiLB game per player in the given season."""
    return read_sql("""
        SELECT DISTINCT ON (player_id)
            player_id, team_id, team_name,
            parent_org_id AS org_id, level,
            game_date AS last_game_date, player_role
        FROM production.fact_milb_player_game
        WHERE season = :season
        ORDER BY player_id, game_date DESC
    """, {"season": season})


def fetch_position_counts() -> pd.DataFrame:
    """Career position game counts from MLB lineups + MiLB batting logs."""
    mlb = read_sql("""
        SELECT player_id, position, COUNT(*) AS games
        FROM production.fact_lineup
        WHERE position NOT IN ('PH', 'PR')
        GROUP BY player_id, position
    """)
    milb = read_sql("""
        SELECT batter_id AS player_id, position, COUNT(*) AS games
        FROM staging.milb_batting_game_logs
        WHERE position NOT IN ('PH', 'PR')
        GROUP BY batter_id, position
    """)
    combined = pd.concat([mlb, milb], ignore_index=True)
    return combined.groupby(["player_id", "position"], as_index=False)["games"].sum()


def fetch_pitcher_classification(season: int) -> pd.DataFrame:
    """Classify pitchers as SP/RP from recent seasons (last 2 years)."""
    mlb = read_sql("""
        SELECT player_id,
               SUM(CASE WHEN pit_is_starter THEN 1 ELSE 0 END) AS starts,
               COUNT(*) AS appearances
        FROM production.fact_player_game_mlb
        WHERE player_role = 'pitcher' AND season >= :s
        GROUP BY player_id
    """, {"s": season - 1})
    milb = read_sql("""
        SELECT player_id,
               SUM(CASE WHEN pit_is_starter THEN 1 ELSE 0 END) AS starts,
               COUNT(*) AS appearances
        FROM production.fact_milb_player_game
        WHERE player_role = 'pitcher' AND season >= :s
        GROUP BY player_id
    """, {"s": season - 1})
    combined = pd.concat([mlb, milb], ignore_index=True)
    agg = combined.groupby("player_id", as_index=False).agg(
        starts=("starts", "sum"), appearances=("appearances", "sum")
    )
    agg["is_starter"] = agg["starts"] / agg["appearances"] > 0.5
    return agg[["player_id", "is_starter"]]


def fetch_current_statuses() -> pd.DataFrame:
    """Most recent status per player from the timeline table."""
    return read_sql("""
        SELECT DISTINCT ON (player_id)
            player_id, status_type, status_start_date
        FROM production.fact_player_status_timeline
        ORDER BY player_id, status_start_date DESC
    """)


def fetch_player_names() -> pd.DataFrame:
    """Player names and API positions from dim_player."""
    return read_sql("""
        SELECT player_id, player_name, primary_position AS api_position
        FROM production.dim_player
    """)


def fetch_milb_player_names() -> pd.DataFrame:
    """Names for MiLB-only players not in dim_player."""
    batters = read_sql("""
        SELECT DISTINCT ON (batter_id)
            batter_id AS player_id, batter_name AS player_name
        FROM staging.milb_batting_game_logs
        ORDER BY batter_id, game_date DESC
    """)
    pitchers = read_sql("""
        SELECT DISTINCT ON (pitcher_id)
            pitcher_id AS player_id, pitcher_name AS player_name
        FROM staging.milb_pitching_game_logs
        ORDER BY pitcher_id, game_date DESC
    """)
    combined = pd.concat([batters, pitchers], ignore_index=True)
    return combined.drop_duplicates("player_id", keep="first")


def fetch_post_season_transactions(season: int) -> pd.DataFrame:
    """Transactions after the regular season for roster overlay."""
    return read_sql("""
        SELECT *
        FROM production.dim_transaction
        WHERE effective_date > :cutoff
          AND to_team_id BETWEEN 108 AND 158
        ORDER BY effective_date, transaction_id
    """, {"cutoff": f"{season}-10-01"})


def fetch_team_names() -> pd.DataFrame:
    return read_sql("SELECT team_id, team_name FROM production.dim_team")


# ---------------------------------------------------------------------------
# Position determination
# ---------------------------------------------------------------------------

def determine_positions(
    pos_counts: pd.DataFrame, min_secondary: int = 10,
) -> pd.DataFrame:
    """Primary = most-played fielding position. Secondary = others with ≥10 games."""
    if pos_counts.empty:
        return pd.DataFrame(columns=["player_id", "primary_position", "secondary_positions"])

    # Prefer fielding positions over DH/P for primary
    fielding = pos_counts[~pos_counts["position"].isin(["DH", "P"])].copy()
    has_fielding = set(fielding["player_id"].unique())
    # Fall back to DH/P for players with no fielding data
    fallback = pos_counts[~pos_counts["player_id"].isin(has_fielding)]
    working = pd.concat([fielding, fallback], ignore_index=True)

    # Primary = position with most games
    idx_max = working.groupby("player_id")["games"].idxmax()
    primary = working.loc[idx_max, ["player_id", "position"]].rename(
        columns={"position": "primary_position"},
    )

    # Secondary = other positions with ≥ min_secondary games (include DH, exclude PH/PR)
    merged = pos_counts.merge(primary, on="player_id")
    sec = merged[
        (merged["position"] != merged["primary_position"])
        & (merged["games"] >= min_secondary)
        & (~merged["position"].isin(["PH", "PR"]))
    ]
    sec_agg = (
        sec.groupby("player_id")["position"]
        .apply(list)
        .rename("secondary_positions")
    )

    result = primary.merge(sec_agg, on="player_id", how="left")
    result["secondary_positions"] = result["secondary_positions"].apply(
        lambda x: x if isinstance(x, list) else [],
    )
    return result


# ---------------------------------------------------------------------------
# Transaction overlay
# ---------------------------------------------------------------------------

def apply_transactions(
    roster: pd.DataFrame,
    txns: pd.DataFrame,
    names_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply post-season transactions to update org assignments and statuses."""
    roster = roster.copy()
    new_players: list[dict] = []

    for _, txn in txns.iterrows():
        pid = txn["player_id"]
        mask = roster["player_id"] == pid
        eff = txn["effective_date"]

        # --- Org-changing transactions ---
        if txn["type_code"] in ORG_CHANGE_TYPES:
            new_org = int(txn["to_team_id"])
            is_minor = "minor league" in str(txn.get("description", "")).lower()
            new_status = "minors" if is_minor else "active"
            new_level = None if is_minor else "MLB"

            if mask.any():
                roster.loc[mask, "org_id"] = new_org
                roster.loc[mask, "team_id"] = new_org
                roster.loc[mask, "team_name"] = txn["to_team_name"]
                roster.loc[mask, "roster_status"] = new_status
                if new_level:
                    roster.loc[mask, "level"] = new_level
                roster.loc[mask, "status_date"] = eff
            else:
                # New player — collect for batch insert
                name_row = names_df[names_df["player_id"] == pid]
                name = (
                    name_row["player_name"].iloc[0]
                    if len(name_row) > 0
                    else txn.get("player_name", "Unknown")
                )
                api_pos = (
                    name_row["api_position"].iloc[0]
                    if len(name_row) > 0 and "api_position" in name_row.columns
                    else "UTIL"
                )
                new_players.append({
                    "player_id": pid,
                    "player_name": name,
                    "org_id": new_org,
                    "roster_status": new_status,
                    "level": new_level,
                    "primary_position": api_pos or "UTIL",
                    "secondary_positions": [],
                    "is_starter": False,
                    "team_id": new_org,
                    "team_name": txn["to_team_name"],
                    "last_game_date": None,
                    "status_date": eff,
                    "player_role": None,
                    "api_position": api_pos,
                    "status_type": None,
                    "status_start_date": None,
                })

        elif txn["type_code"] == "OPT" and mask.any():
            roster.loc[mask, "roster_status"] = "minors"
            roster.loc[mask, "level"] = "AAA"
            roster.loc[mask, "status_date"] = eff

        elif txn["type_code"] == "CU" and mask.any():
            roster.loc[mask, "roster_status"] = "active"
            roster.loc[mask, "level"] = "MLB"
            roster.loc[mask, "status_date"] = eff

        elif txn["type_code"] == "DES" and mask.any():
            roster.loc[mask, "roster_status"] = "restricted"
            roster.loc[mask, "status_date"] = eff

        elif txn["type_code"] == "REL" and mask.any():
            roster.loc[mask, "roster_status"] = "released"
            roster.loc[mask, "status_date"] = eff

        # --- IL transactions (can accompany any type_code) ---
        if txn.get("is_il_placement") and mask.any():
            roster.loc[mask, "roster_status"] = _normalize_il_type(txn.get("il_type"))
            roster.loc[mask, "status_date"] = eff
        elif txn.get("is_il_activation") and mask.any():
            roster.loc[mask, "roster_status"] = "active"
            roster.loc[mask, "status_date"] = eff
        elif txn.get("is_il_transfer") and txn.get("il_type") and mask.any():
            roster.loc[mask, "roster_status"] = _normalize_il_type(txn["il_type"])
            roster.loc[mask, "status_date"] = eff

    # Append new players discovered via transactions
    if new_players:
        new_df = pd.DataFrame(new_players)
        # Deduplicate — keep last transaction per player
        new_df = new_df.drop_duplicates("player_id", keep="last")
        # Exclude any that now exist in roster (from earlier txn in same batch)
        new_df = new_df[~new_df["player_id"].isin(roster["player_id"])]
        roster = pd.concat([roster, new_df], ignore_index=True)
        logger.info("Added %d new players from transactions", len(new_df))

    return roster


# ---------------------------------------------------------------------------
# Write to database
# ---------------------------------------------------------------------------

FINAL_COLS = [
    "player_id", "player_name", "org_id", "roster_status", "level",
    "primary_position", "secondary_positions", "is_starter",
    "team_id", "team_name", "last_game_date", "status_date",
]


def write_roster(df: pd.DataFrame) -> None:
    """Write the roster DataFrame to production.dim_roster via temp table."""
    engine = get_engine()
    out = df[FINAL_COLS].copy()

    # Convert secondary_positions list → PostgreSQL array literal string
    out["secondary_positions"] = out["secondary_positions"].apply(
        lambda x: "{" + ",".join(x) + "}" if isinstance(x, list) and x else "{}",
    )

    # Ensure correct types for to_sql
    out["player_id"] = out["player_id"].astype("int64")
    out["org_id"] = out["org_id"].astype("int64")
    out["is_starter"] = out["is_starter"].astype(bool)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")

    # Write to temp table
    out.to_sql("_tmp_roster", engine, schema="production", if_exists="replace", index=False)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE production.dim_roster"))
        conn.execute(text("""
            INSERT INTO production.dim_roster
                (player_id, player_name, org_id, roster_status, level,
                 primary_position, secondary_positions, is_starter,
                 team_id, team_name, last_game_date, status_date)
            SELECT
                player_id, player_name, org_id::integer, roster_status, level,
                primary_position, secondary_positions::text[], is_starter::boolean,
                team_id::integer, team_name, last_game_date::date, status_date::date
            FROM production._tmp_roster
        """))
        conn.execute(text("DROP TABLE IF EXISTS production._tmp_roster"))

    logger.info("Wrote %d players to production.dim_roster", len(out))


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(season: int = PRIOR_SEASON, rebuild: bool = False) -> None:
    create_table(rebuild=rebuild)
    logger.info("Building roster from %d season data...", season)

    # ---- Fetch all source data ----
    mlb = fetch_mlb_appearances(season)
    milb = fetch_milb_appearances(season)
    pos_counts = fetch_position_counts()
    pitcher_cls = fetch_pitcher_classification(season)
    statuses = fetch_current_statuses()
    dim_names = fetch_player_names()
    milb_names = fetch_milb_player_names()
    teams = fetch_team_names()

    logger.info(
        "Fetched: %d MLB, %d MiLB, %d position records",
        len(mlb), len(milb), len(pos_counts),
    )

    # ---- Build player universe ----
    mlb_df = mlb.copy()
    mlb_df["org_id"] = mlb_df["team_id"]
    mlb_df["level"] = "MLB"

    milb_df = milb[~milb["player_id"].isin(mlb_df["player_id"])].copy()

    roster = pd.concat(
        [
            mlb_df[["player_id", "org_id", "team_id", "level",
                     "last_game_date", "player_role"]],
            milb_df[["player_id", "org_id", "team_id", "team_name",
                      "level", "last_game_date", "player_role"]],
        ],
        ignore_index=True,
    )

    # ---- Player names ----
    all_names = pd.concat(
        [dim_names[["player_id", "player_name"]], milb_names], ignore_index=True,
    ).drop_duplicates("player_id", keep="first")
    roster = roster.merge(all_names, on="player_id", how="left")
    roster["player_name"] = roster["player_name"].fillna("Unknown")

    # Keep api_position for fallback
    roster = roster.merge(
        dim_names[["player_id", "api_position"]], on="player_id", how="left",
    )

    # ---- Team names for MLB teams ----
    roster = roster.merge(
        teams.rename(columns={"team_name": "mlb_team_name"}),
        on="team_id", how="left",
    )
    roster["team_name"] = roster["team_name"].fillna(roster["mlb_team_name"])
    roster.drop(columns=["mlb_team_name"], inplace=True, errors="ignore")

    # ---- Positions ----
    positions = determine_positions(pos_counts)
    roster = roster.merge(positions, on="player_id", how="left")

    # ---- Pitcher handling ----
    pitcher_mask = roster["player_role"] == "pitcher"
    roster = roster.merge(pitcher_cls, on="player_id", how="left")
    roster["is_starter"] = roster["is_starter"].fillna(False)

    # Set SP/RP for pitchers without a fielding-based primary position
    needs_pos = pitcher_mask & roster["primary_position"].isna()
    roster.loc[needs_pos & roster["is_starter"], "primary_position"] = "SP"
    roster.loc[needs_pos & ~roster["is_starter"], "primary_position"] = "RP"

    # Override 'P' primary to SP/RP
    p_primary = pitcher_mask & (roster["primary_position"] == "P")
    roster.loc[p_primary & roster["is_starter"], "primary_position"] = "SP"
    roster.loc[p_primary & ~roster["is_starter"], "primary_position"] = "RP"

    # Fix pitchers misclassified via lineup data (e.g., pitcher who DH'd once)
    api_pitcher = roster["api_position"] == "P"
    misclassified = api_pitcher & roster["primary_position"].isin(["DH", "UTIL"])
    roster.loc[misclassified & roster["is_starter"], "primary_position"] = "SP"
    roster.loc[misclassified & ~roster["is_starter"], "primary_position"] = "RP"

    # ---- Fallback positions ----
    no_pos = roster["primary_position"].isna()
    roster.loc[no_pos, "primary_position"] = roster.loc[no_pos, "api_position"]
    roster.loc[roster["primary_position"].isna(), "primary_position"] = "UTIL"
    roster["secondary_positions"] = roster["secondary_positions"].apply(
        lambda x: x if isinstance(x, list) else [],
    )

    # ---- Status from timeline ----
    roster = roster.merge(
        statuses[["player_id", "status_type", "status_start_date"]],
        on="player_id", how="left",
    )
    roster["roster_status"] = roster["status_type"].map(STATUS_MAP).fillna("active")
    # MiLB players without a status default to 'minors'
    milb_mask = roster["level"] != "MLB"
    roster.loc[milb_mask & roster["status_type"].isna(), "roster_status"] = "minors"

    roster["status_date"] = roster["status_start_date"].fillna(roster["last_game_date"])
    roster["status_date"] = roster["status_date"].fillna(pd.Timestamp(f"{season}-04-01"))

    # ---- Post-season transaction overlay ----
    txns = fetch_post_season_transactions(season)
    logger.info("Applying %d post-season transactions", len(txns))
    roster = apply_transactions(roster, txns, dim_names)

    # ---- Final cleanup ----
    roster = roster[roster["org_id"].between(108, 158)].copy()
    roster = roster.drop_duplicates("player_id", keep="last")
    roster.drop(
        columns=["player_role", "api_position", "status_type",
                 "status_start_date", "mlb_team_name"],
        inplace=True, errors="ignore",
    )

    # ---- Write ----
    write_roster(roster)

    # ---- Summary ----
    by_level = roster["level"].value_counts()
    logger.info(
        "Done: %d players across %d orgs",
        len(roster), roster["org_id"].nunique(),
    )
    for lvl, cnt in sorted(by_level.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %-5s %d", lvl, cnt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dim_roster table")
    parser.add_argument(
        "--season", type=int, default=PRIOR_SEASON,
        help=f"Season to build from (default: {PRIOR_SEASON})",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Drop and recreate the table before building",
    )
    args = parser.parse_args()
    build(season=args.season, rebuild=args.rebuild)
