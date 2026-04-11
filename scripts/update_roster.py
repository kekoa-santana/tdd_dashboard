#!/usr/bin/env python
"""
Daily roster update from transaction data.

Processes new transactions since the last roster update to maintain
production.dim_roster as the source of truth for team rosters.

Handles:
  - Trades, signings, waivers (org changes)
  - Options/recalls (MLB ↔ MiLB)
  - DFA, releases
  - IL placements, activations, transfers

Usage:
    python scripts/update_roster.py                        # since last update
    python scripts/update_roster.py --since 2026-04-01     # from specific date
    python scripts/update_roster.py --refresh-games        # also update from new games
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.db import get_engine, read_sql  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
# Transaction fetching
# ---------------------------------------------------------------------------

def fetch_new_transactions(since: date) -> pd.DataFrame:
    """Get roster-affecting transactions since the given date."""
    # Standard roster moves (filtered to MLB orgs)
    roster_txns = read_sql("""
        SELECT *
        FROM production.dim_transaction
        WHERE effective_date >= :since
          AND type_code IN ('TR', 'SFA', 'SGN', 'CLW', 'SE', 'PUR',
                            'OPT', 'CU', 'DES', 'REL')
          AND to_team_id BETWEEN 108 AND 158
        ORDER BY effective_date, transaction_id
    """, {"since": str(since)})

    # IL transactions (any type_code, any team — applies to known players)
    il_txns = read_sql("""
        SELECT *
        FROM production.dim_transaction
        WHERE effective_date >= :since
          AND (is_il_placement OR is_il_activation OR is_il_transfer)
        ORDER BY effective_date, transaction_id
    """, {"since": str(since)})

    combined = pd.concat([roster_txns, il_txns], ignore_index=True)
    return (
        combined.drop_duplicates("transaction_id")
        .sort_values(["effective_date", "transaction_id"])
    )


# ---------------------------------------------------------------------------
# Transaction processing
# ---------------------------------------------------------------------------

def _get_player_info(conn, player_id: int) -> tuple[str, str]:
    """Look up player name and position from dim_player."""
    result = conn.execute(
        text("""
            SELECT player_name, primary_position
            FROM production.dim_player
            WHERE player_id = :pid
        """),
        {"pid": player_id},
    )
    row = result.fetchone()
    if row:
        return row[0], row[1] or "UTIL"
    return "Unknown", "UTIL"


def _upsert_org_change(
    conn, txn, status: str, level: str | None,
) -> None:
    """Handle transactions that change a player's org (trade, signing, etc.)."""
    pid = int(txn["player_id"])
    org = int(txn["to_team_id"])
    eff = txn["effective_date"]
    team_name = txn["to_team_name"]
    name, pos = _get_player_info(conn, pid)

    # Use player_name from transaction if dim_player lookup failed
    if name == "Unknown" and txn.get("player_name"):
        name = txn["player_name"]

    conn.execute(
        text("""
            INSERT INTO production.dim_roster
                (player_id, player_name, org_id, roster_status, level,
                 primary_position, secondary_positions, is_starter,
                 team_id, team_name, status_date)
            VALUES
                (:pid, :name, :org, :status, :level,
                 :pos, '{}'::text[], FALSE,
                 :org, :team_name, :dt)
            ON CONFLICT (player_id) DO UPDATE SET
                org_id = EXCLUDED.org_id,
                roster_status = EXCLUDED.roster_status,
                level = COALESCE(EXCLUDED.level, production.dim_roster.level),
                team_id = EXCLUDED.team_id,
                team_name = EXCLUDED.team_name,
                status_date = EXCLUDED.status_date,
                updated_at = NOW()
        """),
        {
            "pid": pid, "name": name, "org": org,
            "status": status, "level": level,
            "pos": pos, "team_name": team_name, "dt": eff,
        },
    )


def _update_status(
    conn, player_id: int, status: str, eff_date, **extra,
) -> None:
    """Update roster_status and optionally other fields for an existing player."""
    sets = ["roster_status = :status", "status_date = :dt", "updated_at = NOW()"]
    params: dict = {"pid": player_id, "status": status, "dt": eff_date}
    for key, val in extra.items():
        sets.append(f"{key} = :{key}")
        params[key] = val
    sql = f"UPDATE production.dim_roster SET {', '.join(sets)} WHERE player_id = :pid"
    conn.execute(text(sql), params)


def process_transaction(conn, txn) -> bool:
    """Process a single transaction. Returns True if a change was made."""
    pid = int(txn["player_id"])
    eff = txn["effective_date"]
    changed = False

    # --- Org-changing transactions ---
    if txn["type_code"] in ORG_CHANGE_TYPES:
        desc = str(txn.get("description", "")).lower()
        is_minor = "minor league" in desc
        status = "minors" if is_minor else "active"
        level = None if is_minor else "MLB"
        _upsert_org_change(conn, txn, status, level)
        changed = True

    elif txn["type_code"] == "OPT":
        _update_status(conn, pid, "minors", eff, level="AAA")
        changed = True

    elif txn["type_code"] == "CU":
        _update_status(conn, pid, "active", eff, level="MLB")
        changed = True

    elif txn["type_code"] == "DES":
        _update_status(conn, pid, "restricted", eff)
        changed = True

    elif txn["type_code"] == "REL":
        _update_status(conn, pid, "released", eff)
        changed = True

    # --- IL transactions (can accompany any type_code) ---
    if txn.get("is_il_placement"):
        _update_status(conn, pid, _normalize_il_type(txn.get("il_type")), eff)
        changed = True
    elif txn.get("is_il_activation"):
        _update_status(conn, pid, "active", eff)
        changed = True
    elif txn.get("is_il_transfer") and txn.get("il_type"):
        _update_status(conn, pid, _normalize_il_type(txn["il_type"]), eff)
        changed = True

    return changed


# ---------------------------------------------------------------------------
# Game data refresh
# ---------------------------------------------------------------------------

def refresh_from_games(since: date) -> int:
    """Update last_game_date and team from recent game appearances."""
    engine = get_engine()

    # Get most recent game per player since the cutoff
    recent = read_sql("""
        SELECT DISTINCT ON (player_id)
            player_id, team_id, game_date
        FROM production.fact_player_game_mlb
        WHERE game_date >= :since
        ORDER BY player_id, game_date DESC
    """, {"since": str(since)})

    if recent.empty:
        return 0

    updated = 0
    with engine.begin() as conn:
        for _, row in recent.iterrows():
            result = conn.execute(
                text("""
                    UPDATE production.dim_roster
                    SET last_game_date = :gd,
                        team_id = :tid,
                        org_id = :tid,
                        level = 'MLB',
                        roster_status = CASE
                            WHEN roster_status LIKE 'il_%' THEN roster_status
                            ELSE 'active'
                        END,
                        updated_at = NOW()
                    WHERE player_id = :pid
                      AND (last_game_date IS NULL OR last_game_date < :gd)
                """),
                {
                    "pid": int(row["player_id"]),
                    "tid": int(row["team_id"]),
                    "gd": row["game_date"],
                },
            )
            updated += result.rowcount

    return updated


# ---------------------------------------------------------------------------
# Main update
# ---------------------------------------------------------------------------

def update(since: date | None = None, refresh_games: bool = False) -> None:
    """Process new transactions and optionally refresh from game data."""
    engine = get_engine()

    # Determine cutoff date
    if since is None:
        result = read_sql("""
            SELECT MAX(updated_at)::date AS last_update
            FROM production.dim_roster
        """)
        since = result["last_update"].iloc[0]
        if pd.isna(since):
            logger.error("dim_roster is empty — run build_roster.py first")
            sys.exit(1)

    logger.info("Updating roster from transactions since %s", since)

    # Fetch and process transactions
    txns = fetch_new_transactions(since)
    if txns.empty:
        logger.info("No new transactions since %s", since)
    else:
        logger.info("Processing %d transactions", len(txns))
        changes = 0
        with engine.begin() as conn:
            for _, txn in txns.iterrows():
                if process_transaction(conn, txn):
                    changes += 1
        logger.info("Applied %d roster changes", changes)

    # Optionally refresh from game data
    if refresh_games:
        game_updates = refresh_from_games(since)
        logger.info("Updated %d players from game data", game_updates)

    # Summary
    summary = read_sql("""
        SELECT roster_status, COUNT(*) AS cnt
        FROM production.dim_roster
        WHERE org_id BETWEEN 108 AND 158
        GROUP BY roster_status
        ORDER BY cnt DESC
    """)
    logger.info("Roster summary:")
    for _, row in summary.iterrows():
        logger.info("  %-12s %d", row["roster_status"], row["cnt"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily roster update")
    parser.add_argument(
        "--since", type=str, default=None,
        help="Process transactions from this date (YYYY-MM-DD). Default: last update.",
    )
    parser.add_argument(
        "--refresh-games", action="store_true",
        help="Also update last_game_date from recent game appearances",
    )
    args = parser.parse_args()

    since_date = date.fromisoformat(args.since) if args.since else None
    update(since=since_date, refresh_games=args.refresh_games)
