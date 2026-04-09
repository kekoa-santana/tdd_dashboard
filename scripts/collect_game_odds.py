#!/usr/bin/env python
"""Collect and store game-level odds (moneyline, spread, over/under).

Fetches from DraftKings and Bovada, normalizes into a common schema,
and appends to ``game_odds_history.parquet`` with a snapshot timestamp.

Usage
-----
    python scripts/collect_game_odds.py                  # today
    python scripts/collect_game_odds.py --date 2026-04-15
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.bovada import fetch_bovada_game_odds  # noqa: E402
from lib.draftkings import fetch_dk_game_odds  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
HISTORY_PATH = DASHBOARD_DIR / "game_odds_history.parquet"

# Common output schema
_COLUMNS = [
    "snapshot_ts",       # ISO timestamp of when odds were captured
    "game_date",         # YYYY-MM-DD
    "source",            # "dk" or "bovada"
    "game_description",  # e.g. "NYY @ BOS"
    "market_type",       # "moneyline", "spread", "total"
    "team_or_label",     # team name or Over/Under
    "outcome_type",      # Home/Away for ML/spread, Over/Under for total
    "odds",              # American odds string
    "implied_prob",      # float
    "line",              # float (spread value or total line)
]


def _normalize_bovada(df: pd.DataFrame) -> pd.DataFrame:
    """Add source + outcome_type columns to Bovada game odds."""
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    out = df.copy()
    out["source"] = "bovada"
    # Bovada doesn't include outcome_type; infer from market + label
    out["outcome_type"] = ""
    return out


def _normalize_dk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DraftKings game odds to common schema."""
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    out = df.copy()
    out["source"] = "dk"
    return out


def collect_odds(game_date: str) -> pd.DataFrame:
    """Fetch from all sources, normalize, and return combined DataFrame."""
    snapshot_ts = datetime.now().isoformat()

    frames: list[pd.DataFrame] = []

    # DraftKings
    try:
        dk = fetch_dk_game_odds()
        dk = _normalize_dk(dk)
        if not dk.empty:
            frames.append(dk)
            logger.info("DK: %d odds rows", len(dk))
    except Exception as e:
        logger.warning("DraftKings fetch failed: %s", e)

    # Bovada
    try:
        bov = fetch_bovada_game_odds()
        bov = _normalize_bovada(bov)
        if not bov.empty:
            frames.append(bov)
            logger.info("Bovada: %d odds rows", len(bov))
    except Exception as e:
        logger.warning("Bovada fetch failed: %s", e)

    if not frames:
        logger.warning("No odds collected from any source")
        return pd.DataFrame(columns=_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["snapshot_ts"] = snapshot_ts
    combined["game_date"] = game_date

    # Ensure all columns present
    for col in _COLUMNS:
        if col not in combined.columns:
            combined[col] = None

    return combined[_COLUMNS]


def append_to_history(new_odds: pd.DataFrame) -> int:
    """Append new odds to the cumulative history parquet.

    Returns the total row count after appending.
    """
    if new_odds.empty:
        return 0

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    if HISTORY_PATH.exists():
        existing = pd.read_parquet(HISTORY_PATH)
        combined = pd.concat([existing, new_odds], ignore_index=True)
    else:
        combined = new_odds

    combined.to_parquet(HISTORY_PATH, index=False)
    logger.info(
        "Appended %d rows to game_odds_history.parquet (total: %d)",
        len(new_odds), len(combined),
    )
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect game-level odds")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Game date (YYYY-MM-DD). Default: today.",
    )
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("Collecting game odds for %s", game_date)

    odds = collect_odds(game_date)
    if not odds.empty:
        append_to_history(odds)
    else:
        logger.info("Nothing to store")


if __name__ == "__main__":
    main()
