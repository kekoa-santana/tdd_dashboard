"""Bovada player prop odds scraper.

Fetches MLB player props (K, H, HR, BB, TB) from Bovada's public
JSON API.  Used as a Vegas baseline to compare against TDD model
projections.

No authentication required — uses the same public endpoint that
powers the Bovada website.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BOVADA_MLB_URL = (
    "https://www.bovada.lv/services/sports/event/v2/events/A/"
    "description/baseball/mlb"
)

# Map Bovada market descriptions to our stat names
_MARKET_PATTERNS: list[tuple[str, str]] = [
    (r"Total Strikeouts\s*-\s*(.+)", "K"),
    (r"Total Hits\s*-\s*(.+)", "H"),
    (r"Total Home Runs\s*-\s*(.+)", "HR"),
    (r"Total Bases\s*-\s*(.+)", "TB"),
    (r"Total Pitcher Walks\s*-\s*(.+)", "BB"),
    (r"Total Walks\s*-\s*(.+)", "BB"),
    (r"Total Runs\s*-\s*(.+)", "R"),
]


def _american_to_implied(american: str | int | float) -> float | None:
    """Convert American odds to implied probability."""
    try:
        odds = int(american) if american != "EVEN" else 100
    except (ValueError, TypeError):
        if str(american).upper() == "EVEN":
            odds = 100
        else:
            return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _extract_team_abbr(name: str) -> str:
    """Extract team abbreviation from player name like 'Aaron Judge (NYY)'."""
    m = re.search(r"\((\w{2,3})\)\s*$", name)
    return m.group(1) if m else ""


def _extract_player_name(name: str) -> str:
    """Strip team abbreviation from player name."""
    return re.sub(r"\s*\(\w{2,3}\)\s*$", "", name).strip()


def fetch_bovada_props() -> pd.DataFrame:
    """Fetch current MLB player prop odds from Bovada.

    Returns
    -------
    pd.DataFrame
        Columns: game_description, player_name, team_abbr, stat,
        line, over_odds, under_odds, over_implied, under_implied.
    """
    req = urllib.request.Request(
        BOVADA_MLB_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch Bovada odds: %s", e)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for group in data:
        for event in group.get("events", []):
            game_desc = event.get("description", "")

            for dg in event.get("displayGroups", []):
                dg_desc = dg.get("description", "")
                if "prop" not in dg_desc.lower():
                    continue

                for market in dg.get("markets", []):
                    market_desc = market.get("description", "")

                    # Match market description to a stat
                    stat = None
                    player_raw = None
                    for pattern, stat_name in _MARKET_PATTERNS:
                        m = re.match(pattern, market_desc, re.IGNORECASE)
                        if m:
                            stat = stat_name
                            player_raw = m.group(1).strip()
                            break

                    if not stat or not player_raw:
                        continue

                    player_name = _extract_player_name(player_raw)
                    team_abbr = _extract_team_abbr(player_raw)

                    # Extract over/under outcomes
                    over_odds = None
                    under_odds = None
                    line = None

                    for outcome in market.get("outcomes", []):
                        desc = outcome.get("description", "").lower()
                        price = outcome.get("price", {})
                        american = price.get("american", "")
                        handicap = price.get("handicap", "")

                        if "over" in desc:
                            over_odds = american
                            if handicap:
                                try:
                                    line = float(handicap)
                                except (ValueError, TypeError):
                                    pass
                        elif "under" in desc:
                            under_odds = american
                            if handicap and line is None:
                                try:
                                    line = float(handicap)
                                except (ValueError, TypeError):
                                    pass

                    if line is not None and over_odds is not None:
                        rows.append({
                            "game_description": game_desc,
                            "player_name": player_name,
                            "team_abbr": team_abbr,
                            "stat": stat,
                            "line": line,
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                            "over_implied": _american_to_implied(over_odds),
                            "under_implied": _american_to_implied(under_odds),
                        })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "Fetched %d Bovada props across %d games",
            len(df), df["game_description"].nunique(),
        )
    else:
        logger.warning("No Bovada props found")
    return df
