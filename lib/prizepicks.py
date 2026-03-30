"""Prize Picks MLB player prop scraper.

Fetches MLB player props from the Prize Picks partner API.
No authentication required. Single request returns all current lines.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_URL = (
    "https://partner-api.prizepicks.com/projections"
    "?league_id=2&per_page=1000"
)

# Map Prize Picks stat names to our internal stat keys
_STAT_MAP: dict[str, tuple[str, str]] = {
    "Pitcher Strikeouts": ("K", "pitcher"),
    "Hitter Strikeouts": ("K", "batter"),
    "Hits": ("H", "batter"),
    "Total Bases": ("TB", "batter"),
    "Hits+Runs+RBIs": ("HRR", "batter"),
    "Home Runs": ("HR", "batter"),
    "RBIs": ("RBI", "batter"),
    "Runs": ("R", "batter"),
    "Walks": ("BB", "batter"),
}


def fetch_pp_player_props() -> pd.DataFrame:
    """Fetch current MLB player props from Prize Picks.

    Returns
    -------
    pd.DataFrame
        Columns: player_name, player_type, stat, line, team,
        odds_type, pp_stat_type.
    """
    import urllib.request
    import json

    req = urllib.request.Request(_URL, headers={
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Prize Picks fetch failed: %s", e)
        return pd.DataFrame()

    data = payload.get("data", [])
    included = payload.get("included", [])

    if not data:
        logger.warning("No Prize Picks projections found")
        return pd.DataFrame()

    # Build lookup tables from included objects
    players: dict[str, dict[str, Any]] = {}
    stat_types: dict[str, str] = {}
    for obj in included:
        if obj["type"] == "new_player":
            players[obj["id"]] = obj["attributes"]
        elif obj["type"] == "stat_type":
            stat_types[obj["id"]] = obj["attributes"]["name"]

    rows: list[dict[str, Any]] = []
    for proj in data:
        attrs = proj.get("attributes", {})
        rels = proj.get("relationships", {})

        # Resolve stat type
        st_id = rels.get("stat_type", {}).get("data", {}).get("id", "")
        pp_stat = stat_types.get(st_id, attrs.get("stat_type", ""))

        mapping = _STAT_MAP.get(pp_stat)
        if not mapping:
            continue
        stat_key, player_type = mapping

        # Resolve player
        player_id = rels.get("new_player", {}).get("data", {}).get("id", "")
        player = players.get(player_id, {})
        player_name = player.get("display_name", player.get("name", ""))

        rows.append({
            "player_name": player_name,
            "player_type": player_type,
            "stat": stat_key,
            "line": float(attrs.get("line_score", 0)),
            "team": player.get("team", attrs.get("description", "")),
            "odds_type": attrs.get("odds_type", "standard"),
            "pp_stat_type": pp_stat,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "Fetched %d Prize Picks props across %d players",
            len(df), df["player_name"].nunique(),
        )
    else:
        logger.warning("No matching Prize Picks MLB props found")
    return df
