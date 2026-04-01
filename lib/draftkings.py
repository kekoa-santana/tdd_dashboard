"""DraftKings player prop odds scraper.

Fetches MLB player props (K, H, HR, TB, BB, R) from DraftKings'
sportscontent API. Used as a Vegas baseline to compare against TDD
model projections.

Requires ``curl_cffi`` for TLS fingerprinting (regular urllib/requests
gets 403 from Cloudflare).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_BASE = (
    "https://sportsbook-nash.draftkings.com/api/sportscontent"
    "/dkusoh/v1/leagues/84240"
)

# (category_id, subcategory_id, stat_name, player_type)
_PROP_ENDPOINTS: list[tuple[int, int, str, str]] = [
    (1031, 15221, "K", "pitcher"),     # Strikeouts Thrown O/U
    (1031, 9886, "H", "pitcher"),      # Hits Allowed O/U
    (1031, 15219, "BB", "pitcher"),    # Walks Allowed O/U
    (1031, 17413, "Outs", "pitcher"),  # Outs Recorded O/U
    (743, 6719, "H", "batter"),        # Hits O/U
    (743, 6607, "TB", "batter"),       # Total Bases O/U
    (743, 17411, "BB", "batter"),      # Walks (Batter) O/U
    (743, 17407, "R", "batter"),       # Runs O/U
    (743, 8025, "RBI", "batter"),      # RBIs O/U
    (743, 17406, "HRR", "batter"),     # Hits + Runs + RBIs O/U
]

# HR has no O/U market -- use milestone 1+ as Over 0.5 proxy
_HR_ENDPOINT = (743, 17319, "HR", "batter")  # Home Runs milestone


def _american_to_implied(american: str | int | float) -> float | None:
    """Convert American odds string to implied probability."""
    try:
        cleaned = str(american).replace("\u2212", "-").replace("\u2013", "-")
        odds = int(cleaned)
    except (ValueError, TypeError):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _fetch_subcategory(
    cat_id: int, subcat_id: int
) -> dict[str, Any]:
    """Fetch a single category/subcategory from the DK API."""
    from curl_cffi import requests as cffi_requests

    url = f"{_BASE}/categories/{cat_id}/subcategories/{subcat_id}"
    resp = cffi_requests.get(url, impersonate="chrome110", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_ou_props(
    data: dict[str, Any],
    stat: str,
    player_type: str,
) -> list[dict[str, Any]]:
    """Parse Over/Under selections into prop rows."""
    markets_by_id = {m["id"]: m for m in data.get("markets", [])}
    selections = data.get("selections", [])
    events_by_id = {e["id"]: e for e in data.get("events", [])}

    # Group selections by market
    by_market: dict[str, dict[str, Any]] = {}
    for sel in selections:
        mid = sel["marketId"]
        otype = sel.get("outcomeType", sel.get("label", ""))
        if mid not in by_market:
            by_market[mid] = {}
        by_market[mid][otype] = sel

    rows: list[dict[str, Any]] = []
    for mid, sides in by_market.items():
        over = sides.get("Over")
        under = sides.get("Under")
        if not over:
            continue

        mkt = markets_by_id.get(mid, {})
        event = events_by_id.get(mkt.get("eventId", ""), {})
        player = over["participants"][0] if over.get("participants") else {}

        over_american = over["displayOdds"]["american"]
        under_american = under["displayOdds"]["american"] if under else None

        rows.append({
            "game_description": event.get("name", ""),
            "player_name": player.get("name", ""),
            "player_type": player_type,
            "stat": stat,
            "line": float(over.get("points", 0)),
            "over_odds": over_american,
            "under_odds": under_american,
            "over_implied": _american_to_implied(over_american),
            "under_implied": _american_to_implied(under_american),
        })
    return rows


def _parse_milestone_hr(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse HR milestone 1+ selections as Over 0.5 proxies."""
    markets_by_id = {m["id"]: m for m in data.get("markets", [])}
    events_by_id = {e["id"]: e for e in data.get("events", [])}
    rows: list[dict[str, Any]] = []

    for sel in data.get("selections", []):
        if sel.get("milestoneValue") != 1:
            continue

        mid = sel["marketId"]
        mkt = markets_by_id.get(mid, {})
        event = events_by_id.get(mkt.get("eventId", ""), {})
        player = sel["participants"][0] if sel.get("participants") else {}
        american = sel["displayOdds"]["american"]

        rows.append({
            "game_description": event.get("name", ""),
            "player_name": player.get("name", ""),
            "player_type": "batter",
            "stat": "HR",
            "line": 0.5,
            "over_odds": american,
            "under_odds": None,
            "over_implied": _american_to_implied(american),
            "under_implied": None,
        })
    return rows


def fetch_dk_player_props() -> pd.DataFrame:
    """Fetch current MLB player prop O/U odds from DraftKings.

    Returns
    -------
    pd.DataFrame
        Columns: game_description, player_name, player_type, stat,
        line, over_odds, under_odds, over_implied, under_implied.
    """
    all_rows: list[dict[str, Any]] = []

    # O/U props
    for cat_id, subcat_id, stat, ptype in _PROP_ENDPOINTS:
        try:
            data = _fetch_subcategory(cat_id, subcat_id)
            rows = _parse_ou_props(data, stat, ptype)
            all_rows.extend(rows)
            logger.debug("DK %s: %d props", stat, len(rows))
        except Exception as e:
            logger.warning("DK fetch failed for %s (cat=%d/sub=%d): %s",
                           stat, cat_id, subcat_id, e)

    # HR milestones
    cat_id, subcat_id, stat, ptype = _HR_ENDPOINT
    try:
        data = _fetch_subcategory(cat_id, subcat_id)
        rows = _parse_milestone_hr(data)
        all_rows.extend(rows)
        logger.debug("DK HR milestones: %d props", len(rows))
    except Exception as e:
        logger.warning("DK fetch failed for HR milestones: %s", e)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        logger.info(
            "Fetched %d DK player props across %d players",
            len(df), df["player_name"].nunique(),
        )
    else:
        logger.warning("No DraftKings player props found")
    return df
