"""DraftKings MLB odds & DFS scraper.

Fetches MLB player props (K, H, HR, TB, BB, R) and game-level odds
(moneyline, run line, total) from DraftKings' sportscontent API.

Also fetches DFS classic salaries and tier contest assignments from
DraftKings' public draftgroups API (no auth required).

Requires ``curl_cffi`` for TLS fingerprinting on the sportsbook API
(regular urllib/requests gets 403 from Cloudflare).  The DFS API
uses standard ``requests``.
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


# -----------------------------------------------------------------------
# Game-level odds (moneyline, run line, total)
# -----------------------------------------------------------------------
_GAME_LINES_CATEGORY = 493


def fetch_dk_game_odds() -> pd.DataFrame:
    """Fetch game-level MLB odds from DraftKings.

    Returns
    -------
    pd.DataFrame
        Columns: game_description, market_type, team_or_label,
        outcome_type, odds, implied_prob, line.

        ``market_type`` is one of ``'moneyline'``, ``'spread'``, ``'total'``.
        ``outcome_type`` is ``'Home'``/``'Away'`` for ML/spread,
        ``'Over'``/``'Under'`` for totals.
    """
    from curl_cffi import requests as cffi_requests

    url = f"{_BASE}/categories/{_GAME_LINES_CATEGORY}"
    try:
        resp = cffi_requests.get(url, impersonate="chrome110", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("DK game odds fetch failed: %s", e)
        return pd.DataFrame()

    events_by_id = {e["id"]: e for e in data.get("events", [])}
    markets_by_id = {m["id"]: m for m in data.get("markets", [])}

    _MKT_MAP = {
        "moneyline": "moneyline",
        "run line": "spread",
        "total": "total",
    }

    rows: list[dict[str, Any]] = []
    for sel in data.get("selections", []):
        mid = sel.get("marketId", "")
        mkt = markets_by_id.get(mid, {})
        mkt_name = mkt.get("name", "").lower()

        market_type = _MKT_MAP.get(mkt_name)
        if market_type is None:
            continue

        evt = events_by_id.get(mkt.get("eventId", ""), {})
        odds_str = sel.get("displayOdds", {}).get("american", "")
        points = sel.get("points")
        line_val = float(points) if points not in (None, "") else None

        rows.append({
            "game_description": evt.get("name", ""),
            "market_type": market_type,
            "team_or_label": sel.get("label", ""),
            "outcome_type": sel.get("outcomeType", ""),
            "odds": odds_str,
            "implied_prob": _american_to_implied(odds_str),
            "line": line_val,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "Fetched %d DK game odds across %d games",
            len(df), df["game_description"].nunique(),
        )
    else:
        logger.warning("No DraftKings game odds found")
    return df


# -----------------------------------------------------------------------
# DFS: Classic salaries & Tier assignments
# -----------------------------------------------------------------------
_LOBBY_URL = "https://www.draftkings.com/lobby/getcontests?sport=MLB"
_DRAFTABLES_URL = (
    "https://api.draftkings.com/draftgroups/v1/draftgroups/{dg}/draftables"
)

# DK GameTypeId → label
_GAME_TYPE_CLASSIC = 2
_GAME_TYPE_TIERS = 45

# Tier rosterSlotIds are sequential starting at 278 → Tier 1.
_TIER_SLOT_BASE = 278


def _get_draftgroups() -> list[dict[str, Any]]:
    """Fetch today's MLB draft groups from the DK lobby."""
    import requests as std_requests

    resp = std_requests.get(_LOBBY_URL, timeout=30)
    resp.raise_for_status()
    return resp.json().get("DraftGroups", [])


def _fetch_draftables(draftgroup_id: int) -> list[dict[str, Any]]:
    """Fetch the player pool for a draft group."""
    import requests as std_requests

    url = _DRAFTABLES_URL.format(dg=draftgroup_id)
    resp = std_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("draftables", [])


def _find_main_slate_dg(
    draftgroups: list[dict[str, Any]],
    game_type_id: int,
) -> int | None:
    """Find the main-slate draft group for a given game type.

    The featured/current slate always has the lowest SortOrder (usually 1).
    Stale slates from prior days linger in the lobby with SortOrder=999.
    We sort by SortOrder first, then by game count as tiebreaker.
    """
    candidates = [
        dg for dg in draftgroups if dg.get("GameTypeId") == game_type_id
    ]
    if not candidates:
        return None
    # Primary sort: lowest SortOrder (featured); secondary: most games
    candidates.sort(
        key=lambda d: (d.get("SortOrder", 999), -d.get("GameCount", 0))
    )
    return candidates[0]["DraftGroupId"]


def fetch_dk_salaries(
    draftgroup_id: int | None = None,
) -> pd.DataFrame:
    """Fetch DraftKings classic DFS salaries for today's MLB slate.

    Parameters
    ----------
    draftgroup_id : int, optional
        Specific draft group to fetch.  If *None*, auto-detects the
        main Classic slate (most games).

    Returns
    -------
    pd.DataFrame
        Columns: player_name, player_dk_id, position, team, opponent,
        salary, fppg, game, game_time, status, handedness.
    """
    if draftgroup_id is None:
        dgs = _get_draftgroups()
        draftgroup_id = _find_main_slate_dg(dgs, _GAME_TYPE_CLASSIC)
        if draftgroup_id is None:
            logger.warning("No DK Classic MLB slate found")
            return pd.DataFrame()
        logger.info("Auto-detected Classic main slate: dg=%d", draftgroup_id)

    draftables = _fetch_draftables(draftgroup_id)
    rows: list[dict[str, Any]] = []
    for d in draftables:
        # FPPG is stat attribute id=408
        fppg = None
        for attr in d.get("draftStatAttributes", []):
            if attr.get("id") == 408:
                try:
                    fppg = float(attr["value"])
                except (ValueError, TypeError):
                    pass

        hand = None
        for attr in d.get("playerAttributes", []):
            if attr.get("name") == "Bat-Handedness":
                hand = attr.get("value")

        comp = d.get("competition", {})
        rows.append({
            "player_name": d.get("displayName", ""),
            "player_dk_id": d.get("playerDkId"),
            "position": d.get("position", ""),
            "team": d.get("teamAbbreviation", ""),
            "opponent": _parse_opponent(
                comp.get("name", ""), d.get("teamAbbreviation", "")
            ),
            "salary": d.get("salary"),
            "fppg": fppg,
            "game": comp.get("name", ""),
            "game_time": comp.get("startTime", ""),
            "status": d.get("status", ""),
            "handedness": hand,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "Fetched %d DK salaries (dg=%d), salary range $%s–$%s",
            len(df), draftgroup_id,
            f"{df['salary'].min():,}", f"{df['salary'].max():,}",
        )
    else:
        logger.warning("No DK draftables found for dg=%d", draftgroup_id)
    return df


def fetch_dk_tiers(
    draftgroup_id: int | None = None,
) -> pd.DataFrame:
    """Fetch DraftKings Tiers contest player pool for today's MLB slate.

    Parameters
    ----------
    draftgroup_id : int, optional
        Specific tiers draft group.  If *None*, auto-detects.

    Returns
    -------
    pd.DataFrame
        Columns: player_name, player_dk_id, position, team, opponent,
        tier, fppg, game, game_time, status, handedness.
    """
    if draftgroup_id is None:
        dgs = _get_draftgroups()
        draftgroup_id = _find_main_slate_dg(dgs, _GAME_TYPE_TIERS)
        if draftgroup_id is None:
            logger.warning("No DK Tiers MLB slate found")
            return pd.DataFrame()
        logger.info("Auto-detected Tiers slate: dg=%d", draftgroup_id)

    draftables = _fetch_draftables(draftgroup_id)
    rows: list[dict[str, Any]] = []
    for d in draftables:
        tier = d.get("rosterSlotId", 0) - _TIER_SLOT_BASE + 1
        if tier < 1:
            tier = None

        fppg = None
        for attr in d.get("draftStatAttributes", []):
            if attr.get("id") == 408:
                try:
                    fppg = float(attr["value"])
                except (ValueError, TypeError):
                    pass

        hand = None
        for attr in d.get("playerAttributes", []):
            if attr.get("name") == "Bat-Handedness":
                hand = attr.get("value")

        comp = d.get("competition", {})
        rows.append({
            "player_name": d.get("displayName", ""),
            "player_dk_id": d.get("playerDkId"),
            "position": d.get("position", ""),
            "team": d.get("teamAbbreviation", ""),
            "opponent": _parse_opponent(
                comp.get("name", ""), d.get("teamAbbreviation", "")
            ),
            "tier": tier,
            "fppg": fppg,
            "game": comp.get("name", ""),
            "game_time": comp.get("startTime", ""),
            "status": d.get("status", ""),
            "handedness": hand,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "Fetched %d DK tier players (dg=%d) across %d tiers",
            len(df), draftgroup_id, df["tier"].nunique(),
        )
    else:
        logger.warning("No DK tier draftables found for dg=%d", draftgroup_id)
    return df


def _parse_opponent(game_name: str, team: str) -> str:
    """Extract opponent abbreviation from a game name like 'NYY @ HOU'."""
    parts = game_name.replace(" @ ", "/").replace(" vs ", "/").split("/")
    parts = [p.strip() for p in parts]
    for p in parts:
        if p != team:
            return p
    return ""
