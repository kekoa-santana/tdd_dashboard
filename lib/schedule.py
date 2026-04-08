"""
Schedule and lineup data for in-season dashboard.

Fetches today's games, probable pitchers, and lineups from the MLB Stats API.
Falls back gracefully when data isn't available.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Abbreviation lookup: MLB team ID -> standard abbreviation
_TEAM_ABBR: dict[int, str] = {}


def fetch_recent_transactions(game_date: str) -> pd.DataFrame:
    """Fetch MLB transactions for a given date.

    Parameters
    ----------
    game_date : str
        Date as 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Columns: transaction_id, player_id, player_name, team_abbr,
        type_desc, description.  Empty DataFrame on failure.
    """
    import urllib.request

    url = (
        f"{MLB_API_BASE}/transactions"
        f"?date={game_date}&sportId=1"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch transactions for %s: %s", game_date, e)
        return pd.DataFrame()

    rows = []
    for txn in data.get("transactions", []):
        txn_id = txn.get("id")
        desc = txn.get("description", "")
        type_desc = txn.get("typeDesc", "")

        player = txn.get("person", {})
        team = txn.get("toTeam") or txn.get("fromTeam") or {}

        rows.append({
            "transaction_id": txn_id,
            "player_id": player.get("id"),
            "player_name": player.get("fullName", ""),
            "team_abbr": team.get("abbreviation", ""),
            "type_desc": type_desc,
            "description": desc,
        })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d transactions for %s", len(df), game_date)
    return df


def _fetch_people_bat_hand(player_ids: list[int]) -> dict[int, str]:
    """Map MLB player ID -> plate-appearance bat side (L, R, or S).

    Boxscore ``person`` blobs often omit ``batSide``; the people endpoint is
    reliable. Returns empty dict on failure.
    """
    import urllib.request

    ids = sorted({int(i) for i in player_ids if i is not None})
    if not ids:
        return {}

    url = f"{MLB_API_BASE}/people?personIds={','.join(str(i) for i in ids)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch bat handedness for %d players: %s", len(ids), e)
        return {}

    out: dict[int, str] = {}
    for p in data.get("people", []):
        pid = p.get("id")
        if pid is None:
            continue
        bs = p.get("batSide") or {}
        code = (bs.get("code") or "").strip().upper()
        if code in ("L", "R", "S"):
            out[int(pid)] = code
    return out


def _ensure_team_abbr() -> None:
    """Populate ``_TEAM_ABBR`` from the MLB API (once)."""
    import urllib.request

    if _TEAM_ABBR:
        return
    url = f"{MLB_API_BASE}/teams?sportId=1"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        for t in data.get("teams", []):
            _TEAM_ABBR[t["id"]] = t.get("abbreviation", t.get("teamName", ""))
    except Exception as e:
        logger.warning("Failed to fetch team abbreviations: %s", e)


def fetch_standings(season: int | None = None) -> dict[str, tuple[int, int]]:
    """Fetch current W-L records from the MLB Stats API.

    Parameters
    ----------
    season : int | None
        Season year. Defaults to the current year.

    Returns
    -------
    dict[str, tuple[int, int]]
        ``{team_abbr: (wins, losses)}`` for every MLB team.
    """
    import urllib.request

    if season is None:
        season = date.today().year

    url = (
        f"{MLB_API_BASE}/standings"
        f"?leagueId=103,104&season={season}"
        f"&standingsTypes=regularSeason"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch standings: %s", e)
        return {}

    _ensure_team_abbr()

    standings: dict[str, tuple[int, int]] = {}
    for record in data.get("records", []):
        for entry in record.get("teamRecords", []):
            team_id = entry.get("team", {}).get("id")
            wins = entry.get("wins", 0)
            losses = entry.get("losses", 0)
            abbr = _TEAM_ABBR.get(team_id, entry.get("team", {}).get("name", ""))
            standings[abbr] = (wins, losses)

    return standings


def fetch_todays_schedule(
    game_date: str | None = None,
) -> pd.DataFrame:
    """Fetch today's MLB schedule from the Stats API.

    Parameters
    ----------
    game_date : str | None
        Date as 'YYYY-MM-DD'. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, game_date, game_time, status,
        away_team_id, away_team_name, away_abbr,
        home_team_id, home_team_name, home_abbr,
        away_pitcher_id, away_pitcher_name,
        home_pitcher_id, home_pitcher_name,
        hp_umpire_name, venue_id.
    """
    import urllib.request

    if game_date is None:
        game_date = date.today().isoformat()

    url = (
        f"{MLB_API_BASE}/schedule"
        f"?date={game_date}&sportId=1"
        f"&hydrate=probablePitcher,team,officials"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch schedule from MLB API: %s", e)
        return pd.DataFrame()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            gpk = game.get("gamePk")
            status = game.get("status", {}).get("detailedState", "")
            game_dt = game.get("gameDate", "")

            # Parse game time
            game_time = ""
            if game_dt:
                try:
                    dt = datetime.fromisoformat(game_dt.replace("Z", "+00:00"))
                    # Convert UTC to ET (UTC-4 during DST, UTC-5 otherwise)
                    from datetime import timedelta
                    et_dt = dt - timedelta(hours=4)
                    hour = et_dt.hour
                    minute = et_dt.minute
                    ampm = "AM" if hour < 12 else "PM"
                    display_hour = hour % 12 or 12
                    game_time = f"{display_hour}:{minute:02d} {ampm} ET"
                except Exception:
                    game_time = game_dt

            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})

            away_team = away.get("team", {})
            home_team = home.get("team", {})

            # Probable pitchers
            away_pp = away.get("probablePitcher", {})
            home_pp = home.get("probablePitcher", {})

            # Home plate umpire (available once crew is assigned, usually day-of)
            hp_umpire_name = ""
            for official in game.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    hp_umpire_name = official.get("official", {}).get("fullName", "")
                    break

            rows.append({
                "game_pk": gpk,
                "game_date": game_date,
                "game_time": game_time,
                "status": status,
                "away_team_id": away_team.get("id"),
                "away_team_name": away_team.get("name", ""),
                "away_abbr": away_team.get("abbreviation", ""),
                "home_team_id": home_team.get("id"),
                "home_team_name": home_team.get("name", ""),
                "home_abbr": home_team.get("abbreviation", ""),
                "away_pitcher_id": away_pp.get("id"),
                "away_pitcher_name": away_pp.get("fullName", ""),
                "home_pitcher_id": home_pp.get("id"),
                "home_pitcher_name": home_pp.get("fullName", ""),
                "hp_umpire_name": hp_umpire_name,
                "venue_id": game.get("venue", {}).get("id"),
            })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d games for %s", len(df), game_date)
    return df


def fetch_game_lineups(
    game_pk: int,
) -> pd.DataFrame:
    """Fetch lineup for a specific game from the Stats API.

    Parameters
    ----------
    game_pk : int
        MLB game primary key.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, team_id, team_abbr, batting_order,
        batter_id, batter_name, bat_hand (L / R / S when available).
    """
    import urllib.request

    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch lineup for game %d: %s", game_pk, e)
        return pd.DataFrame()

    rows = []
    for side in ("away", "home"):
        team_data = data.get("teams", {}).get(side, {})
        team_info = team_data.get("team", {})
        team_id = team_info.get("id")
        team_abbr = team_info.get("abbreviation", "")

        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        for order, pid in enumerate(batting_order[:9], 1):
            pid_key = f"ID{pid}"
            player_data = players.get(pid_key, {})
            player_info = player_data.get("person", {})
            game_pos = player_data.get("position", {}).get("abbreviation", "")
            rows.append({
                "game_pk": game_pk,
                "team_id": team_id,
                "team_abbr": team_abbr,
                "batting_order": order,
                "batter_id": pid,
                "batter_name": player_info.get("fullName", "Unknown"),
                "game_position": game_pos,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    hand_map = _fetch_people_bat_hand(df["batter_id"].dropna().astype(int).tolist())
    df["bat_hand"] = df["batter_id"].map(
        lambda x: hand_map.get(int(x), "") if pd.notna(x) else "",
    )
    return df


def fetch_all_lineups(
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch lineups for all games in a schedule.

    Only fetches for games that have started or have lineups posted.

    Parameters
    ----------
    schedule_df : pd.DataFrame
        Schedule with game_pk column.

    Returns
    -------
    pd.DataFrame
        Combined lineup data for all games.
    """
    if schedule_df.empty:
        return pd.DataFrame()

    frames = []
    for gpk in schedule_df["game_pk"].unique():
        lu = fetch_game_lineups(int(gpk))
        if not lu.empty:
            frames.append(lu)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
