"""
Schedule and lineup data for in-season dashboard.

Fetches today's games, probable pitchers, and lineups from the MLB Stats API.
Falls back gracefully when data isn't available.

Synced from: player_profiles/src/data/schedule.py
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_standings(season: int | None = None) -> dict[str, tuple[int, int]]:
    """Fetch current W-L records from the MLB Stats API.

    Returns a dict mapping team abbreviation to (wins, losses).
    """
    import urllib.request

    if season is None:
        season = date.today().year
    url = (
        f"{MLB_API_BASE}/standings"
        f"?leagueId=103,104&season={season}&standingsTypes=regularSeason"
        f"&hydrate=team"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch standings: %s", e)
        return {}

    records: dict[str, tuple[int, int]] = {}
    for division in data.get("records", []):
        for team in division.get("teamRecords", []):
            abbr = team.get("team", {}).get("abbreviation", "")
            if abbr:
                records[abbr] = (
                    int(team.get("wins", 0)),
                    int(team.get("losses", 0)),
                )
    return records


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
        venue_id, venue_name, hp_umpire_name,
        weather_temp, weather_wind, weather_condition.
    """
    import urllib.request

    if game_date is None:
        game_date = date.today().isoformat()

    url = (
        f"{MLB_API_BASE}/schedule"
        f"?date={game_date}&sportId=1"
        f"&hydrate=probablePitcher,team,venue,weather,officials"
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

            # Venue
            venue = game.get("venue", {})

            # Weather
            weather = game.get("weather", {})

            # HP umpire
            hp_ump = ""
            for official in game.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    hp_ump = official.get("official", {}).get("fullName", "")
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
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name", ""),
                "hp_umpire_name": hp_ump,
                "weather_temp": weather.get("temp", ""),
                "weather_wind": weather.get("wind", ""),
                "weather_condition": weather.get("condition", ""),
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
        batter_id, batter_name.
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
            player_info = players.get(pid_key, {}).get("person", {})
            rows.append({
                "game_pk": game_pk,
                "team_id": team_id,
                "team_abbr": team_abbr,
                "batting_order": order,
                "batter_id": pid,
                "batter_name": player_info.get("fullName", "Unknown"),
            })

    return pd.DataFrame(rows)


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


def fetch_live_boxscores(
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch live player stats for all in-progress or final games.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, player_id, player_name, player_type,
        team_abbr, game_status, K, H, HR, BB, TB (batters),
        K, IP, H, BB, HR, Outs (pitchers).
    """
    import urllib.request

    if schedule_df.empty:
        return pd.DataFrame()

    # Only fetch for games that are in progress or final
    active_statuses = {
        "In Progress", "Final", "Game Over", "Mid Inning",
        "End Inning", "Top", "Bottom", "Middle",
    }
    active = schedule_df[
        schedule_df["status"].str.contains(
            "|".join(active_statuses), case=False, na=False,
        )
    ] if "status" in schedule_df.columns else schedule_df

    if active.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for gpk in active["game_pk"].unique():
        url = f"{MLB_API_BASE}/game/{int(gpk)}/boxscore"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning("Failed to fetch boxscore for game %d: %s", gpk, e)
            continue

        # Determine game status from parent schedule row
        game_row = active[active["game_pk"] == gpk].iloc[0]
        game_status = game_row.get("status", "")

        for side in ("away", "home"):
            team_data = data.get("teams", {}).get(side, {})
            team_abbr = team_data.get("team", {}).get("abbreviation", "")
            players = team_data.get("players", {})

            for pid_key, pdata in players.items():
                person = pdata.get("person", {})
                pid = person.get("id")
                name = person.get("fullName", "")
                stats = pdata.get("stats", {})

                batting = stats.get("batting", {})
                pitching = stats.get("pitching", {})

                if batting and batting.get("atBats", 0) + batting.get("baseOnBalls", 0) > 0:
                    rows.append({
                        "game_pk": gpk,
                        "player_id": pid,
                        "player_name": name,
                        "player_type": "batter",
                        "team_abbr": team_abbr,
                        "game_status": game_status,
                        "actual_K": batting.get("strikeOuts", 0),
                        "actual_H": batting.get("hits", 0),
                        "actual_HR": batting.get("homeRuns", 0),
                        "actual_BB": batting.get("baseOnBalls", 0),
                        "actual_TB": batting.get("totalBases", 0),
                    })

                if pitching and pitching.get("inningsPitched"):
                    ip_str = pitching.get("inningsPitched", "0")
                    try:
                        ip_val = float(ip_str)
                    except (ValueError, TypeError):
                        ip_val = 0.0
                    outs = int(ip_val) * 3 + round((ip_val % 1) * 10)
                    rows.append({
                        "game_pk": gpk,
                        "player_id": pid,
                        "player_name": name,
                        "player_type": "pitcher",
                        "team_abbr": team_abbr,
                        "game_status": game_status,
                        "actual_K": pitching.get("strikeOuts", 0),
                        "actual_H": pitching.get("hits", 0),
                        "actual_HR": pitching.get("homeRuns", 0),
                        "actual_BB": pitching.get("baseOnBalls", 0),
                        "actual_TB": 0,
                        "actual_IP": ip_val,
                        "actual_Outs": outs,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(
        "Fetched live boxscores: %d player lines across %d games",
        len(df), df["game_pk"].nunique(),
    )
    return df
