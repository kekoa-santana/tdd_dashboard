#!/usr/bin/env python
"""Return 0 when current ET is inside today's game window.

Game window:
- starts 90 minutes before the earliest first pitch
- ends 5 hours after the latest first pitch

Reads `data/dashboard/todays_games.parquet`.
Exit codes:
- 0: inside window
- 1: outside window, missing file, or no parseable games
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMES_PATH = PROJECT_ROOT / "data" / "dashboard" / "todays_games.parquet"
ET = ZoneInfo("America/New_York")
TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([AP]M)")


def _to_et_datetime(game_date: object, game_time: object) -> datetime | None:
    match = TIME_RE.search(str(game_time).upper())
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    am_pm = match.group(3)

    if am_pm == "AM" and hour == 12:
        hour = 0
    elif am_pm == "PM" and hour < 12:
        hour += 12

    return datetime.fromisoformat(f"{game_date}T{hour:02d}:{minute:02d}:00").replace(
        tzinfo=ET,
    )


def main() -> int:
    if not GAMES_PATH.exists():
        return 1

    games = pd.read_parquet(GAMES_PATH)
    if games.empty or "game_date" not in games.columns or "game_time" not in games.columns:
        return 1

    starts = [
        dt
        for dt in (
            _to_et_datetime(gd, gt)
            for gd, gt in zip(games["game_date"], games["game_time"])
        )
        if dt is not None
    ]
    if not starts:
        return 1

    now_et = datetime.now(ET)
    window_start = min(starts) - timedelta(minutes=90)
    window_end = max(starts) + timedelta(hours=5)

    print(
        f"now_et={now_et.isoformat()} "
        f"window_start={window_start.isoformat()} "
        f"window_end={window_end.isoformat()}",
    )
    return 0 if window_start <= now_et <= window_end else 1


if __name__ == "__main__":
    raise SystemExit(main())
