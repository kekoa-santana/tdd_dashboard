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
# One row per (source, game) for the latest fetch — moneyline, run line, main total
DAILY_WIDE_PATH = DASHBOARD_DIR / "game_odds_daily.parquet"

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


def _pick_primary_total_line(tot: pd.DataFrame, source: str) -> float | None:
    """Choose a single total line when the book posts several (e.g. 7.5, 8, 8.5)."""
    lines = tot["line"].dropna().unique()
    if len(lines) == 0:
        return None
    candidates: list[tuple[float, int, float]] = []
    for line in lines:
        sub = tot[tot["line"] == line]
        if source == "dk":
            has_o = (sub["outcome_type"].astype(str) == "Over").any()
            has_u = (sub["outcome_type"].astype(str) == "Under").any()
        else:
            lab = sub["team_or_label"].astype(str).str.lower()
            has_o = lab.str.contains("over", na=False).any()
            has_u = lab.str.contains("under", na=False).any()
        if has_o and has_u:
            candidates.append((float(line), len(sub), -abs(float(line) - 8.5)))
    if not candidates:
        return None
    # Prefer complete pairs; tie-break toward line count then closeness to 8.5
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]


def build_wide_game_odds(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot one snapshot of long-format odds to one row per source × game."""
    if long_df.empty:
        return pd.DataFrame()

    rows_out: list[dict[str, object]] = []
    group_cols = ["game_date", "source", "game_description", "snapshot_ts"]
    for key, grp in long_df.groupby(group_cols, sort=False):
        gd, src, gdesc, snap = key
        parts = str(gdesc).split(" @ ")
        away_n, home_n = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", "")

        row: dict[str, object] = {
            "snapshot_ts": snap,
            "game_date": gd,
            "source": src,
            "game_description": gdesc,
        }

        ml = grp[grp["market_type"] == "moneyline"]
        if src == "dk":
            a = ml[ml["outcome_type"].astype(str) == "Away"]
            h = ml[ml["outcome_type"].astype(str) == "Home"]
            if not a.empty:
                row["ml_away_odds"] = a.iloc[0]["odds"]
                row["ml_away_implied"] = float(a.iloc[0]["implied_prob"])
            if not h.empty:
                row["ml_home_odds"] = h.iloc[0]["odds"]
                row["ml_home_implied"] = float(h.iloc[0]["implied_prob"])
        else:
            for _, r in ml.iterrows():
                lab = str(r["team_or_label"]).strip()
                if lab == away_n:
                    row["ml_away_odds"] = r["odds"]
                    row["ml_away_implied"] = float(r["implied_prob"])
                elif lab == home_n:
                    row["ml_home_odds"] = r["odds"]
                    row["ml_home_implied"] = float(r["implied_prob"])

        sp = grp[grp["market_type"] == "spread"]
        if src == "dk":
            a = sp[sp["outcome_type"].astype(str) == "Away"]
            h = sp[sp["outcome_type"].astype(str) == "Home"]
            if not a.empty:
                row["spread_away_line"] = a.iloc[0]["line"]
                row["spread_away_odds"] = a.iloc[0]["odds"]
                row["spread_away_implied"] = float(a.iloc[0]["implied_prob"])
            if not h.empty:
                row["spread_home_line"] = h.iloc[0]["line"]
                row["spread_home_odds"] = h.iloc[0]["odds"]
                row["spread_home_implied"] = float(h.iloc[0]["implied_prob"])
        else:
            for _, r in sp.iterrows():
                lab = str(r["team_or_label"]).strip()
                if lab == away_n:
                    row["spread_away_line"] = r["line"]
                    row["spread_away_odds"] = r["odds"]
                    row["spread_away_implied"] = float(r["implied_prob"])
                elif lab == home_n:
                    row["spread_home_line"] = r["line"]
                    row["spread_home_odds"] = r["odds"]
                    row["spread_home_implied"] = float(r["implied_prob"])

        tot = grp[grp["market_type"] == "total"]
        if not tot.empty:
            best_line = _pick_primary_total_line(tot, src)
            if best_line is not None:
                tsub = tot[tot["line"] == best_line]
                row["total_line"] = best_line
                if src == "dk":
                    ovr = tsub[tsub["outcome_type"].astype(str) == "Over"]
                    und = tsub[tsub["outcome_type"].astype(str) == "Under"]
                else:
                    lab = tsub["team_or_label"].astype(str).str.lower()
                    ovr = tsub[lab.str.contains("over", na=False)]
                    und = tsub[lab.str.contains("under", na=False)]
                if not ovr.empty:
                    row["total_over_odds"] = ovr.iloc[0]["odds"]
                    row["total_over_implied"] = float(ovr.iloc[0]["implied_prob"])
                if not und.empty:
                    row["total_under_odds"] = und.iloc[0]["odds"]
                    row["total_under_implied"] = float(und.iloc[0]["implied_prob"])

        rows_out.append(row)

    return pd.DataFrame(rows_out)


def write_game_odds_daily_parquet(long_df: pd.DataFrame) -> None:
    """Overwrite ``game_odds_daily.parquet`` with wide ML / spread / total for this fetch."""
    if long_df.empty:
        return
    wide = build_wide_game_odds(long_df)
    if wide.empty:
        logger.warning("Wide game odds empty — skipping game_odds_daily.parquet")
        return
    DAILY_WIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(DAILY_WIDE_PATH, index=False)
    logger.info("Wrote %d rows to game_odds_daily.parquet", len(wide))


def persist_game_odds(game_date: str) -> pd.DataFrame:
    """Fetch, append to history, and refresh the daily wide snapshot."""
    odds = collect_odds(game_date)
    if odds.empty:
        logger.info("No game odds to persist")
        return odds
    append_to_history(odds)
    write_game_odds_daily_parquet(odds)
    return odds


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

    odds = persist_game_odds(game_date)
    if odds.empty:
        logger.info("Nothing to store")


if __name__ == "__main__":
    main()
