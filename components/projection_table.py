"""Shared rendering for per-game projected stat lines.

Used by the Player Projections page and the Schedule game drill-down so both
show the same numbers in the same shape: a projected average, the range
covering the middle 80% of simulated outcomes, and the actual result once a
game is under way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.html import esc

PITCHER_STATS = ["K", "Outs", "H", "BB", "HR"]
BATTER_STATS = ["H", "TB", "R", "RBI", "BB", "K"]

# The displayed range spans the 10th to 90th percentile of simulated outcomes.
RANGE_LOW_Q = 0.10
RANGE_HIGH_Q = 0.90
P_OVER_COLS = [f"p_over_{k + 0.5:.1f}" for k in range(25)]

# How a cell reads: projection only, projection with a live partial result,
# or the final result against the projection.
MODE_PROJECTION = "projection"
MODE_LIVE = "live"
MODE_FINAL = "final"


def with_outcome_ranges(props: pd.DataFrame) -> pd.DataFrame:
    """Add the 10th to 90th percentile outcome range to each projection row.

    ``p_over_{k+0.5}`` is P(X > k), so the q-quantile is the smallest k with
    P(X > k) <= 1 - q. Rows without distribution columns get no range.

    Outcomes are whole numbers, so the inclusive range usually holds more
    than 80% of the simulated mass. ``range_prob`` records how much, which
    is the coverage the model itself expects.
    """
    frame = props.copy()
    cols = [c for c in P_OVER_COLS if c in frame.columns]
    if not cols or frame.empty:
        frame["range_lo"] = np.nan
        frame["range_hi"] = np.nan
        frame["range_prob"] = np.nan
        return frame
    survival = frame[cols].to_numpy(dtype=float)
    has_dist = ~np.isnan(survival).all(axis=1)
    survival = np.nan_to_num(survival, nan=0.0)

    def first_at_or_below(threshold: float) -> np.ndarray:
        hit = survival <= threshold
        index = hit.argmax(axis=1).astype(float)
        index[~hit.any(axis=1)] = len(cols)
        return index

    lo = first_at_or_below(1 - RANGE_LOW_Q)
    hi = first_at_or_below(1 - RANGE_HIGH_Q)

    # P(lo <= X <= hi) = P(X > lo - 1) - P(X > hi), with P(X > -1) = 1 and
    # P(X > k) = 0 beyond the last tabulated line.
    padded = np.hstack([np.ones((len(frame), 1)), survival, np.zeros((len(frame), 1))])
    rows = np.arange(len(frame))
    range_prob = padded[rows, lo.astype(int)] - padded[rows, hi.astype(int) + 1]

    frame["range_lo"] = np.where(has_dist, lo, np.nan)
    frame["range_hi"] = np.where(has_dist, hi, np.nan)
    frame["range_prob"] = np.where(has_dist, range_prob, np.nan)
    return frame


def _cell(row: pd.Series | None, mode: str) -> str:
    """One stat cell: projection over range, or result over projection."""
    if row is None or pd.isna(row.get("expected")):
        return '<td class="pp-cell"><div class="pp-v pp-muted">-</div></td>'
    expected = float(row["expected"])
    lo, hi = row.get("range_lo"), row.get("range_hi")
    has_range = pd.notna(lo) and pd.notna(hi)
    actual = row.get("actual")

    if mode in (MODE_FINAL, MODE_LIVE) and pd.notna(actual):
        actual = float(actual)
        # A partial result mid-game says nothing about the final range, so
        # only a finished game gets the inside/outside verdict.
        verdict = ""
        if mode == MODE_FINAL and has_range:
            verdict = " pp-in" if lo <= actual <= hi else " pp-out"
        return (
            f'<td class="pp-cell">'
            f'<div class="pp-v{verdict}">{actual:.0f}</div>'
            f'<div class="pp-sub">{expected:.1f}</div>'
            f'</td>'
        )

    range_label = f"{lo:.0f}-{hi:.0f}" if has_range else ""
    return (
        f'<td class="pp-cell">'
        f'<div class="pp-v">{expected:.1f}</div>'
        f'<div class="pp-sub">{range_label}</div>'
        f'</td>'
    )


def _player_rows(rows: pd.DataFrame, stats: list[str], mode: str, order_label: bool) -> str:
    html = ""
    for player_id, player in rows.groupby("player_id", sort=False):
        by_stat = {row["stat"]: row for _, row in player.iterrows()}
        first = player.iloc[0]
        order = first.get("batting_order")
        slot = (
            f'<span class="pp-slot">{int(order)}</span>'
            if order_label and pd.notna(order)
            else ""
        )
        html += (
            f'<tr><td class="pp-name">{slot}{esc(str(first.get("player_name", player_id)))}</td>'
            + "".join(_cell(by_stat.get(stat), mode) for stat in stats)
            + "</tr>"
        )
    return html


def _table(title: str, stats: list[str], body: str) -> str:
    head = "".join(f"<th>{esc(stat)}</th>" for stat in stats)
    return (
        f'<div class="pp-table-wrap"><table class="pp-table">'
        f'<thead><tr><th class="pp-name">{esc(title)}</th>{head}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def team_block(rows: pd.DataFrame, team: str, mode: str, tag: str = "") -> str:
    """Starter and lineup tables for one team in one game."""
    pitchers = rows[rows["player_type"] == "pitcher"]
    batters = rows[rows["player_type"] == "batter"]
    if "batting_order" in batters.columns:
        batters = batters[batters["batting_order"].between(1, 9)].sort_values("batting_order")

    parts = [
        f'<div class="pp-team"><div class="pp-team-head">'
        f'<span class="pp-team-abbr">{esc(team)}</span>{tag}</div>'
    ]
    if not pitchers.empty:
        parts.append(_table("Starter", PITCHER_STATS,
                            _player_rows(pitchers, PITCHER_STATS, mode, order_label=False)))
    if not batters.empty:
        parts.append(_table("Lineup", BATTER_STATS,
                            _player_rows(batters, BATTER_STATS, mode, order_label=True)))
    if pitchers.empty and batters.empty:
        parts.append('<div class="pp-empty-team">No projections for this team.</div>')
    parts.append("</div>")
    return "".join(parts)


def game_block(rows: pd.DataFrame, teams: list[tuple[str, str]], mode: str) -> str:
    """Both teams of one game, side by side. ``teams`` is (abbr, tag) pairs."""
    return (
        '<div class="pp-game">'
        + "".join(team_block(rows[rows["team"] == team], team, mode, tag)
                  for team, tag in teams)
        + "</div>"
    )


def lineup_tag(confirmed: bool | None) -> str:
    if confirmed is None:
        return ""
    if confirmed:
        return '<span class="pp-tag pp-tag-confirmed">Confirmed lineup</span>'
    return '<span class="pp-tag">Projected lineup</span>'
