"""Player Projections -- per-game projected stat lines, and how they turned out."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from components.attribution import build_attribution_panel
from services.data_loader import (
    load_game_props,
    load_prop_attribution,
    load_todays_games,
    load_todays_lineups,
)
from utils.html import esc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PITCHER_STATS = ["K", "Outs", "H", "BB", "HR"]
_BATTER_STATS = ["H", "TB", "R", "RBI", "BB", "K"]

# Stats summarized in the recent-accuracy strip: (player_type, stat, label).
_ACCURACY_STATS = [
    ("pitcher", "K", "Pitcher K"),
    ("pitcher", "Outs", "Pitcher Outs"),
    ("batter", "H", "Batter H"),
    ("batter", "TB", "Batter TB"),
]
_ACCURACY_DAYS = 7

# The displayed range spans the 10th to 90th percentile of simulated outcomes.
_RANGE_LOW_Q = 0.10
_RANGE_HIGH_Q = 0.90
_P_OVER_COLS = [f"p_over_{k + 0.5:.1f}" for k in range(25)]


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def _today_et() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=4)


def _iso(day: datetime) -> str:
    return day.date().isoformat()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _with_ranges(props: pd.DataFrame) -> pd.DataFrame:
    """Add the 10th-90th percentile outcome range to each projection row.

    ``p_over_{k+0.5}`` is P(X > k), so the q-quantile is the smallest k with
    P(X > k) <= 1 - q. Rows without distribution columns get no range.

    Outcomes are whole numbers, so the inclusive range usually holds more
    than 80% of the simulated mass. ``range_prob`` records how much, which
    is the coverage the model itself expects.
    """
    frame = props.copy()
    cols = [c for c in _P_OVER_COLS if c in frame.columns]
    if not cols:
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

    lo = first_at_or_below(1 - _RANGE_LOW_Q)
    hi = first_at_or_below(1 - _RANGE_HIGH_Q)

    # P(lo <= X <= hi) = P(X > lo - 1) - P(X > hi), with P(X > -1) = 1 and
    # P(X > k) = 0 beyond the last tabulated line.
    padded = np.hstack([np.ones((len(frame), 1)), survival, np.zeros((len(frame), 1))])
    rows = np.arange(len(frame))
    range_prob = padded[rows, lo.astype(int)] - padded[rows, hi.astype(int) + 1]

    frame["range_lo"] = np.where(has_dist, lo, np.nan)
    frame["range_hi"] = np.where(has_dist, hi, np.nan)
    frame["range_prob"] = np.where(has_dist, range_prob, np.nan)
    return frame


@st.cache_data(ttl=300)
def _day_projections(game_date: str) -> pd.DataFrame:
    """Projection rows for one date, with outcome ranges."""
    props = load_game_props()
    if props.empty or "game_date" not in props.columns:
        return pd.DataFrame()
    day = props[props["game_date"] == game_date]
    if day.empty:
        return pd.DataFrame()
    return _with_ranges(day)


@st.cache_data(ttl=300)
def _recent_accuracy(end_date: str) -> list[dict]:
    """Error and range coverage over finished games before ``end_date``."""
    props = load_game_props()
    if props.empty or "actual" not in props.columns:
        return []
    end = pd.Timestamp(end_date)
    start = (end - pd.Timedelta(days=_ACCURACY_DAYS)).date().isoformat()
    window = props[
        (props["game_date"] >= start)
        & (props["game_date"] < end_date)
        & (props["game_status"] == "final")
        & props["actual"].notna()
    ]
    if window.empty:
        return []
    window = _with_ranges(window)
    summary: list[dict] = []
    for player_type, stat, label in _ACCURACY_STATS:
        rows = window[(window["player_type"] == player_type) & (window["stat"] == stat)]
        if rows.empty:
            continue
        actual = rows["actual"].astype(float)
        error = (rows["expected"].astype(float) - actual).abs()
        ranged = rows["range_lo"].notna()
        inside = (actual >= rows["range_lo"]) & (actual <= rows["range_hi"])
        summary.append({
            "label": label,
            "mae": float(error.mean()),
            "coverage": float(inside[ranged].mean()) if ranged.any() else None,
            "expected_coverage": float(rows.loc[ranged, "range_prob"].mean()) if ranged.any() else None,
            "n": int(len(rows)),
        })
    return summary


def _game_meta(day: pd.DataFrame, game_date: str) -> list[dict]:
    """One entry per game: teams, time, status, and sort order."""
    schedule = load_todays_games()
    by_pk: dict[int, pd.Series] = {}
    if not schedule.empty and "game_date" in schedule.columns:
        for _, game in schedule[schedule["game_date"] == game_date].iterrows():
            by_pk[int(game["game_pk"])] = game

    games: list[dict] = []
    for game_pk, rows in day.groupby("game_pk"):
        game_pk = int(game_pk)
        status = _game_status(rows)
        scheduled = by_pk.get(game_pk)
        if scheduled is not None:
            away, home = scheduled.get("away_abbr", ""), scheduled.get("home_abbr", "")
            time_label = str(scheduled.get("game_time", "") or "")
            separator = "@"
        else:
            # Past slates have no schedule row, so home and away are unknown.
            teams = sorted(rows["team"].dropna().unique().tolist())
            away, home = (teams + ["", ""])[:2]
            time_label = ""
            separator = "vs"
        games.append({
            "game_pk": game_pk,
            "away": away,
            "home": home,
            "separator": separator,
            "time": time_label,
            "status": status,
            "sort": _time_sort_key(time_label),
        })
    return sorted(games, key=lambda g: (g["sort"], g["away"]))


def _game_status(rows: pd.DataFrame) -> str:
    statuses = set(rows.get("game_status", pd.Series(dtype=object)).dropna())
    if "in_progress" in statuses:
        return "in_progress"
    if statuses == {"final"}:
        return "final"
    return "scheduled"


def _time_sort_key(label: str) -> int:
    try:
        parsed = datetime.strptime(label.replace(" ET", "").strip(), "%I:%M %p")
        return parsed.hour * 60 + parsed.minute
    except ValueError:
        return 24 * 60


def _lineup_confirmed(game_pk: int, team: str) -> bool | None:
    """Whether MLB has posted this team's lineup; None when unknown."""
    lineups = load_todays_lineups()
    if lineups.empty or "lineup_source" not in lineups.columns:
        return None
    rows = lineups[(lineups["game_pk"] == game_pk) & (lineups["team_abbr"] == team)]
    if rows.empty:
        return None
    return bool((rows["lineup_source"] == "api").any())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _number(value: float) -> str:
    return f"{value:.1f}"


def _cell(row: pd.Series | None, final: bool) -> str:
    """One stat cell: projection over range, or actual over projection."""
    if row is None or pd.isna(row.get("expected")):
        return '<td class="pp-cell"><div class="pp-v pp-muted">-</div></td>'
    expected = float(row["expected"])
    lo, hi = row.get("range_lo"), row.get("range_hi")
    has_range = pd.notna(lo) and pd.notna(hi)
    actual = row.get("actual")
    if final and pd.notna(actual):
        actual = float(actual)
        verdict = ""
        if has_range:
            verdict = " pp-in" if lo <= actual <= hi else " pp-out"
        return (
            f'<td class="pp-cell">'
            f'<div class="pp-v{verdict}">{actual:.0f}</div>'
            f'<div class="pp-sub">{_number(expected)}</div>'
            f'</td>'
        )
    range_label = f"{lo:.0f}-{hi:.0f}" if has_range else ""
    return (
        f'<td class="pp-cell">'
        f'<div class="pp-v">{_number(expected)}</div>'
        f'<div class="pp-sub">{range_label}</div>'
        f'</td>'
    )


def _player_rows(rows: pd.DataFrame, stats: list[str], final: bool, order_label: bool) -> str:
    html = ""
    for player_id, player in rows.groupby("player_id", sort=False):
        by_stat = {row["stat"]: row for _, row in player.iterrows()}
        first = player.iloc[0]
        order = first.get("batting_order")
        slot = f'<span class="pp-slot">{int(order)}</span>' if order_label and pd.notna(order) else ""
        html += (
            f'<tr><td class="pp-name">{slot}{esc(str(first.get("player_name", player_id)))}</td>'
            + "".join(_cell(by_stat.get(stat), final) for stat in stats)
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


def _team_block(day: pd.DataFrame, game: dict, team: str, final: bool) -> str:
    rows = day[(day["game_pk"] == game["game_pk"]) & (day["team"] == team)]
    pitchers = rows[rows["player_type"] == "pitcher"]
    batters = rows[(rows["player_type"] == "batter") & (rows["batting_order"].between(1, 9))]
    batters = batters.sort_values("batting_order")

    tag = ""
    if not final:
        confirmed = _lineup_confirmed(game["game_pk"], team)
        if confirmed is not None:
            tag = (
                '<span class="pp-tag pp-tag-confirmed">Confirmed lineup</span>'
                if confirmed
                else '<span class="pp-tag">Projected lineup</span>'
            )

    parts = [f'<div class="pp-team"><div class="pp-team-head">'
             f'<span class="pp-team-abbr">{esc(team)}</span>{tag}</div>']
    if not pitchers.empty:
        parts.append(_table("Starter", _PITCHER_STATS,
                            _player_rows(pitchers, _PITCHER_STATS, final, order_label=False)))
    if not batters.empty:
        parts.append(_table("Lineup", _BATTER_STATS,
                            _player_rows(batters, _BATTER_STATS, final, order_label=True)))
    if pitchers.empty and batters.empty:
        parts.append('<div class="pp-empty-team">No projections for this team.</div>')
    parts.append("</div>")
    return "".join(parts)


def _accuracy_strip(summary: list[dict]) -> str:
    if not summary:
        return ""
    chips = ""
    for item in summary:
        coverage = ""
        if item["coverage"] is not None:
            coverage = f'<span class="pp-acc-cov">{item["coverage"] * 100:.0f}% landed in range</span>'
            if item["expected_coverage"] is not None:
                coverage += f'<span>model expected {item["expected_coverage"] * 100:.0f}%</span>'

        chips += (
            f'<div class="pp-acc-chip">'
            f'<div class="pp-acc-label">{esc(item["label"])}</div>'
            f'<div class="pp-acc-main">Off by {item["mae"]:.2f} on average</div>'
            f'<div class="pp-acc-meta">{coverage}<span>n={item["n"]:,}</span></div>'
            f'</div>'
        )
    return (
        f'<div class="pp-acc">'
        f'<div class="pp-acc-head">How the last {_ACCURACY_DAYS} days of projections landed</div>'
        f'<div class="pp-acc-grid">{chips}</div>'
        f'<div class="pp-acc-note">Stats are whole numbers, so each range holds somewhat more than '
        f'80% of simulated outcomes. A well calibrated model lands about as often as it expects.</div>'
        f'</div>'
    )


def _status_label(status: str) -> str:
    return {"final": "Final", "in_progress": "Live"}.get(status, "")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_player_projections() -> None:
    """Render the Player Projections page."""
    st.markdown(
        '<div class="pp-page-head">'
        '<div class="pp-eyebrow">The Data Diamond</div>'
        '<h1 class="pp-title">Player Projections</h1>'
        '<p class="pp-sub">Projected stat lines for every starter and lineup, '
        'from the same game simulations behind the rest of the site. Each cell shows '
        'the projected average with the range covering the middle 80% of simulated '
        'outcomes. For finished games the actual result is shown with the projection beneath it: '
        '<span class="pp-in">green</span> when it landed inside the range, '
        '<span class="pp-out">orange</span> when it fell outside.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    today = _today_et()
    days = {
        "Yesterday": _iso(today - timedelta(days=1)),
        "Today": _iso(today),
        "Tomorrow": _iso(today + timedelta(days=1)),
    }
    choice = st.segmented_control(
        "Day", list(days), default="Today", key="pp_day", label_visibility="collapsed",
    ) or "Today"
    game_date = days[choice]

    st.markdown(_accuracy_strip(_recent_accuracy(days["Today"])), unsafe_allow_html=True)

    day = _day_projections(game_date)
    if day.empty:
        st.markdown(
            f'<div class="pp-empty">No projections for {esc(choice.lower())} ({esc(game_date)}).</div>',
            unsafe_allow_html=True,
        )
        return

    games = _game_meta(day, game_date)
    teams = sorted({t for g in games for t in (g["away"], g["home"]) if t})
    team_filter = st.selectbox(
        "Team", ["All teams"] + teams, key="pp_team", label_visibility="collapsed",
    )
    if team_filter != "All teams":
        games = [g for g in games if team_filter in (g["away"], g["home"])]

    for game in games:
        final = game["status"] == "final"
        label = f'{game["away"]} {game["separator"]} {game["home"]}'
        details = [part for part in (game["time"], _status_label(game["status"])) if part]
        if details:
            label += "  ·  " + "  ·  ".join(details)
        with st.expander(label, expanded=team_filter != "All teams" or len(games) == 1):
            st.markdown(
                '<div class="pp-game">'
                + _team_block(day, game, game["away"], final)
                + _team_block(day, game, game["home"], final)
                + "</div>",
                unsafe_allow_html=True,
            )

    _render_k_explainer(day)


def _render_k_explainer(day: pd.DataFrame) -> None:
    """Driver breakdown for a starter's strikeout projection."""
    attribution = load_prop_attribution()
    if attribution.empty:
        return
    starters = day[(day["player_type"] == "pitcher") & (day["stat"] == "K")]
    attribution = attribution[
        (attribution["stat"] == "K")
        & (attribution["player_type"] == "pitcher")
        & attribution["game_pk"].isin(starters["game_pk"])
    ]
    if attribution.empty:
        return
    names = dict(zip(starters["player_id"], starters["player_name"]))
    options = {
        f'{names.get(pid, pid)}': pid
        for pid in attribution["player_id"].unique()
        if pid in names
    }
    if not options:
        return
    st.markdown('<div class="pp-explain-head">What drives a strikeout projection</div>',
                unsafe_allow_html=True)
    placeholder = "Select a starter..."
    selected = st.selectbox(
        "Explain a strikeout projection", [placeholder] + sorted(options),
        key="pp_explain", label_visibility="collapsed",
    )
    if selected and selected != placeholder:
        row = attribution[attribution["player_id"] == options[selected]].iloc[0]
        st.markdown(build_attribution_panel(row, name=selected), unsafe_allow_html=True)
