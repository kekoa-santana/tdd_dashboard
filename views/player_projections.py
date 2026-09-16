"""Player Projections -- per-game projected stat lines, and how they turned out."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from components.attribution import build_attribution_panel
from components.projection_table import (
    MODE_FINAL,
    MODE_LIVE,
    MODE_PROJECTION,
    game_block,
    lineup_tag,
    with_outcome_ranges,
)
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

# Stats summarized in the recent-accuracy strip: (player_type, stat, label).
_ACCURACY_STATS = [
    ("pitcher", "K", "Pitcher K"),
    ("pitcher", "Outs", "Pitcher Outs"),
    ("batter", "H", "Batter H"),
    ("batter", "TB", "Batter TB"),
]
_ACCURACY_DAYS = 7

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

@st.cache_data(ttl=300)
def _day_projections(game_date: str) -> pd.DataFrame:
    """Projection rows for one date, with outcome ranges."""
    props = load_game_props()
    if props.empty or "game_date" not in props.columns:
        return pd.DataFrame()
    day = props[props["game_date"] == game_date]
    if day.empty:
        return pd.DataFrame()
    return with_outcome_ranges(day)


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
    window = with_outcome_ranges(window)
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
        mode = {
            "final": MODE_FINAL,
            "in_progress": MODE_LIVE,
        }.get(game["status"], MODE_PROJECTION)
        label = f'{game["away"]} {game["separator"]} {game["home"]}'
        details = [part for part in (game["time"], _status_label(game["status"])) if part]
        if details:
            label += "  ·  " + "  ·  ".join(details)
        with st.expander(label, expanded=team_filter != "All teams" or len(games) == 1):
            rows = day[day["game_pk"] == game["game_pk"]]
            teams = [
                (team, lineup_tag(_lineup_confirmed(game["game_pk"], team))
                       if mode == MODE_PROJECTION else "")
                for team in (game["away"], game["home"])
            ]
            st.markdown(game_block(rows, teams, mode), unsafe_allow_html=True)

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
