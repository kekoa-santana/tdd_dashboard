"""Stats page — Traditional stat leaderboards for any season."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER,
    AVAILABLE_SEASONS, UNRELIABLE_BB_SEASONS, PRIOR_SEASON,
)
from services.data_loader import (
    load_player_teams, load_traditional_stats, load_traditional_stats_all,
)
from components.headshot import headshot_html


# ── Leaderboard definitions ─────────────────────────────────────────
# (title, column, higher_is_better, format, role_filter)

_SP = lambda df: df[df["starts"] >= 3]
_RP = lambda df: df[df["starts"] < 3]

_AL_TEAMS = {"BAL", "BOS", "NYY", "TB", "TOR",
             "CLE", "CWS", "DET", "KC", "MIN",
             "HOU", "LAA", "OAK", "SEA", "TEX"}
_NL_TEAMS = {"ATL", "MIA", "NYM", "PHI", "WSH",
             "CHC", "CIN", "MIL", "PIT", "STL",
             "ARI", "COL", "LAD", "SD", "SF"}

HITTER_LEADERBOARDS = [
    ("AVG", "avg", True, ".000", None),
    ("OPS", "ops", True, ".000", None),
    ("Home Runs", "hr", True, "int", None),
    ("wOBA", "woba", True, ".000", None),
    ("Hits", "hits", True, "int", None),
    ("RBI", "rbi", True, "int", None),
    ("Stolen Bases", "sb", True, "int", None),
    ("Runs", "runs", True, "int", None),
    ("Doubles", "doubles", True, "int", None),
]

PITCHER_LEADERBOARDS = [
    ("ERA", "era", False, "0.00", _SP),
    ("Strikeouts", "k", True, "int", _SP),
    ("WHIP", "whip", False, "0.00", _SP),
    ("FIP", "fip", False, "0.00", _SP),
    ("K/9", "k_per_9", True, "0.00", _SP),
    ("Wins", "w", True, "int", _SP),
    ("Saves", "sv", True, "int", _RP),
    ("Holds", "hld", True, "int", _RP),
    ("IP", "ip", True, "dec1", _SP),
]


# ── CSS ────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
.stats-header {{
    text-align: center;
    margin-bottom: 0.8rem;
}}
.stats-title {{
    color: var(--tdd-cream);
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}}
.lb-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--tdd-dark-border);
}}
.lb-title {{
    color: var(--tdd-gold);
    font-size: 2.0rem;
    font-weight: 700;
    letter-spacing: 0.75px;
}}
.lb-subtitle {{
    color: var(--tdd-slate);
    font-size: 0.70rem;
    font-weight: 400;
    margin-left: 0.4rem;
}}
.lb-row {{
    display: flex;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--tdd-dark-border-faint);
}}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{
    color: var(--tdd-slate);
    font-size: 0.82rem;
    min-width: 1.6rem;
    text-align: right;
    margin-right: 0.5rem;
}}
.lb-rank-top {{ color: var(--tdd-gold); font-weight: 700; }}
.lb-headshot {{
    margin-left: 0.5rem;
    margin-right: 0.5rem;
}}
.lb-name {{
    color: var(--tdd-cream);
    font-size: 0.95rem;
    font-weight: 600;
    flex: 1;
}}
.lb-name a {{
    color: inherit;
    text-decoration: none;
}}
.lb-name a:hover {{
    color: var(--tdd-gold);
    text-decoration: underline;
}}
.lb-team {{
    color: var(--tdd-slate);
    font-size: 0.80rem;
    margin-right: 0.5rem;
}}
.lb-val {{
    color: var(--tdd-sage);
    font-size: 1rem;
    font-weight: 700;
    min-width: 2.5rem;
    text-align: right;
}}

.stSelectbox div[data-baseweb="select"] > div {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}}
.stSelectbox div[data-baseweb="select"] {{
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #C8A96E !important;
    cursor: pointer !important;
}}
.stSelectbox div[data-baseweb="select"] > div:hover {{
    background-color: transparent !important;
}}
</style>
"""


# ── Helpers ────────────────────────────────────────────────────────────

def _fmt_stat(val, fmt: str) -> str:
    if pd.isna(val):
        return "--"
    if fmt == ".000":
        s = f"{val:.3f}"
        return s.lstrip("0") if abs(val) < 1.0 else s
    if fmt == "0.00":
        return f"{val:.2f}"
    if fmt == "int":
        return str(int(val))
    if fmt == "dec1":
        return f"{val:.1f}"
    return str(val)


def _render_stat_leaderboard(
    df: pd.DataFrame,
    title: str,
    col: str,
    higher_is_better: bool,
    fmt: str,
    teams_lookup: dict[int, str],
    id_col: str,
    name_col: str,
    n_show: int = 10,
    role_filter=None,
    link_type: str = "",
) -> None:
    """Render a single stat leaderboard card."""
    work = df.copy()
    if role_filter is not None:
        work = role_filter(work)

    if col not in work.columns:
        return

    work = work[pd.notna(work[col]) & np.isfinite(work[col])]

    ascending = not higher_is_better
    top = work.sort_values(col, ascending=ascending).head(n_show)

    # Subtitle from role filter
    subtitle = ""
    if role_filter is not None:
        fn_str = str(role_filter)
        if ">= 3" in fn_str:
            subtitle = "SP"
        elif "< 3" in fn_str:
            subtitle = "RP"

    rows_html = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        team = teams_lookup.get(pid, "")
        val = _fmt_stat(row[col], fmt)
        rank_class = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"

        hs = f'<span class="lb-headshot">{headshot_html(pid, size=50)}</span>'

        if link_type:
            profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
            name_html = f'<a href="{profile_url}">{name}</a>'
        else:
            name_html = name

        team_html = f'<span class="lb-team" data-team="{team}">{team}</span>' if team else ""

        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{i}.</span>'
            f'{hs}'
            f'<span class="lb-name">{name_html}</span>'
            f'{team_html}'
            f'<span class="lb-val">{val}</span>'
            f'</div>'
        )

    subtitle_html = f'<span class="lb-subtitle">{subtitle}</span>' if subtitle else ""

    html = (
        f'<div class="lb-card">'
        f'<div class="lb-title-row">'
        f'<span class="lb-title">{title}{subtitle_html}</span>'
        f'</div>'
        + "".join(rows_html)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Main page ─────────────────────────────────────────────────────────

def page_stats() -> None:
    """Traditional stat leaderboards for any season."""
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Title ─────────────────────────────────────────────────────
    season_display = st.session_state.get("stats_season", PRIOR_SEASON)
    st.markdown(
        f'<div class="stats-header">'
        f'<div class="stats-title">{season_display} STATS</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Filters ───────────────────────────────────────────────────
    filter_cols = st.columns([2, 2, 2, 2, 2])

    with filter_cols[0]:
        player_type = st.selectbox(
            "Player Type",
            ["Hitter", "Pitcher"],
            key="stats_player_type",
            label_visibility="collapsed",
        )
    with filter_cols[1]:
        season = st.selectbox(
            "Season",
            AVAILABLE_SEASONS,
            key="stats_season",
            label_visibility="collapsed",
        )
    with filter_cols[2]:
        league_filter = st.selectbox(
            "League",
            ["All", "American League", "National League"],
            key="stats_league",
            label_visibility="collapsed",
        )
    with filter_cols[3]:
        n_show = st.selectbox(
            "Show Top",
            [5, 10, 15],
            index=1,
            key="stats_n",
            format_func=lambda x: f"Top {x}",
            label_visibility="collapsed",
        )
    with filter_cols[4]:
        if player_type == "Hitter":
            min_qual = st.selectbox(
                "Min PA",
                [0, 50, 100, 200, 400, 502],
                index=4,
                key="stats_min_pa",
                format_func=lambda x: f"Min {x} PA",
                label_visibility="collapsed",
            )
        else:
            min_qual = st.selectbox(
                "Min IP",
                [0, 10, 30, 50, 100, 162],
                index=3,
                key="stats_min_ip",
                format_func=lambda x: f"Min {x} IP",
                label_visibility="collapsed",
            )

    # ── Load data ─────────────────────────────────────────────────
    pt_key = player_type.lower()
    df = load_traditional_stats_all(pt_key)
    if not df.empty and "season" in df.columns:
        df = df[df["season"] == season].copy()
    else:
        df = load_traditional_stats(pt_key)

    if df.empty:
        st.info(f"No {season} stats data found.")
        return

    if player_type == "Hitter":
        id_col, name_col = "batter_id", "batter_name"
        df = df[df["pa"] >= min_qual]
    else:
        id_col, name_col = "pitcher_id", "pitcher_name"
        df = df[df["ip"] >= min_qual]

    # Team lookup + league filter
    teams_df = load_player_teams()
    teams_lookup: dict[int, str] = {}
    if not teams_df.empty:
        teams_lookup = dict(zip(
            teams_df["player_id"].astype(int), teams_df["team_abbr"]
        ))

    if league_filter != "All" and teams_lookup:
        target_teams = _AL_TEAMS if "American" in league_filter else _NL_TEAMS
        league_ids = {pid for pid, abbr in teams_lookup.items() if abbr in target_teams}
        df = df[df[id_col].isin(league_ids)]

    # ── Leaderboard cards ─────────────────────────────────────────
    leaderboards = HITTER_LEADERBOARDS if player_type == "Hitter" else PITCHER_LEADERBOARDS
    lt = "hitter" if player_type == "Hitter" else "pitcher"

    for i in range(0, len(leaderboards), 3):
        batch = leaderboards[i:i + 3]
        cols = st.columns(len(batch))
        for col_st, (title, stat_col, hib, fmt, role_fn) in zip(cols, batch):
            with col_st:
                _render_stat_leaderboard(
                    df, title, stat_col, hib, fmt,
                    teams_lookup, id_col, name_col,
                    link_type=lt, n_show=n_show, role_filter=role_fn,
                )

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    notes = [f"{season} regular season"]
    if season == 2020:
        notes.append("60-game shortened season")
    if season in UNRELIABLE_BB_SEASONS:
        notes.append("Batted ball metrics unreliable pre-2022")
    qual_label = f"min {min_qual} PA" if player_type == "Hitter" else f"min {min_qual} IP"
    st.caption(
        f"Showing {len(df)} qualified {pt_key}s ({qual_label}) | {' | '.join(notes)}"
    )
