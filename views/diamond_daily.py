"""The Diamond Daily -- observed daily standouts and 14-day heat check."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.alerts import tdd_info, tdd_warn
from components.leaderboard import render_card
from services.data_loader import (
    load_roster,
    load_hitters_daily_standouts,
    load_pitchers_daily_standouts,
    load_hitters_weekly_form,
    load_pitchers_weekly_form,
)


_CSS = """
<style>
.dd-headline-card {
    border: 1px solid var(--tdd-dark-border);
    border-radius: var(--tdd-radius-md);
    background: var(--tdd-dark-card);
    padding: var(--tdd-space-2) var(--tdd-space-3);
    margin-bottom: var(--tdd-space-3);
}
.dd-headline-label {
    color: var(--tdd-slate);
    font-size: var(--tdd-fs-badge);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.dd-headline-name {
    color: var(--tdd-cream);
    font-family: var(--tdd-font-heading);
    font-size: var(--tdd-fs-player-name);
    font-weight: 600;
    line-height: 1.2;
}
.dd-headline-val {
    color: var(--tdd-gold);
    font-size: 1.0rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
}
</style>
"""


def _filter_role(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[col] == value].copy()


_SP_DAILY = lambda df: _filter_role(df, "role", "SP")
_RP_DAILY = lambda df: _filter_role(df, "role", "RP")
_SP_WEEKLY = lambda df: _filter_role(df, "role_14d", "SP")
_RP_WEEKLY = lambda df: _filter_role(df, "role_14d", "RP")


_DAILY_HITTER_LEADERBOARDS = [
    ("Daily Standout Score", "daily_standout_score", True, "dec3", None, ""),
    ("Most Hits", "hits", True, "int", None, ""),
    ("Most HR", "hr", True, "int", None, ""),
    ("Most RBI", "rbi", True, "int", None, ""),
    ("Best OPS", "ops", True, "dec3", None, ""),
    ("Best wOBA", "woba", True, "dec3", None, ""),
]

_DAILY_PITCHER_LEADERBOARDS = [
    ("Daily Standout Score", "daily_standout_score", True, "dec3", _SP_DAILY, "SP"),
    ("Most Ks", "k", True, "int", _SP_DAILY, "SP"),
    ("Best ERA", "era", False, "dec2", _SP_DAILY, "SP"),
    ("Daily Standout Score", "daily_standout_score", True, "dec3", _RP_DAILY, "RP"),
    ("Most Ks", "k", True, "int", _RP_DAILY, "RP"),
    ("Best WHIP", "whip", False, "dec2", _RP_DAILY, "RP"),
]

_WEEKLY_HITTER_LEADERBOARDS = [
    ("Weekly Form Score", "weekly_form_score", True, "dec3", None, ""),
    ("Most Hits (14d)", "hits_14d", True, "int", None, ""),
    ("Most HR (14d)", "hr_14d", True, "int", None, ""),
    ("Best wOBA (14d)", "woba_14d", True, "dec3", None, ""),
    ("Best xwOBA (14d)", "xwoba_14d", True, "dec3", None, ""),
    ("Lowest K Rate (14d)", "k_rate_14d", False, "pct1", None, ""),
]

_WEEKLY_PITCHER_LEADERBOARDS = [
    ("Weekly Form Score", "weekly_form_score", True, "dec3", _SP_WEEKLY, "SP"),
    ("Best K-BB (14d)", "k_minus_bb_14d", True, "pct1", _SP_WEEKLY, "SP"),
    ("Best ERA (14d)", "era_14d", False, "dec2", _SP_WEEKLY, "SP"),
    ("Weekly Form Score", "weekly_form_score", True, "dec3", _RP_WEEKLY, "RP"),
    ("Best K-BB (14d)", "k_minus_bb_14d", True, "pct1", _RP_WEEKLY, "RP"),
    ("Best WHIP (14d)", "whip_14d", False, "dec2", _RP_WEEKLY, "RP"),
]


def _fmt_stat(value: object, fmt: str) -> str:
    if pd.isna(value):
        return ""
    val = float(value)
    if fmt == "int":
        return str(int(round(val)))
    if fmt == "dec2":
        return f"{val:.2f}"
    if fmt == "dec3":
        return f"{val:.3f}"
    if fmt == "pct1":
        return f"{val * 100:.1f}%"
    return str(value)


def _attach_teams(
    df: pd.DataFrame,
    id_col: str,
    teams_lookup: dict[int, str],
) -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return df
    out = df.copy()
    if "team_abbr" not in out.columns:
        out["team_abbr"] = out[id_col].map(
            lambda pid: teams_lookup.get(int(pid), "") if pd.notna(pid) else ""
        )
    else:
        out["team_abbr"] = out["team_abbr"].fillna(
            out[id_col].map(
                lambda pid: teams_lookup.get(int(pid), "") if pd.notna(pid) else ""
            )
        )
    return out


def _render_leaderboard(
    df: pd.DataFrame,
    title: str,
    stat_col: str,
    higher_is_better: bool,
    fmt: str,
    id_col: str,
    name_col: str,
    teams_lookup: dict[int, str],
    link_type: str,
    n_show: int,
    role_filter=None,
    subtitle: str = "",
) -> None:
    work = df.copy()
    if role_filter is not None:
        work = role_filter(work)
    if work.empty or stat_col not in work.columns:
        return

    work = work[pd.notna(work[stat_col])]
    if work.empty:
        return

    top = work.sort_values(stat_col, ascending=not higher_is_better).head(n_show)

    rows: list[dict] = []
    for _, row in top.iterrows():
        if pd.isna(row.get(id_col)):
            continue
        pid = int(row[id_col])
        rows.append({
            "player_id": pid,
            "name": row[name_col],
            "value": _fmt_stat(row[stat_col], fmt),
            "team": teams_lookup.get(pid, row.get("team_abbr", "")),
            "link_type": link_type,
        })

    st.markdown(
        render_card(
            title=f"Top {n_show} {title}",
            rows=rows,
            subtitle=subtitle,
        ),
        unsafe_allow_html=True,
    )


def _top_row(df: pd.DataFrame, col: str, higher_is_better: bool) -> pd.Series | None:
    if df.empty or col not in df.columns:
        return None
    work = df[pd.notna(df[col])]
    if work.empty:
        return None
    idx = work[col].idxmax() if higher_is_better else work[col].idxmin()
    return work.loc[idx]


def _build_daily_headlines(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    farthest_col = next(
        (c for c in ("hit_distance_sc", "hit_distance", "max_hr_distance", "longest_hr_ft") if c in hitters.columns),
        "",
    )
    candidates: list[tuple[str, pd.DataFrame, str, bool, str]] = [
        ("Most Hits", hitters, "hits", True, "int"),
        ("Most HR", hitters, "hr", True, "int"),
        ("Most Ks", pitchers, "k", True, "int"),
    ]
    if farthest_col:
        candidates.append(("Farthest HR", hitters, farthest_col, True, "dec1"))
    else:
        candidates.append(("Best Standout Score", hitters, "daily_standout_score", True, "dec3"))

    for label, df, col, hib, fmt in candidates:
        row = _top_row(df, col, hib)
        if row is None:
            continue
        name = row.get("batter_name") or row.get("pitcher_name") or "N/A"
        val = _fmt_stat(row[col], fmt)
        items.append((label, str(name), val))
    return items[:4]


def _build_weekly_headlines(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    sp = _SP_WEEKLY(pitchers)
    rp = _RP_WEEKLY(pitchers)
    candidates: list[tuple[str, pd.DataFrame, str, bool, str]] = [
        ("Hottest Hitter", hitters, "weekly_form_score", True, "dec3"),
        ("Hottest SP", sp, "weekly_form_score", True, "dec3"),
        ("Hottest RP", rp, "weekly_form_score", True, "dec3"),
        ("Best 14d K-BB", pitchers, "k_minus_bb_14d", True, "pct1"),
    ]
    for label, df, col, hib, fmt in candidates:
        row = _top_row(df, col, hib)
        if row is None:
            continue
        name = row.get("batter_name") or row.get("pitcher_name") or "N/A"
        val = _fmt_stat(row[col], fmt)
        items.append((label, str(name), val))
    return items[:4]


def _render_headlines(items: list[tuple[str, str, str]]) -> None:
    if not items:
        return
    cols = st.columns(min(len(items), 4))
    for col, (label, name, val) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="dd-headline-card">'
                f'<div class="dd-headline-label">{label}</div>'
                f'<div class="dd-headline-name">{name}</div>'
                f'<div class="dd-headline-val">{val}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def page_diamond_daily() -> None:
    """Observed daily best performances and 14-day form boards."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="tdd-page-header">'
        '<div class="tdd-page-title">THE DIAMOND DAILY</div>'
        '<div class="tdd-page-subtitle">Observed leaders from completed games and recent form trends</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_mode, col_type, col_n = st.columns([2, 2, 1.5])
    with col_mode:
        mode = st.selectbox(
            "View",
            ["Daily Standouts", "14-Day Heat Check"],
            key="diamond_daily_mode",
            label_visibility="collapsed",
        )
    with col_type:
        player_view = st.selectbox(
            "Player Type",
            ["All", "Hitters", "Pitchers"],
            key="diamond_daily_player_type",
            label_visibility="collapsed",
        )
    with col_n:
        n_show = st.selectbox(
            "Show Top",
            [5, 10, 15],
            index=1,
            key="diamond_daily_n",
            format_func=lambda x: f"Top {x}",
            label_visibility="collapsed",
        )

    if mode == "Daily Standouts":
        hitters = load_hitters_daily_standouts()
        pitchers = load_pitchers_daily_standouts()
        hitter_boards = _DAILY_HITTER_LEADERBOARDS
        pitcher_boards = _DAILY_PITCHER_LEADERBOARDS
    else:
        hitters = load_hitters_weekly_form()
        pitchers = load_pitchers_weekly_form()
        hitter_boards = _WEEKLY_HITTER_LEADERBOARDS
        pitcher_boards = _WEEKLY_PITCHER_LEADERBOARDS

    if hitters.empty and pitchers.empty:
        tdd_info(
            "No Diamond Daily data found yet. Run precompute to generate "
            "`*_daily_standouts.parquet` and `*_weekly_form.parquet`."
        )
        return

    roster = load_roster()
    teams_lookup: dict[int, str] = {}
    if not roster.empty and "player_id" in roster.columns and "team_abbr" in roster.columns:
        teams_lookup = dict(zip(roster["player_id"].astype(int), roster["team_abbr"]))

    hitters = _attach_teams(hitters, "batter_id", teams_lookup)
    pitchers = _attach_teams(pitchers, "pitcher_id", teams_lookup)

    headlines = (
        _build_daily_headlines(hitters, pitchers)
        if mode == "Daily Standouts"
        else _build_weekly_headlines(hitters, pitchers)
    )
    _render_headlines(headlines)

    if player_view in ("All", "Hitters") and not hitters.empty:
        st.markdown('<div class="tdd-section-hdr">Hitters</div>', unsafe_allow_html=True)
        for i in range(0, len(hitter_boards), 3):
            batch = hitter_boards[i:i + 3]
            cols = st.columns(len(batch))
            for col, (title, stat_col, hib, fmt, role_filter, subtitle) in zip(cols, batch):
                with col:
                    _render_leaderboard(
                        hitters,
                        title=title,
                        stat_col=stat_col,
                        higher_is_better=hib,
                        fmt=fmt,
                        id_col="batter_id",
                        name_col="batter_name",
                        teams_lookup=teams_lookup,
                        link_type="hitter",
                        n_show=n_show,
                        role_filter=role_filter,
                        subtitle=subtitle,
                    )

    if player_view in ("All", "Pitchers") and not pitchers.empty:
        st.markdown('<div class="tdd-section-hdr">Pitchers</div>', unsafe_allow_html=True)
        for i in range(0, len(pitcher_boards), 3):
            batch = pitcher_boards[i:i + 3]
            cols = st.columns(len(batch))
            for col, (title, stat_col, hib, fmt, role_filter, subtitle) in zip(cols, batch):
                with col:
                    _render_leaderboard(
                        pitchers,
                        title=title,
                        stat_col=stat_col,
                        higher_is_better=hib,
                        fmt=fmt,
                        id_col="pitcher_id",
                        name_col="pitcher_name",
                        teams_lookup=teams_lookup,
                        link_type="pitcher",
                        n_show=n_show,
                        role_filter=role_filter,
                        subtitle=subtitle,
                    )

    st.markdown("---")
    if mode == "Daily Standouts":
        st.caption(
            "Daily Standouts celebrate single-game observed performances "
            "from the latest completed slate."
        )
    else:
        st.caption(
            "14-Day Heat Check ranks recent form from rolling two-week production."
        )
