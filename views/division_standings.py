"""Division Standings -- current + preseason projected standings by division."""
from __future__ import annotations


import pandas as pd
import streamlit as st

from utils.alerts import tdd_info, tdd_warn
from config import GOLD, SAGE, SLATE, DARK_BORDER, DASHBOARD_DIR
from components.team_logo import team_logo_html
from services.data_loader import load_team_rankings


# Division display order: AL East/Central/West, NL East/Central/West
_DIVISIONS = [
    ("American League East", "AL East"),
    ("American League Central", "AL Central"),
    ("American League West", "AL West"),
    ("National League East", "NL East"),
    ("National League Central", "NL Central"),
    ("National League West", "NL West"),
]

_TIER_COLORS = {
    "Elite": GOLD,
    "Contender": SAGE,
    "Competitive": SLATE,
    "Rebuilding": "#7B8FA6",
}

_PRESEASON_SNAPSHOT = DASHBOARD_DIR / "snapshots" / "team_rankings_2026_preseason.parquet"


def _load_preseason_rankings() -> pd.DataFrame:
    if not _PRESEASON_SNAPSHOT.exists():
        return pd.DataFrame()
    return pd.read_parquet(_PRESEASON_SNAPSHOT)


def _division_card(teams: pd.DataFrame, short_name: str,
                   standings: dict[str, tuple[int, int]] | None = None) -> str:
    """Render a single division leaderboard card matching the lb-* style."""
    if standings is None:
        standings = {}
    rows_html = ""
    for i, (_, row) in enumerate(teams.iterrows(), 1):
        tid = int(row.get("team_id", 0))
        abbr = row.get("abbreviation", "")
        score = float(row.get("tdd_score", 0))
        tier = row.get("tier", "")
        tier_color = _TIER_COLORS.get(tier, SLATE)
        _rec = standings.get(abbr)
        record_html = (
            f'<span class="lb-elo-val" style="color:var(--tdd-gold);">'
            f'{_rec[0]}-{_rec[1]}</span>'
            if _rec else ""
        )

        rank_cls = "lb-rank-top lb-rank" if i <= 2 else "lb-rank"
        logo = f'<span style="margin:0 0.4rem;">{team_logo_html(tid, size=28)}</span>'

        rows_html += (
            f'<div class="lb-row" style="padding:0.35rem 0;">'
            f'<span class="{rank_cls}">{i}.</span>'
            f'{logo}'
            f'<span class="lb-name tdd-team-abbr" data-team="{abbr}" '
            f'style="flex:1;">{abbr}</span>'
            # Tier pill
            f'<span style="font-size:0.65rem; color:{tier_color}; '
            f'margin-right:0.6rem;">{tier}</span>'
            # TDD score
            f'<span style="color:{SLATE}; font-size:0.75rem; '
            f'margin-right:0.6rem;">{score:.1f}</span>'
            # Record
            f'{record_html}'
            f'</div>'
        )

    return (
        f'<div class="lb-card">'
        f'<div class="lb-title-row"><span class="lb-title">{short_name}</span></div>'
        f'<div class="lb-scroll">{rows_html}</div>'
        f'</div>'
    )


def _render_divisions(rankings: pd.DataFrame,
                      standings: dict[str, tuple[int, int]] | None = None) -> None:
    """Render AL + NL division cards from a rankings dataframe."""
    if standings is None:
        standings = {}

    # Add actual wins for sorting
    rankings = rankings.copy()
    rankings["_sort_wins"] = rankings["abbreviation"].map(
        lambda a: standings[a][0] if a in standings else 0
    )

    al_divs = [d for d in _DIVISIONS if d[0].startswith("American")]
    nl_divs = [d for d in _DIVISIONS if d[0].startswith("National")]

    st.markdown(
        f'<div style="color:var(--tdd-gold); font-size:1.0rem; font-weight:700; '
        f'letter-spacing:0.5px; margin:0.6rem 0 0.4rem; padding-bottom:0.3rem; '
        f'border-bottom:1px solid {DARK_BORDER};">American League</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (div_full, div_short) in zip(cols, al_divs):
        div_teams = rankings[rankings["division"] == div_full].sort_values(
            "_sort_wins", ascending=False,
        ).reset_index(drop=True)
        with col:
            st.markdown(_division_card(div_teams, div_short, standings), unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:var(--tdd-gold); font-size:1.0rem; font-weight:700; '
        f'letter-spacing:0.5px; margin:1.0rem 0 0.4rem; padding-bottom:0.3rem; '
        f'border-bottom:1px solid {DARK_BORDER};">National League</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (div_full, div_short) in zip(cols, nl_divs):
        div_teams = rankings[rankings["division"] == div_full].sort_values(
            "_sort_wins", ascending=False,
        ).reset_index(drop=True)
        with col:
            st.markdown(_division_card(div_teams, div_short, standings), unsafe_allow_html=True)


def page_division_standings() -> None:
    """Render the Division Standings page."""
    from services.data_loader import load_standings

    st.markdown(
        '<div class="section-header">Division Standings</div>',
        unsafe_allow_html=True,
    )

    _standings = load_standings()

    tab_current, tab_preseason = st.tabs(["Current Standings", "Preseason Projections"])

    with tab_current:
        rankings = load_team_rankings()
        if rankings.empty:
            tdd_warn("No team rankings data found. Run precompute first.")
        else:
            _render_divisions(rankings, _standings)

    with tab_preseason:
        preseason = _load_preseason_rankings()
        if preseason.empty:
            tdd_warn("No preseason snapshot found.")
        else:
            st.markdown(
                '<div class="tdd-meta" style="margin-bottom:0.8rem;">'
                'Frozen preseason projections (March 26, 2026)</div>',
                unsafe_allow_html=True,
            )
            _render_divisions(preseason)
