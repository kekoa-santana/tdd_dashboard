"""Division Standings -- preseason projected standings by division."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, SAGE, SLATE, DARK_BORDER
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


def _division_card(teams: pd.DataFrame, short_name: str) -> str:
    """Render a single division leaderboard card matching the lb-* style."""
    rows_html = ""
    for i, (_, row) in enumerate(teams.iterrows(), 1):
        tid = int(row.get("team_id", 0))
        abbr = row.get("abbreviation", "")
        wins = float(row.get("projected_wins", 0))
        score = float(row.get("tdd_score", 0))
        tier = row.get("tier", "")
        tier_color = _TIER_COLORS.get(tier, SLATE)

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
            # Projected wins
            f'<span class="lb-elo-val" style="color:var(--tdd-gold);">'
            f'{wins:.0f}W</span>'
            f'</div>'
        )

    return (
        f'<div class="lb-card">'
        f'<div class="lb-title-row"><span class="lb-title">{short_name}</span></div>'
        f'<div class="lb-scroll">{rows_html}</div>'
        f'</div>'
    )


def page_division_standings() -> None:
    """Render the Division Standings page."""
    st.markdown(
        '<div class="section-header">Division Standings</div>'
        '<div class="tdd-meta" style="margin-top:-0.5rem; margin-bottom:0.8rem;">'
        'Preseason projected standings by division</div>',
        unsafe_allow_html=True,
    )

    rankings = load_team_rankings()
    if rankings.empty:
        st.warning("No team rankings data found. Run precompute first.")
        return

    # AL divisions row
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
            "projected_wins", ascending=False,
        ).reset_index(drop=True)
        with col:
            st.markdown(_division_card(div_teams, div_short), unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:var(--tdd-gold); font-size:1.0rem; font-weight:700; '
        f'letter-spacing:0.5px; margin:1.0rem 0 0.4rem; padding-bottom:0.3rem; '
        f'border-bottom:1px solid {DARK_BORDER};">National League</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (div_full, div_short) in zip(cols, nl_divs):
        div_teams = rankings[rankings["division"] == div_full].sort_values(
            "projected_wins", ascending=False,
        ).reset_index(drop=True)
        with col:
            st.markdown(_division_card(div_teams, div_short), unsafe_allow_html=True)
