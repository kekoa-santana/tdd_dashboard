"""Team Rankings -- standings layout with variant tabs.

Design: eyebrow > masthead > variant tabs (Standings | Tiered | Division)
  > filter chips > full league table with expandable rows > methodology footer.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE
from components.team_logo import team_logo_html
from lib.diamond_rating import score_to_diamonds
from services.data_loader import (
    load_team_rankings,
    load_team_profiles,
    load_team_elo,
    load_standings,
)

# ---------------------------------------------------------------------------
# Division short labels
# ---------------------------------------------------------------------------
_DIV_SHORT: dict[str, str] = {
    "American League East": "AL E",
    "American League Central": "AL C",
    "American League West": "AL W",
    "National League East": "NL E",
    "National League Central": "NL C",
    "National League West": "NL W",
}

_TIER_DESCS: dict[str, str] = {
    "Elite": "World-Series odds in the model's top decile.",
    "Contender": "October-bound if pitching holds; coin-flip rotations apart.",
    "Competitive": "Hovering .500 -- one bat or arm away from contender tier.",
    "Rebuilding": "Sub-.450 win expectancy; rolling young talent forward.",
}


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def _delta_html(d: int) -> str:
    if d == 0:
        return '<span class="delta flat">--</span>'
    arrow = "&#9650;" if d > 0 else "&#9660;"
    cls = "up" if d > 0 else "down"
    return f'<span class="delta {cls}">{arrow}{abs(d)}</span>'


def _comp_bars_html(items: list[tuple[str, float]], max_v: float = 1.0) -> str:
    """Render comparative bar rows. items = [(label, value), ...]."""
    lg_avg_pct = (0.5 / max_v) * 100  # league average at 0.5 on 0-1 scale
    rows = []
    for label, v in items:
        pct = max(0, min(100, (v / max_v) * 100))
        rows.append(
            f'<div class="row">'
            f'<span class="lab">{label}</span>'
            f'<div class="bar">'
            f'<div class="fill" style="width:{pct:.1f}%"></div>'
            f'<div class="lg" style="left:{lg_avg_pct:.1f}%"></div>'
            f'</div>'
            f'<span class="v">{v:.2f}</span>'
            f'</div>'
        )
    return f'<div class="rk-comp">{"".join(rows)}</div>'


def _record_html(wins: int, losses: int, run_diff: float | None = None) -> str:
    rd_html = ""
    if run_diff is not None:
        sign = "+" if run_diff > 0 else ""
        cls = "up" if run_diff > 0 else "down" if run_diff < 0 else ""
        rd_html = f'<span class="rd {cls}">{sign}{run_diff:.0f}</span>'
    return f'<span class="rec">{wins}-{losses}{rd_html}</span>'


# ---------------------------------------------------------------------------
# Variant renderers
# ---------------------------------------------------------------------------

def _render_standings(
    teams: pd.DataFrame,
    standings: dict[str, tuple[int, int]],
    expanded_key: str,
    preseason_ranks: dict[str, int] | None = None,
) -> None:
    """Full league standings table -- sortable, expandable rows."""

    # Header
    head = (
        '<div class="rk-team-head-row">'
        '<span class="ralign">Rk</span>'
        '<span class="calign">&Delta;</span>'
        '<span>Tm</span>'
        '<span>Name</span>'
        '<span class="ralign">&#9670; Score</span>'
        '<span class="ralign col-off">Off</span>'
        '<span class="ralign col-pit">Pit</span>'
        '<span class="ralign col-def">Def</span>'
        '<span class="ralign">Record</span>'
        '<span class="ralign">ELO</span>'
        '<span></span>'
        '</div>'
    )

    rows: list[str] = []
    for _, t in teams.iterrows():
        rank = int(t["rank"])
        abbr = str(t["abbreviation"])
        name = str(t["team_name"])
        div_full = str(t.get("division", ""))
        div_short = _DIV_SHORT.get(div_full, div_full[:4] if div_full else "")
        tdd = float(t.get("tdd_score", 0))
        off = float(t.get("offense_score", 0))
        rot = float(t.get("rotation_score", 0))
        bp = float(t.get("bullpen_score", 0))
        defense = float(t.get("defense_score", 0))
        elo = float(t.get("composite_elo", 1500))
        tier = str(t.get("tier", ""))

        # Rank delta vs preseason (positive = improved)
        if preseason_ranks:
            prev_rank = preseason_ranks.get(abbr, rank)
        else:
            prev_rank = rank
        d_rank = prev_rank - rank

        # Record
        rec = standings.get(abbr)
        wins, losses = rec if rec else (0, 0)

        # Run differential approximation from rpg / ra
        rpg = float(t.get("rpg", 0) or 0)
        ra = float(t.get("ra_per_game", 0) or 0)
        gp = wins + losses
        run_diff = (rpg - ra) * gp if gp > 0 and rpg > 0 and ra > 0 else None

        # Rank class
        rk_cls = "top1" if rank == 1 else ("top5" if rank <= 5 else "")
        rk_color = f"var(--tdd-gold)" if rank == 1 else (
            "var(--tdd-cream)" if rank <= 5 else "var(--tdd-slate)"
        )

        # Team row
        row_html = (
            f'<div class="rk-team-row">'
            f'<span class="rk" style="font-family:var(--tdd-font-heading); '
            f'font-size:1.05rem; font-weight:800; text-align:right; '
            f'font-variant-numeric:tabular-nums; color:{rk_color}">{rank}</span>'
            f'{_delta_html(d_rank)}'
            f'<span class="abbr" data-team="{abbr}">{abbr}</span>'
            f'<span class="nm" data-team="{abbr}">'
            f'<span class="full">{name}</span>'
            f'<span class="div">{div_short}</span>'
            f'</span>'
            f'<span class="sub" style="color:var(--tdd-gold); font-size:1rem;">'
            f'&#9670; {tdd:.1f}</span>'
            f'<span class="sub off">{off:.2f}</span>'
            f'<span class="sub pit">{rot:.2f}</span>'
            f'<span class="sub def">{defense:.2f}</span>'
            f'{_record_html(wins, losses, run_diff)}'
            f'<span class="elo">{elo:.0f}</span>'
            f'<span style="color:var(--tdd-slate); opacity:0.5; text-align:right;">&#8250;</span>'
            f'</div>'
        )

        # Expanded detail
        comp_items = [
            ("Composite", tdd / 10.0),
            ("Offense", off),
            ("Rotation", rot),
            ("Bullpen", bp),
            ("Defense", defense),
        ]
        meta_cells = [
            (tier, "Tier"),
            (div_short, "Div"),
            (f"{wins}-{losses}", "Record"),
            (f"{elo:.0f}", "ELO"),
        ]
        meta_html = "".join(
            f'<div class="c"><div class="v">{v}</div><div class="l">{l}</div></div>'
            for v, l in meta_cells
        )

        styles_html = ""
        for col_name in ["offense_style", "pitching_style", "age_trajectory"]:
            val = t.get(col_name)
            if pd.notna(val) and str(val) != "nan":
                styles_html += (
                    f'<span style="background:rgba(200,169,110,0.08); '
                    f'color:var(--tdd-slate); border:1px solid var(--tdd-dark-border); '
                    f'padding:2px 8px; border-radius:2px; font-size:0.68rem; '
                    f'font-weight:600; margin-right:4px;">{val}</span>'
                )

        styles_block = (
            f'<div style="margin-top:0.5rem;">{styles_html}</div>'
            if styles_html else ""
        )
        expand_html = (
            f'<div class="rk-expand">'
            f'<div>'
            f'<div class="meta-grid">{meta_html}</div>'
            f'{styles_block}'
            f'<a class="open-link" href="?page=team_overview&team={abbr}">'
            f'Open Team Overview &#8594;</a>'
            f'</div>'
            f'<div>'
            f'<div style="font-family:var(--tdd-font-heading); font-size:0.6rem; '
            f'letter-spacing:1.3px; color:var(--tdd-slate); font-weight:700; '
            f'text-transform:uppercase; margin-bottom:0.5rem;">'
            f'Sub-scores &middot; 0-1 &middot; vs lg avg</div>'
            f'{_comp_bars_html(comp_items)}'
            f'</div></div>'
        )

        rows.append(
            f'<details class="rk-team-details">'
            f'<summary>{row_html}</summary>'
            f'{expand_html}'
            f'</details>'
        )

    html = (
        f'<div class="rk-sec"><span class="num">01</span>'
        f'<h2>Power Rankings</h2>'
        f'<span class="sub">Composite &#9670; Score &middot; all {len(teams)} teams</span></div>'
        f'<div class="rk-list">{head}{"".join(rows)}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_tiered(
    teams: pd.DataFrame,
    standings: dict[str, tuple[int, int]],
) -> None:
    """Tiered layout -- Elite/Contender/Competitive/Rebuilding cohorts."""
    tiers = ["Elite", "Contender", "Competitive", "Rebuilding"]

    html = (
        '<div class="rk-sec"><span class="num">01</span>'
        '<h2>Tiers</h2>'
        '<span class="sub">Power-rank cohorts</span></div>'
    )

    for tier in tiers:
        tier_teams = teams[teams["tier"] == tier].sort_values("rank")
        if tier_teams.empty:
            continue

        desc = _TIER_DESCS.get(tier, "")
        cls = tier.lower()

        tier_rows: list[str] = []
        for _, t in tier_teams.iterrows():
            rank = int(t["rank"])
            abbr = str(t["abbreviation"])
            name = str(t["team_name"])
            tdd = float(t.get("tdd_score", 0))
            rec = standings.get(abbr)
            wins, losses = rec if rec else (0, 0)

            rk_color = "var(--tdd-gold)" if rank <= 5 else "var(--tdd-cream)" if rank <= 10 else "var(--tdd-slate)"
            tier_rows.append(
                f'<div class="rk-team-row" style="grid-template-columns:'
                f'2.1rem 2.6rem minmax(0,1fr) 3.2rem 4rem 1.2rem;">'
                f'<span style="font-family:var(--tdd-font-heading); '
                f'font-variant-numeric:tabular-nums; font-size:1.05rem; '
                f'font-weight:800; text-align:right; color:{rk_color}">{rank}</span>'
                f'<span class="abbr" data-team="{abbr}">{abbr}</span>'
                f'<span class="nm" data-team="{abbr}"><span class="full">{name}</span></span>'
                f'<span class="sub" style="color:var(--tdd-gold);">&#9670; {tdd:.1f}</span>'
                f'<span class="rec">{wins}-{losses}</span>'
                f'<span style="color:var(--tdd-slate); opacity:0.5; text-align:right;">&#8250;</span>'
                f'</div>'
            )

        html += (
            f'<div class="rk-tier">'
            f'<div class="rk-tier-head {cls}">'
            f'<span class="label"><span class="star">&#9733;</span>{tier}</span>'
            f'<span class="count">{len(tier_teams)} teams</span>'
            f'<span class="desc">{desc}</span>'
            f'</div>'
            f'<div class="body">{"".join(tier_rows)}</div>'
            f'</div>'
        )

    st.markdown(html, unsafe_allow_html=True)


def _render_division(
    teams: pd.DataFrame,
    standings: dict[str, tuple[int, int]],
) -> None:
    """Division grid -- 6 mini-cards sorted by diamond score."""
    divs = ["AL E", "AL C", "AL W", "NL E", "NL C", "NL W"]
    full_to_short = _DIV_SHORT

    html = (
        '<div class="rk-sec"><span class="num">01</span>'
        '<h2>By Division</h2>'
        '<span class="sub">Sorted within division by &#9670; Score</span></div>'
        '<div class="rk-div-grid">'
    )

    for d in divs:
        # Find teams in this division
        div_teams = teams[
            teams["division"].map(lambda x: _DIV_SHORT.get(str(x), "")) == d
        ].sort_values("tdd_score", ascending=False)

        rows = ""
        for _, t in div_teams.iterrows():
            rank = int(t["rank"])
            abbr = str(t["abbreviation"])
            tdd = float(t.get("tdd_score", 0))
            rec = standings.get(abbr)
            rec_str = f"{rec[0]}-{rec[1]}" if rec else ""

            rows += (
                f'<a href="?page=team_overview&team={abbr}" '
                f'style="text-decoration:none; color:inherit;">'
                f'<div class="rk-div-row">'
                f'<span class="rk{" top" if rank <= 5 else ""}">{rank}</span>'
                f'<span class="ab" data-team="{abbr}">{abbr}</span>'
                f'<span class="rec">{rec_str}</span>'
                f'<span class="sc">{tdd:.1f}</span>'
                f'</div></a>'
            )

        html += f'<div class="rk-div-card"><h3>{d}</h3>{rows}</div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def page_team_rankings() -> None:
    """Render the Team Rankings page."""

    # ---- Data ----
    rankings = load_team_rankings()
    profiles = load_team_profiles()
    elo_df = load_team_elo(preseason=True)
    standings = load_standings()

    # Preseason snapshot for rank deltas
    preseason_ranks: dict[str, int] = {}
    try:
        from config import DASHBOARD_DIR
        pre_path = DASHBOARD_DIR / "snapshots" / "team_rankings_2026_preseason.parquet"
        if pre_path.exists():
            import pandas as _pd
            pre_df = _pd.read_parquet(pre_path, columns=["abbreviation", "rank"])
            preseason_ranks = dict(zip(pre_df["abbreviation"], pre_df["rank"].astype(int)))
    except Exception:
        pass

    if rankings.empty:
        from utils.alerts import tdd_warn
        tdd_warn("No team rankings data found. Run precompute first.")
        return

    n_teams = len(rankings)
    as_of = date.today().strftime("%b %d, %Y")

    # ---- Eyebrow + masthead ----
    masthead = (
        '<div class="rk-page">'
        '<div class="rk-eyebrow">'
        '<span class="gold">&#9733;</span>'
        '<span>The Data Diamond</span>'
        '<span class="sep">/</span>'
        '<span>Rankings</span>'
        '<span class="sep">/</span>'
        '<span>Teams</span>'
        '</div>'
        '<div class="rk-mast">'
        '<div>'
        '<h1>Team Rankings</h1>'
        '<div class="dek">'
        'Composite ratings across <span class="gold">offense, pitching, and defense</span> '
        f'with ELO and division context. &Delta; vs preseason &middot; {as_of}.'
        '</div>'
        '</div>'
        '<div class="meta">'
        f'<div class="stat"><div class="v">{n_teams}</div><div class="l">Teams</div></div>'
        '<div class="stat"><div class="v">5.0</div><div class="l">Lg Avg</div></div>'
        '</div>'
        '</div>'
    )
    st.markdown(masthead, unsafe_allow_html=True)

    # ---- Variant tabs (Streamlit radio as styled tabs) ----
    variant = st.radio(
        "Layout",
        ["Standings", "Tiered", "By Division"],
        horizontal=True,
        key="tr_variant",
        label_visibility="collapsed",
    )

    # ---- Filters (for Standings and Tiered) ----
    filtered = rankings.copy()
    sort_by = "rank"

    if variant != "By Division":
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            league = st.radio(
                "League", ["MLB", "AL", "NL"],
                horizontal=True, key="tr_league",
                label_visibility="collapsed",
            )
        with fc2:
            if variant == "Standings":
                sort_by = st.radio(
                    "Sort", ["&#9670; Score", "Offense", "Pitching", "Defense", "ELO"],
                    horizontal=True, key="tr_sort",
                    label_visibility="collapsed",
                )

        # Apply league filter
        if league == "AL":
            filtered = filtered[filtered["league"] == "AL"]
        elif league == "NL":
            filtered = filtered[filtered["league"] == "NL"]

    # Apply sort
    sort_map = {
        "&#9670; Score": "rank",
        "Offense": "offense_score",
        "Pitching": "rotation_score",
        "Defense": "defense_score",
        "ELO": "composite_elo",
    }
    sort_col = sort_map.get(sort_by, "rank")
    ascending = sort_col == "rank"
    filtered = filtered.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # ---- Render variant ----
    if variant == "Standings":
        _render_standings(filtered, standings, "tr_expanded", preseason_ranks)
    elif variant == "Tiered":
        _render_tiered(filtered, standings)
    else:
        _render_division(rankings, standings)

    # ---- Methodology footer ----
    methodology = (
        '<div class="rk-methodology">'
        '<span class="hdr">Methodology &middot;</span>'
        '&#9670; Score is a 0-10 composite of weighted offense, pitching, and defense '
        'sub-scores normalized vs league average (5.0). ELO is a Bayes-shrunk '
        'baseball-specific rating; week-over-week rank delta and run-differential '
        'carry secondary weight in expected-wins projections.'
        '</div></div>'  # closes rk-page
    )
    st.markdown(methodology, unsafe_allow_html=True)
