"""Player Rankings page — TDD composite rankings for pitchers, batters, and prospects."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from components.metric_cards import metric_card
from components.diamond_rating import diamond_rating_text
from components.headshot import headshot_html
from lib.diamond_rating import score_to_diamonds, diamond_tier
from services.data_loader import (
    load_rankings,
    load_player_teams,
    load_prospect_readiness,
    load_milb_translated,
    load_milb_factors,
)


# ── Tier / score / health color helpers ─────────────────────────────────────

_PROSPECT_TIER_COLORS = {
    "Elite": GOLD,
    "Impact": EMBER,
    "Solid": SAGE,
    "Developing": SLATE,
    "Org Filler": CREAM,
}

_HEALTH_COLORS = {
    "Excellent": SAGE,
    "Good": GOLD,
    "Fair": SLATE,
    "Caution": EMBER,
    "Unknown": CREAM,
}

_READINESS_TIER_COLORS = {
    "Elite": GOLD,
    "Strong": EMBER,
    "Developing": SAGE,
    "Fringe": SLATE,
    "Long Shot": CREAM,
}

_LEVEL_ORDER = ["AAA", "AA", "A+", "A", "ROK"]


def _score_color(val: float) -> str:
    if val >= 4.0:
        return f"color: {GOLD}; font-weight: bold"
    if val >= 3.0:
        return f"color: {SAGE}; font-weight: bold"
    if val >= 2.0:
        return f"color: {SLATE}"
    return f"color: {CREAM}"


def _style_tier(val: str) -> str:
    color = _PROSPECT_TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


def _style_health(val: str) -> str:
    color = _HEALTH_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


# ── CSS ─────────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
.rank-header {{
    text-align: center;
    margin-bottom: 0.8rem;
}}
.rank-title {{
    color: {CREAM};
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}}
.rank-section {{
    color: {GOLD};
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid {DARK_BORDER};
}}
.lb-card {{
    background: {DARK_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 1.5rem;
    max-width: 380px;
}}
.lb-card-wide {{
    max-width: none;
}}
.lb-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {DARK_BORDER};
}}
.lb-title {{
    color: {GOLD};
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
.lb-subtitle {{
    color: {SLATE};
    font-size: 0.70rem;
    font-weight: 400;
    margin-left: 0.4rem;
}}
.lb-scroll {{
    overflow-y: auto;
}}
.lb-scroll::-webkit-scrollbar {{
    width: 6px;
}}
.lb-scroll::-webkit-scrollbar-track {{
    background: transparent;
}}
.lb-scroll::-webkit-scrollbar-thumb {{
    background: rgba(200,169,110,0.3);
    border-radius: 3px;
}}
.lb-row {{
    display: flex;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid {DARK_BORDER}15;
}}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{
    color: {SLATE};
    font-size: 0.82rem;
    min-width: 1.6rem;
    text-align: right;
    margin-right: 0.5rem;
}}
.lb-rank-top {{ color: {GOLD}; font-weight: 700; }}
.lb-headshot {{
    margin-left: 0.5rem;
    margin-right: 0.5rem;
}}
.lb-name {{
    color: {CREAM};
    font-size: 0.95rem;
    font-weight: 600;
    flex: 1;
}}
.lb-name a {{
    color: inherit;
    text-decoration: none;
}}
.lb-name a:hover {{
    color: {GOLD};
    text-decoration: underline;
}}
.lb-info {{
    color: {SLATE};
    font-size: 0.72rem;
    background: rgba(123,143,166,0.12);
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 0.5rem;
}}
.lb-team {{
    color: {SLATE};
    font-size: 0.80rem;
    margin-right: 0.5rem;
}}
.lb-val {{
    display: flex;
    align-items: center;
    min-width: 5rem;
    justify-content: flex-end;
}}
.lb-diamonds {{
    letter-spacing: 1px;
    font-size: 0.7rem;
}}
.lb-rating-num {{
    font-weight: 700;
    font-size: 0.9rem;
    margin-left: 3px;
    min-width: 1.5rem;
    text-align: right;
}}
.lb-stat-cell {{
    margin-left: 0.6rem;
}}
.lb-stat-lbl {{
    color: {SLATE};
    font-size: 0.62rem;
    margin-right: 2px;
}}
.lb-stat-val {{
    color: {CREAM};
    font-size: 0.72rem;
    font-weight: 600;
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


# ── Diamond rating display helpers ──────────────────────────────────────────

def _diamonds_html(rating: float) -> str:
    """Build filled/empty diamond symbols for a 0-5 rating."""
    parts = []
    for i in range(5):
        if i < int(rating) or (i == int(rating) and rating - int(rating) >= 0.5):
            parts.append(f'<span style="color:{GOLD}">&#9670;</span>')
        else:
            parts.append(f'<span style="color:{SLATE}; opacity:0.35">&#9671;</span>')
    return "".join(parts)


def _rating_val_html(score: float) -> str:
    """Format diamond rating as diamond symbols + numeric value for card rows."""
    rating = score_to_diamonds(score)
    color = GOLD if rating >= 4.0 else SAGE if rating >= 3.0 else SLATE
    return (
        f'<span class="lb-diamonds">{_diamonds_html(rating)}</span>'
        f'<span class="lb-rating-num" style="color:{color}">{rating:.1f}</span>'
    )


# ── AL / NL team mapping ──────────────────────────────────────────────────

_AL_TEAMS = {"BAL", "BOS", "NYY", "TB", "TOR",
             "CLE", "CWS", "DET", "KC", "MIN",
             "HOU", "LAA", "OAK", "SEA", "TEX"}
_NL_TEAMS = {"ATL", "MIA", "NYM", "PHI", "WSH",
             "CHC", "CIN", "MIL", "PIT", "STL",
             "ARI", "COL", "LAD", "SD", "SF"}


# ── Detail stat configs for overall cards ──────────────────────────────────
# (label, column_name, format)

BATTER_DETAIL_STATS: list[tuple[str, str, str]] = [
    ("wRC+", "wrc_plus", "int"),
    ("wOBA", "woba", ".000"),
    ("xwOBA", "xwoba", ".000"),
    ("Brl%", "barrel_pct", "pct"),
    ("HH%", "hard_hit_pct", "pct"),
    ("Off", "offense_score", "dec3"),
    ("Fld", "fielding_combined", "dec3"),
]

PITCHER_DETAIL_STATS: list[tuple[str, str, str]] = [
    ("K%", "k_pct", "pct"),
    ("BB%", "bb_pct", "pct"),
    ("SwStr%", "swstr_pct", "pct"),
    ("CSW%", "csw_pct", "pct"),
    ("ERA", "observed_era", "0.00"),
    ("FIP", "observed_fip", "0.00"),
    ("Stuff", "stuff_score", "dec3"),
]


def _fmt_detail(val, fmt: str) -> str:
    if pd.isna(val):
        return "--"
    if fmt == "int":
        return str(int(round(val)))
    if fmt == ".000":
        s = f"{val:.3f}"
        return s.lstrip("0") if abs(val) < 1.0 else s
    if fmt == "0.00":
        return f"{val:.2f}"
    if fmt == "pct":
        return f"{val:.1%}"
    if fmt == "dec3":
        return f"{val:.3f}"
    return str(val)


# ── Generic ranking card renderer ──────────────────────────────────────────

def _render_ranking_card(
    df: pd.DataFrame,
    title: str,
    rank_col: str,
    name_col: str,
    id_col: str,
    score_col: str,
    teams_lookup: dict[int, str],
    *,
    info_col: str | None = None,
    max_height: int = 0,
    n_headshots: int = 5,
    detail_stats: list[tuple[str, str, str]] | None = None,
    wide: bool = False,
    link_type: str = "",
) -> None:
    """Render a scrollable ranking leaderboard card.

    detail_stats: list of (label, column, format) to show as a sub-row.
    wide: if True, card spans full width (no max-width).
    link_type: "hitter" or "pitcher" to make names link to player profile.
    """
    if df.empty:
        return

    work = df.sort_values(rank_col)
    has_detail = detail_stats is not None

    rows_html = []
    for i, (_, row) in enumerate(work.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        rank = int(row[rank_col])
        rank_class = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"

        hs = ""
        if i <= n_headshots:
            hs = f'<span class="lb-headshot">{headshot_html(pid, size=50)}</span>'

        team = teams_lookup.get(pid, "")
        team_html = f'<span class="lb-team">{team}</span>' if team else ""

        info_html = ""
        if info_col and info_col in row.index and pd.notna(row[info_col]):
            info_html = f'<span class="lb-info">{row[info_col]}</span>'

        # Clickable name linking to player profile
        if link_type:
            profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
            name_html = f'<a href="{profile_url}">{name}</a>'
        else:
            name_html = name

        val_html = _rating_val_html(row[score_col])

        # Inline stats (if detail_stats provided)
        stat_inline = ""
        if has_detail:
            for label, col_name, fmt in detail_stats:
                if col_name in row.index:
                    val = _fmt_detail(row[col_name], fmt)
                    stat_inline += (
                        f'<span class="lb-stat-cell">'
                        f'<span class="lb-stat-lbl">{label}</span>'
                        f'<span class="lb-stat-val">{val}</span>'
                        f'</span>'
                    )

        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_class}">{rank}.</span>'
            f'{hs}'
            f'<span class="lb-name">{name_html}</span>'
            f'{info_html}'
            f'{team_html}'
            f'<span class="lb-val">{val_html}</span>'
            f'{stat_inline}'
            f'</div>'
        )

    count_html = f'<span class="lb-subtitle">{len(work)}</span>'
    scroll_style = f' style="max-height:{max_height}px;"' if max_height > 0 else ""
    card_class = "lb-card lb-card-wide" if wide else "lb-card"

    html = (
        f'<div class="{card_class}">'
        f'<div class="lb-title-row">'
        f'<span class="lb-title">{title}{count_html}</span>'
        f'</div>'
        f'<div class="lb-scroll"{scroll_style}>'
        + "".join(rows_html)
        + '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Batter Rankings (leaderboard cards) ────────────────────────────────────

def _render_batter_rankings(df: pd.DataFrame, teams_lookup: dict[int, str]) -> None:
    """Render batter rankings as leaderboard cards with positional breakdowns."""
    search = st.text_input("Search", placeholder="Search player...", key="rank_h_search")
    if search:
        df = df[df["batter_name"].str.contains(search, case=False, na=False)]

    if df.empty:
        st.info("No matching batters found.")
        return

    # Overall rankings (with detail stats, full width)
    _render_ranking_card(
        df, "Overall", "overall_rank", "batter_name", "batter_id",
        "tdd_value_score", teams_lookup,
        info_col="position", max_height=500, n_headshots=10,
        detail_stats=BATTER_DETAIL_STATS, wide=True, link_type="hitter",
    )

    # AL / NL split
    st.markdown('<div class="rank-section">By League</div>', unsafe_allow_html=True)
    al_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _AL_TEAMS}
    nl_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _NL_TEAMS}

    lg_cols = st.columns(2)
    with lg_cols[0]:
        al_df = df[df["batter_id"].isin(al_ids)]
        _render_ranking_card(
            al_df, "American League", "overall_rank", "batter_name", "batter_id",
            "tdd_value_score", teams_lookup,
            info_col="position", max_height=450, n_headshots=5,
            link_type="hitter",
        )
    with lg_cols[1]:
        nl_df = df[df["batter_id"].isin(nl_ids)]
        _render_ranking_card(
            nl_df, "National League", "overall_rank", "batter_name", "batter_id",
            "tdd_value_score", teams_lookup,
            info_col="position", max_height=450, n_headshots=5,
            link_type="hitter",
        )

    # Positional rankings
    st.markdown('<div class="rank-section">By Position</div>', unsafe_allow_html=True)

    positions = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH"]
    for i in range(0, len(positions), 3):
        batch = positions[i:i + 3]
        cols = st.columns(3)
        for col_st, pos in zip(cols, batch):
            with col_st:
                pos_df = df[df["position"] == pos].copy()
                _render_ranking_card(
                    pos_df, pos, "pos_rank", "batter_name", "batter_id",
                    "tdd_value_score", teams_lookup,
                    max_height=400, n_headshots=3, link_type="hitter",
                )


# ── Pitcher Rankings (leaderboard cards) ───────────────────────────────────

def _render_pitcher_rankings(df: pd.DataFrame, teams_lookup: dict[int, str]) -> None:
    """Render pitcher rankings as leaderboard cards with role breakdowns."""
    search = st.text_input("Search", placeholder="Search player...", key="rank_p_search")
    if search:
        df = df[df["pitcher_name"].str.contains(search, case=False, na=False)]

    if df.empty:
        st.info("No matching pitchers found.")
        return

    # Overall rankings (with detail stats, full width)
    _render_ranking_card(
        df, "Overall", "overall_rank", "pitcher_name", "pitcher_id",
        "tdd_value_score", teams_lookup,
        info_col="role", max_height=500, n_headshots=10,
        detail_stats=PITCHER_DETAIL_STATS, wide=True, link_type="pitcher",
    )

    # AL / NL split
    st.markdown('<div class="rank-section">By League</div>', unsafe_allow_html=True)
    al_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _AL_TEAMS}
    nl_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _NL_TEAMS}

    lg_cols = st.columns(2)
    with lg_cols[0]:
        al_df = df[df["pitcher_id"].isin(al_ids)]
        _render_ranking_card(
            al_df, "American League", "overall_rank", "pitcher_name", "pitcher_id",
            "tdd_value_score", teams_lookup,
            info_col="role", max_height=450, n_headshots=5,
            link_type="pitcher",
        )
    with lg_cols[1]:
        nl_df = df[df["pitcher_id"].isin(nl_ids)]
        _render_ranking_card(
            nl_df, "National League", "overall_rank", "pitcher_name", "pitcher_id",
            "tdd_value_score", teams_lookup,
            info_col="role", max_height=450, n_headshots=5,
            link_type="pitcher",
        )

    # Role rankings
    st.markdown('<div class="rank-section">By Role</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for col_st, role in zip(cols, ["SP", "RP"]):
        with col_st:
            role_df = df[df["role"] == role].copy()
            _render_ranking_card(
                role_df, role, "role_rank", "pitcher_name", "pitcher_id",
                "tdd_value_score", teams_lookup,
                max_height=500, n_headshots=5, link_type="pitcher",
            )


# ── Prospect Rankings ────────────────────────────────────────────────────────

def _render_prospect_rankings(df: pd.DataFrame) -> None:
    """Render prospect rankings table with filters."""
    col_tier, col_pos, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tiers = ["All", "Elite", "Impact", "Solid", "Developing", "Org Filler"]
        tier_filter = st.selectbox("Tier", tiers, key="rank_pr_tier")
    with col_pos:
        pos_groups = ["All"] + sorted(df["pos_group"].dropna().unique().tolist()) if "pos_group" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", pos_groups, key="rank_pr_pos")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_pr_level")
    with col_search:
        search = st.text_input("Search prospect", key="rank_pr_search")

    filtered = df.copy()
    if tier_filter != "All" and "tdd_tier" in filtered.columns:
        filtered = filtered[filtered["tdd_tier"] == tier_filter]
    if pos_filter != "All" and "pos_group" in filtered.columns:
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["tdd_tier"] == "Elite").sum() if "tdd_tier" in filtered.columns else 0
        n_impact = (filtered["tdd_tier"] == "Impact").sum() if "tdd_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Impact", f"{n_elite + n_impact}"), unsafe_allow_html=True)
    with cols[2]:
        n_fg = filtered["fg_overall_rank"].notna().sum() if "fg_overall_rank" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_fg)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["tdd_prospect_score"].mean() if "tdd_prospect_score" in filtered.columns and len(filtered) > 0 else 0
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f} / 5"), unsafe_allow_html=True)

    # Compute diamond rating column
    filtered = filtered.copy()
    if "tdd_prospect_score" in filtered.columns:
        filtered["_diamond_rating"] = filtered["tdd_prospect_score"].apply(score_to_diamonds)

    display_map = {
        "tdd_rank": "#",
        "name": "Player",
        "primary_position": "Pos",
        "max_level": "Level",
        "min_age": "Age",
        "_diamond_rating": "Rating",
        "tdd_tier": "Tier",
        "comp_readiness": "Readiness",
        "comp_rate_quality": "Rate Qual",
        "comp_age": "Age Score",
        "comp_trajectory": "Trajectory",
        "comp_positional": "Pos Scarcity",
        "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%",
        "wtd_iso": "ISO",
        "youngest_age_rel": "Age vs Lvl",
        "career_milb_pa": "MiLB PA",
        "fg_future_value": "FG FV",
        "fg_overall_rank": "FG Rank",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    sort_col = "tdd_rank" if "tdd_rank" in available else "_diamond_rating"
    ascending = sort_col == "tdd_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Rating", "{:.1f}"), ("Readiness", "{:.3f}"), ("Rate Qual", "{:.3f}"),
        ("Age Score", "{:.3f}"), ("Trajectory", "{:.3f}"), ("Pos Scarcity", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("ISO", "{:.3f}"),
        ("Age vs Lvl", "{:+.1f}"), ("MiLB PA", "{:,.0f}"),
        ("#", "{:.0f}"), ("Age", "{:.0f}"),
        ("FG FV", "{:.0f}"), ("FG Rank", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Rating" in display_df.columns:
        styler = styler.map(_score_color, subset=["Rating"])
    if "Tier" in display_df.columns:
        styler = styler.map(_style_tier, subset=["Tier"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Rating** = Diamond Rating (0-5) from weighted composite of "
        "Rate Quality (30%), Readiness (25%), Age-Relative (15%), "
        "Trajectory (15%), Positional Scarcity (15%). "
        "**K%/BB%/ISO** are MLB-translated MiLB stats. "
        "**FG FV/Rank** are FanGraphs reference values (not used in TDD scoring)."
    )


# ── Pitching Prospect Rankings ───────────────────────────────────────────────

def _render_pitching_prospect_rankings(df: pd.DataFrame) -> None:
    """Render pitching prospect rankings table with filters."""
    col_tier, col_role, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tiers = ["All", "Elite", "Impact", "Solid", "Developing", "Org Filler"]
        tier_filter = st.selectbox("Tier", tiers, key="rank_pp_tier")
    with col_role:
        roles = ["All", "SP", "RP"]
        role_filter = st.selectbox("Role", roles, key="rank_pp_role")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_pp_level")
    with col_search:
        search = st.text_input("Search prospect", key="rank_pp_search")

    filtered = df.copy()
    if tier_filter != "All" and "tdd_tier" in filtered.columns:
        filtered = filtered[filtered["tdd_tier"] == tier_filter]
    if role_filter != "All" and "pitcher_role" in filtered.columns:
        filtered = filtered[filtered["pitcher_role"] == role_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects Ranked", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["tdd_tier"] == "Elite").sum() if "tdd_tier" in filtered.columns else 0
        n_impact = (filtered["tdd_tier"] == "Impact").sum() if "tdd_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Impact", f"{n_elite + n_impact}"), unsafe_allow_html=True)
    with cols[2]:
        n_fg = filtered["fg_overall_rank"].notna().sum() if "fg_overall_rank" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_fg)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["tdd_prospect_score"].mean() if "tdd_prospect_score" in filtered.columns and len(filtered) > 0 else 0
        avg_diamonds = score_to_diamonds(avg_score)
        st.markdown(metric_card("Avg Rating", f"{avg_diamonds:.1f} / 5"), unsafe_allow_html=True)

    # Compute diamond rating column
    filtered = filtered.copy()
    if "tdd_prospect_score" in filtered.columns:
        filtered["_diamond_rating"] = filtered["tdd_prospect_score"].apply(score_to_diamonds)

    display_map = {
        "tdd_rank": "#",
        "name": "Player",
        "pitcher_role": "Role",
        "max_level": "Level",
        "min_age": "Age",
        "_diamond_rating": "Rating",
        "tdd_tier": "Tier",
        "comp_readiness": "Readiness",
        "comp_rate_quality": "Rate Qual",
        "comp_age": "Age Score",
        "comp_trajectory": "Trajectory",
        "comp_positional": "Pos Scarcity",
        "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%",
        "wtd_hr_bf": "HR/BF",
        "youngest_age_rel": "Age vs Lvl",
        "career_milb_bf": "MiLB BF",
        "sp_pct": "SP%",
        "fg_future_value": "FG FV",
        "fg_overall_rank": "FG Rank",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    sort_col = "tdd_rank" if "tdd_rank" in available else "_diamond_rating"
    ascending = sort_col == "tdd_rank"
    display_df = display_df.sort_values(sort_col, ascending=ascending)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Rating", "{:.1f}"), ("Readiness", "{:.3f}"), ("Rate Qual", "{:.3f}"),
        ("Age Score", "{:.3f}"), ("Trajectory", "{:.3f}"), ("Pos Scarcity", "{:.3f}"),
        ("K%", "{:.1%}"), ("BB%", "{:.1%}"), ("HR/BF", "{:.4f}"),
        ("Age vs Lvl", "{:+.1f}"), ("MiLB BF", "{:,.0f}"), ("SP%", "{:.0%}"),
        ("#", "{:.0f}"), ("Age", "{:.0f}"),
        ("FG FV", "{:.0f}"), ("FG Rank", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Rating" in display_df.columns:
        styler = styler.map(_score_color, subset=["Rating"])
    if "Tier" in display_df.columns:
        styler = styler.map(_style_tier, subset=["Tier"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Rating** = Diamond Rating (0-5) from weighted composite of "
        "Rate Quality (30%), Readiness (25%), Age-Relative (15%), "
        "Trajectory (15%), Positional Scarcity (15%). "
        "**K%/BB%/HR/BF** are MLB-translated MiLB stats. "
        "**SP%** = share of appearances as a starter. "
        "**FG FV/Rank** are FanGraphs reference values (not used in TDD scoring)."
    )


# ── Prospect Readiness ─────────────────────────────────────────────────────

def _style_readiness_tier(val: str) -> str:
    color = _READINESS_TIER_COLORS.get(val, CREAM)
    return f"color: {color}; font-weight: bold"


def _render_prospect_readiness(df: pd.DataFrame) -> None:
    """Render prospect readiness scores with filters."""
    col_tier, col_pos, col_level, col_search = st.columns([1, 1, 1, 2])

    with col_tier:
        tier_options = ["All", "Elite", "Strong", "Developing", "Fringe"]
        tier_filter = st.selectbox("Readiness Tier", tier_options, key="rank_rd_tier")
    with col_pos:
        pos_groups = ["All"] + sorted(df["pos_group"].dropna().unique().tolist()) if "pos_group" in df.columns else ["All"]
        pos_filter = st.selectbox("Position", pos_groups, key="rank_rd_pos")
    with col_level:
        levels = [lv for lv in _LEVEL_ORDER if lv in df["max_level"].unique()] if "max_level" in df.columns else []
        level_filter = st.selectbox("Highest Level", ["All"] + levels, key="rank_rd_level")
    with col_search:
        search = st.text_input("Search player", key="rank_rd_search")

    filtered = df.copy()
    if tier_filter != "All":
        filtered = filtered[filtered["readiness_tier"] == tier_filter]
    if pos_filter != "All" and "pos_group" in filtered.columns:
        filtered = filtered[filtered["pos_group"] == pos_filter]
    if level_filter != "All" and "max_level" in filtered.columns:
        filtered = filtered[filtered["max_level"] == level_filter]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Prospects", f"{len(filtered):,}"), unsafe_allow_html=True)
    with cols[1]:
        n_elite = (filtered["readiness_tier"] == "Elite").sum() if "readiness_tier" in filtered.columns else 0
        n_strong = (filtered["readiness_tier"] == "Strong").sum() if "readiness_tier" in filtered.columns else 0
        st.markdown(metric_card("Elite + Strong", f"{n_elite + n_strong}"), unsafe_allow_html=True)
    with cols[2]:
        n_ranked = filtered["is_ranked"].sum() if "is_ranked" in filtered.columns else 0
        st.markdown(metric_card("FG Ranked", f"{int(n_ranked)}"), unsafe_allow_html=True)
    with cols[3]:
        avg_score = filtered["readiness_score"].mean() if len(filtered) > 0 else 0
        st.markdown(metric_card("Avg Readiness", f"{avg_score:.1%}"), unsafe_allow_html=True)

    display_map = {
        "readiness_score": "Score",
        "readiness_tier": "Tier",
        "name": "Player",
        "pos_group": "Pos",
        "max_level": "Level",
        "wtd_k_pct": "K%",
        "wtd_bb_pct": "BB%",
        "wtd_iso": "ISO",
        "sb_rate": "SB Rate",
        "youngest_age_rel": "Age vs Lvl",
        "min_age": "Age",
        "n_above": "Blocked By",
        "career_milb_pa": "MiLB PA",
    }

    available = [c for c in display_map if c in filtered.columns]
    display_df = filtered[available].copy()
    display_df = display_df.sort_values("readiness_score", ascending=False)
    display_df.columns = [display_map[c] for c in available]

    fmt: dict[str, str] = {}
    for col, f in [
        ("Score", "{:.3f}"), ("K%", "{:.1%}"), ("BB%", "{:.1%}"),
        ("ISO", "{:.3f}"), ("SB Rate", "{:.3f}"), ("Age vs Lvl", "{:+.1f}"),
        ("Blocked By", "{:.0f}"), ("MiLB PA", "{:,.0f}"), ("Age", "{:.0f}"),
    ]:
        if col in display_df.columns:
            fmt[col] = f

    styler = display_df.style.format(fmt, na_rep="—")
    if "Tier" in display_df.columns:
        styler = styler.map(_style_readiness_tier, subset=["Tier"])

    st.dataframe(styler, width='stretch', hide_index=True, height=600)

    st.caption(
        "**Readiness Score** = probability of sticking in MLB (200+ PA season), "
        "combining translated MiLB stats and organizational depth analysis. "
        "**K%/BB%/ISO** are MLB-translated stats from MiLB performance. "
        "**Blocked By** = prospects at same position ahead in the org pipeline."
    )

    with st.expander("Translation Factor Reference"):
        ptype = st.radio(
            "Type", ["Batters", "Pitchers"],
            horizontal=True, key="rank_factor_type",
        )
        factor_type = "batter" if ptype == "Batters" else "pitcher"
        factors = load_milb_factors(factor_type)

        if not factors.empty:
            st.caption(
                "Translation factors convert MiLB stats to MLB equivalents. "
                "Factor > 1.0 means the MiLB stat inflates relative to MLB."
            )
            pooled = factors[factors["pooled"] == True] if "pooled" in factors.columns else factors  # noqa: E712
            factor_cols = {
                "level": "Level", "stat": "Stat", "factor": "Factor",
                "n": "Sample Size", "p25": "P25", "p75": "P75",
            }
            f_avail = [c for c in factor_cols if c in pooled.columns]
            f_display = pooled[f_avail].copy()
            f_display.columns = [factor_cols[c] for c in f_avail]
            if "Level" in f_display.columns:
                level_cat = pd.CategoricalDtype(categories=_LEVEL_ORDER, ordered=True)
                f_display["Level"] = f_display["Level"].astype(level_cat)
                f_display = f_display.sort_values(["Level", "Stat"])
            st.dataframe(
                f_display.style.format({
                    "Factor": "{:.3f}", "P25": "{:.3f}", "P75": "{:.3f}",
                    "Sample Size": "{:,.0f}",
                }, na_rep="—"),
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("Translation factors not available.")


# ── Main page ────────────────────────────────────────────────────────────────

def page_player_rankings() -> None:
    """Render the Player Rankings page."""
    st.markdown(_CSS, unsafe_allow_html=True)

    # Title
    st.markdown(
        '<div class="rank-header">'
        '<div class="rank-title">PLAYER RANKINGS</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    category = st.selectbox(
        "Category",
        ["Batters", "Pitchers", "Hitting Prospects", "Pitching Prospects", "Prospect Readiness"],
        key="rankings_category",
        label_visibility="collapsed",
    )

    # Team lookup for MLB sections
    teams_df = load_player_teams()
    teams_lookup: dict[int, str] = {}
    if not teams_df.empty:
        teams_lookup = dict(zip(
            teams_df["player_id"].astype(int), teams_df["team_abbr"]
        ))

    if category == "Batters":
        df = load_rankings("hitters")
        if df.empty:
            st.warning("No batter rankings data found. Run precompute first.")
            return
        _render_batter_rankings(df, teams_lookup)

    elif category == "Pitchers":
        df = load_rankings("pitchers")
        if df.empty:
            st.warning("No pitcher rankings data found. Run precompute first.")
            return
        _render_pitcher_rankings(df, teams_lookup)

    elif category == "Hitting Prospects":
        df = load_rankings("prospect")
        if df.empty:
            st.warning("No prospect rankings data found. Run precompute first.")
            return
        _render_prospect_rankings(df)

    elif category == "Pitching Prospects":
        df = load_rankings("pitching_prospect")
        if df.empty:
            st.warning("No pitching prospect rankings data found. Run precompute first.")
            return
        _render_pitching_prospect_rankings(df)

    else:  # Prospect Readiness
        df = load_prospect_readiness()
        if df.empty:
            st.warning("No prospect readiness data found. Run precompute first.")
            return
        _render_prospect_readiness(df)
