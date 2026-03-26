"""Player Rankings page — TDD composite rankings for pitchers, batters, and prospects."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER
from components.metric_cards import metric_card
from components.diamond_rating import diamond_rating_text
from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.headshot import headshot_html
from lib.diamond_rating import score_to_diamonds, diamond_tier
from services.data_loader import (
    load_rankings,
    load_player_teams,
    load_prospect_readiness,
    load_milb_translated,
    load_milb_factors,
    load_hitter_archetypes,
    load_pitcher_archetypes,
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

_CSS = """
<style>
.rank-header {
    text-align: center;
    margin-bottom: 0.8rem;
}
.rank-title {
    color: var(--tdd-cream);
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.rank-section {
    color: var(--tdd-gold);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--tdd-dark-border);
}
.lb-title-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--tdd-dark-border);
    background: var(--tdd-dark);
    position: sticky;
    top: 0;
    z-index: 10;
}
.lb-title {
    color: var(--tdd-gold);
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.lb-subtitle {
    color: var(--tdd-slate);
    font-size: 0.70rem;
    font-weight: 400;
    margin-left: 0.4rem;
}
.lb-scroll {
    overflow: visible;
}
.lb-scroll[style*="max-height"] {
    overflow-y: auto;
}
.lb-scroll::-webkit-scrollbar {
    width: 6px;
}
.lb-scroll::-webkit-scrollbar-track {
    background: transparent;
}
.lb-scroll::-webkit-scrollbar-thumb {
    background: rgba(200,169,110,0.3);
    border-radius: 3px;
}
.lb-row {
    display: flex;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--tdd-dark-border-faint);
}
.lb-row:last-child { border-bottom: none; }
.lb-rank {
    color: var(--tdd-slate);
    font-size: 0.82rem;
    min-width: 1.6rem;
    text-align: right;
    margin-right: 0.5rem;
}
.lb-rank-top { color: var(--tdd-gold); font-weight: 700; }
.lb-headshot {
    margin-left: 0.5rem;
    margin-right: 0.5rem;
}
.lb-name {
    color: var(--tdd-cream);
    font-size: 0.95rem;
    font-weight: 600;
    flex: 1;
}
.lb-name a {
    color: inherit;
    text-decoration: none;
}
.lb-name a:hover {
    color: var(--tdd-gold);
    text-decoration: underline;
}
.lb-info {
    color: var(--tdd-slate);
    font-size: 0.72rem;
    background: rgba(123,143,166,0.12);
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 0.5rem;
}
.lb-team {
    color: var(--tdd-slate);
    font-size: 0.80rem;
    margin-right: 0.5rem;
}
.lb-val {
    display: flex;
    align-items: center;
    min-width: 5rem;
    justify-content: flex-end;
}
.lb-diamonds {
    letter-spacing: 1px;
    font-size: 0.7rem;
}
.lb-rating-num {
    font-weight: 700;
    font-size: 0.9rem;
    margin-left: 3px;
    min-width: 1.5rem;
    text-align: right;
}
.lb-stat-cell {
    margin-left: 0.6rem;
}
.lb-stat-lbl {
    color: var(--tdd-slate);
    font-size: 0.62rem;
    margin-right: 2px;
}
.lb-stat-val {
    color: var(--tdd-cream);
    font-size: 0.72rem;
    font-weight: 600;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}
.stSelectbox div[data-baseweb="select"] {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: var(--tdd-gold) !important;
    cursor: pointer !important;
}
.stSelectbox div[data-baseweb="select"] > div:hover {
    background-color: transparent !important;
}

/* ── hover tooltip for compact cards ──────────────── */
.lb-row-tip {
    position: relative;
}
.lb-row-tip .lb-tooltip {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--tdd-dark-card);
    border: 1px solid var(--tdd-dark-border);
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    white-space: nowrap;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    pointer-events: none;
}
.lb-row-tip:hover .lb-tooltip {
    display: flex;
    gap: 0.7rem;
}
.lb-tip-stat {{
    text-align: center;
}}
.lb-tip-val {{
    color: var(--tdd-cream);
    font-size: 0.85rem;
    font-weight: 700;
}}
.lb-tip-lbl {{
    color: var(--tdd-slate);
    font-size: 0.62rem;
}}
/* First-row tooltip: show below instead of above */
.lb-tooltip-below {{
    bottom: auto !important;
    top: 100% !important;
}}
</style>
"""


# ── Diamond rating display helpers ──────────────────────────────────────────

def _diamonds_html(rating: float, fill_color: str = "var(--tdd-gold)") -> str:
    """Build filled/empty diamond symbols for a 0-10 rating."""
    parts = []
    for i in range(10):
        if i < int(rating) or (i == int(rating) and rating - int(rating) >= 0.5):
            parts.append(f'<span style="color:{fill_color}">&#9670;</span>')
        else:
            parts.append(f'<span style="color:var(--tdd-slate); opacity:0.35">&#9671;</span>')
    return "".join(parts)


def _rating_val_html(
    score: float,
    precomputed_rating: float | None = None,
    projected: bool = False,
) -> str:
    """Format diamond rating as diamond symbols + numeric value for card rows.

    Uses pre-computed diamond_rating from scouting grades when available,
    falls back to converting the score via score_to_diamonds.

    When *projected* is True, diamonds render in SAGE (teal-green) instead
    of gold to visually distinguish the 2-3 year projected view.
    """
    rating = precomputed_rating if precomputed_rating is not None else score_to_diamonds(score)
    if projected:
        fill = SAGE
        num_color = SAGE
    else:
        fill = "var(--tdd-gold)"
        num_color = GOLD if rating >= 7.5 else SAGE if rating >= 5.0 else SLATE
    return (
        f'<span class="lb-diamonds">{_diamonds_html(rating, fill_color=fill)}</span>'
        f'<span class="lb-rating-num" style="color:{num_color}">{rating:.1f}</span>'
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

# --- Current Value mode (production-focused) ---
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

# --- Talent Potential mode (scouting/tools-focused) ---
BATTER_TALENT_STATS: list[tuple[str, str, str]] = [
    ("Contact", "contact_skill", "dec3"),
    ("Decisions", "decision_skill", "dec3"),
    ("Damage", "damage_skill", "dec3"),
    ("Proj K%", "projected_k_rate", "pct"),
    ("Proj BB%", "projected_bb_rate", "pct"),
    ("Traj", "trajectory_score", "dec3"),
    ("Fld", "fielding_combined", "dec3"),
]

PITCHER_TALENT_STATS: list[tuple[str, str, str]] = [
    ("Stuff", "stuff_score", "dec3"),
    ("Cmd", "command_score", "dec3"),
    ("Traj", "trajectory_score", "dec3"),
    ("Proj K%", "projected_k_rate", "pct"),
    ("Proj BB%", "projected_bb_rate", "pct"),
    ("pERA", "projected_era", "0.00"),
    ("Hlth", "health_adj", "dec3"),
]


def _fmt_detail(val, fmt: str) -> str:
    if pd.isna(val):
        return ""
    if fmt == "int":
        return str(int(round(val)))
    if fmt == ".000":
        s = f"{val:.3f}"
        return s.lstrip("0") if abs(val) < 1.0 else s
    if fmt == "0.00":
        return f"{val:.2f}"
    if fmt == "pct":
        return f"{val:.1%}"
    if fmt == "dec1":
        return f"{val:.1f}"
    if fmt == "dec2":
        return f"{val:.2f}"
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
    projected: bool = False,
    info_col: str | None = None,
    max_height: int = 0,
    n_headshots: int = 999,
    detail_stats: list[tuple[str, str, str]] | None = None,
    wide: bool = False,
    link_type: str = "",
    expandable: bool = False,
    archetype_lookup: dict[int, tuple[str, str]] | None = None,
    hover_stats: list[tuple[str, str, str]] | None = None,
) -> None:
    """Render a scrollable ranking leaderboard card.

    detail_stats: list of (label, column, format) to show as a sub-row.
    wide: if True, card spans full width (no max-width).
    link_type: "hitter" or "pitcher" to make names link to player profile.
    expandable: if True, each row is a <details>/<summary> expandable card.
    archetype_lookup: {player_id: (archetype_name, archetype_desc)} for expanded view.
    hover_stats: list of (label, column, format) to show as hover tooltip.
    """
    if df.empty:
        return

    # Sort by the precomputed rank column (ascending = rank 1 first).
    # rank_col includes positional adjustments that raw score sorting misses
    # (SS/C boosted, DH/1B penalized).  Fall back to score sort if rank
    # column is missing.
    if rank_col in df.columns and df[rank_col].notna().any():
        work = df.sort_values(rank_col)
    elif score_col in df.columns and df[score_col].notna().any():
        work = df.sort_values(score_col, ascending=False)
    else:
        work = df.copy()
    has_detail = detail_stats is not None

    rows_html = []
    for i, (_, row) in enumerate(work.iterrows(), 1):
        name = row[name_col]
        pid = int(row[id_col])
        rank = i  # display rank matches sort order
        rank_class = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"

        hs = f'<span class="lb-headshot">{headshot_html(pid, size=50)}</span>'

        team = teams_lookup.get(pid, "")
        team_html = f'<span class="lb-team" data-team="{team}">{team}</span>' if team else ""

        info_html = ""
        if info_col and info_col in row.index and pd.notna(row[info_col]):
            info_html = f'<span class="lb-info">{row[info_col]}</span>'

        # Diamond display — always convert score via score_to_diamonds
        val_html = _rating_val_html(row[score_col], precomputed_rating=None, projected=projected)

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

        # When not expandable, keep clickable name link
        if not expandable and link_type:
            profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
            name_content = f'<a href="{profile_url}">{name}</a>'
        else:
            name_content = name

        # Hover tooltip for compact cards
        tooltip_html = ""
        if hover_stats and not expandable:
            tip_cells: list[str] = []
            for label, col_name, fmt in hover_stats:
                if col_name in row.index:
                    val = _fmt_detail(row[col_name], fmt)
                    tip_cells.append(
                        f'<div class="lb-tip-stat">'
                        f'<div class="lb-tip-val">{val}</div>'
                        f'<div class="lb-tip-lbl">{label}</div>'
                        f'</div>'
                    )
            if tip_cells:
                # First row: show tooltip below to avoid clipping at top
                tip_cls = "lb-tooltip lb-tooltip-below" if i == 1 else "lb-tooltip"
                tooltip_html = f'<div class="{tip_cls}">{"".join(tip_cells)}</div>'

        row_class = "lb-row lb-row-tip" if tooltip_html else "lb-row"
        summary_row = (
            f'<div class="{row_class}">'
            f'<span class="{rank_class}">{rank}.</span>'
            f'{hs}'
            f'<span class="lb-name">{name_content}</span>'
            f'{info_html}'
            f'{team_html}'
            f'<span class="lb-val">{val_html}</span>'
            f'{stat_inline}'
            f'{tooltip_html}'
            f'</div>'
        )

        if expandable:
            detail_parts: list[str] = []

            # Larger headshot + name header
            detail_parts.append(
                f'<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">'
                f'{headshot_html(pid, size=80)}'
                f'<div>'
                f'<div style="color:var(--tdd-cream); font-size:1.1rem; font-weight:700;">{name}</div>'
                f'<div style="color:var(--tdd-slate); font-size:0.8rem;">'
                f'{row.get(info_col, "") if info_col else ""}'
                f'{" · " + team if team else ""}</div>'
                f'</div></div>'
            )

            # Archetype badge
            if archetype_lookup and pid in archetype_lookup:
                arch_name, arch_desc = archetype_lookup[pid]
                detail_parts.append(
                    f'<div style="margin-bottom:0.5rem;">'
                    f'<span style="background:{GOLD}22; color:var(--tdd-gold); border:1px solid {GOLD}44; '
                    f'padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">'
                    f'{arch_name}</span>'
                    f'<span style="color:var(--tdd-slate); font-size:0.78rem; margin-left:0.5rem;">'
                    f'{arch_desc}</span>'
                    f'</div>'
                )

            # Key stats from detail_stats
            if has_detail:
                stat_cells: list[str] = []
                for label, col_name, fmt in detail_stats:
                    if col_name in row.index:
                        val = _fmt_detail(row[col_name], fmt)
                        stat_cells.append(
                            f'<div style="text-align:center; min-width:55px;">'
                            f'<div style="color:var(--tdd-cream); font-size:0.9rem; font-weight:700;">{val}</div>'
                            f'<div style="color:var(--tdd-slate); font-size:0.65rem;">{label}</div>'
                            f'</div>'
                        )
                if stat_cells:
                    detail_parts.append(
                        f'<div style="display:flex; flex-wrap:wrap; gap:0.6rem; '
                        f'margin-bottom:0.5rem;">{"".join(stat_cells)}</div>'
                    )

            # Profile link
            if link_type:
                profile_url = f"?page=player_profile&player_id={pid}&player_type={link_type}"
                detail_parts.append(
                    f'<a href="{profile_url}" style="color:var(--tdd-gold); font-size:0.85rem; '
                    f'font-weight:600; text-decoration:none;">'
                    f'View Full Profile &#8594;</a>'
                )

            rows_html.append(expandable_card_html(summary_row, "".join(detail_parts)))
        else:
            rows_html.append(summary_row)

    count_html = f'<span class="lb-subtitle">{len(work)}</span>'
    card_class = "lb-card lb-card-full" if wide else "lb-card"

    if expandable:
        scroll_style = f' style="max-height:{max_height}px;"' if max_height > 0 else ""
        html = (
            f'<div class="{card_class}">'
            f'<div class="lb-scroll"{scroll_style}>'
            f'<div class="lb-title-row">'
            f'<span class="lb-title">{title}{count_html}</span>'
            f'</div>'
            + "".join(rows_html)
            + '</div></div>'
        )
    else:
        scroll_style = f' style="max-height:{max_height}px;"' if max_height > 0 else ""
        html = (
            f'<div class="{card_class}">'
            f'<div class="lb-scroll"{scroll_style}>'
            f'<div class="lb-title-row">'
            f'<span class="lb-title">{title}{count_html}</span>'
            f'</div>'
            + "".join(rows_html)
            + '</div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ── Batter Rankings (leaderboard cards) ────────────────────────────────────

def _render_batter_rankings(
    df: pd.DataFrame, teams_lookup: dict[int, str],
    league_filter: str = "All", search: str = "",
    value_mode: str = "overall",
) -> None:
    """Render batter rankings as leaderboard cards with positional breakdowns."""
    if search:
        df = df[df["batter_name"].str.contains(search, case=False, na=False)]
    if league_filter == "AL":
        al_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _AL_TEAMS}
        df = df[df["batter_id"].isin(al_ids)]
    elif league_filter == "NL":
        nl_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _NL_TEAMS}
        df = df[df["batter_id"].isin(nl_ids)]

    if df.empty:
        st.info("No matching batters found.")
        return

    # Resolve columns based on value mode
    # "overall" (default) = production-dominant current value ranking
    # "projected" = forward-looking 2-3 year dynasty outlook (talent_upside_score)
    is_projected = value_mode == "projected"
    if is_projected:
        score_col = "talent_upside_score" if "talent_upside_score" in df.columns else "tdd_value_score"
        rank_col = "talent_rank" if "talent_rank" in df.columns else "overall_rank"
        detail_stats = BATTER_TALENT_STATS
    else:
        score_col = "current_value_score" if "current_value_score" in df.columns else "tdd_value_score"
        rank_col = "overall_rank"
        detail_stats = BATTER_DETAIL_STATS

    pos_rank_col = "pos_rank"

    # Re-sort by selected score for non-default mode
    if is_projected:
        df = df.sort_values(score_col, ascending=False).copy()

    # Build archetype lookup
    arch_df = load_hitter_archetypes()
    arch_lookup: dict[int, tuple[str, str]] = {}
    if not arch_df.empty and "archetype_name" in arch_df.columns:
        for _, arow in arch_df.iterrows():
            arch_lookup[int(arow["batter_id"])] = (
                str(arow.get("archetype_name", "")),
                str(arow.get("archetype_desc", "")),
            )

    # Overall rankings — expandable cards
    section_title = "Overall" if not is_projected else "Projected Value (2-3 Year Outlook)"
    _render_ranking_card(
        df, section_title, rank_col, "batter_name", "batter_id",
        score_col, teams_lookup,
        projected=is_projected,
        info_col="position", max_height=600,         detail_stats=detail_stats, wide=True, link_type="hitter",
        expandable=True, archetype_lookup=arch_lookup,
    )

    # Positional breakdowns (with hover tooltips)
    st.markdown('<div class="rank-section">By Position</div>', unsafe_allow_html=True)

    positions = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH"]
    for i in range(0, len(positions), 3):
        batch = positions[i:i + 3]
        cols = st.columns(3)
        for col_st, pos in zip(cols, batch):
            with col_st:
                pos_df = df[df["position"] == pos].copy()
                _render_ranking_card(
                    pos_df, pos, pos_rank_col, "batter_name", "batter_id",
                    score_col, teams_lookup,
                    projected=is_projected,
                    max_height=520,  link_type="hitter",
                    hover_stats=detail_stats,
                )


# ── Pitcher Rankings (leaderboard cards) ───────────────────────────────────

def _render_pitcher_rankings(
    df: pd.DataFrame, teams_lookup: dict[int, str],
    league_filter: str = "All", search: str = "",
    value_mode: str = "overall",
) -> None:
    """Render pitcher rankings as leaderboard cards with role breakdowns."""
    if search:
        df = df[df["pitcher_name"].str.contains(search, case=False, na=False)]
    if league_filter == "AL":
        al_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _AL_TEAMS}
        df = df[df["pitcher_id"].isin(al_ids)]
    elif league_filter == "NL":
        nl_ids = {pid for pid, abbr in teams_lookup.items() if abbr in _NL_TEAMS}
        df = df[df["pitcher_id"].isin(nl_ids)]

    if df.empty:
        st.info("No matching pitchers found.")
        return

    # Resolve columns based on value mode
    # "overall" (default) = production-dominant current value ranking
    # "projected" = forward-looking 2-3 year dynasty outlook (talent_upside_score)
    is_projected = value_mode == "projected"
    if is_projected:
        score_col = "talent_upside_score" if "talent_upside_score" in df.columns else "tdd_value_score"
        rank_col = "talent_rank" if "talent_rank" in df.columns else "role_rank"
        detail_stats = PITCHER_TALENT_STATS
    else:
        score_col = "current_value_score" if "current_value_score" in df.columns else "tdd_value_score"
        rank_col = "role_rank"
        detail_stats = PITCHER_DETAIL_STATS

    # Fallback
    if score_col not in df.columns:
        score_col = "tdd_value_score"

    # Re-sort by selected score for non-default mode
    if is_projected:
        df = df.sort_values(score_col, ascending=False).copy()

    # Build archetype lookup
    arch_df = load_pitcher_archetypes()
    arch_lookup: dict[int, tuple[str, str]] = {}
    if not arch_df.empty and "archetype_name" in arch_df.columns:
        for _, arow in arch_df.iterrows():
            arch_lookup[int(arow["pitcher_id"])] = (
                str(arow.get("archetype_name", "")),
                str(arow.get("archetype_desc", "")),
            )

    # Starting Pitchers — expandable cards
    sp_label = "Starting Pitchers" if not is_projected else "SP — Projected Value (2-3 Year Outlook)"
    sp_df = df[df["role"] == "SP"].copy()
    if not sp_df.empty:
        _render_ranking_card(
            sp_df, sp_label, rank_col, "pitcher_name", "pitcher_id",
            score_col, teams_lookup,
            projected=is_projected,
            max_height=600,             detail_stats=detail_stats, wide=True, link_type="pitcher",
            expandable=True, archetype_lookup=arch_lookup,
        )

    # Relief Pitchers — expandable cards
    rp_label = "Relief Pitchers" if not is_projected else "RP — Projected Value (2-3 Year Outlook)"
    rp_df = df[df["role"] == "RP"].copy()
    if not rp_df.empty:
        _render_ranking_card(
            rp_df, rp_label, rank_col, "pitcher_name", "pitcher_id",
            score_col, teams_lookup,
            projected=is_projected,
            max_height=600,             detail_stats=detail_stats, wide=True, link_type="pitcher",
            expandable=True, archetype_lookup=arch_lookup,
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

    styler = display_df.style.format(fmt, na_rep="")
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

    styler = display_df.style.format(fmt, na_rep="")
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

    styler = display_df.style.format(fmt, na_rep="")
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
                }, na_rep=""),
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("Translation factors not available.")


# ── Main page ────────────────────────────────────────────────────────────────

def page_player_rankings() -> None:
    """Render the Player Rankings page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(EXPANDABLE_CARD_CSS, unsafe_allow_html=True)

    # Inline toolbar: Category | Value Mode | League | Search
    tb1, tb2, tb3, tb4 = st.columns([1.5, 1.5, 0.7, 2.5])
    with tb1:
        category = st.selectbox(
            "Category",
            ["Batters", "Pitchers", "Hitting Prospects", "Pitching Prospects", "Prospect Readiness"],
            key="rankings_category",
            label_visibility="collapsed",
        )
    with tb2:
        _mode_options = ["Overall Rating", "Projected Value"]
        value_mode_label = st.selectbox(
            "Mode", _mode_options,
            key="rankings_value_mode",
            label_visibility="collapsed",
            disabled=category not in ("Batters", "Pitchers"),
        )
    with tb3:
        league_filter = st.selectbox(
            "League", ["All", "AL", "NL"],
            key="rank_league",
            label_visibility="collapsed",
            disabled=category not in ("Batters", "Pitchers"),
        )
    with tb4:
        search = st.text_input(
            "Search", placeholder="Search player...",
            key="rank_search", label_visibility="collapsed",
        )

    # "Overall Rating" = scouting-informed comprehensive ranking (default)
    # "Projected Value" = forward-looking 2-3 year outlook (trajectory, age, production)
    value_mode = "projected" if "Projected" in value_mode_label else "overall"

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
        _render_batter_rankings(df, teams_lookup, league_filter, search, value_mode=value_mode)

    elif category == "Pitchers":
        df = load_rankings("pitchers")
        if df.empty:
            st.warning("No pitcher rankings data found. Run precompute first.")
            return
        _render_pitcher_rankings(df, teams_lookup, league_filter, search, value_mode=value_mode)

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
