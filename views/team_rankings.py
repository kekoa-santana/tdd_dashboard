"""Team Rankings — power rankings, ELO leaderboards, tier groups."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER

from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.radar_chart import radar_chart_html
from components.team_logo import team_logo_html
from lib.diamond_rating import score_to_diamonds
from services.data_loader import (
    load_team_rankings,
    load_team_profiles,
    load_team_elo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier badge colours
# ---------------------------------------------------------------------------
_TIER_COLORS: dict[str, str] = {
    "Elite": GOLD,
    "Contender": SAGE,
    "Competitive": SLATE,
    "Rebuilding": EMBER,
}

# ---------------------------------------------------------------------------
# Page-level CSS  (tr- prefix = team rankings)
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
/* ── header ────────────────────────────────────────── */
.tr-header {{ text-align: center; margin-bottom: 0.8rem; }}
.tr-title  {{ color: var(--tdd-cream); font-size: 1.7rem; font-weight: 800; letter-spacing: 1.5px; }}
.tr-section {{ color: var(--tdd-gold); font-size: 1.1rem; font-weight: 700;
              letter-spacing: 0.5px; margin: 1.2rem 0 0.6rem 0;
              padding-bottom: 0.3rem; border-bottom: 1px solid var(--tdd-dark-border); }}

/* ── summary row (inside <summary>) ───────────────── */
.tr-row {{ display: flex; align-items: center; width: 100%; gap: 0; }}
.tr-rank {{ color: var(--tdd-slate); font-size: 0.82rem; min-width: 1.6rem;
           text-align: right; margin-right: 0.5rem; }}
.tr-rank-top {{ color: var(--tdd-gold); font-weight: 700; }}
.tr-logo {{ margin: 0 0.5rem; flex-shrink: 0; }}
.tr-name {{ color: var(--tdd-cream); font-size: 0.95rem; font-weight: 600; flex: 1;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.tr-tier {{ font-size: 0.72rem; font-weight: 600; padding: 2px 8px;
           border-radius: 10px; margin-right: 0.5rem; white-space: nowrap; }}
.tr-diamonds {{ letter-spacing: 1px; font-size: 0.7rem; }}
.tr-rating-num {{ font-weight: 700; font-size: 0.9rem; margin-left: 3px;
                 min-width: 1.5rem; text-align: right; }}
.tr-val {{ display: flex; align-items: center; min-width: 5rem;
          justify-content: flex-end; margin-right: 0.3rem; }}
.tr-stat {{ color: var(--tdd-slate); font-size: 0.72rem; margin-left: 0.5rem;
           white-space: nowrap; }}
.tr-stat b {{ color: var(--tdd-cream); font-weight: 600; }}

/* ── expanded detail ──────────────────────────────── */
.tr-detail-grid {{ display: flex; gap: 1.2rem; align-items: flex-start; }}
.tr-detail-left  {{ flex-shrink: 0; }}
.tr-detail-right {{ flex: 1; min-width: 0; }}
.tr-view-link {{ display: inline-block; margin-top: 0.6rem; color: var(--tdd-gold);
                font-size: 0.85rem; font-weight: 600; text-decoration: none; }}
.tr-view-link:hover {{ text-decoration: underline; }}
.tr-style-pills {{ margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 4px; }}

/* ── leaderboard cards (reuse lb- naming) ─────────── */
.lb-title-row {{ display: flex; justify-content: space-between; align-items: baseline;
                margin-bottom: 0.5rem; padding-bottom: 0.4rem;
                border-bottom: 1px solid var(--tdd-dark-border); }}
.lb-title {{ color: var(--tdd-gold); font-size: 1.0rem; font-weight: 700; letter-spacing: 0.5px; }}
.lb-scroll {{ overflow-y: auto; }}
.lb-scroll::-webkit-scrollbar {{ width: 6px; }}
.lb-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.lb-scroll::-webkit-scrollbar-thumb {{ background: rgba(200,169,110,0.3); border-radius: 3px; }}
.lb-row {{ display: flex; align-items: center; padding: 0.28rem 0;
          border-bottom: 1px solid var(--tdd-dark-border-faint); }}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{ color: var(--tdd-slate); font-size: 0.82rem; min-width: 1.6rem;
           text-align: right; margin-right: 0.5rem; }}
.lb-rank-top {{ color: var(--tdd-gold); font-weight: 700; }}
.lb-name {{ color: var(--tdd-cream); font-size: 0.90rem; font-weight: 600; flex: 1; }}
.lb-elo-val {{ color: var(--tdd-cream); font-size: 0.85rem; font-weight: 700; min-width: 3rem;
              text-align: right; }}

/* ── tier group grid ──────────────────────────────── */
.tr-tier-card {{ background: transparent; border: none;
                border-bottom: 1px solid var(--tdd-dark-border); border-radius: 0; padding: 0.8rem 0; }}
.tr-tier-title {{ font-size: 1.0rem; font-weight: 700; letter-spacing: 0.5px;
                 margin-bottom: 0.6rem; }}
.tr-logo-grid {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
.tr-logo-item {{ display: flex; flex-direction: column; align-items: center;
                width: 50px; }}
.tr-logo-abbr {{ color: var(--tdd-slate); font-size: 0.65rem; font-weight: 600;
                margin-top: 2px; text-align: center; }}

/* ── selectbox override (match player rankings) ───── */
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: transparent !important; border: none !important;
    box-shadow: none !important; padding-left: 0 !important; }}
.stSelectbox div[data-baseweb="select"] {{
    font-size: 1.2rem !important; font-weight: 800 !important;
    color: var(--tdd-gold) !important; cursor: pointer !important; }}

/* ── responsive ───────────────────────────────────── */
@media (max-width: 768px) {{
    .tr-detail-grid {{ flex-direction: column; }}
    .tr-stat {{ margin-left: 0.3rem; font-size: 0.65rem; }}
}}
@media (max-width: 480px) {{
    .tr-stat {{ display: none; }}
    .tr-name {{ font-size: 0.85rem; }}
}}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helper renderers
# ══════════════════════════════════════════════════════════════════════════════

def _diamonds_html(rating: float) -> str:
    """Build filled/empty diamond symbols for a 0-10 rating."""
    parts = []
    for i in range(10):
        if i < int(rating) or (i == int(rating) and rating - int(rating) >= 0.5):
            parts.append(f'<span style="color:var(--tdd-gold)">&#9670;</span>')
        else:
            parts.append(f'<span style="color:var(--tdd-slate); opacity:0.35">&#9671;</span>')
    return "".join(parts)


def _rating_val_html(composite_score: float) -> str:
    """Diamond symbols + numeric value for a 0-1 composite score."""
    rating = score_to_diamonds(composite_score)
    color = GOLD if rating >= 7.5 else SAGE if rating >= 5.0 else SLATE
    return (
        f'<span class="tr-diamonds">{_diamonds_html(rating)}</span>'
        f'<span class="tr-rating-num" style="color:{color}">{rating:.1f}</span>'
    )


def _tier_badge(tier: str) -> str:
    """Colored tier pill."""
    color = _TIER_COLORS.get(tier, SLATE)
    return (
        f'<span class="tr-tier" '
        f'style="background:{color}22; color:{color}; border:1px solid {color}44;">'
        f'{tier}</span>'
    )


def _pill(label: str, color: str) -> str:
    """Small colored pill for style descriptors."""
    return (
        f'<span style="background:{color}22; color:{color}; '
        f'border:1px solid {color}44; padding:3px 10px; border-radius:12px; '
        f'font-size:0.75rem; font-weight:600;">{label}</span>'
    )


def _elo_bar(label: str, value: float, rank: int | None = None,
             min_v: float = 1350, max_v: float = 1650) -> str:
    """Horizontal ELO bar (mirrors team_overview pattern)."""
    pct = max(0, min(100, (value - min_v) / (max_v - min_v) * 100))
    mid_pct = (1500 - min_v) / (max_v - min_v) * 100
    color = GOLD if value >= 1520 else SAGE if value >= 1490 else EMBER if value < 1470 else SLATE
    rank_str = (
        f'<span style="color:var(--tdd-slate); font-size:0.75rem; margin-left:4px;">#{rank}</span>'
        if rank else ""
    )
    return (
        f'<div style="margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:2px;">'
        f'<span style="color:var(--tdd-cream); font-size:0.8rem; font-weight:600;">{label}</span>'
        f'<span style="color:{color}; font-weight:700; font-size:0.85rem;">{value:.0f}{rank_str}</span>'
        f'</div>'
        f'<div style="position:relative; height:14px; background:var(--tdd-dark-card); border-radius:4px; overflow:visible;">'
        f'<div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:4px;"></div>'
        f'<div style="position:absolute; left:{mid_pct:.1f}%; top:0; height:100%; '
        f'width:1px; background:{SLATE}44;"></div>'
        f'</div></div>'
    )


def _score_gauge(label: str, score: float) -> str:
    """Small 0-1 score gauge bar."""
    pct = max(0, min(100, score * 100))
    color = GOLD if score >= 0.70 else SAGE if score >= 0.45 else EMBER if score < 0.30 else SLATE
    return (
        f'<div style="margin-bottom:6px;">'
        f'<div style="display:flex; justify-content:space-between; margin-bottom:1px;">'
        f'<span style="color:var(--tdd-cream); font-size:0.75rem;">{label}</span>'
        f'<span style="color:{color}; font-weight:600; font-size:0.78rem;">{score:.2f}</span>'
        f'</div>'
        f'<div style="height:6px; background:var(--tdd-dark-card); border-radius:3px;">'
        f'<div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:3px;"></div>'
        f'</div></div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Section renderers
# ══════════════════════════════════════════════════════════════════════════════

def _render_power_rankings(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> None:
    """Section 1: expandable power rankings 1-30."""
    # Merge profile scores not in rankings (org_score, schedule_score)
    extra_cols = ["abbreviation"]
    for col in ("org_score", "schedule_score"):
        if col in profiles.columns and col not in df.columns:
            extra_cols.append(col)
    if len(extra_cols) > 1:
        df = df.merge(
            profiles[extra_cols].drop_duplicates("abbreviation"),
            on="abbreviation", how="left", suffixes=("", "_prof"),
        )

    # Merge ELO ranks
    if not elo_df.empty:
        elo_cols = ["team_id", "offense_rank", "pitching_rank", "composite_rank"]
        elo_cols = [c for c in elo_cols if c in elo_df.columns]
        if elo_cols:
            df = df.merge(elo_df[elo_cols].drop_duplicates("team_id"),
                          on="team_id", how="left", suffixes=("", "_elo"))

    df = df.sort_values("rank").reset_index(drop=True)

    cards_html: list[str] = []
    for _, row in df.iterrows():
        rank = int(row.get("rank", 0))
        tid = int(row.get("team_id", 0))
        name = row.get("team_name", "")
        abbr = row.get("abbreviation", "")
        tier = row.get("tier", "")
        tdd = float(row.get("tdd_score", row.get("composite_score", 0)))
        pyth_w = row.get("projected_wins", None)
        off_elo = row.get("offense_elo", None)
        pit_elo = row.get("pitching_elo", None)

        # ── summary row ──
        rank_cls = "tr-rank-top tr-rank" if rank <= 5 else "tr-rank"
        logo = f'<span class="tr-logo">{team_logo_html(tid, size=36)}</span>'
        tier_html = _tier_badge(tier) if tier else ""
        # Use tdd_score (1-10) directly — no conversion needed
        color = GOLD if tdd >= 7.5 else SAGE if tdd >= 5.0 else SLATE
        rating_html = (
            f'<span class="tr-val">'
            f'<span class="tr-diamonds">{_diamonds_html(tdd)}</span>'
            f'<span class="tr-rating-num" style="color:{color}">{tdd:.1f}</span>'
            f'</span>'
        )

        # Summary stats: projected wins + scouting diamond ratings
        stats_parts = []
        if pyth_w is not None and not np.isnan(pyth_w):
            stats_parts.append(f'<span class="tr-stat">W: <b>{pyth_w:.0f}</b></span>')

        # Use team scouting grades from profiles instead of ELO
        prof_row = profiles[profiles["abbreviation"] == abbr]
        if not prof_row.empty:
            lu_d = prof_row["lineup_diamond"].iloc[0] if "lineup_diamond" in prof_row.columns else None
            rot_d = prof_row["rotation_diamond"].iloc[0] if "rotation_diamond" in prof_row.columns else None
            bp_d = prof_row["bullpen_diamond"].iloc[0] if "bullpen_diamond" in prof_row.columns else None
            if lu_d is not None and not np.isnan(lu_d):
                stats_parts.append(f'<span class="tr-stat">Lineup: <b>{lu_d:.1f}</b></span>')
            if rot_d is not None and not np.isnan(rot_d):
                stats_parts.append(f'<span class="tr-stat">Rotation: <b>{rot_d:.1f}</b></span>')
            if bp_d is not None and not np.isnan(bp_d):
                stats_parts.append(f'<span class="tr-stat">Bullpen: <b>{bp_d:.1f}</b></span>')
        elif off_elo is not None and not np.isnan(off_elo):
            stats_parts.append(f'<span class="tr-stat">Off: <b>{off_elo:.0f}</b></span>')
            if pit_elo is not None and not np.isnan(pit_elo):
                stats_parts.append(f'<span class="tr-stat">Pit: <b>{pit_elo:.0f}</b></span>')
        stats_html = "".join(stats_parts)

        summary = (
            f'<div class="tr-row">'
            f'<span class="{rank_cls}">{rank}.</span>'
            f'{logo}'
            f'<span class="tr-name">{name}</span>'
            f'{tier_html}'
            f'{rating_html}'
            f'{stats_html}'
            f'</div>'
        )

        # ── expanded detail ──
        # Radar chart data
        radar_scores = {
            "Offense": float(row.get("offense_score", 0) or 0),
            "Pitching": float(row.get("pitching_score", 0) or 0),
            "Defense": float(row.get("defense_score", 0) or 0),
            "Org": float(row.get("org_score", 0) or 0),
            "Health": float(row.get("health_depth_score", 0) or 0),
            "Schedule": float(row.get("schedule_score", 0) or 0),
        }
        radar_html = radar_chart_html(radar_scores, size=190)

        # Scouting grade bars (replace ELO)
        grade_parts: list[str] = []
        if not prof_row.empty:
            for label, col in [("Lineup", "lineup_diamond"),
                                ("Rotation", "rotation_diamond"),
                                ("Bullpen", "bullpen_diamond")]:
                val = prof_row[col].iloc[0] if col in prof_row.columns else None
                if val is not None and not np.isnan(val):
                    grade_parts.append(_score_gauge(label, val / 10.0))
        elo_html = "".join(grade_parts)

        # Score gauges
        gauges = ""
        for label, col in [("Offense", "offense_score"), ("Pitching", "pitching_score"),
                            ("Defense", "defense_score"), ("Health/Depth", "health_depth_score")]:
            val = row.get(col, None)
            if val is not None and not np.isnan(val):
                gauges += _score_gauge(label, val)

        # Style pills
        pills: list[str] = []
        for col, label_prefix in [("offense_style", ""), ("pitching_style", ""),
                                    ("age_trajectory", "")]:
            val = row.get(col, None)
            if val and isinstance(val, str) and val != "nan":
                pill_color = GOLD if "Power" in val or "Strikeout" in val or "Ascending" in val else (
                    SAGE if "Balanced" in val or "Stable" in val else SLATE
                )
                pills.append(_pill(val, pill_color))
        pills_html = (
            f'<div class="tr-style-pills">{"".join(pills)}</div>'
            if pills else ""
        )

        # View link
        view_link = (
            f'<a class="tr-view-link" href="?page=team_overview&team={abbr}">'
            f'View Team Overview &#8594;</a>'
        )

        detail = (
            f'<div class="tr-detail-grid">'
            f'<div class="tr-detail-left">{radar_html}</div>'
            f'<div class="tr-detail-right">'
            f'{elo_html}{gauges}{pills_html}{view_link}'
            f'</div></div>'
        )

        cards_html.append(expandable_card_html(summary, detail))

    st.markdown("".join(cards_html), unsafe_allow_html=True)


def _render_elo_leaderboard(
    df: pd.DataFrame,
    title: str,
    elo_col: str,
    rank_col: str,
    max_height: int = 400,
    n_logos: int = 30,
) -> None:
    """Scrollable ELO leaderboard card."""
    sorted_df = df.sort_values(elo_col, ascending=False).reset_index(drop=True)

    rows_html: list[str] = []
    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        tid = int(row.get("team_id", 0))
        abbr = row.get("abbreviation", "")
        elo = row.get(elo_col, 1500)
        rank_cls = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"
        logo = f'<span style="margin:0 0.4rem;">{team_logo_html(tid, size=24)}</span>' if i <= n_logos else ""
        color = GOLD if elo >= 1520 else SAGE if elo >= 1490 else EMBER if elo < 1470 else SLATE
        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_cls}">{i}.</span>'
            f'{logo}'
            f'<span class="lb-name tdd-team-abbr" data-team="{abbr}">{abbr}</span>'
            f'<span class="lb-elo-val" style="color:{color}">{elo:.0f}</span>'
            f'</div>'
        )

    height_style = f' style="max-height:{max_height}px;"' if max_height else ""
    html = (
        f'<div class="lb-card">'
        f'<div class="lb-title-row"><span class="lb-title">{title}</span></div>'
        f'<div class="lb-scroll"{height_style}>{"".join(rows_html)}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_diamond_leaderboard(
    df: pd.DataFrame,
    title: str,
    diamond_col: str,
    max_height: int = 400,
) -> None:
    """Scrollable TDD Diamond Rating leaderboard card for teams."""
    if diamond_col not in df.columns:
        st.info(f"No {title} data available.")
        return

    sorted_df = df.dropna(subset=[diamond_col]).sort_values(
        diamond_col, ascending=False,
    ).reset_index(drop=True)

    rows_html: list[str] = []
    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        tid = int(row.get("team_id", 0))
        abbr = row.get("abbreviation", "")
        rating = float(row.get(diamond_col, 5.0))
        rank_cls = "lb-rank-top lb-rank" if i <= 5 else "lb-rank"
        logo = f'<span style="margin:0 0.4rem;">{team_logo_html(tid, size=24)}</span>'
        color = GOLD if rating >= 6.5 else SAGE if rating >= 5.5 else EMBER if rating < 4.5 else SLATE
        rows_html.append(
            f'<div class="lb-row">'
            f'<span class="{rank_cls}">{i}.</span>'
            f'{logo}'
            f'<span class="lb-name tdd-team-abbr" data-team="{abbr}">{abbr}</span>'
            f'<span class="lb-elo-val" style="color:{color}">{rating:.1f}</span>'
            f'</div>'
        )

    height_style = f' style="max-height:{max_height}px;"' if max_height else ""
    html = (
        f'<div class="lb-card">'
        f'<div class="lb-title-row"><span class="lb-title">{title}</span></div>'
        f'<div class="lb-scroll"{height_style}>{"".join(rows_html)}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_tier_groups(df: pd.DataFrame) -> None:
    """Section 3: 2x2 tier group grid with logo clusters."""
    tiers = ["Elite", "Contender", "Competitive", "Rebuilding"]

    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for col_st, tier in zip(cols, tiers[row_start:row_start + 2]):
            with col_st:
                color = _TIER_COLORS.get(tier, SLATE)
                tier_df = df[df["tier"] == tier].sort_values("rank")

                logos: list[str] = []
                for _, row in tier_df.iterrows():
                    tid = int(row.get("team_id", 0))
                    abbr = row.get("abbreviation", "")
                    logos.append(
                        f'<div class="tr-logo-item">'
                        f'{team_logo_html(tid, size=36)}'
                        f'<span class="tr-logo-abbr" data-team="{abbr}">{abbr}</span>'
                        f'</div>'
                    )

                logos_html = "".join(logos) if logos else (
                    f'<span style="color:var(--tdd-slate); font-size:0.8rem;">None</span>'
                )
                html = (
                    f'<div class="tr-tier-card">'
                    f'<div class="tr-tier-title" style="color:{color};">'
                    f'{tier} <span style="color:var(--tdd-slate); font-size:0.75rem; font-weight:400;">'
                    f'({len(tier_df)})</span></div>'
                    f'<div class="tr-logo-grid">{logos_html}</div>'
                    f'</div>'
                )
                st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main page entry
# ══════════════════════════════════════════════════════════════════════════════

def page_team_rankings() -> None:
    """Render the Team Rankings page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(EXPANDABLE_CARD_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="tr-header"><div class="tr-title">TEAM RANKINGS</div></div>',
        unsafe_allow_html=True,
    )

    # ── data ──
    rankings = load_team_rankings()
    profiles = load_team_profiles()
    elo_df = load_team_elo(preseason=True)

    if rankings.empty:
        st.warning("No team rankings data found. Run precompute first.")
        return

    # ── category filter ──
    from utils.division_filter import division_selectbox, apply_division_filter

    category = division_selectbox(key="tr_category")

    filtered = apply_division_filter(rankings, category)

    search = st.text_input("Search", placeholder="Search team...", key="tr_search")
    if search:
        mask = (
            filtered["team_name"].str.contains(search, case=False, na=False)
            | filtered["abbreviation"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.info("No matching teams found.")
        return

    # ── Section 1: Power Rankings ──
    st.markdown('<div class="tr-section">Power Rankings</div>', unsafe_allow_html=True)
    _render_power_rankings(filtered, profiles, elo_df)

    # ── Section 2: TDD Scouting Grade Leaderboards ──
    st.markdown('<div class="tr-section">Team Scouting Grades</div>', unsafe_allow_html=True)

    # Use team_profiles which has the scouting-grade aggregations
    grade_source = apply_division_filter(profiles, category)

    has_grades = all(c in grade_source.columns for c in ["lineup_diamond", "rotation_diamond", "bullpen_diamond"])
    if has_grades:
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_diamond_leaderboard(grade_source, "Top Lineups", "lineup_diamond")
        with c2:
            _render_diamond_leaderboard(grade_source, "Top Rotations", "rotation_diamond")
        with c3:
            _render_diamond_leaderboard(grade_source, "Top Bullpens", "bullpen_diamond")
    else:
        # Fallback to ELO if scouting grades not yet computed
        elo_source = elo_df if not elo_df.empty else rankings
        elo_source = apply_division_filter(elo_source, category)
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_elo_leaderboard(elo_source, "Offense ELO", "offense_elo", "offense_rank")
        with c2:
            _render_elo_leaderboard(elo_source, "Pitching ELO", "pitching_elo", "pitching_rank")
        with c3:
            _render_elo_leaderboard(elo_source, "Composite ELO", "composite_elo", "composite_rank")

    # ── Section 3: Tier Groups ──
    st.markdown('<div class="tr-section">Tier Groups</div>', unsafe_allow_html=True)
    _render_tier_groups(filtered)
