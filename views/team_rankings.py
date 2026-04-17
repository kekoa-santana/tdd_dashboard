"""Team Rankings — power rankings, ELO leaderboards, tier groups."""
from __future__ import annotations


import pandas as pd
import streamlit as st

from config import GOLD, EMBER, SAGE, SLATE, DARK_BORDER
from utils.alerts import tdd_info, tdd_warn

from components.expandable_card import EXPANDABLE_CARD_CSS, expandable_card_html
from components.team_logo import team_logo_html
from lib.diamond_rating import score_to_diamonds
from services.data_loader import (
    load_team_rankings,
    load_team_profiles,
    load_team_elo,
)

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
              padding-bottom: 0.3rem; border-bottom: 1px solid var(--tdd-dark-border);
              text-align: center; }}

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
.tr-detail-grid {{ display: flex; gap: 1.5rem; align-items: flex-start;
                  margin: 0 auto; }}
.tr-detail-left  {{ flex: 3; min-width: 0; }}
.tr-detail-right {{ flex: 2; min-width: 0; }}
.tr-view-link {{ display: inline-block; margin-top: 0.6rem; color: var(--tdd-gold);
                font-size: 0.85rem; font-weight: 600; text-decoration: none; }}
.tr-view-link:hover {{ text-decoration: underline; }}
.tr-style-pills {{ margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 4px; }}

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


def _render_power_rankings(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    elo_df: pd.DataFrame,
    _standings: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Section 1: expandable power rankings 1-30."""
    if _standings is None:
        _standings = {}
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

    # Pre-load rankings + roster for expanded details
    from services.data_loader import load_rankings, load_roster
    try:
        _hr_all = load_rankings("hitters")
        _pr_all = load_rankings("pitchers")
        _roster_loaded = load_roster()
    except Exception:
        _hr_all = _pr_all = _roster_loaded = None

    cards_html: list[str] = []
    for _, row in df.iterrows():
        rank = int(row.get("rank", 0))
        tid = int(row.get("team_id", 0))
        name = row.get("team_name", "")
        abbr = row.get("abbreviation", "")
        tier = row.get("tier", "")
        tdd = float(row.get("tdd_score", row.get("composite_score", 0)))

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

        # Summary stats: record + composition scores (×10 for 1-10 display)
        stats_parts = []
        _rec = _standings.get(abbr)
        if _rec:
            stats_parts.append(f'<span class="tr-stat"><b>{_rec[0]}-{_rec[1]}</b></span>')

        off_s = float(row.get("offense_score", 0) or 0)
        rot_s = float(row.get("rotation_score", 0) or 0)
        bp_s = float(row.get("bullpen_score", 0) or 0)
        if off_s > 0:
            stats_parts.append(f'<span class="tr-stat">Lineup: <b>{off_s * 10:.1f}</b></span>')
        if rot_s > 0:
            stats_parts.append(f'<span class="tr-stat">Rotation: <b>{rot_s * 10:.1f}</b></span>')
        if bp_s > 0:
            stats_parts.append(f'<span class="tr-stat">Bullpen: <b>{bp_s * 10:.1f}</b></span>')
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
        # Composition bars with league rank (replaces radar chart)
        _comp_dims = [
            ("Offense", "offense_score", "offense_score"),
            ("Rotation", "rotation_score", "rotation_score"),
            ("Bullpen", "bullpen_score", "bullpen_score"),
            ("Fielding", "defense_score", "defense_score"),
            ("Health/Depth", "health_depth_score", "health_depth_score"),
        ]

        def _comp_bar(label: str, score: float, league_rank: int) -> str:
            pct = max(0, min(100, score * 100))
            color = GOLD if score >= 0.70 else SAGE if score >= 0.45 else EMBER if score < 0.30 else SLATE
            rank_color = GOLD if league_rank <= 5 else SAGE if league_rank <= 15 else SLATE
            return (
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:2px;">'
                f'<span style="color:var(--tdd-cream); font-size:0.8rem; font-weight:600;">'
                f'{label} <span style="color:{rank_color}; font-size:0.72rem;">(#{league_rank})</span></span>'
                f'<span style="color:{color}; font-weight:700; font-size:0.85rem;">{score:.2f}</span>'
                f'</div>'
                f'<div style="height:10px; background:var(--tdd-dark-card); border-radius:4px;">'
                f'<div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:4px;"></div>'
                f'</div></div>'
            )

        comp_html = ""
        for dim in _comp_dims:
            label, col = dim[0], dim[1]
            val = float(row.get(col, row.get(dim[2], 0)) or 0)
            # Compute league rank for this dimension
            rank_col = col if col in df.columns else dim[2]
            if rank_col in df.columns:
                lg_rank = int((df[rank_col].rank(ascending=False, method="min")).loc[row.name])
            else:
                lg_rank = 0
            comp_html += _comp_bar(label, val, lg_rank)

        # Style pills
        pills: list[str] = []
        for col in ["offense_style", "pitching_style", "age_trajectory"]:
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

        # Top contributing players (right side)
        players_html = ""
        try:
            _roster = _roster_loaded if _roster_loaded is not None else load_roster()
            _team_pids = set(
                _roster[_roster["team_abbr"] == abbr]["player_id"]
            ) if not _roster.empty else set()

            top_list: list[tuple[str, str, int, float]] = []  # name, pos, rank, score

            if _hr_all is not None and not _hr_all.empty and _team_pids:
                _h_pid = "batter_id" if "batter_id" in _hr_all.columns else "player_id"
                _h_sc = next((c for c in ("current_value_score", "tdd_value_score") if c in _hr_all.columns), None)
                if _h_sc:
                    th = _hr_all[_hr_all[_h_pid].isin(_team_pids)]
                    for _, pr in th.iterrows():
                        top_list.append((
                            pr["batter_name"], pr.get("position", ""),
                            int(pr.get("pos_rank", 0)), float(pr[_h_sc]),
                        ))

            if _pr_all is not None and not _pr_all.empty and _team_pids:
                _p_pid = "pitcher_id" if "pitcher_id" in _pr_all.columns else "player_id"
                _p_sc = next((c for c in ("current_value_score", "tdd_value_score") if c in _pr_all.columns), None)
                if _p_sc:
                    tp_ = _pr_all[_pr_all[_p_pid].isin(_team_pids)]
                    for _, pr in tp_.iterrows():
                        top_list.append((
                            pr["pitcher_name"], pr.get("role", "P"),
                            int(pr.get("role_rank", 0)), float(pr[_p_sc]),
                        ))

            top_list.sort(key=lambda x: x[3], reverse=True)
            for name, pos, lg_rank, sc in top_list[:8]:
                rk_color = GOLD if lg_rank <= 5 else SAGE if lg_rank <= 15 else SLATE
                diamonds = score_to_diamonds(sc)
                players_html += (
                    f'<div style="display:flex; align-items:center; gap:6px; '
                    f'padding:3px 0; border-bottom:1px solid {DARK_BORDER};">'
                    f'<span style="color:{rk_color}; font-weight:700; font-size:0.78rem; '
                    f'min-width:2.2rem; text-align:right;">#{lg_rank}</span>'
                    f'<span style="color:{SLATE}; font-size:0.72rem; min-width:1.8rem;">{pos}</span>'
                    f'<span style="color:var(--tdd-cream); font-size:0.82rem; flex:1;">{name}</span>'
                    f'<span style="color:var(--tdd-gold); font-weight:600; font-size:0.82rem;">'
                    f'{diamonds:.1f}</span>'
                    f'</div>'
                )
        except Exception:
            pass  # graceful degradation

        if players_html:
            players_html = (
                f'<div style="margin-top:4px;">'
                f'<div style="color:var(--tdd-slate); font-size:0.72rem; font-weight:600; '
                f'margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;">Top Players</div>'
                f'{players_html}</div>'
            )

        # View link
        view_link = (
            f'<a class="tr-view-link" href="?page=team_overview&team={abbr}">'
            f'View Team Overview &#8594;</a>'
        )

        detail = (
            f'<div class="tr-detail-grid">'
            f'<div class="tr-detail-left">{comp_html}{pills_html}</div>'
            f'<div class="tr-detail-right">'
            f'{players_html}{view_link}'
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
        tdd_info(f"No {title} data available.")
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
    from services.data_loader import load_standings
    rankings = load_team_rankings()
    profiles = load_team_profiles()
    elo_df = load_team_elo(preseason=True)
    _standings = load_standings()

    if rankings.empty:
        tdd_warn("No team rankings data found. Run precompute first.")
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
        tdd_info("No matching teams found.")
        return

    # ── Section 1: Power Rankings ──
    st.markdown('<div class="tr-section">Power Rankings</div>', unsafe_allow_html=True)

    _render_power_rankings(filtered, profiles, elo_df, _standings)

    # ── Section 2: Top Lineups / Rotations / Bullpens (composition scores) ──
    # Use the same scores that drive the power ranking composition bars (×10)
    lb_source = filtered.copy()
    _score_map = {
        "offense_10": "offense_score",
        "rotation_10": "rotation_score",
        "bullpen_10": "bullpen_score",
    }
    for dest, src in _score_map.items():
        if src in lb_source.columns:
            lb_source[dest] = lb_source[src].fillna(0) * 10

    has_scores = all(dest in lb_source.columns for dest in _score_map)
    if has_scores:
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_diamond_leaderboard(lb_source, "Top Lineups", "offense_10")
        with c2:
            _render_diamond_leaderboard(lb_source, "Top Rotations", "rotation_10")
        with c3:
            _render_diamond_leaderboard(lb_source, "Top Bullpens", "bullpen_10")
    else:
        # Fallback to ELO if composition scores not available
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
