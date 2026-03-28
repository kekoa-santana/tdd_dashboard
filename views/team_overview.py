"""Team Overview page — team intelligence, roster, depth chart, trade sim."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    POSITIVE, NEGATIVE,
    PITCHER_STATS, HITTER_STATS,
    CURRENT_SEASON, PRIOR_SEASON,
    DASHBOARD_DIR,
)
from services.data_loader import (
    load_projections, load_counting, load_player_teams, load_roster,
    load_pitcher_arsenal, load_preseason_injuries,
    load_pitcher_offerings, load_hitter_vuln_arch_career,
    load_cluster_metadata, load_baselines_arch,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_rankings, load_position_eligibility,
    load_probable_starters,
    load_team_elo, load_team_profiles, load_team_rankings,
    load_team_elo_history,
)
from utils.helpers import get_team_lookup, get_injury_lookup
from utils.formatters import fmt_stat
from components.diamond_rating import diamond_rating_text_composite

# ── Constants ─────────────────────────────────────────────────────────
_HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
_PITCHER_POSITIONS = {"SP", "RP", "P"}
_MLB_ROSTER_STATUSES = {"active", "il_7", "il_10", "il_15", "il_60"}

_PILL_COLORS = {
    # Pitcher archetypes
    "Command Specialist": SAGE, "Breaking-Ball Heavy": EMBER,
    "Balanced Mix": SLATE, "Power Arm": GOLD,
    "Fastball Dominant": "#3498DB", "Ground-Ball Artist": "#9B59B6",
    # Hitter archetypes
    "Patient Power": GOLD, "Contact-Over-Power": SAGE,
    "Speed Threat": "#3498DB", "Power Slugger": EMBER,
    "Balanced All-Around": SLATE, "Free Swinger": "#9B59B6",
}

_HEALTH_COLORS = {
    "Iron Man": GOLD, "Durable": SAGE,
    "Average": SLATE, "Unknown": SLATE,
    "Questionable": EMBER, "Injury Prone": NEGATIVE,
}

_TIER_COLORS = {
    "Elite": GOLD, "Contender": SAGE,
    "Competitive": SLATE, "Rebuilding": EMBER,
}


# ═══════════════════════════════════════════════════════════════════════
# TEAM PROFILE TAB — new team intelligence from ELO + profiles
# ═══════════════════════════════════════════════════════════════════════

def _elo_bar(label: str, value: float, rank: int | None = None,
             min_v: float = 1350, max_v: float = 1650) -> str:
    """Render a horizontal bar for an ELO rating."""
    pct = max(0, min(100, (value - min_v) / (max_v - min_v) * 100))
    mid_pct = (1500 - min_v) / (max_v - min_v) * 100
    color = GOLD if value >= 1520 else SAGE if value >= 1490 else EMBER if value < 1470 else SLATE
    rank_str = f'<span class="tdd-meta" style="margin-left:6px;">#{rank}</span>' if rank else ""
    return (
        f'<div style="margin-bottom:10px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">'
        f'<span class="tdd-stat-value">{label}</span>'
        f'<span style="color:{color}; font-weight:700;">{value:.0f}{rank_str}</span>'
        f'</div>'
        f'<div style="position:relative; height:18px; background:var(--tdd-dark-card); border-radius:4px; overflow:visible;">'
        f'<div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:4px;"></div>'
        f'<div style="position:absolute; left:{mid_pct:.1f}%; top:0; height:100%; '
        f'width:1px; background:{SLATE}44;"></div>'
        f'</div></div>'
    )


def _score_gauge(label: str, score: float, extra: str = "") -> str:
    """Small score gauge for sub-dimensions (0-1 scale)."""
    pct = max(0, min(100, score * 100))
    color = GOLD if score >= 0.70 else SAGE if score >= 0.45 else EMBER if score < 0.30 else SLATE
    extra_html = f'<span class="tdd-meta" style="margin-left:4px;">{extra}</span>' if extra else ""
    return (
        f'<div style="margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
        f'<span class="tdd-meta" style="color:var(--tdd-cream);">{label}</span>'
        f'<span class="tdd-stat-value" style="color:{color};">'
        f'{score:.2f}{extra_html}</span>'
        f'</div>'
        f'<div style="height:8px; background:var(--tdd-dark-card); border-radius:4px;">'
        f'<div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:4px;"></div>'
        f'</div></div>'
    )


def _pill(label: str, color: str) -> str:
    """Render a colored pill/tag."""
    return (
        f'<span class="tdd-badge" style="background:{color}22; color:{color}; '
        f'border:1px solid {color}44; margin-right:5px;">{label}</span>'
    )


def _stat_row(label: str, value: str, color: str = CREAM) -> str:
    """Key-value stat row."""
    return (
        f'<div style="display:flex; justify-content:space-between; padding:3px 0;">'
        f'<span class="tdd-stat-label">{label}</span>'
        f'<span class="tdd-stat-value" style="color:{color};">{value}</span>'
        f'</div>'
    )


def _render_team_profile_tab(
    selected_team: str,
    teams_df: pd.DataFrame,
) -> None:
    """Render the Team Profile tab — ELO, sub-scores, team identity."""
    rankings = load_team_rankings()
    profiles = load_team_profiles()
    elo_df = load_team_elo(preseason=True)
    elo_history = load_team_elo_history()

    if rankings.empty and profiles.empty and elo_df.empty:
        st.info("No team intelligence data available. Run precompute first.")
        return

    # Find this team's data
    team_rank = rankings[rankings["abbreviation"] == selected_team] if not rankings.empty else pd.DataFrame()
    team_prof = profiles[profiles["abbreviation"] == selected_team] if not profiles.empty else pd.DataFrame()
    team_elo = elo_df[elo_df["abbreviation"] == selected_team] if not elo_df.empty else pd.DataFrame()

    if team_rank.empty and team_elo.empty:
        st.info(f"No profile data for {selected_team}.")
        return

    tr = team_rank.iloc[0] if not team_rank.empty else pd.Series(dtype=float)
    tp = team_prof.iloc[0] if not team_prof.empty else pd.Series(dtype=float)
    te = team_elo.iloc[0] if not team_elo.empty else pd.Series(dtype=float)

    # ── Tier + headline metrics ──────────────────────────────────────
    tier = tr.get("tier", "Competitive") if not tr.empty else "Competitive"
    tier_color = _TIER_COLORS.get(tier, SLATE)
    rank = int(tr["rank"]) if not tr.empty and pd.notna(tr.get("rank")) else None
    from services.data_loader import load_standings
    _standings = load_standings()
    _rec = _standings.get(selected_team)
    record_html = (
        f'<span class="tdd-context" style="color:var(--tdd-cream);">'
        f'<span style="color:var(--tdd-gold); font-weight:700;">{_rec[0]}-{_rec[1]}</span></span>'
        if _rec else ""
    )

    rank_html = f'<span style="color:var(--tdd-gold); font-size:2rem; font-weight:800;">#{rank}</span>' if rank else ""
    tier_html = _pill(tier, tier_color)

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:16px; margin-bottom:16px; flex-wrap:wrap;">'
        f'{rank_html} {tier_html} {record_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Profile scores (diamond scale, from team_profiles) ──────────
    if not tp.empty:
        # Compute league ranks from team_profiles
        all_tp = load_team_profiles()
        _rank_cols = {
            "lineup_diamond": "Offense",
            "rotation_diamond": "Rotation",
            "bullpen_diamond": "Bullpen",
        }
        _ranks: dict[str, int] = {}
        for col in _rank_cols:
            if col in all_tp.columns:
                ranked = all_tp[col].rank(ascending=False, method="min")
                idx = all_tp.index[all_tp["abbreviation"] == selected_team]
                val = ranked.loc[idx].iloc[0] if len(idx) > 0 else 0
                _ranks[col] = int(val) if pd.notna(val) else 0

        # Defense/depth ranks from team_rankings
        all_tr = load_team_rankings()
        for col, label in [("defense_score", "Fielding"), ("health_depth_score", "Depth")]:
            if col in all_tr.columns:
                ranked = all_tr[col].rank(ascending=False, method="min")
                idx = all_tr.index[all_tr["abbreviation"] == selected_team]
                val = ranked.loc[idx].iloc[0] if len(idx) > 0 else 0
                _ranks[col] = int(val) if pd.notna(val) else 0

        # ── Top 10 players with league-wide positional rank ──────────
        st.markdown(
            '<div class="tdd-section-hdr" style="margin-top:16px; margin-bottom:8px;">'
            'Top Players</div>',
            unsafe_allow_html=True,
        )

        # Load rankings for this team — use roster as authoritative team source
        from lib.diamond_rating import score_to_diamonds as _s2d
        _hr = load_rankings("hitters")
        _pr = load_rankings("pitchers")
        _roster = load_roster()
        _team_pids = set()
        if not _roster.empty:
            _team_pids = set(_roster[_roster["team_abbr"] == selected_team]["player_id"])

        def _player_rank_row(name: str, pos: str, league_rank: int, score: float) -> str:
            color = GOLD if league_rank <= 5 else SAGE if league_rank <= 15 else SLATE
            diamonds = _s2d(score)
            return (
                f'<div style="display:flex; align-items:center; gap:8px; '
                f'padding:4px 12px; border-left:3px solid {color}; margin-bottom:2px;">'
                f'<span class="tdd-stat-value" style="color:{color}; min-width:2.5rem; '
                f'font-weight:700;">#{league_rank}</span>'
                f'<span class="tdd-meta" style="min-width:2rem; color:{SLATE};">{pos}</span>'
                f'<span style="color:var(--tdd-cream); flex:1;">{name}</span>'
                f'<span class="tdd-stat-value" style="color:var(--tdd-gold);">{diamonds:.1f}</span>'
                f'</div>'
            )

        top_rows = ""
        top_players = []

        if not _hr.empty and _team_pids:
            _h_pid = "batter_id" if "batter_id" in _hr.columns else "player_id"
            _h_score = next((c for c in ("current_value_score", "tdd_value_score") if c in _hr.columns), None)
            if _h_score:
                team_hitters = _hr[_hr[_h_pid].isin(_team_pids)].copy()
                for _, row in team_hitters.iterrows():
                    top_players.append({
                        "name": row["batter_name"],
                        "pos": row.get("position", ""),
                        "rank": int(row.get("pos_rank", 0)),
                        "score": float(row[_h_score]),
                    })

        if not _pr.empty and _team_pids:
            _p_pid = "pitcher_id" if "pitcher_id" in _pr.columns else "player_id"
            _p_score = next((c for c in ("current_value_score", "tdd_value_score") if c in _pr.columns), None)
            if _p_score:
                team_pitchers = _pr[_pr[_p_pid].isin(_team_pids)].copy()
                for _, row in team_pitchers.iterrows():
                    top_players.append({
                        "name": row["pitcher_name"],
                        "pos": row.get("role", "P"),
                        "rank": int(row.get("role_rank", 0)),
                        "score": float(row[_p_score]),
                    })

        top_players.sort(key=lambda x: x["score"], reverse=True)
        for p in top_players[:10]:
            top_rows += _player_rank_row(p["name"], p["pos"], p["rank"], p["score"])

        if top_rows:
            st.markdown(
                f'<div class="insight-card">{top_rows}</div>',
                unsafe_allow_html=True,
            )

        # Style pills
        styles = []
        off_style = tr.get("offense_style")
        if pd.notna(off_style):
            s_color = GOLD if off_style == "Power" else SAGE if off_style == "Contact" else "#3498DB" if off_style == "Smallball" else SLATE
            styles.append(_pill(f"\u26be {off_style}", s_color))
        pit_style = tr.get("pitching_style")
        if pd.notna(pit_style):
            p_color = GOLD if pit_style == "Strikeout" else SAGE if pit_style == "Contact-mgmt" else SLATE
            styles.append(_pill(f"\u26be {pit_style}", p_color))
        age_traj = tr.get("age_trajectory")
        if pd.notna(age_traj):
            a_color = SAGE if age_traj == "Ascending" else EMBER if age_traj == "Declining" else SLATE
            styles.append(_pill(age_traj, a_color))
        if styles:
            st.markdown(
                f'<div style="margin-top:8px;">{"".join(styles)}</div>',
                unsafe_allow_html=True,
            )

    # ── Offense snapshot ─────────────────────────────────────────────
    st.markdown(
        '<div class="tdd-section-hdr" style="margin-top:16px; margin-bottom:8px;">Offense</div>',
        unsafe_allow_html=True,
    )

    off_cols = st.columns(3)
    with off_cols[0]:
        rpg = tr.get("rpg") if not tr.empty else None
        hr_pg = tr.get("hr_per_game") if not tr.empty else None
        html = ""
        if rpg and pd.notna(rpg):
            html += _stat_row("R/G", f"{rpg:.2f}")
        if hr_pg and pd.notna(hr_pg):
            html += _stat_row("HR/G", f"{hr_pg:.2f}")
        depth = tp.get("lineup_depth") if not tp.empty else None
        if depth and pd.notna(depth):
            html += _stat_row("Lineup Depth", f"{depth:.0%}")
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    with off_cols[1]:
        html = ""
        avg_h = tp.get("avg_hitter_score") if not tp.empty else None
        if avg_h and pd.notna(avg_h):
            html += _stat_row("Avg Hitter Score", f"{avg_h:.2f}")
        sb_pg = tr.get("sb_per_game") if not tr.empty else None
        if sb_pg and pd.notna(sb_pg):
            html += _stat_row("SB/G", f"{sb_pg:.2f}")
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    with off_cols[2]:
        html = ""
        lu_d = tp.get("lineup_diamond") if not tp.empty else None
        if lu_d is not None and pd.notna(lu_d):
            lu_color = GOLD if float(lu_d) >= 6.5 else SAGE if float(lu_d) >= 5.0 else SLATE
            html += _stat_row("Lineup", f"{float(lu_d):.1f}", lu_color)
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    # ── Pitching snapshot ────────────────────────────────────────────
    st.markdown(
        '<div class="tdd-section-hdr" style="margin-top:16px; margin-bottom:8px;">Pitching</div>',
        unsafe_allow_html=True,
    )

    pit_cols = st.columns(3)
    with pit_cols[0]:
        html = ""
        ra_pg = tr.get("ra_per_game") if not tr.empty else None
        if ra_pg and pd.notna(ra_pg):
            html += _stat_row("RA/G", f"{ra_pg:.2f}")
        k_pg = tr.get("k_per_game") if not tr.empty else None
        if k_pg and pd.notna(k_pg):
            html += _stat_row("K/G", f"{k_pg:.1f}")
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    with pit_cols[1]:
        html = ""
        rot = tp.get("rotation_strength") if not tp.empty else None
        if rot and pd.notna(rot):
            html += _stat_row("Rotation Strength", f"{rot:.2f}")
        bp = tp.get("bullpen_depth") if not tp.empty else None
        if bp and pd.notna(bp):
            html += _stat_row("Bullpen Depth", f"{bp:.2f}")
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    with pit_cols[2]:
        html = ""
        rot_d = tp.get("rotation_diamond") if not tp.empty else None
        if rot_d is not None and pd.notna(rot_d):
            rot_color = GOLD if float(rot_d) >= 6.5 else SAGE if float(rot_d) >= 5.0 else SLATE
            html += _stat_row("Rotation", f"{float(rot_d):.1f}", rot_color)
        bp_d = tp.get("bullpen_diamond") if not tp.empty else None
        if bp_d is not None and pd.notna(bp_d):
            bp_color = GOLD if float(bp_d) >= 6.5 else SAGE if float(bp_d) >= 5.0 else SLATE
            html += _stat_row("Bullpen", f"{float(bp_d):.1f}", bp_color)
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    # ── Defense & Organization row ───────────────────────────────────
    def_org_cols = st.columns(2)
    with def_org_cols[0]:
        st.markdown(
            '<div class="tdd-section-hdr" style="margin-top:12px; margin-bottom:8px;">Defense</div>',
            unsafe_allow_html=True,
        )
        html = ""
        oaa = tr.get("team_oaa") if not tr.empty else None
        if oaa is not None and pd.notna(oaa):
            oaa_color = POSITIVE if float(oaa) > 0 else NEGATIVE if float(oaa) < 0 else SLATE
            html += _stat_row("Team OAA", f"{oaa:+.0f}", oaa_color)
        framing = tr.get("catcher_framing_runs") if not tr.empty else None
        if framing is not None and pd.notna(framing):
            fr_color = POSITIVE if float(framing) > 0 else NEGATIVE if float(framing) < 0 else SLATE
            html += _stat_row("Catcher Framing", f"{framing:+.1f} runs", fr_color)
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    with def_org_cols[1]:
        st.markdown(
            '<div class="tdd-section-hdr" style="margin-top:12px; margin-bottom:8px;">Organization</div>',
            unsafe_allow_html=True,
        )
        html = ""
        hg = tr.get("pct_homegrown") if not tr.empty else None
        if hg and pd.notna(hg):
            html += _stat_row("Homegrown %", f"{hg:.0%}")
        farm = tr.get("farm_avg_score") if not tr.empty else None
        if farm and pd.notna(farm):
            html += _stat_row("Farm Avg Score", f"{farm:.2f}")
        n_prosp = tr.get("n_ranked_prospects") if not tr.empty else None
        if n_prosp and pd.notna(n_prosp):
            html += _stat_row("Ranked Prospects", f"{int(n_prosp)}")
        top100 = tr.get("top_100_count") if not tr.empty else None
        if top100 and pd.notna(top100) and int(top100) > 0:
            html += _stat_row("Top-100 Prospects", f"{int(top100)}")
        if html:
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    # ── Health & Age ─────────────────────────────────────────────────
    health_cols = st.columns(2)
    with health_cols[0]:
        il_days = tr.get("total_il_days") if not tr.empty else None
        avg_age = tr.get("avg_age") if not tr.empty else None
        if (il_days and pd.notna(il_days)) or (avg_age and pd.notna(avg_age)):
            st.markdown(
                '<div class="tdd-section-hdr" style="margin-top:12px; margin-bottom:8px;">Health & Age</div>',
                unsafe_allow_html=True,
            )
            html = ""
            if il_days and pd.notna(il_days):
                il_color = NEGATIVE if int(il_days) > 1000 else SLATE
                html += _stat_row(f"IL Days ({PRIOR_SEASON})", f"{int(il_days)}", il_color)
            if avg_age and pd.notna(avg_age):
                html += _stat_row("Avg Roster Age", f"{avg_age:.1f}")
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    with health_cols[1]:
        opp_elo = tr.get("avg_opp_elo") if not tr.empty else None
        big_gm = tr.get("big_game_pct") if not tr.empty else None
        div_elo = tr.get("div_avg_elo") if not tr.empty else None
        if (opp_elo and pd.notna(opp_elo)) or (big_gm and pd.notna(big_gm)):
            st.markdown(
                '<div class="tdd-section-hdr" style="margin-top:12px; margin-bottom:8px;">Schedule Context</div>',
                unsafe_allow_html=True,
            )
            html = ""
            if opp_elo and pd.notna(opp_elo):
                html += _stat_row("Avg Opp ELO", f"{opp_elo:.0f}")
            if big_gm and pd.notna(big_gm):
                html += _stat_row("Big Game W%", f"{big_gm:.0%}")
            if div_elo and pd.notna(div_elo):
                html += _stat_row("Division Avg ELO", f"{div_elo:.0f}")
            st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)

    # ── ELO trend sparkline (last 3 seasons) ─────────────────────────
    if not elo_history.empty:
        team_hist = elo_history[elo_history["abbreviation"] == selected_team] if "abbreviation" in elo_history.columns else pd.DataFrame()
        if team_hist.empty and not te.empty:
            team_id = te.get("team_id")
            if pd.notna(team_id):
                team_hist = elo_history[elo_history["team_id"] == int(team_id)]

        if not team_hist.empty and len(team_hist) > 20:
            st.markdown(
                '<div class="tdd-section-hdr" style="margin-top:16px; margin-bottom:8px;">ELO Trend</div>',
                unsafe_allow_html=True,
            )
            import plotly.graph_objects as go
            chart_data = team_hist[["game_date", "composite_elo"]].copy()
            chart_data["game_date"] = pd.to_datetime(chart_data["game_date"])
            chart_data = chart_data.set_index("game_date").resample("W").last().dropna()

            elo_fig = go.Figure()
            elo_fig.add_trace(go.Scatter(
                x=chart_data.index, y=chart_data["composite_elo"],
                mode="lines", line=dict(color=GOLD, width=2),
                hovertemplate="ELO: %{y:.0f}<extra></extra>",
            ))
            elo_fig.add_hline(y=1500, line_dash="dot", line_color=SLATE, line_width=1)
            elo_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=200, showlegend=False, dragmode=False,
                xaxis=dict(showgrid=False, fixedrange=True,
                           tickfont=dict(color=SLATE, size=9)),
                yaxis=dict(showgrid=True, gridcolor="rgba(123,143,166,0.12)",
                           fixedrange=True, tickfont=dict(color=SLATE, size=9)),
                font=dict(color=CREAM),
            )
            st.plotly_chart(
                elo_fig, use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False},
            )

    # ── League-wide rankings table ───────────────────────────────────
    if not rankings.empty:
        with st.expander("League-Wide Power Rankings"):
            display_cols = ["rank", "abbreviation", "tier", "composite_score",
                            "offense_score", "pitching_score", "defense_score"]
            avail = [c for c in display_cols if c in rankings.columns]
            display = rankings[avail].copy()
            rename_map = {
                "rank": "Rank", "abbreviation": "Team", "tier": "Tier",
                "composite_score": "Score",
                "offense_score": "Off", "pitching_score": "Pit",
                "defense_score": "Field",
            }
            display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

            # Highlight selected team
            def _highlight_team(row: pd.Series) -> list[str]:
                if row.get("Team") == selected_team:
                    return [f"background-color: {GOLD}22; font-weight: bold"] * len(row)
                return [""] * len(row)

            styled = display.style.apply(_highlight_team, axis=1).format(
                {c: "{:.2f}" for c in ["Score", "Off", "Pit", "Def"] if c in display.columns}
            )
            st.dataframe(styled, width='stretch', hide_index=True, height=500)


# ═══════════════════════════════════════════════════════════════════════
# DEPTH CHART TAB (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

def _build_depth_chart(
    selected_team: str, teams_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble per-position player lists from rankings + archetypes."""
    h_rank = load_rankings("hitters")
    p_rank = load_rankings("pitchers")
    if h_rank.empty and p_rank.empty:
        return pd.DataFrame(), pd.DataFrame()

    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )

    if not h_rank.empty:
        team_h = h_rank[h_rank["batter_id"].isin(team_pids)].copy()
        elig = load_position_eligibility()
        if not elig.empty and not team_h.empty:
            team_elig = elig[elig["player_id"].isin(team_pids)].copy()
            team_elig = team_elig.rename(columns={"player_id": "batter_id"})
            base = team_h.drop(columns=["position"], errors="ignore")
            team_h = base.merge(
                team_elig[["batter_id", "position", "starts", "pct", "is_primary"]],
                on="batter_id", how="inner",
            )
            team_h.loc[~team_h["is_primary"], "pos_rank"] = pd.NA
            team_h = team_h.sort_values(
                ["position", "tdd_value_score"], ascending=[True, False],
            )
    else:
        team_h = pd.DataFrame()

    team_p = (
        p_rank[p_rank["pitcher_id"].isin(team_pids)].copy()
        if not p_rank.empty else pd.DataFrame()
    )

    h_arch = load_hitter_archetypes()
    p_arch = load_pitcher_archetypes()
    if not h_arch.empty and not team_h.empty:
        team_h = team_h.merge(
            h_arch[["batter_id", "archetype_name"]].drop_duplicates("batter_id"),
            on="batter_id", how="left",
        )
    if not p_arch.empty and not team_p.empty:
        team_p = team_p.merge(
            p_arch[["pitcher_id", "archetype_name"]].drop_duplicates("pitcher_id"),
            on="pitcher_id", how="left",
        )

    return team_h, team_p


def _player_row_html(
    name: str, rank: int, score: float, arch: str, health: str,
    is_first: bool, *, hand: str = "", secondary: bool = False,
) -> str:
    """Render a single player row for depth chart."""
    if secondary:
        border_color = f"{SLATE}22"
    elif is_first:
        border_color = GOLD
    else:
        border_color = f"{SLATE}44"
    opacity = "0.6" if secondary else "1"

    arch_html = ""
    if pd.notna(arch) and arch:
        color = _PILL_COLORS.get(arch, SLATE)
        arch_html = (
            f'<span class="tdd-badge" style="background:{color}22; color:{color}; '
            f'border:1px solid {color}44; margin-left:8px;">{arch}</span>'
        )

    h_color = _HEALTH_COLORS.get(health, SLATE)
    dot = (
        "\u25cf" if health in ("Iron Man", "Durable")
        else "\u25d0" if health in ("Average", "Unknown")
        else "\u25cb"
    )
    health_html = (
        f'<span class="tdd-meta" style="color:{h_color}; margin-left:8px;">'
        f'{dot} {health}</span>'
    )

    hand_html = ""
    if hand:
        hand_html = (
            f'<span class="tdd-meta" style="min-width:30px;">'
            f'{hand}HP</span>'
        )

    rank_html = (
        f'<span class="tdd-context" style="min-width:60px;">'
        f'#{rank}</span>'
        if rank > 0 else ""
    )

    return (
        f'<div style="padding:6px 12px; border-left:3px solid {border_color}; '
        f'margin-bottom:2px; display:flex; align-items:center; gap:8px; '
        f'flex-wrap:wrap; opacity:{opacity};">'
        f'<span class="tdd-player-name" style="font-weight:{"700" if is_first else "400"}; '
        f'min-width:6rem; flex-shrink:1;">{name}</span>'
        f'{hand_html}'
        f'{rank_html}'
        f'<span class="tdd-stat-value" style="color:var(--tdd-gold); min-width:2.5rem;">{score * 10:.1f}</span>'
        f'{arch_html}{health_html}'
        f'</div>'
    )


def _render_depth_chart_tab(
    selected_team: str,
    teams_df: pd.DataFrame,
    *,
    team_pitchers: pd.DataFrame | None = None,
    p_proj: pd.DataFrame | None = None,
) -> None:
    """Render depth chart from roster.parquet lineup positions."""
    if team_pitchers is None:
        team_pitchers = pd.DataFrame()
    if p_proj is None:
        p_proj = pd.DataFrame()

    # Load roster — contains lineup_position + is_depth_starter from
    # actual current-season lineups (set by precompute run_roster).
    roster = load_roster()
    if roster.empty:
        st.info("No roster data available.")
        return

    team_roster = roster[roster["team_abbr"] == selected_team].copy()
    if team_roster.empty:
        st.info("No roster data for this team.")
        return

    team_pids = set(team_roster["player_id"].astype(int))

    # Pitcher rankings for rotation/bullpen
    p_rank = load_rankings("pitchers")
    team_p = p_rank[p_rank["pitcher_id"].isin(team_pids)].copy() if not p_rank.empty else pd.DataFrame()
    p_arch = load_pitcher_archetypes()
    if not p_arch.empty and not team_p.empty:
        team_p = team_p.merge(
            p_arch[["pitcher_id", "archetype_name"]].drop_duplicates("pitcher_id"),
            on="pitcher_id", how="left",
        )

    # --- Position Depth Chart ---
    st.markdown("### Position Players")

    has_lineup = "lineup_position" in team_roster.columns
    has_starter = "is_depth_starter" in team_roster.columns

    for pos in _HITTER_POSITIONS:
        # Starter: player flagged is_depth_starter at this lineup_position
        starter_row = None
        bench_at_pos: list[pd.Series] = []

        if has_lineup and has_starter:
            starters_here = team_roster[
                (team_roster["lineup_position"] == pos)
                & (team_roster["is_depth_starter"] == True)  # noqa: E712
            ]
            if not starters_here.empty:
                starter_row = starters_here.iloc[0]

            # Bench players who have also started at this position
            bench_here = team_roster[
                (team_roster["lineup_position"] == pos)
                & (team_roster["is_depth_starter"] == False)  # noqa: E712
            ]
            bench_at_pos = [row for _, row in bench_here.iterrows()]

        if starter_row is None and not bench_at_pos:
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:0.5rem; '
                f'padding:0.4rem 0; border-bottom:1px solid var(--tdd-dark-border-faint);">'
                f'<span style="color:var(--tdd-gold); font-weight:700; '
                f'min-width:2.2rem; font-size:0.9rem;">{pos}</span>'
                f'<span class="tdd-context">No data</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            continue

        player_parts: list[str] = []
        if starter_row is not None:
            name = starter_row["player_name"]
            player_parts.append(
                f'<span style="color:var(--tdd-cream); font-weight:600; '
                f'font-size:0.88rem;">{name}</span>'
            )
        for brow in bench_at_pos[:3]:
            player_parts.append(
                f'<span class="tdd-context">{brow["player_name"]}</span>'
            )

        st.markdown(
            f'<div style="display:flex; align-items:center; gap:0.5rem; '
            f'padding:0.4rem 0; border-bottom:1px solid var(--tdd-dark-border-faint); '
            f'flex-wrap:wrap;">'
            f'<span style="color:var(--tdd-gold); font-weight:700; '
            f'min-width:2.2rem; font-size:0.9rem;">{pos}</span>'
            + " <span style='color:var(--tdd-slate); margin:0 0.2rem;'>\u2192</span> ".join(player_parts)
            + '</div>',
            unsafe_allow_html=True,
        )

    # --- Pitching Depth ---
    _render_pitching_depth(team_p)

    # --- Rotation / Bullpen Profiles ---
    if not team_pitchers.empty and not p_proj.empty:
        team_sp = team_pitchers[team_pitchers["is_starter"] == True]  # noqa: E712
        team_rp = team_pitchers[team_pitchers["is_starter"] == False]  # noqa: E712
        lg_sp = p_proj[p_proj["is_starter"] == True]  # noqa: E712
        lg_rp = p_proj[p_proj["is_starter"] == False]  # noqa: E712

        _k_col = "projected_k_rate"
        _bb_col = "projected_bb_rate"

        if not team_sp.empty:
            st.markdown(f"### Rotation Profile")
            _render_staff_metrics(team_sp, lg_sp, _k_col, _bb_col)
        if not team_rp.empty:
            st.markdown(f"### Bullpen Profile")
            _render_staff_metrics(team_rp, lg_rp, _k_col, _bb_col)


def _render_hitter_depth(team_h: pd.DataFrame) -> None:
    """Render hitter depth chart."""
    if team_h.empty:
        return
    st.markdown("### Hitter Depth")

    has_elig = "is_primary" in team_h.columns

    for pos in _HITTER_POSITIONS:
        pos_players = team_h[team_h["position"] == pos].sort_values(
            "tdd_value_score", ascending=False,
        )
        if pos_players.empty:
            st.markdown(
                f'<div style="padding:8px 12px; margin-bottom:6px; '
                f'border-left:3px solid var(--tdd-slate); opacity:0.3;">'
                f'<span class="tdd-stat-label" style="font-weight:600;">{pos}</span> '
                f'<span class="tdd-context">'
                f'\u2014 No ranked player</span></div>',
                unsafe_allow_html=True,
            )
            continue

        rows_html = ""
        for i, (_, row) in enumerate(pos_players.iterrows()):
            name = row["batter_name"]
            is_secondary = has_elig and not row.get("is_primary", True)
            if is_secondary:
                pct = row.get("pct", 0)
                name += f" ({pct:.0%})" if pct else ""
            rank_val = int(row["pos_rank"]) if pd.notna(row.get("pos_rank")) else 0
            rows_html += _player_row_html(
                name=name, rank=rank_val, score=row["tdd_value_score"],
                arch=row.get("archetype_name", ""),
                health=row.get("health_label", "Unknown"),
                is_first=i == 0 and not is_secondary, secondary=is_secondary,
            )

        st.markdown(
            f'<div style="margin-bottom:10px;">'
            f'<div class="tdd-section-hdr" style="margin-bottom:4px; padding-left:15px;">{pos}</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )


def _render_pitching_depth(team_p: pd.DataFrame) -> None:
    """Render pitching depth — rotation + bullpen."""
    if team_p.empty:
        return
    st.markdown("### Pitching Depth")

    for label, subset in [
        ("Starting Rotation",
         team_p[team_p["role"] == "SP"].sort_values("role_rank").head(5)),
        ("Bullpen",
         team_p[team_p["role"] == "RP"].sort_values("tdd_value_score", ascending=False)),
    ]:
        if subset.empty:
            continue
        st.markdown(
            f'<div class="tdd-section-hdr" style="margin-bottom:4px; padding-left:15px;">{label}</div>',
            unsafe_allow_html=True,
        )
        rows_html = ""
        for i, (_, row) in enumerate(subset.iterrows()):
            rows_html += _player_row_html(
                name=row["pitcher_name"], rank=int(row["role_rank"]),
                score=row["tdd_value_score"],
                arch=row.get("archetype_name", ""),
                health=row.get("health_label", "Unknown"),
                is_first=i == 0, hand=row.get("pitch_hand", ""),
            )
        st.markdown(
            f'<div style="margin-bottom:14px;">{rows_html}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR TAB (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

def _identify_weaknesses(
    selected_team: str, teams_df: pd.DataFrame,
) -> list[dict]:
    """Identify positional weaknesses — best player pos_rank > 15."""
    h_rank = load_rankings("hitters")
    if h_rank.empty:
        return []
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )
    team_h = h_rank[h_rank["batter_id"].isin(team_pids)]

    weaknesses: list[dict] = []
    for pos in _HITTER_POSITIONS:
        if pos == "DH":
            continue
        pos_players = team_h[team_h["position"] == pos]
        if pos_players.empty:
            weaknesses.append({"position": pos, "best_rank": None, "player": None})
        else:
            best = pos_players.loc[pos_players["pos_rank"].idxmin()]
            if best["pos_rank"] > 15:
                weaknesses.append({
                    "position": pos,
                    "best_rank": int(best["pos_rank"]),
                    "player": best["batter_name"],
                })
    return weaknesses


def _find_trade_targets(
    selected_team: str, teams_df: pd.DataFrame,
    target_position: str, trade_capital: float,
    offered_positions: list[str] | None = None,
    tolerance: float = 0.15,
) -> pd.DataFrame:
    """Find league-wide trade targets at a position within value tolerance."""
    h_rank = load_rankings("hitters")
    if h_rank.empty:
        return pd.DataFrame()

    other_pids = set(
        teams_df[teams_df["team_abbr"] != selected_team]["player_id"].astype(int)
    )
    candidates = h_rank[
        (h_rank["batter_id"].isin(other_pids))
        & (h_rank["position"] == target_position)
    ].copy()

    if candidates.empty or trade_capital <= 0:
        return pd.DataFrame()

    low = trade_capital * (1 - tolerance)
    high = trade_capital * (1 + tolerance)
    matches = candidates[
        (candidates["tdd_value_score"] >= low) & (candidates["tdd_value_score"] <= high)
    ].copy()

    if matches.empty:
        candidates["_diff"] = (candidates["tdd_value_score"] - trade_capital).abs()
        matches = candidates.nsmallest(10, "_diff").drop(columns="_diff")

    pid_to_team = dict(zip(
        teams_df["player_id"].astype(int), teams_df["team_abbr"],
    ))
    matches["team"] = matches["batter_id"].map(pid_to_team)
    matches["value_match_pct"] = (
        1 - (matches["tdd_value_score"] - trade_capital).abs()
        / max(trade_capital, 0.01)
    ) * 100

    h_arch = load_hitter_archetypes()
    if not h_arch.empty:
        arch_cols = h_arch[["batter_id", "archetype_name"]].drop_duplicates("batter_id")
        if "archetype_name" in matches.columns:
            matches = matches.drop(columns="archetype_name")
        matches = matches.merge(arch_cols, on="batter_id", how="left")

    if offered_positions:
        hitter_pos = [p for p in offered_positions if p in _HITTER_POSITIONS]
        fit_tiers: list[str] = []
        for _, row in matches.iterrows():
            partner_team = row.get("team", "")
            if not partner_team or not hitter_pos:
                fit_tiers.append("Low")
                continue
            partner_pids = set(
                teams_df[teams_df["team_abbr"] == partner_team]["player_id"].astype(int)
            )
            partner_h = h_rank[h_rank["batter_id"].isin(partner_pids)]
            tier = "Low"
            for pos in hitter_pos:
                pos_players = partner_h[partner_h["position"] == pos]
                if pos_players.empty:
                    tier = "High"
                    break
                best_rank = int(pos_players["pos_rank"].min())
                if best_rank > 15:
                    tier = "High"
                    break
                elif best_rank > 10 and tier != "High":
                    tier = "Medium"
            fit_tiers.append(tier)
        matches["trade_fit"] = fit_tiers
    else:
        matches["trade_fit"] = "Low"

    fit_order = {"High": 0, "Medium": 1, "Low": 2}
    matches["_fit_order"] = matches["trade_fit"].map(fit_order)
    return matches.sort_values(["_fit_order", "pos_rank"]).drop(columns="_fit_order").head(10)


def _render_trade_simulator_tab(
    selected_team: str, teams_df: pd.DataFrame,
) -> None:
    """Render the Trade Simulator tab."""
    st.markdown(
        f'<div style="background:{EMBER}22; border:1px solid {EMBER}44; '
        f'border-radius:6px; padding:0.6rem 1rem; margin-bottom:1rem; '
        f'font-size:0.85rem; color:{EMBER}; font-weight:600;">'
        f'WORK IN PROGRESS — This feature is still being refined.</div>',
        unsafe_allow_html=True,
    )
    weaknesses = _identify_weaknesses(selected_team, teams_df)

    st.markdown("#### Positional Needs")
    if weaknesses:
        pills_html = ""
        for w in weaknesses:
            suffix = f' (#{w["best_rank"]})' if w["best_rank"] else " (empty)"
            pills_html += (
                f'<span class="tdd-badge" style="background:{EMBER}22; color:{EMBER}; '
                f'border:1px solid {EMBER}44; margin-right:6px;">'
                f'{w["position"]}{suffix}</span> '
            )
        st.markdown(
            f'<div style="margin-bottom:16px;">{pills_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:var(--tdd-sage);">'
            f'No significant positional weaknesses detected.</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("#### Select Trade Assets")
    h_rank = load_rankings("hitters")
    p_rank = load_rankings("pitchers")
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )

    assets: list[dict] = []
    if not h_rank.empty:
        team_h = h_rank[h_rank["batter_id"].isin(team_pids)].sort_values(
            "tdd_value_score", ascending=False,
        )
        for _, row in team_h.iterrows():
            assets.append({
                "label": (
                    f'{row["batter_name"]} ({row["position"]}, '
                    f'#{int(row["pos_rank"])}) '
                    f'\u2014 {row["tdd_value_score"]:.2f}'
                ),
                "value": row["tdd_value_score"],
                "position": row["position"],
            })
    if not p_rank.empty:
        team_p = p_rank[p_rank["pitcher_id"].isin(team_pids)].sort_values(
            "tdd_value_score", ascending=False,
        )
        for _, row in team_p.iterrows():
            assets.append({
                "label": (
                    f'{row["pitcher_name"]} ({row["role"]}, '
                    f'#{int(row["role_rank"])}) '
                    f'\u2014 {row["tdd_value_score"]:.2f}'
                ),
                "value": row["tdd_value_score"],
                "position": row["role"],
            })

    if not assets:
        st.info("No ranked players available for trade selection.")
        return

    selected_assets = st.multiselect(
        "Players to offer", options=[a["label"] for a in assets], key="trade_assets",
    )
    trade_capital = sum(a["value"] for a in assets if a["label"] in selected_assets)
    st.markdown(
        f'<div style="padding:8px 16px; background:{GOLD}15; '
        f'border:1px solid {GOLD}44; border-radius:8px; margin:8px 0 16px;">'
        f'<span class="tdd-stat-value">Trade Capital:</span> '
        f'<span style="color:var(--tdd-gold); font-weight:700; font-size:1.1rem;">'
        f'{trade_capital:.2f}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Target Need")
    weakness_positions = [w["position"] for w in weaknesses]
    all_positions = weakness_positions + [
        p for p in _HITTER_POSITIONS if p not in weakness_positions
    ]
    target_pos = st.selectbox("Position to upgrade", all_positions, key="trade_target_pos")

    if st.button("Find Trade Matches", type="primary", disabled=trade_capital <= 0):
        offered_positions = [
            a["position"] for a in assets
            if a["label"] in selected_assets and "position" in a
        ]
        matches = _find_trade_targets(
            selected_team, teams_df, target_pos, trade_capital,
            offered_positions=offered_positions,
        )
        if matches.empty:
            st.info(f"No trade candidates found at {target_pos}.")
        else:
            _render_trade_results(matches, trade_capital, target_pos, selected_team, teams_df)


def _render_trade_results(
    matches: pd.DataFrame, trade_capital: float,
    target_pos: str, selected_team: str, teams_df: pd.DataFrame,
) -> None:
    """Display trade match results table."""
    st.markdown(f"#### Trade Candidates \u2014 {target_pos}")

    h_rank = load_rankings("hitters")
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )
    team_at_pos = h_rank[
        (h_rank["batter_id"].isin(team_pids)) & (h_rank["position"] == target_pos)
    ]
    current_best_rank = int(team_at_pos["pos_rank"].min()) if not team_at_pos.empty else None

    rows_html = ""
    for _, row in matches.iterrows():
        name = row["batter_name"]
        team = row.get("team", "?")
        rank = int(row["pos_rank"])
        score = row["tdd_value_score"]
        match_pct = row.get("value_match_pct", 0)
        arch = row.get("archetype_name", "")
        trade_fit = row.get("trade_fit", "Low")
        fit_color = GOLD if trade_fit == "High" else SAGE if trade_fit == "Medium" else SLATE
        fit_label = "Likely" if trade_fit == "High" else "Possible" if trade_fit == "Medium" else "Unlikely"

        if current_best_rank and rank < current_best_rank:
            upgrade = f'<span style="color:{POSITIVE}; font-weight:600;">\u2191 +{current_best_rank - rank}</span>'
        elif current_best_rank:
            upgrade = f'<span style="color:var(--tdd-slate);">\u2014</span>'
        else:
            upgrade = f'<span style="color:{POSITIVE}; font-weight:600;">\u2191 New</span>'

        arch_html = ""
        if pd.notna(arch) and arch:
            color = _PILL_COLORS.get(arch, SLATE)
            arch_html = (
                f'<span class="tdd-badge" style="background:{color}22; color:{color}; '
                f'border:1px solid {color}44;">{arch}</span>'
            )

        m_color = GOLD if match_pct >= 85 else SAGE if match_pct >= 70 else EMBER

        rows_html += (
            f'<tr>'
            f'<td class="tdd-player-name" style="padding:6px 10px;">{name}</td>'
            f'<td class="tdd-stat-label" style="padding:6px 10px;" data-team="{team}">{team}</td>'
            f'<td style="padding:6px 10px;">{arch_html}</td>'
            f'<td class="tdd-stat-value" style="padding:6px 10px;color:var(--tdd-gold);">{score:.2f}</td>'
            f'<td style="padding:6px 10px;">#{rank}</td>'
            f'<td style="padding:6px 10px;color:{m_color};">{match_pct:.0f}%</td>'
            f'<td style="padding:6px 10px;">{upgrade}</td>'
            f'<td style="padding:6px 10px;">'
            f'<span class="tdd-badge" style="background:{fit_color}22; color:{fit_color}; '
            f'border:1px solid {fit_color}44;">'
            f'{fit_label}</span></td>'
            f'</tr>'
        )

    table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        f'<thead><tr style="border-bottom:1px solid {SLATE}44;">'
        f'<th class="tdd-section-hdr" style="text-align:left;padding:6px 10px;">Player</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">Team</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">Archetype</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">TDD Score</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">Pos Rank</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">Match %</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">Upgrade</th>'
        f'<th class="tdd-stat-label" style="padding:6px 10px;">Trade Fit</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )
    st.markdown(f'<div class="insight-card">{table_html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# ROSTER TAB — original overview content (identity, profiles, tables)
# ═══════════════════════════════════════════════════════════════════════

def _render_staff_metrics(
    team_subset: pd.DataFrame, lg_subset: pd.DataFrame,
    k_col: str, bb_col: str,
) -> None:
    """Render strengths/weaknesses for a pitcher subset."""
    pitch_metrics: list[dict] = []
    for label, key, higher_better in [
        ("K%", k_col, True), ("BB%", bb_col, False),
        ("Whiff%", "whiff_rate", True), ("Avg Velo", "avg_velo", True),
        ("Zone%", "zone_pct", True), ("GB%", "gb_pct", True),
    ]:
        if key not in team_subset.columns or key not in lg_subset.columns:
            continue
        team_avg = team_subset[key].dropna().mean()
        league_avg = lg_subset[key].dropna().mean()
        if pd.isna(team_avg) or pd.isna(league_avg) or league_avg == 0:
            continue
        diff = team_avg - league_avg
        if key in (k_col, bb_col, "whiff_rate", "zone_pct", "gb_pct"):
            diff_str = f"{diff * 100:+.1f}pp"
            team_str = f"{team_avg * 100:.1f}%"
            lg_str = f"{league_avg * 100:.1f}%"
        else:
            diff_str = f"{diff:+.1f}"
            team_str = f"{team_avg:.1f}"
            lg_str = f"{league_avg:.1f}"
        is_good = (diff > 0 and higher_better) or (diff < 0 and not higher_better)
        color = POSITIVE if is_good else NEGATIVE
        pitch_metrics.append({
            "Metric": label, "Team": team_str, "League Avg": lg_str,
            "Diff": diff_str, "_color": color, "_is_good": is_good,
        })

    if not pitch_metrics:
        return

    strengths = [m for m in pitch_metrics if m["_is_good"]]
    weaknesses_p = [m for m in pitch_metrics if not m["_is_good"]]
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown(
            f'<div style="color:{POSITIVE}; font-weight:600; margin-bottom:8px;">Strengths</div>',
            unsafe_allow_html=True,
        )
        if strengths:
            for m in strengths:
                st.markdown(
                    f'<div style="padding:4px 0;">'
                    f'<span class="tdd-player-name" style="font-weight:400;">{m["Metric"]}</span>: '
                    f'<span class="tdd-stat-value" style="color:{POSITIVE};">{m["Team"]}</span> '
                    f'<span class="tdd-context">(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span class="tdd-context">None vs league average</span>', unsafe_allow_html=True)
    with col_w:
        st.markdown(
            f'<div style="color:{NEGATIVE}; font-weight:600; margin-bottom:8px;">Weaknesses</div>',
            unsafe_allow_html=True,
        )
        if weaknesses_p:
            for m in weaknesses_p:
                st.markdown(
                    f'<div style="padding:4px 0;">'
                    f'<span class="tdd-player-name" style="font-weight:400;">{m["Metric"]}</span>: '
                    f'<span class="tdd-stat-value" style="color:{NEGATIVE};">{m["Team"]}</span> '
                    f'<span class="tdd-context">(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span class="tdd-context">None vs league average</span>', unsafe_allow_html=True)


def _build_limited_pitcher_rows(
    pitchers: pd.DataFrame, injury_lookup: dict, use_priors: bool,
) -> list[dict]:
    """Build display rows for pitchers without full projections."""
    rows: list[dict] = []
    for _, row in pitchers.sort_values("player_name").iterrows():
        pid = int(row["player_id"])
        inj = injury_lookup.get(pid)
        name = row["player_name"]
        if inj and inj["missed_games"] > 0:
            sev = inj["severity"]
            tag = "[IL-60]" if sev == "major" else "[IL]" if sev == "significant" else "[DTD]"
            name = f"{tag} {name}"
        status = row.get("roster_status", "active")
        badge = "IL" if status.startswith("il_") else "Limited Data"
        r: dict[str, object] = {
            "Name": name,
            "Age": "",
            "Hand": "",
            "Rating": badge,
        }
        if use_priors:
            for label, key, _, _ in PITCHER_STATS:
                r[f"{label} ({PRIOR_SEASON})"] = "--"
            for label, _ in [("Whiff%", "whiff_rate"), ("Avg Velo", "avg_velo")]:
                r[label] = "--"
        else:
            for label, key, _, _ in PITCHER_STATS:
                r[label] = "--"
            r["Proj. K"] = "--"
        rows.append(r)
    return rows


def _build_limited_hitter_rows(
    hitters: pd.DataFrame, injury_lookup: dict, use_priors: bool,
) -> list[dict]:
    """Build display rows for hitters without full projections."""
    rows: list[dict] = []
    for _, row in hitters.sort_values("player_name").iterrows():
        pid = int(row["player_id"])
        inj = injury_lookup.get(pid)
        name = row["player_name"]
        if inj and inj["missed_games"] > 0:
            sev = inj["severity"]
            tag = "[IL-60]" if sev == "major" else "[IL]" if sev == "significant" else "[DTD]"
            name = f"{tag} {name}"
        pos = row.get("primary_position", "")
        status = row.get("roster_status", "active")
        badge = "IL" if status.startswith("il_") else "Limited Data"
        r: dict[str, object] = {
            "Name": name,
            "Age": "",
            "Bats": "",
            "Rating": f"{badge} ({pos})" if pos else badge,
        }
        if use_priors:
            for label, key, _, _ in HITTER_STATS:
                r[f"{label} ({PRIOR_SEASON})"] = "--"
            for label, _ in [("Whiff%", "whiff_rate"), ("Avg EV", "avg_exit_velo"), ("Hard-Hit%", "hard_hit_pct")]:
                r[label] = "--"
        else:
            for label, key, _, _ in HITTER_STATS:
                r[label] = "--"
            for c_label, _ in [("Proj. HR", "total_hr"), ("Proj. BB", "total_bb")]:
                r[c_label] = "--"
        rows.append(r)
    return rows


def _build_pitcher_rows(
    pitchers: pd.DataFrame, injury_lookup: dict, use_priors: bool,
) -> list[dict]:
    """Build display rows for a set of pitchers."""
    rows: list[dict] = []
    # Diamond rating — always derived from tdd_value_score
    from lib.diamond_rating import score_to_diamonds
    _p_ranks = load_rankings("pitchers") if "tdd_value_score" not in pitchers.columns else pd.DataFrame()
    if not _p_ranks.empty and "tdd_value_score" in _p_ranks.columns:
        pitchers = pitchers.merge(
            _p_ranks[["pitcher_id", "tdd_value_score"]].drop_duplicates(),
            on="pitcher_id", how="left",
        )
    sort_col = "tdd_value_score" if "tdd_value_score" in pitchers.columns else "composite_score"
    for _, row in pitchers.sort_values(sort_col, ascending=False).iterrows():
        pid = int(row["pitcher_id"])
        inj = injury_lookup.get(pid)
        name = row["pitcher_name"]
        if inj and inj["missed_games"] > 0:
            sev = inj["severity"]
            tag = "[IL-60]" if sev == "major" else "[IL]" if sev == "significant" else "[DTD]"
            name = f"{tag} {name}"
        tvs = row.get("tdd_value_score")
        rating_str = f"{score_to_diamonds(tvs):.1f}" if pd.notna(tvs) else diamond_rating_text_composite(row.get("composite_score", 0))
        r: dict[str, object] = {
            "Name": name,
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Hand": row.get("pitch_hand", ""),
            "Rating": rating_str,
        }
        if use_priors:
            for label, key, _, _ in PITCHER_STATS:
                obs_col = f"observed_{key}"
                if obs_col in row.index and pd.notna(row.get(obs_col)):
                    r[f"{label} ({PRIOR_SEASON})"] = fmt_stat(row[obs_col], key)
                else:
                    r[f"{label} ({PRIOR_SEASON})"] = "--"
            for label, key in [("Whiff%", "whiff_rate"), ("Avg Velo", "avg_velo")]:
                if key in row.index and pd.notna(row.get(key)):
                    r[label] = fmt_stat(row[key], key)
                else:
                    r[label] = "--"
        else:
            for label, key, _, _ in PITCHER_STATS:
                proj_col = f"projected_{key}"
                delta_col = f"delta_{key}"
                if proj_col in row.index and pd.notna(row.get(proj_col)):
                    proj_val = fmt_stat(row[proj_col], key)
                    delta_pp = row[delta_col] * 100
                    r[label] = f"{proj_val} ({delta_pp:+.1f})" if abs(delta_pp) >= 0.05 else proj_val
                else:
                    r[label] = "--"
            if "total_k_mean" in row.index and pd.notna(row.get("total_k_mean")):
                r["Proj. K"] = int(round(row["total_k_mean"]))
            else:
                r["Proj. K"] = "--"
        rows.append(r)
    return rows


def _render_roster_tab(
    selected_team: str,
    team_hitters: pd.DataFrame,
    team_pitchers: pd.DataFrame,
    h_proj: pd.DataFrame,
    p_proj: pd.DataFrame,
    injury_lookup: dict,
    *,
    missing_hitters: pd.DataFrame | None = None,
    missing_pitchers: pd.DataFrame | None = None,
) -> None:
    """Roster tab — the original overview content (projections, tables)."""
    if missing_hitters is None:
        missing_hitters = pd.DataFrame()
    if missing_pitchers is None:
        missing_pitchers = pd.DataFrame()
    view_mode = st.radio(
        "View",
        [f"{CURRENT_SEASON} Projections", f"{PRIOR_SEASON} Priors (Observed)"],
        horizontal=True, key="team_view_mode",
    )
    use_priors = view_mode == f"{PRIOR_SEASON} Priors (Observed)"

    if use_priors:
        _p_k_col = "observed_k_rate"
        _p_bb_col = "observed_bb_rate"
        _view_label = f"{PRIOR_SEASON} Observed"
    else:
        _p_k_col = "projected_k_rate"
        _p_bb_col = "projected_bb_rate"
        _view_label = f"{CURRENT_SEASON} Projected"

    # ── Pitching staff profile — split by role ──────────────────────
    if not team_pitchers.empty and not p_proj.empty:
        team_sp = team_pitchers[team_pitchers["is_starter"] == True]  # noqa: E712
        team_rp = team_pitchers[team_pitchers["is_starter"] == False]  # noqa: E712
        lg_sp = p_proj[p_proj["is_starter"] == True]  # noqa: E712
        lg_rp = p_proj[p_proj["is_starter"] == False]  # noqa: E712

        if not team_sp.empty:
            st.markdown(f"### Rotation Profile ({_view_label})")
            _render_staff_metrics(team_sp, lg_sp, _p_k_col, _p_bb_col)
        if not team_rp.empty:
            st.markdown(f"### Bullpen Profile ({_view_label})")
            _render_staff_metrics(team_rp, lg_rp, _p_k_col, _p_bb_col)

    # ── Injured players ─────────────────────────────────────────────
    inj_df_full = load_preseason_injuries()
    if not inj_df_full.empty:
        team_inj = inj_df_full[
            (inj_df_full["team_abbr"] == selected_team)
            & (inj_df_full["est_missed_games"] > 0)
        ].sort_values("est_missed_games", ascending=False)
    else:
        team_inj = pd.DataFrame()

    if not team_inj.empty:
        st.markdown("### Injured Players")
        inj_rows = []
        for _, row in team_inj.iterrows():
            inj_rows.append({
                "Player": row["player_name"], "Pos": row["position"],
                "Injury": row["injury"], "Status": row["status"],
                "Est. Return": row["est_return_date"],
                "~Games Missed": int(row["est_missed_games"]),
            })
        st.dataframe(pd.DataFrame(inj_rows), width='stretch', hide_index=True)

    # ── Pitchers tables ─────────────────────────────────────────────
    # Split missing pitchers by role using roster is_starter flag
    if not missing_pitchers.empty and "is_starter" in missing_pitchers.columns:
        miss_sp = missing_pitchers[missing_pitchers["is_starter"] == True]  # noqa: E712
        miss_rp = missing_pitchers[missing_pitchers["is_starter"] == False]  # noqa: E712
    elif not missing_pitchers.empty:
        miss_sp = missing_pitchers[missing_pitchers["primary_position"] == "SP"]
        miss_rp = missing_pitchers[missing_pitchers["primary_position"] != "SP"]
    else:
        miss_sp = pd.DataFrame()
        miss_rp = pd.DataFrame()

    has_any_pitchers = not team_pitchers.empty or not missing_pitchers.empty

    if not has_any_pitchers:
        st.markdown("### Pitchers")
        st.info("No pitcher data for this team.")
    else:
        team_sp_tbl = team_pitchers[team_pitchers["is_starter"] == True] if not team_pitchers.empty else pd.DataFrame()  # noqa: E712
        team_rp_tbl = team_pitchers[team_pitchers["is_starter"] == False] if not team_pitchers.empty else pd.DataFrame()  # noqa: E712

        st.markdown("### Starting Rotation")
        sp_rows = _build_pitcher_rows(team_sp_tbl, injury_lookup, use_priors) if not team_sp_tbl.empty else []
        sp_rows += _build_limited_pitcher_rows(miss_sp, injury_lookup, use_priors) if not miss_sp.empty else []
        if sp_rows:
            st.dataframe(pd.DataFrame(sp_rows), width='stretch', hide_index=True)
        else:
            st.info("No starters for this team.")

        st.markdown("### Bullpen")
        rp_rows = _build_pitcher_rows(team_rp_tbl, injury_lookup, use_priors) if not team_rp_tbl.empty else []
        rp_rows += _build_limited_pitcher_rows(miss_rp, injury_lookup, use_priors) if not miss_rp.empty else []
        if rp_rows:
            st.dataframe(pd.DataFrame(rp_rows), width='stretch', hide_index=True)
        else:
            st.info("No relievers for this team.")

    # ── Hitters table ───────────────────────────────────────────────
    st.markdown("### Hitters")
    has_any_hitters = not team_hitters.empty or not missing_hitters.empty
    if not has_any_hitters:
        st.info("No hitter data for this team.")
    else:
        h_rows = []
        if not team_hitters.empty:
            # Diamond rating — always derived from tdd_value_score
            from lib.diamond_rating import score_to_diamonds
            _h_ranks = load_rankings("hitters") if "tdd_value_score" not in team_hitters.columns else pd.DataFrame()
            if not _h_ranks.empty and "tdd_value_score" in _h_ranks.columns:
                team_hitters = team_hitters.merge(
                    _h_ranks[["batter_id", "tdd_value_score"]].drop_duplicates(),
                    on="batter_id", how="left",
                )
            sort_col = "tdd_value_score" if "tdd_value_score" in team_hitters.columns else "composite_score"
            for _, row in team_hitters.sort_values(sort_col, ascending=False).iterrows():
                pid = int(row["batter_id"])
                inj = injury_lookup.get(pid)
                name = row["batter_name"]
                if inj and inj["missed_games"] > 0:
                    sev = inj["severity"]
                    tag = "[IL-60]" if sev == "major" else "[IL]" if sev == "significant" else "[DTD]"
                    name = f"{tag} {name}"
                tvs = row.get("tdd_value_score")
                rating_str = f"{score_to_diamonds(tvs):.1f}" if pd.notna(tvs) else diamond_rating_text_composite(row.get("composite_score", 0))
                r: dict[str, object] = {
                    "Name": name,
                    "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
                    "Bats": row.get("batter_stand", ""),
                    "Rating": rating_str,
                }
                if use_priors:
                    for label, key, _, _ in HITTER_STATS:
                        obs_col = f"observed_{key}"
                        if obs_col in row.index and pd.notna(row.get(obs_col)):
                            r[f"{label} ({PRIOR_SEASON})"] = fmt_stat(row[obs_col], key)
                        else:
                            r[f"{label} ({PRIOR_SEASON})"] = "--"
                    for label, key in [("Whiff%", "whiff_rate"), ("Avg EV", "avg_exit_velo"), ("Hard-Hit%", "hard_hit_pct")]:
                        if key in row.index and pd.notna(row.get(key)):
                            r[label] = fmt_stat(row[key], key)
                        else:
                            r[label] = "--"
                else:
                    for label, key, _, _ in HITTER_STATS:
                        proj_col = f"projected_{key}"
                        delta_col = f"delta_{key}"
                        if proj_col in row.index and pd.notna(row.get(proj_col)):
                            proj_val = fmt_stat(row[proj_col], key)
                            delta_pp = row[delta_col] * 100
                            r[label] = f"{proj_val} ({delta_pp:+.1f})" if abs(delta_pp) >= 0.05 else proj_val
                        else:
                            r[label] = "--"
                    for c_label, c_prefix in [("Proj. HR", "total_hr"), ("Proj. BB", "total_bb")]:
                        mean_col = f"{c_prefix}_mean"
                        if mean_col in row.index and pd.notna(row.get(mean_col)):
                            r[c_label] = int(round(row[mean_col]))
                        else:
                            r[c_label] = "--"
                h_rows.append(r)
        # Append limited-data hitters at the end
        h_rows += _build_limited_hitter_rows(missing_hitters, injury_lookup, use_priors) if not missing_hitters.empty else []
        st.dataframe(pd.DataFrame(h_rows), width='stretch', hide_index=True)

    st.caption(
        "Strengths/weaknesses compare team averages to league average. "
        + (
            f"Showing {PRIOR_SEASON} observed stats (priors for the Bayesian model)."
            if use_priors
            else f"Deltas shown in parentheses (pp vs {PRIOR_SEASON})."
        )
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN PAGE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def page_team_overview() -> None:
    """Team-level view with team intelligence, roster, depth chart, and trade sim."""
    st.markdown(
        '<div class="section-header">Team Overview</div>',
        unsafe_allow_html=True,
    )

    teams_df = load_roster()
    if teams_df.empty:
        st.warning("No team data found. Run precompute first.")
        return

    team_lookup = get_team_lookup()
    injury_lookup = get_injury_lookup()

    all_teams = sorted(
        teams_df["team_abbr"].replace("", pd.NA).dropna().unique().tolist()
    )
    qp_team = st.query_params.get("team", "")
    default_team_idx = 0
    if qp_team in all_teams:
        default_team_idx = all_teams.index(qp_team)
    selected_team = st.selectbox(
        "Select team", all_teams, index=default_team_idx, key="team_select",
    )
    st.query_params["team"] = selected_team

    # Get all player IDs for this team
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )

    # Load projections
    h_proj = load_projections("hitter")
    p_proj = load_projections("pitcher")
    h_count = load_counting("hitter")
    p_count = load_counting("pitcher")

    # Filter to team
    team_hitters = h_proj[h_proj["batter_id"].isin(team_pids)].copy()
    team_pitchers = p_proj[p_proj["pitcher_id"].isin(team_pids)].copy()

    # Merge counting stats
    if not h_count.empty:
        h_merge_cols = ["batter_id"] + [
            c for c in h_count.columns if c.endswith("_mean") or c.startswith("actual_")
        ]
        available = [c for c in h_merge_cols if c in h_count.columns]
        team_hitters = team_hitters.merge(h_count[available], on="batter_id", how="left")

    if not p_count.empty:
        p_merge_cols = ["pitcher_id"] + [
            c for c in p_count.columns if c.endswith("_mean") or c.startswith("actual_")
        ]
        available = [c for c in p_merge_cols if c in p_count.columns]
        team_pitchers = team_pitchers.merge(p_count[available], on="pitcher_id", how="left")

    # ── Identify rostered players without projections ────────────
    # Only works when roster.parquet is available (has roster_status, primary_position).
    # Falls back gracefully when using player_teams.parquet (no extra columns).
    projected_pids = set(team_hitters["batter_id"]) | set(team_pitchers["pitcher_id"])
    team_roster = teams_df[teams_df["team_abbr"] == selected_team].copy()
    has_roster_detail = (
        "roster_status" in team_roster.columns
        and "primary_position" in team_roster.columns
    )
    if has_roster_detail:
        missing_roster = team_roster[
            (~team_roster["player_id"].isin(projected_pids))
            & (team_roster["roster_status"].isin(_MLB_ROSTER_STATUSES))
        ].copy()
        missing_pitchers = missing_roster[
            missing_roster["primary_position"].isin(_PITCHER_POSITIONS)
        ]
        missing_hitters = missing_roster[
            ~missing_roster["primary_position"].isin(_PITCHER_POSITIONS)
        ]
    else:
        missing_pitchers = pd.DataFrame()
        missing_hitters = pd.DataFrame()

    # ── Header ──────────────────────────────────────────────────────
    # Enrich header with ELO + tier
    rankings = load_team_rankings()
    team_rank_row = rankings[rankings["abbreviation"] == selected_team] if not rankings.empty else pd.DataFrame()

    tier_html = ""
    elo_html = ""
    proj_w_html = ""
    if not team_rank_row.empty:
        tr = team_rank_row.iloc[0]
        tier = tr.get("tier", "")
        tier_color = _TIER_COLORS.get(tier, SLATE)
        if tier:
            tier_html = (
                f'<span class="tdd-badge" style="background:{tier_color}22; color:{tier_color}; '
                f'border:1px solid {tier_color}44; margin-left:10px;">{tier}</span>'
            )
        rank_val = tr.get("rank")
        if pd.notna(rank_val):
            elo_html = (
                f'<span style="color:var(--tdd-gold); font-weight:700; '
                f'margin-left:8px;">#{int(rank_val)}</span>'
            )
    _rec = _standings.get(selected_team)
    if _rec:
        proj_w_html = (
            f'<span class="tdd-context" style="margin-left:10px;">'
            f'<span style="color:var(--tdd-gold); font-weight:700;">{_rec[0]}-{_rec[1]}</span></span>'
        )

    _inj_full = load_preseason_injuries()
    n_injured = len(_inj_full[
        (_inj_full["team_abbr"] == selected_team) & (_inj_full["est_missed_games"] > 0)
    ]) if not _inj_full.empty else 0

    total_pitchers = len(team_pitchers) + len(missing_pitchers)
    total_hitters = len(team_hitters) + len(missing_hitters)
    team_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title"><span class="tdd-team-abbr" data-team="{selected_team}">{selected_team}</span>{elo_html}{tier_html}</div>'
        f'<div class="brand-subtitle">'
        f'{total_pitchers} pitchers | {total_hitters} hitters | '
        f'{n_injured} injured{proj_w_html}</div>'
        f'</div>'
        f'<div class="tdd-context">'
        f'{CURRENT_SEASON} Season</div>'
        f'</div>'
    )
    st.markdown(team_header_html, unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────
    tab_profile, tab_depth, tab_trade = st.tabs(
        ["Team Profile", "Depth Chart", "Trade Simulator"],
    )

    with tab_profile:
        _render_team_profile_tab(selected_team, teams_df)

    with tab_depth:
        _render_depth_chart_tab(
            selected_team, teams_df,
            team_pitchers=team_pitchers,
            p_proj=p_proj,
        )

    with tab_trade:
        _render_trade_simulator_tab(selected_team, teams_df)
