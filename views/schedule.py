"""Schedule page — Today's Games (with live updates) + Game Browser combined."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK, DARK_CARD, DARK_BORDER,
    POSITIVE, NEGATIVE, DASHBOARD_DIR, PRIOR_SEASON, TRAINING_RANGE,
    SCHEDULE_REFRESH_MINUTES, GAME_WINDOW_START_HOUR, GAME_WINDOW_END_HOUR,
)
from services.data_loader import (
    load_todays_games, load_todays_sims, load_todays_lineups,
    load_update_metadata, load_pitcher_arsenal, load_hitter_vulnerability,
    load_projections, load_counting, load_game_info, load_player_teams,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_k_samples, load_bb_samples, load_hr_samples, load_bf_priors,
    load_pitcher_offerings, load_cluster_metadata,
    load_archetype_matchup_matrix,
    load_exit_model, load_pitcher_pitch_count_features,
    load_batter_pitch_count_features, load_tto_profiles,
    load_pitcher_exit_tendencies,
    fetch_live_schedule, fetch_live_lineups,
)
from utils.helpers import get_team_lookup
from components.metric_cards import metric_card
from components.charts import create_game_stat_fig
from components.diamond_rating import diamond_rating_html_composite


def _is_game_window() -> bool:
    """Check if current ET time is within the game window."""
    # Approximate ET as UTC-4 (EDT during baseball season)
    utc_now = datetime.now(timezone.utc)
    et_now = utc_now - timedelta(hours=4)
    return GAME_WINDOW_START_HOUR <= et_now.hour < GAME_WINDOW_END_HOUR


def _staleness_indicator(ts_str: str | None) -> str:
    """Return colored staleness indicator HTML based on timestamp age."""
    if not ts_str:
        return f'<span style="color:{SLATE};">unknown</span>'
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        minutes = age.total_seconds() / 60
        if minutes < 15:
            color = SAGE
        elif minutes < 60:
            color = GOLD
        else:
            color = EMBER
        return (
            f'<span style="color:{color};">●</span> '
            f'{ts.strftime("%I:%M %p")}'
        )
    except Exception:
        return f'<span style="color:{SLATE};">unknown</span>'


def _render_todays_games() -> None:
    """Today's MLB games with matchup analysis and K prop projections.

    During game windows, auto-refreshes schedule/lineup data from MLB API
    every 10 minutes via st.fragment. Projection data stays cached from
    the morning update.
    """
    meta = load_update_metadata()
    sims = load_todays_sims()

    in_game_window = _is_game_window()

    # --- Live schedule data (fragment auto-reruns during game windows) ---
    if in_game_window:
        _render_live_schedule_fragment(meta, sims)
    else:
        _render_cached_schedule(meta, sims)


@st.fragment(run_every=timedelta(minutes=SCHEDULE_REFRESH_MINUTES))
def _render_live_schedule_fragment(
    meta: dict,
    sims: pd.DataFrame,
) -> None:
    """Fragment that auto-reruns to poll MLB API for fresh schedule data."""
    schedule = fetch_live_schedule()
    lineups = fetch_live_lineups(schedule) if not schedule.empty else pd.DataFrame()

    live_ts = datetime.now(timezone.utc).isoformat()
    proj_ts = meta.get("last_updated")

    # Refresh button
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.markdown(
            f'<div style="color:{SLATE}; font-size:0.85rem;">'
            f'Projections from: {_staleness_indicator(proj_ts)} '
            f'&nbsp;|&nbsp; Schedule updated: {_staleness_indicator(live_ts)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("Refresh", key="schedule_refresh"):
            st.cache_data.clear()
            st.rerun()

    _render_schedule_cards(schedule, sims, lineups, meta)


def _render_cached_schedule(meta: dict, sims: pd.DataFrame) -> None:
    """Render schedule from cached parquets (outside game window)."""
    schedule = load_todays_games()
    lineups = load_todays_lineups()

    proj_ts = meta.get("last_updated")
    st.markdown(
        f'<div style="color:{SLATE}; font-size:0.85rem;">'
        f'Projections from: {_staleness_indicator(proj_ts)} '
        f'&nbsp;|&nbsp; <span style="color:{SLATE};">Outside game window — using cached data</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _render_schedule_cards(schedule, sims, lineups, meta)


def _build_projection_lookup() -> dict:
    """Build pitcher_id → projection dict from pitcher_projections.parquet."""
    proj = load_projections("pitcher")
    if proj.empty or "pitcher_id" not in proj.columns:
        return {}
    lookup = {}
    for _, row in proj.iterrows():
        lookup[int(row["pitcher_id"])] = {
            "projected_k_rate": row.get("projected_k_rate"),
            "pitcher_name": row.get("pitcher_name", ""),
            "projected_bb_rate": row.get("projected_bb_rate"),
            "projected_hr_per_bf": row.get("projected_hr_per_bf"),
            "composite_score": row.get("composite_score"),
        }
    return lookup


def _render_schedule_cards(
    schedule: pd.DataFrame,
    sims: pd.DataFrame,
    lineups: pd.DataFrame,
    meta: dict,
) -> None:
    """Render game cards from schedule + sim + lineup data."""
    # Pre-build archetype + projection lookups for enrichment
    _h_arch = load_hitter_archetypes()
    _p_arch = load_pitcher_archetypes()
    _h_proj = load_projections("hitter")
    _h_count = load_counting("hitter")

    # Hitter archetype lookup: batter_id → archetype_name
    _h_arch_lookup: dict[int, str] = {}
    if not _h_arch.empty:
        for _, _r in _h_arch.iterrows():
            _h_arch_lookup[int(_r["batter_id"])] = _r["archetype_name"]

    # Pitcher archetype lookup: pitcher_id → archetype_name
    _p_arch_lookup: dict[int, str] = {}
    if not _p_arch.empty:
        for _, _r in _p_arch.iterrows():
            _p_arch_lookup[int(_r["pitcher_id"])] = _r["archetype_name"]

    # Hitter projection lookup: batter_id → {k_rate, bb_rate, hr, composite_score, ...}
    _h_stat_lookup: dict[int, dict] = {}
    if not _h_proj.empty:
        for _, _r in _h_proj.iterrows():
            _h_stat_lookup[int(_r["batter_id"])] = {
                "k_rate": _r.get("projected_k_rate"),
                "bb_rate": _r.get("projected_bb_rate"),
                "composite_score": _r.get("composite_score"),
            }
    if not _h_count.empty and "batter_id" in _h_count.columns:
        for _, _r in _h_count.iterrows():
            bid = int(_r["batter_id"])
            counting = {
                "hr": _r.get("total_hr_mean"),
                "total_k": _r.get("total_k_mean"),
                "total_bb": _r.get("total_bb_mean"),
                "total_hr": _r.get("total_hr_mean"),
            }
            if bid in _h_stat_lookup:
                _h_stat_lookup[bid].update(counting)
            else:
                _h_stat_lookup[bid] = {
                    "k_rate": None, "bb_rate": None,
                    "composite_score": None, **counting,
                }

    # Game simulator + matchup scoring data
    _k_samples_dict = load_k_samples()
    _bb_samples_dict = load_bb_samples()
    _hr_samples_dict = load_hr_samples()
    _bf_priors = load_bf_priors()
    _arsenal_df = load_pitcher_arsenal()
    _vuln_df = load_hitter_vulnerability(career=True)
    _offerings_df = load_pitcher_offerings()
    _cluster_meta_df = load_cluster_metadata()
    _matchup_matrix_df = load_archetype_matchup_matrix()

    # Game sim v2 component data
    _exit_model = load_exit_model()
    _pitcher_pc = load_pitcher_pitch_count_features()
    _batter_pc = load_batter_pitch_count_features()
    _tto_profiles = load_tto_profiles()
    _exit_tendencies = load_pitcher_exit_tendencies()

    # Always build projection lookup (used by cards + drilldown)
    proj_lookup = _build_projection_lookup()

    if schedule.empty:
        game_date = meta.get("game_date", "")
        st.info(
            f"No games loaded{f' for {game_date}' if game_date else ''}. "
            "Projections update daily during the season."
        )
        return

    game_date = schedule.iloc[0].get("game_date", "")
    n_games = len(schedule)

    # Check if sims are stale (different date than schedule)
    sims_stale = False
    if not sims.empty and "game_pk" in sims.columns:
        sims_gpks = set(sims["game_pk"].tolist())
        sched_gpks = set(schedule["game_pk"].tolist())
        sims_stale = len(sims_gpks & sched_gpks) == 0

    if sims_stale:
        st.markdown(
            f'<div style="color:{EMBER}; font-size:0.85rem; margin-bottom:0.5rem;">'
            f'Simulations are from a previous date — showing base projections only.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="color:{SLATE}; font-size:0.9rem; margin-bottom:1rem;">'
        f'{game_date} | {n_games} games'
        f'</div>',
        unsafe_allow_html=True,
    )

    all_teams = sorted(set(
        schedule["away_abbr"].dropna().tolist() +
        schedule["home_abbr"].dropna().tolist()
    ))
    team_filter = st.selectbox(
        "Filter by team",
        ["All Teams"] + all_teams,
        key="today_team_filter",
    )

    if team_filter != "All Teams":
        schedule = schedule[
            (schedule["away_abbr"] == team_filter) |
            (schedule["home_abbr"] == team_filter)
        ]

    for _, game in schedule.iterrows():
        gpk = game["game_pk"]
        away_abbr = game.get("away_abbr", "?")
        home_abbr = game.get("home_abbr", "?")
        game_time = game.get("game_time", "")
        status = game.get("status", "")

        game_sims = sims[sims["game_pk"] == gpk] if not sims.empty else pd.DataFrame()
        away_sim = game_sims[game_sims["side"] == "away"].iloc[0] if not game_sims.empty and (game_sims["side"] == "away").any() else None
        home_sim = game_sims[game_sims["side"] == "home"].iloc[0] if not game_sims.empty and (game_sims["side"] == "home").any() else None

        # Status styling — highlight live games
        status_str = ""
        if status:
            if status in ("In Progress", "Live"):
                status_str = f' — <span style="color:{SAGE};">● {status}</span>'
            elif "Final" in status:
                status_str = f' — <span style="color:{SLATE};">{status}</span>'
            elif "Scheduled" not in status:
                status_str = f" — {status}"

        # Game context line: venue, umpire, weather
        venue_name = game.get("venue_name", "")
        hp_ump = game.get("hp_umpire_name", "")
        wx_temp = game.get("weather_temp", "")
        wx_cond = game.get("weather_condition", "")
        ctx_parts = []
        if venue_name:
            ctx_parts.append(venue_name)
        if hp_ump:
            ctx_parts.append(f"HP: {hp_ump}")
        if wx_temp:
            wx_str = f"{wx_temp}°F"
            if wx_cond:
                wx_str += f", {wx_cond}"
            ctx_parts.append(wx_str)
        ctx_line = (
            f'<div style="color:{SLATE}; font-size:0.8rem; margin-top:2px;">'
            f'{" · ".join(ctx_parts)}</div>'
            if ctx_parts else ""
        )

        st.markdown(
            f'<div style="background:{DARK_CARD}; border:1px solid {DARK_BORDER}; '
            f'border-radius:8px; padding:1rem 1.5rem; margin-bottom:1rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="color:{GOLD}; font-size:1.1rem; font-weight:700;">'
            f'{away_abbr} @ {home_abbr}</span>'
            f'<span style="color:{SLATE}; font-size:0.85rem;">{game_time}{status_str}</span>'
            f'</div>'
            f'{ctx_line}'
            f'<div style="margin-top:0.6rem;"></div>',
            unsafe_allow_html=True,
        )

        col_a, col_h = st.columns(2)

        for col, sim, side_label, pitcher_name_field, pitcher_id_field in [
            (col_a, away_sim, f"{away_abbr} SP", "away_pitcher_name", "away_pitcher_id"),
            (col_h, home_sim, f"{home_abbr} SP", "home_pitcher_name", "home_pitcher_id"),
        ]:
            with col:
                pp_name = game.get(pitcher_name_field, "TBD")
                if not pp_name:
                    pp_name = "TBD"

                # Append pitcher archetype if available
                _pp_id = game.get(pitcher_id_field)
                _pp_arch = _p_arch_lookup.get(int(_pp_id)) if pd.notna(_pp_id) else None
                _pp_arch_tag = f' <span style="color:{SLATE}; font-size:0.8rem;">({_pp_arch})</span>' if _pp_arch else ""

                if sim is not None:
                    k_rate_pct = sim["projected_k_rate"] * 100
                    exp_k = sim["expected_k"]
                    has_lineup = sim.get("has_lineup", False)
                    lineup_tag = "" if has_lineup else " (no lineup)"

                    # Multi-stat expectations line
                    stat_parts = [f'E[K]: {exp_k:.1f}']
                    exp_bb = sim.get("expected_bb")
                    exp_h = sim.get("expected_h")
                    exp_ip = sim.get("expected_ip")
                    if pd.notna(exp_bb):
                        stat_parts.append(f'BB: {exp_bb:.1f}')
                    if pd.notna(exp_h):
                        stat_parts.append(f'H: {exp_h:.1f}')
                    if pd.notna(exp_ip):
                        stat_parts.append(f'IP: {exp_ip:.1f}')
                    stat_line = " | ".join(stat_parts)

                    # K prop lines
                    p_lines = []
                    for line_col in ["p_over_5_5", "p_over_6_5", "p_over_7_5"]:
                        if line_col in sim.index and pd.notna(sim.get(line_col)):
                            line_val = line_col.replace("p_over_", "").replace("_", ".")
                            p = sim[line_col] * 100
                            color = POSITIVE if p > 55 else NEGATIVE if p < 45 else SLATE
                            p_lines.append(
                                f'<span style="color:{color};">'
                                f'O{line_val}: {p:.0f}%</span>'
                            )
                    p_line_html = " | ".join(p_lines) if p_lines else ""

                    # Fantasy line
                    dk_mean = sim.get("dk_mean")
                    fantasy_html = ""
                    if pd.notna(dk_mean):
                        fantasy_html = (
                            f'<div style="font-size:0.8rem; color:{SLATE}; margin-top:2px;">'
                            f'DK: {dk_mean:.1f} pts</div>'
                        )

                    st.markdown(
                        f'<div style="font-size:0.95rem; font-weight:600; color:{CREAM};">'
                        f'{side_label}: {pp_name}{_pp_arch_tag}</div>'
                        f'<div style="font-size:0.8rem; color:{SLATE}; margin-top:2px;">'
                        f'K%: {k_rate_pct:.1f}% | {stat_line}{lineup_tag}</div>'
                        f'<div style="font-size:0.8rem; margin-top:2px;">{p_line_html}</div>'
                        f'{fantasy_html}',
                        unsafe_allow_html=True,
                    )
                else:
                    # Fallback: show projection K% from pitcher_projections
                    pid = game.get(pitcher_id_field)
                    proj_info = proj_lookup.get(int(pid)) if pd.notna(pid) else None

                    if proj_info and pd.notna(proj_info.get("projected_k_rate")):
                        k_pct = proj_info["projected_k_rate"] * 100
                        bb_pct = proj_info.get("projected_bb_rate")
                        bb_html = f" | BB%: {bb_pct * 100:.1f}%" if pd.notna(bb_pct) else ""
                        st.markdown(
                            f'<div style="font-size:0.95rem; font-weight:600; color:{CREAM};">'
                            f'{side_label}: {pp_name}{_pp_arch_tag}</div>'
                            f'<div style="font-size:0.8rem; color:{SLATE}; margin-top:2px;">'
                            f'K%: {k_pct:.1f}%{bb_html} (projection)</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="font-size:0.95rem; font-weight:600; color:{CREAM};">'
                            f'{side_label}: {pp_name}{_pp_arch_tag}</div>'
                            f'<div style="font-size:0.8rem; color:{SLATE}; margin-top:2px;">'
                            f'K%: 22.0% | BB%: 8.0% (lg avg)</div>',
                            unsafe_allow_html=True,
                        )

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander(f"Game Details — {away_abbr} @ {home_abbr}"):
            _render_game_drilldown(
                game, lineups, _h_arch_lookup, _p_arch_lookup, _h_stat_lookup,
                _k_samples_dict, _bf_priors, _arsenal_df, _vuln_df,
                proj_lookup, gpk,
                _offerings_df, _cluster_meta_df, _matchup_matrix_df,
                bb_samples_dict=_bb_samples_dict,
                hr_samples_dict=_hr_samples_dict,
                exit_model=_exit_model,
                pitcher_pc=_pitcher_pc,
                batter_pc=_batter_pc,
                tto_profiles=_tto_profiles,
                exit_tendencies=_exit_tendencies,
            )

    if not sims.empty and not sims_stale:
        st.markdown("---")
        st.markdown(
            f'<div style="color:{SLATE}; font-size:0.8rem;">'
            f'{len(sims)} pitchers simulated | '
            f'{sims["has_lineup"].sum()} with lineup data | '
            f'10,000 Monte Carlo draws per pitcher</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Game Drill-Down (Phase 2: Game Center)
# ---------------------------------------------------------------------------

def _render_game_drilldown(
    game: pd.Series,
    lineups: pd.DataFrame,
    h_arch_lookup: dict[int, str],
    p_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    k_samples_dict: dict[str, np.ndarray],
    bf_priors: pd.DataFrame,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    proj_lookup: dict[int, dict],
    gpk: int,
    offerings_df: pd.DataFrame | None = None,
    cluster_meta_df: pd.DataFrame | None = None,
    matchup_matrix_df: pd.DataFrame | None = None,
    bb_samples_dict: dict[str, np.ndarray] | None = None,
    hr_samples_dict: dict[str, np.ndarray] | None = None,
    exit_model: object | None = None,
    pitcher_pc: pd.DataFrame | None = None,
    batter_pc: pd.DataFrame | None = None,
    tto_profiles: pd.DataFrame | None = None,
    exit_tendencies: pd.DataFrame | None = None,
) -> None:
    """Rich game drill-down: lineup matchups, archetype analysis, and game simulator."""
    away_abbr = game.get("away_abbr", "?")
    home_abbr = game.get("home_abbr", "?")
    game_lu = lineups[lineups["game_pk"] == gpk] if not lineups.empty else pd.DataFrame()

    # Build per-side info for both tabs
    sides = []
    for side, opp_side in [("away", "home"), ("home", "away")]:
        pitcher_name = game.get(f"{side}_pitcher_name") or "TBD"
        pitcher_id_raw = game.get(f"{side}_pitcher_id")
        pid = int(pitcher_id_raw) if pd.notna(pitcher_id_raw) else None
        side_team_id = game.get(f"{side}_team_id")
        opp_team_id = game.get(f"{opp_side}_team_id")
        opp_abbr = game.get(f"{opp_side}_abbr", "?")
        side_abbr = game.get(f"{side}_abbr", "?")
        opp_lu = (
            game_lu[game_lu["team_id"] == opp_team_id].sort_values("batting_order")
            if not game_lu.empty and pd.notna(opp_team_id) else pd.DataFrame()
        )
        own_lu = (
            game_lu[game_lu["team_id"] == side_team_id].sort_values("batting_order")
            if not game_lu.empty and pd.notna(side_team_id) else pd.DataFrame()
        )
        sides.append({
            "side": side,
            "abbr": side_abbr,
            "opp_abbr": opp_abbr,
            "pitcher_name": pitcher_name,
            "pitcher_id": pid,
            "pitcher_arch": p_arch_lookup.get(pid) if pid else None,
            "pitcher_proj": proj_lookup.get(pid, {}) if pid else {},
            "opp_lineup": opp_lu,
            "own_lineup": own_lu,
        })

    # Resolve umpire + weather context for this game
    game_context = _resolve_game_context(game)

    tab_matchups, tab_hitters, tab_arch, tab_sim = st.tabs(
        ["Lineup Matchups", "Hitter Projections", "Archetype Analysis", "Game Simulator"]
    )

    with tab_matchups:
        _render_matchup_tab(sides, h_arch_lookup, h_stat_lookup,
                            arsenal_df, vuln_df, gpk)

    with tab_hitters:
        _render_hitter_projections_tab(sides, h_arch_lookup, h_stat_lookup, gpk)

    with tab_arch:
        _render_archetype_tab(
            sides, h_arch_lookup, p_arch_lookup,
            offerings_df if offerings_df is not None else pd.DataFrame(),
            cluster_meta_df if cluster_meta_df is not None else pd.DataFrame(),
            matchup_matrix_df if matchup_matrix_df is not None else pd.DataFrame(),
            gpk,
        )

    with tab_sim:
        _render_sim_tab(
            sides, h_stat_lookup, k_samples_dict,
            bf_priors, arsenal_df, vuln_df, gpk,
            game_context=game_context,
            bb_samples_dict=bb_samples_dict or {},
            hr_samples_dict=hr_samples_dict or {},
            exit_model=exit_model,
            pitcher_pc=pitcher_pc if pitcher_pc is not None else pd.DataFrame(),
            batter_pc=batter_pc if batter_pc is not None else pd.DataFrame(),
            tto_profiles=tto_profiles if tto_profiles is not None else pd.DataFrame(),
            exit_tendencies=exit_tendencies if exit_tendencies is not None else pd.DataFrame(),
        )


def _render_hitter_projections_tab(
    sides: list[dict],
    h_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    gpk: int,
) -> None:
    """Hitter projection cards for each team's lineup."""
    for side_info in sides:
        side_abbr = side_info["abbr"]
        own_lu = side_info["own_lineup"]

        st.markdown(
            f'<div style="color:{GOLD}; font-size:1rem; font-weight:600; '
            f'margin:0.5rem 0 0.3rem;">'
            f'{side_abbr} Lineup</div>',
            unsafe_allow_html=True,
        )

        if own_lu.empty:
            st.markdown(
                f'<div style="color:{SLATE}; font-size:0.85rem; '
                f'padding:0.5rem 0;">Lineup not yet available</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            continue

        name_col = "batter_name" if "batter_name" in own_lu.columns else "player_name"
        id_col = "batter_id" if "batter_id" in own_lu.columns else "player_id"

        cards_html: list[str] = []
        for _, brow in own_lu.head(9).iterrows():
            bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
            bname = brow.get(name_col, "Unknown")
            order = int(brow["batting_order"])

            # Look up projection data
            stats = h_stat_lookup.get(bid, {}) if bid else {}
            arch = h_arch_lookup.get(bid, "") if bid else ""

            composite = stats.get("composite_score")
            diamond_html = ""
            if pd.notna(composite):
                diamond_html = diamond_rating_html_composite(composite, size="sm")

            # Counting projections
            proj_k = stats.get("total_k")
            proj_bb = stats.get("total_bb")
            proj_hr = stats.get("total_hr")

            stat_parts: list[str] = []
            if pd.notna(proj_k):
                stat_parts.append(
                    f'<span style="color:{CREAM};">K: '
                    f'<strong>{proj_k:.0f}</strong></span>'
                )
            if pd.notna(proj_bb):
                stat_parts.append(
                    f'<span style="color:{CREAM};">BB: '
                    f'<strong>{proj_bb:.0f}</strong></span>'
                )
            if pd.notna(proj_hr):
                stat_parts.append(
                    f'<span style="color:{CREAM};">HR: '
                    f'<strong>{proj_hr:.0f}</strong></span>'
                )
            stat_line = (
                f' <span style="font-size:0.8rem;">{" | ".join(stat_parts)}</span>'
                if stat_parts
                else f' <span style="color:{SLATE}; font-size:0.8rem;">No projection</span>'
            )

            arch_tag = (
                f' <span style="color:{SLATE}; font-size:0.75rem;">({arch})</span>'
                if arch else ""
            )

            cards_html.append(
                f'<div style="display:flex; align-items:center; gap:0.4rem; '
                f'padding:0.35rem 0.6rem; flex-wrap:wrap; '
                f'border-bottom:1px solid {DARK_BORDER};">'
                f'<span style="color:{SLATE}; font-size:0.8rem; '
                f'min-width:1.2rem; text-align:right;">{order}</span>'
                f'<span style="color:{CREAM}; font-size:0.9rem; font-weight:500; '
                f'min-width:5rem; flex-shrink:1;">{bname}{arch_tag}</span>'
                f'{diamond_html}'
                f'{stat_line}'
                f'</div>'
            )

        if cards_html:
            st.markdown(
                f'<div style="background:{DARK_CARD}; border:1px solid {DARK_BORDER}; '
                f'border-radius:6px; margin-bottom:1rem; overflow:hidden;">'
                + "".join(cards_html)
                + '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")


def _render_matchup_tab(
    sides: list[dict],
    h_arch_lookup: dict[int, str],
    h_stat_lookup: dict[int, dict],
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    gpk: int,
) -> None:
    """Pitcher vs opposing lineup matchup analysis with multi-stat probabilities."""
    from scipy.special import expit, logit as sp_logit
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    def _safe_rate(r: object, default: float) -> float:
        """Ensure rate is valid for logit transform (strictly between 0 and 1)."""
        if pd.notna(r) and 0 < float(r) < 1:
            return float(r)
        return default

    for side_info in sides:
        pitcher_name = side_info["pitcher_name"]
        pid = side_info["pitcher_id"]
        arch = side_info["pitcher_arch"]
        p_proj = side_info["pitcher_proj"]
        opp_lu = side_info["opp_lineup"]
        opp_abbr = side_info["opp_abbr"]
        side_abbr = side_info["abbr"]

        # Pitcher base rates for logit adjustment
        p_k = _safe_rate(p_proj.get("projected_k_rate"), 0.22)
        p_bb = _safe_rate(p_proj.get("projected_bb_rate"), 0.08)
        p_hr = _safe_rate(p_proj.get("projected_hr_per_bf"), 0.03)

        # Pitcher header
        arch_tag = f" ({arch})" if arch else ""
        stats_parts = []
        k_pct = p_proj.get("projected_k_rate")
        bb_pct = p_proj.get("projected_bb_rate")
        score = p_proj.get("composite_score")
        if pd.notna(k_pct):
            stats_parts.append(f"K%: {k_pct * 100:.1f}%")
        if pd.notna(bb_pct):
            stats_parts.append(f"BB%: {bb_pct * 100:.1f}%")
        diamond_span = ""
        if pd.notna(score):
            diamond_span = diamond_rating_html_composite(score, size="sm")

        st.markdown(
            f'<div style="background:{DARK_CARD}; border:1px solid {DARK_BORDER}; '
            f'border-radius:6px; padding:0.8rem 1rem; margin-bottom:0.5rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div style="color:{GOLD}; font-size:1rem; font-weight:600;">'
            f'{side_abbr} SP: {pitcher_name}{arch_tag}</div>'
            f'<div>{diamond_span}</div></div>'
            f'<div style="color:{SLATE}; font-size:0.85rem; margin-top:2px;">'
            f'{" | ".join(stats_parts) if stats_parts else "K%: 22.0% | BB%: 8.0% (lg avg)"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if opp_lu.empty:
            st.caption(f"No lineup available for {opp_abbr}.")
            continue

        # Build lineup table with multi-stat matchup scoring
        name_col = "batter_name" if "batter_name" in opp_lu.columns else "player_name"
        id_col = "batter_id" if "batter_id" in opp_lu.columns else "player_id"

        rows = []
        total_k_lift = total_bb_lift = total_hr_lift = 0.0
        n_scored = 0

        for _, brow in opp_lu.head(9).iterrows():
            bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
            r: dict[str, object] = {
                "#": int(brow["batting_order"]),
                "Batter": brow.get(name_col, "Unknown"),
            }

            # Archetype
            if bid and h_arch_lookup:
                r["Type"] = h_arch_lookup.get(bid, "--")

            # Matchup-adjusted probabilities
            if pid and bid and not arsenal_df.empty and not vuln_df.empty:
                k_result = score_matchup(pid, bid, arsenal_df, vuln_df, baselines_pt)
                bb_result = score_matchup_bb(pid, bid, arsenal_df, vuln_df, baselines_pt)
                hr_result = score_matchup_hr(pid, bid, arsenal_df, vuln_df, baselines_pt)

                k_lift = k_result.get("matchup_k_logit_lift", 0.0)
                bb_lift = bb_result.get("matchup_bb_logit_lift", 0.0)
                hr_lift = hr_result.get("matchup_hr_logit_lift", 0.0)
                if np.isnan(k_lift):
                    k_lift = 0.0
                if np.isnan(bb_lift):
                    bb_lift = 0.0
                if np.isnan(hr_lift):
                    hr_lift = 0.0

                adj_k = float(expit(sp_logit(p_k) + k_lift))
                adj_bb = float(expit(sp_logit(p_bb) + bb_lift))
                adj_hr = float(expit(sp_logit(p_hr) + hr_lift))
                adj_ip = max(0.0, 1.0 - adj_k - adj_bb)

                r["K%"] = f"{adj_k:.1%}"
                r["BB%"] = f"{adj_bb:.1%}"
                r["HR%"] = f"{adj_hr:.1%}"
                r["In-Play%"] = f"{adj_ip:.1%}"
                r["K Lift"] = f"{k_lift:+.3f}"
                r["BB Lift"] = f"{bb_lift:+.3f}"

                total_k_lift += k_lift
                total_bb_lift += bb_lift
                total_hr_lift += hr_lift
                n_scored += 1
            else:
                # Fallback: show hitter projections when matchup data unavailable
                if bid and h_stat_lookup:
                    st_info = h_stat_lookup.get(bid, {})
                    _k = st_info.get("k_rate")
                    _bb = st_info.get("bb_rate")
                    r["K%"] = f"{_k:.1%}" if pd.notna(_k) else "--"
                    r["BB%"] = f"{_bb:.1%}" if pd.notna(_bb) else "--"
                r["HR%"] = "--"
                r["In-Play%"] = "--"
                r["K Lift"] = "--"
                r["BB Lift"] = "--"

            rows.append(r)

        if rows:
            st.markdown(
                f'<div style="color:{CREAM}; font-size:0.9rem; font-weight:500; '
                f'margin-bottom:0.3rem;">vs {opp_abbr} Lineup</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True,
                hide_index=True, height=350,
            )

            if n_scored > 0:
                avg_k = total_k_lift / n_scored
                avg_bb = total_bb_lift / n_scored
                avg_hr = total_hr_lift / n_scored

                if avg_k > 0.05:
                    k_color, k_word = POSITIVE, "favorable"
                elif avg_k < -0.05:
                    k_color, k_word = NEGATIVE, "unfavorable"
                else:
                    k_color, k_word = SLATE, "neutral"

                st.markdown(
                    f'<div style="font-size:0.85rem; margin-bottom:1rem;">'
                    f'<span style="color:{k_color};">Avg K Lift: {avg_k:+.3f} '
                    f'({k_word})</span>'
                    f' &nbsp;|&nbsp; '
                    f'<span style="color:{SLATE};">Avg BB Lift: {avg_bb:+.3f}</span>'
                    f' &nbsp;|&nbsp; '
                    f'<span style="color:{SLATE};">Avg HR Lift: {avg_hr:+.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _render_archetype_tab(
    sides: list[dict],
    h_arch_lookup: dict[int, str],
    p_arch_lookup: dict[int, str],
    offerings_df: pd.DataFrame,
    cluster_meta_df: pd.DataFrame,
    matchup_matrix_df: pd.DataFrame,
    gpk: int,
) -> None:
    """Pitcher archetype vs lineup archetype analysis."""
    if matchup_matrix_df.empty:
        st.info("Archetype matchup data is not yet available.")
        return

    for side_info in sides:
        pitcher_name = side_info["pitcher_name"]
        pid = side_info["pitcher_id"]
        p_arch = side_info["pitcher_arch"]
        opp_lu = side_info["opp_lineup"]
        opp_abbr = side_info["opp_abbr"]
        side_abbr = side_info["abbr"]

        arch_tag = f" ({p_arch})" if p_arch else ""
        st.markdown(
            f'<div style="color:{GOLD}; font-size:1rem; font-weight:600; '
            f'margin:0.5rem 0 0.3rem;">'
            f'{side_abbr} SP: {pitcher_name}{arch_tag}</div>',
            unsafe_allow_html=True,
        )

        # --- Section A: Pitcher Arsenal Profile ---
        if pid and not offerings_df.empty and not cluster_meta_df.empty:
            p_offers = offerings_df[offerings_df["pitcher_id"] == pid].copy()
            if not p_offers.empty:
                total_pitches = p_offers["pitches"].sum()
                p_offers = p_offers.merge(
                    cluster_meta_df[["pitch_archetype", "archetype_name"]],
                    on="pitch_archetype", how="left",
                )
                p_offers["usage"] = p_offers["pitches"] / total_pitches
                p_offers = p_offers.sort_values("usage", ascending=True)

                st.markdown(
                    f'<div style="color:{CREAM}; font-size:0.9rem; font-weight:500; '
                    f'margin:0.5rem 0 0.3rem;">Arsenal Profile</div>',
                    unsafe_allow_html=True,
                )

                fig, ax = plt.subplots(
                    figsize=(5.5, max(1.5, len(p_offers) * 0.4))
                )
                fig.patch.set_facecolor(DARK)
                ax.set_facecolor(DARK)

                colors = [
                    GOLD if row.get("pitch_family") == "Fastball"
                    else EMBER if row.get("pitch_family") == "Breaking"
                    else SAGE
                    for _, row in p_offers.iterrows()
                ]
                ax.barh(
                    range(len(p_offers)),
                    p_offers["usage"],
                    color=colors,
                    height=0.6,
                )

                labels = [
                    f'{row["pitch_name"]} ({row.get("archetype_name", "?")})'
                    for _, row in p_offers.iterrows()
                ]
                ax.set_yticks(range(len(p_offers)))
                ax.set_yticklabels(labels, fontsize=8, color=CREAM)
                ax.set_xlabel("Usage %", fontsize=9, color=SLATE)
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
                )
                ax.tick_params(colors=SLATE, labelsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color(DARK_BORDER)
                ax.spines["left"].set_color(DARK_BORDER)

                for i, (_, row) in enumerate(p_offers.iterrows()):
                    velo = row.get("release_speed")
                    if pd.notna(velo):
                        ax.text(
                            row["usage"] + 0.005, i,
                            f'{velo:.0f} mph', va="center",
                            fontsize=7, color=SLATE,
                        )

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # --- Section B: Archetype Matchup Matrix (filtered) ---
        if p_arch and not matchup_matrix_df.empty:
            p_row = matchup_matrix_df[
                matchup_matrix_df["pitcher_archetype_name"] == p_arch
            ].copy()

            if not p_row.empty:
                # Determine which hitter archetypes are in the opposing lineup
                lineup_archs: set[str] = set()
                if not opp_lu.empty:
                    id_col = (
                        "batter_id" if "batter_id" in opp_lu.columns
                        else "player_id"
                    )
                    for _, brow in opp_lu.head(9).iterrows():
                        bid = (
                            int(brow[id_col])
                            if pd.notna(brow.get(id_col)) else None
                        )
                        if bid and bid in h_arch_lookup:
                            lineup_archs.add(h_arch_lookup[bid])

                stat_choice = st.radio(
                    "Stat", ["K%", "BB%", "HR%"],
                    horizontal=True,
                    key=f"arch_stat_{side_info['side']}_{gpk}",
                )
                stat_col = {
                    "K%": "k_pct", "BB%": "bb_pct", "HR%": "hr_pct",
                }[stat_choice]

                st.markdown(
                    f'<div style="color:{CREAM}; font-size:0.9rem; font-weight:500; '
                    f'margin:0.3rem 0;">'
                    f'Matchup Matrix — {p_arch} vs Hitter Archetypes</div>',
                    unsafe_allow_html=True,
                )

                matrix_rows = []
                for _, mrow in p_row.sort_values(
                    stat_col, ascending=False
                ).iterrows():
                    h_name = mrow["hitter_archetype_name"]
                    in_lineup = h_name in lineup_archs
                    matrix_rows.append({
                        "Hitter Archetype": (
                            f"{'>> ' if in_lineup else ''}{h_name}"
                        ),
                        "K%": f"{mrow['k_pct']:.1%}",
                        "BB%": f"{mrow['bb_pct']:.1%}",
                        "HR%": f"{mrow['hr_pct']:.1%}",
                        "PA": int(mrow["pa"]),
                        "In Lineup": "Yes" if in_lineup else "",
                    })

                st.dataframe(
                    pd.DataFrame(matrix_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        # --- Section C: Lineup Archetype Composition ---
        if not opp_lu.empty and h_arch_lookup:
            id_col = (
                "batter_id" if "batter_id" in opp_lu.columns
                else "player_id"
            )
            arch_counts: dict[str, int] = {}
            for _, brow in opp_lu.head(9).iterrows():
                bid = (
                    int(brow[id_col])
                    if pd.notna(brow.get(id_col)) else None
                )
                if bid:
                    a = h_arch_lookup.get(bid, "Unknown")
                    arch_counts[a] = arch_counts.get(a, 0) + 1

            if arch_counts:
                pills = " &nbsp;".join(
                    f'<span style="background:{DARK_CARD}; '
                    f'border:1px solid {DARK_BORDER}; '
                    f'border-radius:12px; padding:2px 10px; '
                    f'font-size:0.8rem; color:{CREAM};">'
                    f'{count}x {name}</span>'
                    for name, count in sorted(
                        arch_counts.items(), key=lambda x: -x[1]
                    )
                )
                st.markdown(
                    f'<div style="margin:0.5rem 0 1rem;">'
                    f'<span style="color:{SLATE}; font-size:0.85rem;">'
                    f'{opp_abbr} Lineup: </span>{pills}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")


def _parse_temp_bucket(temp_str: object) -> str:
    """Convert temperature string to weather bucket."""
    if not temp_str:
        return "warm"
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        return "warm"
    if temp < 55:
        return "cold"
    if temp < 70:
        return "cool"
    if temp < 85:
        return "warm"
    return "hot"


def _parse_wind_category(wind_str: object) -> str:
    """Convert wind string to category."""
    if not wind_str:
        return "none"
    w = str(wind_str).lower()
    if "calm" in w or w.strip() == "":
        return "none"
    if "out" in w:
        return "out"
    if "in from" in w or "in," in w:
        return "in"
    if "l to r" in w or "r to l" in w:
        return "cross"
    return "none"


def _resolve_game_context(game: pd.Series) -> dict:
    """Look up umpire and weather adjustments from parquet data."""
    from scipy.special import logit as _logit

    ump_name = game.get("hp_umpire_name", "")
    ump_k_lift = 0.0
    ump_detail = ""

    if ump_name:
        ump_path = DASHBOARD_DIR / "umpire_tendencies.parquet"
        if ump_path.exists():
            ump_df = pd.read_parquet(ump_path)
            row = ump_df[ump_df["hp_umpire_name"] == ump_name]
            if not row.empty:
                ump_k_lift = float(row.iloc[0]["k_logit_lift"])
                k_rate = float(row.iloc[0]["k_rate_shrunk"])
                league_k = float(row.iloc[0]["league_k_rate"])
                delta_pp = (k_rate - league_k) * 100
                ump_detail = f"{k_rate:.1%} K-rate ({delta_pp:+.1f}pp vs avg)"

    temp_str = game.get("weather_temp", "")
    wind_str = game.get("weather_wind", "")
    condition = game.get("weather_condition", "")
    wx_k_lift = 0.0
    wx_detail = ""

    temp_bucket = _parse_temp_bucket(temp_str)
    wind_cat = _parse_wind_category(wind_str)

    wx_path = DASHBOARD_DIR / "weather_effects.parquet"
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        wx_row = wx_df[
            (wx_df["temp_bucket"] == temp_bucket) &
            (wx_df["wind_category"] == wind_cat)
        ]
        if not wx_row.empty:
            k_mult = float(wx_row.iloc[0]["k_multiplier"])
            overall_k = float(wx_row.iloc[0]["overall_k_rate"])
            adj_k = np.clip(overall_k * k_mult, 1e-6, 1 - 1e-6)
            wx_k_lift = float(
                _logit(adj_k) - _logit(np.clip(overall_k, 1e-6, 1 - 1e-6))
            )
            k_delta = (k_mult - 1.0) * 100
            if abs(k_delta) > 0.3:
                wx_detail = f"K-rate {k_delta:+.1f}%"

    weather_display = ""
    parts = []
    if temp_str:
        parts.append(f"{temp_str}°F")
    if condition:
        parts.append(condition)
    if wind_str:
        parts.append(wind_str)
    weather_display = " | ".join(parts) if parts else ""

    return {
        "ump_name": ump_name,
        "ump_k_lift": ump_k_lift,
        "ump_detail": ump_detail,
        "wx_k_lift": wx_k_lift,
        "wx_detail": wx_detail,
        "weather_display": weather_display,
        "venue_name": game.get("venue_name", ""),
    }


def _render_sim_tab(
    sides: list[dict],
    h_stat_lookup: dict[int, dict],
    k_samples_dict: dict[str, np.ndarray],
    bf_priors: pd.DataFrame,
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    gpk: int,
    game_context: dict | None = None,
    bb_samples_dict: dict[str, np.ndarray] | None = None,
    hr_samples_dict: dict[str, np.ndarray] | None = None,
    exit_model: object | None = None,
    pitcher_pc: pd.DataFrame | None = None,
    batter_pc: pd.DataFrame | None = None,
    tto_profiles: pd.DataFrame | None = None,
    exit_tendencies: pd.DataFrame | None = None,
) -> None:
    """Multi-stat game simulator using PA-by-PA engine (Layer 3 v2)."""
    from lib.game_sim.simulator import simulate_game
    from lib.game_sim.exit_model import ExitModel
    from lib.game_sim.tto_model import build_all_tto_lifts
    from lib.game_sim.pitch_count_model import build_pitch_count_features
    from lib.game_sim.fantasy_scoring import compute_pitcher_fantasy
    from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    if not k_samples_dict:
        st.info("K% posterior samples are not yet available.")
        return

    bb_samples_dict = bb_samples_dict or {}
    hr_samples_dict = hr_samples_dict or {}
    ctx = game_context or {}
    pitcher_pc = pitcher_pc if pitcher_pc is not None else pd.DataFrame()
    batter_pc = batter_pc if batter_pc is not None else pd.DataFrame()
    tto_profiles_df = tto_profiles if tto_profiles is not None else pd.DataFrame()
    exit_tend_df = exit_tendencies if exit_tendencies is not None else pd.DataFrame()

    # Ensure we have an ExitModel instance
    if exit_model is None:
        exit_model = ExitModel()

    # Game context bar
    context_parts = []
    venue = ctx.get("venue_name")
    if venue:
        context_parts.append(f'<span style="color:{CREAM};">{venue}</span>')
    ump_name = ctx.get("ump_name")
    if ump_name:
        ump_detail = ctx.get("ump_detail", "")
        ump_lift = ctx.get("ump_k_lift", 0.0)
        ump_color = POSITIVE if ump_lift > 0.02 else NEGATIVE if ump_lift < -0.02 else SLATE
        detail_html = f' <span style="color:{ump_color};">({ump_detail})</span>' if ump_detail else ""
        context_parts.append(f'HP: {ump_name}{detail_html}')
    wx_display = ctx.get("weather_display")
    if wx_display:
        wx_detail = ctx.get("wx_detail", "")
        wx_html = f' <span style="color:{SLATE};">({wx_detail})</span>' if wx_detail else ""
        context_parts.append(f'{wx_display}{wx_html}')

    if context_parts:
        st.markdown(
            f'<div style="background:{DARK_CARD}; border:1px solid {DARK_BORDER}; '
            f'border-radius:6px; padding:0.6rem 1rem; margin-bottom:1rem; '
            f'font-size:0.85rem; color:{SLATE};">'
            f'{" &nbsp;|&nbsp; ".join(context_parts)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    baselines_pt = {
        pt: {
            "whiff_rate": vals.get("whiff_rate", 0.25),
            "chase_rate": vals.get("chase_rate", 0.30),
            "barrel_rate": vals.get("barrel_rate", 0.06),
        }
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    ump_k_lift = ctx.get("ump_k_lift", 0.0)
    wx_k_lift = ctx.get("wx_k_lift", 0.0)

    # Stat display config
    _STAT_META = {
        "k":  {"label": "K",  "word": "strikeouts", "lines": (3.5, 10.5), "hi": 6, "vhi": 8},
        "bb": {"label": "BB", "word": "walks",      "lines": (1.5, 5.5),  "hi": 3, "vhi": 4},
        "h":  {"label": "H",  "word": "hits",       "lines": (3.5, 9.5),  "hi": 6, "vhi": 8},
        "hr": {"label": "HR", "word": "home runs",  "lines": (0.5, 2.5),  "hi": 1, "vhi": 2},
    }

    # Helper: fallback rate samples
    _rng = np.random.default_rng(99)

    def _fallback(rate: float, n: int = 4000) -> np.ndarray:
        r = np.clip(rate, 0.01, 0.99)
        return _rng.beta(r * 200, (1 - r) * 200, size=n).astype(np.float32)

    def _pitcher_avg_pitches(pid: int) -> float:
        if exit_tend_df.empty:
            return 88.0
        row = exit_tend_df[
            (exit_tend_df["pitcher_id"] == pid)
            & (exit_tend_df["season"] == PRIOR_SEASON)
        ]
        return float(row.iloc[0]["avg_pitches"]) if not row.empty else 88.0

    rendered = False
    for side_info in sides:
        pid = side_info["pitcher_id"]
        if not pid or str(pid) not in k_samples_dict:
            continue

        rendered = True
        pitcher_name = side_info["pitcher_name"]
        side_abbr = side_info["abbr"]
        opp_abbr = side_info["opp_abbr"]
        opp_lu = side_info["opp_lineup"]
        p_proj = side_info["pitcher_proj"]

        pid_str = str(pid)
        k_samp = k_samples_dict[pid_str]
        proj_bb = float(p_proj.get("projected_bb_rate", 0.08) or 0.08)
        proj_hr = float(p_proj.get("projected_hr_per_bf", 0.03) or 0.03)
        bb_samp = bb_samples_dict.get(pid_str) or _fallback(proj_bb)
        hr_samp = hr_samples_dict.get(pid_str) or _fallback(proj_hr)

        # Compute per-batter matchup lifts
        lineup_matchup_lifts: dict[str, np.ndarray] = {}
        per_batter_details: list[dict] = []
        lineup_batter_ids: list[int] = []
        has_lineup = False

        if not opp_lu.empty and not arsenal_df.empty and not vuln_df.empty:
            id_col = "batter_id" if "batter_id" in opp_lu.columns else "player_id"
            name_col = "batter_name" if "batter_name" in opp_lu.columns else "player_name"

            k_lifts, bb_lifts, hr_lifts = [], [], []
            for _, brow in opp_lu.head(9).iterrows():
                bid = int(brow[id_col]) if pd.notna(brow.get(id_col)) else None
                if bid:
                    lineup_batter_ids.append(bid)
                    k_m = score_matchup(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    kl = k_m.get("matchup_k_logit_lift", 0.0)
                    k_lifts.append(0.0 if np.isnan(kl) else kl)

                    bb_m = score_matchup_bb(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    bl = bb_m.get("matchup_bb_logit_lift", 0.0)
                    bb_lifts.append(0.0 if np.isnan(bl) else bl)

                    hr_m = score_matchup_hr(pid, bid, arsenal_df, vuln_df, baselines_pt)
                    hl = hr_m.get("matchup_hr_logit_lift", 0.0)
                    hr_lifts.append(0.0 if np.isnan(hl) else hl)

                    k_m["batter_name"] = brow.get(name_col, "Unknown")
                    k_m["batting_order"] = int(brow["batting_order"])
                    per_batter_details.append(k_m)
                else:
                    lineup_batter_ids.append(0)
                    k_lifts.append(0.0)
                    bb_lifts.append(0.0)
                    hr_lifts.append(0.0)

            while len(k_lifts) < 9:
                k_lifts.append(0.0)
                bb_lifts.append(0.0)
                hr_lifts.append(0.0)
                lineup_batter_ids.append(0)

            has_lineup = True
            lineup_matchup_lifts = {
                "k": np.array(k_lifts[:9]),
                "bb": np.array(bb_lifts[:9]),
                "hr": np.array(hr_lifts[:9]),
            }

        # TTO lifts
        tto_lifts = build_all_tto_lifts(
            tto_profiles_df if not tto_profiles_df.empty else None,
            pid, PRIOR_SEASON,
        )

        # Pitch count features
        pitcher_ppa_adj = 0.0
        batter_ppa_adjs = np.zeros(9)
        if not pitcher_pc.empty and not batter_pc.empty and lineup_batter_ids:
            pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
                pitcher_pc, batter_pc, pid, lineup_batter_ids[:9], PRIOR_SEASON,
            )

        avg_pitches = _pitcher_avg_pitches(pid)

        # Run PA-by-PA simulation
        result = simulate_game(
            pitcher_k_rate_samples=k_samp,
            pitcher_bb_rate_samples=bb_samp,
            pitcher_hr_rate_samples=hr_samp,
            lineup_matchup_lifts=lineup_matchup_lifts,
            tto_lifts=tto_lifts,
            pitcher_ppa_adj=pitcher_ppa_adj,
            batter_ppa_adjs=batter_ppa_adjs,
            exit_model=exit_model,
            pitcher_avg_pitches=avg_pitches,
            umpire_k_lift=ump_k_lift,
            weather_k_lift=wx_k_lift,
            n_sims=10_000,
            random_seed=42 + gpk + (0 if side_info["side"] == "away" else 1),
        )

        # Fantasy points
        fantasy = compute_pitcher_fantasy(result)
        dk = fantasy.dk_summary()

        # Pitcher header with multi-stat expectations
        arch = side_info.get("pitcher_arch")
        arch_tag = f" ({arch})" if arch else ""
        lineup_tag = f"vs {opp_abbr} lineup" if has_lineup else "league-avg baseline"

        exp_parts = [
            f"E[K]: {np.mean(result.k_samples):.1f}",
            f"BB: {np.mean(result.bb_samples):.1f}",
            f"H: {np.mean(result.h_samples):.1f}",
            f"HR: {np.mean(result.hr_samples):.1f}",
            f"IP: {np.mean(result.ip_samples()):.1f}",
        ]

        st.markdown(
            f'<div style="color:{GOLD}; font-size:1rem; font-weight:600; '
            f'margin:0.8rem 0 0.3rem;">'
            f'{side_abbr} SP: {pitcher_name}{arch_tag}'
            f'<span style="color:{SLATE}; font-size:0.85rem; font-weight:400;">'
            f' — {" | ".join(exp_parts)} | {lineup_tag}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # DK fantasy line
        st.markdown(
            f'<div style="font-size:0.85rem; color:{SLATE}; margin-bottom:0.5rem;">'
            f'DK: {dk["mean"]:.1f} pts (median {dk["median"]:.1f}, '
            f'10th {dk["q10"]:.1f}, 90th {dk["q90"]:.1f})</div>',
            unsafe_allow_html=True,
        )

        # Stat selector
        selected_stat_label = st.radio(
            "Stat", ["K", "BB", "H", "HR"], horizontal=True,
            key=f"sim_stat_{side_info['side']}_{gpk}",
        )
        stat_key = selected_stat_label.lower()
        stat_samples = getattr(result, f"{stat_key}_samples")
        meta = _STAT_META[stat_key]

        # Distribution chart
        fig = create_game_stat_fig(stat_samples, pitcher_name, stat=stat_key)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Combined prop table for all stats
        prop_rows = []
        for skey, smeta in _STAT_META.items():
            s_samples = getattr(result, f"{skey}_samples")
            lo, hi = smeta["lines"]
            lines = [x + 0.5 for x in range(int(lo - 0.5), int(hi - 0.5) + 1)]
            over_df = result.over_probs(skey, lines=lines)

            for _, row in over_df.iterrows():
                p = row["p_over"]
                signal = (
                    "Strong Over" if p > 0.65 else
                    "Lean Over" if p > 0.55 else
                    "Strong Under" if p < 0.35 else
                    "Lean Under" if p < 0.45 else
                    "Toss-up"
                )
                prop_rows.append({
                    "Stat": smeta["label"],
                    "Line": f"Over {row['line']:.1f}",
                    "P(Over)": f"{p:.1%}",
                    "P(Under)": f"{1 - p:.1%}",
                    "Signal": signal,
                })

        st.dataframe(
            pd.DataFrame(prop_rows), use_container_width=True, hide_index=True,
        )

        # Summary cards for selected stat
        summary_cols = st.columns(4)
        stat_label = meta["label"]
        cards = [
            (f"Expected {stat_label}", f"{np.mean(stat_samples):.1f}"),
            ("Std Dev", f"{np.std(stat_samples):.1f}"),
            (f"Median {stat_label}", f"{np.median(stat_samples):.0f}"),
            ("90th Pctile", f"{np.percentile(stat_samples, 90):.0f}"),
        ]
        for col, (label, val) in zip(summary_cols, cards):
            with col:
                st.markdown(metric_card(label, val), unsafe_allow_html=True)

        # Insight box
        mean_val = float(np.mean(stat_samples))
        p_hi = float((stat_samples >= meta["hi"]).sum() / len(stat_samples) * 100)
        p_vhi = float((stat_samples >= meta["vhi"]).sum() / len(stat_samples) * 100)

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-bullet">
                <span class="dot" style="background:{GOLD};"></span>
                Expect around <strong>{mean_val:.0f} {meta['word']}</strong>
                (±{np.std(stat_samples):.0f}).
            </div>
            <div class="insight-bullet">
                <span class="dot" style="background:{SAGE};"></span>
                <strong>{p_hi:.0f}%</strong> chance of {meta['hi']}+ {meta['word']},
                <strong>{p_vhi:.0f}%</strong> chance of {meta['vhi']}+.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-batter matchup breakdown
        if per_batter_details:
            with st.expander(f"Lineup Matchup Breakdown — {pitcher_name} vs {opp_abbr}"):
                bd_rows = []
                for d in per_batter_details:
                    bname = d.get("batter_name", "Unknown")
                    mwhiff = d.get("matchup_whiff_rate", np.nan)
                    bwhiff = d.get("baseline_whiff_rate", np.nan)
                    lift = d.get("matchup_k_logit_lift", 0.0)
                    rel = d.get("avg_reliability", 0.0)
                    bd_rows.append({
                        "#": d.get("batting_order", ""),
                        "Batter": bname,
                        "Matchup Whiff%": f"{mwhiff:.1%}" if pd.notna(mwhiff) else "--",
                        "Baseline Whiff%": f"{bwhiff:.1%}" if pd.notna(bwhiff) else "--",
                        "K Lift": f"{lift:+.3f}",
                        "Reliability": f"{rel:.0%}",
                    })
                st.dataframe(
                    pd.DataFrame(bd_rows), use_container_width=True,
                    hide_index=True,
                )

                avg_lift = float(np.mean([
                    d.get("matchup_k_logit_lift", 0.0) for d in per_batter_details
                ]))
                if avg_lift > 0.05:
                    color, word = POSITIVE, "favorable"
                elif avg_lift < -0.05:
                    color, word = NEGATIVE, "unfavorable"
                else:
                    color, word = SLATE, "neutral"
                st.markdown(
                    f'<div style="color:{color}; font-size:0.85rem;">'
                    f'Avg K Lift: {avg_lift:+.3f} — this lineup is {word} '
                    f'for strikeouts</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

    if not rendered:
        st.info("No K% samples available for this game's pitchers.")


def _render_game_browser() -> None:
    """Browse historical games grouped by game_pk with team and pitcher selectors."""
    from lib.matchup import score_matchup
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    st.caption(
        f"Browse {PRIOR_SEASON} games by team. Select a game, then view each pitcher's "
        "matchup breakdown against the opposing lineup."
    )

    game_logs_path = DASHBOARD_DIR / "pitcher_game_logs.parquet"
    lineups_path = DASHBOARD_DIR / "game_lineups.parquet"
    batter_ks_path = DASHBOARD_DIR / "game_batter_ks.parquet"

    if not all(p.exists() for p in [game_logs_path, lineups_path, batter_ks_path]):
        st.warning("Game browser data is not yet available.")
        return

    game_logs = pd.read_parquet(game_logs_path)
    all_lineups = pd.read_parquet(lineups_path)
    all_batter_ks = pd.read_parquet(batter_ks_path)

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    if arsenal_df.empty or vuln_df.empty:
        st.warning("Matchup profile data not found.")
        return

    @st.cache_data
    def _enrich_game_logs(_game_logs: pd.DataFrame) -> pd.DataFrame:
        game_info = load_game_info()
        if game_info.empty:
            return _game_logs
        return _game_logs.merge(game_info, on="game_pk", how="left")

    game_logs = _enrich_game_logs(game_logs)

    if "game_date" not in game_logs.columns:
        st.warning("Game info is not yet available.")
        return

    team_lookup = get_team_lookup()

    all_team_names = set()
    if "home_team_name" in game_logs.columns:
        all_team_names.update(game_logs["home_team_name"].dropna().unique())
    if "away_team_name" in game_logs.columns:
        all_team_names.update(game_logs["away_team_name"].dropna().unique())
    all_team_names = sorted(t for t in all_team_names if t)

    selected_team = st.selectbox(
        "Select team", all_team_names, key="gb_team"
    )

    team_games = game_logs[
        (game_logs["home_team_name"] == selected_team) |
        (game_logs["away_team_name"] == selected_team)
    ].copy()

    if team_games.empty:
        st.info("No games found for this team.")
        return

    game_summary = (
        team_games.groupby("game_pk")
        .agg(
            game_date=("game_date", "first"),
            home_team_name=("home_team_name", "first"),
            away_team_name=("away_team_name", "first"),
        )
        .reset_index()
        .sort_values("game_date", ascending=False)
    )

    game_options = {}
    for _, g in game_summary.iterrows():
        gpk = int(g["game_pk"])
        date_str = str(g["game_date"])[:10]
        home = g["home_team_name"] or "?"
        away = g["away_team_name"] or "?"
        label = f"{date_str} — {away} @ {home}"
        game_options[label] = gpk

    selected_game_label = st.selectbox(
        "Select Game", list(game_options.keys()), key="gb_game",
    )
    selected_gpk = game_options[selected_game_label]

    game_pitchers = game_logs[game_logs["game_pk"] == selected_gpk].copy()
    if game_pitchers.empty:
        st.info("No pitcher data for this game.")
        return

    game_meta = game_summary[game_summary["game_pk"] == selected_gpk].iloc[0]
    home_name = game_meta["home_team_name"] or "Home"
    away_name = game_meta["away_team_name"] or "Away"

    pitcher_opts = {}
    for _, pr in game_pitchers.sort_values(
        ["is_starter", "pitcher_name"], ascending=[False, True]
    ).iterrows():
        pid = int(pr["pitcher_id"])
        pname = pr["pitcher_name"]
        team = team_lookup.get(pid, "")
        role = "SP" if pr.get("is_starter") else "RP"
        ks = int(pr["strike_outs"]) if pd.notna(pr.get("strike_outs")) else 0
        ip = pr.get("innings_pitched", 0)
        dname = f"{pname} ({team}, {role}) — {ks} K, {ip} IP" if team else f"{pname} ({role}) — {ks} K, {ip} IP"
        pitcher_opts[dname] = pid

    selected_pitcher_display = st.selectbox(
        "Select Pitcher", list(pitcher_opts.keys()), key="gb_pitcher",
    )
    pitcher_id = pitcher_opts[selected_pitcher_display]

    game_row = game_pitchers[game_pitchers["pitcher_id"] == pitcher_id].iloc[0]
    game_lineups_this = all_lineups[all_lineups["game_pk"] == selected_gpk]

    if game_lineups_this.empty:
        st.warning("No lineup data found for this game.")
        return

    bk_game = all_batter_ks[
        (all_batter_ks["game_pk"] == selected_gpk) &
        (all_batter_ks["pitcher_id"] == pitcher_id)
    ]
    faced_batters = set(bk_game["batter_id"].tolist())
    home_tid = game_row.get("home_team_id")

    opposing_lineup = None
    opponent_name = ""
    for tid in game_lineups_this["team_id"].unique():
        team_lineup = game_lineups_this[
            game_lineups_this["team_id"] == tid
        ].sort_values("batting_order")
        lineup_batters = set(team_lineup["player_id"].tolist())
        if lineup_batters & faced_batters:
            opposing_lineup = team_lineup
            opponent_name = home_name if tid == home_tid else away_name
            break

    if opposing_lineup is None or opposing_lineup.empty:
        st.warning("Could not determine opposing lineup for this pitcher.")
        return

    actual_ks_game = bk_game
    actual_k_map = dict(zip(actual_ks_game["batter_id"], actual_ks_game["k"]))
    actual_pa_map = dict(zip(actual_ks_game["batter_id"], actual_ks_game["pa"]))

    baselines_pt = {
        pt: {"whiff_rate": vals.get("whiff_rate", 0.25)}
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    display_rows = []
    total_actual_k = int(game_row.get("strike_outs", 0))
    total_matchup_lift = 0.0
    n_scored = 0

    for _, brow in opposing_lineup.iterrows():
        bid = int(brow["player_id"])
        bname = brow.get("batter_name", "Unknown")
        order = int(brow["batting_order"])
        bteam = team_lookup.get(bid, "")

        matchup = score_matchup(
            pitcher_id=pitcher_id,
            batter_id=bid,
            pitcher_arsenal=arsenal_df,
            hitter_vuln=vuln_df,
            baselines_pt=baselines_pt,
        )

        lift = matchup.get("matchup_k_logit_lift", 0.0)
        if np.isnan(lift):
            lift = 0.0
        mwhiff = matchup.get("matchup_whiff_rate", np.nan)
        bwhiff = matchup.get("baseline_whiff_rate", np.nan)
        reliability = matchup.get("avg_reliability", 0.0)

        actual_k = actual_k_map.get(bid, 0)
        actual_pa = actual_pa_map.get(bid, 0)

        if not np.isnan(lift):
            total_matchup_lift += lift
            n_scored += 1

        row = {
            "#": order,
            "Batter": f"{bname} ({bteam})" if bteam else bname,
            "Matchup Whiff%": f"{mwhiff:.1%}" if pd.notna(mwhiff) else "--",
            "Baseline Whiff%": f"{bwhiff:.1%}" if pd.notna(bwhiff) else "--",
            "K Lift": f"{lift:+.3f}" if lift != 0 else "0.000",
            "Reliability": f"{reliability:.0%}",
            "PA": actual_pa,
            "K": actual_k,
        }
        display_rows.append(row)

    date_str = str(game_meta["game_date"])[:10]
    ip = game_row.get("innings_pitched", 0)
    bf = int(game_row.get("batters_faced", 0)) if pd.notna(game_row.get("batters_faced")) else 0
    avg_lift = total_matchup_lift / n_scored if n_scored > 0 else 0.0
    pitcher_name = game_row.get("pitcher_name", "Unknown")
    pitcher_team = team_lookup.get(pitcher_id, "")

    gb_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{pitcher_name}{f" ({pitcher_team})" if pitcher_team else ""}</div>'
        f'<div class="brand-subtitle">{date_str} — {away_name} @ {home_name} | {ip} IP, {bf} BF</div>'
        f'</div>'
        f'<div style="font-size:1.2rem; font-weight:600;">'
        f'<span style="color:{GOLD};">{total_actual_k} K</span>'
        f'<span style="color:{SLATE};"> | Avg Lift: {avg_lift:+.3f}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(gb_header_html, unsafe_allow_html=True)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    lineup_ks = sum(r["K"] for r in display_rows)
    lineup_pa = sum(r["PA"] for r in display_rows)
    k_rate_actual = lineup_ks / lineup_pa if lineup_pa > 0 else 0

    if avg_lift > 0.05:
        lift_color = POSITIVE
        lift_word = "favorable"
    elif avg_lift < -0.05:
        lift_color = NEGATIVE
        lift_word = "unfavorable"
    else:
        lift_color = SLATE
        lift_word = "neutral"

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-bullet">
            <span class="dot" style="background:{GOLD};"></span>
            Actual: <strong>{total_actual_k} K</strong> in {bf} BF
            ({k_rate_actual:.1%} K rate)
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{lift_color};"></span>
            Matchup model rated this lineup as
            <strong style="color:{lift_color};">{lift_word}</strong>
            for strikeouts (avg logit lift: {avg_lift:+.3f})
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SLATE};"></span>
            K Lift = matchup-driven K% advantage above baseline (logit scale).
            Positive = hitter is more vulnerable to this pitcher's arsenal.
            Negative = hitter handles it better than average.
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_schedule() -> None:
    """Combined Schedule page — Today's Games + Game Browser."""
    st.markdown('<div class="section-header">Schedule</div>',
                unsafe_allow_html=True)

    view = st.radio(
        "View",
        ["Today's Games", "Game Browser"],
        horizontal=True,
        key="schedule_view",
    )

    if view == "Today's Games":
        _render_todays_games()
    else:
        _render_game_browser()
