"""Player Profile page |Deep dive into a single player's projections."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, POSITIVE, NEGATIVE,
    CURRENT_SEASON, PRIOR_SEASON, TRAIN_START, PROJECTION_LABEL,
    AVAILABLE_SEASONS, UNRELIABLE_BB_SEASONS,
    PITCHER_STATS, HITTER_STATS,
    PITCHER_COUNTING_DISPLAY, HITTER_COUNTING_DISPLAY,
    HITTER_TRAD_STATS, HITTER_TRAD_COUNTING,
    PITCHER_TRAD_STATS, PITCHER_TRAD_COUNTING,
    HITTER_ADVANCED_STATS, PITCHER_ADVANCED_STATS,
    HITTER_EOS_DELTA_STATS, PITCHER_EOS_DELTA_STATS,
    PITCH_DISPLAY,
)
from utils.alerts import tdd_info, tdd_warn
from services.data_loader import (
    load_projections, load_counting, load_player_teams, load_rankings,
    load_k_samples, load_traditional_stats, load_traditional_stats_all,
    load_pitcher_arsenal, load_pitcher_arsenal_all,
    load_hitter_vulnerability, load_hitter_vulnerability_all,
    load_pitcher_location_grid, load_pitcher_location_grid_all,
    load_hitter_zone_grid, load_hitter_zone_grid_all,
    load_pitcher_offerings, load_hitter_vuln_arch, load_hitter_vuln_arch_career,
    load_cluster_metadata, load_baselines_arch,
    load_hitter_aggressiveness, load_hitter_aggressiveness_all,
    load_pitcher_efficiency, load_pitcher_efficiency_all,
    load_full_stats, load_advanced_stats, load_hitter_strength,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_archetype_matchup_matrix,
    load_hitter_breakout_candidates,
    load_hitter_grade_ci, load_pitcher_grade_ci,
    season_selector,
)
from utils.helpers import get_team_lookup, get_injury_lookup
from utils.html import esc
from utils.team_names import team_full
from utils.formatters import fmt_stat, fmt_trad
from components.metric_cards import (
    metric_card, percentile_rank, hybrid_percentile_rank,
    stat_chip, stat_chip_row, grade_bar_block, QUALIFIED_PA, MIN_PA,
)
from components.charts import (
    create_posterior_fig, create_arsenal_donut,
    create_pitch_density_plotly, create_hitter_zone_plotly,
)
from components.tables import (
    combine_platoon_vuln,
    build_hitter_profile_table, build_pitcher_profile_table,
)
from components.scouting import generate_scouting_bullets, generate_scouting_card, render_scouting_card
from components.headshot import render_headshot
from components.diamond_rating import diamond_rating_html, diamond_rating_html_composite


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def render_approach_efficiency(
    player_type: str,
    player_id: int,
    id_col: str,
    selected_season: int | None = None,
    is_career: bool = False,
) -> None:
    """Render Approach & Efficiency cards for any season or career."""
    label = "Career" if is_career else str(selected_season) if selected_season else str(PRIOR_SEASON)
    st.markdown(
        f'<div class="tdd-section-hdr" style="margin-top:1rem;">'
        f'{label} Approach &amp; Efficiency</div>',
        unsafe_allow_html=True,
    )
    if player_type == "Hitter":
        _agg_all = load_hitter_aggressiveness_all()
        if _agg_all.empty:
            _agg_all = load_hitter_aggressiveness()  # fallback to single-season
        if not _agg_all.empty:
            if is_career:
                _agg_player = _agg_all[_agg_all["batter_id"] == player_id]
                if not _agg_player.empty:
                    # Weighted average by PA (approx: use pitches_per_pa count as proxy)
                    _agg_data = _agg_player.select_dtypes(include="number").mean()
                else:
                    st.caption("No aggressiveness data for this player.")
                    return
            else:
                _season = selected_season if selected_season else PRIOR_SEASON
                if "season" in _agg_all.columns:
                    _agg_player = _agg_all[
                        (_agg_all["batter_id"] == player_id) & (_agg_all["season"] == _season)
                    ]
                else:
                    _agg_player = _agg_all[_agg_all["batter_id"] == player_id]
                if _agg_player.empty:
                    st.caption(f"No aggressiveness data for this player in {_season}.")
                    return
                _agg_data = _agg_player.iloc[0]

            _agg_items = [
                ("FP Swing%", "first_pitch_swing_pct", True, True),
                ("Chase%", "chase_rate", True, False),
                ("2-Strike Chase%", "two_strike_chase_rate", True, False),
                ("2-Strike Whiff%", "two_strike_whiff_rate", True, False),
                ("Zone Swing%", "zone_swing_pct", True, True),
                ("P/PA", "pitches_per_pa", False, None),
            ]
            # Build season population for percentiles
            if not is_career and "season" in _agg_all.columns:
                _agg_pop = _agg_all[_agg_all["season"] == _season]
            else:
                _agg_pop = _agg_all
            for _chunk_start in range(0, len(_agg_items), 3):
                _chunk = _agg_items[_chunk_start:_chunk_start + 3]
                _a_cols = st.columns(len(_chunk))
                for _ac, (_albl, _acol, _is_pct, _hib) in zip(_a_cols, _chunk):
                    _aval = _agg_data.get(_acol)
                    if pd.notna(_aval):
                        _disp = f"{_aval:.1%}" if _is_pct else f"{_aval:.1f}"
                    else:
                        _disp = ""
                    _pct = None
                    if pd.notna(_aval) and _hib is not None and _acol in _agg_pop.columns:
                        _pct = percentile_rank(_agg_pop[_acol], float(_aval), _hib)
                    with _ac:
                        st.markdown(metric_card(_albl, _disp, pctile=_pct), unsafe_allow_html=True)
    else:
        _eff_all = load_pitcher_efficiency_all()
        if _eff_all.empty:
            _eff_all = load_pitcher_efficiency()  # fallback
        if not _eff_all.empty:
            if is_career:
                _eff_player = _eff_all[_eff_all["pitcher_id"] == player_id]
                if not _eff_player.empty:
                    _eff_data = _eff_player.select_dtypes(include="number").mean()
                else:
                    st.caption("No efficiency data for this player.")
                    return
            else:
                _season = selected_season if selected_season else PRIOR_SEASON
                if "season" in _eff_all.columns:
                    _eff_player = _eff_all[
                        (_eff_all["pitcher_id"] == player_id) & (_eff_all["season"] == _season)
                    ]
                else:
                    _eff_player = _eff_all[_eff_all["pitcher_id"] == player_id]
                if _eff_player.empty:
                    st.caption(f"No efficiency data for this player in {_season}.")
                    return
                _eff_data = _eff_player.iloc[0]

            _eff_items = [
                ("F-Strike%", "first_strike_pct", True, True),
                ("Zone%", "zone_pct", True, True),
                ("Putaway%", "putaway_rate", True, True),
                ("P/PA", "pitches_per_pa", False, False),
            ]
            # Build season population for percentiles
            if not is_career and "season" in _eff_all.columns:
                _eff_pop = _eff_all[_eff_all["season"] == _season]
            else:
                _eff_pop = _eff_all
            _e_cols = st.columns(len(_eff_items))
            for _ec, (_elbl, _ecol, _is_pct, _hib) in zip(_e_cols, _eff_items):
                _eval = _eff_data.get(_ecol)
                if pd.notna(_eval):
                    _disp = f"{_eval:.1%}" if _is_pct else f"{_eval:.1f}"
                else:
                    _disp = ""
                _pct = None
                if pd.notna(_eval) and _ecol in _eff_pop.columns:
                    _pct = percentile_rank(_eff_pop[_ecol], float(_eval), _hib)
                with _ec:
                    st.markdown(metric_card(_elbl, _disp, pctile=_pct), unsafe_allow_html=True)


def render_pitch_profiles(
    player_type: str,
    player_id: int,
    selected_name: str,
    selected_season: int | None = None,
    is_career: bool = False,
) -> None:
    """Render pitch arsenal / vulnerability tables and zone charts for any season."""
    season_label = "Career" if is_career else str(selected_season) if selected_season else str(PRIOR_SEASON)

    if player_type == "Pitcher":
        # Platoon split toggle for pitcher arsenal
        platoon_choice = st.radio(
            "Batter handedness",
            ["All Batters", "vs LHH", "vs RHH"],
            horizontal=True,
            key="hist_pitcher_platoon_split",
        )
        _platoon_stand = {"vs LHH": "L", "vs RHH": "R"}.get(platoon_choice)
        _platoon_suffix = f" |{platoon_choice}" if _platoon_stand else ""

        # Arsenal
        if is_career:
            arsenal_all = load_pitcher_arsenal_all()
            if not arsenal_all.empty:
                p_ars = arsenal_all[arsenal_all["pitcher_id"] == player_id]
                if _platoon_stand and "batter_stand" in p_ars.columns:
                    p_ars = p_ars[p_ars["batter_stand"] == _platoon_stand]
                if not p_ars.empty:
                    # Aggregate career arsenal: sum counts, recompute rates
                    sum_cols = ["pitches", "total_pitches", "swings", "whiffs",
                                "called_strikes", "csw", "bip", "barrels_proxy", "hard_hits"]
                    sum_cols = [c for c in sum_cols if c in p_ars.columns]
                    grp = p_ars.groupby(["pitcher_id", "pitch_hand", "pitch_type", "pitch_family"])[sum_cols].sum().reset_index()
                    total_p = grp["pitches"].sum()
                    grp["usage_pct"] = grp["pitches"] / total_p if total_p > 0 else 0
                    grp["whiff_rate"] = grp["whiffs"] / grp["swings"].replace(0, np.nan)
                    grp["csw_pct"] = grp["csw"] / grp["pitches"].replace(0, np.nan)
                    grp["barrel_rate_against"] = grp["barrels_proxy"] / grp["bip"].replace(0, np.nan)
                    grp["hard_hit_rate_against"] = grp["hard_hits"] / grp["bip"].replace(0, np.nan)
                    # Weighted avg velo
                    if "avg_velo" in p_ars.columns:
                        velo_grp = p_ars.groupby("pitch_type").apply(
                            lambda g: (g["avg_velo"] * g["pitches"]).sum() / g["pitches"].sum()
                            if g["pitches"].sum() > 0 else np.nan,
                            include_groups=False,
                        ).reset_index(name="avg_velo")
                        grp = grp.merge(velo_grp, on="pitch_type", how="left")
                    p_arsenal = grp
                else:
                    p_arsenal = pd.DataFrame()
            else:
                p_arsenal = pd.DataFrame()
        else:
            arsenal_all = load_pitcher_arsenal_all()
            if not arsenal_all.empty and "season" in arsenal_all.columns:
                _season = selected_season if selected_season else PRIOR_SEASON
                _mask = (arsenal_all["pitcher_id"] == player_id) & (arsenal_all["season"] == _season)
                if _platoon_stand and "batter_stand" in arsenal_all.columns:
                    _mask = _mask & (arsenal_all["batter_stand"] == _platoon_stand)
                p_arsenal = arsenal_all[_mask].copy()
                # Recompute usage_pct within the filtered subset
                if not p_arsenal.empty and _platoon_stand:
                    total_p = p_arsenal["pitches"].sum()
                    p_arsenal["usage_pct"] = p_arsenal["pitches"] / total_p if total_p > 0 else 0
            else:
                # Fallback to single-season file (no platoon split available)
                arsenal_df = load_pitcher_arsenal()
                p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == player_id].copy() if not arsenal_df.empty else pd.DataFrame()

        if not p_arsenal.empty:
            st.markdown(f'<div class="tdd-section-hdr">Pitch Arsenal ({season_label}{_platoon_suffix})</div>',
                        unsafe_allow_html=True)
            donut_col, table_col = st.columns([2, 3])
            with donut_col:
                donut_fig = create_arsenal_donut(p_arsenal, season_label=f"{season_label}{_platoon_suffix}")
                st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
            with table_col:
                table_html = build_pitcher_profile_table(p_arsenal)
                if table_html:
                    st.markdown(f'<div class="insight-card">{table_html}</div>', unsafe_allow_html=True)

        # Arsenal Archetype Map
        offerings_df = load_pitcher_offerings()
        cluster_meta = load_cluster_metadata()
        if not offerings_df.empty and not cluster_meta.empty:
            p_off = offerings_df[offerings_df["pitcher_id"] == player_id].copy()
            if not p_off.empty:
                p_off = p_off.merge(
                    cluster_meta[["pitch_archetype", "archetype_name"]],
                    on="pitch_archetype", how="left",
                )
                p_off = p_off.sort_values("pitches", ascending=False)
                arch_rows = []
                for _, row in p_off.iterrows():
                    pt_name = PITCH_DISPLAY.get(row.get("pitch_type", ""), row.get("pitch_name", ""))
                    velo = f'{row["release_speed"]:.1f} mph' if pd.notna(row.get("release_speed")) else ""
                    ivb = f'{row["pfx_z"]:.1f}"' if pd.notna(row.get("pfx_z")) else ""
                    hb = f'{row["pfx_x"]:.1f}"' if pd.notna(row.get("pfx_x")) else ""
                    arch_rows.append({
                        "Pitch": pt_name,
                        "Archetype": row.get("archetype_name", f'Cluster {row["pitch_archetype"]}'),
                        "Velo": velo,
                        "IVB": ivb,
                        "HB": hb,
                    })
                if arch_rows:
                    st.markdown(f'<div class="tdd-section-hdr">Arsenal Archetype Map</div>',
                                unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

        # Location heatmap |prefer raw coordinates, fall back to grid
        from services.data_loader import load_pitcher_pitch_locations
        _raw_locs = load_pitcher_pitch_locations()
        _has_raw = not _raw_locs.empty and player_id in _raw_locs["pitcher_id"].values

        if _has_raw and (is_career or selected_season is None):
            # Raw coordinates |projection or career view
            p_loc = _raw_locs[_raw_locs["pitcher_id"] == player_id]
        else:
            # Grid data |historical seasons or raw not available
            if is_career:
                ploc_all = load_pitcher_location_grid_all()
                if not ploc_all.empty:
                    p_loc = ploc_all[ploc_all["pitcher_id"] == player_id]
                    if not p_loc.empty:
                        sum_cols = [c for c in ["pitches", "swings", "whiffs", "called_strikes", "bip"] if c in p_loc.columns]
                        grp_cols = ["pitcher_id", "pitcher_name", "pitch_type", "batter_stand", "grid_col", "grid_row"]
                        grp_cols = [c for c in grp_cols if c in p_loc.columns]
                        p_loc = p_loc.groupby(grp_cols)[sum_cols].sum().reset_index()
                else:
                    p_loc = pd.DataFrame()
            else:
                ploc_all = load_pitcher_location_grid_all()
                _season = selected_season if selected_season else PRIOR_SEASON
                if not ploc_all.empty and "season" in ploc_all.columns:
                    p_loc = ploc_all[(ploc_all["pitcher_id"] == player_id) & (ploc_all["season"] == _season)]
                else:
                    p_loc_df = load_pitcher_location_grid()
                    p_loc = p_loc_df[p_loc_df["pitcher_id"] == player_id] if not p_loc_df.empty else pd.DataFrame()

        if not p_loc.empty:
            st.markdown(f'<div class="tdd-section-hdr">Pitch Density ({season_label})</div>',
                        unsafe_allow_html=True)
            loc_cols = st.columns([1, 3])
            with loc_cols[0]:
                loc_stand = st.radio(
                    "Batter handedness", ["All", "vs LHH", "vs RHH"],
                    horizontal=True, key="hist_pitcher_loc_stand",
                    label_visibility="collapsed",
                )
            stand_filter = {"vs LHH": "L", "vs RHH": "R"}.get(loc_stand)

            # Pitch-type filter |works for both raw and grid data
            if "plate_x" in p_loc.columns:
                _avail_pts = p_loc.groupby("pitch_type").size().sort_values(ascending=False)
            else:
                _avail_pts = p_loc.groupby("pitch_type")["pitches"].sum().sort_values(ascending=False)
            _pt_options = [pt for pt in _avail_pts.index if pt in PITCH_DISPLAY]
            _pt_labels = {pt: PITCH_DISPLAY.get(pt, pt) for pt in _pt_options}
            with loc_cols[1]:
                _selected_pts = st.multiselect(
                    "Pitch types",
                    options=_pt_options,
                    format_func=lambda pt: _pt_labels[pt],
                    default=[],
                    key="hist_pitch_type_filter",
                    placeholder="All pitch types (top 4)",
                    label_visibility="collapsed",
                )

            # Detect pitch hand from arsenal data
            _p_hand = None
            _ars_for_hand = load_pitcher_arsenal()
            if not _ars_for_hand.empty and "pitch_hand" in _ars_for_hand.columns:
                _hand_rows = _ars_for_hand[_ars_for_hand["pitcher_id"] == player_id]
                if not _hand_rows.empty:
                    _p_hand = _hand_rows.iloc[0]["pitch_hand"]

            fig_loc = create_pitch_density_plotly(
                p_loc, pitch_types=_selected_pts or None,
                pitcher_name=selected_name, batter_stand=stand_filter,
                pitch_hand=_p_hand,
            )
            st.plotly_chart(fig_loc, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    else:
        # Hitter vulnerability
        if is_career:
            vuln_df = load_hitter_vulnerability(career=True)
        else:
            vuln_all = load_hitter_vulnerability_all()
            _season = selected_season if selected_season else PRIOR_SEASON
            if not vuln_all.empty and "season" in vuln_all.columns:
                vuln_df = vuln_all[vuln_all["season"] == _season]
            else:
                vuln_df = load_hitter_vulnerability(career=False)

        if not vuln_df.empty:
            h_vuln_all = vuln_df[vuln_df["batter_id"] == player_id].copy()
            if not h_vuln_all.empty:
                side_counts = h_vuln_all.groupby("batter_stand")["pitches"].sum()
                is_switch = len(side_counts) > 1 and all(v >= 50 for v in side_counts.values)
                section_label = f"Pitch-Type Profile ({season_label})"
                if is_switch:
                    platoon_side = st.radio(
                        "Batter side",
                        ["vs RHP (bats L)", "vs LHP (bats R)", "Combined"],
                        horizontal=True, key="hist_profile_platoon",
                    )
                    if platoon_side.startswith("vs RHP"):
                        h_vuln = h_vuln_all[h_vuln_all["batter_stand"] == "L"].copy()
                    elif platoon_side.startswith("vs LHP"):
                        h_vuln = h_vuln_all[h_vuln_all["batter_stand"] == "R"].copy()
                    else:
                        h_vuln = combine_platoon_vuln(h_vuln_all)
                else:
                    h_vuln = h_vuln_all
                    platoon_side = None

                st.markdown(f'<div class="tdd-section-hdr">{section_label}</div>',
                            unsafe_allow_html=True)
                table_html = build_hitter_profile_table(h_vuln)
                if table_html:
                    st.markdown(f'<div class="insight-card">{table_html}</div>', unsafe_allow_html=True)

                # Batted ball warning for pre-2022
                if selected_season and selected_season in UNRELIABLE_BB_SEASONS:
                    st.caption(
                        f"Note: Batted ball coverage was limited in {selected_season}. "
                        "xwOBA and barrel metrics may be unreliable."
                    )

        # Archetype Vulnerability
        cluster_meta = load_cluster_metadata()
        baselines = load_baselines_arch()
        if is_career:
            vuln_arch_df = load_hitter_vuln_arch_career()
            min_swings = 20
        else:
            vuln_arch_df = load_hitter_vuln_arch()
            min_swings = 10

        if not vuln_arch_df.empty and not cluster_meta.empty and not baselines.empty:
            h_vuln_arch = vuln_arch_df[vuln_arch_df["batter_id"] == player_id].copy()
            if not h_vuln_arch.empty and not is_career and "season" in h_vuln_arch.columns:
                _season = selected_season if selected_season else PRIOR_SEASON
                h_vuln_arch = h_vuln_arch[h_vuln_arch["season"] == _season]

            if not h_vuln_arch.empty:
                # Aggregate across platoon sides if multiple rows per archetype
                agg_cols = {"swings": "sum", "whiffs": "sum", "out_of_zone_pitches": "sum", "chase_swings": "sum"}
                available_agg = {k: v for k, v in agg_cols.items() if k in h_vuln_arch.columns}
                if available_agg:
                    h_vuln_arch = h_vuln_arch.groupby("pitch_archetype").agg(available_agg).reset_index()
                    h_vuln_arch["whiff_rate"] = h_vuln_arch["whiffs"] / h_vuln_arch["swings"].clip(lower=1)
                    h_vuln_arch["chase_rate"] = h_vuln_arch["chase_swings"] / h_vuln_arch["out_of_zone_pitches"].clip(lower=1)

                h_vuln_arch = h_vuln_arch[h_vuln_arch["swings"] >= min_swings]

                if not h_vuln_arch.empty:
                    # Get league baselines (average across batter hands)
                    bl_avg = baselines.groupby("pitch_archetype").agg(
                        lg_whiff=("whiff_rate", "mean"),
                        lg_chase=("chase_rate", "mean"),
                    ).reset_index()

                    h_vuln_arch = h_vuln_arch.merge(
                        cluster_meta[["pitch_archetype", "archetype_name"]],
                        on="pitch_archetype", how="left",
                    ).merge(bl_avg, on="pitch_archetype", how="left")

                    arch_vuln_rows = []
                    for _, row in h_vuln_arch.sort_values("whiff_rate", ascending=False).iterrows():
                        whiff_delta = row["whiff_rate"] - row["lg_whiff"] if pd.notna(row.get("lg_whiff")) else 0
                        chase_delta = row["chase_rate"] - row["lg_chase"] if pd.notna(row.get("lg_chase")) else 0
                        arch_vuln_rows.append({
                            "Archetype": row.get("archetype_name", f'Cluster {row["pitch_archetype"]}'),
                            "Whiff%": row["whiff_rate"],
                            "Whiff Δ": whiff_delta,
                            "Chase%": row["chase_rate"],
                            "Chase Δ": chase_delta,
                            "Swings": int(row["swings"]),
                        })

                    if arch_vuln_rows:
                        st.markdown(f'<div class="tdd-section-hdr">Archetype Vulnerability ({season_label})</div>',
                                    unsafe_allow_html=True)
                        st.caption(f"Min {min_swings} swings. Δ = vs league avg for that pitch archetype. Green = handles well, orange = vulnerable.")
                        av_df = pd.DataFrame(arch_vuln_rows)

                        def _color_delta(val: float) -> str:
                            """Negative delta (fewer whiffs/chases) is good for hitter."""
                            if val < -0.01:
                                return f"color: {POSITIVE}"
                            elif val > 0.01:
                                return f"color: {NEGATIVE}"
                            return f"color: {SLATE}"

                        styled = (
                            av_df.style
                            .format({
                                "Whiff%": "{:.1%}", "Chase%": "{:.1%}",
                                "Whiff Δ": "{:+.1%}", "Chase Δ": "{:+.1%}",
                            })
                            .map(_color_delta, subset=["Whiff Δ", "Chase Δ"])
                        )
                        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Hitter zone grid
        if is_career:
            hzone_df = load_hitter_zone_grid(career=True)
        else:
            hzone_all = load_hitter_zone_grid_all()
            _season = selected_season if selected_season else PRIOR_SEASON
            if not hzone_all.empty and "season" in hzone_all.columns:
                hzone_df = hzone_all[hzone_all["season"] == _season]
            else:
                hzone_df = load_hitter_zone_grid(career=False)

        if not hzone_df.empty:
            h_zone = hzone_df[hzone_df["batter_id"] == player_id]
            if not h_zone.empty:
                st.markdown(f'<div class="tdd-section-hdr">Zone Profile ({season_label})</div>',
                            unsafe_allow_html=True)

                # Inline controls: pitcher hand filter + pitch type filter
                _hz_ctrl_cols = st.columns([1, 3])
                with _hz_ctrl_cols[0]:
                    _hz_hand_choice = st.radio(
                        "Pitcher hand", ["All", "vs LHP", "vs RHP"],
                        horizontal=True, key="hitter_zone_hand_filter",
                        label_visibility="collapsed",
                    )
                # Map selection to batter_stand (vs LHP → batter hits from R side)
                zone_stand = {"vs LHP": "R", "vs RHP": "L"}.get(_hz_hand_choice)

                # Pitch-type filter (only when data includes pitch_type column)
                _hz_selected_pts: list[str] | None = None
                if "pitch_type" in h_zone.columns:
                    _hz_avail_pts = (
                        h_zone.groupby("pitch_type")["pitches"].sum()
                        .sort_values(ascending=False)
                    )
                    _hz_pt_options = [pt for pt in _hz_avail_pts.index if pt in PITCH_DISPLAY]
                    _hz_pt_labels = {pt: PITCH_DISPLAY.get(pt, pt) for pt in _hz_pt_options}
                    with _hz_ctrl_cols[1]:
                        _hz_selected_pts = st.multiselect(
                            "Pitch types",
                            options=_hz_pt_options,
                            format_func=lambda pt: _hz_pt_labels[pt],
                            default=[],
                            key="hitter_zone_pitch_type_filter",
                            placeholder="All pitch types",
                            label_visibility="collapsed",
                        )

                zone_cols = st.columns(2)
                with zone_cols[0]:
                    fig_whiff = create_hitter_zone_plotly(
                        h_zone, metric="whiff_rate",
                        batter_name=selected_name, batter_stand=zone_stand,
                        pitch_types=_hz_selected_pts or None,
                    )
                    st.plotly_chart(fig_whiff, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False}, key="hzone_whiff")
                with zone_cols[1]:
                    fig_xwoba = create_hitter_zone_plotly(
                        h_zone, metric="xwoba",
                        batter_name=selected_name, batter_stand=zone_stand,
                        pitch_types=_hz_selected_pts or None,
                    )
                    st.plotly_chart(fig_xwoba, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False}, key="hzone_xwoba")

                if selected_season and selected_season in UNRELIABLE_BB_SEASONS:
                    st.caption(
                        f"Note: Batted ball coverage was limited in {selected_season}. "
                        "xwOBA zone data may be unreliable."
                    )


def render_season_trends(
    player_type: str,
    player_id: int,
    selected_name: str,
    selected_season: int | None = None,
) -> None:
    """Render year-over-year trend charts for key stats."""
    full_df = load_full_stats(player_type.lower())
    if full_df.empty:
        tdd_info("No historical stats available for season trends.")
        return

    id_col = "pitcher_id" if player_type == "Pitcher" else "batter_id"

    # Supplement with woba from traditional stats (not in full_stats)
    if player_type == "Hitter":
        trad_all = load_traditional_stats_all("hitter")
        if not trad_all.empty and "woba" in trad_all.columns and "woba" not in full_df.columns:
            woba_df = trad_all[[id_col, "season", "woba"]].drop_duplicates()
            full_df = full_df.merge(woba_df, on=[id_col, "season"], how="left")

    player_data = full_df[full_df[id_col] == player_id].sort_values("season")
    if len(player_data) < 2:
        tdd_info("Not enough seasons to display trends (need at least two).")
        return

    seasons = player_data["season"].values

    if player_type == "Pitcher":
        trend_stats = [
            ("K%", "k_rate", True, True),
            ("BB%", "bb_rate", False, True),
            ("Avg Velo", "avg_velo", True, False),
            ("Whiff%", "whiff_rate", True, True),
            ("CSW%", "csw_pct", True, True),
            ("Zone%", "zone_pct", True, True),
        ]
    else:
        trend_stats = [
            ("K%", "k_rate", False, True),
            ("BB%", "bb_rate", True, True),
            ("wOBA", "woba", True, False),
            ("Avg EV", "avg_exit_velo", True, False),
            ("Whiff%", "whiff_rate", False, True),
            ("Chase%", "chase_rate", False, True),
            ("Barrel%", "barrel_pct", True, True),
        ]

    # Filter to stats that have data
    available_stats = []
    for label, key, hb, is_pct in trend_stats:
        if key in player_data.columns and player_data[key].notna().sum() >= 2:
            available_stats.append((label, key, hb, is_pct))
    if not available_stats:
        return

    # League-wide y-axis ranges so small personal changes don't look extreme
    # These represent the typical range a viewer should mentally anchor to
    _Y_RANGES: dict[str, tuple[float, float]] = {
        "k_rate": (10, 35),      # K% in pct
        "bb_rate": (3, 15),      # BB% in pct
        "avg_velo": (88, 100),   # mph
        "avg_exit_velo": (83, 95),  # mph
        "whiff_rate": (15, 40),  # pct
        "chase_rate": (20, 40),  # pct
        "csw_pct": (22, 38),     # pct
        "zone_pct": (38, 55),    # pct
        "barrel_pct": (2, 18),   # pct
        "woba": (0.280, 0.420),  # raw scale (not pct)
    }

    # Build plotly trend subplots
    def _build_trend_plotly(stats: list, n_cols: int, row_height: int = 180):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        n = len(stats)
        n_r = (n + n_cols - 1) // n_cols
        subplot_titles = [s[0] for s in stats]

        fig = make_subplots(
            rows=n_r, cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12 if n_r > 1 else 0.2,
            horizontal_spacing=0.08,
        )

        for idx, (label, key, higher_better, is_pct) in enumerate(stats):
            row_idx = idx // n_cols + 1
            col_idx = idx % n_cols + 1

            vals = player_data[key].values
            valid_mask = ~pd.isna(vals)
            valid_seasons = seasons[valid_mask]
            valid_vals = vals[valid_mask].astype(float)
            if is_pct:
                valid_vals = valid_vals * 100

            # Main trend line
            fig.add_trace(go.Scatter(
                x=valid_seasons.tolist(),
                y=valid_vals.tolist(),
                mode="lines+markers",
                line=dict(color=GOLD, width=2),
                marker=dict(size=5, color=GOLD),
                hovertemplate="%{x}: %{y:.1f}" + ("%" if is_pct else "") + "<extra></extra>",
                showlegend=False,
            ), row=row_idx, col=col_idx)

            # Highlight selected season
            if selected_season and selected_season in valid_seasons:
                sel_idx = list(valid_seasons).index(selected_season)
                fig.add_trace(go.Scatter(
                    x=[int(selected_season)],
                    y=[float(valid_vals[sel_idx])],
                    mode="markers",
                    marker=dict(size=9, color=SAGE),
                    hoverinfo="skip",
                    showlegend=False,
                ), row=row_idx, col=col_idx)

            # Y-axis range
            if key in _Y_RANGES:
                y_lo, y_hi = _Y_RANGES[key]
                data_lo, data_hi = float(valid_vals.min()), float(valid_vals.max())
                y_lo = min(y_lo, data_lo - 1)
                y_hi = max(y_hi, data_hi + 1)
            else:
                data_lo, data_hi = float(valid_vals.min()), float(valid_vals.max())
                pad = max((data_hi - data_lo) * 0.3, 2.0)
                y_lo, y_hi = data_lo - pad, data_hi + pad

            suffix = "%" if is_pct else ""
            xaxis_key = f"xaxis{idx + 1}" if idx > 0 else "xaxis"
            yaxis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
            fig.update_layout(**{
                xaxis_key: dict(
                    range=[TRAIN_START - 0.5, PRIOR_SEASON + 0.5],
                    tickfont=dict(color=SLATE, size=9),
                    showgrid=False,
                    dtick=2,
                    showline=True, linecolor=SLATE, linewidth=0.5,
                ),
                yaxis_key: dict(
                    range=[y_lo, y_hi],
                    tickfont=dict(color=SLATE, size=9),
                    ticksuffix=suffix,
                    showgrid=True, gridcolor="rgba(123,143,166,0.12)", gridwidth=0.5,
                    showline=True, linecolor=SLATE, linewidth=0.5,
                    zeroline=False,
                ),
            })

            # Delta annotation
            if len(valid_vals) >= 2:
                delta = valid_vals[-1] - valid_vals[-2]
                if abs(delta) > 0.1:
                    arrow = "+" if delta > 0 else ""
                    clr = SAGE if (delta > 0) == higher_better else EMBER
                    fmt_delta = f"{arrow}{delta:.1f}{suffix}"
                    fig.add_annotation(
                        x=int(valid_seasons[-1]),
                        y=float(valid_vals[-1]),
                        text=f"<b>{fmt_delta}</b>",
                        showarrow=False,
                        font=dict(color=clr, size=10),
                        xanchor="left", xshift=8, yshift=8,
                        xref=f"x{idx + 1}" if idx > 0 else "x",
                        yref=f"y{idx + 1}" if idx > 0 else "y",
                    )

        # Style subplot titles
        for ann in fig.layout.annotations:
            ann.update(font=dict(color=CREAM, size=12))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=row_height * n_r,
            showlegend=False,
        )

        return fig

    st.markdown('<div class="tdd-section-hdr">Season Trends</div>',
                unsafe_allow_html=True)

    n_stats = len(available_stats)
    n_cols = min(3, n_stats)
    trend_fig = _build_trend_plotly(available_stats, n_cols=n_cols, row_height=200)

    st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
    st.caption(
        f"Year-over-year trends for {selected_name}. "
        f"{'Green dot' if selected_season else 'Gold line'} = {'selected season' if selected_season else 'trajectory'}. "
        f"Delta annotation shows change from prior season."
    )


def render_arsenal_evolution(
    player_id: int,
    selected_name: str,
    selected_season: int,
) -> None:
    """Show pitcher arsenal changes vs prior year."""
    arsenal_all = load_pitcher_arsenal_all()
    if arsenal_all.empty or "season" in arsenal_all.columns and selected_season <= 2018:
        tdd_info("No arsenal data available for comparison.")
        return

    curr = arsenal_all[
        (arsenal_all["pitcher_id"] == player_id) & (arsenal_all["season"] == selected_season)
    ]
    prev = arsenal_all[
        (arsenal_all["pitcher_id"] == player_id) & (arsenal_all["season"] == selected_season - 1)
    ]

    if curr.empty or prev.empty:
        tdd_info(f"No arsenal comparison available (need data for both {selected_season} and {selected_season - 1}).")
        return

    # Merge on pitch_type
    merged = curr.merge(
        prev[["pitch_type", "usage_pct", "avg_velo", "whiff_rate", "csw_pct"]],
        on="pitch_type", how="outer", suffixes=("", "_prev"),
    )

    if merged.empty:
        return

    # Build delta table
    rows = []
    for _, r in merged.sort_values("usage_pct", ascending=False, na_position="last").iterrows():
        pt = r["pitch_type"]
        usage = r.get("usage_pct")
        usage_prev = r.get("usage_pct_prev")
        velo = r.get("avg_velo")
        velo_prev = r.get("avg_velo_prev")
        whiff = r.get("whiff_rate")
        whiff_prev = r.get("whiff_rate_prev")
        csw = r.get("csw_pct")
        csw_prev = r.get("csw_pct_prev")

        def _delta_fmt(curr_v, prev_v, is_pct=True, decimals=1):
            if pd.isna(curr_v) and pd.isna(prev_v):
                return "", "", ""
            if pd.isna(prev_v):
                val_str = f"{curr_v*100:.{decimals}f}%" if is_pct else f"{curr_v:.{decimals}f}"
                return val_str, "", "NEW"
            if pd.isna(curr_v):
                return "DROPPED", f"{prev_v*100:.{decimals}f}%" if is_pct else f"{prev_v:.{decimals}f}", ""
            d = curr_v - prev_v
            val_str = f"{curr_v*100:.{decimals}f}%" if is_pct else f"{curr_v:.{decimals}f}"
            if is_pct:
                delta_str = f"{d*100:+.{decimals}f}pp"
            else:
                delta_str = f"{d:+.{decimals}f}"
            return val_str, f"{prev_v*100:.{decimals}f}%" if is_pct else f"{prev_v:.{decimals}f}", delta_str

        u_curr, _, u_delta = _delta_fmt(usage, usage_prev)
        v_curr, _, v_delta = _delta_fmt(velo, velo_prev, is_pct=False)
        w_curr, _, w_delta = _delta_fmt(whiff, whiff_prev)
        c_curr, _, c_delta = _delta_fmt(csw, csw_prev)

        rows.append({
            "Pitch": pt,
            "Usage%": u_curr,
            "Usage \u0394": u_delta,
            "Velo": v_curr,
            "Velo \u0394": v_delta,
            "Whiff%": w_curr,
            "Whiff \u0394": w_delta,
            "CSW%": c_curr,
            "CSW \u0394": c_delta,
        })

    if rows:
        st.markdown(
            f'<div class="tdd-section-hdr">Arsenal Changes vs {selected_season - 1}</div>',
            unsafe_allow_html=True,
        )
        delta_df = pd.DataFrame(rows)
        st.dataframe(delta_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Year-over-year arsenal evolution. "
            f"NEW = pitch added in {selected_season}. "
            f"DROPPED = pitch no longer thrown."
        )


# ---------------------------------------------------------------------------
# Page: Player Profile (Editorial Layout)
# ---------------------------------------------------------------------------


def _editorial_hero_html(
    name: str,
    team: str,
    header_parts: list[str],
    diamond_html: str,
    tools_html: str,
    injury_html: str,
    vitals: list[tuple[str, str, str]],
    scouting_text: str,
    player_id: int | None = None,
) -> str:
    """Build the editorial hero: headshot portrait + identity + vitals + scouting callout."""
    from components.headshot import _headshot_url

    # Vitals grid
    vitals_cells = ""
    for v, l, d in vitals:
        d_cls = "pos" if d.startswith("+") else "neg" if d.startswith("-") else ""
        d_html = f'<div class="d {d_cls}">{esc(d)}</div>' if d else ""
        vitals_cells += (
            '<div class="vstat">'
            f'<div class="v">{esc(v)}</div>'
            f'<div class="l">{esc(l)}</div>'
            f'{d_html}'
            '</div>'
        )

    # Scouting callout
    scout_html = ""
    if scouting_text:
        scout_html = (
            '<div class="scouting">'
            '<span class="eyebrow">Model Read</span>'
            f'<p>{esc(scouting_text)}</p>'
            '</div>'
        )

    team_attr = f' data-team="{esc(team)}"' if team else ""
    sub_line = " | ".join(header_parts) if header_parts else ""

    # Portrait: real MLB headshot
    if player_id:
        hs_url = _headshot_url(player_id, 400)
        portrait_content = (
            f'<img src="{hs_url}" alt="{esc(name)}" '
            f'style="width:100%;height:100%;object-fit:cover;object-position:top center;" '
            f'onerror="this.style.display=\'none\'" />'
        )
    else:
        initials = "".join(w[0] for w in name.split()[:2]) if name else ""
        portrait_content = f'<span class="initials">{esc(initials)}</span>'

    return (
        '<div class="tdd-hero-player editorial">'
        # Portrait column
        f'<div class="portrait" style="background-color:var(--tdd-dark-card)">'
        f'{portrait_content}'
        f'<div class="team-strip"{team_attr}>{esc(team)}</div>'
        '</div>'
        # Body column
        '<div class="body">'
        '<div class="top">'
        '<div class="idblock">'
        '<div class="eyebrow">Player Profile</div>'
        f'<h1>{esc(name)}</h1>'
        f'<div class="sub">{esc(sub_line)}</div>'
        f'{injury_html}'
        '</div>'
        f'<div class="rating">{diamond_html}{tools_html}</div>'
        '</div>'
        # Vitals grid
        f'<div class="vitals">{vitals_cells}</div>'
        # Scouting callout
        f'{scout_html}'
        '</div>'
        '</div>'
    )


def _section_head(title: str, sub: str = "") -> str:
    """Return a .p-section header."""
    sub_html = f'<span class="p-shead-sub">{esc(sub)}</span>' if sub else ""
    return (
        '<div class="p-shead">'
        f'<h2>{esc(title)}</h2>'
        f'{sub_html}'
        '</div>'
    )


def page_player_profile() -> None:
    """Deep dive into a single player's projections -- editorial layout."""

    qp_player_id = st.query_params.get("player_id", "")

    # Load both pitcher and hitter projections into a unified list
    pitcher_df = load_projections("pitcher")
    hitter_df = load_projections("hitter")
    team_lookup = get_team_lookup()

    # Build unified player list: (display_name, player_id, player_type)
    # Two-way players (e.g. Ohtani) appear in both dataframes |detect and
    # merge into a single "Two-Way" entry instead of showing duplicates.
    pitcher_ids = set(pitcher_df["pitcher_id"].astype(int))
    hitter_ids = set(hitter_df["batter_id"].astype(int))
    two_way_ids = pitcher_ids & hitter_ids

    _all_players: list[tuple[str, int, str]] = []
    for _, pr in pitcher_df.iterrows():
        pid = int(pr["pitcher_id"])
        if pid in two_way_ids:
            continue  # handled below as Two-Way
        pname = pr["pitcher_name"]
        team = team_lookup.get(pid, "")
        _all_players.append((f"{pname} ({team})" if team else pname, pid, "Pitcher"))
    for _, pr in hitter_df.iterrows():
        pid = int(pr["batter_id"])
        pname = pr["batter_name"]
        team = team_lookup.get(pid, "")
        if pid in two_way_ids:
            _all_players.append((f"{pname} ({team}) [Two-Way]" if team else f"{pname} [Two-Way]", pid, "Two-Way"))
        else:
            _all_players.append((f"{pname} ({team})" if team else pname, pid, "Hitter"))

    if not _all_players:
        tdd_warn("No projection data found.")
        return

    # Collect all teams for filter
    all_teams = sorted({team_lookup.get(pid, "") for _, pid, _ in _all_players} - {""})

    # Deep link support
    deep_link_player_id: int | None = None
    if qp_player_id.isdigit():
        deep_link_player_id = int(qp_player_id)

    # Inline toolbar: Team | Player | Season
    sel_cols = st.columns([1, 3, 1])
    with sel_cols[0]:
        team_display = ["All Teams"] + [team_full(a) for a in all_teams]
        default_team_idx = 0
        if deep_link_player_id is not None:
            linked_team = team_lookup.get(deep_link_player_id, "")
            linked_display = team_full(linked_team) if linked_team else ""
            if linked_display in team_display:
                default_team_idx = team_display.index(linked_display)
        profile_team_display = st.selectbox(
            "Team", team_display, index=default_team_idx, key="profile_team",
            label_visibility="collapsed",
        )
        # Map display name back to abbreviation for filtering
        _abbr_map = {team_full(a): a for a in all_teams}
        profile_team_filter = _abbr_map.get(profile_team_display, profile_team_display)

    # Filter players by team
    if profile_team_filter != "All Teams":
        filtered_players = [
            (dname, pid, pt) for dname, pid, pt in _all_players
            if team_lookup.get(pid, "") == profile_team_filter
        ]
    else:
        filtered_players = _all_players
    filtered_players.sort(key=lambda x: x[0])

    with sel_cols[1]:
        display_names = [dname for dname, _, _ in filtered_players]
        default_player_idx = 0
        if deep_link_player_id is not None:
            for i, (_, pid, _) in enumerate(filtered_players):
                if pid == deep_link_player_id:
                    default_player_idx = i
                    break
        selected_display = st.selectbox(
            "Player", display_names, index=default_player_idx,
            key="profile_player", label_visibility="collapsed",
        )
    selected_idx = display_names.index(selected_display)
    _, player_id, player_type = filtered_players[selected_idx]

    # Resolve player row from the correct dataframe
    is_two_way_player = player_type == "Two-Way"
    two_way_pitcher_row = None
    if is_two_way_player:
        # Two-way: load both profiles, display hitter as primary
        name_col, id_col, hand_col = "batter_name", "batter_id", "batter_stand"
        stat_configs = HITTER_STATS
        df = hitter_df
        # Also grab pitcher row for pitching grades
        p_match = pitcher_df[pitcher_df["pitcher_id"] == player_id]
        if not p_match.empty:
            two_way_pitcher_row = p_match.iloc[0]
        player_type = "Hitter"  # render as hitter for downstream
    elif player_type == "Pitcher":
        name_col, id_col, hand_col = "pitcher_name", "pitcher_id", "pitch_hand"
        stat_configs = PITCHER_STATS
        df = pitcher_df
    else:
        name_col, id_col, hand_col = "batter_name", "batter_id", "batter_stand"
        stat_configs = HITTER_STATS
        df = hitter_df

    player_row = df[df[id_col] == player_id].iloc[0]
    selected_name = player_row[name_col]

    st.query_params["player_id"] = str(player_id)
    st.query_params["player_type"] = player_type.lower()

    with sel_cols[2]:
        season_choice = season_selector("profile", include_career=True, label_visibility="collapsed")
    is_projection = season_choice == PROJECTION_LABEL
    is_career = season_choice == "Career"
    selected_season = None if is_projection or is_career else int(season_choice)

    # === EDITORIAL LAYOUT ==========================================
    # Wrap entire page in .tdd-profile container
    st.markdown('<div class="tdd-profile">', unsafe_allow_html=True)

    # --- Header card ---
    teams_df = load_player_teams()
    player_team = ""
    if not teams_df.empty:
        team_row = teams_df[teams_df["player_id"] == player_id]
        if not team_row.empty:
            player_team = team_row.iloc[0].get("team_abbr", "")

    hand = player_row.get(hand_col, "")
    age = int(player_row["age"]) if pd.notna(player_row.get("age")) else "?"
    role = ""
    if player_type == "Pitcher" and "is_starter" in player_row.index:
        role = "SP" if player_row["is_starter"] else "RP"

    # Skill tier label
    _TIER_LABELS = {0: "Below-Avg", 1: "Average", 2: "Above-Avg", 3: "Elite"}
    skill_tier = int(player_row.get("skill_tier", 1)) if pd.notna(player_row.get("skill_tier")) else None
    tier_label = _TIER_LABELS.get(skill_tier, "") if skill_tier is not None else ""

    header_parts = []
    if player_team:
        header_parts.append(
            f'<span class="tdd-team-abbr" data-team="{player_team}">{team_full(player_team)}</span>'
        )
    header_parts.append(f"Age {age}")
    if hand:
        if player_type == "Pitcher":
            header_parts.append("LHP" if hand == "L" else "RHP")
        else:
            header_parts.append(f"Bats {'L' if hand == 'L' else 'R'}")
    if role:
        header_parts.append(role)
    if tier_label:
        header_parts.append(f"Skill Tier: {tier_label}")

    # Archetype badge
    if player_type == "Pitcher":
        _arch_df = load_pitcher_archetypes()
        if not _arch_df.empty:
            _arch_row = _arch_df[_arch_df["pitcher_id"] == player_id]
            if not _arch_row.empty:
                header_parts.append(_arch_row.iloc[0]["archetype_name"])
    else:
        _arch_df = load_hitter_archetypes()
        if not _arch_df.empty:
            _arch_row = _arch_df[_arch_df["batter_id"] == player_id]
            if not _arch_row.empty:
                header_parts.append(_arch_row.iloc[0]["archetype_name"])

    # Breakout tier badge (hitters only)
    _breakout_tier_label = ""
    _breakout_hole = ""
    if player_type == "Hitter":
        _bo_df = load_hitter_breakout_candidates()
        if not _bo_df.empty:
            _bo_row = _bo_df[_bo_df["batter_id"] == player_id]
            if not _bo_row.empty:
                _bt = _bo_row.iloc[0].get("breakout_tier", "")
                _bh = _bo_row.iloc[0].get("breakout_hole", "")
                _b_type = _bo_row.iloc[0].get("breakout_type", "")
                if _bt:
                    _breakout_tier_label = f"Breakout: {_bt} ({_b_type})"
                    header_parts.append(_breakout_tier_label)
                if pd.notna(_bh) and _bh:
                    _breakout_hole = str(_bh)

    # Park factor for hitters
    if player_type == "Hitter":
        counting_df = load_counting("hitter")
        if not counting_df.empty:
            c_row = counting_df[counting_df["batter_id"] == player_id]
            if not c_row.empty and "hr_park_factor" in c_row.columns:
                pf = c_row.iloc[0].get("hr_park_factor")
                if pd.notna(pf) and abs(pf - 1.0) > 0.005:
                    pf_label = f"HR Park: {pf:.3f}"
                    header_parts.append(pf_label)

    # Diamond rating — always derived from tdd_value_score via score_to_diamonds
    from lib.diamond_rating import score_to_diamonds
    _ranks = load_rankings("hitters") if player_type != "Pitcher" else load_rankings("pitchers")
    _id_col = "batter_id" if player_type != "Pitcher" else "pitcher_id"
    _rank_row = _ranks[_ranks[_id_col] == player_id] if not _ranks.empty else pd.DataFrame()
    if not _rank_row.empty and "tdd_value_score" in _rank_row.columns and pd.notna(_rank_row["tdd_value_score"].iloc[0]):
        _tvs = float(_rank_row["tdd_value_score"].iloc[0])
        _dr = score_to_diamonds(_tvs)
        diamond_html = diamond_rating_html(0, size="lg", precomputed=_dr)
    else:
        composite = player_row["composite_score"]
        diamond_html = diamond_rating_html_composite(composite, size="lg")

    # Load grade confidence intervals
    _grade_ci_df = load_pitcher_grade_ci() if player_type == "Pitcher" else load_hitter_grade_ci()
    _grade_ci_row = pd.DataFrame()
    if not _grade_ci_df.empty:
        _grade_ci_row = _grade_ci_df[_grade_ci_df["player_id"] == player_id]

    # Tools Grade — scouting-based rating (shown below diamond rating with tooltip)
    _tools_html = ""
    if not _rank_row.empty and "tools_rating" in _rank_row.columns and pd.notna(_rank_row["tools_rating"].iloc[0]):
        _tools_val = float(_rank_row["tools_rating"].iloc[0])
        _role_label = "pitchers" if player_type == "Pitcher" else "hitters"
        # Add diamond rating CI range if available
        _dr_ci_html = ""
        if not _grade_ci_row.empty:
            _ci = _grade_ci_row.iloc[0]
            _dr_lo = _ci.get("diamond_rating_lo")
            _dr_hi = _ci.get("diamond_rating_hi")
            if pd.notna(_dr_lo) and pd.notna(_dr_hi):
                _dr_ci_html = (
                    f' <span style="color:{SLATE}; font-size:0.65rem; opacity:0.7;">'
                    f'({_dr_lo:.1f}-{_dr_hi:.1f})</span>'
                )
        _tools_html = (
            f'<div style="margin-top:4px; font-size:0.8rem; cursor:help;" '
            f'title="Tools Grade: scaled 1-10 against all {_role_label} in the system based on scouting grades (20-80 scale)">'
            f'<span style="color:{SLATE}; font-size:0.7rem;">TOOLS </span>'
            f'<span style="color:{GOLD}; font-weight:600;">{_tools_val:.1f}</span>'
            f'{_dr_ci_html}'
            f'</div>'
        )

    # Enrich role/position in header with rank from rankings data
    if not _rank_row.empty:
        _rr = _rank_row.iloc[0]
        if player_type == "Pitcher" and role:
            _role_rank = _rr.get("role_rank")
            if pd.notna(_role_rank):
                for i, part in enumerate(header_parts):
                    if part == role:
                        header_parts[i] = f"{role} #{int(_role_rank)}"
                        break
        elif player_type != "Pitcher":
            _pos = _rr.get("position", "")
            _pos_rank = _rr.get("pos_rank")
            if _pos and pd.notna(_pos_rank):
                header_parts.append(f"{_pos} #{int(_pos_rank)}")

    # Injury status
    injury_lookup = get_injury_lookup()
    inj_info = injury_lookup.get(player_id)
    injury_html = ""
    if inj_info and inj_info["missed_games"] > 0:
        inj_color = EMBER if inj_info["severity"] == "major" else GOLD
        injury_html = (
            f'<div class="tdd-meta" style="color:{inj_color}; margin-top:4px;">'
            f'{inj_info["status"]} |{inj_info["injury"]} '
            f'(est. return: {inj_info["est_return"]}, ~{inj_info["missed_games"]}G missed)'
            f'</div>'
        )

    # Build vitals from Bayesian projections (available before trad_all_df loads)
    vitals: list[tuple[str, str, str]] = []
    if player_type in ("Hitter", "Two-Way"):
        for col, label in [("projected_k_rate", "K%"), ("projected_bb_rate", "BB%")]:
            v = player_row.get(col)
            if pd.notna(v):
                vitals.append((f"{float(v)*100:.1f}%", f"Proj {label}", ""))
    else:
        for col, label in [("projected_k_rate", "K%"), ("projected_bb_rate", "BB%"), ("projected_hr_per_bf", "HR/BF")]:
            v = player_row.get(col)
            if pd.notna(v):
                if col == "projected_hr_per_bf":
                    vitals.append((f"{float(v)*100:.1f}%", f"Proj {label}", ""))
                else:
                    vitals.append((f"{float(v)*100:.1f}%", f"Proj {label}", ""))

    # Get scouting summary for callout
    scouting_text = ""
    try:
        _is_h = player_type in ("Hitter", "Two-Way")
        _sc_arch_df = load_hitter_archetypes() if _is_h else load_pitcher_archetypes()
        _arch_id_col = "batter_id" if _is_h else "pitcher_id"
        if not _sc_arch_df.empty:
            _ar = _sc_arch_df[_sc_arch_df[_arch_id_col] == player_id]
            if not _ar.empty:
                arch_name = _ar.iloc[0].get("archetype_name", "")
                if arch_name:
                    scouting_text = f"{arch_name} profile."
    except Exception:
        pass

    # Render editorial hero
    st.markdown(
        _editorial_hero_html(
            name=selected_name,
            team=player_team,
            header_parts=header_parts,
            diamond_html=diamond_html,
            tools_html=_tools_html,
            injury_html=injury_html,
            vitals=vitals,
            scouting_text=scouting_text,
            player_id=player_id,
        ),
        unsafe_allow_html=True,
    )

    # === SECTIONS (editorial single-column flow) ===================

    # --- Scouting grades ---
    if not is_two_way_player and not _rank_row.empty:
        _rr_grades = _rank_row.iloc[0]
        _ci_data = _grade_ci_row.iloc[0] if not _grade_ci_row.empty else None
        if player_type == "Pitcher":
            _grade_skills = [("Stuff", "grade_stuff"), ("Command", "grade_command"), ("Durability", "grade_durability")]
        else:
            _grade_skills = [("Contact", "grade_hit"), ("Power", "grade_power"), ("Speed", "grade_speed"), ("Fielding", "grade_fielding"), ("Discipline", "grade_discipline")]

        _grade_bars: list[tuple[str, float, float | None, float | None]] = []
        for _lbl, _col in _grade_skills:
            _v = _rr_grades.get(_col)
            if pd.notna(_v):
                _lo = _ci_data.get(f"{_col}_lo") if _ci_data is not None else None
                _hi = _ci_data.get(f"{_col}_hi") if _ci_data is not None else None
                _grade_bars.append((_lbl, float(_v), float(_lo) if pd.notna(_lo) else None, float(_hi) if pd.notna(_hi) else None))
        if _grade_bars:
            st.markdown(grade_bar_block(_grade_bars), unsafe_allow_html=True)

    # --- Two-Way Player: show both batting + pitching scouting grades ---
    if is_two_way_player and two_way_pitcher_row is not None:
        # Load rankings for scouting grades
        h_ranks = load_rankings("hitters")
        p_ranks = load_rankings("pitchers")
        h_row = h_ranks[h_ranks["batter_id"] == player_id]
        p_row = p_ranks[p_ranks["pitcher_id"] == player_id]

        # Load grade CIs for two-way player
        _tw_h_ci_df = load_hitter_grade_ci()
        _tw_p_ci_df = load_pitcher_grade_ci()
        _tw_h_ci = _tw_h_ci_df[_tw_h_ci_df["player_id"] == player_id] if not _tw_h_ci_df.empty else pd.DataFrame()
        _tw_p_ci = _tw_p_ci_df[_tw_p_ci_df["player_id"] == player_id] if not _tw_p_ci_df.empty else pd.DataFrame()

        tw_parts = []
        if not h_row.empty:
            hr = h_row.iloc[0]
            _tw_hci = _tw_h_ci.iloc[0] if not _tw_h_ci.empty else None
            h_tvs = hr.get("tdd_value_score")
            h_dr_s = f"{score_to_diamonds(h_tvs):.1f}" if pd.notna(h_tvs) else ""
            _h_grade_parts = []
            for _lbl, _col in [("Contact", "grade_hit"), ("Power", "grade_power"), ("Speed", "grade_speed"), ("Discipline", "grade_discipline")]:
                _gv = int(hr.get(_col, 0))
                _ci_str = ""
                if _tw_hci is not None:
                    _lo = _tw_hci.get(f"{_col}_lo")
                    _hi = _tw_hci.get(f"{_col}_hi")
                    if pd.notna(_lo) and pd.notna(_hi):
                        _ci_str = f'<span style="opacity:0.6; font-size:0.8em;">({int(_lo)}-{int(_hi)})</span>'
                _h_grade_parts.append(f'{_lbl}:{_gv}{_ci_str}')
            tw_parts.append(
                f'<span style="color:var(--tdd-gold); font-weight:700;">Batting: {h_dr_s}</span>'
                f' <span style="color:var(--tdd-slate); font-size:0.85em;">'
                f'{" ".join(_h_grade_parts)}</span>'
            )
        if not p_row.empty:
            pr = p_row.iloc[0]
            _tw_pci = _tw_p_ci.iloc[0] if not _tw_p_ci.empty else None
            p_tvs = pr.get("tdd_value_score")
            p_dr_s = f"{score_to_diamonds(p_tvs):.1f}" if pd.notna(p_tvs) else ""
            _p_grade_parts = []
            for _lbl, _col in [("Stuff", "grade_stuff"), ("Command", "grade_command"), ("Durability", "grade_durability")]:
                _gv = int(pr.get(_col, 0))
                _ci_str = ""
                if _tw_pci is not None:
                    _lo = _tw_pci.get(f"{_col}_lo")
                    _hi = _tw_pci.get(f"{_col}_hi")
                    if pd.notna(_lo) and pd.notna(_hi):
                        _ci_str = f'<span style="opacity:0.6; font-size:0.8em;">({int(_lo)}-{int(_hi)})</span>'
                _p_grade_parts.append(f'{_lbl}:{_gv}{_ci_str}')
            tw_parts.append(
                f'<span style="color:var(--tdd-gold); font-weight:700;">Pitching: {p_dr_s}</span>'
                f' <span style="color:var(--tdd-slate); font-size:0.85em;">'
                f'{" ".join(_p_grade_parts)}</span>'
            )
        if tw_parts:
            tw_html = (
                f'<div style="background:transparent; border:none; '
                f'border-left:3px solid var(--tdd-gold); padding:10px 16px; margin:8px 0;">'
                f'<span style="color:var(--tdd-gold); font-size:0.8rem; font-weight:600;">TWO-WAY PLAYER</span>'
                f'<div style="display:flex; gap:24px; margin-top:4px;">'
                + "".join(f"<div>{p}</div>" for p in tw_parts)
                + f'</div></div>'
            )
            st.markdown(tw_html, unsafe_allow_html=True)

    # ===================================================================
    # LAYOUT: Temporal zones
    #   1. Career Context (compact, always visible)
    #   2. Season (projections + advanced + traditional + counting/EOS)
    #   3. Scouting / Tools
    #   4. Deep Dive (season selector controls pitch profiles, trends)
    # ===================================================================

    # Load data needed for multiple sections
    trad_all_df = load_traditional_stats_all(player_type.lower())
    trad_current = load_traditional_stats(player_type.lower())
    if not trad_current.empty:
        # Append current-season rows (avoid duplicates if already present)
        if not trad_all_df.empty:
            existing_seasons = set(trad_all_df["season"].unique())
            trad_new = trad_current[~trad_current["season"].isin(existing_seasons)]
            if not trad_new.empty:
                trad_all_df = pd.concat([trad_all_df, trad_new], ignore_index=True)
        else:
            trad_all_df = trad_current
    counting_df = load_counting(player_type.lower())

    # Detect in-season state: if current season has games, show that instead
    _current_season_games = 0
    if not trad_all_df.empty:
        _cs_data = trad_all_df[
            (trad_all_df[id_col] == player_id)
            & (trad_all_df["season"] == CURRENT_SEASON)
        ]
        if not _cs_data.empty and "games" in _cs_data.columns:
            _current_season_games = int(_cs_data.iloc[0].get("games", 0))

    _in_season = _current_season_games >= 5
    _recent_season = CURRENT_SEASON if _in_season else PRIOR_SEASON
    _recent_label = f"{CURRENT_SEASON} ({_current_season_games}G)" if _in_season else str(PRIOR_SEASON)

    # ── 1. CAREER CONTEXT (compact) ──────────────────────────────────
    # Derive rates from counting stats since trad parquet has counts not rates
    if not trad_all_df.empty:
        career_player = trad_all_df[trad_all_df[id_col] == player_id]
        if not career_player.empty and len(career_player) >= 1:
            n_seasons = career_player["season"].nunique()
            _exp_col = "pa" if "pa" in career_player.columns else "bf"
            _c_pa = career_player[_exp_col].sum()
            _c_k = career_player["k"].sum() if "k" in career_player.columns else 0
            _c_bb = career_player["bb"].sum() if "bb" in career_player.columns else 0
            _c_hr = int(career_player.get("hr", career_player.get("hr_allowed", pd.Series([0]))).sum())
            _c_k_pct = _c_k / _c_pa if _c_pa > 0 else 0
            _c_bb_pct = _c_bb / _c_pa if _c_pa > 0 else 0
            # PA/BF-weighted averages for rate stats
            _wt = career_player[_exp_col]
            _c_avg = (_wt * career_player["avg"]).sum() / _c_pa if _c_pa > 0 and "avg" in career_player.columns else 0
            _c_obp = (_wt * career_player["obp"]).sum() / _c_pa if _c_pa > 0 and "obp" in career_player.columns else 0
            _c_slg = (_wt * career_player["slg"]).sum() / _c_pa if _c_pa > 0 and "slg" in career_player.columns else 0
            _c_woba = (_wt * career_player["woba"]).sum() / _c_pa if _c_pa > 0 and "woba" in career_player.columns else 0

            if player_type in ("Hitter", "Two-Way"):
                career_line = (
                    f'<span style="color:{CREAM}; font-weight:600;">{n_seasons} seasons</span>'
                    f' <span style="color:{SLATE};">|</span> '
                    f'.{int(_c_avg * 1000):03d}'
                    f'/.{int(_c_obp * 1000):03d}'
                    f'/.{int(_c_slg * 1000):03d}'
                    f' <span style="color:{SLATE};">|</span> '
                    f'.{int(_c_woba * 1000):03d} wOBA'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_k_pct:.1%} K'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_bb_pct:.1%} BB'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_hr} HR'
                )
            else:
                _c_ip = career_player["ip"].sum() if "ip" in career_player.columns else 0
                _c_er = career_player["er"].sum() if "er" in career_player.columns else 0
                _c_era = (_c_er * 9 / _c_ip) if _c_ip > 0 else 0
                career_line = (
                    f'<span style="color:{CREAM}; font-weight:600;">{n_seasons} seasons</span>'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_era:.2f} ERA'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_k_pct:.1%} K'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_bb_pct:.1%} BB'
                    f' <span style="color:{SLATE};">|</span> '
                    f'{_c_ip:.0f} IP'
                )

            st.markdown(
                f'<div style="padding:6px 0; margin:4px 0;">'
                f'<span style="color:{SLATE}; font-size:0.7rem; font-weight:600; '
                f'letter-spacing:1px;">CAREER</span>'
                f'<div style="font-size:0.9rem; margin-top:2px;">{career_line}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── SEASON STATS ────────────────────────────────────────────────
    st.markdown('<div class="p-section">' + _section_head("Season Stats", "Observed rates and counting stats"), unsafe_allow_html=True)
    _player_seasons: list[int] = []
    if not trad_all_df.empty:
        _ps = trad_all_df[trad_all_df[id_col] == player_id]["season"].dropna().unique()
        _player_seasons = sorted([int(s) for s in _ps], reverse=True)

    if _player_seasons:
        _season_opts = [f"{s} Season" for s in _player_seasons]
        _season_pick = st.selectbox(
            "Season",
            _season_opts,
            index=0,
            key="profile_season_pick",
            label_visibility="collapsed",
        )
        _pick_season = int(_season_pick.split()[0])
    else:
        _pick_season = _recent_season

    if not trad_all_df.empty:
        recent_data = trad_all_df[
            (trad_all_df[id_col] == player_id)
            & (trad_all_df["season"] == _pick_season)
        ]
        if not recent_data.empty:
            rd = recent_data.iloc[0]
            _trad_pop = trad_all_df[trad_all_df["season"] == _pick_season]

            if player_type in ("Hitter", "Two-Way"):
                rate_configs_t = HITTER_TRAD_STATS
                counting_configs_t = HITTER_TRAD_COUNTING
            else:
                rate_configs_t = PITCHER_TRAD_STATS
                counting_configs_t = PITCHER_TRAD_COUNTING

            # Stat tooltips
            _rate_tips = {
                "avg": "Batting average", "obp": "On-base percentage",
                "slg": "Slugging percentage", "ops": "On-base plus slugging",
                "woba": "Weighted on-base average", "babip": "Batting average on balls in play",
                "iso": "Isolated power (SLG minus AVG)",
                "era": "Earned run average", "fip": "Fielding independent pitching",
                "whip": "Walks + hits per inning pitched",
            }

            # ── A. BAYESIAN RATE PROJECTIONS (current season only) ────
            if _pick_season == CURRENT_SEASON:
                _proj_chips: list[str] = []
                _proj_label = f"{CURRENT_SEASON} Updated Projection" if _in_season else f"{CURRENT_SEASON} Projection"
                for label, key, higher_better, desc in stat_configs:
                    proj_col = f"projected_{key}"
                    sd_col = f"projected_{key}_sd"
                    if proj_col in player_row.index and pd.notna(player_row.get(proj_col)):
                        proj_val = player_row[proj_col]
                        sd_val = player_row.get(sd_col)
                        display = fmt_stat(proj_val, key)
                        ci_tip = ""
                        if pd.notna(sd_val):
                            lo = proj_val - 1.96 * sd_val
                            hi = proj_val + 1.96 * sd_val
                            ci_tip = f" (95% CI: {fmt_stat(lo, key)} to {fmt_stat(hi, key)})"
                        _proj_chips.append(stat_chip(display, f"Proj {label}", f"{desc}{ci_tip}", SAGE))
                if _proj_chips:
                    st.markdown(
                        f'<div style="text-align:center; color:{GOLD}; font-size:0.75rem; '
                        f'font-weight:600; margin:8px 0 2px; letter-spacing:1px;">{_proj_label}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(stat_chip_row(_proj_chips, margin="0 0 8px"),
                                unsafe_allow_html=True)

            # ── B. ADVANCED SNAPSHOT (from parquet, with percentiles) ────
            _adv_type = "hitter" if player_type in ("Hitter", "Two-Way") else "pitcher"
            _adv_df = load_advanced_stats(_adv_type)
            _adv_configs = HITTER_ADVANCED_STATS if _adv_type == "hitter" else PITCHER_ADVANCED_STATS
            _adv_id_col = "batter_id" if _adv_type == "hitter" else "pitcher_id"
            _adv_pa_col = "pa" if _adv_type == "hitter" else "batters_faced"

            if not _adv_df.empty:
                _adv_player = _adv_df[_adv_df[_adv_id_col] == player_id]
                if not _adv_player.empty:
                    _adv_row = _adv_player.iloc[0]
                    _adv_cards: list[str] = []
                    for a_label, a_col, a_hib, a_fmt in _adv_configs:
                        a_val = _adv_row.get(a_col)
                        if pd.isna(a_val):
                            continue
                        # Format value
                        if a_fmt == "int":
                            a_disp = str(int(a_val))
                        elif a_fmt == "xwoba":
                            a_disp = fmt_stat(a_val, "xwoba")
                        elif a_fmt in ("avg_exit_velo", "avg_velo"):
                            a_disp = fmt_stat(a_val, a_fmt)
                        else:  # pct
                            a_disp = fmt_stat(a_val, a_col)
                        # Percentile against population
                        _a_pct = hybrid_percentile_rank(
                            _adv_df, float(a_val), a_col, a_hib,
                            min_pa=QUALIFIED_PA if a_fmt == "pct" else MIN_PA,
                            pa_col=_adv_pa_col,
                        )
                        _adv_cards.append(metric_card(a_label, a_disp, pctile=_a_pct))

                    if _adv_cards:
                        for i in range(0, len(_adv_cards), 4):
                            chunk = _adv_cards[i:i + 4]
                            cols = st.columns(len(chunk))
                            for col_w, card_html in zip(cols, chunk):
                                with col_w:
                                    st.markdown(card_html, unsafe_allow_html=True)

            # ── C. TRADITIONAL RATES (with percentiles, existing pattern) ─
            for i in range(0, len(rate_configs_t), 4):
                chunk = rate_configs_t[i:i + 4]
                r_cols = st.columns(len(chunk))
                for rc, (label, col_name, higher_better, fmt) in zip(r_cols, chunk):
                    val = rd.get(col_name) if hasattr(rd, 'get') else rd[col_name] if col_name in rd.index else None
                    _pct = None
                    if pd.notna(val) and col_name in _trad_pop.columns:
                        _pct = percentile_rank(_trad_pop[col_name], float(val), higher_better)
                    tip = _rate_tips.get(col_name, label)
                    with rc:
                        st.markdown(
                            metric_card(
                                f'<span title="{tip}" style="cursor:help;">{label}</span>',
                                fmt_trad(val, fmt) if pd.notna(val) else "N/A",
                                pctile=_pct,
                            ),
                            unsafe_allow_html=True,
                        )

            # ── D. COUNTING STATS with EOS DELTA ─────────────────────────
            _eos_configs = HITTER_EOS_DELTA_STATS if player_type in ("Hitter", "Two-Way") else PITCHER_EOS_DELTA_STATS
            _show_eos = _in_season and _pick_season == CURRENT_SEASON

            # Get counting sim row for EOS projections
            _sim_row = None
            if _show_eos and not counting_df.empty:
                _sim_match = counting_df[counting_df[id_col] == player_id]
                if not _sim_match.empty:
                    _sim_row = _sim_match.iloc[0]

            if _show_eos and _sim_row is not None:
                # EOS delta format: current / proj with pace indicator
                _games_played = rd.get("games", 0)
                _games_played = int(_games_played) if pd.notna(_games_played) else 0
                _season_frac = _games_played / 162.0 if _games_played > 0 else 0

                _eos_parts: list[str] = []
                for e_label, e_trad_col, e_sim_prefix, e_hib in _eos_configs:
                    e_current = rd.get(e_trad_col)
                    e_proj_col = f"{e_sim_prefix}_mean"
                    e_proj = _sim_row.get(e_proj_col) if e_proj_col in _sim_row.index else None
                    if pd.isna(e_current) or pd.isna(e_proj):
                        continue
                    e_cur_int = int(e_current)
                    e_proj_int = int(round(e_proj))
                    e_rem = e_proj_int - e_cur_int
                    e_rem_color = SAGE if (e_rem > 0) == e_hib else EMBER if (e_rem < 0) == e_hib else SLATE
                    e_rem_str = f"+{e_rem}" if e_rem >= 0 else str(e_rem)

                    # Pace indicator: compare current to expected at this point
                    pace_html = ""
                    if _season_frac >= 0.05 and e_proj_int > 0:
                        expected_now = e_proj_int * _season_frac
                        pace_ratio = e_cur_int / expected_now if expected_now > 0 else 1.0
                        if pace_ratio >= 1.10:
                            pace_html = f'<span style="color:{SAGE}; font-size:0.6rem;"> &#9650;</span>'
                        elif pace_ratio <= 0.90:
                            pace_html = f'<span style="color:{EMBER}; font-size:0.6rem;"> &#9660;</span>'

                    _eos_parts.append(
                        f'<div style="text-align:center; padding:4px 10px;">'
                        f'<div style="color:{CREAM}; font-size:1.1rem; font-weight:700;">'
                        f'{e_cur_int} <span style="color:{SLATE}; font-size:0.85rem;">/</span> '
                        f'<span style="color:{GOLD};">{e_proj_int}</span>{pace_html}</div>'
                        f'<div style="color:{e_rem_color}; font-size:0.7rem; font-weight:600;">'
                        f'{e_rem_str} rem</div>'
                        f'<div style="color:{SLATE}; font-size:0.65rem; text-transform:uppercase; '
                        f'letter-spacing:1px;">{e_label}</div></div>'
                    )
                if _eos_parts:
                    st.markdown(
                        f'<div style="display:flex; flex-wrap:wrap; gap:12px; margin-top:6px; '
                        f'justify-content:center;">'
                        + "".join(_eos_parts) + '</div>',
                        unsafe_allow_html=True,
                    )
            else:
                # Plain counting row (historical seasons or no sim data)
                count_vals = []
                for label, col_name in counting_configs_t[:8]:
                    val = rd.get(col_name) if hasattr(rd, 'get') else rd[col_name] if col_name in rd.index else None
                    if pd.notna(val):
                        v = f"{val:.1f}" if col_name == "ip" else str(int(val))
                        count_vals.append(
                            f'<span style="color:{CREAM}; font-weight:600;">{v}</span> '
                            f'<span style="color:{SLATE}; font-size:0.7rem;">{label}</span>'
                        )
                if count_vals:
                    st.markdown(
                        f'<div style="display:flex; flex-wrap:wrap; gap:12px; margin-top:6px; '
                        f'justify-content:center;">'
                        + "".join(f"<div>{v}</div>" for v in count_vals)
                        + '</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption(f"No stats found for {_pick_season}.")

    st.markdown('</div>', unsafe_allow_html=True)  # close season stats p-section

    # ── SCOUTING REPORT ──────────────────────────────────────────
    st.markdown('<div class="p-section">' + _section_head("Scouting Report", "Model-generated analysis"), unsafe_allow_html=True)

    # Gather data for scouting card
    _is_hitter = player_type in ("Hitter", "Two-Way")
    _sc_arch = load_hitter_archetypes() if _is_hitter else load_pitcher_archetypes()
    _sc_grades = load_hitter_grade_ci() if _is_hitter else load_pitcher_grade_ci()
    _sc_str = load_hitter_strength(career=True) if _is_hitter else None
    _sc_vuln = load_hitter_vulnerability(career=True) if _is_hitter else None
    _sc_arsenal = load_pitcher_arsenal() if not _is_hitter else None
    _sc_adv = load_advanced_stats("hitter" if _is_hitter else "pitcher")
    _sc_breakout = load_hitter_breakout_candidates() if _is_hitter else None

    # Get counting sim row for outlook
    _sc_counting_row = None
    if not counting_df.empty:
        _sc_cm = counting_df[counting_df[id_col] == player_id]
        if not _sc_cm.empty:
            _sc_counting_row = _sc_cm.iloc[0]

    # Get current-season trad row for outlook
    _sc_trad_row = None
    if not trad_all_df.empty:
        _sc_tr = trad_all_df[
            (trad_all_df[id_col] == player_id)
            & (trad_all_df["season"] == CURRENT_SEASON)
        ]
        if not _sc_tr.empty:
            _sc_trad_row = _sc_tr.iloc[0]

    _sc_card = generate_scouting_card(
        player_id, player_type, player_row, _sc_trad_row, _sc_counting_row,
        stat_configs, df,
        archetype_df=_sc_arch,
        grade_df=_sc_grades,
        str_df=_sc_str,
        vuln_df=_sc_vuln,
        arsenal_df=_sc_arsenal,
        adv_df=_sc_adv,
        breakout_df=_sc_breakout,
    )
    render_scouting_card(_sc_card)

    st.markdown('</div>', unsafe_allow_html=True)  # close scouting p-section

    # ── APPROACH & EFFICIENCY ────────────────────────────────────
    st.markdown('<div class="p-section">' + _section_head("Approach & Efficiency", "Plate discipline and efficiency metrics"), unsafe_allow_html=True)
    render_approach_efficiency(
        player_type, player_id, id_col,
        selected_season=_recent_season, is_career=False,
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close approach p-section

    # ── DEEP DIVE ────────────────────────────────────────────────
    st.markdown('<div class="p-section">' + _section_head("Deep Dive", "Season trends, pitch profiles, and arsenal"), unsafe_allow_html=True)
    deep_season = st.selectbox(
        "Season", [str(s) for s in AVAILABLE_SEASONS] + ["Career"],
        index=len(AVAILABLE_SEASONS) - 1,  # default to most recent
        key="profile_deep_dive_season",
        label_visibility="collapsed",
    )
    _deep_is_career = deep_season == "Career"
    _deep_season = None if _deep_is_career else int(deep_season)

    # Season trends (multi-year chart)
    render_season_trends(player_type, player_id, selected_name)

    # Pitch profiles, zones, vulnerabilities
    render_pitch_profiles(
        player_type, player_id, selected_name,
        selected_season=_deep_season, is_career=_deep_is_career,
    )

    # Arsenal evolution (pitcher only)
    if player_type == "Pitcher" and _deep_season and not _deep_is_career:
        render_arsenal_evolution(player_id, selected_name, _deep_season)

    st.markdown('</div>', unsafe_allow_html=True)  # close deep dive p-section

    # ── K% POSTERIOR (pitcher only) ──────────────────────────────
    k_samples = load_k_samples()
    sample_key = str(player_id)
    if player_type == "Pitcher" and sample_key in k_samples:
        st.markdown(
            '<div class="tdd-section-hdr">K% Posterior Distribution</div>',
            unsafe_allow_html=True,
        )
        samples = k_samples[sample_key]
        obs_k = player_row.get("observed_k_rate")
        fig = create_posterior_fig(
            samples,
            observed=obs_k if pd.notna(obs_k) else None,
            stat_label=f"Projected K% ({CURRENT_SEASON})",
        )
        _, chart_col, _ = st.columns([1, 3, 1])
        with chart_col:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    # ── STAT BREAKDOWN ──────────────────────────────────────────────
    st.markdown('<div class="p-section">' + _section_head("Stat Breakdown", "Observed vs projected with credible intervals"), unsafe_allow_html=True)
    detail_rows = []
    for label, key, higher_better, desc in stat_configs:
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        sd_col = f"projected_{key}_sd"
        lo_col = f"projected_{key}_2_5"
        hi_col = f"projected_{key}_97_5"
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            detail_rows.append({
                "Stat": label,
                f"{PRIOR_SEASON} Observed": fmt_stat(player_row[obs_col], key),
                f"{CURRENT_SEASON} Projected": fmt_stat(player_row[proj_col], key),
                "Delta": f"{player_row[f'delta_{key}'] * 100:+.1f}pp",
                "95% Cred. Int.": (
                    f"[{fmt_stat(player_row[lo_col], key)}, "
                    f"{fmt_stat(player_row[hi_col], key)}]"
                    if lo_col in player_row.index and pd.notna(player_row.get(lo_col))
                    else ""
                ),
                "Description": desc,
            })
    if detail_rows:
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close stat breakdown p-section
    st.markdown('</div>', unsafe_allow_html=True)  # close .tdd-profile wrapper
    return  # End of profile page
