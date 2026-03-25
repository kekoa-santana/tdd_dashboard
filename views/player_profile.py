"""Player Profile page |Deep dive into a single player's projections."""
from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st


def _fig_to_b64_img(fig: Figure) -> str:
    """Convert a matplotlib Figure to a base64 <img> tag for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" />'

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK, DARK_BORDER, DARK_CARD,
    POSITIVE, NEGATIVE,
    CURRENT_SEASON, PRIOR_SEASON, TRAIN_START, PROJECTION_LABEL,
    AVAILABLE_SEASONS, UNRELIABLE_BB_SEASONS,
    PITCHER_STATS, HITTER_STATS,
    PITCHER_OBSERVED_STATS, HITTER_OBSERVED_STATS,
    PITCHER_COUNTING_DISPLAY, HITTER_COUNTING_DISPLAY,
    HITTER_TRAD_STATS, HITTER_TRAD_COUNTING,
    PITCHER_TRAD_STATS, PITCHER_TRAD_COUNTING,
    PITCH_DISPLAY,
)
from services.data_loader import (
    load_projections, load_counting, load_player_teams, load_rankings,
    load_k_samples, load_traditional_stats_all,
    load_pitcher_arsenal, load_pitcher_arsenal_all,
    load_hitter_vulnerability, load_hitter_vulnerability_all,
    load_hitter_strength,
    load_pitcher_location_grid, load_pitcher_location_grid_all,
    load_hitter_zone_grid, load_hitter_zone_grid_all,
    load_pitcher_offerings, load_hitter_vuln_arch, load_hitter_vuln_arch_career,
    load_cluster_metadata, load_baselines_arch,
    load_hitter_aggressiveness, load_hitter_aggressiveness_all,
    load_pitcher_efficiency, load_pitcher_efficiency_all,
    load_full_stats, load_preseason_injuries,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_archetype_matchup_matrix,
    load_hitter_breakout_candidates,
    season_selector,
)
from utils.helpers import get_team_lookup, get_injury_lookup
from utils.formatters import fmt_stat, fmt_pct, fmt_trad, delta_html
from components.metric_cards import (
    metric_card, percentile_rank, pctile_color,
    pctile_bar_html, observed_pctile_bar_html,
)
from components.charts import (
    create_posterior_fig, create_arsenal_donut,
    create_pitch_density_plotly, create_hitter_zone_plotly,
)
from components.tables import (
    combine_platoon_vuln,
    build_hitter_profile_table, build_pitcher_profile_table,
)
from components.scouting import generate_scouting_bullets
from components.headshot import render_headshot
from components.diamond_rating import diamond_rating_html, diamond_rating_html_composite


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def career_aggregate_trad(trad_player: pd.DataFrame, player_type: str) -> pd.Series:
    """Aggregate multi-season traditional stats into career totals."""
    numeric_cols = trad_player.select_dtypes(include="number").columns.tolist()
    # Separate counting vs rate columns
    if player_type == "Hitter":
        counting = ["pa", "ab", "hits", "doubles", "triples", "hr", "rbi",
                     "bb", "k", "hbp", "sb", "cs", "sac_fly"]
        denom_col = "pa"
    else:
        counting = ["games", "starts", "wins", "losses", "saves", "holds",
                     "ip", "k", "bb", "hr", "hits_allowed", "er",
                     "batters_faced", "go", "ao"]
        denom_col = "batters_faced" if "batters_faced" in trad_player.columns else "ip"

    counting = [c for c in counting if c in numeric_cols]
    result = trad_player[counting].sum()

    # Recompute rate stats from totals
    if player_type == "Hitter":
        ab = result.get("ab", 0)
        pa = result.get("pa", 0)
        h = result.get("hits", 0)
        bb = result.get("bb", 0)
        hbp = result.get("hbp", 0)
        sf = result.get("sac_fly", 0)
        hr = result.get("hr", 0)
        doubles = result.get("doubles", 0)
        triples = result.get("triples", 0)
        result["avg"] = h / ab if ab > 0 else 0
        result["obp"] = (h + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0
        tb = h + doubles + 2 * triples + 3 * hr
        result["slg"] = tb / ab if ab > 0 else 0
        result["ops"] = result["obp"] + result["slg"]
        result["iso"] = result["slg"] - result["avg"]
    else:
        ip = result.get("ip", 0)
        bf = result.get("batters_faced", 0)
        er = result.get("er", 0)
        k = result.get("k", 0)
        bb = result.get("bb", 0)
        hr = result.get("hr", 0)
        ha = result.get("hits_allowed", 0)
        go = result.get("go", 0)
        ao = result.get("ao", 0)
        result["era"] = (er / ip * 9) if ip > 0 else 0
        result["whip"] = (ha + bb) / ip if ip > 0 else 0
        result["k_per_9"] = (k / ip * 9) if ip > 0 else 0
        result["bb_per_9"] = (bb / ip * 9) if ip > 0 else 0
        result["hr_per_9"] = (hr / ip * 9) if ip > 0 else 0
        result["k_bb_ratio"] = (k / bb) if bb > 0 else 0
        result["go_ao_ratio"] = (go / ao) if ao > 0 else 0
        # FIP
        c_fip = 3.20
        result["fip"] = ((13 * hr + 3 * bb - 2 * k) / ip + c_fip) if ip > 0 else 0

    return result


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
            _a_cols = st.columns(len(_agg_items))
            for _ac, (_albl, _acol, _is_pct, _hib) in zip(_a_cols, _agg_items):
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
                    st.dataframe(pd.DataFrame(arch_rows), width='stretch', hide_index=True)

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
                        st.dataframe(styled, width='stretch', hide_index=True)

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


def render_observed_percentiles(
    player_type: str,
    player_id: int,
    selected_season: int | None = None,
    is_career: bool = False,
) -> None:
    """Render observed percentile bars for any season, ranked within that season's population."""
    obs_stat_configs = PITCHER_OBSERVED_STATS if player_type == "Pitcher" else HITTER_OBSERVED_STATS
    id_col = "pitcher_id" if player_type == "Pitcher" else "batter_id"

    full_df = load_full_stats(player_type.lower())
    if full_df.empty:
        return

    if is_career:
        # Average across seasons for the player; rank against all players' career averages
        career_avg = full_df.groupby(id_col).mean(numeric_only=True).reset_index()
        player_vals = career_avg[career_avg[id_col] == player_id]
        pop_df = career_avg
        season_label = "Career"
    else:
        _season = selected_season if selected_season else PRIOR_SEASON
        season_df = full_df[full_df["season"] == _season]
        player_vals = season_df[season_df[id_col] == player_id]
        pop_df = season_df
        season_label = str(_season)

    if player_vals.empty:
        return

    player_data = player_vals.iloc[0]

    # Filter out unreliable batted ball stats for pre-2022
    unreliable_keys = {"hard_hit_pct", "avg_exit_velo", "barrel_pct", "fb_pct"}
    is_unreliable_bb = (selected_season is not None and selected_season in UNRELIABLE_BB_SEASONS)

    obs_bars_html = ""
    for label, key, higher_better, _ in obs_stat_configs:
        if is_unreliable_bb and key in unreliable_keys:
            continue
        if key not in player_data.index or pd.isna(player_data.get(key)):
            continue
        val = player_data[key]
        if key not in pop_df.columns:
            continue
        pctile = percentile_rank(pop_df[key], val, higher_better)
        obs_bars_html += observed_pctile_bar_html(label, pctile, val, key)

    if obs_bars_html:
        st.markdown(
            f'<div class="tdd-section-hdr">{season_label} Observed Percentiles</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight-card">{obs_bars_html}</div>',
            unsafe_allow_html=True,
        )
        note = (
            f"Skill profile based on {season_label} observed data, "
            f"ranked among {len(pop_df)} {player_type.lower()}s in {season_label}. "
        )
        if is_unreliable_bb:
            note += "Batted ball metrics hidden (insufficient Statcast coverage pre-2022). "
        note += (
            "100th = best, 1st = worst. "
            f"<span style='color:var(--tdd-sage);'>Green</span> = elite (80+), "
            f"<span style='color:var(--tdd-gold);'>gold</span> = above-avg (60-79), "
            f"<span style='color:var(--tdd-slate);'>gray</span> = mid-tier (40-59), "
            f"<span style='color:var(--tdd-ember);'>orange</span> = below-avg (&lt;40)."
        )
        st.caption(note, unsafe_allow_html=True)


def render_season_trends(
    player_type: str,
    player_id: int,
    selected_name: str,
    selected_season: int | None = None,
) -> None:
    """Render year-over-year trend charts for key stats."""
    full_df = load_full_stats(player_type.lower())
    if full_df.empty:
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
        return  # Need at least 2 seasons for a trend

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
        return

    curr = arsenal_all[
        (arsenal_all["pitcher_id"] == player_id) & (arsenal_all["season"] == selected_season)
    ]
    prev = arsenal_all[
        (arsenal_all["pitcher_id"] == player_id) & (arsenal_all["season"] == selected_season - 1)
    ]

    if curr.empty or prev.empty:
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

        u_curr, u_prev, u_delta = _delta_fmt(usage, usage_prev)
        v_curr, v_prev, v_delta = _delta_fmt(velo, velo_prev, is_pct=False)
        w_curr, w_prev, w_delta = _delta_fmt(whiff, whiff_prev)
        c_curr, c_prev, c_delta = _delta_fmt(csw, csw_prev)

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
        st.dataframe(delta_df, width='stretch', hide_index=True)
        st.caption(
            f"Year-over-year arsenal evolution. "
            f"NEW = pitch added in {selected_season}. "
            f"DROPPED = pitch no longer thrown."
        )


# ---------------------------------------------------------------------------
# Page: Player Profile
# ---------------------------------------------------------------------------
def page_player_profile() -> None:
    """Deep dive into a single player's projections."""
    st.markdown('<div class="tdd-section-hdr">Player Profile</div>',
                unsafe_allow_html=True)

    qp_player_type = st.query_params.get("player_type", "")
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
        st.warning("No projection data found.")
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
        team_opts = ["All Teams"] + all_teams
        default_team_idx = 0
        if deep_link_player_id is not None:
            linked_team = team_lookup.get(deep_link_player_id, "")
            if linked_team in team_opts:
                default_team_idx = team_opts.index(linked_team)
        profile_team_filter = st.selectbox(
            "Team", team_opts, index=default_team_idx, key="profile_team",
            label_visibility="collapsed",
        )

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
    show_trad = not is_projection  # backwards-compat for header logic

    # --- Header card ---
    # Team abbreviation
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
            f'<span class="tdd-team-abbr" data-team="{player_team}">{player_team}</span>'
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

    # Use pre-computed diamond_rating from rankings as single source of truth.
    # Shows both Overall Rating (gold) and Projected Value (sage) side by side.
    _ranks = load_rankings("hitters") if player_type != "Pitcher" else load_rankings("pitchers")
    _id_col = "batter_id" if player_type != "Pitcher" else "pitcher_id"
    _rank_row = _ranks[_ranks[_id_col] == player_id] if not _ranks.empty else pd.DataFrame()
    if not _rank_row.empty and "diamond_rating" in _rank_row.columns and pd.notna(_rank_row["diamond_rating"].iloc[0]):
        _precomputed_dr = float(_rank_row["diamond_rating"].iloc[0])
        diamond_html = diamond_rating_html(0, size="lg", precomputed=_precomputed_dr)
    else:
        composite = player_row["composite_score"]
        diamond_html = diamond_rating_html_composite(composite, size="lg")

    # Projected Value rating (SAGE colored, side by side with overall)
    _proj_html = ""
    if not _rank_row.empty and "current_value_score" in _rank_row.columns:
        from lib.diamond_rating import score_to_diamonds
        _proj_score = float(_rank_row["current_value_score"].iloc[0])
        _proj_rating = score_to_diamonds(_proj_score)
        _proj_diamonds = ""
        for _di in range(10):
            if _di < int(_proj_rating) or (_di == int(_proj_rating) and _proj_rating - int(_proj_rating) >= 0.5):
                _proj_diamonds += f'<span style="color:{SAGE}">&#9670;</span>'
            else:
                _proj_diamonds += f'<span style="color:{SLATE}; opacity:0.35">&#9671;</span>'
        _proj_html = (
            f'<div style="margin-top:4px; font-size:0.85rem;">'
            f'<span style="color:{SLATE}; font-size:0.7rem;">PROJECTED </span>'
            f'<span style="letter-spacing:2px;">{_proj_diamonds}</span>'
            f' <span style="color:{SAGE}; font-size:0.8rem;">{_proj_rating:.1f}</span>'
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

    header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{selected_name}</div>'
        f'<div class="brand-subtitle">{" | ".join(header_parts)} | '
        f'{PROJECTION_LABEL if is_projection else "Career" if is_career else f"{selected_season} Season"}</div>'
        f'{injury_html}'
        f'</div>'
        f'<div style="text-align:right;">'
        f'{diamond_html}'
        f'{_proj_html}'
        f'</div>'
        f'</div>'
    )
    hdr_left, hdr_right = st.columns([1, 11])
    with hdr_left:
        render_headshot(player_id, size=80)
    with hdr_right:
        st.markdown(header_html, unsafe_allow_html=True)

    # --- Two-Way Player: show both batting + pitching scouting grades ---
    if is_two_way_player and two_way_pitcher_row is not None:
        # Load rankings for scouting grades
        h_ranks = load_rankings("hitters")
        p_ranks = load_rankings("pitchers")
        h_row = h_ranks[h_ranks["batter_id"] == player_id]
        p_row = p_ranks[p_ranks["pitcher_id"] == player_id]

        tw_parts = []
        if not h_row.empty:
            hr = h_row.iloc[0]
            h_dr = hr.get("diamond_rating", "")
            h_dr_s = f"{h_dr:.1f}" if pd.notna(h_dr) else ""
            tw_parts.append(
                f'<span style="color:var(--tdd-gold); font-weight:700;">Batting: {h_dr_s}</span>'
                f' <span style="color:var(--tdd-slate); font-size:0.85em;">'
                f'H:{int(hr.get("grade_hit", 0))} P:{int(hr.get("grade_power", 0))} '
                f'Sp:{int(hr.get("grade_speed", 0))} D:{int(hr.get("grade_discipline", 0))}</span>'
            )
        if not p_row.empty:
            pr = p_row.iloc[0]
            p_dr = pr.get("diamond_rating", "")
            p_dr_s = f"{p_dr:.1f}" if pd.notna(p_dr) else ""
            tw_parts.append(
                f'<span style="color:var(--tdd-gold); font-weight:700;">Pitching: {p_dr_s}</span>'
                f' <span style="color:var(--tdd-slate); font-size:0.85em;">'
                f'St:{int(pr.get("grade_stuff", 0))} Cm:{int(pr.get("grade_command", 0))} '
                f'Du:{int(pr.get("grade_durability", 0))}</span>'
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
    # RESTRUCTURED LAYOUT: Temporal zones
    #   1. Career Context (compact, always visible)
    #   2. Side-by-side: Recent Season | 2026 Projections
    #   3. Scouting / Tools
    #   4. Deep Dive (season selector controls pitch profiles, trends)
    # ===================================================================

    # Load data needed for multiple sections
    trad_all_df = load_traditional_stats_all(player_type.lower())
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

    # ── 2. PROJECTIONS (compact, above season) ──────────────────────
    _proj_label = f"{CURRENT_SEASON} Updated Projection" if _in_season else f"{CURRENT_SEASON} Projection"
    st.markdown(
        f'<div style="text-align:center; color:{GOLD}; font-size:0.9rem; '
        f'font-weight:600; margin:12px 0 4px;">{_proj_label}</div>',
        unsafe_allow_html=True,
    )

    # Build all projection chips in one row
    _proj_chips: list[str] = []

    # Bayesian rate projections (K%, BB%)
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
            tip = f"{desc}{ci_tip}"
            _proj_chips.append(
                f'<div style="text-align:center; padding:4px 10px;" title="{tip}">'
                f'<div style="color:{SAGE}; font-size:1.4rem; font-weight:700; cursor:help;">{display}</div>'
                f'<div style="color:{SLATE}; font-size:0.65rem; text-transform:uppercase; '
                f'letter-spacing:1px;">{label}</div></div>'
            )

    # Counting stat projections
    if not counting_df.empty:
        c_row = counting_df[counting_df[id_col] == player_id]
        if not c_row.empty:
            c_data = c_row.iloc[0]
            counting_display = PITCHER_COUNTING_DISPLAY if player_type == "Pitcher" else HITTER_COUNTING_DISPLAY

            # Filter out SV/HLD for starting pitchers
            if player_type == "Pitcher" and role == "SP":
                counting_display = [
                    item for item in counting_display
                    if not any(x in item[0].lower() for x in ("sv", "hld", "save", "hold"))
                ]

            _stat_tips = {
                "K": "Projected strikeouts", "BB": "Projected walks",
                "HR": "Projected home runs", "R": "Projected runs scored",
                "RBI": "Projected runs batted in", "IP": "Projected innings pitched",
                "wRC+": "Weighted runs created plus (100 = league average)",
                "FIP-ERA": "Fielding independent pitching minus ERA",
            }

            for item in counting_display:
                c_label = item[0]
                c_prefix, c_actual, c_hb = item[1], item[2], item[3]
                confidence = item[4] if len(item) == 5 else "med"
                mean_col = f"{c_prefix}_mean"
                p10_col = f"{c_prefix}_p10"
                p90_col = f"{c_prefix}_p90"
                if mean_col not in c_data.index or pd.isna(c_data.get(mean_col)):
                    continue
                val = int(round(c_data[mean_col]))
                lo = int(round(c_data.get(p10_col, val)))
                hi = int(round(c_data.get(p90_col, val)))

                val_color = {
                    "high": SAGE, "med": GOLD, "range": CREAM, "low": SLATE,
                }.get(confidence, GOLD)

                short_label = c_label.replace("Proj. ", "")
                tip_base = _stat_tips.get(short_label, short_label)
                tip = f"{tip_base}. 80% range: {lo} to {hi}"

                if confidence == "range":
                    disp = f"{lo} to {hi}"
                    tip = f"{tip_base}. Mean: {val}"
                else:
                    disp = str(val)

                _proj_chips.append(
                    f'<div style="text-align:center; padding:4px 10px;" title="{tip}">'
                    f'<div style="color:{val_color}; font-size:1.4rem; font-weight:700; cursor:help;">{disp}</div>'
                    f'<div style="color:{SLATE}; font-size:0.65rem; text-transform:uppercase; '
                    f'letter-spacing:1px;">{short_label}</div></div>'
                )

    if _proj_chips:
        st.markdown(
            f'<div style="display:flex; flex-wrap:wrap; gap:8px; '
            f'justify-content:center; margin-bottom:12px;">'
            + "".join(_proj_chips) + '</div>',
            unsafe_allow_html=True,
        )

    # ── 3. SEASON (full width, with dropdown) ────────────────────────
    _player_seasons: list[int] = []
    if not trad_all_df.empty:
        _ps = trad_all_df[trad_all_df[id_col] == player_id]["season"].dropna().unique()
        _player_seasons = sorted([int(s) for s in _ps], reverse=True)

    if _player_seasons:
        _season_opts = [f"{s} Season" for s in _player_seasons]
        with st.container(key="gold_season_pick"):
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

            # Build all stats as chips (rate + counting + advanced)
            all_chips: list[str] = []

            # Advanced stats first (most important: wRC+, xwOBA, K%, BB%)
            _exp_col = "pa" if "pa" in rd.index else "bf"
            _exp_v = rd.get(_exp_col, 0)
            _k_v = rd.get("k", 0)
            _bb_v = rd.get("bb", 0)

            # Load advanced Statcast for this season
            _adv_data: dict = {}
            try:
                from src.data.db import read_sql as _adv_sql
                _adv_table = "fact_batting_advanced" if player_type != "Pitcher" else "fact_pitching_advanced"
                _adv_id = "batter_id" if player_type != "Pitcher" else "pitcher_id"
                _adv_pa = "pa" if player_type != "Pitcher" else "batters_faced"
                _ar = _adv_sql(f"""
                    SELECT xwoba, barrel_pct, hard_hit_pct, sweet_spot_pct, wrc_plus
                    FROM production.{_adv_table}
                    WHERE {_adv_id} = {player_id}
                      AND season = {_pick_season} AND {_adv_pa} >= 50
                    LIMIT 1
                """, {})
                if not _ar.empty:
                    _adv_data = _ar.iloc[0].to_dict()
            except Exception:
                pass

            # Priority-ordered stat chips for hitters
            _ordered_stats: list[tuple[str, str, str]] = []  # (value, label, tooltip)
            if player_type in ("Hitter", "Two-Way"):
                if _adv_data.get("wrc_plus") and pd.notna(_adv_data["wrc_plus"]):
                    _ordered_stats.append((str(int(_adv_data["wrc_plus"])), "wRC+", "Weighted runs created plus (100 = league avg)"))
                if _exp_v > 0:
                    _ordered_stats.append((f"{_k_v / _exp_v:.1%}", "K%", "Strikeout rate"))
                    _ordered_stats.append((f"{_bb_v / _exp_v:.1%}", "BB%", "Walk rate"))
                if _adv_data.get("xwoba") and pd.notna(_adv_data["xwoba"]):
                    _ordered_stats.append((f".{int(_adv_data['xwoba'] * 1000):03d}", "xwOBA", "Expected weighted on-base average (Statcast)"))
                if _adv_data.get("barrel_pct") and pd.notna(_adv_data["barrel_pct"]):
                    _ordered_stats.append((f"{_adv_data['barrel_pct']:.1%}", "Brl%", "Barrel rate (95+ mph EV, optimal launch angle)"))
                if _adv_data.get("hard_hit_pct") and pd.notna(_adv_data["hard_hit_pct"]):
                    _ordered_stats.append((f"{_adv_data['hard_hit_pct']:.1%}", "HH%", "Hard hit rate (95+ mph exit velocity)"))
            else:
                # Pitcher priority
                if _exp_v > 0:
                    _ordered_stats.append((f"{_k_v / _exp_v:.1%}", "K%", "Strikeout rate"))
                    _ordered_stats.append((f"{_bb_v / _exp_v:.1%}", "BB%", "Walk rate"))
                if _adv_data.get("xwoba") and pd.notna(_adv_data["xwoba"]):
                    _ordered_stats.append((f".{int(_adv_data['xwoba'] * 1000):03d}", "xwOBA", "Expected wOBA against"))

            for _sv, _sl, _st in _ordered_stats:
                all_chips.append(
                    f'<div style="text-align:center; padding:4px 10px;" title="{_st}">'
                    f'<div style="color:{CREAM}; font-size:1.4rem; font-weight:700; cursor:help;">{_sv}</div>'
                    f'<div style="color:{SLATE}; font-size:0.65rem; text-transform:uppercase; '
                    f'letter-spacing:1px;">{_sl}</div></div>'
                )

            if all_chips:
                st.markdown(
                    f'<div style="display:flex; flex-wrap:wrap; gap:8px; '
                    f'justify-content:center; margin:4px 0 8px;">'
                    + "".join(all_chips) + '</div>',
                    unsafe_allow_html=True,
                )

            # Traditional rate stats with percentiles
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

            # Counting stats (compact centered row)
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

    # ── 3. SCOUTING / TOOLS ──────────────────────────────────────────
    # Scouting report bullets
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    bullets = generate_scouting_bullets(stat_configs, player_row, df, player_type)

    if player_type in ("Hitter", "Two-Way"):
        _cnt = load_counting("hitter")
        if not _cnt.empty:
            _c_r = _cnt[_cnt["batter_id"] == player_id]
            if not _c_r.empty and "hr_park_factor" in _c_r.columns:
                _pf = _c_r.iloc[0].get("hr_park_factor")
                if pd.notna(_pf) and _pf > 1.03:
                    bullets.append((POSITIVE, f"Home park boosts HR rate (park factor {_pf:.3f})."))
                elif pd.notna(_pf) and _pf < 0.97:
                    bullets.append((NEGATIVE, f"Home park suppresses HR rate (park factor {_pf:.3f})."))

    if player_type in ("Hitter", "Two-Way") and _breakout_hole:
        _bo_df_scout = load_hitter_breakout_candidates()
        if not _bo_df_scout.empty:
            _bo_r = _bo_df_scout[_bo_df_scout["batter_id"] == player_id]
            if not _bo_r.empty:
                _bo_score = _bo_r.iloc[0].get("breakout_score", 0)
                _bo_tier = _bo_r.iloc[0].get("breakout_tier", "")
                _bo_type = _bo_r.iloc[0].get("breakout_type", "")
                _tier_color = GOLD if _bo_tier == "High" else SAGE
                bullets.append((
                    _tier_color,
                    f"<b>{_bo_tier} breakout candidate</b> ({_bo_type}, score {_bo_score:.2f}). "
                    f"Key hole: <b>{_breakout_hole}</b>.",
                ))

    _arch_df_scout = load_hitter_archetypes() if player_type in ("Hitter", "Two-Way") else load_pitcher_archetypes()
    _id_col_arch = "batter_id" if player_type in ("Hitter", "Two-Way") else "pitcher_id"
    if not _arch_df_scout.empty:
        _arch_match = _arch_df_scout[_arch_df_scout[_id_col_arch] == player_id]
        if not _arch_match.empty:
            _a_name = _arch_match.iloc[0]["archetype_name"]
            _a_desc = _arch_match.iloc[0]["archetype_desc"]
            bullets.append((GOLD, f"Classified as <b>{_a_name}</b> |{_a_desc.lower()}."))

            _mm = load_archetype_matchup_matrix()
            if not _mm.empty:
                if player_type in ("Hitter", "Two-Way"):
                    _mm_sub = _mm[_mm["hitter_archetype_name"] == _a_name]
                    if not _mm_sub.empty:
                        _worst = _mm_sub.loc[_mm_sub["k_pct"].idxmax()]
                        _best = _mm_sub.loc[_mm_sub["k_pct"].idxmin()]
                        bullets.append((
                            SLATE,
                            f"Highest K% vs <b>{_worst['pitcher_archetype_name']}</b> ({_worst['k_pct']:.1%}), "
                            f"lowest vs <b>{_best['pitcher_archetype_name']}</b> ({_best['k_pct']:.1%})."
                        ))
                else:
                    _mm_sub = _mm[_mm["pitcher_archetype_name"] == _a_name]
                    if not _mm_sub.empty:
                        _best_k = _mm_sub.loc[_mm_sub["k_pct"].idxmax()]
                        _worst_k = _mm_sub.loc[_mm_sub["k_pct"].idxmin()]
                        bullets.append((
                            SLATE,
                            f"Highest K% vs <b>{_best_k['hitter_archetype_name']}</b> ({_best_k['k_pct']:.1%}), "
                            f"lowest vs <b>{_worst_k['hitter_archetype_name']}</b> ({_worst_k['k_pct']:.1%})."
                        ))

    if bullets:
        bullet_html = "".join(
            f'<div class="insight-bullet">'
            f'<span class="dot" style="background:{color};"></span>'
            f'{text}</div>'
            for color, text in bullets
        )
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Scouting Report</div>
            {bullet_html}
        </div>
        """, unsafe_allow_html=True)

    # Approach & efficiency
    render_approach_efficiency(
        player_type, player_id, id_col,
        selected_season=_recent_season, is_career=False,
    )

    # (Observed + projected percentile bars are now inline in the columns above)

    # ── 4. DEEP DIVE (season selector controls this) ─────────────────
    st.markdown(
        f'<div class="tdd-section-hdr" style="margin-top:1.5rem;">Deep Dive</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="gold_deep_season"):
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

    # K% posterior distribution (pitcher only)
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

    # Stat breakdown table
    st.markdown(
        '<div class="tdd-section-hdr">Stat Breakdown</div>',
        unsafe_allow_html=True,
    )
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
        st.dataframe(pd.DataFrame(detail_rows), width='stretch', hide_index=True)

    return  # End of profile page
