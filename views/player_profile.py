"""Player Profile page — Deep dive into a single player's projections."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK, DARK_BORDER,
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
    load_projections, load_counting, load_player_teams,
    load_k_samples, load_traditional_stats_all,
    load_pitcher_arsenal, load_pitcher_arsenal_all,
    load_hitter_vulnerability, load_hitter_vulnerability_all,
    load_hitter_strength,
    load_pitcher_location_grid, load_pitcher_location_grid_all,
    load_hitter_zone_grid, load_hitter_zone_grid_all,
    load_hitter_aggressiveness, load_hitter_aggressiveness_all,
    load_pitcher_efficiency, load_pitcher_efficiency_all,
    load_full_stats, load_preseason_injuries,
    season_selector,
)
from utils.helpers import get_team_lookup, get_injury_lookup
from utils.formatters import fmt_stat, fmt_pct, fmt_trad, delta_html
from components.metric_cards import (
    metric_card, percentile_rank, pctile_color,
    pctile_bar_html, observed_pctile_bar_html,
)
from components.charts import create_posterior_fig
from components.tables import (
    combine_platoon_vuln,
    build_hitter_profile_table, build_pitcher_profile_table,
)
from components.scouting import generate_scouting_bullets
from lib.theme import add_watermark


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
        f'<div class="section-header" style="font-size:1rem; margin-top:1rem;">'
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
                    _disp = "--"
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
                    _disp = "--"
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
        _platoon_suffix = f" — {platoon_choice}" if _platoon_stand else ""

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
            st.markdown(f'<div class="section-header">Pitch Arsenal ({season_label}{_platoon_suffix})</div>',
                        unsafe_allow_html=True)
            table_html = build_pitcher_profile_table(p_arsenal)
            if table_html:
                st.markdown(f'<div class="insight-card">{table_html}</div>', unsafe_allow_html=True)

        # Location heatmap
        if is_career:
            ploc_all = load_pitcher_location_grid_all()
            if not ploc_all.empty:
                p_loc = ploc_all[ploc_all["pitcher_id"] == player_id]
                # Sum across seasons
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
            st.markdown(f'<div class="section-header">Pitch Location ({season_label})</div>',
                        unsafe_allow_html=True)
            from lib.zone_charts import plot_pitcher_location_heatmap
            loc_stand = st.radio(
                "Batter handedness", ["All", "vs LHH", "vs RHH"],
                horizontal=True, key="hist_pitcher_loc_stand",
            )
            stand_filter = {"vs LHH": "L", "vs RHH": "R"}.get(loc_stand)

            # Pitch-type filter
            _avail_pts = (
                p_loc.groupby("pitch_type")["pitches"].sum()
                .sort_values(ascending=False)
            )
            _pt_options = [pt for pt in _avail_pts.index if pt in PITCH_DISPLAY]
            _pt_labels = {pt: PITCH_DISPLAY.get(pt, pt) for pt in _pt_options}
            _selected_pts = st.multiselect(
                "Pitch types",
                options=_pt_options,
                format_func=lambda pt: _pt_labels[pt],
                default=[],
                key="hist_pitch_type_filter",
                help="Leave empty to auto-select top 4 pitch types by volume",
            )

            fig_loc = plot_pitcher_location_heatmap(
                p_loc, pitch_types=_selected_pts or None,
                pitcher_name=selected_name, batter_stand=stand_filter,
            )
            st.pyplot(fig_loc, use_container_width=True)
            plt.close(fig_loc)

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

                st.markdown(f'<div class="section-header">{section_label}</div>',
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
                st.markdown(f'<div class="section-header">Zone Profile ({season_label})</div>',
                            unsafe_allow_html=True)
                from lib.zone_charts import plot_hitter_zone_grid
                zone_stand = None
                if 'is_switch' in dir() and is_switch and platoon_side:
                    if platoon_side.startswith("vs RHP"):
                        zone_stand = "L"
                    elif platoon_side.startswith("vs LHP"):
                        zone_stand = "R"

                # Pitch-type filter (only when data includes pitch_type column)
                _hz_selected_pts: list[str] | None = None
                if "pitch_type" in h_zone.columns:
                    _hz_avail_pts = (
                        h_zone.groupby("pitch_type")["pitches"].sum()
                        .sort_values(ascending=False)
                    )
                    _hz_pt_options = [pt for pt in _hz_avail_pts.index if pt in PITCH_DISPLAY]
                    _hz_pt_labels = {pt: PITCH_DISPLAY.get(pt, pt) for pt in _hz_pt_options}
                    _hz_selected_pts = st.multiselect(
                        "Pitch types",
                        options=_hz_pt_options,
                        format_func=lambda pt: _hz_pt_labels[pt],
                        default=[],
                        key="hitter_zone_pitch_type_filter",
                        help="Leave empty to show all pitch types combined",
                    )

                zone_col1, zone_col2 = st.columns(2)
                with zone_col1:
                    fig_whiff = plot_hitter_zone_grid(
                        h_zone, metric="whiff_rate",
                        batter_name=selected_name, batter_stand=zone_stand,
                        pitch_types=_hz_selected_pts or None,
                    )
                    st.pyplot(fig_whiff, use_container_width=True)
                    plt.close(fig_whiff)
                with zone_col2:
                    fig_xwoba = plot_hitter_zone_grid(
                        h_zone, metric="xwoba",
                        batter_name=selected_name, batter_stand=zone_stand,
                        pitch_types=_hz_selected_pts or None,
                    )
                    st.pyplot(fig_xwoba, use_container_width=True)
                    plt.close(fig_xwoba)

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
            f'<div class="section-header">{season_label} Observed Percentiles</div>',
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
            f"<span style='color:{SAGE};'>Green</span> = elite (80+), "
            f"<span style='color:{GOLD};'>gold</span> = above-avg (60-79), "
            f"<span style='color:{SLATE};'>gray</span> = mid-tier (40-59), "
            f"<span style='color:{EMBER};'>orange</span> = below-avg (&lt;40)."
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
    }

    # Create multi-panel chart — compact size matching other dashboard charts
    n_stats = len(available_stats)
    n_cols = min(3, n_stats)
    n_rows = (n_stats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7, 1.5 * n_rows),
        facecolor=DARK, squeeze=False,
    )

    for idx, (label, key, higher_better, is_pct) in enumerate(available_stats):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_facecolor(DARK)

        vals = player_data[key].values
        valid_mask = ~pd.isna(vals)
        valid_seasons = seasons[valid_mask]
        valid_vals = vals[valid_mask].astype(float)

        if is_pct:
            valid_vals = valid_vals * 100

        # Main line
        ax.plot(valid_seasons, valid_vals, color=GOLD, linewidth=1.5, marker="o",
                markersize=4, zorder=3)

        # Highlight selected season
        if selected_season and selected_season in valid_seasons:
            sel_idx = list(valid_seasons).index(selected_season)
            ax.plot(selected_season, valid_vals[sel_idx], "o",
                    color=SAGE, markersize=7, zorder=4)

        # Set y-axis range: use league-wide range, expanded to include player data
        if key in _Y_RANGES:
            y_lo, y_hi = _Y_RANGES[key]
            data_lo, data_hi = float(valid_vals.min()), float(valid_vals.max())
            y_lo = min(y_lo, data_lo - 1)
            y_hi = max(y_hi, data_hi + 1)
            ax.set_ylim(y_lo, y_hi)
        else:
            # Fallback: pad by 20% of data range (minimum 2 units)
            data_lo, data_hi = float(valid_vals.min()), float(valid_vals.max())
            pad = max((data_hi - data_lo) * 0.3, 2.0)
            ax.set_ylim(data_lo - pad, data_hi + pad)

        # Style
        ax.set_title(label, color=CREAM, fontsize=9, fontweight="bold", pad=4)
        ax.tick_params(colors=SLATE, labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(SLATE)
        ax.spines["left"].set_color(SLATE)
        ax.grid(axis="y", color=SLATE, alpha=0.15, linewidth=0.5)
        ax.set_xlim(TRAIN_START - 0.5, PRIOR_SEASON + 0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

        # Trend arrow annotation
        if len(valid_vals) >= 2:
            delta = valid_vals[-1] - valid_vals[-2]
            if abs(delta) > 0.1:
                arrow = "+" if delta > 0 else ""
                color = SAGE if (delta > 0) == higher_better else EMBER
                fmt_delta = f"{arrow}{delta:.1f}{'%' if is_pct else ''}"
                ax.annotate(
                    fmt_delta,
                    xy=(valid_seasons[-1], valid_vals[-1]),
                    xytext=(5, 6), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold",
                )

    # Hide empty subplots
    for idx in range(n_stats, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout(pad=1.0)
    st.markdown('<div class="section-header">Season Trends</div>',
                unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
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
                return "--", "--", ""
            if pd.isna(prev_v):
                val_str = f"{curr_v*100:.{decimals}f}%" if is_pct else f"{curr_v:.{decimals}f}"
                return val_str, "--", "NEW"
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
            f'<div class="section-header">Arsenal Changes vs {selected_season - 1}</div>',
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
# Page: Player Profile
# ---------------------------------------------------------------------------
def page_player_profile() -> None:
    """Deep dive into a single player's projections."""
    st.markdown('<div class="section-header">Player Profile</div>',
                unsafe_allow_html=True)

    player_type = st.radio(
        "Player type",
        ["Pitcher", "Hitter"],
        horizontal=True,
        key="profile_type",
    )

    df = load_projections(player_type.lower())
    if df.empty:
        st.warning(
            "No projection data found. "
            "Run `python scripts/precompute_dashboard_data.py` first."
        )
        return

    if player_type == "Pitcher":
        name_col, id_col, hand_col = "pitcher_name", "pitcher_id", "pitch_hand"
        stat_configs = PITCHER_STATS
    else:
        name_col, id_col, hand_col = "batter_name", "batter_id", "batter_stand"
        stat_configs = HITTER_STATS

    # Player selector (with team filter and team abbreviations)
    team_lookup = get_team_lookup()

    # Add team_abbr to df for filtering
    df["_team"] = df[id_col].apply(lambda x: team_lookup.get(int(x), ""))

    sel_cols = st.columns([1, 3])
    with sel_cols[0]:
        profile_team_opts = ["All"] + sorted(df["_team"].replace("", pd.NA).dropna().unique().tolist())
        profile_team_filter = st.selectbox("Team", profile_team_opts, key="profile_team")
    with sel_cols[1]:
        filtered_df = df if profile_team_filter == "All" else df[df["_team"] == profile_team_filter]
        profile_display = {}
        for _, pr in filtered_df.iterrows():
            pid = int(pr[id_col])
            pname = pr[name_col]
            team = team_lookup.get(pid, "")
            dname = f"{pname} ({team})" if team else pname
            profile_display[dname] = pname
        selected_display = st.selectbox("Select player", sorted(profile_display.keys()), key="profile_player")
    selected_name = profile_display[selected_display]

    player_row = df[df[name_col] == selected_name].iloc[0]
    player_id = int(player_row[id_col])

    # --- Season selector ---
    season_choice = season_selector("profile", include_career=True)
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
        header_parts.append(player_team)
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

    composite = player_row["composite_score"]
    comp_color = POSITIVE if composite > 0 else NEGATIVE if composite < 0 else SLATE

    # Injury status
    injury_lookup = get_injury_lookup()
    inj_info = injury_lookup.get(player_id)
    injury_html = ""
    if inj_info and inj_info["missed_games"] > 0:
        inj_color = EMBER if inj_info["severity"] == "major" else GOLD
        injury_html = (
            f'<div style="color:{inj_color}; font-size:0.85rem; margin-top:4px;">'
            f'{inj_info["status"]} — {inj_info["injury"]} '
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
        f'<div style="color:{comp_color}; font-size:1.2rem; font-weight:600;">'
        f'Composite: {composite:+.2f}'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # --- Historical / Career stats view (non-projection) ---
    if show_trad:
        _trad_season_label = "Career" if is_career else str(selected_season)

        # Load traditional stats for the selected season or career
        trad_all_df = load_traditional_stats_all(player_type.lower())
        if not trad_all_df.empty:
            if is_career:
                trad_player = trad_all_df[trad_all_df[id_col] == player_id]
            else:
                trad_player = trad_all_df[
                    (trad_all_df[id_col] == player_id)
                    & (trad_all_df["season"] == selected_season)
                ]

            if not trad_player.empty:
                if is_career:
                    # Aggregate career: sum counting, weighted-average rates
                    trad_data = career_aggregate_trad(trad_player, player_type)
                else:
                    trad_data = trad_player.iloc[0]

                if player_type == "Hitter":
                    rate_configs_t = HITTER_TRAD_STATS
                    counting_configs_t = HITTER_TRAD_COUNTING
                else:
                    rate_configs_t = PITCHER_TRAD_STATS
                    counting_configs_t = PITCHER_TRAD_COUNTING

                # Build season population for percentiles
                if is_career:
                    _trad_pop = trad_all_df  # rank against all player-seasons
                else:
                    _trad_pop = trad_all_df[trad_all_df["season"] == selected_season]

                # Rate stat cards
                st.markdown(
                    f'<div class="section-header" style="font-size:1rem; margin-top:1rem;">'
                    f'{_trad_season_label} Rate Stats</div>',
                    unsafe_allow_html=True,
                )
                rate_cols = st.columns(len(rate_configs_t))
                for col, (label, col_name, higher_better, fmt) in zip(rate_cols, rate_configs_t):
                    val = trad_data.get(col_name) if hasattr(trad_data, 'get') else trad_data[col_name] if col_name in trad_data.index else None
                    _pct = None
                    if pd.notna(val) and col_name in _trad_pop.columns:
                        _pct = percentile_rank(_trad_pop[col_name], float(val), higher_better)
                    with col:
                        st.markdown(
                            metric_card(label, fmt_trad(val, fmt), pctile=_pct),
                            unsafe_allow_html=True,
                        )

                # Counting stat higher_is_better lookup
                _counting_hib = {
                    "games": True, "pa": True, "ab": True, "hits": True,
                    "doubles": True, "triples": True, "hr": True, "runs": True,
                    "rbi": True, "bb": True, "k": player_type == "Pitcher",
                    "sb": True, "cs": False,
                    "starts": True, "w": True, "l": False, "sv": True,
                    "hld": True, "ip": True, "bf": True, "hits_allowed": False,
                    "er": False, "hr_allowed": False, "hbp": False,
                }

                # Counting stat cards
                st.markdown(
                    f'<div class="section-header" style="font-size:1rem; margin-top:1rem;">'
                    f'{_trad_season_label} Counting Stats</div>',
                    unsafe_allow_html=True,
                )
                for i in range(0, len(counting_configs_t), 7):
                    chunk = counting_configs_t[i:i + 7]
                    c_cols = st.columns(len(chunk))
                    for col, (label, col_name) in zip(c_cols, chunk):
                        val = trad_data.get(col_name) if hasattr(trad_data, 'get') else trad_data[col_name] if col_name in trad_data.index else None
                        if pd.notna(val):
                            display_val = f"{val:.1f}" if col_name == "ip" else str(int(val))
                        else:
                            display_val = "--"
                        _pct = None
                        if pd.notna(val) and col_name in _trad_pop.columns:
                            _pct = percentile_rank(
                                _trad_pop[col_name], float(val),
                                _counting_hib.get(col_name, True),
                            )
                        with col:
                            st.markdown(
                                metric_card(label, display_val, pctile=_pct),
                                unsafe_allow_html=True,
                            )

                # Batted ball data warning for pre-2022
                if selected_season is not None and selected_season in UNRELIABLE_BB_SEASONS:
                    st.caption(
                        f"Note: Batted ball data coverage was limited in {selected_season} "
                        f"(Statcast coverage ~{21 + (selected_season - 2018) * 15}%). "
                        "Barrel rate, xwOBA, and hard-hit metrics may be unreliable."
                    )
            else:
                st.info(f"No stats found for {selected_name} in {_trad_season_label}.")
        else:
            st.info("No traditional stats data found. Run precompute with multi-season data first.")

        # --- Approach & Efficiency cards ---
        render_approach_efficiency(
            player_type, player_id, id_col,
            selected_season=selected_season, is_career=is_career,
        )

        # --- Observed percentiles for selected season ---
        render_observed_percentiles(
            player_type, player_id,
            selected_season=selected_season, is_career=is_career,
        )

        # --- Season Trends ---
        render_season_trends(
            player_type, player_id, selected_name,
            selected_season=selected_season,
        )

        # --- Pitch profile + zone charts for the selected season ---
        render_pitch_profiles(
            player_type, player_id, selected_name,
            selected_season=selected_season, is_career=is_career,
        )

        # --- Arsenal evolution (pitcher only, specific season) ---
        if player_type == "Pitcher" and selected_season and not is_career:
            render_arsenal_evolution(player_id, selected_name, selected_season)

        return  # Skip the projection view below

    # --- Comparison baseline toggle ---
    compare_to = st.radio(
        "Compare projection to",
        ["Career Avg", str(PRIOR_SEASON)],
        horizontal=True,
        key="compare_baseline",
    )

    # --- Stat metric cards ---
    cols = st.columns(len(stat_configs))
    for col, (label, key, higher_better, _) in zip(cols, stat_configs):
        obs_col = f"observed_{key}"
        career_col = f"career_{key}"
        proj_col = f"projected_{key}"

        if compare_to == "Career Avg" and career_col in player_row.index and pd.notna(player_row.get(career_col)):
            baseline = player_row[career_col]
            baseline_label = "Career"
        elif obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            baseline = player_row[obs_col]
            baseline_label = str(PRIOR_SEASON)
        else:
            baseline = None
            baseline_label = ""

        if baseline is not None and proj_col in player_row.index and pd.notna(player_row.get(proj_col)):
            proj_str = fmt_stat(player_row[proj_col], key)
            delta = player_row[proj_col] - baseline
            base_str = fmt_stat(baseline, key)
            delta_str = (
                f"{baseline_label}: {base_str} ({delta_html(delta, higher_better)})"
            )
            _pct = None
            if proj_col in df.columns:
                _pct = percentile_rank(df[proj_col], float(player_row[proj_col]), higher_better)
            with col:
                st.markdown(
                    metric_card(f"Proj. {label}", proj_str, delta_str, pctile=_pct),
                    unsafe_allow_html=True,
                )
        else:
            with col:
                st.markdown(
                    metric_card(f"Proj. {label}", "--"),
                    unsafe_allow_html=True,
                )

    # --- Counting Stat Cards ---
    counting_display = PITCHER_COUNTING_DISPLAY if player_type == "Pitcher" else HITTER_COUNTING_DISPLAY
    counting_df = load_counting(player_type.lower())
    if not counting_df.empty:
        c_row = counting_df[counting_df[id_col] == player_id]
        if not c_row.empty:
            c_data = c_row.iloc[0]
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            c_cols = st.columns(len(counting_display))
            for col, (c_label, c_prefix, c_actual, c_hb) in zip(c_cols, counting_display):
                mean_col = f"{c_prefix}_mean"
                p10_col = f"{c_prefix}_p10"
                p90_col = f"{c_prefix}_p90"
                if mean_col in c_data.index and pd.notna(c_data.get(mean_col)):
                    val = int(round(c_data[mean_col]))
                    lo = int(round(c_data.get(p10_col, val)))
                    hi = int(round(c_data.get(p90_col, val)))
                    # Baseline: Career Avg uses rate x projected PA, prior season uses actual
                    actual_val = c_data.get(c_actual)
                    if compare_to == "Career Avg":
                        # Derive career-pace counting total from career rate x projected PA
                        rate_key = c_prefix.replace("total_", "") + "_rate"
                        if rate_key == "sb_rate":
                            rate_key = "sb_per_game"  # SB uses games not PA
                        career_rate_col = f"career_{rate_key}"
                        proj_pa = c_data.get("projected_pa_mean", c_data.get("projected_bf_mean"))
                        if (career_rate_col in player_row.index
                                and pd.notna(player_row.get(career_rate_col))
                                and pd.notna(proj_pa)):
                            career_count = int(round(player_row[career_rate_col] * proj_pa))
                            delta = val - career_count
                            delta_str = f"Career pace: {career_count} ({delta:+d}) | 80%: {lo} – {hi}"
                        elif pd.notna(actual_val):
                            actual_int = int(actual_val)
                            delta = val - actual_int
                            delta_str = f"{PRIOR_SEASON}: {actual_int} ({delta:+d}) | 80%: {lo} – {hi}"
                        else:
                            delta_str = f"80% range: {lo} – {hi}"
                    elif pd.notna(actual_val):
                        actual_int = int(actual_val)
                        delta = val - actual_int
                        delta_str = f"{PRIOR_SEASON}: {actual_int} ({delta:+d}) | 80%: {lo} – {hi}"
                    else:
                        delta_str = f"80% range: {lo} – {hi}"
                    _pct = None
                    if mean_col in counting_df.columns:
                        _pct = percentile_rank(counting_df[mean_col], float(c_data[mean_col]), c_hb)
                    with col:
                        st.markdown(
                            metric_card(c_label, str(val), delta_str, pctile=_pct),
                            unsafe_allow_html=True,
                        )
                else:
                    with col:
                        st.markdown(
                            metric_card(c_label, "--"),
                            unsafe_allow_html=True,
                        )

    # --- Scouting Report (plain English) ---
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    bullets = generate_scouting_bullets(stat_configs, player_row, df, player_type)

    # Add park factor scouting note for hitters
    if player_type == "Hitter":
        _cnt = load_counting("hitter")
        if not _cnt.empty:
            _c_r = _cnt[_cnt["batter_id"] == player_id]
            if not _c_r.empty and "hr_park_factor" in _c_r.columns:
                _pf = _c_r.iloc[0].get("hr_park_factor")
                if pd.notna(_pf) and _pf > 1.03:
                    bullets.append((POSITIVE, f"Home park boosts HR rate (park factor {_pf:.3f}). Projected HRs adjusted up."))
                elif pd.notna(_pf) and _pf < 0.97:
                    bullets.append((NEGATIVE, f"Home park suppresses HR rate (park factor {_pf:.3f}). Projected HRs adjusted down."))

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

    # --- Approach & Efficiency (on default projection view -- shows prior season) ---
    render_approach_efficiency(
        player_type, player_id, id_col,
        selected_season=PRIOR_SEASON, is_career=False,
    )

    # --- Prior Season Observed Percentiles ---
    obs_stat_configs = PITCHER_OBSERVED_STATS if player_type == "Pitcher" else HITTER_OBSERVED_STATS
    obs_bars_html = ""
    for label, key, higher_better, _ in obs_stat_configs:
        if key not in player_row.index or pd.isna(player_row.get(key)):
            continue
        val = player_row[key]
        # Rank among all players in the projection set
        if key in df.columns:
            pctile = percentile_rank(df[key], val, higher_better)
        else:
            continue
        obs_bars_html += observed_pctile_bar_html(label, pctile, val, key)

    if obs_bars_html:
        st.markdown(f'<div class="section-header">{PRIOR_SEASON} Observed Percentiles</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-card">{obs_bars_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Current skill profile based on {PRIOR_SEASON} observed data. "
            "100th = best, 1st = worst. "
            "Green = elite (80+), gold = above-avg (60-79), "
            "gray = mid-tier (40-59), orange = below-avg (<40)."
        )

    # --- Projected Percentiles ---
    proj_bars_html = ""
    for label, key, higher_better, _ in stat_configs:
        proj_col = f"projected_{key}"
        obs_col = f"observed_{key}"
        ci_lo_col = f"projected_{key}_2_5"
        ci_hi_col = f"projected_{key}_97_5"

        if proj_col not in player_row.index or pd.isna(player_row.get(proj_col)):
            continue

        pctile = percentile_rank(df[proj_col], player_row[proj_col], higher_better)
        ci_lo = player_row.get(ci_lo_col, player_row[proj_col])
        ci_hi = player_row.get(ci_hi_col, player_row[proj_col])

        # Prior season observed percentile as reference line
        pctile_prior = None
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            pctile_prior = percentile_rank(
                df[obs_col], player_row[obs_col], higher_better,
            )

        proj_bars_html += pctile_bar_html(label, pctile, ci_lo, ci_hi, key, pctile_prior)

    if proj_bars_html:
        st.markdown(f'<div class="section-header">{CURRENT_SEASON} Projected Percentiles</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-card">{proj_bars_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Bayesian-projected K% and BB% for {CURRENT_SEASON}. "
            "100th = best, 1st = worst. "
            f"Dashed line = {PRIOR_SEASON} observed percentile. "
            "Green = elite (80+), gold = above-avg (60-79), "
            "gray = mid-tier (40-59), orange = below-avg (<40). "
            "Range = 95% credible interval."
        )

    # --- Season Trends (also on projection view) ---
    render_season_trends(player_type, player_id, selected_name)

    # --- Pitch Profile Tables ---
    if player_type == "Pitcher":
        # Platoon split toggle for pitcher arsenal (projection view)
        proj_platoon_choice = st.radio(
            "Batter handedness",
            ["All Batters", "vs LHH", "vs RHH"],
            horizontal=True,
            key="profile_platoon_split",
        )
        _proj_platoon_stand = {"vs LHH": "L", "vs RHH": "R"}.get(proj_platoon_choice)
        _proj_platoon_suffix = f" — {proj_platoon_choice}" if _proj_platoon_stand else ""

        if _proj_platoon_stand:
            # Load split data from the _all file and filter by batter_stand
            arsenal_all = load_pitcher_arsenal_all()
            if not arsenal_all.empty and "batter_stand" in arsenal_all.columns:
                _mask = (arsenal_all["pitcher_id"] == player_id) & (arsenal_all["batter_stand"] == _proj_platoon_stand)
                # Use most recent season if season column exists
                if "season" in arsenal_all.columns:
                    _mask = _mask & (arsenal_all["season"] == PRIOR_SEASON)
                p_arsenal = arsenal_all[_mask].copy()
                # Recompute usage_pct within the filtered subset
                if not p_arsenal.empty:
                    total_p = p_arsenal["pitches"].sum()
                    p_arsenal["usage_pct"] = p_arsenal["pitches"] / total_p if total_p > 0 else 0
            else:
                p_arsenal = pd.DataFrame()
        else:
            arsenal_df = load_pitcher_arsenal()
            p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == player_id].copy() if not arsenal_df.empty else pd.DataFrame()

        if not p_arsenal.empty:
            st.markdown(f'<div class="section-header">Pitch Arsenal{_proj_platoon_suffix}</div>',
                        unsafe_allow_html=True)
            table_html = build_pitcher_profile_table(p_arsenal)
            if table_html:
                st.markdown(
                    f'<div class="insight-card">{table_html}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Colors relative to league average per pitch type. "
                    f'<span style="color:{SAGE};">Green</span> = above avg, '
                    f'<span style="color:{GOLD};">gold</span> = avg, '
                    f'<span style="color:{EMBER};">orange</span> = below avg. '
                    "CSW% = called strikes + whiffs / pitches.",
                    unsafe_allow_html=True,
                )

        # --- Pitcher Location Heatmap ---
        ploc_df = load_pitcher_location_grid()
        if not ploc_df.empty:
            p_loc = ploc_df[ploc_df["pitcher_id"] == player_id]
            if not p_loc.empty:
                st.markdown('<div class="section-header">Pitch Location Profile</div>',
                            unsafe_allow_html=True)
                from lib.zone_charts import plot_pitcher_location_heatmap

                loc_stand = st.radio(
                    "Batter handedness",
                    ["All", "vs LHH", "vs RHH"],
                    horizontal=True,
                    key="profile_pitcher_loc_stand",
                )
                stand_filter = {"vs LHH": "L", "vs RHH": "R"}.get(loc_stand)

                # Pitch-type filter
                _avail_pts = (
                    p_loc.groupby("pitch_type")["pitches"].sum()
                    .sort_values(ascending=False)
                )
                _pt_options = [pt for pt in _avail_pts.index if pt in PITCH_DISPLAY]
                _pt_labels = {pt: PITCH_DISPLAY.get(pt, pt) for pt in _pt_options}
                _selected_pts = st.multiselect(
                    "Pitch types",
                    options=_pt_options,
                    format_func=lambda pt: _pt_labels[pt],
                    default=[],
                    key="profile_pitch_type_filter",
                    help="Leave empty to auto-select top 4 pitch types by volume",
                )

                fig_loc = plot_pitcher_location_heatmap(
                    p_loc, pitch_types=_selected_pts or None,
                    pitcher_name=selected_name, batter_stand=stand_filter,
                )
                st.pyplot(fig_loc, use_container_width=True)
                plt.close(fig_loc)
                st.caption("Pitch count shown per cell. Catcher's perspective.")
    else:
        vuln_df = load_hitter_vulnerability(career=True)
        if not vuln_df.empty:
            h_vuln_all = vuln_df[vuln_df["batter_id"] == player_id].copy()
            if not h_vuln_all.empty:
                # Detect switch hitter: significant data on both sides
                side_counts = h_vuln_all.groupby("batter_stand")["pitches"].sum()
                is_switch = (
                    len(side_counts) > 1
                    and all(v >= 50 for v in side_counts.values)
                )

                section_label = "Pitch-Type Profile (Career)"
                if is_switch:
                    platoon_side = st.radio(
                        "Batter side",
                        ["vs RHP (bats L)", "vs LHP (bats R)", "Combined"],
                        horizontal=True,
                        key="profile_platoon",
                    )
                    if platoon_side.startswith("vs RHP"):
                        h_vuln = h_vuln_all[h_vuln_all["batter_stand"] == "L"].copy()
                        section_label = "Pitch-Type Profile (Career — vs RHP)"
                    elif platoon_side.startswith("vs LHP"):
                        h_vuln = h_vuln_all[h_vuln_all["batter_stand"] == "R"].copy()
                        section_label = "Pitch-Type Profile (Career — vs LHP)"
                    else:
                        # Combined: sum raw counts across sides, recompute rates
                        h_vuln = combine_platoon_vuln(h_vuln_all)
                        section_label = "Pitch-Type Profile (Career — Combined)"
                else:
                    h_vuln = h_vuln_all

                st.markdown(f'<div class="section-header">{section_label}</div>',
                            unsafe_allow_html=True)
                table_html = build_hitter_profile_table(h_vuln)
                if table_html:
                    st.markdown(
                        f'<div class="insight-card">{table_html}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Colors relative to league average per pitch type. "
                        f'<span style="color:{EMBER};">Orange</span> = exploitable, '
                        f'<span style="color:{SAGE};">green</span> = strength. '
                        "CStr% = called strikes / pitches. "
                        "Bar opacity reflects sample confidence.",
                        unsafe_allow_html=True,
                    )

        # --- Hitter Zone Grid ---
        hzone_df = load_hitter_zone_grid(career=True)
        if not hzone_df.empty:
            h_zone = hzone_df[hzone_df["batter_id"] == player_id]
            if not h_zone.empty:
                st.markdown('<div class="section-header">Zone Profile (Career)</div>',
                            unsafe_allow_html=True)
                from lib.zone_charts import plot_hitter_zone_grid

                # Determine platoon stand for zone charts (match pitch-type profile)
                zone_stand = None
                if is_switch:
                    if platoon_side.startswith("vs RHP"):
                        zone_stand = "L"
                    elif platoon_side.startswith("vs LHP"):
                        zone_stand = "R"
                    # else Combined -> zone_stand = None

                # Pitch-type filter (only when data includes pitch_type column)
                _hz_career_selected_pts: list[str] | None = None
                if "pitch_type" in h_zone.columns:
                    _hz_career_avail_pts = (
                        h_zone.groupby("pitch_type")["pitches"].sum()
                        .sort_values(ascending=False)
                    )
                    _hz_career_pt_options = [pt for pt in _hz_career_avail_pts.index if pt in PITCH_DISPLAY]
                    _hz_career_pt_labels = {pt: PITCH_DISPLAY.get(pt, pt) for pt in _hz_career_pt_options}
                    _hz_career_selected_pts = st.multiselect(
                        "Pitch types",
                        options=_hz_career_pt_options,
                        format_func=lambda pt: _hz_career_pt_labels[pt],
                        default=[],
                        key="hitter_zone_career_pitch_type_filter",
                        help="Leave empty to show all pitch types combined",
                    )

                zone_col1, zone_col2 = st.columns(2)
                with zone_col1:
                    fig_whiff = plot_hitter_zone_grid(
                        h_zone, metric="whiff_rate",
                        batter_name=selected_name, batter_stand=zone_stand,
                        pitch_types=_hz_career_selected_pts or None,
                    )
                    st.pyplot(fig_whiff, use_container_width=True)
                    plt.close(fig_whiff)
                with zone_col2:
                    fig_xwoba = plot_hitter_zone_grid(
                        h_zone, metric="xwoba",
                        batter_name=selected_name, batter_stand=zone_stand,
                        pitch_types=_hz_career_selected_pts or None,
                    )
                    st.pyplot(fig_xwoba, use_container_width=True)
                    plt.close(fig_xwoba)
                st.caption(
                    f"Left: whiff vulnerability (<span style='color:{EMBER};'>orange</span> = exploitable). "
                    f"Right: damage on contact (<span style='color:{SAGE};'>green</span> = hitter strength). "
                    "Catcher's perspective. Cells with insufficient data dimmed.",
                    unsafe_allow_html=True,
                )

    # --- Posterior KDE (for pitchers with K% samples) ---
    k_samples = load_k_samples()
    sample_key = str(player_id)

    if player_type == "Pitcher" and sample_key in k_samples:
        st.markdown('<div class="section-header">K% Posterior Distribution</div>',
                    unsafe_allow_html=True)
        samples = k_samples[sample_key]
        obs_k = player_row.get("observed_k_rate")
        fig = create_posterior_fig(
            samples,
            observed=obs_k if pd.notna(obs_k) else None,
            stat_label=f"Projected K% ({CURRENT_SEASON})",
        )
        _, chart_col, _ = st.columns([1, 3, 1])
        with chart_col:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        ci_lo, ci_hi = np.percentile(samples * 100, [2.5, 97.5])
        st.caption(
            f"Dashed gold = projected mean | Dotted gray = {PRIOR_SEASON} observed | "
            f"Shaded = 95% credible interval [{ci_lo:.1f}%, {ci_hi:.1f}%]"
        )

    # --- Stat detail table ---
    st.markdown('<div class="section-header">Stat Breakdown</div>',
                unsafe_allow_html=True)
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
                "95% CI": (
                    f"[{fmt_stat(player_row[lo_col], key)}, "
                    f"{fmt_stat(player_row[hi_col], key)}]"
                    if lo_col in player_row.index and pd.notna(player_row.get(lo_col))
                    else "--"
                ),
                "Description": desc,
            })
    # Add counting stat rows to detail table
    counting_df_detail = load_counting(player_type.lower())
    if not counting_df_detail.empty:
        c_row_detail = counting_df_detail[counting_df_detail[id_col] == player_id]
        if not c_row_detail.empty:
            c_data_detail = c_row_detail.iloc[0]
            for c_label, c_prefix, c_actual, c_hb in counting_display:
                mean_col = f"{c_prefix}_mean"
                p2_5_col = f"{c_prefix}_p2_5"
                p97_5_col = f"{c_prefix}_p97_5"
                if mean_col in c_data_detail.index and pd.notna(c_data_detail.get(mean_col)):
                    proj_val = int(round(c_data_detail[mean_col]))
                    ci_lo = int(round(c_data_detail.get(p2_5_col, proj_val)))
                    ci_hi = int(round(c_data_detail.get(p97_5_col, proj_val)))
                    actual_val = c_data_detail.get(c_actual)
                    if pd.notna(actual_val):
                        actual_int = int(actual_val)
                        delta = proj_val - actual_int
                        delta_str = f"{delta:+d}"
                        obs_str = str(actual_int)
                    else:
                        delta_str = "--"
                        obs_str = "--"
                    detail_rows.append({
                        "Stat": c_label,
                        f"{PRIOR_SEASON} Observed": obs_str,
                        f"{CURRENT_SEASON} Projected": str(proj_val),
                        "Delta": delta_str,
                        "95% CI": f"[{ci_lo}, {ci_hi}]",
                        "Description": "Season total (Bayesian rate x playing time)",
                    })

    if detail_rows:
        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
        )
