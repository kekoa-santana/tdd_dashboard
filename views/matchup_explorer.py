"""Matchup Explorer page | head-to-head pitcher vs hitter breakdown."""

from __future__ import annotations

import html
import numpy as np
import pandas as pd
import streamlit as st

from config import EMBER, SAGE, SLATE, PITCH_DISPLAY
from services.data_loader import (
    load_baselines_arch,
    load_cluster_metadata,
    load_hitter_strength,
    load_hitter_vuln_arch_career,
    load_hitter_vulnerability,
    load_hitter_zone_grid,
    load_pitcher_arsenal,
    load_pitcher_location_grid,
    load_pitcher_offerings,
    load_pitcher_pitch_locations,
    load_projections,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_archetype_matchup_matrix,
)
from components.charts import (
    create_hitter_zone_plotly,
    create_matchup_whiff_bars_fig,
    create_pitch_density_plotly,
    matchup_hitter_zone_peak_caption,
    matchup_pitcher_peak_caption,
)
from utils.formatters import delta_html, fmt_pct
from utils.helpers import get_team_lookup
from components.metric_cards import metric_card
from components.tables import (
    build_hitter_profile_table,
    build_matchup_table,
    build_pitcher_profile_table,
)
from components.headshot import headshot_html
from components.scouting import build_matchup_scouting_bullets


def page_matchup_explorer() -> None:
    """Head-to-head pitcher vs hitter matchup breakdown."""
    from lib.matchup import (
        score_matchup,
        score_matchup_advantage,
        score_matchup_by_archetype,
    )
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE
    from services.data_loader import (
        load_batter_platoon_splits,
        load_batter_glicko,
        load_pitcher_glicko,
        load_pitcher_gb_pct,
        load_matchup_baselines,
    )

    st.markdown('<div class="tdd-section-hdr">Matchup Explorer</div>',
                unsafe_allow_html=True)

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    str_df = load_hitter_strength(career=True)
    pitcher_proj = load_projections("pitcher")
    hitter_proj = load_projections("hitter")

    # Load archetype data (may be empty if not precomputed)
    offerings_df = load_pitcher_offerings()
    vuln_arch_df = load_hitter_vuln_arch_career()
    cluster_meta_df = load_cluster_metadata()
    baselines_arch_df = load_baselines_arch()
    archetype_available = (
        not offerings_df.empty
        and not vuln_arch_df.empty
        and not baselines_arch_df.empty
    )

    if arsenal_df.empty or vuln_df.empty:
        st.warning(
            "Matchup data not found. Re-run "
            "`python scripts/precompute_dashboard_data.py` to generate it."
        )
        return

    team_lookup = get_team_lookup()
    qp_pitcher_id = st.query_params.get("pitcher_id", "")
    qp_hitter_id = st.query_params.get("hitter_id", "")

    col1, col2 = st.columns(2)
    with col1:
        if pitcher_proj.empty:
            st.warning("No pitcher projections available.")
            return
        p_display = {}
        for _, pr in pitcher_proj.iterrows():
            pid = int(pr["pitcher_id"])
            pname = pr["pitcher_name"]
            team = team_lookup.get(pid, "")
            dname = f"{pname} ({team})" if team else pname
            p_display[dname] = pname
        sorted_p_display = sorted(p_display.keys())
        default_p_idx = 0
        if qp_pitcher_id.isdigit():
            target_pid = int(qp_pitcher_id)
            for i, dname in enumerate(sorted_p_display):
                matched = pitcher_proj[pitcher_proj["pitcher_name"] == p_display[dname]]
                if not matched.empty and int(matched.iloc[0]["pitcher_id"]) == target_pid:
                    default_p_idx = i
                    break
        selected_pitcher_display = st.selectbox(
            "Select Pitcher", sorted_p_display, index=default_p_idx,
            key="mu_pitcher",
        )
        selected_pitcher = p_display[selected_pitcher_display]
    with col2:
        if hitter_proj.empty:
            st.warning("No hitter projections available.")
            return
        h_display = {}
        for _, hr_ in hitter_proj.iterrows():
            hid = int(hr_["batter_id"])
            hname = hr_["batter_name"]
            team = team_lookup.get(hid, "")
            dname = f"{hname} ({team})" if team else hname
            h_display[dname] = hname
        sorted_h_display = sorted(h_display.keys())
        default_h_idx = 0
        if qp_hitter_id.isdigit():
            target_hid = int(qp_hitter_id)
            for i, dname in enumerate(sorted_h_display):
                matched = hitter_proj[hitter_proj["batter_name"] == h_display[dname]]
                if not matched.empty and int(matched.iloc[0]["batter_id"]) == target_hid:
                    default_h_idx = i
                    break
        selected_hitter_display = st.selectbox(
            "Select Hitter", sorted_h_display, index=default_h_idx,
            key="mu_hitter",
        )
        selected_hitter = h_display[selected_hitter_display]

    pitcher_row = pitcher_proj[pitcher_proj["pitcher_name"] == selected_pitcher].iloc[0]
    hitter_row = hitter_proj[hitter_proj["batter_name"] == selected_hitter].iloc[0]
    pitcher_id = int(pitcher_row["pitcher_id"])
    batter_id = int(hitter_row["batter_id"])
    st.query_params["pitcher_id"] = str(pitcher_id)
    st.query_params["hitter_id"] = str(batter_id)
    pitcher_hand = pitcher_row.get("pitch_hand", "R")

    # --- Platoon-aware filtering for switch hitters ---
    # Determine which side the hitter bats from against this pitcher
    batter_vuln_all = vuln_df[vuln_df["batter_id"] == batter_id]
    side_counts = batter_vuln_all.groupby("batter_stand")["pitches"].sum() if not batter_vuln_all.empty else pd.Series(dtype=float)
    is_switch = len(side_counts) > 1 and all(v >= 50 for v in side_counts.values)

    if is_switch:
        # Switch hitter bats from opposite side of pitcher
        platoon_side = "L" if pitcher_hand == "R" else "R"
        vuln_filtered = vuln_df[
            (vuln_df["batter_id"] != batter_id)
            | (vuln_df["batter_stand"] == platoon_side)
        ].copy()
        str_filtered = str_df[
            (str_df["batter_id"] != batter_id)
            | (str_df["batter_stand"] == platoon_side)
        ].copy() if not str_df.empty else pd.DataFrame()
        hitter_hand = platoon_side
    else:
        vuln_filtered = vuln_df
        str_filtered = str_df
        hitter_hand = hitter_row.get("batter_stand", "?")

    # Build pitch-type baselines (needed for both modes)
    baselines_pt: dict[str, dict[str, float]] = {}
    for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items():
        baselines_pt[pt] = vals if isinstance(vals, dict) else {"whiff_rate": 0.25}

    # --- Matchup model toggle ---
    if archetype_available:
        matchup_mode = st.radio(
            "Matchup Model",
            ["Pitch-Type", "Archetype"],
            horizontal=True,
            key="matchup_mode",
            help="Pitch-Type matches individual pitches. Archetype groups similar pitches by movement/velo profile.",
        )
    else:
        matchup_mode = "Pitch-Type"

    # Score the matchup (archetype or pitch-type)
    using_archetype = matchup_mode == "Archetype"
    if using_archetype:
        baselines_arch_dict: dict[int, dict[str, float]] = {}
        for _, brow in baselines_arch_df.iterrows():
            arch_id = int(brow["pitch_archetype"])
            baselines_arch_dict[arch_id] = {"whiff_rate": float(brow["whiff_rate"])}

        matchup = score_matchup_by_archetype(
            pitcher_id, batter_id,
            offerings_df, vuln_arch_df, baselines_arch_dict,
            cluster_metadata=cluster_meta_df if not cluster_meta_df.empty else None,
            hitter_vuln_pt=vuln_filtered,
            baselines_pt=baselines_pt,
        )
    else:
        matchup = score_matchup(
            pitcher_id, batter_id, arsenal_df, vuln_filtered, baselines_pt,
        )

    # Pitch-type rows for tables, scouting, charts (always pitch-type arsenal)
    p_arsenal = arsenal_df[arsenal_df["pitcher_id"] == pitcher_id].copy()
    h_vuln = vuln_filtered[vuln_filtered["batter_id"] == batter_id].copy()
    h_str = (
        str_filtered[str_filtered["batter_id"] == batter_id].copy()
        if not str_filtered.empty else pd.DataFrame()
    )

    # --- Unified matchup advantage ---
    # Load auxiliary data for enriched scoring
    platoon_df = load_batter_platoon_splits()
    pitcher_glicko_df = load_pitcher_glicko()
    batter_glicko_df = load_batter_glicko()
    pitcher_gb_df = load_pitcher_gb_pct()
    matchup_baselines = load_matchup_baselines()

    # Build batter platoon splits dict
    batter_platoon_dict = None
    if not platoon_df.empty:
        bp = platoon_df[platoon_df["batter_id"] == batter_id]
        if not bp.empty:
            batter_platoon_dict = {}
            for _, row in bp.iterrows():
                batter_platoon_dict[row["pitch_hand"]] = {
                    "k_rate": float(row["k_rate"]),
                    "bb_rate": float(row["bb_rate"]),
                    "pa": int(row["pa"]),
                }
            batter_platoon_dict["overall_k_rate"] = float(bp["overall_k_rate"].iloc[0])
            batter_platoon_dict["overall_bb_rate"] = float(bp["overall_bb_rate"].iloc[0])

    # Look up Glicko ratings
    p_glicko_mu = None
    if not pitcher_glicko_df.empty:
        pg = pitcher_glicko_df[pitcher_glicko_df["pitcher_id"] == pitcher_id]
        if not pg.empty:
            p_glicko_mu = float(pg["mu"].iloc[0])
    b_glicko_mu = None
    if not batter_glicko_df.empty:
        bg = batter_glicko_df[batter_glicko_df["batter_id"] == batter_id]
        if not bg.empty:
            b_glicko_mu = float(bg["mu"].iloc[0])

    # Look up pitcher GB% and batter GB/FB rates
    pitcher_gb_val = None
    if not pitcher_gb_df.empty:
        pgb = pitcher_gb_df[pitcher_gb_df["pitcher_id"] == pitcher_id]
        if not pgb.empty:
            pitcher_gb_val = float(pgb["gb_pct"].iloc[0])
    batter_fb_val = None
    if not hitter_proj.empty:
        hp = hitter_proj[hitter_proj["batter_id"] == batter_id]
        if not hp.empty:
            batter_fb_val = float(hp["fb_rate"].iloc[0]) if "fb_rate" in hp.columns else None

    # Build matchup scales dict
    scales_dict = None
    if matchup_baselines:
        scales_dict = {
            "trajectory_scale": matchup_baselines.get("trajectory_scale", 5.0),
            "glicko_scale": matchup_baselines.get("glicko_scale", 0.001),
        }

    advantage_result = score_matchup_advantage(
        pitcher_id, batter_id, arsenal_df, vuln_filtered, baselines_pt,
        hitter_str=str_filtered if not str_filtered.empty else None,
        pitcher_hand=pitcher_hand,
        batter_platoon_splits=batter_platoon_dict,
        pitcher_gb_pct=pitcher_gb_val,
        batter_fb_rate=batter_fb_val,
        pitcher_glicko_mu=p_glicko_mu,
        batter_glicko_mu=b_glicko_mu,
        league_platoon=matchup_baselines if matchup_baselines else None,
        matchup_scales=scales_dict,
    )

    # Extract results for display
    lift = advantage_result["breakdown"]["k_lift"]
    bb_lift_h = advantage_result["breakdown"]["bb_lift"]
    hr_lift_h = advantage_result["breakdown"]["hr_lift"]
    edge_label = {
        "pitcher": "Pitcher Advantage",
        "hitter": "Hitter Advantage",
        "neutral": "Neutral Matchup",
    }[advantage_result["advantage"]]
    edge_color = {"pitcher": EMBER, "hitter": SAGE, "neutral": SLATE}[advantage_result["advantage"]]

    pitcher_label = "LHP" if pitcher_hand == "L" else "RHP"
    hitter_label = "LHH" if hitter_hand == "L" else "RHH"
    switch_tag = " (switch)" if is_switch else ""
    hand_str = f"{pitcher_label} vs {hitter_label}{switch_tag}"

    mwhiff = matchup["matchup_whiff_rate"]
    bwhiff = matchup["baseline_whiff_rate"]
    whiff_delta = (mwhiff - bwhiff) if pd.notna(mwhiff) and pd.notna(bwhiff) else 0.0
    whiff_delta_pp = whiff_delta * 100.0
    whiff_delta_html = delta_html(whiff_delta, higher_is_better=True) if whiff_delta != 0 else ""

    p_headshot = headshot_html(pitcher_id, size=50)
    h_headshot = headshot_html(batter_id, size=50)
    _edge_expl = (
        f"Unified matchup advantage: {advantage_result['reason']}. "
        f"Signals: K/BB/HR pitch-type lifts, platoon, primary pitch, contact quality, "
        f"trajectory, skill rating. Confidence: {advantage_result['confidence']}."
    )
    _k_logit_tip = (
        f"K lift (logit scale): {lift:+.3f} — technical scale from the pitch-type matchup model; "
        "compare to whiff delta in percentage points below."
    )
    _edge_expl_esc = html.escape(_edge_expl, quote=True)
    _k_logit_tip_esc = html.escape(_k_logit_tip, quote=True)
    matchup_header_html = (
        f'<div class="brand-header">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'{p_headshot}'
        f'<div>'
        f'<div class="brand-title">{selected_pitcher} vs {selected_hitter}</div>'
        f'<div class="brand-subtitle">{hand_str} | {"Archetype" if using_archetype else "Pitch-type"} profiles</div>'
        f'</div>'
        f'{h_headshot}'
        f'</div>'
        f'<div style="text-align:right; max-width:22rem;">'
        f'<div class="tdd-edge-label" style="color:{edge_color};">{edge_label}</div>'
        f'<div class="tdd-meta" style="margin-top:0.25rem;">'
        f'<b>Whiff vs baseline:</b> {whiff_delta_pp:+.1f} pp'
        f' <span class="tdd-tip" title="{_k_logit_tip_esc}">(K lift ⓘ)</span>'
        f'</div>'
        f'<div class="tdd-meta" style="font-size:0.78rem; color:var(--tdd-slate); margin-top:0.2rem;">'
        f'Model lifts (pitch-type): K {lift:+.3f} | BB {bb_lift_h:+.3f} | HR {hr_lift_h:+.3f}'
        f'</div>'
        f'<div class="tdd-meta" style="font-size:0.72rem; color:var(--tdd-slate); margin-top:0.15rem;" '
        f'title="{_edge_expl_esc}">Why this badge? (hover)</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(matchup_header_html, unsafe_allow_html=True)

    # --- Key pitches (top scouting bullets, above the fold) ---
    if not p_arsenal.empty and not h_vuln.empty:
        _kp = build_matchup_scouting_bullets(
            p_arsenal, h_vuln, h_str, selected_pitcher, selected_hitter,
        )[:3]
        if _kp:
            kp_html = "".join(
                f'<div style="color:{c}; font-size:0.82rem; margin:0.12rem 0; '
                f'padding-left:0.65rem; border-left:2px solid {c};">{t}</div>'
                for c, t in _kp
            )
            st.markdown(
                f'<div style="margin:0.35rem 0 0.6rem;">'
                f'<div style="color:var(--tdd-cream); font-size:0.85rem; font-weight:600; '
                f'margin-bottom:0.25rem;">Key pitches</div>{kp_html}</div>',
                unsafe_allow_html=True,
            )

    # --- Summary metrics ---

    m_cols = st.columns(4)
    with m_cols[0]:
        st.markdown(
            metric_card(
                "Matchup Whiff",
                fmt_pct(mwhiff) if pd.notna(mwhiff) else "",
                whiff_delta_html,
            ),
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            metric_card("Baseline Whiff", fmt_pct(bwhiff) if pd.notna(bwhiff) else ""),
            unsafe_allow_html=True,
        )
    with m_cols[2]:
        st.markdown(
            metric_card("Pitch Types", str(matchup["n_pitch_types"])),
            unsafe_allow_html=True,
        )
    with m_cols[3]:
        st.markdown(
            metric_card("Data Reliability", f"{matchup['avg_reliability']:.0%}"),
            unsafe_allow_html=True,
        )

    # --- Combined matchup breakdown table ---
    breakdown_label = "Archetype Matchup" if using_archetype else "Pitch-by-Pitch Matchup"
    st.markdown(f'<div class="tdd-section-hdr">{breakdown_label}</div>',
                unsafe_allow_html=True)

    if using_archetype:
        # Build archetype breakdown table
        p_off = offerings_df[offerings_df["pitcher_id"] == pitcher_id].copy()
        p_off = p_off[p_off["pitches"] >= 20].copy()
        if p_off.empty:
            st.info("Insufficient archetype offering data for detailed breakdown.")
        else:
            # Build archetype label map from cluster metadata
            arch_label_map: dict[int, str] = {}
            if not cluster_meta_df.empty:
                for _, cmrow in cluster_meta_df.iterrows():
                    aid = int(cmrow["pitch_archetype"])
                    label = cmrow.get("archetype_name", f"Cluster {aid}")
                    arch_label_map[aid] = str(label)

            # Aggregate offerings by archetype
            # pitcher_offerings lacks swings/whiffs | join from arsenal
            p_ars_for_arch = arsenal_df[
                (arsenal_df["pitcher_id"] == pitcher_id) & (arsenal_df["pitches"] >= 10)
            ].copy()
            if "pitch_archetype" not in p_ars_for_arch.columns and not p_off.empty:
                pt_to_arch = p_off.set_index("pitch_type")["pitch_archetype"].to_dict()
                p_ars_for_arch["pitch_archetype"] = p_ars_for_arch["pitch_type"].map(pt_to_arch)

            if "whiff_rate" in p_ars_for_arch.columns and "pitch_archetype" in p_ars_for_arch.columns:
                # Pitches-weighted whiff_rate by archetype
                p_ars_for_arch["weighted_whiff"] = p_ars_for_arch["whiff_rate"] * p_ars_for_arch["pitches"]
                arch_agg = p_ars_for_arch.groupby("pitch_archetype", as_index=False).agg(
                    pitches=("pitches", "sum"),
                    weighted_whiff=("weighted_whiff", "sum"),
                )
                arch_agg["whiff_rate"] = arch_agg["weighted_whiff"] / arch_agg["pitches"]
                arch_agg = arch_agg.drop(columns=["weighted_whiff"])
            else:
                arch_agg = p_off.groupby("pitch_archetype", as_index=False).agg(
                    pitches=("pitches", "sum"),
                )
                arch_agg["whiff_rate"] = np.nan
            total_p = arch_agg["pitches"].sum()
            arch_agg["usage_pct"] = arch_agg["pitches"] / total_p if total_p > 0 else 0.0
            arch_agg = arch_agg.sort_values("usage_pct", ascending=False)

            # Build HTML table
            rows_html = ""
            for _, arow in arch_agg.iterrows():
                aid = int(arow["pitch_archetype"])
                label = arch_label_map.get(aid, f"Cluster {aid}")
                usage = arow["usage_pct"]
                p_whiff = arow["whiff_rate"]

                # Hitter whiff for this archetype
                h_arch_rows = vuln_arch_df[
                    (vuln_arch_df["batter_id"] == batter_id)
                    & (vuln_arch_df["pitch_archetype"] == aid)
                ]
                h_whiff = h_arch_rows["whiff_rate"].iloc[0] if not h_arch_rows.empty else np.nan

                # Baseline for edge coloring
                league_w = baselines_arch_dict.get(aid, {}).get("whiff_rate", 0.25)
                edge_val = 0.0
                if pd.notna(p_whiff) and pd.notna(h_whiff):
                    edge_val = (p_whiff - league_w) + (h_whiff - league_w)

                if edge_val > 0.02:
                    edge_color_cell = EMBER
                    edge_sym = "+"
                elif edge_val < -0.02:
                    edge_color_cell = SAGE
                    edge_sym = "-"
                else:
                    edge_color_cell = SLATE
                    edge_sym = "="

                p_whiff_str = f"{p_whiff:.1%}" if pd.notna(p_whiff) else ""
                h_whiff_str = f"{h_whiff:.1%}" if pd.notna(h_whiff) else ""

                rows_html += (
                    f"<tr>"
                    f'<td style="text-align:left; padding:6px 10px;">{label}</td>'
                    f'<td style="text-align:center; padding:6px 10px;">{usage:.1%}</td>'
                    f'<td style="text-align:center; padding:6px 10px;">{p_whiff_str}</td>'
                    f'<td style="text-align:center; padding:6px 10px;">{h_whiff_str}</td>'
                    f'<td style="text-align:center; padding:6px 10px; color:{edge_color_cell}; font-weight:600;">{edge_sym}</td>'
                    f"</tr>"
                )

            arch_table_html = (
                '<table style="width:100%; border-collapse:collapse; font-size:0.9rem;">'
                "<thead><tr>"
                '<th style="text-align:left; padding:6px 10px; border-bottom:1px solid #333;">Archetype</th>'
                '<th style="text-align:center; padding:6px 10px; border-bottom:1px solid #333;">Usage</th>'
                '<th style="text-align:center; padding:6px 10px; border-bottom:1px solid #333;">P Whiff%</th>'
                '<th style="text-align:center; padding:6px 10px; border-bottom:1px solid #333;">H Whiff%</th>'
                '<th style="text-align:center; padding:6px 10px; border-bottom:1px solid #333;">Edge</th>'
                "</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>"
            )
            st.markdown(
                f'<div class="insight-card">{arch_table_html}</div>',
                unsafe_allow_html=True,
            )
            platoon_note = f" Hitter stats from {hitter_label} side." if is_switch else ""
            st.caption(
                "P Whiff% = pitcher's whiff rate for that archetype | "
                "H Whiff% = hitter's whiff rate against that archetype | "
                f"Edge: green = pitcher advantage, red = hitter advantage.{platoon_note}"
            )
    elif p_arsenal.empty:
        st.info("Insufficient arsenal data for detailed breakdown.")
    else:
        _whiff_fig = create_matchup_whiff_bars_fig(
            p_arsenal, h_vuln, h_str, selected_pitcher, selected_hitter,
        )
        if _whiff_fig is not None:
            st.markdown(
                '<div class="tdd-meta" style="margin:0.2rem 0 0.4rem;">'
                "Whiff% by pitch — bar opacity scales with usage; dashed line = league whiff.</div>",
                unsafe_allow_html=True,
            )
            st.pyplot(_whiff_fig, clear_figure=True)

        matchup_html = build_matchup_table(p_arsenal, h_vuln, h_str)
        if matchup_html:
            st.markdown(
                f'<div class="insight-card">{matchup_html}</div>',
                unsafe_allow_html=True,
            )
            platoon_note = f" Hitter stats from {hitter_label} side." if is_switch else ""
            st.caption(
                "P Whiff% = pitcher's whiff rate with that pitch | "
                "H Whiff% = hitter's whiff rate against that pitch type | "
                "H Chase% = hitter's chase rate | H xwOBA = xwOBA on contact | "
                f"Edge: green = pitcher advantage, red = hitter advantage.{platoon_note}"
            )

    # --- Location Matchup (Plotly contours — same as schedule matchup analysis) ---
    ploc_df = load_pitcher_location_grid()
    hzone_df = load_hitter_zone_grid(career=True)
    _raw_locs = load_pitcher_pitch_locations()

    if not ploc_df.empty and not hzone_df.empty:
        p_loc = ploc_df[ploc_df["pitcher_id"] == pitcher_id]
        h_zone = hzone_df[hzone_df["batter_id"] == batter_id]
        _p_raw = _raw_locs[_raw_locs["pitcher_id"] == pitcher_id] if not _raw_locs.empty else pd.DataFrame()
        _pitch_loc_data = _p_raw if not _p_raw.empty else p_loc

        _pitch_hand = pitcher_hand if pitcher_hand in ("L", "R") else None
        _batter_stand = hitter_hand if hitter_hand in ("L", "R") else None

        p_arsenal_me = arsenal_df[arsenal_df["pitcher_id"] == pitcher_id] if not arsenal_df.empty else pd.DataFrame()

        if not _pitch_loc_data.empty and not p_arsenal_me.empty:
            total_pitches = p_arsenal_me["pitches"].sum()
            p_arsenal_sorted = p_arsenal_me.copy()
            p_arsenal_sorted["usage"] = p_arsenal_sorted["pitches"] / total_pitches
            p_arsenal_sorted = p_arsenal_sorted.sort_values("usage", ascending=False)

            primary_pts = p_arsenal_sorted[p_arsenal_sorted["usage"] >= 0.10]["pitch_type"].tolist()

            if primary_pts:
                st.markdown(
                    '<div class="tdd-section-hdr">Location Matchup</div>',
                    unsafe_allow_html=True,
                )

                for rank, pt in enumerate(primary_pts, 1):
                    pt_name = PITCH_DISPLAY.get(pt, pt)
                    _pt_row = p_arsenal_sorted[p_arsenal_sorted["pitch_type"] == pt]
                    _pt_usage = f" ({_pt_row.iloc[0]['usage']:.0%})" if not _pt_row.empty else ""
                    st.markdown(
                        f'<div class="tdd-meta" style="margin:0.8rem 0 0.2rem;">'
                        f'#{rank} {pt_name}{_pt_usage}</div>',
                        unsafe_allow_html=True,
                    )
                    matchup_cols = st.columns(2, gap="medium")
                    with matchup_cols[0]:
                        fig_density = create_pitch_density_plotly(
                            _pitch_loc_data, pitch_types=[pt],
                            pitcher_name=selected_pitcher,
                            batter_stand=_batter_stand,
                            pitch_hand=_pitch_hand,
                        )
                        st.plotly_chart(
                            fig_density, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": False},
                            key=f"me_density_{pitcher_id}_{batter_id}_{pt}",
                        )
                    with matchup_cols[1]:
                        if not h_zone.empty:
                            fig_xwoba = create_hitter_zone_plotly(
                                h_zone, metric="xwoba",
                                batter_name=selected_hitter,
                                batter_stand=_batter_stand,
                                pitch_types=[pt] if "pitch_type" in h_zone.columns else None,
                            )
                            st.plotly_chart(
                                fig_xwoba, use_container_width=True,
                                config={"displayModeBar": False, "scrollZoom": False},
                                key=f"me_xwoba_{pitcher_id}_{batter_id}_{pt}",
                            )
                        else:
                            st.markdown(
                                f'<div class="tdd-meta">No zone data for {selected_hitter}</div>',
                                unsafe_allow_html=True,
                            )

                    _cap_p = matchup_pitcher_peak_caption(
                        _pitch_loc_data, pt, _batter_stand,
                    )
                    _cap_h = (
                        matchup_hitter_zone_peak_caption(
                            h_zone, pt, _batter_stand, metric="xwoba",
                        )
                        if not h_zone.empty
                        else None
                    )
                    _caps = [x for x in (_cap_p, _cap_h) if x]
                    if _caps:
                        st.caption(" · ".join(_caps))

                # Secondary pitches
                secondary_pts = p_arsenal_sorted[p_arsenal_sorted["usage"] < 0.10]
                if not secondary_pts.empty:
                    sec_parts = [
                        f"{PITCH_DISPLAY.get(row['pitch_type'], row['pitch_type'])} ({row['usage']:.0%})"
                        for _, row in secondary_pts.iterrows()
                    ]
                    st.markdown(
                        f'<div class="tdd-context" style="margin:0.3rem 0 0.8rem;">'
                        f'Also throws: {", ".join(sec_parts)} | limited usage, not shown above'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # --- Side-by-side profile tables ---
    st.markdown('<div class="tdd-section-hdr">Individual Profiles</div>',
                unsafe_allow_html=True)

    prof_col1, prof_col2 = st.columns(2)
    with prof_col1:
        st.markdown(
            f'<div class="tdd-section-hdr" style="margin-bottom:0.5rem;">'
            f'{selected_pitcher} | Arsenal</div>',
            unsafe_allow_html=True,
        )
        if not p_arsenal.empty:
            p_table = build_pitcher_profile_table(p_arsenal)
            if p_table:
                st.markdown(
                    f'<div class="insight-card">{p_table}</div>',
                    unsafe_allow_html=True,
                )

    with prof_col2:
        vuln_label = f"{selected_hitter} | Vulnerabilities"
        if is_switch:
            vuln_label += f" (batting {hitter_hand})"
        st.markdown(
            f'<div class="tdd-section-hdr" style="margin-bottom:0.5rem;">'
            f'{vuln_label}</div>',
            unsafe_allow_html=True,
        )
        if not h_vuln.empty:
            h_table = build_hitter_profile_table(h_vuln)
            if h_table:
                st.markdown(
                    f'<div class="insight-card">{h_table}</div>',
                    unsafe_allow_html=True,
                )

    # --- Archetype Matchup Matrix ---
    _mm_df = load_archetype_matchup_matrix()
    if not _mm_df.empty:
        st.markdown('<div class="tdd-section-hdr">Archetype Matchup Matrix</div>',
                    unsafe_allow_html=True)

        # Determine selected player archetypes for highlighting
        _p_arch_df = load_pitcher_archetypes()
        _h_arch_df = load_hitter_archetypes()
        _sel_p_arch = None
        _sel_h_arch = None
        if not _p_arch_df.empty:
            _pa_row = _p_arch_df[_p_arch_df["pitcher_id"] == pitcher_id]
            if not _pa_row.empty:
                _sel_p_arch = _pa_row.iloc[0]["archetype_name"]
        if not _h_arch_df.empty:
            _ha_row = _h_arch_df[_h_arch_df["batter_id"] == batter_id]
            if not _ha_row.empty:
                _sel_h_arch = _ha_row.iloc[0]["archetype_name"]

        _mm_stat = st.radio(
            "Stat", ["K%", "BB%", "HR%"], horizontal=True, key="mm_stat",
        )
        _stat_col = {"K%": "k_pct", "BB%": "bb_pct", "HR%": "hr_pct"}[_mm_stat]

        # Pivot to matrix
        _pivot = _mm_df.pivot(
            index="pitcher_archetype_name",
            columns="hitter_archetype_name",
            values=_stat_col,
        )
        _p_names = _pivot.index.tolist()
        _h_names = _pivot.columns.tolist()

        # Color scale: for K% higher → ember, lower → sage; for BB%/HR% higher → ember, lower → sage
        _all_vals = _pivot.values.flatten()
        _vmin, _vmax = float(_all_vals.min()), float(_all_vals.max())

        def _cell_color(val: float) -> str:
            if _vmax == _vmin:
                return SLATE
            t = (val - _vmin) / (_vmax - _vmin)
            if _mm_stat == "K%":
                # Higher K% → ember (pitcher advantage)
                return EMBER if t > 0.66 else SAGE if t < 0.33 else SLATE
            else:
                # Higher BB%/HR% → sage (hitter advantage)
                return SAGE if t > 0.66 else EMBER if t < 0.33 else SLATE

        # Build HTML table
        _hdr = "".join(
            f'<th style="padding:6px 8px; font-size:0.75rem; color:'
            f'{"var(--tdd-gold)" if name == _sel_h_arch else "var(--tdd-cream)"}; '
            f'{"font-weight:700; text-decoration:underline;" if name == _sel_h_arch else ""}'
            f'border-bottom:1px solid var(--tdd-dark-border);">{name}</th>'
            for name in _h_names
        )
        _rows = ""
        for p_name in _p_names:
            _is_sel_row = (p_name == _sel_p_arch)
            _row_label_style = (
                f'color:var(--tdd-gold); font-weight:700; text-decoration:underline;'
                if _is_sel_row else f'color:var(--tdd-cream);'
            )
            _cells = f'<td style="padding:6px 8px; {_row_label_style} font-size:0.8rem; white-space:nowrap;">{p_name}</td>'
            for h_name in _h_names:
                val = _pivot.loc[p_name, h_name]
                color = _cell_color(val)
                _is_highlight = _is_sel_row or (h_name == _sel_h_arch)
                _bg = f"{color}33" if _is_highlight else f"{color}18"
                _fw = "700" if _is_highlight else "400"
                _cells += (
                    f'<td style="padding:6px 8px; text-align:center; font-size:0.8rem; '
                    f'color:{color}; font-weight:{_fw}; background:{_bg};">'
                    f'{val:.1%}</td>'
                )
            _rows += f'<tr style="border-bottom:1px solid var(--tdd-dark-border)22;">{_cells}</tr>'

        _matrix_html = (
            f'<table style="width:100%; border-collapse:collapse;">'
            f'<thead><tr><th style="padding:6px 8px;"></th>{_hdr}</tr></thead>'
            f'<tbody>{_rows}</tbody></table>'
        )
        st.markdown(f'<div class="insight-card">{_matrix_html}</div>', unsafe_allow_html=True)

        _highlight_parts = []
        if _sel_p_arch:
            _highlight_parts.append(f"{selected_pitcher} → **{_sel_p_arch}** (row highlighted)")
        if _sel_h_arch:
            _highlight_parts.append(f"{selected_hitter} → **{_sel_h_arch}** (column highlighted)")
        if _highlight_parts:
            st.caption(" | ".join(_highlight_parts))
        st.caption(
            f"6×6 matrix of historical {_mm_stat} by pitcher/hitter archetype pair. "
            "Red = higher rate, green = lower rate."
        )

    # --- Scouting report ---
    st.markdown('<div class="tdd-section-hdr">Matchup Scouting Report</div>',
                unsafe_allow_html=True)

    # Overall summary | use blended edge (whiff lift + contact quality)
    bullets_html = ""
    if pd.notna(mwhiff) and pd.notna(bwhiff):
        whiff_delta_pp = (mwhiff - bwhiff) * 100
        if blended_edge > 0.15:
            summary = (
                f"Favorable matchup for <b>{selected_pitcher}</b>. "
                f"The hitter's pitch-type vulnerabilities align well with the arsenal"
            )
            if whiff_delta_pp > 1:
                summary += f", boosting the expected whiff rate by {whiff_delta_pp:.1f}pp above baseline."
            else:
                summary += "."
        elif blended_edge < -0.15:
            summary = (
                f"Tough matchup for <b>{selected_pitcher}</b>. "
            )
            if damage_score > 0.03:
                summary += (
                    f"<b>{selected_hitter}</b> does significant damage on contact against this arsenal"
                )
                if whiff_delta_pp > 1:
                    summary += f" despite an elevated whiff rate (+{whiff_delta_pp:.1f}pp)."
                else:
                    summary += f" and handles the pitch mix well."
            else:
                summary += (
                    f"<b>{selected_hitter}</b> handles this arsenal well, pulling the expected "
                    f"whiff rate {abs(whiff_delta_pp):.1f}pp below baseline."
                )
        else:
            summary = (
                f"Neutral matchup | no strong edge either way. "
                f"The whiff rate shifts by only {whiff_delta_pp:+.1f}pp from baseline."
            )

        bullets_html += (
            f'<div class="insight-bullet">'
            f'<span class="dot" style="background:{edge_color};"></span>'
            f'{summary}</div>'
        )

    # Per-pitch scouting bullets
    if not p_arsenal.empty and not h_vuln.empty:
        pitch_bullets = build_matchup_scouting_bullets(
            p_arsenal, h_vuln, h_str, selected_pitcher, selected_hitter,
        )
        for color, text in pitch_bullets:
            bullets_html += (
                f'<div class="insight-bullet">'
                f'<span class="dot" style="background:{color};"></span>'
                f'{text}</div>'
            )

    # Reliability note
    bullets_html += (
        f'<div class="insight-bullet">'
        f'<span class="dot" style="background:{SLATE};"></span>'
        f'Data reliability: {matchup["avg_reliability"]:.0%} '
        f'(based on sample sizes across {matchup["n_pitch_types"]} pitch types)</div>'
    )

    st.markdown(
        f'<div class="insight-card">'
        f'<div class="insight-title">Matchup Summary</div>'
        f'{bullets_html}</div>',
        unsafe_allow_html=True,
    )
