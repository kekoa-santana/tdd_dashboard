"""Matchup Explorer page — head-to-head pitcher vs hitter breakdown."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import EMBER, GOLD, SAGE, SLATE
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
    load_projections,
)
from utils.formatters import delta_html, fmt_pct
from utils.helpers import get_team_lookup
from components.metric_cards import metric_card
from components.tables import (
    build_hitter_profile_table,
    build_matchup_table,
    build_pitcher_profile_table,
)
from components.scouting import build_matchup_scouting_bullets


def page_matchup_explorer() -> None:
    """Head-to-head pitcher vs hitter matchup breakdown."""
    from lib.matchup import score_matchup, score_matchup_by_archetype
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    st.markdown('<div class="section-header">Matchup Explorer</div>',
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

    # --- Selectors (with team abbreviations) ---
    team_lookup = get_team_lookup()
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
        selected_pitcher_display = st.selectbox("Select Pitcher", sorted(p_display.keys()), key="mu_pitcher")
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
        selected_hitter_display = st.selectbox("Select Hitter", sorted(h_display.keys()), key="mu_hitter")
        selected_hitter = h_display[selected_hitter_display]

    pitcher_row = pitcher_proj[pitcher_proj["pitcher_name"] == selected_pitcher].iloc[0]
    hitter_row = hitter_proj[hitter_proj["batter_name"] == selected_hitter].iloc[0]
    pitcher_id = int(pitcher_row["pitcher_id"])
    batter_id = int(hitter_row["batter_id"])
    pitcher_hand = pitcher_row.get("pitch_hand", "R")

    # --- Platoon-aware filtering for switch hitters ---
    # Determine which side the hitter bats from against this pitcher
    batter_vuln_all = vuln_df[vuln_df["batter_id"] == batter_id]
    batter_str_all = str_df[str_df["batter_id"] == batter_id] if not str_df.empty else pd.DataFrame()
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

    # Score the matchup
    using_archetype = matchup_mode == "Archetype"
    if using_archetype:
        # Convert baselines_arch DataFrame to dict format
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

    # --- Compute contact-quality adjustment ---
    # Usage-weighted xwOBA and hard-hit delta vs league baselines
    p_ars = arsenal_df[
        (arsenal_df["pitcher_id"] == pitcher_id) & (arsenal_df["pitches"] >= 20)
    ].copy()
    h_str_filt = str_filtered[str_filtered["batter_id"] == batter_id] if not str_filtered.empty else pd.DataFrame()
    damage_score = 0.0
    total_usage = p_ars["usage_pct"].sum() if not p_ars.empty else 0.0
    if total_usage > 0 and not h_str_filt.empty:
        for _, prow in p_ars.iterrows():
            pt = prow["pitch_type"]
            usage_w = prow["usage_pct"] / total_usage
            lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {})
            lg_xwoba = lg.get("xwoba_contact", 0.320)
            lg_hh = lg.get("hard_hit_rate", 0.33)
            s_row = h_str_filt[h_str_filt["pitch_type"] == pt]
            if s_row.empty:
                continue
            h_xwoba = s_row["xwoba_contact"].iloc[0] if "xwoba_contact" in s_row.columns else np.nan
            h_hh = s_row["hard_hit_rate"].iloc[0] if "hard_hit_rate" in s_row.columns else np.nan
            if pd.notna(h_xwoba):
                damage_score += usage_w * (h_xwoba - lg_xwoba)
            if pd.notna(h_hh):
                damage_score += usage_w * (h_hh - lg_hh) * 0.5

    # --- Matchup header ---
    # Blend whiff lift (pitcher-favorable when positive) with damage score
    # (hitter-favorable when positive). Convert damage to same scale as logit lift.
    lift = matchup["matchup_k_logit_lift"]
    blended_edge = lift - damage_score * 6.0  # scale damage to logit-lift magnitude
    if blended_edge > 0.15:
        edge_label = "Pitcher Advantage"
        edge_color = SAGE
    elif blended_edge < -0.15:
        edge_label = "Hitter Advantage"
        edge_color = EMBER
    else:
        edge_label = "Neutral Matchup"
        edge_color = SLATE

    pitcher_label = "LHP" if pitcher_hand == "L" else "RHP"
    hitter_label = "LHH" if hitter_hand == "L" else "RHH"
    switch_tag = " (switch)" if is_switch else ""
    hand_str = f"{pitcher_label} vs {hitter_label}{switch_tag}"

    matchup_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{selected_pitcher} vs {selected_hitter}</div>'
        f'<div class="brand-subtitle">{hand_str} | {"Archetype" if using_archetype else "Pitch-type"} profiles</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="color:{edge_color}; font-size:1.1rem; font-weight:600;">{edge_label}</div>'
        f'<div style="color:{SLATE}; font-size:0.85rem;">K Lift: {lift:+.3f} logit</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(matchup_header_html, unsafe_allow_html=True)

    # --- Summary metrics ---
    mwhiff = matchup["matchup_whiff_rate"]
    bwhiff = matchup["baseline_whiff_rate"]
    whiff_delta = (mwhiff - bwhiff) if pd.notna(mwhiff) and pd.notna(bwhiff) else 0.0
    whiff_delta_html = delta_html(whiff_delta, higher_is_better=True) if whiff_delta != 0 else ""

    m_cols = st.columns(4)
    with m_cols[0]:
        st.markdown(
            metric_card(
                "Matchup Whiff",
                fmt_pct(mwhiff) if pd.notna(mwhiff) else "--",
                whiff_delta_html,
            ),
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            metric_card("Baseline Whiff", fmt_pct(bwhiff) if pd.notna(bwhiff) else "--"),
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
    st.markdown(f'<div class="section-header">{breakdown_label}</div>',
                unsafe_allow_html=True)

    p_arsenal = arsenal_df[
        arsenal_df["pitcher_id"] == pitcher_id
    ].copy()
    h_vuln = vuln_filtered[vuln_filtered["batter_id"] == batter_id].copy()
    h_str = str_filtered[str_filtered["batter_id"] == batter_id].copy() if not str_filtered.empty else pd.DataFrame()

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
                    label = cmrow.get("label", cmrow.get("archetype_label", f"Archetype {aid}"))
                    arch_label_map[aid] = str(label)

            # Aggregate offerings by archetype
            arch_agg = p_off.groupby("pitch_archetype", as_index=False).agg(
                pitches=("pitches", "sum"),
                swings=("swings", "sum"),
                whiffs=("whiffs", "sum"),
            )
            arch_agg["whiff_rate"] = arch_agg["whiffs"] / arch_agg["swings"].replace(0, np.nan)
            total_p = arch_agg["pitches"].sum()
            arch_agg["usage_pct"] = arch_agg["pitches"] / total_p if total_p > 0 else 0.0
            arch_agg = arch_agg.sort_values("usage_pct", ascending=False)

            # Build HTML table
            rows_html = ""
            for _, arow in arch_agg.iterrows():
                aid = int(arow["pitch_archetype"])
                label = arch_label_map.get(aid, f"Archetype {aid}")
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
                    edge_color_cell = SAGE
                    edge_sym = "+"
                elif edge_val < -0.02:
                    edge_color_cell = EMBER
                    edge_sym = "-"
                else:
                    edge_color_cell = SLATE
                    edge_sym = "="

                p_whiff_str = f"{p_whiff:.1%}" if pd.notna(p_whiff) else "--"
                h_whiff_str = f"{h_whiff:.1%}" if pd.notna(h_whiff) else "--"

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

    # --- Location Matchup Overlay ---
    ploc_df = load_pitcher_location_grid()
    hzone_df = load_hitter_zone_grid(career=True)
    if not ploc_df.empty and not hzone_df.empty:
        p_loc = ploc_df[ploc_df["pitcher_id"] == pitcher_id]
        h_zone = hzone_df[hzone_df["batter_id"] == batter_id]
        if not p_loc.empty and not h_zone.empty:
            from lib.zone_charts import plot_matchup_overlay

            st.markdown('<div class="section-header">Location Matchup</div>',
                        unsafe_allow_html=True)
            st.caption(
                f"Gold dots = where the pitcher throws each pitch. "
                f'Background = hitter whiff vulnerability '
                f'(<span style="color:{EMBER};">orange</span> = exploitable, '
                f'<span style="color:{SAGE};">green</span> = strong). '
                f"Catcher's perspective.",
                unsafe_allow_html=True,
            )

            # Show top pitch types by usage
            pt_totals = p_loc.groupby("pitch_type")["pitches"].sum().sort_values(ascending=False)
            top_pts = pt_totals.head(4).index.tolist()

            stand_for_overlay = hitter_hand if hitter_hand in ("L", "R") else None

            overlay_cols = st.columns(min(len(top_pts), 4))
            for i, pt in enumerate(top_pts):
                with overlay_cols[i]:
                    fig_ov = plot_matchup_overlay(
                        p_loc, h_zone, pitch_type=pt,
                        pitcher_name=selected_pitcher, hitter_name=selected_hitter,
                        batter_stand=stand_for_overlay,
                    )
                    st.pyplot(fig_ov, use_container_width=True)
                    plt.close(fig_ov)

    # --- Side-by-side profile tables ---
    st.markdown('<div class="section-header">Individual Profiles</div>',
                unsafe_allow_html=True)

    prof_col1, prof_col2 = st.columns(2)
    with prof_col1:
        st.markdown(
            f'<div style="color:{GOLD}; font-size:0.95rem; font-weight:600; '
            f'margin-bottom:0.5rem;">{selected_pitcher} — Arsenal</div>',
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
        vuln_label = f"{selected_hitter} — Vulnerabilities"
        if is_switch:
            vuln_label += f" (batting {hitter_hand})"
        st.markdown(
            f'<div style="color:{GOLD}; font-size:0.95rem; font-weight:600; '
            f'margin-bottom:0.5rem;">{vuln_label}</div>',
            unsafe_allow_html=True,
        )
        if not h_vuln.empty:
            h_table = build_hitter_profile_table(h_vuln)
            if h_table:
                st.markdown(
                    f'<div class="insight-card">{h_table}</div>',
                    unsafe_allow_html=True,
                )

    # --- Scouting report ---
    st.markdown('<div class="section-header">Matchup Scouting Report</div>',
                unsafe_allow_html=True)

    # Overall summary — use blended edge (whiff lift + contact quality)
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
                f"Neutral matchup — no strong edge either way. "
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
