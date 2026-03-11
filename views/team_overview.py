"""Team Overview page — team-level view of projected pitchers and hitters."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    POSITIVE, NEGATIVE,
    PITCHER_STATS, HITTER_STATS,
    CURRENT_SEASON, PRIOR_SEASON,
)
from services.data_loader import (
    load_projections, load_counting, load_player_teams,
    load_pitcher_arsenal, load_preseason_injuries,
)
from utils.helpers import get_team_lookup, get_injury_lookup
from utils.formatters import fmt_stat


def page_team_overview() -> None:
    """Team-level view of projected pitchers and hitters with strengths/weaknesses."""
    st.markdown('<div class="section-header">Team Overview</div>',
                unsafe_allow_html=True)

    # Load data
    teams_df = load_player_teams()
    if teams_df.empty:
        st.warning("No team data found. Run precompute first.")
        return

    team_lookup = get_team_lookup()
    injury_lookup = get_injury_lookup()

    # Team selector
    all_teams = sorted(teams_df["team_abbr"].replace("", pd.NA).dropna().unique().tolist())
    selected_team = st.selectbox("Select team", all_teams, key="team_select")

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
            c for c in h_count.columns
            if c.endswith("_mean") or c.startswith("actual_")
        ]
        available = [c for c in h_merge_cols if c in h_count.columns]
        team_hitters = team_hitters.merge(h_count[available], on="batter_id", how="left")

    if not p_count.empty:
        p_merge_cols = ["pitcher_id"] + [
            c for c in p_count.columns
            if c.endswith("_mean") or c.startswith("actual_")
        ]
        available = [c for c in p_merge_cols if c in p_count.columns]
        team_pitchers = team_pitchers.merge(p_count[available], on="pitcher_id", how="left")

    # ── Header ──────────────────────────────────────────────────────
    _inj_full = load_preseason_injuries()
    n_injured = len(_inj_full[
        (_inj_full["team_abbr"] == selected_team) & (_inj_full["est_missed_games"] > 0)
    ]) if not _inj_full.empty else 0
    team_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{selected_team}</div>'
        f'<div class="brand-subtitle">{len(team_pitchers)} pitchers | {len(team_hitters)} hitters | {n_injured} injured</div>'
        f'</div>'
        f'<div style="color:{SLATE}; font-size:0.9rem;">{CURRENT_SEASON} Season</div>'
        f'</div>'
    )
    st.markdown(team_header_html, unsafe_allow_html=True)

    # View toggle: Projections vs 2025 Priors
    view_mode = st.radio(
        "View",
        [f"{CURRENT_SEASON} Projections", f"{PRIOR_SEASON} Priors (Observed)"],
        horizontal=True,
        key="team_view_mode",
    )
    use_priors = view_mode == f"{PRIOR_SEASON} Priors (Observed)"

    # Select which K%/BB% columns to use based on view mode
    if use_priors:
        _h_k_col = "observed_k_rate"
        _h_bb_col = "observed_bb_rate"
        _p_k_col = "observed_k_rate"
        _p_bb_col = "observed_bb_rate"
        _view_label = f"{PRIOR_SEASON} Observed"
    else:
        _h_k_col = "projected_k_rate"
        _h_bb_col = "projected_bb_rate"
        _p_k_col = "projected_k_rate"
        _p_bb_col = "projected_bb_rate"
        _view_label = f"{CURRENT_SEASON} Projected"

    # ── Team Identity Tags ──────────────────────────────────────────
    identity_tags: list[tuple[str, str]] = []  # (label, color)

    if not team_hitters.empty and not h_proj.empty:
        # Power vs contact tags (only if Statcast cols available)
        _has_statcast = all(c in h_proj.columns for c in ("hard_hit_pct", "avg_exit_velo", "whiff_rate", "z_contact_pct"))
        if _has_statcast:
            lg_hh = h_proj["hard_hit_pct"].dropna().mean()
            lg_ev = h_proj["avg_exit_velo"].dropna().mean()
            lg_whiff = h_proj["whiff_rate"].dropna().mean()
            lg_zcon = h_proj["z_contact_pct"].dropna().mean()
            team_hh = team_hitters["hard_hit_pct"].dropna().mean()
            team_ev = team_hitters["avg_exit_velo"].dropna().mean()
            team_whiff = team_hitters["whiff_rate"].dropna().mean()
            team_zcon = team_hitters["z_contact_pct"].dropna().mean()

            power_score = (team_hh - lg_hh) / max(h_proj["hard_hit_pct"].dropna().std(), 0.001) \
                        + (team_ev - lg_ev) / max(h_proj["avg_exit_velo"].dropna().std(), 0.001)
            contact_score = (team_zcon - lg_zcon) / max(h_proj["z_contact_pct"].dropna().std(), 0.001) \
                          + (lg_whiff - team_whiff) / max(h_proj["whiff_rate"].dropna().std(), 0.001)

            if power_score > 1.0:
                identity_tags.append(("Power Offense", GOLD))
            elif power_score < -1.0:
                identity_tags.append(("Low-Power Offense", SLATE))
            if contact_score > 1.0:
                identity_tags.append(("Contact Offense", SAGE))
            elif contact_score < -1.0:
                identity_tags.append(("Swing-and-Miss Offense", EMBER))

        # Lineup handedness
        n_left = (team_hitters["batter_stand"] == "L").sum()
        n_right = (team_hitters["batter_stand"] == "R").sum()
        n_switch = (team_hitters["batter_stand"] == "S").sum()
        total_h = len(team_hitters)
        if total_h > 0:
            left_pct = (n_left + n_switch * 0.5) / total_h
            if left_pct >= 0.55:
                identity_tags.append(("LHB-Heavy Lineup", SLATE))
            elif left_pct <= 0.30:
                identity_tags.append(("RHB-Heavy Lineup", SLATE))
            else:
                identity_tags.append(("Balanced Lineup", SLATE))

    if not team_pitchers.empty and not p_proj.empty:
        # Staff handedness
        n_lhp = (team_pitchers["pitch_hand"] == "L").sum()
        n_rhp = (team_pitchers["pitch_hand"] == "R").sum()
        total_p = len(team_pitchers)
        if total_p > 0:
            lhp_pct = n_lhp / total_p
            if lhp_pct >= 0.45:
                identity_tags.append(("LHP-Heavy Staff", SLATE))
            elif lhp_pct <= 0.20:
                identity_tags.append(("RHP-Heavy Staff", SLATE))

        # Strikeout staff vs control staff
        lg_k = p_proj["projected_k_rate"].dropna().mean()
        lg_bb = p_proj["projected_bb_rate"].dropna().mean()
        team_k = team_pitchers["projected_k_rate"].dropna().mean()
        team_bb = team_pitchers["projected_bb_rate"].dropna().mean()
        k_z = (team_k - lg_k) / max(p_proj["projected_k_rate"].dropna().std(), 0.001)
        bb_z = (team_bb - lg_bb) / max(p_proj["projected_bb_rate"].dropna().std(), 0.001)
        if k_z > 0.8:
            identity_tags.append(("High-K Staff", GOLD))
        elif k_z < -0.8:
            identity_tags.append(("Low-K Staff", EMBER))
        if bb_z < -0.8:
            identity_tags.append(("Control Staff", SAGE))
        elif bb_z > 0.8:
            identity_tags.append(("Walk-Prone Staff", EMBER))

    # Staff arsenal breakdown
    arsenal_df = load_pitcher_arsenal()
    team_arsenal_summary = None
    if not arsenal_df.empty and not team_pitchers.empty:
        team_pitcher_ids = set(team_pitchers["pitcher_id"].astype(int))
        team_ars = arsenal_df[arsenal_df["pitcher_id"].isin(team_pitcher_ids)]
        if not team_ars.empty:
            # Aggregate by pitch family — weighted by pitches thrown
            family_agg = (
                team_ars.groupby("pitch_family")
                .agg(pitches=("pitches", "sum"), whiffs=("whiffs", "sum"), swings=("swings", "sum"))
                .reset_index()
            )
            family_agg["pct"] = family_agg["pitches"] / family_agg["pitches"].sum()
            family_agg["whiff_rate"] = family_agg["whiffs"] / family_agg["swings"].clip(lower=1)
            team_arsenal_summary = family_agg.sort_values("pct", ascending=False)

            # Tag heavy arsenal leans
            fb_pct = family_agg.loc[family_agg["pitch_family"] == "fastball", "pct"]
            brk_pct = family_agg.loc[family_agg["pitch_family"] == "breaking", "pct"]
            if not fb_pct.empty and float(fb_pct.iloc[0]) >= 0.55:
                identity_tags.append(("Fastball-Heavy Staff", SLATE))
            if not brk_pct.empty and float(brk_pct.iloc[0]) >= 0.35:
                identity_tags.append(("Breaking-Heavy Staff", SLATE))

    # Render identity tags
    if identity_tags:
        tags_html = " ".join(
            f'<span style="background:{color}22; color:{color}; border:1px solid {color}44; '
            f'padding:4px 12px; border-radius:16px; font-size:0.85rem; font-weight:600; '
            f'margin-right:6px;">{label}</span>'
            for label, color in identity_tags
        )
        st.markdown(f'<div style="margin-bottom:16px;">{tags_html}</div>',
                    unsafe_allow_html=True)

    # ── Staff Arsenal Breakdown ─────────────────────────────────────
    if team_arsenal_summary is not None and not team_arsenal_summary.empty:
        st.markdown("### Staff Arsenal Mix")
        ars_cols = st.columns(len(team_arsenal_summary))
        for col, (_, row) in zip(ars_cols, team_arsenal_summary.iterrows()):
            family = row["pitch_family"].title()
            pct = row["pct"]
            whiff = row["whiff_rate"]
            col.metric(family, f"{pct:.0%}", f"Whiff: {whiff:.1%}")

        # Pitch type detail
        if not team_ars.empty:
            pt_agg = (
                team_ars.groupby("pitch_type")
                .agg(pitches=("pitches", "sum"), whiffs=("whiffs", "sum"), swings=("swings", "sum"))
                .reset_index()
            )
            pt_agg["pct"] = pt_agg["pitches"] / pt_agg["pitches"].sum()
            pt_agg["whiff_rate"] = pt_agg["whiffs"] / pt_agg["swings"].clip(lower=1)
            pt_agg = pt_agg.sort_values("pct", ascending=False)
            pt_rows = []
            for _, r in pt_agg.iterrows():
                if r["pct"] >= 0.02:  # Only show pitch types with >= 2% usage
                    pt_rows.append({
                        "Pitch": r["pitch_type"],
                        "Usage": f"{r['pct']:.1%}",
                        "Whiff%": f"{r['whiff_rate']:.1%}",
                        "Pitches": int(r["pitches"]),
                    })
            with st.expander("Pitch type detail"):
                st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

    # ── Team Strengths & Weaknesses (offense) ───────────────────────
    # Compare team averages to league averages across all projected hitters
    if not team_hitters.empty and not h_proj.empty:
        st.markdown(f"### Offense Profile ({_view_label})")
        st.caption(
            f"Based on {'projected' if not use_priors else f'{PRIOR_SEASON} observed'} rates "
            "and Statcast metrics. Does not account for defense."
        )

        offense_metrics = []
        for label, key, higher_better in [
            ("K%", _h_k_col, False),
            ("BB%", _h_bb_col, True),
            ("Whiff%", "whiff_rate", False),
            ("Chase%", "chase_rate", False),
            ("Avg EV", "avg_exit_velo", True),
            ("Hard-Hit%", "hard_hit_pct", True),
        ]:
            if key not in team_hitters.columns or key not in h_proj.columns:
                continue
            team_avg = team_hitters[key].dropna().mean()
            league_avg = h_proj[key].dropna().mean()
            if pd.isna(team_avg) or pd.isna(league_avg) or league_avg == 0:
                continue
            diff = team_avg - league_avg
            # For rates shown as percentages
            if key in (_h_k_col, _h_bb_col, "whiff_rate", "chase_rate", "hard_hit_pct"):
                diff_str = f"{diff * 100:+.1f}pp"
                team_str = f"{team_avg * 100:.1f}%"
                lg_str = f"{league_avg * 100:.1f}%"
            else:
                diff_str = f"{diff:+.1f}"
                team_str = f"{team_avg:.1f}"
                lg_str = f"{league_avg:.1f}"

            is_good = (diff > 0 and higher_better) or (diff < 0 and not higher_better)
            color = POSITIVE if is_good else NEGATIVE if abs(diff) > 0.001 else SLATE
            offense_metrics.append({
                "Metric": label,
                "Team": team_str,
                "League Avg": lg_str,
                "Diff": diff_str,
                "_color": color,
                "_is_good": is_good,
            })

        if offense_metrics:
            strengths = [m for m in offense_metrics if m["_is_good"]]
            weaknesses = [m for m in offense_metrics if not m["_is_good"]]

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown(f'<div style="color:{POSITIVE}; font-weight:600; margin-bottom:8px;">Strengths</div>',
                            unsafe_allow_html=True)
                if strengths:
                    for m in strengths:
                        st.markdown(
                            f'<div style="padding:4px 0;"><span style="color:{CREAM};">{m["Metric"]}</span>: '
                            f'<span style="color:{POSITIVE}; font-weight:600;">{m["Team"]}</span> '
                            f'<span style="color:{SLATE};">(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(f'<span style="color:{SLATE};">None vs league average</span>',
                                unsafe_allow_html=True)
            with col_w:
                st.markdown(f'<div style="color:{NEGATIVE}; font-weight:600; margin-bottom:8px;">Weaknesses</div>',
                            unsafe_allow_html=True)
                if weaknesses:
                    for m in weaknesses:
                        st.markdown(
                            f'<div style="padding:4px 0;"><span style="color:{CREAM};">{m["Metric"]}</span>: '
                            f'<span style="color:{NEGATIVE}; font-weight:600;">{m["Team"]}</span> '
                            f'<span style="color:{SLATE};">(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(f'<span style="color:{SLATE};">None vs league average</span>',
                                unsafe_allow_html=True)

    # ── Pitching staff profile ──────────────────────────────────────
    if not team_pitchers.empty and not p_proj.empty:
        st.markdown(f"### Pitching Staff Profile ({_view_label})")
        pitch_metrics = []
        for label, key, higher_better in [
            ("K%", _p_k_col, True),
            ("BB%", _p_bb_col, False),
            ("Whiff%", "whiff_rate", True),
            ("Avg Velo", "avg_velo", True),
            ("Zone%", "zone_pct", True),
            ("GB%", "gb_pct", True),
        ]:
            if key not in team_pitchers.columns or key not in p_proj.columns:
                continue
            team_avg = team_pitchers[key].dropna().mean()
            league_avg = p_proj[key].dropna().mean()
            if pd.isna(team_avg) or pd.isna(league_avg) or league_avg == 0:
                continue
            diff = team_avg - league_avg
            if key in (_p_k_col, _p_bb_col, "whiff_rate", "zone_pct", "gb_pct"):
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

        if pitch_metrics:
            strengths = [m for m in pitch_metrics if m["_is_good"]]
            weaknesses = [m for m in pitch_metrics if not m["_is_good"]]
            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown(f'<div style="color:{POSITIVE}; font-weight:600; margin-bottom:8px;">Strengths</div>',
                            unsafe_allow_html=True)
                if strengths:
                    for m in strengths:
                        st.markdown(
                            f'<div style="padding:4px 0;"><span style="color:{CREAM};">{m["Metric"]}</span>: '
                            f'<span style="color:{POSITIVE}; font-weight:600;">{m["Team"]}</span> '
                            f'<span style="color:{SLATE};">(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(f'<span style="color:{SLATE};">None vs league average</span>',
                                unsafe_allow_html=True)
            with col_w:
                st.markdown(f'<div style="color:{NEGATIVE}; font-weight:600; margin-bottom:8px;">Weaknesses</div>',
                            unsafe_allow_html=True)
                if weaknesses:
                    for m in weaknesses:
                        st.markdown(
                            f'<div style="padding:4px 0;"><span style="color:{CREAM};">{m["Metric"]}</span>: '
                            f'<span style="color:{NEGATIVE}; font-weight:600;">{m["Team"]}</span> '
                            f'<span style="color:{SLATE};">(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(f'<span style="color:{SLATE};">None vs league average</span>',
                                unsafe_allow_html=True)

    # ── Injured players (from injury parquet by team, not just player_teams) ──
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
                "Player": row["player_name"],
                "Pos": row["position"],
                "Injury": row["injury"],
                "Status": row["status"],
                "Est. Return": row["est_return_date"],
                "~Games Missed": int(row["est_missed_games"]),
            })
        st.dataframe(pd.DataFrame(inj_rows), use_container_width=True, hide_index=True)

    # ── Pitchers table ──────────────────────────────────────────────
    st.markdown("### Pitchers")

    if team_pitchers.empty:
        st.info("No pitcher projections for this team.")
    else:
        p_rows = []
        for _, row in team_pitchers.sort_values("composite_score", ascending=False).iterrows():
            pid = int(row["pitcher_id"])
            inj = injury_lookup.get(pid)
            name = row["pitcher_name"]
            if inj and inj["missed_games"] > 0:
                sev = inj["severity"]
                tag = "[IL-60]" if sev == "major" else "[IL]" if sev == "significant" else "[DTD]"
                name = f"{tag} {name}"
            role_str = "SP" if row.get("is_starter") else "RP"
            r: dict[str, object] = {
                "Name": name,
                "Role": role_str,
                "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
                "Hand": row.get("pitch_hand", ""),
                "Score": round(row["composite_score"], 2),
            }
            if use_priors:
                # Show 2025 observed stats
                for label, key, _, _ in PITCHER_STATS:
                    obs_col = f"observed_{key}"
                    if obs_col in row.index and pd.notna(row.get(obs_col)):
                        r[f"{label} ({PRIOR_SEASON})"] = fmt_stat(row[obs_col], key)
                    else:
                        r[f"{label} ({PRIOR_SEASON})"] = "--"
                # Observed Statcast
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
            p_rows.append(r)
        st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

    # ── Hitters table ───────────────────────────────────────────────
    st.markdown("### Hitters")

    if team_hitters.empty:
        st.info("No hitter projections for this team.")
    else:
        h_rows = []
        for _, row in team_hitters.sort_values("composite_score", ascending=False).iterrows():
            pid = int(row["batter_id"])
            inj = injury_lookup.get(pid)
            name = row["batter_name"]
            if inj and inj["missed_games"] > 0:
                sev = inj["severity"]
                tag = "[IL-60]" if sev == "major" else "[IL]" if sev == "significant" else "[DTD]"
                name = f"{tag} {name}"
            r: dict[str, object] = {
                "Name": name,
                "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
                "Bats": row.get("batter_stand", ""),
                "Score": round(row["composite_score"], 2),
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
        st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

    st.caption(
        "Strengths/weaknesses compare team averages to league average across all projected players. "
        "Offense profile reflects batting projections only — does not account for defensive value. "
        + (f"Showing {PRIOR_SEASON} observed stats (priors for the Bayesian model)." if use_priors
           else f"Deltas shown in parentheses (pp vs {PRIOR_SEASON}).")
    )
