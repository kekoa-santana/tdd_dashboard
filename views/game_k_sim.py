"""Game K Simulator page — interactive K prop simulator with lineup,
umpire, and weather controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    POSITIVE, NEGATIVE, DASHBOARD_DIR,
    PRIOR_SEASON, TRAINING_RANGE,
)
from services.data_loader import (
    load_k_samples, load_bf_priors, load_projections,
    load_pitcher_arsenal, load_hitter_vulnerability,
)
from utils.helpers import get_team_lookup
from utils.formatters import fmt_pct
from components.metric_cards import metric_card, percentile_rank
from components.charts import create_game_k_fig


def page_game_k_sim() -> None:
    """Simulate game K totals for a selected pitcher."""
    from lib.bf_model import get_bf_distribution
    from lib.game_k_model import compute_k_over_probs, simulate_game_ks

    st.markdown('<div class="section-header">Game K Simulator</div>',
                unsafe_allow_html=True)

    k_samples_dict = load_k_samples()
    bf_priors = load_bf_priors()

    if not k_samples_dict:
        st.warning(
            "No K% posterior samples found. "
            "Run `python scripts/precompute_dashboard_data.py` first."
        )
        return

    pitcher_proj = load_projections("pitcher")
    if pitcher_proj.empty:
        st.warning("No pitcher projections found.")
        return

    # Filter to pitchers with K% samples
    available_ids = set(k_samples_dict.keys())
    pitchers_with_samples = pitcher_proj[
        pitcher_proj["pitcher_id"].astype(str).isin(available_ids)
    ].sort_values("pitcher_name")

    if pitchers_with_samples.empty:
        st.warning("No pitchers with K% samples found.")
        return

    # Pitcher selector (with team abbreviation)
    team_lookup = get_team_lookup()
    display_names = []
    display_to_id = {}
    for _, prow in pitchers_with_samples.iterrows():
        pid = int(prow["pitcher_id"])
        pname = prow["pitcher_name"]
        team = team_lookup.get(pid, "")
        dname = f"{pname} ({team})" if team else pname
        display_names.append(dname)
        display_to_id[dname] = pid

    selected_display = st.selectbox(
        "Select pitcher",
        sorted(display_names),
        key="gamek_pitcher",
    )
    pitcher_id = int(display_to_id[selected_display])
    selected_name = pitchers_with_samples[
        pitchers_with_samples["pitcher_id"] == pitcher_id
    ].iloc[0]["pitcher_name"]
    k_rate_samples = k_samples_dict[str(pitcher_id)]

    # BF parameters
    bf_info = get_bf_distribution(pitcher_id, PRIOR_SEASON, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    col1, col2 = st.columns(2)
    with col1:
        bf_mu_adj = st.slider(
            "Expected batters faced",
            min_value=10, max_value=35, value=int(round(bf_mu)),
            help="Adjust based on expected workload",
        )
    with col2:
        _k_mean = np.mean(k_rate_samples)
        _k_pct = None
        if "projected_k_rate" in pitcher_proj.columns:
            _k_pct = percentile_rank(pitcher_proj["projected_k_rate"], float(_k_mean), True)
        st.markdown(
            metric_card("Projected K%", fmt_pct(_k_mean), pctile=_k_pct),
            unsafe_allow_html=True,
        )

    # Umpire selector
    ump_lift = 0.0
    ump_path = DASHBOARD_DIR / "umpire_tendencies.parquet"
    if ump_path.exists():
        ump_df = pd.read_parquet(ump_path)
        ump_names = ["League Average"] + sorted(ump_df["hp_umpire_name"].tolist())
        selected_ump = st.selectbox(
            "HP Umpire",
            ump_names,
            key="gamek_umpire",
            help="Select the home plate umpire to adjust K-rate prediction",
        )
        if selected_ump != "League Average":
            ump_row = ump_df[ump_df["hp_umpire_name"] == selected_ump]
            if not ump_row.empty:
                ump_lift = float(ump_row.iloc[0]["k_logit_lift"])
                ump_k_rate = float(ump_row.iloc[0]["k_rate_shrunk"])
                league_k = float(ump_row.iloc[0]["league_k_rate"])
                delta_pp = (ump_k_rate - league_k) * 100
                if abs(delta_pp) > 0.3:
                    color = POSITIVE if delta_pp > 0 else NEGATIVE
                    direction = "above" if delta_pp > 0 else "below"
                    st.markdown(
                        f'<div style="color:{color}; font-size:0.85rem; margin-top:-0.5rem;">'
                        f'{selected_ump}: {ump_k_rate:.1%} K-rate ({delta_pp:+.1f}pp {direction} avg)'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Weather controls
    weather_lift = 0.0
    weather_hr_mult = 1.0
    wx_path = DASHBOARD_DIR / "weather_effects.parquet"
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        wx_col1, wx_col2, wx_col3 = st.columns(3)
        with wx_col1:
            is_dome = st.checkbox("Dome / Retractable Roof (closed)", key="gamek_dome")
        if not is_dome:
            with wx_col2:
                temp_bucket = st.selectbox(
                    "Temperature",
                    ["warm (70-84°F)", "cool (55-69°F)", "hot (85+°F)", "cold (<55°F)"],
                    key="gamek_temp",
                )
                temp_key = temp_bucket.split(" ")[0]
            with wx_col3:
                wind_cat = st.selectbox(
                    "Wind",
                    ["none", "out", "cross", "in"],
                    key="gamek_wind",
                    format_func=lambda x: {
                        "none": "Calm / None",
                        "out": "Out (to CF/LF/RF)",
                        "cross": "Cross (L to R / R to L)",
                        "in": "In (from OF)",
                    }.get(x, x),
                )
            wx_row = wx_df[
                (wx_df["temp_bucket"] == temp_key) & (wx_df["wind_category"] == wind_cat)
            ]
            if not wx_row.empty:
                k_mult = float(wx_row.iloc[0]["k_multiplier"])
                weather_hr_mult = float(wx_row.iloc[0]["hr_multiplier"])
                # Convert K multiplier to logit lift
                overall_k = float(wx_row.iloc[0]["overall_k_rate"])
                adj_k = overall_k * k_mult
                from scipy.special import logit as _logit_fn
                weather_lift = float(
                    _logit_fn(np.clip(adj_k, 1e-6, 1 - 1e-6))
                    - _logit_fn(np.clip(overall_k, 1e-6, 1 - 1e-6))
                )
                # Show weather impact
                k_delta = (k_mult - 1.0) * 100
                hr_delta = (weather_hr_mult - 1.0) * 100
                parts = []
                if abs(k_delta) > 0.3:
                    k_color = POSITIVE if k_delta > 0 else NEGATIVE
                    parts.append(f'<span style="color:{k_color}">K-rate {k_delta:+.1f}%</span>')
                if abs(hr_delta) > 1:
                    hr_color = POSITIVE if hr_delta > 0 else NEGATIVE
                    parts.append(f'<span style="color:{hr_color}">HR-rate {hr_delta:+.0f}%</span>')
                if parts:
                    st.markdown(
                        f'<div style="font-size:0.85rem; margin-top:-0.5rem;">'
                        f'Weather impact: {" | ".join(parts)}</div>',
                        unsafe_allow_html=True,
                    )

    # --- Lineup matchup adjustment ---
    from lib.matchup import score_matchup as _score_matchup
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE as _LEAGUE_AVG

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    hitter_proj = load_projections("hitter")

    lineup_lifts = None
    per_batter_details: list[dict] = []

    if not arsenal_df.empty and not vuln_df.empty:
        st.markdown("---")
        st.markdown("**Opposing Lineup**")

        lineup_mode = st.radio(
            "Lineup source",
            ["League Average (no lineup)", "Manual (9 hitters)"],
            horizontal=True,
            key="gamek_lineup_mode",
        )

        baselines_pt = {
            pt: {"whiff_rate": vals.get("whiff_rate", 0.25)}
            for pt, vals in _LEAGUE_AVG.items()
        }

        if lineup_mode == "Manual (9 hitters)":
            if not hitter_proj.empty:
                # Build hitter options with teams
                h_options = {}
                for _, hr_ in hitter_proj.iterrows():
                    hid = int(hr_["batter_id"])
                    hname = hr_["batter_name"]
                    team = team_lookup.get(hid, "")
                    dname = f"{hname} ({team})" if team else hname
                    h_options[dname] = hid
                sorted_hitters = sorted(h_options.keys())

                st.caption("Select 9 hitters in batting order:")
                manual_ids = []
                cols = st.columns(3)
                for i in range(9):
                    with cols[i % 3]:
                        sel = st.selectbox(
                            f"#{i+1}",
                            sorted_hitters,
                            key=f"gamek_manual_{i}",
                        )
                        manual_ids.append(h_options[sel])

                # Score matchups
                lifts = np.zeros(9)
                details = []
                for i, bid in enumerate(manual_ids):
                    m = _score_matchup(
                        pitcher_id, bid, arsenal_df, vuln_df, baselines_pt,
                    )
                    lift = m.get("matchup_k_logit_lift", 0.0)
                    if np.isnan(lift):
                        lift = 0.0
                    lifts[i] = lift
                    # Get name
                    bname = next(
                        (k.split(" (")[0] for k, v in h_options.items() if v == bid),
                        "Unknown",
                    )
                    m["batter_name"] = bname
                    m["batting_order"] = i + 1
                    details.append(m)
                lineup_lifts = lifts
                per_batter_details = details

    # Simulate
    game_ks = simulate_game_ks(
        pitcher_k_rate_samples=k_rate_samples,
        bf_mu=float(bf_mu_adj),
        bf_sigma=bf_sigma,
        lineup_matchup_lifts=lineup_lifts,
        umpire_k_logit_lift=ump_lift,
        weather_k_logit_lift=weather_lift,
        n_draws=10000,
        random_seed=42,
    )

    # K distribution chart
    fig = create_game_k_fig(game_ks, selected_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # P(over X.5) table
    st.markdown('<div class="section-header">K Prop Lines</div>',
                unsafe_allow_html=True)
    k_over = compute_k_over_probs(game_ks)
    k_over = k_over[(k_over["line"] >= 2.5) & (k_over["line"] <= 10.5)].copy()

    display_lines = []
    for _, row in k_over.iterrows():
        line = row["line"]
        p_over = row["p_over"]
        if p_over > 0.65:
            signal = "Strong Over"
        elif p_over > 0.55:
            signal = "Lean Over"
        elif p_over < 0.35:
            signal = "Strong Under"
        elif p_over < 0.45:
            signal = "Lean Under"
        else:
            signal = "Toss-up"
        display_lines.append({
            "Line": f"Over {line:.1f}",
            "P(Over)": f"{p_over:.1%}",
            "P(Under)": f"{1 - p_over:.1%}",
            "Edge Signal": signal,
        })

    st.dataframe(
        pd.DataFrame(display_lines),
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    st.markdown("---")
    summary_cols = st.columns(4)
    stats = [
        ("Expected K", f"{np.mean(game_ks):.1f}"),
        ("Std Dev", f"{np.std(game_ks):.1f}"),
        ("Median K", f"{np.median(game_ks):.0f}"),
        ("90th Pctile", f"{np.percentile(game_ks, 90):.0f}"),
    ]
    for col, (label, val) in zip(summary_cols, stats):
        with col:
            st.markdown(metric_card(label, val), unsafe_allow_html=True)

    # Plain English summary
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    mean_k = np.mean(game_ks)
    p_over_5 = (game_ks >= 6).sum() / len(game_ks) * 100
    p_over_7 = (game_ks >= 8).sum() / len(game_ks) * 100

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">What This Means</div>
        <div class="insight-bullet">
            <span class="dot" style="background:{GOLD};"></span>
            On an average night, expect around <strong>{mean_k:.0f} strikeouts</strong>
            (give or take {np.std(game_ks):.0f}).
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SAGE};"></span>
            There's a <strong>{p_over_5:.0f}% chance</strong> of 6+ Ks
            and a <strong>{p_over_7:.0f}% chance</strong> of 8+ Ks.
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SLATE};"></span>
            Based on {len(game_ks):,} Monte Carlo simulations using the Bayesian
            K% posterior ({TRAINING_RANGE} training data).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-batter matchup details (when lineup is active)
    if per_batter_details:
        st.markdown("---")
        st.markdown('<div class="section-header">Lineup Matchup Breakdown</div>',
                    unsafe_allow_html=True)
        lineup_rows = []
        for d in per_batter_details:
            bname = d.get("batter_name", "Unknown")
            bteam = team_lookup.get(d.get("batter_id", 0), "")
            mwhiff = d.get("matchup_whiff_rate", np.nan)
            bwhiff = d.get("baseline_whiff_rate", np.nan)
            lift = d.get("matchup_k_logit_lift", 0.0)
            rel = d.get("avg_reliability", 0.0)
            lineup_rows.append({
                "#": d.get("batting_order", ""),
                "Batter": f"{bname} ({bteam})" if bteam else bname,
                "Matchup Whiff%": f"{mwhiff:.1%}" if pd.notna(mwhiff) else "--",
                "Baseline Whiff%": f"{bwhiff:.1%}" if pd.notna(bwhiff) else "--",
                "K Lift": f"{lift:+.3f}",
                "Reliability": f"{rel:.0%}",
            })
        st.dataframe(
            pd.DataFrame(lineup_rows),
            use_container_width=True,
            hide_index=True,
        )
        avg_lift = np.mean([d.get("matchup_k_logit_lift", 0.0) for d in per_batter_details])
        if avg_lift > 0.05:
            st.markdown(f'<div style="color:{POSITIVE}; font-size:0.9rem;">'
                        f'This lineup is favorable for strikeouts (avg lift: {avg_lift:+.3f})</div>',
                        unsafe_allow_html=True)
        elif avg_lift < -0.05:
            st.markdown(f'<div style="color:{NEGATIVE}; font-size:0.9rem;">'
                        f'This lineup is unfavorable for strikeouts (avg lift: {avg_lift:+.3f})</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:{SLATE}; font-size:0.9rem;">'
                        f'This lineup is neutral for strikeouts (avg lift: {avg_lift:+.3f})</div>',
                        unsafe_allow_html=True)
