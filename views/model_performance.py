"""Model Performance page -- backtest results, game sim validation, projection movers, preseason comparison."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, EMBER, SLATE, CREAM, DASHBOARD_DIR,
    PITCHER_STATS, HITTER_STATS,
)
from components.metric_cards import metric_card
from components.backtest_charts import (
    PLOTLY_CONFIG,
    create_coverage_chart,
    create_game_k_model_comparison,
    create_movers_chart,
    create_projection_timeline,
    create_pred_vs_actual_scatter,
    create_error_histogram,
    create_rolling_mae_chart,
    create_prop_calibration_chart,
)
from services.data_loader import (
    load_backtest,
    load_projections,
    load_weekly_snapshots,
    load_pitcher_sim_log,
    load_batter_sim_log,
)
from utils.alerts import tdd_info, tdd_warn
from utils.formatters import fmt_stat, delta_html
from utils.helpers import strip_accents

from components.diamond_rating import diamond_rating_text_composite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_movers(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    id_col: str,
    name_col: str,
    stat_col: str,
    n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute top improvers and decliners between two projection snapshots."""
    merged = current.merge(
        previous[[id_col, stat_col]], on=id_col, suffixes=("", "_prev"),
    )
    merged["delta"] = merged[stat_col] - merged[f"{stat_col}_prev"]
    merged = merged.dropna(subset=["delta"])

    display_cols = [name_col, stat_col, f"{stat_col}_prev", "delta"]
    available = [c for c in display_cols if c in merged.columns]

    improvers = merged.nlargest(n, "delta")[available]
    decliners = merged.nsmallest(n, "delta")[available]
    return improvers, decliners


def _load_preseason(player_type: str) -> pd.DataFrame:
    """Load preseason snapshot from the snapshots directory."""
    from config import CURRENT_SEASON
    fname = f"{player_type}_projections_{CURRENT_SEASON}_preseason.parquet"
    path = DASHBOARD_DIR / "snapshots" / fname
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# In-Season Accuracy
# ---------------------------------------------------------------------------

_PITCHER_STAT_CONFIGS = {
    "K": ("expected_k", "actual_k"),
    "BB": ("expected_bb", "actual_bb"),
    "H": ("expected_h", "actual_h"),
    "HR": ("expected_hr", "actual_hr"),
}

_BATTER_STAT_CONFIGS = {
    "K": ("expected_k", "actual_k"),
    "H": ("expected_h", "actual_h"),
    "HR": ("expected_hr", "actual_hr"),
    "TB": ("expected_tb", "actual_tb"),
}


def _render_in_season_accuracy_tab() -> None:
    """In-season game sim accuracy: pitcher and batter predictions vs actuals."""

    player_type = st.radio(
        "Player type", ["Pitcher", "Hitter"],
        horizontal=True, key="in_season_type",
        label_visibility="collapsed",
    )

    if player_type == "Pitcher":
        df = load_pitcher_sim_log()
        stat_map = _PITCHER_STAT_CONFIGS
        name_col = "pitcher_name"
    else:
        df = load_batter_sim_log()
        stat_map = _BATTER_STAT_CONFIGS
        name_col = "batter_name"

    if df.empty:
        tdd_info(
            "No in-season prediction data yet. The accuracy log is populated "
            "daily once the season starts and games have been played."
        )
        return

    # Ensure numeric types
    for _, (pred_col, act_col) in stat_map.items():
        if pred_col in df.columns:
            df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
        if act_col in df.columns:
            df[act_col] = pd.to_numeric(df[act_col], errors="coerce")

    # Summary metric cards
    n_games = len(df)
    pred_k, act_k = df.get("expected_k"), df.get("actual_k")
    k_mae = float((pred_k - act_k).abs().mean()) if pred_k is not None and act_k is not None else 0
    k_corr = float(pred_k.corr(act_k)) if pred_k is not None and act_k is not None else 0
    k_bias = float((pred_k - act_k).mean()) if pred_k is not None and act_k is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{'Starts' if player_type == 'Pitcher' else 'Games'} Tracked",
                    str(n_games), color=GOLD)
    with c2:
        metric_card("K MAE", f"{k_mae:.2f}")
    with c3:
        metric_card("K Correlation", f"{k_corr:.3f}")
    with c4:
        metric_card("K Bias", f"{k_bias:+.2f}")

    # Stat selector
    stat_label = st.radio(
        "Stat", list(stat_map.keys()),
        horizontal=True, key="in_season_stat",
        label_visibility="collapsed",
    )
    pred_col, act_col = stat_map[stat_label]

    if pred_col not in df.columns or act_col not in df.columns:
        tdd_warn(f"No {stat_label} data available in the log.")
        return

    valid = df[[pred_col, act_col]].dropna()
    if valid.empty:
        tdd_warn(f"No valid {stat_label} predictions with actuals.")
        return

    predicted = valid[pred_col]
    actual = valid[act_col]
    has_lineup = df.loc[valid.index, "has_lineup"] if "has_lineup" in df.columns else None

    # Scatter + Error histogram
    col_l, col_r = st.columns(2)
    with col_l:
        fig = create_pred_vs_actual_scatter(predicted, actual, stat_label, has_lineup)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L173")
    with col_r:
        errors = predicted - actual
        fig = create_error_histogram(errors, stat_label)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L177")

    # K prop calibration (pitcher only)
    if player_type == "Pitcher" and stat_label == "K":
        prop_lines = [
            ("p_over_4_5", 4.5), ("p_over_5_5", 5.5),
            ("p_over_6_5", 6.5), ("p_over_7_5", 7.5),
        ]
        probs_list, overs_list, labels_list = [], [], []
        for col, line_val in prop_lines:
            if col in df.columns and "actual_k" in df.columns:
                valid_prop = df[[col, "actual_k"]].dropna()
                if len(valid_prop) >= 10:
                    probs_list.append(valid_prop[col])
                    overs_list.append((valid_prop["actual_k"] > line_val).astype(float))
                    labels_list.append(f"{line_val}")
        if probs_list:
            st.markdown(f'<div style="color:{GOLD}; font-weight:600; margin-top:1rem;">'
                        f'Prop Calibration (K)</div>', unsafe_allow_html=True)
            fig = create_prop_calibration_chart(probs_list, overs_list, labels_list)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L197")

    # Rolling MAE trend
    if "prediction_date" in df.columns:
        st.markdown(f'<div style="color:{GOLD}; font-weight:600; margin-top:1rem;">'
                    f'Accuracy Over Time</div>', unsafe_allow_html=True)
        stat_configs = [(pred_col, act_col, stat_label)]
        fig = create_rolling_mae_chart(df, "prediction_date", stat_configs, window=7)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L205")

    # Biggest hits and misses
    df_scored = df[[name_col, "prediction_date", pred_col, act_col]].dropna().copy()
    if not df_scored.empty:
        df_scored["error"] = (df_scored[pred_col] - df_scored[act_col]).abs()
        df_scored["signed_error"] = df_scored[pred_col] - df_scored[act_col]

        st.markdown(f'<div style="color:{GOLD}; font-weight:600; margin-top:1rem;">'
                    f'Biggest Hits and Misses</div>', unsafe_allow_html=True)

        hit_col, miss_col = st.columns(2)
        with hit_col:
            st.markdown(f"**Closest Predictions** ({stat_label})")
            hits = df_scored.nsmallest(10, "error")[
                [name_col, "prediction_date", pred_col, act_col, "error"]
            ].reset_index(drop=True)
            hits.columns = ["Player", "Date", "Predicted", "Actual", "Error"]
            st.dataframe(hits, use_container_width=True, hide_index=True)

        with miss_col:
            st.markdown(f"**Biggest Misses** ({stat_label})")
            misses = df_scored.nlargest(10, "error")[
                [name_col, "prediction_date", pred_col, act_col, "signed_error"]
            ].reset_index(drop=True)
            misses.columns = ["Player", "Date", "Predicted", "Actual", "Error"]
            st.dataframe(misses, use_container_width=True, hide_index=True)

    # Raw data expander
    with st.expander("Full prediction log"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    """Render the Model Performance page."""
    st.markdown(
        f'<div class="brand-header">'
        f'<div><div class="brand-title">Model Performance</div>'
        f'<div class="brand-subtitle">Backtest accuracy, model comparisons, and projection movers</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    (tab_in_season, tab_backtest, tab_game_k, tab_hits_misses, tab_movers,
     tab_preseason) = st.tabs([
        "In-Season Accuracy", "Backtest Results", "Game Sim Backtests",
        "Biggest Hits & Misses", "Projection Movers",
        "Weekly Snapshots",
    ])

    # ===================================================================
    # Tab 0: In-Season Accuracy
    # ===================================================================
    with tab_in_season:
        _render_in_season_accuracy_tab()

    # ===================================================================
    # Tab 1: Backtest Results
    # ===================================================================
    with tab_backtest:
        _render_backtest_top10_tab()

    # ===================================================================
    # Tab 2: Game Sim Backtests
    # ===================================================================
    with tab_game_k:
        _render_game_sim_backtest_tab()

    # ===================================================================
    # Tab 3: Biggest Hits & Misses
    # ===================================================================
    with tab_hits_misses:
        _render_hits_misses_tab()

    # ===================================================================
    # Tab 4: Projection Movers
    # ===================================================================
    with tab_movers:
        _render_movers_tab()

    # ===================================================================
    # Tab 5: Weekly Snapshots
    # ===================================================================
    with tab_preseason:
        _render_weekly_snapshots_tab()


# ---------------------------------------------------------------------------
# Tab 1: Backtest Top 10
# ---------------------------------------------------------------------------

def _render_backtest_top10_tab() -> None:
    """Leaderboard-style backtest results: most/least of each stat + our prediction."""
    from components.headshot import headshot_html

    bip_df = load_backtest("bip_backtest_results")
    if bip_df.empty:
        tdd_info("No player-level backtest data available.")
        return

    # Deduplicate: keep one variant per player per season (K/BB/HR are identical across variants)
    if "variant" in bip_df.columns:
        bip_df = bip_df.drop_duplicates(subset=["batter_id", "test_season"], keep="first")

    # Player type toggle
    player_type = st.radio(
        "Player type", ["Batter", "Pitcher"],
        horizontal=True, key="bt_type",
        label_visibility="collapsed",
    )

    if player_type == "Pitcher":
        tdd_info("Pitcher-level backtest leaderboards coming soon. Use the Game Sim Backtests tab for pitcher accuracy.")
        return

    # Stat selector based on player type
    # (label, pred_col, actual_col, higher_is_better_for_batter)
    batter_stats = [
        ("K", "total_k_mean", "k", False),
        ("BB", "total_bb_mean", "bb", True),
        ("HR", "total_hr_mean", "hr", True),
        ("H", "total_h_mean", "hits", True),
        ("SB", "total_sb_mean", "sb", True) if "total_sb_mean" in bip_df.columns and "sb" in bip_df.columns else None,
    ]
    batter_stats = [s for s in batter_stats if s is not None]

    stat_choice = st.selectbox(
        "Stat", [s[0] for s in batter_stats], key="bt_stat_choice",
    )
    cfg = next(s for s in batter_stats if s[0] == stat_choice)
    stat_label, pred_col, act_col, higher_is_good = cfg

    # Season filter
    seasons = sorted(bip_df["test_season"].unique(), reverse=True)
    season = st.selectbox("Season", seasons, key="bt_season")

    df = bip_df[bip_df["test_season"] == season].copy()
    if pred_col not in df.columns or act_col not in df.columns:
        tdd_info(f"No data for {stat_label}.")
        return

    df = df[["batter_id", "batter_name", pred_col, act_col]].dropna().copy()
    df["predicted"] = df[pred_col].round(1)
    df["actual"] = df[act_col].astype(int)
    df["error"] = df["predicted"] - df["actual"]

    # Summary cards
    mae = df["error"].abs().mean()
    corr = df["predicted"].corr(df["actual"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("Players", f"{len(df):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(f"{stat_label} MAE", f"{mae:.1f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Correlation", f"{corr:.3f}"), unsafe_allow_html=True)

    # Leaderboard builder using CSS classes from styles.css
    def _build_lb(subset: pd.DataFrame, title: str, is_good: bool) -> str:
        tier = "good" if is_good else "bad"
        rows = []
        for i, (_, row) in enumerate(subset.iterrows(), 1):
            pid = int(row["batter_id"])
            name = row["batter_name"]
            actual = int(row["actual"])
            pred = row["predicted"]
            err = row["error"]
            err_color = SAGE if abs(err) < mae else EMBER
            rank_class = f"lb-rank-{tier} lb-rank"

            hs = f'<span class="lb-headshot">{headshot_html(pid, size=50)}</span>'
            rows.append(
                f'<div class="lb-row">'
                f'<span class="{rank_class}">{i}.</span>'
                f'{hs}'
                f'<span class="lb-name">{name}</span>'
                f'<span class="lb-val lb-val-{tier}">{actual}</span>'
                f'<span class="lb-range" style="color:{err_color};">'
                f'pred: {pred:.0f} ({err:+.0f})</span>'
                f'</div>'
            )
        title_class = "lb-val-good" if is_good else "lb-val-bad"
        return (
            f'<div class="lb-card lb-card-wide">'
            f'<div class="lb-title-row">'
            f'<span class="lb-title {title_class}">{title}</span>'
            f'</div>'
            + "".join(rows)
            + '</div>'
        )

    # Most and Least: "good" = gold, "bad" = slate
    most = df.nlargest(10, "actual")
    least = df[df["actual"] > 0].nsmallest(10, "actual")

    # For K: most is bad (slate), fewest is good (gold)
    # For HR/H/BB/SB: most is good (gold), fewest is bad (slate)
    col_l, col_r = st.columns(2)
    with col_l:
        html = _build_lb(most, f"Most {stat_label} ({season})", is_good=higher_is_good)
        st.markdown(html, unsafe_allow_html=True)
    with col_r:
        html = _build_lb(least, f"Fewest {stat_label} ({season})", is_good=not higher_is_good)
        st.markdown(html, unsafe_allow_html=True)

    # Scatter
    from components.backtest_charts import create_pred_vs_actual_scatter
    fig = create_pred_vs_actual_scatter(df["predicted"], df["actual"], stat_label)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"bt_scatter_{stat_label}_{season}")

def _render_game_sim_backtest_tab() -> None:
    """Game sim backtest results for all projected stats individually."""
    summary = load_backtest("game_sim_backtest_summary")
    game_k = load_backtest("game_k_backtest")

    if summary.empty and game_k.empty:
        tdd_info("No game simulation backtest data available.")
        return

    # Stat selector from summary columns -- check for _mae OR _bias (schema varies)
    if not summary.empty:
        stat_names = ["K", "BB", "H", "HR", "BF", "Pitches", "IP", "Outs", "Runs"]
        stat_keys = ["k", "bb", "h", "hr", "bf", "pitches", "ip", "outs", "runs"]
        available = [
            (name, key) for name, key in zip(stat_names, stat_keys)
            if f"{key}_mae" in summary.columns or f"{key}_bias" in summary.columns
        ]
    else:
        available = [("K", "k")]

    if not available:
        available = [("K", "k")]

    stat_choice = st.radio(
        "Stat", [a[0] for a in available],
        horizontal=True, key="gsim_stat",
        label_visibility="collapsed",
    )
    stat_key = next((k for n, k in available if n == stat_choice), available[0][1])

    # Summary cards from the summary table
    if not summary.empty:
        mae_col = f"{stat_key}_mae"
        rmse_col = f"{stat_key}_rmse"
        bias_col = f"{stat_key}_bias"
        corr_col = f"{stat_key}_corr"

        avg_mae = summary[mae_col].mean() if mae_col in summary.columns else None
        avg_rmse = summary[rmse_col].mean() if rmse_col in summary.columns else None
        avg_bias = summary[bias_col].mean() if bias_col in summary.columns else None
        avg_corr = summary[corr_col].mean() if corr_col in summary.columns else None
        total_games = int(summary["n_games"].sum())

        card_cols = st.columns(5)
        with card_cols[0]:
            st.markdown(metric_card("Games Tested", f"{total_games:,}"), unsafe_allow_html=True)
        if avg_mae is not None:
            with card_cols[1]:
                st.markdown(metric_card(f"{stat_choice} MAE", f"{avg_mae:.2f}"), unsafe_allow_html=True)
        if avg_rmse is not None:
            with card_cols[2]:
                st.markdown(metric_card(f"{stat_choice} RMSE", f"{avg_rmse:.2f}"), unsafe_allow_html=True)
        if avg_corr is not None:
            with card_cols[3]:
                st.markdown(metric_card("Correlation", f"{avg_corr:.3f}"), unsafe_allow_html=True)
        if avg_bias is not None:
            with card_cols[4]:
                st.markdown(metric_card("Bias", f"{avg_bias:+.2f}"), unsafe_allow_html=True)

        # Per-season breakdown table
        display_cols = ["test_season", "n_games"]
        for prefix in [stat_key]:
            for suffix in ["mae", "rmse", "bias", "corr"]:
                c = f"{prefix}_{suffix}"
                if c in summary.columns:
                    display_cols.append(c)
        # K-specific extras
        if stat_key == "k":
            for c in ["k_avg_brier", "k_coverage_50", "k_coverage_80", "k_coverage_90"]:
                if c in summary.columns:
                    display_cols.append(c)

        st.dataframe(
            summary[display_cols].style.format(
                {c: "{:.3f}" for c in display_cols if c not in ["test_season", "n_games"]},
                na_rep="",
            ),
            use_container_width=True, hide_index=True,
        )

    # K-specific: model tier comparison and coverage from game_k_backtest
    if stat_key == "k" and not game_k.empty:
        if "full_model_avg_brier" in game_k.columns:
            st.markdown('<div class="tdd-section-hdr">Model Tier Comparison (Legacy K Backtest)</div>',
                        unsafe_allow_html=True)
            fig = create_game_k_model_comparison(game_k)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="gsim_tier")

        cov_cols = ["coverage_50", "coverage_80", "coverage_90"]
        cov_labels = ["50% CI", "80% CI", "90% CI"]
        existing = [(c, l) for c, l in zip(cov_cols, cov_labels) if c in game_k.columns]
        if existing:
            st.markdown('<div class="tdd-section-hdr">Interval Coverage (Legacy K Backtest)</div>',
                        unsafe_allow_html=True)
            fig = create_coverage_chart(game_k, [c for c, _ in existing], [l for _, l in existing])
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="gsim_cov")

    # Predicted vs actual scatter from game-level predictions
    preds = load_backtest("game_sim_backtest_predictions")
    if not preds.empty:
        pred_col = f"expected_{stat_key}"
        act_col = f"actual_{stat_key}"
        if pred_col in preds.columns and act_col in preds.columns:
            valid = preds[[pred_col, act_col]].dropna()
            if not valid.empty:
                from components.backtest_charts import create_pred_vs_actual_scatter
                st.markdown(f'<div class="tdd-section-hdr">{stat_choice}: Predicted vs Actual (Game Level)</div>',
                            unsafe_allow_html=True)
                fig = create_pred_vs_actual_scatter(valid[pred_col], valid[act_col], stat_choice)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"gsim_scatter_{stat_key}")


# ---------------------------------------------------------------------------
# Tab 3: Biggest Hits & Misses
# ---------------------------------------------------------------------------

def _render_hits_misses_tab() -> None:
    """Show players with largest prediction errors (positive and negative)."""
    st.markdown(
        f'<div style="color:{GOLD}; font-weight:600; font-size:0.85rem; '
        f'margin-bottom:0.8rem;">'
        f'2026 Season (Live) &mdash; updates every time a player plays a game'
        f'</div>',
        unsafe_allow_html=True,
    )

    player_type = st.radio(
        "Player Type", ["Pitcher", "Hitter"],
        horizontal=True, key="hm_player_type",
    )
    ptype = player_type.lower()

    id_col = "pitcher_id" if ptype == "pitcher" else "batter_id"
    name_col = "pitcher_name" if ptype == "pitcher" else "batter_name"

    current = load_projections(ptype)
    if current.empty:
        tdd_info("No projection data available.")
        return

    stat_col = st.selectbox(
        "Stat", ["projected_k_rate", "projected_bb_rate"],
        format_func=lambda x: "K%" if "k_rate" in x else "BB%",
        key="hm_stat",
    )
    observed_col = stat_col.replace("projected_", "observed_")

    if observed_col not in current.columns:
        tdd_info(
            f"No observed data for {stat_col.replace('projected_', '')} yet. "
            "This tab populates as 2026 games are played and projections update."
        )
        return

    # Compute errors: predicted - actual
    df = current[[id_col, name_col, stat_col, observed_col]].dropna(
        subset=[stat_col, observed_col]
    ).copy()

    if df.empty:
        tdd_info("No players with both projected and observed values.")
        return

    df["error"] = df[stat_col] - df[observed_col]
    df["abs_error"] = df["error"].abs()

    stat_label = "K%" if "k_rate" in stat_col else "BB%"
    n_show = 10

    # Biggest Misses (largest absolute error)
    biggest_misses = df.nlargest(n_show, "abs_error")

    # Biggest Hits (smallest absolute error, with meaningful projection)
    biggest_hits = df.nsmallest(n_show, "abs_error")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            f'<div class="tdd-section-hdr" style="color:var(--tdd-sage);">Biggest Hits ({stat_label})</div>',
            unsafe_allow_html=True,
        )
        if not biggest_hits.empty:
            fig = create_movers_chart(
                biggest_hits[name_col].tolist(),
                (biggest_hits["error"] * 100).tolist(),
                f"Closest Projections ({stat_label})",
                positive_color=SAGE, negative_color=SAGE,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L574")
            pass

            display = biggest_hits[[name_col, stat_col, observed_col, "error"]].copy()
            for c in [stat_col, observed_col, "error"]:
                display[c] = (display[c] * 100).round(1)
            display.columns = [
                "Player",
                f"Projected {stat_label}",
                f"Actual {stat_label}",
                "Error (pp)",
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)

    with chart_col2:
        st.markdown(
            f'<div class="tdd-section-hdr" style="color:var(--tdd-ember);">Biggest Misses ({stat_label})</div>',
            unsafe_allow_html=True,
        )
        if not biggest_misses.empty:
            fig = create_movers_chart(
                biggest_misses[name_col].tolist(),
                (biggest_misses["error"] * 100).tolist(),
                f"Largest Errors ({stat_label})",
                positive_color=EMBER, negative_color=EMBER,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L600")
            pass

            display = biggest_misses[[name_col, stat_col, observed_col, "error"]].copy()
            for c in [stat_col, observed_col, "error"]:
                display[c] = (display[c] * 100).round(1)
            display.columns = [
                "Player",
                f"Projected {stat_label}",
                f"Actual {stat_label}",
                "Error (pp)",
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)

    # Summary stats
    cols = st.columns(3)
    with cols[0]:
        st.markdown(metric_card(
            "Mean Abs Error", f"{df['abs_error'].mean() * 100:.1f}pp",
        ), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card(
            "Median Abs Error", f"{df['abs_error'].median() * 100:.1f}pp",
        ), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card(
            "Players Evaluated", f"{len(df):,}",
        ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4: Projection Movers
# ---------------------------------------------------------------------------

def _render_movers_tab() -> None:
    """Projection movers: biggest changes vs preseason or weekly snapshots."""
    player_type = st.radio(
        "Player Type", ["Pitcher", "Hitter"],
        horizontal=True, key="movers_player_type",
    )
    ptype = player_type.lower()

    id_col = "pitcher_id" if ptype == "pitcher" else "batter_id"
    name_col = "pitcher_name" if ptype == "pitcher" else "batter_name"

    current = load_projections(ptype)
    preseason = _load_preseason(ptype)
    weekly_snaps = load_weekly_snapshots(ptype)

    # Build comparison options
    options = []
    if not preseason.empty:
        options.append("vs Preseason")
    for date_str in sorted(weekly_snaps.keys(), reverse=True):
        options.append(f"vs {date_str}")

    if not options:
        tdd_info(
            "No comparison snapshots available yet. Preseason snapshots and "
            "weekly snapshots will appear here as the season progresses."
        )
        return

    if current.empty:
        tdd_warn("Current projections not loaded.")
        return

    comparison = st.selectbox("Compare to", options, key="movers_comparison")

    stat_col = st.selectbox(
        "Stat", ["projected_k_rate", "projected_bb_rate"],
        format_func=lambda x: "K%" if "k_rate" in x else "BB%",
        key="movers_stat",
    )

    # Resolve previous snapshot
    if comparison == "vs Preseason":
        previous = preseason
    else:
        date_str = comparison.replace("vs ", "")
        previous = weekly_snaps.get(date_str, pd.DataFrame())

    if previous.empty:
        tdd_warn("Selected snapshot is empty.")
        return

    # Compute movers
    improvers, decliners = _compute_movers(
        current, previous, id_col, name_col, stat_col,
    )

    # For pitchers K%: higher = improver (more Ks = good)
    # For hitters K%: lower = improver (fewer Ks = good)
    stat_label = "K%" if "k_rate" in stat_col else "BB%"
    is_higher_better = (ptype == "pitcher" and "k_rate" in stat_col) or \
                       (ptype == "hitter" and "bb_rate" in stat_col)

    if is_higher_better:
        up_label, down_label = "Top Improvers", "Top Decliners"
        up_df, down_df = improvers, decliners
    else:
        # Invert: for hitter K%, decliners (lower K%) are "improvers"
        up_label, down_label = "Top Improvers", "Top Decliners"
        up_df, down_df = decliners, improvers

    # Visual movers charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if not up_df.empty and name_col in up_df.columns and "delta" in up_df.columns:
            fig = create_movers_chart(
                up_df[name_col].tolist(),
                (up_df["delta"] * 100).tolist(),
                f"{up_label} ({stat_label})",
                positive_color=SAGE, negative_color=SAGE,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L715")
            pass

    with chart_col2:
        if not down_df.empty and name_col in down_df.columns and "delta" in down_df.columns:
            fig = create_movers_chart(
                down_df[name_col].tolist(),
                (down_df["delta"] * 100).tolist(),
                f"{down_label} ({stat_label})",
                positive_color=EMBER, negative_color=EMBER,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L726")
            pass

    # Data tables
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="tdd-section-hdr" style="color:var(--tdd-sage);">{up_label} ({stat_label})</div>',
            unsafe_allow_html=True,
        )
        if not up_df.empty:
            display = up_df.copy()
            for c in [stat_col, f"{stat_col}_prev", "delta"]:
                if c in display.columns:
                    display[c] = (display[c] * 100).round(1)
            display.columns = [c.replace("projected_", "").replace("_prev", " (prev)")
                               .replace("delta", "Change").replace("_", " ").title()
                               for c in display.columns]
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.caption("No data")

    with col2:
        st.markdown(
            f'<div class="tdd-section-hdr" style="color:var(--tdd-ember);">{down_label} ({stat_label})</div>',
            unsafe_allow_html=True,
        )
        if not down_df.empty:
            display = down_df.copy()
            for c in [stat_col, f"{stat_col}_prev", "delta"]:
                if c in display.columns:
                    display[c] = (display[c] * 100).round(1)
            display.columns = [c.replace("projected_", "").replace("_prev", " (prev)")
                               .replace("delta", "Change").replace("_", " ").title()
                               for c in display.columns]
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.caption("No data")

    # Player search with timeline
    st.markdown('<div class="tdd-section-hdr">Player Timeline</div>',
                unsafe_allow_html=True)

    all_snapshots = {}
    if not preseason.empty:
        all_snapshots["Preseason"] = preseason
    all_snapshots.update(weekly_snaps)
    # Add current as latest
    all_snapshots["Current"] = current

    if len(all_snapshots) < 2:
        st.caption("Need at least 2 snapshots to show a timeline.")
        return

    # Player search
    player_options = current[[id_col, name_col]].drop_duplicates()
    if player_options.empty:
        return

    search = st.text_input("Search player", key="movers_search")
    if search:
        mask = player_options[name_col].str.contains(search, case=False, na=False)
        player_options = player_options[mask]

    if player_options.empty:
        st.caption("No matching players.")
        return

    selected_name = st.selectbox(
        "Player", player_options[name_col].tolist(), key="movers_player",
    )
    selected_id = int(
        player_options[player_options[name_col] == selected_name][id_col].iloc[0]
    )

    fig = create_projection_timeline(
        all_snapshots, selected_id, id_col, name_col,
        [stat_col],
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L806")
    else:
        st.caption("Player not found in snapshots.")


# ---------------------------------------------------------------------------
# Tab 5: Preseason Comparison
# ---------------------------------------------------------------------------

def _build_comparison_df(
    preseason: pd.DataFrame,
    current: pd.DataFrame,
    id_col: str,
    name_col: str,
    stat_configs: list,
) -> pd.DataFrame:
    """Merge preseason and current projections, compute deltas."""
    stat_cols = [f"projected_{cfg[1]}" for cfg in stat_configs]
    pre_cols = [id_col, name_col] + [c for c in stat_cols if c in preseason.columns]
    cur_cols = [id_col] + [c for c in stat_cols if c in current.columns]

    if "composite_score" in current.columns:
        cur_cols.append("composite_score")

    merged = preseason[pre_cols].merge(
        current[cur_cols], on=id_col, suffixes=("_pre", ""),
    )

    for cfg in stat_configs:
        col = f"projected_{cfg[1]}"
        pre_col = f"{col}_pre"
        if col in merged.columns and pre_col in merged.columns:
            merged[f"delta_{cfg[1]}"] = merged[col] - merged[pre_col]

    return merged


def _render_weekly_snapshots_tab() -> None:
    """Weekly snapshot comparison with week-over-week dropdown."""
    player_type = st.radio(
        "Player type", ["Pitcher", "Hitter"],
        horizontal=True, key="weekly_type",
    )
    ptype = player_type.lower()

    if player_type == "Hitter":
        name_col, id_col = "batter_name", "batter_id"
        stat_configs = HITTER_STATS
    else:
        name_col, id_col = "pitcher_name", "pitcher_id"
        stat_configs = PITCHER_STATS

    preseason = _load_preseason(ptype)
    current = load_projections(ptype)
    weekly = load_weekly_snapshots(ptype)

    # Build ordered snapshot list: Preseason, Week 1, Week 2, ...
    all_snapshots: dict[str, pd.DataFrame] = {}
    if not preseason.empty:
        all_snapshots["Preseason"] = preseason
    for i, (date_str, df) in enumerate(sorted(weekly.items()), start=1):
        all_snapshots[f"Week {i} ({date_str})"] = df

    if not all_snapshots:
        tdd_info("No snapshots available yet. Preseason and weekly snapshots will appear here as the season progresses.")
        return

    # Snapshot selector
    snap_labels = list(all_snapshots.keys())
    default_idx = max(0, len(snap_labels) - 2) if len(snap_labels) >= 2 else 0

    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        from_snap = st.selectbox(
            "From", snap_labels, index=default_idx, key="weekly_from",
        )
    with compare_col2:
        to_options = snap_labels
        to_default = len(snap_labels) - 1
        to_snap = st.selectbox(
            "To", to_options, index=to_default, key="weekly_to",
        )

    prev_df = all_snapshots[from_snap]
    curr_df = all_snapshots[to_snap] if to_snap != "Current" else current
    # If "Current" projections exist and aren't already a snapshot, add as option
    if not current.empty and to_snap not in all_snapshots:
        curr_df = current

    if prev_df.empty or curr_df.empty:
        tdd_warn("Selected snapshots are empty.")
        return

    # Sub-tabs: Comparison Table | Biggest Movers | Player Lookup
    sub_overview, sub_movers, sub_lookup = st.tabs([
        "Comparison Table", "Biggest Movers", "Player Lookup",
    ])

    # --- Comparison Table ---
    with sub_overview:
        merged = _build_comparison_df(prev_df, curr_df, id_col, name_col, stat_configs)

        if merged.empty:
            tdd_warn("No matching players between selected snapshots.")
            return

        st.markdown(
            f'<div class="insight-card">'
            f'<span class="tdd-meta">Comparing </span>'
            f'<span class="tdd-stat-value">{from_snap}</span>'
            f'<span class="tdd-meta"> to </span>'
            f'<span class="tdd-stat-value">{to_snap}</span>'
            f'<span class="tdd-meta"> | {len(merged)} {ptype}s matched</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        search = st.text_input("Search player", "", placeholder="Type a name...",
                               key="preseason_overview_search")
        if search:
            _norm = strip_accents(search)
            merged = merged[
                merged[name_col].apply(lambda x: _norm.lower() in strip_accents(str(x)).lower())
            ]

        display_rows = []
        for _, row in merged.iterrows():
            r: dict[str, object] = {"Name": row[name_col]}
            if "composite_score" in row.index and pd.notna(row.get("composite_score")):
                r["Rating"] = diamond_rating_text_composite(row["composite_score"])
            for label, key, higher_better, _ in stat_configs:
                proj_col = f"projected_{key}"
                pre_col = f"{proj_col}_pre"
                delta_col = f"delta_{key}"
                if pre_col in row.index and pd.notna(row.get(pre_col)):
                    r[f"Pre {label}"] = fmt_stat(row[pre_col], key)
                else:
                    r[f"Pre {label}"] = ""
                if proj_col in row.index and pd.notna(row.get(proj_col)):
                    r[f"Now {label}"] = fmt_stat(row[proj_col], key)
                else:
                    r[f"Now {label}"] = ""
                if delta_col in row.index and pd.notna(row.get(delta_col)):
                    d = row[delta_col] * 100
                    r[f"{label} Chg"] = f"{d:+.1f}pp"
                else:
                    r[f"{label} Chg"] = ""
            display_rows.append(r)

        display_df = pd.DataFrame(display_rows)
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)

    # --- Biggest Movers ---
    with sub_movers:
        stat_choice = st.radio(
            "Stat", [cfg[0] for cfg in stat_configs],
            horizontal=True, key="weekly_movers_stat",
        )
        stat_cfg = next(cfg for cfg in stat_configs if cfg[0] == stat_choice)
        stat_key = stat_cfg[1]
        higher_better = stat_cfg[2]

        merged_full = _build_comparison_df(prev_df, curr_df, id_col, name_col, stat_configs)
        delta_col = f"delta_{stat_key}"

        if delta_col not in merged_full.columns or merged_full[delta_col].isna().all():
            tdd_info(f"No delta data available for {stat_choice}.")
        else:
            merged_sorted = merged_full.dropna(subset=[delta_col])
            n_show = min(10, len(merged_sorted) // 2)
            if n_show < 3:
                n_show = min(3, len(merged_sorted))

            merged_sorted["delta_pp"] = merged_sorted[delta_col] * 100

            if higher_better:
                improvers = merged_sorted.nlargest(n_show, "delta_pp")
                decliners = merged_sorted.nsmallest(n_show, "delta_pp")
            else:
                improvers = merged_sorted.nsmallest(n_show, "delta_pp")
                decliners = merged_sorted.nlargest(n_show, "delta_pp")

            col1, col2 = st.columns(2)
            with col1:
                if not improvers.empty:
                    fig = create_movers_chart(
                        improvers[name_col].tolist(),
                        improvers["delta_pp"].tolist(),
                        f"Top {stat_choice} Improvers",
                        positive_color=SAGE,
                        negative_color=SAGE,
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L983")
                    pass

            with col2:
                if not decliners.empty:
                    fig = create_movers_chart(
                        decliners[name_col].tolist(),
                        decliners["delta_pp"].tolist(),
                        f"Top {stat_choice} Decliners",
                        positive_color=EMBER,
                        negative_color=EMBER,
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="mp_L995")
                    pass

    # --- Player Lookup with sparkline timeline ---
    with sub_lookup:
        st.markdown(
            f'<div class="insight-card">'
            f'<span class="tdd-meta">Search for a player to see their projection '
            f'evolution from preseason through weekly snapshots.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Use the "to" snapshot for names since it's the most recent
        _names_df = curr_df if not curr_df.empty else prev_df
        all_names = _names_df[[id_col, name_col]].drop_duplicates()
        search_lu = st.text_input("Search player", key="weekly_lu_search")
        if search_lu:
            _norm = strip_accents(search_lu)
            all_names = all_names[
                all_names[name_col].apply(lambda x: _norm.lower() in strip_accents(str(x)).lower())
            ]

        if all_names.empty:
            st.caption("No matching players.")
            return

        selected_name = st.selectbox(
            "Player", all_names[name_col].tolist(), key="weekly_lu_player",
        )
        selected_id = int(
            all_names[all_names[name_col] == selected_name][id_col].iloc[0]
        )

        snapshots_ordered = dict(all_snapshots)  # reuse the ordered snapshot dict

        if len(snapshots_ordered) < 2:
            st.caption("Need at least 2 snapshots to show a timeline.")
            return

        for stat_label, stat_key, higher_better, _ in stat_configs:
            proj_col = f"projected_{stat_key}"
            values = []
            dates = []
            for label, snap_df in snapshots_ordered.items():
                row = snap_df[snap_df[id_col] == selected_id]
                if not row.empty and proj_col in row.columns:
                    val = row.iloc[0][proj_col]
                    if pd.notna(val):
                        values.append(float(val))
                        dates.append(label)

            if len(values) < 2:
                continue

            delta = values[-1] - values[0]
            if higher_better:
                color = SAGE if delta > 0 else EMBER
            else:
                color = SAGE if delta < 0 else EMBER

            m_col, spark_col = st.columns([1, 2])
            with m_col:
                cur_val = values[-1] * 100
                pre_val = values[0] * 100
                delta_pp = (values[-1] - values[0]) * 100
                st.markdown(
                    metric_card(
                        f"{stat_label}",
                        f"{cur_val:.1f}%",
                        delta_html(values[-1] - values[0], higher_is_better=higher_better),
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(f"Pre: {pre_val:.1f}% | Now: {cur_val:.1f}% | {delta_pp:+.1f}pp")

            with spark_col:
                import plotly.graph_objects as go
                from components.backtest_charts import _hex_to_rgba

                plot_vals = [v * 100 for v in values]
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=list(dates), y=plot_vals,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                    showlegend=False,
                ))

                # CI bands if available
                lo_col = f"{proj_col}_2_5"
                hi_col = f"{proj_col}_97_5"
                lo_vals, hi_vals = [], []
                for label_d, snap_df in snapshots_ordered.items():
                    if label_d not in dates:
                        continue
                    row = snap_df[snap_df[id_col] == selected_id]
                    if not row.empty and lo_col in row.columns and hi_col in row.columns:
                        lo_v = row.iloc[0].get(lo_col)
                        hi_v = row.iloc[0].get(hi_col)
                        if pd.notna(lo_v) and pd.notna(hi_v):
                            lo_vals.append(float(lo_v) * 100)
                            hi_vals.append(float(hi_v) * 100)
                if len(lo_vals) == len(values):
                    fig.add_trace(go.Scatter(
                        x=list(dates) + list(dates)[::-1],
                        y=hi_vals + lo_vals[::-1],
                        fill="toself",
                        fillcolor=_hex_to_rgba(color, 0.12),
                        line=dict(width=0),
                        hoverinfo="skip", showlegend=False,
                    ))

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=250,
                    showlegend=False,
                    dragmode=False,
                    title=dict(
                        text=f"{selected_name} | {stat_label} Evolution",
                        font=dict(color=CREAM, size=12), x=0.5, xanchor="center",
                    ),
                    xaxis=dict(
                        tickfont=dict(color=SLATE, size=8), tickangle=-30,
                        showgrid=False, zeroline=False, showline=False,
                    ),
                    yaxis=dict(
                        tickfont=dict(color=SLATE, size=8),
                        showgrid=True, gridcolor="rgba(123,143,166,0.12)",
                        zeroline=False, showline=False,
                        title=dict(text=f"{stat_label} (%)", font=dict(color=SLATE, size=9)),
                    ),
                    font=dict(color=CREAM),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"mp_spark_{stat_label}")


