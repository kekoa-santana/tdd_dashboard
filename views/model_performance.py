"""Model Performance page -- backtest results, game K model, projection movers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, EMBER, SLATE, CREAM,
    DARK_CARD, DARK_BORDER,
    DASHBOARD_DIR,
)
from components.metric_cards import metric_card
from components.backtest_charts import (
    create_accuracy_bars,
    create_coverage_chart,
    create_game_k_model_comparison,
    create_movers_chart,
    create_projection_timeline,
)
from services.data_loader import (
    load_backtest,
    load_projections,
    load_weekly_snapshots,
)


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

    tab_backtest, tab_game_k, tab_movers = st.tabs([
        "Backtest Results", "Game K Model", "Projection Movers",
    ])

    # ===================================================================
    # Tab 1: Backtest Results
    # ===================================================================
    with tab_backtest:
        player_type = st.radio(
            "Player Type", ["Pitcher", "Hitter"],
            horizontal=True, key="bt_player_type",
        )
        ptype = player_type.lower()

        stat_category = st.radio(
            "Stat Category", ["Rate Stats (K%, BB%)", "Counting Stats", "Game K Props"],
            horizontal=True, key="bt_stat_cat",
        )

        if stat_category == "Rate Stats (K%, BB%)":
            _render_rate_backtest(ptype)
        elif stat_category == "Counting Stats":
            _render_counting_backtest(ptype)
        else:
            _render_game_k_tab_inline()

    # ===================================================================
    # Tab 2: Game K Model
    # ===================================================================
    with tab_game_k:
        _render_game_k_tab()

    # ===================================================================
    # Tab 3: Projection Movers
    # ===================================================================
    with tab_movers:
        _render_movers_tab()


# ---------------------------------------------------------------------------
# Tab 1 renderers
# ---------------------------------------------------------------------------

def _render_rate_backtest(ptype: str) -> None:
    """Rate stat backtest: K% and BB%."""
    # Try multi-stat first, fall back to single-stat
    df_multi = load_backtest(f"{ptype}_multi_stat_backtest")
    df_single = load_backtest(f"{ptype}_k_backtest")

    if df_multi.empty and df_single.empty:
        st.info(f"No rate stat backtest data found for {ptype}s.")
        return

    # Use multi-stat if available (has both k_rate and bb_rate)
    if not df_multi.empty:
        for stat_name in ["k_rate", "bb_rate"]:
            df_stat = df_multi[df_multi["stat"] == stat_name]
            if df_stat.empty:
                continue

            label = "K%" if stat_name == "k_rate" else "BB%"
            st.markdown(f'<div class="section-header">{label} Backtest</div>',
                        unsafe_allow_html=True)
            _render_backtest_summary(df_stat, label)
    elif not df_single.empty:
        st.markdown('<div class="section-header">K% Backtest</div>',
                    unsafe_allow_html=True)
        _render_backtest_summary(df_single, "K%")


def _render_counting_backtest(ptype: str) -> None:
    """Counting stat backtest."""
    df = load_backtest(f"{ptype}_counting_backtest")
    if df.empty:
        st.info(f"No counting stat backtest data found for {ptype}s.")
        return

    stats_available = df["stat"].unique().tolist()
    for stat_name in stats_available:
        df_stat = df[df["stat"] == stat_name]
        label = stat_name.replace("total_", "").upper()
        st.markdown(f'<div class="section-header">{label} Counting Backtest</div>',
                    unsafe_allow_html=True)
        _render_backtest_summary(df_stat, label, is_counting=True)


def _render_backtest_summary(
    df: pd.DataFrame, label: str, is_counting: bool = False,
) -> None:
    """Render metric cards, comparison table, and charts for a backtest df."""
    # Summary metric cards
    avg_mae_imp = df["mae_improvement_pct"].mean()
    avg_rmse_imp = df["rmse_improvement_pct"].mean()
    avg_coverage = df["coverage_95"].mean() * 100 if "coverage_95" in df.columns else None
    total_n = int(df["n_players"].sum()) if "n_players" in df.columns else 0

    cols = st.columns(4 if avg_coverage is not None else 3)
    with cols[0]:
        color = SAGE if avg_mae_imp > 0 else EMBER
        sign = "+" if avg_mae_imp > 0 else ""
        st.markdown(metric_card(
            f"MAE Improvement", f"{sign}{avg_mae_imp:.1f}%",
        ), unsafe_allow_html=True)
    with cols[1]:
        color = SAGE if avg_rmse_imp > 0 else EMBER
        sign = "+" if avg_rmse_imp > 0 else ""
        st.markdown(metric_card(
            f"RMSE Improvement", f"{sign}{avg_rmse_imp:.1f}%",
        ), unsafe_allow_html=True)
    if avg_coverage is not None:
        with cols[2]:
            st.markdown(metric_card(
                "95% Coverage", f"{avg_coverage:.1f}%",
            ), unsafe_allow_html=True)
    with cols[-1]:
        st.markdown(metric_card(
            "Sample Size", f"{total_n:,}",
        ), unsafe_allow_html=True)

    # Comparison table
    display_cols = ["test_season", "bayes_mae", "marcel_mae", "mae_improvement_pct",
                    "bayes_rmse", "marcel_rmse", "rmse_improvement_pct"]
    if "coverage_95" in df.columns:
        display_cols.append("coverage_95")
    if "n_players" in df.columns:
        display_cols.append("n_players")
    if is_counting and "coverage_80" in df.columns:
        display_cols.insert(-1, "coverage_80")

    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].style.format({
            "bayes_mae": "{:.4f}",
            "marcel_mae": "{:.4f}",
            "mae_improvement_pct": "{:+.1f}%",
            "bayes_rmse": "{:.4f}",
            "marcel_rmse": "{:.4f}",
            "rmse_improvement_pct": "{:+.1f}%",
            "coverage_95": "{:.1%}",
            "coverage_80": "{:.1%}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = create_accuracy_bars(df, "bayes_mae", "marcel_mae", f"{label} MAE by Season")
        st.pyplot(fig, use_container_width=True)
    with col2:
        cov_cols, cov_labels = [], []
        if "coverage_95" in df.columns:
            cov_cols.append("coverage_95")
            cov_labels.append("95% CI")
        if "coverage_80" in df.columns:
            cov_cols.append("coverage_80")
            cov_labels.append("80% CI")
        if cov_cols:
            fig = create_coverage_chart(df, cov_cols, cov_labels)
            st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Game K Model
# ---------------------------------------------------------------------------

def _render_game_k_tab_inline() -> None:
    """Inline game K backtest for Tab 1 stat category selector."""
    _render_game_k_tab()


def _render_game_k_tab() -> None:
    """Dedicated Game K model performance section."""
    df = load_backtest("game_k_backtest")
    if df.empty:
        st.info("No game K backtest data available.")
        return

    # Summary cards
    total_games = int(df["n_games"].sum())
    best_brier = df["full_model_avg_brier"].min() if "full_model_avg_brier" in df.columns else df["avg_brier"].min()
    avg_log = df["log_score"].mean() if "log_score" in df.columns else None

    cols = st.columns(4 if avg_log is not None else 3)
    with cols[0]:
        st.markdown(metric_card("Games Tested", f"{total_games:,}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Best Brier", f"{best_brier:.4f}"), unsafe_allow_html=True)
    if avg_log is not None:
        with cols[2]:
            st.markdown(metric_card("Avg Log Score", f"{avg_log:.3f}"), unsafe_allow_html=True)

    # Coverage cards
    cov_cols = ["coverage_50", "coverage_80", "coverage_90"]
    cov_labels = ["50% CI", "80% CI", "90% CI"]
    existing = [(c, l) for c, l in zip(cov_cols, cov_labels) if c in df.columns]
    if existing:
        cov_c = st.columns(len(existing) + 1)
        for i, (col, lbl) in enumerate(existing):
            with cov_c[i]:
                avg = df[col].mean() * 100
                st.markdown(metric_card(f"{lbl} Coverage", f"{avg:.1f}%"), unsafe_allow_html=True)

    # Model tier comparison chart
    st.markdown('<div class="section-header">Model Tier Comparison</div>',
                unsafe_allow_html=True)
    fig = create_game_k_model_comparison(df)
    st.pyplot(fig, use_container_width=True)

    # Coverage chart
    if existing:
        st.markdown('<div class="section-header">Interval Coverage</div>',
                    unsafe_allow_html=True)
        fig = create_coverage_chart(
            df,
            [c for c, _ in existing],
            [l for _, l in existing],
        )
        st.pyplot(fig, use_container_width=True)

    # Raw data
    with st.expander("Raw Backtest Data"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Projection Movers
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
        st.info(
            "No comparison snapshots available yet. Preseason snapshots and "
            "weekly snapshots will appear here as the season progresses."
        )
        return

    if current.empty:
        st.warning("Current projections not loaded.")
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
        st.warning("Selected snapshot is empty.")
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
    import matplotlib.pyplot as plt

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if not up_df.empty and name_col in up_df.columns and "delta" in up_df.columns:
            fig = create_movers_chart(
                up_df[name_col].tolist(),
                (up_df["delta"] * 100).tolist(),
                f"{up_label} ({stat_label})",
                positive_color=SAGE, negative_color=SAGE,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with chart_col2:
        if not down_df.empty and name_col in down_df.columns and "delta" in down_df.columns:
            fig = create_movers_chart(
                down_df[name_col].tolist(),
                (down_df["delta"] * 100).tolist(),
                f"{down_label} ({stat_label})",
                positive_color=EMBER, negative_color=EMBER,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Data tables
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="section-header" style="color:{SAGE};">{up_label} ({stat_label})</div>',
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
            f'<div class="section-header" style="color:{EMBER};">{down_label} ({stat_label})</div>',
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
    st.markdown('<div class="section-header">Player Timeline</div>',
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
        st.pyplot(fig, use_container_width=True)
    else:
        st.caption("Player not found in snapshots.")
