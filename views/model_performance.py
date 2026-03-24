"""Model Performance page -- backtest results, game K model, projection movers, preseason comparison."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, EMBER, SLATE, CREAM, DARK,
    DARK_CARD, DARK_BORDER,
    DASHBOARD_DIR,
    PITCHER_STATS, HITTER_STATS,
    PRIOR_SEASON, CURRENT_SEASON,
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
from utils.formatters import fmt_stat, delta_html
from utils.helpers import strip_accents
from lib.theme import add_watermark
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

    (tab_backtest, tab_game_k, tab_hits_misses, tab_movers,
     tab_preseason, tab_season_tracker) = st.tabs([
        "Backtest Results", "Game K Model", "Biggest Hits & Misses",
        "Projection Movers", "Preseason Comparison", "Season Tracker",
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
    # Tab 5: Preseason Comparison
    # ===================================================================
    with tab_preseason:
        _render_preseason_comparison_tab()

    # ===================================================================
    # Tab 6: Season Tracker
    # ===================================================================
    with tab_season_tracker:
        _render_season_tracker_tab()


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
            st.markdown(f'<div class="tdd-section-hdr">{label} Backtest</div>',
                        unsafe_allow_html=True)
            _render_backtest_summary(df_stat, label)
    elif not df_single.empty:
        st.markdown('<div class="tdd-section-hdr">K% Backtest</div>',
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
        st.markdown(f'<div class="tdd-section-hdr">{label} Counting Backtest</div>',
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
        width='stretch',
        hide_index=True,
    )

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = create_accuracy_bars(df, "bayes_mae", "marcel_mae", f"{label} MAE by Season")
        st.pyplot(fig, width='stretch')
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
            st.pyplot(fig, width='stretch')


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
    st.markdown('<div class="tdd-section-hdr">Model Tier Comparison</div>',
                unsafe_allow_html=True)
    fig = create_game_k_model_comparison(df)
    st.pyplot(fig, width='stretch')

    # Coverage chart
    if existing:
        st.markdown('<div class="tdd-section-hdr">Interval Coverage</div>',
                    unsafe_allow_html=True)
        fig = create_coverage_chart(
            df,
            [c for c, _ in existing],
            [l for _, l in existing],
        )
        st.pyplot(fig, width='stretch')

    # Raw data
    with st.expander("Raw Backtest Data"):
        st.dataframe(df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Biggest Hits & Misses
# ---------------------------------------------------------------------------

def _render_hits_misses_tab() -> None:
    """Show players with largest prediction errors (positive and negative)."""
    player_type = st.radio(
        "Player Type", ["Pitcher", "Hitter"],
        horizontal=True, key="hm_player_type",
    )
    ptype = player_type.lower()

    id_col = "pitcher_id" if ptype == "pitcher" else "batter_id"
    name_col = "pitcher_name" if ptype == "pitcher" else "batter_name"

    current = load_projections(ptype)
    if current.empty:
        st.info("No projection data available.")
        return

    stat_col = st.selectbox(
        "Stat", ["projected_k_rate", "projected_bb_rate"],
        format_func=lambda x: "K%" if "k_rate" in x else "BB%",
        key="hm_stat",
    )
    observed_col = stat_col.replace("projected_", "observed_")

    if observed_col not in current.columns:
        st.info(
            f"No observed data for {stat_col.replace('projected_', '')}. "
            "Observed stats appear once the season starts."
        )
        return

    # Compute errors: predicted - actual
    df = current[[id_col, name_col, stat_col, observed_col]].dropna(
        subset=[stat_col, observed_col]
    ).copy()

    if df.empty:
        st.info("No players with both projected and observed values.")
        return

    df["error"] = df[stat_col] - df[observed_col]
    df["abs_error"] = df["error"].abs()

    stat_label = "K%" if "k_rate" in stat_col else "BB%"
    n_show = 10

    # Biggest Misses (largest absolute error)
    biggest_misses = df.nlargest(n_show, "abs_error")

    # Biggest Hits (smallest absolute error, with meaningful projection)
    biggest_hits = df.nsmallest(n_show, "abs_error")

    import matplotlib.pyplot as plt

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
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            display = biggest_hits[[name_col, stat_col, observed_col, "error"]].copy()
            for c in [stat_col, observed_col, "error"]:
                display[c] = (display[c] * 100).round(1)
            display.columns = [
                "Player",
                f"Projected {stat_label}",
                f"Actual {stat_label}",
                "Error (pp)",
            ]
            st.dataframe(display, width='stretch', hide_index=True)

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
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            display = biggest_misses[[name_col, stat_col, observed_col, "error"]].copy()
            for c in [stat_col, observed_col, "error"]:
                display[c] = (display[c] * 100).round(1)
            display.columns = [
                "Player",
                f"Projected {stat_label}",
                f"Actual {stat_label}",
                "Error (pp)",
            ]
            st.dataframe(display, width='stretch', hide_index=True)

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
            st.pyplot(fig, width='stretch')
            plt.close(fig)

    with chart_col2:
        if not down_df.empty and name_col in down_df.columns and "delta" in down_df.columns:
            fig = create_movers_chart(
                down_df[name_col].tolist(),
                (down_df["delta"] * 100).tolist(),
                f"{down_label} ({stat_label})",
                positive_color=EMBER, negative_color=EMBER,
            )
            st.pyplot(fig, width='stretch')
            plt.close(fig)

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
            st.dataframe(display, width='stretch', hide_index=True)
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
            st.dataframe(display, width='stretch', hide_index=True)
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
        st.pyplot(fig, width='stretch')
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


def _render_preseason_comparison_tab() -> None:
    """Compare preseason projections to current with comparison table, movers, and sparklines."""
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    if not snapshot_dir.exists():
        st.warning("No preseason snapshots found. Run precompute first.")
        return

    h_snaps = sorted(snapshot_dir.glob("hitter_projections_*_preseason.parquet"))
    p_snaps = sorted(snapshot_dir.glob("pitcher_projections_*_preseason.parquet"))

    if not h_snaps and not p_snaps:
        st.warning("No preseason snapshots found. Run precompute first.")
        return

    player_type = st.radio(
        "Player type", ["Pitcher", "Hitter"],
        horizontal=True, key="preseason_type",
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

    if preseason.empty:
        st.info("No preseason snapshot available for this player type yet.")
        return

    if current.empty:
        _render_preseason_only_table(preseason, name_col, id_col, stat_configs)
        return

    # Sub-tabs: Comparison Table | Biggest Movers | Player Lookup
    sub_overview, sub_movers, sub_lookup = st.tabs([
        "Comparison Table", "Biggest Movers", "Player Lookup",
    ])

    # --- Comparison Table ---
    with sub_overview:
        merged = _build_comparison_df(preseason, current, id_col, name_col, stat_configs)

        if merged.empty:
            st.warning("No matching players between preseason and current projections.")
            return

        snap_date = preseason["snapshot_date"].iloc[0] if "snapshot_date" in preseason.columns else "Preseason"

        st.markdown(
            f'<div class="insight-card">'
            f'<span class="tdd-meta">Comparing </span>'
            f'<span class="tdd-stat-value">preseason ({snap_date})</span>'
            f'<span class="tdd-meta"> to </span>'
            f'<span class="tdd-stat-value">current projections</span>'
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
                    r[f"Pre {label}"] = "--"
                if proj_col in row.index and pd.notna(row.get(proj_col)):
                    r[f"Now {label}"] = fmt_stat(row[proj_col], key)
                else:
                    r[f"Now {label}"] = "--"
                if delta_col in row.index and pd.notna(row.get(delta_col)):
                    d = row[delta_col] * 100
                    r[f"{label} Chg"] = f"{d:+.1f}pp"
                else:
                    r[f"{label} Chg"] = "--"
            display_rows.append(r)

        display_df = pd.DataFrame(display_rows)
        st.dataframe(display_df, width='stretch', hide_index=True, height=600)

    # --- Biggest Movers vs Preseason ---
    with sub_movers:
        stat_choice = st.radio(
            "Stat", [cfg[0] for cfg in stat_configs],
            horizontal=True, key="preseason_movers_stat",
        )
        stat_cfg = next(cfg for cfg in stat_configs if cfg[0] == stat_choice)
        stat_key = stat_cfg[1]
        higher_better = stat_cfg[2]

        merged_full = _build_comparison_df(preseason, current, id_col, name_col, stat_configs)
        delta_col = f"delta_{stat_key}"

        if delta_col not in merged_full.columns or merged_full[delta_col].isna().all():
            st.info(f"No delta data available for {stat_choice}.")
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
                        f"Top {stat_choice} Improvers vs Preseason",
                        positive_color=SAGE,
                        negative_color=SAGE,
                    )
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)

            with col2:
                if not decliners.empty:
                    fig = create_movers_chart(
                        decliners[name_col].tolist(),
                        decliners["delta_pp"].tolist(),
                        f"Top {stat_choice} Decliners vs Preseason",
                        positive_color=EMBER,
                        negative_color=EMBER,
                    )
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)

    # --- Player Lookup with sparkline timeline ---
    with sub_lookup:
        st.markdown(
            f'<div class="insight-card">'
            f'<span class="tdd-meta">Search for a player to see their projection '
            f'evolution from preseason through weekly snapshots.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        all_names = current[[id_col, name_col]].drop_duplicates()
        search_lu = st.text_input("Search player", key="preseason_lu_search")
        if search_lu:
            _norm = strip_accents(search_lu)
            all_names = all_names[
                all_names[name_col].apply(lambda x: _norm.lower() in strip_accents(str(x)).lower())
            ]

        if all_names.empty:
            st.caption("No matching players.")
            return

        selected_name = st.selectbox(
            "Player", all_names[name_col].tolist(), key="preseason_lu_player",
        )
        selected_id = int(
            all_names[all_names[name_col] == selected_name][id_col].iloc[0]
        )

        snapshots_ordered: dict[str, pd.DataFrame] = {}
        if not preseason.empty:
            snapshots_ordered["Preseason"] = preseason
        for date_str in sorted(weekly.keys()):
            snapshots_ordered[date_str] = weekly[date_str]
        snapshots_ordered["Current"] = current

        if len(snapshots_ordered) < 2:
            st.caption("Need at least preseason + current to show a timeline.")
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
                fig, ax = plt.subplots(figsize=(5.5, 2.5))
                fig.patch.set_facecolor(DARK)
                ax.set_facecolor(DARK)

                plot_vals = [v * 100 for v in values]
                x = range(len(plot_vals))
                ax.plot(x, plot_vals, color=color, linewidth=2, marker="o", markersize=5)
                ax.fill_between(x, min(plot_vals) - 0.5, plot_vals, alpha=0.1, color=color)

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
                    ax.fill_between(x, lo_vals, hi_vals, alpha=0.15, color=color)

                ax.set_xticks(list(x))
                ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel(f"{stat_label} (%)", color=SLATE, fontsize=9)
                ax.set_title(
                    f"{selected_name} — {stat_label} Evolution",
                    color=CREAM, fontsize=11, fontweight="bold", pad=8,
                )
                ax.tick_params(colors=SLATE, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                add_watermark(fig)
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)


def _render_preseason_only_table(
    df: pd.DataFrame,
    name_col: str,
    id_col: str,
    stat_configs: list,
) -> None:
    """Simple preseason projection table when no current data is available."""
    snap_date = df["snapshot_date"].iloc[0] if "snapshot_date" in df.columns else "Unknown"
    target_season = df["target_season"].iloc[0] if "target_season" in df.columns else "?"

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">Projection Snapshot</div>
        <div class="insight-bullet">
            <span class="dot" style="background:{GOLD};"></span>
            Target season: {target_season} | Snapshot date: {snap_date}
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SLATE};"></span>
            These projections are frozen from preseason. Current projections not yet available for comparison.
        </div>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("Search player", "", placeholder="Type a name...",
                           key="preseason_search_fallback")
    if search:
        _search_norm = strip_accents(search)
        df = df[df[name_col].apply(lambda x: _search_norm.lower() in strip_accents(str(x)).lower())]

    display_rows = []
    for _, row in df.iterrows():
        r: dict[str, object] = {
            "Rank": len(display_rows) + 1,
            "Name": row[name_col],
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Rating": diamond_rating_text_composite(row["composite_score"]),
        }
        for label, key, higher_better, _ in stat_configs:
            proj_col = f"projected_{key}"
            obs_col = f"observed_{key}"
            if proj_col in row.index and pd.notna(row.get(proj_col)):
                r[f"Proj {label}"] = fmt_stat(row[proj_col], key)
            else:
                r[f"Proj {label}"] = "--"
            if obs_col in row.index and pd.notna(row.get(obs_col)):
                r[f"{PRIOR_SEASON} {label}"] = fmt_stat(row[obs_col], key)
            else:
                r[f"{PRIOR_SEASON} {label}"] = "--"
        display_rows.append(r)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df, width='stretch', hide_index=True, height=600)

    st.caption(
        f"Showing {len(display_df)} players from preseason projection. "
        "These are locked in and won't change — use for end-of-season accuracy review."
    )


# ---------------------------------------------------------------------------
# Tab 6: Season Tracker
# ---------------------------------------------------------------------------

def _render_season_tracker_tab() -> None:
    """In-season accuracy tracker — stub until games begin."""
    st.markdown(
        f'<div class="tdd-section-hdr">In-Season Accuracy Tracker</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Tracking begins Opening Day. Once the season starts, this tab will "
        "show how well the Bayesian projections track real performance as "
        "observed data accumulates."
    )

    st.markdown(
        f'<div class="tdd-stat-value" style="margin-top:16px; '
        f'margin-bottom:8px;">What will be tracked:</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
- **Running MAE** for K% and BB% projections — updated weekly as new data arrives
- **Projected vs Actual scatter plots** — one point per player, diagonal = perfect accuracy
- **Calibration curve** — predicted probability vs actual frequency across deciles
- **Weekly accuracy snapshots** — trend line showing whether the model improves as the season progresses
    """)

    st.markdown(
        f'<div style="margin-top:16px; padding:12px 16px; '
        f'border-left:3px solid var(--tdd-gold); background:transparent;">'
        f'<span class="tdd-meta">'
        f'Data will auto-populate from weekly projection snapshots and '
        f'the daily update pipeline. No manual setup required.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
