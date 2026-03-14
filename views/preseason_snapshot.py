"""Preseason Snapshot page — Compare preseason projections to current, with evolution sparklines."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, EMBER, SLATE, CREAM, DARK,
    DASHBOARD_DIR,
    PITCHER_STATS, HITTER_STATS,
    PRIOR_SEASON, CURRENT_SEASON,
)
from services.data_loader import (
    load_projections,
    load_weekly_snapshots,
)
from utils.formatters import fmt_stat, delta_html
from utils.helpers import strip_accents
from components.metric_cards import metric_card
from components.backtest_charts import create_movers_chart
from lib.theme import add_watermark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_preseason(player_type: str) -> pd.DataFrame:
    """Load preseason snapshot from the snapshots directory."""
    fname = f"{player_type}_projections_{CURRENT_SEASON}_preseason.parquet"
    path = DASHBOARD_DIR / "snapshots" / fname
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


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



# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_preseason_snapshot() -> None:
    """Compare preseason projections to current with evolution charts."""
    st.markdown('<div class="section-header">Preseason Snapshot</div>',
                unsafe_allow_html=True)

    # Check for any preseason snapshots
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
        horizontal=True, key="snap_type",
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
        # Fall back to just showing preseason table
        _render_preseason_table(preseason, name_col, id_col, stat_configs)
        return

    # ------------------------------------------------------------------
    # Tab layout: Overview | Biggest Movers | Player Lookup
    # ------------------------------------------------------------------
    tab_overview, tab_movers, tab_lookup = st.tabs([
        "Comparison Table", "Biggest Movers", "Player Lookup",
    ])

    # ==================================================================
    # Tab 1: Side-by-side comparison table
    # ==================================================================
    with tab_overview:
        merged = _build_comparison_df(preseason, current, id_col, name_col, stat_configs)

        if merged.empty:
            st.warning("No matching players between preseason and current projections.")
            return

        snap_date = preseason["snapshot_date"].iloc[0] if "snapshot_date" in preseason.columns else "Preseason"

        st.markdown(
            f'<div class="insight-card">'
            f'<span style="color:{SLATE};">Comparing </span>'
            f'<span style="color:{GOLD}; font-weight:600;">preseason ({snap_date})</span>'
            f'<span style="color:{SLATE};"> to </span>'
            f'<span style="color:{GOLD}; font-weight:600;">current projections</span>'
            f'<span style="color:{SLATE};"> | {len(merged)} {ptype}s matched</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Search
        search = st.text_input("Search player", "", placeholder="Type a name...",
                               key="snap_overview_search")
        if search:
            _norm = strip_accents(search)
            merged = merged[
                merged[name_col].apply(lambda x: _norm.lower() in strip_accents(str(x)).lower())
            ]

        # Build display table
        display_rows = []
        for _, row in merged.iterrows():
            r: dict[str, object] = {"Name": row[name_col]}
            if "composite_score" in row.index and pd.notna(row.get("composite_score")):
                r["Score"] = round(row["composite_score"], 1)
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
                    d = row[delta_col] * 100  # to percentage points
                    r[f"{label} Chg"] = f"{d:+.1f}pp"
                else:
                    r[f"{label} Chg"] = "--"
            display_rows.append(r)

        display_df = pd.DataFrame(display_rows)
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)

    # ==================================================================
    # Tab 2: Biggest Movers
    # ==================================================================
    with tab_movers:
        stat_choice = st.radio(
            "Stat", [cfg[0] for cfg in stat_configs],
            horizontal=True, key="snap_movers_stat",
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

            # For display: convert to percentage points
            merged_sorted["delta_pp"] = merged_sorted[delta_col] * 100

            # Improvers: depends on higher_better
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
                    st.pyplot(fig, use_container_width=True)
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
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    # ==================================================================
    # Tab 3: Player Lookup with sparkline timeline
    # ==================================================================
    with tab_lookup:
        st.markdown(
            f'<div class="insight-card">'
            f'<span style="color:{SLATE};">Search for a player to see their projection '
            f'evolution from preseason through weekly snapshots.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Player search
        all_names = current[[id_col, name_col]].drop_duplicates()
        search_lu = st.text_input("Search player", key="snap_lu_search")
        if search_lu:
            _norm = strip_accents(search_lu)
            all_names = all_names[
                all_names[name_col].apply(lambda x: _norm.lower() in strip_accents(str(x)).lower())
            ]

        if all_names.empty:
            st.caption("No matching players.")
            return

        selected_name = st.selectbox(
            "Player", all_names[name_col].tolist(), key="snap_lu_player",
        )
        selected_id = int(
            all_names[all_names[name_col] == selected_name][id_col].iloc[0]
        )

        # Build timeline from preseason + weekly snapshots + current
        snapshots_ordered: dict[str, pd.DataFrame] = {}
        if not preseason.empty:
            snapshots_ordered["Preseason"] = preseason
        for date_str in sorted(weekly.keys()):
            snapshots_ordered[date_str] = weekly[date_str]
        snapshots_ordered["Current"] = current

        if len(snapshots_ordered) < 2:
            st.caption("Need at least preseason + current to show a timeline.")
            return

        # Extract stat values across snapshots
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

            # Determine color based on direction of change
            delta = values[-1] - values[0]
            if higher_better:
                color = SAGE if delta > 0 else EMBER
            else:
                color = SAGE if delta < 0 else EMBER

            # Metric card + sparkline side by side
            m_col, spark_col = st.columns([1, 2])
            with m_col:
                pre_val = values[0] * 100
                cur_val = values[-1] * 100
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
                # Full timeline chart (not just sparkline — more useful)
                fig, ax = plt.subplots(figsize=(7, 2.5))
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
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ---------------------------------------------------------------------------
# Fallback: preseason-only table (no current projections)
# ---------------------------------------------------------------------------

def _render_preseason_table(
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
                           key="snap_search_fallback")
    if search:
        _search_norm = strip_accents(search)
        df = df[df[name_col].apply(lambda x: _search_norm.lower() in strip_accents(str(x)).lower())]

    display_rows = []
    for _, row in df.iterrows():
        r: dict[str, object] = {
            "Rank": len(display_rows) + 1,
            "Name": row[name_col],
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Score": round(row["composite_score"], 2),
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
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)

    st.caption(
        f"Showing {len(display_df)} players from preseason projection. "
        "These are locked in and won't change — use for end-of-season accuracy review."
    )
