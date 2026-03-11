"""Preseason Snapshot page — View frozen preseason projections."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, SLATE, DASHBOARD_DIR,
    PITCHER_STATS, HITTER_STATS,
    PRIOR_SEASON,
)
from services.data_loader import load_projections
from utils.formatters import fmt_stat
from utils.helpers import strip_accents


def page_preseason_snapshot() -> None:
    """View frozen preseason projections for end-of-season comparison."""
    st.markdown('<div class="section-header">Preseason Snapshot</div>',
                unsafe_allow_html=True)

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
        "Player type",
        ["Hitter", "Pitcher"],
        horizontal=True,
        key="snap_type",
    )

    snaps = h_snaps if player_type == "Hitter" else p_snaps
    if not snaps:
        st.warning(f"No {player_type.lower()} snapshots available.")
        return

    snap_labels = []
    for s in snaps:
        season = s.stem.split("_")[2]
        snap_labels.append(f"{season} Preseason")

    selected_label = st.selectbox("Snapshot", snap_labels, key="snap_select")
    selected_snap = snaps[snap_labels.index(selected_label)]

    df = pd.read_parquet(selected_snap)

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
            These projections are frozen from preseason. Compare to actual results at end of season.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if player_type == "Hitter":
        name_col, id_col = "batter_name", "batter_id"
        stat_configs = HITTER_STATS
    else:
        name_col, id_col = "pitcher_name", "pitcher_id"
        stat_configs = PITCHER_STATS

    search = st.text_input("Search player", "", placeholder="Type a name...",
                           key="snap_search")
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
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.caption(
        f"Showing {len(display_df)} {player_type.lower()}s from preseason projection. "
        "These are locked in and won't change — use for end-of-season accuracy review."
    )
