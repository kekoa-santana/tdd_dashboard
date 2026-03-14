"""Data Health page -- Artifact freshness, inventory, and manifest validation."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    GOLD, SAGE, EMBER, SLATE, CREAM,
    DARK_CARD, DARK_BORDER,
    DASHBOARD_DIR,
    CURRENT_SEASON, PRIOR_SEASON, TRAIN_START, TRAIN_END,
    TRAINING_RANGE, AVAILABLE_SEASONS,
)
from services.data_loader import load_update_metadata
from components.metric_cards import metric_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(nbytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def _freshness_color(hours: float | None) -> str:
    """Return color hex based on data freshness."""
    if hours is None:
        return EMBER
    if hours < 24:
        return SAGE
    if hours < 48:
        return GOLD
    return EMBER


@st.cache_data(ttl=300)
def _scan_artifacts(directory: str) -> pd.DataFrame:
    """Scan a directory for parquet/npz/json files and return inventory DataFrame."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for p in sorted(dir_path.iterdir()):
        if not p.is_file():
            continue
        if p.suffix not in (".parquet", ".npz", ".json"):
            continue

        stat = p.stat()
        row: dict = {
            "filename": p.name,
            "size": _human_size(stat.st_size),
            "size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "rows": None,
        }

        if p.suffix == ".parquet":
            try:
                row["rows"] = len(pd.read_parquet(p))
            except Exception:
                row["rows"] = "error"

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_data_health() -> None:
    """Dashboard data health: freshness, inventory, manifest validation."""

    st.markdown(
        '<div class="section-header">Data Health</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Section 1: Update Status
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Update Status</div>',
        unsafe_allow_html=True,
    )

    meta = load_update_metadata()

    if not meta:
        st.warning("No update metadata found. Run the update pipeline to generate one.")
    else:
        # Compute freshness
        hours_since: float | None = None
        last_ts_str = meta.get("last_updated", "")
        if last_ts_str:
            try:
                ts = datetime.fromisoformat(last_ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                hours_since = (
                    datetime.now(timezone.utc) - ts
                ).total_seconds() / 3600.0
            except (ValueError, TypeError):
                pass

        fresh_color = _freshness_color(hours_since)

        # Freshness badge
        if hours_since is not None:
            if hours_since < 24:
                freshness_label = f"{hours_since:.1f}h ago"
            else:
                freshness_label = f"{hours_since:.0f}h ago"
        else:
            freshness_label = "Unknown"

        ts_display = ""
        if last_ts_str:
            try:
                ts = datetime.fromisoformat(last_ts_str)
                ts_display = ts.strftime("%b %d, %Y %I:%M %p")
            except Exception:
                ts_display = last_ts_str

        st.markdown(
            f'<div class="insight-card">'
            f'<span style="color:{SLATE};">Last Updated: </span>'
            f'<span style="color:{fresh_color}; font-weight:600;">'
            f'{ts_display} ({freshness_label})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Metric cards row
        cols = st.columns(5)
        cards = [
            ("Game Date", meta.get("game_date", "N/A")),
            ("Season", str(meta.get("season", "N/A"))),
            ("Hitters Updated", str(meta.get("hitters_updated", "N/A"))),
            ("Pitchers Updated", str(meta.get("pitchers_updated", "N/A"))),
            ("K Samples", str(meta.get("k_samples_count", "N/A"))),
        ]
        for col, (label, value) in zip(cols, cards):
            with col:
                st.markdown(metric_card(label, value), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Section 2: Artifact Inventory
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Artifact Inventory</div>',
        unsafe_allow_html=True,
    )

    df_main = _scan_artifacts(str(DASHBOARD_DIR))
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    df_snap = _scan_artifacts(str(snapshot_dir))

    if df_main.empty:
        st.info("No artifacts found in data/dashboard/.")
    else:
        st.markdown(
            f'<span style="color:{SLATE}; font-size:0.85rem;">'
            f'{len(df_main)} files in data/dashboard/ '
            f'({_human_size(df_main["size_bytes"].sum())} total)'
            f'</span>',
            unsafe_allow_html=True,
        )
        display_df = df_main[["filename", "size", "last_modified", "rows"]].copy()
        display_df.columns = ["Filename", "Size", "Last Modified", "Rows"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if not df_snap.empty:
        st.markdown(
            f'<span style="color:{SLATE}; font-size:0.85rem;">'
            f'{len(df_snap)} files in snapshots/ '
            f'({_human_size(df_snap["size_bytes"].sum())} total)'
            f'</span>',
            unsafe_allow_html=True,
        )
        display_snap = df_snap[["filename", "size", "last_modified", "rows"]].copy()
        display_snap.columns = ["Filename", "Size", "Last Modified", "Rows"]
        st.dataframe(display_snap, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Section 3: Manifest Validation
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Manifest Validation</div>',
        unsafe_allow_html=True,
    )

    try:
        from services.manifest import validate_manifest

        manifest_path = DASHBOARD_DIR / "manifest.json"
        if not manifest_path.exists():
            st.info(
                "No manifest found. Run update pipeline to generate one."
            )
        else:
            status = validate_manifest(DASHBOARD_DIR)

            if status.valid and not status.warnings:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<span style="color:{SAGE}; font-weight:700;">'
                    f'PASS</span>'
                    f'<span style="color:{CREAM};"> -- '
                    f'All manifest checks passed.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif status.valid and status.warnings:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<span style="color:{GOLD}; font-weight:700;">'
                    f'WARNINGS</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for w in status.warnings:
                    st.markdown(
                        f'<span style="color:{GOLD};">- {w}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<span style="color:{EMBER}; font-weight:700;">'
                    f'ERRORS</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if status.missing_artifacts:
                    st.markdown(
                        f'<span style="color:{EMBER};">Missing artifacts:</span>',
                        unsafe_allow_html=True,
                    )
                    for m in status.missing_artifacts:
                        st.markdown(
                            f'<span style="color:{CREAM};">  - {m}</span>',
                            unsafe_allow_html=True,
                        )
                if status.row_count_mismatches:
                    st.markdown(
                        f'<span style="color:{EMBER};">Row count mismatches:</span>',
                        unsafe_allow_html=True,
                    )
                    for r in status.row_count_mismatches:
                        st.markdown(
                            f'<span style="color:{CREAM};">  - {r}</span>',
                            unsafe_allow_html=True,
                        )
                if status.warnings:
                    for w in status.warnings:
                        st.markdown(
                            f'<span style="color:{GOLD};">- {w}</span>',
                            unsafe_allow_html=True,
                        )

            if status.manifest_age_hours is not None:
                age_color = _freshness_color(status.manifest_age_hours)
                st.markdown(
                    f'<span style="color:{SLATE}; font-size:0.85rem;">'
                    f'Manifest age: '
                    f'<span style="color:{age_color};">'
                    f'{status.manifest_age_hours:.1f}h</span></span>',
                    unsafe_allow_html=True,
                )

    except ImportError:
        st.info("Manifest validation is not yet available (services/manifest.py not found).")

    # ------------------------------------------------------------------
    # Section 4: Config Summary
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Config Summary</div>',
        unsafe_allow_html=True,
    )

    config_items = [
        ("CURRENT_SEASON", str(CURRENT_SEASON)),
        ("PRIOR_SEASON", str(PRIOR_SEASON)),
        ("TRAIN_START", str(TRAIN_START)),
        ("TRAIN_END", str(TRAIN_END)),
        ("TRAINING_RANGE", TRAINING_RANGE),
        ("AVAILABLE_SEASONS", ", ".join(str(s) for s in AVAILABLE_SEASONS)),
    ]

    config_html = '<div class="insight-card">'
    for label, value in config_items:
        config_html += (
            f'<div style="margin:4px 0;">'
            f'<span style="color:{SLATE}; font-size:0.85rem;">{label}: </span>'
            f'<span style="color:{CREAM}; font-size:0.85rem;">{value}</span>'
            f'</div>'
        )
    config_html += "</div>"

    st.markdown(config_html, unsafe_allow_html=True)
