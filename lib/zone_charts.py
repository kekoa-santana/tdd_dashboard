"""
Pitch location and zone visualization charts.

Three public chart functions:
- plot_pitcher_location_heatmap: where a pitcher throws each pitch type
- plot_hitter_zone_grid: batter whiff vulnerability or damage by zone
- plot_matchup_overlay: pitcher location density over hitter vulnerability

Canonical source. Dashboard copy at: tdd-dashboard/lib/zone_charts.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from lib.constants import ZONE_BOUNDARIES, ZONE_GRID
from lib.theme import GOLD, EMBER, SAGE, SLATE, CREAM, DARK, add_watermark

# Grid geometry
_N = ZONE_GRID["n_rows"]  # 5
_X_MIN, _X_MAX = ZONE_GRID["x_min"], ZONE_GRID["x_max"]
_Z_MIN, _Z_MAX = ZONE_GRID["z_min"], ZONE_GRID["z_max"]
_X_EDGES = np.linspace(_X_MIN, _X_MAX, _N + 1)
_Z_EDGES = np.linspace(_Z_MIN, _Z_MAX, _N + 1)

# Strike zone rectangle
_SZ_LEFT = ZONE_BOUNDARIES["plate_x_left"]
_SZ_RIGHT = ZONE_BOUNDARIES["plate_x_right"]
_SZ_BOT = ZONE_BOUNDARIES["plate_z_bot_avg"]
_SZ_TOP = ZONE_BOUNDARIES["plate_z_top_avg"]


def _draw_zone_frame(ax: plt.Axes) -> None:
    """Draw 5x5 grid lines and strike zone rectangle on an axes."""
    # Grid lines
    for x in _X_EDGES:
        ax.axvline(x, color=SLATE, alpha=0.2, linewidth=0.5, zorder=1)
    for z in _Z_EDGES:
        ax.axhline(z, color=SLATE, alpha=0.2, linewidth=0.5, zorder=1)

    # Strike zone rectangle
    from matplotlib.patches import Rectangle
    sz_rect = Rectangle(
        (_SZ_LEFT, _SZ_BOT),
        _SZ_RIGHT - _SZ_LEFT,
        _SZ_TOP - _SZ_BOT,
        linewidth=1.5,
        edgecolor=CREAM,
        facecolor="none",
        linestyle="--",
        alpha=0.6,
        zorder=5,
    )
    ax.add_patch(sz_rect)

    # Axes styling
    ax.set_xlim(_X_MIN, _X_MAX)
    ax.set_ylim(_Z_MIN, _Z_MAX)
    ax.set_aspect("equal")
    ax.tick_params(colors=SLATE, labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _build_grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """Pivot a dataframe with grid_row/grid_col into a 5x5 numpy array."""
    grid = np.full((_N, _N), np.nan)
    for _, row in df.iterrows():
        r = int(row["grid_row"])
        c = int(row["grid_col"])
        if 0 <= r < _N and 0 <= c < _N:
            grid[r, c] = row[value_col]
    return grid


def plot_pitcher_location_heatmap(
    location_df: pd.DataFrame,
    pitch_types: list[str] | None = None,
    pitcher_name: str = "",
    batter_stand: str | None = None,
) -> plt.Figure:
    """Heatmap of pitch density by location for a single pitcher.

    Parameters
    ----------
    location_df : pd.DataFrame
        Filtered to one pitcher. Columns: pitch_type, batter_stand,
        grid_row, grid_col, pitches.
    pitch_types : list[str] | None
        Which pitch types to show. If None, uses top 4 by volume.
    pitcher_name : str
        For chart title.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all batters combined.

    Returns
    -------
    plt.Figure
    """
    df = location_df.copy()
    if batter_stand:
        df = df[df["batter_stand"] == batter_stand]

    if df.empty:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(DARK)
        ax.set_facecolor(DARK)
        ax.text(0.5, 0.5, "No location data", color=SLATE,
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Aggregate across batter_stand if not filtered
    agg = df.groupby(["pitch_type", "grid_row", "grid_col"]).agg(
        pitches=("pitches", "sum"),
    ).reset_index()

    # Determine pitch types to show
    if pitch_types is None:
        pt_totals = agg.groupby("pitch_type")["pitches"].sum().sort_values(ascending=False)
        pitch_types = pt_totals.head(4).index.tolist()

    n_pt = len(pitch_types)
    if n_pt == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(DARK)
        ax.set_facecolor(DARK)
        return fig

    ncols = min(n_pt, 4)
    nrows = (n_pt + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows))
    fig.patch.set_facecolor(DARK)

    if n_pt == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    # Gold-based colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gold_heat", [DARK, "#4a3d25", GOLD, "#f5e6c8"], N=256
    )

    for idx, pt in enumerate(pitch_types):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx, col_idx]
        ax.set_facecolor(DARK)

        pt_df = agg[agg["pitch_type"] == pt]
        total = pt_df["pitches"].sum()
        if total == 0:
            ax.text(0.5, 0.5, "No data", color=SLATE,
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            _draw_zone_frame(ax)
            ax.set_title(pt, color=CREAM, fontsize=9, fontweight="bold")
            continue

        # Build density grid (fraction of pitches in each cell)
        pt_df = pt_df.copy()
        pt_df["pct"] = pt_df["pitches"] / total
        grid = _build_grid(pt_df, "pct")

        # Draw heatmap
        im = ax.pcolormesh(
            _X_EDGES, _Z_EDGES, grid,
            cmap=cmap, vmin=0, vmax=max(0.12, np.nanmax(grid)),
            shading="flat", zorder=2,
        )

        # Annotate cells with pitch count
        for r in range(_N):
            for c in range(_N):
                cell_row = pt_df[
                    (pt_df["grid_row"] == r) & (pt_df["grid_col"] == c)
                ]
                if not cell_row.empty:
                    cnt = int(cell_row["pitches"].iloc[0])
                    if cnt >= 5:
                        cx = (_X_EDGES[c] + _X_EDGES[c + 1]) / 2
                        cz = (_Z_EDGES[r] + _Z_EDGES[r + 1]) / 2
                        ax.text(
                            cx, cz, str(cnt), color=CREAM,
                            ha="center", va="center", fontsize=6,
                            alpha=0.7, zorder=6,
                        )

        _draw_zone_frame(ax)
        stand_label = f" vs {'LHH' if batter_stand == 'L' else 'RHH'}" if batter_stand else ""
        ax.set_title(f"{pt}{stand_label}", color=CREAM, fontsize=9, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_pt, nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx, col_idx].set_visible(False)

    title = f"{pitcher_name} — Pitch Location" if pitcher_name else "Pitch Location"
    fig.suptitle(title, color=GOLD, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_hitter_zone_grid(
    zone_df: pd.DataFrame,
    metric: str = "whiff_rate",
    batter_name: str = "",
    batter_stand: str | None = None,
    pitch_types: list[str] | None = None,
) -> plt.Figure:
    """Hot/cold zone grid for a single hitter.

    Parameters
    ----------
    zone_df : pd.DataFrame
        Filtered to one batter. Columns: batter_stand, grid_row, grid_col,
        pitches, swings, whiffs, bip, xwoba_sum, xwoba_count, hard_hits, barrels.
    metric : str
        'whiff_rate' or 'xwoba'.
    batter_name : str
        For chart title.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all combined.
    pitch_types : list[str] | None
        Filter to specific pitch types. Requires pitch_type column in zone_df.

    Returns
    -------
    plt.Figure
    """
    df = zone_df.copy()
    if batter_stand:
        df = df[df["batter_stand"] == batter_stand]

    # Filter by pitch type(s) if requested and column exists
    if pitch_types and "pitch_type" in df.columns:
        df = df[df["pitch_type"].isin(pitch_types)]
        if df.empty:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor(DARK)
            ax.set_facecolor(DARK)
            ax.text(0.5, 0.5, "No data for selected pitch types", color=SLATE,
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            _draw_zone_frame(ax)
            add_watermark(fig)
            fig.tight_layout()
            return fig

    # Aggregate across stands/pitch_types if not filtered
    agg_cols = ["pitches", "swings", "whiffs", "bip"]
    extra_cols = ["xwoba_sum", "xwoba_count", "hard_hits", "barrels"]
    for col in extra_cols:
        if col in df.columns:
            agg_cols.append(col)

    agg = df.groupby(["grid_row", "grid_col"])[agg_cols].sum().reset_index()

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    if agg.empty:
        ax.text(0.5, 0.5, "No zone data", color=SLATE,
                ha="center", va="center", transform=ax.transAxes)
        _draw_zone_frame(ax)
        return fig

    # Compute metric per cell
    if metric == "whiff_rate":
        agg["value"] = np.where(agg["swings"] >= 10, agg["whiffs"] / agg["swings"], np.nan)
        label = "Whiff%"
        # Diverging: SAGE (low whiff) -> neutral -> EMBER (high whiff)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "whiff_div", [SAGE, "#2a2a2a", EMBER], N=256
        )
        vmin, vmax = 0.10, 0.45
        center = 0.25
        fmt = lambda v: f"{v:.0%}"
        min_sample = 10  # swings
        sample_col = "swings"
    elif metric == "xwoba":
        agg["value"] = np.where(
            agg.get("xwoba_count", 0) >= 8,
            agg["xwoba_sum"] / agg["xwoba_count"],
            np.nan,
        )
        label = "xwOBA"
        # Diverging: EMBER (low = pitcher wins) -> neutral -> SAGE (high = hitter wins)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "xwoba_div", [EMBER, "#2a2a2a", SAGE], N=256
        )
        vmin, vmax = 0.150, 0.500
        center = 0.315
        fmt = lambda v: f".{int(v*1000):03d}"
        min_sample = 8
        sample_col = "xwoba_count" if "xwoba_count" in agg.columns else "bip"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    grid = _build_grid(agg, "value")
    sample_grid = _build_grid(agg, sample_col)

    # Draw heatmap
    norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(
        _X_EDGES, _Z_EDGES, grid,
        cmap=cmap, norm=norm, shading="flat", zorder=2,
    )

    # Alpha overlay for small samples
    for r in range(_N):
        for c in range(_N):
            n_samp = sample_grid[r, c] if not np.isnan(sample_grid[r, c]) else 0
            val = grid[r, c]
            cx = (_X_EDGES[c] + _X_EDGES[c + 1]) / 2
            cz = (_Z_EDGES[r] + _Z_EDGES[r + 1]) / 2

            if np.isnan(val) or n_samp < min_sample:
                # Dim overlay for insufficient data
                from matplotlib.patches import Rectangle as Rect
                rect = Rect(
                    (_X_EDGES[c], _Z_EDGES[r]),
                    _X_EDGES[c + 1] - _X_EDGES[c],
                    _Z_EDGES[r + 1] - _Z_EDGES[r],
                    facecolor=DARK, alpha=0.7, zorder=3,
                )
                ax.add_patch(rect)
                ax.text(cx, cz, "--", color=SLATE, ha="center", va="center",
                        fontsize=7, alpha=0.5, zorder=6)
            else:
                ax.text(cx, cz, fmt(val), color=CREAM, ha="center", va="center",
                        fontsize=7, fontweight="bold", zorder=6)

    _draw_zone_frame(ax)

    stand_label = ""
    if batter_stand:
        stand_label = f" ({'LHB' if batter_stand == 'L' else 'RHB'})"
    title = f"{batter_name}{stand_label} — {label}" if batter_name else label
    ax.set_title(title, color=GOLD, fontsize=10, fontweight="bold", pad=8)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_matchup_overlay(
    pitcher_loc_df: pd.DataFrame,
    hitter_zone_df: pd.DataFrame,
    pitch_type: str,
    pitcher_name: str = "",
    hitter_name: str = "",
    batter_stand: str | None = None,
) -> plt.Figure:
    """Overlay pitcher location density on hitter whiff vulnerability.

    Parameters
    ----------
    pitcher_loc_df : pd.DataFrame
        Filtered to one pitcher. Columns: pitch_type, batter_stand,
        grid_row, grid_col, pitches.
    hitter_zone_df : pd.DataFrame
        Filtered to one batter. Columns: batter_stand, grid_row, grid_col,
        swings, whiffs.
    pitch_type : str
        Which pitch type to show.
    pitcher_name, hitter_name : str
        For chart title.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all combined.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    # --- Hitter whiff vulnerability background ---
    h_df = hitter_zone_df.copy()
    if batter_stand:
        h_df = h_df[h_df["batter_stand"] == batter_stand]
    h_agg = h_df.groupby(["grid_row", "grid_col"])[["swings", "whiffs"]].sum().reset_index()
    h_agg["whiff_rate"] = np.where(
        h_agg["swings"] >= 10, h_agg["whiffs"] / h_agg["swings"], np.nan
    )

    whiff_grid = _build_grid(h_agg, "whiff_rate")
    cmap_bg = mcolors.LinearSegmentedColormap.from_list(
        "whiff_bg", [SAGE, "#1a1a1a", EMBER], N=256
    )
    norm = mcolors.TwoSlopeNorm(vcenter=0.25, vmin=0.10, vmax=0.45)

    ax.pcolormesh(
        _X_EDGES, _Z_EDGES, whiff_grid,
        cmap=cmap_bg, norm=norm, shading="flat", zorder=1, alpha=0.5,
    )

    # --- Pitcher location density overlay ---
    p_df = pitcher_loc_df.copy()
    p_df = p_df[p_df["pitch_type"] == pitch_type]
    if batter_stand:
        p_df = p_df[p_df["batter_stand"] == batter_stand]

    p_agg = p_df.groupby(["grid_row", "grid_col"])["pitches"].sum().reset_index()
    total_pitches = p_agg["pitches"].sum()

    if total_pitches > 0:
        max_pitches = p_agg["pitches"].max()
        for _, row in p_agg.iterrows():
            r, c = int(row["grid_row"]), int(row["grid_col"])
            cnt = int(row["pitches"])
            if cnt < 3:
                continue
            cx = (_X_EDGES[c] + _X_EDGES[c + 1]) / 2
            cz = (_Z_EDGES[r] + _Z_EDGES[r + 1]) / 2
            # Size proportional to density, min 30, max 300
            size = 30 + 270 * (cnt / max_pitches)
            ax.scatter(
                cx, cz, s=size, c=GOLD, alpha=0.7,
                edgecolors=CREAM, linewidth=0.5, zorder=4,
            )
            ax.text(
                cx, cz, str(cnt), color=DARK, ha="center", va="center",
                fontsize=6, fontweight="bold", zorder=5,
            )

    _draw_zone_frame(ax)

    # Whiff rate annotations (lighter, behind scatter)
    for r in range(_N):
        for c in range(_N):
            val = whiff_grid[r, c]
            if not np.isnan(val):
                cx = (_X_EDGES[c] + _X_EDGES[c + 1]) / 2
                cz = _Z_EDGES[r] + 0.05  # slightly below center
                ax.text(
                    cx, cz, f"{val:.0%}", color=SLATE,
                    ha="center", va="bottom", fontsize=5, alpha=0.5, zorder=3,
                )

    stand_label = f" vs {'LHH' if batter_stand == 'L' else 'RHH'}" if batter_stand else ""
    title = f"{pitcher_name} {pitch_type}{stand_label}"
    subtitle = f"vs {hitter_name}" if hitter_name else ""
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 color=GOLD, fontsize=9, fontweight="bold", pad=8)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_pitcher_location_compact(
    location_df: pd.DataFrame,
    pitch_types: list[str] | None = None,
    pitcher_name: str = "",
    batter_stand: str | None = None,
    figsize: tuple[float, float] = (14, 3.5),
) -> plt.Figure:
    """Compact single-row pitch location heatmap for Game Prep cards.

    Similar to :func:`plot_pitcher_location_heatmap` but optimised for
    small inline display: one row of subplots, no cell annotations,
    no watermark.

    Parameters
    ----------
    location_df : pd.DataFrame
        Filtered to one pitcher. Columns: pitch_type, batter_stand,
        grid_row, grid_col, pitches.
    pitch_types : list[str] | None
        Which pitch types to show. If None, uses top 4 by volume.
    pitcher_name : str
        Optional suptitle text.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all batters combined.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    plt.Figure
    """
    from config import PITCH_DISPLAY

    df = location_df.copy()

    # Compute per-pitch-type usage splits by batter hand (before any filtering)
    _usage_by_stand: dict[str, dict[str, float]] = {}
    if "batter_stand" in df.columns:
        stand_totals = df.groupby("batter_stand")["pitches"].sum()
        for pt_code in df["pitch_type"].unique():
            pt_data = df[df["pitch_type"] == pt_code]
            _usage_by_stand[pt_code] = {}
            for stand in ("L", "R"):
                st_total = stand_totals.get(stand, 0)
                if st_total > 0:
                    pt_stand_n = pt_data[pt_data["batter_stand"] == stand]["pitches"].sum()
                    _usage_by_stand[pt_code][stand] = pt_stand_n / st_total
                else:
                    _usage_by_stand[pt_code][stand] = 0.0

    if batter_stand:
        df = df[df["batter_stand"] == batter_stand]

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.text(0.5, 0.5, "No location data", color=SLATE,
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return fig

    # Aggregate across batter_stand if not filtered
    agg = df.groupby(["pitch_type", "grid_row", "grid_col"]).agg(
        pitches=("pitches", "sum"),
    ).reset_index()

    # Determine pitch types — sorted by overall usage (descending)
    pt_totals = agg.groupby("pitch_type")["pitches"].sum().sort_values(ascending=False)
    if pitch_types is None:
        pitch_types = pt_totals.head(4).index.tolist()
    else:
        # Re-sort provided pitch types by usage
        pitch_types = sorted(
            pitch_types,
            key=lambda pt: pt_totals.get(pt, 0),
            reverse=True,
        )

    n_pt = len(pitch_types)
    if n_pt == 0:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        return fig

    fig, axes = plt.subplots(1, n_pt, figsize=figsize)
    fig.patch.set_alpha(0)

    if n_pt == 1:
        axes = [axes]

    # Gold-based colormap — use card bg as the "zero" color so it blends
    _CARD_BG = "#1A1917"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gold_heat", [_CARD_BG, "#4a3d25", GOLD, "#f5e6c8"], N=256,
    )

    # Grand total for overall usage %
    grand_total = pt_totals.sum()

    for idx, pt in enumerate(pitch_types):
        ax = axes[idx]
        ax.set_facecolor(_CARD_BG)

        pt_df = agg[agg["pitch_type"] == pt]
        total = pt_df["pitches"].sum()
        if total == 0:
            ax.text(0.5, 0.5, "No data", color=SLATE,
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            _draw_zone_frame(ax)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            label = PITCH_DISPLAY.get(pt, pt)
            ax.set_title(label, color=CREAM, fontsize=12, fontweight="bold")
            continue

        # Build density grid (fraction of pitches in each cell)
        pt_df = pt_df.copy()
        pt_df["pct"] = pt_df["pitches"] / total
        grid = _build_grid(pt_df, "pct")

        ax.pcolormesh(
            _X_EDGES, _Z_EDGES, grid,
            cmap=cmap, vmin=0, vmax=max(0.12, np.nanmax(grid)),
            shading="flat", zorder=2,
        )

        _draw_zone_frame(ax)

        # Remove technical tick labels, add context labels for first subplot
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        if idx == 0:
            ax.set_ylabel("High → Low", color=SLATE, fontsize=8, labelpad=4)
            if batter_stand == "L":
                ax.set_xlabel("Outside → Inside", color=SLATE, fontsize=8, labelpad=4)
            else:
                ax.set_xlabel("Inside → Outside", color=SLATE, fontsize=8, labelpad=4)

        # Title: pitch name + usage subtitle
        label = PITCH_DISPLAY.get(pt, pt)
        stand_usage = _usage_by_stand.get(pt, {})
        l_pct = stand_usage.get("L", 0) * 100
        r_pct = stand_usage.get("R", 0) * 100
        if l_pct > 0 or r_pct > 0:
            usage_line = f"LHB {l_pct:.0f}%  ·  RHB {r_pct:.0f}%"
        else:
            overall_pct = (pt_totals.get(pt, 0) / grand_total * 100) if grand_total > 0 else 0
            usage_line = f"{overall_pct:.0f}% usage"
        ax.set_title(f"{label}\n{usage_line}", color=CREAM, fontsize=12,
                     fontweight="bold", pad=6)
        # Color the usage subtitle line (second line of title) in SLATE
        title_obj = ax.title
        title_obj.set_fontsize(12)
        # Use a two-line title with smaller second line via text instead
        ax.set_title("", pad=0)
        ax.text(0.5, 1.08, label, transform=ax.transAxes, ha="center",
                va="bottom", color=CREAM, fontsize=12, fontweight="bold")
        ax.text(0.5, 1.02, usage_line, transform=ax.transAxes, ha="center",
                va="bottom", color=SLATE, fontsize=8)

    if pitcher_name:
        fig.suptitle(pitcher_name, color=GOLD, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def plot_pitcher_location_split(
    location_df: pd.DataFrame,
    pitch_types: list[str] | None = None,
    pitcher_name: str = "",
    figsize: tuple[float, float] = (14, 6.5),
) -> plt.Figure:
    """Split pitch location heatmap: top row vs RHB, bottom row vs LHB.

    Same pitch type order in both rows (sorted by overall usage) so the
    reader can visually compare each pitch's location across batter hands.

    Parameters
    ----------
    location_df : pd.DataFrame
        Filtered to one pitcher. Columns: pitch_type, batter_stand,
        grid_row, grid_col, pitches.
    pitch_types : list[str] | None
        Which pitch types to show. If None, uses top 4 by overall volume.
    pitcher_name : str
        Optional suptitle text.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    plt.Figure
    """
    from config import PITCH_DISPLAY

    df = location_df.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.text(0.5, 0.5, "No location data", color=SLATE,
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return fig

    # Aggregate across both hands for overall ordering
    agg_all = df.groupby(["pitch_type", "grid_row", "grid_col"]).agg(
        pitches=("pitches", "sum"),
    ).reset_index()
    pt_totals = agg_all.groupby("pitch_type")["pitches"].sum().sort_values(ascending=False)

    if pitch_types is None:
        pitch_types = pt_totals.head(4).index.tolist()
    else:
        pitch_types = sorted(
            pitch_types,
            key=lambda pt: pt_totals.get(pt, 0),
            reverse=True,
        )

    n_pt = len(pitch_types)
    if n_pt == 0:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        return fig

    # Per-stand aggregations
    stand_aggs: dict[str, pd.DataFrame] = {}
    stand_totals: dict[str, pd.Series] = {}
    for stand in ("R", "L"):
        s_df = df[df["batter_stand"] == stand]
        if s_df.empty:
            stand_aggs[stand] = pd.DataFrame()
            stand_totals[stand] = pd.Series(dtype=float)
        else:
            s_agg = s_df.groupby(["pitch_type", "grid_row", "grid_col"]).agg(
                pitches=("pitches", "sum"),
            ).reset_index()
            stand_aggs[stand] = s_agg
            stand_totals[stand] = s_agg.groupby("pitch_type")["pitches"].sum()

    # Grand total per stand (for usage %)
    grand_by_stand: dict[str, float] = {}
    for stand in ("R", "L"):
        gt = stand_totals[stand].sum() if not stand_totals[stand].empty else 0
        grand_by_stand[stand] = float(gt)

    fig, axes = plt.subplots(2, n_pt, figsize=figsize)
    fig.patch.set_alpha(0)

    if n_pt == 1:
        axes = axes.reshape(2, 1)

    _CARD_BG = "#1A1917"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gold_heat", [_CARD_BG, "#4a3d25", GOLD, "#f5e6c8"], N=256,
    )

    stands = ["R", "L"]
    row_labels = ["vs RHB", "vs LHB"]

    for row_idx, (stand, row_label) in enumerate(zip(stands, row_labels)):
        s_agg = stand_aggs[stand]
        s_grand = grand_by_stand[stand]

        for col_idx, pt in enumerate(pitch_types):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(_CARD_BG)

            pt_df = s_agg[s_agg["pitch_type"] == pt] if not s_agg.empty else pd.DataFrame()
            total = pt_df["pitches"].sum() if not pt_df.empty else 0

            if total == 0:
                ax.text(0.5, 0.5, "No data", color=SLATE,
                        ha="center", va="center", transform=ax.transAxes, fontsize=7)
                _draw_zone_frame(ax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(length=0)
            else:
                pt_df = pt_df.copy()
                pt_df["pct"] = pt_df["pitches"] / total
                grid = _build_grid(pt_df, "pct")

                ax.pcolormesh(
                    _X_EDGES, _Z_EDGES, grid,
                    cmap=cmap, vmin=0, vmax=max(0.12, np.nanmax(grid)),
                    shading="flat", zorder=2,
                )
                _draw_zone_frame(ax)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(length=0)

            # Context labels on first column of each row
            if col_idx == 0:
                ax.set_ylabel("High \u2192 Low", color=SLATE, fontsize=8, labelpad=4)
                # Inside/Outside flips by batter hand (plate_x orientation)
                if stand == "R":
                    ax.set_xlabel("Inside \u2192 Outside", color=SLATE, fontsize=8, labelpad=4)
                else:
                    ax.set_xlabel("Outside \u2192 Inside", color=SLATE, fontsize=8, labelpad=4)

            # Column titles only on top row
            if row_idx == 0:
                label = PITCH_DISPLAY.get(pt, pt)
                # Hand-specific usage %
                pt_stand_n = float(stand_totals[stand].get(pt, 0)) if not stand_totals[stand].empty else 0
                usage_pct = (pt_stand_n / s_grand * 100) if s_grand > 0 else 0
                usage_line = f"RHB {usage_pct:.0f}%"
                ax.text(0.5, 1.12, label, transform=ax.transAxes, ha="center",
                        va="bottom", color=CREAM, fontsize=12, fontweight="bold")
                ax.text(0.5, 1.05, usage_line, transform=ax.transAxes, ha="center",
                        va="bottom", color=SLATE, fontsize=8)
            else:
                # Bottom row: just the hand-specific usage
                pt_stand_n = float(stand_totals[stand].get(pt, 0)) if not stand_totals[stand].empty else 0
                usage_pct = (pt_stand_n / s_grand * 100) if s_grand > 0 else 0
                usage_line = f"LHB {usage_pct:.0f}%"
                ax.text(0.5, 1.05, usage_line, transform=ax.transAxes, ha="center",
                        va="bottom", color=SLATE, fontsize=8)

    # Row labels on the left margin
    fig.text(0.01, 0.73, "vs RHB", color=GOLD, fontsize=10, fontweight="bold",
             va="center", rotation=90)
    fig.text(0.01, 0.30, "vs LHB", color=GOLD, fontsize=10, fontweight="bold",
             va="center", rotation=90)

    if pitcher_name:
        fig.suptitle(pitcher_name, color=GOLD, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout(rect=[0.03, 0, 1, 1])
    return fig
