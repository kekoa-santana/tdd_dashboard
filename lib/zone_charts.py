"""
Pitch location and zone visualization charts.

Three public chart functions for the dashboard:
- plot_pitcher_location_heatmap: where a pitcher throws each pitch type
- plot_hitter_zone_grid: batter whiff vulnerability or damage by zone
- plot_matchup_overlay: pitcher location density over hitter vulnerability

Synced from: player_profiles/src/viz/zone_charts.py
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
        Optionally includes pitch_type column for filtering.
    metric : str
        'whiff_rate' or 'xwoba'.
    batter_name : str
        For chart title.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all combined.
    pitch_types : list[str] | None
        Filter to specific pitch types. Requires pitch_type column in zone_df.
        If None or empty, aggregates across all pitch types (default behavior).

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
        # Diverging: SAGE (low whiff) → neutral → EMBER (high whiff)
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
        # Diverging: EMBER (low = pitcher wins) → neutral → SAGE (high = hitter wins)
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

    # Dim overlay for cells with insufficient data
    from matplotlib.patches import Rectangle as Rect
    for r in range(_N):
        for c in range(_N):
            n_samp = sample_grid[r, c] if not np.isnan(sample_grid[r, c]) else 0
            val = grid[r, c]

            if np.isnan(val) or n_samp < min_sample:
                rect = Rect(
                    (_X_EDGES[c], _Z_EDGES[r]),
                    _X_EDGES[c + 1] - _X_EDGES[c],
                    _Z_EDGES[r + 1] - _Z_EDGES[r],
                    facecolor=DARK, alpha=0.7, zorder=3,
                )
                ax.add_patch(rect)

    _draw_zone_frame(ax)

    stand_label = ""
    if batter_stand:
        stand_label = f" ({'LHB' if batter_stand == 'L' else 'RHB'})"
    title = f"{batter_name}{stand_label} — {label}" if batter_name else label
    ax.set_title(title, color=GOLD, fontsize=10, fontweight="bold", pad=8)

    fig.tight_layout()
    return fig


def plot_matchup_overlay(
    pitcher_loc_df: pd.DataFrame,
    hitter_zone_df: pd.DataFrame,
    pitch_type: str,
    pitcher_name: str = "",
    hitter_name: str = "",
    batter_stand: str | None = None,
    pitch_display_name: str | None = None,
) -> plt.Figure:
    """Two-layer matchup zone chart.

    Layer 1 (KDE): Pitcher location density as a smooth gold heatmap with
    transparency — bright where the pitcher concentrates, fading to
    transparent where he doesn't.

    Layer 2 (dashed grid): Hitter vulnerability on the 5×5 grid. Dashed
    borders colored sage (pitcher advantage — hitter whiffs) or ember
    (hitter advantage — hitter handles it). Danger zones (low density +
    hitter strong) get a thicker ember dash.

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
    pitch_display_name : str | None
        Full display name (e.g. "4-Seam Fastball"). Falls back to pitch_type.

    Returns
    -------
    plt.Figure
    """
    from matplotlib.patches import Rectangle as Rect
    from scipy.stats import gaussian_kde

    pt_label = pitch_display_name or pitch_type

    fig, ax = plt.subplots(figsize=(3.2, 3.5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    # --- Hitter whiff vulnerability (grid-based) ---
    h_df = hitter_zone_df.copy()
    if batter_stand:
        h_df = h_df[h_df["batter_stand"] == batter_stand]
    h_agg = h_df.groupby(["grid_row", "grid_col"])[["swings", "whiffs"]].sum().reset_index()
    h_agg["whiff_rate"] = np.where(
        h_agg["swings"] >= 10, h_agg["whiffs"] / h_agg["swings"], np.nan
    )
    whiff_grid = _build_grid(h_agg, "whiff_rate")

    # --- Layer 1: Pitcher density KDE (gold, transparent → opaque) ---
    p_df = pitcher_loc_df.copy()
    p_df = p_df[p_df["pitch_type"] == pitch_type]
    if batter_stand:
        p_df = p_df[p_df["batter_stand"] == batter_stand]
    p_agg = p_df.groupby(["grid_row", "grid_col"])["pitches"].sum().reset_index()

    # Reconstruct point cloud from grid counts for KDE
    points_x: list[float] = []
    points_z: list[float] = []
    for _, row in p_agg.iterrows():
        r, c = int(row["grid_row"]), int(row["grid_col"])
        if 0 <= r < _N and 0 <= c < _N:
            cx = (_X_EDGES[c] + _X_EDGES[c + 1]) / 2
            cz = (_Z_EDGES[r] + _Z_EDGES[r + 1]) / 2
            cnt = int(row["pitches"])
            points_x.extend([cx] * cnt)
            points_z.extend([cz] * cnt)

    # Evaluate KDE on fine grid
    n_fine = 80
    xi = np.linspace(_X_MIN, _X_MAX, n_fine)
    zi = np.linspace(_Z_MIN, _Z_MAX, n_fine)
    Xi, Zi = np.meshgrid(xi, zi)

    if len(points_x) >= 5:
        kde = gaussian_kde(np.vstack([points_x, points_z]), bw_method=0.3)
        density_fine = kde(np.vstack([Xi.ravel(), Zi.ravel()])).reshape(Xi.shape)
        density_fine /= density_fine.max()  # normalize to [0, 1]
    else:
        density_fine = np.zeros_like(Xi)

    # Gold KDE with transparency: alpha scales with density
    gold_rgb = mcolors.to_rgb(GOLD)
    rgba = np.zeros((*Xi.shape, 4))
    rgba[:, :, 0] = gold_rgb[0]
    rgba[:, :, 1] = gold_rgb[1]
    rgba[:, :, 2] = gold_rgb[2]
    rgba[:, :, 3] = np.clip(density_fine * 0.85, 0, 0.85)

    ax.imshow(
        rgba, extent=[_X_MIN, _X_MAX, _Z_MIN, _Z_MAX],
        origin="lower", aspect="equal", zorder=2,
        interpolation="bilinear",
    )

    # --- Layer 2: Hitter vulnerability grid (dashed borders) ---
    # Compute grid-level density for danger zone detection
    density_grid = _build_grid(p_agg, "pitches") if not p_agg.empty else np.zeros((_N, _N))
    density_grid = np.where(np.isnan(density_grid), 0.0, density_grid)
    total_pitches = density_grid.sum()
    if total_pitches > 0:
        density_frac = density_grid / total_pitches
    else:
        density_frac = np.zeros((_N, _N))
    density_vals = density_frac[density_frac > 0]
    density_q25 = np.percentile(density_vals, 25) if len(density_vals) > 0 else 0.0

    for r in range(_N):
        for c in range(_N):
            whiff = whiff_grid[r, c]
            if np.isnan(whiff):
                continue

            dens = density_frac[r, c]
            is_danger = dens <= density_q25 and whiff < 0.20

            # Color: sage = pitcher advantage (high whiff), ember = hitter advantage
            if whiff >= 0.30:
                color = SAGE
            elif whiff <= 0.18:
                color = EMBER
            else:
                color = SLATE

            lw = 2.0 if is_danger else 1.2
            ls = (0, (4, 3)) if not is_danger else (0, (3, 2))

            border = Rect(
                (_X_EDGES[c], _Z_EDGES[r]),
                _X_EDGES[c + 1] - _X_EDGES[c],
                _Z_EDGES[r + 1] - _Z_EDGES[r],
                facecolor="none",
                edgecolor=color,
                linewidth=lw,
                linestyle=ls,
                alpha=0.9 if is_danger else 0.6,
                zorder=5,
            )
            ax.add_patch(border)

    _draw_zone_frame(ax)

    stand_label = f" vs {'LHH' if batter_stand == 'L' else 'RHH'}" if batter_stand else ""
    title = f"{pitcher_name} {pt_label}{stand_label}"
    subtitle = f"vs {hitter_name}" if hitter_name else ""
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 color=GOLD, fontsize=9, fontweight="bold", pad=8)

    fig.tight_layout(pad=1.2)
    return fig
