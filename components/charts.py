"""Chart builders for the TDD Dashboard (matplotlib + plotly)."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK,
    PITCH_DISPLAY, PITCH_ORDER, PITCH_TYPE_COLORS,
    PRIOR_SEASON,
)
from lib.theme import add_watermark as _theme_watermark


def add_watermark(fig) -> None:
    """Dashboard-specific watermark that scales to figure size."""
    try:
        from tdd_theme import LOGO_PATH
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        if not LOGO_PATH.exists():
            return
        logo_img = mpimg.imread(str(LOGO_PATH))
        # Scale zoom to figure size — smaller charts get smaller watermarks
        w, h = fig.get_size_inches()
        zoom = min(w, h) * 0.04  # ~0.12–0.16 for typical 3–4 inch charts
        zoom = max(0.08, min(zoom, 0.25))  # clamp range
        imagebox = OffsetImage(logo_img, zoom=zoom, alpha=0.025)
        imagebox.image.axes = fig.axes[0] if fig.axes else None
        ab = AnnotationBbox(
            imagebox, (0.5, 0.5),
            xycoords="figure fraction",
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=0,
        )
        fig.add_artist(ab)
    except (ImportError, Exception):
        _theme_watermark(fig)


def apply_dark_mpl() -> None:
    """Apply dark-themed matplotlib settings for dashboard charts."""
    mpl.rcParams.update({
        "figure.facecolor": DARK,
        "axes.facecolor": DARK,
        "savefig.facecolor": DARK,
        "text.color": CREAM,
        "axes.labelcolor": SLATE,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "font.size": 14,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


# Apply at import time
apply_dark_mpl()


def create_posterior_fig(
    samples: np.ndarray,
    observed: float | None = None,
    stat_label: str = "K%",
    color: str = SAGE,
):
    """Create a posterior KDE plot with brand styling. Returns plotly Figure."""
    import plotly.graph_objects as go

    pct_samples = samples * 100
    kde = gaussian_kde(pct_samples, bw_method=0.3)
    x = np.linspace(max(0, pct_samples.min() - 2), pct_samples.max() + 2, 300)
    y = kde(x)

    ci_lo, ci_hi = np.percentile(pct_samples, [2.5, 97.5])
    mean_val = float(np.mean(pct_samples))
    y_max = float(y.max())

    fig = go.Figure()

    # Full KDE curve with light fill
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.25),
        hoverinfo="skip",
        showlegend=False,
    ))

    # 95% CI shaded region (darker fill)
    ci_mask = (x >= ci_lo) & (x <= ci_hi)
    x_ci = x[ci_mask]
    y_ci = y[ci_mask]
    fig.add_trace(go.Scatter(
        x=x_ci, y=y_ci,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.15),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Mean vertical line
    fig.add_shape(
        type="line", x0=mean_val, x1=mean_val, y0=0, y1=y_max,
        line=dict(color=GOLD, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=mean_val, y=y_max * 0.92,
        text=f"<b>{mean_val:.1f}%</b>",
        showarrow=False, font=dict(color=GOLD, size=12),
        xanchor="left", xshift=6,
    )

    # Observed line
    if observed is not None:
        obs_pct = observed * 100
        fig.add_shape(
            type="line", x0=obs_pct, x1=obs_pct, y0=0, y1=y_max,
            line=dict(color=SLATE, width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=obs_pct, y=y_max * 0.75,
            text=f"{obs_pct:.1f}% ({PRIOR_SEASON})",
            showarrow=False, font=dict(color=SLATE, size=10),
            xanchor="left", xshift=6,
        )

    # 95% CI annotation
    fig.add_annotation(
        x=1.0, y=1.0, xref="paper", yref="paper",
        text=f"95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]",
        showarrow=False, font=dict(color=SLATE, size=10),
        xanchor="right", yanchor="top",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        xaxis=dict(
            title=dict(text=stat_label, font=dict(color=SLATE, size=11)),
            tickfont=dict(color=SLATE, size=9),
            showgrid=False, zeroline=False,
            showline=False, rangemode="nonnegative",
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False,
            showline=False,
        ),
    )

    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string for plotly fills."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


_STAT_CHART_CONFIG = {
    "k":  {"xlabel": "Strikeouts", "title_stat": "K",  "color": SAGE},
    "bb": {"xlabel": "Walks",      "title_stat": "BB", "color": EMBER},
    "h":  {"xlabel": "Hits Allowed", "title_stat": "H", "color": SLATE},
    "hr": {"xlabel": "Home Runs",  "title_stat": "HR", "color": GOLD},
}

def create_arsenal_donut(
    arsenal_df: pd.DataFrame,
    season_label: str = "",
):
    """Pitch usage donut chart — colored by pitch type. Returns plotly Figure."""
    import plotly.graph_objects as go
    from config import PITCH_ORDER

    df = arsenal_df.copy()
    df = df[df["pitches"] >= 20]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20), height=300,
        )
        return fig

    df["_order"] = df["pitch_type"].map(
        {pt: i for i, pt in enumerate(PITCH_ORDER)}
    ).fillna(99)
    df = df.sort_values("_order")

    labels = df["pitch_type"].map(PITCH_DISPLAY).fillna(df["pitch_type"]).tolist()
    sizes = df["usage_pct"].tolist()
    colors = [
        PITCH_TYPE_COLORS.get(pt, SLATE)
        for pt in df["pitch_type"]
    ]
    total = int(df["pitches"].sum())

    text_labels = [f"{label} {s * 100:.0f}%" for label, s in zip(labels, sizes)]

    fig = go.Figure(data=[go.Pie(
        labels=text_labels,
        values=sizes,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=DARK, width=1.5)),
        textinfo="label",
        textposition="outside",
        textfont=dict(color=CREAM, size=11),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
        direction="clockwise",
        sort=False,
    )])

    title_text = f"Pitch Mix{f' ({season_label})' if season_label else ''}"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
        showlegend=False,
        title=dict(text=title_text, font=dict(color=CREAM, size=13), x=0.5, xanchor="center"),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br><span style='font-size:10px;color:var(--tdd-slate)'>pitches</span>",
                x=0.5, y=0.5, font=dict(size=16, color=GOLD),
                showarrow=False, xanchor="center", yanchor="middle",
            ),
        ],
    )

    return fig


# ---------------------------------------------------------------------------
# Plotly KDE pitch density charts
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string for plotly."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_pitch_density_plotly(
    location_df: pd.DataFrame,
    pitch_types: list[str] | None = None,
    pitcher_name: str = "",
    batter_stand: str | None = None,
    pitch_hand: str | None = None,
):
    """Plotly KDE pitch density subplots — one panel per pitch type.

    Accepts either raw pitch coordinates (plate_x, plate_z columns) or
    pre-aggregated grid data (grid_row, grid_col, pitches columns).
    Raw coordinates produce smoother, more accurate density maps.

    Parameters
    ----------
    location_df : pd.DataFrame
        Filtered to one pitcher. Either:
        - Raw: columns plate_x, plate_z, pitch_type, batter_stand
        - Grid: columns pitch_type, batter_stand, grid_row, grid_col, pitches
    pitch_types : list[str] | None
        Which pitch types to show. If None, uses top 4 by volume.
    pitcher_name : str
        For chart title.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all batters combined.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from lib.constants import ZONE_BOUNDARIES, ZONE_GRID

    _N = ZONE_GRID["n_rows"]
    _X_MIN, _X_MAX = ZONE_GRID["x_min"], ZONE_GRID["x_max"]
    _Z_MIN, _Z_MAX = ZONE_GRID["z_min"], ZONE_GRID["z_max"]
    _X_EDGES = np.linspace(_X_MIN, _X_MAX, _N + 1)
    _Z_EDGES = np.linspace(_Z_MIN, _Z_MAX, _N + 1)
    _SZ_LEFT = ZONE_BOUNDARIES["plate_x_left"]
    _SZ_RIGHT = ZONE_BOUNDARIES["plate_x_right"]
    _SZ_BOT = ZONE_BOUNDARIES["plate_z_bot_avg"]
    _SZ_TOP = ZONE_BOUNDARIES["plate_z_top_avg"]

    # Detect data format
    has_raw = "plate_x" in location_df.columns and "plate_z" in location_df.columns

    df = location_df.copy()
    if batter_stand:
        df = df[df["batter_stand"] == batter_stand]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No location data", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=SLATE))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Determine pitch types to show
    if has_raw:
        pt_totals = df.groupby("pitch_type").size().sort_values(ascending=False)
    else:
        agg = df.groupby(["pitch_type", "grid_row", "grid_col"]).agg(
            pitches=("pitches", "sum"),
        ).reset_index()
        pt_totals = agg.groupby("pitch_type")["pitches"].sum().sort_values(ascending=False)

    if pitch_types is None:
        pitch_types = pt_totals.head(4).index.tolist()

    n_pt = len(pitch_types)
    if n_pt == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    ncols = min(n_pt, 2)
    nrows = (n_pt + ncols - 1) // ncols

    # Titles with usage %
    subplot_titles = []
    for pt in pitch_types:
        pt_label = PITCH_DISPLAY.get(pt, pt)
        pt_count = int(pt_totals.get(pt, 0))
        total_all = int(pt_totals.sum())
        pct = pt_count / total_all * 100 if total_all > 0 else 0
        subplot_titles.append(f"{pt_label} ({pct:.0f}%)")
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Extended range — show full pitch distribution beyond the zone
    _VIEW_X_MIN, _VIEW_X_MAX = -2.5, 2.5
    _VIEW_Z_MIN, _VIEW_Z_MAX = -0.5, 5.0

    # Fine grid for KDE evaluation over the extended range
    n_fine = 80
    xi = np.linspace(_VIEW_X_MIN, _VIEW_X_MAX, n_fine)
    zi = np.linspace(_VIEW_Z_MIN, _VIEW_Z_MAX, n_fine)

    for idx, pt in enumerate(pitch_types):
        r_idx = idx // ncols + 1
        c_idx = idx % ncols + 1

        if has_raw:
            pt_df = df[df["pitch_type"] == pt]
            points_x = pt_df["plate_x"].dropna().values
            points_z = pt_df["plate_z"].dropna().values
        else:
            pt_agg = agg[agg["pitch_type"] == pt]
            rng = np.random.default_rng(42)
            px_list: list[float] = []
            pz_list: list[float] = []
            for _, row in pt_agg.iterrows():
                r, c = int(row["grid_row"]), int(row["grid_col"])
                if 0 <= r < _N and 0 <= c < _N:
                    cnt = int(row["pitches"])
                    px_list.extend(rng.uniform(_X_EDGES[c], _X_EDGES[c + 1], size=cnt).tolist())
                    pz_list.extend(rng.uniform(_Z_EDGES[r], _Z_EDGES[r + 1], size=cnt).tolist())
            points_x = np.array(px_list)
            points_z = np.array(pz_list)

        if len(points_x) < 5:
            continue

        # KDE
        bw = 0.2 if has_raw else 0.3
        kde = gaussian_kde(np.vstack([points_x, points_z]), bw_method=bw)
        Xi, Zi = np.meshgrid(xi, zi)
        density = kde(np.vstack([Xi.ravel(), Zi.ravel()])).reshape(Xi.shape)

        # Pitch-type color
        pt_color = PITCH_TYPE_COLORS.get(pt, GOLD)

        # Contour-style density — filled contours like Savant
        colorscale = [
            [0.0, "rgba(0,0,0,0)"],
            [0.15, _hex_to_rgba(pt_color, 0.10)],
            [0.3, _hex_to_rgba(pt_color, 0.25)],
            [0.5, _hex_to_rgba(pt_color, 0.50)],
            [0.7, _hex_to_rgba(pt_color, 0.75)],
            [0.85, _hex_to_rgba(pt_color, 0.90)],
            [1.0, "#ffffff"],
        ]

        fig.add_trace(
            go.Contour(
                z=density, x=xi, y=zi,
                colorscale=colorscale,
                showscale=False,
                contours=dict(
                    coloring="fill",
                    showlines=False,
                    start=density.max() * 0.05,
                    end=density.max(),
                    size=density.max() / 8,
                ),
                hoverinfo="skip",
            ),
            row=r_idx, col=c_idx,
        )

        # Hover overlay — pitch % per grid cell
        if not has_raw:
            _hover_agg = agg
        else:
            _hover_agg = df[df["pitch_type"] == pt].copy()
        _cell_x, _cell_z, _cell_pct = [], [], []
        _pt_total = len(points_x) if has_raw else int(pt_totals.get(pt, 1))
        for _r in range(_N):
            for _c in range(_N):
                cx = (_X_EDGES[_c] + _X_EDGES[_c + 1]) / 2
                cz = (_Z_EDGES[_r] + _Z_EDGES[_r + 1]) / 2
                if has_raw:
                    _cnt = int(((df["pitch_type"] == pt) &
                                (df["plate_x"] >= _X_EDGES[_c]) & (df["plate_x"] < _X_EDGES[_c + 1]) &
                                (df["plate_z"] >= _Z_EDGES[_r]) & (df["plate_z"] < _Z_EDGES[_r + 1])).sum())
                else:
                    _cell = _hover_agg[(_hover_agg["pitch_type"] == pt) &
                                       (_hover_agg["grid_row"] == _r) & (_hover_agg["grid_col"] == _c)]
                    _cnt = int(_cell["pitches"].sum()) if not _cell.empty else 0
                if _cnt > 0:
                    _cell_x.append(cx)
                    _cell_z.append(cz)
                    _cell_pct.append(_cnt / _pt_total * 100)
        if _cell_x:
            fig.add_trace(
                go.Scatter(
                    x=_cell_x, y=_cell_z, mode="markers",
                    marker=dict(size=25, opacity=0),
                    customdata=[[f"{p:.1f}%"] for p in _cell_pct],
                    hovertemplate="Pitch %: %{customdata[0]}<extra></extra>",
                    showlegend=False,
                ),
                row=r_idx, col=c_idx,
            )

        # Strike zone rectangle
        fig.add_shape(
            type="rect",
            x0=_SZ_LEFT, x1=_SZ_RIGHT, y0=_SZ_BOT, y1=_SZ_TOP,
            line=dict(color=CREAM, width=1.5, dash="dash"),
            opacity=0.5,
            row=r_idx, col=c_idx,
        )

        # Home plate pentagon (larger)
        _hp_w = 0.42
        fig.add_shape(
            type="path",
            path=(
                f"M {-_hp_w} 0.18 L {_hp_w} 0.18 "
                f"L {_hp_w} 0.0 L 0 -0.22 L {-_hp_w} 0.0 Z"
            ),
            line=dict(color=SLATE, width=1.5),
            fillcolor="rgba(123,143,166,0.12)",
            row=r_idx, col=c_idx,
        )

        # Pitcher silhouette (catcher's perspective — pitcher is behind the zone)
        _hand = pitch_hand or "R"
        _arm_dir = 1 if _hand == "R" else -1
        _px, _py = 0, 4.2
        _sl = "rgba(123,143,166,0.18)"
        # Head
        fig.add_shape(
            type="circle",
            x0=_px - 0.10, x1=_px + 0.10,
            y0=_py + 0.25, y1=_py + 0.45,
            line=dict(color=SLATE, width=1), fillcolor=_sl,
            row=r_idx, col=c_idx,
        )
        # Torso
        fig.add_shape(
            type="line",
            x0=_px, y0=_py + 0.25, x1=_px, y1=_py - 0.15,
            line=dict(color=SLATE, width=1.5),
            row=r_idx, col=c_idx,
        )
        # Throwing arm
        fig.add_shape(
            type="path",
            path=(
                f"M {_px} {_py + 0.15} "
                f"L {_px + _arm_dir * 0.25} {_py + 0.25} "
                f"L {_px + _arm_dir * 0.45} {_py + 0.35}"
            ),
            line=dict(color=SLATE, width=1.5),
            row=r_idx, col=c_idx,
        )
        # Glove arm
        fig.add_shape(
            type="path",
            path=f"M {_px} {_py + 0.15} L {_px - _arm_dir * 0.18} {_py + 0.05}",
            line=dict(color=SLATE, width=1.5),
            row=r_idx, col=c_idx,
        )
        # Stride leg
        fig.add_shape(
            type="line",
            x0=_px, y0=_py - 0.15,
            x1=_px + _arm_dir * 0.12, y1=_py - 0.40,
            line=dict(color=SLATE, width=1.5),
            row=r_idx, col=c_idx,
        )
        # Plant leg
        fig.add_shape(
            type="line",
            x0=_px, y0=_py - 0.15,
            x1=_px - _arm_dir * 0.15, y1=_py - 0.35,
            line=dict(color=SLATE, width=1.5),
            row=r_idx, col=c_idx,
        )
        # RHP / LHP label above pitcher
        _hand_label = "LHP" if _hand == "L" else "RHP"
        fig.add_annotation(
            x=_px, y=_py + 0.58,
            text=_hand_label,
            showarrow=False,
            font=dict(color=SLATE, size=9),
            row=r_idx, col=c_idx,
        )

        # Batter silhouette — only when a specific handedness is selected
        if batter_stand in ("L", "R"):
            # Catcher's perspective: LHB on right side, RHB on left side
            _bx = 1.5 if batter_stand == "L" else -1.5
            _by = 1.8
            _bat_dir = -1 if batter_stand == "L" else 1
            _batter_label = "LHH" if batter_stand == "L" else "RHH"
            # Head
            fig.add_shape(
                type="circle",
                x0=_bx - 0.10, x1=_bx + 0.10,
                y0=_by + 0.50, y1=_by + 0.70,
                line=dict(color=SLATE, width=1), fillcolor=_sl,
                row=r_idx, col=c_idx,
            )
            # Body
            fig.add_shape(
                type="path",
                path=(
                    f"M {_bx} {_by + 0.50} "
                    f"L {_bx - 0.12} {_by - 0.10} "
                    f"L {_bx + 0.12} {_by - 0.10} Z"
                ),
                line=dict(color=SLATE, width=1), fillcolor=_sl,
                row=r_idx, col=c_idx,
            )
            # Bat
            fig.add_shape(
                type="line",
                x0=_bx + _bat_dir * 0.05, y0=_by + 0.30,
                x1=_bx + _bat_dir * 0.50, y1=_by + 0.65,
                line=dict(color=SLATE, width=2.5),
                row=r_idx, col=c_idx,
            )
            # Label
            fig.add_annotation(
                x=_bx, y=_by + 0.82,
                text=_batter_label,
                showarrow=False,
                font=dict(color=SLATE, size=9),
                row=r_idx, col=c_idx,
            )

    # Style all subplots — no ticks, no labels, no grid
    _ax_style = dict(showgrid=False, zeroline=False, showticklabels=False, ticks="", showline=False)
    for idx in range(n_pt):
        x_ax = f"xaxis{idx + 1}" if idx > 0 else "xaxis"
        y_ax = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
        y_anchor = "y" if idx == 0 else f"y{idx + 1}"
        fig.update_layout(**{
            x_ax: dict(range=[_VIEW_X_MIN, _VIEW_X_MAX], scaleanchor=y_anchor, **_ax_style),
            y_ax: dict(range=[_VIEW_Z_MIN, _VIEW_Z_MAX], **_ax_style),
        })

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(color=CREAM, size=12)

    height = 630 * nrows
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=60),
        height=height,
        showlegend=False,
    )

    return fig


def create_hitter_zone_plotly(
    zone_df: pd.DataFrame,
    metric: str = "whiff_rate",
    batter_name: str = "",
    batter_stand: str | None = None,
    pitch_types: list[str] | None = None,
):
    """Plotly KDE contour of hitter zone profile (whiff rate or xwOBA).

    Parameters
    ----------
    zone_df : pd.DataFrame
        Filtered to one batter. Columns: batter_stand, grid_row, grid_col,
        pitches, swings, whiffs, bip, xwoba_sum, xwoba_count.
    metric : str
        'whiff_rate' or 'xwoba'.
    batter_name : str
        For chart title.
    batter_stand : str | None
        Filter to 'L' or 'R'. None = all combined.
    pitch_types : list[str] | None
        Filter to specific pitch types if pitch_type column exists.
    """
    import plotly.graph_objects as go
    from lib.constants import ZONE_BOUNDARIES, ZONE_GRID

    _N = ZONE_GRID["n_rows"]
    _X_EDGES = np.linspace(ZONE_GRID["x_min"], ZONE_GRID["x_max"], _N + 1)
    _Z_EDGES = np.linspace(ZONE_GRID["z_min"], ZONE_GRID["z_max"], _N + 1)
    _SZ_LEFT = ZONE_BOUNDARIES["plate_x_left"]
    _SZ_RIGHT = ZONE_BOUNDARIES["plate_x_right"]
    _SZ_BOT = ZONE_BOUNDARIES["plate_z_bot_avg"]
    _SZ_TOP = ZONE_BOUNDARIES["plate_z_top_avg"]

    _VIEW_X = (-2.5, 2.5)
    _VIEW_Z = (-0.5, 5.0)

    df = zone_df.copy()
    if batter_stand:
        df = df[df["batter_stand"] == batter_stand]
    if pitch_types and "pitch_type" in df.columns:
        df = df[df["pitch_type"].isin(pitch_types)]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No zone data", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=SLATE))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Aggregate
    agg_cols = ["pitches", "swings", "whiffs", "bip"]
    for c in ["xwoba_sum", "xwoba_count", "hard_hits", "barrels"]:
        if c in df.columns:
            agg_cols.append(c)
    agg = df.groupby(["grid_row", "grid_col"])[agg_cols].sum().reset_index()

    # Metric config
    if metric == "whiff_rate":
        label = "Whiff%"
        color_hot = EMBER  # high whiff = hitter weakness
    else:
        label = "xwOBA"
        color_hot = SAGE   # high xwOBA = hitter strength

    # Build weighted point cloud
    rng = np.random.default_rng(42)
    pts_x: list[float] = []
    pts_z: list[float] = []
    wts: list[float] = []

    for _, row in agg.iterrows():
        r, c = int(row["grid_row"]), int(row["grid_col"])
        if not (0 <= r < _N and 0 <= c < _N):
            continue
        if metric == "whiff_rate":
            if row["swings"] < 10:
                continue
            val = row["whiffs"] / row["swings"]
        else:
            xcount = row.get("xwoba_count", 0)
            if xcount < 8:
                continue
            val = row["xwoba_sum"] / xcount

        cnt = max(int(row["pitches"]), 1)
        jx = rng.uniform(_X_EDGES[c], _X_EDGES[c + 1], size=cnt)
        jz = rng.uniform(_Z_EDGES[r], _Z_EDGES[r + 1], size=cnt)
        pts_x.extend(jx.tolist())
        pts_z.extend(jz.tolist())
        wts.extend([val] * cnt)

    if len(pts_x) < 5:
        fig = go.Figure()
        fig.add_annotation(text=f"Not enough data for {label}", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=SLATE))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    kde = gaussian_kde(np.vstack([pts_x, pts_z]), bw_method=0.35, weights=np.array(wts))
    n_fine = 80
    xi = np.linspace(*_VIEW_X, n_fine)
    zi = np.linspace(*_VIEW_Z, n_fine)
    Xi, Zi = np.meshgrid(xi, zi)
    density = kde(np.vstack([Xi.ravel(), Zi.ravel()])).reshape(Xi.shape)

    colorscale = [
        [0.0, "rgba(0,0,0,0)"],
        [0.15, _hex_to_rgba(color_hot, 0.10)],
        [0.3, _hex_to_rgba(color_hot, 0.25)],
        [0.5, _hex_to_rgba(color_hot, 0.50)],
        [0.7, _hex_to_rgba(color_hot, 0.75)],
        [0.85, _hex_to_rgba(color_hot, 0.90)],
        [1.0, "#ffffff"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=density, x=xi, y=zi,
        colorscale=colorscale, showscale=False,
        contours=dict(
            coloring="fill", showlines=False,
            start=density.max() * 0.05, end=density.max(),
            size=density.max() / 8,
        ),
        hoverinfo="skip",
    ))

    # Hover overlay — metric value per grid cell
    _cell_x, _cell_z, _cell_vals = [], [], []
    for _, row in agg.iterrows():
        r, c = int(row["grid_row"]), int(row["grid_col"])
        if not (0 <= r < _N and 0 <= c < _N):
            continue
        if metric == "whiff_rate":
            if row["swings"] < 10:
                continue
            val = row["whiffs"] / row["swings"]
            _cell_vals.append(f"{val:.0%}")
        else:
            xcount = row.get("xwoba_count", 0)
            if xcount < 8:
                continue
            val = row["xwoba_sum"] / xcount
            _cell_vals.append(f".{int(val * 1000):03d}")
        _cell_x.append((_X_EDGES[c] + _X_EDGES[c + 1]) / 2)
        _cell_z.append((_Z_EDGES[r] + _Z_EDGES[r + 1]) / 2)
    if _cell_x:
        _hover_label = "Whiff%" if metric == "whiff_rate" else "xwOBA"
        fig.add_trace(go.Scatter(
            x=_cell_x, y=_cell_z, mode="markers",
            marker=dict(size=25, opacity=0),
            customdata=[[v] for v in _cell_vals],
            hovertemplate=f"{_hover_label}: %{{customdata[0]}}<extra></extra>",
            showlegend=False,
        ))

    # Strike zone
    fig.add_shape(type="rect",
                  x0=_SZ_LEFT, x1=_SZ_RIGHT, y0=_SZ_BOT, y1=_SZ_TOP,
                  line=dict(color=CREAM, width=1.5, dash="dash"), opacity=0.5)

    # Home plate
    _hp = 0.42
    fig.add_shape(type="path",
                  path=f"M {-_hp} 0.18 L {_hp} 0.18 L {_hp} 0.0 L 0 -0.22 L {-_hp} 0.0 Z",
                  line=dict(color=SLATE, width=1.5), fillcolor="rgba(123,143,166,0.12)")

    # Batter silhouette
    _stand = batter_stand or "R"
    _bx = 1.5 if _stand == "L" else -1.5
    _by = 1.8
    _bat_dir = -1 if _stand == "L" else 1
    _bl = "LHH" if _stand == "L" else "RHH"
    _sl = "rgba(123,143,166,0.18)"
    fig.add_shape(type="circle", x0=_bx-0.10, x1=_bx+0.10,
                  y0=_by+0.50, y1=_by+0.70,
                  line=dict(color=SLATE, width=1), fillcolor=_sl)
    fig.add_shape(type="path",
                  path=f"M {_bx} {_by+0.50} L {_bx-0.12} {_by-0.10} L {_bx+0.12} {_by-0.10} Z",
                  line=dict(color=SLATE, width=1), fillcolor=_sl)
    fig.add_shape(type="line",
                  x0=_bx+_bat_dir*0.05, y0=_by+0.30,
                  x1=_bx+_bat_dir*0.50, y1=_by+0.65,
                  line=dict(color=SLATE, width=2.5))
    fig.add_annotation(x=_bx, y=_by+0.82, text=_bl,
                       showarrow=False, font=dict(color=SLATE, size=9))

    stand_sfx = f" ({'LHB' if batter_stand == 'L' else 'RHB'})" if batter_stand else ""
    title = f"{batter_name}{stand_sfx} — {label}" if batter_name else label

    fig.update_layout(
        title=dict(text=title, font=dict(color=CREAM, size=12), x=0.5),
        xaxis=dict(range=list(_VIEW_X), showgrid=False, zeroline=False,
                   showticklabels=False, ticks="", showline=False, scaleanchor="y"),
        yaxis=dict(range=list(_VIEW_Z), showgrid=False, zeroline=False,
                   showticklabels=False, ticks="", showline=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=60),
        height=630, showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Matchup: whiff comparison chart + location peak captions
# ---------------------------------------------------------------------------


def _plate_bucket_label(px: float, pz: float, batter_stand: str | None) -> str:
    """Short catcher-view bucket (inside/away × height) for tooltips and captions."""
    from lib.constants import ZONE_BOUNDARIES

    sz_b = ZONE_BOUNDARIES["plate_z_bot_avg"]
    sz_t = ZONE_BOUNDARIES["plate_z_top_avg"]
    if pz < sz_b + 0.35:
        z_lab = "low"
    elif pz > sz_t - 0.35:
        z_lab = "high"
    else:
        z_lab = "middle"
    thr = 0.28
    if batter_stand == "R":
        if px > thr:
            x_lab = "in"
        elif px < -thr:
            x_lab = "away"
        else:
            x_lab = "mid"
    elif batter_stand == "L":
        if px < -thr:
            x_lab = "in"
        elif px > thr:
            x_lab = "away"
        else:
            x_lab = "mid"
    else:
        x_lab = "3B side" if px < -thr else "1B side" if px > thr else "center"
    return f"{x_lab}, {z_lab}"


def matchup_pitcher_peak_caption(
    location_df: pd.DataFrame,
    pitch_type: str,
    batter_stand: str | None,
) -> str | None:
    """Densest pitch-location bucket for one pitch type (raw coords or grid)."""
    from collections import Counter

    from lib.constants import ZONE_GRID

    if location_df.empty or "pitch_type" not in location_df.columns:
        return None
    df = location_df[location_df["pitch_type"] == pitch_type].copy()
    if batter_stand and "batter_stand" in df.columns:
        df = df[df["batter_stand"] == batter_stand]
    if df.empty:
        return None

    _nc = ZONE_GRID["n_cols"]
    _nr = ZONE_GRID["n_rows"]
    _xe = np.linspace(ZONE_GRID["x_min"], ZONE_GRID["x_max"], _nc + 1)
    _ze = np.linspace(ZONE_GRID["z_min"], ZONE_GRID["z_max"], _nr + 1)

    has_raw = "plate_x" in df.columns and "plate_z" in df.columns
    if has_raw:
        px = df["plate_x"].dropna().values
        pz = df["plate_z"].dropna().values
        if len(px) < 8:
            return None
        ci = np.clip(np.searchsorted(_xe, px, side="right") - 1, 0, _nc - 1)
        ri = np.clip(np.searchsorted(_ze, pz, side="right") - 1, 0, _nr - 1)
        (r, c), _ = Counter(zip(ri, ci)).most_common(1)[0]
        r, c = int(r), int(c)
    else:
        need = {"grid_row", "grid_col", "pitches"}
        if not need.issubset(df.columns):
            return None
        sub = df.groupby(["grid_row", "grid_col"], as_index=False)["pitches"].sum()
        if sub.empty or sub["pitches"].max() < 1:
            return None
        idx = int(sub["pitches"].idxmax())
        r, c = int(sub.loc[idx, "grid_row"]), int(sub.loc[idx, "grid_col"])
        if not (0 <= r < _nr and 0 <= c < _nc):
            return None

    cx = (_xe[c] + _xe[c + 1]) / 2.0
    cz = (_ze[r] + _ze[r + 1]) / 2.0
    return f"Pitcher peak: {_plate_bucket_label(cx, cz, batter_stand)}"


def matchup_hitter_zone_peak_caption(
    zone_df: pd.DataFrame,
    pitch_type: str | None,
    batter_stand: str | None,
    metric: str = "xwoba",
) -> str | None:
    """Highest-xwOBA zone bucket (min contact sample)."""
    from lib.constants import ZONE_GRID

    if zone_df.empty or metric != "xwoba":
        return None
    df = zone_df.copy()
    if batter_stand and "batter_stand" in df.columns:
        df = df[df["batter_stand"] == batter_stand]
    if pitch_type and "pitch_type" in df.columns:
        df = df[df["pitch_type"] == pitch_type]
    if df.empty:
        return None

    agg_cols = ["pitches", "xwoba_sum", "xwoba_count"]
    if not all(c in df.columns for c in ("xwoba_sum", "xwoba_count")):
        return None
    agg = df.groupby(["grid_row", "grid_col"], as_index=False)[agg_cols].sum()
    best_val = -1.0
    best_r: int | None = None
    best_c: int | None = None
    for _, row in agg.iterrows():
        xc = float(row.get("xwoba_count", 0) or 0)
        if xc < 8:
            continue
        val = float(row["xwoba_sum"]) / xc
        if val > best_val:
            best_val = val
            best_r = int(row["grid_row"])
            best_c = int(row["grid_col"])

    if best_r is None or best_c is None:
        return None

    _nc = ZONE_GRID["n_cols"]
    _nr = ZONE_GRID["n_rows"]
    if not (0 <= best_r < _nr and 0 <= best_c < _nc):
        return None
    _xe = np.linspace(ZONE_GRID["x_min"], ZONE_GRID["x_max"], _nc + 1)
    _ze = np.linspace(ZONE_GRID["z_min"], ZONE_GRID["z_max"], _nr + 1)
    cx = (_xe[best_c] + _xe[best_c + 1]) / 2.0
    cz = (_ze[best_r] + _ze[best_r + 1]) / 2.0
    loc = _plate_bucket_label(cx, cz, batter_stand)
    return f"Hitter peak xwOBA: {loc} (.{int(best_val * 1000):03d})"


def create_matchup_whiff_bars_fig(
    arsenal_df: pd.DataFrame,
    vuln_df: pd.DataFrame,
    _str_df: pd.DataFrame,
    pitcher_name: str,
    hitter_name: str,
) -> Figure | None:
    """Grouped horizontal bars: pitcher vs hitter whiff% by pitch (usage → alpha).

    ``_str_df`` is accepted for parity with ``build_matchup_table`` call sites.
    """
    from lib.constants import LEAGUE_AVG_OVERALL
    from matplotlib.patches import Patch

    p_df = arsenal_df.copy()
    p_df = p_df[p_df["pitches"] >= 20]
    if p_df.empty:
        return None

    v_df = vuln_df.copy()
    v_df = v_df[v_df["pitches"] >= 15]
    if not v_df.empty:
        v_df = v_df.sort_values("pitches", ascending=False).drop_duplicates(
            subset=["pitch_type"], keep="first",
        )
    p_df["_order"] = p_df["pitch_type"].map(
        {pt: i for i, pt in enumerate(PITCH_ORDER)},
    ).fillna(99)
    p_df = p_df.sort_values("_order")

    labels: list[str] = []
    p_pct: list[float] = []
    h_pct: list[float] = []
    usages: list[float] = []

    for _, row in p_df.iterrows():
        pt = row["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        u = float(row.get("usage_pct", 0) or 0)
        labels.append(f"{pt_name} ({u * 100:.0f}%)")
        usages.append(u)

        pw = row.get("whiff_rate", np.nan)
        p_pct.append(float(pw * 100) if pd.notna(pw) else float("nan"))

        h_row = v_df[v_df["pitch_type"] == pt]
        if len(h_row) > 0 and "swings" in h_row.columns:
            sw = h_row["swings"].iloc[0]
            wh = h_row["whiffs"].iloc[0] if "whiffs" in h_row.columns else 0
            hr = float(wh / sw * 100) if pd.notna(sw) and sw > 0 else float("nan")
        else:
            hr = float("nan")
        h_pct.append(hr)

    n = len(labels)
    y = np.arange(n, dtype=float)
    fig = Figure(figsize=(7, max(3.0, 0.38 * n + 1.2)))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    h_bar = 0.16
    for i in range(n):
        alpha = 0.38 + 0.62 * min(max(usages[i], 0.0), 1.0)
        alpha = min(alpha, 1.0)
        if pd.notna(p_pct[i]):
            ax.barh(
                y[i] - 0.19, p_pct[i], height=h_bar,
                color=SAGE, alpha=alpha, zorder=2,
            )
        if pd.notna(h_pct[i]):
            ax.barh(
                y[i] + 0.19, h_pct[i], height=h_bar,
                color=EMBER, alpha=alpha, zorder=2,
            )

    ax.set_yticks(y, labels)
    ax.set_xlabel("Whiff%", color=SLATE, fontsize=10)
    ax.set_xlim(0, 52)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=SLATE, labelsize=9)
    ax.grid(axis="x", color=SLATE, alpha=0.15, linestyle="-", linewidth=0.5)

    lg_w = LEAGUE_AVG_OVERALL.get("whiff_rate", 0.25) * 100
    ax.axvline(lg_w, color=GOLD, linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

    leg = [
        Patch(facecolor=SAGE, label="Pitcher whiff%", alpha=0.85),
        Patch(facecolor=EMBER, label="Hitter whiff%", alpha=0.85),
    ]
    ax.legend(
        handles=leg, loc="lower right", fontsize=8,
        framealpha=0.15, labelcolor=CREAM, facecolor=DARK,
    )
    ax.set_title(
        f"P vs H whiff — {pitcher_name} · {hitter_name}",
        color=CREAM, fontsize=11, pad=8,
    )
    fig.tight_layout()
    add_watermark(fig)
    return fig
