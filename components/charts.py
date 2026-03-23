"""Matplotlib chart builders for the TDD Dashboard."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK,
    PITCH_DISPLAY, PITCH_TYPE_TO_FAMILY, PITCH_FAMILY_COLORS, PITCH_TYPE_COLORS,
    PRIOR_SEASON, CURRENT_SEASON,
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
from utils.formatters import whiff_quality_color, xwoba_quality_color


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
) -> Figure:
    """Create a posterior KDE plot with brand styling."""
    fig = Figure(figsize=(5.5, 3.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    pct_samples = samples * 100
    kde = gaussian_kde(pct_samples, bw_method=0.3)
    x = np.linspace(pct_samples.min() - 2, pct_samples.max() + 2, 300)
    y = kde(x)

    ax.fill_between(x, y, alpha=0.3, color=color)
    ax.plot(x, y, color=color, linewidth=2)

    ci_lo, ci_hi = np.percentile(pct_samples, [2.5, 97.5])
    ci_mask = (x >= ci_lo) & (x <= ci_hi)
    ax.fill_between(x[ci_mask], y[ci_mask], alpha=0.15, color=color)

    mean_val = np.mean(pct_samples)
    ax.axvline(mean_val, color=GOLD, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(
        mean_val, ax.get_ylim()[1] * 0.92,
        f" {mean_val:.1f}%",
        color=GOLD, fontsize=11, fontweight="bold", va="top",
    )

    if observed is not None:
        obs_pct = observed * 100
        ax.axvline(obs_pct, color=SLATE, linewidth=1.5, linestyle=":", alpha=0.8)
        ax.text(
            obs_pct, ax.get_ylim()[1] * 0.75,
            f" {obs_pct:.1f}% ({PRIOR_SEASON})",
            color=SLATE, fontsize=9, va="top",
        )

    ax.text(
        0.98, 0.95,
        f"95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]",
        transform=ax.transAxes,
        color=SLATE, fontsize=9, ha="right", va="top",
    )

    ax.set_xlabel(stat_label, color=SLATE, fontsize=10)
    ax.set_ylabel("", color=SLATE, fontsize=10)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    add_watermark(fig)
    fig.tight_layout()
    return fig


_STAT_CHART_CONFIG = {
    "k":  {"xlabel": "Strikeouts", "title_stat": "K",  "color": SAGE},
    "bb": {"xlabel": "Walks",      "title_stat": "BB", "color": EMBER},
    "h":  {"xlabel": "Hits Allowed", "title_stat": "H", "color": SLATE},
    "hr": {"xlabel": "Home Runs",  "title_stat": "HR", "color": GOLD},
}


def create_game_stat_fig(
    samples: np.ndarray,
    pitcher_name: str,
    stat: str = "k",
) -> Figure:
    """Create a game stat distribution histogram, parameterized by stat."""
    cfg = _STAT_CHART_CONFIG.get(stat, _STAT_CHART_CONFIG["k"])
    bar_color = cfg["color"]
    xlabel = cfg["xlabel"]
    title_stat = cfg["title_stat"]

    fig = Figure(figsize=(5.5, 3.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    max_val = int(samples.max()) + 1
    bins = np.arange(-0.5, max_val + 1.5, 1)
    counts, _, bars = ax.hist(
        samples, bins=bins, density=True,
        color=bar_color, alpha=0.7, edgecolor=DARK, linewidth=0.5,
    )

    mode_val = int(np.median(samples))
    for bar in bars:
        if abs(bar.get_x() + 0.5 - mode_val) < 0.5:
            bar.set_facecolor(GOLD)
            bar.set_alpha(0.9)

    mean_val = np.mean(samples)
    ax.axvline(mean_val, color=GOLD, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(
        mean_val + 0.3, ax.get_ylim()[1] * 0.9,
        f"E[{title_stat}] = {mean_val:.1f}",
        color=GOLD, fontsize=11, fontweight="bold", va="top",
    )

    ax.set_xlabel(xlabel, color=SLATE, fontsize=11)
    ax.set_ylabel("Probability", color=SLATE, fontsize=10)
    ax.set_title(
        f"{pitcher_name} -- Projected {title_stat} Distribution ({CURRENT_SEASON})",
        color=CREAM, fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def create_game_k_fig(
    k_samples: np.ndarray,
    pitcher_name: str,
) -> Figure:
    """Create a game K distribution histogram (backward-compat wrapper)."""
    return create_game_stat_fig(k_samples, pitcher_name, stat="k")


def create_arsenal_fig(
    arsenal_df: pd.DataFrame,
    pitcher_name: str,
) -> Figure:
    """Pitcher arsenal chart."""
    df = arsenal_df.sort_values("usage_pct", ascending=True).copy()
    df["label"] = (
        df["pitch_type"].map(PITCH_DISPLAY).fillna(df["pitch_type"])
        + "  " + (df["usage_pct"] * 100).round(0).astype(int).astype(str) + "%"
    )

    n = len(df)
    max_usage = df["usage_pct"].max()
    min_thickness, max_thickness = 0.25, 0.85
    df["thickness"] = min_thickness + (
        (df["usage_pct"] / max_usage) * (max_thickness - min_thickness)
    )

    fig = Figure(figsize=(5.5, max(2.5, n * 0.6)))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    y_positions = np.arange(n)

    for i, (_, row) in enumerate(df.iterrows()):
        velo = row.get("avg_velo", np.nan)
        whiff = row.get("whiff_rate", np.nan)
        xwoba = row.get("xwoba_against", np.nan)
        bar_len = velo if pd.notna(velo) else 0
        color = whiff_quality_color(whiff) if pd.notna(whiff) else SLATE
        thickness = row["thickness"]

        if bar_len > 0:
            rsize = min(thickness * 0.7, 2.0)
            bar = FancyBboxPatch(
                (0, y_positions[i] - thickness / 2),
                bar_len, thickness,
                boxstyle=f"round,pad=0,rounding_size={rsize:.2f}",
                facecolor=color, alpha=0.85, zorder=2,
            )
            ax.add_patch(bar)

        if pd.notna(xwoba) and bar_len > 0:
            dot_color = xwoba_quality_color(xwoba)
            ax.scatter(
                bar_len - 3, y_positions[i],
                s=20, color=dot_color, edgecolors="none",
                zorder=3,
            )

        parts = []
        if pd.notna(whiff):
            parts.append(f"Whiff {whiff*100:.0f}%")
        if pd.notna(xwoba):
            parts.append(f"xwOBA .{int(xwoba*1000):03d}")
        if parts:
            ax.text(
                104, y_positions[i], "  " + " | ".join(parts),
                color=CREAM, fontsize=8.5, va="center", ha="left",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, 103)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Velocity (mph)", color=SLATE, fontsize=10)
    ax.set_title(
        f"{pitcher_name} -- Pitch Arsenal ({PRIOR_SEASON})",
        color=CREAM, fontsize=12, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="y", colors=CREAM, labelsize=10)
    ax.tick_params(axis="x", colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    fig.subplots_adjust(right=0.60)
    return fig


def create_arsenal_donut(
    arsenal_df: pd.DataFrame,
    season_label: str = "",
) -> Figure:
    """Pitch usage donut chart — colored by pitch family."""
    from config import PITCH_ORDER

    df = arsenal_df.copy()
    df = df[df["pitches"] >= 20]
    if df.empty:
        fig = Figure(figsize=(3.5, 3.5))
        fig.patch.set_facecolor(DARK)
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

    fig = Figure(figsize=(3.5, 3.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor=DARK, linewidth=1.5),
    )

    # Annotate each wedge outside the donut
    for i, (wedge, label, size) in enumerate(zip(wedges, labels, sizes)):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        ha = "left" if x >= 0 else "right"
        ax.annotate(
            f"{label} {size * 100:.0f}%",
            xy=(0.7 * x, 0.7 * y),
            xytext=(1.15 * x, 1.15 * y),
            fontsize=8,
            color=CREAM,
            fontweight="600",
            ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6),
        )

    # Center label
    total = int(df["pitches"].sum())
    ax.text(
        0, 0.06, f"{total:,}",
        ha="center", va="center", fontsize=14, fontweight="bold", color=GOLD,
    )
    ax.text(
        0, -0.12, "pitches",
        ha="center", va="center", fontsize=8, color=SLATE,
    )

    ax.set_title(
        f"Pitch Mix{f' ({season_label})' if season_label else ''}",
        color=CREAM, fontsize=10, fontweight="bold", pad=8,
    )

    fig.tight_layout()
    return fig


def blend_whiff_rate(row: pd.Series) -> float:
    """Blend raw whiff rate toward league baseline by sample reliability."""
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

    raw = row.get("whiff_rate", 0) or 0
    swings = row.get("swings", 0) or 0
    pt = row.get("pitch_type", "")
    league = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get(
        "whiff_rate", LEAGUE_AVG_OVERALL.get("whiff_rate", 0.25),
    )
    reliability = min(swings, 50) / 50
    return reliability * raw + (1 - reliability) * league


def create_hitter_vuln_fig(
    vuln_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    hitter_name: str,
) -> Figure:
    """Dual bar chart: vulnerabilities (whiff rate, sample-blended) and
    strengths (xwOBA on contact)."""
    merged = vuln_df[vuln_df["swings"] >= 10].copy() if "swings" in vuln_df.columns else vuln_df.copy()
    merged["label"] = merged["pitch_type"].map(PITCH_DISPLAY).fillna(merged["pitch_type"])

    merged["blended_whiff"] = merged.apply(blend_whiff_rate, axis=1)

    if "xwoba_contact" not in merged.columns or merged["xwoba_contact"].isna().all():
        if strength_df is not None and not strength_df.empty and "xwoba_contact" in strength_df.columns:
            xwoba_map = strength_df.set_index("pitch_type")["xwoba_contact"].to_dict()
            merged["xwoba_contact"] = merged["pitch_type"].map(xwoba_map)

    merged = merged.sort_values("blended_whiff", ascending=True)

    n_rows = len(merged)
    fig = Figure(figsize=(5.5, max(2.2, n_rows * 0.5)))
    axes = fig.subplots(1, 2)
    fig.patch.set_facecolor(DARK)

    ax1 = axes[0]
    ax1.set_facecolor(DARK)
    y_pos = np.arange(n_rows)
    for i, (_, row) in enumerate(merged.iterrows()):
        w = row["blended_whiff"]
        swings = row.get("swings", 50) or 50
        color = EMBER if w >= 0.30 else SAGE if w >= 0.20 else GOLD
        alpha = min(1.0, 0.4 + 0.6 * (min(swings, 100) - 10) / 90)
        bar_w = w * 100
        rsize = min(0.3, bar_w / 5) if bar_w > 0 else 0.1
        bar = FancyBboxPatch(
            (0, y_pos[i] - 0.3), bar_w, 0.6,
            boxstyle=f"round,pad=0,rounding_size={rsize:.2f}",
            facecolor=color, alpha=alpha, zorder=2,
        )
        ax1.add_patch(bar)
        ax1.text(
            w * 100 + 1, y_pos[i], f"n={int(swings)}",
            color=SLATE, fontsize=7, va="center", alpha=0.7,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(merged["label"])
    ax1.set_xlim(0, max(merged["blended_whiff"].max() * 100 + 12, 45))
    ax1.set_ylim(-0.6, n_rows - 0.4)
    ax1.set_xlabel("Whiff Rate %", color=SLATE, fontsize=9)
    ax1.set_title("Vulnerability", color=EMBER, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=CREAM, labelsize=9)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2 = axes[1]
    ax2.set_facecolor(DARK)
    xlabel = "xwOBA on Contact"

    if "xwoba_contact" in merged.columns:
        vals = merged["xwoba_contact"].fillna(0)
    else:
        vals = pd.Series([0.0] * n_rows, index=merged.index)

    for i, (_, row) in enumerate(merged.iterrows()):
        v = vals.iloc[i]
        if v > 0 and pd.notna(v):
            bar_width = v * 100
            color = GOLD if v >= 0.400 else SAGE if v >= 0.340 else SLATE
            rsize = min(0.3, bar_width / 5)
            bar = FancyBboxPatch(
                (0, y_pos[i] - 0.3), bar_width, 0.6,
                boxstyle=f"round,pad=0,rounding_size={rsize:.2f}",
                facecolor=color, alpha=0.85, zorder=2,
            )
            ax2.add_patch(bar)

    ax2.set_yticks(y_pos)
    ax2.set_ylim(-0.6, n_rows - 0.4)
    max_val = vals.max() if vals.max() > 0 else 0.5
    x_max = min(max_val * 100 + 8, 100)
    ax2.set_xlim(0, x_max)
    if max_val > 0.6:
        tick_vals = [0, 0.200, 0.400, 0.600, 0.800]
    else:
        tick_vals = [0, 0.100, 0.200, 0.300, 0.400, 0.500]
    ax2.set_xticks([t * 100 for t in tick_vals])
    ax2.set_xticklabels([f".{int(t*1000):03d}" for t in tick_vals])
    ax2.set_xlabel(xlabel, color=SLATE, fontsize=9)
    ax2.set_title("Contact Quality", color=SAGE, fontsize=11, fontweight="bold")
    ax2.tick_params(colors=CREAM, labelsize=9)
    ax2.set_yticklabels([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{hitter_name} -- Pitch-Type Profile",
        color=CREAM, fontsize=12, fontweight="bold", y=1.02,
    )

    add_watermark(fig)
    fig.tight_layout()
    return fig


# Brand-adjacent palette for archetype slices (supports up to 8 archetypes)
_ARCHETYPE_COLORS = [
    "#C8A96E",  # gold
    "#D4562A",  # ember
    "#6BA38E",  # sage
    "#7B8FA6",  # slate
    "#E8D5B0",  # light gold
    "#A67B5B",  # brown
    "#8B6FA6",  # purple
    "#5BA3A3",  # teal
]


def create_archetype_donut_fig(
    arch_counts: dict[str, int],
    title: str = "",
) -> Figure:
    """Lineup archetype composition as a donut chart.

    Parameters
    ----------
    arch_counts : dict[str, int]
        Mapping of archetype name to count (e.g. {"Patient Power": 2, ...}).
    title : str, optional
        Unused (caller adds its own header); kept for API symmetry.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not arch_counts:
        fig = Figure(figsize=(4, 4))
        fig.patch.set_facecolor(DARK)
        return fig

    # Sort descending by count for consistent visual ordering
    sorted_items = sorted(arch_counts.items(), key=lambda x: -x[1])
    names = [name for name, _ in sorted_items]
    counts = [count for _, count in sorted_items]
    total = sum(counts)
    colors = _ARCHETYPE_COLORS[: len(names)]

    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor=DARK, linewidth=1.5),
    )

    # Annotate each wedge outside the donut
    for wedge, label, count, color in zip(wedges, names, counts, colors):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        ha = "left" if x >= 0 else "right"
        ax.annotate(
            f"{count}x {label}",
            xy=(0.7 * x, 0.7 * y),
            xytext=(1.15 * x, 1.15 * y),
            fontsize=8,
            color=color,
            fontweight="600",
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6),
        )

    # Center text: total count
    ax.text(
        0, 0.06, str(total),
        ha="center", va="center", fontsize=16, fontweight="bold", color=GOLD,
    )
    ax.text(
        0, -0.12, "hitters",
        ha="center", va="center", fontsize=8, color=SLATE,
    )

    fig.tight_layout()
    return fig
