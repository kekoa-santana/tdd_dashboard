"""Backtest visualization charts for the Model Performance page."""
from __future__ import annotations

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from config import GOLD, EMBER, SAGE, SLATE, CREAM, DARK
from lib.theme import add_watermark


def create_accuracy_bars(
    df: pd.DataFrame,
    bayes_col: str,
    marcel_col: str,
    title: str,
) -> Figure:
    """Grouped bars: Bayes vs Marcel by test_season.

    Parameters
    ----------
    df : DataFrame with ``test_season``, *bayes_col*, and *marcel_col* columns.
    bayes_col / marcel_col : column names for the two systems.
    title : chart title string.
    """
    seasons = df["test_season"].astype(str).tolist()
    bayes_vals = df[bayes_col].tolist()
    marcel_vals = df[marcel_col].tolist()

    x = np.arange(len(seasons))
    width = 0.35

    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    ax.bar(x - width / 2, bayes_vals, width, label="Bayes", color=SAGE, alpha=0.85)
    ax.bar(x + width / 2, marcel_vals, width, label="Marcel", color=SLATE, alpha=0.85)

    # Value labels
    for i, (bv, mv) in enumerate(zip(bayes_vals, marcel_vals)):
        ax.text(i - width / 2, bv, f"{bv:.3f}", ha="center", va="bottom",
                color=SAGE, fontsize=8, fontweight="bold")
        ax.text(i + width / 2, mv, f"{mv:.3f}", ha="center", va="bottom",
                color=SLATE, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_title(title, color=CREAM, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def create_coverage_chart(
    df: pd.DataFrame,
    coverage_cols: list[str],
    labels: list[str],
) -> Figure:
    """Horizontal bars showing CI coverage vs expected targets.

    Parameters
    ----------
    df : DataFrame (one row per test_season).
    coverage_cols : list of column names holding coverage fractions.
    labels : display labels for each coverage level (e.g. ["95% CI"]).
    """
    # Average across seasons for a single summary bar per level
    avgs = [df[col].mean() for col in coverage_cols]
    targets = []
    for lbl in labels:
        if "95" in lbl:
            targets.append(0.95)
        elif "80" in lbl:
            targets.append(0.80)
        elif "90" in lbl:
            targets.append(0.90)
        elif "50" in lbl:
            targets.append(0.50)
        else:
            targets.append(0.95)

    y = np.arange(len(labels))
    fig = Figure(figsize=(7, max(2, len(labels) * 0.8)))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    colors = [SAGE if avg >= tgt * 0.95 else EMBER for avg, tgt in zip(avgs, targets)]
    ax.barh(y, [a * 100 for a in avgs], height=0.5, color=colors, alpha=0.85)

    for i, (avg, tgt) in enumerate(zip(avgs, targets)):
        ax.plot([tgt * 100, tgt * 100], [i - 0.35, i + 0.35],
                color=GOLD, linewidth=2, linestyle="--")
        ax.text(avg * 100 + 1, i, f"{avg*100:.1f}%", va="center",
                color=CREAM, fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Coverage %", color=SLATE, fontsize=10)
    ax.set_title("Interval Coverage (dashed = target)", color=CREAM,
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=CREAM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def create_game_k_model_comparison(df: pd.DataFrame) -> Figure:
    """Grouped bars comparing 4 model tiers by Brier score.

    Expects columns: naive_avg_brier, poisson_avg_brier,
    model_no_matchup_avg_brier, full_model_avg_brier.
    """
    tier_cols = [
        ("naive_avg_brier", "Naive"),
        ("poisson_avg_brier", "Poisson"),
        ("model_no_matchup_avg_brier", "No Matchup"),
        ("full_model_avg_brier", "Full Model"),
    ]

    seasons = df["test_season"].astype(str).tolist()
    n_seasons = len(seasons)
    n_tiers = len(tier_cols)
    width = 0.18
    x = np.arange(n_seasons)

    tier_colors = [SLATE, EMBER, GOLD, SAGE]

    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    for j, ((col, label), color) in enumerate(zip(tier_cols, tier_colors)):
        if col not in df.columns:
            continue
        offset = (j - n_tiers / 2 + 0.5) * width
        vals = df[col].tolist()
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=color, fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Avg Brier Score", color=SLATE, fontsize=10)
    ax.set_title("Game K Model Comparison by Brier Score", color=CREAM,
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def create_movers_chart(
    names: list[str],
    deltas: list[float],
    title: str,
    positive_color: str = SAGE,
    negative_color: str = EMBER,
) -> Figure:
    """Horizontal lollipop chart showing biggest projection movers.

    Parameters
    ----------
    names : player names (top to bottom).
    deltas : change values in percentage points.
    title : chart title.
    positive_color / negative_color : bar colors.
    """
    n = len(names)
    fig = Figure(figsize=(7, max(2.5, n * 0.35)))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    y = np.arange(n)
    colors = [positive_color if d >= 0 else negative_color for d in deltas]

    ax.barh(y, deltas, height=0.5, color=colors, alpha=0.85)

    for i, (d, name) in enumerate(zip(deltas, names)):
        align = "left" if d >= 0 else "right"
        offset = 0.1 if d >= 0 else -0.1
        ax.text(d + offset, i, f"{d:+.1f}pp", ha=align, va="center",
                color=CREAM, fontsize=8, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color=SLATE, linewidth=0.5, alpha=0.5)
    ax.set_title(title, color=CREAM, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Change (pp)", color=SLATE, fontsize=9)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.invert_yaxis()
    add_watermark(fig)
    fig.tight_layout()
    return fig


def create_projection_timeline(
    snapshots: dict[str, pd.DataFrame],
    player_id: int,
    id_col: str,
    name_col: str,
    stat_cols: list[str],
) -> Figure | None:
    """Line chart of projected rate over time with CI band.

    Parameters
    ----------
    snapshots : {date_str: df} dict from weekly snapshot loader.
    player_id : player to plot.
    id_col / name_col : column names for player ID and display name.
    stat_cols : list of stat column names to plot (e.g. ["projected_k_rate"]).
    """
    dates = sorted(snapshots.keys())
    if not dates:
        return None

    stat_data: dict[str, list[float]] = {s: [] for s in stat_cols}
    valid_dates: list[str] = []
    player_name = ""

    for d in dates:
        df = snapshots[d]
        row = df[df[id_col] == player_id]
        if row.empty:
            continue
        valid_dates.append(d)
        if not player_name and name_col in row.columns:
            player_name = str(row.iloc[0][name_col])
        for s in stat_cols:
            stat_data[s].append(float(row.iloc[0].get(s, 0)))

    if not valid_dates:
        return None

    colors = [SAGE, GOLD, EMBER, SLATE]
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    for i, s in enumerate(stat_cols):
        color = colors[i % len(colors)]
        vals = [v * 100 for v in stat_data[s]]
        ax.plot(range(len(valid_dates)), vals, color=color,
                linewidth=2, marker="o", markersize=5, label=s.replace("projected_", ""))

        # CI band if available
        lo_col = s + "_2_5"
        hi_col = s + "_97_5"
        lo_vals, hi_vals = [], []
        for d in valid_dates:
            row = snapshots[d][snapshots[d][id_col] == player_id]
            if not row.empty and lo_col in row.columns and hi_col in row.columns:
                lo_vals.append(float(row.iloc[0][lo_col]) * 100)
                hi_vals.append(float(row.iloc[0][hi_col]) * 100)
        if len(lo_vals) == len(valid_dates):
            ax.fill_between(range(len(valid_dates)), lo_vals, hi_vals,
                            alpha=0.15, color=color)

    ax.set_xticks(range(len(valid_dates)))
    ax.set_xticklabels(valid_dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rate (%)", color=SLATE, fontsize=10)
    title = f"{player_name} -- Projection Timeline" if player_name else "Projection Timeline"
    ax.set_title(title, color=CREAM, fontsize=12, fontweight="bold", pad=10)
    if len(stat_cols) > 1:
        ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_watermark(fig)
    fig.tight_layout()
    return fig
