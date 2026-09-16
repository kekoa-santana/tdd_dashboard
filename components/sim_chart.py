"""Simulation distribution chart for pitcher/batter game projections."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components.projection_table import with_outcome_ranges
from config import GOLD, EMBER, SAGE, SLATE, CREAM

PITCHER_STAT_META = {
    "k":    {"label": "K",    "xlabel": "Pitcher Strikeouts", "color": SAGE,  "lines": (3.5, 10.5), "hi": 6, "vhi": 8},
    "bb":   {"label": "BB",   "xlabel": "Walks Issued",      "color": EMBER, "lines": (1.5, 5.5),  "hi": 3, "vhi": 4},
    "h":    {"label": "H",    "xlabel": "Hits Allowed",      "color": SLATE, "lines": (3.5, 9.5),  "hi": 6, "vhi": 8},
    "hr":   {"label": "HR",   "xlabel": "Home Runs Allowed", "color": GOLD,  "lines": (0.5, 2.5),  "hi": 1, "vhi": 2},
    "outs": {"label": "Outs", "xlabel": "Outs Recorded",     "color": SLATE, "lines": (11.5, 21.5), "hi": 17, "vhi": 20},
}

BATTER_STAT_META = {
    "k":  {"label": "K",  "xlabel": "Batter Strikeouts", "color": EMBER, "lines": (0.5, 3.5), "hi": 2, "vhi": 3},
    "bb": {"label": "BB", "xlabel": "Batter Walks",      "color": SAGE,  "lines": (0.5, 2.5), "hi": 1, "vhi": 2},
    "h":  {"label": "H",  "xlabel": "Batter Hits",       "color": SAGE,  "lines": (0.5, 3.5), "hi": 2, "vhi": 3},
    "hr": {"label": "HR", "xlabel": "Batter Home Runs",  "color": GOLD,  "lines": (0.5, 1.5), "hi": 1, "vhi": 2},
}


def _pmf_from_p_over(row: pd.Series, max_k: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct integer PMF from p_over_{k-0.5} columns.

    P(X=0) = 1 - p_over_0.5
    P(X=k) = p_over_{k-0.5} - p_over_{k+0.5}
    """
    p_over: list[float] = []
    for k in range(max_k + 1):
        col = f"p_over_{k + 0.5:.1f}"
        val = row.get(col)
        if val is None or pd.isna(val):
            break
        p_over.append(float(val))
    if not p_over:
        return np.array([0]), np.array([1.0])
    # P(X=0) = 1 - p_over_0.5; P(X=k) = p_over_{k-0.5} - p_over_{k+0.5}
    pmf = [1.0 - p_over[0]]
    for i in range(len(p_over) - 1):
        pmf.append(max(p_over[i] - p_over[i + 1], 0.0))
    pmf.append(max(p_over[-1], 0.0))  # tail >= last line
    pmf_arr = np.array(pmf, dtype=float)
    total = pmf_arr.sum()
    if total > 0:
        pmf_arr = pmf_arr / total
    # Trim trailing near-zero bins for a cleaner chart
    last_nonzero = np.where(pmf_arr > 1e-4)[0]
    cutoff = int(last_nonzero[-1]) + 1 if len(last_nonzero) else len(pmf_arr)
    return np.arange(cutoff), pmf_arr[:cutoff]


def _p_at_least(row: pd.Series, count: int) -> float:
    """P(X >= count), read off the p_over half-line below ``count``."""
    if count <= 0:
        return 1.0
    col = f"p_over_{count - 0.5:.1f}"
    value = row.get(col)
    return float(value) if value is not None and pd.notna(value) else 0.0


@st.fragment
def render_player_sim_from_props(
    rows: pd.DataFrame,
    stat_meta: dict[str, dict],
    stat_options: list[str],
    widget_key: str,
) -> None:
    """Render the projected outcome distribution for one player.

    `rows` is a DataFrame slice with one row per stat for a single player.
    The PMF is reconstructed from the p_over half-integer columns so the
    chart matches what Player Projections shows (same source of truth).
    """
    import plotly.graph_objects as go

    tb_stat, tb_count, _ = st.columns([1, 2, 3])
    with tb_stat:
        stat_label = st.selectbox(
            "Stat", stat_options,
            key=f"psim_stat_{widget_key}", label_visibility="collapsed",
        )
    stat_key = stat_label.lower()
    meta = stat_meta[stat_key]

    # rows.stat uses the confident_picks labels ("K", "BB", "H", "HR", "Outs")
    row_match = rows[rows["stat"].str.lower() == stat_key]
    if row_match.empty:
        st.caption(f"No {stat_label} projection for this player.")
        return
    row = row_match.iloc[0]

    lo, hi = meta["lines"]
    count_opts = list(range(int(lo + 0.5), int(hi + 0.5) + 1))
    count_labels = [f"{c}+ {meta['label']}" for c in count_opts]
    mean_val = float(row.get("expected", 0.0))
    default_idx = 0
    for i, c in enumerate(count_opts):
        if c <= mean_val:
            default_idx = i
    with tb_count:
        selected = st.selectbox(
            "Chance of", count_labels, index=default_idx,
            key=f"psim_count_{widget_key}", label_visibility="collapsed",
        )
    count = count_opts[count_labels.index(selected)]
    p_at_least = _p_at_least(row, count)

    bar_color = meta.get("color", SAGE)
    xs, pmf = _pmf_from_p_over(row)
    at_least_y = np.where(xs >= count, pmf, 0)
    below_y = np.where(xs < count, pmf, 0)

    pfig = go.Figure()
    pfig.add_trace(go.Bar(
        x=xs, y=below_y, name=f"under {count}",
        marker_color="rgba(123,143,166,0.53)", width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_trace(go.Bar(
        x=xs, y=at_least_y, name=f"{count}+",
        marker_color=bar_color, width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_vline(
        x=mean_val, line_dash="dash", line_color=GOLD, line_width=2,
        annotation_text=f"Projected {mean_val:.1f}",
        annotation_position="top left",
        annotation_font=dict(color=GOLD, size=12),
    )
    pfig.update_layout(
        barmode="stack", showlegend=False, dragmode=False,
        xaxis_title=meta.get("xlabel", meta["label"]), yaxis_title="",
        yaxis=dict(showticklabels=False, showgrid=False, fixedrange=True),
        xaxis=dict(dtick=1, gridcolor="rgba(0,0,0,0)", fixedrange=True, rangemode="nonnegative"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CREAM),
        margin=dict(l=10, r=10, t=30, b=40),
        height=260,
    )
    st.plotly_chart(pfig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False}, key=f"psim_chart_{widget_key}_{stat_key}")

    ranged = with_outcome_ranges(row_match.iloc[[0]]).iloc[0]
    _render_sim_summary(
        meta=meta,
        mean_val=mean_val,
        std_val=float(row.get("std", 0.0)),
        count=count,
        p_at_least=p_at_least,
        range_lo=ranged.get("range_lo"),
        range_hi=ranged.get("range_hi"),
        p_hi=_p_at_least(row, meta["hi"]) * 100,
        p_vhi=_p_at_least(row, meta["vhi"]) * 100,
    )


def _render_sim_summary(
    *, meta: dict, mean_val: float, std_val: float, count: int,
    p_at_least: float, range_lo, range_hi, p_hi: float, p_vhi: float,
) -> None:
    """Projection headline: expected value, likely range, and one chance."""
    range_html = ""
    if pd.notna(range_lo) and pd.notna(range_hi):
        range_html = (
            f'<span style="color:var(--tdd-slate); font-size:0.75rem;">'
            f'likely {range_lo:.0f}-{range_hi:.0f}</span>'
        )
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0; flex-wrap:wrap;">'
        f'<span style="color:var(--tdd-cream); font-size:1rem; font-weight:700;">'
        f'{mean_val:.1f} {meta["label"]}</span>'
        f'<span style="color:var(--tdd-slate); font-size:0.75rem;">projected</span>'
        f'{range_html}'
        f'<span style="color:var(--tdd-cream); font-size:1rem; font-weight:700;">'
        f'{p_at_least:.1%}</span>'
        f'<span style="color:var(--tdd-slate); font-size:0.75rem;">'
        f'chance of {count}+</span>'
        f'</div>'
        f'<div style="color:var(--tdd-slate); font-size:0.78rem; margin-top:0.3rem;">'
        f'Spread ±{std_val:.1f}'
        f' | {p_hi:.0f}% chance {meta["hi"]}+'
        f' | {p_vhi:.0f}% chance {meta["vhi"]}+'
        f'</div>',
        unsafe_allow_html=True,
    )
