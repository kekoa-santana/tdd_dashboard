"""Simulation distribution chart for pitcher/batter game projections."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

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


@st.fragment
def render_player_sim(
    samples_dict: dict[str, np.ndarray],
    stat_meta: dict[str, dict],
    stat_options: list[str],
    widget_key: str,
) -> None:
    """Render sim distribution chart + summary for one player."""
    import plotly.graph_objects as go

    tb_stat, tb_line, _ = st.columns([1, 2, 3])
    with tb_stat:
        stat_label = st.selectbox(
            "Stat", stat_options,
            key=f"psim_stat_{widget_key}", label_visibility="collapsed",
        )
    stat_key = stat_label.lower()
    stat_samples = samples_dict[stat_key]
    meta = stat_meta[stat_key]

    lo, hi = meta["lines"]
    line_opts = [x + 0.5 for x in range(int(lo - 0.5), int(hi - 0.5) + 1)]
    line_labels = [f"Over {v:.1f}" for v in line_opts]
    _mean = float(np.mean(stat_samples))
    _default_idx = 0
    for i, v in enumerate(line_opts):
        if v <= _mean:
            _default_idx = i
    with tb_line:
        sel_line = st.selectbox(
            "Line", line_labels, index=_default_idx,
            key=f"psim_line_{widget_key}", label_visibility="collapsed",
        )
    threshold = line_opts[line_labels.index(sel_line)]

    p_over = float(np.mean(stat_samples > threshold))
    p_under = 1.0 - p_over

    if p_over > 0.65:
        signal, sig_color = "Strong Over", SAGE
    elif p_over > 0.55:
        signal, sig_color = "Lean Over", SAGE
    elif p_over < 0.35:
        signal, sig_color = "Strong Under", EMBER
    elif p_over < 0.45:
        signal, sig_color = "Lean Under", EMBER
    else:
        signal, sig_color = "Toss-up", SLATE

    bar_color = meta.get("color", SAGE)
    max_val = int(stat_samples.max()) + 1
    bins = np.arange(0, max_val + 2)
    counts, _ = np.histogram(stat_samples, bins=bins - 0.5, density=True)
    over_counts = np.where(bins[:-1] > threshold, counts, 0)
    under_counts = np.where(bins[:-1] <= threshold, counts, 0)

    pfig = go.Figure()
    pfig.add_trace(go.Bar(
        x=bins[:-1], y=under_counts, name="Under",
        marker_color="rgba(123,143,166,0.53)", width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_trace(go.Bar(
        x=bins[:-1], y=over_counts, name="Over",
        marker_color=bar_color, width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_vline(
        x=threshold, line_dash="dash", line_color=GOLD, line_width=2,
        annotation_text=f"P(Over {threshold:.1f}) = {p_over:.1%}",
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

    mean_val = float(np.mean(stat_samples))
    std_val = float(np.std(stat_samples))
    p_hi = float((stat_samples >= meta["hi"]).sum() / len(stat_samples) * 100)
    p_vhi = float((stat_samples >= meta["vhi"]).sum() / len(stat_samples) * 100)
    _render_sim_summary(
        widget_key, stat_key, signal, sig_color, p_over, p_under, threshold,
        meta, mean_val, std_val, p_hi, p_vhi,
    )


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


@st.fragment
def render_player_sim_from_props(
    rows: pd.DataFrame,
    stat_meta: dict[str, dict],
    stat_options: list[str],
    widget_key: str,
) -> None:
    """Render distribution chart using game_props p_over_X.X columns.

    `rows` is a DataFrame slice with one row per stat for a single player.
    The PMF is reconstructed from the p_over half-integer columns so the
    chart matches what Props Lab shows (same source of truth).
    """
    import plotly.graph_objects as go

    tb_stat, tb_line, _ = st.columns([1, 2, 3])
    with tb_stat:
        stat_label = st.selectbox(
            "Stat", stat_options,
            key=f"psim_stat_{widget_key}", label_visibility="collapsed",
        )
    stat_key = stat_label.lower()
    meta = stat_meta[stat_key]

    # rows.stat uses the confident_picks labels ("K", "BB", "H", "HR", "Outs")
    stat_label_norm = stat_label
    row_match = rows[rows["stat"].str.lower() == stat_key]
    if row_match.empty:
        st.caption(f"No {stat_label_norm} projection for this player.")
        return
    row = row_match.iloc[0]

    lo, hi = meta["lines"]
    line_opts = [x + 0.5 for x in range(int(lo - 0.5), int(hi - 0.5) + 1)]
    line_labels = [f"Over {v:.1f}" for v in line_opts]
    mean_val = float(row.get("expected", 0.0))
    _default_idx = 0
    for i, v in enumerate(line_opts):
        if v <= mean_val:
            _default_idx = i
    with tb_line:
        sel_line = st.selectbox(
            "Line", line_labels, index=_default_idx,
            key=f"psim_line_{widget_key}", label_visibility="collapsed",
        )
    threshold = line_opts[line_labels.index(sel_line)]

    col = f"p_over_{threshold:.1f}"
    p_over = float(row[col]) if col in row.index and pd.notna(row.get(col)) else 0.0
    p_under = 1.0 - p_over

    if p_over > 0.65:
        signal, sig_color = "Strong Over", SAGE
    elif p_over > 0.55:
        signal, sig_color = "Lean Over", SAGE
    elif p_over < 0.35:
        signal, sig_color = "Strong Under", EMBER
    elif p_over < 0.45:
        signal, sig_color = "Lean Under", EMBER
    else:
        signal, sig_color = "Toss-up", SLATE

    bar_color = meta.get("color", SAGE)
    xs, pmf = _pmf_from_p_over(row)
    over_y = np.where(xs > threshold, pmf, 0)
    under_y = np.where(xs <= threshold, pmf, 0)

    pfig = go.Figure()
    pfig.add_trace(go.Bar(
        x=xs, y=under_y, name="Under",
        marker_color="rgba(123,143,166,0.53)", width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_trace(go.Bar(
        x=xs, y=over_y, name="Over",
        marker_color=bar_color, width=0.85,
        hovertemplate="%{x} " + meta["label"] + ": %{y:.1%}<extra></extra>",
    ))
    pfig.add_vline(
        x=threshold, line_dash="dash", line_color=GOLD, line_width=2,
        annotation_text=f"P(Over {threshold:.1f}) = {p_over:.1%}",
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

    std_val = float(row.get("std", 0.0))
    # P(X >= hi) via the nearest half-line below hi
    hi_col = f"p_over_{meta['hi'] - 0.5:.1f}"
    vhi_col = f"p_over_{meta['vhi'] - 0.5:.1f}"
    p_hi = float(row[hi_col]) * 100 if hi_col in row.index and pd.notna(row.get(hi_col)) else 0.0
    p_vhi = float(row[vhi_col]) * 100 if vhi_col in row.index and pd.notna(row.get(vhi_col)) else 0.0
    _render_sim_summary(
        widget_key, stat_key, signal, sig_color, p_over, p_under, threshold,
        meta, mean_val, std_val, p_hi, p_vhi,
    )


def _render_sim_summary(
    widget_key: str, stat_key: str, signal: str, sig_color: str,
    p_over: float, p_under: float, threshold: float,
    meta: dict, mean_val: float, std_val: float, p_hi: float, p_vhi: float,
) -> None:

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0;">'
        f'<span style="background:{sig_color}22; color:{sig_color}; '
        f'border:1px solid {sig_color}44; padding:3px 10px; border-radius:12px; '
        f'font-size:0.82rem; font-weight:700;">{signal}</span>'
        f'<span style="color:var(--tdd-cream); font-size:1rem; font-weight:700;">'
        f'{p_over:.1%}</span>'
        f'<span style="color:var(--tdd-slate); font-size:0.75rem;">'
        f'Over {threshold:.1f}</span>'
        f'<span style="color:var(--tdd-cream); font-size:1rem; font-weight:700;">'
        f'{p_under:.1%}</span>'
        f'<span style="color:var(--tdd-slate); font-size:0.75rem;">'
        f'Under {threshold:.1f}</span>'
        f'</div>'
        f'<div style="color:var(--tdd-slate); font-size:0.78rem; margin-top:0.3rem;">'
        f'E[{meta["label"]}] {mean_val:.1f} (\u00b1{std_val:.1f})'
        f' | {p_hi:.0f}% chance {meta["hi"]}+'
        f' | {p_vhi:.0f}% chance {meta["vhi"]}+'
        f'</div>',
        unsafe_allow_html=True,
    )
