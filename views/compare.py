"""Player Comparison page -- side-by-side projection comparison."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, SLATE, CREAM, DARK, DARK_BORDER,
    CURRENT_SEASON, PROJECTION_LABEL,
    PITCHER_STATS, HITTER_STATS,
    PITCHER_OBSERVED_STATS, HITTER_OBSERVED_STATS,
    PITCHER_COUNTING_DISPLAY, HITTER_COUNTING_DISPLAY,
)
from services.data_loader import (
    load_projections,
    load_counting,
    load_rankings,
    load_hitter_archetypes,
    load_pitcher_archetypes,
)
from utils.helpers import get_team_lookup
from utils.formatters import fmt_stat
from components.headshot import headshot_html
from components.metric_cards import percentile_rank, pctile_color
from components.diamond_rating import diamond_rating_html_composite


# ---------------------------------------------------------------------------
# Stat row renderers
# ---------------------------------------------------------------------------

def _stat_row_html(label: str, values: list[str], colors: list[str]) -> str:
    """Build a single comparison row: label on the left, values in columns."""
    val_cells = "".join(
        f'<td style="text-align:center; padding:6px 8px; color:{c}; font-weight:600;">'
        f'{v}</td>'
        for v, c in zip(values, colors)
    )
    return (
        f'<tr>'
        f'<td style="padding:6px 8px; color:var(--tdd-slate); white-space:nowrap;">{label}</td>'
        f'{val_cells}'
        f'</tr>'
    )


def _pctile_bar_inline(pctile: float, width_px: int = 120) -> str:
    """Tiny inline percentile bar for comparison tables."""
    color = pctile_color(pctile)
    fill_w = int(pctile / 100 * width_px)
    return (
        f'<div style="display:inline-flex; align-items:center; gap:6px;">'
        f'<div style="width:{width_px}px; height:6px; background:{DARK_BORDER}; '
        f'border-radius:3px; overflow:hidden;">'
        f'<div style="width:{fill_w}px; height:100%; background:{color}; '
        f'border-radius:3px;"></div>'
        f'</div>'
        f'<span style="color:{color}; font-size:0.8rem; font-weight:600;">'
        f'{pctile:.0f}th</span>'
        f'</div>'
    )


def _highlight_best(
    values: list[float | None],
    higher_is_better: bool,
) -> list[str]:
    """Return a list of colors: gold for best value, cream for others."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None and pd.notna(v)]
    if not valid:
        return [CREAM] * len(values)
    if higher_is_better:
        best_idx = max(valid, key=lambda x: x[1])[0]
    else:
        best_idx = min(valid, key=lambda x: x[1])[0]
    return [GOLD if i == best_idx else CREAM for i in range(len(values))]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_compare() -> None:
    st.markdown(
        '<div class="brand-header">'
        '<div><div class="brand-title">Compare Players</div>'
        '<div class="brand-subtitle">'
        'Side-by-side Bayesian projection comparison for 2-3 players</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Read query params for deep linking
    qp_type = st.query_params.get("player_type", "")
    qp_ids = st.query_params.get("player_ids", "")

    type_options = ["Pitcher", "Hitter"]
    default_type_idx = 1 if qp_type.lower() == "hitter" else 0

    player_type = st.radio(
        "Player type", type_options,
        index=default_type_idx,
        horizontal=True,
        key="cmp_type",
    )

    proj_df = load_projections(player_type.lower())
    if proj_df.empty:
        st.info("No projection data available.")
        return

    if player_type == "Pitcher":
        name_col, id_col = "pitcher_name", "pitcher_id"
    else:
        name_col, id_col = "batter_name", "batter_id"

    team_lookup = get_team_lookup()

    # Build display name -> player_id mapping
    display_map: dict[str, int] = {}
    for _, row in proj_df.iterrows():
        pid = int(row[id_col])
        pname = row[name_col]
        team = team_lookup.get(pid, "")
        dname = f"{pname} ({team})" if team else pname
        display_map[dname] = pid

    sorted_names = sorted(display_map.keys())

    # Resolve deep-linked player IDs to display names
    default_selections: list[str] = []
    if qp_ids:
        for pid_str in qp_ids.split(","):
            pid_str = pid_str.strip()
            if pid_str.isdigit():
                pid_int = int(pid_str)
                for dname, pid_val in display_map.items():
                    if pid_val == pid_int:
                        default_selections.append(dname)
                        break

    selected = st.multiselect(
        "Select 2-3 players to compare",
        sorted_names,
        default=default_selections[:3],
        max_selections=3,
        key="cmp_players",
    )

    if len(selected) < 2:
        st.info("Select at least 2 players to compare.")
        return

    selected_ids = [display_map[s] for s in selected]

    # Write query params for deep linking
    st.query_params["player_type"] = player_type.lower()
    st.query_params["player_ids"] = ",".join(str(pid) for pid in selected_ids)

    # Gather data for selected players
    players = proj_df[proj_df[id_col].isin(selected_ids)].copy()

    # Load supplementary data
    counting_df = load_counting(player_type.lower())
    arch_df = (
        load_pitcher_archetypes() if player_type == "Pitcher"
        else load_hitter_archetypes()
    )

    stat_configs = PITCHER_STATS if player_type == "Pitcher" else HITTER_STATS
    obs_configs = (
        PITCHER_OBSERVED_STATS if player_type == "Pitcher"
        else HITTER_OBSERVED_STATS
    )
    counting_display = (
        PITCHER_COUNTING_DISPLAY if player_type == "Pitcher"
        else HITTER_COUNTING_DISPLAY
    )

    # Load rankings for role/position rank display
    _rank_type = "pitchers" if player_type == "Pitcher" else "hitters"
    _rank_df = load_rankings(_rank_type)
    _rank_id_col = "pitcher_id" if player_type == "Pitcher" else "batter_id"

    # ----- Player header cards -----
    cols = st.columns(len(selected))
    for col_st, display_name in zip(cols, selected):
        pid = display_map[display_name]
        match = players[players[id_col] == pid]
        if match.empty:
            continue
        row = match.iloc[0]

        with col_st:
            # Headshot + name + team
            team = team_lookup.get(pid, "")
            st.markdown(
                f'<div style="text-align:center; padding:8px 0;">'
                f'{headshot_html(pid, size=80)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div class="tdd-player-name">{row[name_col]}</div>'
                f'<div class="tdd-meta" data-team="{team}">{team}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Diamond rating
            composite = row.get("composite_score")
            if pd.notna(composite):
                diamonds_html = diamond_rating_html_composite(composite, size="md")
                st.markdown(
                    f'<div style="text-align:center; margin:4px 0;">'
                    f'{diamonds_html}</div>',
                    unsafe_allow_html=True,
                )

            # Age + hand/role info (with positional rank)
            age = int(row["age"]) if pd.notna(row.get("age")) else "?"
            info_parts = [f"Age {age}"]
            _rk_row = _rank_df[_rank_df[_rank_id_col] == pid] if not _rank_df.empty else pd.DataFrame()
            if player_type == "Pitcher":
                hand = row.get("pitch_hand", "")
                if hand:
                    info_parts.append(f"{'LHP' if hand == 'L' else 'RHP'}")
                if "is_starter" in row.index:
                    _role_label = "SP" if row["is_starter"] else "RP"
                    if not _rk_row.empty:
                        _rr = _rk_row.iloc[0].get("role_rank")
                        if pd.notna(_rr):
                            _role_label = f"{_role_label} #{int(_rr)}"
                    info_parts.append(_role_label)
            else:
                stand = row.get("batter_stand", "")
                if stand:
                    info_parts.append(f"Bats {stand}")
                if not _rk_row.empty:
                    _pos = _rk_row.iloc[0].get("position", "")
                    _pr = _rk_row.iloc[0].get("pos_rank")
                    if _pos and pd.notna(_pr):
                        info_parts.append(f"{_pos} #{int(_pr)}")

            st.markdown(
                f'<div class="tdd-meta" style="text-align:center; '
                f'margin-bottom:4px;">{" | ".join(info_parts)}</div>',
                unsafe_allow_html=True,
            )

            # Archetype badge
            if not arch_df.empty:
                arch_id_col = "pitcher_id" if player_type == "Pitcher" else "batter_id"
                arch_match = arch_df[arch_df[arch_id_col] == pid]
                if not arch_match.empty:
                    arch_name = arch_match.iloc[0]["archetype_name"]
                    st.markdown(
                        f'<div style="text-align:center;">'
                        f'<span class="tdd-badge">{arch_name}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ----- Comparison table -----
    st.markdown("---")

    # Build ordered list of player rows (matching selected order)
    ordered_rows: list[pd.Series] = []
    for pid in selected_ids:
        match = players[players[id_col] == pid]
        if not match.empty:
            ordered_rows.append(match.iloc[0])

    n_players = len(ordered_rows)

    # Table header
    header_cells = "".join(
        f'<th style="text-align:center; padding:6px 8px; color:var(--tdd-gold); '
        f'font-weight:700; font-size:0.9rem;">{r[name_col]}</th>'
        for r in ordered_rows
    )
    table_header = (
        f'<tr>'
        f'<th style="padding:6px 8px; color:var(--tdd-slate);">Stat</th>'
        f'{header_cells}'
        f'</tr>'
    )

    table_rows: list[str] = []

    # Section: Bayesian Projected Rates
    table_rows.append(
        f'<tr><td colspan="{n_players + 1}" class="tdd-section-hdr" '
        f'style="padding:10px 8px 4px; '
        f'border-bottom:1px solid var(--tdd-dark-border);">'
        f'{PROJECTION_LABEL} Rates</td></tr>'
    )

    for label, key, higher_better, desc in stat_configs:
        proj_col = f"projected_{key}"
        vals: list[float | None] = []
        display_vals: list[str] = []
        for r in ordered_rows:
            v = r.get(proj_col)
            if pd.notna(v):
                vals.append(float(v))
                display_vals.append(fmt_stat(v, key))
            else:
                vals.append(None)
                display_vals.append("")

        # Percentile sub-row
        pctile_vals: list[str] = []
        for r in ordered_rows:
            v = r.get(proj_col)
            if pd.notna(v) and proj_col in proj_df.columns:
                pct = percentile_rank(proj_df[proj_col], float(v), higher_better)
                pctile_vals.append(_pctile_bar_inline(pct))
            else:
                pctile_vals.append("")

        colors = _highlight_best(vals, higher_better)
        table_rows.append(_stat_row_html(label, display_vals, colors))
        # Percentile row
        table_rows.append(_stat_row_html(
            "",
            pctile_vals,
            [SLATE] * n_players,
        ))

    # Pitcher extra: FIP/ERA projections
    if player_type == "Pitcher":
        for label, col_key in [("Proj. FIP", "projected_fip"), ("Proj. ERA", "projected_era")]:
            vals = []
            display_vals = []
            for r in ordered_rows:
                v = r.get(col_key)
                if pd.notna(v):
                    vals.append(float(v))
                    display_vals.append(f"{v:.2f}")
                else:
                    vals.append(None)
                    display_vals.append("")
            colors = _highlight_best(vals, higher_is_better=False)
            table_rows.append(_stat_row_html(label, display_vals, colors))

    # Hitter extra: wOBA projection
    if player_type == "Hitter":
        for label, col_key, hib in [("Proj. wOBA", "projected_woba", True)]:
            vals = []
            display_vals = []
            for r in ordered_rows:
                v = r.get(col_key)
                if pd.notna(v):
                    vals.append(float(v))
                    display_vals.append(f"{v:.3f}")
                else:
                    vals.append(None)
                    display_vals.append("")
            colors = _highlight_best(vals, higher_is_better=hib)
            table_rows.append(_stat_row_html(label, display_vals, colors))

    # Section: Observed Skill Metrics
    table_rows.append(
        f'<tr><td colspan="{n_players + 1}" class="tdd-section-hdr" '
        f'style="padding:10px 8px 4px; '
        f'border-bottom:1px solid var(--tdd-dark-border);">'
        f'Observed Skill Profile</td></tr>'
    )

    for label, key, higher_better, desc in obs_configs:
        vals = []
        display_vals = []
        for r in ordered_rows:
            v = r.get(key)
            if pd.notna(v):
                vals.append(float(v))
                display_vals.append(fmt_stat(v, key))
            else:
                vals.append(None)
                display_vals.append("")
        colors = _highlight_best(vals, higher_better)
        table_rows.append(_stat_row_html(label, display_vals, colors))

    # Section: Counting Projections
    if not counting_df.empty:
        counting_rows_exist = False
        counting_row_html: list[str] = []

        for c_label, c_prefix, _c_actual, c_hb in counting_display:
            mean_col = f"{c_prefix}_mean"
            p10_col = f"{c_prefix}_p10"
            p90_col = f"{c_prefix}_p90"

            vals = []
            display_vals = []
            for r in ordered_rows:
                pid = int(r[id_col])
                c_match = counting_df[counting_df[id_col] == pid]
                if not c_match.empty and mean_col in c_match.columns:
                    c_data = c_match.iloc[0]
                    v = c_data.get(mean_col)
                    if pd.notna(v):
                        mean_v = int(round(v))
                        lo = int(round(c_data.get(p10_col, v)))
                        hi = int(round(c_data.get(p90_col, v)))
                        vals.append(float(mean_v))
                        display_vals.append(
                            f'{mean_v} <span class="tdd-context">'
                            f'({lo}-{hi})</span>'
                        )
                        counting_rows_exist = True
                        continue
                vals.append(None)
                display_vals.append("")

            colors = _highlight_best(vals, c_hb)
            counting_row_html.append(_stat_row_html(c_label, display_vals, colors))

        if counting_rows_exist:
            table_rows.append(
                f'<tr><td colspan="{n_players + 1}" class="tdd-section-hdr" '
                f'style="padding:10px 8px 4px; '
                f'border-bottom:1px solid var(--tdd-dark-border);">'
                f'{PROJECTION_LABEL} Counting Stats</td></tr>'
            )
            table_rows.extend(counting_row_html)

    # Assemble table
    all_rows = "\n".join(table_rows)
    table_html = (
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%; border-collapse:collapse; '
        f'background:transparent; border:none; '
        f'border-bottom:1px solid var(--tdd-dark-border); overflow:hidden;">'
        f'<thead style="background:{DARK};">{table_header}</thead>'
        f'<tbody>{all_rows}</tbody>'
        f'</table>'
        f'</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # ----- Percentile Comparison Bars -----
    st.markdown(
        f'<div style="margin-top:1.5rem;">'
        f'<div class="section-header">'
        f'Projected Percentile Comparison</div></div>',
        unsafe_allow_html=True,
    )

    for label, key, higher_better, desc in stat_configs:
        proj_col = f"projected_{key}"
        bar_parts: list[str] = []

        for r in ordered_rows:
            v = r.get(proj_col)
            pname = r[name_col]
            if pd.notna(v) and proj_col in proj_df.columns:
                pct = percentile_rank(proj_df[proj_col], float(v), higher_better)
                color = pctile_color(pct)
                fill_w = pct
                bar_parts.append(
                    f'<div style="margin-bottom:4px;">'
                    f'<div style="display:flex; align-items:center; gap:8px;">'
                    f'<span style="color:var(--tdd-cream); font-size:0.8rem; min-width:4rem; '
                    f'text-align:right; flex-shrink:1;">{pname}</span>'
                    f'<div style="flex:1; height:10px; background:{DARK_BORDER}; '
                    f'border-radius:5px; overflow:hidden;">'
                    f'<div style="width:{fill_w:.0f}%; height:100%; background:{color}; '
                    f'border-radius:5px;"></div>'
                    f'</div>'
                    f'<span style="color:{color}; font-size:0.8rem; font-weight:600; '
                    f'min-width:2.5rem;">{pct:.0f}th</span>'
                    f'</div>'
                    f'</div>'
                )

        if bar_parts:
            bars_html = "".join(bar_parts)
            st.markdown(
                f'<div style="background:transparent; border:none; '
                f'border-bottom:1px solid var(--tdd-dark-border); padding:12px 0; margin-bottom:12px;">'
                f'<div class="tdd-section-hdr" style="margin-bottom:8px;">{label}</div>'
                f'{bars_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        f"Projected percentiles rank each player among all {player_type.lower()}s "
        f"in the {CURRENT_SEASON} projection set. "
        f"Gold highlight = best value in comparison group. "
        f"Green = elite (80+), gold = above-avg (60-79), "
        f"gray = mid-tier (40-59), orange = below-avg (<40)."
    )
