"""Team Overview page — team-level view of projected pitchers and hitters."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    POSITIVE, NEGATIVE,
    PITCHER_STATS, HITTER_STATS,
    CURRENT_SEASON, PRIOR_SEASON,
)
from services.data_loader import (
    load_projections, load_counting, load_player_teams, load_roster,
    load_pitcher_arsenal, load_preseason_injuries,
    load_pitcher_offerings, load_hitter_vuln_arch_career,
    load_cluster_metadata, load_baselines_arch,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_rankings, load_position_eligibility,
    load_probable_starters,
)
from utils.helpers import get_team_lookup, get_injury_lookup
from utils.formatters import fmt_stat
from components.diamond_rating import diamond_rating_text_composite

# ── Constants ─────────────────────────────────────────────────────────
_HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]

_PILL_COLORS = {
    # Pitcher archetypes
    "Command Specialist": SAGE, "Breaking-Ball Heavy": EMBER,
    "Balanced Mix": SLATE, "Power Arm": GOLD,
    "Fastball Dominant": "#3498DB", "Ground-Ball Artist": "#9B59B6",
    # Hitter archetypes
    "Patient Power": GOLD, "Contact-Over-Power": SAGE,
    "Speed Threat": "#3498DB", "Power Slugger": EMBER,
    "Balanced All-Around": SLATE, "Free Swinger": "#9B59B6",
}

_HEALTH_COLORS = {
    "Iron Man": GOLD, "Durable": SAGE,
    "Average": SLATE, "Unknown": SLATE,
    "Questionable": EMBER, "Injury Prone": NEGATIVE,
}


# ── Depth Chart helpers ───────────────────────────────────────────────

def _build_depth_chart(
    selected_team: str, teams_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble per-position player lists from rankings + archetypes.

    Uses the position eligibility table when available so that utility
    players appear at every position they qualify for.  Falls back to
    the single ``position`` column in the rankings if the eligibility
    parquet hasn't been generated yet.
    """
    h_rank = load_rankings("hitters")
    p_rank = load_rankings("pitchers")
    if h_rank.empty and p_rank.empty:
        return pd.DataFrame(), pd.DataFrame()

    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )

    # ── Hitters: expand to all eligible positions ────────────────
    if not h_rank.empty:
        team_h = h_rank[h_rank["batter_id"].isin(team_pids)].copy()
        elig = load_position_eligibility()
        if not elig.empty and not team_h.empty:
            # Keep only this team's eligibility rows
            team_elig = elig[elig["player_id"].isin(team_pids)].copy()
            team_elig = team_elig.rename(columns={"player_id": "batter_id"})
            # Drop position (replaced by eligibility); keep pos_rank (MLB-wide)
            base = team_h.drop(columns=["position"], errors="ignore")
            team_h = base.merge(
                team_elig[["batter_id", "position", "starts", "pct", "is_primary"]],
                on="batter_id", how="inner",
            )
            # Null out pos_rank for secondary positions (rank is primary-pos specific)
            team_h.loc[~team_h["is_primary"], "pos_rank"] = pd.NA
            team_h = team_h.sort_values(
                ["position", "tdd_value_score"], ascending=[True, False],
            )
        # If no eligibility data, team_h keeps its original position column
    else:
        team_h = pd.DataFrame()

    # ── Pitchers: unchanged (single role) ────────────────────────
    team_p = (
        p_rank[p_rank["pitcher_id"].isin(team_pids)].copy()
        if not p_rank.empty else pd.DataFrame()
    )

    # Merge archetypes
    h_arch = load_hitter_archetypes()
    p_arch = load_pitcher_archetypes()
    if not h_arch.empty and not team_h.empty:
        team_h = team_h.merge(
            h_arch[["batter_id", "archetype_name"]].drop_duplicates("batter_id"),
            on="batter_id", how="left",
        )
    if not p_arch.empty and not team_p.empty:
        team_p = team_p.merge(
            p_arch[["pitcher_id", "archetype_name"]].drop_duplicates("pitcher_id"),
            on="pitcher_id", how="left",
        )

    return team_h, team_p


def _player_row_html(
    name: str, rank: int, score: float, arch: str, health: str,
    is_first: bool, *, hand: str = "", secondary: bool = False,
) -> str:
    """Render a single player row for depth chart."""
    if secondary:
        border_color = f"{SLATE}22"
    elif is_first:
        border_color = GOLD
    else:
        border_color = f"{SLATE}44"
    opacity = "0.6" if secondary else "1"

    arch_html = ""
    if pd.notna(arch) and arch:
        color = _PILL_COLORS.get(arch, SLATE)
        arch_html = (
            f'<span style="background:{color}22; color:{color}; border:1px solid {color}44; '
            f'padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:600; '
            f'margin-left:8px;">{arch}</span>'
        )

    h_color = _HEALTH_COLORS.get(health, SLATE)
    dot = (
        "\u25cf" if health in ("Iron Man", "Durable")
        else "\u25d0" if health in ("Average", "Unknown")
        else "\u25cb"
    )
    health_html = (
        f'<span style="color:{h_color}; font-size:0.75rem; margin-left:8px;">'
        f'{dot} {health}</span>'
    )

    hand_html = ""
    if hand:
        hand_html = (
            f'<span style="color:{SLATE}; font-size:0.8rem; min-width:30px;">'
            f'{hand}HP</span>'
        )

    rank_html = (
        f'<span style="color:{SLATE}; font-size:0.85rem; min-width:60px;">'
        f'#{rank}</span>'
        if rank > 0 else ""
    )

    return (
        f'<div style="padding:6px 12px; border-left:3px solid {border_color}; '
        f'margin-bottom:2px; display:flex; align-items:center; gap:8px; '
        f'flex-wrap:wrap; opacity:{opacity};">'
        f'<span style="color:{CREAM}; font-weight:{"700" if is_first else "400"}; '
        f'min-width:6rem; flex-shrink:1;">{name}</span>'
        f'{hand_html}'
        f'{rank_html}'
        f'<span style="color:{GOLD}; font-size:0.85rem; min-width:2.5rem;">{score:.2f}</span>'
        f'{arch_html}{health_html}'
        f'</div>'
    )


def _render_depth_chart_tab(selected_team: str, teams_df: pd.DataFrame) -> None:
    """Render the Depth Chart tab."""
    team_h, team_p = _build_depth_chart(selected_team, teams_df)
    if team_h.empty and team_p.empty:
        st.info("No rankings data available. Run the rankings precompute first.")
        return

    # ── Diamond chart (probable starters on a field) ──────────────
    probable = load_probable_starters()
    if not probable.empty:
        team_starters = probable[probable["team_abbr"] == selected_team]
        if not team_starters.empty:
            from components.depth_chart_diamond import render_diamond_chart
            render_diamond_chart(team_starters, selected_team)

    # ── Summary metric cards ─────────────────────────────────────
    if not team_h.empty:
        # For summary cards, use only primary-position rows when available
        if "is_primary" in team_h.columns:
            primary_h = team_h[team_h["is_primary"]].copy()
        else:
            primary_h = team_h.copy()

        if not primary_h.empty:
            best_per_pos = primary_h.loc[
                primary_h.groupby("position")["tdd_value_score"].idxmax()
            ]
        else:
            best_per_pos = pd.DataFrame(columns=team_h.columns)

        filled_positions = set(best_per_pos["position"]) if not best_per_pos.empty else set()
        non_dh = best_per_pos[best_per_pos["position"] != "DH"] if not best_per_pos.empty else best_per_pos

        strongest = non_dh.loc[non_dh["tdd_value_score"].idxmax()] if not non_dh.empty else None
        weakest = non_dh.loc[non_dh["tdd_value_score"].idxmin()] if not non_dh.empty else None
        avg_score = non_dh["tdd_value_score"].mean() if not non_dh.empty else 0
        gaps = [
            p for p in _HITTER_POSITIONS
            if p != "DH" and p not in filled_positions
        ]

        cols = st.columns(4)
        with cols[0]:
            if strongest is not None:
                s_rank = int(strongest["pos_rank"]) if pd.notna(strongest.get("pos_rank")) else None
                val = (
                    f'{strongest["position"]} (#{s_rank})'
                    if s_rank else f'{strongest["position"]}'
                )
            else:
                val = "--"
            st.metric("Strongest Position", val)
        with cols[1]:
            if weakest is not None:
                w_rank = int(weakest["pos_rank"]) if pd.notna(weakest.get("pos_rank")) else None
                val = (
                    f'{weakest["position"]} (#{w_rank})'
                    if w_rank else f'{weakest["position"]}'
                )
            else:
                val = "--"
            st.metric("Weakest Position", val)
        with cols[2]:
            st.metric("Avg Value Score", f"{avg_score:.2f}" if avg_score else "--")
        with cols[3]:
            st.metric("Position Gaps", str(len(gaps)) if gaps else "0")

    _render_hitter_depth(team_h)
    _render_pitching_depth(team_p)


def _render_hitter_depth(team_h: pd.DataFrame) -> None:
    """Render hitter depth chart — 9 positions with ranked players."""
    if team_h.empty:
        return
    st.markdown("### Hitter Depth")

    has_elig = "is_primary" in team_h.columns

    for pos in _HITTER_POSITIONS:
        pos_players = team_h[team_h["position"] == pos].sort_values(
            "tdd_value_score", ascending=False,
        )

        if pos_players.empty:
            st.markdown(
                f'<div style="padding:8px 12px; margin-bottom:6px; '
                f'border-left:3px solid {SLATE}44; opacity:0.5;">'
                f'<span style="color:{SLATE}; font-weight:600;">{pos}</span> '
                f'<span style="color:{SLATE}; font-size:0.85rem;">'
                f'\u2014 No ranked player</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            continue

        rows_html = ""
        for i, (_, row) in enumerate(pos_players.iterrows()):
            name = row["batter_name"]
            is_secondary = has_elig and not row.get("is_primary", True)
            if is_secondary:
                pct = row.get("pct", 0)
                name += f" ({pct:.0%})" if pct else ""

            rank_val = int(row["pos_rank"]) if pd.notna(row.get("pos_rank")) else 0
            rows_html += _player_row_html(
                name=name,
                rank=rank_val,
                score=row["tdd_value_score"],
                arch=row.get("archetype_name", ""),
                health=row.get("health_label", "Unknown"),
                is_first=i == 0 and not is_secondary,
                secondary=is_secondary,
            )

        st.markdown(
            f'<div style="margin-bottom:10px;">'
            f'<div style="color:{GOLD}; font-weight:700; font-size:0.9rem; '
            f'margin-bottom:4px; padding-left:15px;">{pos}</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )


def _render_pitching_depth(team_p: pd.DataFrame) -> None:
    """Render pitching depth — rotation + bullpen."""
    if team_p.empty:
        return
    st.markdown("### Pitching Depth")

    for label, subset in [
        ("Starting Rotation",
         team_p[team_p["role"] == "SP"].sort_values("role_rank").head(5)),
        ("Bullpen",
         team_p[team_p["role"] == "RP"].sort_values(
             "tdd_value_score", ascending=False)),
    ]:
        if subset.empty:
            continue
        st.markdown(
            f'<div style="color:{GOLD}; font-weight:700; font-size:0.9rem; '
            f'margin-bottom:4px; padding-left:15px;">{label}</div>',
            unsafe_allow_html=True,
        )
        rows_html = ""
        for i, (_, row) in enumerate(subset.iterrows()):
            rows_html += _player_row_html(
                name=row["pitcher_name"],
                rank=int(row["role_rank"]),
                score=row["tdd_value_score"],
                arch=row.get("archetype_name", ""),
                health=row.get("health_label", "Unknown"),
                is_first=i == 0,
                hand=row.get("pitch_hand", ""),
            )
        st.markdown(
            f'<div style="margin-bottom:14px;">{rows_html}</div>',
            unsafe_allow_html=True,
        )


# ── Trade Simulator helpers ───────────────────────────────────────────

def _identify_weaknesses(
    selected_team: str, teams_df: pd.DataFrame,
) -> list[dict]:
    """Identify positional weaknesses — best player pos_rank > 15."""
    h_rank = load_rankings("hitters")
    if h_rank.empty:
        return []

    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )
    team_h = h_rank[h_rank["batter_id"].isin(team_pids)]

    weaknesses: list[dict] = []
    for pos in _HITTER_POSITIONS:
        if pos == "DH":
            continue
        pos_players = team_h[team_h["position"] == pos]
        if pos_players.empty:
            weaknesses.append({
                "position": pos, "best_rank": None, "player": None,
            })
        else:
            best = pos_players.loc[pos_players["pos_rank"].idxmin()]
            if best["pos_rank"] > 15:
                weaknesses.append({
                    "position": pos,
                    "best_rank": int(best["pos_rank"]),
                    "player": best["batter_name"],
                })
    return weaknesses


def _find_trade_targets(
    selected_team: str,
    teams_df: pd.DataFrame,
    target_position: str,
    trade_capital: float,
    offered_positions: list[str] | None = None,
    tolerance: float = 0.15,
) -> pd.DataFrame:
    """Find league-wide trade targets at a position within value tolerance."""
    h_rank = load_rankings("hitters")
    if h_rank.empty:
        return pd.DataFrame()

    other_pids = set(
        teams_df[teams_df["team_abbr"] != selected_team]["player_id"].astype(int)
    )
    candidates = h_rank[
        (h_rank["batter_id"].isin(other_pids))
        & (h_rank["position"] == target_position)
    ].copy()

    if candidates.empty or trade_capital <= 0:
        return pd.DataFrame()

    low = trade_capital * (1 - tolerance)
    high = trade_capital * (1 + tolerance)
    matches = candidates[
        (candidates["tdd_value_score"] >= low)
        & (candidates["tdd_value_score"] <= high)
    ].copy()

    if matches.empty:
        candidates["_diff"] = (
            candidates["tdd_value_score"] - trade_capital
        ).abs()
        matches = candidates.nsmallest(10, "_diff").drop(columns="_diff")

    # Add team info
    pid_to_team = dict(zip(
        teams_df["player_id"].astype(int), teams_df["team_abbr"],
    ))
    matches["team"] = matches["batter_id"].map(pid_to_team)

    # Value match percentage
    matches["value_match_pct"] = (
        1 - (matches["tdd_value_score"] - trade_capital).abs()
        / max(trade_capital, 0.01)
    ) * 100

    # Merge archetypes
    h_arch = load_hitter_archetypes()
    if not h_arch.empty:
        arch_cols = h_arch[["batter_id", "archetype_name"]].drop_duplicates(
            "batter_id",
        )
        if "archetype_name" in matches.columns:
            matches = matches.drop(columns="archetype_name")
        matches = matches.merge(arch_cols, on="batter_id", how="left")

    # Assess mutual fit — do partner teams need the offered positions?
    if offered_positions:
        hitter_pos = [p for p in offered_positions if p in _HITTER_POSITIONS]
        fit_tiers: list[str] = []
        for _, row in matches.iterrows():
            partner_team = row.get("team", "")
            if not partner_team or not hitter_pos:
                fit_tiers.append("Low")
                continue
            partner_pids = set(
                teams_df[teams_df["team_abbr"] == partner_team]["player_id"].astype(int)
            )
            partner_h = h_rank[h_rank["batter_id"].isin(partner_pids)]
            tier = "Low"
            for pos in hitter_pos:
                pos_players = partner_h[partner_h["position"] == pos]
                if pos_players.empty:
                    tier = "High"
                    break
                best_rank = int(pos_players["pos_rank"].min())
                if best_rank > 15:
                    tier = "High"
                    break
                elif best_rank > 10 and tier != "High":
                    tier = "Medium"
            fit_tiers.append(tier)
        matches["trade_fit"] = fit_tiers
    else:
        matches["trade_fit"] = "Low"

    fit_order = {"High": 0, "Medium": 1, "Low": 2}
    matches["_fit_order"] = matches["trade_fit"].map(fit_order)
    return matches.sort_values(
        ["_fit_order", "pos_rank"],
    ).drop(columns="_fit_order").head(10)


def _render_trade_simulator_tab(
    selected_team: str, teams_df: pd.DataFrame,
) -> None:
    """Render the Trade Simulator tab."""
    # Step 1 — Weakness display
    weaknesses = _identify_weaknesses(selected_team, teams_df)

    st.markdown("#### Positional Needs")
    if weaknesses:
        pills_html = ""
        for w in weaknesses:
            suffix = f' (#{w["best_rank"]})' if w["best_rank"] else " (empty)"
            pills_html += (
                f'<span style="background:{EMBER}22; color:{EMBER}; '
                f'border:1px solid {EMBER}44; padding:4px 12px; '
                f'border-radius:16px; font-size:0.85rem; font-weight:600; '
                f'margin-right:6px;">{w["position"]}{suffix}</span> '
            )
        st.markdown(
            f'<div style="margin-bottom:16px;">{pills_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{SAGE};">'
            f'No significant positional weaknesses detected.</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Step 2 — Select trade assets
    st.markdown("#### Select Trade Assets")
    h_rank = load_rankings("hitters")
    p_rank = load_rankings("pitchers")
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )

    assets: list[dict] = []
    if not h_rank.empty:
        team_h = h_rank[h_rank["batter_id"].isin(team_pids)].sort_values(
            "tdd_value_score", ascending=False,
        )
        for _, row in team_h.iterrows():
            assets.append({
                "label": (
                    f'{row["batter_name"]} ({row["position"]}, '
                    f'#{int(row["pos_rank"])}) '
                    f'\u2014 {row["tdd_value_score"]:.2f}'
                ),
                "value": row["tdd_value_score"],
                "position": row["position"],
            })
    if not p_rank.empty:
        team_p = p_rank[p_rank["pitcher_id"].isin(team_pids)].sort_values(
            "tdd_value_score", ascending=False,
        )
        for _, row in team_p.iterrows():
            assets.append({
                "label": (
                    f'{row["pitcher_name"]} ({row["role"]}, '
                    f'#{int(row["role_rank"])}) '
                    f'\u2014 {row["tdd_value_score"]:.2f}'
                ),
                "value": row["tdd_value_score"],
                "position": row["role"],
            })

    if not assets:
        st.info("No ranked players available for trade selection.")
        return

    selected_assets = st.multiselect(
        "Players to offer",
        options=[a["label"] for a in assets],
        key="trade_assets",
    )

    trade_capital = sum(
        a["value"] for a in assets if a["label"] in selected_assets
    )
    st.markdown(
        f'<div style="padding:8px 16px; background:{GOLD}15; '
        f'border:1px solid {GOLD}44; border-radius:8px; margin:8px 0 16px;">'
        f'<span style="color:{CREAM}; font-weight:600;">Trade Capital:</span> '
        f'<span style="color:{GOLD}; font-weight:700; font-size:1.1rem;">'
        f'{trade_capital:.2f}</span></div>',
        unsafe_allow_html=True,
    )

    # Step 3 — Select target need
    st.markdown("#### Target Need")
    weakness_positions = [w["position"] for w in weaknesses]
    all_positions = weakness_positions + [
        p for p in _HITTER_POSITIONS if p not in weakness_positions
    ]
    target_pos = st.selectbox(
        "Position to upgrade",
        all_positions,
        key="trade_target_pos",
    )

    # Step 4 — Find matches
    if st.button(
        "Find Trade Matches", type="primary", disabled=trade_capital <= 0,
    ):
        offered_positions = [
            a["position"] for a in assets
            if a["label"] in selected_assets and "position" in a
        ]
        matches = _find_trade_targets(
            selected_team, teams_df, target_pos, trade_capital,
            offered_positions=offered_positions,
        )
        if matches.empty:
            st.info(f"No trade candidates found at {target_pos}.")
        else:
            _render_trade_results(
                matches, trade_capital, target_pos,
                selected_team, teams_df,
            )


def _render_trade_results(
    matches: pd.DataFrame,
    trade_capital: float,
    target_pos: str,
    selected_team: str,
    teams_df: pd.DataFrame,
) -> None:
    """Display trade match results table."""
    st.markdown(f"#### Trade Candidates \u2014 {target_pos}")

    # Current best at target position for upgrade indicator
    h_rank = load_rankings("hitters")
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )
    team_at_pos = h_rank[
        (h_rank["batter_id"].isin(team_pids))
        & (h_rank["position"] == target_pos)
    ]
    current_best_rank = (
        int(team_at_pos["pos_rank"].min())
        if not team_at_pos.empty else None
    )

    rows_html = ""
    for _, row in matches.iterrows():
        name = row["batter_name"]
        team = row.get("team", "?")
        rank = int(row["pos_rank"])
        score = row["tdd_value_score"]
        match_pct = row.get("value_match_pct", 0)
        arch = row.get("archetype_name", "")
        trade_fit = row.get("trade_fit", "Low")
        fit_color = (
            GOLD if trade_fit == "High"
            else SAGE if trade_fit == "Medium"
            else SLATE
        )
        fit_label = (
            "Likely" if trade_fit == "High"
            else "Possible" if trade_fit == "Medium"
            else "Unlikely"
        )

        if current_best_rank and rank < current_best_rank:
            upgrade = (
                f'<span style="color:{POSITIVE}; font-weight:600;">'
                f'\u2191 +{current_best_rank - rank}</span>'
            )
        elif current_best_rank:
            upgrade = f'<span style="color:{SLATE};">\u2014</span>'
        else:
            upgrade = (
                f'<span style="color:{POSITIVE}; font-weight:600;">'
                f'\u2191 New</span>'
            )

        arch_html = ""
        if pd.notna(arch) and arch:
            color = _PILL_COLORS.get(arch, SLATE)
            arch_html = (
                f'<span style="background:{color}22; color:{color}; '
                f'border:1px solid {color}44; padding:2px 8px; '
                f'border-radius:10px; font-size:0.75rem;">{arch}</span>'
            )

        m_color = (
            GOLD if match_pct >= 85
            else SAGE if match_pct >= 70
            else EMBER
        )

        rows_html += (
            f'<tr>'
            f'<td style="color:{CREAM};padding:6px 10px;font-weight:600;">'
            f'{name}</td>'
            f'<td style="padding:6px 10px;color:{SLATE};">{team}</td>'
            f'<td style="padding:6px 10px;">{arch_html}</td>'
            f'<td style="padding:6px 10px;color:{GOLD};">{score:.2f}</td>'
            f'<td style="padding:6px 10px;">#{rank}</td>'
            f'<td style="padding:6px 10px;color:{m_color};">'
            f'{match_pct:.0f}%</td>'
            f'<td style="padding:6px 10px;">{upgrade}</td>'
            f'<td style="padding:6px 10px;">'
            f'<span style="background:{fit_color}22; color:{fit_color}; '
            f'border:1px solid {fit_color}44; padding:2px 8px; '
            f'border-radius:10px; font-size:0.75rem; font-weight:600;">'
            f'{fit_label}</span></td>'
            f'</tr>'
        )

    table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        f'<thead><tr style="border-bottom:1px solid {SLATE}44;">'
        f'<th style="text-align:left;padding:6px 10px;color:{GOLD};">Player</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">Team</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">Archetype</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">TDD Score</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">Pos Rank</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">Match %</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">Upgrade</th>'
        f'<th style="padding:6px 10px;color:{SLATE};">Trade Fit</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )
    st.markdown(
        f'<div class="insight-card">{table_html}</div>',
        unsafe_allow_html=True,
    )


# ── Pitching staff helpers ────────────────────────────────────────────

def _render_staff_metrics(
    team_subset: pd.DataFrame,
    lg_subset: pd.DataFrame,
    k_col: str,
    bb_col: str,
) -> None:
    """Render strengths/weaknesses comparison for a pitcher subset (SP or RP)."""
    pitch_metrics: list[dict] = []
    for label, key, higher_better in [
        ("K%", k_col, True),
        ("BB%", bb_col, False),
        ("Whiff%", "whiff_rate", True),
        ("Avg Velo", "avg_velo", True),
        ("Zone%", "zone_pct", True),
        ("GB%", "gb_pct", True),
    ]:
        if key not in team_subset.columns or key not in lg_subset.columns:
            continue
        team_avg = team_subset[key].dropna().mean()
        league_avg = lg_subset[key].dropna().mean()
        if pd.isna(team_avg) or pd.isna(league_avg) or league_avg == 0:
            continue
        diff = team_avg - league_avg
        if key in (k_col, bb_col, "whiff_rate", "zone_pct", "gb_pct"):
            diff_str = f"{diff * 100:+.1f}pp"
            team_str = f"{team_avg * 100:.1f}%"
            lg_str = f"{league_avg * 100:.1f}%"
        else:
            diff_str = f"{diff:+.1f}"
            team_str = f"{team_avg:.1f}"
            lg_str = f"{league_avg:.1f}"
        is_good = (diff > 0 and higher_better) or (diff < 0 and not higher_better)
        color = POSITIVE if is_good else NEGATIVE
        pitch_metrics.append({
            "Metric": label, "Team": team_str, "League Avg": lg_str,
            "Diff": diff_str, "_color": color, "_is_good": is_good,
        })

    if not pitch_metrics:
        return

    strengths = [m for m in pitch_metrics if m["_is_good"]]
    weaknesses_p = [m for m in pitch_metrics if not m["_is_good"]]
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown(
            f'<div style="color:{POSITIVE}; font-weight:600; '
            f'margin-bottom:8px;">Strengths</div>',
            unsafe_allow_html=True,
        )
        if strengths:
            for m in strengths:
                st.markdown(
                    f'<div style="padding:4px 0;">'
                    f'<span style="color:{CREAM};">{m["Metric"]}</span>: '
                    f'<span style="color:{POSITIVE}; font-weight:600;">'
                    f'{m["Team"]}</span> '
                    f'<span style="color:{SLATE};">'
                    f'(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<span style="color:{SLATE};">None vs league average</span>',
                unsafe_allow_html=True,
            )
    with col_w:
        st.markdown(
            f'<div style="color:{NEGATIVE}; font-weight:600; '
            f'margin-bottom:8px;">Weaknesses</div>',
            unsafe_allow_html=True,
        )
        if weaknesses_p:
            for m in weaknesses_p:
                st.markdown(
                    f'<div style="padding:4px 0;">'
                    f'<span style="color:{CREAM};">{m["Metric"]}</span>: '
                    f'<span style="color:{NEGATIVE}; font-weight:600;">'
                    f'{m["Team"]}</span> '
                    f'<span style="color:{SLATE};">'
                    f'(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<span style="color:{SLATE};">None vs league average</span>',
                unsafe_allow_html=True,
            )


def _build_pitcher_rows(
    pitchers: pd.DataFrame,
    injury_lookup: dict,
    use_priors: bool,
) -> list[dict]:
    """Build display rows for a set of pitchers (SP or RP)."""
    rows: list[dict] = []
    for _, row in pitchers.sort_values(
        "composite_score", ascending=False,
    ).iterrows():
        pid = int(row["pitcher_id"])
        inj = injury_lookup.get(pid)
        name = row["pitcher_name"]
        if inj and inj["missed_games"] > 0:
            sev = inj["severity"]
            tag = (
                "[IL-60]" if sev == "major"
                else "[IL]" if sev == "significant"
                else "[DTD]"
            )
            name = f"{tag} {name}"
        r: dict[str, object] = {
            "Name": name,
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Hand": row.get("pitch_hand", ""),
            "Rating": diamond_rating_text_composite(row["composite_score"]),
        }
        if use_priors:
            for label, key, _, _ in PITCHER_STATS:
                obs_col = f"observed_{key}"
                if obs_col in row.index and pd.notna(row.get(obs_col)):
                    r[f"{label} ({PRIOR_SEASON})"] = fmt_stat(row[obs_col], key)
                else:
                    r[f"{label} ({PRIOR_SEASON})"] = "--"
            for label, key in [
                ("Whiff%", "whiff_rate"), ("Avg Velo", "avg_velo"),
            ]:
                if key in row.index and pd.notna(row.get(key)):
                    r[label] = fmt_stat(row[key], key)
                else:
                    r[label] = "--"
        else:
            for label, key, _, _ in PITCHER_STATS:
                proj_col = f"projected_{key}"
                delta_col = f"delta_{key}"
                if proj_col in row.index and pd.notna(row.get(proj_col)):
                    proj_val = fmt_stat(row[proj_col], key)
                    delta_pp = row[delta_col] * 100
                    r[label] = (
                        f"{proj_val} ({delta_pp:+.1f})"
                        if abs(delta_pp) >= 0.05
                        else proj_val
                    )
                else:
                    r[label] = "--"
            if "total_k_mean" in row.index and pd.notna(row.get("total_k_mean")):
                r["Proj. K"] = int(round(row["total_k_mean"]))
            else:
                r["Proj. K"] = "--"
        rows.append(r)
    return rows


# ── Overview tab (existing content) ──────────────────────────────────

def _render_overview_tab(
    selected_team: str,
    team_hitters: pd.DataFrame,
    team_pitchers: pd.DataFrame,
    h_proj: pd.DataFrame,
    p_proj: pd.DataFrame,
    injury_lookup: dict,
) -> None:
    """All original team overview content — identity, profiles, tables."""
    # View toggle: Projections vs Priors
    view_mode = st.radio(
        "View",
        [f"{CURRENT_SEASON} Projections", f"{PRIOR_SEASON} Priors (Observed)"],
        horizontal=True,
        key="team_view_mode",
    )
    use_priors = view_mode == f"{PRIOR_SEASON} Priors (Observed)"

    if use_priors:
        _h_k_col = "observed_k_rate"
        _h_bb_col = "observed_bb_rate"
        _p_k_col = "observed_k_rate"
        _p_bb_col = "observed_bb_rate"
        _view_label = f"{PRIOR_SEASON} Observed"
    else:
        _h_k_col = "projected_k_rate"
        _h_bb_col = "projected_bb_rate"
        _p_k_col = "projected_k_rate"
        _p_bb_col = "projected_bb_rate"
        _view_label = f"{CURRENT_SEASON} Projected"

    # ── Team Identity Tags ──────────────────────────────────────────
    identity_tags: list[tuple[str, str]] = []

    if not team_hitters.empty and not h_proj.empty:
        _has_statcast = all(
            c in h_proj.columns
            for c in ("hard_hit_pct", "avg_exit_velo", "whiff_rate", "z_contact_pct")
        )
        if _has_statcast:
            lg_hh = h_proj["hard_hit_pct"].dropna().mean()
            lg_ev = h_proj["avg_exit_velo"].dropna().mean()
            lg_whiff = h_proj["whiff_rate"].dropna().mean()
            lg_zcon = h_proj["z_contact_pct"].dropna().mean()
            team_hh = team_hitters["hard_hit_pct"].dropna().mean()
            team_ev = team_hitters["avg_exit_velo"].dropna().mean()
            team_whiff = team_hitters["whiff_rate"].dropna().mean()
            team_zcon = team_hitters["z_contact_pct"].dropna().mean()

            power_score = (
                (team_hh - lg_hh) / max(h_proj["hard_hit_pct"].dropna().std(), 0.001)
                + (team_ev - lg_ev) / max(h_proj["avg_exit_velo"].dropna().std(), 0.001)
            )
            contact_score = (
                (team_zcon - lg_zcon) / max(h_proj["z_contact_pct"].dropna().std(), 0.001)
                + (lg_whiff - team_whiff) / max(h_proj["whiff_rate"].dropna().std(), 0.001)
            )

            if power_score > 1.0:
                identity_tags.append(("Power Offense", GOLD))
            elif power_score < -1.0:
                identity_tags.append(("Low-Power Offense", SLATE))
            if contact_score > 1.0:
                identity_tags.append(("Contact Offense", SAGE))
            elif contact_score < -1.0:
                identity_tags.append(("Swing-and-Miss Offense", EMBER))

        n_left = (team_hitters["batter_stand"] == "L").sum()
        n_right = (team_hitters["batter_stand"] == "R").sum()
        n_switch = (team_hitters["batter_stand"] == "S").sum()
        total_h = len(team_hitters)
        if total_h > 0:
            left_pct = (n_left + n_switch * 0.5) / total_h
            if left_pct >= 0.55:
                identity_tags.append(("LHB-Heavy Lineup", SLATE))
            elif left_pct <= 0.30:
                identity_tags.append(("RHB-Heavy Lineup", SLATE))
            else:
                identity_tags.append(("Balanced Lineup", SLATE))

    if not team_pitchers.empty and not p_proj.empty:
        n_lhp = (team_pitchers["pitch_hand"] == "L").sum()
        n_rhp = (team_pitchers["pitch_hand"] == "R").sum()
        total_p = len(team_pitchers)
        if total_p > 0:
            lhp_pct = n_lhp / total_p
            if lhp_pct >= 0.45:
                identity_tags.append(("LHP-Heavy Staff", SLATE))
            elif lhp_pct <= 0.20:
                identity_tags.append(("RHP-Heavy Staff", SLATE))

        lg_k = p_proj["projected_k_rate"].dropna().mean()
        lg_bb = p_proj["projected_bb_rate"].dropna().mean()
        team_k = team_pitchers["projected_k_rate"].dropna().mean()
        team_bb = team_pitchers["projected_bb_rate"].dropna().mean()
        k_z = (team_k - lg_k) / max(p_proj["projected_k_rate"].dropna().std(), 0.001)
        bb_z = (team_bb - lg_bb) / max(p_proj["projected_bb_rate"].dropna().std(), 0.001)
        if k_z > 0.8:
            identity_tags.append(("High-K Staff", GOLD))
        elif k_z < -0.8:
            identity_tags.append(("Low-K Staff", EMBER))
        if bb_z < -0.8:
            identity_tags.append(("Control Staff", SAGE))
        elif bb_z > 0.8:
            identity_tags.append(("Walk-Prone Staff", EMBER))

    # Staff arsenal breakdown
    arsenal_df = load_pitcher_arsenal()
    team_arsenal_summary = None
    team_ars = pd.DataFrame()
    if not arsenal_df.empty and not team_pitchers.empty:
        team_pitcher_ids = set(team_pitchers["pitcher_id"].astype(int))
        team_ars = arsenal_df[arsenal_df["pitcher_id"].isin(team_pitcher_ids)]
        if not team_ars.empty:
            family_agg = (
                team_ars.groupby("pitch_family")
                .agg(
                    pitches=("pitches", "sum"),
                    whiffs=("whiffs", "sum"),
                    swings=("swings", "sum"),
                )
                .reset_index()
            )
            family_agg["pct"] = family_agg["pitches"] / family_agg["pitches"].sum()
            family_agg["whiff_rate"] = (
                family_agg["whiffs"] / family_agg["swings"].clip(lower=1)
            )
            team_arsenal_summary = family_agg.sort_values("pct", ascending=False)

            fb_pct = family_agg.loc[
                family_agg["pitch_family"] == "fastball", "pct"
            ]
            brk_pct = family_agg.loc[
                family_agg["pitch_family"] == "breaking", "pct"
            ]
            if not fb_pct.empty and float(fb_pct.iloc[0]) >= 0.55:
                identity_tags.append(("Fastball-Heavy Staff", SLATE))
            if not brk_pct.empty and float(brk_pct.iloc[0]) >= 0.35:
                identity_tags.append(("Breaking-Heavy Staff", SLATE))

    # Render identity tags
    if identity_tags:
        tags_html = " ".join(
            f'<span style="background:{color}22; color:{color}; '
            f'border:1px solid {color}44; padding:4px 12px; border-radius:16px; '
            f'font-size:0.85rem; font-weight:600; margin-right:6px;">{label}</span>'
            for label, color in identity_tags
        )
        st.markdown(
            f'<div style="margin-bottom:16px;">{tags_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Roster Archetype Composition ────────────────────────────────
    _p_arch_df = load_pitcher_archetypes()
    _h_arch_df = load_hitter_archetypes()

    if not _p_arch_df.empty or not _h_arch_df.empty:
        sp_pills = ""
        bp_pills = ""
        lineup_pills = ""

        if not _p_arch_df.empty and not team_pitchers.empty:
            _sp_ids = set(
                team_pitchers[team_pitchers["is_starter"] == True]["pitcher_id"].astype(int)
            )
            _rp_ids = set(
                team_pitchers[team_pitchers["is_starter"] == False]["pitcher_id"].astype(int)
            )

            _sp_arch = _p_arch_df[_p_arch_df["pitcher_id"].isin(_sp_ids)]
            if not _sp_arch.empty:
                for name, count in _sp_arch["archetype_name"].value_counts().items():
                    color = _PILL_COLORS.get(name, SLATE)
                    sp_pills += (
                        f'<span style="background:{color}22; color:{color}; '
                        f'border:1px solid {color}44; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.8rem; font-weight:600; '
                        f'margin-right:5px; white-space:nowrap;">'
                        f'{count}\u00d7 {name}</span>'
                    )

            _rp_arch = _p_arch_df[_p_arch_df["pitcher_id"].isin(_rp_ids)]
            if not _rp_arch.empty:
                for name, count in _rp_arch["archetype_name"].value_counts().items():
                    color = _PILL_COLORS.get(name, SLATE)
                    bp_pills += (
                        f'<span style="background:{color}22; color:{color}; '
                        f'border:1px solid {color}44; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.8rem; font-weight:600; '
                        f'margin-right:5px; white-space:nowrap;">'
                        f'{count}\u00d7 {name}</span>'
                    )

        if not _h_arch_df.empty and not team_hitters.empty:
            _th_ids = set(team_hitters["batter_id"].astype(int))
            _th_arch = _h_arch_df[_h_arch_df["batter_id"].isin(_th_ids)]
            if not _th_arch.empty:
                for name, count in _th_arch["archetype_name"].value_counts().items():
                    color = _PILL_COLORS.get(name, SLATE)
                    lineup_pills += (
                        f'<span style="background:{color}22; color:{color}; '
                        f'border:1px solid {color}44; padding:3px 10px; '
                        f'border-radius:12px; font-size:0.8rem; font-weight:600; '
                        f'margin-right:5px; white-space:nowrap;">'
                        f'{count}\u00d7 {name}</span>'
                    )

        if sp_pills or bp_pills or lineup_pills:
            st.markdown("### Roster Composition")
            if sp_pills:
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="color:{CREAM}; font-size:0.85rem; '
                    f'font-weight:600; margin-right:8px;">SP:</span>'
                    f'{sp_pills}</div>',
                    unsafe_allow_html=True,
                )
            if bp_pills:
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="color:{CREAM}; font-size:0.85rem; '
                    f'font-weight:600; margin-right:8px;">Bullpen:</span>'
                    f'{bp_pills}</div>',
                    unsafe_allow_html=True,
                )
            if lineup_pills:
                st.markdown(
                    f'<div style="margin-bottom:12px;">'
                    f'<span style="color:{CREAM}; font-size:0.85rem; '
                    f'font-weight:600; margin-right:8px;">Lineup:</span>'
                    f'{lineup_pills}</div>',
                    unsafe_allow_html=True,
                )

    # ── Staff Arsenal Breakdown ─────────────────────────────────────
    if team_arsenal_summary is not None and not team_arsenal_summary.empty:
        st.markdown("### Staff Arsenal Mix")
        ars_cols = st.columns(len(team_arsenal_summary))
        for col, (_, row) in zip(ars_cols, team_arsenal_summary.iterrows()):
            family = row["pitch_family"].title()
            pct = row["pct"]
            whiff = row["whiff_rate"]
            col.metric(family, f"{pct:.0%}", f"Whiff: {whiff:.1%}")

        if not team_ars.empty:
            pt_agg = (
                team_ars.groupby("pitch_type")
                .agg(
                    pitches=("pitches", "sum"),
                    whiffs=("whiffs", "sum"),
                    swings=("swings", "sum"),
                )
                .reset_index()
            )
            pt_agg["pct"] = pt_agg["pitches"] / pt_agg["pitches"].sum()
            pt_agg["whiff_rate"] = (
                pt_agg["whiffs"] / pt_agg["swings"].clip(lower=1)
            )
            pt_agg = pt_agg.sort_values("pct", ascending=False)
            pt_rows = []
            for _, r in pt_agg.iterrows():
                if r["pct"] >= 0.02:
                    pt_rows.append({
                        "Pitch": r["pitch_type"],
                        "Usage": f"{r['pct']:.1%}",
                        "Whiff%": f"{r['whiff_rate']:.1%}",
                        "Pitches": int(r["pitches"]),
                    })
            with st.expander("Pitch type detail"):
                st.dataframe(
                    pd.DataFrame(pt_rows),
                    use_container_width=True, hide_index=True,
                )

    # ── Staff Archetype Mix ─────────────────────────────────────────
    offerings_df = load_pitcher_offerings()
    cluster_meta = load_cluster_metadata()
    if not offerings_df.empty and not cluster_meta.empty and not team_pitchers.empty:
        team_pitcher_ids = set(team_pitchers["pitcher_id"].astype(int))
        team_off = offerings_df[offerings_df["pitcher_id"].isin(team_pitcher_ids)]
        if not team_off.empty and "pitches" in team_off.columns:
            arch_agg = (
                team_off.groupby("pitch_archetype")
                .agg(pitches=("pitches", "sum"))
                .reset_index()
            )
            arch_agg = arch_agg.merge(
                cluster_meta[["pitch_archetype", "archetype_name"]],
                on="pitch_archetype", how="left",
            )
            arch_agg["pct"] = arch_agg["pitches"] / arch_agg["pitches"].sum()
            arch_agg = arch_agg.sort_values("pct", ascending=False)

            st.markdown("### Staff Archetype Mix")
            st.caption(
                "Pitch archetype distribution across the team's pitching staff."
            )

            _bar_colors = [
                GOLD, EMBER, SAGE, SLATE, CREAM,
                "#9B59B6", "#3498DB", "#E67E22",
            ]
            bars_html = ""
            for i, (_, row) in enumerate(arch_agg.iterrows()):
                color = _bar_colors[i % len(_bar_colors)]
                pct = row["pct"]
                name = row.get(
                    "archetype_name", f'Cluster {row["pitch_archetype"]}',
                )
                bars_html += (
                    f'<div style="display:flex;align-items:center;'
                    f'margin-bottom:6px;">'
                    f'<div style="width:140px;color:{CREAM};font-size:0.85rem;'
                    f'flex-shrink:0;">{name}</div>'
                    f'<div style="flex:1;background:#1a1d24;border-radius:4px;'
                    f'height:22px;overflow:hidden;">'
                    f'<div style="width:{pct*100:.1f}%;background:{color};'
                    f'height:100%;border-radius:4px;display:flex;'
                    f'align-items:center;padding-left:6px;">'
                    f'<span style="color:#fff;font-size:0.75rem;'
                    f'font-weight:600;">{pct:.0%}</span>'
                    f'</div></div></div>'
                )
            st.markdown(
                f'<div class="insight-card">{bars_html}</div>',
                unsafe_allow_html=True,
            )

    # ── Team Strengths & Weaknesses (offense) ───────────────────────
    if not team_hitters.empty and not h_proj.empty:
        st.markdown(f"### Offense Profile ({_view_label})")
        st.caption(
            f"Based on {'projected' if not use_priors else f'{PRIOR_SEASON} observed'}"
            " rates and Statcast metrics. Does not account for defense."
        )

        offense_metrics = []
        for label, key, higher_better in [
            ("K%", _h_k_col, False),
            ("BB%", _h_bb_col, True),
            ("Whiff%", "whiff_rate", False),
            ("Chase%", "chase_rate", False),
            ("Avg EV", "avg_exit_velo", True),
            ("Hard-Hit%", "hard_hit_pct", True),
        ]:
            if key not in team_hitters.columns or key not in h_proj.columns:
                continue
            team_avg = team_hitters[key].dropna().mean()
            league_avg = h_proj[key].dropna().mean()
            if pd.isna(team_avg) or pd.isna(league_avg) or league_avg == 0:
                continue
            diff = team_avg - league_avg
            if key in (
                _h_k_col, _h_bb_col, "whiff_rate", "chase_rate", "hard_hit_pct",
            ):
                diff_str = f"{diff * 100:+.1f}pp"
                team_str = f"{team_avg * 100:.1f}%"
                lg_str = f"{league_avg * 100:.1f}%"
            else:
                diff_str = f"{diff:+.1f}"
                team_str = f"{team_avg:.1f}"
                lg_str = f"{league_avg:.1f}"

            is_good = (
                (diff > 0 and higher_better)
                or (diff < 0 and not higher_better)
            )
            color = (
                POSITIVE if is_good
                else NEGATIVE if abs(diff) > 0.001
                else SLATE
            )
            offense_metrics.append({
                "Metric": label, "Team": team_str, "League Avg": lg_str,
                "Diff": diff_str, "_color": color, "_is_good": is_good,
            })

        if offense_metrics:
            strengths = [m for m in offense_metrics if m["_is_good"]]
            weaknesses_off = [m for m in offense_metrics if not m["_is_good"]]

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown(
                    f'<div style="color:{POSITIVE}; font-weight:600; '
                    f'margin-bottom:8px;">Strengths</div>',
                    unsafe_allow_html=True,
                )
                if strengths:
                    for m in strengths:
                        st.markdown(
                            f'<div style="padding:4px 0;">'
                            f'<span style="color:{CREAM};">{m["Metric"]}</span>: '
                            f'<span style="color:{POSITIVE}; font-weight:600;">'
                            f'{m["Team"]}</span> '
                            f'<span style="color:{SLATE};">'
                            f'(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f'<span style="color:{SLATE};">None vs league average</span>',
                        unsafe_allow_html=True,
                    )
            with col_w:
                st.markdown(
                    f'<div style="color:{NEGATIVE}; font-weight:600; '
                    f'margin-bottom:8px;">Weaknesses</div>',
                    unsafe_allow_html=True,
                )
                if weaknesses_off:
                    for m in weaknesses_off:
                        st.markdown(
                            f'<div style="padding:4px 0;">'
                            f'<span style="color:{CREAM};">{m["Metric"]}</span>: '
                            f'<span style="color:{NEGATIVE}; font-weight:600;">'
                            f'{m["Team"]}</span> '
                            f'<span style="color:{SLATE};">'
                            f'(lg: {m["League Avg"]}, {m["Diff"]})</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f'<span style="color:{SLATE};">None vs league average</span>',
                        unsafe_allow_html=True,
                    )

    # ── Offense Archetype Vulnerability ─────────────────────────────
    vuln_arch_career = load_hitter_vuln_arch_career()
    baselines_arch = load_baselines_arch()
    if (
        not vuln_arch_career.empty and not baselines_arch.empty
        and not cluster_meta.empty and not team_hitters.empty
    ):
        team_hitter_ids = set(team_hitters["batter_id"].astype(int))
        team_vuln = vuln_arch_career[
            vuln_arch_career["batter_id"].isin(team_hitter_ids)
        ]
        if not team_vuln.empty and "swings" in team_vuln.columns:
            team_arch = (
                team_vuln.groupby("pitch_archetype")
                .agg(
                    swings=("swings", "sum"),
                    whiffs=("whiffs", "sum"),
                    out_of_zone_pitches=("out_of_zone_pitches", "sum"),
                    chase_swings=("chase_swings", "sum"),
                )
                .reset_index()
            )
            team_arch["whiff_rate"] = (
                team_arch["whiffs"] / team_arch["swings"].clip(lower=1)
            )
            team_arch["chase_rate"] = (
                team_arch["chase_swings"]
                / team_arch["out_of_zone_pitches"].clip(lower=1)
            )

            bl_avg = baselines_arch.groupby("pitch_archetype").agg(
                lg_whiff=("whiff_rate", "mean"),
                lg_chase=("chase_rate", "mean"),
            ).reset_index()

            team_arch = team_arch.merge(
                cluster_meta[["pitch_archetype", "archetype_name"]],
                on="pitch_archetype", how="left",
            ).merge(bl_avg, on="pitch_archetype", how="left")
            team_arch = team_arch.sort_values("whiff_rate", ascending=False)

            st.markdown("### Offense Archetype Vulnerability")
            st.caption(
                "Team batting rates vs each pitch archetype, "
                "compared to league average."
            )

            vuln_rows = []
            for _, row in team_arch.iterrows():
                w_delta = (
                    row["whiff_rate"] - row["lg_whiff"]
                    if pd.notna(row.get("lg_whiff")) else 0
                )
                c_delta = (
                    row["chase_rate"] - row["lg_chase"]
                    if pd.notna(row.get("lg_chase")) else 0
                )
                w_color = (
                    NEGATIVE if w_delta > 0.01
                    else POSITIVE if w_delta < -0.01
                    else SLATE
                )
                c_color = (
                    NEGATIVE if c_delta > 0.01
                    else POSITIVE if c_delta < -0.01
                    else SLATE
                )
                name = row.get(
                    "archetype_name", f'Cluster {row["pitch_archetype"]}',
                )
                vuln_rows.append(
                    f'<tr>'
                    f'<td style="color:{CREAM};padding:6px 10px;">{name}</td>'
                    f'<td style="padding:6px 10px;">{row["whiff_rate"]:.1%}</td>'
                    f'<td style="color:{w_color};padding:6px 10px;">'
                    f'{w_delta:+.1%}</td>'
                    f'<td style="padding:6px 10px;">{row["chase_rate"]:.1%}</td>'
                    f'<td style="color:{c_color};padding:6px 10px;">'
                    f'{c_delta:+.1%}</td>'
                    f'</tr>'
                )
            if vuln_rows:
                table_html = (
                    f'<table style="width:100%;border-collapse:collapse;'
                    f'font-size:0.85rem;">'
                    f'<thead><tr style="border-bottom:1px solid {SLATE}44;">'
                    f'<th style="text-align:left;padding:6px 10px;'
                    f'color:{GOLD};">Archetype</th>'
                    f'<th style="padding:6px 10px;color:{SLATE};">Whiff%</th>'
                    f'<th style="padding:6px 10px;color:{SLATE};">'
                    f'\u0394 Lg</th>'
                    f'<th style="padding:6px 10px;color:{SLATE};">Chase%</th>'
                    f'<th style="padding:6px 10px;color:{SLATE};">'
                    f'\u0394 Lg</th>'
                    f'</tr></thead><tbody>{"".join(vuln_rows)}</tbody></table>'
                )
                st.markdown(
                    f'<div class="insight-card">{table_html}</div>',
                    unsafe_allow_html=True,
                )

    # ── Pitching staff profile — split by role ──────────────────────
    if not team_pitchers.empty and not p_proj.empty:
        team_sp = team_pitchers[team_pitchers["is_starter"] == True]
        team_rp = team_pitchers[team_pitchers["is_starter"] == False]
        lg_sp = p_proj[p_proj["is_starter"] == True]
        lg_rp = p_proj[p_proj["is_starter"] == False]

        if not team_sp.empty:
            st.markdown(f"### Rotation Profile ({_view_label})")
            _render_staff_metrics(team_sp, lg_sp, _p_k_col, _p_bb_col)

        if not team_rp.empty:
            st.markdown(f"### Bullpen Profile ({_view_label})")
            _render_staff_metrics(team_rp, lg_rp, _p_k_col, _p_bb_col)

    # ── Injured players ─────────────────────────────────────────────
    inj_df_full = load_preseason_injuries()
    if not inj_df_full.empty:
        team_inj = inj_df_full[
            (inj_df_full["team_abbr"] == selected_team)
            & (inj_df_full["est_missed_games"] > 0)
        ].sort_values("est_missed_games", ascending=False)
    else:
        team_inj = pd.DataFrame()

    if not team_inj.empty:
        st.markdown("### Injured Players")
        inj_rows = []
        for _, row in team_inj.iterrows():
            inj_rows.append({
                "Player": row["player_name"],
                "Pos": row["position"],
                "Injury": row["injury"],
                "Status": row["status"],
                "Est. Return": row["est_return_date"],
                "~Games Missed": int(row["est_missed_games"]),
            })
        st.dataframe(
            pd.DataFrame(inj_rows),
            use_container_width=True, hide_index=True,
        )

        # ── Injury impact notes ──────────────────────────────────────
        impact_lines: list[str] = []
        for _, row in team_inj.iterrows():
            pid = int(row["player_id"]) if pd.notna(row.get("player_id")) else None
            if pid is None:
                continue
            # Find this player in hitter or pitcher projections
            p_match = team_pitchers[team_pitchers["pitcher_id"] == pid] if not team_pitchers.empty else pd.DataFrame()
            h_match = team_hitters[team_hitters["batter_id"] == pid] if not team_hitters.empty else pd.DataFrame()

            if not p_match.empty:
                pr = p_match.iloc[0]
                score = pr.get("composite_score")
                k_rate = pr.get("projected_k_rate")
                bb_rate = pr.get("projected_bb_rate")
            elif not h_match.empty:
                hr = h_match.iloc[0]
                score = hr.get("composite_score")
                k_rate = hr.get("projected_k_rate")
                bb_rate = hr.get("projected_bb_rate")
            else:
                continue

            if pd.isna(score):
                continue

            # Build diamond string (0-5 scale from composite 0-1)
            diamond_val = score * 5
            filled = int(diamond_val)
            diamonds = "\u25c6" * filled + "\u25c7" * (5 - filled)

            parts = [f"{diamond_val:.1f}"]
            if pd.notna(k_rate):
                parts.append(f"{k_rate * 100:.1f}% K%")
            if pd.notna(bb_rate):
                parts.append(f"{bb_rate * 100:.1f}% BB%")

            player_name = row["player_name"]
            sep = " \u2014 "
            parts_str = sep.join(parts)

            impact_lines.append(
                f'<div style="padding:3px 0; font-size:0.85rem;">'
                f'<span style="color:{CREAM};">{player_name}</span> '
                f'<span style="color:{SLATE};">\u2014 Loses </span>'
                f'<span style="color:{GOLD};">{diamonds} {parts_str}</span>'
                f'</div>'
            )

        if impact_lines:
            joined_lines = "".join(impact_lines)
            st.markdown(
                f'<div style="margin-top:8px; padding:10px 14px; '
                f'border-left:3px solid {EMBER}; background:{EMBER}08;">'
                f'<div style="color:{EMBER}; font-weight:600; font-size:0.85rem; '
                f'margin-bottom:6px;">Projected Impact</div>'
                f'{joined_lines}</div>',
                unsafe_allow_html=True,
            )

    # ── Pitchers tables — split by role ─────────────────────────────
    if team_pitchers.empty:
        st.markdown("### Pitchers")
        st.info("No pitcher projections for this team.")
    else:
        team_sp_tbl = team_pitchers[team_pitchers["is_starter"] == True]
        team_rp_tbl = team_pitchers[team_pitchers["is_starter"] == False]

        st.markdown("### Starting Rotation")
        if team_sp_tbl.empty:
            st.info("No starters projected for this team.")
        else:
            sp_rows = _build_pitcher_rows(team_sp_tbl, injury_lookup, use_priors)
            st.dataframe(
                pd.DataFrame(sp_rows),
                use_container_width=True, hide_index=True,
            )

        st.markdown("### Bullpen")
        if team_rp_tbl.empty:
            st.info("No relievers projected for this team.")
        else:
            rp_rows = _build_pitcher_rows(team_rp_tbl, injury_lookup, use_priors)
            st.dataframe(
                pd.DataFrame(rp_rows),
                use_container_width=True, hide_index=True,
            )

    # ── Hitters table ───────────────────────────────────────────────
    st.markdown("### Hitters")

    if team_hitters.empty:
        st.info("No hitter projections for this team.")
    else:
        h_rows = []
        for _, row in team_hitters.sort_values(
            "composite_score", ascending=False,
        ).iterrows():
            pid = int(row["batter_id"])
            inj = injury_lookup.get(pid)
            name = row["batter_name"]
            if inj and inj["missed_games"] > 0:
                sev = inj["severity"]
                tag = (
                    "[IL-60]" if sev == "major"
                    else "[IL]" if sev == "significant"
                    else "[DTD]"
                )
                name = f"{tag} {name}"
            r: dict[str, object] = {
                "Name": name,
                "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
                "Bats": row.get("batter_stand", ""),
                "Rating": diamond_rating_text_composite(row["composite_score"]),
            }
            if use_priors:
                for label, key, _, _ in HITTER_STATS:
                    obs_col = f"observed_{key}"
                    if obs_col in row.index and pd.notna(row.get(obs_col)):
                        r[f"{label} ({PRIOR_SEASON})"] = fmt_stat(
                            row[obs_col], key,
                        )
                    else:
                        r[f"{label} ({PRIOR_SEASON})"] = "--"
                for label, key in [
                    ("Whiff%", "whiff_rate"),
                    ("Avg EV", "avg_exit_velo"),
                    ("Hard-Hit%", "hard_hit_pct"),
                ]:
                    if key in row.index and pd.notna(row.get(key)):
                        r[label] = fmt_stat(row[key], key)
                    else:
                        r[label] = "--"
            else:
                for label, key, _, _ in HITTER_STATS:
                    proj_col = f"projected_{key}"
                    delta_col = f"delta_{key}"
                    if proj_col in row.index and pd.notna(row.get(proj_col)):
                        proj_val = fmt_stat(row[proj_col], key)
                        delta_pp = row[delta_col] * 100
                        r[label] = (
                            f"{proj_val} ({delta_pp:+.1f})"
                            if abs(delta_pp) >= 0.05
                            else proj_val
                        )
                    else:
                        r[label] = "--"
                for c_label, c_prefix in [
                    ("Proj. HR", "total_hr"), ("Proj. BB", "total_bb"),
                ]:
                    mean_col = f"{c_prefix}_mean"
                    if mean_col in row.index and pd.notna(row.get(mean_col)):
                        r[c_label] = int(round(row[mean_col]))
                    else:
                        r[c_label] = "--"
            h_rows.append(r)
        st.dataframe(
            pd.DataFrame(h_rows),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Strengths/weaknesses compare team averages to league average "
        "across all projected players. "
        "Offense profile reflects batting projections only — does not "
        "account for defensive value. "
        + (
            f"Showing {PRIOR_SEASON} observed stats "
            f"(priors for the Bayesian model)."
            if use_priors
            else f"Deltas shown in parentheses (pp vs {PRIOR_SEASON})."
        )
    )


# ── Main page entry point ────────────────────────────────────────────

def page_team_overview() -> None:
    """Team-level view of projected pitchers and hitters with strengths/weaknesses."""
    st.markdown(
        '<div class="section-header">Team Overview</div>',
        unsafe_allow_html=True,
    )

    # Load data — dim_roster is the source of truth for team rosters
    teams_df = load_roster()
    if teams_df.empty:
        st.warning("No team data found. Run precompute first.")
        return

    team_lookup = get_team_lookup()
    injury_lookup = get_injury_lookup()

    all_teams = sorted(
        teams_df["team_abbr"].replace("", pd.NA).dropna().unique().tolist()
    )
    qp_team = st.query_params.get("team", "")
    default_team_idx = 0
    if qp_team in all_teams:
        default_team_idx = all_teams.index(qp_team)
    selected_team = st.selectbox(
        "Select team", all_teams, index=default_team_idx, key="team_select",
    )
    st.query_params["team"] = selected_team

    # Get all player IDs for this team
    team_pids = set(
        teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int)
    )

    # Load projections
    h_proj = load_projections("hitter")
    p_proj = load_projections("pitcher")
    h_count = load_counting("hitter")
    p_count = load_counting("pitcher")

    # Filter to team
    team_hitters = h_proj[h_proj["batter_id"].isin(team_pids)].copy()
    team_pitchers = p_proj[p_proj["pitcher_id"].isin(team_pids)].copy()

    # Merge counting stats
    if not h_count.empty:
        h_merge_cols = ["batter_id"] + [
            c for c in h_count.columns
            if c.endswith("_mean") or c.startswith("actual_")
        ]
        available = [c for c in h_merge_cols if c in h_count.columns]
        team_hitters = team_hitters.merge(
            h_count[available], on="batter_id", how="left",
        )

    if not p_count.empty:
        p_merge_cols = ["pitcher_id"] + [
            c for c in p_count.columns
            if c.endswith("_mean") or c.startswith("actual_")
        ]
        available = [c for c in p_merge_cols if c in p_count.columns]
        team_pitchers = team_pitchers.merge(
            p_count[available], on="pitcher_id", how="left",
        )

    # ── Header ──────────────────────────────────────────────────────
    _inj_full = load_preseason_injuries()
    n_injured = len(_inj_full[
        (_inj_full["team_abbr"] == selected_team)
        & (_inj_full["est_missed_games"] > 0)
    ]) if not _inj_full.empty else 0
    team_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{selected_team}</div>'
        f'<div class="brand-subtitle">'
        f'{len(team_pitchers)} pitchers | {len(team_hitters)} hitters | '
        f'{n_injured} injured</div>'
        f'</div>'
        f'<div style="color:{SLATE}; font-size:0.9rem;">'
        f'{CURRENT_SEASON} Season</div>'
        f'</div>'
    )
    st.markdown(team_header_html, unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────
    tab_overview, tab_depth, tab_trade = st.tabs(
        ["Overview", "Depth Chart", "Trade Simulator"],
    )

    with tab_overview:
        _render_overview_tab(
            selected_team, team_hitters, team_pitchers,
            h_proj, p_proj, injury_lookup,
        )

    with tab_depth:
        _render_depth_chart_tab(selected_team, teams_df)

    with tab_trade:
        _render_trade_simulator_tab(selected_team, teams_df)
