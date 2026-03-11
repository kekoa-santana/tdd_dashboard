"""Schedule page — Today's Games + Game Browser combined."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM, DARK_CARD, DARK_BORDER,
    POSITIVE, NEGATIVE, DASHBOARD_DIR, PRIOR_SEASON,
)
from services.data_loader import (
    load_todays_games, load_todays_sims, load_todays_lineups,
    load_update_metadata, load_pitcher_arsenal, load_hitter_vulnerability,
    load_projections, load_game_info, load_player_teams,
)
from utils.helpers import get_team_lookup
from components.metric_cards import metric_card


def _render_todays_games() -> None:
    """Today's MLB games with matchup analysis and K prop projections."""
    meta = load_update_metadata()
    schedule = load_todays_games()
    sims = load_todays_sims()
    lineups = load_todays_lineups()

    if schedule.empty:
        game_date = meta.get("game_date", "")
        st.info(
            f"No games loaded{f' for {game_date}' if game_date else ''}. "
            "Run `python scripts/update_in_season.py` to fetch today's schedule."
        )
        return

    game_date = schedule.iloc[0].get("game_date", "")
    n_games = len(schedule)

    updated_str = ""
    if meta.get("last_updated"):
        try:
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(meta["last_updated"])
            updated_str = f" | Updated {ts.strftime('%I:%M %p')}"
        except Exception:
            pass

    st.markdown(
        f'<div style="color:{SLATE}; font-size:0.9rem; margin-bottom:1rem;">'
        f'{game_date} | {n_games} games{updated_str}'
        f'</div>',
        unsafe_allow_html=True,
    )

    all_teams = sorted(set(
        schedule["away_abbr"].dropna().tolist() +
        schedule["home_abbr"].dropna().tolist()
    ))
    team_filter = st.selectbox(
        "Filter by team",
        ["All Teams"] + all_teams,
        key="today_team_filter",
    )

    if team_filter != "All Teams":
        schedule = schedule[
            (schedule["away_abbr"] == team_filter) |
            (schedule["home_abbr"] == team_filter)
        ]

    for _, game in schedule.iterrows():
        gpk = game["game_pk"]
        away_abbr = game.get("away_abbr", "?")
        home_abbr = game.get("home_abbr", "?")
        game_time = game.get("game_time", "")
        status = game.get("status", "")

        game_sims = sims[sims["game_pk"] == gpk] if not sims.empty else pd.DataFrame()
        away_sim = game_sims[game_sims["side"] == "away"].iloc[0] if not game_sims.empty and (game_sims["side"] == "away").any() else None
        home_sim = game_sims[game_sims["side"] == "home"].iloc[0] if not game_sims.empty and (game_sims["side"] == "home").any() else None

        status_str = f" — {status}" if status and "Scheduled" not in status else ""
        st.markdown(
            f'<div style="background:{DARK_CARD}; border:1px solid {DARK_BORDER}; '
            f'border-radius:8px; padding:1rem 1.5rem; margin-bottom:1rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">'
            f'<span style="color:{GOLD}; font-size:1.1rem; font-weight:700;">'
            f'{away_abbr} @ {home_abbr}</span>'
            f'<span style="color:{SLATE}; font-size:0.85rem;">{game_time}{status_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_a, col_h = st.columns(2)

        for col, sim, side_label, pitcher_name_field in [
            (col_a, away_sim, f"{away_abbr} SP", "away_pitcher_name"),
            (col_h, home_sim, f"{home_abbr} SP", "home_pitcher_name"),
        ]:
            with col:
                pp_name = game.get(pitcher_name_field, "TBD")
                if not pp_name:
                    pp_name = "TBD"

                if sim is not None:
                    k_rate_pct = sim["projected_k_rate"] * 100
                    exp_k = sim["expected_k"]
                    has_lineup = sim.get("has_lineup", False)
                    lineup_tag = "" if has_lineup else " (no lineup)"

                    p_lines = []
                    for line_col in ["p_over_5_5", "p_over_6_5", "p_over_7_5"]:
                        if line_col in sim.index and pd.notna(sim.get(line_col)):
                            line_val = line_col.replace("p_over_", "").replace("_", ".")
                            p = sim[line_col] * 100
                            color = POSITIVE if p > 55 else NEGATIVE if p < 45 else SLATE
                            p_lines.append(
                                f'<span style="color:{color};">'
                                f'O{line_val}: {p:.0f}%</span>'
                            )

                    p_line_html = " | ".join(p_lines) if p_lines else ""

                    st.markdown(
                        f'<div style="font-size:0.95rem; font-weight:600; color:{CREAM};">'
                        f'{side_label}: {pp_name}</div>'
                        f'<div style="font-size:0.8rem; color:{SLATE}; margin-top:2px;">'
                        f'K%: {k_rate_pct:.1f}% | E[K]: {exp_k:.1f}{lineup_tag}</div>'
                        f'<div style="font-size:0.8rem; margin-top:2px;">{p_line_html}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="font-size:0.95rem; font-weight:600; color:{CREAM};">'
                        f'{side_label}: {pp_name}</div>'
                        f'<div style="font-size:0.8rem; color:{SLATE};">No projection available</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('</div>', unsafe_allow_html=True)

        if not lineups.empty:
            game_lu = lineups[lineups["game_pk"] == gpk]
            if not game_lu.empty:
                with st.expander(f"Lineup Details — {away_abbr} @ {home_abbr}"):
                    for sim, side, opp_label in [
                        (away_sim, "away", home_abbr),
                        (home_sim, "home", away_abbr),
                    ]:
                        if sim is None:
                            continue

                        pitcher_name = sim["pitcher_name"]
                        opp_team_id = game.get(f"{'home' if side == 'away' else 'away'}_team_id")
                        opp_lu = game_lu[game_lu["team_id"] == opp_team_id].sort_values("batting_order")

                        if opp_lu.empty:
                            continue

                        st.markdown(
                            f'<div style="color:{GOLD}; font-size:0.9rem; font-weight:600; '
                            f'margin:0.5rem 0 0.3rem 0;">'
                            f'{pitcher_name} vs {opp_label} Lineup</div>',
                            unsafe_allow_html=True,
                        )

                        lu_rows = []
                        for _, brow in opp_lu.head(9).iterrows():
                            lu_rows.append({
                                "#": int(brow["batting_order"]),
                                "Batter": brow.get("batter_name", "Unknown"),
                            })

                        if lu_rows:
                            st.dataframe(
                                pd.DataFrame(lu_rows),
                                use_container_width=True,
                                hide_index=True,
                                height=350,
                            )

    if not sims.empty:
        st.markdown("---")
        st.markdown(
            f'<div style="color:{SLATE}; font-size:0.8rem;">'
            f'{len(sims)} pitchers simulated | '
            f'{sims["has_lineup"].sum()} with lineup data | '
            f'10,000 Monte Carlo draws per pitcher</div>',
            unsafe_allow_html=True,
        )


def _render_game_browser() -> None:
    """Browse historical games grouped by game_pk with team and pitcher selectors."""
    from lib.matchup import score_matchup
    from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

    st.caption(
        f"Browse {PRIOR_SEASON} games by team. Select a game, then view each pitcher's "
        "matchup breakdown against the opposing lineup."
    )

    game_logs_path = DASHBOARD_DIR / "pitcher_game_logs.parquet"
    lineups_path = DASHBOARD_DIR / "game_lineups.parquet"
    batter_ks_path = DASHBOARD_DIR / "game_batter_ks.parquet"

    if not all(p.exists() for p in [game_logs_path, lineups_path, batter_ks_path]):
        st.warning(
            "Game browser data not found. Re-run "
            "`python scripts/precompute_dashboard_data.py` to generate it."
        )
        return

    game_logs = pd.read_parquet(game_logs_path)
    all_lineups = pd.read_parquet(lineups_path)
    all_batter_ks = pd.read_parquet(batter_ks_path)

    arsenal_df = load_pitcher_arsenal()
    vuln_df = load_hitter_vulnerability(career=True)
    if arsenal_df.empty or vuln_df.empty:
        st.warning("Matchup profile data not found.")
        return

    @st.cache_data
    def _enrich_game_logs(_game_logs: pd.DataFrame) -> pd.DataFrame:
        game_info = load_game_info()
        if game_info.empty:
            return _game_logs
        return _game_logs.merge(game_info, on="game_pk", how="left")

    game_logs = _enrich_game_logs(game_logs)

    if "game_date" not in game_logs.columns:
        st.warning("Game info not available. Re-run precompute.")
        return

    team_lookup = get_team_lookup()

    all_team_names = set()
    if "home_team_name" in game_logs.columns:
        all_team_names.update(game_logs["home_team_name"].dropna().unique())
    if "away_team_name" in game_logs.columns:
        all_team_names.update(game_logs["away_team_name"].dropna().unique())
    all_team_names = sorted(t for t in all_team_names if t)

    selected_team = st.selectbox(
        "Select team", all_team_names, key="gb_team"
    )

    team_games = game_logs[
        (game_logs["home_team_name"] == selected_team) |
        (game_logs["away_team_name"] == selected_team)
    ].copy()

    if team_games.empty:
        st.info("No games found for this team.")
        return

    game_summary = (
        team_games.groupby("game_pk")
        .agg(
            game_date=("game_date", "first"),
            home_team_name=("home_team_name", "first"),
            away_team_name=("away_team_name", "first"),
        )
        .reset_index()
        .sort_values("game_date", ascending=False)
    )

    game_options = {}
    for _, g in game_summary.iterrows():
        gpk = int(g["game_pk"])
        date_str = str(g["game_date"])[:10]
        home = g["home_team_name"] or "?"
        away = g["away_team_name"] or "?"
        label = f"{date_str} — {away} @ {home}"
        game_options[label] = gpk

    selected_game_label = st.selectbox(
        "Select Game", list(game_options.keys()), key="gb_game",
    )
    selected_gpk = game_options[selected_game_label]

    game_pitchers = game_logs[game_logs["game_pk"] == selected_gpk].copy()
    if game_pitchers.empty:
        st.info("No pitcher data for this game.")
        return

    game_meta = game_summary[game_summary["game_pk"] == selected_gpk].iloc[0]
    home_name = game_meta["home_team_name"] or "Home"
    away_name = game_meta["away_team_name"] or "Away"

    pitcher_opts = {}
    for _, pr in game_pitchers.sort_values(
        ["is_starter", "pitcher_name"], ascending=[False, True]
    ).iterrows():
        pid = int(pr["pitcher_id"])
        pname = pr["pitcher_name"]
        team = team_lookup.get(pid, "")
        role = "SP" if pr.get("is_starter") else "RP"
        ks = int(pr["strike_outs"]) if pd.notna(pr.get("strike_outs")) else 0
        ip = pr.get("innings_pitched", 0)
        dname = f"{pname} ({team}, {role}) — {ks} K, {ip} IP" if team else f"{pname} ({role}) — {ks} K, {ip} IP"
        pitcher_opts[dname] = pid

    selected_pitcher_display = st.selectbox(
        "Select Pitcher", list(pitcher_opts.keys()), key="gb_pitcher",
    )
    pitcher_id = pitcher_opts[selected_pitcher_display]

    game_row = game_pitchers[game_pitchers["pitcher_id"] == pitcher_id].iloc[0]
    game_lineups_this = all_lineups[all_lineups["game_pk"] == selected_gpk]

    if game_lineups_this.empty:
        st.warning("No lineup data found for this game.")
        return

    bk_game = all_batter_ks[
        (all_batter_ks["game_pk"] == selected_gpk) &
        (all_batter_ks["pitcher_id"] == pitcher_id)
    ]
    faced_batters = set(bk_game["batter_id"].tolist())
    home_tid = game_row.get("home_team_id")

    opposing_lineup = None
    opponent_name = ""
    for tid in game_lineups_this["team_id"].unique():
        team_lineup = game_lineups_this[
            game_lineups_this["team_id"] == tid
        ].sort_values("batting_order")
        lineup_batters = set(team_lineup["player_id"].tolist())
        if lineup_batters & faced_batters:
            opposing_lineup = team_lineup
            opponent_name = home_name if tid == home_tid else away_name
            break

    if opposing_lineup is None or opposing_lineup.empty:
        st.warning("Could not determine opposing lineup for this pitcher.")
        return

    actual_ks_game = bk_game
    actual_k_map = dict(zip(actual_ks_game["batter_id"], actual_ks_game["k"]))
    actual_pa_map = dict(zip(actual_ks_game["batter_id"], actual_ks_game["pa"]))

    baselines_pt = {
        pt: {"whiff_rate": vals.get("whiff_rate", 0.25)}
        for pt, vals in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    display_rows = []
    total_actual_k = int(game_row.get("strike_outs", 0))
    total_matchup_lift = 0.0
    n_scored = 0

    for _, brow in opposing_lineup.iterrows():
        bid = int(brow["player_id"])
        bname = brow.get("batter_name", "Unknown")
        order = int(brow["batting_order"])
        bteam = team_lookup.get(bid, "")

        matchup = score_matchup(
            pitcher_id=pitcher_id,
            batter_id=bid,
            pitcher_arsenal=arsenal_df,
            hitter_vuln=vuln_df,
            baselines_pt=baselines_pt,
        )

        lift = matchup.get("matchup_k_logit_lift", 0.0)
        if np.isnan(lift):
            lift = 0.0
        mwhiff = matchup.get("matchup_whiff_rate", np.nan)
        bwhiff = matchup.get("baseline_whiff_rate", np.nan)
        reliability = matchup.get("avg_reliability", 0.0)

        actual_k = actual_k_map.get(bid, 0)
        actual_pa = actual_pa_map.get(bid, 0)

        if not np.isnan(lift):
            total_matchup_lift += lift
            n_scored += 1

        row = {
            "#": order,
            "Batter": f"{bname} ({bteam})" if bteam else bname,
            "Matchup Whiff%": f"{mwhiff:.1%}" if pd.notna(mwhiff) else "--",
            "Baseline Whiff%": f"{bwhiff:.1%}" if pd.notna(bwhiff) else "--",
            "K Lift": f"{lift:+.3f}" if lift != 0 else "0.000",
            "Reliability": f"{reliability:.0%}",
            "PA": actual_pa,
            "K": actual_k,
        }
        display_rows.append(row)

    date_str = str(game_meta["game_date"])[:10]
    ip = game_row.get("innings_pitched", 0)
    bf = int(game_row.get("batters_faced", 0)) if pd.notna(game_row.get("batters_faced")) else 0
    avg_lift = total_matchup_lift / n_scored if n_scored > 0 else 0.0
    pitcher_name = game_row.get("pitcher_name", "Unknown")
    pitcher_team = team_lookup.get(pitcher_id, "")

    gb_header_html = (
        f'<div class="brand-header">'
        f'<div>'
        f'<div class="brand-title">{pitcher_name}{f" ({pitcher_team})" if pitcher_team else ""}</div>'
        f'<div class="brand-subtitle">{date_str} — {away_name} @ {home_name} | {ip} IP, {bf} BF</div>'
        f'</div>'
        f'<div style="font-size:1.2rem; font-weight:600;">'
        f'<span style="color:{GOLD};">{total_actual_k} K</span>'
        f'<span style="color:{SLATE};"> | Avg Lift: {avg_lift:+.3f}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(gb_header_html, unsafe_allow_html=True)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    lineup_ks = sum(r["K"] for r in display_rows)
    lineup_pa = sum(r["PA"] for r in display_rows)
    k_rate_actual = lineup_ks / lineup_pa if lineup_pa > 0 else 0

    if avg_lift > 0.05:
        lift_color = POSITIVE
        lift_word = "favorable"
    elif avg_lift < -0.05:
        lift_color = NEGATIVE
        lift_word = "unfavorable"
    else:
        lift_color = SLATE
        lift_word = "neutral"

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-bullet">
            <span class="dot" style="background:{GOLD};"></span>
            Actual: <strong>{total_actual_k} K</strong> in {bf} BF
            ({k_rate_actual:.1%} K rate)
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{lift_color};"></span>
            Matchup model rated this lineup as
            <strong style="color:{lift_color};">{lift_word}</strong>
            for strikeouts (avg logit lift: {avg_lift:+.3f})
        </div>
        <div class="insight-bullet">
            <span class="dot" style="background:{SLATE};"></span>
            Positive K Lift = hitter is more vulnerable to this pitcher's arsenal.
            Negative = hitter handles it better than average.
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_schedule() -> None:
    """Combined Schedule page — Today's Games + Game Browser."""
    st.markdown('<div class="section-header">Schedule</div>',
                unsafe_allow_html=True)

    view = st.radio(
        "View",
        ["Today's Games", "Game Browser"],
        horizontal=True,
        key="schedule_view",
    )

    if view == "Today's Games":
        _render_todays_games()
    else:
        _render_game_browser()
