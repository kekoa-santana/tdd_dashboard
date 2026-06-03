"""Game-sim projection utilities.

Pure projection/data-shaping helpers extracted from the (removed) matchup
narrative engine.  No matchup/scouting logic lives here -- this module only
reshapes pre-computed ``game_props`` into a sims-like frame for downstream
prediction archiving and dashboard sim rendering.
"""
from __future__ import annotations

import pandas as pd


def _build_sims_from_game_props(
    game_props: pd.DataFrame,
    game_date: str | None = None,
) -> pd.DataFrame:
    """Pivot game_props pitcher rows into a sims-like DataFrame.

    game_props has one row per (player, stat). This function pivots the
    stat dimension into columns so downstream code can access a single
    row per pitcher with expected_k, expected_bb, etc.

    Parameters
    ----------
    game_props : pd.DataFrame
        Full game_props.parquet (all dates).
    game_date : str, optional
        Filter to this date. If None, uses the latest date.

    Returns
    -------
    pd.DataFrame
        One row per pitcher appearance with columns matching the old
        todays_sims schema: pitcher_id, game_pk, team_abbr, opp_abbr,
        expected_k, expected_bb, expected_hr, expected_ip, dk_mean,
        projected_k_rate, avg_matchup_lift, has_lineup.
    """
    if game_props.empty:
        return pd.DataFrame()

    if game_date is None:
        game_date = game_props["game_date"].max()

    pitchers = game_props[
        (game_props["game_date"] == game_date)
        & (game_props["player_type"] == "pitcher")
    ].copy()

    if pitchers.empty:
        return pd.DataFrame()

    # Pivot stat → expected value
    stat_pivot = pitchers.pivot_table(
        index="player_id", columns="stat", values="expected", aggfunc="first",
    )
    stat_map = {"K": "expected_k", "BB": "expected_bb", "HR": "expected_hr",
                "H": "expected_h", "Outs": "expected_outs"}
    stat_pivot = stat_pivot.rename(columns=stat_map)

    # Get shared pitcher-level columns from the first stat row per pitcher
    meta = pitchers.drop_duplicates("player_id").set_index("player_id")[
        ["player_name", "game_pk", "team", "opponent", "expected_ip",
         "expected_bf", "dk_mean", "side"]
    ].rename(columns={"team": "team_abbr", "opponent": "opp_abbr",
                       "player_name": "pitcher_name"})

    if "dk_mean" not in meta.columns:
        meta["dk_mean"] = 0.0

    result = meta.join(stat_pivot, how="left").reset_index().rename(
        columns={"player_id": "pitcher_id"},
    )

    # Derive projected_k_rate from expected_k / expected_bf
    if "expected_k" in result.columns and "expected_bf" in result.columns:
        bf = result["expected_bf"].fillna(22.0).clip(lower=1)
        result["projected_k_rate"] = result["expected_k"].fillna(0) / bf
    else:
        result["projected_k_rate"] = 0.0

    # Columns not available in game_props — safe defaults
    result["avg_matchup_lift"] = 0.0
    result["has_lineup"] = True  # precompute always has lineup context

    return result
