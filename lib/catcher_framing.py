"""Catcher framing lookup utilities.

Builds per-game catcher framing logit lifts for K and BB rates.
Catcher framing affects called-strike rates on borderline pitches,
which translates to +-0.5-1.0 pp on K% for elite/poor framers.

Synced from player_profiles/src/data/catcher_framing.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FRAMING_WEIGHT: float = 0.3


def get_catcher_framing_lift(
    catcher_id: int,
    season: int,
    framing_data: pd.DataFrame,
    weight: float = _FRAMING_WEIGHT,
) -> dict[str, float]:
    """Return logit lifts for K and BB from catcher framing effects."""
    if framing_data is None or framing_data.empty:
        return {"k_logit_lift": 0.0, "bb_logit_lift": 0.0}

    mask = framing_data["catcher_id"] == catcher_id
    catcher_rows = framing_data.loc[mask]

    if catcher_rows.empty:
        return {"k_logit_lift": 0.0, "bb_logit_lift": 0.0}

    exact = catcher_rows.loc[catcher_rows["season"] == season]
    if not exact.empty:
        raw_lift = float(exact.iloc[0]["logit_lift"])
    else:
        prior = catcher_rows.loc[catcher_rows["season"] <= season]
        if prior.empty:
            return {"k_logit_lift": 0.0, "bb_logit_lift": 0.0}
        raw_lift = float(prior.sort_values("season").iloc[-1]["logit_lift"])

    weighted_lift = raw_lift * weight

    return {
        "k_logit_lift": weighted_lift,
        "bb_logit_lift": -weighted_lift,
    }


def build_catcher_framing_lookup(
    train_seasons: list[int],
    test_season: int,
) -> dict[str, dict[tuple[int, int], float]]:
    """Build (game_pk, pitcher_id) -> catcher framing logit lifts.

    Requires database access (player_profiles queries). Only usable
    in the precompute pipeline, not at dashboard runtime.
    """
    from lib.db import read_sql  # noqa: F401 — deferred to avoid hard dep

    try:
        from src.data.queries import (
            get_catcher_framing_effects,
            get_catcher_game_assignments,
            get_game_starter_teams,
        )
    except ImportError:
        logger.warning(
            "Cannot import src.data.queries; build_catcher_framing_lookup "
            "is only available when running from the player_profiles repo."
        )
        return {"k": {}, "bb": {}}

    framing_data = get_catcher_framing_effects(seasons=train_seasons)
    if framing_data.empty:
        logger.warning("No catcher framing data for seasons %s", train_seasons)
        return {"k": {}, "bb": {}}

    catcher_assignments = get_catcher_game_assignments(int(test_season))

    if catcher_assignments.empty:
        logger.warning("No catcher lineup data for season %d", test_season)
        return {"k": {}, "bb": {}}

    pitcher_teams = get_game_starter_teams(int(test_season))

    if pitcher_teams.empty:
        logger.warning("No starter data for season %d", test_season)
        return {"k": {}, "bb": {}}

    catcher_lift_by_team: dict[tuple[int, int], dict[str, float]] = {}
    last_train = max(train_seasons)
    for _, row in catcher_assignments.iterrows():
        gpk = int(row["game_pk"])
        catcher_id_val = int(row["catcher_id"])
        team_id = int(row["team_id"])

        lifts = get_catcher_framing_lift(
            catcher_id=catcher_id_val,
            season=last_train,
            framing_data=framing_data,
        )
        catcher_lift_by_team[(gpk, team_id)] = lifts

    k_lifts: dict[tuple[int, int], float] = {}
    bb_lifts: dict[tuple[int, int], float] = {}

    for _, row in pitcher_teams.iterrows():
        gpk = int(row["game_pk"])
        pid = int(row["pitcher_id"])
        team_id = int(row["team_id"])
        team_key = (gpk, team_id)

        if team_key in catcher_lift_by_team:
            lifts = catcher_lift_by_team[team_key]
            k_lifts[(gpk, pid)] = lifts["k_logit_lift"]
            bb_lifts[(gpk, pid)] = lifts["bb_logit_lift"]

    n_entries = len(k_lifts)
    non_zero_k = sum(1 for v in k_lifts.values() if abs(v) > 0.001)
    logger.info(
        "Catcher framing lookup: %d pitcher-games, %d non-zero K lifts",
        n_entries, non_zero_k,
    )
    return {"k": k_lifts, "bb": bb_lifts}
