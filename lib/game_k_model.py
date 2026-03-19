"""
Step 14: Game-level K posterior Monte Carlo engine.

Combines:
- Pitcher K% posterior samples (Layer 1)
- BF distribution (Step 13)
- Per-batter matchup logit lifts (Layer 2)

to produce a full posterior over game strikeout totals.

Synced from: player_profiles/src/models/game_k_model.py
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from lib.bf_model import draw_bf_samples, get_bf_distribution
from lib.matchup import score_matchup, score_matchup_for_stat
from lib.rest_adjustment import (
    apply_rest_to_bf,
    compute_rest_for_game,
    get_rest_adjustment,
)

logger = logging.getLogger(__name__)

# Clip bounds for logit transform (avoid infinities)
_CLIP_LO = 1e-6
_CLIP_HI = 1 - 1e-6

# ---------------------------------------------------------------------------
# League-average TTO logit lifts (relative to overall rate).
# Computed from 2018-2025 fact_pa:
#   TTO1 K=.2378  BB=.0848  HR=.0298   overall K=.2256  BB=.0812  HR=.0316
#   TTO2 K=.2095  BB=.0748  HR=.0339
#   TTO3 K=.1942  BB=.0747  HR=.0366
# Lift = logit(tto_rate) - logit(overall_rate)
# ---------------------------------------------------------------------------
_LEAGUE_TTO_LOGIT_LIFTS: dict[str, np.ndarray] = {
    "k": np.array([
        logit(0.23782) - logit(0.22557),   # TTO1: +0.066
        logit(0.20952) - logit(0.22557),   # TTO2: -0.085
        logit(0.19421) - logit(0.22557),   # TTO3: -0.171
    ]),
    "bb": np.array([
        logit(0.08483) - logit(0.08115),   # TTO1: +0.047
        logit(0.07479) - logit(0.08115),   # TTO2: -0.082
        logit(0.07470) - logit(0.08115),   # TTO3: -0.083
    ]),
    "hr": np.array([
        logit(0.02979) - logit(0.03162),   # TTO1: -0.062
        logit(0.03385) - logit(0.03162),   # TTO2: +0.072
        logit(0.03658) - logit(0.03162),   # TTO3: +0.160
    ]),
}

# Number of BF in each TTO block (9 batters per time through)
_BF_PER_TTO = 9


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Logit with clipping."""
    return logit(np.clip(p, _CLIP_LO, _CLIP_HI))


def build_tto_logit_lifts(
    tto_profiles: pd.DataFrame | None,
    pitcher_id: int,
    season: int,
    stat_name: str = "k",
) -> np.ndarray:
    """Get TTO logit lifts for a pitcher, falling back to league average.

    Parameters
    ----------
    tto_profiles : pd.DataFrame or None
        Output of ``get_tto_adjustment_profiles()``.  Must contain columns:
        pitcher_id, season, tto, {stat}_rate, overall_{stat}_rate.
        If None, returns league-average lifts.
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season to look up.
    stat_name : str
        One of 'k', 'bb', 'hr'.

    Returns
    -------
    np.ndarray
        Shape (3,) logit lifts for TTO 1, 2, 3.
    """
    sn = stat_name.lower()
    league_lifts = _LEAGUE_TTO_LOGIT_LIFTS.get(sn)
    if league_lifts is None:
        return np.zeros(3)

    if tto_profiles is None or tto_profiles.empty:
        return league_lifts.copy()

    mask = (
        (tto_profiles["pitcher_id"] == pitcher_id)
        & (tto_profiles["season"] == season)
    )
    pitcher_data = tto_profiles[mask]

    if len(pitcher_data) < 3:
        return league_lifts.copy()

    rate_col = f"{sn}_rate"
    overall_col = f"overall_{sn}_rate"

    if rate_col not in pitcher_data.columns or overall_col not in pitcher_data.columns:
        return league_lifts.copy()

    pitcher_data = pitcher_data.sort_values("tto")
    tto_rates = pitcher_data[rate_col].values.astype(float)
    overall_rate = pitcher_data[overall_col].values[0].astype(float)

    # Avoid degenerate rates
    if overall_rate < _CLIP_LO or overall_rate > _CLIP_HI:
        return league_lifts.copy()

    overall_logit = logit(np.clip(overall_rate, _CLIP_LO, _CLIP_HI))
    tto_logits = logit(np.clip(tto_rates, _CLIP_LO, _CLIP_HI))
    pitcher_lifts = tto_logits - overall_logit

    # Reliability-weight toward league average based on PA
    pa_counts = pitcher_data["pa_count"].values.astype(float)
    reliability = np.clip(pa_counts / 100.0, 0.0, 1.0)  # full weight at 100 PA
    blended = reliability * pitcher_lifts + (1.0 - reliability) * league_lifts

    return blended


def _assign_bf_to_tto(bf: int) -> np.ndarray:
    """Assign each BF to a TTO block (0-indexed: 0=TTO1, 1=TTO2, 2=TTO3).

    Parameters
    ----------
    bf : int
        Total batters faced in the game.

    Returns
    -------
    np.ndarray
        Shape (bf,) with values 0, 1, or 2 indicating TTO block.
    """
    tto_assignments = np.zeros(bf, dtype=int)
    for i in range(bf):
        tto_assignments[i] = min(i // _BF_PER_TTO, 2)
    return tto_assignments


def simulate_game_ks(
    pitcher_k_rate_samples: np.ndarray,
    bf_mu: float,
    bf_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    umpire_k_logit_lift: float = 0.0,
    weather_k_logit_lift: float = 0.0,
    tto_logit_lifts: np.ndarray | None = None,
    rest_k_logit_lift: float = 0.0,
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo simulation of game strikeout totals.

    Parameters
    ----------
    pitcher_k_rate_samples : np.ndarray
        K% posterior samples from Layer 1 (values in [0, 1]).
    bf_mu : float
        Mean batters faced for this pitcher (rest-adjusted if applicable).
    bf_sigma : float
        Std of batters faced (rest-adjusted if applicable).
    lineup_matchup_lifts : np.ndarray or None
        Shape (9,) logit-scale lifts per batting order slot.
        Positive = batter more vulnerable → more Ks.
        None = no matchup adjustment (baseline mode).
    umpire_k_logit_lift : float
        Logit-scale shift for HP umpire K-rate tendency.
        Positive = umpire calls more Ks than average.
    weather_k_logit_lift : float
        Logit-scale shift for weather effect on K-rate.
        Positive = weather conditions increase Ks (e.g. cold).
    tto_logit_lifts : np.ndarray or None
        Shape (3,) logit-scale lifts for TTO 1, 2, 3+.
        Applied per-BF based on times-through-order block.
        None = no TTO adjustment (flat rate, backward compatible).
    rest_k_logit_lift : float
        Logit-scale shift for days-rest effect on K-rate.
        From ``rest_adjustment.get_rest_adjustment()``.
    n_draws : int
        Number of Monte Carlo draws.
    bf_min : int
        Minimum BF per game.
    bf_max : int
        Maximum BF per game.
    random_seed : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer K totals per simulated game.
    """
    rng = np.random.default_rng(random_seed)

    # Resample pitcher K% to n_draws if needed
    if len(pitcher_k_rate_samples) != n_draws:
        idx = rng.choice(len(pitcher_k_rate_samples), size=n_draws, replace=True)
        k_rate_draws = pitcher_k_rate_samples[idx]
    else:
        k_rate_draws = pitcher_k_rate_samples.copy()

    # Draw BF samples
    bf_draws = draw_bf_samples(
        mu_bf=bf_mu, sigma_bf=bf_sigma,
        n_draws=n_draws, bf_min=bf_min, bf_max=bf_max, rng=rng,
    )

    # Default: no matchup adjustment
    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(9)

    # Convert pitcher K% to logit scale and apply context adjustments
    k_logit = (
        _safe_logit(k_rate_draws)
        + umpire_k_logit_lift
        + weather_k_logit_lift
        + rest_k_logit_lift
    )

    # Vectorize by grouping draws with same BF value
    k_totals = np.zeros(n_draws, dtype=int)

    unique_bf = np.unique(bf_draws)
    for bf_val in unique_bf:
        mask = bf_draws == bf_val
        n_bf_draws = mask.sum()
        bf_int = int(bf_val)

        game_ks = np.zeros(n_bf_draws, dtype=int)
        k_logit_subset = k_logit[mask]

        if tto_logit_lifts is not None:
            # TTO-aware: iterate over each BF position, applying the
            # correct TTO lift and matchup lift per batter faced.
            # BF 0-8 → TTO1, 9-17 → TTO2, 18+ → TTO3.
            # Group consecutive BF positions by (tto, slot) to batch
            # Bernoulli draws into Binomial where possible.
            #
            # Build a (tto_block, slot) → count mapping for this bf_int.
            tto_slot_counts: dict[tuple[int, int], int] = {}
            for bf_idx in range(bf_int):
                tto_block = min(bf_idx // _BF_PER_TTO, 2)
                slot = bf_idx % 9
                key = (tto_block, slot)
                tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1

            for (tto_block, slot), count in tto_slot_counts.items():
                adjusted_logit = (
                    k_logit_subset
                    + lineup_matchup_lifts[slot]
                    + tto_logit_lifts[tto_block]
                )
                adjusted_p = expit(adjusted_logit)
                game_ks += rng.binomial(n=count, p=adjusted_p)
        else:
            # Original flat-rate path (no TTO adjustment)
            base_pa = bf_int // 9
            extra = bf_int % 9
            pa_per_slot = np.full(9, base_pa, dtype=int)
            pa_per_slot[:extra] += 1

            for slot in range(9):
                if pa_per_slot[slot] == 0:
                    continue
                adjusted_logit = k_logit_subset + lineup_matchup_lifts[slot]
                adjusted_p = expit(adjusted_logit)
                slot_ks = rng.binomial(n=pa_per_slot[slot], p=adjusted_p)
                game_ks += slot_ks

        k_totals[mask] = game_ks

    return k_totals


def simulate_game_outcomes(
    k_rate_samples: np.ndarray,
    bb_rate_samples: np.ndarray | None,
    hr_rate_samples: np.ndarray | None,
    bf_mu: float,
    bf_sigma: float,
    lineup_k_lifts: np.ndarray | None = None,
    lineup_bb_lifts: np.ndarray | None = None,
    lineup_hr_lifts: np.ndarray | None = None,
    umpire_k_logit_lift: float = 0.0,
    weather_k_logit_lift: float = 0.0,
    tto_k_lifts: np.ndarray | None = None,
    tto_bb_lifts: np.ndarray | None = None,
    tto_hr_lifts: np.ndarray | None = None,
    rest_k_logit_lift: float = 0.0,
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Monte Carlo simulation of game K, BB, and HR totals.

    Shares BF draws across all stats. BB and HR are optional — when None,
    only K results are returned.

    Parameters
    ----------
    k_rate_samples : np.ndarray
        K% posterior samples (values in [0, 1]).
    bb_rate_samples : np.ndarray or None
        BB% posterior samples. None → K-only mode.
    hr_rate_samples : np.ndarray or None
        HR/BF posterior samples. None → no HR output.
    bf_mu, bf_sigma : float
        BF distribution parameters.
    lineup_k_lifts, lineup_bb_lifts, lineup_hr_lifts : np.ndarray or None
        Shape (9,) logit-scale lifts per batting order slot.
    umpire_k_logit_lift, weather_k_logit_lift : float
        K-only context lifts.
    tto_k_lifts, tto_bb_lifts, tto_hr_lifts : np.ndarray or None
        Shape (3,) logit lifts per TTO block.
    rest_k_logit_lift : float
        Days-rest K-rate logit lift.
    n_draws, bf_min, bf_max, random_seed : int
        Simulation parameters.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"k"`` (always), optionally ``"bb"`` and ``"hr"``.
    """
    rng = np.random.default_rng(random_seed)

    # Shared BF draws
    bf_draws = draw_bf_samples(
        mu_bf=bf_mu, sigma_bf=bf_sigma,
        n_draws=n_draws, bf_min=bf_min, bf_max=bf_max, rng=rng,
    )

    # Build per-stat config: (rate_samples, matchup_lifts, tto_lifts, context_lift)
    stats_config: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, float]] = [
        ("k", k_rate_samples,
         lineup_k_lifts if lineup_k_lifts is not None else np.zeros(9),
         tto_k_lifts,
         umpire_k_logit_lift + weather_k_logit_lift + rest_k_logit_lift),
    ]
    if bb_rate_samples is not None:
        stats_config.append((
            "bb", bb_rate_samples,
            lineup_bb_lifts if lineup_bb_lifts is not None else np.zeros(9),
            tto_bb_lifts,
            0.0,  # no umpire/weather/rest lift for BB
        ))
    if hr_rate_samples is not None:
        stats_config.append((
            "hr", hr_rate_samples,
            lineup_hr_lifts if lineup_hr_lifts is not None else np.zeros(9),
            tto_hr_lifts,
            0.0,  # no umpire/weather/rest lift for HR
        ))

    results: dict[str, np.ndarray] = {}

    for stat_name, rate_samples, matchup_lifts, tto_lifts, ctx_lift in stats_config:
        # Resample rate draws to n_draws
        if len(rate_samples) != n_draws:
            idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
            rate_draws = rate_samples[idx]
        else:
            rate_draws = rate_samples.copy()

        base_logit = _safe_logit(rate_draws) + ctx_lift
        totals = np.zeros(n_draws, dtype=int)
        unique_bf = np.unique(bf_draws)

        for bf_val in unique_bf:
            mask = bf_draws == bf_val
            n_bf_draws = mask.sum()
            bf_int = int(bf_val)
            game_counts = np.zeros(n_bf_draws, dtype=int)
            logit_subset = base_logit[mask]

            if tto_lifts is not None:
                tto_slot_counts: dict[tuple[int, int], int] = {}
                for bf_idx in range(bf_int):
                    tto_block = min(bf_idx // _BF_PER_TTO, 2)
                    slot = bf_idx % 9
                    key = (tto_block, slot)
                    tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1

                for (tto_block, slot), count in tto_slot_counts.items():
                    adjusted = logit_subset + matchup_lifts[slot] + tto_lifts[tto_block]
                    game_counts += rng.binomial(n=count, p=expit(adjusted))
            else:
                base_pa = bf_int // 9
                extra = bf_int % 9
                pa_per_slot = np.full(9, base_pa, dtype=int)
                pa_per_slot[:extra] += 1
                for slot in range(9):
                    if pa_per_slot[slot] == 0:
                        continue
                    adjusted = logit_subset + matchup_lifts[slot]
                    game_counts += rng.binomial(n=pa_per_slot[slot], p=expit(adjusted))

            totals[mask] = game_counts

        results[stat_name] = totals

    return results


def compute_over_probs(
    samples: np.ndarray,
    lines: list[float] | None = None,
) -> pd.DataFrame:
    """Compute P(over) for a set of lines — stat-agnostic version.

    Parameters
    ----------
    samples : np.ndarray
        Monte Carlo game total samples.
    lines : list[float] or None
        Lines to evaluate. Default: [0.5, 1.5, ..., 12.5].

    Returns
    -------
    pd.DataFrame
        Columns: line, p_over, p_under, expected, std.
    """
    if lines is None:
        lines = [x + 0.5 for x in range(13)]

    expected = float(np.mean(samples))
    std = float(np.std(samples))

    records = []
    for line in lines:
        p_over = float(np.mean(samples > line))
        records.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            "expected": expected,
            "std": std,
        })

    return pd.DataFrame(records)


def compute_k_over_probs(
    k_samples: np.ndarray,
    lines: list[float] | None = None,
) -> pd.DataFrame:
    """Compute P(over X.5) for standard K prop lines.

    Parameters
    ----------
    k_samples : np.ndarray
        Monte Carlo K total samples.
    lines : list[float] or None
        Lines to evaluate. Default: [0.5, 1.5, ..., 12.5].

    Returns
    -------
    pd.DataFrame
        Columns: line, p_over, p_under, expected_k, std_k.
    """
    if lines is None:
        lines = [x + 0.5 for x in range(13)]

    expected_k = float(np.mean(k_samples))
    std_k = float(np.std(k_samples))

    records = []
    for line in lines:
        p_over = float(np.mean(k_samples > line))
        records.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            "expected_k": expected_k,
            "std_k": std_k,
        })

    return pd.DataFrame(records)
