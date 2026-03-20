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


# ---------------------------------------------------------------------------
# Generalized simulation functions (all prop types: K, BB, HR, H, Outs)
# ---------------------------------------------------------------------------

# Mapping from stat_name -> matchup lift key in score_matchup_for_stat results
_STAT_LIFT_KEYS: dict[str, str] = {
    "k": "matchup_k_logit_lift",
    "bb": "matchup_bb_logit_lift",
    "hr": "matchup_hr_logit_lift",
}


def simulate_game_stat(
    rate_samples: np.ndarray,
    opp_mu: float,
    opp_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    context_logit_lift: float = 0.0,
    tto_logit_lifts: np.ndarray | None = None,
    n_draws: int = 4000,
    opp_min: int = 3,
    opp_max: int = 35,
    n_slots: int = 9,
    random_seed: int = 42,
) -> np.ndarray:
    """Stat-agnostic Binomial Monte Carlo simulation.

    Works for any rate-based stat (K, BB, H) for either pitchers or batters.
    For pitchers: opp = BF, n_slots = 9 (lineup).
    For batters: opp = PA, n_slots = 1 (no lineup splitting needed).

    Parameters
    ----------
    rate_samples : np.ndarray
        Rate posterior samples (values in [0, 1]).
    opp_mu : float
        Mean opportunities (BF for pitchers, PA for batters).
    opp_sigma : float
        Std of opportunities.
    lineup_matchup_lifts : np.ndarray or None
        Shape (n_slots,) logit-scale lifts per slot.
    context_logit_lift : float
        Additional logit-scale context shift (umpire, weather, park).
    tto_logit_lifts : np.ndarray or None
        Shape (3,) logit-scale lifts for TTO 1, 2, 3+.
        Only used in pitcher mode (n_slots == 9).
    n_draws, opp_min, opp_max, n_slots, random_seed : int
        Simulation parameters.

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer stat totals.
    """
    rng = np.random.default_rng(random_seed)

    if len(rate_samples) != n_draws:
        idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
        rate_draws = rate_samples[idx]
    else:
        rate_draws = rate_samples.copy()

    opp_draws = draw_bf_samples(
        mu_bf=opp_mu, sigma_bf=opp_sigma,
        n_draws=n_draws, bf_min=opp_min, bf_max=opp_max, rng=rng,
    )

    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(n_slots)

    rate_logit = _safe_logit(rate_draws) + context_logit_lift
    stat_totals = np.zeros(n_draws, dtype=int)

    if n_slots == 1:
        adjusted_logit = rate_logit + lineup_matchup_lifts[0]
        adjusted_p = expit(adjusted_logit)
        stat_totals = rng.binomial(n=opp_draws.astype(int), p=adjusted_p)
    else:
        unique_opp = np.unique(opp_draws)
        for opp_val in unique_opp:
            mask = opp_draws == opp_val
            n_opp_draws = mask.sum()
            opp_int = int(opp_val)
            game_stats = np.zeros(n_opp_draws, dtype=int)
            rate_logit_subset = rate_logit[mask]

            if tto_logit_lifts is not None:
                tto_slot_counts: dict[tuple[int, int], int] = {}
                for bf_idx in range(opp_int):
                    tto_block = min(bf_idx // _BF_PER_TTO, 2)
                    slot = bf_idx % n_slots
                    key = (tto_block, slot)
                    tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1
                for (tto_block, slot), count in tto_slot_counts.items():
                    adjusted_logit = (
                        rate_logit_subset
                        + lineup_matchup_lifts[slot]
                        + tto_logit_lifts[tto_block]
                    )
                    adjusted_p = expit(adjusted_logit)
                    game_stats += rng.binomial(n=count, p=adjusted_p)
            else:
                base_pa = opp_int // n_slots
                extra = opp_int % n_slots
                pa_per_slot = np.full(n_slots, base_pa, dtype=int)
                pa_per_slot[:extra] += 1
                for slot in range(n_slots):
                    if pa_per_slot[slot] == 0:
                        continue
                    adjusted_logit = rate_logit_subset + lineup_matchup_lifts[slot]
                    adjusted_p = expit(adjusted_logit)
                    game_stats += rng.binomial(n=pa_per_slot[slot], p=adjusted_p)

            stat_totals[mask] = game_stats

    return stat_totals


def simulate_game_stat_poisson(
    rate_samples: np.ndarray,
    opp_mu: float,
    opp_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    context_logit_lift: float = 0.0,
    park_factor: float = 1.0,
    tto_logit_lifts: np.ndarray | None = None,
    n_draws: int = 4000,
    opp_min: int = 3,
    opp_max: int = 35,
    n_slots: int = 9,
    random_seed: int = 42,
) -> np.ndarray:
    """Poisson Monte Carlo simulation for rare events (HR).

    lambda = rate * opportunities * park_factor

    Parameters
    ----------
    rate_samples : np.ndarray
        Rate posterior samples (values in [0, 1]).
    opp_mu, opp_sigma : float
        Opportunity distribution parameters.
    lineup_matchup_lifts : np.ndarray or None
        Shape (n_slots,) logit-scale lifts per slot.
    context_logit_lift : float
        Additional logit-scale context shift.
    park_factor : float
        Multiplicative park factor for the stat (e.g., HR park factor).
    tto_logit_lifts : np.ndarray or None
        Shape (3,) logit-scale lifts for TTO 1, 2, 3+.
    n_draws, opp_min, opp_max, n_slots, random_seed : int
        Simulation parameters.

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer stat totals.
    """
    rng = np.random.default_rng(random_seed)

    if len(rate_samples) != n_draws:
        idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
        rate_draws = rate_samples[idx]
    else:
        rate_draws = rate_samples.copy()

    opp_draws = draw_bf_samples(
        mu_bf=opp_mu, sigma_bf=opp_sigma,
        n_draws=n_draws, bf_min=opp_min, bf_max=opp_max, rng=rng,
    )

    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(n_slots)

    rate_logit = _safe_logit(rate_draws) + context_logit_lift
    stat_totals = np.zeros(n_draws, dtype=int)

    if n_slots == 1:
        adjusted_logit = rate_logit + lineup_matchup_lifts[0]
        adjusted_rate = expit(adjusted_logit)
        lam = adjusted_rate * opp_draws * park_factor
        stat_totals = rng.poisson(lam=lam)
    else:
        unique_opp = np.unique(opp_draws)
        for opp_val in unique_opp:
            mask = opp_draws == opp_val
            n_opp_draws = mask.sum()
            opp_int = int(opp_val)
            game_stats = np.zeros(n_opp_draws, dtype=int)
            rate_logit_subset = rate_logit[mask]

            if tto_logit_lifts is not None:
                tto_slot_counts: dict[tuple[int, int], int] = {}
                for bf_idx in range(opp_int):
                    tto_block = min(bf_idx // _BF_PER_TTO, 2)
                    slot = bf_idx % n_slots
                    key = (tto_block, slot)
                    tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1
                for (tto_block, slot), count in tto_slot_counts.items():
                    adjusted_logit = (
                        rate_logit_subset
                        + lineup_matchup_lifts[slot]
                        + tto_logit_lifts[tto_block]
                    )
                    adjusted_rate = expit(adjusted_logit)
                    lam = adjusted_rate * count * park_factor
                    game_stats += rng.poisson(lam=lam)
            else:
                base_pa = opp_int // n_slots
                extra = opp_int % n_slots
                pa_per_slot = np.full(n_slots, base_pa, dtype=int)
                pa_per_slot[:extra] += 1
                for slot in range(n_slots):
                    if pa_per_slot[slot] == 0:
                        continue
                    adjusted_logit = rate_logit_subset + lineup_matchup_lifts[slot]
                    adjusted_rate = expit(adjusted_logit)
                    lam = adjusted_rate * pa_per_slot[slot] * park_factor
                    game_stats += rng.poisson(lam=lam)

            stat_totals[mask] = game_stats

    return stat_totals


def _compute_lineup_matchup_lifts_for_stat(
    stat_name: str,
    pitcher_id: int,
    lineup_batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score matchups for a 9-batter lineup for any stat type.

    Parameters
    ----------
    stat_name : str
        Stat to score ('k', 'bb', 'hr', etc.).
    pitcher_id : int
        Pitcher MLB ID.
    lineup_batter_ids : list[int]
        Exactly 9 batter IDs in batting order.
    pitcher_arsenal, hitter_vuln : pd.DataFrame
        Arsenal and vulnerability profiles.
    baselines_pt : dict
        League baselines per pitch type.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        (lifts array shape (9,), list of per-batter matchup dicts)
    """
    lift_key = _STAT_LIFT_KEYS.get(stat_name.lower(), "matchup_logit_lift")
    lifts = np.zeros(len(lineup_batter_ids))
    matchup_details: list[dict[str, Any]] = []

    for i, batter_id in enumerate(lineup_batter_ids):
        result = score_matchup_for_stat(
            stat_name=stat_name,
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        lift = result.get(lift_key, 0.0)
        if np.isnan(lift):
            lift = 0.0
        lifts[i] = lift
        matchup_details.append(result)

    return lifts, matchup_details


def predict_game_stat(
    stat_name: str,
    pitcher_id: int,
    season: int,
    lineup_batter_ids: list[int] | None,
    rate_samples: np.ndarray,
    bf_priors: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    context_logit_lift: float = 0.0,
    lineup_proneness_lift: float = 0.0,
    park_factor: float = 1.0,
    model_type: str = "binomial",
    days_rest: int | None = None,
    tto_logit_lifts: np.ndarray | None = None,
    n_draws: int = 4000,
    random_seed: int = 42,
    bf_min: int = 3,
    bf_max: int = 35,
) -> dict[str, Any]:
    """Full game stat prediction for a pitcher, combining all layers.

    Parameters
    ----------
    stat_name : str
        Stat being predicted ('k', 'bb', 'hr', 'h', 'outs').
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season for BF lookup.
    lineup_batter_ids : list[int] or None
        9 batter IDs in order. None = no matchup adjustment.
    rate_samples : np.ndarray
        Posterior rate samples from Layer 1 (values in [0, 1]).
    bf_priors : pd.DataFrame
        BF priors from ``compute_pitcher_bf_priors``.
    pitcher_arsenal, hitter_vuln : pd.DataFrame or None
        Required if lineup given.
    baselines_pt : dict or None
        League baselines per pitch type.
    context_logit_lift : float
        Logit-scale context shift (umpire + weather + park combined).
    lineup_proneness_lift : float
        Logit-scale lift from aggregate lineup proneness.
    park_factor : float
        Multiplicative park factor. Used only for Poisson model (HR).
    model_type : str
        'binomial' or 'poisson'.
    days_rest : int or None
        Days since pitcher's last start.
    tto_logit_lifts : np.ndarray or None
        Shape (3,) logit lifts for TTO 1, 2, 3+.
    n_draws, random_seed, bf_min, bf_max : int
        Simulation parameters.

    Returns
    -------
    dict
        Keys: stat_samples, over_probs, expected_{stat_name},
        std_{stat_name}, bf_mu, bf_sigma, lineup_matchup_lifts,
        per_batter_details, context_logit_lift, lineup_proneness_lift,
        park_factor, days_rest, rest_bucket.
    """
    bf_info = get_bf_distribution(pitcher_id, season, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    _REST_STATS = {"k", "bb", "hr"}
    rest_adj = get_rest_adjustment(days_rest)
    sn_lower = stat_name.lower()
    rest_lift = (
        rest_adj.get(f"{sn_lower}_lift", 0.0)
        if sn_lower in _REST_STATS else 0.0
    )
    if sn_lower in _REST_STATS:
        bf_mu, bf_sigma = apply_rest_to_bf(bf_mu, bf_sigma, days_rest)

    lineup_lifts = None
    per_batter_details: list[dict[str, Any]] = []
    if lineup_batter_ids is not None and len(lineup_batter_ids) == 9:
        if (pitcher_arsenal is not None and hitter_vuln is not None
                and baselines_pt is not None):
            lineup_lifts, per_batter_details = (
                _compute_lineup_matchup_lifts_for_stat(
                    stat_name, pitcher_id, lineup_batter_ids,
                    pitcher_arsenal, hitter_vuln, baselines_pt,
                )
            )

    total_context = context_logit_lift + lineup_proneness_lift + rest_lift

    if model_type == "poisson":
        stat_samples = simulate_game_stat_poisson(
            rate_samples=rate_samples, opp_mu=bf_mu, opp_sigma=bf_sigma,
            lineup_matchup_lifts=lineup_lifts,
            context_logit_lift=total_context, park_factor=park_factor,
            tto_logit_lifts=tto_logit_lifts,
            n_draws=n_draws, opp_min=bf_min, opp_max=bf_max,
            n_slots=9, random_seed=random_seed,
        )
    else:
        stat_samples = simulate_game_stat(
            rate_samples=rate_samples, opp_mu=bf_mu, opp_sigma=bf_sigma,
            lineup_matchup_lifts=lineup_lifts,
            context_logit_lift=total_context,
            tto_logit_lifts=tto_logit_lifts,
            n_draws=n_draws, opp_min=bf_min, opp_max=bf_max,
            n_slots=9, random_seed=random_seed,
        )

    over_probs = compute_over_probs(stat_samples)

    sn = stat_name.lower()
    return {
        "stat_samples": stat_samples,
        "over_probs": over_probs,
        f"expected_{sn}": float(np.mean(stat_samples)),
        f"std_{sn}": float(np.std(stat_samples)),
        "bf_mu": bf_mu,
        "bf_sigma": bf_sigma,
        "lineup_matchup_lifts": lineup_lifts,
        "per_batter_details": per_batter_details,
        "context_logit_lift": context_logit_lift,
        "lineup_proneness_lift": lineup_proneness_lift,
        "park_factor": park_factor,
        "days_rest": days_rest,
        "rest_bucket": rest_adj["rest_bucket"],
    }


def simulate_batter_game_stat(
    rate_samples: np.ndarray,
    pa_mu: float,
    pa_sigma: float,
    matchup_logit_lift: float = 0.0,
    context_logit_lift: float = 0.0,
    park_factor: float = 1.0,
    model_type: str = "binomial",
    n_draws: int = 4000,
    pa_min: int = 1,
    pa_max: int = 7,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo simulation for a single batter's game stat total.

    Parameters
    ----------
    rate_samples : np.ndarray
        Rate posterior samples (values in [0, 1]).
    pa_mu, pa_sigma : float
        PA distribution parameters.
    matchup_logit_lift : float
        Logit-scale matchup adjustment vs the opposing pitcher.
    context_logit_lift : float
        Additional logit-scale context shift (umpire, weather).
    park_factor : float
        Multiplicative park factor. Used only for Poisson model (HR).
    model_type : str
        'binomial' or 'poisson'.
    n_draws, pa_min, pa_max, random_seed : int
        Simulation parameters.

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer stat totals.
    """
    matchup_array = np.array([matchup_logit_lift])

    if model_type == "poisson":
        return simulate_game_stat_poisson(
            rate_samples=rate_samples, opp_mu=pa_mu, opp_sigma=pa_sigma,
            lineup_matchup_lifts=matchup_array,
            context_logit_lift=context_logit_lift, park_factor=park_factor,
            n_draws=n_draws, opp_min=pa_min, opp_max=pa_max,
            n_slots=1, random_seed=random_seed,
        )
    else:
        return simulate_game_stat(
            rate_samples=rate_samples, opp_mu=pa_mu, opp_sigma=pa_sigma,
            lineup_matchup_lifts=matchup_array,
            context_logit_lift=context_logit_lift,
            n_draws=n_draws, opp_min=pa_min, opp_max=pa_max,
            n_slots=1, random_seed=random_seed,
        )


def predict_batter_game(
    stat_name: str,
    batter_id: int,
    pitcher_id: int,
    rate_samples: np.ndarray,
    pa_mu: float,
    pa_sigma: float,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    context_logit_lift: float = 0.0,
    opposing_pitcher_lift: float = 0.0,
    park_factor: float = 1.0,
    model_type: str = "binomial",
    default_lines: list[float] | None = None,
    n_draws: int = 4000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Full game stat prediction for a single batter.

    Parameters
    ----------
    stat_name : str
        Stat being predicted ('k', 'bb', 'hr', 'h').
    batter_id : int
        Batter MLB ID.
    pitcher_id : int
        Opposing pitcher MLB ID.
    rate_samples : np.ndarray
        Posterior rate samples from Layer 1 (values in [0, 1]).
    pa_mu, pa_sigma : float
        PA distribution parameters.
    pitcher_arsenal, hitter_vuln : pd.DataFrame or None
        Required for matchup scoring.
    baselines_pt : dict or None
        League baselines per pitch type.
    context_logit_lift : float
        Logit-scale context shift (umpire + weather).
    opposing_pitcher_lift : float
        Logit-scale lift from opposing pitcher quality.
    park_factor : float
        Multiplicative park factor. Used only for Poisson model (HR).
    model_type : str
        'binomial' or 'poisson'.
    default_lines : list[float] or None
        Lines for P(over) computation.
    n_draws, random_seed : int
        Simulation parameters.

    Returns
    -------
    dict
        Keys: stat_samples, over_probs, expected_{stat_name},
        std_{stat_name}, pa_mu, pa_sigma, matchup_logit_lift,
        matchup_detail, opposing_pitcher_lift.
    """
    sn = stat_name.lower()
    lift_key = _STAT_LIFT_KEYS.get(sn, "matchup_logit_lift")

    matchup_lift = 0.0
    matchup_detail: dict[str, Any] = {}
    if (pitcher_arsenal is not None and hitter_vuln is not None
            and baselines_pt is not None):
        matchup_detail = score_matchup_for_stat(
            stat_name=stat_name,
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        matchup_lift = matchup_detail.get(lift_key, 0.0)
        if np.isnan(matchup_lift):
            matchup_lift = 0.0

    total_context = context_logit_lift + opposing_pitcher_lift

    stat_samples = simulate_batter_game_stat(
        rate_samples=rate_samples, pa_mu=pa_mu, pa_sigma=pa_sigma,
        matchup_logit_lift=matchup_lift,
        context_logit_lift=total_context, park_factor=park_factor,
        model_type=model_type, n_draws=n_draws, random_seed=random_seed,
    )

    over_probs = compute_over_probs(stat_samples, lines=default_lines)

    return {
        "stat_samples": stat_samples,
        "over_probs": over_probs,
        f"expected_{sn}": float(np.mean(stat_samples)),
        f"std_{sn}": float(np.std(stat_samples)),
        "pa_mu": pa_mu,
        "pa_sigma": pa_sigma,
        "matchup_logit_lift": matchup_lift,
        "matchup_detail": matchup_detail,
        "opposing_pitcher_lift": opposing_pitcher_lift,
    }


# ---------------------------------------------------------------------------
# Catcher framing lift
# ---------------------------------------------------------------------------

_FRAMING_WEIGHT: float = 0.3


def get_catcher_framing_lift(
    catcher_id: int,
    season: int,
    framing_data: pd.DataFrame,
    weight: float = _FRAMING_WEIGHT,
) -> dict[str, float]:
    """Return logit lifts for K and BB from catcher framing effects.

    Parameters
    ----------
    catcher_id : int
        Catcher MLB ID.
    season : int
        Season to look up.
    framing_data : pd.DataFrame
        Must contain columns: catcher_id, season, logit_lift.
    weight : float
        Scaling factor applied to raw framing logit lift.

    Returns
    -------
    dict[str, float]
        Keys: ``k_logit_lift`` (positive = more Ks),
        ``bb_logit_lift`` (negative = fewer BBs when framing is good).
    """
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

    Parameters
    ----------
    train_seasons : list[int]
        Seasons to compute framing effects from.
    test_season : int
        Season whose games to map.

    Returns
    -------
    dict[str, dict[tuple[int, int], float]]
        ``{"k": {(game_pk, pitcher_id): lift},
          "bb": {(game_pk, pitcher_id): lift}}``.
    """
    try:
        from lib.db import read_sql
        from lib.queries import get_catcher_framing_effects
    except ImportError:
        logger.warning(
            "lib.db or lib.queries not available for catcher framing lookup"
        )
        return {"k": {}, "bb": {}}

    framing_data = get_catcher_framing_effects(seasons=train_seasons)
    if framing_data.empty:
        logger.warning("No catcher framing data for seasons %s", train_seasons)
        return {"k": {}, "bb": {}}

    catcher_assignments = read_sql(f"""
        SELECT fl.game_pk, fl.player_id AS catcher_id, fl.team_id
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE fl.position = 'C'
          AND fl.is_starter = true
          AND dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    if catcher_assignments.empty:
        logger.warning("No catcher lineup data for season %d", test_season)
        return {"k": {}, "bb": {}}

    pitcher_teams = read_sql(f"""
        SELECT fpg.game_pk, fpg.player_id AS pitcher_id, fpg.team_id
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND dg.season = {int(test_season)}
          AND dg.game_type = 'R'
    """, {})

    if pitcher_teams.empty:
        logger.warning("No starter data for season %d", test_season)
        return {"k": {}, "bb": {}}

    catcher_lift_by_team: dict[tuple[int, int], dict[str, float]] = {}
    last_train = max(train_seasons)
    for _, row in catcher_assignments.iterrows():
        gpk = int(row["game_pk"])
        catcher_id = int(row["catcher_id"])
        team_id = int(row["team_id"])
        lifts = get_catcher_framing_lift(
            catcher_id=catcher_id, season=last_train,
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
