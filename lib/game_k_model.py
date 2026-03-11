"""
Game-level K posterior Monte Carlo engine.

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
from lib.matchup import score_matchup

logger = logging.getLogger(__name__)

# Clip bounds for logit transform (avoid infinities)
_CLIP_LO = 1e-6
_CLIP_HI = 1 - 1e-6


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Logit with clipping."""
    return logit(np.clip(p, _CLIP_LO, _CLIP_HI))


def simulate_game_ks(
    pitcher_k_rate_samples: np.ndarray,
    bf_mu: float,
    bf_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    umpire_k_logit_lift: float = 0.0,
    weather_k_logit_lift: float = 0.0,
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
        Mean batters faced for this pitcher.
    bf_sigma : float
        Std of batters faced.
    lineup_matchup_lifts : np.ndarray or None
        Shape (9,) logit-scale lifts per batting order slot.
        Positive = batter more vulnerable → more Ks.
        None = no matchup adjustment (baseline mode).
    umpire_k_logit_lift : float
        Logit-scale shift for HP umpire K-rate tendency.
    weather_k_logit_lift : float
        Logit-scale shift for weather effect on K-rate.
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

    # Convert pitcher K% to logit scale and apply umpire + weather adjustments
    k_logit = _safe_logit(k_rate_draws) + umpire_k_logit_lift + weather_k_logit_lift

    # Vectorize by grouping draws with same BF value
    k_totals = np.zeros(n_draws, dtype=int)

    unique_bf = np.unique(bf_draws)
    for bf_val in unique_bf:
        mask = bf_draws == bf_val
        n_bf_draws = mask.sum()
        bf_int = int(bf_val)

        # Allocate PA across 9 batting order slots
        base_pa = bf_int // 9
        extra = bf_int % 9
        pa_per_slot = np.full(9, base_pa, dtype=int)
        pa_per_slot[:extra] += 1

        # For each slot with PA > 0, simulate Ks
        game_ks = np.zeros(n_bf_draws, dtype=int)
        k_logit_subset = k_logit[mask]

        for slot in range(9):
            if pa_per_slot[slot] == 0:
                continue
            # Adjust K rate by matchup lift for this slot
            adjusted_logit = k_logit_subset + lineup_matchup_lifts[slot]
            adjusted_p = expit(adjusted_logit)
            # Binomial draw: K per slot
            slot_ks = rng.binomial(n=pa_per_slot[slot], p=adjusted_p)
            game_ks += slot_ks

        k_totals[mask] = game_ks

    return k_totals


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
