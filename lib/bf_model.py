"""
Batters-Faced distribution model with empirical Bayes partial pooling.

BF for starters is very stable (mean~22, std~4.5). A shrinkage estimator
is appropriate — no MCMC needed.

Synced from: player_profiles/src/models/bf_model.py
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Population defaults (validated against 2022-2025 starter data)
DEFAULT_POP_BF_MU = 22.0
DEFAULT_POP_WITHIN_STD = 3.4
DEFAULT_SHRINKAGE_K = 2.4  # sigma^2 / tau^2 ≈ 11.56 / 4.84

# Pitches-per-PA adjustment
POP_MEAN_PPA = 3.91        # League avg P/PA for starters (2023-2025)
PPA_ADJ_WEIGHT = 0.15      # Blend weight for P/PA implied BF adjustment


def get_bf_distribution(
    pitcher_id: int,
    season: int,
    bf_priors: pd.DataFrame,
    pop_mu: float = DEFAULT_POP_BF_MU,
    pop_within_std: float = DEFAULT_POP_WITHIN_STD,
) -> dict[str, Any]:
    """Look up BF distribution parameters for a pitcher.

    Parameters
    ----------
    pitcher_id : int
        MLB pitcher ID.
    season : int
        Season to look up.
    bf_priors : pd.DataFrame
        Output of ``compute_pitcher_bf_priors``.
    pop_mu : float
        Fallback population mean.
    pop_within_std : float
        Fallback population std.

    Returns
    -------
    dict
        Keys: mu_bf, sigma_bf, reliability, dist_type.
    """
    mask = (bf_priors["pitcher_id"] == pitcher_id) & (bf_priors["season"] == season)
    rows = bf_priors[mask]

    if rows.empty:
        return {
            "mu_bf": pop_mu,
            "sigma_bf": pop_within_std,
            "reliability": 0.0,
            "dist_type": "population_fallback",
        }

    row = rows.iloc[0]
    return {
        "mu_bf": float(row["mu_bf"]),
        "sigma_bf": float(row["sigma_bf"]),
        "reliability": float(row["reliability"]),
        "dist_type": "shrinkage",
    }


def draw_bf_samples(
    mu_bf: float,
    sigma_bf: float,
    n_draws: int,
    bf_min: int = 3,
    bf_max: int = 35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw integer BF samples from a truncated normal.

    Parameters
    ----------
    mu_bf : float
        Mean BF.
    sigma_bf : float
        Std BF.
    n_draws : int
        Number of samples to draw.
    bf_min : int
        Minimum BF (inclusive).
    bf_max : int
        Maximum BF (inclusive).
    rng : numpy Generator, optional
        For reproducibility. If None, uses default.

    Returns
    -------
    np.ndarray
        Integer BF values, shape (n_draws,).
    """
    if sigma_bf <= 0:
        return np.full(n_draws, int(np.clip(np.round(mu_bf), bf_min, bf_max)), dtype=int)

    a = (bf_min - mu_bf) / sigma_bf
    b = (bf_max - mu_bf) / sigma_bf

    if rng is None:
        rng = np.random.default_rng()

    # Use scipy truncnorm with numpy rng
    samples = stats.truncnorm.rvs(
        a, b, loc=mu_bf, scale=sigma_bf, size=n_draws,
        random_state=rng.integers(0, 2**31),
    )

    return np.clip(np.round(samples), bf_min, bf_max).astype(int)
