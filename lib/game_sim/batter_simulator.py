"""
Batter game simulator.

Simulates a batter's plate appearances across a game, determining
for each PA whether the batter faces the starter (with specific
matchup adjustments) or a reliever (team bullpen aggregate rates),
then resolving outcomes.

Produces joint distributions over K, BB, H, HR, TB, R, RBI per game.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from lib.game_sim.bip_model import (
    BIP_DOUBLE,
    BIP_SINGLE,
    BIP_TRIPLE,
    BIPOutcomeModel,
)
from lib.game_sim.pa_outcome_model import (
    PA_DOUBLE,
    PA_HBP,
    PA_HOME_RUN,
    PA_OUT,
    PA_SINGLE,
    PA_STRIKEOUT,
    PA_TRIPLE,
    PA_WALK,
    PAOutcomeModel,
)
from lib.game_sim.batter_pa_model import (
    draw_total_pa,
    split_pa_starter_reliever,
)
from lib.bf_model import draw_bf_samples

logger = logging.getLogger(__name__)

_CLIP_LO = 1e-6
_CLIP_HI = 1 - 1e-6

# Max PAs a batter can get in a game
MAX_BATTER_PA = 7

# League average rates (2022-2025) for logit centering
LEAGUE_K_RATE = 0.226
LEAGUE_BB_RATE = 0.082
LEAGUE_HR_RATE = 0.031

# ---------------------------------------------------------------------------
# League-average TTO logit lifts (from batter's perspective).
# Mirrors _LEAGUE_TTO_LOGIT_LIFTS in game_k_model.py.
# TTO1: pitcher is sharper (higher K/BB, lower HR)
# TTO2/3: pitcher fatigues (lower K/BB, higher HR)
# Lift = logit(tto_rate) - logit(overall_rate)
# ---------------------------------------------------------------------------
_TTO_K_LIFTS = np.array([
    logit(0.23782) - logit(0.22557),   # TTO1: +0.066
    logit(0.20952) - logit(0.22557),   # TTO2: -0.085
    logit(0.19421) - logit(0.22557),   # TTO3: -0.171
])
_TTO_BB_LIFTS = np.array([
    logit(0.08483) - logit(0.08115),   # TTO1: +0.047
    logit(0.07479) - logit(0.08115),   # TTO2: -0.082
    logit(0.07470) - logit(0.08115),   # TTO3: -0.083
])
_TTO_HR_LIFTS = np.array([
    logit(0.02979) - logit(0.03162),   # TTO1: -0.062
    logit(0.03385) - logit(0.03162),   # TTO2: +0.072
    logit(0.03658) - logit(0.03162),   # TTO3: +0.160
])
_BF_PER_TTO = 9


def _safe_logit(p: float | np.ndarray) -> float | np.ndarray:
    return logit(np.clip(p, _CLIP_LO, _CLIP_HI))


@dataclass
class BatterSimulationResult:
    """Results from a batter game simulation.

    All arrays have shape (n_sims,).
    """

    k_samples: np.ndarray
    bb_samples: np.ndarray
    h_samples: np.ndarray
    hr_samples: np.ndarray
    single_samples: np.ndarray
    double_samples: np.ndarray
    triple_samples: np.ndarray
    tb_samples: np.ndarray
    r_samples: np.ndarray
    rbi_samples: np.ndarray
    hbp_samples: np.ndarray
    pa_samples: np.ndarray
    pa_vs_starter_samples: np.ndarray
    pa_vs_reliever_samples: np.ndarray
    n_sims: int = 0

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all stats."""
        stats = {}
        for name in [
            "k", "bb", "h", "hr", "single", "double", "triple",
            "tb", "r", "rbi", "hbp", "pa",
        ]:
            samples = getattr(self, f"{name}_samples")
            stats[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "q10": float(np.percentile(samples, 10)),
                "q90": float(np.percentile(samples, 90)),
            }
        return stats

    def over_probs(
        self,
        stat: str,
        lines: list[float] | None = None,
    ) -> pd.DataFrame:
        """Compute P(over X.5) for prop lines."""
        samples = getattr(self, f"{stat}_samples")
        if lines is None:
            max_val = int(np.percentile(samples, 99)) + 2
            lines = [x + 0.5 for x in range(max_val)]

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


def simulate_batter_game(
    batter_k_rate_samples: np.ndarray,
    batter_bb_rate_samples: np.ndarray,
    batter_hr_rate_samples: np.ndarray,
    batting_order: int,
    starter_k_rate: float,
    starter_bb_rate: float,
    starter_hr_rate: float,
    starter_bf_mu: float,
    starter_bf_sigma: float,
    matchup_k_lift: float = 0.0,
    matchup_bb_lift: float = 0.0,
    matchup_hr_lift: float = 0.0,
    bullpen_k_rate: float = 0.253,
    bullpen_bb_rate: float = 0.084,
    bullpen_hr_rate: float = 0.024,
    batter_babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    weather_k_lift: float = 0.0,
    tto_k_lifts: np.ndarray | None = None,
    tto_bb_lifts: np.ndarray | None = None,
    tto_hr_lifts: np.ndarray | None = None,
    n_sims: int = 10_000,
    random_seed: int = 42,
) -> BatterSimulationResult:
    """Run vectorized Monte Carlo batter game simulation.

    Applies times-through-order (TTO) adjustments to starter PAs:
    the pitcher is sharper the first time through (higher K, lower HR)
    and fatigues on subsequent passes (lower K, higher HR). TTO lifts
    default to league-average constants when not provided.

    Parameters
    ----------
    batter_k_rate_samples : np.ndarray
        Batter K% posterior samples from Layer 1.
    batter_bb_rate_samples : np.ndarray
        Batter BB% posterior samples.
    batter_hr_rate_samples : np.ndarray
        Batter HR/PA posterior samples.
    batting_order : int
        Batting order position (1-9).
    starter_k_rate : float
        Opposing starter's K% posterior mean.
    starter_bb_rate : float
        Opposing starter's BB% posterior mean.
    starter_hr_rate : float
        Opposing starter's HR/BF posterior mean.
    starter_bf_mu : float
        Starter's expected BF for this game.
    starter_bf_sigma : float
        Starter's BF std.
    matchup_k_lift : float
        Batter-vs-starter matchup logit lift for K.
    matchup_bb_lift : float
        Batter-vs-starter matchup logit lift for BB.
    matchup_hr_lift : float
        Batter-vs-starter matchup logit lift for HR.
    bullpen_k_rate : float
        Opposing team bullpen aggregate K rate.
    bullpen_bb_rate : float
        Opposing team bullpen aggregate BB rate.
    bullpen_hr_rate : float
        Opposing team bullpen aggregate HR rate.
    batter_babip_adj : float
        Batter BABIP adjustment for BIP outcomes.
    umpire_k_lift, umpire_bb_lift, park_hr_lift, weather_k_lift : float
        Context adjustments.
    tto_k_lifts : np.ndarray or None
        Shape (3,) logit lifts for K by TTO block (1st, 2nd, 3rd pass).
        None uses league-average TTO constants.
    tto_bb_lifts : np.ndarray or None
        Shape (3,) logit lifts for BB by TTO block.
        None uses league-average TTO constants.
    tto_hr_lifts : np.ndarray or None
        Shape (3,) logit lifts for HR by TTO block.
        None uses league-average TTO constants.
    n_sims : int
        Number of simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    BatterSimulationResult
        Joint distributions over batter counting stats.
    """
    rng = np.random.default_rng(random_seed)
    pa_model = PAOutcomeModel()

    # Resample batter posteriors to n_sims
    def _resample(arr: np.ndarray) -> np.ndarray:
        if len(arr) == n_sims:
            return arr.copy()
        idx = rng.choice(len(arr), size=n_sims, replace=True)
        return arr[idx]

    batter_k = _resample(batter_k_rate_samples)
    batter_bb = _resample(batter_bb_rate_samples)
    batter_hr = _resample(batter_hr_rate_samples)

    # Resolve TTO lifts: use caller-provided or league-average defaults
    eff_tto_k = tto_k_lifts if tto_k_lifts is not None else _TTO_K_LIFTS
    eff_tto_bb = tto_bb_lifts if tto_bb_lifts is not None else _TTO_BB_LIFTS
    eff_tto_hr = tto_hr_lifts if tto_hr_lifts is not None else _TTO_HR_LIFTS

    # --- 1. Draw total PAs and starter BF ---
    total_pa = draw_total_pa(batting_order, rng, n_sims)

    starter_bf = draw_bf_samples(
        mu_bf=starter_bf_mu,
        sigma_bf=starter_bf_sigma,
        n_draws=n_sims,
        bf_min=3,
        bf_max=35,
        rng=rng,
    )

    # Split PAs into vs-starter and vs-reliever
    pa_vs_starter, pa_vs_reliever = split_pa_starter_reliever(
        total_pa, batting_order, starter_bf,
    )

    # --- 2. Compute per-PA rates ---
    # Pitcher quality lift: how far is the starter/reliever from league avg?
    starter_k_lift = _safe_logit(starter_k_rate) - _safe_logit(LEAGUE_K_RATE)
    starter_bb_lift = _safe_logit(starter_bb_rate) - _safe_logit(LEAGUE_BB_RATE)
    starter_hr_lift = _safe_logit(starter_hr_rate) - _safe_logit(LEAGUE_HR_RATE)

    bullpen_k_lift = _safe_logit(bullpen_k_rate) - _safe_logit(LEAGUE_K_RATE)
    bullpen_bb_lift = _safe_logit(bullpen_bb_rate) - _safe_logit(LEAGUE_BB_RATE)
    bullpen_hr_lift = _safe_logit(bullpen_hr_rate) - _safe_logit(LEAGUE_HR_RATE)

    # --- 3. Accumulators ---
    k_total = np.zeros(n_sims, dtype=np.int32)
    bb_total = np.zeros(n_sims, dtype=np.int32)
    h_total = np.zeros(n_sims, dtype=np.int32)
    hr_total = np.zeros(n_sims, dtype=np.int32)
    single_total = np.zeros(n_sims, dtype=np.int32)
    double_total = np.zeros(n_sims, dtype=np.int32)
    triple_total = np.zeros(n_sims, dtype=np.int32)
    hbp_total = np.zeros(n_sims, dtype=np.int32)
    r_total = np.zeros(n_sims, dtype=np.int32)
    rbi_total = np.zeros(n_sims, dtype=np.int32)

    # --- 4. Simulate each PA ---
    for pa_num in range(MAX_BATTER_PA):
        active = pa_num < total_pa
        n_active = active.sum()
        if n_active == 0:
            break

        # Determine if facing starter or reliever for this PA
        # PA pa_num happens at global BF = batting_order + 9 * pa_num
        global_pos = batting_order + 9 * pa_num
        vs_starter = active & (global_pos <= starter_bf)

        # Determine TTO block for this PA (scalar since global_pos is scalar)
        # global_pos 1-9: TTO1, 10-18: TTO2, 19+: TTO3
        tto_block = min((global_pos - 1) // _BF_PER_TTO, 2)
        tto_k = eff_tto_k[tto_block]
        tto_bb = eff_tto_bb[tto_block]
        tto_hr = eff_tto_hr[tto_block]

        # Build rates: use batter posteriors as base,
        # add pitcher quality lift + matchup lift + TTO lift
        k_logit_base = _safe_logit(batter_k[active])
        bb_logit_base = _safe_logit(batter_bb[active])
        hr_logit_base = _safe_logit(batter_hr[active])

        # Pitcher quality + matchup + TTO lifts (TTO only for starter PAs)
        vs_starter_active = vs_starter[active]

        k_pitcher_lift = np.where(
            vs_starter_active,
            starter_k_lift + matchup_k_lift + tto_k,
            bullpen_k_lift,
        )
        bb_pitcher_lift = np.where(
            vs_starter_active,
            starter_bb_lift + matchup_bb_lift + tto_bb,
            bullpen_bb_lift,
        )
        hr_pitcher_lift = np.where(
            vs_starter_active,
            starter_hr_lift + matchup_hr_lift + tto_hr,
            bullpen_hr_lift,
        )

        # Final adjusted rates
        k_rate_adj = expit(
            k_logit_base + k_pitcher_lift + umpire_k_lift + weather_k_lift
        )
        bb_rate_adj = expit(
            bb_logit_base + bb_pitcher_lift + umpire_bb_lift
        )
        hr_rate_adj = expit(
            hr_logit_base + hr_pitcher_lift + park_hr_lift
        )

        # Draw outcomes
        probs = pa_model.compute_pa_probs(
            pitcher_k_rate=k_rate_adj,
            pitcher_bb_rate=bb_rate_adj,
            pitcher_hr_rate=hr_rate_adj,
        )
        outcomes = pa_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active,
            babip_adj=batter_babip_adj,
        )

        # Accumulate
        k_total[active] += (outcomes == PA_STRIKEOUT).astype(np.int32)
        bb_total[active] += (outcomes == PA_WALK).astype(np.int32)
        hbp_total[active] += (outcomes == PA_HBP).astype(np.int32)
        hr_total[active] += (outcomes == PA_HOME_RUN).astype(np.int32)
        single_total[active] += (outcomes == PA_SINGLE).astype(np.int32)
        double_total[active] += (outcomes == PA_DOUBLE).astype(np.int32)
        triple_total[active] += (outcomes == PA_TRIPLE).astype(np.int32)

        is_hit = np.isin(outcomes, [PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN])
        h_total[active] += is_hit.astype(np.int32)

        # Simplified R/RBI: HR always scores batter (R+1, RBI+1)
        # Other hits sometimes score runners — use population averages
        is_hr = (outcomes == PA_HOME_RUN)
        r_total[active] += is_hr.astype(np.int32)
        rbi_total[active] += is_hr.astype(np.int32)

        # Non-HR hits: ~15% chance of scoring a run, ~30% chance of RBI
        non_hr_hit = is_hit & ~is_hr
        n_non_hr = non_hr_hit.sum()
        if n_non_hr > 0:
            r_total[active] += (non_hr_hit & (rng.random(n_active) < 0.15)).astype(np.int32)
            rbi_total[active] += (non_hr_hit & (rng.random(n_active) < 0.30)).astype(np.int32)

        # BB/HBP: ~10% chance of eventually scoring
        on_base_no_hit = np.isin(outcomes, [PA_WALK, PA_HBP])
        n_ob = on_base_no_hit.sum()
        if n_ob > 0:
            r_total[active] += (on_base_no_hit & (rng.random(n_active) < 0.10)).astype(np.int32)

    # Compute total bases
    tb_total = (
        single_total
        + 2 * double_total
        + 3 * triple_total
        + 4 * hr_total
    )

    return BatterSimulationResult(
        k_samples=k_total,
        bb_samples=bb_total,
        h_samples=h_total,
        hr_samples=hr_total,
        single_samples=single_total,
        double_samples=double_total,
        triple_samples=triple_total,
        tb_samples=tb_total,
        r_samples=r_total,
        rbi_samples=rbi_total,
        hbp_samples=hbp_total,
        pa_samples=total_pa,
        pa_vs_starter_samples=pa_vs_starter,
        pa_vs_reliever_samples=pa_vs_reliever,
        n_sims=n_sims,
    )
