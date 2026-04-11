"""
Vectorized Monte Carlo game simulator.

Simulates plate appearances sequentially through the batting order,
advancing all N simulations in lockstep. Each PA draws a pitch count
and outcome, updates game state, and checks for pitcher exit.

Produces joint distributions over all pitcher counting stats from
a single simulation run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lib.game_sim.exit_model import ExitModel
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
    compute_fatigue_adjustments,
)
from lib.game_sim.pitch_count_model import PitchCountModel
from lib.game_sim.tto_model import BF_PER_TTO

# League-average bullpen fallback rates (used when a team has no entry in
# team_bullpen_rates.parquet). Mirrors src.utils.constants in player_profiles.
_LEAGUE_BULLPEN_K_RATE = 0.253
_LEAGUE_BULLPEN_BB_RATE = 0.084
_LEAGUE_BULLPEN_HR_RATE = 0.024

# Safety cap on bullpen tail PAs.
_MAX_BULLPEN_PA = 80

# Runner advancement probabilities (league-average, see CLAUDE discussion
# 2026-04-10).
_SAC_FLY_PROB_0OUT = 0.50
_SAC_FLY_PROB_1OUT = 0.30
_R2_SCORES_ON_SINGLE = 0.60
_R1_SCORES_ON_DOUBLE = 0.40


def _advance_runners(
    outcomes: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    inning_outs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply PA outcomes to base state and return runs + new base occupancy.

    See ``player_profiles/src/models/game_sim/simulator.py`` for the full
    docstring. Advancement rules (approved 2026-04-10):
    - K / non-K out: no movement, except R3 scores on non-K out with prob
      0.50 (0 outs) / 0.30 (1 out) / 0.0 (2 outs) for sac fly modeling.
    - Walk/HBP: forced advancement only.
    - Single: batter→1B, R3 scores, R2 scores 60% else→3B, R1→2B.
    - Double: batter→2B, R3 scores, R2 scores, R1 scores 40% else→3B.
    - Triple: all on base score, batter→3B.
    - HR: clears bases, all score.
    """
    n = outcomes.shape[0]
    runs = np.zeros(n, dtype=np.int32)
    new_r1 = r1.copy()
    new_r2 = r2.copy()
    new_r3 = r3.copy()

    # Non-K out: sac fly with runner on 3B
    out_mask = outcomes == PA_OUT
    sf_prob = np.where(
        out_mask & (r3 == 1) & (inning_outs == 0),
        _SAC_FLY_PROB_0OUT,
        np.where(
            out_mask & (r3 == 1) & (inning_outs == 1),
            _SAC_FLY_PROB_1OUT,
            0.0,
        ),
    )
    sf_draw = rng.random(n) < sf_prob
    runs += sf_draw.astype(np.int32)
    new_r3 = np.where(sf_draw, 0, new_r3)

    # Walk / HBP: forced advancement
    walk_mask = (outcomes == PA_WALK) | (outcomes == PA_HBP)
    bases_loaded_walk = walk_mask & (r1 == 1) & (r2 == 1) & (r3 == 1)
    runs += bases_loaded_walk.astype(np.int32)
    walk_push_r3 = walk_mask & (r1 == 1) & (r2 == 1)
    walk_push_r2 = walk_mask & (r1 == 1)
    new_r3 = np.where(walk_push_r3, 1, new_r3)
    new_r2 = np.where(walk_push_r2, 1, new_r2)
    new_r1 = np.where(walk_mask, 1, new_r1)

    # Single
    single_mask = outcomes == PA_SINGLE
    runs += (single_mask & (r3 == 1)).astype(np.int32)

    r2_on_single = single_mask & (r2 == 1)
    r2_score_draw = rng.random(n) < _R2_SCORES_ON_SINGLE
    runs += (r2_on_single & r2_score_draw).astype(np.int32)
    r2_holds_at_3_single = r2_on_single & ~r2_score_draw

    r1_on_single = single_mask & (r1 == 1)

    new_r3 = np.where(single_mask, r2_holds_at_3_single.astype(np.int32), new_r3)
    new_r2 = np.where(single_mask, r1_on_single.astype(np.int32), new_r2)
    new_r1 = np.where(single_mask, 1, new_r1)

    # Double
    double_mask = outcomes == PA_DOUBLE
    runs += (double_mask & (r3 == 1)).astype(np.int32)
    runs += (double_mask & (r2 == 1)).astype(np.int32)

    r1_on_double = double_mask & (r1 == 1)
    r1_score_draw = rng.random(n) < _R1_SCORES_ON_DOUBLE
    runs += (r1_on_double & r1_score_draw).astype(np.int32)
    r1_holds_at_3_double = r1_on_double & ~r1_score_draw

    new_r3 = np.where(double_mask, r1_holds_at_3_double.astype(np.int32), new_r3)
    new_r2 = np.where(double_mask, 1, new_r2)
    new_r1 = np.where(double_mask, 0, new_r1)

    # Triple
    triple_mask = outcomes == PA_TRIPLE
    runs += (triple_mask & (r3 == 1)).astype(np.int32)
    runs += (triple_mask & (r2 == 1)).astype(np.int32)
    runs += (triple_mask & (r1 == 1)).astype(np.int32)
    new_r3 = np.where(triple_mask, 1, new_r3)
    new_r2 = np.where(triple_mask, 0, new_r2)
    new_r1 = np.where(triple_mask, 0, new_r1)

    # Home run
    hr_mask = outcomes == PA_HOME_RUN
    hr_runs = (
        hr_mask.astype(np.int32)
        + (hr_mask & (r1 == 1)).astype(np.int32)
        + (hr_mask & (r2 == 1)).astype(np.int32)
        + (hr_mask & (r3 == 1)).astype(np.int32)
    )
    runs += hr_runs
    new_r3 = np.where(hr_mask, 0, new_r3)
    new_r2 = np.where(hr_mask, 0, new_r2)
    new_r1 = np.where(hr_mask, 0, new_r1)

    return runs, new_r1, new_r2, new_r3

logger = logging.getLogger(__name__)

_CLIP_LO = 1e-6
_CLIP_HI = 1 - 1e-6

# Maximum PA per game (safety valve)
MAX_PA_PER_GAME = 45


@dataclass
class SimulationResult:
    """Results from a game simulation run.

    All arrays have shape (n_sims,) — one value per simulated game.

    Starter counting stats (k/bb/h/hr/hbp/bf/pitch_count/outs) back pitcher
    prop probabilities. ``runs_samples`` is the full-game run total
    (starter + bullpen); ``starter_runs_samples`` / ``bullpen_runs_samples``
    are diagnostic components that sum to ``runs_samples``.
    """

    k_samples: np.ndarray
    bb_samples: np.ndarray
    h_samples: np.ndarray
    hr_samples: np.ndarray
    hbp_samples: np.ndarray
    bf_samples: np.ndarray
    pitch_count_samples: np.ndarray
    outs_samples: np.ndarray
    runs_samples: np.ndarray
    starter_runs_samples: np.ndarray | None = None
    bullpen_runs_samples: np.ndarray | None = None
    n_sims: int = 0

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all stats.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict with mean, std, median, q10, q90 per stat.
        """
        stats = {}
        for name in [
            "k", "bb", "h", "hr", "hbp", "bf", "pitch_count", "outs", "runs",
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
        """Compute P(over X.5) for prop lines.

        Parameters
        ----------
        stat : str
            Stat name (e.g., 'k', 'bb', 'h', 'hr').
        lines : list[float], optional
            Lines to evaluate. Default: [0.5, 1.5, ..., 12.5].

        Returns
        -------
        pd.DataFrame
            Columns: line, p_over, p_under, expected, std.
        """
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

    def ip_samples(self) -> np.ndarray:
        """Compute innings pitched from outs."""
        full_innings = self.outs_samples // 3
        partial = self.outs_samples % 3
        return full_innings + partial / 10.0  # Baseball IP notation


def simulate_bullpen_tail(
    *,
    outs: np.ndarray,
    inning: np.ndarray,
    inning_outs: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    bullpen_k_rate: float,
    bullpen_bb_rate: float,
    bullpen_hr_rate: float,
    umpire_k_lift: float,
    umpire_bb_lift: float,
    park_hr_lift: float,
    weather_k_lift: float,
    babip_adj: float,
    rng: np.random.Generator,
    n_sims: int,
) -> np.ndarray:
    """Continue a game simulation after the starter exits using team bullpen rates.

    Runs PA-by-PA with team-aggregate bullpen rates until every sim reaches
    27 outs (or ``_MAX_BULLPEN_PA`` is hit as a safety valve). Mutates the
    passed-in state vectors (``outs``, ``inning``, ``inning_outs``,
    ``r1``, ``r2``, ``r3``) in place.

    Matchup, TTO, fatigue, pitch counts, and the exit model are all disabled
    in the tail. Park/umpire/weather lifts carry through since the
    environment doesn't change mid-game.

    Returns
    -------
    np.ndarray
        ``bullpen_runs`` of shape ``(n_sims,)``.
    """
    pa_outcome_model = PAOutcomeModel()

    bullpen_runs = np.zeros(n_sims, dtype=np.int32)

    # Active = sims that still need more outs (skip complete games)
    active = outs < 27

    for _ in range(_MAX_BULLPEN_PA):
        n_active = int(active.sum())
        if n_active == 0:
            break

        zeros_active = np.zeros(n_active, dtype=np.float64)
        k_rates = np.full(n_active, bullpen_k_rate, dtype=np.float64)
        bb_rates = np.full(n_active, bullpen_bb_rate, dtype=np.float64)
        hr_rates = np.full(n_active, bullpen_hr_rate, dtype=np.float64)

        probs = pa_outcome_model.compute_pa_probs(
            pitcher_k_rate=k_rates,
            pitcher_bb_rate=bb_rates,
            pitcher_hr_rate=hr_rates,
            matchup_k_lift=zeros_active,
            matchup_bb_lift=zeros_active,
            matchup_hr_lift=zeros_active,
            tto_k_lift=zeros_active,
            tto_bb_lift=zeros_active,
            tto_hr_lift=zeros_active,
            fatigue_k_lift=zeros_active,
            fatigue_bb_lift=zeros_active,
            fatigue_hr_lift=zeros_active,
            umpire_k_lift=umpire_k_lift,
            umpire_bb_lift=umpire_bb_lift,
            park_hr_lift=park_hr_lift,
            weather_k_lift=weather_k_lift,
        )

        outcomes = pa_outcome_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active, babip_adj=babip_adj,
        )

        # Runner advancement (pre-out-increment for sac fly eligibility)
        runs_scored, new_r1, new_r2, new_r3 = _advance_runners(
            outcomes=outcomes,
            r1=r1[active],
            r2=r2[active],
            r3=r3[active],
            inning_outs=inning_outs[active],
            rng=rng,
        )

        bullpen_runs[active] += runs_scored
        r1[active] = new_r1
        r2[active] = new_r2
        r3[active] = new_r3

        # Outs (after sac fly check)
        is_out = np.isin(outcomes, [PA_STRIKEOUT, PA_OUT])
        outs[active] += is_out.astype(np.int32)
        inning_outs[active] += is_out.astype(np.int32)

        # Inning rollover
        inning_over = inning_outs[active] >= 3
        inning[active] = np.where(inning_over, inning[active] + 1, inning[active])
        inning_outs[active] = np.where(inning_over, 0, inning_outs[active])
        r1[active] = np.where(inning_over, 0, r1[active])
        r2[active] = np.where(inning_over, 0, r2[active])
        r3[active] = np.where(inning_over, 0, r3[active])

        active = outs < 27

    return bullpen_runs


def simulate_game(
    pitcher_k_rate_samples: np.ndarray,
    pitcher_bb_rate_samples: np.ndarray,
    pitcher_hr_rate_samples: np.ndarray,
    lineup_matchup_lifts: dict[str, np.ndarray],
    tto_lifts: dict[str, np.ndarray],
    pitcher_ppa_adj: float,
    batter_ppa_adjs: np.ndarray,
    exit_model: ExitModel,
    pitcher_avg_pitches: float = 88.0,
    babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    weather_k_lift: float = 0.0,
    bullpen_k_rate: float = _LEAGUE_BULLPEN_K_RATE,
    bullpen_bb_rate: float = _LEAGUE_BULLPEN_BB_RATE,
    bullpen_hr_rate: float = _LEAGUE_BULLPEN_HR_RATE,
    n_sims: int = 10_000,
    random_seed: int = 42,
) -> SimulationResult:
    """Run vectorized Monte Carlo game simulation.

    Advances all simulations one PA at a time through the lineup order.
    Simulations where the pitcher has exited are masked out.

    Parameters
    ----------
    pitcher_k_rate_samples : np.ndarray
        K% posterior samples from Layer 1.
    pitcher_bb_rate_samples : np.ndarray
        BB% posterior samples from Layer 1.
    pitcher_hr_rate_samples : np.ndarray
        HR/BF posterior samples from Layer 1.
    lineup_matchup_lifts : dict[str, np.ndarray]
        Per-stat matchup logit lifts. Keys: 'k', 'bb', 'hr'.
        Each value shape (9,) for 9 batting order slots.
    tto_lifts : dict[str, np.ndarray]
        TTO logit lifts. Keys: 'k', 'bb', 'hr'.
        Each value shape (3,) for TTO 1, 2, 3.
    pitcher_ppa_adj : float
        Pitcher pitches-per-PA adjustment.
    batter_ppa_adjs : np.ndarray
        Shape (9,) batter P/PA adjustments.
    exit_model : ExitModel
        Trained pitcher exit model.
    pitcher_avg_pitches : float
        Pitcher's historical average exit pitch count.
    babip_adj : float
        Pitcher BABIP adjustment for BIP outcomes.
    umpire_k_lift : float
        Umpire K-rate logit lift.
    umpire_bb_lift : float
        Umpire BB-rate logit lift.
    park_hr_lift : float
        Park HR logit lift.
    weather_k_lift : float
        Weather K logit lift.
    n_sims : int
        Number of Monte Carlo simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    SimulationResult
        Joint distributions over all pitcher counting stats.
    """
    rng = np.random.default_rng(random_seed)

    # Initialize component models
    pitch_count_model = PitchCountModel()
    pa_outcome_model = PAOutcomeModel()

    # Resample posterior draws to n_sims
    def _resample(arr: np.ndarray) -> np.ndarray:
        if len(arr) == n_sims:
            return arr.copy()
        idx = rng.choice(len(arr), size=n_sims, replace=True)
        return arr[idx]

    k_rates = _resample(pitcher_k_rate_samples)
    bb_rates = _resample(pitcher_bb_rate_samples)
    hr_rates = _resample(pitcher_hr_rate_samples)

    # Default matchup lifts to zeros if missing stat keys
    for stat in ("k", "bb", "hr"):
        if stat not in lineup_matchup_lifts:
            lineup_matchup_lifts[stat] = np.zeros(9)
        if stat not in tto_lifts:
            tto_lifts[stat] = np.zeros(3)

    # --- Game state arrays (all shape n_sims) ---
    pitches = np.zeros(n_sims, dtype=np.int32)
    outs = np.zeros(n_sims, dtype=np.int32)
    inning = np.ones(n_sims, dtype=np.int32)
    inning_outs = np.zeros(n_sims, dtype=np.int32)  # 0, 1, 2 within inning
    lineup_pos = np.zeros(n_sims, dtype=np.int32)    # 0-8
    bf_count = np.zeros(n_sims, dtype=np.int32)
    # Base occupancy flags (exit model still takes a total count, computed
    # on the fly as r1+r2+r3).
    r1 = np.zeros(n_sims, dtype=np.int32)
    r2 = np.zeros(n_sims, dtype=np.int32)
    r3 = np.zeros(n_sims, dtype=np.int32)
    runs = np.zeros(n_sims, dtype=np.int32)
    score_diff = np.zeros(n_sims, dtype=np.int32)     # pitcher team perspective

    # Accumulators
    k_total = np.zeros(n_sims, dtype=np.int32)
    bb_total = np.zeros(n_sims, dtype=np.int32)
    h_total = np.zeros(n_sims, dtype=np.int32)
    hr_total = np.zeros(n_sims, dtype=np.int32)
    hbp_total = np.zeros(n_sims, dtype=np.int32)

    # Recent trouble tracker (last 2 PA: BB, H, HBP)
    recent_trouble = np.zeros(n_sims, dtype=np.int32)
    prev_trouble = np.zeros(n_sims, dtype=np.int32)

    # Active mask — simulations where pitcher is still in the game
    active = np.ones(n_sims, dtype=bool)

    # --- Main simulation loop ---
    for pa_num in range(MAX_PA_PER_GAME):
        n_active = active.sum()
        if n_active == 0:
            break

        # Current batter slot (0-8)
        slot = lineup_pos[active] % 9
        # Current TTO
        tto = np.minimum(bf_count[active] // BF_PER_TTO, 2)

        # --- 1. Draw pitch count for this PA ---
        # Get per-batter P/PA adjustments for active sims
        batter_adj_active = batter_ppa_adjs[slot]
        pa_pitches = pitch_count_model.draw_pitches(
            pitcher_adj=pitcher_ppa_adj,
            batter_adj=batter_adj_active,
            rng=rng,
            n_draws=n_active,
        )
        pitches[active] += pa_pitches

        # --- 2. Compute PA outcome probabilities ---
        fatigue = compute_fatigue_adjustments(pitches[active])

        # Gather per-batter matchup lifts
        k_matchup = np.array([lineup_matchup_lifts["k"][s] for s in slot])
        bb_matchup = np.array([lineup_matchup_lifts["bb"][s] for s in slot])
        hr_matchup = np.array([lineup_matchup_lifts["hr"][s] for s in slot])

        # Gather TTO lifts
        k_tto = np.array([tto_lifts["k"][t] for t in tto])
        bb_tto = np.array([tto_lifts["bb"][t] for t in tto])
        hr_tto = np.array([tto_lifts["hr"][t] for t in tto])

        probs = pa_outcome_model.compute_pa_probs(
            pitcher_k_rate=k_rates[active],
            pitcher_bb_rate=bb_rates[active],
            pitcher_hr_rate=hr_rates[active],
            matchup_k_lift=k_matchup,
            matchup_bb_lift=bb_matchup,
            matchup_hr_lift=hr_matchup,
            tto_k_lift=k_tto,
            tto_bb_lift=bb_tto,
            tto_hr_lift=hr_tto,
            fatigue_k_lift=fatigue["k"],
            fatigue_bb_lift=fatigue["bb"],
            fatigue_hr_lift=fatigue["hr"],
            umpire_k_lift=umpire_k_lift,
            umpire_bb_lift=umpire_bb_lift,
            park_hr_lift=park_hr_lift,
            weather_k_lift=weather_k_lift,
        )

        # --- 3. Draw PA outcomes ---
        outcomes = pa_outcome_model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n_active, babip_adj=babip_adj,
        )

        # --- 4. Update game state ---
        # Update accumulators
        k_total[active] += (outcomes == PA_STRIKEOUT).astype(np.int32)
        bb_total[active] += (outcomes == PA_WALK).astype(np.int32)
        hbp_total[active] += (outcomes == PA_HBP).astype(np.int32)
        hr_total[active] += (outcomes == PA_HOME_RUN).astype(np.int32)

        is_hit = np.isin(outcomes, [PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN])
        h_total[active] += is_hit.astype(np.int32)

        # --- Runner advancement (pre-out-increment so sac flies can see
        #     the current inning_outs as 0 or 1).
        runs_scored, new_r1, new_r2, new_r3 = _advance_runners(
            outcomes=outcomes,
            r1=r1[active],
            r2=r2[active],
            r3=r3[active],
            inning_outs=inning_outs[active],
            rng=rng,
        )

        runs[active] += runs_scored
        score_diff[active] -= runs_scored
        r1[active] = new_r1
        r2[active] = new_r2
        r3[active] = new_r3

        # Update outs (after sac fly check)
        is_out = np.isin(outcomes, [PA_STRIKEOUT, PA_OUT])
        outs[active] += is_out.astype(np.int32)
        inning_outs[active] += is_out.astype(np.int32)

        # Check for inning change (3 outs in inning)
        inning_over = inning_outs[active] >= 3
        inning[active] = np.where(inning_over, inning[active] + 1, inning[active])
        inning_outs[active] = np.where(inning_over, 0, inning_outs[active])
        r1[active] = np.where(inning_over, 0, r1[active])
        r2[active] = np.where(inning_over, 0, r2[active])
        r3[active] = np.where(inning_over, 0, r3[active])

        # Update recent trouble
        is_trouble = np.isin(
            outcomes,
            [PA_WALK, PA_HBP, PA_SINGLE, PA_DOUBLE, PA_TRIPLE, PA_HOME_RUN],
        ).astype(np.int32)
        new_recent = prev_trouble[active] + is_trouble
        prev_trouble[active] = is_trouble
        recent_trouble[active] = new_recent

        # Advance lineup position and BF count
        bf_count[active] += 1
        lineup_pos[active] += 1

        # --- 5. Exit check ---
        # Force exit on complete game (27 outs)
        force_exit = outs[active] >= 27
        # Force exit on pitch count hard cap
        force_exit |= pitches[active] >= 130

        # Model-based exit probability
        current_tto = np.minimum(bf_count[active] // BF_PER_TTO + 1, 3)
        exit_prob = exit_model.predict_exit_prob(
            cumulative_pitches=pitches[active],
            inning=inning[active],
            inning_outs=inning_outs[active],
            score_diff=score_diff[active],
            runners=r1[active] + r2[active] + r3[active],
            tto=current_tto,
            recent_trouble=recent_trouble[active],
            pitcher_avg_pitches=pitcher_avg_pitches,
        )

        # Draw exit decisions
        exit_draw = rng.random(n_active) < exit_prob
        exits = force_exit | exit_draw

        # Update active mask
        active_indices = np.where(active)[0]
        active[active_indices[exits]] = False

    # --- Bullpen tail: finish the game to 27 outs using team bullpen rates ---
    # Snapshot starter totals before the tail mutates shared state arrays.
    starter_runs = runs.copy()
    starter_outs = outs.copy()

    bullpen_runs = simulate_bullpen_tail(
        outs=outs,
        inning=inning,
        inning_outs=inning_outs,
        r1=r1,
        r2=r2,
        r3=r3,
        bullpen_k_rate=bullpen_k_rate,
        bullpen_bb_rate=bullpen_bb_rate,
        bullpen_hr_rate=bullpen_hr_rate,
        umpire_k_lift=umpire_k_lift,
        umpire_bb_lift=umpire_bb_lift,
        park_hr_lift=park_hr_lift,
        weather_k_lift=weather_k_lift,
        babip_adj=babip_adj,
        rng=rng,
        n_sims=n_sims,
    )

    total_runs = starter_runs + bullpen_runs

    return SimulationResult(
        k_samples=k_total,
        bb_samples=bb_total,
        h_samples=h_total,
        hr_samples=hr_total,
        hbp_samples=hbp_total,
        bf_samples=bf_count,
        pitch_count_samples=pitches,
        outs_samples=starter_outs,
        runs_samples=total_runs,
        starter_runs_samples=starter_runs,
        bullpen_runs_samples=bullpen_runs,
        n_sims=n_sims,
    )


def predict_game(
    pitcher_id: int,
    season: int,
    lineup_batter_ids: list[int],
    pitcher_k_rate_samples: np.ndarray,
    pitcher_bb_rate_samples: np.ndarray,
    pitcher_hr_rate_samples: np.ndarray,
    lineup_matchup_lifts: dict[str, np.ndarray],
    tto_lifts: dict[str, np.ndarray],
    pitcher_features: pd.DataFrame,
    batter_features: pd.DataFrame,
    exit_model: ExitModel,
    pitcher_avg_pitches: float = 88.0,
    babip_adj: float = 0.0,
    umpire_k_lift: float = 0.0,
    umpire_bb_lift: float = 0.0,
    park_hr_lift: float = 0.0,
    weather_k_lift: float = 0.0,
    bullpen_k_rate: float = _LEAGUE_BULLPEN_K_RATE,
    bullpen_bb_rate: float = _LEAGUE_BULLPEN_BB_RATE,
    bullpen_hr_rate: float = _LEAGUE_BULLPEN_HR_RATE,
    n_sims: int = 10_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """High-level game prediction interface.

    Computes pitch count features, runs simulation, and returns
    comprehensive results including prop line probabilities.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season for feature lookup.
    lineup_batter_ids : list[int]
        9 batter IDs in batting order.
    pitcher_k_rate_samples : np.ndarray
        K% posterior samples.
    pitcher_bb_rate_samples : np.ndarray
        BB% posterior samples.
    pitcher_hr_rate_samples : np.ndarray
        HR/BF posterior samples.
    lineup_matchup_lifts : dict[str, np.ndarray]
        Per-stat matchup lifts, shape (9,) each.
    tto_lifts : dict[str, np.ndarray]
        TTO lifts, shape (3,) each.
    pitcher_features : pd.DataFrame
        Pitcher pitch count features.
    batter_features : pd.DataFrame
        Batter pitch count features.
    exit_model : ExitModel
        Trained exit model.
    pitcher_avg_pitches : float
        Average exit pitch count.
    babip_adj : float
        Pitcher BABIP adjustment.
    umpire_k_lift, umpire_bb_lift, park_hr_lift, weather_k_lift : float
        Context adjustments.
    n_sims : int
        Number of simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict[str, Any]
        Keys: 'result' (SimulationResult), 'summary', 'k_over_probs',
        'bb_over_probs', 'h_over_probs', 'hr_over_probs'.
    """
    from lib.game_sim.pitch_count_model import build_pitch_count_features

    # Build pitch count adjustments
    pitcher_ppa_adj, batter_ppa_adjs = build_pitch_count_features(
        pitcher_features=pitcher_features,
        batter_features=batter_features,
        pitcher_id=pitcher_id,
        batter_ids=lineup_batter_ids,
        season=season,
    )

    # Run simulation
    result = simulate_game(
        pitcher_k_rate_samples=pitcher_k_rate_samples,
        pitcher_bb_rate_samples=pitcher_bb_rate_samples,
        pitcher_hr_rate_samples=pitcher_hr_rate_samples,
        lineup_matchup_lifts=lineup_matchup_lifts,
        tto_lifts=tto_lifts,
        pitcher_ppa_adj=pitcher_ppa_adj,
        batter_ppa_adjs=batter_ppa_adjs,
        exit_model=exit_model,
        pitcher_avg_pitches=pitcher_avg_pitches,
        babip_adj=babip_adj,
        umpire_k_lift=umpire_k_lift,
        umpire_bb_lift=umpire_bb_lift,
        park_hr_lift=park_hr_lift,
        weather_k_lift=weather_k_lift,
        bullpen_k_rate=bullpen_k_rate,
        bullpen_bb_rate=bullpen_bb_rate,
        bullpen_hr_rate=bullpen_hr_rate,
        n_sims=n_sims,
        random_seed=random_seed,
    )

    return {
        "result": result,
        "summary": result.summary(),
        "k_over_probs": result.over_probs("k"),
        "bb_over_probs": result.over_probs("bb"),
        "h_over_probs": result.over_probs("h"),
        "hr_over_probs": result.over_probs("hr"),
    }
