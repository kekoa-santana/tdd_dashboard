"""Methodology page — explains the Bayesian projection system."""
from __future__ import annotations

import streamlit as st

from config import CURRENT_SEASON, TRAINING_RANGE


def page_methodology() -> None:
    """Overview of the projection methodology and model architecture."""
    st.markdown(
        '<div class="tdd-section-hdr">Methodology</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
**The Data Diamond** projections are built on a Bayesian statistical framework
trained on {TRAINING_RANGE} MLB data.

### Projection System

- **Prior distributions** — Preseason priors derived from career performance,
  age curves, park factors, and batted ball profiles
- **Conjugate updating** — Beta-Binomial in-season updates for rate stats
  (K%, BB%, HR/BF) as new data arrives daily
- **Posterior samples** — 4,000 Monte Carlo draws per player per stat,
  giving full uncertainty ranges (not just point estimates)

### Game Simulator

- **PA-by-PA engine** — Simulates each plate appearance using pitcher and
  batter posterior distributions
- **Matchup adjustments** — Pitch-type-level log-odds scoring blends
  pitcher arsenal with hitter vulnerability profiles
- **Context factors** — Umpire tendencies, weather, park factors, rest days,
  times-through-order progression, pitch count fatigue

### Data Sources

- **Statcast** — Pitch-level data (location, velocity, movement, outcomes)
- **MLB Stats API** — Schedules, lineups, rosters, game states
- **Historical** — {TRAINING_RANGE} seasons of complete batting and pitching data

### Model Validation

See the **Model Performance** page for predicted-vs-actual tracking,
calibration curves, and biggest hits/misses.

See the **Data Health** page for data freshness monitoring and artifact
validation.

---

*Built by Kekoa Santana. {CURRENT_SEASON} season.*
    """)
