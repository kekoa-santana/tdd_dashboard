"""Diamond Rating — TDD's 0-10 diamond scale for player quality.

The display diamond rating is always derived from ``tdd_value_score`` (0-1)
via ``score_to_diamonds()``.  This production-weighted composite is the
single source of truth for all leaderboards, team pages, and game cards.

A separate ``tools_rating`` column (0-10, league-relative) exists in the
rankings parquets for scouting-only evaluation.  It is shown on the player
profile page as "Tools Grade" and is NOT used for sorting or display
diamond ratings elsewhere.

Scale: 0-10 (continuous, 1 decimal)
  8.0+ = Elite
  6.5+ = Above Average
  5.0  = MLB Average
  3.5+ = Below Average
  <3.5 = Developing / Fringe
"""
from __future__ import annotations


def score_to_diamonds(score: float) -> float:
    """Convert TDD composite score (0-1) to diamond rating (0-10).

    Fallback for when pre-computed ``diamond_rating`` column is not available.
    Uses piecewise linear mapping calibrated to the scouting grade scale.

    Args:
        score: A 0-1 normalized score (e.g. tdd_value_score, tdd_prospect_score,
               or a composite_score that has been passed through
               ``normalize_composite()`` first).
    """
    if score < 0.20:
        return 2.0 + (score / 0.20) * 1.0        # 2.0-3.0
    elif score < 0.40:
        return 3.0 + ((score - 0.20) / 0.20) * 2.0  # 3.0-5.0
    elif score < 0.55:
        return 5.0 + ((score - 0.40) / 0.15) * 1.5  # 5.0-6.5
    elif score < 0.70:
        return 6.5 + ((score - 0.55) / 0.15) * 1.0  # 6.5-7.5
    else:
        return 7.5 + ((score - 0.70) / 0.30) * 2.5  # 7.5-10.0


def normalize_composite(value: float, low: float = -1.0, high: float = 1.2) -> float:
    """Rescale a signed composite_score (roughly -1 to +1.2) to the 0-1 range.

    The projection composite_score is a weighted delta average that can be
    negative. This maps it linearly into [0, 1] so it can feed into
    ``score_to_diamonds()``.

    Args:
        value: Raw signed composite_score.
        low: Value that maps to 0.0 (floor).
        high: Value that maps to 1.0 (ceiling).
    """
    clamped = max(low, min(high, value))
    return (clamped - low) / (high - low)


def diamond_tier(rating: float) -> str:
    """Return tier label for a diamond rating (0-10 scale)."""
    if rating >= 8.0:
        return "Elite"
    elif rating >= 6.5:
        return "Above Average"
    elif rating >= 5.0:
        return "Average"
    elif rating >= 3.5:
        return "Below Average"
    return "Developing"
