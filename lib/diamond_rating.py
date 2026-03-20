"""Diamond Rating — TDD's 0-5 diamond scale for player quality."""
from __future__ import annotations


def score_to_diamonds(score: float) -> float:
    """Convert TDD composite score (0-1) to diamond rating (0-5).

    Args:
        score: A 0-1 normalized score (e.g. tdd_value_score, tdd_prospect_score,
               or a composite_score that has been passed through
               ``normalize_composite()`` first).
    """
    if score < 0.20:
        return 1.0 + (score / 0.20) * 0.0  # 1.0 floor
    elif score < 0.40:
        return 1.0 + ((score - 0.20) / 0.20) * 1.0  # 1.0-2.0
    elif score < 0.55:
        return 2.0 + ((score - 0.40) / 0.15) * 1.0  # 2.0-3.0
    elif score < 0.70:
        return 3.0 + ((score - 0.55) / 0.15) * 1.0  # 3.0-4.0
    else:
        return 4.0 + ((score - 0.70) / 0.30) * 1.0  # 4.0-5.0


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
    """Return tier label for a diamond rating."""
    if rating >= 4.5:
        return "Elite"
    elif rating >= 3.5:
        return "Above Average"
    elif rating >= 2.5:
        return "Average"
    elif rating >= 1.5:
        return "Below Average"
    return "Developing"
