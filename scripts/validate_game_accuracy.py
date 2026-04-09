#!/usr/bin/env python
"""Validate game-level prediction accuracy: moneyline, spread, over/under.

Reads the cumulative ``game_prediction_log.parquet`` (built by
``update_in_season.archive_yesterdays_game_predictions``) and reports
hit-rates at multiple confidence thresholds plus aggregate CRPS.

Usage
-----
    python scripts/validate_game_accuracy.py
    python scripts/validate_game_accuracy.py --min-games 20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
LOG_PATH = DASHBOARD_DIR / "game_prediction_log.parquet"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _hit_rate_at_threshold(
    prob_col: pd.Series,
    correct_col: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    """Compute hit rate for picks where max(prob, 1-prob) >= threshold."""
    confidence = prob_col.clip(lower=1 - prob_col, upper=prob_col)
    # For probabilities < 0.5, confidence = 1 - prob
    confidence = prob_col.where(prob_col >= 0.5, 1 - prob_col)
    mask = confidence >= threshold
    n = mask.sum()
    if n == 0:
        return {"threshold": threshold, "n_picks": 0, "hit_rate": np.nan}
    hits = correct_col[mask].sum()
    return {"threshold": threshold, "n_picks": int(n), "hit_rate": float(hits / n)}


def build_moneyline_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Moneyline accuracy at various confidence thresholds."""
    if df.empty or "ml_correct" not in df.columns:
        return pd.DataFrame()

    rows = []
    for t in THRESHOLDS:
        rows.append(_hit_rate_at_threshold(df["p_away_win"], df["ml_correct"], t))
    return pd.DataFrame(rows)


def build_spread_accuracy(df: pd.DataFrame, spread_label: str = "m1_5") -> pd.DataFrame:
    """Spread (run-line) accuracy at various confidence thresholds."""
    cover_col = f"p_away_cover_{spread_label}"
    correct_col = f"spread_{spread_label}_correct"
    if df.empty or correct_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for t in THRESHOLDS:
        rows.append(_hit_rate_at_threshold(df[cover_col], df[correct_col], t))
    return pd.DataFrame(rows)


def build_ou_accuracy(df: pd.DataFrame, total_label: str = "8_5") -> pd.DataFrame:
    """Over/under accuracy at various confidence thresholds."""
    prob_col = f"p_over_{total_label}"
    correct_col = f"ou_{total_label}_correct"
    if df.empty or correct_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for t in THRESHOLDS:
        rows.append(_hit_rate_at_threshold(df[prob_col], df[correct_col], t))
    return pd.DataFrame(rows)


def print_accuracy_report(df: pd.DataFrame) -> None:
    """Print a formatted accuracy report to stdout."""
    n_games = len(df)
    dates = df["prediction_date"].nunique() if "prediction_date" in df.columns else "?"
    print(f"\n{'=' * 60}")
    print(f"Game Prediction Accuracy Report")
    print(f"  {n_games} games across {dates} prediction dates")
    print(f"{'=' * 60}")

    # CRPS summary
    if "crps_total" in df.columns:
        crps_t = df["crps_total"].dropna()
        crps_m = df["crps_margin"].dropna()
        if not crps_t.empty:
            print(f"\n--- CRPS (lower = better, 0 = perfect) ---")
            print(f"  Total runs:  mean={crps_t.mean():.3f}  median={crps_t.median():.3f}  (n={len(crps_t)})")
        if not crps_m.empty:
            print(f"  Run margin:  mean={crps_m.mean():.3f}  median={crps_m.median():.3f}  (n={len(crps_m)})")

    # Moneyline
    ml = build_moneyline_accuracy(df)
    if not ml.empty:
        print(f"\n--- Moneyline ---")
        for _, row in ml.iterrows():
            if row["n_picks"] > 0:
                print(f"  ≥{row['threshold']:.0%} conf:  {row['hit_rate']:.1%}  ({row['n_picks']:.0f} picks)")

    # Spread -1.5
    sp = build_spread_accuracy(df, "m1_5")
    if not sp.empty:
        print(f"\n--- Spread (away -1.5) ---")
        for _, row in sp.iterrows():
            if row["n_picks"] > 0:
                print(f"  ≥{row['threshold']:.0%} conf:  {row['hit_rate']:.1%}  ({row['n_picks']:.0f} picks)")

    # Spread +1.5
    sp2 = build_spread_accuracy(df, "p1_5")
    if not sp2.empty:
        print(f"\n--- Spread (away +1.5) ---")
        for _, row in sp2.iterrows():
            if row["n_picks"] > 0:
                print(f"  ≥{row['threshold']:.0%} conf:  {row['hit_rate']:.1%}  ({row['n_picks']:.0f} picks)")

    # Over/Under at each line
    for label in ["6_5", "7_5", "8_5", "9_5"]:
        ou = build_ou_accuracy(df, label)
        if not ou.empty:
            line_str = label.replace("_", ".")
            print(f"\n--- Over/Under {line_str} ---")
            for _, row in ou.iterrows():
                if row["n_picks"] > 0:
                    print(f"  ≥{row['threshold']:.0%} conf:  {row['hit_rate']:.1%}  ({row['n_picks']:.0f} picks)")

    # Overall moneyline accuracy (naive)
    if "ml_correct" in df.columns:
        naive = df["ml_correct"].mean()
        print(f"\n  Overall moneyline accuracy (no threshold): {naive:.1%}")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Game prediction accuracy report")
    parser.add_argument("--min-games", type=int, default=5,
                        help="Minimum games required to report.")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        logger.warning("No game prediction log found at %s. Run update pipeline first.", LOG_PATH)
        return

    df = pd.read_parquet(LOG_PATH)
    if len(df) < args.min_games:
        logger.info("Only %d games in log (need %d). Skipping report.", len(df), args.min_games)
        return

    print_accuracy_report(df)


if __name__ == "__main__":
    main()
