#!/usr/bin/env python
"""Fit a calibration curve for breakout probabilities against real outcomes.

The breakout model ranks players well but its raw probabilities run about
twice as high as observed breakout rates. This script scores the frozen
preseason candidate lists against what actually happened and fits a Platt
scaling (two-parameter logistic on the log-odds), which corrects the level
without disturbing the ranking.

Writes ``data/dashboard/breakout_calibration.json`` with the fitted
parameters, the outcome definition, and the reliability table, so the
dashboard can show calibrated numbers and say how they were derived.

Usage:
    python scripts/calibrate_breakouts.py                # fit and write
    python scripts/calibrate_breakouts.py --dry-run      # report only
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
SNAPSHOTS = DASHBOARD_DIR / "snapshots"
OUTPUT = DASHBOARD_DIR / "breakout_calibration.json"

# A breakout has to be both real and actually witnessed: a player who never
# got the playing time did not break out, so the minimum is part of the label.
HITTER_MIN_PA = 200
HITTER_WRC_GAIN = 15
HITTER_WRC_FLOOR = 110

PITCHER_MIN_BF = 300
PITCHER_ERA_GAIN = 0.75
PITCHER_ERA_CEILING = 4.00

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(prob: np.ndarray, outcome: np.ndarray, iterations: int = 500) -> tuple[float, float]:
    """Fit outcome ~ sigmoid(a * logit(prob) + b) by gradient descent.

    Two parameters on a few hundred rows: enough to move the level and the
    slope, not enough to chase noise the way isotonic regression would.
    """
    x = _logit(prob)
    y = outcome.astype(float)
    a, b = 1.0, 0.0
    learning_rate = 0.05
    for _ in range(iterations):
        pred = _sigmoid(a * x + b)
        error = pred - y
        a -= learning_rate * float(np.mean(error * x))
        b -= learning_rate * float(np.mean(error))
    return a, b


def reliability(prob: np.ndarray, outcome: np.ndarray, edges=(0, .2, .4, .6, .8, 1.0)) -> list[dict]:
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            continue
        table.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "n": int(mask.sum()),
            "predicted": round(float(prob[mask].mean()), 4),
            "observed": round(float(outcome[mask].mean()), 4),
        })
    return table


def score_hitters() -> pd.DataFrame | None:
    candidates = SNAPSHOTS / f"hitter_breakout_candidates_{SEASON}_preseason.parquet"
    if not candidates.exists():
        return None
    brk = pd.read_parquet(candidates)
    advanced = pd.read_parquet(DASHBOARD_DIR / "hitter_advanced.parquet")
    actual = advanced[["batter_id", "pa", "wrc_plus"]].rename(
        columns={"pa": "pa_actual", "wrc_plus": "wrc_actual"}
    )
    frame = brk.merge(actual, on="batter_id", how="left")
    frame["pa_actual"] = frame["pa_actual"].fillna(0)
    frame["gain"] = frame["wrc_actual"] - frame["wrc_plus"]
    frame["outcome"] = (
        (frame["pa_actual"] >= HITTER_MIN_PA)
        & (frame["gain"] >= HITTER_WRC_GAIN)
        & (frame["wrc_actual"] >= HITTER_WRC_FLOOR)
    )
    return frame


def score_pitchers() -> pd.DataFrame | None:
    candidates = SNAPSHOTS / f"pitcher_breakout_candidates_{SEASON}_preseason.parquet"
    if not candidates.exists():
        return None
    brk = pd.read_parquet(candidates)
    traditional = pd.read_parquet(DASHBOARD_DIR / "pitcher_traditional.parquet")
    actual = traditional[["pitcher_id", "bf", "era"]].rename(
        columns={"bf": "bf_actual", "era": "era_actual"}
    )
    frame = brk.merge(actual, on="pitcher_id", how="left")
    frame["bf_actual"] = frame["bf_actual"].fillna(0)
    frame["gain"] = frame["era"] - frame["era_actual"]          # ERA down is good
    frame["outcome"] = (
        (frame["bf_actual"] >= PITCHER_MIN_BF)
        & (frame["gain"] >= PITCHER_ERA_GAIN)
        & (frame["era_actual"] <= PITCHER_ERA_CEILING)
    )
    return frame


def summarize(label: str, frame: pd.DataFrame) -> dict:
    prob = frame["breakout_prob"].to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy(dtype=bool)
    a, b = fit_platt(prob, outcome)
    calibrated = _sigmoid(a * _logit(prob) + b)

    print(f"\n=== {label} ===")
    print(f"  candidates scored: {len(frame)}   observed breakouts: {outcome.sum()} "
          f"({outcome.mean():.0%})")
    print(f"  raw mean probability: {prob.mean():.0%}   calibrated: {calibrated.mean():.0%}")
    print(f"  Platt a={a:.3f} b={b:.3f}")
    print(f"  {'bucket':10} {'n':>4} {'predicted':>10} {'observed':>9} {'calibrated':>11}")
    raw_table = reliability(prob, outcome)
    cal_table = reliability(calibrated, outcome)
    for row in raw_table:
        mask = (prob >= float(row["bucket"].split("-")[0].rstrip("%")) / 100) & (
            prob <= float(row["bucket"].split("-")[1].rstrip("%")) / 100
        )
        print(f"  {row['bucket']:10} {row['n']:4d} {row['predicted']:9.0%} "
              f"{row['observed']:8.0%} {calibrated[mask].mean():10.0%}")

    # Ranking quality is what survives calibration; report it explicitly.
    order = np.argsort(-prob)
    top_decile = max(1, len(frame) // 10)
    lift = outcome[order][:top_decile].mean() / outcome.mean() if outcome.mean() else float("nan")
    print(f"  top-decile lift over base rate: {lift:.1f}x")

    return {
        "n": int(len(frame)),
        "observed_rate": round(float(outcome.mean()), 4),
        "raw_mean": round(float(prob.mean()), 4),
        "calibrated_mean": round(float(calibrated.mean()), 4),
        "platt_a": round(a, 4),
        "platt_b": round(b, 4),
        "top_decile_lift": round(float(lift), 2),
        "reliability_raw": raw_table,
        "reliability_calibrated": cal_table,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    payload: dict = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fitted_on_seasons": [SEASON],
        "method": "platt",
        "caveat": (
            f"Fitted on {SEASON} outcomes only, so these are in-sample. Refit as "
            "more seasons finish."
        ),
        "definitions": {
            "hitter": (
                f"{HITTER_MIN_PA}+ PA, wRC+ up {HITTER_WRC_GAIN}+ over the prior "
                f"season, finishing at {HITTER_WRC_FLOOR}+"
            ),
            "pitcher": (
                f"{PITCHER_MIN_BF}+ batters faced, ERA down {PITCHER_ERA_GAIN}+ "
                f"from the prior season, finishing at {PITCHER_ERA_CEILING} or below"
            ),
        },
    }

    for label, frame in (("hitter", score_hitters()), ("pitcher", score_pitchers())):
        if frame is None or frame.empty:
            print(f"No {label} candidates for {SEASON}; skipping")
            continue
        payload[label] = summarize(label.capitalize() + "s", frame)

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from config import CURRENT_SEASON as SEASON  # noqa: E402

    raise SystemExit(main())
