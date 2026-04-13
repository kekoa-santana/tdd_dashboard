#!/usr/bin/env python
"""Build per-stat confidence tier thresholds from backtest data.

For each (player_type, stat), sweeps model confidence from 0.50 to 0.95 and
finds the minimum threshold where the cumulative observed hit rate meets each
tier target. Writes:

  - data/dashboard/stat_tier_thresholds.json  (tier lookups consumed by UI)
  - data/dashboard/prop_calibration.parquet   (binned calibration curve)

Usage:
    python scripts/build_prop_calibration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dashboard"

# Minimum observed hit rate to qualify for each tier.
TIER_TARGETS = {
    "Lock":   0.65,
    "Strong": 0.58,
    "Lean":   0.53,
}

BIN_WIDTH = 0.05     # width of each confidence bin
MIN_N_PER_BIN = 25   # ignore bins with fewer picks
MIN_DISTINCT = 2     # need at least this many isotonic levels to trust calibration

# Batter stats are over-only on PrizePicks (and Pick6).  Unders on these
# stats inflate calibration accuracy with picks that cannot actually be taken.
_OVER_ONLY_STATS: set[tuple[str, str]] = {
    ("batter", "H"),
    ("batter", "HR"),
    ("batter", "TB"),
    ("batter", "R"),
    ("batter", "RBI"),
    ("batter", "BB"),
    ("batter", "HRR"),
    ("batter", "K"),
}


# ---------------------------------------------------------------------------
# Loading / joining (mirrors validate_prop_accuracy.py)
# ---------------------------------------------------------------------------

def _load_props() -> pd.DataFrame:
    dk = pd.read_parquet(DATA_DIR / "dk_props_history.parquet")
    pp = pd.read_parquet(DATA_DIR / "pp_props_history.parquet")
    pp = pp[pp.odds_type == "standard"]

    keep = ["player_id", "player_name", "player_type", "stat", "line", "game_date"]
    dk_norm = dk[keep].copy()
    dk_norm["source"] = "dk"
    pp_norm = pp[keep].copy()
    pp_norm["source"] = "pp"

    combined = pd.concat([dk_norm, pp_norm], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["player_id", "stat", "line", "game_date"]
    )
    combined = combined[combined.line % 1 == 0.5].copy()
    return combined


def _match_props_to_game_props(
    props: pd.DataFrame, gp: pd.DataFrame
) -> pd.DataFrame:
    join_cols = ["player_id", "stat"]

    merged_same = props.merge(
        gp, on=join_cols + ["game_date"], how="inner", suffixes=("", "_gp")
    )

    matched_keys = set(
        zip(
            merged_same.player_id,
            merged_same.stat,
            merged_same.game_date,
            merged_same.line,
        )
    )
    props_remaining = props[
        ~props.apply(
            lambda r: (r.player_id, r.stat, r.game_date, r.line) in matched_keys,
            axis=1,
        )
    ].copy()

    if not props_remaining.empty:
        props_remaining["game_date_shifted"] = (
            pd.to_datetime(props_remaining.game_date) + pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
        merged_next = props_remaining.merge(
            gp.rename(columns={"game_date": "game_date_gp"}),
            left_on=join_cols + ["game_date_shifted"],
            right_on=join_cols + ["game_date_gp"],
            how="inner",
            suffixes=("", "_gp"),
        )
        merged_next = merged_next.drop(
            columns=["game_date_shifted", "game_date_gp"], errors="ignore"
        )
    else:
        merged_next = pd.DataFrame()

    return pd.concat([merged_same, merged_next], ignore_index=True)


def load_merged() -> pd.DataFrame:
    props = _load_props()
    gp = pd.read_parquet(DATA_DIR / "game_props.parquet")
    gp = gp[gp.game_status == "final"].copy()

    merged = _match_props_to_game_props(props, gp)

    def _lookup_p_over(row: pd.Series) -> float | None:
        col = f"p_over_{row['line']:.1f}"
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
        return None

    merged["p_over"] = merged.apply(_lookup_p_over, axis=1)
    merged = merged.dropna(subset=["p_over", "actual"])
    merged = merged[merged.actual != merged.line]  # drop pushes

    # Direction + confidence + hit
    merged["pick_over"] = merged["p_over"] >= 0.5
    merged["confidence"] = np.where(
        merged["pick_over"], merged["p_over"], 1 - merged["p_over"]
    )
    merged["hit"] = np.where(
        merged["pick_over"],
        merged["actual"] > merged["line"],
        merged["actual"] < merged["line"],
    ).astype(int)

    # Drop under picks on batter stats (over-only on PrizePicks / Pick6)
    before = len(merged)
    is_restricted_under = (
        ~merged["pick_over"]
        & merged.apply(
            lambda r: (r["player_type"], r["stat"]) in _OVER_ONLY_STATS, axis=1
        )
    )
    merged = merged[~is_restricted_under].copy()
    dropped = before - len(merged)
    if dropped:
        print(f"Excluded {dropped} batter under picks (over-only stats)")

    return merged


# ---------------------------------------------------------------------------
# Tier fitting
# ---------------------------------------------------------------------------

def compute_tier_thresholds(sub: pd.DataFrame) -> dict:
    """Find min confidence threshold where calibrated hit rate meets target.

    1. Bin picks into BIN_WIDTH-wide confidence windows.
    2. Drop bins with < MIN_N_PER_BIN picks (noisy tail).
    3. Fit weighted isotonic regression on the surviving bins.
       This makes the rate curve monotone non-decreasing while pooling
       adjacent violators weighted by sample size (PAV algorithm).
    4. If PAV produces < MIN_DISTINCT distinct fitted levels, the model's
       confidence signal is uninformative for this stat -- return all None.
    5. Sweep the fitted curve (no extrapolation beyond measured bins) to
       find each tier's minimum threshold.
    """
    result: dict[str, float | int | None] = {"n_total": int(len(sub))}
    none_result = {**result, **{t: None for t in TIER_TARGETS}}

    bins = np.arange(0.50, 1.0, BIN_WIDTH)
    bin_data: list[tuple[float, int, float]] = []
    for b in bins:
        mask = (sub.confidence >= b) & (sub.confidence < b + BIN_WIDTH)
        n = int(mask.sum())
        if n >= MIN_N_PER_BIN:
            bin_data.append((round(float(b), 2), n, float(sub.loc[mask, "hit"].mean())))

    if len(bin_data) < 2:
        return none_result

    x_bins = np.array([d[0] for d in bin_data])
    y_bins = np.array([d[2] for d in bin_data])
    w_bins = np.array([d[1] for d in bin_data])

    ir = IsotonicRegression(increasing=True)
    y_fit = ir.fit_transform(x_bins, y_bins, sample_weight=w_bins)

    if len(set(np.round(y_fit, 4))) < MIN_DISTINCT:
        return none_result

    for tier, target in TIER_TARGETS.items():
        threshold: float | None = None
        for x, y in zip(x_bins, y_fit):
            if y >= target:
                threshold = round(float(x), 2)
                break
        result[tier] = threshold

    return result


def build_calibration_table(sub: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0.50, 1.01, 0.02)
    sub = sub.copy()
    sub["conf_bin"] = pd.cut(sub.confidence, bins=bins, include_lowest=True)
    grp = (
        sub.groupby("conf_bin", observed=True)
        .agg(n=("hit", "size"), hits=("hit", "sum"), rate=("hit", "mean"))
        .reset_index()
    )
    grp["conf_bin"] = grp["conf_bin"].astype(str)
    return grp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    merged = load_merged()
    print(f"Merged props: {len(merged)}")
    print(f"Stats: {merged.stat.value_counts().to_dict()}")
    print(f"Player types: {merged.player_type.value_counts().to_dict()}")
    print()

    thresholds: dict = {}
    calibration_rows: list[pd.DataFrame] = []

    for (ptype, stat), sub in merged.groupby(["player_type", "stat"]):
        key = f"{ptype}_{stat}"
        thresholds[key] = compute_tier_thresholds(sub)

        cal = build_calibration_table(sub)
        cal["player_type"] = ptype
        cal["stat"] = stat
        calibration_rows.append(cal)

    out_json = DATA_DIR / "stat_tier_thresholds.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "tier_targets": TIER_TARGETS,
                "min_n_per_bin": MIN_N_PER_BIN,
            "bin_width": BIN_WIDTH,
                "thresholds": thresholds,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    if calibration_rows:
        cal_df = pd.concat(calibration_rows, ignore_index=True)
        cal_df.to_parquet(DATA_DIR / "prop_calibration.parquet", index=False)
        print(f"Wrote {DATA_DIR / 'prop_calibration.parquet'}")

    print()
    print("=" * 68)
    print("TIER THRESHOLDS  (min model confidence to qualify)")
    print("=" * 68)
    print(f"  {'key':<14} {'n':>5}   {'Lock':>6}  {'Strong':>7}  {'Lean':>6}")
    print(f"  {'-' * 14} {'-' * 5}   {'-' * 6}  {'-' * 7}  {'-' * 6}")
    for key in sorted(thresholds):
        info = thresholds[key]
        n = info["n_total"]
        def fmt(t):
            v = info.get(t)
            return f"{v:.2f}" if isinstance(v, float) else "  --"
        print(f"  {key:<14} {n:>5}   {fmt('Lock'):>6}  {fmt('Strong'):>7}  {fmt('Lean'):>6}")
    print()


if __name__ == "__main__":
    main()
