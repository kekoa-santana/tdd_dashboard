"""Validate model accuracy against actual sportsbook prop lines.

Loads DK + PP (standard only) prop history, deduplicates across sources,
joins to game_props model predictions and actuals, then reports hit rates
at 60%/70%/80% confidence thresholds for both over and under picks.

Usage:
    python scripts/validate_prop_accuracy.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dashboard"

# Confidence thresholds to evaluate
THRESHOLDS = [0.60, 0.70, 0.80]


def load_props() -> pd.DataFrame:
    """Load DK + PP (standard only) props, deduplicated across sources."""
    dk = pd.read_parquet(DATA_DIR / "dk_props_history.parquet")
    pp = pd.read_parquet(DATA_DIR / "pp_props_history.parquet")

    # PP: keep only standard odds type
    pp = pp[pp.odds_type == "standard"]

    # Normalize to common columns
    keep = ["player_id", "player_name", "player_type", "stat", "line", "game_date"]
    dk_norm = dk[keep].copy()
    dk_norm["source"] = "dk"
    pp_norm = pp[keep].copy()
    pp_norm["source"] = "pp"

    combined = pd.concat([dk_norm, pp_norm], ignore_index=True)

    # Deduplicate: same player + stat + line + date = one prop
    combined = combined.drop_duplicates(subset=["player_id", "stat", "line", "game_date"])

    # Drop whole-number lines -- no matching p_over column and pushes possible
    combined = combined[combined.line % 1 == 0.5].copy()

    return combined


def load_game_props() -> pd.DataFrame:
    """Load game_props with model predictions and actuals."""
    gp = pd.read_parquet(DATA_DIR / "game_props.parquet")
    # Only final games have reliable actuals
    return gp[gp.game_status == "final"].copy()


def match_props_to_game_props(
    props: pd.DataFrame, gp: pd.DataFrame
) -> pd.DataFrame:
    """Join props to game_props, handling a 1-day date offset.

    game_props dates are sometimes +1 day relative to prop dates (likely a
    timezone / schedule-cutoff issue).  We try same-date first, then +1 day
    for any unmatched props, preferring the match where the player actually
    appeared (has an actual result).
    """
    join_cols = ["player_id", "stat"]

    # Attempt 1: same-date join
    merged_same = props.merge(
        gp,
        on=join_cols + ["game_date"],
        how="inner",
        suffixes=("", "_gp"),
    )

    matched_keys = set(
        zip(merged_same.player_id, merged_same.stat, merged_same.game_date, merged_same.line)
    )

    # Attempt 2: +1 day join for unmatched props
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
        # Keep original prop date, drop helper columns
        merged_next = merged_next.drop(
            columns=["game_date_shifted", "game_date_gp"], errors="ignore"
        )
    else:
        merged_next = pd.DataFrame()

    merged = pd.concat([merged_same, merged_next], ignore_index=True)
    return merged


def lookup_p_over(row: pd.Series) -> float | None:
    """Look up model p(over) for a specific line from the p_over_X.5 columns."""
    col = f"p_over_{row['line']:.1f}"
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return None


def build_accuracy_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Build accuracy summary across confidence thresholds and directions."""
    rows = []
    for threshold in THRESHOLDS:
        # Over picks: model says p_over >= threshold
        over_mask = merged.p_over_at_line >= threshold
        over = merged[over_mask]
        over_hits = (over.actual > over.line).sum()
        over_n = len(over)
        over_acc = over_hits / over_n if over_n > 0 else None

        # Under picks: model says p_under >= threshold (p_over <= 1 - threshold)
        under_mask = merged.p_over_at_line <= (1 - threshold)
        under = merged[under_mask]
        under_hits = (under.actual < under.line).sum()
        under_n = len(under)
        under_acc = under_hits / under_n if under_n > 0 else None

        # Combined: any confident pick in either direction
        either_mask = over_mask | under_mask
        either = merged[either_mask]
        either_correct = (
            ((either.p_over_at_line >= threshold) & (either.actual > either.line))
            | ((either.p_over_at_line <= (1 - threshold)) & (either.actual < either.line))
        )
        either_hits = either_correct.sum()
        either_n = len(either)
        either_acc = either_hits / either_n if either_n > 0 else None

        rows.append({
            "threshold": f"{threshold:.0%}+",
            "over_n": over_n,
            "over_hits": over_hits,
            "over_acc": over_acc,
            "under_n": under_n,
            "under_hits": under_hits,
            "under_acc": under_acc,
            "combined_n": either_n,
            "combined_hits": either_hits,
            "combined_acc": either_acc,
        })

    return pd.DataFrame(rows)


def build_stat_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    """Break down accuracy by stat type at each threshold."""
    rows = []
    for threshold in THRESHOLDS:
        for stat in sorted(merged.stat.unique()):
            sub = merged[merged.stat == stat]

            over_mask = sub.p_over_at_line >= threshold
            under_mask = sub.p_over_at_line <= (1 - threshold)
            either_mask = over_mask | under_mask
            either = sub[either_mask]

            if len(either) == 0:
                continue

            correct = (
                ((either.p_over_at_line >= threshold) & (either.actual > either.line))
                | ((either.p_over_at_line <= (1 - threshold)) & (either.actual < either.line))
            )
            rows.append({
                "threshold": f"{threshold:.0%}+",
                "stat": stat,
                "n": len(either),
                "hits": correct.sum(),
                "acc": correct.sum() / len(either),
            })

    return pd.DataFrame(rows)


def main() -> None:
    # --- Load and join ---
    props = load_props()
    gp = load_game_props()

    print(f"Props loaded: {len(props)} (DK + PP standard, deduplicated, .5 hooks only)")
    print(f"Game props (final): {len(gp)}")
    print(f"Prop dates: {sorted(props.game_date.unique())}")
    print()

    # Join props to game_props (handles 1-day date offset)
    merged = match_props_to_game_props(props, gp)
    print(f"Matched props to game_props: {len(merged)}")

    # Look up model p(over) at the actual prop line
    merged["p_over_at_line"] = merged.apply(lookup_p_over, axis=1)

    # Drop rows without a model prediction or actual
    before = len(merged)
    merged = merged.dropna(subset=["p_over_at_line", "actual"])
    print(f"With model prediction + actual: {len(merged)} (dropped {before - len(merged)})")

    # Exclude pushes (shouldn't happen with .5 hooks)
    pushes = (merged.actual == merged.line).sum()
    if pushes > 0:
        print(f"Excluding {pushes} pushes")
        merged = merged[merged.actual != merged.line]

    print(f"Final evaluation set: {len(merged)} props")
    print(f"  Dates: {sorted(merged.game_date.unique())}")
    print(f"  Stats: {merged.stat.value_counts().to_dict()}")
    print(f"  Player types: {merged.player_type.value_counts().to_dict()}")
    print()

    # --- Overall accuracy ---
    acc = build_accuracy_table(merged)
    print("=" * 70)
    print("OVERALL ACCURACY BY CONFIDENCE THRESHOLD")
    print("=" * 70)
    for _, row in acc.iterrows():
        print(f"\n  {row['threshold']} confidence:")
        print(f"    Over  -- {row['over_hits']:>4}/{row['over_n']:<4} = {row['over_acc']:.1%}" if row["over_acc"] is not None else f"    Over  -- 0 props")
        print(f"    Under -- {row['under_hits']:>4}/{row['under_n']:<4} = {row['under_acc']:.1%}" if row["under_acc"] is not None else f"    Under -- 0 props")
        print(f"    Both  -- {row['combined_hits']:>4}/{row['combined_n']:<4} = {row['combined_acc']:.1%}" if row["combined_acc"] is not None else f"    Both  -- 0 props")

    # --- Breakdown by stat ---
    stat_df = build_stat_breakdown(merged)
    if not stat_df.empty:
        print()
        print("=" * 70)
        print("ACCURACY BY STAT TYPE")
        print("=" * 70)
        for threshold in THRESHOLDS:
            label = f"{threshold:.0%}+"
            sub = stat_df[stat_df.threshold == label]
            if sub.empty:
                continue
            print(f"\n  {label} confidence:")
            for _, row in sub.iterrows():
                print(f"    {row['stat']:<5} -- {row['hits']:>4}/{row['n']:<4} = {row['acc']:.1%}")

    print()


if __name__ == "__main__":
    main()
