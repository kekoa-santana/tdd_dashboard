"""Hit rate for the new softmax sim on the last N days of PP + DK props.

Consumes ``softmax_predictions.parquet`` produced by
``player_profiles/scripts/validate_softmax_recent.py`` and scores each
PP / DK line using a Poisson (count stats) or Normal (outs) CDF from the
sim's per-pitcher ``(expected, std)``. Output format mirrors the existing
``validate_props_lab_hits.py`` so numbers are directly comparable.

Also scores game totals (O/U) against ``game_odds_history.parquet``.

Usage
-----
    python scripts/validate_softmax_hits.py
    python scripts/validate_softmax_hits.py --min-confidence 0.65
    python scripts/validate_softmax_hits.py --include-dk-market
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "dashboard"

# Dashboard stat label → sim prefix (exact p_{prefix}_over_X.Y columns).
# Separate lookups for pitcher vs batter since same stat label (K / H / HR)
# applies to different sim outputs on each side.
_PITCHER_STAT_TO_SIM: dict[str, str] = {
    "K": "k", "BB": "bb", "H": "h", "HR": "hr", "Outs": "outs",
}
_BATTER_STAT_TO_SIM: dict[str, str] = {
    "K": "k", "BB": "bb", "H": "h", "HR": "hr", "TB": "tb",
    "R": "r", "RBI": "rbi", "HRR": "hrr",
    "2B": "double", "3B": "triple",
}

# Union of pitcher + batter supported stats for initial filtering
_ALL_SUPPORTED_STATS = set(_PITCHER_STAT_TO_SIM.keys()) | set(_BATTER_STAT_TO_SIM.keys())

# PP standard batter props where Less is not offered (mirror of existing)
_PP_STANDARD_BATTER_NO_UNDER_STATS = frozenset({
    "H", "HR", "TB", "R", "RBI", "BB", "HRR", "K",
})


def _unders_allowed(line_tier: str, player_type: str, stat: str) -> bool:
    lt = (line_tier or "").strip().lower()
    pt = (player_type or "").strip().lower()
    if lt in ("demon", "goblin"):
        return False
    if lt == "market":
        return True
    if lt == "standard":
        if pt == "pitcher":
            return True
        if pt == "batter" and stat in _PP_STANDARD_BATTER_NO_UNDER_STATS:
            return False
        return True
    return False


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


# Override for teams whose roster.team_name doesn't match the nickname books use.
# 109 (Arizona) ships as "D-backs" in roster but books write "Diamondbacks".
_ALIAS_OVERRIDES: dict[int, list[str]] = {
    109: ["Diamondbacks", "D-backs"],
}


def _load_team_aliases() -> dict[int, list[str]]:
    """Build team_id -> list of name substrings that can appear in a book's
    `game_description`. Combines roster team_name (MLB-style nickname like
    "Diamondbacks") with team_abbr, applying `_ALIAS_OVERRIDES` for teams
    whose roster nickname doesn't match the book wording. Longest name first
    so multi-word nicknames ("White Sox", "Red Sox", "Blue Jays") match
    before any shorter overlapping alias.
    """
    from pathlib import Path
    dash_dir = Path(__file__).resolve().parents[1] / "data" / "dashboard"
    try:
        roster = pd.read_parquet(dash_dir / "roster.parquet")
    except Exception:
        return {}
    if "org_id" not in roster.columns or "team_name" not in roster.columns:
        return {}
    pairs = (
        roster[["org_id", "team_name", "team_abbr"]]
        .dropna()
        .drop_duplicates(subset=["org_id"])
    )
    aliases: dict[int, list[str]] = {}
    for _, r in pairs.iterrows():
        tid = int(r["org_id"])
        names = list(_ALIAS_OVERRIDES.get(tid, []))
        names.append(str(r["team_name"]))
        if pd.notna(r.get("team_abbr")):
            names.append(str(r["team_abbr"]))
        aliases[tid] = sorted(set(names), key=len, reverse=True)
    return aliases


def _infer_team_ids(
    game_description: object, aliases: dict[int, list[str]],
) -> frozenset[int] | None:
    """Return the pair of team_ids whose nicknames both appear in the
    `game_description` string (e.g. 'Arizona Diamondbacks @ New York Mets'
    or 'ARI Diamondbacks @ NY Mets'). Returns None if not exactly 2 teams match.
    """
    if not isinstance(game_description, str) or not aliases:
        return None
    desc_lower = game_description.lower()
    matched: set[int] = set()
    # Try every nickname alias for each team (longest first), not just the
    # first one, so teams with multiple name forms (e.g. 109 = Diamondbacks
    # or D-backs) match whichever the book uses.
    for tid, names in aliases.items():
        for alias in names:
            # Skip short abbreviations (<=3 chars); collision-prone
            # ("NY" vs "NYY"/"NYM", "LA" vs "LAA"/"LAD", "SD" vs "STL").
            if len(alias) <= 3:
                continue
            if alias.lower() in desc_lower:
                matched.add(tid)
                break
    if len(matched) != 2:
        return None
    return frozenset(matched)


def _confidence_tier(
    thresholds: dict, player_type: str, stat: str, model_p: float,
) -> str:
    confidence = max(model_p, 1 - model_p)
    lookup = thresholds.get("thresholds", {}).get(f"{player_type}_{stat}")
    if not lookup:
        return "Pass"
    for tier in ("Lock", "Strong", "Lean"):
        thr = lookup.get(tier)
        if thr is not None and confidence >= thr:
            return tier
    return "Pass"


def build_book_props(dk: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    """Same dedup logic as validate_props_lab_hits.build_book_props."""
    d_norm = pd.DataFrame()
    p_norm = pd.DataFrame()
    if not dk.empty:
        d = dk.copy()
        d["game_date"] = _norm_date(d["game_date"])
        d["line_tier"] = "market"
        d["line"] = pd.to_numeric(d["line"], errors="coerce")
        d = d.dropna(subset=["player_id", "stat", "line", "game_date"])
        d["player_id"] = d["player_id"].astype(int)
        d_norm = d[["player_id", "player_type", "stat", "line",
                    "game_date", "line_tier"]]
    if not pp.empty:
        p = pp.copy()
        p["game_date"] = _norm_date(p["game_date"])
        p["line"] = pd.to_numeric(p["line"], errors="coerce")
        p = p.dropna(subset=["player_id", "stat", "line", "game_date"])
        p["player_id"] = p["player_id"].astype(int)
        if "odds_type" not in p.columns:
            p["odds_type"] = "standard"
        p["line_tier"] = p["odds_type"].astype(str).str.lower().str.strip()
        p_norm = p[["player_id", "player_type", "stat", "line",
                    "game_date", "line_tier"]]
    if not d_norm.empty and not p_norm.empty:
        dk_keys = d_norm[["player_id", "stat", "line", "game_date"]].drop_duplicates()
        dk_keys["_dk_match"] = True
        p_norm = p_norm.merge(
            dk_keys, on=["player_id", "stat", "line", "game_date"], how="left",
        )
        p_norm = p_norm[p_norm["_dk_match"].isna()].drop(columns=["_dk_match"])
    parts = [df for df in (d_norm, p_norm) if not df.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[out["line"] % 1 == 0.5].copy()
    out = out.drop_duplicates(
        subset=["player_id", "stat", "line", "line_tier", "game_date"],
        keep="first",
    )
    return out


def _p_over_for_line(
    stat: str, row: pd.Series, line: float, player_type: str,
) -> float:
    """Look up exact P(over line) for a stat from the sim sample columns."""
    table = _PITCHER_STAT_TO_SIM if player_type == "pitcher" else _BATTER_STAT_TO_SIM
    sim_key = table.get(stat)
    if sim_key is None:
        return float("nan")
    snapped = round(line * 2) / 2
    col = f"p_{sim_key}_over_{snapped:.1f}"
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return float("nan")


def score_props(
    pitcher_preds: pd.DataFrame,
    batter_preds: pd.DataFrame,
    props: pd.DataFrame,
    actuals_gp: pd.DataFrame,
) -> pd.DataFrame:
    """Join book lines to pitcher + batter sim predictions; compute hit."""
    # Split props by player_type (pitcher vs batter) and score separately
    pitcher_props = props[props["player_type"] == "pitcher"].copy()
    batter_props = props[props["player_type"] == "batter"].copy()
    # Filter to stats we can score on each side
    pitcher_props = pitcher_props[pitcher_props["stat"].isin(_PITCHER_STAT_TO_SIM)]
    batter_props = batter_props[batter_props["stat"].isin(_BATTER_STAT_TO_SIM)]

    def _join_with_offset(props_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
        """Join on (player_id, game_date) same-date then +1 day offset.

        Book line dates are recorded when the line is offered (day before),
        so ``prop.game_date + 1 day`` often equals ``sim.game_date``. The
        result always carries ``game_date`` = the actual (sim) game date so
        downstream joins (actuals) line up correctly.
        """
        sim_renamed = sim_df.rename(columns={"game_date": "actual_game_date"})
        # Same-date attempt
        m_same = props_df.merge(
            sim_renamed, left_on=["player_id", "game_date"],
            right_on=["player_id", "actual_game_date"],
            how="inner", suffixes=("", "_sim"),
        )
        matched = set(
            zip(m_same["player_id"], m_same["stat"],
                m_same["game_date"], m_same["line"])
        )
        remaining = props_df[~props_df.apply(
            lambda r: (r["player_id"], r["stat"],
                       r["game_date"], r["line"]) in matched, axis=1,
        )].copy()
        if not remaining.empty:
            remaining["game_date_shifted"] = (
                pd.to_datetime(remaining["game_date"]) + pd.Timedelta(days=1)
            ).dt.strftime("%Y-%m-%d")
            m_next = remaining.merge(
                sim_renamed, left_on=["player_id", "game_date_shifted"],
                right_on=["player_id", "actual_game_date"], how="inner",
            ).drop(columns=["game_date_shifted"], errors="ignore")
            result = pd.concat([m_same, m_next], ignore_index=True)
        else:
            result = m_same
        # Replace prop-date with actual game date so actuals joins line up
        result["game_date"] = result["actual_game_date"]
        result = result.drop(columns=["actual_game_date"], errors="ignore")
        return result

    merged_parts = []
    if not pitcher_props.empty and not pitcher_preds.empty:
        sim = pitcher_preds.rename(columns={"pitcher_id": "player_id"}).copy()
        sim["game_date"] = _norm_date(sim["game_date"])
        sim["player_id"] = sim["player_id"].astype(int)
        m = _join_with_offset(pitcher_props, sim)
        logger.info("Pitcher props joined: %d", len(m))
        if not m.empty:
            m["model_p"] = m.apply(
                lambda r: _p_over_for_line(
                    str(r["stat"]), r, float(r["line"]), "pitcher",
                ), axis=1,
            )
            merged_parts.append(m)

    if not batter_props.empty and not batter_preds.empty:
        sim = batter_preds.rename(columns={"batter_id": "player_id"}).copy()
        sim["game_date"] = _norm_date(sim["game_date"])
        sim["player_id"] = sim["player_id"].astype(int)
        m = _join_with_offset(batter_props, sim)
        logger.info("Batter props joined: %d", len(m))
        if not m.empty:
            m["model_p"] = m.apply(
                lambda r: _p_over_for_line(
                    str(r["stat"]), r, float(r["line"]), "batter",
                ), axis=1,
            )
            merged_parts.append(m)

    if not merged_parts:
        return pd.DataFrame()
    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.dropna(subset=["model_p"]).copy()
    merged["model_p"] = np.clip(merged["model_p"], 0.0, 1.0)

    # Attach actuals from game_props on (player_id, game_date, stat)
    actuals = actuals_gp[["player_id", "stat", "game_date", "actual"]].copy()
    actuals["game_date"] = _norm_date(actuals["game_date"])
    merged = merged.merge(actuals, on=["player_id", "stat", "game_date"], how="left")
    merged["actual"] = pd.to_numeric(merged["actual"], errors="coerce")
    merged = merged.dropna(subset=["actual"])

    # Drop pushes
    merged = merged[merged["actual"] != merged["line"]].copy()
    return merged


def _print_trust_lines(df: pd.DataFrame, min_n: int = 5) -> None:
    print("TRUST LINES (hit rate by stat / tier / player_type)")
    print("-" * 72)
    if df.empty:
        print("  (no picks)")
        return
    keys = ["stat", "line_tier", "player_type"]
    rows = []
    for k, sub in df.groupby(keys):
        if len(sub) < min_n:
            continue
        h = int(sub["hit"].sum())
        n = len(sub)
        rows.append((k, n, h, h / n))
    rows.sort(key=lambda r: (-r[3], -r[1]))
    for k, n, h, r in rows:
        stat, tier, pt = k
        ptc = "P" if pt == "pitcher" else "H"
        print(f"  {stat:5s} {tier:9s} {ptc}  n={n:5d}  hits={h:5d}  ({r:.1%})")


def validate_game_totals(
    preds: pd.DataFrame,
    odds: pd.DataFrame,
    actuals_gp: pd.DataFrame,
    min_confidence: float,
) -> None:
    """Score game O/U totals vs book using paired pitcher predictions."""
    if odds.empty:
        print("\n(no game_odds_history available; skipping game totals)")
        return

    # Pair pitchers per game → total expected + variance
    grouped = preds.groupby("game_pk").agg(
        total_exp=("expected_runs", "sum"),
        total_var=("std_runs", lambda s: float(np.sum(s.astype(float) ** 2))),
        game_date=("game_date", "first"),
        n_pitchers=("pitcher_id", "count"),
        team_ids=("pitcher_team_id", lambda s: frozenset(int(x) for x in s.dropna())),
    ).reset_index()
    grouped = grouped[grouped["n_pitchers"] == 2].copy()
    grouped = grouped[grouped["team_ids"].apply(len) == 2].copy()
    grouped["total_std"] = np.sqrt(np.maximum(grouped["total_var"], 1.0))
    grouped["game_date"] = _norm_date(grouped["game_date"])

    # Actual game totals: the `actual_total_runs` column is per-pitcher (runs
    # scored against that pitcher's team = the OPPOSING team's runs), so each
    # game has two complementary values that sum to the real game total.
    # Example: game 822828 -> pitcher@team141=8, pitcher@team142=2, total=10.
    actual_runs = preds.groupby("game_pk")["actual_total_runs"].sum().reset_index()
    grouped = grouped.merge(actual_runs, on="game_pk", how="left")
    grouped = grouped.dropna(subset=["actual_total_runs"])

    # Filter odds to totals only
    tot = odds[odds["market_type"].str.contains("total", case=False, na=False)].copy()
    if tot.empty:
        print("\n(no total markets in game_odds_history; skipping)")
        return
    tot["game_date"] = _norm_date(tot["game_date"])
    # Use latest snapshot per (game, line)
    tot = tot.sort_values("snapshot_ts").drop_duplicates(
        subset=["game_date", "game_description", "line", "outcome_type"],
        keep="last",
    )
    # Get over rows; book line. Use team_or_label (populated by both DK and Bovada)
    # rather than outcome_type (DK-only).
    over = tot[tot["team_or_label"].str.lower() == "over"][
        ["game_date", "game_description", "line"]
    ].drop_duplicates()
    over["game_date"] = _norm_date(over["game_date"])

    # Map each odds row's `game_description` to the pair of team_ids it represents,
    # then join on (game_date, frozenset(team_ids)) so each predicted game is paired
    # with its own book line rather than the nearest line from any game that day.
    team_aliases = _load_team_aliases()  # team_id -> list[str] of name substrings
    over["team_ids"] = over["game_description"].apply(
        lambda d: _infer_team_ids(d, team_aliases),
    )
    over = over[over["team_ids"].apply(lambda s: isinstance(s, frozenset) and len(s) == 2)]
    if over.empty:
        print("\n(could not match any game_description to team_ids; skipping)")
        return

    # Per-game consensus book line: median across sources (DK + Bovada)
    over = (
        over.groupby(["game_date", "team_ids"])["line"]
        .median()
        .reset_index()
    )
    joined = grouped.merge(over, on=["game_date", "team_ids"], how="inner")
    if joined.empty:
        print("\n(no predicted games matched to book totals; skipping)")
        return

    joined["model_p_over"] = joined.apply(
        lambda r: float(
            1.0 - norm.cdf(r["line"], loc=r["total_exp"], scale=r["total_std"])
        ), axis=1,
    )
    joined["pick_over"] = joined["model_p_over"] > 0.5
    joined["confidence"] = joined["model_p_over"].where(
        joined["pick_over"], 1.0 - joined["model_p_over"],
    )
    joined["hit"] = np.where(
        joined["pick_over"],
        joined["actual_total_runs"] > joined["line"],
        joined["actual_total_runs"] < joined["line"],
    )
    joined = joined[joined["actual_total_runs"] != joined["line"]].copy()
    eligible = joined[joined["confidence"] >= min_confidence]

    print()
    print("=" * 72)
    print(f"GAME TOTALS (O/U) — book: game_odds_history")
    print("=" * 72)
    print(f"  paired games with book line: {len(joined)}")
    print(f"  eligible (conf >= {min_confidence:.0%}): {len(eligible)}")
    if not eligible.empty:
        h = int(eligible["hit"].sum())
        n = len(eligible)
        print(f"  hits: {h} / {n} ({h / n:.1%})")
        overs = eligible[eligible["pick_over"]]
        unders = eligible[~eligible["pick_over"]]
        if not overs.empty:
            ho = int(overs["hit"].sum())
            print(f"    overs:  {ho} / {len(overs)} ({ho / len(overs):.1%})")
        if not unders.empty:
            hu = int(unders["hit"].sum())
            print(f"    unders: {hu} / {len(unders)} ({hu / len(unders):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--over-only", action="store_true")
    parser.add_argument("--include-dk-market", action="store_true")
    parser.add_argument("--min-cell-n", type=int, default=5)
    args = parser.parse_args()

    data_dir = args.data_dir
    pp_pred_path = data_dir / "softmax_pitcher_predictions.parquet"
    ba_pred_path = data_dir / "softmax_batter_predictions.parquet"
    gp_path = data_dir / "game_props.parquet"
    pp_path = data_dir / "pp_props_history.parquet"
    dk_path = data_dir / "dk_props_history.parquet"
    odds_path = data_dir / "game_odds_history.parquet"
    tier_path = data_dir / "stat_tier_thresholds.json"

    if not gp_path.exists() or not pp_path.exists():
        logger.error("Missing game_props or pp_props_history")
        sys.exit(1)

    pitcher_preds = pd.read_parquet(pp_pred_path) if pp_pred_path.exists() else pd.DataFrame()
    batter_preds = pd.read_parquet(ba_pred_path) if ba_pred_path.exists() else pd.DataFrame()
    logger.info(
        "Loaded softmax predictions — pitcher: %d rows / %d games | batter: %d rows / %d games",
        len(pitcher_preds),
        pitcher_preds["game_pk"].nunique() if not pitcher_preds.empty else 0,
        len(batter_preds),
        batter_preds["game_pk"].nunique() if not batter_preds.empty else 0,
    )

    gp = pd.read_parquet(gp_path)
    gp = gp[gp["game_status"].astype(str).str.lower() == "final"].copy()

    pp = pd.read_parquet(pp_path)
    dk = pd.read_parquet(dk_path) if (args.include_dk_market and dk_path.exists()) else pd.DataFrame()

    props = build_book_props(dk, pp)
    if props.empty:
        logger.error("No props rows after stacking")
        sys.exit(1)

    # Tag player_type from game_props so we know which side to score
    pt_lookup = gp[["player_id", "player_type"]].drop_duplicates("player_id")
    props = props.drop(columns=["player_type"], errors="ignore").merge(
        pt_lookup, on="player_id", how="left",
    )
    props["player_type"] = props["player_type"].fillna("pitcher")

    scored = score_props(pitcher_preds, batter_preds, props, gp)
    if scored.empty:
        logger.error("No props matched to softmax predictions")
        sys.exit(1)

    thresholds = {}
    if tier_path.exists():
        with open(tier_path, encoding="utf-8") as f:
            thresholds = json.load(f)

    scored["conf_tier"] = scored.apply(
        lambda r: _confidence_tier(
            thresholds, str(r.get("player_type", "")),
            str(r["stat"]), float(r["model_p"]),
        ), axis=1,
    )

    scored = scored[scored["model_p"] != 0.5].copy()
    scored["pick_over"] = scored["model_p"] > 0.5
    over = scored[scored["pick_over"]].copy()
    over["confidence"] = over["model_p"]
    over["hit"] = over["actual"] > over["line"]

    if args.over_only:
        picks = over
    else:
        under = scored[~scored["pick_over"]].copy()
        allow = under.apply(
            lambda r: _unders_allowed(
                str(r["line_tier"]), str(r["player_type"]), str(r["stat"]),
            ), axis=1,
        )
        under = under[allow].copy()
        under["confidence"] = 1.0 - under["model_p"]
        under["hit"] = under["actual"] < under["line"]
        picks = pd.concat([over, under], ignore_index=True)

    eligible = picks[picks["confidence"] >= args.min_confidence].copy()
    eligible["side"] = np.where(eligible["pick_over"], "Over", "Under")

    scope = "PP + DK market" if args.include_dk_market else "PrizePicks only"
    print()
    print("=" * 72)
    print(f"SOFTMAX SIM HIT RATE - {scope}")
    print(f"  (confidence >= {args.min_confidence:.0%})")
    print("=" * 72)
    if eligible.empty:
        print("No picks meet the confidence threshold.")
    else:
        n = len(eligible)
        h = int(eligible["hit"].sum())
        print(f"  n picks:  {n}")
        print(f"  hits:     {h}")
        print(f"  hit rate: {h / n:.1%}")
        print(f"  overs: {int(eligible['pick_over'].sum())}  |  unders: {int((~eligible['pick_over']).sum())}")
        print()
        _print_trust_lines(eligible, min_n=args.min_cell_n)

    # Per-side breakdown (pitcher vs batter)
    if not eligible.empty:
        print()
        print("BY PLAYER TYPE")
        print("-" * 72)
        for pt, sub in eligible.groupby("player_type"):
            h = int(sub["hit"].sum())
            n = len(sub)
            print(f"  {str(pt):8s}  n={n:5d}  hits={h:5d}  ({h / n:.1%})")
        print()

    # Game totals (uses only pitcher preds; runs come from pitcher sim)
    odds = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
    if not pitcher_preds.empty:
        validate_game_totals(pitcher_preds, odds, gp, args.min_confidence)


if __name__ == "__main__":
    main()
