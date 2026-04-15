#!/usr/bin/env python
"""Hit rate for Props Lab–style picks vs actual results.

Joins Prize Picks (and optionally DraftKings) book lines to ``game_props``,
then reports hit rates on finalized games.

**Default (PrizePicks-only)**  
    DraftKings rows (``line_tier=market``) are **excluded** — use
    ``--include-dk-market`` if you want them. Typical CA / no-DK workflows stay
    PP-only.

**Sides**  
    By default, both **over** and **under** picks are included where the book
    offers that side (see ``_unders_allowed``). Demon/Goblin are **More only**
    per PrizePicks Help Center. Use ``--over-only`` for overs alone.

**Confidence**  
    Filter: ``--min-confidence`` applies to P(over) on overs and to P(under) on
    unders. **Conf tier** (Lock / Strong / Lean / Pass) still uses
    ``max(model_p, 1-model_p)`` like the dashboard.

Requires under ``data/dashboard/``:

- ``game_props.parquet``, ``pp_props_history.parquet``
- ``dk_props_history.parquet`` (only if ``--include-dk-market``)
- ``stat_tier_thresholds.json`` (optional)

**TRUST LINES** (readable buckets for talking to users)::

    P(over) >= 60%:
      TB    demon    H    n=167  hits=64  (38.4%)

  Stat, PrizePicks tier (or ``market`` for DK), ``H``/``P`` = batter/pitcher.
  Sorted by hit rate (then sample size). Use ``--compact`` for this + header only.

Usage
-----
cd tdd-dashboard
python scripts/validate_props_lab_hits.py --compact
python scripts/validate_props_lab_hits.py
python scripts/validate_props_lab_hits.py --min-confidence 0.65 --over-only
python scripts/validate_props_lab_hits.py --include-dk-market
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "dashboard"

# PP *standard* batter props: many pick'em markets (hits, H+R+RBI, RBI, etc.)
# are only offered on More in practice; do not count under-side for these.
# (Pitcher standard: unders are common, e.g. K / BB / outs.)
_PP_STANDARD_BATTER_NO_UNDER_STATS: frozenset[str] = frozenset({
    "H",
    "HR",
    "TB",
    "R",
    "RBI",
    "BB",
    "HRR",
    "K",
})


def _unders_allowed(line_tier: str, player_type: str, stat: str) -> bool:
    """Whether a **Less / under** pick is offered for this book row.

    PrizePicks Help Center: Demon and Goblin squares only allow **More**.
    Standard PP projections allow **More** and **Less**. DraftKings market lines
    are modeled as full over/under for MLB props.

    Parameters
    ----------
    line_tier
        ``market`` | ``standard`` | ``goblin`` | ``demon``.
    player_type
        ``batter`` | ``pitcher``.
    stat
        Game props stat label (``K``, ``H``, ``HRR``, ...).
    """
    lt = (line_tier or "").strip().lower()
    pt = (player_type or "").strip().lower()
    st = str(stat).strip()

    if lt in ("demon", "goblin"):
        return False
    if lt == "market":
        return True
    if lt == "standard":
        if pt == "pitcher":
            return True
        if pt == "batter" and st in _PP_STANDARD_BATTER_NO_UNDER_STATS:
            return False
        return True
    return False


def _float_match(a: float, b: object, tol: float = 1e-6) -> bool:
    if pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


def _p_over_column(line: float) -> str:
    x = float(line)
    x = round(x * 2) / 2.0
    return f"p_over_{x:.1f}"


def _p_over_from_legacy_triplet(row: pd.Series, book: float) -> float:
    for lv_col, p_col in (
        ("line_low", "p_over_low"),
        ("line_mid", "p_over_mid"),
        ("line_high", "p_over_high"),
    ):
        if lv_col not in row.index or p_col not in row.index:
            continue
        if _float_match(book, row[lv_col]) and pd.notna(row[p_col]):
            return float(row[p_col])
    return float("nan")


def _p_over_nearest_grid(row: pd.Series, book: float) -> float:
    best_d = float("inf")
    best_p = float("nan")
    for c in row.index:
        if not isinstance(c, str) or not c.startswith("p_over_"):
            continue
        if c in ("p_over_low", "p_over_mid", "p_over_high"):
            continue
        try:
            lv = float(c.replace("p_over_", ""))
        except ValueError:
            continue
        if pd.isna(row[c]):
            continue
        d = abs(lv - book)
        if d < best_d:
            best_d = d
            best_p = float(row[c])
    return best_p


def _attach_p_over_at_book(df: pd.DataFrame, line_col: str = "line") -> pd.DataFrame:
    """Add ``model_p`` = P(over) at ``line_col`` from grid / legacy / nearest."""

    def lookup(row: pd.Series) -> float:
        book = float(row[line_col])
        col = _p_over_column(book)
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
        legacy = _p_over_from_legacy_triplet(row, book)
        if not np.isnan(legacy):
            return legacy
        return _p_over_nearest_grid(row, book)

    out = df.copy()
    out["model_p"] = out.apply(lookup, axis=1)
    return out


def match_props_to_game_props(
    props: pd.DataFrame, gp: pd.DataFrame
) -> pd.DataFrame:
    """Join book rows to game_props; same date + one-day offset as validate_prop_accuracy."""
    join_cols = ["player_id", "stat"]
    merged_same = props.merge(
        gp,
        on=join_cols + ["game_date"],
        how="inner",
        suffixes=("", "_gp"),
    )
    matched_keys = set(
        zip(
            merged_same.player_id,
            merged_same.stat,
            merged_same.game_date,
            merged_same["line"],
        )
    )
    props_remaining = props[
        ~props.apply(
            lambda r: (r.player_id, r.stat, r.game_date, r["line"]) in matched_keys,
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
            columns=["game_date_shifted", "game_date_gp"],
            errors="ignore",
        )
    else:
        merged_next = pd.DataFrame()
    return pd.concat([merged_same, merged_next], ignore_index=True)


def _confidence_tier(
    thresholds: dict,
    player_type: str,
    stat: str,
    model_p: float,
) -> str:
    """Lock / Strong / Lean / Pass — mirrors ``views/projected_performers._confidence_tier``."""
    confidence = max(model_p, 1 - model_p)
    lookup = thresholds.get("thresholds", {}).get(f"{player_type}_{stat}")
    if not lookup:
        return "Pass"
    for tier in ("Lock", "Strong", "Lean"):
        thr = lookup.get(tier)
        if thr is not None and confidence >= thr:
            return tier
    return "Pass"


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _fmt_hit(sub: pd.DataFrame) -> str:
    if sub.empty:
        return "n=0"
    h = int(sub["hit"].sum())
    n = len(sub)
    return f"n={n:5d}  hits={h:5d}  ({h / n:.1%})"


def _print_breakdown_by(
    df: pd.DataFrame,
    title: str,
    key: str,
    order: list[str] | None = None,
) -> None:
    print(title)
    print("-" * 72)
    keys = order if order is not None else sorted(df[key].astype(str).unique(), key=str)
    for k in keys:
        sub = df[df[key].astype(str) == str(k)]
        lab = str(k)
        if sub.empty:
            print(f"  {lab:12s}  n=0")
        else:
            print(f"  {lab:12s}  {_fmt_hit(sub)}")
    print()


def _print_crosstab(
    df: pd.DataFrame,
    title: str,
    row_key: str,
    col_key: str,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
) -> None:
    print(title)
    print("-" * 72)
    rows = row_order if row_order is not None else sorted(df[row_key].astype(str).unique(), key=str)
    cols = col_order if col_order is not None else sorted(df[col_key].astype(str).unique(), key=str)
    for r in rows:
        parts: list[str] = []
        for c in cols:
            sub = df[(df[row_key].astype(str) == str(r)) & (df[col_key].astype(str) == str(c))]
            if sub.empty:
                parts.append(f"{c}: -")
            else:
                h = int(sub["hit"].sum())
                n = len(sub)
                parts.append(f"{c}: {h}/{n} ({h / n:.0%})")
        print(f"  {str(r):8s}  " + "  |  ".join(parts))
    print()


def _ptype_letter(player_type: str) -> str:
    """Single-letter role for display: H=batter, P=pitcher."""
    pt = (player_type or "").strip().lower()
    if pt == "batter":
        return "H"
    if pt == "pitcher":
        return "P"
    return "?"


def _print_trust_lines(
    eligible: pd.DataFrame,
    *,
    min_confidence: float,
    min_n: int,
    include_under_block: bool,
) -> None:
    """Compact lines: ``STAT tier H n=.. hits=.. (pct)`` for day-to-day trust.

    One block for overs (P(over) at or above threshold) and one for unders
    when applicable.
    """
    thr_pct = min_confidence * 100.0

    over_el = eligible[eligible["pick_over"]].copy()
    if not over_el.empty:
        over_el["ptype"] = over_el["player_type"].map(_ptype_letter)
        print("TRUST LINES (overs)")
        print(f"P(over) >= {thr_pct:g}%:")
        rows: list[tuple[str, str, str, int, int, float]] = []
        for (stat, lt, ph), sub in over_el.groupby(
            ["stat", "line_tier", "ptype"],
            dropna=False,
        ):
            n = len(sub)
            if n < min_n:
                continue
            h = int(sub["hit"].sum())
            rows.append((str(stat), str(lt), str(ph), n, h, h / n if n else 0.0))
        # Best hit rates first (then larger n for ties)
        rows.sort(key=lambda x: (-x[5], -x[3], x[0], x[1], x[2]))
        if not rows:
            print(f"  (no cells with n >= {min_n})")
        else:
            for stat, lt, ph, n, h, rate in rows:
                print(
                    f"  {stat:4s}  {lt:8s}  {ph}  n={n:5d}  hits={h:5d}  ({rate:.1%})",
                )
        print()

    if include_under_block:
        under_el = eligible[~eligible["pick_over"]].copy()
        if not under_el.empty:
            under_el["ptype"] = under_el["player_type"].map(_ptype_letter)
            print("TRUST LINES (unders)")
            print(f"P(under) >= {thr_pct:g}%:")
            rows_u: list[tuple[str, str, str, int, int, float]] = []
            for (stat, lt, ph), sub in under_el.groupby(
                ["stat", "line_tier", "ptype"],
                dropna=False,
            ):
                n = len(sub)
                if n < min_n:
                    continue
                h = int(sub["hit"].sum())
                rows_u.append((str(stat), str(lt), str(ph), n, h, h / n if n else 0.0))
            rows_u.sort(key=lambda x: (-x[5], -x[3], x[0], x[1], x[2]))
            if not rows_u:
                print(f"  (no cells with n >= {min_n})")
            else:
                for stat, lt, ph, n, h, rate in rows_u:
                    print(
                        f"  {stat:4s}  {lt:8s}  {ph}  n={n:5d}  hits={h:5d}  ({rate:.1%})",
                    )
            print()


def _print_three_way(
    df: pd.DataFrame,
    title: str,
    min_cell_n: int,
) -> None:
    print(title)
    print("-" * 72)
    keys = ["side", "conf_tier", "stat"]
    any_printed = False
    for (s, t, st), sub in sorted(
        df.groupby(keys, dropna=False),
        key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2])),
    ):
        if len(sub) < min_cell_n:
            continue
        any_printed = True
        h = int(sub["hit"].sum())
        n = len(sub)
        print(f"  {s:5s}  {t:8s}  {st:6s}  n={n:5d}  hits={h:5d}  ({h / n:.1%})")
    if not any_printed:
        print(f"  (no cells with n >= {min_cell_n})")
    print()


def build_book_props(dk: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    """Long stack of DK + PP offers (same spirit as Props Lab concat).

    When a PrizePicks line value matches a DraftKings market line for the
    same player + stat + date, the PP row is dropped — DK has vig-bearing
    odds and carries more information, so the PP duplicate is redundant.
    PP lines at values DK does not offer (typical for goblin / demon
    off-market) are retained.
    """
    d_norm = pd.DataFrame()
    p_norm = pd.DataFrame()
    if not dk.empty:
        d = dk.copy()
        d["game_date"] = _norm_date(d["game_date"])
        d["line_tier"] = "market"
        d["line"] = pd.to_numeric(d["line"], errors="coerce")
        d = d.dropna(subset=["player_id", "stat", "line", "game_date"])
        d["player_id"] = d["player_id"].astype(int)
        d_norm = d[["player_id", "player_type", "stat", "line", "game_date", "line_tier"]]
    if not pp.empty:
        p = pp.copy()
        p["game_date"] = _norm_date(p["game_date"])
        p["line"] = pd.to_numeric(p["line"], errors="coerce")
        p = p.dropna(subset=["player_id", "stat", "line", "game_date"])
        p["player_id"] = p["player_id"].astype(int)
        if "odds_type" not in p.columns:
            p["odds_type"] = "standard"
        p["line_tier"] = p["odds_type"].astype(str).str.lower().str.strip()
        p_norm = p[["player_id", "player_type", "stat", "line", "game_date", "line_tier"]]

    # Drop PP rows whose line value duplicates a DK market line for
    # the same player / stat / date. Goblin and demon at non-market
    # values stay, as do DK-only lines.
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
    # Half-point lines only (avoid push / missing p_over grid)
    out = out[out["line"] % 1 == 0.5].copy()
    # Defense-in-depth: drop exact duplicates within a source (rare)
    out = out.drop_duplicates(
        subset=["player_id", "stat", "line", "line_tier", "game_date"],
        keep="first",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Props Lab hit rate (historical DK + PP vs game_props actuals)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA,
        help="Dashboard data directory (default: ../data/dashboard)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Min P(over) on overs and min P(under) on unders (default: 0.60)",
    )
    parser.add_argument(
        "--over-only",
        action="store_true",
        help="Only More/over picks (no unders). Default: include unders where allowed.",
    )
    parser.add_argument(
        "--include-dk-market",
        action="store_true",
        help="Include DraftKings rows (line_tier=market). Default: PrizePicks only.",
    )
    parser.add_argument(
        "--min-cell-n",
        type=int,
        default=5,
        help="Minimum n for TRUST LINES cells and the 3-way table (default: 5)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only header + TRUST LINES (skip side/tier/stat crosstabs).",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    gp_path = data_dir / "game_props.parquet"
    dk_path = data_dir / "dk_props_history.parquet"
    pp_path = data_dir / "pp_props_history.parquet"
    tier_path = data_dir / "stat_tier_thresholds.json"

    for p in (gp_path, pp_path):
        if not p.exists():
            logger.error("Missing required file: %s", p)
            sys.exit(1)
    if args.include_dk_market and not dk_path.exists():
        logger.error("--include-dk-market requires %s", dk_path)
        sys.exit(1)

    gp = pd.read_parquet(gp_path)
    gp = gp[gp["game_status"].astype(str).str.lower() == "final"].copy()
    gp["game_date"] = _norm_date(gp["game_date"])
    gp["player_type"] = gp["player_type"].astype(str).str.lower().str.strip()

    if args.include_dk_market:
        dk = pd.read_parquet(dk_path)
    else:
        dk = pd.DataFrame()
        logger.info("Excluding DraftKings (use --include-dk-market to include market lines)")
    pp = pd.read_parquet(pp_path) if pp_path.exists() else pd.DataFrame()
    logger.info(
        "Loaded game_props (final): %d | DK rows: %d | PP history: %d",
        len(gp),
        len(dk),
        len(pp),
    )

    props = build_book_props(dk, pp)
    if props.empty:
        logger.error("No book prop rows after stacking DK + PP")
        sys.exit(1)

    merged = match_props_to_game_props(props, gp)
    logger.info("Matched book rows to game_props: %d", len(merged))
    if merged.empty:
        logger.error("No joins — check game_date alignment")
        sys.exit(1)

    # Book line column name after merge: props used ``line``
    book_col = "line"
    if book_col not in merged.columns:
        logger.error("Expected merged column 'line' for book line")
        sys.exit(1)

    merged = _attach_p_over_at_book(merged, line_col=book_col)
    merged = merged.dropna(subset=["model_p", "actual"])
    merged["model_p"] = np.clip(merged["model_p"].astype(float), 0.0, 1.0)
    merged["actual"] = pd.to_numeric(merged["actual"], errors="coerce")
    merged = merged.dropna(subset=["actual"])

    # Pushes
    line_vals = merged[book_col].astype(float)
    merged = merged[merged["actual"].values != line_vals.values].copy()

    thresholds: dict = {}
    if tier_path.exists():
        with open(tier_path, encoding="utf-8") as f:
            thresholds = json.load(f)

    merged["conf_tier"] = merged.apply(
        lambda r: _confidence_tier(
            thresholds,
            str(r.get("player_type", "")),
            str(r.get("stat", "")),
            float(r["model_p"]),
        ),
        axis=1,
    )

    if not args.include_dk_market:
        n_mk = int((merged["line_tier"].astype(str).str.lower() == "market").sum())
        merged = merged[merged["line_tier"].astype(str).str.lower() != "market"].copy()
        logger.info("Excluded %d DraftKings (market) rows", n_mk)

    # Exclude coin flip
    merged = merged[merged["model_p"] != 0.5].copy()
    merged["pick_over"] = merged["model_p"] > 0.5

    over_df = merged[merged["pick_over"]].copy()
    over_df["confidence"] = over_df["model_p"].astype(float)
    over_df["hit"] = over_df["actual"].astype(float) > over_df[book_col].astype(float)

    if args.over_only:
        picks = over_df
    else:
        under_df = merged[~merged["pick_over"]].copy()
        allow = under_df.apply(
            lambda r: _unders_allowed(
                str(r["line_tier"]),
                str(r["player_type"]),
                str(r["stat"]),
            ),
            axis=1,
        )
        n_dropped = int((~allow).sum())
        if n_dropped:
            logger.info(
                "Excluded %d under picks (not offered for this tier/stat per _unders_allowed)",
                n_dropped,
            )
        under_df = under_df[allow].copy()
        under_df["confidence"] = 1.0 - under_df["model_p"].astype(float)
        under_df["hit"] = under_df["actual"].astype(float) < under_df[book_col].astype(float)
        picks = pd.concat([over_df, under_df], ignore_index=True)

    eligible = picks[picks["confidence"] >= args.min_confidence].copy()
    eligible["side"] = np.where(eligible["pick_over"], "Over", "Under")
    mode = "over only" if args.over_only else "over + allowed unders"
    logger.info(
        "%s: picks with confidence >= %.0f%%: %d (of %d candidate rows)",
        mode,
        args.min_confidence * 100,
        len(eligible),
        len(picks),
    )

    if eligible.empty:
        print("No picks meet the confidence threshold.")
        sys.exit(0)

    n = len(eligible)
    hits = int(eligible["hit"].sum())
    print()
    print("=" * 72)
    scope = "PP + DK market" if args.include_dk_market else "PrizePicks only (no DraftKings market)"
    subtitle = (
        f"P(over) >= {args.min_confidence:.0%} on overs"
        if args.over_only
        else f"min confidence {args.min_confidence:.0%} (P(over) / P(under) on respective sides)"
    )
    print(f"PROPS LAB HIT RATE - {scope}")
    print(f"  ({subtitle})")
    print("=" * 72)
    print(f"  n picks:     {n}")
    print(f"  hits:        {hits}")
    print(f"  hit rate:    {hits / n:.1%}")
    no = int(eligible["pick_over"].sum())
    nu = int((~eligible["pick_over"]).sum())
    print(f"  overs: {no}  |  unders: {nu}")
    print()

    _print_trust_lines(
        eligible,
        min_confidence=args.min_confidence,
        min_n=args.min_cell_n,
        include_under_block=not args.over_only,
    )

    if args.compact:
        sys.exit(0)

    _print_breakdown_by(
        eligible,
        "By side (Over vs Under)",
        "side",
        order=["Over", "Under"],
    )
    _print_breakdown_by(
        eligible,
        "By confidence tier (Lock / Strong / Lean / Pass)",
        "conf_tier",
        order=["Lock", "Strong", "Lean", "Pass"],
    )
    _print_breakdown_by(
        eligible,
        "By stat",
        "stat",
        order=sorted(eligible["stat"].astype(str).unique(), key=str),
    )

    _CONF_ORDER = ["Lock", "Strong", "Lean", "Pass"]
    _stat_u = sorted(eligible["stat"].astype(str).unique(), key=str)
    _print_crosstab(
        eligible,
        "By side x confidence tier",
        "side",
        "conf_tier",
        row_order=["Over", "Under"],
        col_order=[c for c in _CONF_ORDER if c in set(eligible["conf_tier"].unique())]
        + [c for c in sorted(eligible["conf_tier"].unique(), key=str) if c not in _CONF_ORDER],
    )
    _print_crosstab(
        eligible,
        "By side x stat",
        "side",
        "stat",
        row_order=["Over", "Under"],
        col_order=_stat_u,
    )
    _print_crosstab(
        eligible,
        "By confidence tier x stat",
        "conf_tier",
        "stat",
        row_order=[c for c in _CONF_ORDER if c in set(eligible["conf_tier"].unique())]
        + [c for c in sorted(eligible["conf_tier"].unique(), key=str) if c not in _CONF_ORDER],
        col_order=_stat_u,
    )

    _tier_order_pp = ("standard", "goblin", "demon", "market")
    _seen_lt = set(eligible["line_tier"].astype(str).unique())
    if _seen_lt:
        print("By PrizePicks / DK line tier (standard, goblin, demon; market only if --include-dk-market)")
        print("-" * 72)
        for lt in [x for x in _tier_order_pp if x in _seen_lt] + sorted(
            _seen_lt - set(_tier_order_pp), key=str
        ):
            sub = eligible[eligible["line_tier"].astype(str) == lt]
            print(f"  {lt:10s}  {_fmt_hit(sub)}")
        print()

    _print_three_way(
        eligible,
        f"By side x confidence tier x stat (cells with n >= {args.min_cell_n})",
        args.min_cell_n,
    )


if __name__ == "__main__":
    main()
