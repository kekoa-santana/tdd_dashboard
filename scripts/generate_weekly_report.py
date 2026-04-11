#!/usr/bin/env python
"""
Generate a weekly report summary for The Data Diamond newsletter.

Produces a clean markdown file with projection risers/fallers,
breakout candidates, Diamond Rating leaders, and placeholders for
in-season model accuracy and matchup highlights.

Usage
-----
    python scripts/generate_weekly_report.py                    # today's date
    python scripts/generate_weekly_report.py --date 2026-04-15  # specific date
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import CURRENT_SEASON  # noqa: E402
from lib.diamond_rating import (  # noqa: E402
    diamond_tier,
    score_to_diamonds,
)

log = logging.getLogger("weekly_report")

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(name: str) -> pd.DataFrame | None:
    """Load a parquet from DASHBOARD_DIR, returning None if missing."""
    path = DASHBOARD_DIR / name
    if not path.exists():
        log.warning("Missing parquet: %s", path)
        return None
    return pd.read_parquet(path)


def _fmt_diamonds(rating: float) -> str:
    """Format diamond rating as e.g. '4.2 (Above Average)'."""
    return f"{rating:.1f} ({diamond_tier(rating)})"


def _signed(val: float, fmt: str = ".1f") -> str:
    """Format a signed delta, e.g. '+2.3' or '-1.1'."""
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:{fmt}}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_risers_fallers(lines: list[str]) -> None:
    """Top 10 risers and fallers by composite_score shift vs preseason."""
    lines.append("## Projection Movers")
    lines.append("")

    # --- Pitchers ---
    p_curr = _load_parquet("pitcher_projections.parquet")
    p_snap = _load_parquet(str(Path("snapshots") / f"pitcher_projections_{CURRENT_SEASON}_preseason.parquet"))

    # --- Hitters ---
    h_curr = _load_parquet("hitter_projections.parquet")
    h_snap = _load_parquet(str(Path("snapshots") / f"hitter_projections_{CURRENT_SEASON}_preseason.parquet"))

    movers: list[tuple[str, str, float, float, float]] = []  # (name, type, current, preseason, delta)

    if p_curr is not None and p_snap is not None:
        merged = p_curr[["pitcher_id", "pitcher_name", "composite_score"]].merge(
            p_snap[["pitcher_id", "composite_score"]],
            on="pitcher_id",
            suffixes=("_curr", "_pre"),
        )
        merged["delta"] = merged["composite_score_curr"] - merged["composite_score_pre"]
        for _, row in merged.iterrows():
            movers.append((
                row["pitcher_name"], "P",
                row["composite_score_curr"], row["composite_score_pre"], row["delta"],
            ))

    if h_curr is not None and h_snap is not None:
        merged = h_curr[["batter_id", "batter_name", "composite_score"]].merge(
            h_snap[["batter_id", "composite_score"]],
            on="batter_id",
            suffixes=("_curr", "_pre"),
        )
        merged["delta"] = merged["composite_score_curr"] - merged["composite_score_pre"]
        for _, row in merged.iterrows():
            movers.append((
                row["batter_name"], "H",
                row["composite_score_curr"], row["composite_score_pre"], row["delta"],
            ))

    if not movers:
        lines.append("*No snapshot data available for comparison.*")
        lines.append("")
        return

    movers_df = pd.DataFrame(movers, columns=["Name", "Type", "Current", "Preseason", "Delta"])

    # Top 10 Risers
    risers = movers_df.nlargest(10, "Delta")
    lines.append("### Top 10 Risers")
    lines.append("")
    lines.append("| Rank | Player | Type | Composite Change |")
    lines.append("|-----:|--------|:----:|-----------------:|")
    for i, (_, row) in enumerate(risers.iterrows(), 1):
        lines.append(
            f"| {i} | {row['Name']} | {row['Type']} | {_signed(row['Delta'])} |"
        )
    lines.append("")

    # Top 10 Fallers
    fallers = movers_df.nsmallest(10, "Delta")
    lines.append("### Top 10 Fallers")
    lines.append("")
    lines.append("| Rank | Player | Type | Composite Change |")
    lines.append("|-----:|--------|:----:|-----------------:|")
    for i, (_, row) in enumerate(fallers.iterrows(), 1):
        lines.append(
            f"| {i} | {row['Name']} | {row['Type']} | {_signed(row['Delta'])} |"
        )
    lines.append("")


def _section_breakout_candidates(lines: list[str]) -> None:
    """Top 5 hitter and pitcher breakout candidates."""
    lines.append("## Breakout Candidates")
    lines.append("")

    # Hitter breakouts
    h_bo = _load_parquet("hitter_breakout_candidates.parquet")
    if h_bo is not None and not h_bo.empty:
        top = h_bo.nlargest(5, "breakout_score")
        lines.append("### Hitters")
        lines.append("")
        lines.append("| Rank | Player | Age | Breakout Type | Score | Tier |")
        lines.append("|-----:|--------|----:|:--------------|------:|:-----|")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"| {i} | {row['batter_name']} | {row['age']} "
                f"| {row['breakout_type']} | {row['breakout_score']:.3f} "
                f"| {row['breakout_tier']} |"
            )
        lines.append("")
        # Add narrative for top candidate
        best = top.iloc[0]
        lines.append(f"> **{best['batter_name']}:** {best['breakout_narrative']}")
        lines.append("")
    else:
        lines.append("*No hitter breakout data available.*")
        lines.append("")

    # Pitcher breakouts
    p_bo = _load_parquet("pitcher_breakout_candidates.parquet")
    if p_bo is not None and not p_bo.empty:
        top = p_bo.nlargest(5, "breakout_score")
        lines.append("### Pitchers")
        lines.append("")
        lines.append("| Rank | Player | Age | Breakout Type | Score | Tier |")
        lines.append("|-----:|--------|----:|:--------------|------:|:-----|")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            role = "SP" if row.get("is_starter") else "RP"
            lines.append(
                f"| {i} | {row['pitcher_name']} ({role}) | {row['age']} "
                f"| {row['breakout_type']} | {row['breakout_score']:.3f} "
                f"| {row['breakout_tier']} |"
            )
        lines.append("")
        best = top.iloc[0]
        lines.append(f"> **{best['pitcher_name']}:** {best['breakout_narrative']}")
        lines.append("")
    else:
        lines.append("*No pitcher breakout data available.*")
        lines.append("")


def _section_diamond_leaders(lines: list[str]) -> None:
    """Top 10 hitters and pitchers by Diamond Rating."""
    lines.append("## Diamond Rating Leaders")
    lines.append("")

    # Hitter leaders
    h_rank = _load_parquet("hitters_rankings.parquet")
    if h_rank is not None and not h_rank.empty:
        top = h_rank.nlargest(10, "tdd_value_score")
        lines.append("### Hitters")
        lines.append("")
        lines.append("| Rank | Player | Pos | Age | Diamond Rating | Value Score |")
        lines.append("|-----:|--------|:----|----:|:--------------:|------------:|")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            rating = score_to_diamonds(row["tdd_value_score"])
            lines.append(
                f"| {i} | {row['batter_name']} | {row['position']} | {row['age']} "
                f"| {_fmt_diamonds(rating)} | {row['tdd_value_score']:.3f} |"
            )
        lines.append("")
    else:
        lines.append("*No hitter rankings data available.*")
        lines.append("")

    # Pitcher leaders
    p_rank = _load_parquet("pitchers_rankings.parquet")
    if p_rank is not None and not p_rank.empty:
        top = p_rank.nlargest(10, "tdd_value_score")
        lines.append("### Pitchers")
        lines.append("")
        lines.append("| Rank | Player | Role | Age | Diamond Rating | Value Score |")
        lines.append("|-----:|--------|:-----|----:|:--------------:|------------:|")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            rating = score_to_diamonds(row["tdd_value_score"])
            lines.append(
                f"| {i} | {row['pitcher_name']} | {row['role']} | {row['age']} "
                f"| {_fmt_diamonds(rating)} | {row['tdd_value_score']:.3f} |"
            )
        lines.append("")
    else:
        lines.append("*No pitcher rankings data available.*")
        lines.append("")


def _section_model_accuracy(lines: list[str]) -> None:
    """Model accuracy summary (placeholder until season starts)."""
    lines.append("## Model Accuracy")
    lines.append("")
    lines.append("*Tracking begins Opening Day. This section will include:*")
    lines.append("")
    lines.append("- Pitcher K% predicted vs actual (MAE, calibration)")
    lines.append("- Hitter K%/BB% predicted vs actual")
    lines.append("- Game K simulator hit rate on over/under lines")
    lines.append("- Biggest hits and misses of the week")
    lines.append("")


def _section_premium_matchups(lines: list[str]) -> None:
    """Upcoming premium matchups (placeholder until season starts)."""
    lines.append("## Premium Matchups This Week")
    lines.append("")
    lines.append("*Schedule-based matchup highlights will appear once the season begins.*")
    lines.append("")
    lines.append("This section will feature:")
    lines.append("")
    lines.append("- Top-rated pitcher vs lineup matchups")
    lines.append("- High-K-probability game stacks")
    lines.append("- Breakout candidate spotlight starts")
    lines.append("")


# ---------------------------------------------------------------------------
# Main report assembly
# ---------------------------------------------------------------------------

def generate_report(report_date: date | None = None) -> str:
    """Generate full markdown report content."""
    report_date = report_date or date.today()
    week_start = report_date - timedelta(days=report_date.weekday())  # Monday
    week_end = week_start + timedelta(days=6)  # Sunday

    lines: list[str] = []

    # Header
    lines.append("# The Data Diamond -- Weekly Report")
    lines.append("")
    lines.append(
        f"**Week of {week_start.strftime('%B %d')} - "
        f"{week_end.strftime('%B %d, %Y')}**"
    )
    lines.append("")
    lines.append(f"*Generated {report_date.strftime('%Y-%m-%d')} "
                 f"| {CURRENT_SEASON} Season Projections*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sections
    _section_risers_fallers(lines)
    _section_breakout_candidates(lines)
    _section_diamond_leaders(lines)
    _section_model_accuracy(lines)
    _section_premium_matchups(lines)

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by The Data Diamond projection system. "
                 "All projections are Bayesian posterior estimates.*")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate a weekly TDD report in markdown."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Report date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    report_date = date.fromisoformat(args.date) if args.date else date.today()
    content = generate_report(report_date)

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"weekly_{report_date.isoformat()}.md"
    output_path.write_text(content, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
