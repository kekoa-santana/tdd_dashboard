#!/usr/bin/env python
"""Generate a daily game preview report.

Usage:
    python scripts/generate_daily_preview.py
    python scripts/generate_daily_preview.py --date 2026-04-09
    python scripts/generate_daily_preview.py --stdout
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.fantasy_report import generate_daily_report

DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate daily game preview report",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Report date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing a file.",
    )
    args = parser.parse_args()

    report_date = args.date or date.today().isoformat()

    content = generate_daily_report(DASHBOARD_DIR, report_date)

    if args.stdout:
        print(content)
    else:
        output_dir = PROJECT_ROOT / "reports"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"daily_preview_{report_date}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
