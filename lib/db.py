"""
Database connection helper for the dashboard.

Provides read_sql() for scripts that need to query the database
(e.g., update_in_season.py for fetching observed 2026 totals).

Synced from: player_profiles/src/data/db.py (subset)
"""
from __future__ import annotations

import os
import logging

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# Database connection — reads from environment or falls back to defaults
DB_USER = os.getenv("DB_USER", "kekoa")
DB_PASS = os.getenv("DB_PASS", "goatez")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "mlb_fantasy")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

_engine = None


def get_engine():
    """Get or create a SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame.

    Parameters
    ----------
    query : str
        SQL query string (can use :param_name for parameters).
    params : dict, optional
        Query parameters.

    Returns
    -------
    pd.DataFrame
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = pd.read_sql(text(query), conn, params=params)
    return result
