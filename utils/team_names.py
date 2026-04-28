"""Shared team abbreviation-to-name mapping.

Single source of truth so every page shows consistent full team names
instead of raw abbreviations.
"""
from __future__ import annotations

# abbr -> (full name, short name)
# full = "Los Angeles Dodgers", short = "Dodgers"
_TEAMS: dict[str, tuple[str, str]] = {
    "ARI": ("Arizona Diamondbacks", "D-backs"),
    "ATH": ("Athletics", "Athletics"),
    "ATL": ("Atlanta Braves", "Braves"),
    "AZ":  ("Arizona Diamondbacks", "D-backs"),
    "BAL": ("Baltimore Orioles", "Orioles"),
    "BOS": ("Boston Red Sox", "Red Sox"),
    "CHC": ("Chicago Cubs", "Cubs"),
    "CIN": ("Cincinnati Reds", "Reds"),
    "CLE": ("Cleveland Guardians", "Guardians"),
    "COL": ("Colorado Rockies", "Rockies"),
    "CWS": ("Chicago White Sox", "White Sox"),
    "DET": ("Detroit Tigers", "Tigers"),
    "HOU": ("Houston Astros", "Astros"),
    "KC":  ("Kansas City Royals", "Royals"),
    "LAA": ("Los Angeles Angels", "Angels"),
    "LAD": ("Los Angeles Dodgers", "Dodgers"),
    "MIA": ("Miami Marlins", "Marlins"),
    "MIL": ("Milwaukee Brewers", "Brewers"),
    "MIN": ("Minnesota Twins", "Twins"),
    "NYM": ("New York Mets", "Mets"),
    "NYY": ("New York Yankees", "Yankees"),
    "OAK": ("Oakland Athletics", "Athletics"),
    "PHI": ("Philadelphia Phillies", "Phillies"),
    "PIT": ("Pittsburgh Pirates", "Pirates"),
    "SD":  ("San Diego Padres", "Padres"),
    "SEA": ("Seattle Mariners", "Mariners"),
    "SF":  ("San Francisco Giants", "Giants"),
    "STL": ("St. Louis Cardinals", "Cardinals"),
    "TB":  ("Tampa Bay Rays", "Rays"),
    "TEX": ("Texas Rangers", "Rangers"),
    "TOR": ("Toronto Blue Jays", "Blue Jays"),
    "WSH": ("Washington Nationals", "Nationals"),
}


def team_full(abbr: str) -> str:
    """Return full name like 'Los Angeles Dodgers'. Falls back to abbr."""
    entry = _TEAMS.get(abbr)
    return entry[0] if entry else abbr


def team_short(abbr: str) -> str:
    """Return short name like 'Dodgers'. Falls back to abbr."""
    entry = _TEAMS.get(abbr)
    return entry[1] if entry else abbr


def abbr_from_full(full_name: str) -> str:
    """Reverse lookup: 'Los Angeles Dodgers' -> 'LAD'. Falls back to input."""
    for abbr, (full, _short) in _TEAMS.items():
        if full == full_name:
            return abbr
    return full_name


def team_select_options(abbrs: list[str]) -> list[str]:
    """Convert a list of abbreviations to full names for selectbox display."""
    return [team_full(a) for a in abbrs]


def team_select_to_abbr(selected_full: str) -> str:
    """Convert a selectbox full-name selection back to abbreviation."""
    return abbr_from_full(selected_full)
