"""Shared configuration constants for the TDD Dashboard."""
from __future__ import annotations
from pathlib import Path

import yaml

from lib.theme import GOLD, EMBER, SAGE, SLATE, CREAM, DARK

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"
ICON_PATH = PROJECT_ROOT / "iconTransparent.png"

# ---------------------------------------------------------------------------
# Runtime config (loaded from config/runtime.yaml)
# ---------------------------------------------------------------------------
_runtime_path = PROJECT_ROOT / "config" / "runtime.yaml"
with open(_runtime_path, "r") as _f:
    RUNTIME: dict = yaml.safe_load(_f)

CURRENT_SEASON: int = RUNTIME["current_season"]           # e.g. 2026
PRIOR_SEASON: int = CURRENT_SEASON - 1                    # e.g. 2025
TRAIN_START: int = RUNTIME["train_start_season"]           # e.g. 2018
TRAIN_END: int = RUNTIME["train_end_season"]               # e.g. 2025
PROJECTION_LABEL: str = f"{CURRENT_SEASON} Projection"     # "2026 Projection"
TRAINING_RANGE: str = f"{TRAIN_START}-{TRAIN_END}"          # "2018-2025"

# Live schedule refresh
SCHEDULE_REFRESH_MINUTES: int = RUNTIME.get("schedule_refresh_interval_minutes", 10)
GAME_WINDOW_START_HOUR: int = RUNTIME.get("game_window_start_hour", 11)
GAME_WINDOW_END_HOUR: int = RUNTIME.get("game_window_end_hour", 24)

# Derived colors for dashboard dark theme
DARK_CARD = "#181b23"
DARK_BORDER = "#2a2e3a"
POSITIVE = SAGE
NEGATIVE = EMBER

# Season config
AVAILABLE_SEASONS = list(range(PRIOR_SEASON, TRAIN_START - 1, -1))  # [2025, ..., 2018]
UNRELIABLE_BB_SEASONS = {2018, 2019, 2020, 2021}

# Pitch display config
PITCH_DISPLAY: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "CU": "Curveball", "ST": "Sweeper",
    "CH": "Changeup", "FS": "Splitter", "KC": "Knuckle-Curve",
    "SV": "Slurve", "CS": "Slow Curve", "FO": "Forkball",
    "EP": "Eephus", "KN": "Knuckleball",
}
PITCH_ORDER: list[str] = [
    "FF", "SI", "FC",
    "SL", "ST", "CU", "KC", "SV", "CS",
    "CH", "FS", "FO",
    "EP", "KN",
]
PITCH_FAMILY_COLORS: dict[str, str] = {
    "fastball": "#E8575A",
    "breaking": "#5B9BD5",
    "offspeed": "#70AD47",
}
PITCH_TYPE_COLORS: dict[str, str] = {
    # Fastballs — warm tones, distinct from each other
    "FF": "#E8575A",   # 4-Seam: bright red
    "SI": "#D4862A",   # Sinker: burnt orange
    "FC": "#D98880",   # Cutter: salmon
    # Breaking — cool tones, distinct from each other
    "SL": "#5B9BD5",   # Slider: medium blue
    "ST": "#4EC6C1",   # Sweeper: teal
    "CU": "#8E7CC3",   # Curveball: purple
    "KC": "#A78BDA",   # Knuckle-Curve: lavender
    "SV": "#3A6FB0",   # Slurve: deep blue
    "CS": "#7B68AE",   # Slow Curve: dark violet
    # Offspeed — green/yellow tones
    "CH": "#70AD47",   # Changeup: green
    "FS": "#A8C256",   # Splitter: yellow-green
    "FO": "#8B9A46",   # Forkball: olive
    "EP": "#D4C75A",   # Eephus: gold-yellow
    "KN": "#5BA886",   # Knuckleball: teal-green
}
PITCH_TYPE_TO_FAMILY: dict[str, str] = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "breaking", "CU": "breaking", "ST": "breaking",
    "KC": "breaking", "SV": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed",
    "EP": "offspeed", "KN": "offspeed",
}

# Projected stat configs per player type (Bayesian model)
PITCHER_STATS = [
    ("K%", "k_rate", True, "Higher K% = more strikeout stuff"),
    ("BB%", "bb_rate", False, "Lower BB% = better control"),
]
HITTER_STATS = [
    ("K%", "k_rate", False, "Lower K% = better contact ability"),
    ("BB%", "bb_rate", True, "Higher BB% = better plate discipline"),
]

# Observed stats (no Bayesian projection — displayed as current percentiles)
HITTER_OBSERVED_STATS = [
    ("Whiff%", "whiff_rate", False, "Lower whiff rate = better contact"),
    ("Chase%", "chase_rate", False, "Lower chase rate = better discipline"),
    ("Z-Contact%", "z_contact_pct", True, "Zone contact rate"),
    ("Avg EV", "avg_exit_velo", True, "Average exit velocity (mph)"),
    ("Hard-Hit%", "hard_hit_pct", True, "Exit velo >= 95 mph rate"),
    ("Sprint Speed", "sprint_speed", True, "Baserunning speed (ft/s)"),
    ("FB%", "fb_pct", True, "Fly ball rate"),
]
PITCHER_OBSERVED_STATS = [
    ("Whiff%", "whiff_rate", True, "Higher whiff rate = better stuff"),
    ("Avg Velo", "avg_velo", True, "Average fastball velocity (mph)"),
    ("Extension", "release_extension", True, "Release extension (ft)"),
    ("Zone%", "zone_pct", True, "Pitch in-zone rate"),
    ("GB%", "gb_pct", True, "Ground ball rate"),
]

# Counting stat display configs: (label, column_prefix, actual_col, higher_better)
# Ordered by validated accuracy (MAPE): most reliable stats first
HITTER_COUNTING_DISPLAY = [
    ("wRC+", "projected_wrc_plus", None, True),        # MAPE=11%, r=0.739
    ("Proj. K", "total_k", "actual_k", False),         # MAPE=25%, r=0.783
    ("Proj. HR", "total_hr", "actual_hr", True),       # MAPE=28%, r=0.936
    ("Proj. BB", "total_bb", "actual_bb", True),       # MAPE=30%, r=0.871
    ("Proj. R", "total_r", None, True),                # MAPE=30%, r=0.786
    ("Proj. RBI", "total_rbi", None, True),            # MAPE=30%, r=0.817
    ("Proj. SB", "total_sb", None, True),              # r=0.937
]
PITCHER_COUNTING_DISPLAY = [
    ("FIP-ERA", "projected_fip_era", None, False),     # best rate metric, r=0.297
    ("Proj. K", "total_k", "actual_k", True),          # MAPE=47%, r=0.777
    ("Proj. IP", "projected_ip", None, True),          # MAPE=47%, r=0.769
    ("Proj. BB", "total_bb", "actual_bb", False),      # MAPE=48%, r=0.692
    ("Proj. SV", "total_sv", None, True),              # role-specific
    ("Proj. HLD", "total_hld", None, True),            # role-specific
]

# Traditional stat display configs
HITTER_TRAD_STATS = [
    ("AVG", "avg", True, ".000"),
    ("OBP", "obp", True, ".000"),
    ("SLG", "slg", True, ".000"),
    ("OPS", "ops", True, ".000"),
    ("ISO", "iso", True, ".000"),
    ("wOBA", "woba", True, ".000"),
    ("BABIP", "babip", True, ".000"),
]
HITTER_TRAD_COUNTING = [
    ("G", "games"), ("PA", "pa"), ("AB", "ab"), ("H", "hits"),
    ("2B", "doubles"), ("3B", "triples"), ("HR", "hr"),
    ("R", "runs"), ("RBI", "rbi"), ("BB", "bb"), ("K", "k"),
    ("SB", "sb"), ("CS", "cs"),
]
PITCHER_TRAD_STATS = [
    ("ERA", "era", False, "0.00"),
    ("FIP", "fip", False, "0.00"),
    ("WHIP", "whip", False, "0.00"),
    ("K/9", "k_per_9", True, "0.00"),
    ("BB/9", "bb_per_9", False, "0.00"),
    ("HR/9", "hr_per_9", False, "0.00"),
    ("K/BB", "k_per_bb", True, "0.00"),
    ("GO/AO", "go_ao", True, "0.00"),
]
PITCHER_TRAD_COUNTING = [
    ("G", "games"), ("GS", "starts"), ("W", "w"), ("L", "l"),
    ("SV", "sv"), ("HLD", "hld"), ("IP", "ip"), ("BF", "bf"),
    ("H", "hits_allowed"), ("ER", "er"), ("HR", "hr_allowed"),
    ("K", "k"), ("BB", "bb"), ("HBP", "hbp"),
]

# Scouting report helpers
STAT_NAMES = {
    "k_rate": "strikeout rate",
    "bb_rate": "walk rate",
}
GOOD_DIRECTION_LABEL = {
    ("k_rate", True): "miss more bats",
    ("k_rate", False): "make more contact",
    ("bb_rate", True): "draw more walks",
    ("bb_rate", False): "improve control",
}
