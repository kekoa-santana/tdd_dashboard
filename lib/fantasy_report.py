"""
Daily game preview engine.

Generates game-by-game scouting narratives from pre-computed parquet data.
Each game produces bullet points covering:
  - Pitcher struggles against this lineup
  - Pitcher advantages over this lineup
  - Key batters (hitter-favored matchups)
  - Struggling batters (pitcher-favored matchups)
  - Bullpen impact (help or hurt)

Used by:
  - Daily Preview dashboard page (full slate overview)
  - Schedule page game drilldown (per-game scouting report)
  - CLI script (scripts/generate_daily_preview.py)

Uses varied, baseball-vernacular templates driven by the underlying data
signals rather than monotonous stat dumps.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

PITCH_DISPLAY: dict[str, str] = {
    "FF": "4-seam", "SI": "sinker", "FC": "cutter",
    "SL": "slider", "CU": "curveball", "ST": "sweeper",
    "CH": "changeup", "FS": "splitter", "KC": "knuckle-curve",
    "SV": "slurve", "CS": "slow curve", "FO": "forkball",
    "EP": "eephus", "KN": "knuckleball",
}

# ---------------------------------------------------------------------------
# Vocabulary tiers - data-driven phrasing
# ---------------------------------------------------------------------------

def _whiff_adj(rate: float) -> str:
    """Describe a pitch's whiff quality."""
    if rate >= 0.38:
        return random.choice(["wipeout", "devastating", "unhittable", "filthy"])
    if rate >= 0.30:
        return random.choice(["nasty", "sharp", "swing-and-miss", "putaway"])
    if rate >= 0.24:
        return random.choice(["solid", "respectable", "decent"])
    if rate >= 0.18:
        return random.choice(["hittable", "below-average", "mediocre"])
    return random.choice(["flat", "get-me-over", "meatball-prone"])


def _k_rate_desc(rate: float) -> str:
    """Describe a pitcher's K rate in context."""
    if rate >= 0.30:
        return random.choice([
            "an elite strikeout artist",
            "one of the game's best at generating whiffs",
            "a premier K-rate arm",
        ])
    if rate >= 0.27:
        return random.choice([
            "a high-strikeout arm",
            "a pitcher who racks up punchouts",
            "a strong K-rate pitcher",
        ])
    if rate >= 0.23:
        return random.choice([
            "a solid strikeout pitcher",
            "a pitcher with average swing-and-miss stuff",
        ])
    if rate >= 0.19:
        return random.choice([
            "a below-average strikeout pitcher",
            "a contact-oriented arm",
            "a pitcher who relies on weak contact over whiffs",
        ])
    return random.choice([
        "a low-K arm who needs his defense",
        "a pitcher with minimal swing-and-miss",
    ])


def _control_desc(bb_rate: float) -> str:
    """Describe a pitcher's walk rate."""
    if bb_rate <= 0.055:
        return random.choice([
            "elite control", "pinpoint command",
            "rarely issues free passes",
        ])
    if bb_rate <= 0.075:
        return random.choice([
            "solid command", "above-average control",
            "keeps the walks in check",
        ])
    if bb_rate <= 0.090:
        return random.choice(["average control", "manageable walk rate"])
    if bb_rate <= 0.105:
        return random.choice([
            "shaky command", "walk-prone tendencies",
            "below-average control",
        ])
    return random.choice([
        "poor command", "a walk machine",
        "liable to put runners on via free pass",
    ])


def _damage_desc(xwoba: float) -> str:
    """Describe hitter contact quality."""
    if xwoba >= 0.420:
        return random.choice([
            "does serious damage on contact",
            "squares it up hard",
            "barrels everything he touches",
        ])
    if xwoba >= 0.370:
        return random.choice([
            "hits the ball hard",
            "makes quality contact",
            "puts a charge into it",
        ])
    if xwoba >= 0.320:
        return random.choice(["makes average contact", "puts the ball in play"])
    return random.choice([
        "makes weak contact", "doesn't do much damage when he connects",
    ])


def _lineup_k_tendency(avg_k_rate: float) -> str:
    """Describe a lineup's collective K tendency."""
    if avg_k_rate >= 0.26:
        return random.choice([
            "a free-swinging lineup", "a strikeout-heavy group",
            "a lineup full of whiff-prone bats",
        ])
    if avg_k_rate >= 0.23:
        return random.choice([
            "a lineup with average contact ability",
            "a group that strikes out at a normal clip",
        ])
    return random.choice([
        "a tough lineup to punch out",
        "a disciplined group that puts the ball in play",
        "a contact-heavy lineup",
    ])


def _bullpen_quality(k_rate: float) -> str:
    """Describe bullpen K quality."""
    if k_rate >= 0.27:
        return random.choice([
            "a dominant bullpen", "a shutdown pen",
            "a lights-out relief corps",
        ])
    if k_rate >= 0.24:
        return random.choice([
            "a solid bullpen", "a reliable pen",
            "a dependable relief unit",
        ])
    if k_rate >= 0.21:
        return random.choice([
            "an average bullpen", "a league-average pen",
        ])
    return random.choice([
        "a weak bullpen", "a shaky pen",
        "a below-average relief corps",
    ])


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PitcherReport:
    pitcher_name: str
    pitcher_id: int
    team_abbr: str
    opp_abbr: str
    side: str
    projected_k_rate: float = 0.0
    projected_bb_rate: float = 0.0
    expected_k: float = 0.0
    expected_ip: float = 0.0
    expected_bb: float = 0.0
    expected_hr: float = 0.0
    dk_mean: float = 0.0
    archetype: str = ""
    struggles: list[str] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    key_batters: list[str] = field(default_factory=list)
    struggling_batters: list[str] = field(default_factory=list)
    bullpen_notes: list[str] = field(default_factory=list)
    batting_outlook: list[str] = field(default_factory=list)
    has_lineup: bool = False


@dataclass
class GameReport:
    game_pk: int
    game_time: str
    away_abbr: str
    home_abbr: str
    away: PitcherReport | None = None
    home: PitcherReport | None = None


# ---------------------------------------------------------------------------
# Data loading (for standalone script use -- no Streamlit)
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    games: pd.DataFrame
    sims: pd.DataFrame
    lineups: pd.DataFrame
    batter_sims: pd.DataFrame
    pitcher_proj: pd.DataFrame
    hitter_proj: pd.DataFrame
    arsenal: pd.DataFrame
    vuln: pd.DataFrame
    hitter_str: pd.DataFrame
    platoon: pd.DataFrame
    team_bullpen: pd.DataFrame
    team_profiles: pd.DataFrame
    player_teams: pd.DataFrame
    pitcher_arch: pd.DataFrame
    baselines_pt: dict[str, dict[str, float]]


def _build_sims_from_game_props(
    game_props: pd.DataFrame,
    game_date: str | None = None,
) -> pd.DataFrame:
    """Pivot game_props pitcher rows into a sims-like DataFrame.

    game_props has one row per (player, stat). This function pivots the
    stat dimension into columns so downstream code can access a single
    row per pitcher with expected_k, expected_bb, etc.

    Parameters
    ----------
    game_props : pd.DataFrame
        Full game_props.parquet (all dates).
    game_date : str, optional
        Filter to this date. If None, uses the latest date.

    Returns
    -------
    pd.DataFrame
        One row per pitcher appearance with columns matching the old
        todays_sims schema: pitcher_id, game_pk, team_abbr, opp_abbr,
        expected_k, expected_bb, expected_hr, expected_ip, dk_mean,
        projected_k_rate, avg_matchup_lift, has_lineup.
    """
    if game_props.empty:
        return pd.DataFrame()

    if game_date is None:
        game_date = game_props["game_date"].max()

    pitchers = game_props[
        (game_props["game_date"] == game_date)
        & (game_props["player_type"] == "pitcher")
    ].copy()

    if pitchers.empty:
        return pd.DataFrame()

    # Pivot stat → expected value
    stat_pivot = pitchers.pivot_table(
        index="player_id", columns="stat", values="expected", aggfunc="first",
    )
    stat_map = {"K": "expected_k", "BB": "expected_bb", "HR": "expected_hr",
                "H": "expected_h", "Outs": "expected_outs"}
    stat_pivot = stat_pivot.rename(columns=stat_map)

    # Get shared pitcher-level columns from the first stat row per pitcher
    meta = pitchers.drop_duplicates("player_id").set_index("player_id")[
        ["player_name", "game_pk", "team", "opponent", "expected_ip",
         "expected_bf", "dk_mean", "side"]
    ].rename(columns={"team": "team_abbr", "opponent": "opp_abbr",
                       "player_name": "pitcher_name"})

    if "dk_mean" not in meta.columns:
        meta["dk_mean"] = 0.0

    result = meta.join(stat_pivot, how="left").reset_index().rename(
        columns={"player_id": "pitcher_id"},
    )

    # Derive projected_k_rate from expected_k / expected_bf
    if "expected_k" in result.columns and "expected_bf" in result.columns:
        bf = result["expected_bf"].fillna(22.0).clip(lower=1)
        result["projected_k_rate"] = result["expected_k"].fillna(0) / bf
    else:
        result["projected_k_rate"] = 0.0

    # Columns not available in game_props — safe defaults
    result["avg_matchup_lift"] = 0.0
    result["has_lineup"] = True  # precompute always has lineup context

    return result


def load_report_data(dashboard_dir: Path) -> ReportData:
    """Load all parquets needed for the daily report."""
    def _load(name: str) -> pd.DataFrame:
        p = dashboard_dir / name
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    baselines_pt = {
        pt: {
            "whiff_rate": v.get("whiff_rate", 0.25),
            "chase_rate": v.get("chase_rate", 0.30),
            "barrel_rate": v.get("barrel_rate", 0.06),
        }
        for pt, v in LEAGUE_AVG_BY_PITCH_TYPE.items()
    }

    # Build sims from game_props (single source of truth).
    # Use the date from todays_games so sims align with batter_sims/lineups
    # (game_props may contain future dates that don't match).
    game_props = _load("game_props.parquet")
    games = _load("todays_games.parquet")
    _today_date: str | None = None
    if not games.empty and "game_date" in games.columns:
        _today_date = str(games["game_date"].iloc[0])
    sims = _build_sims_from_game_props(game_props, game_date=_today_date)

    return ReportData(
        games=_load("todays_games.parquet"),
        sims=sims,
        lineups=_load("todays_lineups.parquet"),
        batter_sims=_load("todays_batter_sims.parquet"),
        pitcher_proj=_load("pitcher_projections.parquet"),
        hitter_proj=_load("hitter_projections.parquet"),
        arsenal=_load("pitcher_arsenal.parquet"),
        vuln=_load("hitter_vuln_career.parquet"),
        hitter_str=_load("hitter_str_career.parquet"),
        platoon=_load("batter_platoon_splits.parquet"),
        team_bullpen=_load("team_bullpen_rates.parquet"),
        team_profiles=_load("team_profiles.parquet"),
        player_teams=_load("player_teams.parquet"),
        pitcher_arch=_load("pitcher_archetypes.parquet"),
        baselines_pt=baselines_pt,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pitcher_top_pitches(
    pitcher_id: int, arsenal: pd.DataFrame, n: int = 3,
    min_usage: float = 0.10,
) -> pd.DataFrame:
    """Return a pitcher's top N pitches by usage (above *min_usage*)."""
    p = arsenal[arsenal["pitcher_id"] == pitcher_id].copy()
    if "pitches" in p.columns:
        p = p[p["pitches"] >= 20]
    if "usage_pct" in p.columns:
        p = p[p["usage_pct"] >= min_usage]
    return p.sort_values("usage_pct", ascending=False).head(n)


def _lineup_whiff_vs_pitch(
    pitch_type: str,
    batter_ids: list[int],
    vuln: pd.DataFrame,
) -> float | None:
    """Average whiff rate of a lineup against a specific pitch type."""
    v = vuln[
        (vuln["pitch_type"] == pitch_type)
        & (vuln["batter_id"].isin(batter_ids))
    ]
    if v.empty or "swings" not in v.columns:
        return None
    total_sw = v["swings"].sum()
    total_wh = v["whiffs"].sum()
    if total_sw < 20:
        return None
    return total_wh / total_sw


def _lineup_chase_vs_pitch(
    pitch_type: str,
    batter_ids: list[int],
    vuln: pd.DataFrame,
) -> float | None:
    """Average chase rate of a lineup against a specific pitch type."""
    v = vuln[
        (vuln["pitch_type"] == pitch_type)
        & (vuln["batter_id"].isin(batter_ids))
    ]
    if v.empty or "out_of_zone_pitches" not in v.columns:
        return None
    oz = v["out_of_zone_pitches"].sum()
    cs = v["chase_swings"].sum() if "chase_swings" in v.columns else 0
    if oz < 15:
        return None
    return cs / oz


def _lineup_xwoba_vs_pitch(
    pitch_type: str,
    batter_ids: list[int],
    hitter_str: pd.DataFrame,
) -> float | None:
    """Average xwOBA on contact for a lineup against a pitch type."""
    s = hitter_str[
        (hitter_str["pitch_type"] == pitch_type)
        & (hitter_str["batter_id"].isin(batter_ids))
    ]
    if s.empty or "xwoba_contact" not in s.columns:
        return None
    vals = s["xwoba_contact"].dropna()
    if len(vals) < 3:
        return None
    return vals.mean()


def _get_bullpen_rates(
    team_abbr: str,
    data: ReportData,
    current_season: int = 2025,
) -> dict[str, float] | None:
    """Get bullpen K/BB/HR rates for a team."""
    # Map team_abbr to team_id via player_teams or team_profiles
    bp = data.team_bullpen
    if bp.empty:
        return None

    # Find team_id from team_profiles (column is 'abbreviation')
    tp = data.team_profiles
    abbr_col = "team_abbr" if "team_abbr" in tp.columns else "abbreviation"
    if not tp.empty and abbr_col in tp.columns:
        team_row = tp[tp[abbr_col] == team_abbr]
        if not team_row.empty and "team_id" in team_row.columns:
            tid = int(team_row["team_id"].iloc[0])
            rates = bp[
                (bp["team_id"] == tid) & (bp["season"] == current_season)
            ]
            if rates.empty:
                rates = bp[
                    (bp["team_id"] == tid)
                    & (bp["season"] == current_season - 1)
                ]
            if not rates.empty:
                r = rates.iloc[0]
                return {
                    "k_rate": float(r["k_rate"]),
                    "bb_rate": float(r["bb_rate"]),
                    "hr_rate": float(r["hr_rate"]),
                }
    return None


def _get_bullpen_score(team_abbr: str, data: ReportData) -> float | None:
    """Get bullpen_score from team_profiles (0-1 scale)."""
    tp = data.team_profiles
    abbr_col = "team_abbr" if "team_abbr" in tp.columns else "abbreviation"
    if tp.empty or abbr_col not in tp.columns:
        return None
    row = tp[tp[abbr_col] == team_abbr]
    if row.empty or "bullpen_score" not in row.columns:
        return None
    val = row["bullpen_score"].iloc[0]
    return float(val) if pd.notna(val) else None


# ---------------------------------------------------------------------------
# Bullet generators
# ---------------------------------------------------------------------------

def _generate_struggles(
    sim_row: pd.Series,
    batter_sims: pd.DataFrame,
    top_pitches: pd.DataFrame,
    batter_ids: list[int],
    data: ReportData,
    pitcher_name: str,
) -> list[str]:
    """Generate bullets about what the pitcher will struggle with."""
    bullets: list[str] = []
    k_rate = sim_row.get("projected_k_rate", 0.0)
    bb_rate_proj = 0.0

    # Get projected BB rate from pitcher projections
    pid = sim_row.get("pitcher_id")
    pp = data.pitcher_proj
    if not pp.empty and pid is not None:
        pr = pp[pp["pitcher_id"] == pid]
        if not pr.empty:
            bb_rate_proj = float(pr["projected_bb_rate"].iloc[0] or 0.0)

    # 1. Low K rate limits upside
    if k_rate < 0.20:
        templates = [
            f"{pitcher_name} is {_k_rate_desc(k_rate)} ({k_rate:.1%} K rate) "
            f"-- don't count on a big strikeout number",
            f"Limited swing-and-miss stuff ({k_rate:.1%} projected K rate) "
            f"caps his upside today",
            f"With a {k_rate:.1%} K rate, {pitcher_name} needs weak contact "
            f"and defense to get through lineups",
        ]
        bullets.append(random.choice(templates))

    # 2. Negative matchup lift -- lineup resists Ks
    avg_lift = sim_row.get("avg_matchup_lift", 0.0)
    if pd.notna(avg_lift) and avg_lift < -0.02:
        templates = [
            f"This {sim_row.get('opp_abbr', '')} lineup matches up well "
            f"against his arsenal -- expect fewer Ks than the projection suggests",
            f"The matchup data is unfavorable here -- this lineup resists "
            f"his pitch mix better than most",
            f"Tough draw: this lineup's pitch-type strengths neutralize "
            f"what {pitcher_name} does best",
        ]
        bullets.append(random.choice(templates))

    # 3. Walk-prone
    if bb_rate_proj > 0.090:
        exp_bb = sim_row.get("expected_bb", 0)
        templates = [
            f"Control is a question mark ({bb_rate_proj:.1%} BB rate) "
            f"-- free baserunners will inflate pitch count and shorten the outing",
            f"Command is a concern at {bb_rate_proj:.1%} BB rate -- "
            f"walks will put runners on and drive up his pitch count",
            f"{pitcher_name} has {_control_desc(bb_rate_proj)} and projects "
            f"for {exp_bb:.0f} walks today",
        ]
        bullets.append(random.choice(templates))

    # 4. Primary pitch gets neutralized by lineup
    if not top_pitches.empty and batter_ids:
        top_pitch = top_pitches.iloc[0]
        pt = top_pitch["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        usage = top_pitch.get("usage_pct", 0)
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)

        lu_whiff = _lineup_whiff_vs_pitch(pt, batter_ids, data.vuln)
        if lu_whiff is not None and lu_whiff < lg_whiff * 0.85 and usage > 0.25:
            templates = [
                f"His go-to {pt_name} ({usage:.0%} usage) won't get the "
                f"whiffs he's used to -- this lineup only chases it "
                f"{lu_whiff:.0%} of the time vs {lg_whiff:.0%} league average",
                f"The {pt_name} is his bread and butter ({usage:.0%} of pitches), "
                f"but this lineup has a {lu_whiff:.0%} whiff rate against it -- "
                f"well below the {lg_whiff:.0%} norm",
                f"Trouble: {sim_row.get('opp_abbr', '')}'s hitters only whiff "
                f"at {lu_whiff:.0%} on the {pt_name}, and he throws it "
                f"{usage:.0%} of the time",
            ]
            bullets.append(random.choice(templates))

        # Lineup does damage on the primary pitch
        lu_xwoba = _lineup_xwoba_vs_pitch(pt, batter_ids, data.hitter_str)
        if lu_xwoba is not None and lu_xwoba > 0.370 and usage > 0.25:
            lg_xwoba = lg.get("xwoba_contact", 0.320)
            templates = [
                f"When this lineup connects on the {pt_name}, they do damage "
                f"(.{int(lu_xwoba * 1000):03d} xwOBA on contact vs "
                f".{int(lg_xwoba * 1000):03d} league avg)",
                f"The {pt_name} might get swings, but this group "
                f"{_damage_desc(lu_xwoba)} when they put it in play "
                f"(.{int(lu_xwoba * 1000):03d} xwOBA)",
            ]
            bullets.append(random.choice(templates))

    # 5. Count of batters with hitter advantage
    if not batter_sims.empty and "matchup_k_lift" in batter_sims.columns:
        hitter_edge = batter_sims[batter_sims["matchup_k_lift"] < -0.03]
        n_edge = len(hitter_edge)
        if n_edge >= 4:
            templates = [
                f"{n_edge} of {len(batter_sims)} batters in this lineup hold "
                f"matchup edges against his arsenal -- that's a lot of tough at-bats",
                f"Depth of problem: {n_edge} batters have favorable pitch-type "
                f"matchups, so there's no easy stretch in this order",
            ]
            bullets.append(random.choice(templates))

    return bullets[:4]  # Cap at 4 bullets


def _generate_advantages(
    sim_row: pd.Series,
    batter_sims: pd.DataFrame,
    top_pitches: pd.DataFrame,
    batter_ids: list[int],
    data: ReportData,
    pitcher_name: str,
) -> list[str]:
    """Generate bullets about pitcher advantages over the lineup."""
    bullets: list[str] = []
    k_rate = sim_row.get("projected_k_rate", 0.0)
    expected_k = sim_row.get("expected_k", 0.0)
    expected_hr = sim_row.get("expected_hr", 0.0)

    # Get projected BB rate
    pid = sim_row.get("pitcher_id")
    bb_rate_proj = 0.0
    pp = data.pitcher_proj
    if not pp.empty and pid is not None:
        pr = pp[pp["pitcher_id"] == pid]
        if not pr.empty:
            bb_rate_proj = float(pr["projected_bb_rate"].iloc[0] or 0.0)

    # 1. High K rate
    if k_rate >= 0.25:
        templates = [
            f"{pitcher_name} is {_k_rate_desc(k_rate)} -- {k_rate:.1%} K rate "
            f"with {expected_k:.0f} projected punchouts today",
            f"The strikeout upside is real: {k_rate:.1%} K rate translates "
            f"to {expected_k:.0f} projected Ks",
            f"Big K upside: {pitcher_name} projects for {expected_k:.0f} Ks "
            f"on a {k_rate:.1%} strikeout rate",
        ]
        bullets.append(random.choice(templates))

    # 2. Positive matchup lift -- lineup is K-prone
    avg_lift = sim_row.get("avg_matchup_lift", 0.0)
    if pd.notna(avg_lift) and avg_lift > 0.02:
        templates = [
            f"Matchup boost: this lineup's pitch-type vulnerabilities "
            f"play right into his arsenal",
            f"The matchup data loves this spot -- the opposing hitters "
            f"are susceptible to what he throws",
            f"His pitch mix exploits this lineup's weaknesses -- "
            f"expect K production above his baseline",
        ]
        bullets.append(random.choice(templates))

    # 3. Good control
    if bb_rate_proj <= 0.070:
        templates = [
            f"Excellent command ({bb_rate_proj:.1%} BB rate) "
            f"means efficient innings and a longer outing",
            f"{pitcher_name} has {_control_desc(bb_rate_proj)} -- he'll attack "
            f"the zone and keep his pitch count manageable",
            f"He walks very few hitters ({bb_rate_proj:.1%} BB rate) -- "
            f"expect a clean, efficient outing",
        ]
        bullets.append(random.choice(templates))

    # 4. Primary pitch dominates this lineup
    if not top_pitches.empty and batter_ids:
        top_pitch = top_pitches.iloc[0]
        pt = top_pitch["pitch_type"]
        pt_name = PITCH_DISPLAY.get(pt, pt)
        usage = top_pitch.get("usage_pct", 0)
        p_whiff = top_pitch.get("whiff_rate", 0)
        lg = LEAGUE_AVG_BY_PITCH_TYPE.get(pt, LEAGUE_AVG_OVERALL)
        lg_whiff = lg.get("whiff_rate", 0.25)

        lu_whiff = _lineup_whiff_vs_pitch(pt, batter_ids, data.vuln)
        if (
            lu_whiff is not None
            and lu_whiff > lg_whiff * 1.10
            and p_whiff > lg_whiff
            and usage > 0.20
        ):
            templates = [
                f"His {_whiff_adj(p_whiff)} {pt_name} ({p_whiff:.0%} whiff rate) "
                f"attacks a lineup that whiffs at {lu_whiff:.0%} against the pitch "
                f"-- that's a recipe for punchouts",
                f"The {pt_name} is {_whiff_adj(p_whiff)} ({p_whiff:.0%} whiff) "
                f"and this group swings through it at {lu_whiff:.0%} -- "
                f"above the {lg_whiff:.0%} league norm",
                f"Look for the {pt_name} to be the weapon here: "
                f"{p_whiff:.0%} whiff rate meets a lineup that can't "
                f"lay off ({lu_whiff:.0%} whiff rate on the pitch)",
            ]
            bullets.append(random.choice(templates))

        # Check secondary pitch too
        if len(top_pitches) >= 2:
            sec = top_pitches.iloc[1]
            s_pt = sec["pitch_type"]
            s_name = PITCH_DISPLAY.get(s_pt, s_pt)
            s_whiff = sec.get("whiff_rate", 0)
            s_lg = LEAGUE_AVG_BY_PITCH_TYPE.get(
                s_pt, LEAGUE_AVG_OVERALL
            ).get("whiff_rate", 0.25)
            s_lu_whiff = _lineup_whiff_vs_pitch(s_pt, batter_ids, data.vuln)
            if (
                s_lu_whiff is not None
                and s_lu_whiff > s_lg * 1.15
                and s_whiff > s_lg
            ):
                templates = [
                    f"Two-pitch problem for hitters: the {s_name} is also "
                    f"{_whiff_adj(s_whiff)} ({s_whiff:.0%} whiff) and "
                    f"this lineup swings through it at {s_lu_whiff:.0%}",
                    f"It's not just the primary pitch -- the {s_name} "
                    f"({s_whiff:.0%} whiff) also gives this lineup fits "
                    f"({s_lu_whiff:.0%} whiff rate)",
                ]
                bullets.append(random.choice(templates))

    # 5. Many batters are K-vulnerable
    if not batter_sims.empty and "matchup_k_lift" in batter_sims.columns:
        k_vuln = batter_sims[batter_sims["matchup_k_lift"] > 0.04]
        n_vuln = len(k_vuln)
        if n_vuln >= 4:
            templates = [
                f"{n_vuln} of {len(batter_sims)} hitters are K-vulnerable "
                f"against this pitch mix -- soft spots throughout the lineup",
                f"This order has holes: {n_vuln} batters are strikeout-prone "
                f"vs his repertoire",
            ]
            bullets.append(random.choice(templates))

    return bullets[:4]


def _classify_batters(
    batter_sims: pd.DataFrame,
    pitcher_id: int,
    data: ReportData,
    pitcher_name: str,
    pitcher_hand: str | None = None,
) -> tuple[list[str], list[str]]:
    """Classify batters as key (hitter edge) or struggling (pitcher edge).

    Returns (key_batter_bullets, struggling_batter_bullets).
    """
    key_bullets: list[str] = []
    struggle_bullets: list[str] = []

    if batter_sims.empty:
        return key_bullets, struggle_bullets

    top_pitches = _get_pitcher_top_pitches(pitcher_id, data.arsenal, n=2)
    primary_pt = None
    primary_name = None
    if not top_pitches.empty:
        primary_pt = top_pitches.iloc[0]["pitch_type"]
        primary_name = PITCH_DISPLAY.get(primary_pt, primary_pt)

    for _, brow in batter_sims.iterrows():
        bid = brow.get("batter_id")
        bname = brow.get("batter_name", "Unknown")
        order = brow.get("batting_order", "?")
        k_lift = brow.get("matchup_k_lift", 0.0)
        bb_lift = brow.get("matchup_bb_lift", 0.0)
        hr_lift = brow.get("matchup_hr_lift", 0.0)

        if any(pd.isna(x) for x in [k_lift, bb_lift, hr_lift]):
            k_lift = 0.0 if pd.isna(k_lift) else k_lift
            bb_lift = 0.0 if pd.isna(bb_lift) else bb_lift
            hr_lift = 0.0 if pd.isna(hr_lift) else hr_lift

        # Net score: negative k_lift = hitter advantage on Ks,
        # positive bb/hr lift = hitter advantage
        hitter_net = -k_lift + 0.5 * bb_lift + 0.5 * hr_lift

        # Get batter-specific pitch vuln for narrative detail
        reason = ""
        if bid is not None and primary_pt:
            bv = data.vuln[
                (data.vuln["batter_id"] == bid)
                & (data.vuln["pitch_type"] == primary_pt)
            ]
            b_whiff = None
            if not bv.empty and "swings" in bv.columns:
                sw = bv["swings"].iloc[0]
                wh = bv["whiffs"].iloc[0]
                if pd.notna(sw) and sw > 10:
                    b_whiff = wh / sw

            bs = data.hitter_str[
                (data.hitter_str["batter_id"] == bid)
                & (data.hitter_str["pitch_type"] == primary_pt)
            ] if not data.hitter_str.empty else pd.DataFrame()
            b_xwoba = None
            if not bs.empty and "xwoba_contact" in bs.columns:
                b_xwoba = bs["xwoba_contact"].iloc[0]

            # Build the reason from the dominant signal
            if hitter_net > 0.04:  # hitter favored
                if b_xwoba is not None and b_xwoba >= 0.380:
                    reason = (
                        f"{_damage_desc(b_xwoba)} vs the {primary_name} "
                        f"(.{int(b_xwoba * 1000):03d} xwOBA)"
                    )
                elif k_lift < -0.04:
                    reason = f"tough to strike out with this arsenal"
                elif bb_lift > 0.03:
                    reason = "good eye, will work deep counts and draw walks"
                else:
                    reason = "pitch-type matchup favors him across the board"
            elif hitter_net < -0.04:  # pitcher favored
                if b_whiff is not None and b_whiff > 0.30 and primary_name:
                    reason = (
                        f"whiffs at {b_whiff:.0%} on the {primary_name} "
                        f"-- that's trouble"
                    )
                elif k_lift > 0.06:
                    reason = "K-vulnerable against this pitch mix"
                elif bb_lift < -0.02:
                    reason = "chases out of the zone against this type of stuff"
                else:
                    reason = f"{pitcher_name}'s arsenal has a clear edge"

        # Platoon context
        platoon_note = ""
        if pitcher_hand and bid is not None and not data.platoon.empty:
            plat = data.platoon[
                (data.platoon["batter_id"] == bid)
                & (data.platoon["pitch_hand"] == pitcher_hand)
            ]
            if not plat.empty:
                pk = plat["k_rate"].iloc[0]
                ok = plat["overall_k_rate"].iloc[0]
                if pd.notna(pk) and pd.notna(ok):
                    diff = pk - ok
                    if diff > 0.04:
                        platoon_note = (
                            f" (K rate jumps to {pk:.0%} vs "
                            f"{'lefties' if pitcher_hand == 'L' else 'righties'})"
                        )
                    elif diff < -0.04:
                        platoon_note = (
                            f" (only {pk:.0%} K rate vs "
                            f"{'lefties' if pitcher_hand == 'L' else 'righties'})"
                        )

        order_label = f"#{int(order)}" if pd.notna(order) else ""

        if hitter_net > 0.04:
            bullet = f"{bname} ({order_label})"
            if reason:
                bullet += f" -- {reason}"
            bullet += platoon_note
            key_bullets.append((hitter_net, bullet))
        elif hitter_net < -0.04:
            bullet = f"{bname} ({order_label})"
            if reason:
                bullet += f" -- {reason}"
            bullet += platoon_note
            struggle_bullets.append((abs(hitter_net), bullet))

    # Sort by magnitude and cap at 3
    key_bullets.sort(key=lambda x: x[0], reverse=True)
    struggle_bullets.sort(key=lambda x: x[0], reverse=True)

    return (
        [b for _, b in key_bullets[:3]],
        [b for _, b in struggle_bullets[:3]],
    )


def _generate_bullpen_notes(
    sim_row: pd.Series,
    data: ReportData,
    pitcher_team_abbr: str,
    opp_team_abbr: str,
) -> list[str]:
    """Generate bullets about bullpen impact."""
    bullets: list[str] = []

    expected_ip = sim_row.get("expected_ip", 5.0)
    bp_rates = _get_bullpen_rates(pitcher_team_abbr, data)

    # Pitcher's own team bullpen
    if bp_rates:
        bp_k = bp_rates["k_rate"]
        bp_bb = bp_rates["bb_rate"]
        bp_hr = bp_rates["hr_rate"]

        # Strong bullpen
        if bp_k >= 0.26:
            templates = [
                f"{pitcher_team_abbr} has {_bullpen_quality(bp_k)} "
                f"({bp_k:.1%} K rate) -- any lead is in good hands",
                f"The bullpen is a plus here: {pitcher_team_abbr} relievers "
                f"punch out {bp_k:.1%} of hitters they face",
            ]
            bullets.append(random.choice(templates))
        # Weak bullpen
        elif bp_k < 0.22:
            templates = [
                f"{pitcher_team_abbr} has {_bullpen_quality(bp_k)} "
                f"({bp_k:.1%} K rate) -- he needs to go deep to protect "
                f"any lead",
                f"The pen is a liability: {pitcher_team_abbr} relievers "
                f"struggle to miss bats ({bp_k:.1%} K rate)",
            ]
            bullets.append(random.choice(templates))

        # Walk-heavy bullpen
        if bp_bb > 0.095:
            templates = [
                f"Bullpen walks are a problem ({bp_bb:.1%} BB rate) -- "
                f"late-game leads can unravel quickly",
                f"The relief corps puts runners on at a {bp_bb:.1%} clip -- "
                f"a short outing could get ugly",
            ]
            bullets.append(random.choice(templates))

        # Short outing + bad pen compound
        if expected_ip < 5.0 and bp_k < 0.23:
            templates = [
                f"Short leash expected ({expected_ip:.0f} IP projected) "
                f"and {_bullpen_quality(bp_k)} behind him -- risky combo",
                f"Only {expected_ip:.0f} IP projected, handing the ball to "
                f"a pen that doesn't miss bats -- proceed with caution",
            ]
            bullets.append(random.choice(templates))

    return bullets[:3]


def _generate_batting_outlook(
    sim_row: pd.Series,
    data: ReportData,
    pitcher_team_abbr: str,
    opp_team_abbr: str,
    pitcher_name: str,
) -> list[str]:
    """Generate bullets from the opposing lineup's batting perspective.

    Frames how ``opp_team_abbr``'s batters match up against the starter
    *and* what they can expect from ``pitcher_team_abbr``'s bullpen once
    the starter exits.
    """
    bullets: list[str] = []

    k_rate = sim_row.get("projected_k_rate", 0.0)
    expected_ip = sim_row.get("expected_ip", 5.0)
    bp_rates = _get_bullpen_rates(pitcher_team_abbr, data)

    if not bp_rates:
        return bullets

    bp_k = bp_rates["k_rate"]
    bp_bb = bp_rates.get("bb_rate", 0.0)
    bp_hr = bp_rates.get("hr_rate", 0.0)

    tough_starter = k_rate >= 0.25
    soft_starter = k_rate < 0.20
    weak_pen = bp_k < 0.21
    strong_pen = bp_k >= 0.26
    short_outing = expected_ip < 5.0
    walky_pen = bp_bb > 0.095
    hr_pen = bp_hr > 0.032

    # Tough starter + weak bullpen -- survive and thrive
    if tough_starter and weak_pen:
        templates = [
            f"{opp_team_abbr} batters will struggle against {pitcher_name} "
            f"({k_rate:.1%} K rate) but the bullpen is a different story "
            f"({bp_k:.1%} K rate)",
            f"Tough matchup early against {pitcher_name}, but "
            f"{pitcher_team_abbr}'s pen ({bp_k:.1%} K rate) gives "
            f"{opp_team_abbr}'s lineup a window late",
        ]
        bullets.append(random.choice(templates))

    # Soft starter + weak bullpen -- feast all game
    elif soft_starter and weak_pen:
        templates = [
            f"If {opp_team_abbr} can get to {pitcher_name} early, they are "
            f"going to feast on the bullpen ({bp_k:.1%} K rate)",
            f"{pitcher_name} doesn't miss many bats ({k_rate:.1%} K rate) "
            f"and the pen behind him is no better -- "
            f"{opp_team_abbr}'s lineup should stay aggressive all game",
        ]
        bullets.append(random.choice(templates))

    # Tough starter + strong bullpen -- tough all game
    elif tough_starter and strong_pen:
        templates = [
            f"Tough sledding for {opp_team_abbr} all game -- "
            f"{pitcher_name} is dominant ({k_rate:.1%} K rate) and "
            f"the bullpen is just as nasty ({bp_k:.1%} K rate)",
            f"{opp_team_abbr} needs to capitalize on mistakes -- "
            f"{pitcher_name} and {pitcher_team_abbr}'s pen "
            f"({bp_k:.1%} K rate) don't give many free passes",
        ]
        bullets.append(random.choice(templates))

    # Short outing + weak bullpen -- late-game opportunity
    if short_outing and weak_pen and not (tough_starter and weak_pen):
        templates = [
            f"{pitcher_name} is on a short leash ({expected_ip:.0f} IP "
            f"projected) and {pitcher_team_abbr}'s pen is hittable "
            f"({bp_k:.1%} K rate) -- late innings could open up for "
            f"{opp_team_abbr}",
            f"Get through {pitcher_name} early and the game flips: "
            f"{pitcher_team_abbr}'s bullpen ({bp_k:.1%} K rate) "
            f"won't shut the door",
        ]
        bullets.append(random.choice(templates))

    # Walk-heavy pen -- free baserunners
    if walky_pen and not strong_pen and len(bullets) < 2:
        bullets.append(
            f"{pitcher_team_abbr}'s relievers put runners on "
            f"({bp_bb:.1%} BB rate) -- patience pays for "
            f"{opp_team_abbr} in the late innings"
        )

    return bullets[:2]


# ---------------------------------------------------------------------------
# Per-game analysis
# ---------------------------------------------------------------------------

def _analyze_pitcher_side(
    game_row: pd.Series,
    side: str,
    data: ReportData,
) -> PitcherReport | None:
    """Analyze one pitcher in a game (home or away)."""
    pid_col = f"{side}_pitcher_id"
    name_col = f"{side}_pitcher_name"
    opp_side = "home" if side == "away" else "away"

    pid = game_row.get(pid_col)
    if pd.isna(pid):
        return None
    pid = int(pid)
    pname = game_row.get(name_col, "TBD")
    team_abbr = game_row.get(f"{side}_abbr", "")
    opp_abbr = game_row.get(f"{opp_side}_abbr", "")

    # Get sim row for this pitcher
    sim = data.sims[data.sims["pitcher_id"] == pid]
    if sim.empty:
        return PitcherReport(
            pitcher_name=pname, pitcher_id=pid,
            team_abbr=team_abbr, opp_abbr=opp_abbr, side=side,
        )
    sim_row = sim.iloc[0]

    # Get pitcher hand
    pitcher_hand = None
    pr = data.pitcher_proj[data.pitcher_proj["pitcher_id"] == pid]
    if not pr.empty and "pitch_hand" in pr.columns:
        pitcher_hand = pr["pitch_hand"].iloc[0]

    # Get archetype
    archetype = ""
    if not data.pitcher_arch.empty:
        pa = data.pitcher_arch[data.pitcher_arch["pitcher_id"] == pid]
        if not pa.empty and "archetype_name" in pa.columns:
            archetype = str(pa["archetype_name"].iloc[0] or "")

    # Get projected BB rate for this pitcher
    bb_rate_proj = 0.0
    if not pr.empty and "projected_bb_rate" in pr.columns:
        val = pr["projected_bb_rate"].iloc[0]
        bb_rate_proj = float(val) if pd.notna(val) else 0.0

    # Get opposing batter sims (cap to lineup spots 1-9)
    gpk = game_row["game_pk"]
    bs = data.batter_sims[
        (data.batter_sims["game_pk"] == gpk)
        & (data.batter_sims["opp_starter_id"] == pid)
    ] if not data.batter_sims.empty else pd.DataFrame()
    if not bs.empty and "batting_order" in bs.columns:
        bs = bs[bs["batting_order"] <= 9].sort_values("batting_order")

    batter_ids = list(bs["batter_id"].dropna().astype(int)) if not bs.empty else []

    # Top pitches
    top_pitches = _get_pitcher_top_pitches(pid, data.arsenal, n=3)

    # Generate all bullet sections
    struggles = _generate_struggles(
        sim_row, bs, top_pitches, batter_ids, data, pname,
    )
    advantages = _generate_advantages(
        sim_row, bs, top_pitches, batter_ids, data, pname,
    )
    key_batters, struggling_batters = _classify_batters(
        bs, pid, data, pname, pitcher_hand,
    )
    bullpen_notes = _generate_bullpen_notes(
        sim_row, data, team_abbr, opp_abbr,
    )
    batting_outlook = _generate_batting_outlook(
        sim_row, data, team_abbr, opp_abbr, pname,
    )

    return PitcherReport(
        pitcher_name=pname,
        pitcher_id=pid,
        team_abbr=team_abbr,
        opp_abbr=opp_abbr,
        side=side,
        projected_k_rate=float(sim_row.get("projected_k_rate", 0.0)),
        projected_bb_rate=bb_rate_proj,
        expected_k=float(sim_row.get("expected_k", 0.0)),
        expected_ip=float(sim_row.get("expected_ip", 0.0)),
        expected_bb=float(sim_row.get("expected_bb", 0.0)),
        expected_hr=float(sim_row.get("expected_hr", 0.0)),
        dk_mean=float(sim_row.get("dk_mean", 0.0)),
        archetype=archetype,
        struggles=struggles,
        advantages=advantages,
        key_batters=key_batters,
        struggling_batters=struggling_batters,
        bullpen_notes=bullpen_notes,
        batting_outlook=batting_outlook,
        has_lineup=bool(sim_row.get("has_lineup", False)),
    )


def analyze_game(game_row: pd.Series, data: ReportData) -> GameReport:
    """Analyze a full game, both sides."""
    return GameReport(
        game_pk=int(game_row["game_pk"]),
        game_time=str(game_row.get("game_time", "")),
        away_abbr=str(game_row.get("away_abbr", "")),
        home_abbr=str(game_row.get("home_abbr", "")),
        away=_analyze_pitcher_side(game_row, "away", data),
        home=_analyze_pitcher_side(game_row, "home", data),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_pitcher_section(report: PitcherReport) -> list[str]:
    """Format a single pitcher's analysis into markdown lines."""
    lines: list[str] = []
    label = f"vs {report.opp_abbr}" if report.side == "away" else f"vs {report.opp_abbr}"
    header = f"### {report.pitcher_name} ({report.team_abbr}) {label}"
    lines.append(header)

    # Quick stat line
    parts = []
    if report.expected_k > 0:
        parts.append(f"{report.expected_k:.0f} K")
    if report.expected_ip > 0:
        parts.append(f"{report.expected_ip:.0f} IP")
    if report.expected_bb > 0:
        parts.append(f"{report.expected_bb:.0f} BB")
    if report.dk_mean > 0:
        parts.append(f"{report.dk_mean:.1f} DK pts")
    if report.archetype:
        parts.append(report.archetype)
    if parts:
        lines.append(f"*Projected: {' | '.join(parts)}*")
    if not report.has_lineup:
        lines.append("*(Lineup not yet confirmed -- projections based on roster)*")
    lines.append("")

    if report.advantages:
        lines.append("**Advantages:**")
        for b in report.advantages:
            lines.append(f"- {b}")
        lines.append("")

    if report.struggles:
        lines.append("**Struggles:**")
        for b in report.struggles:
            lines.append(f"- {b}")
        lines.append("")

    if report.key_batters:
        lines.append("**Batters to Watch (hitter favored):**")
        for b in report.key_batters:
            lines.append(f"- {b}")
        lines.append("")

    if report.struggling_batters:
        lines.append("**Batters Who Will Struggle (pitcher favored):**")
        for b in report.struggling_batters:
            lines.append(f"- {b}")
        lines.append("")

    if report.bullpen_notes:
        lines.append("**Bullpen:**")
        for b in report.bullpen_notes:
            lines.append(f"- {b}")
        lines.append("")

    if report.batting_outlook:
        lines.append(f"**{report.opp_abbr} Batting Outlook:**")
        for b in report.batting_outlook:
            lines.append(f"- {b}")
        lines.append("")

    return lines


def format_game_report(game: GameReport) -> str:
    """Format a full game report into markdown."""
    lines: list[str] = []
    lines.append(f"## {game.away_abbr} @ {game.home_abbr} -- {game.game_time}")
    lines.append("")

    if game.away:
        lines.extend(_format_pitcher_section(game.away))
    else:
        lines.append(f"### {game.away_abbr} starter: TBD")
        lines.append("")

    if game.home:
        lines.extend(_format_pitcher_section(game.home))
    else:
        lines.append(f"### {game.home_abbr} starter: TBD")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_pitcher_scouting(
    pitcher_id: int,
    pitcher_name: str,
    team_abbr: str,
    opp_abbr: str,
    data: ReportData,
) -> PitcherReport:
    """Generate scouting bullets for a single pitcher.

    Convenience wrapper for the schedule page drilldown -- avoids needing
    to construct a full game row from todays_games.
    """
    # Find sim row
    sim = data.sims[data.sims["pitcher_id"] == pitcher_id]
    if sim.empty:
        return PitcherReport(
            pitcher_name=pitcher_name, pitcher_id=pitcher_id,
            team_abbr=team_abbr, opp_abbr=opp_abbr, side="",
        )
    sim_row = sim.iloc[0]
    gpk = sim_row.get("game_pk")

    # Pitcher hand
    pitcher_hand = None
    pr = data.pitcher_proj[data.pitcher_proj["pitcher_id"] == pitcher_id]
    if not pr.empty and "pitch_hand" in pr.columns:
        pitcher_hand = pr["pitch_hand"].iloc[0]

    bb_rate_proj = 0.0
    if not pr.empty and "projected_bb_rate" in pr.columns:
        val = pr["projected_bb_rate"].iloc[0]
        bb_rate_proj = float(val) if pd.notna(val) else 0.0

    # Archetype
    archetype = ""
    if not data.pitcher_arch.empty:
        pa = data.pitcher_arch[data.pitcher_arch["pitcher_id"] == pitcher_id]
        if not pa.empty and "archetype_name" in pa.columns:
            archetype = str(pa["archetype_name"].iloc[0] or "")

    # Batter sims (cap to lineup spots 1-9)
    bs = data.batter_sims[
        (data.batter_sims["game_pk"] == gpk)
        & (data.batter_sims["opp_starter_id"] == pitcher_id)
    ] if not data.batter_sims.empty and pd.notna(gpk) else pd.DataFrame()
    if not bs.empty and "batting_order" in bs.columns:
        bs = bs[bs["batting_order"] <= 9].sort_values("batting_order")

    batter_ids = list(bs["batter_id"].dropna().astype(int)) if not bs.empty else []
    top_pitches = _get_pitcher_top_pitches(pitcher_id, data.arsenal, n=3)

    struggles = _generate_struggles(sim_row, bs, top_pitches, batter_ids, data, pitcher_name)
    advantages = _generate_advantages(sim_row, bs, top_pitches, batter_ids, data, pitcher_name)
    key_batters, struggling_batters = _classify_batters(bs, pitcher_id, data, pitcher_name, pitcher_hand)
    bullpen_notes = _generate_bullpen_notes(sim_row, data, team_abbr, opp_abbr)
    batting_outlook = _generate_batting_outlook(sim_row, data, team_abbr, opp_abbr, pitcher_name)

    return PitcherReport(
        pitcher_name=pitcher_name,
        pitcher_id=pitcher_id,
        team_abbr=team_abbr,
        opp_abbr=opp_abbr,
        side="",
        projected_k_rate=float(sim_row.get("projected_k_rate", 0.0)),
        projected_bb_rate=bb_rate_proj,
        expected_k=float(sim_row.get("expected_k", 0.0)),
        expected_ip=float(sim_row.get("expected_ip", 0.0)),
        expected_bb=float(sim_row.get("expected_bb", 0.0)),
        expected_hr=float(sim_row.get("expected_hr", 0.0)),
        dk_mean=float(sim_row.get("dk_mean", 0.0)),
        archetype=archetype,
        struggles=struggles,
        advantages=advantages,
        key_batters=key_batters,
        struggling_batters=struggling_batters,
        bullpen_notes=bullpen_notes,
        batting_outlook=batting_outlook,
        has_lineup=bool(sim_row.get("has_lineup", False)),
    )


def generate_daily_report(
    dashboard_dir: Path,
    report_date: str | None = None,
) -> str:
    """Generate the full daily preview report as markdown.

    Parameters
    ----------
    dashboard_dir : Path
        Path to the data/dashboard/ directory.
    report_date : str, optional
        Date string for the header. Defaults to "Today".

    Returns
    -------
    str
        Full markdown report.
    """
    data = load_report_data(dashboard_dir)

    if data.games.empty:
        return "# Daily Preview\n\nNo games scheduled today.\n"

    header_date = report_date or "Today"
    lines: list[str] = [
        f"# Daily Preview -- {header_date}",
        "",
        f"*{len(data.games)} games on the slate*",
        "",
    ]

    game_reports: list[GameReport] = []
    for _, game_row in data.games.iterrows():
        gr = analyze_game(game_row, data)
        game_reports.append(gr)

    for gr in game_reports:
        lines.append(format_game_report(gr))

    return "\n".join(lines)
