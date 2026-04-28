"""Team Overview page -- Press Box data wall."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from config import (
    GOLD, EMBER, SAGE, SLATE, CREAM,
    CURRENT_SEASON, PRIOR_SEASON,
)
from utils.alerts import tdd_info, tdd_warn
from services.data_loader import (
    load_projections, load_counting, load_roster,
    load_preseason_injuries,
    load_hitter_archetypes, load_pitcher_archetypes,
    load_rankings, load_team_elo, load_team_profiles, load_team_rankings,
    load_standings,
)
from utils.html import esc, esc_attr
from utils.team_names import team_full, abbr_from_full
from components.team_logo import team_logo_html


# ── Constants ─────────────────────────────────────────────────────────
_HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]

_TIER_COLORS = {
    "Elite": GOLD, "Contender": SAGE,
    "Competitive": SLATE, "Rebuilding": EMBER,
}

# Position coords as % within the depth-chart diamond (facing up)
_FIELD_POS = {
    "C":  (50, 84),
    "1B": (80, 58),
    "2B": (68, 40),
    "SS": (32, 40),
    "3B": (20, 58),
    "LF": (12, 20),
    "CF": (50, 8),
    "RF": (88, 20),
}

# Score labels for radar / diamond scores
_SCORE_KEYS = [
    ("lineup_diamond",   "Lineup"),
    ("rotation_diamond", "Rotation"),
    ("bullpen_diamond",  "Bullpen"),
]


# ── Helpers ───────────────────────────────────────────────────────────

def _safe(val, default=None):
    """Return val if not NaN/None, else default."""
    if val is None:
        return default
    if isinstance(val, float) and (pd.isna(val) or math.isnan(val)):
        return default
    return val


def _section(num: str, title: str, sub: str = "") -> str:
    sub_html = f'<span class="sub">{esc(sub)}</span>' if sub else ""
    return (
        f'<div class="to-sec">'
        f'<span class="num">{esc(num)}</span>'
        f'<h2>{esc(title)}</h2>{sub_html}'
        f'</div>'
    )


def _blk_head(title: str, meta: str = "") -> str:
    meta_html = f'<span class="meta">{esc(meta)}</span>' if meta else ""
    return f'<div class="blk-head"><span>{esc(title)}</span>{meta_html}</div>'


# ── Rendering functions ──────────────────────────────────────────────

def _render_eyebrow(division: str, team_name: str) -> str:
    return (
        f'<div class="to-eyebrow">'
        f'<span>The Data Diamond</span>'
        f'<span class="sep">/</span>'
        f'<span>Teams</span>'
        f'<span class="sep">/</span>'
        f'<span>{esc(division)}</span>'
        f'<span class="sep">/</span>'
        f'<span class="gold">{esc(team_name)}</span>'
        f'</div>'
    )


def _render_team_header(
    abbr: str,
    team_name: str,
    division: str,
    record: tuple[int, int] | None,
    tr: pd.Series,
    te: pd.Series,
    team_id: int | None = None,
) -> str:
    # Record line
    w, l = record if record else (0, 0)
    pct = w / (w + l) if (w + l) > 0 else 0
    pct_str = f"{pct:.3f}".lstrip("0")

    run_diff = _safe(tr.get("run_diff"), "")
    if isinstance(run_diff, (int, float)):
        run_diff = f"{run_diff:+.0f}" if run_diff != 0 else "0"
    streak = _safe(tr.get("streak"), "")
    last10 = _safe(tr.get("last_10"), "")

    sub_parts = [f'<span class="rec">{w}-{l}</span>', f'<span>{pct_str} W%</span>']
    if run_diff:
        sub_parts.append(f'<span class="gold">RD {run_diff}</span>')
    if last10:
        sub_parts.append(f'<span>L10 {last10}</span>')
    if streak:
        sub_parts.append(f'<span class="gold">{streak}</span>')

    sub_html = '<span class="dot">&middot;</span>'.join(sub_parts)

    # Stat boxes
    elo = _safe(tr.get("elo"), _safe(te.get("composite_elo")))
    elo_delta = _safe(tr.get("elo_delta"), "")
    rank = _safe(tr.get("rank"))
    proj_wins = _safe(tr.get("proj_wins"))

    stats_html = ""
    if elo:
        delta_cls = ' class="delta up"' if str(elo_delta).startswith("+") else ' class="delta"'
        stats_html += (
            f'<div class="stat">'
            f'<div class="v">{int(elo)}</div>'
            f'<div class="l">ELO</div>'
            f'<div{delta_cls}>{elo_delta}</div>'
            f'</div>'
        )
    if rank:
        stats_html += (
            f'<div class="stat">'
            f'<div class="v neutral">#{int(rank)}</div>'
            f'<div class="l">Lg Rank</div>'
            f'<div class="delta">&middot;</div>'
            f'</div>'
        )
    if proj_wins:
        stats_html += (
            f'<div class="stat">'
            f'<div class="v">{float(proj_wins):.1f}</div>'
            f'<div class="l">Proj Wins</div>'
            f'<div class="delta">&middot;</div>'
            f'</div>'
        )

    return (
        f'<header class="to-header">'
        f'<div class="crest" data-team="{esc_attr(abbr)}">'
        f'{team_logo_html(team_id, size=72) if team_id else esc(abbr)}'
        f'</div>'
        f'<div>'
        f'<div class="name">{esc(division)} &middot; {CURRENT_SEASON} Season</div>'
        f'<h1>{esc(team_name)}</h1>'
        f'<div class="sub">{sub_html}</div>'
        f'</div>'
        f'<div class="stats">{stats_html}</div>'
        f'</header>'
    )


def _render_tier_row(tier: str, div_rank: int | None) -> str:
    tier_cls = "gold" if tier == "Elite" else "sage" if tier == "Contender" else ""
    pills = [f'<span class="pill {tier_cls}">{tier} Tier</span>']
    if div_rank:
        pills.append(f'<span class="pill">Div #{div_rank}</span>')
    return f'<div class="to-tier-row">{"".join(pills)}</div>'


def _render_identity_card(
    tier: str, rank: int | None, division: str, div_rank: int | None,
    off_style: str, pit_style: str, age_traj: str,
) -> str:
    rows = [
        ("Tier", tier, "gold" if tier == "Elite" else ""),
        ("League #", f"{rank} of 30" if rank else "--", "gold" if rank and rank <= 5 else ""),
        ("Division", f"{division} &middot; #{div_rank}" if div_rank else division, ""),
        ("Style", " &middot; ".join(filter(None, [off_style, pit_style])) or "--", ""),
        ("Tempo", age_traj or "--", "slate"),
    ]
    rows_html = ""
    for k, v, cls in rows:
        v_cls = f' class="v {cls}"' if cls else ' class="v"'
        rows_html += (
            f'<div class="to-identity-row">'
            f'<span class="k">{esc(k)}</span>'
            f'<span{v_cls}>{v}</span>'
            f'</div>'
        )
    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Team Identity", "profile")}'
        f'<div class="to-identity-list">{rows_html}</div>'
        f'</div>'
    )


def _render_scores_card(scores: list[dict]) -> str:
    rows_html = ""
    for s in scores:
        v = s["v"]
        pct = (v / 10) * 100
        lg_pct = (s["lg_avg"] / 10) * 100
        blurb = s.get("blurb", "")
        rows_html += (
            f'<div class="to-score-row">'
            f'<span class="lab">{esc(s["label"])}</span>'
            f'<div class="bar">'
            f'<div class="fill" style="width:{pct:.0f}%"></div>'
            f'<div class="lg-tick" style="left:{lg_pct:.0f}%"></div>'
            f'</div>'
            f'<div class="right">'
            f'<span class="v">{v:.1f}</span>'
            f'<span class="rk">#{s["rank"]}</span>'
            f'</div>'
            f'</div>'
        )
        if blurb:
            rows_html += f'<div class="to-score-blurb">{esc(blurb)}</div>'
    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Diamond Scores", "0-10 vs lg avg")}'
        f'<div class="to-scores">{rows_html}</div>'
        f'</div>'
    )


def _render_radar_card(scores: list[dict]) -> str:
    """5-axis radar SVG."""
    labels = [s["label"] for s in scores]
    values = [s["v"] / 10 for s in scores]
    lg_vals = [s["lg_avg"] / 10 for s in scores]
    cx, cy, R = 140, 130, 92
    n = len(labels)

    def angle(i: int) -> float:
        return (-math.pi / 2) + (i * 2 * math.pi / n)

    def pt(i: int, r: float) -> tuple[float, float]:
        return cx + math.cos(angle(i)) * R * r, cy + math.sin(angle(i)) * R * r

    def poly_pts(vs: list[float]) -> str:
        return " ".join(f"{pt(i, v)[0]:.1f},{pt(i, v)[1]:.1f}" for i, v in enumerate(vs))

    # Grid rings
    rings = ""
    for r in (0.25, 0.5, 0.75, 1.0):
        rings += f'<polygon points="{poly_pts([r]*n)}" fill="none" stroke="#2A2E3A" stroke-width="1"/>'

    # Spokes
    spokes = ""
    for i in range(n):
        x, y = pt(i, 1)
        spokes += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#2A2E3A" stroke-width="1"/>'

    # League avg polygon
    lg_poly = f'<polygon points="{poly_pts(lg_vals)}" fill="rgba(123,143,166,0.15)" stroke="#7B8FA6" stroke-width="1" stroke-dasharray="3 3"/>'

    # Team polygon
    team_poly = f'<polygon points="{poly_pts(values)}" fill="var(--tdd-gold)" fill-opacity="0.18" stroke="var(--tdd-gold)" stroke-width="1.5"/>'

    # Dots
    dots = ""
    for i, v in enumerate(values):
        x, y = pt(i, v)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--tdd-gold)"/>'

    # Labels
    label_els = ""
    for i, lab in enumerate(labels):
        x, y = pt(i, 1.18)
        label_els += (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="middle" '
            f'fill="#7B8FA6" font-size="9.5" font-family="Inter" font-weight="700" '
            f'letter-spacing="1">{lab.upper()}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 280 260">'
        f'{rings}{spokes}{lg_poly}{team_poly}{dots}{label_els}'
        f'</svg>'
    )

    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Composite Profile", "team vs lg avg")}'
        f'<div class="to-radar-wrap">{svg}</div>'
        f'</div>'
    )


def _render_sw_list(items: list[dict], kind: str) -> str:
    title = "Strengths" if kind == "up" else "Weaknesses"
    rows_html = ""
    for it in items:
        rows_html += (
            f'<div class="to-sw-row {kind}">'
            f'<div class="lab">{esc(it["label"])}'
            f'<span class="lg">{esc(it["metric"])} &middot; {esc(it["lg"])}</span>'
            f'</div>'
            f'<div class="delta">{esc(it["delta"])}</div>'
            f'<div class="rk">#{it["rank"]}</div>'
            f'</div>'
        )
    return (
        f'<div class="to-card-block">'
        f'{_blk_head(title, "vs league avg")}'
        f'<div class="to-sw-list">{rows_html}</div>'
        f'</div>'
    )


def _render_top_players(players: list[dict]) -> str:
    rows_html = ""
    for i, p in enumerate(players, 1):
        rk_cls = "rk gold" if p["rank"] <= 5 else "rk sage" if p["rank"] <= 15 else "rk"
        note_html = f'<span class="il">&middot; {esc(p.get("note", ""))}</span>' if p.get("note") else ""
        arch_html = f'<span class="arch">{esc(p.get("arch", ""))}</span>' if p.get("arch") else ""
        rows_html += (
            f'<div class="to-top-row">'
            f'<span class="{rk_cls}">{str(i).zfill(2)}</span>'
            f'<span class="pos">{esc(p["pos"])}</span>'
            f'<span class="nm">{esc(p["name"])}{arch_html}{note_html}</span>'
            f'<span class="diamond"><span class="glyph">&#9670;</span>{p["score"]:.1f}</span>'
            f'<span class="rk">#{p["rank"]}</span>'
            f'</div>'
        )
    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Top Players", "league rank")}'
        f'<div class="to-top-list">{rows_html}</div>'
        f'</div>'
    )


def _render_depth_field(depth: dict[str, list[dict]]) -> str:
    """Baseball diamond SVG with positioned player labels."""
    # SVG background
    svg = (
        '<svg viewBox="0 0 540 514" preserveAspectRatio="xMidYMid meet">'
        '<defs>'
        '<pattern id="grass" x="0" y="0" width="14" height="14" patternUnits="userSpaceOnUse">'
        '<rect width="14" height="14" fill="#161922"/>'
        '<circle cx="2" cy="2" r="0.6" fill="#1f2330"/>'
        '</pattern>'
        '</defs>'
        '<path d="M 30 280 A 240 240 0 0 1 510 280 L 270 514 Z" '
        'fill="url(#grass)" stroke="#2A2E3A" stroke-width="1"/>'
        '<polygon points="270,160 400,290 270,420 140,290" '
        'fill="#1c2030" stroke="#3a3f4f" stroke-width="1"/>'
        '<circle cx="270" cy="290" r="60" fill="none" stroke="#3a3f4f" '
        'stroke-width="1" stroke-dasharray="2 4"/>'
    )
    # Bases
    for bx, by in [(270, 160), (400, 290), (270, 420), (140, 290)]:
        svg += (
            f'<rect x="{bx-5}" y="{by-5}" width="10" height="10" '
            f'fill="#C8A96E" transform="rotate(45 {bx} {by})"/>'
        )
    # Pitcher mound
    svg += '<circle cx="270" cy="290" r="9" fill="#2A2E3A" stroke="#3a3f4f"/>'
    svg += '</svg>'

    # Position labels
    pos_els = ""
    for pos, (x, y) in _FIELD_POS.items():
        players = depth.get(pos, [])
        if not players:
            continue
        starter = players[0]
        backup = players[1] if len(players) > 1 else None
        bench_html = ""
        if backup:
            last_name = backup["name"].split()[-1]
            bench_html = f'<span class="bench">-> {esc(last_name)}</span>'
        pos_els += (
            f'<div class="to-field-pos" style="left:{x}%;top:{y}%">'
            f'<div class="pin">{pos}</div>'
            f'<div class="nm" title="{esc_attr(starter["name"])}">'
            f'{esc(starter["name"])}'
            f'<span class="sc">{starter["score"]:.1f}</span>'
            f'</div>'
            f'{bench_html}'
            f'</div>'
        )

    # DH row below the diamond
    dh_players = depth.get("DH", [])
    dh_html = ""
    if dh_players:
        dh = dh_players[0]
        dh_html = (
            f'<div style="margin-top:0.7rem; display:flex; align-items:center; '
            f'justify-content:center; gap:0.55rem; '
            f'font-family:var(--tdd-font-heading); font-size:0.7rem; '
            f'color:var(--tdd-slate); letter-spacing:1px; '
            f'border-top:1px solid var(--tdd-dark-border-faint); padding-top:0.55rem;">'
            f'<span style="background:rgba(123,143,166,0.1); padding:2px 6px; '
            f'border-radius:2px; font-size:0.58rem; letter-spacing:1.2px; font-weight:700;">DH</span>'
            f'<span style="color:var(--tdd-cream);">{esc(dh["name"])}</span>'
            f'<span style="color:var(--tdd-gold); font-weight:800; '
            f'font-variant-numeric:tabular-nums;">&#9670; {dh["score"]:.1f}</span>'
            f'</div>'
        )

    return (
        f'<div class="to-card-block to-field-card">'
        f'{_blk_head("Depth Chart", "starter / next-up")}'
        f'<div class="to-field-wrap">{svg}{pos_els}</div>'
        f'{dh_html}'
        f'</div>'
    )


def _render_pitching_staff(rotation: list[dict], bullpen: list[dict]) -> str:
    def _row(slot: str, p: dict) -> str:
        hand = p.get("hand", "")
        hand_html = f'<span class="h">{hand}HP</span>' if hand else ""
        era = p.get("era")
        era_str = f"{era:.2f}" if era is not None else "--"
        arch = p.get("arch", "")
        return (
            f'<div class="to-staff-row">'
            f'<span class="slot">{esc(slot)}</span>'
            f'<span class="nm">{esc(p["name"])} {hand_html}</span>'
            f'<span class="era">{era_str}</span>'
            f'<span class="sc">{p["score"]:.1f}</span>'
            f'<span class="arch">{esc(arch)}</span>'
            f'</div>'
        )

    rot_html = ""
    for i, p in enumerate(rotation):
        rot_html += _row(f"SP{i+1}", p)

    bp_html = ""
    for i, p in enumerate(bullpen[:4]):
        slot = "CL" if i == 0 else "SU" if i == 1 else "RP"
        bp_html += _row(slot, p)

    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Pitching Staff", "rotation & bullpen")}'
        f'<div class="to-staff-divider">Rotation</div>'
        f'<div class="to-staff">{rot_html}</div>'
        f'<div class="to-staff-divider">Bullpen &middot; High Leverage</div>'
        f'<div class="to-staff">{bp_html}</div>'
        f'</div>'
    )


def _render_health_card(il_list: list[dict], il_days: int | None, avg_age: float | None) -> str:
    org_cells = ""
    if il_days is not None:
        org_cells += (
            f'<div class="to-org-cell">'
            f'<span class="v">{il_days:,}</span>'
            f'<span class="l">IL Days \'{PRIOR_SEASON % 100}</span>'
            f'</div>'
        )
    if avg_age is not None:
        org_cells += (
            f'<div class="to-org-cell">'
            f'<span class="v">{avg_age:.1f}</span>'
            f'<span class="l">Avg Age</span>'
            f'</div>'
        )

    il_rows = ""
    for p in il_list:
        il_rows += (
            f'<div class="to-il-row">'
            f'<span class="nm">{esc(p["name"])}<span class="pos">{esc(p.get("pos", ""))}</span></span>'
            f'<span class="status">{esc(p.get("status", "IL"))}</span>'
            f'<span class="note">{esc(p.get("note", ""))}</span>'
            f'</div>'
        )

    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Health / Injury List")}'
        f'<div class="to-org-grid">{org_cells}</div>'
        f'<div class="to-il-list">{il_rows}</div>'
        f'</div>'
    )


def _render_org_card(
    pct_hg: float | None, farm_avg: float | None,
    n_ranked: int | None, top100: int | None,
) -> str:
    cells = ""
    if pct_hg is not None:
        cells += f'<div class="to-org-cell"><span class="v">{pct_hg:.0%}</span><span class="l">Homegrown</span></div>'
    if farm_avg is not None:
        cells += f'<div class="to-org-cell"><span class="v">{farm_avg:.1f}</span><span class="l">Farm Score</span></div>'
    if n_ranked is not None:
        cells += f'<div class="to-org-cell"><span class="v">{int(n_ranked)}</span><span class="l">Ranked</span></div>'
    if top100 is not None and int(top100) > 0:
        cells += f'<div class="to-org-cell"><span class="v">{int(top100)}</span><span class="l">Top 100</span></div>'

    return (
        f'<div class="to-card-block">'
        f'{_blk_head("Organization / Farm")}'
        f'<div class="to-org-grid">{cells}</div>'
        f'</div>'
    )


def _render_league_ranks(
    rankings: pd.DataFrame, selected_team: str,
) -> str:
    if rankings.empty:
        return ""

    header = (
        '<div class="to-league-row head">'
        '<span class="rk">RK</span>'
        '<span class="ab">TEAM</span>'
        '<span class="nm">Tier</span>'
        '<span class="v">Score</span>'
        '<span class="v">Off</span>'
        '<span class="v">Pit</span>'
        '<span class="v">Def</span>'
        '</div>'
    )

    rows_html = ""
    display = rankings.head(8)
    for _, r in display.iterrows():
        abbr = r.get("abbreviation", "")
        rank = _safe(r.get("rank"), 0)
        tier = _safe(r.get("tier"), "")
        comp = _safe(r.get("composite_score"), 0)
        off = _safe(r.get("offense_score"), 0)
        pit = _safe(r.get("pitching_score"), 0)
        defense = _safe(r.get("defense_score"), 0)
        is_this = abbr == selected_team
        row_cls = "to-league-row this" if is_this else "to-league-row"
        rk_cls = "rk top" if rank <= 4 else "rk"
        v_cls = "v gold" if is_this else "v"
        rows_html += (
            f'<div class="{row_cls}">'
            f'<span class="{rk_cls}">{int(rank)}</span>'
            f'<span class="ab">{esc(abbr)}</span>'
            f'<span class="nm">{esc(tier)}</span>'
            f'<span class="{v_cls}">{float(comp):.1f}</span>'
            f'<span class="v">{float(off):.1f}</span>'
            f'<span class="v">{float(pit):.1f}</span>'
            f'<span class="v">{float(defense):.1f}</span>'
            f'</div>'
        )

    return f'<div class="to-league-strip">{header}{rows_html}</div>'


# ── Data assembly helpers ─────────────────────────────────────────────

def _build_scores(
    tp: pd.Series, all_tp: pd.DataFrame, all_tr: pd.DataFrame,
    selected_team: str,
) -> list[dict]:
    """Build diamond sub-scores list from team profiles + rankings."""
    scores = []

    for col, label in _SCORE_KEYS:
        v = _safe(tp.get(col) if not tp.empty else None, 5.0)
        # Compute league rank
        rank = 15
        if col in all_tp.columns:
            ranked = all_tp[col].rank(ascending=False, method="min")
            idx = all_tp.index[all_tp["abbreviation"] == selected_team]
            if len(idx) > 0:
                rank = int(ranked.loc[idx].iloc[0])
        scores.append({
            "key": col, "label": label,
            "v": float(v), "rank": rank, "lg_avg": 5.0,
        })

    # Defense from team_rankings
    for col, label in [("defense_score", "Fielding"), ("health_depth_score", "Depth")]:
        v = 5.0
        rank = 15
        if col in all_tr.columns:
            tr_row = all_tr[all_tr["abbreviation"] == selected_team]
            if not tr_row.empty:
                v = _safe(tr_row.iloc[0].get(col), 5.0)
            ranked = all_tr[col].rank(ascending=False, method="min")
            idx = all_tr.index[all_tr["abbreviation"] == selected_team]
            if len(idx) > 0:
                rank = int(ranked.loc[idx].iloc[0])
        scores.append({
            "key": col, "label": label,
            "v": float(v), "rank": rank, "lg_avg": 5.0,
        })

    return scores


def _build_strengths_weaknesses(tr: pd.Series) -> tuple[list[dict], list[dict]]:
    """Build strengths and weaknesses lists from team_rankings data."""
    strengths = []
    weaknesses = []

    # Define metrics: (field, label, format, lg_avg, higher_is_better)
    metrics = [
        ("rpg", "Run scoring", "{:.2f} R/G", 4.5, True),
        ("hr_per_game", "Power", "{:.2f} HR/G", 1.12, True),
        ("k_per_game", "Strikeout stuff", "{:.1f} K/G", 8.7, True),
        ("ra_per_game", "Run prevention", "{:.2f} RA/G", 4.5, False),
        ("team_oaa", "Defense (OAA)", "{:+.0f} OAA", 0, True),
        ("avg_age", "Roster age", "{:.1f} yrs", 28.4, False),
    ]

    for field, label, fmt, lg_avg, higher_better in metrics:
        val = _safe(tr.get(field))
        if val is None:
            continue
        val = float(val)
        diff = val - lg_avg
        if not higher_better:
            diff = -diff

        # Compute display delta
        raw_diff = val - lg_avg
        if isinstance(raw_diff, float):
            delta_str = f"{raw_diff:+.2f}" if abs(raw_diff) < 10 else f"{raw_diff:+.0f}"
        else:
            delta_str = f"{raw_diff:+.0f}"

        entry = {
            "label": label,
            "metric": fmt.format(val),
            "lg": f"{lg_avg} lg",
            "delta": delta_str,
            "rank": 15,  # placeholder
        }

        if diff > 0:
            strengths.append(entry)
        else:
            weaknesses.append(entry)

    return strengths[:4], weaknesses[:4]


def _build_top_players(
    selected_team: str, team_pids: set[int],
) -> list[dict]:
    """Build top 10 roster players sorted by score."""
    from lib.diamond_rating import score_to_diamonds

    hr = load_rankings("hitters")
    pr = load_rankings("pitchers")
    h_arch = load_hitter_archetypes()
    p_arch = load_pitcher_archetypes()

    h_arch_map = {}
    if not h_arch.empty and "batter_id" in h_arch.columns:
        h_arch_map = dict(zip(h_arch["batter_id"].astype(int), h_arch["archetype_name"]))
    p_arch_map = {}
    if not p_arch.empty and "pitcher_id" in p_arch.columns:
        p_arch_map = dict(zip(p_arch["pitcher_id"].astype(int), p_arch["archetype_name"]))

    players: list[dict] = []

    if not hr.empty:
        pid_col = "batter_id" if "batter_id" in hr.columns else "player_id"
        score_col = next((c for c in ("current_value_score", "tdd_value_score") if c in hr.columns), None)
        if score_col:
            team_h = hr[hr[pid_col].isin(team_pids)]
            for _, row in team_h.iterrows():
                pid = int(row[pid_col])
                players.append({
                    "name": row.get("batter_name", ""),
                    "pos": row.get("position", ""),
                    "rank": int(row.get("pos_rank", 0) or 0),
                    "score": score_to_diamonds(float(row[score_col])),
                    "arch": h_arch_map.get(pid, ""),
                })

    if not pr.empty:
        pid_col = "pitcher_id" if "pitcher_id" in pr.columns else "player_id"
        score_col = next((c for c in ("current_value_score", "tdd_value_score") if c in pr.columns), None)
        if score_col:
            team_p = pr[pr[pid_col].isin(team_pids)]
            for _, row in team_p.iterrows():
                pid = int(row[pid_col])
                players.append({
                    "name": row.get("pitcher_name", ""),
                    "pos": row.get("role", "P"),
                    "rank": int(row.get("role_rank", 0) or 0),
                    "score": score_to_diamonds(float(row[score_col])),
                    "arch": p_arch_map.get(pid, ""),
                })

    players.sort(key=lambda x: x["score"], reverse=True)
    return players[:10]


def _build_depth_chart(
    roster: pd.DataFrame, selected_team: str,
) -> dict[str, list[dict]]:
    """Build depth chart from roster data."""
    from lib.diamond_rating import score_to_diamonds

    team_roster = roster[roster["team_abbr"] == selected_team].copy()
    if team_roster.empty:
        return {}

    hr = load_rankings("hitters")
    pr = load_rankings("pitchers")

    # Build score lookup
    score_map: dict[int, float] = {}
    if not hr.empty:
        pid_col = "batter_id" if "batter_id" in hr.columns else "player_id"
        score_col = next((c for c in ("current_value_score", "tdd_value_score") if c in hr.columns), None)
        if score_col:
            for _, r in hr.iterrows():
                score_map[int(r[pid_col])] = score_to_diamonds(float(r[score_col]))

    if not pr.empty:
        pid_col = "pitcher_id" if "pitcher_id" in pr.columns else "player_id"
        score_col = next((c for c in ("current_value_score", "tdd_value_score") if c in pr.columns), None)
        if score_col:
            for _, r in pr.iterrows():
                score_map[int(r[pid_col])] = score_to_diamonds(float(r[score_col]))

    has_lineup = "lineup_position" in team_roster.columns
    has_starter = "is_depth_starter" in team_roster.columns

    depth: dict[str, list[dict]] = {}
    for pos in _HITTER_POSITIONS:
        entries: list[dict] = []
        if has_lineup and has_starter:
            starters = team_roster[
                (team_roster["lineup_position"] == pos)
                & (team_roster["is_depth_starter"] == True)  # noqa: E712
            ]
            for _, row in starters.iterrows():
                pid = int(row["player_id"])
                entries.append({
                    "name": row["player_name"],
                    "score": score_map.get(pid, 5.0),
                })
            bench = team_roster[
                (team_roster["lineup_position"] == pos)
                & (team_roster["is_depth_starter"] == False)  # noqa: E712
            ]
            for _, row in bench.iterrows():
                pid = int(row["player_id"])
                entries.append({
                    "name": row["player_name"],
                    "score": score_map.get(pid, 4.0),
                })
        if entries:
            depth[pos] = entries

    return depth


def _build_pitching_staff(
    team_pids: set[int],
) -> tuple[list[dict], list[dict]]:
    """Build rotation and bullpen lists."""
    from lib.diamond_rating import score_to_diamonds

    pr = load_rankings("pitchers")
    p_arch = load_pitcher_archetypes()
    p_proj = load_projections("pitcher")

    p_arch_map = {}
    if not p_arch.empty and "pitcher_id" in p_arch.columns:
        p_arch_map = dict(zip(p_arch["pitcher_id"].astype(int), p_arch["archetype_name"]))

    if pr.empty:
        return [], []

    pid_col = "pitcher_id" if "pitcher_id" in pr.columns else "player_id"
    score_col = next((c for c in ("current_value_score", "tdd_value_score") if c in pr.columns), None)
    if not score_col:
        return [], []

    team_p = pr[pr[pid_col].isin(team_pids)].copy()
    if team_p.empty:
        return [], []

    # Determine starter status
    is_starter_col = "is_starter" if "is_starter" in team_p.columns else None
    role_col = "role" if "role" in team_p.columns else None

    # Get ERA from projections
    era_map: dict[int, float] = {}
    hand_map: dict[int, str] = {}
    if not p_proj.empty:
        for _, r in p_proj.iterrows():
            pid = int(r.get("pitcher_id", 0))
            # Try projected ERA or actual ERA
            era = _safe(r.get("projected_era"), _safe(r.get("era")))
            if era is not None:
                era_map[pid] = float(era)
            throws = _safe(r.get("throws"))
            if throws:
                hand_map[pid] = str(throws)[0]

    rotation: list[dict] = []
    bullpen: list[dict] = []

    for _, row in team_p.iterrows():
        pid = int(row[pid_col])
        entry = {
            "name": row.get("pitcher_name", ""),
            "score": score_to_diamonds(float(row[score_col])),
            "era": era_map.get(pid),
            "hand": hand_map.get(pid, ""),
            "arch": p_arch_map.get(pid, ""),
        }

        is_sp = False
        if is_starter_col and pd.notna(row.get(is_starter_col)):
            is_sp = bool(row[is_starter_col])
        elif role_col and pd.notna(row.get(role_col)):
            is_sp = str(row[role_col]).upper() in ("SP", "STARTER")

        if is_sp:
            rotation.append(entry)
        else:
            bullpen.append(entry)

    rotation.sort(key=lambda x: x["score"], reverse=True)
    bullpen.sort(key=lambda x: x["score"], reverse=True)

    return rotation[:5], bullpen[:6]


def _build_il_list(selected_team: str) -> list[dict]:
    """Build injury list from preseason injuries data."""
    inj = load_preseason_injuries()
    if inj.empty:
        return []

    team_inj = inj[
        (inj["team_abbr"] == selected_team)
        & (inj["est_missed_games"] > 0)
    ]
    result = []
    for _, row in team_inj.iterrows():
        pos = _safe(row.get("position"), "")
        result.append({
            "name": row.get("player_name", ""),
            "pos": pos,
            "status": f"IL-{int(row.get('il_type', 15))}" if _safe(row.get("il_type")) else "IL",
            "note": _safe(row.get("injury_description"), ""),
        })
    return result[:5]


# ── Team name lookup ──────────────────────────────────────────────────

_TEAM_NAMES: dict[str, tuple[str, str]] = {
    "ARI": ("Arizona Diamondbacks", "NL West"),
    "ATL": ("Atlanta Braves", "NL East"),
    "BAL": ("Baltimore Orioles", "AL East"),
    "BOS": ("Boston Red Sox", "AL East"),
    "CHC": ("Chicago Cubs", "NL Central"),
    "CWS": ("Chicago White Sox", "AL Central"),
    "CIN": ("Cincinnati Reds", "NL Central"),
    "CLE": ("Cleveland Guardians", "AL Central"),
    "COL": ("Colorado Rockies", "NL West"),
    "DET": ("Detroit Tigers", "AL Central"),
    "HOU": ("Houston Astros", "AL West"),
    "KC":  ("Kansas City Royals", "AL Central"),
    "LAA": ("Los Angeles Angels", "AL West"),
    "LAD": ("Los Angeles Dodgers", "NL West"),
    "MIA": ("Miami Marlins", "NL East"),
    "MIL": ("Milwaukee Brewers", "NL Central"),
    "MIN": ("Minnesota Twins", "AL Central"),
    "NYM": ("New York Mets", "NL East"),
    "NYY": ("New York Yankees", "AL East"),
    "OAK": ("Oakland Athletics", "AL West"),
    "PHI": ("Philadelphia Phillies", "NL East"),
    "PIT": ("Pittsburgh Pirates", "NL Central"),
    "SD":  ("San Diego Padres", "NL West"),
    "SF":  ("San Francisco Giants", "NL West"),
    "SEA": ("Seattle Mariners", "AL West"),
    "STL": ("St. Louis Cardinals", "NL Central"),
    "TB":  ("Tampa Bay Rays", "AL East"),
    "TEX": ("Texas Rangers", "AL West"),
    "TOR": ("Toronto Blue Jays", "AL East"),
    "WSH": ("Washington Nationals", "NL East"),
}


# ═══════════════════════════════════════════════════════════════════════
# Main page
# ═══════════════════════════════════════════════════════════════════════

def page_team_overview() -> None:
    """Team Overview -- Press Box data wall."""

    # ── Team selector ──────────────────────────────────────────────
    teams_df = load_roster()
    if teams_df.empty:
        tdd_warn("No team data found. Run precompute first.")
        return

    all_abbrs = sorted(
        teams_df["team_abbr"].replace("", pd.NA).dropna().unique().tolist()
    )
    display_names = [team_full(a) for a in all_abbrs]
    qp_team = st.query_params.get("team", "")
    default_idx = 0
    if qp_team in all_abbrs:
        default_idx = all_abbrs.index(qp_team)
    selected_name = st.selectbox(
        "Select team", display_names, index=default_idx, key="team_select",
        label_visibility="collapsed",
    )
    selected_team = abbr_from_full(selected_name)
    st.query_params["team"] = selected_team

    # ── Load data ──────────────────────────────────────────────────
    rankings = load_team_rankings()
    profiles = load_team_profiles()
    elo_df = load_team_elo(preseason=True)
    standings = load_standings()

    tr_row = rankings[rankings["abbreviation"] == selected_team] if not rankings.empty else pd.DataFrame()
    tp_row = profiles[profiles["abbreviation"] == selected_team] if not profiles.empty else pd.DataFrame()
    te_row = elo_df[elo_df["abbreviation"] == selected_team] if not elo_df.empty else pd.DataFrame()

    tr = tr_row.iloc[0] if not tr_row.empty else pd.Series(dtype=float)
    tp = tp_row.iloc[0] if not tp_row.empty else pd.Series(dtype=float)
    te = te_row.iloc[0] if not te_row.empty else pd.Series(dtype=float)

    team_pids = set(teams_df[teams_df["team_abbr"] == selected_team]["player_id"].astype(int))
    record = standings.get(selected_team)

    team_info = _TEAM_NAMES.get(selected_team, (selected_team, ""))
    team_name, division = team_info

    tier = _safe(tr.get("tier"), "Competitive") if not tr.empty else "Competitive"
    rank = int(_safe(tr.get("rank"), 0)) if not tr.empty else 0
    div_rank = int(_safe(tr.get("div_rank"), 0)) if not tr.empty else 0

    # ── Assemble page HTML ─────────────────────────────────────────
    html_parts: list[str] = ['<div class="to-page">']

    # Eyebrow
    html_parts.append(_render_eyebrow(division, team_name))

    # Team header masthead
    team_id = int(tr.get("team_id", 0)) if not tr.empty and "team_id" in tr.index else None
    html_parts.append(_render_team_header(
        selected_team, team_name, division, record, tr, te,
        team_id=team_id,
    ))

    # Tier row
    html_parts.append(_render_tier_row(tier, div_rank if div_rank else None))

    # ── Section 01: Profile ────────────────────────────────────────
    html_parts.append(_section("01", "Profile", "identity / scores / composite"))

    scores = _build_scores(tp, profiles, rankings, selected_team)

    # Get style info
    off_style = _safe(tr.get("offense_style"), "") if not tr.empty else ""
    pit_style = _safe(tr.get("pitching_style"), "") if not tr.empty else ""
    age_traj = _safe(tr.get("age_trajectory"), "") if not tr.empty else ""

    identity_html = _render_identity_card(
        tier, rank if rank else None, division, div_rank if div_rank else None,
        off_style, pit_style, age_traj,
    )
    scores_html = _render_scores_card(scores)
    radar_html = _render_radar_card(scores)

    html_parts.append(
        f'<div class="to-row-identity">'
        f'{identity_html}{scores_html}{radar_html}'
        f'</div>'
    )

    # ── Section 02: Strengths & Weaknesses ─────────────────────────
    html_parts.append(_section("02", "Strengths & Weaknesses", "vs MLB average"))

    strengths, weaknesses = _build_strengths_weaknesses(tr)
    html_parts.append(
        f'<div class="to-row-sw">'
        f'{_render_sw_list(strengths, "up")}'
        f'{_render_sw_list(weaknesses, "down")}'
        f'</div>'
    )

    # ── Section 03: Roster ─────────────────────────────────────────
    html_parts.append(_section("03", "Roster", "position players / pitching"))

    depth = _build_depth_chart(teams_df, selected_team)
    rotation, bullpen = _build_pitching_staff(team_pids)
    top_players = _build_top_players(selected_team, team_pids)

    html_parts.append(
        f'<div class="to-row-identity" style="grid-template-columns:1.2fr 1fr 1fr">'
        f'{_render_depth_field(depth)}'
        f'{_render_pitching_staff(rotation, bullpen)}'
        f'{_render_top_players(top_players)}'
        f'</div>'
    )

    # ── Section 04: Schedule & Context ─────────────────────────────
    html_parts.append(_section("04", "Context", "health / organization"))

    il_list = _build_il_list(selected_team)
    il_days = int(_safe(tr.get("total_il_days"))) if _safe(tr.get("total_il_days")) is not None else None
    avg_age = float(_safe(tr.get("avg_age"))) if _safe(tr.get("avg_age")) is not None else None
    pct_hg = float(_safe(tr.get("pct_homegrown"))) if _safe(tr.get("pct_homegrown")) is not None else None
    farm_avg = float(_safe(tr.get("farm_avg_score"))) if _safe(tr.get("farm_avg_score")) is not None else None
    n_ranked = int(_safe(tr.get("n_ranked_prospects"))) if _safe(tr.get("n_ranked_prospects")) is not None else None
    top100 = int(_safe(tr.get("top_100_count"))) if _safe(tr.get("top_100_count")) is not None else None

    html_parts.append(
        f'<div class="to-row-sw" style="margin-top:0.6rem">'
        f'{_render_health_card(il_list, il_days, avg_age)}'
        f'{_render_org_card(pct_hg, farm_avg, n_ranked, top100)}'
        f'</div>'
    )

    # ── Section 05: League Standing ────────────────────────────────
    html_parts.append(_section("05", "League Standing", "top 8 by composite"))

    if not rankings.empty:
        sort_col = "rank" if "rank" in rankings.columns else "composite_score"
        asc = True if sort_col == "rank" else False
        sorted_ranks = rankings.sort_values(sort_col, ascending=asc)
        html_parts.append(_render_league_ranks(sorted_ranks, selected_team))

    html_parts.append('</div>')  # close .to-page

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)
