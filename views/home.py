"""Home page -- daily dashboard overview with click-through sections."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape

import pandas as pd
import streamlit as st

from config import CURRENT_SEASON
from services.data_loader import (
    load_game_props,
    load_hitters_daily_standouts,
    load_todays_batter_sims,
    load_todays_games,
    load_update_metadata,
    load_backtest,
    load_dk_props,
)
from utils.team_names import team_short


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nav_link(page_name: str, text: str) -> str:
    """Return an anchor tag that navigates to a dashboard page."""
    slug = page_name.lower().replace(" ", "_")
    return f'<a href="?page={slug}" target="_self" class="home-cta">{escape(text)} <span class="home-cta-arrow">&rarr;</span></a>'


def _nav_url(page_name: str) -> str:
    slug = page_name.lower().replace(" ", "_")
    return f"?page={slug}"


def _format_odds(odds: str | float | None) -> str:
    if odds is None or (isinstance(odds, float) and pd.isna(odds)):
        return ""
    s = str(odds)
    # Ensure leading + for positive
    if s and s[0].isdigit():
        s = "+" + s
    return s


def _edge_direction(edge: float) -> str:
    return "over" if edge > 0 else "under"


def _stat_chip_color(stat: str) -> tuple[str, str]:
    """Return (background, text-color) for a stat chip."""
    colors = {
        "H": ("var(--tdd-gold)", "var(--tdd-dark)"),
        "HR": ("var(--tdd-ember)", "var(--tdd-dark)"),
        "TB": ("var(--tdd-gold)", "var(--tdd-dark)"),
        "HRR": ("var(--tdd-gold)", "var(--tdd-dark)"),
        "BB": ("var(--tdd-sage)", "var(--tdd-dark)"),
        "Outs": ("var(--tdd-sage)", "var(--tdd-dark)"),
        "R": ("var(--tdd-gold)", "var(--tdd-dark)"),
        "RBI": ("var(--tdd-gold)", "var(--tdd-dark)"),
        "K": ("transparent", "var(--tdd-slate)"),
    }
    return colors.get(stat, ("var(--tdd-slate)", "var(--tdd-dark)"))


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _assemble_home_data() -> dict:
    """Pull together all data needed for the home page from existing loaders."""
    meta = load_update_metadata()
    games = load_todays_games()
    props = load_game_props()
    dk = load_dk_props()
    batter_sims = load_todays_batter_sims()
    standouts = load_hitters_daily_standouts()

    game_date = meta.get("game_date", datetime.now().strftime("%Y-%m-%d"))

    # --- Today's props ---
    today_props = props[props["game_date"] == game_date] if not props.empty else pd.DataFrame()

    # --- Schedule (games only, no predictions) ---
    schedule = []
    if not games.empty:
        for _, row in games.iterrows():
            schedule.append({
                "game_pk": row["game_pk"],
                "time": row.get("game_time", ""),
                "status": row.get("status", "scheduled"),
                "away": row.get("away_abbr", ""),
                "home": row.get("home_abbr", ""),
                "away_pitcher": row.get("away_pitcher_name", ""),
                "home_pitcher": row.get("home_pitcher_name", ""),
            })

    # --- Featured game predictions (DISABLED) ---
    # TODO: Re-enable when game-level prediction accuracy is validated.
    # The old hero section with win%, projected runs, and pitcher props is
    # commented out until we are confident in the model output.
    # See _render_hero() for the original implementation.

    # --- Best matchups of the day (top 2 games by total projected TB) ---
    best_matchups: list[dict] = []
    if not batter_sims.empty and not games.empty:
        game_agg = (
            batter_sims
            .groupby("game_pk")
            .agg(
                sum_tb=("expected_tb", "sum"),
                sum_h=("expected_h", "sum"),
                n_hitters=("batter_id", "count"),
            )
            .sort_values("sum_tb", ascending=False)
        )
        for gpk in game_agg.index[:2]:
            gi = games[games["game_pk"] == gpk]
            if gi.empty:
                continue
            gi = gi.iloc[0]
            g_sims = batter_sims[batter_sims["game_pk"] == gpk]
            top3 = g_sims.nlargest(3, "expected_h")
            star_hitters = []
            for _, h in top3.iterrows():
                star_hitters.append({
                    "name": h["batter_name"],
                    "team": h["team_abbr"],
                    "proj_h": round(h["expected_h"], 2),
                    "proj_tb": round(h["expected_tb"], 2),
                    "p_h_1": round(h.get("p_h_over_0_5", 0) * 100),
                })
            best_matchups.append({
                "game_pk": int(gpk),
                "time": gi.get("game_time", ""),
                "away": gi.get("away_abbr", ""),
                "home": gi.get("home_abbr", ""),
                "away_pitcher": gi.get("away_pitcher_name", "") or "TBD",
                "home_pitcher": gi.get("home_pitcher_name", "") or "TBD",
                "sum_tb": round(float(game_agg.loc[gpk, "sum_tb"]), 1),
                "sum_h": round(float(game_agg.loc[gpk, "sum_h"]), 1),
                "n_hitters": int(game_agg.loc[gpk, "n_hitters"]),
                "stars": star_hitters,
            })

    # --- Top hitters today (from batter sims) ---
    top_hitters = []
    if not batter_sims.empty:
        ranked = batter_sims.nlargest(8, "expected_h")
        for _, h in ranked.iterrows():
            top_hitters.append({
                "name": h.get("batter_name", ""),
                "team": h.get("team_abbr", ""),
                "opp": h.get("opp_abbr", ""),
                "order": int(h.get("batting_order", 0)),
                "proj_h": round(h.get("expected_h", 0), 2),
                "proj_hr": round(h.get("expected_hr", 0), 2),
                "proj_tb": round(h.get("expected_tb", 0), 2),
                "proj_bb": round(h.get("expected_bb", 0), 2),
                "p_h_1": round(h.get("p_h_over_0_5", 0) * 100),
                "p_h_2": round(h.get("p_h_over_1_5", 0) * 100),
                "p_hr_1": round(h.get("p_hr_over_0_5", 0) * 100),
            })

    # --- Top props edges ---
    top_edges = []
    if not today_props.empty:
        with_edge = today_props[
            today_props["model_edge"].notna()
            & today_props["vegas_line"].notna()
        ].copy()
        if not with_edge.empty:
            with_edge["abs_edge"] = with_edge["model_edge"].abs()
            best_edges = with_edge.nlargest(8, "abs_edge")
            for _, e in best_edges.iterrows():
                top_edges.append({
                    "name": e.get("player_name", ""),
                    "player_type": e.get("player_type", ""),
                    "team": e.get("team", ""),
                    "stat": e.get("stat", ""),
                    "expected": round(e["expected"], 2),
                    "line": e.get("vegas_line"),
                    "edge": round(e["model_edge"], 3),
                    "odds": e.get("vegas_odds", ""),
                    "direction": _edge_direction(e["model_edge"]),
                })

    # --- Model performance from backtests ---
    bt_summary = load_backtest("game_prop_summary")
    model_stats = {}
    if not bt_summary.empty:
        latest = bt_summary[bt_summary["test_season"] == bt_summary["test_season"].max()]
        for _, row in latest.iterrows():
            key = f"{row.get('side', '')}_{row.get('stat', '')}"
            model_stats[key] = {
                "ece": round(row["ece"], 3) if pd.notna(row.get("ece")) else None,
                "coverage_80": round(row["coverage_80"] * 100, 1) if pd.notna(row.get("coverage_80")) else None,
                "brier": round(row["avg_brier"], 3) if pd.notna(row.get("avg_brier")) else None,
            }

    # --- Yesterday standouts ---
    yesterday_hit = None
    yesterday_miss = None
    if not standouts.empty:
        best = standouts.iloc[0]
        yesterday_hit = {
            "player": best.get("batter_name", ""),
            "line": f"{best.get('hits', 0)}-for-{best.get('ab', 0)}, {best.get('hr', 0)} HR, {best.get('rbi', 0)} RBI",
            "score": round(best.get("daily_standout_score", 0), 2),
        }

    # --- Data freshness ---
    last_updated = meta.get("last_updated", "")
    last_schedule = meta.get("last_schedule_refresh", "")

    data_feeds = []
    if last_schedule:
        try:
            sched_dt = datetime.fromisoformat(last_schedule)
            age = datetime.now() - sched_dt
            age_str = f"{int(age.total_seconds() // 60)} min" if age < timedelta(hours=1) else f"{int(age.total_seconds() // 3600)} hr"
            data_feeds.append({"name": "Lineups", "age": age_str, "status": "ok" if age < timedelta(hours=1) else "warn"})
        except Exception:
            data_feeds.append({"name": "Lineups", "age": "-", "status": "warn"})
    if last_updated:
        try:
            upd_dt = datetime.fromisoformat(last_updated)
            age = datetime.now() - upd_dt
            age_str = f"{int(age.total_seconds() // 60)} min" if age < timedelta(hours=1) else f"{int(age.total_seconds() // 3600)} hr"
            data_feeds.append({"name": "Projections", "age": age_str, "status": "ok" if age < timedelta(hours=24) else "warn"})
        except Exception:
            data_feeds.append({"name": "Projections", "age": "-", "status": "warn"})

    # DK props freshness
    if not dk.empty and "game_date" in dk.columns:
        has_today = (dk["game_date"] == game_date).any()
        data_feeds.append({
            "name": "Market",
            "age": "today" if has_today else "stale",
            "status": "ok" if has_today else "warn",
        })

    return {
        "game_date": game_date,
        "meta": meta,
        "schedule": schedule,
        "best_matchups": best_matchups,
        "top_hitters": top_hitters,
        "top_edges": top_edges,
        "model_stats": model_stats,
        "yesterday_hit": yesterday_hit,
        "yesterday_miss": yesterday_miss,
        "data_feeds": data_feeds,
        "n_games": len(schedule),
        "n_hitters": len(batter_sims) if not batter_sims.empty else 0,
    }


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _render_ticker(data: dict) -> str:
    """Scrolling ticker strip at top."""
    items = []
    for g in data["schedule"][:6]:
        status = g["status"].lower()
        if "progress" in status or "live" in status:
            score_html = f'<span class="home-ticker-score">{escape(str(g.get("status", "")))}</span>'
        elif "final" in status:
            score_html = '<span class="home-ticker-score">FINAL</span>'
        else:
            score_html = f'<span class="home-ticker-score">{escape(g["time"])}</span>'
        items.append(
            f'<span class="home-ticker-item">'
            f'<span data-team="{escape(g["away"])}">{escape(team_short(g["away"]))}</span>'
            f' @ '
            f'<span data-team="{escape(g["home"])}">{escape(team_short(g["home"]))}</span>'
            f' {score_html}'
            f'<span class="home-ticker-dot">&middot;</span>'
            f'</span>'
        )
    # Duplicate for seamless loop
    all_items = "".join(items) * 2
    return f'''
    <div class="home-ticker">
        <span class="home-ticker-live-dot"></span>
        <span class="home-ticker-label">LIVE</span>
        <div class="home-ticker-items">{all_items}</div>
    </div>
    '''


def _render_masthead(data: dict) -> str:
    """Date masthead with slate-level stats."""
    try:
        d = datetime.strptime(data["game_date"], "%Y-%m-%d")
        date_str = d.strftime("%a %b %d")
        year_str = d.strftime("%Y")
    except Exception:
        date_str = data["game_date"]
        year_str = ""

    return f'''
    <div class="home-masthead">
        <div>
            <div class="home-masthead-date">{date_str}<span class="home-masthead-ord">&middot; {year_str}</span></div>
        </div>
        <div class="home-masthead-stats">
            <div class="home-masthead-stat">
                <div class="home-masthead-stat-v">{data["n_games"]}</div>
                <div class="home-masthead-stat-l">Games</div>
            </div>
            <div class="home-masthead-stat">
                <div class="home-masthead-stat-v">{data["n_hitters"]}</div>
                <div class="home-masthead-stat-l">Hitters</div>
            </div>
        </div>
    </div>
    '''


def _render_hero(featured: dict) -> str:
    """Featured matchup hero section."""
    if not featured:
        return ""

    f = featured
    away_stats = f.get("away_stats", {})
    home_stats = f.get("home_stats", {})

    def _pitcher_stats_html(stats: dict) -> str:
        parts = []
        for stat, label in [("Outs", "Outs"), ("H", "H"), ("HR", "HR"), ("BB", "BB")]:
            v = stats.get(stat, "-")
            parts.append(f'<span><span class="home-hero-pl">{label}</span> <span class="home-hero-pk">{v}</span></span>')
        k_val = stats.get("K", "-")
        parts.append(f'<span class="home-hero-dim"><span class="home-hero-pl">K</span> <span class="home-hero-pk">{k_val}</span></span>')
        return " ".join(parts)

    win_label = f["away"] if f["p_away_win"] > f["p_home_win"] else f["home"]
    win_pct = max(f["p_away_win"], f["p_home_win"])

    return f'''
    <a href="{_nav_url("Game Analysis")}" target="_self" class="home-hero-link">
    <div class="home-hero">
        <div class="home-hero-left">
            <div class="home-hero-eyebrow">
                <span class="home-hero-dot"></span>
                <span>Featured Matchup</span>
                <span class="home-hero-time">&middot; {escape(f["time"])}</span>
            </div>
            <div class="home-hero-teams">
                <div class="home-hero-team home-hero-team-away">
                    <div class="home-hero-abbr" data-team="{escape(f["away"])}">{escape(team_short(f["away"]))}</div>
                </div>
                <div class="home-hero-at">@</div>
                <div class="home-hero-team home-hero-team-home">
                    <div class="home-hero-abbr" data-team="{escape(f["home"])}">{escape(team_short(f["home"]))}</div>
                </div>
            </div>
            <div class="home-hero-pitchers">
                <div class="home-hero-pitcher">
                    <div class="home-hero-role">Away</div>
                    <div class="home-hero-pname">{escape(f["away_pitcher"])}</div>
                    <div class="home-hero-pline">{_pitcher_stats_html(away_stats)}</div>
                </div>
                <div class="home-hero-pitcher home-hero-pitcher-right">
                    <div class="home-hero-role">Home</div>
                    <div class="home-hero-pname">{escape(f["home_pitcher"])}</div>
                    <div class="home-hero-pline">{_pitcher_stats_html(home_stats)}</div>
                </div>
            </div>
        </div>
        <div class="home-hero-right">
            <div class="home-hero-kheadline">
                <div class="home-hero-lbl">Projected combined runs</div>
                <div class="home-hero-big">{f["proj_runs"]}</div>
                <div class="home-hero-ci">
                    <span>({f["proj_runs_lo"]}&ndash;{f["proj_runs_hi"]})</span>
                </div>
            </div>
            <div class="home-hero-metrics">
                <div class="home-hero-m">
                    <div class="home-hero-ml">Win%</div>
                    <div class="home-hero-mv home-hero-gold">{win_label} {win_pct}%</div>
                </div>
                <div class="home-hero-m">
                    <div class="home-hero-ml">Total</div>
                    <div class="home-hero-mv">{f["proj_runs"]}</div>
                </div>
                <div class="home-hero-m">
                    <div class="home-hero-ml">Away R</div>
                    <div class="home-hero-mv">{f["away_runs"]}</div>
                </div>
            </div>
            <div class="home-hero-cta">Open Game Analysis <span class="home-cta-arrow">&rarr;</span></div>
        </div>
    </div>
    </a>
    '''


def _render_best_matchups(matchups: list[dict]) -> str:
    """Two best matchups of the day based on hitter projections."""
    if not matchups:
        return ""

    cards = []
    for m in matchups[:2]:
        # Star hitters list
        stars_html = ""
        for s in m["stars"]:
            stars_html += (
                f'<div class="home-mu-star">'
                f'<span class="home-mu-star-nm" data-team="{escape(s["team"])}">{escape(s["name"])}</span>'
                f'<span class="home-mu-star-stat">{s["proj_h"]} H</span>'
                f'<span class="home-mu-star-ci">{s["p_h_1"]}%</span>'
                f'</div>'
            )

        cards.append(f'''
        <a href="{_nav_url("Schedule")}" target="_self" class="home-mu-card">
            <div class="home-mu-header">
                <div class="home-mu-teams">
                    <span class="home-mu-abbr" data-team="{escape(m["away"])}">{escape(team_short(m["away"]))}</span>
                    <span class="home-mu-at">@</span>
                    <span class="home-mu-abbr" data-team="{escape(m["home"])}">{escape(team_short(m["home"]))}</span>
                </div>
                <div class="home-mu-time">{escape(m["time"])}</div>
            </div>
            <div class="home-mu-pitchers">
                <span>{escape(m["away_pitcher"])}</span>
                <span class="home-mu-vs">vs</span>
                <span>{escape(m["home_pitcher"])}</span>
            </div>
            <div class="home-mu-agg">
                <span class="home-mu-agg-v">{m["sum_h"]}</span>
                <span class="home-mu-agg-l">Proj H</span>
                <span class="home-mu-agg-v">{m["sum_tb"]}</span>
                <span class="home-mu-agg-l">Proj TB</span>
                <span class="home-mu-agg-v">{m["n_hitters"]}</span>
                <span class="home-mu-agg-l">Hitters</span>
            </div>
            <div class="home-mu-stars-label">Top hitters</div>
            <div class="home-mu-stars">{stars_html}</div>
        </a>
        ''')

    return (
        f'<div class="home-mu-section">'
        f'<div class="home-mu-eyebrow">Best Matchups Today</div>'
        f'<div class="home-mu-row">{"".join(cards)}</div>'
        f'</div>'
    )


def _render_quick_rail(data: dict) -> str:
    """Quick-nav tile rail below hero."""
    tiles = []

    # Games tile
    tiles.append(f'''
        <a href="{_nav_url("Schedule")}" target="_self" class="home-rail-tile">
            <div class="home-rail-l">Games</div>
            <div class="home-rail-t">{data["n_games"]} today</div>
            <div class="home-rail-v">{data["n_hitters"]} hitters</div>
        </a>
    ''')

    # Top hitter tile
    if data["top_hitters"]:
        top = data["top_hitters"][0]
        tiles.append(f'''
            <a href="{_nav_url("Schedule")}" target="_self" class="home-rail-tile">
                <div class="home-rail-l">Top Hitter</div>
                <div class="home-rail-t">{escape(top["name"])}</div>
                <div class="home-rail-v">{top["proj_h"]} H &middot; {top["p_h_1"]}% for 1+</div>
            </a>
        ''')

    # Props tile -- link only, no edge advice while WIP
    tiles.append(f'''
        <a href="{_nav_url("Props Lab")}" target="_self" class="home-rail-tile">
            <div class="home-rail-l">Props</div>
            <div class="home-rail-t">Props Lab</div>
            <div class="home-rail-v">Explore &rarr;</div>
        </a>
    ''')

    # Model tile
    model = data.get("model_stats", {})
    batter_h = model.get("batter_H", model.get("batter_h", {}))
    ece = batter_h.get("ece", "-")
    tiles.append(f'''
        <a href="{_nav_url("Model Performance")}" target="_self" class="home-rail-tile">
            <div class="home-rail-l">Calibration</div>
            <div class="home-rail-t">Batter H</div>
            <div class="home-rail-v">ECE {ece}</div>
        </a>
    ''')

    # Methodology tile
    tiles.append(f'''
        <a href="{_nav_url("Methodology")}" target="_self" class="home-rail-tile">
            <div class="home-rail-l">Method</div>
            <div class="home-rail-t">Bayesian</div>
            <div class="home-rail-v">Multi-stat</div>
        </a>
    ''')

    return f'<div class="home-rail">{"".join(tiles)}</div>'


def _render_section_head(title: str, sub: str, cta_text: str, cta_page: str) -> str:
    cta = _nav_link(cta_page, cta_text) if cta_text else ""
    sub_html = f'<span class="home-sec-sub">{escape(sub)}</span>' if sub else ""
    return f'''
    <div class="home-sec-head">
        <div class="home-sec-left">
            <h3>{escape(title)}</h3>
            {sub_html}
        </div>
        {cta}
    </div>
    '''


def _render_schedule_strip(schedule: list[dict]) -> str:
    """Compact game list for the home page."""
    rows = []
    for g in schedule[:6]:
        status_cls = "live" if "progress" in g["status"].lower() else g["status"].lower()
        if "progress" in g["status"].lower():
            status_html = f'&bull; LIVE'
        elif "final" in g["status"].lower():
            status_html = "FINAL"
        else:
            status_html = escape(g["time"])

        pitcher_away = escape(g.get("away_pitcher", "")) or "TBD"
        pitcher_home = escape(g.get("home_pitcher", "")) or "TBD"
        pitcher_html = f'{pitcher_away} vs {pitcher_home}'

        rows.append(f'''
        <a href="{_nav_url("Schedule")}" target="_self" class="home-game-card">
            <div class="home-game-time">{escape(g["time"])}</div>
            <div class="home-game-teams">
                <span data-team="{escape(g["away"])}" class="home-game-abbr">{escape(team_short(g["away"]))}</span>
                <span class="home-game-at">@</span>
                <span data-team="{escape(g["home"])}" class="home-game-abbr">{escape(team_short(g["home"]))}</span>
            </div>
            <div class="home-game-detail">
                <span class="home-game-pitchers">{pitcher_html}</span>
                <span class="home-game-status home-game-status-{status_cls}">{status_html}</span>
            </div>
        </a>
        ''')
    return f'<div class="home-game-list">{"".join(rows)}</div>'


def _render_top_hitters(hitters: list[dict]) -> str:
    """Top projected hitters for today with confidence intervals."""
    if not hitters:
        return '<div class="home-perf-empty">No hitter sims available for today</div>'
    rows = []
    for i, h in enumerate(hitters[:6]):
        rk_cls = "home-perf-rk-top" if i < 3 else ""
        rows.append(f'''
        <a href="{_nav_url("Schedule")}" target="_self" class="home-perf-row">
            <div class="home-perf-rk {rk_cls}">{i + 1}</div>
            <div class="home-perf-who">
                <div class="home-perf-nm">{escape(h["name"])}</div>
                <div class="home-perf-meta">
                    <span data-team="{escape(h["team"])}">{escape(h["team"])}</span>
                    vs {escape(h["opp"])} &middot; #{h["order"]}
                </div>
            </div>
            <div class="home-perf-proj">
                <div class="home-perf-main">{h["proj_h"]} H</div>
                <div class="home-perf-sub">{h["proj_hr"]} HR &middot; {h["proj_tb"]} TB &middot; {h["proj_bb"]} BB</div>
                <div class="home-perf-ci">{h["p_h_1"]}% for 1+ H &middot; {h["p_h_2"]}% for 2+</div>
            </div>
        </a>
        ''')
    return "".join(rows)


def _render_edges_list(edges: list[dict]) -> str:
    """Props edges list with stat-colored chips."""
    rows = []
    for e in edges[:6]:
        bg, fg = _stat_chip_color(e["stat"])
        border = "var(--tdd-slate)" if e["stat"] == "K" else bg
        dir_cls = "home-edge-over" if e["direction"] == "over" else "home-edge-under"
        edge_sign = "+" if e["edge"] > 0 else ""

        name_display = e["name"]
        # player_name might be an ID, try to use it as-is
        if isinstance(name_display, (int, float)):
            name_display = str(int(name_display))

        rows.append(f'''
        <a href="{_nav_url("Props Lab")}" target="_self" class="home-edge-row">
            <div class="home-edge-chip" style="background:{bg};color:{fg};border:1px solid {border};">{escape(e["stat"])}</div>
            <div class="home-edge-who">
                <div class="home-edge-nm">{escape(str(name_display))}</div>
                <div class="home-edge-meta">
                    <span data-team="{escape(e["team"])}">{escape(e["team"])}</span>
                    &middot; Proj {e["expected"]} &middot; Line {e["line"]}
                </div>
            </div>
            <div class="home-edge-prop">
                <div class="home-edge-val {dir_cls}">{edge_sign}{e["edge"]:.2f} {"&uarr;" if e["direction"] == "over" else "&darr;"}</div>
            </div>
        </a>
        ''')
    return "".join(rows)


def _render_model_perf(model_stats: dict) -> str:
    """Model performance strip using calibration metrics.

    ECE (Expected Calibration Error) measures how well the model's predicted
    probabilities match observed hit rates. Lower is better -- 0.0 is perfect.
    Coverage shows how often actual outcomes fall within the model's predicted
    confidence intervals. 80% coverage at 80% CI means the model is well-calibrated.
    """
    tiles = []

    # Batter H calibration -- our best-calibrated stat
    bh = model_stats.get("batter_H", model_stats.get("batter_h", {}))
    ece_val = bh.get("ece", "-")
    tiles.append(f'''
        <div class="home-model-m">
            <div class="home-model-l">Batter H ECE</div>
            <div class="home-model-v home-model-v-gold">{ece_val}</div>
            <div class="home-model-sub home-model-sage">0.0 = perfect</div>
        </div>
    ''')

    # Batter H coverage
    cov_val = bh.get("coverage_80", "-")
    cov_display = f"{cov_val}%" if cov_val != "-" else "-"
    tiles.append(f'''
        <div class="home-model-m">
            <div class="home-model-l">H 80% Coverage</div>
            <div class="home-model-v">{cov_display}</div>
            <div class="home-model-sub">target 80%</div>
        </div>
    ''')

    # Batter BB calibration
    bbb = model_stats.get("batter_bb", model_stats.get("batter_BB", {}))
    tiles.append(f'''
        <div class="home-model-m">
            <div class="home-model-l">Batter BB ECE</div>
            <div class="home-model-v">{bbb.get("ece", "-")}</div>
            <div class="home-model-sub home-model-sage">0.0 = perfect</div>
        </div>
    ''')

    # Pitcher K calibration (dimmed -- noisy)
    pk = model_stats.get("pitcher_k", model_stats.get("pitcher_K", {}))
    tiles.append(f'''
        <div class="home-model-m home-model-dim">
            <div class="home-model-l">Pitcher K ECE</div>
            <div class="home-model-v">{pk.get("ece", "-")}</div>
            <div class="home-model-sub">noisier signal</div>
        </div>
    ''')

    explainer = (
        '<div class="home-model-explainer">'
        '<b>ECE</b> (Expected Calibration Error) measures how well predicted probabilities '
        'match actual outcomes. Lower = better. <b>Coverage</b> is how often actuals fall '
        'within the model\'s confidence intervals.'
        '</div>'
    )

    return f'<div class="home-model-strip">{"".join(tiles)}</div>{explainer}'


def _render_hit_miss(hit: dict | None, miss: dict | None) -> str:
    """Yesterday's biggest hit and miss."""
    hit_html = ""
    if hit:
        hit_html = f'''
        <div class="home-hm-side">
            <div class="home-hm-tag home-hm-tag-hit">&blacktriangle; Biggest Hit</div>
            <div class="home-hm-player">{escape(hit["player"])}</div>
            <div class="home-hm-detail">{escape(hit["line"])}</div>
            <div class="home-hm-score home-hm-score-hit">Standout Score {hit["score"]}</div>
        </div>
        '''

    miss_html = ""
    # We may not always have a "miss" from standouts data
    if miss:
        miss_html = f'''
        <div class="home-hm-side">
            <div class="home-hm-tag home-hm-tag-miss">&blacktriangledown; Biggest Miss</div>
            <div class="home-hm-player">{escape(miss["player"])}</div>
            <div class="home-hm-detail">{escape(miss["line"])}</div>
        </div>
        '''

    if not hit_html and not miss_html:
        return ""

    return f'<div class="home-hm-card">{hit_html}{miss_html}</div>'


def _render_data_health(feeds: list[dict]) -> str:
    """Data freshness strip at bottom."""
    feed_html = []
    for f in feeds:
        status_cls = f"home-feed-{f['status']}" if f["status"] != "ok" else ""
        feed_html.append(f'''
        <div class="home-feed {status_cls}">
            <span class="home-feed-dot"></span>
            <span class="home-feed-name">{escape(f["name"])}</span>
            <span class="home-feed-age">{escape(f["age"])}</span>
        </div>
        ''')

    return f'''
    <div class="home-data-health">
        <div class="home-feeds">{"".join(feed_html)}</div>
        <div class="home-disclaimer">For entertainment &amp; research purposes. Not financial advice.</div>
    </div>
    '''


# ---------------------------------------------------------------------------
# Main page function
# ---------------------------------------------------------------------------

def _html(s: str) -> str:
    """Collapse indentation so Streamlit doesn't treat it as a code block."""
    import re
    # Strip leading/trailing whitespace, collapse runs of whitespace between tags
    s = s.strip()
    s = re.sub(r">\s+<", "><", s)
    return s


def page_home() -> None:
    """Render the Home dashboard overview page."""
    data = _assemble_home_data()

    parts: list[str] = []

    # Ticker
    if data["schedule"]:
        parts.append(_render_ticker(data))

    # Masthead
    parts.append(_render_masthead(data))

    # Hero matchup -- DISABLED until game prediction accuracy is validated.
    # TODO: Re-enable _render_hero() when we are confident in game-level predictions.
    # if data["featured"]:
    #     parts.append(_render_hero(data["featured"]))

    # Best matchups of the day (replaces hero while game predictions are disabled)
    if data["best_matchups"]:
        parts.append(_render_best_matchups(data["best_matchups"]))

    # Quick rail
    parts.append(_render_quick_rail(data))

    # 2-column section: Schedule | Top Hitters
    schedule_html = _render_section_head("Today's Slate", f'{data["n_games"]} games', "Schedule", "Schedule")
    schedule_html += _render_schedule_strip(data["schedule"])

    hitters_html = _render_section_head("Top Projected Hitters", "Today's game sims", "Schedule", "Schedule")
    hitters_html += _render_top_hitters(data["top_hitters"])

    parts.append(
        f'<div class="home-section">'
        f'<div class="home-row home-row-2col">'
        f'<div>{schedule_html}</div>'
        f'<div>{hitters_html}</div>'
        f'</div></div>'
    )

    # Props Lab -- WIP notice
    parts.append(
        f'<div class="home-section">'
        f'<div class="home-wip-box">'
        f'<div class="home-wip-label">Props Lab Edges</div>'
        f'<div class="home-wip-note">'
        f'Game odds and prop edges are a work in progress. '
        f'We are still validating edge accuracy against market lines before surfacing recommendations. '
        f'Visit <a href="{_nav_url("Props Lab")}" target="_self">Props Lab</a> '
        f'to explore the raw model output.'
        f'</div>'
        f'</div></div>'
    )

    # 2-column section: Model Perf | Yesterday Hit/Miss
    model_html = _render_section_head("Model Performance", "", "Full Report", "Model Performance")
    model_html += _render_model_perf(data["model_stats"])

    hm_html = _render_section_head("Today's Standouts", "Top performers", "", "")
    hm_html += _render_hit_miss(data["yesterday_hit"], data["yesterday_miss"])

    parts.append(
        f'<div class="home-section">'
        f'<div class="home-row home-row-2col">'
        f'<div>{model_html}</div>'
        f'<div>{hm_html}</div>'
        f'</div></div>'
    )

    # Data health
    if data["data_feeds"]:
        parts.append(_render_data_health(data["data_feeds"]))

    # Assemble and strip indentation so Streamlit doesn't treat it as code
    full = '<div class="home-page">' + "".join(parts) + '</div>'
    st.markdown(_html(full), unsafe_allow_html=True)
