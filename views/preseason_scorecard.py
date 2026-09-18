"""Preseason Scorecard -- how the frozen preseason projections actually did."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import CURRENT_SEASON
from services.data_loader import (
    calibrated_breakout_prob,
    load_player_teams,
    load_traditional_stats_all,
    load_advanced_stats,
    load_breakout_calibration,
    load_preseason_breakout_candidates,
    load_preseason_counting,
    load_preseason_counting_sim,
    load_preseason_projections,
    load_traditional_stats,
)
from utils.alerts import tdd_info
from utils.html import esc

# Playing-time bars for a fair read. Below these a player's rate stats are
# mostly noise, and a projection should not be graded on 40 plate appearances.
MIN_PA = 200
MIN_BF = 300
GROUP_SIZE = 50

# wRC+ and FIP lines that separate a star call from an ordinary one. 100 is
# league average by construction, so 120 is a clear step above it.
STAR_WRC = 120
DUD_WRC = 85
ACE_FIP = 3.60
WEAK_FIP = 4.60

_HITTER_STATS = [
    ("AVG", "avg", "rate3"),
    ("OBP", "obp", "rate3"),
    ("SLG", "slg", "rate3"),
    ("OPS", "ops", "rate3"),
    ("wOBA", "woba", "rate3"),
]
_PITCHER_STATS = [
    ("ERA", "era", "dec2"),
    ("WHIP", "whip", "dec2"),
    ("FIP", "fip_era", "dec2", "fip"),
]


def _fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "-"
    if kind == "rate3":
        text = f"{value:.3f}"
        return text[1:] if text.startswith("0.") else text
    if kind == "rate3_signed":
        return f"{value:+.3f}"
    if kind == "dec0":
        return f"{value:.0f}"
    if kind == "dec0_signed":
        return f"{value:+.0f}"
    if kind == "dec2":
        return f"{value:.2f}"
    if kind == "dec2_signed":
        return f"{value:+.2f}"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "pct_signed":
        return f"{value:+.1%}"
    return f"{value:.2f}"


def _metrics(projected: pd.Series, actual: pd.Series, lo=None, hi=None) -> dict:
    projected = projected.astype(float)
    actual = actual.astype(float)
    coverage = np.nan
    if lo is not None and hi is not None:
        coverage = float(((actual >= lo) & (actual <= hi)).mean())
    return {
        "proj": projected.mean(),
        "actual": actual.mean(),
        "bias": (projected - actual).mean(),
        "mae": (projected - actual).abs().mean(),
        "corr": projected.corr(actual),
        "coverage": coverage,
        "n": len(projected),
    }


def _team_lookup() -> dict[int, str]:
    teams = load_player_teams()
    if teams.empty or "player_id" not in teams.columns:
        return {}
    return dict(zip(teams["player_id"].astype(int), teams["team_abbr"]))


def _hitter_verdict(projected: float, actual: float) -> tuple[str, str]:
    """Label a star or dud call by where the player actually finished.

    Compares the rounded values so a line reading 120 is never described as
    falling short of 120.
    """
    projected, actual = round(projected), round(actual)
    if projected >= STAR_WRC:
        if actual >= STAR_WRC:
            return "delivered", "hit"
        if actual >= 100:
            return "above average, short of the call", "soft"
        return "bust", "miss"
    if actual <= DUD_WRC:
        return "as projected", "hit"
    if actual >= 110:
        return "we were wrong", "miss"
    return "better than projected", "soft"


def _pitcher_verdict(projected: float, actual: float) -> tuple[str, str]:
    projected, actual = round(projected, 2), round(actual, 2)
    if projected <= ACE_FIP:
        if actual <= ACE_FIP:
            return "delivered", "hit"
        if actual <= 4.20:
            return "useful, short of the call", "soft"
        return "bust", "miss"
    if actual >= WEAK_FIP:
        return "as projected", "hit"
    if actual <= 3.90:
        return "we were wrong", "miss"
    return "better than projected", "soft"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _hitter_scorecard() -> dict:
    sim = load_preseason_counting_sim("hitter")
    actual = load_traditional_stats("hitter")
    if sim.empty or actual.empty:
        return {}
    qualified = actual[actual["pa"] >= MIN_PA]
    merged = sim.merge(
        qualified[["batter_id", "batter_name", "pa", "avg", "obp", "slg", "ops", "woba"]],
        on="batter_id", suffixes=("", "_act"),
    )
    if merged.empty:
        return {}

    rows = []
    for label, stat, fmt in _HITTER_STATS:
        rows.append({
            "label": label, "fmt": fmt,
            **_metrics(merged[f"projected_{stat}_mean"], merged[stat],
                       merged.get(f"projected_{stat}_p10"), merged.get(f"projected_{stat}_p90")),
        })

    ranked = merged.sort_values("projected_ops_mean", ascending=False)
    groups = []
    for label, frame in (
        (f"Top {GROUP_SIZE}", ranked.head(GROUP_SIZE)),
        ("Middle", ranked.iloc[GROUP_SIZE:-GROUP_SIZE]),
        (f"Bottom {GROUP_SIZE}", ranked.tail(GROUP_SIZE)),
    ):
        groups.append({
            "label": label,
            **_metrics(frame["projected_ops_mean"], frame["ops"],
                       frame["projected_ops_p10"], frame["projected_ops_p90"]),
        })

    # Playing time by projected tier, across everyone projected, to show how
    # much the qualified sample is shaped by who kept their job.
    everyone = sim.merge(
        actual[["batter_id", "pa"]].rename(columns={"pa": "pa_actual"}),
        on="batter_id", how="left",
    )
    everyone["pa_actual"] = everyone["pa_actual"].fillna(0)
    everyone = everyone.sort_values("projected_ops_mean", ascending=False)
    survival = []
    for label, frame in (
        (f"Top {GROUP_SIZE}", everyone.head(GROUP_SIZE)),
        ("Middle", everyone.iloc[GROUP_SIZE:-GROUP_SIZE]),
        (f"Bottom {GROUP_SIZE}", everyone.tail(GROUP_SIZE)),
    ):
        survival.append({
            "label": label,
            "played": float((frame["pa_actual"] >= MIN_PA).mean()),
            "median_pa": float(frame["pa_actual"].median()),
        })

    rates = []
    rate_stamp = ""
    projections = load_preseason_projections("hitter")
    advanced = load_advanced_stats("hitter")
    if not projections.empty and "snapshot_date" in projections.columns:
        stamps = projections["snapshot_date"].dropna().astype(str)
        if not stamps.empty:
            rate_stamp = stamps.max()
    if not projections.empty and not advanced.empty:
        adv = advanced[advanced["pa"] >= MIN_PA][["batter_id", "k_pct", "bb_pct"]]
        rate_merged = projections.drop_duplicates("batter_id").merge(adv, on="batter_id")
        for label, proj_col, act_col in (("K%", "projected_k_rate", "k_pct"),
                                         ("BB%", "projected_bb_rate", "bb_pct")):
            if proj_col not in rate_merged or act_col not in rate_merged:
                continue
            act = rate_merged[act_col].astype(float)
            if act.mean() > 1:
                act = act / 100.0
            rates.append({"label": label, **_metrics(rate_merged[proj_col], act)})

    # Player level: who the model called a star or a dud, and what happened.
    advanced_all = load_advanced_stats("hitter")
    players: dict[str, list[dict]] = {"stars": [], "duds": [], "over": [], "under": []}
    if not advanced_all.empty:
        wrc = advanced_all[advanced_all["pa"] >= MIN_PA][["batter_id", "pa", "wrc_plus"]]
        named = sim.merge(wrc, on="batter_id")
        teams = _team_lookup()
        named = named.assign(
            projected=named["projected_wrc_plus_mean"].astype(float),
            actual=named["wrc_plus"].astype(float),
        )
        named["delta"] = named["actual"] - named["projected"]
        named["team"] = named["batter_id"].astype(int).map(teams).fillna("")

        def _rows(frame: pd.DataFrame) -> list[dict]:
            out = []
            for _, row in frame.iterrows():
                verdict, kind = _hitter_verdict(row["projected"], row["actual"])
                out.append({
                    "name": row["batter_name"], "team": row["team"],
                    "playing_time": int(row["pa"]),
                    "projected": row["projected"], "actual": row["actual"],
                    "delta": row["delta"], "verdict": verdict, "kind": kind,
                })
            return out

        stars = named[named["projected"] >= STAR_WRC].sort_values("projected", ascending=False)
        duds = named[named["projected"] <= DUD_WRC].sort_values("projected")
        players["stars"] = _rows(stars)
        players["duds"] = _rows(duds)
        players["over"] = _rows(named.nsmallest(8, "delta"))
        players["under"] = _rows(named.nlargest(8, "delta"))

    return {"stats": rows, "groups": groups, "survival": survival, "rates": rates,
            "rate_stamp": rate_stamp,
            "players": players, "n": len(merged), "games": int(actual["games"].max())}


@st.cache_data(ttl=600)
def _pitcher_scorecard() -> dict:
    sim = load_preseason_counting_sim("pitcher")
    actual = load_traditional_stats("pitcher")
    if sim.empty or actual.empty:
        return {}
    qualified = actual[actual["bf"] >= MIN_BF]
    merged = sim.merge(
        qualified[["pitcher_id", "bf", "ip", "era", "whip", "fip", "k", "bb",
                   "hits_allowed"]],
        on="pitcher_id", suffixes=("", "_act"),
    )
    if merged.empty:
        return {}

    rows = []
    for entry in _PITCHER_STATS:
        label, stat, fmt = entry[0], entry[1], entry[2]
        actual_col = entry[3] if len(entry) > 3 else entry[1]
        proj_col = f"projected_{stat}_mean"
        if proj_col not in merged or actual_col not in merged:
            continue
        rows.append({
            "label": label, "fmt": fmt,
            **_metrics(merged[proj_col], merged[actual_col],
                       merged.get(f"projected_{stat}_p10"), merged.get(f"projected_{stat}_p90")),
        })

    rates = []
    if {"total_k_mean", "total_bf_mean"} <= set(merged.columns):
        merged["k_proj"] = merged["total_k_mean"] / merged["total_bf_mean"]
        merged["bb_proj"] = merged["total_bb_mean"] / merged["total_bf_mean"]
        merged["k_act"] = merged["k"] / merged["bf"]
        merged["bb_act"] = merged["bb"] / merged["bf"]
        rates.append({"label": "K%", **_metrics(merged["k_proj"], merged["k_act"])})
        rates.append({"label": "BB%", **_metrics(merged["bb_proj"], merged["bb_act"])})

    # Where the ERA miss comes from: baserunners allowed vs runs conceded.
    diagnosis = {}
    if {"total_h_mean", "total_outs_mean"} <= set(merged.columns):
        innings = merged["total_outs_mean"] / 3
        diagnosis = {
            "hits9_proj": float((merged["total_h_mean"] / innings * 9).mean()),
            "hits9_act": float((merged["hits_allowed"] / merged["ip"] * 9).mean())
            if "hits_allowed" in merged else np.nan,
            "runs9_proj": float((merged["total_runs_mean"] / innings * 9).mean())
            if "total_runs_mean" in merged else np.nan,
            "runs9_act": float(merged["era"].mean()),
        }

    players: dict[str, list[dict]] = {"stars": [], "duds": [], "over": [], "under": []}
    if "projected_fip_era_mean" in merged.columns and "fip" in merged.columns:
        teams = _team_lookup()
        named = merged.assign(
            projected=merged["projected_fip_era_mean"].astype(float),
            actual=merged["fip"].astype(float),
        )
        # For pitchers a negative delta is an improvement, so flip the sign
        # to keep "delta" meaning "better than projected" everywhere.
        named["delta"] = named["projected"] - named["actual"]
        named["team"] = named["pitcher_id"].astype(int).map(teams).fillna("")

        def _rows(frame: pd.DataFrame) -> list[dict]:
            out = []
            for _, row in frame.iterrows():
                verdict, kind = _pitcher_verdict(row["projected"], row["actual"])
                out.append({
                    "name": row["pitcher_name"], "team": row["team"],
                    "playing_time": float(row["ip"]),
                    "projected": row["projected"], "actual": row["actual"],
                    "delta": row["delta"], "verdict": verdict, "kind": kind,
                })
            return out

        aces = named[named["projected"] <= ACE_FIP].sort_values("projected")
        weak = named[named["projected"] >= WEAK_FIP].sort_values("projected", ascending=False)
        players["stars"] = _rows(aces)
        players["duds"] = _rows(weak)
        players["over"] = _rows(named.nsmallest(8, "delta"))
        players["under"] = _rows(named.nlargest(8, "delta"))

    return {"stats": rows, "rates": rates, "diagnosis": diagnosis,
            "players": players, "n": len(merged)}


@st.cache_data(ttl=600)
def _pitcher_benchmark() -> dict:
    """Compare the projections against simply reusing last season's numbers.

    Restricted to pitchers who qualified in both seasons, so the model and
    the naive baseline are judged on exactly the same players.
    """
    actual = load_traditional_stats("pitcher")
    history = load_traditional_stats_all("pitcher")
    sim = load_preseason_counting_sim("pitcher")
    counting = load_preseason_counting("pitcher")
    if actual.empty or history.empty or sim.empty:
        return {}

    current = actual[actual["bf"] >= MIN_BF][["pitcher_id", "era", "fip"]].rename(
        columns={"era": "era_now", "fip": "fip_now"})
    prior_season = history[(history["season"] == CURRENT_SEASON - 1)
                           & (history["bf"] >= MIN_BF)][["pitcher_id", "era", "fip"]]
    prior_season = prior_season.rename(columns={"era": "era_prior", "fip": "fip_prior"})

    frame = current.merge(prior_season, on="pitcher_id").merge(
        sim[["pitcher_id", "projected_era_mean", "projected_fip_era_mean"]], on="pitcher_id")
    if not counting.empty and "projected_era_mean" in counting.columns:
        frame = frame.merge(
            counting[["pitcher_id", "projected_era_mean"]].rename(
                columns={"projected_era_mean": "siera"}), on="pitcher_id", how="left")
    if len(frame) < 20:
        return {}

    def row(label, predictor, target, naive=False):
        pred, act = frame[predictor].astype(float), frame[target].astype(float)
        return {"label": label, "naive": naive, "corr": pred.corr(act),
                "bias": (pred - act).mean(), "mae": (pred - act).abs().mean()}

    era_rows = [
        row("Last season's ERA", "era_prior", "era_now", naive=True),
        row("Last season's FIP", "fip_prior", "era_now", naive=True),
        row("Our ERA projection", "projected_era_mean", "era_now"),
        row("Our FIP projection", "projected_fip_era_mean", "era_now"),
    ]
    fip_rows = [
        row("Last season's FIP", "fip_prior", "fip_now", naive=True),
        row("Our FIP projection", "projected_fip_era_mean", "fip_now"),
    ]
    if "siera" in frame.columns and frame["siera"].notna().sum() > 20:
        era_rows.insert(3, row("Our SIERA projection", "siera", "era_now"))
    return {"n": len(frame), "era": era_rows, "fip": fip_rows}


@st.cache_data(ttl=600)
def _breakout_scorecard(player_type: str) -> dict:
    candidates = load_preseason_breakout_candidates(player_type)
    if candidates.empty:
        return {}
    calibration = load_breakout_calibration()

    if player_type == "hitter":
        advanced = load_advanced_stats("hitter")
        if advanced.empty:
            return {}
        actual = advanced[["batter_id", "pa", "wrc_plus"]].rename(
            columns={"pa": "played", "wrc_plus": "after"})
        frame = candidates.merge(actual, on="batter_id", how="left")
        frame["before"] = frame["wrc_plus"]
        frame["name"] = frame["batter_name"]
        threshold, floor, gain = MIN_PA, 110, 15
        frame["improved"] = frame["after"] - frame["before"]
        frame["hit"] = (frame["played"].fillna(0) >= threshold) & \
                       (frame["improved"] >= gain) & (frame["after"] >= floor)
        unit, metric = "PA", "wRC+"
    else:
        actual = load_traditional_stats("pitcher")
        if actual.empty:
            return {}
        act = actual[["pitcher_id", "bf", "era"]].rename(
            columns={"bf": "played", "era": "after"})
        frame = candidates.merge(act, on="pitcher_id", how="left")
        frame["before"] = frame["era"]
        frame["name"] = frame["pitcher_name"]
        frame["improved"] = frame["before"] - frame["after"]     # ERA down is good
        frame["hit"] = (frame["played"].fillna(0) >= MIN_BF) & \
                       (frame["improved"] >= 0.75) & (frame["after"] <= 4.00)
        unit, metric = "BF", "ERA"

    frame["played"] = frame["played"].fillna(0)
    frame["calibrated"] = calibrated_breakout_prob(
        frame["breakout_prob"], player_type, calibration)

    tiers = []
    for tier in ("Breakout Candidate", "On the Radar", ""):
        group = frame[frame["breakout_tier"] == tier]
        if group.empty:
            continue
        tiers.append({
            "label": tier or "Not flagged",
            "n": len(group),
            "hit_rate": float(group["hit"].mean()),
            "never_played": float((group["played"] < (MIN_PA if player_type == "hitter" else MIN_BF)).mean()),
            "raw": float(group["breakout_prob"].mean()),
            "calibrated": float(group["calibrated"].mean()),
        })

    params = calibration.get(player_type, {}) if calibration else {}
    named = frame[frame["breakout_tier"] == "Breakout Candidate"].sort_values("breakout_rank")

    # Bucket by what the model said, then report what happened and what the
    # calibration now says for those same players. Bucketing the calibrated
    # values separately would compare different sets of players.
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    reliability = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (frame["breakout_prob"] >= lo) & (
            frame["breakout_prob"] < hi if hi < 1.0 else frame["breakout_prob"] <= hi)
        if not mask.any():
            continue
        reliability.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "n": int(mask.sum()),
            "predicted": float(frame.loc[mask, "breakout_prob"].mean()),
            "observed": float(frame.loc[mask, "hit"].mean()),
            "calibrated": float(frame.loc[mask, "calibrated"].mean()),
        })

    return {
        "tiers": tiers,
        "reliability": reliability,
        "lift": params.get("top_decile_lift"),
        "observed_rate": params.get("observed_rate"),
        "named": named[["name", "breakout_rank", "breakout_prob", "calibrated",
                        "before", "after", "played", "hit"]].to_dict("records"),
        "unit": unit, "metric": metric,
        "caveat": calibration.get("caveat", "") if calibration else "",
        "definition": (calibration.get("definitions", {}) or {}).get(player_type, ""),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _accuracy_table(rows: list[dict], signed_kind: str) -> str:
    head = ("<tr><th class='sc-left'>Stat</th><th>Projected</th><th>Actual</th>"
            "<th>Bias</th><th>Error</th><th>Corr</th><th>In range</th></tr>")
    body = ""
    for row in rows:
        coverage = "-" if pd.isna(row["coverage"]) else f"{row['coverage']:.0%}"
        body += (
            f"<tr><td class='sc-left'>{esc(row['label'])}</td>"
            f"<td>{_fmt(row['proj'], row['fmt'])}</td>"
            f"<td>{_fmt(row['actual'], row['fmt'])}</td>"
            f"<td>{_fmt(row['bias'], signed_kind)}</td>"
            f"<td>{_fmt(row['mae'], row['fmt'])}</td>"
            f"<td>{row['corr']:.2f}</td><td>{coverage}</td></tr>"
        )
    return f"<div class='sc-wrap'><table class='sc-table'>{head}{body}</table></div>"


def _rate_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    head = ("<tr><th class='sc-left'>Rate</th><th>Projected</th><th>Actual</th>"
            "<th>Bias</th><th>Error</th><th>Corr</th></tr>")
    body = ""
    for row in rows:
        body += (
            f"<tr><td class='sc-left'>{esc(row['label'])}</td>"
            f"<td>{_fmt(row['proj'], 'pct')}</td><td>{_fmt(row['actual'], 'pct')}</td>"
            f"<td>{_fmt(row['bias'], 'pct_signed')}</td><td>{_fmt(row['mae'], 'pct')}</td>"
            f"<td>{row['corr']:.2f}</td></tr>"
        )
    return f"<div class='sc-wrap'><table class='sc-table'>{head}{body}</table></div>"


_VERDICT_CLASS = {"hit": "sc-hit", "miss": "sc-miss", "soft": "sc-muted"}


def _player_table(rows: list[dict], metric: str, unit: str, fmt: str,
                  limit: int = 12) -> str:
    """Name-by-name table of projected vs actual with a verdict."""
    if not rows:
        return "<div class='sc-note'>Nothing to show yet.</div>"
    head = (f"<tr><th class='sc-left'>Player</th><th>{esc(unit)}</th>"
            f"<th>Projected {esc(metric)}</th><th>Actual</th><th>Diff</th>"
            f"<th class='sc-left'>Verdict</th></tr>")
    body = ""
    for row in rows[:limit]:
        team = f" <span class='sc-muted'>{esc(row['team'])}</span>" if row["team"] else ""
        playing = (f"{row['playing_time']:.0f}" if unit == "PA"
                   else f"{row['playing_time']:.0f}")
        body += (
            f"<tr><td class='sc-left'>{esc(str(row['name']))}{team}</td>"
            f"<td class='sc-muted'>{playing}</td>"
            f"<td>{_fmt(row['projected'], fmt)}</td>"
            f"<td class='sc-strong'>{_fmt(row['actual'], fmt)}</td>"
            f"<td class='{_VERDICT_CLASS.get(row['kind'], '')}'>"
            f"{_fmt(row['delta'], fmt + '_signed')}</td>"
            f"<td class='sc-left {_VERDICT_CLASS.get(row['kind'], '')}'>"
            f"{esc(row['verdict'])}</td></tr>"
        )
    return f"<div class='sc-wrap'><table class='sc-table'>{head}{body}</table></div>"


def _call_summary(rows: list[dict], noun: str) -> str:
    if not rows:
        return ""
    hits = sum(1 for row in rows if row["kind"] == "hit")
    return (f"<div class='sc-note'><b class='sc-strong'>{hits} of {len(rows)}</b> "
            f"{esc(noun)} landed.</div>")


def _render_player_sections(players: dict, metric: str, unit: str, fmt: str,
                            star_label: str, dud_label: str, star_noun: str) -> None:
    st.markdown(f'<div class="tdd-section-hdr">{esc(star_label)}</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sc-note">Diff is how the player finished against the projection: '
        'positive means better than projected.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_call_summary(players.get("stars", []), star_noun), unsafe_allow_html=True)
    st.markdown(_player_table(players.get("stars", []), metric, unit, fmt),
                unsafe_allow_html=True)
    if len(players.get("stars", [])) > 12:
        with st.expander(f"All {len(players['stars'])}", expanded=False):
            st.markdown(_player_table(players["stars"], metric, unit, fmt, limit=999),
                        unsafe_allow_html=True)

    st.markdown(f'<div class="tdd-section-hdr">{esc(dud_label)}</div>',
                unsafe_allow_html=True)
    st.markdown(_call_summary(players.get("duds", []), "low calls"), unsafe_allow_html=True)
    st.markdown(_player_table(players.get("duds", []), metric, unit, fmt),
                unsafe_allow_html=True)
    if len(players.get("duds", [])) > 12:
        with st.expander(f"All {len(players['duds'])}", expanded=False):
            st.markdown(_player_table(players["duds"], metric, unit, fmt, limit=999),
                        unsafe_allow_html=True)

    st.markdown('<div class="tdd-section-hdr">The worst calls in both directions</div>',
                unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='sc-note'>Projected far too high</div>", unsafe_allow_html=True)
        st.markdown(_player_table(players.get("over", []), metric, unit, fmt, limit=8),
                    unsafe_allow_html=True)
    with right:
        st.markdown("<div class='sc-note'>Projected far too low</div>", unsafe_allow_html=True)
        st.markdown(_player_table(players.get("under", []), metric, unit, fmt, limit=8),
                    unsafe_allow_html=True)


def _render_hitters() -> None:
    data = _hitter_scorecard()
    if not data:
        tdd_info("No preseason hitter snapshot to score yet.")
        return

    st.markdown(
        f'<div class="sc-note">{data["n"]} hitters with {MIN_PA}+ plate appearances, '
        f'scored through {data["games"]} team games.</div>',
        unsafe_allow_html=True,
    )
    _render_player_sections(
        data.get("players", {}), "wRC+", "PA", "dec0",
        f"Projected stars (wRC+ {STAR_WRC}+)",
        f"Projected duds (wRC+ {DUD_WRC} or below)",
        "star calls",
    )

    st.markdown('<div class="tdd-section-hdr">Accuracy across the board</div>',
                unsafe_allow_html=True)
    st.markdown(_accuracy_table(data["stats"], "rate3_signed"), unsafe_allow_html=True)
    st.markdown(_rate_table(data["rates"]), unsafe_allow_html=True)
    note = ('Rate stats are modeled directly and hold up best. '
            'Slash lines inherit batted-ball luck on top of those rates.')
    if data.get("rate_stamp"):
        # The K%/BB% rows come from the rate-projection file, which a full
        # precompute rewrites. It is still trained through the prior season
        # only and carries no in-season updates, but it is not the Opening
        # Day artifact the counting lines use, so it is labelled separately.
        note += (f' The K% and BB% rows above are rebuilt from the same '
                 f'2018-{CURRENT_SEASON - 1} fit rather than read from the Opening Day '
                 f'snapshot; this copy was written {esc(data["rate_stamp"])}.')
    st.markdown(f'<div class="sc-note">{note}</div>', unsafe_allow_html=True)

    st.markdown('<div class="tdd-section-hdr">Best vs worst projected</div>',
                unsafe_allow_html=True)
    head = ("<tr><th class='sc-left'>Group</th><th>n</th><th>Projected OPS</th>"
            "<th>Actual OPS</th><th>Bias</th><th>Error</th><th>In range</th></tr>")
    body = ""
    for row in data["groups"]:
        body += (
            f"<tr><td class='sc-left'>{esc(row['label'])}</td><td>{row['n']}</td>"
            f"<td>{_fmt(row['proj'], 'rate3')}</td><td>{_fmt(row['actual'], 'rate3')}</td>"
            f"<td>{_fmt(row['bias'], 'rate3_signed')}</td><td>{_fmt(row['mae'], 'rate3')}</td>"
            f"<td>{row['coverage']:.0%}</td></tr>"
        )
    st.markdown(f"<div class='sc-wrap'><table class='sc-table'>{head}{body}</table></div>",
                unsafe_allow_html=True)

    survival = "".join(
        f"<tr><td class='sc-left'>{esc(row['label'])}</td>"
        f"<td>{row['played']:.0%}</td><td>{row['median_pa']:.0f}</td></tr>"
        for row in data["survival"]
    )
    st.markdown(
        '<div class="sc-callout"><div class="sc-callout-head">Read this with the playing time</div>'
        '<div class="sc-callout-body">The weakest projected hitters who reach the plate-appearance '
        'bar are the ones who hit well enough to keep the job, which flatters their group. '
        'Playing time across every projected hitter:</div>'
        f"<div class='sc-wrap'><table class='sc-table'>"
        f"<tr><th class='sc-left'>Group</th><th>Reached {MIN_PA} PA</th><th>Median PA</th></tr>"
        f"{survival}</table></div></div>",
        unsafe_allow_html=True,
    )


def _render_pitchers() -> None:
    data = _pitcher_scorecard()
    if not data:
        tdd_info("No preseason pitcher snapshot to score yet.")
        return

    st.markdown(
        f'<div class="sc-note">{data["n"]} pitchers with {MIN_BF}+ batters faced.</div>',
        unsafe_allow_html=True,
    )
    _render_player_sections(
        data.get("players", {}), "FIP", "IP", "dec2",
        f"Projected aces (FIP {ACE_FIP:.2f} or better)",
        f"Projected weak arms (FIP {WEAK_FIP:.2f}+)",
        "ace calls",
    )
    st.markdown(
        '<div class="sc-note">Judged on FIP rather than ERA: the ERA projection is '
        'the one metric here that does not work, for the reason below.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="tdd-section-hdr">Accuracy across the board</div>',
                unsafe_allow_html=True)
    st.markdown(_accuracy_table(data["stats"], "dec2_signed"), unsafe_allow_html=True)
    st.markdown(_rate_table(data["rates"]), unsafe_allow_html=True)

    diagnosis = data.get("diagnosis") or {}
    if diagnosis and not pd.isna(diagnosis.get("runs9_proj", np.nan)):
        st.markdown(
            '<div class="sc-callout sc-warn">'
            '<div class="sc-callout-head">Why ERA is not the yardstick here</div>'
            '<div class="sc-callout-body">'
            'ERA depends on sequencing and defense, so it is a poor target no matter who '
            'projects it: the best predictor below manages a 0.35 correlation with next '
            'season ERA while the same method correlates 0.50 with FIP. That is the metric, '
            'not the model. '
            f'On top of that our own run conversion is off. The simulator allows '
            f'<b>{diagnosis["hits9_proj"]:.1f}</b> hits per nine against an actual '
            f'<b>{diagnosis["hits9_act"]:.1f}</b>, then turns those baserunners into only '
            f'<b>{diagnosis["runs9_proj"]:.1f}</b> runs per nine where real pitchers concede '
            f'<b>{diagnosis["runs9_act"]:.2f}</b>. The component rates are sound; the step that '
            'converts baserunners into runs is too kind to the pitcher, which is a real bug and '
            'shows up as the level bias in the table above. Everything on this site judges '
            'pitchers on FIP.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    benchmark = _pitcher_benchmark()
    if benchmark:
        st.markdown('<div class="tdd-section-hdr">Is it better than reusing last season?</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="sc-note">The honest benchmark for any projection is the lazy '
            f'alternative. Both are scored on the same {benchmark["n"]} pitchers who qualified '
            f'in {CURRENT_SEASON - 1} and {CURRENT_SEASON}.</div>',
            unsafe_allow_html=True,
        )
        for title, rows in (("Predicting this season's ERA", benchmark["era"]),
                            ("Predicting this season's FIP", benchmark["fip"])):
            body = ""
            for row in rows:
                tag = " <span class='sc-muted'>naive</span>" if row["naive"] else ""
                body += (
                    f"<tr><td class='sc-left'>{esc(row['label'])}{tag}</td>"
                    f"<td>{row['corr']:+.2f}</td><td>{_fmt(row['bias'], 'dec2_signed')}</td>"
                    f"<td>{_fmt(row['mae'], 'dec2')}</td></tr>"
                )
            st.markdown(
                f"<div class='sc-note'>{esc(title)}</div>"
                "<div class='sc-wrap'><table class='sc-table'>"
                "<tr><th class='sc-left'>Predictor</th><th>Corr</th><th>Bias</th><th>Error</th></tr>"
                f"{body}</table></div>",
                unsafe_allow_html=True,
            )


def _render_breakouts(player_type: str) -> None:
    data = _breakout_scorecard(player_type)
    if not data:
        tdd_info("No preseason breakout list to score yet.")
        return

    if data.get("definition"):
        st.markdown(f'<div class="sc-note">Counted as a breakout: {esc(data["definition"])}.</div>',
                    unsafe_allow_html=True)

    head = ("<tr><th class='sc-left'>Tier</th><th>Flagged</th><th>Hit rate</th>"
            f"<th>Never reached the {esc(data['unit'])} bar</th>"
            "<th>Model said</th><th>Calibrated</th></tr>")
    body = ""
    for row in data["tiers"]:
        body += (
            f"<tr><td class='sc-left'>{esc(row['label'])}</td><td>{row['n']}</td>"
            f"<td class='sc-strong'>{row['hit_rate']:.0%}</td>"
            f"<td>{row['never_played']:.0%}</td>"
            f"<td class='sc-muted'>{row['raw']:.0%}</td><td>{row['calibrated']:.0%}</td></tr>"
        )
    st.markdown(f"<div class='sc-wrap'><table class='sc-table'>{head}{body}</table></div>",
                unsafe_allow_html=True)

    if data.get("reliability"):
        rows = ""
        for row in data["reliability"]:
            rows += (
                f"<tr><td class='sc-left'>{esc(row['bucket'])}</td><td>{row['n']}</td>"
                f"<td class='sc-muted'>{row['predicted']:.0%}</td>"
                f"<td class='sc-strong'>{row['observed']:.0%}</td>"
                f"<td>{row['calibrated']:.0%}</td></tr>"
            )
        st.markdown(
            '<div class="tdd-section-hdr">Were the probabilities honest?</div>'
            "<div class='sc-wrap'><table class='sc-table'>"
            "<tr><th class='sc-left'>Model said</th><th>n</th><th>Predicted</th>"
            "<th>Actually happened</th><th>After calibration</th></tr>"
            f"{rows}</table></div>",
            unsafe_allow_html=True,
        )
        lift = data.get("lift")
        st.markdown(
            '<div class="sc-note">The raw probabilities ran far above what happened, but they '
            'ordered players correctly, which is the part worth keeping'
            + (f": the top decile broke out {lift:.1f}x as often as the field. " if lift else ". ")
            + f'{esc(data.get("caveat", ""))}</div>',
            unsafe_allow_html=True,
        )

    named = data.get("named") or []
    if named:
        with st.expander(f"Every named candidate ({len(named)})", expanded=False):
            rows = ""
            for row in named:
                bar = MIN_PA if player_type == "hitter" else MIN_BF
                if row["played"] < bar:
                    verdict = "<span class='sc-muted'>never played enough</span>"
                elif row["hit"]:
                    verdict = "<span class='sc-hit'>hit</span>"
                else:
                    verdict = "<span class='sc-miss'>miss</span>"
                after = "-" if pd.isna(row["after"]) else f"{row['after']:.0f}" \
                    if player_type == "hitter" else f"{row['after']:.2f}"
                before = f"{row['before']:.0f}" if player_type == "hitter" else f"{row['before']:.2f}"
                rows += (
                    f"<tr><td class='sc-left'>{esc(str(row['name']))}</td>"
                    f"<td class='sc-muted'>{row['breakout_prob']:.0%}</td>"
                    f"<td>{row['calibrated']:.0%}</td>"
                    f"<td>{before} &rarr; {after}</td>"
                    f"<td>{row['played']:.0f}</td><td>{verdict}</td></tr>"
                )
            st.markdown(
                "<div class='sc-wrap'><table class='sc-table'>"
                f"<tr><th class='sc-left'>Player</th><th>Model said</th><th>Calibrated</th>"
                f"<th>{esc(data['metric'])}</th><th>{esc(data['unit'])}</th><th>Result</th></tr>"
                f"{rows}</table></div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page_preseason_scorecard() -> None:
    """Grade the frozen preseason projections against the season so far."""
    st.markdown(
        f'<div class="sc-head">'
        f'<div class="sc-eyebrow">The Data Diamond</div>'
        f'<h1 class="sc-title">Preseason Scorecard</h1>'
        f'<p class="sc-lede">Every projection here was built from 2018-{CURRENT_SEASON - 1} data '
        f'with no knowledge of this season, and the counting and wRC+ lines come from snapshots '
        f'frozen on Opening Day. This page scores them against what has actually happened, '
        f'including the parts that did not work.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    hitters, pitchers, breakouts = st.tabs(["Hitters", "Pitchers", "Breakout calls"])
    with hitters:
        _render_hitters()
    with pitchers:
        _render_pitchers()
    with breakouts:
        kind = st.radio("Player type", ["Hitters", "Pitchers"], horizontal=True,
                        key="sc_brk_type", label_visibility="collapsed")
        _render_breakouts("hitter" if kind == "Hitters" else "pitcher")
