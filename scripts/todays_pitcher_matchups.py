"""Analyze today's pitcher matchups using TDD projection system."""
import sys
sys.path.insert(0, r'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard')

import pandas as pd
import numpy as np
from pathlib import Path
from lib.schedule import fetch_todays_schedule, fetch_game_lineups
from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE
from config import PITCH_DISPLAY

DASHBOARD_DIR = Path(r'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard')

# Load core data
pitcher_proj = pd.read_parquet(DASHBOARD_DIR / 'pitcher_projections.parquet')
arsenal_df = pd.read_parquet(DASHBOARD_DIR / 'pitcher_arsenal.parquet')
vuln_df = pd.read_parquet(DASHBOARD_DIR / 'hitter_vuln_career.parquet')
pitcher_arch = pd.read_parquet(DASHBOARD_DIR / 'pitcher_archetypes.parquet')

# Build baselines
baselines_pt = {
    pt: {
        "whiff_rate": v.get("whiff_rate", 0.25),
        "chase_rate": v.get("chase_rate", 0.30),
        "barrel_rate": v.get("barrel_rate", 0.06),
    }
    for pt, v in LEAGUE_AVG_BY_PITCH_TYPE.items()
}

# Fetch today's schedule
schedule = fetch_todays_schedule()
print(f"Games today: {len(schedule)}")

# Build lookups
proj_lookup = {}
for _, row in pitcher_proj.iterrows():
    proj_lookup[int(row['pitcher_id'])] = {
        'pitcher_name': row.get('pitcher_name', ''),
        'projected_k_rate': row.get('projected_k_rate'),
        'projected_bb_rate': row.get('projected_bb_rate'),
        'projected_hr_per_bf': row.get('projected_hr_per_bf'),
    }

arch_lookup = {}
for _, r in pitcher_arch.iterrows():
    arch_lookup[int(r['pitcher_id'])] = r.get('archetype_name', '')

# Score every pitcher vs opposing lineup
results = []
for _, game in schedule.iterrows():
    gpk = game['game_pk']
    lineups = fetch_game_lineups(gpk)

    for side in ['away', 'home']:
        pid_col = f'{side}_pitcher_id'
        name_col = f'{side}_pitcher_name'
        opp_side = 'home' if side == 'away' else 'away'

        pid = game.get(pid_col)
        if pd.isna(pid):
            continue
        pid = int(pid)
        pname = game.get(name_col, 'TBD')

        opp_tid = game.get(f'{opp_side}_team_id')
        opp_lu = (
            lineups[lineups['team_id'] == opp_tid]
            if not lineups.empty and pd.notna(opp_tid)
            else pd.DataFrame()
        )

        proj = proj_lookup.get(pid, {})
        k_rate = proj.get('projected_k_rate', np.nan)
        bb_rate = proj.get('projected_bb_rate', np.nan)

        total_k_lift = total_bb_lift = total_hr_lift = 0.0
        n_scored = 0

        if not opp_lu.empty:
            for _, brow in opp_lu.head(9).iterrows():
                bid = int(brow['batter_id']) if pd.notna(brow.get('batter_id')) else None
                if bid is None:
                    continue
                kr = score_matchup(pid, bid, arsenal_df, vuln_df, baselines_pt)
                br = score_matchup_bb(pid, bid, arsenal_df, vuln_df, baselines_pt)
                hr = score_matchup_hr(pid, bid, arsenal_df, vuln_df, baselines_pt)

                kl = kr.get('matchup_k_logit_lift', 0.0)
                bl = br.get('matchup_bb_logit_lift', 0.0)
                hl = hr.get('matchup_hr_logit_lift', 0.0)

                total_k_lift += 0.0 if np.isnan(kl) else kl
                total_bb_lift += 0.0 if np.isnan(bl) else bl
                total_hr_lift += 0.0 if np.isnan(hl) else hl
                n_scored += 1

        avg_k = total_k_lift / n_scored if n_scored > 0 else 0.0
        avg_bb = total_bb_lift / n_scored if n_scored > 0 else 0.0
        avg_hr = total_hr_lift / n_scored if n_scored > 0 else 0.0
        net_score = avg_k - avg_bb * 0.5 - avg_hr * 0.5

        results.append({
            'pitcher_name': pname,
            'pitcher_id': pid,
            'team': game.get(f'{side}_abbr', ''),
            'opponent': game.get(f'{opp_side}_abbr', ''),
            'archetype': arch_lookup.get(pid, ''),
            'proj_k_rate': k_rate,
            'proj_bb_rate': bb_rate,
            'avg_k_lift': avg_k,
            'avg_bb_lift': avg_bb,
            'avg_hr_lift': avg_hr,
            'net_matchup_score': net_score,
            'lineup_scored': n_scored,
            'game_time': game.get('game_time', ''),
        })

matchups = pd.DataFrame(results).sort_values('net_matchup_score', ascending=False)


def fmt_pct(v):
    if pd.notna(v):
        return f"{v*100:.1f}%"
    return "N/A"


def lift_word(val, positive_good=True):
    threshold = 0.03
    if positive_good:
        if val > threshold:
            return "favorable"
        elif val < -threshold:
            return "unfavorable"
    else:
        if val < -threshold:
            return "favorable"
        elif val > threshold:
            return "unfavorable"
    return "neutral"


print("=" * 90)
print("PITCHER MATCHUP RANKINGS — TODAY")
print("=" * 90)

top = matchups[matchups['lineup_scored'] > 0]
for i, (_, row) in enumerate(top.iterrows(), 1):
    pid = row['pitcher_id']
    pname = row['pitcher_name']
    k_w = lift_word(row['avg_k_lift'], positive_good=True)
    bb_w = lift_word(row['avg_bb_lift'], positive_good=False)
    hr_w = lift_word(row['avg_hr_lift'], positive_good=False)
    arch = row['archetype'] or 'N/A'

    print()
    print(f"  #{i}  {pname} ({row['team']} vs {row['opponent']}) — {row['game_time']}")
    print(f"       Archetype: {arch}")
    print(f"       Proj K%: {fmt_pct(row['proj_k_rate'])}  |  Proj BB%: {fmt_pct(row['proj_bb_rate'])}")
    print(f"       K matchup: {row['avg_k_lift']:+.3f} ({k_w})  |  BB: {row['avg_bb_lift']:+.3f} ({bb_w})  |  HR: {row['avg_hr_lift']:+.3f} ({hr_w})")
    print(f"       NET score: {row['net_matchup_score']:+.3f}")

    # Arsenal summary
    p_ars = arsenal_df[arsenal_df['pitcher_id'] == pid].sort_values('usage_pct', ascending=False)
    if not p_ars.empty:
        pitches = []
        for _, pa in p_ars.head(4).iterrows():
            pt = pa['pitch_type']
            pt_name = PITCH_DISPLAY.get(pt, pt)
            usage = pa.get('usage_pct', 0)
            whiff = pa.get('whiff_rate', 0)
            pitches.append(f"{pt_name} ({usage:.0%}, {whiff:.0%} whiff)")
        print(f"       Arsenal: {' | '.join(pitches)}")

# Pitchers without lineup data
no_lu = matchups[matchups['lineup_scored'] == 0]
if not no_lu.empty:
    print()
    print("--- No lineup available yet ---")
    for _, row in no_lu.iterrows():
        arch = row['archetype'] or 'N/A'
        print(f"  {row['pitcher_name']} ({row['team']} vs {row['opponent']}) — Proj K%: {fmt_pct(row['proj_k_rate'])}  |  {arch}")

print("=" * 90)
print()
print("KEY:")
print("  K Lift > 0 = pitcher gets MORE Ks vs this lineup than baseline")
print("  BB Lift < 0 = pitcher gives FEWER walks (favorable)")
print("  HR Lift < 0 = pitcher allows FEWER HRs (favorable)")
print("  NET = K_lift - 0.5*BB_lift - 0.5*HR_lift  (higher = better pitcher matchup)")
