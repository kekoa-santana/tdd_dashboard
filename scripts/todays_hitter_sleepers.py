"""Find sleeper hitters with favorable matchups today using TDD system."""
import sys
sys.path.insert(0, r'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard')

import pandas as pd
import numpy as np
from pathlib import Path
from lib.schedule import fetch_todays_schedule, fetch_game_lineups
from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

DASHBOARD_DIR = Path(r'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard')

# Load core data
hitter_proj = pd.read_parquet(DASHBOARD_DIR / 'hitter_projections.parquet')
hitter_count = pd.read_parquet(DASHBOARD_DIR / 'hitter_counting_sim.parquet')
hitter_rankings = pd.read_parquet(DASHBOARD_DIR / 'hitters_rankings.parquet')
hitter_arch = pd.read_parquet(DASHBOARD_DIR / 'hitter_archetypes.parquet')
arsenal_df = pd.read_parquet(DASHBOARD_DIR / 'pitcher_arsenal.parquet')
vuln_df = pd.read_parquet(DASHBOARD_DIR / 'hitter_vuln_career.parquet')
pitcher_proj = pd.read_parquet(DASHBOARD_DIR / 'pitcher_projections.parquet')
pitcher_arch = pd.read_parquet(DASHBOARD_DIR / 'pitcher_archetypes.parquet')

# Quick column inspection
print("Hitter rankings columns:", list(hitter_rankings.columns))
print("Hitter proj columns:", list(hitter_proj.columns))
print("Hitter counting columns:", list(hitter_count.columns))
print()

# Build baselines
baselines_pt = {
    pt: {
        "whiff_rate": v.get("whiff_rate", 0.25),
        "chase_rate": v.get("chase_rate", 0.30),
        "barrel_rate": v.get("barrel_rate", 0.06),
    }
    for pt, v in LEAGUE_AVG_BY_PITCH_TYPE.items()
}

# Build lookups
rank_lookup = {}
for _, r in hitter_rankings.iterrows():
    bid = int(r['batter_id'])
    rank_lookup[bid] = {
        'tdd_value_score': r.get('tdd_value_score', np.nan),
        'rank': r.get('rank', np.nan),
    }

proj_lookup = {}
for _, r in hitter_proj.iterrows():
    bid = int(r['batter_id'])
    proj_lookup[bid] = {
        'batter_name': r.get('batter_name', ''),
        'projected_k_rate': r.get('projected_k_rate'),
        'projected_bb_rate': r.get('projected_bb_rate'),
    }

count_lookup = {}
for _, r in hitter_count.iterrows():
    bid = int(r['batter_id'])
    count_lookup[bid] = {
        'total_hr_mean': r.get('total_hr_mean'),
        'total_k_mean': r.get('total_k_mean'),
        'total_bb_mean': r.get('total_bb_mean'),
        'projected_wrc_plus': r.get('projected_wrc_plus'),
    }

arch_lookup = {}
for _, r in hitter_arch.iterrows():
    arch_lookup[int(r['batter_id'])] = r.get('archetype_name', '')

p_arch_lookup = {}
for _, r in pitcher_arch.iterrows():
    p_arch_lookup[int(r['pitcher_id'])] = r.get('archetype_name', '')

p_proj_lookup = {}
for _, r in pitcher_proj.iterrows():
    p_proj_lookup[int(r['pitcher_id'])] = {
        'pitcher_name': r.get('pitcher_name', ''),
        'projected_k_rate': r.get('projected_k_rate'),
    }

# Fetch today's schedule & lineups
schedule = fetch_todays_schedule()
print(f"Games today: {len(schedule)}")

# Score every hitter vs opposing SP
results = []
for _, game in schedule.iterrows():
    gpk = game['game_pk']
    lineups = fetch_game_lineups(gpk)

    for side in ['away', 'home']:
        opp_side = 'home' if side == 'away' else 'away'
        opp_pid = game.get(f'{opp_side}_pitcher_id')
        opp_pname = game.get(f'{opp_side}_pitcher_name', 'TBD')

        if pd.isna(opp_pid):
            continue
        opp_pid = int(opp_pid)

        team_tid = game.get(f'{side}_team_id')
        team_abbr = game.get(f'{side}_abbr', '')
        opp_abbr = game.get(f'{opp_side}_abbr', '')

        team_lu = (
            lineups[lineups['team_id'] == team_tid]
            if not lineups.empty and pd.notna(team_tid)
            else pd.DataFrame()
        )

        if team_lu.empty:
            continue

        for _, brow in team_lu.head(9).iterrows():
            bid = int(brow['batter_id']) if pd.notna(brow.get('batter_id')) else None
            if bid is None:
                continue

            bname = brow.get('batter_name', 'Unknown')
            order = int(brow.get('batting_order', 0))

            # Score matchup vs opposing SP
            kr = score_matchup(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            br = score_matchup_bb(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            hr = score_matchup_hr(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)

            k_lift = kr.get('matchup_k_logit_lift', 0.0)
            bb_lift = br.get('matchup_bb_logit_lift', 0.0)
            hr_lift = hr.get('matchup_hr_logit_lift', 0.0)
            k_lift = 0.0 if np.isnan(k_lift) else k_lift
            bb_lift = 0.0 if np.isnan(bb_lift) else bb_lift
            hr_lift = 0.0 if np.isnan(hr_lift) else hr_lift

            # From hitter POV: negative k_lift = hitter advantage (fewer Ks)
            # positive bb_lift = more walks for hitter
            # positive hr_lift = more HR for hitter
            # Hitter net: -k_lift + bb_lift*0.5 + hr_lift*0.5
            hitter_net = -k_lift + bb_lift * 0.5 + hr_lift * 0.5

            rank_info = rank_lookup.get(bid, {})
            proj_info = proj_lookup.get(bid, {})
            cnt_info = count_lookup.get(bid, {})
            tdd_score = rank_info.get('tdd_value_score', np.nan)

            results.append({
                'batter_name': bname,
                'batter_id': bid,
                'team': team_abbr,
                'vs_pitcher': opp_pname,
                'vs_pitcher_id': opp_pid,
                'vs_pitcher_k_rate': p_proj_lookup.get(opp_pid, {}).get('projected_k_rate'),
                'opponent': opp_abbr,
                'batting_order': order,
                'archetype': arch_lookup.get(bid, ''),
                'tdd_value_score': tdd_score,
                'proj_k_rate': proj_info.get('projected_k_rate'),
                'proj_bb_rate': proj_info.get('projected_bb_rate'),
                'proj_hr': cnt_info.get('total_hr_mean'),
                'proj_wrc_plus': cnt_info.get('projected_wrc_plus'),
                'k_lift': k_lift,
                'bb_lift': bb_lift,
                'hr_lift': hr_lift,
                'hitter_net_matchup': hitter_net,
                'game_time': game.get('game_time', ''),
            })

df = pd.DataFrame(results)

# "Sleeper" filter: lower tdd_value_score = less likely rostered
# Use value score percentile to approximate ownership
# Bottom 50-75% of ranked hitters are likely available in many leagues
if df.empty:
    print("No lineup data available yet.")
    exit()

# Determine a sleeper threshold: below median tdd_value_score among today's hitters
scored = df[df['tdd_value_score'].notna()].copy()
median_score = scored['tdd_value_score'].median()
q75 = scored['tdd_value_score'].quantile(0.75)
q25 = scored['tdd_value_score'].quantile(0.25)

print(f"TDD value score distribution (today's hitters): median={median_score:.1f}, Q25={q25:.1f}, Q75={q75:.1f}")
print()

# Sleepers: below median value score BUT favorable matchup
sleepers = df[
    (df['tdd_value_score'].isna() | (df['tdd_value_score'] < median_score))
    & (df['hitter_net_matchup'] > 0.0)
].sort_values('hitter_net_matchup', ascending=False)

# Also show unranked hitters (no tdd_value_score) with good matchups — these are deep sleepers
deep = df[df['tdd_value_score'].isna() & (df['hitter_net_matchup'] > 0.0)].sort_values('hitter_net_matchup', ascending=False)

def fmt_pct(v):
    return f"{v*100:.1f}%" if pd.notna(v) else "N/A"

def fmt_hr(v):
    return f"{v:.0f}" if pd.notna(v) else "N/A"

def fmt_wrc(v):
    return f"{v:.0f}" if pd.notna(v) else "N/A"


print("=" * 95)
print("HITTER SLEEPER PICKS — FAVORABLE MATCHUPS TODAY")
print("Lower TDD value score = more likely available on waivers")
print("=" * 95)

for i, (_, row) in enumerate(sleepers.head(25).iterrows(), 1):
    arch = row['archetype'] or 'N/A'
    vs_k = fmt_pct(row['vs_pitcher_k_rate'])
    score_str = f"{row['tdd_value_score']:.1f}" if pd.notna(row['tdd_value_score']) else "UNRANKED"
    print()
    print(f"  #{i}  {row['batter_name']} ({row['team']}) — bats {row['batting_order']}th")
    print(f"       vs {row['vs_pitcher']} ({row['opponent']}, K% {vs_k}) — {row['game_time']}")
    print(f"       TDD Score: {score_str}  |  Archetype: {arch}")
    print(f"       Proj K%: {fmt_pct(row['proj_k_rate'])}  |  BB%: {fmt_pct(row['proj_bb_rate'])}  |  HR: {fmt_hr(row['proj_hr'])}  |  wRC+: {fmt_wrc(row['proj_wrc_plus'])}")
    print(f"       Matchup K lift: {row['k_lift']:+.3f}  |  BB lift: {row['bb_lift']:+.3f}  |  HR lift: {row['hr_lift']:+.3f}")
    print(f"       HITTER NET: {row['hitter_net_matchup']:+.3f}")

print()
print("=" * 95)
print("KEY:")
print("  K lift < 0 = hitter strikes out LESS vs this pitcher (hitter advantage)")
print("  BB lift > 0 = hitter draws MORE walks vs this pitcher")
print("  HR lift > 0 = hitter hits MORE HRs vs this pitcher")
print("  HITTER NET = -K_lift + 0.5*BB_lift + 0.5*HR_lift  (higher = better hitter matchup)")
print("  TDD Score: lower = less fantasy value = more likely on your waiver wire")
print("=" * 95)
