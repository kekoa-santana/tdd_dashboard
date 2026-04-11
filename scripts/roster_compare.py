"""Compare user's fantasy roster vs sleeper picks using TDD system."""
import sys
sys.path.insert(0, r'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard')

import pandas as pd
import numpy as np
from pathlib import Path
from lib.schedule import fetch_todays_schedule, fetch_game_lineups
from lib.matchup import score_matchup, score_matchup_bb, score_matchup_hr
from lib.constants import LEAGUE_AVG_BY_PITCH_TYPE

DASHBOARD_DIR = Path(r'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard')

# Load data
hitter_proj = pd.read_parquet(DASHBOARD_DIR / 'hitter_projections.parquet')
hitter_count = pd.read_parquet(DASHBOARD_DIR / 'hitter_counting_sim.parquet')
hitter_rankings = pd.read_parquet(DASHBOARD_DIR / 'hitters_rankings.parquet')
hitter_arch = pd.read_parquet(DASHBOARD_DIR / 'hitter_archetypes.parquet')
pitcher_proj = pd.read_parquet(DASHBOARD_DIR / 'pitcher_projections.parquet')
pitcher_rankings = pd.read_parquet(DASHBOARD_DIR / 'pitchers_rankings.parquet')
pitcher_arch = pd.read_parquet(DASHBOARD_DIR / 'pitcher_archetypes.parquet')
pitcher_count = pd.read_parquet(DASHBOARD_DIR / 'pitcher_counting_sim.parquet')
arsenal_df = pd.read_parquet(DASHBOARD_DIR / 'pitcher_arsenal.parquet')
vuln_df = pd.read_parquet(DASHBOARD_DIR / 'hitter_vuln_career.parquet')

baselines_pt = {
    pt: {"whiff_rate": v.get("whiff_rate", 0.25), "chase_rate": v.get("chase_rate", 0.30), "barrel_rate": v.get("barrel_rate", 0.06)}
    for pt, v in LEAGUE_AVG_BY_PITCH_TYPE.items()
}

# User's roster - search by name fragments
my_hitters = [
    "Ramirez",    # Augustin Ramirez
    "Rice",       # Ben Rice
    "Guerrero",   # Vlad Jr
    "Turang",     # Brice Turang
    "Riley",      # Austin Riley
    "Lindor",     # Francisco Lindor
    "Frelick",    # Sal Frelick
    "Pages",      # Andy Pages
    "Rooker",     # Brent Rooker
    "Murakami",   # Munetaka Murakami
    "Sanoja",     # Javier Sanoja
]

my_pitchers = [
    "Ashcraft",   # Braxton Ashcraft
    "Luzardo",    # Jesus Luzardo
    "Gausman",    # Kevin Gausman
    "Crochet",    # Garrett Crochet
    "Cabrera",    # Edward Cabrera (or similar)
]

# Find hitters in system
print("=" * 100)
print("YOUR HITTER ROSTER — TDD SYSTEM LOOKUP")
print("=" * 100)

def find_hitter(name_frag):
    """Search across hitter datasets for a player."""
    matches = hitter_proj[hitter_proj['batter_name'].str.contains(name_frag, case=False, na=False)]
    if matches.empty:
        matches = hitter_count[hitter_count['batter_name'].str.contains(name_frag, case=False, na=False)]
    if matches.empty:
        matches = hitter_rankings[hitter_rankings['batter_name'].str.contains(name_frag, case=False, na=False)] if 'batter_name' in hitter_rankings.columns else pd.DataFrame()
    return matches

def find_pitcher(name_frag):
    matches = pitcher_proj[pitcher_proj['pitcher_name'].str.contains(name_frag, case=False, na=False)]
    if matches.empty:
        matches = pitcher_count[pitcher_count['pitcher_name'].str.contains(name_frag, case=False, na=False)]
    return matches

# Build rank/score lookups
h_rank_lookup = {}
for _, r in hitter_rankings.iterrows():
    bid = int(r['batter_id'])
    h_rank_lookup[bid] = {
        'tdd_value_score': r.get('tdd_value_score', np.nan),
        'rank': r.get('rank', np.nan),
    }
    # Also try name
    if 'batter_name' in r.index:
        h_rank_lookup[r['batter_name']] = h_rank_lookup[bid]

h_arch_lookup = {}
for _, r in hitter_arch.iterrows():
    h_arch_lookup[int(r['batter_id'])] = r.get('archetype_name', '')

h_count_lookup = {}
for _, r in hitter_count.iterrows():
    bid = int(r['batter_id'])
    h_count_lookup[bid] = {
        'total_hr_mean': r.get('total_hr_mean'),
        'total_k_mean': r.get('total_k_mean'),
        'total_bb_mean': r.get('total_bb_mean'),
        'total_r_mean': r.get('total_r_mean'),
        'total_rbi_mean': r.get('total_rbi_mean'),
        'total_sb_mean': r.get('total_sb_mean'),
        'projected_wrc_plus': r.get('projected_wrc_plus'),
    }

p_rank_lookup = {}
for _, r in pitcher_rankings.iterrows():
    pid = int(r['pitcher_id'])
    p_rank_lookup[pid] = {
        'tdd_value_score': r.get('tdd_value_score', np.nan),
        'rank': r.get('rank', np.nan),
    }

p_count_lookup = {}
for _, r in pitcher_count.iterrows():
    pid = int(r['pitcher_id'])
    p_count_lookup[pid] = {
        'total_k_mean': r.get('total_k_mean'),
        'total_sv_mean': r.get('total_sv_mean'),
        'total_hld_mean': r.get('total_hld_mean'),
        'projected_ip': r.get('projected_ip'),
        'projected_fip_era': r.get('projected_fip_era'),
    }

p_arch_lookup = {}
for _, r in pitcher_arch.iterrows():
    p_arch_lookup[int(r['pitcher_id'])] = r.get('archetype_name', '')

def fmt_pct(v):
    return f"{v*100:.1f}%" if pd.notna(v) else "N/A"

def fmt_f(v, dec=0):
    return f"{v:.{dec}f}" if pd.notna(v) else "N/A"

# Fetch today's schedule for matchup context
schedule = fetch_todays_schedule()
all_lineups = {}
for _, game in schedule.iterrows():
    gpk = game['game_pk']
    lu = fetch_game_lineups(gpk)
    all_lineups[gpk] = lu

# Build today's matchup map: batter_id -> opposing pitcher info
todays_matchups = {}
for _, game in schedule.iterrows():
    gpk = game['game_pk']
    lu = all_lineups.get(gpk, pd.DataFrame())
    for side in ['away', 'home']:
        opp_side = 'home' if side == 'away' else 'away'
        opp_pid = game.get(f'{opp_side}_pitcher_id')
        opp_pname = game.get(f'{opp_side}_pitcher_name', 'TBD')
        team_tid = game.get(f'{side}_team_id')
        team_abbr = game.get(f'{side}_abbr', '')
        opp_abbr = game.get(f'{opp_side}_abbr', '')
        gt = game.get('game_time', '')

        if pd.isna(opp_pid):
            continue
        opp_pid = int(opp_pid)

        team_lu = lu[lu['team_id'] == team_tid] if not lu.empty and pd.notna(team_tid) else pd.DataFrame()
        if team_lu.empty:
            continue

        for _, brow in team_lu.head(9).iterrows():
            bid = int(brow['batter_id']) if pd.notna(brow.get('batter_id')) else None
            if bid is None:
                continue
            order = int(brow.get('batting_order', 0))

            # Score matchup
            kr = score_matchup(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            br = score_matchup_bb(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            hr = score_matchup_hr(opp_pid, bid, arsenal_df, vuln_df, baselines_pt)
            k_lift = kr.get('matchup_k_logit_lift', 0.0)
            bb_lift = br.get('matchup_bb_logit_lift', 0.0)
            hr_lift = hr.get('matchup_hr_logit_lift', 0.0)
            k_lift = 0.0 if np.isnan(k_lift) else k_lift
            bb_lift = 0.0 if np.isnan(bb_lift) else bb_lift
            hr_lift = 0.0 if np.isnan(hr_lift) else hr_lift
            hitter_net = -k_lift + bb_lift * 0.5 + hr_lift * 0.5

            todays_matchups[bid] = {
                'vs_pitcher': opp_pname,
                'vs_pitcher_id': opp_pid,
                'team': team_abbr,
                'opponent': opp_abbr,
                'batting_order': order,
                'k_lift': k_lift,
                'bb_lift': bb_lift,
                'hr_lift': hr_lift,
                'hitter_net': hitter_net,
                'game_time': gt,
            }

# Also build pitcher matchup map
todays_pitcher_matchups = {}
for _, game in schedule.iterrows():
    for side in ['away', 'home']:
        pid = game.get(f'{side}_pitcher_id')
        if pd.isna(pid):
            continue
        pid = int(pid)
        pname = game.get(f'{side}_pitcher_name', 'TBD')
        opp_side = 'home' if side == 'away' else 'away'
        team_abbr = game.get(f'{side}_abbr', '')
        opp_abbr = game.get(f'{opp_side}_abbr', '')
        gt = game.get('game_time', '')
        todays_pitcher_matchups[pid] = {
            'pitcher_name': pname,
            'team': team_abbr,
            'opponent': opp_abbr,
            'game_time': gt,
        }

# Display hitter roster
for name_frag in my_hitters:
    matches = find_hitter(name_frag)
    if matches.empty:
        print(f"\n  {name_frag}: NOT FOUND in system")
        continue

    row = matches.iloc[0]
    bid = int(row['batter_id'])
    bname = row.get('batter_name', name_frag)
    proj_k = row.get('projected_k_rate')
    proj_bb = row.get('projected_bb_rate')
    arch = h_arch_lookup.get(bid, 'N/A')
    rank_info = h_rank_lookup.get(bid, {})
    tdd = rank_info.get('tdd_value_score', np.nan)
    rnk = rank_info.get('rank', np.nan)
    cnt = h_count_lookup.get(bid, {})

    tdd_str = f"{tdd:.1f}" if pd.notna(tdd) else "UNRANKED"
    rnk_str = f"#{int(rnk)}" if pd.notna(rnk) else "N/A"

    print(f"\n  {bname} (ID: {bid})")
    print(f"    TDD Score: {tdd_str}  |  Rank: {rnk_str}  |  Archetype: {arch}")
    print(f"    Proj K%: {fmt_pct(proj_k)}  |  BB%: {fmt_pct(proj_bb)}")
    print(f"    Proj HR: {fmt_f(cnt.get('total_hr_mean'))}  |  R: {fmt_f(cnt.get('total_r_mean'))}  |  RBI: {fmt_f(cnt.get('total_rbi_mean'))}  |  SB: {fmt_f(cnt.get('total_sb_mean'))}  |  wRC+: {fmt_f(cnt.get('projected_wrc_plus'))}")

    # Today's matchup
    m = todays_matchups.get(bid)
    if m:
        print(f"    TODAY: {m['team']} vs {m['vs_pitcher']} ({m['opponent']}) — {m['game_time']} — bat {m['batting_order']}")
        print(f"           K lift: {m['k_lift']:+.3f}  |  BB lift: {m['bb_lift']:+.3f}  |  HR lift: {m['hr_lift']:+.3f}  |  NET: {m['hitter_net']:+.3f}")
    else:
        print(f"    TODAY: Not in a confirmed lineup / no game found")

print()
print("=" * 100)
print("YOUR PITCHER ROSTER — TDD SYSTEM LOOKUP")
print("=" * 100)

for name_frag in my_pitchers:
    matches = find_pitcher(name_frag)
    if matches.empty:
        print(f"\n  {name_frag}: NOT FOUND in system")
        continue

    row = matches.iloc[0]
    pid = int(row['pitcher_id'])
    pname = row.get('pitcher_name', name_frag)
    proj_k = row.get('projected_k_rate')
    proj_bb = row.get('projected_bb_rate')
    arch = p_arch_lookup.get(pid, 'N/A')
    rank_info = p_rank_lookup.get(pid, {})
    tdd = rank_info.get('tdd_value_score', np.nan)
    rnk = rank_info.get('rank', np.nan)
    cnt = p_count_lookup.get(pid, {})

    tdd_str = f"{tdd:.1f}" if pd.notna(tdd) else "UNRANKED"
    rnk_str = f"#{int(rnk)}" if pd.notna(rnk) else "N/A"

    print(f"\n  {pname} (ID: {pid})")
    print(f"    TDD Score: {tdd_str}  |  Rank: {rnk_str}  |  Archetype: {arch}")
    print(f"    Proj K%: {fmt_pct(proj_k)}  |  BB%: {fmt_pct(proj_bb)}")
    print(f"    Proj K: {fmt_f(cnt.get('total_k_mean'))}  |  IP: {fmt_f(cnt.get('projected_ip'))}  |  FIP-ERA: {fmt_f(cnt.get('projected_fip_era'), 2)}")

    # Today's start?
    pm = todays_pitcher_matchups.get(pid)
    if pm:
        print(f"    TODAY: {pm['team']} vs {pm['opponent']} — {pm['game_time']}")
    else:
        print(f"    TODAY: Not starting")

# Now the comparison: sleepers vs weakest roster spots
print()
print("=" * 100)
print("ROSTER COMPARISON: YOUR WEAKEST SPOTS vs BEST SLEEPERS")
print("=" * 100)

# Gather your hitter data with scores
my_hitter_data = []
for name_frag in my_hitters:
    matches = find_hitter(name_frag)
    if matches.empty:
        my_hitter_data.append({'name': name_frag, 'bid': None, 'tdd': np.nan, 'proj_hr': np.nan, 'today_net': np.nan, 'in_lineup': False})
        continue
    row = matches.iloc[0]
    bid = int(row['batter_id'])
    rank_info = h_rank_lookup.get(bid, {})
    tdd = rank_info.get('tdd_value_score', np.nan)
    cnt = h_count_lookup.get(bid, {})
    m = todays_matchups.get(bid)
    my_hitter_data.append({
        'name': row.get('batter_name', name_frag),
        'bid': bid,
        'tdd': tdd,
        'proj_hr': cnt.get('total_hr_mean', np.nan),
        'proj_wrc': cnt.get('projected_wrc_plus', np.nan),
        'today_net': m['hitter_net'] if m else np.nan,
        'in_lineup': m is not None,
        'matchup_info': m,
    })

# Sort your hitters by TDD score (weakest first)
my_hitter_data.sort(key=lambda x: x['tdd'] if pd.notna(x['tdd']) else -999)

print("\nYour hitters ranked by TDD value (weakest first):")
for i, h in enumerate(my_hitter_data, 1):
    tdd_str = f"{h['tdd']:.1f}" if pd.notna(h['tdd']) else "UNRANKED"
    net_str = f"{h['today_net']:+.3f}" if pd.notna(h['today_net']) else "no game/LU"
    lu_str = "IN LINEUP" if h['in_lineup'] else "NOT IN LINEUP"
    print(f"  {i}. {h['name']:<25} TDD: {tdd_str:<10} Today NET: {net_str:<10} {lu_str}")

# Top sleeper candidates (from previous analysis)
sleeper_names = [
    "Chandler Simpson", "Jacob Wilson", "Nick Fortes", "Nathan Lukes",
    "Evan Carter", "Masataka Yoshida", "Liam Hicks", "Edgar Quero",
    "Marcelo Mayer", "Nolan Schanuel", "Alec Burleson", "Danny Jansen",
    "Xavier Edwards",
]

print("\nTop available sleepers with today's matchup edge:")
for sname in sleeper_names:
    # Find in hitter data
    sm = hitter_proj[hitter_proj['batter_name'].str.contains(sname.split()[-1], case=False, na=False)]
    if sm.empty:
        sm = hitter_count[hitter_count['batter_name'].str.contains(sname.split()[-1], case=False, na=False)]
    if sm.empty:
        continue
    srow = sm.iloc[0]
    sbid = int(srow['batter_id'])
    srank = h_rank_lookup.get(sbid, {})
    stdd = srank.get('tdd_value_score', np.nan)
    scnt = h_count_lookup.get(sbid, {})
    smatch = todays_matchups.get(sbid)
    if smatch is None:
        continue
    tdd_str = f"{stdd:.1f}" if pd.notna(stdd) else "UNRANKED"
    hr_str = fmt_f(scnt.get('total_hr_mean'))
    print(f"  {srow.get('batter_name', sname):<25} TDD: {tdd_str:<10} Today NET: {smatch['hitter_net']:+.3f}  vs {smatch['vs_pitcher']} ({smatch['opponent']})  HR proj: {hr_str}  bat {smatch['batting_order']}")
