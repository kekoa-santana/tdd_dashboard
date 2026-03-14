# CLAUDE.md — The Data Diamond Dashboard

## Project Overview
Interactive Streamlit dashboard for MLB player analytics, powered by Bayesian projections from the `player_profiles` projection engine. Displays season-level projections, traditional stats, matchup analysis, game-level K predictions, and daily game coverage. Built by The Data Diamond (Koa).

**This is the display/presentation layer.** All Bayesian model training, backtesting, and precomputation happens in the sibling `player_profiles` repo. This repo consumes pre-computed parquet/npz files and handles live updates + UI.

**⚠️ ARCHITECTURE NOTE:** The current implementation has drifted from the intended architecture. The dashboard's `update_in_season.py` currently performs projection updating that should be moved to `player_profiles`. See PLAN.md for the target architecture.

## Related Repos
- **Projection engine:** `C:/Users/kekoa/Documents/data_analytics/player_profiles/` — Bayesian models, backtesting, precompute
- **Theme package:** `tdd_theme` — shared brand colors/utilities (pip-installed)

## Tech Stack
- **Language:** Python 3.11+
- **UI:** Streamlit
- **Data:** Pre-computed parquets + npz (no direct model training)
- **Database:** PostgreSQL (`mlb_fantasy` on `localhost:5433`) — used by `update_in_season.py` for fetching observed 2026 totals and performing projection updates
- **API:** MLB Stats API (schedule, lineups, probable pitchers)
- **Visualization:** matplotlib, scipy (KDE plots)
- **Computation:** numpy, pandas, scipy

## Project Structure
```
tdd-dashboard/
├── app.py                        # Main Streamlit dashboard (~383 lines, modularized)
├── .streamlit/config.toml        # Dark theme config
├── iconTransparent.png           # Brand icon
├── SYNC_GUIDE.md                 # How to sync lib/ from projection engine
├── lib/                          # Computation modules (synced from player_profiles)
│   ├── __init__.py
│   ├── constants.py              # Pitch maps, zone boundaries, league avgs
│   ├── theme.py                  # TDD brand colors + watermark
│   ├── matchup.py                # Pitch-type matchup scoring (log-odds)
│   ├── bf_model.py               # Batters-faced distribution lookup
│   ├── game_k_model.py           # Game K Monte Carlo simulator
│   ├── zone_charts.py            # Pitcher location + hitter zone heatmaps
│   ├── in_season_updater.py      # Beta-Binomial conjugate updating
│   ├── schedule.py               # MLB Stats API schedule/lineup fetcher
│   └── db.py                     # SQLAlchemy read_sql helper
├── pages/                        # Modular dashboard pages
│   ├── schedule.py               # Today's Games + Game Browser (unified)
│   ├── projections.py            # Projections + Stats (unified)
│   ├── player_profile.py         # Full player analytics page
│   ├── team_overview.py          # Team identity and roster analysis
│   ├── matchup_explorer.py       # Pitcher vs batter matchup scoring
│   ├── game_k_sim.py             # Interactive K prop simulator
│   ├── preseason_snapshot.py     # Preseason vs current comparison
│   ├── data_health.py            # Data freshness and manifest validation
│   └── model_performance.py      # Model accuracy and backtest results
├── components/                   # Shared UI components
│   ├── charts.py                 # Common chart utilities
│   ├── tables.py                 # Data table components
│   ├── metric_cards.py           # Stat card displays
│   ├── scouting.py               # Scouting report visualizations
│   └── backtest_charts.py        # Model performance charts
├── services/                     # Data loading and validation
│   ├── data_loader.py            # Cached parquet loaders
│   └── manifest.py               # Artifact contract validation
├── scripts/
│   ├── update_in_season.py       # Daily update pipeline (CURRENTLY DOES PROJECTION WORK)
│   └── sync_lib.py                # Scripted lib/ sync from player_profiles
└── data/dashboard/               # Pre-computed data (~43 files)
    ├── *_projections.parquet     # Rate projections (K%, BB%)
    ├── *_counting.parquet        # Count projections (K, BB, HR)
    ├── *_traditional*.parquet    # Observed stats (AVG, ERA, etc.)
    ├── pitcher_k_samples.npz    # Posterior K% samples
    ├── bf_priors.parquet         # BF distribution priors
    ├── todays_*.parquet          # Daily schedule/sims/lineups
    ├── update_metadata.json      # Last update timestamp
    └── snapshots/                # Frozen preseason baselines
```

## Dashboard Pages
1. **Schedule** — Today's Games + historical Game Browser (unified)
2. **Projections** — Hitter/pitcher rate + counting stat projections with search/filter (includes Stats)
3. **Player Profile** — Full player page: projections, percentiles, scouting report, approach/efficiency, arsenal/vulnerability, zone charts, season trends
4. **Team Overview** — Team identity, roster strengths, injury list
5. **Matchup Explorer** — Pitcher vs batter matchup scoring with zone overlay
6. **Game K Simulator** — Interactive K prop simulator (lineup, umpire, weather controls)
7. **Preseason Snapshot** — Compare current vs preseason projections
8. **Data Health** — Data freshness, artifact inventory, manifest validation
9. **Model Performance** — Predicted vs actual tracking, backtest results, calibration curves

## Data Flow

**CURRENT IMPLEMENTATION (⚠️ ARCHITECTURE VIOLATION):**
```
tdd-dashboard/scripts/update_in_season.py
  ├── Queries database directly for 2026 season totals
  ├── Performs Beta-Binomial conjugate updating
  ├── Updates projections and K% samples
  ├── Fetches daily schedule/lineups from MLB API
  ├── Runs game simulations
  └── Writes all parquets + manifest
         |
         +---> data/dashboard/*.parquet
                  |
                  v
              app.py (reads parquets, zero DB calls)
```

**INTENDED ARCHITECTURE (see PLAN.md):**
```
player_profiles/                         tdd-dashboard/
  precompute_dashboard_data.py             data/dashboard/*.parquet
  update_in_season.py (model work)                  |
         |                                          |
         +---> writes parquets -------->            |
                                              app.py (reads parquets, zero DB calls)
                                              scripts/update_in_season.py (bookkeeping only)
```

### Three projection tiers:
| Tier | Source | Frequency |
|------|--------|-----------|
| Preseason projections | `player_profiles` precompute | Once (frozen in snapshots/) |
| Updated projections | `tdd-dashboard` conjugate updates (⚠️ should move to player_profiles) | Daily during season |
| Daily game projections | Live simulator in app.py | On-demand per game |

## lib/ — Synced Modules

These files are copied from `player_profiles/src/` with `from src.` changed to `from lib.`. See `SYNC_GUIDE.md` for the full mapping.

**When to sync:** Only when function signatures or behavior change in the projection engine. Model training changes (priors, covariates, MCMC structure) do NOT require syncing — they only affect the parquet outputs.

**How to sync:**
1. Copy the file from `player_profiles/src/...` to `tdd-dashboard/lib/`
2. Replace `from src.` imports with `from lib.`
3. Test: `python -c "from lib.<module> import <function>"`

## Running the Dashboard

```bash
# First time: copy data from projection engine
cp -r ../player_profiles/data/dashboard/* data/dashboard/

# Run dashboard
streamlit run app.py

# Daily in-season update (currently does projection work + dashboard bookkeeping)
python scripts/update_in_season.py
python scripts/update_in_season.py --date 2026-04-15  # specific date
python scripts/update_in_season.py --skip-schedule    # skip MLB API calls
python scripts/update_in_season.py --snapshot          # force a weekly projection snapshot
```

## Coding Standards
- **Python 3.11+**, type hints on function signatures
- **No database calls in app.py** — all data comes from pre-computed parquets
- **Dark theme** — all matplotlib charts use `DARK` background, not the cream `apply_theme()`
- **Standard chart size:** `figsize=(7, 3)` for dashboard matplotlib charts
- **Colors from theme only** — import from `lib.theme`, never hardcode hex values
- **Brand colors:** GOLD (#C8A96E), EMBER (#D4562A), SAGE (#6BA38E), SLATE (#7B8FA6), CREAM (#F5F2EE), DARK (#0F1117)

## Database Context
- **app.py has zero database calls** — purely reads parquets for instant load times
- **update_in_season.py makes database calls** — queries 2026 season totals for projection updating (⚠️ should move to player_profiles)
- **DB details:** See `player_profiles/CLAUDE.md` for `mlb_fantasy` on `localhost:5433`

## Key Design Decisions
- **app.py has zero database calls** — purely reads parquets for instant load times
- **Modular architecture** — app.py reduced to 383 lines, pages split into separate modules
- **Conjugate updating** (Beta-Binomial) for in-season projection updates — instant vs re-running MCMC (currently in dashboard, should move to player_profiles)
- **Preseason snapshots** frozen for honest before/after comparison as season progresses
- **Season selector** on Player Profile, Projections, Stats pages — any season 2018-2025 + career + 2026 projection
- **Pre-2022 batted ball warning** — Statcast coverage unreliable before 2022, affected stats are hidden
- **Contract validation** — manifest.json validates artifact schemas and row counts on startup
- **Automated testing** — 41 smoke tests ensure all pages render with fixture data
