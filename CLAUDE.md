# CLAUDE.md — The Data Diamond Dashboard

## Project Overview
Streamlit dashboard for MLB player projections, powered by Bayesian models from the sibling `player_profiles` repo. Shows season-long projections, per-game simulated stat lines, matchup analysis, and model accuracy tracking.

**This is the presentation layer plus the daily pipeline.** Model training, backtesting and precompute live in `player_profiles`. This repo renders pre-computed artifacts and orchestrates the daily run that produces and publishes them.

See `README.md` for the public-facing description of the model and architecture.

## Related Repos
- **Projection engine:** `C:/Users/kekoa/Documents/data_analytics/player_profiles/` — Bayesian models, game simulator, precompute
- **ETL:** `C:/Users/kekoa/Documents/data_analytics/mlb_fantasy_ETL/` — Statcast + MLB API into PostgreSQL
- **Theme package:** `tdd_theme` — shared brand colors (pip-installed)

## Tech Stack
- **Language:** Python 3.11+
- **UI:** Streamlit (dark theme, custom CSS in `assets/styles.css`)
- **Data:** Pre-computed parquet/npz/json artifacts. No model training, no DB calls in the app.
- **Storage:** Cloudflare R2 (public bucket). Artifacts are NOT committed to git.
- **Database:** PostgreSQL `mlb_fantasy` on `localhost:5433` — a Docker container (`mlb_postgres`), used only by pipeline scripts
- **API:** MLB Stats API (schedule, lineups, boxscores, season series)
- **Visualization:** plotly (primary), matplotlib (zone charts)

## Where data comes from

```
data/dashboard/          local artifacts, gitignored, produced by the pipeline
services/artifacts.py    resolves a name -> local file, else container cache, else R2
services/data_loader.py  cached loaders; every read goes through artifact_path()
```

`artifact_path()` never raises. A missing artifact resolves to a path that does not
exist, and callers degrade with `if not path.exists(): return pd.DataFrame()`.
Local files always win, so development stays offline.

**Never read `DASHBOARD_DIR / name` directly in app code** — use `artifact_path(name)`,
or the deployed app (which has no local data) breaks.

## Project Structure
```
app.py                  entry point: nav, routing, page registry (PAGES / PAGE_URL_MAP)
config.py               seasons, colors, paths (values come from runtime.yaml)
views/                  one module per page (19 pages)
components/             shared renderers: projection_table, sim_chart, leaderboard,
                        attribution, metric_cards, headshot, grades, team_logo
services/               artifacts.py, data_loader.py, manifest.py
lib/                    computation modules synced from player_profiles (see docs/SYNC_GUIDE.md)
scripts/                daily_update.bat, update_in_season.py, publish_artifacts.py,
                        setup_scheduled_tasks.ps1, validation scripts
assets/styles.css       all CSS; app.py only injects :root color variables
tests/                  91 tests
```

## Pages
Schedule, Game Analysis, Player Projections, Home, News, Player Profile, Player Rankings,
Stats, The Diamond Daily, Projections, Breakout Candidates, Team Overview, Team Rankings,
Division Standings, Compare Players, Lineup Creator, Model Performance, Data Health, Methodology.

`PAGE_URL_MAP` keeps old slugs working (`?page=props_lab` resolves to Player Projections).

## Daily pipeline
Two Windows scheduled tasks, both running `scripts/daily_update.bat`:

| Task | When | Mode |
|---|---|---|
| TDD Full Daily Update | 6:00 AM | ETL, precompute, projections, team-run scoring, full sims, publish |
| TDD Intraday Refresh | every 15 min, 9 AM to midnight | `--intraday`: re-sim changed games, live standouts, publish |

- **Intraday is change-driven.** `confident_picks.run(incremental=True)` re-simulates only
  pre-game games whose probable starters, HP umpire, or confirmed lineups changed.
  Exit code 3 means nothing changed. Fingerprints live in `data/dashboard/history/sim_input_state.json`.
- **Publishing goes to R2, never git.** `publish_artifacts.py` uploads only changed objects
  and only the set the dashboard reads; it writes `index.json` so the app can enumerate.
- **Step order matters.** Precompute must run BEFORE `update_in_season.py`: both write
  `hitter_traditional.parquet` / `pitcher_traditional.parquet`, precompute with the last
  completed season and the in-season step with the current one. The dashboard reads those
  as current-season stats.
- **The lock holds the owning PID.** A killed run is detected and cleared by the next run.

## Coding Standards
- **Python 3.11+**, type hints on function signatures
- **No database calls in app code** — pipeline scripts only
- **No direct artifact paths in app code** — always `artifact_path()`
- **Colors from CSS variables / `config`**, never hardcoded hex in views
- **Brand colors:** GOLD `#C8A96E`, EMBER `#D4562A`, SAGE `#6BA38E`, SLATE `#7B8FA6`, CREAM `#F5F2EE`, DARK `#0F1117`
- **No em dashes** anywhere in code, comments or UI text
- **No betting language.** This site shows projections: expected values, ranges and
  probabilities of outcomes. No lines, no over/under framing, no edges or picks.
- Mobile matters: pages must not scroll horizontally at ~400px wide

## Key Design Decisions
- **Projections carry ranges.** Displayed ranges are the 10th-90th percentile of simulated
  outcomes, taken from `p_over_{k}` columns in `game_props.parquet`, which is the single
  source of truth for game-level distributions.
- **Accuracy is reported honestly.** Because counting stats are integers, an inclusive
  range holds more than 80% of the mass, so observed coverage is always shown next to the
  model's own expected coverage.
- **Preseason projections never see in-season data.** Full-season rate lines come from
  `snapshots/*_counting_sim_{season}_preseason.parquet`, frozen before Opening Day.
- **Conjugate updating** (Beta-Binomial) for in-season rate updates, instead of re-running MCMC.
- **Contract validation** — `manifest.json` validates artifact schemas and row counts; the
  pipeline fails manifest generation on an unreadable artifact.

## Gotchas
- `data/dashboard/` is gitignored. Do not commit artifacts.
- Cumulative parquet files must be written atomically (temp file + `os.replace`); an
  interrupted in-place write corrupts them and breaks manifest generation.
- Pipeline metadata timestamps are UTC-aware; the deployed app runs in UTC.
- `check_roster_moves` is a no-op (`fetch_recent_transactions` never existed).
- Streamlit Cloud reruns the entry script on push but may keep already-imported modules
  until the container restarts; module-level changes can take a few minutes to appear.

## Running
```bash
streamlit run app.py                      # local, reads data/dashboard
TDD_ARTIFACT_BASE_URL=https://pub-507219cfdcb94d45b71784dbf880db4e.r2.dev streamlit run app.py
pytest -q                                 # 91 tests
python scripts/publish_artifacts.py --dry-run
```
