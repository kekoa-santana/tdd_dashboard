# One-Stop Baseball Analytics Plan (ETL + Modeling + Dashboard)

## Summary
Build a unified, production-ready analytics platform across three repos:
1. `mlb_fantasy_ETL` as the canonical data platform.
2. `player_profiles` as the canonical modeling/backtest platform.
3. `tdd-dashboard` as the canonical product/UI platform.

The roadmap has two phases: **Foundation** (reliability, contracts, automation) then **Expansion** (new analytics features, decision tools, content generation). Foundation comes first because current growth is bottlenecked by pipeline hardening gaps, manual sync drift, and a 5000-line monolith.

**Ownership:** All code changes happen in `tdd-dashboard`. The other repos (`mlb_fantasy_ETL`, `player_profiles`) are read-only references — their issues are noted here for awareness but changes to them are made separately by the owner.

## Scope
In scope:
1. End-to-end data-to-product reliability.
2. Shared data contracts and season/version configuration.
3. Automated update operations (daily + intraday).
4. Product expansion for "one-stop shop" analytics pages.
5. New modeling capabilities (contact quality, team wins, player comps).
6. Decision support tools (betting edge finder, lineup optimizer).
7. Measurable QA and observability.

Out of scope for this plan window:
1. Replacing Streamlit with a new frontend framework.
2. Replacing PostgreSQL or replatforming infrastructure.
3. Rewriting Bayesian model families from scratch.
4. User accounts / authentication.
5. Public API layer (future consideration).

## Target Architecture Decisions
1. **Source-of-truth boundaries**
   - ETL truth: `mlb_fantasy_ETL`.
   - Model truth: `player_profiles`.
   - UI truth: `tdd-dashboard`.
   - Dashboard must not add private model logic; it consumes precomputed artifacts.

2. **Contract-first integration**
   - Introduce versioned artifact contracts for all files in `data/dashboard/`.
   - Every producer run emits schema/version metadata.
   - Every consumer run validates contract before use.

3. **Season/config centralization**
   - Remove hardcoded season constants from runtime logic.
   - Add centralized season/runtime config loaded by all repos.

4. **No manual drift**
   - Replace manual "sync-by-copy" habits with scripted sync + verification checks for shared modules.

5. **Modular dashboard**
   - Break `app.py` (~5000 lines) into page modules + shared data services.
   - Each page is a self-contained module with its own data loader and render function.

## Public APIs / Interfaces / Type Changes
1. Add `config/runtime.yaml` in each repo (same schema).
   - Fields: `current_season`, `train_start_season`, `train_end_season`, `supported_historical_seasons`, `game_types_in_scope`, `refresh_windows`, `schema_version`.
   - Default: `current_season=2026`.

2. Add artifact manifest contract `data/dashboard/manifest.json`.
   - Fields: `artifact_name`, `schema_version`, `row_count`, `column_hash`, `generated_at`, `producer_repo`, `producer_commit`.
   - Produced by `player_profiles` precompute and `update_in_season`.
   - Validated by `tdd-dashboard` loaders on startup.

3. Add ETL run metadata table `production.etl_run_log` and DQ table `production.etl_dq_results`.
   - Types: `run_id`, `pipeline_step`, `status`, `started_at`, `ended_at`, `row_count`, `error_text`, `dq_check_name`, `dq_status`, `dq_details`.

4. Add CLI contracts:
   - `mlb_fantasy_ETL`: `python full_pipeline.py --mode {daily,intraday,backfill} --validate-only`.
   - `player_profiles`: `python scripts/precompute_dashboard_data.py --contract-out data/dashboard/manifest.json`.
   - `tdd-dashboard`: startup flag `--strict-contracts`.

5. Add dashboard page interfaces:
   - `Model Performance` page: consumes backtest outputs and contract metadata.
   - `Data Health` page: consumes ETL/model run logs and freshness checks.
   - `Stats` page added to nav as a first-class route.
   - `Team Outlook` page: consumes aggregated roster projections.

---

## Phase 1: Foundation (Weeks 1-4)

### Milestone 1 (Weeks 1-2): Reliability Baseline

**Workstream: ETL Awareness (read-only — changes made separately)**
- Fix `full_pipeline.py` skip-ingestion variable pathing
- Fix DQ import path mismatch in analysis scripts
- Convert print-based diagnostics to structured logging
- Add deterministic batch size settings for staging upserts
- Add explicit upsert policy switch (`prefer_incoming` vs `fill_null_only`)

**Workstream: Dashboard Testing** ✅ COMPLETE (2026-03-11)
- [x] Add dashboard page smoke tests (each page route renders with fixture data) — 41 tests in tests/test_smoke.py
- [x] Add CI workflow for `tdd-dashboard` (lint + smoke tests + contract checks) — .github/workflows/ci.yml

**Workstream: Dashboard Modularization** ✅ COMPLETE (2026-03-11)
- [x] Break `app.py` into page modules: `pages/schedule.py` (Today's Games + Game Browser merged), `pages/projections.py`, `pages/player_profile.py`, `pages/team_overview.py`, `pages/matchup_explorer.py`, `pages/game_k_sim.py`, `pages/preseason_snapshot.py`
- [x] Extract shared data loading into `services/data_loader.py`
- [x] Extract shared UI components into `components/` (stat cards, tables, charts)
- [x] Add `Stats` page to sidebar navigation as first-class route (merged into Projections via season dropdown)
- [x] Merge "Today's Games" and "Game Browser" into unified `Schedule` page

Exit criteria:
- ✅ `app.py` is under 500 lines (routing + config only) — reduced to 383 lines
- ✅ All pages render correctly after modularization — visually verified
- ✅ Dashboard smoke tests pass (41 tests) — CI workflow configured
- ✅ Schedule page combines today's games view with historical game browsing

### Milestone 2 (Weeks 3-4): Contracted Integration + Quick Wins

**Workstream: Contracts** ✅ COMPLETE (2026-03-11)
- [ ] Implement manifest production in `player_profiles` precompute (out of scope — done in player_profiles repo)
- [x] Implement manifest validation in `tdd-dashboard` startup ✅ — services/manifest.py
- [x] Add shared runtime config loading (`config/runtime.yaml`) ✅
- [x] Scripted sync/verification for `lib/` modules ✅ — scripts/sync_lib.py (--check, --sync, --verify)
- [x] Remove hardcoded 2025/2026 references from dashboard logic and labels ✅

**Workstream: Dashboard Quick Wins (low effort, existing data)** ✅ COMPLETE (2026-03-11)
- [x] Add platoon split toggle on Player Profile (vs LHH / vs RHH / All) ✅ — pitcher arsenal sections
- [x] Add pitch-type multiselect on pitcher zone charts ✅ — both historical and projection views
- [x] Add pitch-type multiselect on hitter zone grid ✅ — added pitch_type to queries.py + precompute, wired into zone_charts.py + player_profile.py
- [x] Wire pitch archetype matchups into Matchup Explorer ✅ — archetype precompute added to player_profiles, Pitch-Type/Archetype toggle on Matchup Explorer
- [x] Add starter/reliever filter on Projections page ✅ — already existed from modularization

Exit criteria:
- ✅ Dashboard startup validates manifest when present (lenient by default)
- ✅ Manifest warnings stored in session state for Data Health page
- ✅ Season/year labels sourced from config, not constants
- ✅ Platoon and pitch-type toggles functional on Player Profile
- ✅ Hitter zone grid supports pitch-type filtering (query + precompute + UI)
- ✅ Archetype matchup toggle functional on Matchup Explorer (with fallback to pitch-type)

---

## Phase 2: Product Expansion (Weeks 5-10)

### Milestone 3 (Weeks 5-6): Observability + Predicted vs Actual

**Workstream: New Pages — Trust & Transparency**
- [x] `Data Health` page: freshness status, last update timestamps, artifact counts, manifest validation ✅ (2026-03-11)
- [ ] `Model Performance` page: predicted vs actual tracker AND backtest results
  - Running Brier scores by stat (K%, BB%) and timeframe (weekly, monthly, season)
  - Calibration curves (predicted probability vs observed frequency)
  - Full backtest results display (walk-forward folds, MAE/RMSE/Brier by stat, vs Marcel comparison)
  - Backtest summary table (imported from `player_profiles`)
  - Biggest misses + biggest hits (model accountability)
- [ ] Preseason vs current projection comparison (sparkline evolution charts)

**Workstream: Season Narrative**
- [ ] Week-over-week projection movement tracking (store weekly snapshots)
- [ ] Auto-generated "biggest movers" section (use existing `plot_pitcher_k_movers()` / `plot_hitter_k_movers()` from player_profiles)
- [ ] Player projection timeline: sparkline of projected K% with narrowing CI bands through season

Exit criteria:
- Predicted-vs-actual metrics visible by stat and timeframe
- Freshness and run status visible to end users
- Weekly snapshots auto-saved by update pipeline

### Milestone 4 (Weeks 7-8): Intraday Automation + Spring Training Data

**Workstream: Spring Training Roster Population**
- [ ] Ingest spring training data from MLB Stats API for early-season roster/lineup population
- [ ] Use spring training appearances to identify Opening Day rosters before regular season starts
- [ ] Surface spring training stats as context on Player Profile (clearly labeled as ST data)

**Workstream: Intraday Automation**
- [ ] Scheduled daily ETL + model update pipeline (overnight cron/task scheduler)
- [ ] Intraday schedule/lineup/sim refresh every 15 minutes during game windows (configurable in `runtime.yaml`)
- [ ] Stale data alerting: flag when artifacts exceed freshness thresholds
- [ ] Graceful partial refresh: if API fails, use last-known data with staleness label

Exit criteria:
- Spring training data available for roster population before Opening Day
- Intraday updates execute automatically during configured windows
- Stale data is clearly labeled, never silently served

### Milestone 5 (Weeks 9-10): Team Intelligence + Optimizer

**Workstream: Team-Level Analytics**
- [ ] `Team Outlook` page: projected team wins via roster aggregation
  - Aggregate hitter/pitcher projections → team run production/prevention estimates
  - Pythagorean win expectation with uncertainty bands
  - Projected standings (division + league)
  - Roster strength/weakness radar chart
- [ ] Team comparison tool: side-by-side roster strength analysis
- [ ] Injury impact calculator: "what does losing Player X cost this team in projected wins?"

**Workstream: Lineup Optimizer Prototype**
- [ ] Lineup optimizer sandbox: given a roster, find optimal batting order
  - Uses matchup scores + counting projections
  - "Swap Player A for Player B" impact display
  - Trade scenario: "what if Team X acquires Player Y?"
- [ ] Fantasy draft helper: rank players by projected counting stats with upside (90th percentile outcomes)

Exit criteria:
- Team-level output is reproducible and documented
- Optimizer scenarios return stable outputs with guardrails
- Fantasy helper ranks available for standard league formats

---

## Phase 3: Advanced Modeling (Weeks 11-16)

### Milestone 6 (Weeks 11-13): Expanded Projections (Dashboard Surfacing)

> **Blocked on:** Modeling work in `player_profiles` (contact quality, ERA components, SB fix). See `player_profiles/PLAN.md` for that roadmap.

**Workstream: Surface New Projections on Dashboard**
- [ ] Surface contact quality projections (xwOBA, barrel rate, AVG, OBP, SLG, HR) on Player Profile + Projections pages
- [ ] Surface ERA component projections (ERA/FIP/xFIP with uncertainty) on pitcher profiles and projections
- [ ] Surface updated SB counting projections
- [ ] Add new projection columns to leaderboard tables and export formats

Exit criteria:
- New projection types display correctly when parquets are available
- Pages gracefully degrade when new projection parquets are not yet available

### Milestone 7 (Weeks 14-16): Intelligence Layer

> **Blocked on:** Player similarity/comps model and fielding ingestion in `player_profiles`. See `player_profiles/PLAN.md`.

**Workstream: Surface Player Comps (waiting on `player_profiles`)**
- [ ] Surface "Similar Players" section on Player Profile page
- [ ] Display comp trajectories for long-range projection context

**Workstream: Surface Fielding Data (waiting on `player_profiles` / ETL)**
- [ ] Display OAA on Player Profile and Team Overview
- [ ] Factor into team-level WAR-like composite display

**Workstream: Advanced Matchup Features**
- [ ] Velocity trend tracking (acceleration/deceleration as injury signal)
  - Already implemented in `player_profiles` queries, needs dashboard surfacing
- [ ] Contact suppression vs whiff skill separation for pitchers
- [ ] Count-state matchup scoring (ahead/behind in count)

Exit criteria:
- Player comps display when parquet data is available
- OAA visible on player and team pages when data is available
- Velocity trends flagged for injury risk on pitcher profiles

---

## Future / Monetization (Parked — Not Scheduled)

### Betting Edge Finder
> **Status:** Parked pending monetization decision. The infrastructure already exists (game K posteriors, P(over) in Today's Games). The monetizable piece is edge-vs-line comparison and bankroll management.

- [ ] New page: `Betting Edge Finder`
  - Input: user enters sportsbook K lines (or paste from a source)
  - Display: model P(over) vs implied odds, highlight edges > 3%
  - Kelly criterion sizing (fraction of bankroll per bet)
  - Daily edge summary: sorted by expected value
- [ ] ROI tracker: simulated bankroll tracking over time (paper trading mode)
- [ ] Historical edge validation: backtest edge finder against 2022-2025 game K data

**Options:**
1. Build internally first, validate ROI over a season, then decide on monetization
2. Release a "teaser" version (model confidence without specific edge sizing) as marketing
3. Gate behind paywall from day one

---

## Phase 4: Polish & Content (Ongoing)

### UX Improvements
- [ ] Mobile-responsive layouts for all pages
- [ ] Collapsible sections for dense pages (Player Profile)
- [ ] Batter/pitcher silhouettes on zone heatmaps (handedness indicator)
- [ ] Loading states and skeleton screens for slow computations

### Content Generation
- [ ] Daily highlights auto-generation (biggest edges, projection movers, breakout candidates)
- [ ] Shareable player cards (exportable PNG via `save_card()` from tdd_theme)
- [ ] Season recap / weekly digest format

### Data Expansion
- [ ] Historical MLB API data for more seasons (descriptive stats pre-Statcast)
- [ ] Postseason data as optional quality signal (not used in projection, used in evaluation)

---

## Testing Plan

### ETL Tests
- Unit: spec coercion, bounds handling, PK dedup behavior, retry behavior
- Integration: one-day run into test DB schemas, verify row counts and keys
- Contract: schema and nullability checks on raw/staging/production handoffs

### Dashboard Tests
- Loader tests: missing/invalid manifest, schema mismatch, stale data detection
- Page smoke tests: each route renders with fixture artifacts
- E2E: startup → load page → inspect core metrics/cards
- Visual regression: key charts render without errors

### Ops Tests
- Scheduled job dry-run in non-production mode
- Alert simulation on failed ETL step, missing artifact, stale timestamp
- Recovery scenario: resume from failed intraday run

---

## Rollout and Monitoring
1. Environments: local dev → staging → production.
2. Deployment gate: CI green + contract validation + smoke tests.
3. Monitoring metrics:
   - ETL success rate by step
   - Artifact freshness lag
   - Dashboard render errors by page
   - Contract failure counts
4. Alert thresholds:
   - Daily pipeline failure: immediate alert
   - Intraday lag > 20 minutes during game windows: alert
   - Contract mismatch: blocking alert

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Schema drift between repos | Strict manifest contracts + CI contract checks |
| Intraday API instability | Retry policy + partial-refresh fallback + stale-data labeling |
| Large monolith slows delivery | Modularize app.py in Milestone 1 (prerequisite for all feature work) |
| Hidden ETL correctness issues | Formalize DQ scripts into CI-grade tests and run logs |
| Contact quality model is slow to converge | Start with moment-matched empirical Bayes before full MCMC |
| Betting edge finder creates false confidence | Parked in Future/Monetization section; paper trading mode first if built |
| Feature creep delays foundation | Phase 1 is pure infrastructure; no feature work until contracts are in place |

---

## Assumptions and Defaults
1. Database remains PostgreSQL `mlb_fantasy` on `localhost:5433`.
2. Core game scope defaults to regular season for projections; postseason remains optional context input.
3. Primary update cadence defaults:
   - Daily full pipeline overnight.
   - Intraday updates every 15 minutes during configured game windows.
4. Existing parquet artifact naming is preserved initially; versioning is added via manifest.
5. Existing Bayesian model families remain in place; this plan focuses on pipeline/product reliability and integration, then expands modeling in Phase 3.

---

## Immediate Next Sprint Backlog (Decision-Complete) ✅ ALL DONE
1. ~~Begin `app.py` modularization~~ ✅ DONE
2. ~~Add dashboard page smoke tests~~ ✅ DONE (41 tests)
3. ~~Add CI workflow for `tdd-dashboard`~~ ✅ DONE (.github/workflows/ci.yml)
4. ~~Add shared runtime config and remove hardcoded season constants~~ ✅ DONE
5. ~~Add manifest validation path~~ ✅ DONE (services/manifest.py + lenient validation on startup)
6. ~~Add dashboard `Stats` page to sidebar navigation~~ ✅ DONE (merged into Projections)
7. ~~Add initial `Data Health` page~~ ✅ DONE (pages/data_health.py — freshness, inventory, manifest, config)

Success criteria — ALL MET:
1. ✅ `app.py` reduced to routing + config (383 lines), all pages in `pages/` modules.
2. ✅ Today's Games and Game Browser unified into Schedule page.
3. ✅ Dashboard smoke tests pass (41 tests).
4. ✅ Dashboard displays current season from config and surfaces data freshness on Data Health page.
