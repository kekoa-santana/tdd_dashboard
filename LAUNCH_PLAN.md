# TDD Dashboard — Launch Sprint Plan (3/19 → 3/23)

**Target:** Live and polished by March 23, 2026 (3 days before Opening Day)
**Executor:** Claude Code
**Decision log:** All items below reflect Koa's sign-off from the 3/19 review session.

---

## Locked-In Decisions

| Decision | Choice |
|----------|--------|
| Composite score | **Diamond Rating (0–5 diamonds)** — visual diamond icons, fractional (e.g., 4.8◆). Unique to TDD brand. |
| Player headshots | **Hotlink from MLB CDN** (`img.mlb.com/headshots/current/60x60/{player_id}.png`). No local hosting. Safe for free site; consult IP attorney before paywall launch. |
| Projections page | **Overhaul into Leaderboard + Discovery** — top risers, breakout candidates, biggest shifts. Less spreadsheet, more editorial. |
| Page reload fix | **URL deep linking via `st.query_params`** — each page writes params, app.py reads on load. Fixes reload-to-landing issue. |
| Game Explorer | **Expand existing schedule tabs** — add hitter props alongside pitcher K props. Read-only, no user inputs. Each game = mini analytics hub. |
| Archetype names | **Human-readable everywhere** — replace "Archetype 4 (SL primary)" with "Breaking-Ball Heavy" etc. on all pages. |
| Game K Simulator | **Keep inside Game Explorer tabs** (not standalone page). Read-only per-game view. |
| Preseason snapshot | **Freeze right before season (3/25)** — not now, not after games start. |
| Architecture | **Migrate projection logic to player_profiles** before season. Dashboard update script becomes thin wrapper. |
| Newsletter | **Exportable summary** — auto-generated, manually shared by Koa. |
| Monetization | **Freemium** — free base site, paid tier for better bets + deeper analysis (future). |

---

## Day-by-Day Sprint

### Day 1 — Thursday 3/19: Bug Fixes & Foundation

**Goal:** Eliminate all visible bugs. Ship a clean version of what exists today.

| # | Task | Est. Time | Priority |
|---|------|-----------|----------|
| 1.1 | **Fix `<div c` HTML leak** on Matchup Explorer — find the broken `st.markdown` call in `matchup_explorer.py`, fix the unclosed/malformed HTML in the metric card rendering | 15 min | P0 |
| 1.2 | **Fix Data Health metric card text overflow** — abbreviate labels ("Hitters Updated" → "Hitters", "K Samples" → "K Samp.") or switch to smaller font / wider cards | 20 min | P0 |
| 1.3 | **Remove developer-facing update message** from Schedule page — replace "Run `python scripts/update_in_season.py`" with "Projections update daily during the season" or hide entirely | 10 min | P0 |
| 1.4 | **Fix truncated filter dropdowns** on Projections — ensure "All Teams", "All Roles", "All" labels display fully instead of "A." | 20 min | P0 |
| 1.5 | **Standardize archetype names** across all pages — map generic names ("Archetype 4 (SL primary)") to human-readable names ("Breaking-Ball Heavy") using the archetype metadata parquets. Ensure Team Overview, Schedule, Player Profile all use the same names. | 45 min | P0 |
| 1.6 | **Fix "No projection available" fallback** — show league-average baseline with "(lg avg)" disclaimer instead of blank gap for unprojected pitchers on Schedule | 30 min | P1 |
| 1.7 | **Implement URL deep linking** — add `st.query_params` to app.py routing + each page. Support `?page=player_profile&player_id=XXX`, `?page=team_overview&team=NYY`, etc. Fix reload-to-landing bug. | 1.5 hr | P0 |

**Day 1 total: ~3.5 hours**

---

### Day 2 — Friday 3/20: Diamond Rating, Headshots, Player Profile Polish

**Goal:** Transform the visual identity. Diamond Rating everywhere, headshots on profiles, score consistency.

| # | Task | Est. Time | Priority |
|---|------|-----------|----------|
| 2.1 | **Implement Diamond Rating system** — create `lib/diamond_rating.py` with conversion function (TDD score 0–1 → 0.0–5.0 diamond scale). Build `components/diamond_rating.py` with HTML/CSS rendering: filled ◆, half ◆, empty ◇ icons in gold. Add tooltip showing numeric value on hover. | 1.5 hr | P0 |
| 2.2 | **Replace composite score with Diamond Rating** on: Player Profile header, Player Rankings table, Projections/Leaderboard page, Team Overview roster tables, Schedule game cards, Matchup Explorer | 1 hr | P0 |
| 2.3 | **Add player headshots** — create `components/headshot.py` that renders `<img>` from MLB CDN URL with fallback silhouette SVG for missing images. Add to: Player Profile header, Matchup Explorer pitcher/hitter cards. | 45 min | P0 |
| 2.4 | **Sticky name column** on all data tables — use custom CSS to freeze the first column (player name) when scrolling horizontally on Projections, Stats, Rankings tables | 30 min | P1 |
| 2.5 | **Add tooltips/glossary** for non-obvious terms — "pp" (percentage points), "95% CI" (credible interval), "K Lift" (matchup advantage). Use `st.help` or `(?)` hover icons inline. | 45 min | P1 |
| 2.6 | **Fix color direction consistency** — ensure deltas for stats where lower-is-better (BB% for pitchers, K% for hitters) show green for improvement regardless of +/- direction | 30 min | P1 |

**Day 2 total: ~5 hours**

---

### Day 3 — Saturday 3/21: Projections Overhaul & Comparison Tool

**Goal:** Transform Projections into a discovery page. Build the player comparison tool.

| # | Task | Est. Time | Priority |
|---|------|-----------|----------|
| 3.1 | **Overhaul Projections page → "Leaderboard & Discovery"** — redesign with sections: (a) Top 10 K% Risers / Fallers, (b) Breakout Watch (surface existing `hitter_breakout_candidates.parquet` + in-progress pitcher breakouts), (c) Diamond Rating Leaders by position, (d) Filterable stat leaderboards (K%, BB%, Proj K, etc.), (e) Biggest Projection Shifts (delta from preseason). Keep a compact full-table mode behind an expander for power users who want the raw data. | 3 hr | P0 |
| 3.2 | **Build Player Comparison tool** — new page or modal accessible from Player Profile + Rankings. Select 2–3 players, show side-by-side: Diamond Rating, K%/BB% projections with CIs, percentile bars, radar chart of key stats, archetype comparison, arsenal overlap (for pitchers). Use `st.columns` for layout. | 2.5 hr | P0 |
| 3.3 | **Add Comparison to sidebar navigation** — add "Compare Players" to nav | 10 min | P0 |
| 3.4 | **Surface breakout candidates** — if pitcher breakout model data is ready, integrate into the new Leaderboard page. If not, wire up hitter breakout data that already exists. | 45 min | P1 |

**Day 3 total: ~6.5 hours**

---

### Day 4 — Sunday 3/22: Game Explorer Expansion & Launch Polish

**Goal:** Expand game-day experience. Final QA pass. Ship it.

| # | Task | Est. Time | Priority |
|---|------|-----------|----------|
| 4.1 | **Expand Game Explorer hitter props** — within the existing schedule game tabs, add hitter projection cards: projected H, HR, RBI, TB, K for each lineup slot. Show Diamond Rating + archetype for each hitter. Read-only, pulls from counting projections + matchup scores. | 2 hr | P0 |
| 4.2 | **Injury impact stubs** — on Team Overview, when a player is IL-flagged, show a brief "Impact" note: who replaces them in the lineup/rotation and the projected stat change. Can be simple delta calculation from roster data. | 1 hr | P1 |
| 4.3 | **In-season accuracy tracker stub** — add a "Season Tracker" tab to Model Performance. Pre-season it shows "Tracking begins Opening Day" with explanation of what will be tracked (running MAE, projected vs actual scatter, calibration curve). Wire the data pipeline to append weekly accuracy snapshots once games start. | 45 min | P1 |
| 4.4 | **Weekly report template** — create `scripts/generate_weekly_report.py` that pulls: top 10 risers/fallers, breakout candidates, model accuracy snapshot, upcoming premium matchups. Outputs a styled markdown or HTML file suitable for sharing. | 1 hr | P1 |
| 4.5 | **Full QA pass** — click through every page on the live deployed app. Check: no HTML leaks, all archetype names human-readable, Diamond Rating renders correctly, headshots load, deep links work, mobile layout is at least functional, all data loads without errors. | 1 hr | P0 |
| 4.6 | **Architecture migration plan** — document the exact steps to move projection updating from `tdd-dashboard/scripts/update_in_season.py` to `player_profiles/`. Don't execute the migration yet (do it 3/24–3/25 after QA is clean), but have the plan written so it's a clean cut. | 30 min | P1 |

**Day 4 total: ~6.25 hours**

---

### Day 5 — Monday 3/23: Launch Day

| # | Task | Est. Time | Priority |
|---|------|-----------|----------|
| 5.1 | **Execute architecture migration** — move projection logic to player_profiles, make dashboard update script a thin wrapper | 2 hr | P0 |
| 5.2 | **Deploy final version** to Streamlit Cloud | 15 min | P0 |
| 5.3 | **Freeze preseason snapshot** (or schedule for 3/25 just before Opening Day) | 15 min | P0 |
| 5.4 | **Smoke test deployed app** — verify all pages load, data is fresh, deep links work | 30 min | P0 |

**Day 5 total: ~3 hours**

---

## Post-Launch Roadmap (Season Weeks 1–4)

These are features discussed in the audit that are valuable but not launch-critical:

| Week | Feature | Notes |
|------|---------|-------|
| 1 | Fantasy scoring columns on rankings | Koa building improved fantasy model on Bayesian base |
| 1 | Prospect ETA countdown + comps display | Data exists, needs UI |
| 2 | Daily betting/DFS summary page | Consolidate K props + matchup edges + park/umpire factors |
| 2 | Umpire tendency page | Data exists in `umpire_tendencies.parquet` |
| 2 | Park factors page | Data exists in `park_factors.parquet` |
| 3 | Team power rankings + division projections | Extension of team similarity concept |
| 3 | MiLB translation explainer visual | Educational content for prospect page |
| 3 | Mobile responsiveness pass | Test and optimize key pages for phone screens |
| 4 | Weekly auto-report polish | Iterate on format based on first few manual shares |
| 4 | Global search bar | Sidebar search that jumps to any player profile |

---

## Diamond Rating Specification

```
Score → Diamonds mapping:
  TDD Score 0.00–0.20  →  ◆◇◇◇◇  (1.0)
  TDD Score 0.20–0.40  →  ◆◆◇◇◇  (2.0)
  TDD Score 0.40–0.55  →  ◆◆◆◇◇  (3.0)
  TDD Score 0.55–0.70  →  ◆◆◆◆◇  (4.0)
  TDD Score 0.70–1.00  →  ◆◆◆◆◆  (5.0)

Fractional diamonds (e.g., 4.3◆) rendered as partially filled.
Color: TDD Gold (#C8A96E) for filled, Slate (#7B8FA6) for empty.
Hover tooltip: "Diamond Rating: 4.3 / 5.0 — Elite"

Tier labels:
  4.5–5.0  →  "Elite"
  3.5–4.4  →  "Above Average"
  2.5–3.4  →  "Average"
  1.5–2.4  →  "Below Average"
  0.0–1.4  →  "Developing"
```

---

## File Changes Summary

**New files to create:**
- `lib/diamond_rating.py` — score conversion + tier labels
- `components/diamond_rating.py` — HTML/CSS rendering
- `components/headshot.py` — MLB CDN image with fallback
- `views/compare.py` — player comparison page
- `scripts/generate_weekly_report.py` — exportable summary generator

**Files to modify heavily:**
- `app.py` — URL deep linking, nav routing, add Compare page
- `views/projections.py` — full overhaul into Leaderboard & Discovery
- `views/schedule.py` — hitter props in game tabs, remove dev message
- `views/player_profile.py` — Diamond Rating, headshot, score fix
- `views/team_overview.py` — archetype name fix, injury impact
- `views/matchup_explorer.py` — HTML leak fix, headshots
- `views/player_rankings.py` — Diamond Rating column
- `views/model_performance.py` — Season Tracker tab stub
- `views/data_health.py` — metric card overflow fix
- `components/metric_cards.py` — Diamond Rating integration, delta direction fix

**Files to modify lightly:**
- `components/tables.py` — sticky name column CSS
- `components/scouting.py` — tooltip additions
- `config.py` — diamond rating tier thresholds
- `services/data_loader.py` — breakout candidates loader

---

*Plan locked 3/19/2026. Execute in Claude Code starting Day 1.*
