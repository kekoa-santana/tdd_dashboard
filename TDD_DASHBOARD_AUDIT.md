# The Data Diamond Dashboard — Comprehensive Audit Report

**Date:** March 19, 2026
**Reviewer:** Claude (commissioned by Koa)
**Scope:** Full UX, feature, and strategic review of the live Streamlit dashboard + codebase

---

## Executive Summary

The Data Diamond is an ambitious and largely well-executed MLB analytics dashboard. It combines Bayesian projections, Statcast-derived metrics, pitch-level matchup analysis, prospect rankings, and live game coverage into a single Streamlit application. The depth of the player profile page alone rivals paid platforms. However, several UI polish issues, missing fan-centric features, and architectural gaps prevent it from being the true "one-stop shop" you're aiming for. This report covers what's working, what needs fixing, and what features would elevate TDD from a strong projection tool to an indispensable daily-use platform for MLB fans.

---

## Part 1: What's Working Well

### Strengths MLB Fans Will Love

**Player Profile Page** — This is the crown jewel. The combination of Bayesian posterior distributions, percentile bars with prior-season comparison dashes, pitch arsenal tables with color-coded spark bars, zone heatmaps, scouting reports in plain English, and season trend sparklines is genuinely best-in-class. Most free tools give you a stat table; TDD gives you a full analytical picture with uncertainty quantification. The "Compare projection to Career Avg / 2025" toggle is a great touch.

**Schedule Page with K Props** — Showing O/U probabilities (O5.5, O6.5, O7.5) alongside probable pitchers with archetype tags and K% projections is exactly what daily fantasy and prop bettors want. The live auto-refresh during game windows is well-implemented. Venue, umpire, and weather context on every game card adds real value.

**Archetype System** — Classifying pitchers (Command Specialist, Breaking-Ball Heavy, Fastball Dominant, etc.) and hitters (Patient Power, Contact-Over-Power, Power Slugger, etc.) gives fans an intuitive mental model. The archetype matchup matrix (which archetypes dominate which) is a unique differentiator no competitor offers at this level.

**Team Overview** — The Pitching Staff Profile with strengths/weaknesses vs league average, the staff archetype mix distribution chart, and the offense profile are all things fans can quickly consume to understand a team's identity. This page answers "what kind of team is this?" in seconds.

**Model Transparency** — The Model Performance page showing Bayes vs Marcel backtest comparisons, coverage calibration, and biggest hits/misses is a trust-builder. Most projection sites hide their track record. Showing it, even when the numbers aren't perfect, builds credibility.

**Composite Rankings Methodology** — The TDD Score (Stuff 50%, Command 20%, Workload 15%, Trajectory 15%) is well-constructed and clearly explained. The breakdown into sub-scores gives fans something to debate.

**Dark Theme & Branding** — The dark UI with gold/sage/ember accent colors feels polished and modern. The consistent brand identity (logo, watermark on charts, "THE DATA DIAMOND" header) is professional.

---

## Part 2: Bugs & UI Issues to Fix

### Critical Bugs

1. **HTML Tag Leak on Matchup Explorer** — A raw `<div c` string is visibly rendered below the "MATCHUP WHIFF" metric card. This is an escaped HTML fragment bleeding through `st.markdown(unsafe_allow_html=True)`. This is the most visible bug on the site and will immediately undermine credibility with new visitors.

2. **Data Health Metric Card Text Overflow** — The metric cards on the Data Health page have severe label wrapping: "SEASO N", "HITTERS UPDAT ED", "PITCHE RS UPDAT ED", "K SAMPL ES". The cards are too narrow for the label text. Either abbreviate the labels, reduce font size, or widen the cards.

3. **Stale Data Warning on Schedule Page** — The Schedule page shows "Simulations are from a previous date — showing base projections. Run `python scripts/update_in_season.py` to update." This is an internal developer message exposed to public users who have no ability to run that script. It should either be hidden on the deployed version or replaced with a user-friendly message like "Projections will refresh when the season begins."

4. **"No projection available" for Some Pitchers** — Multiple spring training game cards show "No projection available" for one side of the matchup. While expected for fringe players, it would be better to show a league-average fallback with a disclaimer rather than leaving a blank gap.

### UI Polish Issues

5. **Projections Table Missing Player Names When Scrolled Right** — When users scroll the projections table horizontally to see Proj. K, Proj. BB, Proj. Outs, the player name column scrolls off-screen. The Name column should be frozen/sticky so users always know which player they're looking at.

6. **Filter Dropdowns Show "A." Instead of Full Labels** — On the Projections page, the filter dropdowns display as "A. ▼" which is truncated. Users can't tell what filter each dropdown controls without clicking it. These should show "All Teams", "All Roles", etc. or at least show the full label text.

7. **Archetype Names Are Generic** — On the Team Overview page, the Staff Archetype Mix uses names like "Archetype 4 (SL primary)" and "Archetype 3 (FF primary)" instead of the human-readable names (Command Specialist, Breaking-Ball Heavy, etc.) that appear elsewhere. These should be consistent across all pages.

8. **No Player Headshots** — The Player Profile page has a header section but no player photos. Adding MLB headshot images (available via the MLB Stats API or similar sources) would make profiles feel more complete and recognizable.

9. **Composite Score on Player Profile Shows Negative** — Aaron Ashby's composite is shown as "-0.07" in red, which is confusing. The Rankings page uses a 0-1 scale (TDD Score), but the Player Profile shows a different "Composite" number. These should be unified or the profile should explain what the composite means.

---

## Part 3: Optimization Opportunities

### Performance & UX

10. **Page Load Speed** — Streamlit reruns the entire page on every interaction. With 10 pages and heavy parquet loading, cold starts are noticeable. Consider implementing `st.cache_resource` for heavier objects (NPZ sample arrays), lazy-loading sections behind expanders, and pre-aggregating data where possible.

11. **Mobile Responsiveness** — The dashboard's wide layout with multi-column metric cards and horizontal tables doesn't adapt well to mobile screens. Since many fans will access this on phones (especially during games), consider testing and optimizing the mobile experience or adding a mobile-friendly layout mode.

12. **Search/Navigation** — There's no global search. A fan who wants to quickly find "Juan Soto" has to know which page to go to, then use that page's search box. A global search bar in the sidebar that jumps to the relevant player profile would be a huge UX win.

13. **URL Deep Linking** — Streamlit's session state means you can't share a direct link to a specific player's profile or a specific matchup. Consider using query parameters (`?player=juan-soto`) so fans can share and bookmark specific views.

### Data Presentation

14. **Projections Table Is Dense** — The table shows Rank, Name, Age, Hand, Score, K%, BB%, Proj. K, Proj. BB, Proj. Outs all at once. For casual fans, this is overwhelming. Consider adding a "Simple / Advanced" toggle that shows a cleaner subset by default.

15. **Missing Context for Casual Fans** — Terms like "pp" (percentage points), "logit lift", "95% CI", "posterior distribution" appear without explanation. Add tooltips or a glossary page. The scouting report bullets are a great example of translating stats to plain English — do more of this everywhere.

16. **Color Legend Inconsistency** — Percentile bars use green (80+), gold (60-79), gray (40-59), orange (<40). But metric card deltas use green for positive and red for negative regardless of stat direction. For stats where lower is better (BB% for pitchers), a decrease should be green, not red. Make sure direction-awareness is consistent.

---

## Part 4: Features MLB Fans Would Love

### High-Impact Additions (Within Your Current Scope)

17. **Fantasy Baseball Integration** — Add a "Fantasy Value" column to rankings and projections. Map your Bayesian projections to standard fantasy scoring formats (5x5 roto, H2H points). Fantasy players are the most engaged daily users of projection tools — this alone could double your audience.

18. **Daily Lineup Optimizer Suggestions** — You already have today's games, K props, and matchup scores. Bundle these into a "Today's Best Bets" or "DFS Stacks" summary page that highlights the top pitcher starts, best hitter matchups, and value plays for the day.

19. **Comparison Tool** — Let fans select 2-3 players and see their projections, percentiles, and arsenals side-by-side. "Should I start Skubal or Glasnow this week?" is the kind of question fans ask daily. Your data already supports this; it just needs a UI.

20. **Waiver Wire / Breakout Candidates** — You have `hitter_breakout_candidates.parquet` in your data directory but it doesn't appear on any page. Surface this as a "Breakout Watch" section — fans love discovering under-the-radar players before they pop.

21. **Injury Impact Analysis** — You track health scores and IL status. When a key player goes on the IL, show how it affects team projections and which replacement players benefit. "Acuña is out — here's how ATL's offense profile changes and who fills his role."

22. **Historical Accuracy Tracker (In-Season)** — During the season, add a running "How are our projections doing?" tracker that compares projected K% vs actual K% for the season-to-date, updated weekly. This is your biggest credibility tool.

### Medium-Impact Additions

23. **Team Similarity Clusters** — You mention wanting team similarity in your goals. Use the team archetype distributions and offensive/pitching profiles to compute team-level similarity scores. "The 2026 Orioles play most like the 2024 Dodgers" is the kind of insight that generates social media engagement.

24. **Pitcher Fatigue / Workload Monitor** — Track pitch counts, innings paced, and rest days across the season. Flag pitchers approaching workload thresholds. "Skubal is 15 innings ahead of last year's pace at this point — monitor for fatigue" is valuable for fantasy managers.

25. **Umpire Tendency Page** — You have `umpire_tendencies.parquet` but it's only surfaced in game cards. Give it a dedicated page or section: umpire strike zone overlay, K% impact, historical tendencies. Fans and bettors actively seek this.

26. **Park Factors Page** — Similarly, `park_factors.parquet` exists but isn't prominently surfaced. A park factors leaderboard (most K-friendly, most HR-friendly) would be useful for both bettors and fantasy managers.

27. **Weekly Newsletter / Report Generator** — Auto-generate a weekly summary: biggest projection movers, breakout candidates, model accuracy update, upcoming premium matchups. Could be exported as a shareable image or email.

### Prospect-Specific Enhancements

28. **Prospect ETA Countdown** — You have readiness tiers (Elite, Strong, Developing, etc.) but no estimated call-up dates. Adding "likely call-up window" estimates (e.g., "April 2026", "September 2026", "2027") would make the prospect page much more actionable.

29. **MiLB-to-MLB Translation Explainer** — The translated stats are powerful but most fans don't understand how MiLB translations work. A brief visual explainer ("here's how we convert AA stats to MLB equivalents") would build trust and engagement.

30. **Prospect Comparison to Current MLB Players** — "This prospect's translated line looks like a young version of X" comps are catnip for baseball fans. You have prospect comps in the data — make sure they're prominently displayed.

---

## Part 5: Strategic Recommendations

### Positioning as a One-Stop Shop

To truly be the "one-stop shop for MLB fans," TDD needs to cover three user personas:

| Persona | What They Want | Current Coverage | Gap |
|---------|---------------|-----------------|-----|
| **Fantasy Manager** | Rankings, projections, waiver targets, start/sit advice | Strong projections, weak fantasy framing | Fantasy scoring, DFS integration, start/sit tool |
| **Prop Bettor** | K props, matchup edges, umpire/park factors | Excellent K props, good matchups | Consolidate into a daily betting dashboard |
| **Baseball Nerd** | Deep analytics, model methodology, prospect scouting | Excellent player profiles, good model page | Add more educational content, glossary |

### Competitive Differentiation

Your strongest differentiators vs established tools (FanGraphs, Baseball Savant, etc.):

- **Bayesian uncertainty quantification** — No one else shows posterior distributions to fans. Lean into this. Make "the range of outcomes" your brand identity.
- **Archetype matchup matrix** — Completely unique. Market this heavily.
- **Live game-level K simulation** — The Monte Carlo simulator is more sophisticated than anything freely available.
- **Integrated prospect translations** — Having MiLB translations in the same tool as MLB projections is rare.

### What NOT to Build

- Don't build a play-by-play tracker — MLB.com and ESPN own this space.
- Don't build a social/community feature — focus on being the best data source, not a social network.
- Don't try to cover every stat — your K%/BB%/HR focus with Bayesian updating is your niche. Own it deeply rather than spreading thin.

---

## Part 6: Technical Debt & Architecture Notes

31. **Archetype Naming** — The codebase uses both generic names ("Archetype 4 (SL primary)") and human-readable names ("Breaking-Ball Heavy") inconsistently. Standardize to always display human-readable names on all pages.

32. **Projection vs Stats Page Overlap** — Having both a "Projections" page and a "Stats" page is slightly confusing. Consider merging them with a "Projected / Observed" toggle, or make the distinction clearer in the sidebar labels (e.g., "2026 Projections" vs "Historical Stats").

33. **Game K Simulator Missing from Sidebar** — The CLAUDE.md mentions a "Game K Simulator" page with interactive controls (lineup, umpire, weather), but it's not visible in the sidebar navigation. Either it was removed or it's not yet connected. This would be a high-value interactive feature — prioritize surfacing it.

34. **Preseason Snapshot Page Missing** — Similarly, the CLAUDE.md mentions a "Preseason Snapshot" comparison page but it doesn't appear in the sidebar. The Model Performance page has a "Preseason Comparison" tab, which may have absorbed this. Clarify or consolidate.

35. **Architecture Drift** — Per CLAUDE.md, the `update_in_season.py` script currently handles projection updating that should live in the `player_profiles` repo. This is flagged in your own docs but worth reiterating — as the season progresses and updates become daily, having the model logic split across repos will create maintenance headaches.

---

## Priority Action Items

### Do Now (Pre-Season Launch)
1. Fix the `<div c` HTML leak on Matchup Explorer
2. Fix Data Health metric card text overflow
3. Remove the developer-facing "Run python scripts/update_in_season.py" message from Schedule
4. Fix truncated filter dropdown labels on Projections page
5. Standardize archetype names across all pages

### Do Soon (Early Season)
6. Add a global search bar in the sidebar
7. Surface breakout candidates data that already exists
8. Add fantasy scoring columns to rankings
9. Create a player comparison tool
10. Build a daily betting/DFS summary page

### Do Eventually (Mid-Season)
11. Add team similarity clusters
12. Build umpire and park factor pages
13. Add URL deep linking for shareability
14. Optimize mobile responsiveness
15. Weekly auto-generated report feature

---

*Report generated from live site review + full codebase analysis. All observations verified against both the deployed Streamlit app and source code.*
