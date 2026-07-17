# Mobile Responsive Overhaul Plan

**Branch:** `feat/mobile-responsive-overhaul`

**Status:** In progress - Steps 1 and 2 implemented

**Primary review sizes:** 390 x 844 phone, 768 x 1024 tablet, 1440 x 900 desktop

## 1. Objective

Make the dashboard intentionally usable on phones and tablets without reducing
desktop information density. The overhaul will replace broad, page-agnostic
responsive rules with scoped layout behavior, beginning with the Player Profile
and then moving through the rest of the dashboard in reviewable increments.

The work is complete when:

- Every user-facing page fits the viewport without page-level horizontal scroll
  at 360, 390, 430, 768, and 1024 CSS pixels.
- Player photos, logos, controls, cards, charts, and tables resize or reflow
  according to an explicit component contract.
- Dense content remains readable without turning every compact card into a
  full-width row.
- All primary actions have touch targets at least 44 x 44 CSS pixels.
- Mobile content order follows task priority rather than desktop source order.
- Desktop layouts at 1280 and 1440 pixels do not regress.
- Existing smoke tests pass, and responsive browser checks cover the critical
  pages and breakpoints.

## 2. Scope

### Included

- Global shell, content width, spacing, navigation, and responsive tokens.
- Player Profile hero, filters, metric grids, charts, and tables.
- Home, Schedule, game pages, player pages, team pages, rankings, projections,
  Props Lab, Model Performance, Data Health, and informational pages.
- Custom HTML components and Streamlit `st.columns` layouts.
- Images, Plotly charts, dataframes, custom tables, and long text handling.
- Touch, keyboard, reduced-motion, and basic mobile accessibility behavior.
- A repeatable responsive QA matrix and regression checklist.

### Not included

- Changes to projection models or dashboard datasets.
- A visual rebrand or a desktop redesign unrelated to responsiveness.
- Native-mobile gestures that duplicate existing controls.
- Fixing unrelated data-contract failures, except where they block responsive
  testing. Those failures will be recorded and handled separately.

## 3. Current Baseline

The initial audit found the following concrete issues:

- The Player Profile portrait renders at 364 x 240 on a 390px phone and
  734 x 240 on a 768px tablet. `object-fit: cover` crops most of the face.
- The `.tdd-profile` opening and closing tags are emitted in separate Streamlit
  markdown calls. The resulting wrapper is empty, its descendant CSS does not
  match, and it creates an 88px blank region above the hero.
- The profile identity/rating row requires 401px inside a 364px content area.
- The global phone rule stacks every Streamlit column to 100% width. Compact
  metrics become unnecessarily long single-column lists.
- On Stats, a three-leaderboard batch becomes approximately 2,773px tall on a
  phone.
- Team Overview has page-level horizontal overflow from the seven-column league
  rankings grid and long depth-chart labels.
- Schedule and Player Rankings already avoid page-level horizontal overflow and
  should be treated as reference implementations, not rewritten wholesale.
- Home currently raises `KeyError: 'model_edge'`, preventing a complete visual
  audit of that page.

## 4. Responsive Design Contract

### Breakpoints

Use a small, documented breakpoint set. Component-level exceptions should be
rare and justified next to the rule.

| Range | Name | Default behavior |
| --- | --- | --- |
| 0-479px | Phone | Single-column task flow; two-up compact metrics |
| 480-767px | Large phone | Same content order with slightly wider grids |
| 768-1023px | Tablet | Two-column layouts where each pane remains usable |
| 1024px+ | Desktop | Existing desktop information density |

### Layout rules

- Filters: one column on phones; two columns where space permits on tablets.
- Compact metric cards: two columns on phones, up to four on desktop.
- Long cards and charts: one column on phones.
- Split analytical views: stack on phones; remain split only when both panes are
  at least 320px wide.
- Tables: never expand the page. Use an internal scroller, a compact mobile
  column set, or a card representation.
- Images: never rely on intrinsic or CDN dimensions for layout. Each image class
  owns `width`, `max-width`, `aspect-ratio`, `height`, and `object-fit` behavior.
- Text in grid cells: define wrapping, ellipsis, or a mobile alternate; never
  allow implicit minimum-content width to control the page.

### CSS architecture

- Keep design tokens in `:root`; add explicit responsive spacing and image tokens.
- Replace global `stHorizontalBlock > stColumn` overrides with scoped rules.
- Wrap related Streamlit layouts in keyed containers so generated `st-key-*`
  classes provide stable page/component hooks.
- Do not use unmatched opening/closing HTML tags across Streamlit calls.
- Prefer component classes over inline styles. Inline values should be limited to
  genuinely dynamic data such as progress percentages or coordinates.
- Place component responsive rules next to the component's base styles.
- Keep one final, small utilities section; do not accumulate competing global
  media queries in multiple locations.

## 5. Implementation Sequence

Each step is a review boundary. Do not begin the next step until the current
step's acceptance criteria have been checked at phone, tablet, and desktop sizes.

### Step 0 - Establish guardrails and baseline

Tasks:

1. Record screenshots and DOM measurements for critical pages at 390, 768, and
   1440 pixels.
2. Add a responsive audit checklist under `docs/` and a lightweight browser audit
   helper if it can be done without introducing a heavy runtime dependency.
3. Inventory every `st.columns`, fixed CSS grid, fixed width, custom table, image,
   Plotly chart, and dataframe.
4. Classify each occurrence as filter, compact metrics, analytical split, chart,
   table, navigation, or decorative layout.
5. Record the Home `model_edge` failure as a separate blocker.

Acceptance criteria:

- Baseline measurements are reproducible.
- The page inventory names the owner file and expected phone behavior.
- Existing user data changes are not staged or committed with responsive work.

### Step 1 - Repair the Player Profile structure

Tasks:

1. Remove the synthetic `.tdd-profile` opening and closing markdown tags.
2. Introduce a real page marker or keyed Streamlit containers for profile-only
   width, spacing, filter, metric, and chart rules.
3. Change descendant selectors that currently depend on the nonexistent wrapper.
4. Remove the 88px phantom gap.
5. Fix the subtitle construction so the team abbreviation renders as markup rather
   than escaped HTML while all user/data text remains escaped safely.

Acceptance criteria:

- No empty `.tdd-profile` element exists.
- Hero and section selectors match the intended rendered elements.
- No literal HTML appears in the player subtitle.
- Desktop spacing remains visually equivalent or improves intentionally.

### Step 2 - Rebuild the Player Profile hero and headshot behavior

Tasks:

1. Replace the portrait's inline width/height styles with semantic classes.
2. Define explicit portrait sizes:
   - Desktop: approximately 280 x 360.
   - Tablet: approximately 160-180px wide with a controlled aspect ratio.
   - Phone: 112-144px square or near-square thumbnail.
3. On phones, place the portrait beside the identity when space permits, then put
   rating, vitals, and scouting content on full-width rows.
4. Stack the identity and rating blocks when their combined minimum width does not
   fit. Left-align the rating on phones.
5. Give the image a stable focal point and verify real and fallback headshots.
6. Request an appropriately sized MLB CDN image for each display tier rather than
   downloading a 400px asset for all viewports.
7. Add `loading="lazy"` where it does not delay the primary hero image.

Acceptance criteria:

- The player's face is visible at 360, 390, 430, and 768px.
- The portrait does not dominate the first phone viewport.
- Hero content has no horizontal overflow.
- The player name, team, role, rating, and first key metrics are visible in a
  sensible order without excessive blank space.

### Step 3 - Replace global column stacking with responsive primitives

Tasks:

1. Remove or narrow the global phone rule that makes all Streamlit columns 100%.
2. Create scoped patterns for:
   - `.responsive-filters`
   - `.responsive-metrics`
   - `.responsive-split`
   - `.responsive-charts`
   - `.responsive-actions`
3. Use keyed Streamlit containers to apply those patterns reliably.
4. Convert repeated `st.columns(len(chunk))` metric loops to a predictable
   responsive grid where practical.
5. Confirm that visual order and keyboard order remain the same.

Acceptance criteria:

- Compact metrics render two-up on phones.
- Filters and analytical splits stack cleanly.
- No component depends on the number of unrelated columns elsewhere on the page.
- Rules do not target transient Emotion class names.

### Step 4 - Finish Player Profile content sections

Tasks:

1. Apply the new primitives to filter controls and season selectors.
2. Make scouting grades, percentile bars, approach metrics, and discipline cards
   phone-readable without excessive vertical expansion.
3. Stack pitch-mix/chart-and-table splits below their usable minimum width.
4. Set phone-specific Plotly heights, margins, legend behavior, and label sizes.
5. Give dataframes internal horizontal scrolling and a useful initial column set.
6. Consider a compact card representation for the final Stat Breakdown if the
   dataframe still requires frequent horizontal scrolling.

Acceptance criteria:

- The full profile has no page-level horizontal overflow.
- No chart labels or legends are clipped.
- Important values remain visible without horizontal scrolling.
- Scrolling length is materially reduced from the baseline.

### Step 5 - Global shell and navigation

Tasks:

1. Verify topbar logo, brand text, hamburger, focus states, and open-menu width.
2. Give every mobile menu item a minimum 44px touch height.
3. Confirm the open menu scrolls independently and does not trap page scrolling
   after it closes.
4. Normalize `.block-container` phone/tablet padding and top offset.
5. Ensure focus outlines, reduced motion, and long translated labels remain usable.

Acceptance criteria:

- Navigation works by touch and keyboard at every breakpoint.
- Menu content is neither clipped nor wider than the viewport.
- Shell changes do not alter desktop navigation behavior.

### Step 6 - Dense list and leaderboard pages

Pages:

- Stats
- Projections
- Projected Performers
- Player Rankings
- Breakout Candidates
- Team Rankings

Tasks:

1. Replace simultaneous multi-leaderboard phone layouts with one of:
   - a metric selector and one visible leaderboard,
   - horizontally scrollable snap cards, or
   - progressively disclosed sections.
2. Preserve the existing multi-column desktop layout.
3. Collapse secondary leaderboard columns and maintain player-name priority.
4. Make filters sticky only if testing shows that it improves navigation without
   consuming too much phone viewport space.
5. Add ellipsis plus accessible full-name text where row space is constrained.

Acceptance criteria:

- No phone view renders multiple full-height leaderboards sequentially by default.
- A user can change the displayed metric without returning to the top of the page.
- Row names, ranks, values, and navigation targets remain readable and tappable.

### Step 7 - Team pages and fixed-grid components

Tasks:

1. Fix Team Overview league-rank overflow by using a compact phone column set or
   an explicit internal scroller.
2. Add truncation/wrapping rules for depth-chart names and archetypes.
3. Reflow team header statistics and roster summaries by priority.
4. Audit Team Rankings cards and division tables for the same minimum-content
   width problems.
5. Verify team logos never force a row wider or taller than intended.

Acceptance criteria:

- Team pages have no page-level horizontal scroll at 360px.
- The selected team and primary score remain visible without scrolling a table.
- Long player and archetype names do not overlap adjacent cells.

### Step 8 - Games, schedule, Props Lab, and lineup tools

Pages:

- Schedule
- Game Analysis
- Daily Preview
- Props Lab
- Lineup Creator

Tasks:

1. Preserve Schedule's existing successful mobile behavior and add regression
   coverage before touching shared styles.
2. Stack game matchup panes and prioritize score, starters, status, and top edges.
3. Reduce dense stat grids to essential phone columns with details available on
   demand.
4. Ensure lineup selectors and action buttons have adequate touch size.
5. Test expanded panels, chart controls, and long player names.

Acceptance criteria:

- Primary game information is visible in the first phone viewport.
- Expanded content stays within its card.
- Controls remain usable without zooming or horizontal panning.

### Step 9 - Home, editorial, performance, and utility pages

Pages:

- Home
- The Diamond Daily
- News
- Model Performance
- Data Health
- Compare
- Methodology

Tasks:

1. Resolve or isolate the Home `model_edge` data-contract failure before visual
   responsive work begins on Home.
2. Reflow editorial mastheads, pull quotes, and multi-column story layouts.
3. Turn large metric rows into two-up grids.
4. Stack performance chart pairs and reduce mobile chart decoration.
5. Make diagnostic tables internally scrollable and keep key identifying columns
   visible.
6. Verify long explanatory text uses comfortable line length and spacing.

Acceptance criteria:

- Every page renders successfully with the current dashboard data contract.
- Editorial hierarchy remains clear on phones.
- Diagnostics remain usable without shrinking text below readable sizes.

### Step 10 - Cross-page polish and final regression pass

Tasks:

1. Consolidate duplicate or conflicting media queries.
2. Remove obsolete inline responsive styles and dead selectors.
3. Audit font sizes, line heights, touch targets, focus states, and contrast.
4. Audit image request sizes and unnecessary above-the-fold rendering.
5. Run the complete page/breakpoint matrix.
6. Run existing tests and add targeted regression tests for any Python rendering
   logic changed during the overhaul.
7. Update `docs/frontend_style_system.md` with the responsive contract and examples.

Acceptance criteria:

- All Definition of Done checks pass.
- No responsive rule is knowingly dependent on a transient generated class.
- Desktop screenshots show no unintended layout changes.
- Documentation explains how future pages choose the correct responsive primitive.

## 6. Page and Breakpoint Test Matrix

### Required viewport widths

- 360px: small Android/iPhone layout stress test.
- 390px: primary phone review size.
- 430px: large phone.
- 768px: portrait tablet.
- 1024px: tablet/desktop boundary.
- 1440px: desktop regression.

### Critical pages tested at every width

- Home
- Schedule
- Player Profile: hitter, pitcher, two-way player, missing headshot
- Player Rankings
- Stats: hitter and pitcher
- Projections
- Team Overview
- Team Rankings
- Game Analysis
- Props Lab
- Model Performance

### Checks for every tested state

- `documentElement.scrollWidth <= documentElement.clientWidth` and equivalent
  check for Streamlit's main scrolling section.
- Images remain inside their component and display the intended focal area.
- No control, label, badge, or rating is clipped.
- Tap targets meet the 44px minimum.
- Keyboard focus remains visible and follows visual order.
- Open menus, expanders, dropdowns, and details panels stay within the viewport.
- Charts resize after navigation and control changes.
- Dataframes scroll internally rather than widening the page.
- Loading, empty, error, and sparse-data states are also checked.

## 7. Commit and Review Strategy

Keep commits narrow and stage only files owned by the responsive task. The working
tree already contains unrelated data and source changes; those must not be included
in responsive commits.

Suggested commit sequence:

1. `docs: add responsive contract and audit baseline`
2. `fix: repair player profile container structure`
3. `feat: add responsive player profile hero`
4. `refactor: scope responsive column primitives`
5. `feat: optimize player profile mobile content`
6. `fix: remove team page mobile overflow`
7. `feat: add mobile leaderboard presentation`
8. `feat: optimize game and tool pages for mobile`
9. `feat: optimize editorial and utility pages for mobile`
10. `test: add responsive regression coverage`
11. `docs: document responsive component usage`

After each implementation commit:

- Run the directly affected smoke tests.
- Review 390, 768, and 1440px before moving on.
- Record any intentionally deferred issue in this plan.

## 8. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Global CSS fixes one page and breaks another | Scope rules through component classes and keyed containers |
| Streamlit DOM changes across versions | Avoid Emotion classes; use stable `data-testid`, keys, and owned HTML classes |
| CSS breakpoint cascade becomes contradictory | Keep component media rules beside base rules and consolidate old global queries |
| Mobile layout hides useful analytical context | Prioritize key values and expose secondary detail progressively |
| Long tables remain unusable | Use compact column sets, sticky identifiers, or card representations |
| Inline HTML escaping introduces regressions | Separate trusted markup construction from escaped data values and test both |
| Existing dirty work enters responsive commits | Stage explicit paths and inspect every staged diff before committing |
| Current data errors block page audits | Log separately and fix/isolate only the minimum required for responsive testing |

## 9. Definition of Done

- [ ] All critical pages pass the full breakpoint matrix.
- [ ] No page-level horizontal overflow at 360px or wider.
- [ ] Player headshots have explicit responsive sizes and correct focal cropping.
- [ ] Compact metrics are two-up on phones where appropriate.
- [ ] Filters, charts, and analytical splits stack intentionally.
- [ ] Tables and dataframes scroll internally or use compact mobile presentations.
- [ ] Navigation and primary actions meet touch-target requirements.
- [ ] Empty, loading, sparse, and error states remain usable.
- [ ] Existing tests pass.
- [ ] Responsive documentation is updated.
- [ ] Desktop behavior has been visually regression-tested.
- [ ] Responsive commits contain no unrelated data or model changes.

## 10. First Review Checkpoint

Begin with **Steps 1 and 2 only**: repair the Player Profile structure and rebuild
the responsive hero/headshot. Stop after verifying hitter, pitcher, two-way, and
fallback-image states at 390, 768, and 1440px. Review those results before changing
the global Streamlit column behavior in Step 3.

**Checkpoint result (2026-07-16):** Steps 1 and 2 are implemented. Hitter, pitcher,
and two-way profiles were verified at 360, 390, 768, and 1440px with no page-level
horizontal overflow. Fallback headshot and balanced markup behavior are covered by
targeted tests. Step 3 remains intentionally deferred pending review.
