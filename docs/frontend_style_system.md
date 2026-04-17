# Frontend Style System

This document describes the CSS architecture for the TDD Dashboard:
what the tokens mean, where styles live, and how to name new classes.

## Source of truth

| Layer | File | Owns |
|-------|------|------|
| Static CSS + tokens | `assets/styles.css` | All structural rules, default palette, component CSS |
| Dynamic overrides | `app.py` (top of file) | Google Fonts `<link>`, `:root` overrides for the user-selected palette and font pairing |
| Per-page CSS | `views/*.py` (`_CSS = f"""..."""`) | Legacy. Projections, Stats migrated. Diamond Daily minimal. Remaining views migrated as touched. |

**Rule:** if a rule does not depend on a Python value, it belongs in `styles.css`.
`app.py` only injects `:root` variable *values*, never structural selectors.

## Token catalog (`:root` in `styles.css`)

### Color (palette-driven)
Overridden by `app.py` when a non-default palette is selected.

| Token | Default | Purpose |
|-------|---------|---------|
| `--tdd-gold` | `#C8A96E` | Brand primary, section headers, "good" direction |
| `--tdd-ember` | `#D4562A` | Accent, "bad" direction, negative delta |
| `--tdd-sage` | `#6BA38E` | Positive delta, secondary accent |
| `--tdd-slate` | `#7B8FA6` | Secondary / meta text |
| `--tdd-cream` | `#F5F2EE` | Primary text on dark backgrounds |
| `--tdd-dark` | `#0F1117` | App background |
| `--tdd-dark-card` | `#181b23` | Raised card surface |
| `--tdd-dark-border` | `#2a2e3a` | Dividers, card borders |
| `--tdd-dark-border-faint` | `#2a2e3a22` | Subtle row dividers (`AA` alpha) |
| `--tdd-positive` | `--tdd-sage` | Semantic "good" |
| `--tdd-negative` | `--tdd-ember` | Semantic "bad" |

### Typography

| Token | Default | Purpose |
|-------|---------|---------|
| `--tdd-font-heading` | `"Inter", ...` | Brand, section headers, player/team names, `h1`..`h3` |
| `--tdd-font-body` | `"IBM Plex Mono", ...` | Everything else |

Font sizes use `--tdd-fs-*` tokens (team-abbr, player-name, stat-value, stat-label, meta, badge, section-header, context).

### Spacing (4px base)
Use instead of inline `rem` or `px` values.

| Token | Value |
|-------|-------|
| `--tdd-space-1` | `0.25rem` (4px) |
| `--tdd-space-2` | `0.5rem`  (8px) |
| `--tdd-space-3` | `0.75rem` (12px) |
| `--tdd-space-4` | `1rem`    (16px) |
| `--tdd-space-5` | `1.5rem`  (24px) |
| `--tdd-space-6` | `2rem`    (32px) |
| `--tdd-space-7` | `3rem`    (48px) |
| `--tdd-space-8` | `4rem`    (64px) |

### Radius, shadow, z-index

- `--tdd-radius-sm | md | lg | pill`
- `--tdd-shadow-sm | md | lg`
- `--tdd-z-base | sticky | dropdown | nav | nav-menu | mobile-nav | modal`

Always use a z-index token, never a magic number. If none fits, add one here first.

### Sizing (logos / headshots)
`--tdd-logo-sm | md | lg`, `--tdd-headshot-sm | md | lg`. Applied via utility classes `.tdd-logo-*` and `.tdd-headshot-*`.

### Breakpoints
CSS `@media` rules cannot reference custom properties. These values are referenced directly across the stylesheet; keep them in sync.

| Name | Value | Use |
|------|-------|-----|
| Phone | `480px` | Single-column, hide secondary content |
| Tablet | `768px` | 2-up columns, compact metrics |
| Desktop | `1024px` | Hide mobile-only affordances, full table density |

## Class naming conventions

Two-tier namespace: **utility** and **component**.

### Utility classes: `.tdd-*`
Small, reusable, token-wired. One job each. Think font size plus color, or a single layout hint.

Examples: `.tdd-team-abbr`, `.tdd-stat-value`, `.tdd-meta`, `.tdd-badge`, `.tdd-logo-md`.

### Page-level patterns: `.tdd-page-*`, `.tdd-callout*`, `.tdd-empty-state`
Shared across all views. Use instead of per-page header/callout classes.

| Class | Purpose |
|-------|---------|
| `.tdd-page-header` | Centered page title wrapper |
| `.tdd-page-title` | Large page title text |
| `.tdd-page-subtitle` | Slate subtitle under title |
| `.tdd-callout` | Left-accent banner (gold, default) |
| `.tdd-callout-info` | Slate info callout |
| `.tdd-callout-warn` | Ember warning callout |
| `.tdd-empty-state` | Centered empty/no-data message |
| `.tdd-status-live` | Green live game badge |
| `.tdd-status-final` | Slate final game badge |
| `.tdd-game-count` | Game count below schedule header |
| `.tdd-sim-summary` | Simulation summary footer text |
| `.tdd-props-header` | Section header for props sections |
| `.tdd-footer-disclaimer` | Site-wide footer disclaimer |

### Component classes: `.c-*` (new, Phase 2+)
Multi-element widgets with their own rule blocks. BEM-ish: `.c-<component>__<element>--<modifier>`.

Planned migration targets:

| Legacy prefix | New namespace | Scope |
|---------------|---------------|-------|
| `.lb-*` | `.c-leaderboard__*` | Leaderboard cards (projections, rankings, stats) |
| `.topbar-*` | `.c-nav__*` | Top navigation bar and dropdowns |
| `.metric-*` | `.c-metric__*` | Borderless metric cards |
| `.insight-*` | `.c-insight__*` | Left-accent insight cards |
| `.pctile-*` | `.c-pctile__*` | Percentile bars |
| `.pitch-table`, `.matchup-table` | `.c-table--pitch`, `.c-table--matchup` | Tabular views |

**Migration is not automatic.** Legacy classes stay in `styles.css` until the matching Phase 2 shared renderer lands. When a renderer is introduced, the page-level `_CSS` block is removed and the markup emits `.c-*` classes.

### State / modifier classes
Prefer attribute selectors (`[aria-current="page"]`, `[data-team="NYY"]`) over state-in-classname (`.is-active`). The `data-team` attribute pattern already drives team-color theming.

### What not to name
- Do not add per-page CSS under a page-specific prefix (e.g. `.proj-*`, `.sched-*`). Those are the `_CSS` blocks being removed.
- Do not add `!important` except where Streamlit's inline styles force the issue. Existing uses against `[data-baseweb]` and `stSelectbox` are the precedent; new ones should be justified in a comment.

## Dynamic vs. static split

`app.py` currently runs, in order:

1. `st.markdown(<styles.css contents>)`: loads everything static.
2. `<link>` for Google Fonts if the active pairing needs them.
3. A single `:root { ... }` override block **only if** the active palette or font pairing differs from the styles.css defaults.

Anything else going into `app.py` as raw CSS is a smell.

## HTML safety contract

All views use `st.markdown(..., unsafe_allow_html=True)` to render custom HTML.
Any string interpolated into that HTML must be categorized as **safe** or **data-derived**.

### Safe (no escaping needed)
- CSS class names and inline style values (code-controlled)
- Integer IDs in URLs (`player_id=12345`)
- Output of component renderers (`headshot_html()`, `team_logo_html()`, `render_card()`, etc.) that already handle their own escaping
- Color hex values from `config.py`

### Data-derived (MUST escape)
- Player names, team abbreviations, venue names, umpire names
- Stat labels or values that originate from DataFrame columns
- Any string read from parquet/CSV/API that appears in text content or attributes

### How to escape
Import `from utils.html import esc, esc_attr`.
- `esc(value)`: for text between HTML tags. Escapes `&`, `<`, `>`.
- `esc_attr(value)`: for values inside HTML attributes (quoted). Also escapes `"` and `'`.

Both convert to `str` first and return `""` for `None`.

### Where escaping is applied
The shared `components/leaderboard.py` renderer escapes all data fields internally.
The five highest-volume views (`schedule`, `projected_performers`, `player_rankings`,
`team_overview`, `compare`) have been audited and patched. Remaining views should be
updated as they are touched.

## Theme alignment

`.streamlit/config.toml` sets native Streamlit widget colors. These must match the
default palette in `styles.css :root`:

| config.toml key | CSS token | Default |
|-----------------|-----------|---------|
| `primaryColor` | `--tdd-gold` | `#C8A96E` |
| `backgroundColor` | `--tdd-dark` | `#0F1117` |
| `secondaryBackgroundColor` | `--tdd-dark-card` | `#181b23` |
| `textColor` | `--tdd-cream` | `#F5F2EE` |
| `font` | `--tdd-font-body` | `monospace` |

**Known limitation:** when a user switches palette via the Settings expander,
CSS variables update but `config.toml` stays fixed. Native Streamlit widgets
(progress bars, sliders, toggle switches) keep the default palette colors.
Custom HTML (leaderboards, cards, nav) recolors correctly because it uses
`var(--tdd-*)`. This is acceptable; palette switching is cosmetic and the
affected native widgets are minimal.

## Visual regression checklist

Run through this checklist after CSS or layout changes. Test at three widths:
desktop (1280px+), tablet (768px), phone (480px).

### Pages to check
- [ ] Schedule: game cards, prop cards, lineup rows
- [ ] Projections: leaderboard cards (3-up columns), watch-list rows
- [ ] Player Profile: metric cards, percentile bars, plotly charts, zone contours
- [ ] Player Rankings: ranking cards, diamond ratings, expandable detail cards
- [ ] Team Overview: depth chart rows, roster table, trade sim results
- [ ] Compare: side-by-side player headers, stat comparison rows

### Things to verify
- [ ] Topbar: all dropdown menus open on hover AND keyboard Tab
- [ ] Topbar: focus ring visible when tabbing through nav items
- [ ] Topbar: mobile hamburger opens/closes, all links work
- [ ] Leaderboard cards: rank gold on top 5, headshot visible, team colored
- [ ] Text: no clipping, no overflow, readable contrast on dark background
- [ ] Columns: stack vertically on phone, 2-up on tablet, 3-up on desktop
- [ ] Charts: respect container width, no horizontal scroll
- [ ] Font: heading font on titles/names, body font everywhere else

### Accessibility spot checks
- [ ] Tab through entire topbar without mouse; every link reachable
- [ ] Active page indicated via `aria-current="page"` (inspect DOM)
- [ ] Logo `<img>` has `alt` text
- [ ] `prefers-reduced-motion`: verify no transitions when enabled
- [ ] Color contrast: gold-on-dark and slate-on-dark pass WCAG AA (4.5:1 for text)

## Adding a new component (checklist)

1. Name it `.c-<component>`. Keep rules co-located with other `.c-*` blocks in `styles.css`.
2. Use tokens. No raw hex, no raw pixel spacing.
3. If it needs Python-side assembly, put the renderer in `components/<name>.py` and import it from views.
4. If it is page-specific, consider whether it is really a component. One-off markup in a view is fine; duplicated markup across views is not.
5. Update this doc's legacy-to-new table if the component replaces one.
