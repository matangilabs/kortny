# Authoring Guide — styled-report-pdf v2.0

This reference documents every component class you can use in the HTML you
write for `render_pdf.py`. For each component: what it is, when to use it,
the exact HTML to write, and a rendered example.

The script handles all page geometry, fonts, colors, and page numbers — you
only write content-level markup using these classes.

---

## Quick-start: fragment vs full-document mode

**Fragment mode (recommended):** write only the `<body>` content — no
`<html>`, `<head>`, or `<style>` tags. The script wraps it in the full
document template, applies the theme, and prepends a dark cover when
`--title` is passed.

**Full-document mode:** write a complete `<!DOCTYPE html>…</html>` document.
The script injects the theme CSS into your `<head>`. Your own `<style>` wins
for anything not covered by the theme.

---

## Page geometry

All three themes render to **A4 (210 × 297 mm)**. Margins are handled by the
theme — content is inset via `.page-body` padding (24 mm top/bottom, 22 mm
sides). The **cover** and **section-divider** elements live outside
`.page-body` and span the full page width for a true full-bleed effect.

In fragment mode the script inserts the `.page-body` wrapper automatically.
In full-document mode you must wrap normal content in `<main class="page-body">`.

---

## Component reference

### `.kicker` / `.eyebrow` — mono CAPS section label

The mono uppercase label that sits above every `<h2>`. It names the section
type (e.g. "Executive Summary", "Sector Analysis", "Key Risks") and is the
signature move of the Editorial Mono style.

**When to use:** before every `<h2>`. Never use it mid-paragraph.

```html
<p class="kicker">Executive Summary</p>
<h2>The shift to self-hosted AI coworkers</h2>
```

---

### `<h2>` / `<h3>` / `<h4>` — heading scale

- `<h2>` — major section heading; auto-registers as running header in the
  page footer. Always preceded by `.kicker`.
- `<h3>` — sub-section heading within a section.
- `<h4>` — tertiary label (renders as mono CAPS, smaller than `<h3>`).

```html
<p class="kicker">Financial Overview</p>
<h2>Revenue trajectory</h2>
<p class="lead">ARR grew 3× in twelve months …</p>

<h3>Unit economics</h3>
<p>Gross margin expanded from 61% to 74% …</p>

<h4>Cost of goods sold</h4>
<p>Infra spend declined as a share of revenue …</p>
```

---

### `<p class="lead">` — standfirst / opening sentence

A larger, airier paragraph used as the opening sentence under a heading.
Limit to one per section.

```html
<h2>Capital formation</h2>
<p class="lead">The private AI market absorbed $75B in 2025, a 38% increase
over 2024 — driven almost entirely by growth-stage rounds above $100M.</p>
<p>Seed activity remained flat, suggesting the market has entered a
consolidation phase …</p>
```

---

### `.stat-grid` + `.stat-card` — N-up metric cards

Four (or three, or two) key statistics displayed prominently at the top of a
section. Each card has a large display number, a mono CAPS label, and an
optional delta / sub-line.

**When to use:** near the top of a major section, before prose. Use 4-up for
high-level KPI rows, 2-up or 3-up for more focused comparisons.

**Structure:**
```html
<div class="stat-grid stat-grid--4">
  <div class="stat-card">
    <span class="stat-card__label">Capital Raised</span>
    <span class="stat-card__value">$75<span class="stat-card__unit">B</span></span>
    <span class="stat-card__sub stat-card__sub--up">↑ 38% YoY</span>
  </div>
  <div class="stat-card">
    <span class="stat-card__label">Active Deployments</span>
    <span class="stat-card__value num">12,400</span>
    <span class="stat-card__sub stat-card__sub--up">↑ 2.1×</span>
  </div>
  <div class="stat-card">
    <span class="stat-card__label">Median Payback</span>
    <span class="stat-card__value">4.2<span class="stat-card__unit">mo</span></span>
    <span class="stat-card__sub stat-card__sub--down">↓ 1.1 mo</span>
  </div>
  <div class="stat-card">
    <span class="stat-card__label">Self-Hosted Share</span>
    <span class="stat-card__value">61<span class="stat-card__unit">%</span></span>
    <span class="stat-card__sub">flat</span>
  </div>
</div>
```

Use `.stat-grid--3` or `.stat-grid--2` for 3-up or 2-up layouts. For 3-up,
the value font is implicitly larger because there is more horizontal space —
the number still renders at 40pt.

**Rules:**
- `.stat-card__unit` is for short suffixes (B, M, K, %, mo, ×).
- `.stat-card__sub--up` renders green; `--down` renders red.
- Add class `.num` to `.stat-card__value` when the number has no unit suffix
  and should use tabular figures (e.g. "12,400").

---

### `<table class="data-table">` — financial data table

The primary data display component. Hairline rules (no box border), mono CAPS
headers, bold first column, right-aligned tabular numerics. Zebra rows. The
`<thead>` repeats on page breaks automatically.

**Numeric alignment is mandatory.** Add class `.num` to every cell (`<td>` or
`<th>`) that contains a number. This right-aligns it and applies tabular
figures so digits line up in columns.

**Structure:**
```html
<table class="data-table">
  <caption>Table 1 — Funding by stage, 2024–2026 (US$ M)</caption>
  <thead>
    <tr>
      <th scope="col">Stage</th>
      <th scope="col" class="num">2024</th>
      <th scope="col" class="num">2025</th>
      <th scope="col" class="num">2026E</th>
      <th scope="col" class="num">CAGR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Seed</th>
      <td class="num">1,240</td>
      <td class="num">1,810</td>
      <td class="num">2,300</td>
      <td class="num">+36%</td>
    </tr>
    <tr>
      <th scope="row">Series A</th>
      <td class="num">3,900</td>
      <td class="num">6,200</td>
      <td class="num">9,100</td>
      <td class="num">+53%</td>
    </tr>
    <tr>
      <th scope="row">Growth</th>
      <td class="num">11,400</td>
      <td class="num">22,800</td>
      <td class="num">41,500</td>
      <td class="num">+91%</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <th scope="row">Total</th>
      <td class="num">16,540</td>
      <td class="num">30,810</td>
      <td class="num">52,900</td>
      <td class="num">+78%</td>
    </tr>
  </tfoot>
</table>
<p class="source">Source: Kortny Research estimates; PitchBook. E = estimate.</p>
```

**Rules:**
- Always include a `<caption>` — it renders as a mono CAPS label above the
  table.
- Always use `<thead>`, `<tbody>`, and `<tfoot>` (totals row).
- Row headers (`<th scope="row">`) render bold in the first column.
- Follow every table with a `<p class="source">` attribution line.
- Use a consistent decimal count per column (e.g. all percentages to one
  decimal, all dollar amounts with comma thousands separator).

---

### `.pull-quote` — isolated key sentence

A memorable sentence pulled from the body and displayed with a left accent
border. Use sparingly — 1 or 2 per document maximum. Never use it for a
statistic; use a stat card for numbers.

```html
<blockquote class="pull-quote">
  <p>"Self-hosting collapses the per-seat economics that defined the first
     wave of AI tools."</p>
  <cite>— Internal analysis, Section 2.3</cite>
</blockquote>
```

---

### `.key-takeaway` — insight callout box

A 1–2 sentence synthesis with a tinted background and left accent border. Use
at the end of a section to surface the conclusion for a skimming reader.

```html
<aside class="key-takeaway">
  <span class="key-takeaway__tag">Key Takeaway</span>
  <p>Mid-market teams that self-host reach payback approximately 40% faster
     than SaaS adopters, driven by elimination of per-seat fees above roughly
     twenty users.</p>
</aside>
```

---

### `.bar` + `.bar-row` + `.bar-legend` — CSS concentration bar

A stacked horizontal bar showing a part-to-whole split. Use instead of a pie
chart when there are ≤ 4 segments and the point is the proportion, not a
trend. Renders perfectly in WeasyPrint with no font-embedding risk.

Widths are inline-style percentages. Background colors reference the CSS
custom properties defined by the theme.

```html
<div class="bar" role="img"
     aria-label="Deployment mix: 61% self-hosted, 22% cloud, 17% hybrid">
  <span class="bar-row" style="width:61%; background:var(--accent)">61%</span>
  <span class="bar-row" style="width:22%; background:var(--accent-2)">22%</span>
  <span class="bar-row" style="width:17%; background:var(--ink-soft)">17%</span>
</div>
<div class="bar-legend">
  <span><i style="background:var(--accent)"></i>Self-hosted</span>
  <span><i style="background:var(--accent-2)"></i>Cloud</span>
  <span><i style="background:var(--ink-soft)"></i>Hybrid</span>
</div>
```

---

### `.figure` — chart or image wrapper

Wraps any chart image or inline SVG with a mono CAPS caption above. Use for
time-series charts, multi-series bars, or any visual that needs a number.

```html
<figure class="figure">
  <p class="figure__cap">Figure 2 — Monthly ARR growth (US$ M), 2024–2026</p>
  <img src="/workspace/charts/arr_growth.png" alt="ARR growth chart">
  <p class="source">Source: Internal finance; Kortny Research.</p>
</figure>
```

For inline SVG (e.g. from `vl-convert`), replace the `<img>` with the raw
`<svg>…</svg>` block. Both render as vector in WeasyPrint.

---

### `<p class="source">` — attribution line

A micro mono line below a table, figure, or CSS bar citing data origin. Use
after every exhibit.

```html
<p class="source">Source: PitchBook; Kortny Research estimates. E = estimate.</p>
```

---

### `.sources` — end-of-section references block

A numbered list of citations at the end of a section or the document.

```html
<div class="sources">
  <p class="sources__head">Notes &amp; Sources</p>
  <ol>
    <li>All figures are management estimates unless cited.</li>
    <li>PitchBook, "Global AI Funding Report," May 2026.</li>
    <li>Kortny internal deployment data, April 2026.</li>
  </ol>
</div>
```

---

### `.section-divider` — full-bleed dark section page

A full-page dark interstitial page that appears before a major section. This
is the "Viktor-style" premium touch that signals professional investment
research. Use for reports with 3 or more major sections.

The element must live **outside** a `.page-body` div — place it directly in
`<body>` (or between two `<main class="page-body">` blocks in full-doc mode).

```html
<!-- end of previous section's .page-body -->

<div class="section-divider">
  <span class="section-divider__num">02</span>
  <p class="section-divider__kicker">Part Two</p>
  <h1 class="section-divider__title">Adoption &amp; Economics</h1>
  <p class="section-divider__deck">How self-hosting changes the cost curve
     for mid-market teams.</p>
</div>

<!-- .page-body for the next section follows -->
```

**Rules:**
- `__num` is a two-digit mono label (01, 02, 03…).
- `__kicker` is the part label in muted mono CAPS.
- `__title` is the big display title (max 18 characters per line; use `<br>`
  for longer titles).
- `__deck` is one optional sentence of context.

---

## Cover (fragment mode)

In fragment mode the script generates the cover from the CLI flags — you do
**not** write cover HTML in your fragment. The cover is always:
- Full-bleed dark background.
- Top bar: "KORTNY RESEARCH" left, date right (mono CAPS).
- Kicker: `--kicker` text in accent color.
- Title: `--title` in large display serif.
- Deck: `--subtitle` in body serif, muted.
- Bottom bar: "CONFIDENTIAL" left.

In full-document mode, write the cover yourself using the class structure
below (place it before the first `<main class="page-body">`):

```html
<section class="cover">
  <div class="cover__top">
    <span>KORTNY RESEARCH</span>
    <span>CONFIDENTIAL — 13 JUN 2026</span>
  </div>
  <div class="cover__main">
    <p class="cover__kicker">Investment Research — Confidential</p>
    <h1 class="cover__title">SpaceX IPO<br>Readiness Analysis</h1>
    <p class="cover__deck">A field analysis of capital formation, market
       timing, and competitive moats ahead of a potential public offering.</p>
  </div>
  <div class="cover__bottom">
    <span>Prepared for Acme Capital Partners</span>
    <span></span>
  </div>
</section>
```

---

## Images

There is no network at render time. Save any image (e.g. a `chart-maker` PNG)
under the workbench dir and reference it by absolute path:

```html
<img src="/workspace/charts/revenue_growth.png" alt="Revenue growth" style="width:100%">
```

Remote `src` URLs will fail to load silently — always use absolute local paths.

---

## CLI flags reference

| Flag | Description |
|---|---|
| `--html PATH` | Path to HTML fragment or full document |
| `--html-stdin` | Read HTML from stdin |
| `--out PATH` | Output .pdf path |
| `--style NAME` | `editorial-mono` (default) / `minimal` / `editorial-feature` |
| `--title TEXT` | Cover title (fragment mode) |
| `--subtitle TEXT` | Cover deck text |
| `--kicker TEXT` | Mono CAPS eyebrow above cover title (default: "Research Report") |
| `--date TEXT` | Cover date string (default: today) |
| `--accent HEX` | Override theme accent color (e.g. `#1D4ED8`) |

---

## Worked example — 3-section SpaceX-style IPO report

This is a pattern-match template the model can adapt for any investment
research report. It demonstrates: cover (via flags), stat cards, a ruled
table, a CSS bar, a pull-quote, a key takeaway, a section divider, and sources.

**Fragment file (`spacex_ipo.html`):**

```html
<!-- SECTION 1 BODY -->
<p class="kicker">Executive Summary</p>
<h2>The case for a SpaceX public offering</h2>
<p class="lead">SpaceX enters the public markets with an unassailable
technical moat, $9.2B in contracted backlog, and a cost-per-kg-to-orbit
advantage of 4× over the nearest competitor.</p>

<p>After seventeen years of private capital formation, a 2026 IPO window
opens at a moment of peak commercial momentum: Starlink has crossed 5 million
subscribers, Starship has completed its first commercial payload mission, and
the government-services pipeline is fully funded through 2030.</p>

<div class="stat-grid stat-grid--4">
  <div class="stat-card">
    <span class="stat-card__label">2025 Revenue</span>
    <span class="stat-card__value">$11<span class="stat-card__unit">B</span></span>
    <span class="stat-card__sub stat-card__sub--up">↑ 42% YoY</span>
  </div>
  <div class="stat-card">
    <span class="stat-card__label">Starlink Subscribers</span>
    <span class="stat-card__value">5.1<span class="stat-card__unit">M</span></span>
    <span class="stat-card__sub stat-card__sub--up">↑ 2.4×</span>
  </div>
  <div class="stat-card">
    <span class="stat-card__label">Launch Cost / kg LEO</span>
    <span class="stat-card__value">$1<span class="stat-card__unit">K</span></span>
    <span class="stat-card__sub stat-card__sub--down">↓ 74% vs. 2020</span>
  </div>
  <div class="stat-card">
    <span class="stat-card__label">Valuation (est.)</span>
    <span class="stat-card__value">$350<span class="stat-card__unit">B</span></span>
    <span class="stat-card__sub">pre-IPO</span>
  </div>
</div>

<blockquote class="pull-quote">
  <p>"The cost curve is the moat. No incumbent can replicate Falcon 9
     economics without a decade of iteration — and Starship resets the
     baseline again."</p>
  <cite>— Kortny Research, Section 3.1</cite>
</blockquote>

<h3>Revenue mix</h3>
<p>Starlink internet services now represent the majority of group revenue,
   with launch services and government contracts making up the remainder.</p>

<div class="bar" role="img"
     aria-label="Revenue mix: 61% Starlink, 26% Launch Services, 13% Government">
  <span class="bar-row" style="width:61%; background:var(--accent)">61%</span>
  <span class="bar-row" style="width:26%; background:var(--accent-2)">26%</span>
  <span class="bar-row" style="width:13%; background:var(--ink-soft)">13%</span>
</div>
<div class="bar-legend">
  <span><i style="background:var(--accent)"></i>Starlink</span>
  <span><i style="background:var(--accent-2)"></i>Launch Services</span>
  <span><i style="background:var(--ink-soft)"></i>Government</span>
</div>

<table class="data-table">
  <caption>Table 1 — Revenue by segment, 2023–2025 (US$ B)</caption>
  <thead>
    <tr>
      <th scope="col">Segment</th>
      <th scope="col" class="num">2023</th>
      <th scope="col" class="num">2024</th>
      <th scope="col" class="num">2025</th>
      <th scope="col" class="num">CAGR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Starlink</th>
      <td class="num">3.0</td>
      <td class="num">5.2</td>
      <td class="num">6.7</td>
      <td class="num">+49%</td>
    </tr>
    <tr>
      <th scope="row">Launch Services</th>
      <td class="num">2.1</td>
      <td class="num">2.5</td>
      <td class="num">2.9</td>
      <td class="num">+17%</td>
    </tr>
    <tr>
      <th scope="row">Government</th>
      <td class="num">1.1</td>
      <td class="num">1.2</td>
      <td class="num">1.4</td>
      <td class="num">+13%</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <th scope="row">Total</th>
      <td class="num">6.2</td>
      <td class="num">8.9</td>
      <td class="num">11.0</td>
      <td class="num">+33%</td>
    </tr>
  </tfoot>
</table>
<p class="source">Source: Kortny Research estimates; SpaceX filings (private). E = estimate.</p>

<aside class="key-takeaway">
  <span class="key-takeaway__tag">Key Takeaway</span>
  <p>Starlink is the revenue engine. Its subscriber growth rate, at 2.4×
     year-over-year, creates a durable annuity that de-risks the launch
     services segment's lumpier government-contract revenue.</p>
</aside>

<div class="sources">
  <p class="sources__head">Notes &amp; Sources</p>
  <ol>
    <li>Revenue figures are Kortny Research estimates based on public filings
        and management commentary. SpaceX has not published audited financials.</li>
    <li>Starlink subscriber count: SpaceX press release, March 2026.</li>
    <li>LEO launch cost per kg: Bryce Space & Technology, 2025 Global Launch
        Report.</li>
  </ol>
</div>


<!-- SECTION DIVIDER — lives outside .page-body in full-doc mode -->
<!-- In fragment mode, the script auto-wraps in a single page-body,
     so place this HTML between two logical sections;
     the break-before/break-after CSS ensures page isolation. -->

<div class="section-divider">
  <span class="section-divider__num">02</span>
  <p class="section-divider__kicker">Part Two</p>
  <h1 class="section-divider__title">Competitive<br>Landscape</h1>
  <p class="section-divider__deck">Mapping the orbital-economy incumbents
     against SpaceX's cost and cadence advantages.</p>
</div>


<!-- SECTION 2 BODY -->
<p class="kicker">Competitive Analysis</p>
<h2>Structural advantages versus incumbents</h2>
<p class="lead">SpaceX operates at a cost structure no legacy launch
provider can match without greenfield investment — a 4× cost advantage
in launch and a fully-owned end-to-end internet service.</p>

<p>United Launch Alliance, Arianespace, and RocketLab each compete on
   specific orbital regimes but lack the fully-reusable first-stage
   economics that allow SpaceX to price below production cost when
   capturing anchor contracts.</p>

<h3>Launch cadence comparison</h3>

<table class="data-table">
  <caption>Table 2 — Global orbital launches, 2025</caption>
  <thead>
    <tr>
      <th scope="col">Operator</th>
      <th scope="col" class="num">Launches</th>
      <th scope="col" class="num">Success Rate</th>
      <th scope="col" class="num">Cost / kg LEO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">SpaceX</th>
      <td class="num">96</td>
      <td class="num">99.4%</td>
      <td class="num">$1,000</td>
    </tr>
    <tr>
      <th scope="row">RocketLab</th>
      <td class="num">18</td>
      <td class="num">94.4%</td>
      <td class="num">$7,500</td>
    </tr>
    <tr>
      <th scope="row">ULA</th>
      <td class="num">7</td>
      <td class="num">100%</td>
      <td class="num">$18,000</td>
    </tr>
    <tr>
      <th scope="row">Arianespace</th>
      <td class="num">5</td>
      <td class="num">100%</td>
      <td class="num">$10,000</td>
    </tr>
  </tbody>
</table>
<p class="source">Source: Bryce Space &amp; Technology; Kortny Research.</p>

<aside class="key-takeaway">
  <span class="key-takeaway__tag">Key Takeaway</span>
  <p>SpaceX's launch cadence is 5× the nearest commercial competitor and its
     cost-per-kg is 7–18× lower. This is not a temporary lead — it reflects
     a decade of iterative reusability development that cannot be fast-followed.</p>
</aside>

<div class="sources">
  <p class="sources__head">Notes &amp; Sources</p>
  <ol>
    <li>Bryce Space and Technology, "State of the Space Industrial Base 2025."</li>
    <li>Cost per kg estimates for non-SpaceX providers reflect list price; actual
        contracted rates may vary.</li>
  </ol>
</div>
```

**Render command:**

```bash
python scripts/render_pdf.py \
  --html spacex_ipo.html \
  --out spacex_ipo_report.pdf \
  --style editorial-mono \
  --title "SpaceX IPO Readiness Analysis" \
  --subtitle "A field analysis of capital formation, market timing, and competitive moats." \
  --kicker "Investment Research — Confidential" \
  --date "13 Jun 2026"
```

Expected output: a multi-page A4 PDF with:
- Page 1: full-bleed black cover with display title, mono kicker, deck text.
- Pages 2+: warm off-white body pages with running page number (bottom-right),
  "KORTNY RESEARCH" (bottom-left), running section name (top-right).
- Section 1: kicker + H2 + lead + 4-up stat cards + pull-quote + CSS bar +
  data table + key takeaway + sources.
- Section divider: full-bleed black page with section number, kicker, and title.
- Section 2: kicker + H2 + lead + data table + key takeaway + sources.

---

## WeasyPrint compatibility notes

All component classes are WeasyPrint-compatible. Specifically:

- **Named pages** (`page: cover` on `.cover` and `.section-divider`) and
  `@page cover { margin: 0 }` work in WeasyPrint — this is what produces
  true full-bleed covers without a Chromium/Gotenberg backend.
- **`@page` margin boxes** (`@bottom-right`, `@top-right`, `@bottom-left`)
  provide automatic page numbers and running headers — WeasyPrint-only;
  Chromium ignores these (for Chromium, a `footer.html` template would be
  needed, noted in the research as HIG-244 Gotenberg upgrade).
- **`string-set`** on `<h2>` registers the section name for the `@top-right`
  margin box running header — WeasyPrint only.
- **CSS Grid** (`display: grid`) for stat cards — supported since WeasyPrint
  v53.
- **No `color-mix()`** — all tinted backgrounds use literal `rgba()` values
  (e.g. `rgba(250,250,248,0.70)`) for maximum compatibility across WeasyPrint
  versions.
- **`break-inside: avoid`** and `page-break-inside: avoid` (legacy alias) are
  both set for maximum engine coverage.
- **`orphans` / `widows`** are set globally — honored by WeasyPrint; Chromium
  support is partial.
