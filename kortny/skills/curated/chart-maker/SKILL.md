---
name: chart-maker
description: Use when asked to make, plot, or generate a chart, graph, or data visualization image to share in Slack — a bar chart, line chart, or trend plot saved as a PNG, with labeled axes and a clear takeaway, not a table of numbers.
metadata:
  version: 1.0.0
  display_name: Chart Maker
  tags: chart, graph, plot, visualization, bar chart, line chart, png, matplotlib, data viz
---

## Goal

Produce one clean, presentation-grade chart as a PNG — labeled axes, a title
that states the takeaway, no chart junk — and upload it to the Slack thread.

## Steps

1. **Pick the right chart for the comparison.** Categories → bar (`bar`) or
   horizontal bar (`hbar` when labels are long). Change over time → line
   (`line`). Parts of a whole, and only when explicitly asked, with ≤ 5
   slices → `pie`. When in doubt, use a bar chart.
2. **Write the title as the takeaway**, not the metric: "Signups doubled
   after the referral launch", not "Signups by week".
3. **Label both axes** with units, and name the data source.
4. **Render with the script** (`scripts/make_chart.py`) by passing a JSON
   spec. It applies a restrained colorblind-friendly palette, a light grid on
   the value axis only, thousands-separated ticks, and the source caption.
5. **Upload the PNG to the thread** with a one-line read of what the chart
   shows. If it belongs in a deck or report, hand the file path to
   `deck-builder` or `document_studio`.

## Visualization standards (non-negotiable)

- **No pie-chart abuse.** Pies only on explicit request and only with ≤ 5
  slices; the script refuses more. Default to bars for category comparisons.
- **Labeled axes, always.** A chart with bare axes is not done.
- **The title carries the point** — a takeaway, not a restatement of the axis.
- **No chart junk** — no 3D, no gradients, no shadows; grid on the value axis
  only.
- **Honest numbers** — only values that are in the data; don't smooth or
  invent points to make a trend look cleaner.

## Pairing

- `data-brief` / `data-digest` → the prose read of the same numbers.
- `deck-builder` / `document_studio` → embed the PNG by file path.

## Script

- `scripts/make_chart.py` — inputs: `--spec spec.json` (or JSON on stdin),
  `--out chart.png`. Chart `type`: `bar` (grouped if multiple series),
  `hbar`, `line`, `pie` (≤ 5 slices). Fields: `title`, `x_label`, `y_label`,
  `labels`, `series` (`[{name, values}]`), `source`. Deps: matplotlib (Agg
  backend, headless). No network. See `references/spec-format.md`.
