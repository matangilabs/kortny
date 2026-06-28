# Provenance

**Concept source**: Anthropic proprietary internal document / visualization skills
**Script authorship**: Original (clean-room rewrite)
**Adaptation date**: 2026-06-12
**Adapted by**: Agent D (HIG-239)

## What was adapted

The SKILL.md was written fresh. The script (`make_chart.py`) is an original
implementation using matplotlib (Agg headless backend) — no code or wording
was copied from the upstream source. The upstream skill is proprietary; only
the *visualization standards* (unprotectable ideas) informed this skill:
labeled axes, takeaway-style titles, no chart junk, and the no-pie-chart-abuse
discipline (the script caps pies at 5 slices).

## Script dependencies

- `make_chart.py`: matplotlib only. argparse, JSON spec in (file or stdin),
  `.png` file out, no network access.

## Slack-first adaptations

- Output targets a Slack file upload (the PNG itself), or hand-off to
  `deck-builder` / `document_studio` by file path.
- File paths use workbench paths, not local development paths.

## License

Original concept source is proprietary — this skill and script are a
clean-room rewrite. This file and all files in this directory are subject to
the Kortny project license.
