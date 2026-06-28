# Audience-Tiering Table

Use this table to determine the appropriate version(s) of a report for each audience.

| Audience | Length | What they want | Format | Tone |
|---|---|---|---|---|
| **Executive / Board** | 1 page max | Decision + recommendation. Why it matters. What it costs to act or not act. | Bullet-led, no tables unless essential | Direct, confident |
| **Manager / Team lead** | 2-3 pages | Findings + implications for their team. Action items they own. | Structured sections, short tables OK | Collegial, action-oriented |
| **Analyst / Practitioner** | Full length | Full methodology, raw findings, caveats, data tables, sources | Sections with subsections, full tables, appendices | Precise, hedged where appropriate |
| **All-hands / Wide audience** | 1 page | What changed and why it matters to them personally | Prose-led, no jargon | Conversational |

## Tiering workflow

1. Ask: who reads this? If multiple audiences, produce the executive version first.
2. Trim for exec: cut everything that doesn't answer "what should we do?" or "what does this mean for us?"
3. Expand for practitioner: restore methodology, data tables, source citations.
4. Never combine: a report that tries to serve executives and practitioners serves neither.

## Upgrade options

- For styled PDF output: use the `document_studio` tool (`format: "pdf"`).
- For slide deck version: pair with the `deck-builder` skill.
- For spreadsheet appendix: pair with the `spreadsheet-builder` skill.
