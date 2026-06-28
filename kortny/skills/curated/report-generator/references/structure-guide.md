# Report Structure Guide

## Standard structure

### Executive summary (always first)
One paragraph. The conclusion. What the reader needs to know even if they read nothing else.
- Finding: [The main thing the data shows]
- Implication: [What it means for the business]
- Recommendation: [What to do next]

### Key findings
3-7 findings, each with:
- The finding (conclusion-first, not observation-first)
- The evidence (data point, source)
- The implication

### Methodology (for practitioner audience; omit for exec)
How the analysis was done. Data sources, time period, limitations.

### Detailed analysis
Expand on each finding with supporting data, tables, and charts (described in markdown for the file version).

### Recommendations
Prioritized list. Each recommendation:
- What to do
- Who owns it
- Why (tied to a specific finding)
- Rough effort/impact estimate

### Appendix (optional)
Raw data tables, source list, glossary.

---

## Slack mrkdwn summary format (always included inline)

Lead with a bold title and date. Follow with TL;DR (one sentence), then 3-5 key finding bullets, then the primary recommendation. Close with "Full report attached." Keep this under 200 words.

## File upload format
Upload as `[report-title]-[date].md` for markdown.
For PDF: call the `document_studio` tool with this report's content as block IR.
