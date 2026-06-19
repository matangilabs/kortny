---
name: slack-block-kit
description: Use whenever a Slack reply carries structured data — a list of entities with attributes, key-value facts or metrics, status, comparisons, or rows that share columns. Teaches how to render that data as native Slack Block Kit (cards, fields, tables, context) instead of flat prose, by emitting a small presentation hint. Trigger this for any answer that lists things, reports numbers, shows statuses, or compares options — even when the user didn't ask for "formatting."
metadata:
  version: 1.0.0
  display_name: Slack Block Kit Presentation
  tags: slack, block kit, formatting, presentation, cards, table, fields, data
---

## Goal

Make Slack answers scannable. Conversational *voice* stays as prose, but
*structured data* — a list of entities, a set of facts, metrics, statuses,
comparisons, rows — renders as native Slack Block Kit so the reader takes it in
at a glance instead of parsing a wall of bullets.

You do this by adding an optional `presentation` object next to your `message`.
You never write Slack Block Kit JSON yourself — you describe the data in a tiny
schema and deterministic code builds and validates the real blocks. This keeps
you safe (no malformed payloads) and lets you focus on *what* the data is.

## The contract

Return your normal object, optionally with `presentation`:

```json
{"message": "<the full Slack answer in prose>",
 "presentation": {"elements": [ ... ]}}
```

Three rules that keep this trustworthy:

1. **Voice in the message, data in the presentation — never both.** The message
   is the conversational framing: a lead-in, the gist, and any insight or
   recommendation. The per-item detail (the attributes you'd put in cards, the
   rows of a table, the key-values of fields) goes in the presentation ONLY. Do
   not also bullet-list those same items/attributes in the message — the reader
   should never see the same data twice. The message should still stand on its
   own as a short summary (notifications and screen readers see only the text),
   so you may *name* the items in a sentence ("3 schedules: A, B, and C"), but
   keep the attributes (cadence, status, cost…) to the presentation.
2. **Never invent data for the presentation.** Every value comes from the
   evidence you were given. No new numbers, names, or claims.
3. **Never put Block Kit JSON, block types, `action_id`, or the presentation
   object inside the `message` string.** Buttons, links-with-actions, and IDs
   are out of scope here — display only.

## When to use which element

The instinct to build: *if you just wrote bullets describing several things, or
a run of "Label: value" facts, or a little table — that's a signal to add a
presentation element.* Plain conversation is not.

- **items** — *the default for a list of entities* (issues, schedules,
  accounts, providers, deals, PRs, candidates). Each item is a title + a few
  facts + an optional one-line meta. Renders as Slack's native section+context
  rows — readable in a narrow thread, never a wall of bullets. Reach for this
  first whenever the answer lists several things, each with attributes.
- **table** — rows that share the *same columns* and the point is comparison
  (usage by model, spend by vendor, headcount by team). When you emit a table,
  do **not** also write a markdown table in the message.
- **fields** — key-value facts, metrics, or status about *one* thing (a single
  schedule's cadence/next-run/delivery; a provider's health/models/tier).
- **cards** — use *sparingly* for 2-5 important objects that each deserve a
  compact tile. For a longer or plainer list, prefer **items**.
- **sources** — citations / references the answer draws on. Give a `source_ref`
  for each (the sources you were provided); Kortny attaches the real link. Never
  write a URL yourself. Only use this when the answer actually cites sources.
- **context** — provenance / freshness footnotes ("Checked 2 minutes ago",
  "Partial: web search timed out").

**Skip presentation entirely** for plain conversation, a greeting, an apology,
an explanation, or a single short fact. Do not turn chit-chat into a dashboard —
over-formatting is as bad as under-formatting. One good element beats five.

## Element shapes

```json
{"type":"items","title":"optional","items":[
  {"title":"HIG-276 Project layer","facts":[{"label":"Status","value":"In progress"},{"label":"Owner","value":"Aneesh"}],"context":["Source: Linear"]}]}

{"type":"table","title":"optional","columns":["Model","Cost"],
 "rows":[["gpt-4o","$12.30"],["deepseek","$3.10"]]}

{"type":"fields","title":"optional","items":[
  {"label":"Status","value":"Active"},{"label":"Next run","value":"09:00"}]}

{"type":"cards","title":"optional","items":[
  {"title":"Name","subtitle":"optional","body":"optional short line",
   "fields":[{"label":"Cadence","value":"Every 6 hours"}]}]}

{"type":"sources","items":[
  {"source_ref":"source:0","title":"optional shorter title","body":"why this source matters"}]}

{"type":"context","items":["Checked 2 minutes ago"]}
```

## Worked examples

**Example 1 — a list of entities → items.** Note the message *names* them in one
sentence but does NOT bullet their attributes — the items list carries that, so
there's no duplication.
Request: "what's the status of my schedules?"
```json
{"message":"Yep, you've got 3 active schedules running — Integration catalog sync, Memory consolidation, and Witness scan. All healthy.",
 "presentation":{"elements":[{"type":"items","items":[
   {"title":"Integration catalog sync","facts":[{"label":"Cadence","value":"Every 6 hours"},{"label":"Delivery","value":"Dashboard"}]},
   {"title":"Memory consolidation","facts":[{"label":"Cadence","value":"Daily"},{"label":"Delivery","value":"Dashboard"}]},
   {"title":"Witness scan","facts":[{"label":"Cadence","value":"Every 6 hours"},{"label":"Delivery","value":"Dashboard"}]}]}]}}
```

**Example 5 — a source-backed answer → sources.** Only when the answer actually
draws on sources you were given. Reference them by `source_ref`; never write the
URL — Kortny attaches the real link.
Request: "what's the latest on the Artemis II mission?"
```json
{"message":"Artemis II splashed down safely after a 9-day crewed lunar flyby — the first since Apollo. Here's what I drew on:",
 "presentation":{"elements":[{"type":"sources","items":[
   {"source_ref":"source:0","body":"Official NASA mission overview."},
   {"source_ref":"source:1","body":"Splashdown report, Apr 10."}]}]}}
```

**Example 2 — rows that share columns → table.**
Request: "break down my LLM usage by model this week"
```json
{"message":"Here's your usage by model this week:",
 "presentation":{"elements":[{"type":"table","columns":["Model","Cost","Tokens"],
   "rows":[["gpt-4o","$12.30","1.2M"],["deepseek","$3.10","800k"],["haiku","$0.90","400k"]]}]}}
```

**Example 3 — facts about one thing → fields (+ provenance → context).**
Request: "what's the status of the billing integration?"
```json
{"message":"The billing integration is healthy and connected.",
 "presentation":{"elements":[
   {"type":"fields","title":"Billing integration","items":[{"label":"Status","value":"Connected"},{"label":"Last sync","value":"4 min ago"},{"label":"Account","value":"acme-prod"}]},
   {"type":"context","items":["Source: Composio connection status"]}]}}
```

**Example 4 — plain conversation → no presentation.**
Request: "hey, you around?"
```json
{"message":"Yep, around and ready — what do you need?"}
```

## Limits & validation

The renderer enforces Slack's hard limits and drops/degrades anything invalid
(it will fall back to plain prose rather than drop your answer), so you don't
have to count characters. For the exact caps and surface rules (tables are
message-only, 50 blocks/message, char limits) see `references/elements.md`. A
standalone validator lives in `scripts/validate_blocks.py` for anyone building
Block Kit outside this pipeline.
