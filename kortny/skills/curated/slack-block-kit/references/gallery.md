# Layout gallery — scenario → composition

Distilled from Slack's own Block Kit templates. Each entry maps a common
coworker scenario to the presentation elements that render it well. Different
teams need different layouts; pick by *what the data is*, not by department.

The presentation hint composes a SEQUENCE of elements, so a rich answer is just
the right elements in order (e.g. `header` → `items` → `context`). Voice (the
conversational framing) always stays in the `message`; these elements carry the
structured data.

## Answer with sources (research / lookup / "what's the latest on X")
Pattern: rich markdown body in the message → `sources`.
- message: the synthesized answer in markdown (headings, a table, bullets, a
  quote all render natively in the markdown block).
- `sources`: the citations, by `source_ref` (URLs bound server-side).
- optional `context`: "Generated from N sources · verify before sharing."
Slack template: deep_search_result, catalog_search.

## Entity list ("my open issues", "deals at risk", "schedules", "tickets")
Pattern: `items` (one section+context row per entity, divider-separated).
- Each item: title + a few facts (status/owner/amount/cadence) + a one-line
  context (source/freshness). Optionally a leading `header`.
Slack template: ticket_list, lunch_poll, itinerary, project_tracker.

## Comparison ("A vs B", "spend by vendor", "usage by model")
Pattern: `table` when rows share columns and comparison is the job; otherwise
`fields` for a 2-way at-a-glance.
Slack template: (general tabular).

## Single record / status ("status of the Acme deal", "is prod healthy")
Pattern: `fields` (key-values about one thing), optional `context` for freshness.
Slack template: expense_app (per-item section+fields).

## Dashboard / multi-section briefing ("morning digest", "project check-in")
Pattern: `header` → `fields` (headline metrics) → `items` (the list) →
`context` (provenance). Use `divider` between sections if it aids scanning.
Slack template: itinerary, project_tracker, expense_app.

## A few important objects (2-5 entities each worth a tile)
Pattern: `cards` (compact tiles). For longer/plainer lists prefer `items`.
Slack template: approval, deep_search_result (source cards).

## Plain conversation / opinion / explanation / ack
Pattern: NO presentation. Just the message. Most replies are this.

---

## Not yet available (interactive — next slice)

Slack's templates also show buttons, overflow menus, selects, datepickers,
checkboxes, and modal forms (todo_app, lunch_poll actions, app_menu,
calendar, workplace_poll). These are deferred to the interactivity slice
because each carries server state (approval keys, action ids) that must be
bound to server-owned records and acknowledged within Slack's 3-second window —
they are not display-only. Do not emit interactive elements from the
presentation hint yet.

## Hard rule recap

- Layout is unlimited; **egress is not**. Never author a URL, image URL, action
  id, or block id. Links/citations come through `source_ref` → server-resolved
  evidence URLs. Images are deferred.
- One good composition beats a dashboard. Match the layout to the data; when in
  doubt, prose.
