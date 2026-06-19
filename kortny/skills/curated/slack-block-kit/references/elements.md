# Slack Block Kit element reference

Deeper reference for the presentation elements. The SKILL.md body covers the
decision rule and shapes; this file holds the platform limits, surface rules,
and the full element catalog for anyone extending the renderer or building Block
Kit directly.

## Hard limits (Slack-enforced; over-limit payloads are rejected)

| Scope | Limit |
|---|---|
| Blocks per message | 50 |
| Blocks per modal / Home tab | 100 |
| Markdown block (cumulative per payload) | 12,000 chars |
| Section text | 3,000 chars |
| Section fields | 10 fields × 2,000 chars |
| Context block | 10 elements |
| Header (plain_text) | 150 chars |
| Table | 100 rows × 20 columns; 10,000 chars/table; 10,000 aggregate/message |
| Button text / value / url | 75 / 2,000 / 3,000 chars |

The deterministic renderer validates against these and degrades gracefully
(drop context → drop secondary parts → truncate table → cards→sections →
prose-only) so a bad layout never drops the answer.

## Surface rules

- **`table` blocks are message-only.** They do not render in modals or the App
  Home tab. Off the message surface the renderer degrades a table to a markdown
  table inside a markdown block.
- **`markdown` block** is Slack's LLM-friendly primitive: it renders ordinary
  markdown (bold, italics, links, lists, code, quotes, dividers, task lists)
  natively, so the conversational voice needs no special handling.

## Element catalog (presentation hint → Slack block)

| Hint element | Renders as | Use for |
|---|---|---|
| `items` | per item: `section`(title+facts) + `context`(meta), divider-separated | the default for an entity list ("list my X") |
| `table` | `table` block (message) / markdown table (modal/Home) | rows that share columns, 3+ rows |
| `fields` | `section` with a fields array | key-value facts / metrics / status about one thing |
| `cards` | `card` blocks (one per item; section+fields fallback if a card overflows) | 2-5 important objects that each deserve a tile (sparingly) |
| `sources` | carousel of source cards / stacked sections, links only | citations — URLs resolved server-side from evidence, never LLM-authored |
| `context` | `context` block | provenance, freshness, source footnotes |
| `header` | `header` block (plain_text) | title a section of a longer answer |
| `divider` | `divider` block | separate sections of a longer answer |

### Source binding (the `sources` element)

The LLM gives only a `source_ref` (e.g. `source:0`) chosen from the
`available_sources` it was shown; the renderer resolves that ref to a
server-built URL from the response record's evidence. The hint may not carry a
URL (the schema forbids stray keys). An unresolved ref is dropped; a `sources`
element with no resolvable refs renders nothing. This is the trust guarantee: a
hallucinated or phishing link can never reach the user.

## Voice vs. data (the product principle)

Conversational voice stays prose; structured data becomes native primitives.
The win is scannability, not decoration. Signs you are over-formatting:

- wrapping a one-line answer in a card,
- a "table" with one row, or a "fields" with one item,
- formatting a greeting, an apology, or a follow-up question,
- adding a card per bullet in a casual list.

When in doubt, prefer prose. One precise element beats a dashboard.

## Interactivity (not in this slice)

Buttons, selects, overflow menus, and modals are deliberately out of scope for
the presentation hint. They carry server state (approval keys, action ids) and
must be built from server-owned records, not LLM-authored values — that arrives
in a later slice. Do not emit interactive elements or links that trigger
actions from the presentation hint.
