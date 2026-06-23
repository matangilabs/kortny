"""Labeled cases for the response-pipeline substance-drop eval (HIG-287).

Each case captures:
- the raw answer the agent produced
- what the humanizer put in `message`
- how many presentation elements the LLM returned
- what render_blocks produced (pre-computed; None means blocks=0)
- tokens that MUST appear in the final posted text for the answer to be intact
- whether the substance-drop guard is expected to fire

Expand from real agent outputs over time; this seed set is the floor that
_is_substance_dropped_prerender() and the posting pipeline are held to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResponsePipelineCase:
    name: str
    raw_answer: str
    humanized_text: str
    presentation_element_count: int
    rendered_blocks: list[dict] | None
    key_tokens: list[str]
    expects_guard_trigger: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SKILLS_RAW = (
    "Here are the skills I currently have access to:\n\n"
    "**Research & Analysis**\n"
    "- `web_search` — search the web for current information\n"
    "- `notion_search` — search your connected Notion workspace\n"
    "- `document_analysis` — analyse uploaded documents (PDF, DOCX, images)\n\n"
    "**Communication & Scheduling**\n"
    "- `send_email` — compose and send emails via connected Gmail or Outlook\n"
    "- `calendar_check` — query your Google or Outlook calendar for free/busy\n"
    "- `schedule_meeting` — create calendar events with invites\n\n"
    "**Project & Task Management**\n"
    "- `linear_create_issue` — create issues in your Linear workspace\n"
    "- `linear_search` — search Linear projects, cycles, and issues\n"
    "- `github_search_code` — search code in connected GitHub repositories\n"
    "- `github_create_pr` — open a pull request on your behalf\n\n"
    "**Data & Code**\n"
    "- `code_exec` — run Python in a sandboxed environment\n"
    "- `sql_query` — query connected databases via natural language\n"
    "- `csv_parse` — analyse CSV files you upload\n\n"
    "**Knowledge & Memory**\n"
    "- `memory_recall` — retrieve facts and episodes from Kortny's memory\n"
    "- `memory_save` — store a new fact for future recall\n"
    "- `knowledge_graph_query` — query the team knowledge graph\n\n"
    "**Utilities**\n"
    "- `summarise` — condense long content (docs, threads, web pages)\n"
    "- `translate` — translate text between languages\n"
    "- `image_describe` — describe or extract text from images\n\n"
    "You can ask me to use any of these directly, or just describe your goal "
    "and I'll pick the right tool for the job."
)

_LONG_LIST_RAW = (
    "Current sprint items:\n\n"
    "1. Refactor authentication middleware — assigned to @alice, due Friday\n"
    "2. Add CSV export to the reporting dashboard — assigned to @bob, due Monday\n"
    "3. Fix the race condition in the job queue — assigned to @carol, due Wednesday\n"
    "4. Write integration tests for the Slack webhook handler — unassigned, blocked\n"
    "5. Update the deployment runbook after the K8s migration — assigned to @dave\n"
    "6. Audit MCP server permissions for prod — in review\n"
    "7. Backfill missing OpenTelemetry spans in the worker loop — candidate\n"
    "8. Upgrade LiteLLM to 1.47 and run regression tests — assigned to @alice\n"
)

_METRICS_RAW = (
    "System metrics for the past 24 hours:\n\n"
    "- P50 latency: 142ms\n"
    "- P95 latency: 891ms\n"
    "- P99 latency: 2,340ms\n"
    "- Error rate: 0.12%\n"
    "- Throughput: 4,821 req/min\n"
    "- Active worker processes: 8\n"
    "- Queue depth (current): 14\n"
    "- Queue depth (peak): 203\n"
    "- Memory usage: 61% of 16 GB\n"
    "- CPU usage: 34% average across pods\n"
)

_MULTISTEP_RAW = (
    "I completed the following steps:\n\n"
    "1. Searched Notion for 'Q2 pipeline' — found 3 candidate pages.\n"
    "2. Opened each page and checked the last-edited date — two were stale "
    "(edited > 90 days ago), one was edited last Tuesday.\n"
    "3. Read the current page: it contains the Q2 pipeline numbers for "
    "Enterprise and SMB segments but the SMB row is missing May data.\n"
    "4. Created a Linear issue HIG-999 to track the missing row.\n"
    "5. Posted a summary to #finance-ops with a link to the Notion page.\n"
)

# A pre-computed valid fields block (simulates a successfully rendered element).
_FIELDS_BLOCK: dict = {
    "type": "section",
    "fields": [
        {"type": "mrkdwn", "text": "*Status*\nActive"},
    ],
}

# ---------------------------------------------------------------------------
# Seed cases
# ---------------------------------------------------------------------------

SEED_RESPONSE_PIPELINE_CASES: tuple[ResponsePipelineCase, ...] = (
    # 1. The exact bug: skills list, intro-only message, 1 element, 0 blocks.
    ResponsePipelineCase(
        name="skills_list",
        raw_answer=_SKILLS_RAW,
        humanized_text="Here's what I can do for you:",
        presentation_element_count=1,
        rendered_blocks=None,
        key_tokens=["web_search", "code_exec", "memory_recall"],
        expects_guard_trigger=True,
        notes=(
            "Exact reproduction of the HIG-287 bug: gemini-3.1-flash-lite put "
            "the skills list only in presentation, the message was a 107-char "
            "intro. Guard must fire and replace text with the raw answer."
        ),
    ),
    # 2. Plain short answer — no drop possible.
    ResponsePipelineCase(
        name="plain_answer",
        raw_answer="The meeting is at 3pm.",
        humanized_text="The meeting is at 3pm.",
        presentation_element_count=0,
        rendered_blocks=None,
        key_tokens=["3pm"],
        expects_guard_trigger=False,
        notes="Short raw answer under 200 chars; guard must never fire.",
    ),
    # 3. Long list with a summary humanized text and valid rendered blocks.
    ResponsePipelineCase(
        name="long_list_with_valid_blocks",
        raw_answer=_LONG_LIST_RAW,
        humanized_text=(
            "Here are the 8 active sprint items. Highlights: 3 due this week, "
            "1 blocked waiting for assignment, 1 currently in review."
        ),
        presentation_element_count=1,
        rendered_blocks=[_FIELDS_BLOCK],
        key_tokens=["sprint"],
        expects_guard_trigger=False,
        notes=(
            "Summary is >= 40% of raw chars and blocks rendered successfully. "
            "Guard must not fire."
        ),
    ),
    # 4. Key-value metrics with full prose humanization — no presentation.
    ResponsePipelineCase(
        name="key_value_metrics",
        raw_answer=_METRICS_RAW,
        humanized_text=(
            "System is healthy over the last 24h. P95 latency is 891ms, error "
            "rate 0.12%, throughput 4,821 req/min. Queue peaked at 203 but is "
            "down to 14 now. Memory at 61%, CPU at 34%."
        ),
        presentation_element_count=0,
        rendered_blocks=None,
        key_tokens=["891ms", "0.12%", "4,821"],
        expects_guard_trigger=False,
        notes=(
            "Humanized prose covers the key numbers; no presentation. "
            "Guard must not fire."
        ),
    ),
    # 5. One-line answer — raw is trivially short.
    ResponsePipelineCase(
        name="one_line_answer",
        raw_answer="Done.",
        humanized_text="Done.",
        presentation_element_count=0,
        rendered_blocks=None,
        key_tokens=["Done"],
        expects_guard_trigger=False,
        notes="Raw under 200 chars; guard must not fire regardless of text shape.",
    ),
    # 6. Multi-step recap: intro-only message, 1 element, 0 blocks — guard fires.
    ResponsePipelineCase(
        name="intro_only_with_no_blocks",
        raw_answer=_MULTISTEP_RAW,
        humanized_text="Here's a summary:",
        presentation_element_count=1,
        rendered_blocks=None,
        key_tokens=["Notion", "Linear", "SMB"],
        expects_guard_trigger=True,
        notes=(
            "Lead-in preamble 'Here's a summary:' with 1 presentation element "
            "and 0 rendered blocks. Guard must fire and fall back to raw answer."
        ),
    ),
)
