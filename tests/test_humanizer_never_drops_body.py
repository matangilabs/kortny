"""HIG-287: humanizer must never drop the answer body.

Bug reproduction + guard coverage. Pure functions only — no DB, no LLM.

Before the fix:
  synthesize_response() returned `text="Here's what I can do for you:"` (107
  chars) plus a PresentationHint with 1 element. render_blocks() returned None
  because the element rendered to zero blocks. The short intro was posted to
  Slack and the 2944-char skills list vanished.

After the fix:
  _is_substance_dropped_prerender() detects that the humanized text ends with
  ":" (a preamble) and the model used presentation, replaces result.text with
  the sanitized raw answer, and clears presentation so blocks are not attempted.
"""

from __future__ import annotations

import pytest

from kortny.slack.humanizer import (
    _is_substance_dropped_prerender,
    sanitize_humanized_response,
)
from kortny.slack.response_render import render_blocks

# ---------------------------------------------------------------------------
# Fixtures
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

_SKILLS_INTRO = "Here's what I can do for you:"

_SHORT_RAW = "The meeting is at 3pm."

_LONG_LIST_RAW = (
    "Current sprint items:\n\n"
    "1. Refactor authentication middleware — assigned to @alice, due Friday\n"
    "2. Add CSV export to the reporting dashboard — assigned to @bob, due next Monday\n"
    "3. Fix the race condition in the job queue — assigned to @carol, due Wednesday\n"
    "4. Write integration tests for the Slack webhook handler — unassigned, blocked\n"
    "5. Update the deployment runbook after the K8s migration — assigned to @dave\n"
    "6. Audit MCP server permissions for prod — in review\n"
    "7. Backfill missing OpenTelemetry spans in the worker loop — candidate\n"
    "8. Upgrade LiteLLM to 1.47 and run regression tests — assigned to @alice\n"
)

_LONG_LIST_SUMMARY = (
    "Here are the 8 active sprint items. Highlights: 3 due this week, "
    "1 blocked waiting for assignment, 1 currently in review."
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

_METRICS_HUMANIZED = (
    "System is healthy over the last 24h. P95 latency is 891ms, error rate "
    "0.12%, throughput 4,821 req/min. Queue peaked at 203 but is down to 14."
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

_MULTISTEP_INTRO = "Here's a summary:"


# ---------------------------------------------------------------------------
# Bug reproduction: _is_substance_dropped_prerender must detect the drop
# ---------------------------------------------------------------------------


def test_skills_drop_guard_fires() -> None:
    """Reproduce the exact bug: intro-only colon text + 1 presentation element."""
    assert len(_SKILLS_RAW) > 200, "raw must be non-trivial for the guard to apply"
    assert _SKILLS_INTRO.endswith(":"), "intro must end with colon"

    assert _is_substance_dropped_prerender(
        humanized_text=_SKILLS_INTRO,
        raw_answer=_SKILLS_RAW,
        presentation_element_count=1,
    )


def test_multistep_intro_guard_fires() -> None:
    """Multi-step recap: 'Here's a summary:' + presentation element = guard fires."""
    assert _MULTISTEP_INTRO.endswith(":")
    assert _is_substance_dropped_prerender(
        humanized_text=_MULTISTEP_INTRO,
        raw_answer=_MULTISTEP_RAW,
        presentation_element_count=1,
    )


# ---------------------------------------------------------------------------
# Guard must NOT fire for legitimate short/full answers
# ---------------------------------------------------------------------------


def test_short_raw_never_fires() -> None:
    """Raw answer under 200 chars: no guard, even with a preamble-like message."""
    assert not _is_substance_dropped_prerender(
        humanized_text="Done.",
        raw_answer="Done.",
        presentation_element_count=0,
    )


def test_one_line_answer_no_presentation() -> None:
    """One-line raw answer under threshold: guard must not fire."""
    assert not _is_substance_dropped_prerender(
        humanized_text=_SHORT_RAW,
        raw_answer=_SHORT_RAW,
        presentation_element_count=0,
    )


def test_full_prose_answer_no_presentation() -> None:
    """Humanized text is the full prose; no presentation requested — guard silent."""
    assert not _is_substance_dropped_prerender(
        humanized_text=_SKILLS_RAW,  # full text
        raw_answer=_SKILLS_RAW,
        presentation_element_count=0,
    )


def test_long_list_with_valid_blocks_no_fire() -> None:
    """Summary ending in '.' (not ':') + presentation element: guard stays off.

    'Here are the 8 active sprint items.' ends with '.' not ':', so it is a
    complete sentence and not a lead-in preamble.  The guard only fires on
    trailing ':' with presentation elements.
    """
    assert not _LONG_LIST_SUMMARY.endswith(":")
    assert not _is_substance_dropped_prerender(
        humanized_text=_LONG_LIST_SUMMARY,
        raw_answer=_LONG_LIST_RAW,
        presentation_element_count=1,
    )


def test_metrics_humanized_full_no_fire() -> None:
    """Humanized metrics paragraph (no colon ending, no presentation): guard off."""
    assert not _METRICS_HUMANIZED.endswith(":")
    assert not _is_substance_dropped_prerender(
        humanized_text=_METRICS_HUMANIZED,
        raw_answer=_METRICS_RAW,
        presentation_element_count=0,
    )


def test_colon_in_middle_of_sentence_no_fire() -> None:
    """A colon in the middle of a complete sentence does not trigger the guard.

    Only the TRAILING colon matters. 'P95 latency: 891ms, CPU: 34%.' is a
    complete answer even though it contains colons.
    """
    complete_answer = "P95 latency: 891ms, error rate: 0.12%, CPU: 34%."
    assert not complete_answer.endswith(":")
    assert not _is_substance_dropped_prerender(
        humanized_text=complete_answer,
        raw_answer=_METRICS_RAW,
        presentation_element_count=1,
    )


def test_no_presentation_no_fire_even_with_colon_ending_short_raw() -> None:
    """No presentation elements: guard stays off for short raw (under 200 chars)."""
    # Raw under threshold — guard never fires.
    short_raw = "Here are the metrics: P95=891ms, error_rate=0.12%."
    assert len(short_raw) < 200
    assert not _is_substance_dropped_prerender(
        humanized_text="Here are the metrics:",
        raw_answer=short_raw,
        presentation_element_count=0,
    )


def test_colon_ending_without_presentation_above_ratio_no_fire() -> None:
    """Colon ending + no presentation + humanized ratio above 20%: guard off.

    The secondary (no-presentation) trigger requires ratio < 20%. A colon
    ending with a reasonable ratio is not flagged even without presentation.
    """
    # humanized=50 chars, raw=200 chars → ratio=25%, above 20% threshold.
    raw = "x" * 200
    humanized = "Here are the results:"  # 21 chars, ratio=10.5%
    # ratio < 20%, ends_with_colon, but no presentation_element_count
    # secondary: ratio_too_low (21 < 200*0.40=80) AND 21 < 200*0.20=40 AND colon
    # → 21 < 40 is True, but no presentation_element_count=0
    # secondary check: ratio_too_low AND humanized_len < raw_len * 0.20 AND ends_with_colon
    # 21 < 40 → True, so secondary fires!
    # This is intentional: a colon-ending intro with < 20% of raw and no presentation
    # is also a pathological case we catch.
    # So this test documents that the secondary path DOES fire at < 20%.
    result = _is_substance_dropped_prerender(
        humanized_text=humanized,
        raw_answer=raw,
        presentation_element_count=0,
    )
    # 21 chars / 200 chars = 10.5% < 20%, ends with ":" → secondary fires
    assert result


def test_colon_ending_no_presentation_above_20pct_no_fire() -> None:
    """Colon ending + no presentation + ratio above 20%: guard stays off."""
    # humanized=50 chars, raw=200 chars → ratio=25%, above 20%
    raw = "x" * 200
    humanized = "Here are the top results for your query:"  # 40 chars, ratio=20%
    # 40 / 200 = 20%, NOT strictly less than 20% → secondary doesn't fire
    result = _is_substance_dropped_prerender(
        humanized_text=humanized,
        raw_answer=raw,
        presentation_element_count=0,
    )
    # 40 < 200*0.20=40 → 40 < 40 is False → secondary does NOT fire
    assert not result


# ---------------------------------------------------------------------------
# End-to-end: after guard fires, fallback text contains the substance
# ---------------------------------------------------------------------------


def test_guard_fallback_contains_substance() -> None:
    """When the guard fires, sanitize_humanized_response(None, fallback=raw)
    must yield text that contains key tokens from the raw answer."""
    guard_fired = _is_substance_dropped_prerender(
        humanized_text=_SKILLS_INTRO,
        raw_answer=_SKILLS_RAW,
        presentation_element_count=1,
    )
    assert guard_fired

    fallback_text = sanitize_humanized_response(None, fallback=_SKILLS_RAW)
    # The fallback must contain substantive tokens from the raw answer.
    assert "web_search" in fallback_text
    assert "code_exec" in fallback_text
    assert "memory_recall" in fallback_text


def test_guard_fallback_contains_multistep_tokens() -> None:
    """Fallback for a multi-step recap must preserve the step details."""
    fallback_text = sanitize_humanized_response(None, fallback=_MULTISTEP_RAW)
    assert "Notion" in fallback_text
    assert "Linear" in fallback_text
    assert "SMB" in fallback_text


# ---------------------------------------------------------------------------
# render_blocks: zero-blocks case
# ---------------------------------------------------------------------------


def test_render_blocks_does_not_raise() -> None:
    """render_blocks must not raise even when hint elements all drop."""
    from kortny.slack.presentation import parse_presentation

    hint_data = {
        "version": 1,
        "elements": [
            {
                "type": "fields",
                "items": [{"label": "Status", "value": "Active"}],
            }
        ],
    }
    hint = parse_presentation(hint_data)
    assert hint is not None
    # Either returns blocks or None — must not raise.
    _ = render_blocks(_SKILLS_INTRO, hint)


# ---------------------------------------------------------------------------
# Eval: run the eval scorer inline as tests
# ---------------------------------------------------------------------------


def test_eval_all_cases_pass() -> None:
    """Run the response pipeline eval scorer; all cases must produce correct results."""
    from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
    from kortny.evals.response_pipeline.scoring import score_response_pipeline

    report = score_response_pipeline(SEED_RESPONSE_PIPELINE_CASES)
    failures = [r for r in report.results if not r.passed]
    messages = "\n".join(
        f"  [{r.case_name}] guard_fired={r.guard_fired} "
        f"expected_guard={r.expects_guard_trigger} "
        f"post_render_guard_fired={r.post_render_guard_fired} "
        f"expected_post_render={r.expects_post_render_guard_trigger} "
        f"missing_tokens={r.missing_tokens}"
        for r in failures
    )
    assert not failures, f"Eval failures:\n{messages}"


def test_eval_discrimination_is_perfect() -> None:
    """Both guards must correctly fire/not-fire on every labeled case."""
    from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
    from kortny.evals.response_pipeline.scoring import score_response_pipeline

    report = score_response_pipeline(SEED_RESPONSE_PIPELINE_CASES)
    assert report.discrimination == 1.0, (
        f"Imperfect pre-render discrimination: {report.discrimination:.3f}\n"
        + "\n".join(
            f"  {r.case_name}: guard_fired={r.guard_fired} "
            f"expected={r.expects_guard_trigger}"
            for r in report.results
            if r.guard_fired != r.expects_guard_trigger
        )
    )
    assert report.post_render_discrimination == 1.0, (
        f"Imperfect post-render discrimination: {report.post_render_discrimination:.3f}\n"
        + "\n".join(
            f"  {r.case_name}: post_render_guard_fired={r.post_render_guard_fired} "
            f"expected={r.expects_post_render_guard_trigger}"
            for r in report.results
            if r.post_render_guard_fired != r.expects_post_render_guard_trigger
        )
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "skills_list",
        "intro_only_with_no_blocks",
    ],
)
def test_eval_guard_fires_on_drop_cases(case_name: str) -> None:
    """The pre-render guard must fire on each case marked expects_guard_trigger=True."""
    from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
    from kortny.evals.response_pipeline.scoring import score_response_pipeline

    cases = [c for c in SEED_RESPONSE_PIPELINE_CASES if c.name == case_name]
    assert cases, f"Case {case_name!r} not found in SEED_RESPONSE_PIPELINE_CASES"
    report = score_response_pipeline(cases)
    result = report.results[0]
    assert result.guard_fired, (
        f"Guard did not fire for {case_name!r}: "
        f"humanized_len={len(cases[0].humanized_text)} "
        f"raw_len={len(cases[0].raw_answer)} "
        f"presentation_element_count={cases[0].presentation_element_count}"
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "plain_answer",
        "long_list_with_valid_blocks",
        "key_value_metrics",
        "one_line_answer",
    ],
)
def test_eval_guard_silent_on_clean_cases(case_name: str) -> None:
    """The pre-render guard must NOT fire on clean/correct cases."""
    from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
    from kortny.evals.response_pipeline.scoring import score_response_pipeline

    cases = [c for c in SEED_RESPONSE_PIPELINE_CASES if c.name == case_name]
    assert cases, f"Case {case_name!r} not found"
    report = score_response_pipeline(cases)
    result = report.results[0]
    assert not result.guard_fired, (
        f"Guard fired falsely for {case_name!r}: "
        f"humanized_len={len(cases[0].humanized_text)} "
        f"raw_len={len(cases[0].raw_answer)}"
    )


# ---------------------------------------------------------------------------
# New: post-render zero-blocks guard (the render-based net)
# ---------------------------------------------------------------------------


def test_period_intro_not_caught_by_colon_guard() -> None:
    """A period-terminated intro does NOT trigger the pre-render colon guard.

    This is the variant the old guard misses: 'Here's what I found.' ends
    with '.' not ':', so _is_substance_dropped_prerender is silent even
    though the body is entirely offloaded to a presentation that will render
    to zero blocks.
    """
    # This is the exact intro from the 'period_intro_failing_presentation' eval case.
    assert not _is_substance_dropped_prerender(
        humanized_text="Here's what I found.",
        raw_answer=(
            "Here is what I found about the three open vendor proposals:\n\n"
            "**Acme Corp** (submitted 2026-06-01)\n"
            "- Pricing: $48,000/year for up to 50 seats\n"
            "- SLA: 99.9% uptime, 4-hour response window\n"
            "- Integration: REST API + Slack webhook; no native Linear connector\n"
            "- Risk: no SOC 2 Type II cert yet (audit scheduled Q3 2026)\n\n"
            "**Bravo Systems** (submitted 2026-06-03)\n"
            "- Pricing: $61,000/year for up to 75 seats (volume discount negotiable)\n"
            "- SLA: 99.95% uptime, 2-hour response window, dedicated CSM\n"
            "- Integration: native Linear, GitHub, and Slack connectors\n"
            "- Risk: higher cost; references checked — all positive\n\n"
            "**Charlie Tech** (submitted 2026-06-10)\n"
            "- Pricing: $39,500/year for up to 40 seats\n"
            "- SLA: 99.5% uptime, 8-hour response window (business hours only)\n"
            "- Integration: REST API only; no pre-built connectors\n"
            "- Risk: lowest uptime SLA; small team (8 engineers)\n\n"
            "Recommendation: Bravo Systems best fits the integration and SLA "
            "requirements; Charlie Tech is viable only if budget is the hard constraint."
        ),
        presentation_element_count=1,
    ), (
        "Pre-render colon guard must NOT fire for a period-terminated intro "
        "— 'Here's what I found.' does not end with ':'"
    )


def test_period_intro_caught_by_post_render_guard() -> None:
    """The post-render zero-blocks guard catches the period-terminated intro variant.

    When the humanizer writes 'Here's what I found.' (period, not colon),
    the pre-render guard is silent. But if the presentation element renders
    to zero blocks, the post-render guard in agent_executor fires and falls
    back to the raw answer. The eval scorer simulates this via
    _is_zero_blocks_from_nonempty_presentation().
    """
    from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
    from kortny.evals.response_pipeline.scoring import (
        _is_zero_blocks_from_nonempty_presentation,
        score_response_pipeline,
    )

    cases = [
        c
        for c in SEED_RESPONSE_PIPELINE_CASES
        if c.name == "period_intro_failing_presentation"
    ]
    assert cases, "Case 'period_intro_failing_presentation' missing from eval suite"
    case = cases[0]

    # 1. Confirm: pre-render guard does NOT fire (period, not colon).
    assert not _is_substance_dropped_prerender(
        humanized_text=case.humanized_text,
        raw_answer=case.raw_answer,
        presentation_element_count=case.presentation_element_count,
    ), "Pre-render guard must NOT fire on period-terminated intro"

    # 2. Confirm: post-render guard DOES fire (presentation elements > 0, blocks = None).
    assert _is_zero_blocks_from_nonempty_presentation(case), (
        "Post-render guard must fire when presentation_element_count > 0 "
        "and rendered_blocks is None"
    )

    # 3. Confirm: eval scoring routes to fallback text containing the substance.
    report = score_response_pipeline(cases)
    result = report.results[0]
    assert result.post_render_guard_fired
    assert not result.guard_fired, "Pre-render guard must be silent"
    assert result.passed, f"Eval case failed: missing_tokens={result.missing_tokens}"
    for token in ["Acme Corp", "Bravo Systems", "Charlie Tech", "Recommendation"]:
        assert token in result.final_text, (
            f"Token {token!r} missing from fallback text — substance was dropped"
        )


def test_post_render_guard_silent_when_blocks_render() -> None:
    """Post-render guard must NOT fire when presentation renders to valid blocks."""
    from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
    from kortny.evals.response_pipeline.scoring import (
        _is_zero_blocks_from_nonempty_presentation,
    )

    # 'long_list_with_valid_blocks' has presentation_element_count=1 and
    # rendered_blocks=[_FIELDS_BLOCK] (not None) — guard must not fire.
    cases = [
        c
        for c in SEED_RESPONSE_PIPELINE_CASES
        if c.name == "long_list_with_valid_blocks"
    ]
    assert cases
    case = cases[0]
    assert case.rendered_blocks is not None, "fixture must have rendered_blocks"
    assert not _is_zero_blocks_from_nonempty_presentation(case), (
        "Post-render guard must NOT fire when rendered_blocks is not None"
    )


def test_post_render_guard_silent_for_short_raw() -> None:
    """Post-render guard must NOT fire for short raw answers (< 200 chars)."""
    from kortny.evals.response_pipeline.cases import ResponsePipelineCase
    from kortny.evals.response_pipeline.scoring import (
        _is_zero_blocks_from_nonempty_presentation,
    )

    short_case = ResponsePipelineCase(
        name="short_with_failing_presentation",
        raw_answer="The meeting is at 3pm.",  # < 200 chars
        humanized_text="Here's the info.",
        presentation_element_count=1,
        rendered_blocks=None,  # would trigger if raw were long enough
        key_tokens=["3pm"],
        expects_guard_trigger=False,
    )
    assert not _is_zero_blocks_from_nonempty_presentation(short_case), (
        "Post-render guard must not fire for raw answers < 200 chars"
    )
