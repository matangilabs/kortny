from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

from kortny.db.models import Schedule, Task
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.scheduler import (
    LLMScheduleParser,
    ScheduleCreationContext,
    infer_schedule_delivery,
    looks_like_schedule_request,
    parse_schedule_request,
)
from kortny.scheduler.creation import format_schedule_proposal


class FakeScheduleParserLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] = (),
        response_format: dict[str, Any] | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        self.calls.append(
            {
                "task_id": task_id,
                "messages": messages,
                "tools": tools,
                "response_format": response_format,
                "prompt_name": prompt_name,
                "prompt_source": prompt_source,
            }
        )
        return Completion(
            content=self.content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=20),
            cost_usd=Decimal("0.000001"),
            model="test/model",
        )


def test_parse_weekly_schedule_request_extracts_cron_contract() -> None:
    draft = parse_schedule_request(
        (
            "Every Monday morning, check for unresolved decisions I was involved "
            "in and DM me only if there is something specific."
        ),
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
    )

    assert draft is not None
    assert draft.spec_kind == "cron"
    assert draft.cron_expr == "0 9 * * 1"
    assert draft.cadence_label == "Every Monday morning"
    assert draft.next_run_at == datetime(2026, 6, 8, 9, 0, tzinfo=UTC)
    assert draft.task_input == (
        "check for unresolved decisions I was involved in and DM me only if "
        "there is something specific."
    )
    assert draft.needs_confirmation is False


def test_parse_explicit_draft_schedule_still_extracts_schedule_shape() -> None:
    draft = parse_schedule_request(
        "Draft a schedule for every Friday morning to check PYPL.",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
    )

    assert draft is not None
    assert draft.spec_kind == "cron"
    assert draft.cron_expr == "0 9 * * 5"
    assert draft.needs_confirmation is False


def test_parse_every_morning_preserves_following_task_words() -> None:
    draft = parse_schedule_request(
        "Every morning can you check on PYPL ticker and give me a market summary",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
    )

    assert draft is not None
    assert draft.spec_kind == "cron"
    assert draft.cron_expr == "0 9 * * *"
    assert draft.cadence_label == "Every morning"
    assert (
        draft.task_input == "can you check on PYPL ticker and give me a market summary"
    )


def test_parse_daily_schedule_with_explicit_central_time_humanizes_response() -> None:
    draft = parse_schedule_request(
        "Every morning at 8AM central time I want a stock market update.",
        now=datetime(2026, 6, 4, 19, 29, tzinfo=UTC),
    )

    assert draft is not None
    assert draft.spec_kind == "cron"
    assert draft.cron_expr == "0 8 * * *"
    assert draft.timezone == "America/Chicago"
    assert draft.next_run_at == datetime(2026, 6, 5, 13, 0, tzinfo=UTC)
    assert draft.cadence_label == "Every morning at 8:00 AM Central time"
    assert draft.task_input == "send a stock market update."

    response = format_schedule_proposal(
        schedule=cast(Schedule, None),
        draft=draft,
        delivery_surface="dm",
        needs_confirmation=False,
        now=datetime(2026, 6, 4, 19, 29, tzinfo=UTC),
    )

    assert (
        "I'll send a stock market update every morning at 8:00 AM Central time"
        in response
    )
    assert "First check is tomorrow at 8:00 AM Central time" in response
    assert "I'll at" not in response


def test_infer_schedule_delivery_from_surface_and_text() -> None:
    source_task_id = uuid.uuid4()
    dm_context = ScheduleCreationContext(
        installation_id=uuid.uuid4(),
        slack_channel_id="D123",
        slack_user_id="U123",
        slack_thread_ts="D123",
        source_surface="dm",
        source_task_id=source_task_id,
    )
    channel_context = ScheduleCreationContext(
        installation_id=uuid.uuid4(),
        slack_channel_id="C123",
        slack_user_id="U123",
        slack_thread_ts="1716400000.000001",
        source_surface="app_mention",
        source_task_id=source_task_id,
    )

    dm_delivery = infer_schedule_delivery(context=dm_context, text="Every morning")
    thread_delivery = infer_schedule_delivery(
        context=channel_context,
        text="Every morning check the market",
    )
    channel_delivery = infer_schedule_delivery(
        context=channel_context,
        text="Every morning post the market update in this channel and attach files",
    )

    assert dm_delivery.kind == "slack_dm"
    assert dm_delivery.response_label == "in this DM"
    assert thread_delivery.kind == "slack_thread"
    assert thread_delivery.slack_thread_ts == "1716400000.000001"
    assert channel_delivery.kind == "slack_channel"
    assert channel_delivery.slack_thread_ts is None
    assert channel_delivery.artifact_policy == "attach_files"


def test_schedule_detector_ignores_plain_work_requests() -> None:
    assert looks_like_schedule_request("summarize this channel") is False
    assert looks_like_schedule_request("Every Friday, summarize this channel") is True
    assert (
        looks_like_schedule_request("Weekdays at 8 AM send me a market update") is True
    )


def test_llm_schedule_parser_accepts_valid_weekday_cron() -> None:
    llm = FakeScheduleParserLLM(
        """
        {
          "is_schedule": true,
          "schedule_kind": "cron",
          "cron_expr": "0 8 * * 1-5",
          "interval_seconds": null,
          "run_at": null,
          "timezone": "America/Chicago",
          "cadence_label": "Every weekday at 8:00 AM Central time",
          "task_input": "send a stock market update",
          "needs_confirmation": false,
          "confidence": 0.94,
          "clarifying_question": null
        }
        """
    )
    context = ScheduleCreationContext(
        installation_id=uuid.uuid4(),
        slack_channel_id="D123",
        slack_user_id="U123",
        slack_thread_ts="D123",
        source_surface="dm",
        source_task_id=uuid.uuid4(),
    )

    draft = LLMScheduleParser(llm=llm).parse(
        task=cast(Task, SimpleNamespace(id=uuid.uuid4())),
        context=context,
        text="Every weekday at 8 AM central time send me a stock market update",
        now=datetime(2026, 6, 4, 20, 0, tzinfo=UTC),
    )

    assert draft is not None
    assert draft.spec_kind == "cron"
    assert draft.cron_expr == "0 8 * * 1-5"
    assert draft.timezone == "America/Chicago"
    assert draft.next_run_at == datetime(2026, 6, 5, 13, 0, tzinfo=UTC)
    assert draft.task_input == "send a stock market update"
    assert draft.parse_strategy == "llm_schedule_parser"
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
    assert llm.calls[0]["prompt_name"] == "kortny.schedule_parser"


def test_llm_schedule_parser_rejects_low_confidence_payload() -> None:
    llm = FakeScheduleParserLLM(
        """
        {
          "is_schedule": true,
          "schedule_kind": "unsupported",
          "cron_expr": null,
          "interval_seconds": null,
          "run_at": null,
          "timezone": "UTC",
          "cadence_label": "",
          "task_input": "send a stock market update",
          "needs_confirmation": false,
          "confidence": 0.4,
          "clarifying_question": "Which weekday?"
        }
        """
    )
    context = ScheduleCreationContext(
        installation_id=uuid.uuid4(),
        slack_channel_id="D123",
        slack_user_id="U123",
        slack_thread_ts="D123",
        source_surface="dm",
        source_task_id=uuid.uuid4(),
    )

    draft = LLMScheduleParser(llm=llm).parse(
        task=cast(Task, SimpleNamespace(id=uuid.uuid4())),
        context=context,
        text="check this sometimes",
        now=datetime(2026, 6, 4, 20, 0, tzinfo=UTC),
    )

    assert draft is None
