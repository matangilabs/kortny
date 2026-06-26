"""LLM-backed Witness candidate extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from kortny.db.models import ObserveChannelProfile, SlackChannelMembership, Task
from kortny.llm import ChatMessage, LLMService
from kortny.tools.types import JsonObject
from kortny.witness.opportunities import (
    ALLOWED_AUTOMATION_KINDS,
    ALLOWED_CANDIDATE_TYPES,
    WitnessOpportunityCandidateInput,
)

WITNESS_TASK_RESPONSE_EXTRACTOR_PROMPT_NAME = "kortny.witness_task_response_extractor"
WITNESS_TASK_RESPONSE_EXTRACTOR_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
WITNESS_CHANNEL_PROFILE_EXTRACTOR_PROMPT_NAME = (
    "kortny.witness_channel_profile_extractor"
)
WITNESS_CHANNEL_PROFILE_EXTRACTOR_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
MAX_EXTRACTED_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class WitnessTaskResponseExtraction:
    """Structured result from the Witness extractor."""

    candidates: tuple[WitnessOpportunityCandidateInput, ...]
    skipped_reason: str | None
    raw_candidate_count: int


class WitnessTaskResponseExtractor:
    """Ask an LLM whether a completed task contains Witness opportunities."""

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def extract(
        self,
        *,
        task: Task,
        response_text: str,
    ) -> WitnessTaskResponseExtraction:
        completion = self.llm.complete(
            task_id=task.id,
            messages=_task_response_messages(task=task, response_text=response_text),
            response_format=WITNESS_TASK_RESPONSE_EXTRACTOR_RESPONSE_FORMAT,
            prompt_name=WITNESS_TASK_RESPONSE_EXTRACTOR_PROMPT_NAME,
        )
        return parse_witness_task_response_extraction(completion.content)


class WitnessChannelProfileExtractor:
    """Ask an LLM whether a channel profile contains Witness opportunities."""

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def extract(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
    ) -> WitnessTaskResponseExtraction:
        completion = self.llm.complete(
            task_id=task.id,
            messages=_channel_profile_messages(
                task=task,
                membership=membership,
                profile=profile,
            ),
            response_format=WITNESS_CHANNEL_PROFILE_EXTRACTOR_RESPONSE_FORMAT,
            prompt_name=WITNESS_CHANNEL_PROFILE_EXTRACTOR_PROMPT_NAME,
        )
        return parse_witness_channel_profile_extraction(completion.content)


def parse_witness_task_response_extraction(
    content: str | None,
) -> WitnessTaskResponseExtraction:
    """Parse and validate model output from the Witness extractor."""

    return _parse_witness_extraction(
        content,
        extractor_prompt_name=WITNESS_TASK_RESPONSE_EXTRACTOR_PROMPT_NAME,
    )


def parse_witness_channel_profile_extraction(
    content: str | None,
) -> WitnessTaskResponseExtraction:
    """Parse and validate model output from the channel profile extractor."""

    return _parse_witness_extraction(
        content,
        extractor_prompt_name=WITNESS_CHANNEL_PROFILE_EXTRACTOR_PROMPT_NAME,
    )


def _parse_witness_extraction(
    content: str | None,
    *,
    extractor_prompt_name: str,
) -> WitnessTaskResponseExtraction:
    """Parse and validate model output from a Witness extractor."""

    if not content:
        return WitnessTaskResponseExtraction(
            candidates=(),
            skipped_reason="empty_model_output",
            raw_candidate_count=0,
        )
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError):
        return WitnessTaskResponseExtraction(
            candidates=(),
            skipped_reason="invalid_json",
            raw_candidate_count=0,
        )
    if not isinstance(payload, dict):
        return WitnessTaskResponseExtraction(
            candidates=(),
            skipped_reason="invalid_payload",
            raw_candidate_count=0,
        )
    skipped_reason = _optional_text(payload.get("skipped_reason"), max_chars=160)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return WitnessTaskResponseExtraction(
            candidates=(),
            skipped_reason=skipped_reason or "missing_candidates",
            raw_candidate_count=0,
        )

    candidates: list[WitnessOpportunityCandidateInput] = []
    for raw_candidate in raw_candidates:
        candidate = _candidate_from_payload(
            raw_candidate,
            extractor_prompt_name=extractor_prompt_name,
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        if len(candidates) >= MAX_EXTRACTED_CANDIDATES:
            break

    return WitnessTaskResponseExtraction(
        candidates=tuple(candidates),
        skipped_reason=None if candidates else skipped_reason or "no_valid_candidates",
        raw_candidate_count=len(raw_candidates),
    )


def _task_response_messages(
    *,
    task: Task,
    response_text: str,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            content=(
                "You are Kortny's Witness extractor. Kortny is an AI coworker in "
                "Slack. Decide whether a completed Slack answer contains future "
                "things Kortny should watch for, proactively help with, or remember "
                "as candidate opportunities. Use semantic judgment; do not require "
                "specific headings or phrases. "
                "Return only the JSON object — no prose, markdown, or comments. "
                'Schema: {"candidates":[{"candidate_type":"workflow_gap|'
                "artifact_followup|unresolved_decision|data_quality_issue|"
                'recurring_check|project_status_gap|general_help",'
                '"title":"short title","summary":"what Kortny should watch '
                'for or help with","suggested_action":"operator-facing action",'
                '"suggested_message":"low-pressure Slack DM or channel suggestion",'
                '"automation_kind":"recurring|one_shot|watch",'
                '"cadence_suggestion":"natural language cadence or empty",'
                '"deliverable":"concrete recurring output, e.g. \'post a trading '
                "summary in this channel'\","
                '"evidence":["short evidence from the answer or request"],'
                '"confidence_score":0.0,"confidence_reason":"why"}],'
                '"skipped_reason":"only when no candidates"}. '
                "Frame opportunities as concrete deliverables with a cadence when "
                "the evidence suggests recurrence; default automation_kind to "
                "watch when unsure. "
                "Only create candidates that would make Kortny more useful later. "
                "Return no candidates (set skipped_reason) for routine greetings, "
                "generic answers, or claims without evidence. "
                "Extract ONLY opportunities grounded in the provided answer and "
                "request; never invent facts, user ids, or context not present in "
                "the input. confidence_score must be 0.0..1.0. "
                "Examples: "
                '{"slack_surface":"channel","kortny_response":"I pulled the weekly P&L report. Revenue is up 12% WoW.","user_request":"show me the P&L","max_candidates":5} '
                '-> {"candidates":[{"candidate_type":"recurring_check","title":"Weekly P&L report","summary":"User requests P&L updates; could automate weekly delivery.","suggested_action":"Schedule a weekly P&L summary post","suggested_message":"Want me to post the P&L summary here every Monday?","automation_kind":"recurring","cadence_suggestion":"weekly on Monday","deliverable":"Weekly P&L summary in channel","evidence":["pulled the weekly P&L report","Revenue is up 12% WoW"],"confidence_score":0.80,"confidence_reason":"Explicit recurring report pattern."}]} '
                '{"slack_surface":"channel","kortny_response":"You\'re welcome!","user_request":"thanks","max_candidates":5} '
                '-> {"candidates":[],"skipped_reason":"routine_greeting"} '
                "Ground every field in the input; abstain when unsupported."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                {
                    "slack_surface": (
                        "dm"
                        if task.slack_channel_id
                        and task.slack_channel_id.startswith("D")
                        else "channel"
                    ),
                    "channel_id": task.slack_channel_id,
                    "user_id": task.slack_user_id,
                    "user_request": task.input,
                    "kortny_response": response_text,
                    "allowed_candidate_types": sorted(ALLOWED_CANDIDATE_TYPES),
                    "max_candidates": MAX_EXTRACTED_CANDIDATES,
                },
                sort_keys=True,
            ),
        ),
    )


def _channel_profile_messages(
    *,
    task: Task,
    membership: SlackChannelMembership,
    profile: ObserveChannelProfile,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            content=(
                "You are Kortny's Witness extractor for channel assessments. "
                "Kortny is an AI coworker in Slack. Decide whether a saved channel "
                "profile contains future things Kortny should watch for, "
                "proactively help with, or remember as candidate opportunities. "
                "Use semantic judgment from the provided evidence; do not depend "
                "on headings, regexes, or fixed phrases. Do not infer private DMs "
                "or cross-channel facts that are not in the payload. "
                "Return only the JSON object — no prose, markdown, or comments. "
                'Schema: {"candidates":[{"candidate_type":"'
                "workflow_gap|artifact_followup|unresolved_decision|"
                "data_quality_issue|recurring_check|project_status_gap|"
                'general_help","title":"short title","summary":"what '
                'Kortny should watch for or help with","suggested_action":'
                '"operator-facing action","suggested_message":"low-pressure '
                'Slack DM or channel suggestion",'
                '"automation_kind":"recurring|one_shot|watch",'
                '"cadence_suggestion":"natural language cadence or empty",'
                '"deliverable":"concrete recurring output, e.g. \'post a trading '
                "summary in this channel'\","
                '"evidence":["short evidence '
                'from the profile"],"confidence_score":0.0,'
                '"confidence_reason":"why"}],"skipped_reason":"only when '
                'no candidates"}. Frame opportunities as concrete deliverables '
                "with a cadence when the evidence suggests recurrence; default "
                "automation_kind to watch when unsure. "
                "Only create candidates that would make Kortny "
                "more useful later. Return no candidates (set skipped_reason) "
                "when the profile is too thin, too speculative, or lacks "
                "actionable future help. "
                "Extract ONLY opportunities grounded in the provided profile; "
                "never invent facts, channel ids, or context not present in "
                "the input. confidence_score must be 0.0..1.0. "
                "Examples: "
                '{"channel_id":"C1","profile":{"summary":"Daily trading ops channel. Team posts P&L reports every morning and reviews open positions.","message_count":240},"max_candidates":5} '
                '-> {"candidates":[{"candidate_type":"recurring_check","title":"Daily P&L summary","summary":"Team posts P&L every morning; Kortny could automate or summarize.","suggested_action":"Post a morning P&L digest in #trading-ops","suggested_message":"I can post a daily P&L digest here each morning. Want me to set that up?","automation_kind":"recurring","cadence_suggestion":"daily at market open","deliverable":"Morning P&L summary post in channel","evidence":["Team posts P&L reports every morning","reviews open positions"],"confidence_score":0.83,"confidence_reason":"Clear daily recurring pattern in profile."}]} '
                '{"channel_id":"C2","profile":{"summary":"","message_count":2},"max_candidates":5} '
                '-> {"candidates":[],"skipped_reason":"profile_too_thin"} '
                "Ground every field in the input; abstain when unsupported."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                _channel_profile_payload(
                    task=task,
                    membership=membership,
                    profile=profile,
                ),
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )


def _channel_profile_payload(
    *,
    task: Task,
    membership: SlackChannelMembership,
    profile: ObserveChannelProfile,
) -> JsonObject:
    return {
        "slack_surface": "channel",
        "channel_id": membership.channel_id,
        "channel_name": membership.channel_name,
        "channel_type": membership.channel_type,
        "added_by_user_id": membership.added_by_user_id,
        "assessment_request": task.input,
        "profile": {
            "id": str(profile.id),
            "version": profile.profile_version,
            "status": profile.profile_status,
            "summary": _optional_text(profile.summary, max_chars=4000),
            "message_count": profile.message_count,
            "file_count": profile.file_count,
            "fresh_window_days": profile.fresh_window_days,
            "archive_window_days": profile.archive_window_days,
            "observed_range_start_ts": profile.observed_range_start_ts,
            "observed_range_end_ts": profile.observed_range_end_ts,
            "confidence_score": str(profile.confidence_score),
            "confidence_reason": _optional_text(
                profile.confidence_reason,
                max_chars=500,
            ),
        },
        "semantic_extraction": _semantic_extraction_payload(profile),
        "assumptions": _json_list(profile.assumptions_json, limit=5, max_chars=240),
        "evidence_refs": _json_list(profile.evidence_refs_json, limit=8, max_chars=500),
        "allowed_candidate_types": sorted(ALLOWED_CANDIDATE_TYPES),
        "max_candidates": MAX_EXTRACTED_CANDIDATES,
    }


def _semantic_extraction_payload(profile: ObserveChannelProfile) -> JsonObject:
    profile_payload = (
        profile.profile_json if isinstance(profile.profile_json, dict) else {}
    )
    extraction = profile_payload.get("semantic_extraction")
    if not isinstance(extraction, dict):
        metadata = (
            profile.metadata_json if isinstance(profile.metadata_json, dict) else {}
        )
        extraction = metadata.get("semantic_extraction")
    if not isinstance(extraction, dict):
        return {}
    return {
        "likely_purpose": _optional_text(
            extraction.get("likely_purpose"), max_chars=260
        ),
        "recurring_topics": _string_tuple(
            extraction.get("recurring_topics"),
            max_items=5,
            max_chars=180,
        ),
        "workflows": _string_tuple(
            extraction.get("workflows"),
            max_items=5,
            max_chars=220,
        ),
        "important_entities": _string_tuple(
            extraction.get("important_entities"),
            max_items=8,
            max_chars=180,
        ),
        "assumptions": _string_tuple(
            extraction.get("assumptions"),
            max_items=5,
            max_chars=220,
        ),
        "help_opportunities": _string_tuple(
            extraction.get("help_opportunities"),
            max_items=5,
            max_chars=220,
        ),
        "evidence": _string_tuple(
            extraction.get("evidence"),
            max_items=8,
            max_chars=240,
        ),
        "confidence": _optional_text(extraction.get("confidence"), max_chars=40),
    }


def _json_list(
    value: object,
    *,
    limit: int,
    max_chars: int,
) -> tuple[object, ...]:
    if not isinstance(value, list):
        return ()
    output: list[object] = []
    for item in value[:limit]:
        if isinstance(item, str):
            text = _optional_text(item, max_chars=max_chars)
            if text is not None:
                output.append(text)
        elif isinstance(item, dict):
            output.append(_compact_json_object(item, max_chars=max_chars))
    return tuple(output)


def _compact_json_object(value: dict[object, object], *, max_chars: int) -> JsonObject:
    output: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, str):
            text = _optional_text(item, max_chars=max_chars)
            if text is not None:
                output[key] = text
        elif isinstance(item, int | float | bool) or item is None:
            output[key] = item
    return output


def _candidate_from_payload(
    value: object,
    *,
    extractor_prompt_name: str,
) -> WitnessOpportunityCandidateInput | None:
    if not isinstance(value, dict):
        return None
    candidate_type = _optional_text(value.get("candidate_type"), max_chars=80)
    title = _optional_text(value.get("title"), max_chars=140)
    summary = _optional_text(value.get("summary"), max_chars=1000)
    if (
        candidate_type not in ALLOWED_CANDIDATE_TYPES
        or title is None
        or summary is None
    ):
        return None
    confidence_score = _confidence(value.get("confidence_score"))
    confidence_reason = _optional_text(
        value.get("confidence_reason"),
        max_chars=500,
    )
    evidence = _string_tuple(value.get("evidence"), max_items=5, max_chars=300)
    return WitnessOpportunityCandidateInput(
        candidate_type=candidate_type,
        title=title,
        summary=summary,
        suggested_action=_optional_text(value.get("suggested_action"), max_chars=500),
        suggested_message=_optional_text(value.get("suggested_message"), max_chars=500),
        evidence=evidence,
        confidence_score=confidence_score,
        confidence_reason=confidence_reason or "Witness extractor proposed this.",
        metadata_json={
            "extractor": extractor_prompt_name,
        },
        automation_kind=_automation_kind(value.get("automation_kind")),
        cadence_suggestion=_optional_text(
            value.get("cadence_suggestion"), max_chars=160
        ),
        deliverable=_optional_text(value.get("deliverable"), max_chars=300),
    )


def _automation_kind(value: object) -> str | None:
    """Lenient automation kind parsing: missing/invalid means None (watch)."""

    text = _optional_text(value, max_chars=40)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in ALLOWED_AUTOMATION_KINDS:
        return None
    return normalized


def _optional_text(value: object, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text:
        return None
    return text[:max_chars].strip()


def _string_tuple(
    value: object,
    *,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _optional_text(item, max_chars=max_chars)
        if text is None:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= max_items:
            break
    return tuple(output)


def _confidence(value: object) -> Decimal:
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0.500")
    if score < 0:
        return Decimal("0.000")
    if score > 1:
        return Decimal("1.000")
    return score.quantize(Decimal("0.001"))


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]
