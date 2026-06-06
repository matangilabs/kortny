"""Dedicated channel graph refresh pipeline.

This path is intentionally narrower than the general agent runtime. Dashboard
graph refreshes should behave like extraction jobs: bounded source read,
deterministic profile synthesis, validation, and projection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from kortny.db.models import Task, TaskEventType
from kortny.knowledge_graph.refresh import KG_REFRESH_SOURCE
from kortny.llm import ChatMessage, LLMService
from kortny.observability import log_observation
from kortny.observe.assessment import (
    channel_assessment_request_event,
    request_payload_channel_id,
)
from kortny.tasks import TaskService
from kortny.tools import SlackChannelHistoryTool
from kortny.tools.types import JsonObject, ToolResult

KG_CHANNEL_REFRESH_PIPELINE_STARTED_MESSAGE = "kg_channel_refresh_pipeline_started"
KG_CHANNEL_REFRESH_HISTORY_LOADED_MESSAGE = "kg_channel_refresh_history_loaded"
KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE = (
    "kg_channel_refresh_profile_synthesized"
)
KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE = "kg_channel_refresh_semantic_extracted"
KG_CHANNEL_REFRESH_SEMANTIC_FALLBACK_MESSAGE = "kg_channel_refresh_semantic_fallback"
KG_CHANNEL_REFRESH_PIPELINE_COMPLETED_MESSAGE = "kg_channel_refresh_pipeline_completed"
GRAPH_REFRESH_HISTORY_LIMIT = 80
SEMANTIC_EXTRACTOR_PROMPT_NAME = "kortny.knowledge_graph.channel_semantic_extractor"
SEMANTIC_EXTRACTOR_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
SEMANTIC_EXTRACTOR_MAX_MESSAGES = 40
SEMANTIC_EXTRACTOR_MAX_MESSAGE_TEXT_CHARS = 500
SEMANTIC_EXTRACTOR_MAX_TOTAL_TEXT_CHARS = 6000

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_&.-]{2,}")
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_STOP_WORDS = frozenset(
    {
        "about",
        "above",
        "after",
        "also",
        "and",
        "any",
        "are",
        "around",
        "because",
        "been",
        "before",
        "between",
        "channel",
        "could",
        "for",
        "from",
        "have",
        "here",
        "into",
        "just",
        "last",
        "like",
        "more",
        "need",
        "only",
        "over",
        "please",
        "should",
        "that",
        "the",
        "their",
        "there",
        "this",
        "through",
        "what",
        "when",
        "where",
        "which",
        "with",
        "work",
        "would",
        "your",
    }
)
_META_TEXT_RE = re.compile(
    r"\b(adk|branch outputs|final slack-native answer|kortny_root_orchestrator|"
    r"planned[_ -]?workflow|quick_response_agent|route_reason|tool selector)\b",
    re.I,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChannelGraphRefreshPipelineResult:
    """Result from a direct channel graph refresh extraction run."""

    result_summary: str
    artifact_count: int
    message_count: int
    file_count: int


def is_dashboard_graph_refresh_task(session: Session, task: Task) -> bool:
    """Return whether this task should bypass the general agent runtime."""

    request_event = channel_assessment_request_event(session, task)
    if request_event is None:
        return False
    return request_event.payload.get("source") == KG_REFRESH_SOURCE


class ChannelGraphRefreshPipeline:
    """Run a bounded Graphify-style extraction pass for one Slack channel."""

    def __init__(
        self,
        *,
        session: Session,
        task_service: TaskService,
        history_tool: SlackChannelHistoryTool,
        llm: LLMService | None = None,
    ) -> None:
        self.session = session
        self.task_service = task_service
        self.history_tool = history_tool
        self.llm = llm

    def run(self, task: Task) -> ChannelGraphRefreshPipelineResult:
        """Load bounded channel context and synthesize a graph profile seed."""

        request_event = channel_assessment_request_event(self.session, task)
        request_payload = request_event.payload if request_event is not None else {}
        channel_id = (
            request_payload_channel_id(request_payload) or task.slack_channel_id or ""
        )
        if not channel_id:
            raise ValueError("channel graph refresh requires a Slack channel ID")

        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_REFRESH_PIPELINE_STARTED_MESSAGE,
                "pipeline": "channel_graph_refresh",
                "pipeline_version": "hig_181_graphify_slice_1",
                "source": KG_REFRESH_SOURCE,
                "channel_id": channel_id,
            },
        )
        log_observation(
            logger,
            "kg_channel_refresh_pipeline_started",
            task=task,
            pipeline="channel_graph_refresh",
            pipeline_version="hig_181_graphify_slice_1",
            channel_id=channel_id,
        )

        args: JsonObject = {
            "channel_id": channel_id,
            "limit": GRAPH_REFRESH_HISTORY_LIMIT,
            "include_threads": True,
            "source": "auto",
        }
        tool_call_id = f"kg-refresh-{uuid.uuid4().hex[:12]}"
        normalized_args_hash = _normalized_tool_args_hash(args)
        self.task_service.append_event(
            task,
            TaskEventType.tool_call,
            {
                "turn": 0,
                "tool_call_id": tool_call_id,
                "tool": self.history_tool.name,
                "runtime": "kg_channel_refresh_pipeline",
                "step_id": "bounded_slack_history",
                "normalized_args_hash": normalized_args_hash,
                "attempt_no": 1,
                "argument_keys": sorted(args),
                "arguments": args,
            },
        )

        started = time.perf_counter()
        result = self.history_tool.invoke(args)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        result_payload = _tool_result_payload(self.history_tool.name, result)
        self.task_service.append_event(
            task,
            TaskEventType.tool_result,
            {
                "turn": 0,
                "tool_call_id": tool_call_id,
                "tool": self.history_tool.name,
                "runtime": "kg_channel_refresh_pipeline",
                "step_id": "bounded_slack_history",
                "normalized_args_hash": normalized_args_hash,
                "attempt_no": 1,
                "latency_ms": latency_ms,
                "output_shape": _output_shape(result.output),
                "artifact_count": len(result.artifacts),
                "recoverable": _recoverable_tool_result(result.output),
                **result_payload,
            },
        )

        messages = _history_messages(result.output)
        file_count = _file_count(messages)
        context_source = _safe_str(result.output.get("context_source"))
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_REFRESH_HISTORY_LOADED_MESSAGE,
                "channel_id": channel_id,
                "message_count": len(messages),
                "file_count": file_count,
                "context_source": context_source,
                "slack_api_called": result.output.get("slack_api_called"),
                "cache_hit": result.output.get("cache_hit"),
                "limit": GRAPH_REFRESH_HISTORY_LIMIT,
            },
        )

        deterministic_summary = _synthesize_profile_summary(
            channel_id=channel_id,
            messages=messages,
            context_source=context_source,
        )
        semantic_result = self._semantic_profile_summary(
            task=task,
            channel_id=channel_id,
            messages=messages,
            context_source=context_source,
        )
        synthesis = "semantic_llm" if semantic_result is not None else "deterministic"
        summary = semantic_result or deterministic_summary
        summary = _validated_profile_summary(
            summary,
            channel_id=channel_id,
            message_count=len(messages),
        )
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE,
                "channel_id": channel_id,
                "message_count": len(messages),
                "file_count": file_count,
                "summary_chars": len(summary),
                "synthesis": synthesis,
            },
        )
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_REFRESH_PIPELINE_COMPLETED_MESSAGE,
                "channel_id": channel_id,
                "artifact_count": len(result.artifacts),
            },
        )
        log_observation(
            logger,
            "kg_channel_refresh_pipeline_completed",
            task=task,
            channel_id=channel_id,
            message_count=len(messages),
            file_count=file_count,
            context_source=context_source,
            latency_ms=latency_ms,
        )
        return ChannelGraphRefreshPipelineResult(
            result_summary=summary,
            artifact_count=len(result.artifacts),
            message_count=len(messages),
            file_count=file_count,
        )

    def _semantic_profile_summary(
        self,
        *,
        task: Task,
        channel_id: str,
        messages: tuple[JsonObject, ...],
        context_source: str | None,
    ) -> str | None:
        if self.llm is None:
            return None
        if not messages:
            self._record_semantic_fallback(
                task=task,
                channel_id=channel_id,
                reason="no_messages",
            )
            return None

        try:
            completion = self.llm.complete(
                task_id=task.id,
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "You extract a conservative workspace knowledge graph "
                            "profile from bounded Slack channel evidence. Return "
                            "only JSON. Do not mention internal runtimes, agents, "
                            "prompts, tools, or orchestration. Do not infer DMs or "
                            "private context. Keep every field short and grounded "
                            "in the provided message snippets. Never use em "
                            "dashes in JSON string values. Use commas, colons, "
                            "semicolons, periods, or simple hyphens instead."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            _semantic_extraction_payload(
                                channel_id=channel_id,
                                messages=messages,
                                context_source=context_source,
                            ),
                            default=str,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    ),
                ),
                prompt_name=SEMANTIC_EXTRACTOR_PROMPT_NAME,
                response_format=SEMANTIC_EXTRACTOR_RESPONSE_FORMAT,
            )
            extraction = _parse_semantic_extraction(completion.content)
            summary = _semantic_summary_text(
                channel_id=channel_id,
                message_count=len(messages),
                context_source=context_source,
                extraction=extraction,
            )
        except Exception as exc:
            self._record_semantic_fallback(
                task=task,
                channel_id=channel_id,
                reason="semantic_extraction_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

        if _META_TEXT_RE.search(summary):
            self._record_semantic_fallback(
                task=task,
                channel_id=channel_id,
                reason="semantic_output_rejected_meta_text",
            )
            return None

        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE,
                "channel_id": channel_id,
                "prompt_name": SEMANTIC_EXTRACTOR_PROMPT_NAME,
                "confidence": extraction.confidence,
                "topic_count": len(extraction.recurring_topics),
                "workflow_count": len(extraction.workflows),
                "entity_count": len(extraction.important_entities),
                "help_opportunity_count": len(extraction.help_opportunities),
                "extraction": _semantic_extraction_event_payload(extraction),
            },
        )
        return summary

    def _record_semantic_fallback(
        self,
        *,
        task: Task,
        channel_id: str,
        reason: str,
        error_type: str | None = None,
        error: str | None = None,
    ) -> None:
        payload: JsonObject = {
            "message": KG_CHANNEL_REFRESH_SEMANTIC_FALLBACK_MESSAGE,
            "channel_id": channel_id,
            "prompt_name": SEMANTIC_EXTRACTOR_PROMPT_NAME,
            "reason": reason,
        }
        if error_type is not None:
            payload["error_type"] = error_type
        if error is not None:
            payload["error"] = error[:500]
        self.task_service.append_event(task, TaskEventType.log, payload)


def _synthesize_profile_summary(
    *,
    channel_id: str,
    messages: tuple[JsonObject, ...],
    context_source: str | None,
) -> str:
    if not messages:
        return (
            f"Channel graph profile for {channel_id}: no recent observed Slack "
            "messages were available yet. Confidence is low; keep the channel "
            "membership active and refresh again after Kortny observes more "
            "channel activity."
        )

    stats = _message_stats(messages)
    topic_text = _human_list(stats.top_terms) if stats.top_terms else "general updates"
    entity_text = _human_list(stats.entities) if stats.entities else "none detected"
    file_text = (
        f"{stats.file_count} file attachment(s) were observed"
        if stats.file_count
        else "no file attachments were observed"
    )
    evidence_text = _evidence_examples(messages)
    confidence = "medium" if len(messages) >= 10 else "low"
    source_text = context_source or "unknown"

    return (
        f"Channel graph profile for {channel_id} based on {len(messages)} recent "
        f"Slack message(s) from {source_text}. Confidence is {confidence}.\n\n"
        f"Likely purpose: recent activity appears to center on {topic_text}.\n\n"
        f"Observed structure: {stats.user_count} participant(s), "
        f"{stats.thread_count} threaded message(s), and {file_text}.\n\n"
        f"Notable entities or symbols: {entity_text}.\n\n"
        f"Evidence snippets: {evidence_text}\n\n"
        "Potential Kortny help: summarize recent decisions, compare recurring "
        "updates over time, surface unresolved follow-ups, inspect shared files, "
        "and turn repeated channel workflows into candidate automations."
    )


def _validated_profile_summary(
    summary: str,
    *,
    channel_id: str,
    message_count: int,
) -> str:
    if not _META_TEXT_RE.search(summary):
        return summary[:8000]
    return (
        f"Channel graph profile for {channel_id} based on {message_count} recent "
        "Slack message(s). The initial profile synthesis was rejected because it "
        "contained runtime orchestration text, so this conservative seed only "
        "records that Kortny can observe this channel and should refresh again "
        "with bounded channel evidence."
    )


def _semantic_extraction_payload(
    *,
    channel_id: str,
    messages: tuple[JsonObject, ...],
    context_source: str | None,
) -> JsonObject:
    return {
        "channel_id": channel_id,
        "context_source": context_source,
        "instructions": {
            "task": "Extract a grounded channel profile seed.",
            "output_schema": {
                "likely_purpose": "short string",
                "recurring_topics": ["short strings"],
                "workflows": ["short strings"],
                "important_entities": ["people, tools, projects, symbols, or systems"],
                "assumptions": ["careful assumptions with evidence basis"],
                "help_opportunities": ["ways Kortny can help in this channel"],
                "evidence": ["short message snippets or paraphrases"],
                "confidence": "low|medium|high",
            },
            "limits": {
                "max_items_per_list": 5,
                "max_string_chars": 160,
                "no_private_or_dm_context": True,
                "no_runtime_or_prompt_commentary": True,
            },
        },
        "messages": _semantic_messages(messages),
    }


def _semantic_messages(messages: tuple[JsonObject, ...]) -> list[JsonObject]:
    output: list[JsonObject] = []
    remaining_text_chars = SEMANTIC_EXTRACTOR_MAX_TOTAL_TEXT_CHARS
    for message in messages[-SEMANTIC_EXTRACTOR_MAX_MESSAGES:]:
        if remaining_text_chars <= 0:
            break
        formatted = _semantic_message(
            message,
            max_text_chars=min(
                SEMANTIC_EXTRACTOR_MAX_MESSAGE_TEXT_CHARS,
                remaining_text_chars,
            ),
        )
        text = formatted.get("text")
        if isinstance(text, str):
            remaining_text_chars -= len(text)
        output.append(formatted)
    return output


def _semantic_message(message: JsonObject, *, max_text_chars: int) -> JsonObject:
    return {
        "ts": _safe_str(message.get("ts")),
        "thread_ts": _safe_str(message.get("thread_ts")),
        "user": _safe_str(message.get("user")) or _safe_str(message.get("author")),
        "text": (_safe_str(message.get("text")) or "")[:max_text_chars],
        "file_count": (
            len([file for file in files if isinstance(file, Mapping)])
            if isinstance((files := message.get("files")), list)
            else 0
        ),
    }


def _parse_semantic_extraction(content: str | None) -> _SemanticExtraction:
    if not content:
        raise ValueError("semantic extractor returned empty content")
    raw = json.loads(_extract_json_object(content))
    if not isinstance(raw, dict):
        raise ValueError("semantic extractor returned a non-object payload")

    likely_purpose = _bounded_required_text(raw.get("likely_purpose"), "likely_purpose")
    confidence = _bounded_required_text(raw.get("confidence"), "confidence").lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("semantic extractor confidence must be low, medium, or high")

    extraction = _SemanticExtraction(
        likely_purpose=likely_purpose,
        recurring_topics=_bounded_text_tuple(raw.get("recurring_topics")),
        workflows=_bounded_text_tuple(raw.get("workflows")),
        important_entities=_bounded_text_tuple(raw.get("important_entities")),
        assumptions=_bounded_text_tuple(raw.get("assumptions")),
        help_opportunities=_bounded_text_tuple(raw.get("help_opportunities")),
        evidence=_bounded_text_tuple(raw.get("evidence")),
        confidence=confidence,
    )
    joined = " ".join(
        (
            extraction.likely_purpose,
            *extraction.recurring_topics,
            *extraction.workflows,
            *extraction.important_entities,
            *extraction.assumptions,
            *extraction.help_opportunities,
            *extraction.evidence,
        )
    )
    if _META_TEXT_RE.search(joined):
        raise ValueError("semantic extractor output contained runtime meta text")
    return extraction


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("semantic extractor response did not contain a JSON object")
    return stripped[start : end + 1]


def _bounded_required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"semantic extractor missing {field_name}")
    return _bounded_text(value, max_chars=200)


def _bounded_text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        bounded = _bounded_text(item, max_chars=180)
        if bounded:
            items.append(bounded)
        if len(items) >= 5:
            break
    return tuple(items)


def _bounded_text(value: str, *, max_chars: int) -> str:
    return re.sub(r"\s+", " ", value).strip()[:max_chars].strip()


def _semantic_summary_text(
    *,
    channel_id: str,
    message_count: int,
    context_source: str | None,
    extraction: _SemanticExtraction,
) -> str:
    return (
        f"Channel graph profile for {channel_id} based on {message_count} recent "
        f"Slack message(s) from {context_source or 'unknown'}. Confidence is "
        f"{extraction.confidence}.\n\n"
        f"Likely purpose: {extraction.likely_purpose}\n\n"
        f"Recurring topics: {_human_list(extraction.recurring_topics) or 'none detected'}.\n\n"
        f"Observed workflows: {_human_list(extraction.workflows) or 'none detected'}.\n\n"
        "Notable entities or systems: "
        f"{_human_list(extraction.important_entities) or 'none detected'}.\n\n"
        f"Careful assumptions: {_human_list(extraction.assumptions) or 'none'}.\n\n"
        "Potential Kortny help: "
        f"{_human_list(extraction.help_opportunities) or 'summarize activity and surface follow-ups'}.\n\n"
        f"Evidence snippets: {_human_list(extraction.evidence) or 'none provided'}."
    )


def _semantic_extraction_event_payload(extraction: _SemanticExtraction) -> JsonObject:
    return {
        "likely_purpose": extraction.likely_purpose,
        "recurring_topics": list(extraction.recurring_topics),
        "workflows": list(extraction.workflows),
        "important_entities": list(extraction.important_entities),
        "assumptions": list(extraction.assumptions),
        "help_opportunities": list(extraction.help_opportunities),
        "evidence": list(extraction.evidence),
        "confidence": extraction.confidence,
    }


@dataclass(frozen=True, slots=True)
class _MessageStats:
    top_terms: tuple[str, ...]
    entities: tuple[str, ...]
    user_count: int
    thread_count: int
    file_count: int


@dataclass(frozen=True, slots=True)
class _SemanticExtraction:
    likely_purpose: str
    recurring_topics: tuple[str, ...]
    workflows: tuple[str, ...]
    important_entities: tuple[str, ...]
    assumptions: tuple[str, ...]
    help_opportunities: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: str


def _message_stats(messages: tuple[JsonObject, ...]) -> _MessageStats:
    tokens: Counter[str] = Counter()
    entities: Counter[str] = Counter()
    users: set[str] = set()
    thread_count = 0
    file_count = 0
    for message in messages:
        text = _safe_str(message.get("text")) or ""
        for token in _TOKEN_RE.findall(text):
            normalized = token.lower().strip(".-_")
            if len(normalized) < 3 or normalized in _STOP_WORDS:
                continue
            tokens[normalized] += 1
        for ticker in _TICKER_RE.findall(text):
            if ticker not in {"HTTP", "HTTPS"}:
                entities[ticker] += 1
        user = _safe_str(message.get("user"))
        if user:
            users.add(user)
        ts = _safe_str(message.get("ts"))
        thread_ts = _safe_str(message.get("thread_ts"))
        if thread_ts and ts and thread_ts != ts:
            thread_count += 1
        files = message.get("files")
        if isinstance(files, list):
            file_count += sum(1 for file in files if isinstance(file, Mapping))

    return _MessageStats(
        top_terms=tuple(term for term, _ in tokens.most_common(6)),
        entities=tuple(entity for entity, _ in entities.most_common(8)),
        user_count=len(users),
        thread_count=thread_count,
        file_count=file_count,
    )


def _evidence_examples(messages: tuple[JsonObject, ...]) -> str:
    snippets: list[str] = []
    for message in messages[-5:]:
        text = _safe_str(message.get("text"))
        if not text:
            continue
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            continue
        snippets.append(cleaned[:140])
        if len(snippets) >= 3:
            break
    if not snippets:
        return "No text snippets were available in the observed messages."
    return "; ".join(f'"{snippet}"' for snippet in snippets)


def _history_messages(output: Mapping[str, Any]) -> tuple[JsonObject, ...]:
    messages = output.get("messages")
    if not isinstance(messages, list):
        return ()
    return tuple(message for message in messages if isinstance(message, dict))


def _file_count(messages: tuple[JsonObject, ...]) -> int:
    return _message_stats(messages).file_count


def _human_list(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _safe_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _normalized_tool_args_hash(arguments: JsonObject) -> str:
    canonical = json.dumps(
        arguments,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tool_result_payload(tool_name: str, result: ToolResult) -> JsonObject:
    return {
        "tool": tool_name,
        "output": result.output,
        "cost_usd": str(result.cost_usd),
        "artifacts": [
            {
                "filename": artifact.filename,
                "path": artifact.path,
                "mime_type": artifact.mime_type,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in result.artifacts
        ],
    }


def _output_shape(output: JsonObject) -> JsonObject:
    return {
        "type": "object",
        "keys": sorted(output),
    }


def _recoverable_tool_result(output: JsonObject) -> bool | None:
    error = output.get("error")
    if not isinstance(error, dict):
        return None
    recoverable = error.get("recoverable")
    return recoverable if isinstance(recoverable, bool) else None
