"""Slack-native final response synthesis."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import Artifact, Task, TaskEvent, TaskEventType
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.llm import (
    ChatMessage,
    LLMProvider,
    LLMService,
    ModelRouter,
    ModelRouteTier,
)
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.observe.style_cards import ChannelStyleCard, load_channel_style
from kortny.persona import personalize
from kortny.skills import (
    RESPONSE_HUMANIZER_INVOCATION,
    SkillActivation,
    SkillRegistryService,
)
from kortny.slack.synthesis import (
    EvidenceKind,
    EvidenceTrust,
    SlackRef,
    SynthesisApprovalState,
    SynthesisContext,
    SynthesisEvidence,
    SynthesisOutcome,
)
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject

RESPONSE_HUMANIZER_PROMPT_NAME = "kortny.response_humanizer"
RESPONSE_HUMANIZER_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
RESPONSE_HUMANIZER_SYSTEM_PROMPT = """You write __AGENT_NAME__'s final Slack response from a typed ResponseRecord and SynthesisContext.

Return exactly one JSON object:
{"message":"Slack-ready message"}

The message value must contain only the Slack-ready response. Do not include
notes, reasoning, draft analysis, labels like final_mode, or explanations of
your rewrite.

Rules:
- Use only facts, actions, artifacts, failures, uncertainties, links, and the raw
  answer present in the ResponseRecord, bounded by the evidence and outcome in
  SynthesisContext.
- Treat SynthesisContext as the evidence boundary. If its outcome is no_result,
  partial_failure, needs_approval, or error, make that state clear and do not
  imply completed work that the context does not support.
- Do not add new facts, numbers, source claims, tools, or conclusions.
- Lead with the answer, not with boilerplate.
- Write as __AGENT_NAME__, one Slack-native coworker. Internalize schedules, Witness,
  memory, knowledge graph, Slack context, and integrations as __AGENT_NAME__'s own
  abilities. Do not make the response sound like a wrapper around subsystems.
- Prefer natural coworker language over tool/report language. For example,
  say "Yep, I have..." instead of "I found records in the scheduler database";
  say "I can..." instead of "native tools are available".
- Do not expose implementation labels like "native tools", "always on",
  "scheduler DB", "workspace graph", "planned workflow", "runtime", "agent",
  "branch", or "source of truth" unless the user explicitly asks about
  internals.
- For schedules and recurring work, speak in user-facing terms: what __AGENT_NAME__
  will do, when it runs, where it will be delivered, and the next useful time.
- Follow the selected response_shape. Include required elements when the
  ResponseRecord contains enough evidence; when it does not, state the limit
  instead of inventing support.
- For analyst_audit responses, use a consulting-grade shape: bottom line, scope,
  evaluation lens when relevant, ranked findings, concrete recommendations,
  highest-leverage move, and a specific next step.
- For comparison_memo responses, make the recommendation explicit and then show
  the tradeoffs.
- Make tool usage sound natural when it helps, not mechanical.
- For memory no-match cases, use normal coworker language: say __AGENT_NAME__ checked
  what it remembers, does not see that saved right now, and has nothing to
  remove. Do not expose raw tool phrases.
- For context_profile responses, answer as a workspace-aware coworker. Explain
  what __AGENT_NAME__ knows, why __AGENT_NAME__ believes it, and the confidence/limits in
  natural language. Do not lead with implementation terms like "workspace graph"
  unless the user specifically asks about internals.
- For capability inventory responses, turn tool registry and integration data
  into natural first-person groups. Say what __AGENT_NAME__ can do and which connected
  apps are available. Do not expose field names like native_tools,
  connected_integrations, toolkit_slug, connected_account_id, or scope_note.
- Use Slack mrkdwn: *bold*, simple bullets, and <https://url|label> links.
- Do not use Markdown headings with #.
- Avoid repetitive endings like "If you want..." unless a next step is
  genuinely useful and specific.
- Keep it concise for Slack, but do not omit important recommendations.
- Apply human editing principles: remove inflated/promotional language, cut
  chatbot artifacts, vary rhythm naturally, and preserve substance.
- The style_profile may include channel_voice: match its register (formality,
  brevity, emoji norms). Never imitate a specific person.
"""
MAX_RAW_ANSWER_CHARS = 8000
MAX_TRACE_OUTPUT_CHARS = 1200
MAX_HUMANIZED_CHARS = 12000
MAX_SYNTHESIS_EVIDENCE_CHARS = 900
HUMANIZER_LEAK_MARKERS = frozenset(
    {
        "_mode is",
        "answer_shape",
        "according to my guidelines",
        "branch outputs",
        "final_mode",
        "i should",
        "i'm the planned_workflow_merger",
        "i am the planned_workflow_merger",
        "i can see that the integration branch",
        "i can see that the workspace branch",
        "i can see that the research branch",
        "i'll present this as kortny's final answer",
        "i’ll present this as kortny's final answer",
        "i'll keep it",
        "i’ll keep it",
        "let me check",
        "let me write",
        "my job is to merge",
        "planned_integration_worker",
        "planned_research_worker",
        "planned_workflow",
        "planned_workflow_merger",
        "planned_workspace_worker",
        "raw_answer",
        "renderer_constraints",
        "response_record",
        "the user said",
        "the user is asking",
    }
)
FINAL_ANSWER_MARKERS = (
    "final answer:",
    "here is the final slack-ready answer:",
    "here's the final slack-ready answer:",
    "i'll present this as kortny's final answer.",
    "i’ll present this as kortny's final answer.",
)
FINAL_ANSWER_LINE_PREFIXES = (
    ":",
    "*",
    "bottom line",
    "here's",
    "here is",
    "i'm",
    "i’m",
    "quick take",
    "recommendation",
    "short version",
    "the short version",
    "yep",
    "yes",
)
MEMORY_TOOL_NAMES = frozenset(
    {"remember_fact", "recall_fact", "inspect_memory", "forget_fact"}
)
logger = logging.getLogger(__name__)


class ResponseMode(StrEnum):
    """High-level response shape selected from execution evidence."""

    quick_answer = "quick_answer"
    research_summary = "research_summary"
    file_analysis = "file_analysis"
    artifact_delivery = "artifact_delivery"
    failure_recovery = "failure_recovery"
    memory_recall = "memory_recall"
    context_answer = "context_answer"
    multi_step_recap = "multi_step_recap"


class ResponseShape(StrEnum):
    """Concrete Slack response pattern selected for the final answer."""

    quick_reply = "quick_reply"
    research_brief = "research_brief"
    analyst_audit = "analyst_audit"
    comparison_memo = "comparison_memo"
    file_review = "file_review"
    document_delivery = "document_delivery"
    memory_note = "memory_note"
    context_profile = "context_profile"
    status_recap = "status_recap"
    failure_note = "failure_note"


@dataclass(frozen=True, slots=True)
class SlackSurface:
    """Slack delivery surface for the response."""

    kind: str
    threaded: bool

    def to_payload(self) -> JsonObject:
        return {"kind": self.kind, "threaded": self.threaded}


@dataclass(frozen=True, slots=True)
class ResponseStyleProfile:
    """Small, typed style profile for response synthesis."""

    tone: str = "approachable, steady, direct"
    brevity: str = "concise"
    polish: str = "professional"
    humor: str = "off_by_default"
    proactive_suggestions: str = "only_when_clearly_useful"
    channel_voice: str = ""

    def to_payload(self) -> JsonObject:
        payload: JsonObject = {
            "tone": self.tone,
            "brevity": self.brevity,
            "polish": self.polish,
            "humor": self.humor,
            "proactive_suggestions": self.proactive_suggestions,
        }
        if self.channel_voice:
            payload["channel_voice"] = self.channel_voice
        return payload

    @staticmethod
    def from_style_card(
        card: ChannelStyleCard,
        surface: SlackSurface,
    ) -> ResponseStyleProfile:
        """Map channel register dimensions onto the response style profile.

        DMs keep the static default profile: per-user style is explicitly out
        of scope, and a channel card never describes a DM surface.
        """

        if surface.kind != "channel":
            return ResponseStyleProfile()
        return ResponseStyleProfile(
            tone=CHANNEL_VOICE_TONES.get(
                card.formality, "approachable, steady, direct"
            ),
            brevity=CHANNEL_VOICE_BREVITY.get(card.brevity, "concise"),
            polish="relaxed" if card.punctuation == "relaxed" else "professional",
            channel_voice=render_channel_voice(card),
        )


CHANNEL_VOICE_TONES = {
    "casual": "casual, friendly, direct",
    "neutral": "approachable, steady, direct",
    "formal": "polished, professional, direct",
}
CHANNEL_VOICE_BREVITY = {
    "terse": "very concise",
    "moderate": "concise",
    "expansive": "thorough but tight",
}
CHANNEL_VOICE_EMOJI = {
    "none": "no emoji",
    "light": "light emoji use",
    "heavy": "emoji are welcome",
}
CHANNEL_VOICE_MAX_CHARS = 240


def render_channel_voice(card: ChannelStyleCard) -> str:
    """Render the card's dims + notes as one bounded instruction line."""

    dims = (
        f"Match this channel's register: {card.formality} formality, "
        f"{card.brevity} replies, "
        f"{CHANNEL_VOICE_EMOJI.get(card.emoji_culture, 'light emoji use')}, "
        f"{card.punctuation} punctuation."
    )
    line = f"{dims} {card.notes}".strip() if card.notes else dims
    return _shorten(" ".join(line.split()), max_chars=CHANNEL_VOICE_MAX_CHARS)


@dataclass(frozen=True, slots=True)
class ResponseShapeProfile:
    """Quality contract for the selected response shape."""

    shape: ResponseShape
    label: str
    selected_reason: str
    required_elements: list[str]
    quality_checks: list[str]
    avoid: list[str]
    framework_hint: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "shape": self.shape.value,
            "label": self.label,
            "selected_reason": self.selected_reason,
            "required_elements": self.required_elements,
            "quality_checks": self.quality_checks,
            "avoid": self.avoid,
            "framework_hint": self.framework_hint,
        }


@dataclass(frozen=True, slots=True)
class ResponseAction:
    """One action the agent took while completing the task."""

    tool: str
    status: str
    argument_keys: list[str]
    summary: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "tool": self.tool,
            "status": self.status,
            "argument_keys": self.argument_keys,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ResponseEvidence:
    """Evidence available to the synthesizer."""

    source_type: str
    source_id: str
    tool: str | None = None
    urls: list[str] | None = None
    preview: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "tool": self.tool,
            "urls": self.urls or [],
            "preview": self.preview,
        }


@dataclass(frozen=True, slots=True)
class ResponseArtifact:
    """Artifact produced during the task."""

    filename: str
    mime_type: str | None
    size_bytes: int | None
    posted: bool

    def to_payload(self) -> JsonObject:
        return {
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "posted": self.posted,
        }


@dataclass(frozen=True, slots=True)
class ResponseFailure:
    """Tool or execution failure that may need user-facing caveats."""

    source: str
    code: str | None
    message: str | None
    recoverable: bool | None
    recovery_action: str | None

    def to_payload(self) -> JsonObject:
        return {
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "recovery_action": self.recovery_action,
        }


@dataclass(frozen=True, slots=True)
class ResponseSkill:
    """Procedural skill selected for response synthesis."""

    slug: str
    name: str
    version: str
    owner_type: str
    trust_level: str
    selected_reason: str
    instructions_md: str

    def to_payload(self) -> JsonObject:
        return {
            "slug": self.slug,
            "name": self.name,
            "version": self.version,
            "owner_type": self.owner_type,
            "trust_level": self.trust_level,
            "selected_reason": self.selected_reason,
            "instructions_md": self.instructions_md,
        }


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    """Typed terminal response contract for the Slack humanizer."""

    user_request: str
    raw_answer: str
    response_mode: ResponseMode
    response_shape: ResponseShapeProfile
    task_status: str
    slack_surface: SlackSurface
    style_profile: ResponseStyleProfile
    actions_taken: list[ResponseAction]
    evidence: list[ResponseEvidence]
    artifacts: list[ResponseArtifact]
    failures: list[ResponseFailure]
    uncertainties: list[str]
    suggested_next_actions: list[str]
    procedural_skills: list[ResponseSkill]

    def to_payload(self) -> JsonObject:
        return {
            "user_request": self.user_request,
            "raw_answer": _shorten(
                self.raw_answer,
                max_chars=MAX_RAW_ANSWER_CHARS,
            ),
            "response_mode": self.response_mode.value,
            "response_shape": self.response_shape.to_payload(),
            "task_status": self.task_status,
            "slack_surface": self.slack_surface.to_payload(),
            "style_profile": self.style_profile.to_payload(),
            "actions_taken": [action.to_payload() for action in self.actions_taken],
            "evidence": [item.to_payload() for item in self.evidence],
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "failures": [failure.to_payload() for failure in self.failures],
            "uncertainties": self.uncertainties,
            "suggested_next_actions": self.suggested_next_actions,
            "procedural_skills": [
                skill.to_payload() for skill in self.procedural_skills
            ],
        }

    def summary_payload(self) -> JsonObject:
        """Return a compact trace payload for task events."""

        return {
            "response_mode": self.response_mode.value,
            "response_shape": self.response_shape.shape.value,
            "response_shape_reason": self.response_shape.selected_reason,
            "required_element_count": len(self.response_shape.required_elements),
            "task_status": self.task_status,
            "action_count": len(self.actions_taken),
            "evidence_count": len(self.evidence),
            "artifact_count": len(self.artifacts),
            "failure_count": len(self.failures),
            "uncertainty_count": len(self.uncertainties),
            "suggested_next_action_count": len(self.suggested_next_actions),
            "procedural_skill_count": len(self.procedural_skills),
            "procedural_skill_slugs": [skill.slug for skill in self.procedural_skills],
        }


@dataclass(frozen=True, slots=True)
class ResponseSynthesisResult:
    """Result of a Slack response synthesis pass."""

    text: str
    changed: bool
    reason: str


class ChannelStyleResolver(Protocol):
    """Resolves the per-channel response style for a task, if any."""

    def resolve(
        self,
        *,
        session: Session,
        task: Task,
        surface: SlackSurface,
    ) -> ResponseStyleProfile | None:
        """Return a channel-adapted style profile, or None for the default."""


class ChannelStyleCardResolver:
    """Builds the response style from the channel's learned style card.

    Only constructed when KORTNY_STYLE_CARDS_ENABLED is on; with no resolver,
    no card, or a non-channel surface the static default profile is used and
    behavior is byte-identical to the pre-style-card path.
    """

    def resolve(
        self,
        *,
        session: Session,
        task: Task,
        surface: SlackSurface,
    ) -> ResponseStyleProfile | None:
        if surface.kind != "channel":
            return None
        style = load_channel_style(
            session,
            installation_id=task.installation_id,
            channel_id=task.slack_channel_id,
        )
        profile: ResponseStyleProfile | None = None
        if style.card is not None:
            profile = ResponseStyleProfile.from_style_card(style.card, surface)
        if style.pinned_style:
            pinned_voice = _shorten(
                " ".join(style.pinned_style.split()),
                max_chars=CHANNEL_VOICE_MAX_CHARS,
            )
            profile = replace(
                profile if profile is not None else ResponseStyleProfile(),
                channel_voice=pinned_voice,
            )
        return profile


class ResponseSynthesizer(Protocol):
    """Rewrites raw coordinator output into Slack-facing text."""

    def synthesize(
        self,
        *,
        session: Session,
        task: Task,
        response_record: ResponseRecord,
        synthesis_context: SynthesisContext,
        task_service: TaskService,
    ) -> ResponseSynthesisResult:
        """Return Slack-ready text."""


class StaticResponseSynthesizer:
    """Deterministic fallback that strips unusable response preambles."""

    uses_procedural_skills = False

    def synthesize(
        self,
        *,
        session: Session,
        task: Task,
        response_record: ResponseRecord,
        synthesis_context: SynthesisContext,
        task_service: TaskService,
    ) -> ResponseSynthesisResult:
        del session, task, synthesis_context, task_service
        raw_text = response_record.raw_answer
        normalized = sanitize_humanized_response(None, fallback=raw_text)
        return ResponseSynthesisResult(
            text=normalized,
            changed=normalized != raw_text,
            reason="static_response_cleanup",
        )


class LLMResponseSynthesizer:
    """LLM-backed final response synthesizer."""

    uses_procedural_skills = True

    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
        min_chars: int | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.provider_name = DbLLMProvider(provider_name) if provider_name else None
        self.min_chars = (
            settings.response_humanizer_min_chars
            if min_chars is None
            else max(0, min_chars)
        )

    def synthesize(
        self,
        *,
        session: Session,
        task: Task,
        response_record: ResponseRecord,
        synthesis_context: SynthesisContext,
        task_service: TaskService,
    ) -> ResponseSynthesisResult:
        if _should_skip(response_record, min_chars=self.min_chars):
            normalized = sanitize_humanized_response(
                None,
                fallback=response_record.raw_answer,
            )
            return ResponseSynthesisResult(
                text=normalized,
                changed=normalized != response_record.raw_answer,
                reason="skipped_short_or_artifact",
            )

        model_route = ModelRouter(self.settings).route_for_tier(
            _route_tier(response_record),
            reason="response_humanizer",
        )
        if self.provider is None:
            selection = select_runtime_model(
                session=session,
                settings=self.settings,
                installation_id=task.installation_id,
                model_route=model_route,
            )
            provider = create_provider_for_selection(
                settings=self.settings,
                selection=selection,
            )
            provider_name = self.provider_name or selection.provider_name
            model_route = selection.model_route
        else:
            provider = self.provider
            provider_name = self.provider_name or DbLLMProvider(
                self.settings.llm_provider
            )
        completion = LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=model_route,
        ).complete(
            task_id=task.id,
            messages=(
                ChatMessage(
                    role="system",
                    content=personalize(
                        RESPONSE_HUMANIZER_SYSTEM_PROMPT,
                        self.settings.agent_display_name,
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        _synthesis_payload(response_record, synthesis_context),
                        default=str,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            ),
            prompt_name=RESPONSE_HUMANIZER_PROMPT_NAME,
            response_format=RESPONSE_HUMANIZER_RESPONSE_FORMAT,
        )
        text = sanitize_humanized_response(
            completion.content,
            fallback=response_record.raw_answer,
        )
        return ResponseSynthesisResult(
            text=text,
            changed=text != response_record.raw_answer,
            reason="llm_humanizer",
        )


def synthesize_response(
    synthesizer: ResponseSynthesizer,
    *,
    session: Session,
    task: Task,
    raw_text: str,
    task_service: TaskService,
    style_resolver: ChannelStyleResolver | None = None,
) -> str:
    """Generate Slack-facing response text, failing open to the raw answer."""

    response_record = build_response_record(
        session=session,
        task=task,
        raw_text=raw_text,
        style_resolver=style_resolver,
    )
    if getattr(synthesizer, "uses_procedural_skills", False):
        activations = SkillRegistryService(
            session,
            task_service=task_service,
        ).select_for_response(
            task,
            response_mode=response_record.response_mode.value,
            response_shape=response_record.response_shape.shape.value,
            invocation_kind=RESPONSE_HUMANIZER_INVOCATION,
        )
        response_record = replace(
            response_record,
            procedural_skills=_response_skills_from_activations(activations),
        )
    synthesis_context = build_synthesis_context(
        session=session,
        task=task,
        raw_text=raw_text,
        response_record=response_record,
    )
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "response_record_built",
            **response_record.summary_payload(),
        },
    )
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "synthesis_context_built",
            **synthesis_context.summary_payload(),
        },
    )
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "response_humanizer_started",
            "raw_chars": len(raw_text),
            "response_mode": response_record.response_mode.value,
        },
    )
    try:
        result = synthesizer.synthesize(
            session=session,
            task=task,
            response_record=response_record,
            synthesis_context=synthesis_context,
            task_service=task_service,
        )
    except Exception as exc:
        logger.exception("response humanizer failed task_id=%s", task.id)
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "response_humanizer_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "fallback": "sanitized_raw_answer",
            },
        )
        return sanitize_humanized_response(None, fallback=raw_text)

    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "response_humanizer_completed",
            "changed": result.changed,
            "reason": result.reason,
            "raw_chars": len(raw_text),
            "output_chars": len(result.text),
            "response_mode": response_record.response_mode.value,
        },
    )
    return result.text


def sanitize_humanized_response(text: str | None, *, fallback: str) -> str:
    """Return usable Slack-facing response text without formatting normalization."""

    safe_fallback = strip_internal_response_preamble(fallback).strip()
    if text is None:
        return safe_fallback
    message = _json_message(text)
    normalized = (
        (message if message is not None else text).strip().strip('"').strip("'").strip()
    )
    if not normalized:
        return safe_fallback
    if _looks_like_humanizer_leak(normalized):
        return safe_fallback
    if len(normalized) > MAX_HUMANIZED_CHARS:
        normalized = normalized[: MAX_HUMANIZED_CHARS - 1].rstrip() + "."
    return normalized


def strip_internal_response_preamble(text: str) -> str:
    """Remove ADK/agent scratchpad text before a Slack-facing final answer."""

    raw = text.strip()
    if not raw:
        return raw
    head = raw[:2500].casefold()
    if not _looks_like_humanizer_leak(head):
        return raw

    for marker in FINAL_ANSWER_MARKERS:
        index = head.find(marker)
        if index == -1:
            continue
        candidate = raw[index + len(marker) :].strip()
        if _usable_final_answer_candidate(candidate):
            return candidate

    lines = raw.splitlines()
    for index, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_final_answer_start(stripped):
            candidate = "\n".join(lines[index:]).strip()
            if _usable_final_answer_candidate(
                candidate
            ) or _usable_short_final_answer_candidate(candidate):
                return candidate
    for candidate in reversed(_paragraphs(raw)[1:]):
        if _usable_final_answer_candidate(candidate) and not _looks_like_humanizer_leak(
            candidate
        ):
            return candidate
    return raw


def _json_message(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message
    return None


def _looks_like_humanizer_leak(text: str) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in HUMANIZER_LEAK_MARKERS)


def _looks_like_final_answer_start(line: str) -> bool:
    normalized = line.casefold()
    return normalized.startswith(FINAL_ANSWER_LINE_PREFIXES)


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
            continue
        if current:
            paragraphs.append("\n".join(current).strip())
            current = []
    if current:
        paragraphs.append("\n".join(current).strip())
    return paragraphs


def _usable_final_answer_candidate(text: str) -> bool:
    return len(text.strip()) >= 40


def _usable_short_final_answer_candidate(text: str) -> bool:
    return len(text.strip()) >= 8


def build_response_record(
    *,
    session: Session,
    task: Task,
    raw_text: str,
    style_resolver: ChannelStyleResolver | None = None,
) -> ResponseRecord:
    """Build the typed response contract from task events and artifacts."""

    events = _task_events(session, task)
    calls_by_id = _tool_calls_by_id(events)
    actions: list[ResponseAction] = []
    evidence: list[ResponseEvidence] = []
    failures: list[ResponseFailure] = []
    uncertainties: list[str] = []

    for event in events:
        if event.type is TaskEventType.tool_result:
            action, evidence_item, failure = _response_items_from_tool_result(
                event,
                calls_by_id,
            )
            if action is not None:
                actions.append(action)
            if evidence_item is not None:
                evidence.append(evidence_item)
            if failure is not None:
                failures.append(failure)
                if failure.message:
                    uncertainties.append(failure.message)
        elif event.type is TaskEventType.error:
            failure = _response_failure_from_error_event(event)
            failures.append(failure)
            if failure.message:
                uncertainties.append(failure.message)

    artifacts = _artifact_summary(session, task)
    response_mode = _select_response_mode(
        raw_text=raw_text,
        actions=actions,
        evidence=evidence,
        artifacts=artifacts,
        failures=failures,
    )
    response_shape = _select_response_shape(
        user_request=task.input,
        raw_text=raw_text,
        response_mode=response_mode,
        actions=actions,
        evidence=evidence,
        artifacts=artifacts,
        failures=failures,
    )
    slack_surface = SlackSurface(
        kind="dm" if task.slack_channel_id.startswith("D") else "channel",
        threaded=task.slack_thread_ts != task.slack_message_ts,
    )
    style_profile: ResponseStyleProfile | None = None
    if style_resolver is not None:
        try:
            style_profile = style_resolver.resolve(
                session=session,
                task=task,
                surface=slack_surface,
            )
        except Exception:
            logger.exception("channel style resolution failed task_id=%s", task.id)
            style_profile = None
    return ResponseRecord(
        user_request=task.input,
        raw_answer=raw_text.strip(),
        response_mode=response_mode,
        response_shape=response_shape,
        task_status=_response_status(failures),
        slack_surface=slack_surface,
        style_profile=style_profile or ResponseStyleProfile(),
        actions_taken=actions[-10:],
        evidence=evidence[-10:],
        artifacts=artifacts,
        failures=failures[-10:],
        uncertainties=list(dict.fromkeys(uncertainties))[-8:],
        suggested_next_actions=_suggested_next_actions(
            response_mode,
            failures,
            artifacts,
        ),
        procedural_skills=[],
    )


def build_synthesis_context(
    *,
    session: Session,
    task: Task,
    raw_text: str,
    response_record: ResponseRecord,
) -> SynthesisContext:
    """Build the typed evidence pack for final Slack response synthesis."""

    events = _task_events(session, task)
    evidence = _synthesis_evidence_from_events(events, task)
    approvals = _approval_states_from_events(events)
    outcome, outcome_reason = _select_synthesis_outcome(
        raw_text=raw_text,
        response_record=response_record,
        events=events,
        approvals=approvals,
    )
    uncertainty = _synthesis_uncertainty(
        response_record=response_record,
        outcome=outcome,
        outcome_reason=outcome_reason,
    )
    return SynthesisContext(
        user_intent=task.input,
        outcome=outcome,
        outcome_reason=outcome_reason,
        slack_surface=response_record.slack_surface.kind,
        threaded=response_record.slack_surface.threaded,
        addressee_user_id=task.slack_user_id,
        evidence=evidence[-10:],
        approvals=approvals[-5:],
        uncertainty=uncertainty[-8:],
        skills_loaded=[skill.slug for skill in response_record.procedural_skills],
        allowed_claim_sources=[item.source_id for item in evidence[-10:]],
        forbidden_claims=_default_forbidden_claims(outcome),
    )


def _should_skip(response_record: ResponseRecord, *, min_chars: int) -> bool:
    raw_text = response_record.raw_answer
    if response_record.response_mode is ResponseMode.artifact_delivery:
        return True
    if (
        response_record.actions_taken
        or response_record.evidence
        or response_record.failures
    ):
        return False
    return len(raw_text) < min_chars


def _response_skills_from_activations(
    activations: Sequence[SkillActivation],
) -> list[ResponseSkill]:
    return [
        ResponseSkill(
            slug=activation.slug,
            name=activation.name,
            version=activation.version,
            owner_type=activation.owner_type,
            trust_level=activation.trust_level,
            selected_reason=activation.selected_reason,
            instructions_md=activation.instructions_md,
        )
        for activation in activations
    ]


def _route_tier(response_record: ResponseRecord) -> ModelRouteTier:
    del response_record
    return ModelRouteTier.humanizer


def _synthesis_payload(
    response_record: ResponseRecord,
    synthesis_context: SynthesisContext,
) -> JsonObject:
    return {
        "response_record": response_record.to_payload(),
        "synthesis_context": synthesis_context.to_payload(),
        "renderer_constraints": {
            "target": "Slack mrkdwn",
            "avoid": ["GitHub Markdown headings", "Markdown tables"],
        },
    }


def _synthesis_evidence_from_events(
    events: Sequence[TaskEvent],
    task: Task,
) -> list[SynthesisEvidence]:
    evidence: list[SynthesisEvidence] = []
    calls_by_id = _tool_calls_by_id(events)
    slack_ref = SlackRef(
        channel_id=task.slack_channel_id,
        thread_ts=task.slack_thread_ts,
        message_ts=task.slack_message_ts,
        user_id=task.slack_user_id,
    )
    for event in events:
        payload = event.payload
        if event.type is TaskEventType.tool_result:
            tool_call_id = _string(payload.get("tool_call_id")) or f"event-{event.id}"
            call = calls_by_id.get(tool_call_id, {})
            tool = _string(payload.get("tool")) or _string(call.get("tool")) or "tool"
            output = payload.get("output")
            content = _synthesis_tool_content(tool, output)
            if content is None:
                content = _tool_result_summary(output) or "Tool returned a result."
            urls = _extract_urls(output)
            evidence.append(
                SynthesisEvidence(
                    source_id=tool_call_id,
                    kind=_evidence_kind_for_tool(tool),
                    content=_shorten(content, max_chars=MAX_SYNTHESIS_EVIDENCE_CHARS),
                    trust=_trust_for_tool(tool),
                    confidence=_confidence_for_tool_result(payload, output),
                    tool=tool,
                    urls=urls,
                    slack_ref=slack_ref,
                    metadata=_synthesis_tool_metadata(payload, output),
                )
            )
        elif event.type is TaskEventType.error:
            failure = _response_failure_from_error_event(event)
            content = failure.message or failure.code or "Task recorded an error."
            evidence.append(
                SynthesisEvidence(
                    source_id=f"error-{event.id}",
                    kind=EvidenceKind.error,
                    content=_shorten(content, max_chars=MAX_SYNTHESIS_EVIDENCE_CHARS),
                    trust=EvidenceTrust.trusted,
                    confidence=1.0,
                    slack_ref=slack_ref,
                    metadata={
                        "source": failure.source,
                        "code": failure.code,
                        "recovery_action": failure.recovery_action,
                    },
                )
            )
    return evidence


def _synthesis_tool_content(tool: str, output: object) -> str | None:
    if not isinstance(output, dict):
        return None
    error = _tool_error_payload(output)
    if error is not None:
        return _string(error.get("message")) or "Tool reported an error."

    if tool == "inspect_memory":
        count = _optional_int(output.get("count")) or 0
        scope = _string(output.get("scope")) or "memory"
        facts = output.get("facts")
        if count == 0 or not isinstance(facts, list) or not facts:
            return f"Inspected {scope} memory and found no active facts."
        fact_summaries = [
            summary
            for fact in facts[:5]
            if isinstance(fact, dict)
            if (summary := _memory_fact_summary(fact)) is not None
        ]
        if fact_summaries:
            return (
                f"Inspected {scope} memory and found {count} active fact(s): "
                + "; ".join(fact_summaries)
            )
        return f"Inspected {scope} memory and found {count} active fact(s)."

    if tool == "recall_fact":
        key = _string(output.get("key")) or "requested key"
        if output.get("found") is False:
            return f"No active memory fact was found for {key}."
        value_text = _string(output.get("value_text"))
        if value_text:
            return f"Recalled memory fact {key}: {value_text}"
        return f"Recalled memory fact {key}."

    if tool == "forget_fact":
        key = _string(output.get("key")) or "requested key"
        forgotten_count = _optional_int(output.get("forgotten_count")) or 0
        if forgotten_count == 0:
            return f"No active memory fact matched {key}."
        return f"Forgot memory fact {key}."

    if tool == "remember_fact":
        status = _string(output.get("status"))
        key = _string(output.get("key")) or "memory fact"
        value_text = _string(output.get("value_text"))
        if status == "pending_confirmation":
            if value_text:
                return f"Proposed memory fact {key} for confirmation: {value_text}"
            return f"Proposed memory fact {key} for confirmation."

    if tool == "query_workspace_graph":
        return _workspace_graph_summary(output)

    summary = (
        _string(output.get("assistant_summary"))
        or _string(output.get("summary"))
        or _string(output.get("message"))
    )
    if summary:
        return summary
    titles = _result_titles(output)
    if titles:
        return "Tool returned these result titles: " + "; ".join(titles[:5])
    return None


def _memory_fact_summary(fact: JsonObject) -> str | None:
    key = _string(fact.get("key"))
    value_text = _string(fact.get("value_text"))
    if key and value_text:
        return f"{key}: {value_text}"
    if key:
        return key
    return value_text


def _workspace_graph_summary(output: JsonObject) -> str:
    entity_count = _optional_int(output.get("entity_count")) or 0
    edge_count = _optional_int(output.get("edge_count")) or 0
    destination = output.get("destination")
    destination_label = _workspace_graph_destination_label(destination)
    omitted_reasons = _string_list(output.get("omitted_reasons"))
    if entity_count == 0 and edge_count == 0:
        if omitted_reasons:
            return (
                f"Workspace context lookup for {destination_label} returned no "
                f"visible active graph rows ({', '.join(omitted_reasons)})."
            )
        return (
            f"Workspace context lookup for {destination_label} returned no "
            "visible active graph rows."
        )

    entity_bits = [
        bit
        for entity in output.get("entities", [])[:6]
        if isinstance(entity, dict)
        if (bit := _workspace_graph_entity_bit(entity)) is not None
    ]
    relationship_bits = [
        bit
        for relationship in output.get("relationships", [])[:4]
        if isinstance(relationship, dict)
        if (bit := _workspace_graph_relationship_bit(relationship)) is not None
    ]
    pieces = [
        (
            f"Workspace context lookup for {destination_label} returned "
            f"{entity_count} active graph {_plural('entity', entity_count)} "
            f"and {edge_count} {_plural('relationship', edge_count)}."
        )
    ]
    if entity_bits:
        pieces.append("Entities: " + "; ".join(entity_bits))
    if relationship_bits:
        pieces.append("Relationships: " + "; ".join(relationship_bits))
    if omitted_reasons:
        pieces.append("Limits: " + ", ".join(omitted_reasons))
    return " ".join(pieces)


def _workspace_graph_destination_label(destination: object) -> str:
    if not isinstance(destination, dict):
        return "this Slack surface"
    surface_type = _string(destination.get("surface_type")) or "surface"
    surface_id = _string(destination.get("surface_id"))
    if surface_id:
        return f"{surface_type} {surface_id}"
    return surface_type


def _workspace_graph_entity_bit(entity: JsonObject) -> str | None:
    label = _string(entity.get("display_name")) or _string(entity.get("canonical_key"))
    entity_type = _string(entity.get("entity_type"))
    confidence = _string(entity.get("confidence_score"))
    evidence_count = _optional_int(entity.get("evidence_count"))
    if label is None:
        return None
    suffixes: list[str] = []
    if entity_type:
        suffixes.append(entity_type)
    if confidence:
        suffixes.append(f"confidence {confidence}")
    if evidence_count:
        suffixes.append(f"{evidence_count} evidence item(s)")
    if suffixes:
        return f"{label} ({', '.join(suffixes)})"
    return label


def _workspace_graph_relationship_bit(relationship: JsonObject) -> str | None:
    source = _string(relationship.get("source_label"))
    target = _string(relationship.get("target_label"))
    relationship_type = _string(relationship.get("relationship_type"))
    confidence = _string(relationship.get("confidence_score"))
    if source is None or target is None or relationship_type is None:
        return None
    bit = f"{source} {relationship_type} {target}"
    if confidence:
        bit += f" (confidence {confidence})"
    return bit


def _plural(noun: str, count: int) -> str:
    return noun if count == 1 else f"{noun}s"


def _result_titles(value: object) -> list[str]:
    titles: list[str] = []

    def walk(item: object) -> None:
        if len(titles) >= 8:
            return
        if isinstance(item, dict):
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                titles.append(title.strip())
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return list(dict.fromkeys(titles))


def _evidence_kind_for_tool(tool: str) -> EvidenceKind:
    if tool == "query_workspace_graph":
        return EvidenceKind.workspace_graph
    if tool in MEMORY_TOOL_NAMES:
        return EvidenceKind.memory
    return EvidenceKind.tool_result


def _trust_for_tool(tool: str) -> EvidenceTrust:
    if tool == "query_workspace_graph":
        return EvidenceTrust.trusted
    if tool in MEMORY_TOOL_NAMES:
        return EvidenceTrust.trusted
    return EvidenceTrust.untrusted


def _confidence_for_tool_result(payload: JsonObject, output: object) -> float:
    if payload.get("recoverable") is True or _tool_error_payload(output) is not None:
        return 0.3
    if isinstance(output, dict) and output.get("successful") is False:
        return 0.3
    return 0.9


def _synthesis_tool_metadata(payload: JsonObject, output: object) -> JsonObject:
    metadata: JsonObject = {
        "artifact_count": payload.get("artifact_count"),
        "recoverable": payload.get("recoverable"),
        "error_category": payload.get("error_category"),
        "recovery_action": payload.get("recovery_action"),
    }
    if isinstance(output, dict):
        for key in (
            "scope",
            "key",
            "count",
            "found",
            "forgotten_count",
            "status",
            "successful",
            "entity_count",
            "edge_count",
            "destination",
            "omitted_count",
            "omitted_reasons",
        ):
            if key in output:
                metadata[key] = output[key]
    return {key: value for key, value in metadata.items() if value is not None}


def _approval_states_from_events(
    events: Sequence[TaskEvent],
) -> list[SynthesisApprovalState]:
    approvals: list[SynthesisApprovalState] = []
    for event in events:
        payload = event.payload
        if event.type is not TaskEventType.log:
            continue
        message = _string(payload.get("message"))
        if message not in {
            "tool_approval_required",
            "tool_approval_waiting",
            "tool_approval_decision",
        }:
            continue
        approvals.append(
            SynthesisApprovalState(
                tool=_string(payload.get("tool")),
                status=_approval_status(message, payload),
                reason=_string(payload.get("reason")),
                approver_user_id=_string(payload.get("user_id"))
                or _string(payload.get("approver_user_id")),
                metadata={
                    "message": message,
                    "tool_call_id": payload.get("tool_call_id"),
                    "decision": payload.get("decision"),
                },
            )
        )
    return approvals


def _approval_status(message: str, payload: JsonObject) -> str:
    if message == "tool_approval_decision":
        decision = _string(payload.get("decision"))
        return decision or "decided"
    if message == "tool_approval_waiting":
        return "waiting"
    return "required"


def _select_synthesis_outcome(
    *,
    raw_text: str,
    response_record: ResponseRecord,
    events: Sequence[TaskEvent],
    approvals: Sequence[SynthesisApprovalState],
) -> tuple[SynthesisOutcome, str]:
    if any(approval.status in {"required", "waiting"} for approval in approvals):
        return SynthesisOutcome.needs_approval, "tool approval is pending"
    if _has_no_result_signal(events) and _no_result_signal_is_terminal(response_record):
        return SynthesisOutcome.no_result, "tool evidence reported no matching result"
    if response_record.failures and response_record.actions_taken:
        return (
            SynthesisOutcome.partial_failure,
            "some tool or execution evidence failed",
        )
    if response_record.failures:
        return SynthesisOutcome.error, "task recorded failure evidence"
    if not raw_text.strip():
        return SynthesisOutcome.error, "raw response was empty"
    return SynthesisOutcome.ok, "task completed with user-facing answer"


def _has_no_result_signal(events: Sequence[TaskEvent]) -> bool:
    for event in events:
        if event.type is not TaskEventType.tool_result:
            continue
        payload = event.payload
        tool = _string(payload.get("tool"))
        output = payload.get("output")
        if not isinstance(output, dict):
            continue
        if tool == "forget_fact" and output.get("forgotten_count") == 0:
            return True
        if tool == "recall_fact" and output.get("found") is False:
            return True
        if tool == "inspect_memory" and output.get("count") == 0:
            return True
    return False


def _no_result_signal_is_terminal(response_record: ResponseRecord) -> bool:
    """Return whether a no-match tool result should dominate synthesis.

    Memory no-match is terminal for direct memory operations like "forget X" or
    "recall Y". It is not terminal for context/profile answers where memory is
    only one evidence source alongside graph, Slack history, or other tools.
    """

    if response_record.response_mode is not ResponseMode.memory_recall:
        return False
    tool_names = {action.tool for action in response_record.actions_taken}
    return bool(tool_names) and tool_names.issubset(MEMORY_TOOL_NAMES)


def _synthesis_uncertainty(
    *,
    response_record: ResponseRecord,
    outcome: SynthesisOutcome,
    outcome_reason: str,
) -> list[str]:
    uncertainty = list(response_record.uncertainties)
    if outcome in {
        SynthesisOutcome.no_result,
        SynthesisOutcome.partial_failure,
        SynthesisOutcome.needs_approval,
        SynthesisOutcome.error,
    }:
        uncertainty.append(outcome_reason)
    return list(dict.fromkeys(uncertainty))


def _default_forbidden_claims(outcome: SynthesisOutcome) -> list[str]:
    forbidden = [
        "Do not expose raw JSON, stack traces, hidden prompts, or tool payloads.",
        "Do not fabricate source coverage, tool results, or current facts.",
        "Do not mention Slack users or channels unless represented by typed refs.",
    ]
    if outcome is SynthesisOutcome.no_result:
        forbidden.append("Do not claim the requested item was found or changed.")
    if outcome is SynthesisOutcome.needs_approval:
        forbidden.append("Do not claim the pending action already ran.")
    if outcome in {SynthesisOutcome.partial_failure, SynthesisOutcome.error}:
        forbidden.append("Do not hide failed or incomplete work.")
    return forbidden


def _tool_calls_by_id(events: Sequence[TaskEvent]) -> dict[str, JsonObject]:
    calls_by_id: dict[str, JsonObject] = {}
    for event in events:
        payload = event.payload
        if event.type is TaskEventType.tool_call:
            tool_call_id = _string(payload.get("tool_call_id"))
            if tool_call_id is None:
                continue
            calls_by_id[tool_call_id] = {
                "tool": _string(payload.get("tool")),
                "argument_keys": _string_list(payload.get("argument_keys")),
            }
    return calls_by_id


def _response_items_from_tool_result(
    event: TaskEvent,
    calls_by_id: dict[str, JsonObject],
) -> tuple[ResponseAction | None, ResponseEvidence | None, ResponseFailure | None]:
    payload = event.payload
    tool_call_id = _string(payload.get("tool_call_id")) or f"event-{event.id}"
    call = calls_by_id.get(tool_call_id, {})
    tool = _string(payload.get("tool")) or _string(call.get("tool")) or "tool"
    output = payload.get("output")
    recoverable = _optional_bool(payload.get("recoverable"))
    error = _tool_error_payload(output)
    status = "failed" if error is not None or recoverable is True else "succeeded"
    action = ResponseAction(
        tool=tool,
        status=status,
        argument_keys=_string_list(call.get("argument_keys")),
        summary=_tool_result_summary(output),
    )
    urls = _extract_urls(output)
    evidence = ResponseEvidence(
        source_type="tool_result",
        source_id=tool_call_id,
        tool=tool,
        urls=urls,
        preview=_output_preview(output),
    )
    failure = None
    if error is not None:
        failure = ResponseFailure(
            source=tool,
            code=_string(error.get("code")),
            message=_string(error.get("message")),
            recoverable=_optional_bool(error.get("recoverable")),
            recovery_action=_string(error.get("recovery_action"))
            or _string(payload.get("recovery_action")),
        )
    return action, evidence, failure


def _response_failure_from_error_event(event: TaskEvent) -> ResponseFailure:
    payload = event.payload
    return ResponseFailure(
        source=_string(payload.get("phase")) or "task",
        code=_string(payload.get("type")),
        message=_string(payload.get("message")) or _string(payload.get("error")),
        recoverable=False,
        recovery_action=_string(payload.get("recovery_action")),
    )


def _artifact_summary(session: Session, task: Task) -> list[ResponseArtifact]:
    artifacts = session.scalars(
        select(Artifact)
        .where(Artifact.task_id == task.id)
        .order_by(Artifact.created_at)
    )
    return [
        ResponseArtifact(
            filename=artifact.filename,
            mime_type=artifact.mime_type,
            size_bytes=artifact.size_bytes,
            posted=artifact.posted_at is not None,
        )
        for artifact in artifacts
    ]


def _select_response_mode(
    *,
    raw_text: str,
    actions: Sequence[ResponseAction],
    evidence: Sequence[ResponseEvidence],
    artifacts: Sequence[ResponseArtifact],
    failures: Sequence[ResponseFailure],
) -> ResponseMode:
    tool_names = {action.tool for action in actions}
    if artifacts:
        return ResponseMode.artifact_delivery
    if failures:
        return ResponseMode.failure_recovery
    if "slack_file_read" in tool_names:
        return ResponseMode.file_analysis
    if "query_workspace_graph" in tool_names:
        return ResponseMode.context_answer
    if tool_names & MEMORY_TOOL_NAMES:
        return ResponseMode.memory_recall
    if _has_research_evidence(tool_names, evidence):
        return ResponseMode.research_summary
    if len(actions) >= 2 or len(raw_text) > 1800:
        return ResponseMode.multi_step_recap
    return ResponseMode.quick_answer


def _select_response_shape(
    *,
    user_request: str,
    raw_text: str,
    response_mode: ResponseMode,
    actions: Sequence[ResponseAction],
    evidence: Sequence[ResponseEvidence],
    artifacts: Sequence[ResponseArtifact],
    failures: Sequence[ResponseFailure],
) -> ResponseShapeProfile:
    del raw_text, actions, evidence
    request = user_request.casefold()
    framework_hint = _framework_hint(request)

    if response_mode is ResponseMode.failure_recovery or failures:
        return _shape_profile(
            ResponseShape.failure_note,
            selected_reason="task has failures or recoverable caveats",
            framework_hint=framework_hint,
        )
    if response_mode is ResponseMode.artifact_delivery or artifacts:
        return _shape_profile(
            ResponseShape.document_delivery,
            selected_reason="task produced one or more artifacts",
            framework_hint=framework_hint,
        )
    if response_mode is ResponseMode.memory_recall:
        return _shape_profile(
            ResponseShape.memory_note,
            selected_reason="task is about memory recall or memory state",
            framework_hint=framework_hint,
        )
    if response_mode is ResponseMode.context_answer:
        return _shape_profile(
            ResponseShape.context_profile,
            selected_reason="task uses workspace context or graph-backed profile evidence",
            framework_hint=framework_hint,
        )
    if _is_comparison_request(request):
        return _shape_profile(
            ResponseShape.comparison_memo,
            selected_reason="user asked for a comparison or choice",
            framework_hint=framework_hint,
        )
    if _is_analyst_audit_request(request):
        return _shape_profile(
            ResponseShape.analyst_audit,
            selected_reason="user asked for an audit, review, critique, or framework analysis",
            framework_hint=framework_hint,
        )
    if response_mode is ResponseMode.file_analysis:
        return _shape_profile(
            ResponseShape.file_review,
            selected_reason="task uses file analysis evidence",
            framework_hint=framework_hint,
        )
    if response_mode is ResponseMode.research_summary:
        return _shape_profile(
            ResponseShape.research_brief,
            selected_reason="task uses research or source evidence",
            framework_hint=framework_hint,
        )
    if response_mode is ResponseMode.multi_step_recap:
        return _shape_profile(
            ResponseShape.status_recap,
            selected_reason="task involved multiple actions or a long recap",
            framework_hint=framework_hint,
        )
    return _shape_profile(
        ResponseShape.quick_reply,
        selected_reason="default concise Slack reply",
        framework_hint=framework_hint,
    )


def _shape_profile(
    shape: ResponseShape,
    *,
    selected_reason: str,
    framework_hint: str | None,
) -> ResponseShapeProfile:
    if shape is ResponseShape.analyst_audit:
        return ResponseShapeProfile(
            shape=shape,
            label="Analyst audit",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=[
                "bottom_line",
                "scope",
                "evaluation_lens",
                "ranked_findings",
                "evidence_or_limits",
                "concrete_recommendations",
                "highest_leverage_move",
                "next_step",
            ],
            quality_checks=[
                "findings are specific and ranked",
                "recommendations are concrete enough to act on",
                "scope and evidence limits are explicit",
            ],
            avoid=[
                "generic advice",
                "unranked laundry lists",
                "invented source coverage",
            ],
        )
    if shape is ResponseShape.comparison_memo:
        return ResponseShapeProfile(
            shape=shape,
            label="Comparison memo",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=[
                "recommendation",
                "scope",
                "tradeoffs",
                "when_to_choose_each",
                "decision_risk",
                "next_step",
            ],
            quality_checks=[
                "recommendation is explicit",
                "tradeoffs explain why, not just what",
                "decision criteria are visible",
            ],
            avoid=["hedging without a pick", "false precision", "tables in Slack"],
        )
    if shape is ResponseShape.research_brief:
        return ResponseShapeProfile(
            shape=shape,
            label="Research brief",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=[
                "bottom_line",
                "top_findings",
                "source_context",
                "limits",
                "next_step",
            ],
            quality_checks=[
                "findings synthesize across sources",
                "source limitations are visible",
                "answer avoids link dumping",
            ],
            avoid=["raw search-result lists", "unsupported recency claims"],
        )
    if shape is ResponseShape.file_review:
        return ResponseShapeProfile(
            shape=shape,
            label="File review",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=[
                "bottom_line",
                "file_scope",
                "key_points",
                "gaps_or_caveats",
                "next_step",
            ],
            quality_checks=[
                "file scope is explicit",
                "summary distinguishes content from interpretation",
            ],
            avoid=["pretending unseen files were reviewed"],
        )
    if shape is ResponseShape.document_delivery:
        return ResponseShapeProfile(
            shape=shape,
            label="Document delivery",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=["artifact", "what_changed", "review_prompt"],
            quality_checks=["message is short because artifact carries detail"],
            avoid=["repeating the whole document in Slack"],
        )
    if shape is ResponseShape.status_recap:
        return ResponseShapeProfile(
            shape=shape,
            label="Status recap",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=["what_matters", "groups", "open_items"],
            quality_checks=["recap is grouped by topic, not chronological noise"],
            avoid=["activity logs", "overclaiming blockers"],
        )
    if shape is ResponseShape.memory_note:
        return ResponseShapeProfile(
            shape=shape,
            label="Memory note",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=[
                "remembered_fact_or_limit",
                "scope",
                "natural_no_match_explanation_when_relevant",
            ],
            quality_checks=["memory scope is clear"],
            avoid=["raw ids unless necessary", "raw tool-result phrasing"],
        )
    if shape is ResponseShape.context_profile:
        return ResponseShapeProfile(
            shape=shape,
            label="Context profile",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=[
                "what_kortny_knows",
                "why_kortny_believes_it",
                "confidence_or_limits",
            ],
            quality_checks=[
                "clearly separates observed activity from inferred profile",
                "uses provenance naturally instead of implementation labels",
                "does not overstate stale or thin context",
            ],
            avoid=[
                "starting with graph internals",
                "raw ids unless needed for disambiguation",
                "claiming remembered facts were found when only history was used",
            ],
        )
    if shape is ResponseShape.failure_note:
        return ResponseShapeProfile(
            shape=shape,
            label="Failure note",
            selected_reason=selected_reason,
            framework_hint=framework_hint,
            required_elements=["what_failed", "impact", "next_safe_step"],
            quality_checks=["failure is user-safe and non-diagnostic by default"],
            avoid=["stack traces", "blamey language"],
        )
    return ResponseShapeProfile(
        shape=shape,
        label="Quick reply",
        selected_reason=selected_reason,
        framework_hint=framework_hint,
        required_elements=["answer"],
        quality_checks=["answer is direct and concise"],
        avoid=["boilerplate"],
    )


def _is_comparison_request(value: str) -> bool:
    return any(phrase in value for phrase in COMPARISON_PHRASES)


def _is_analyst_audit_request(value: str) -> bool:
    return any(phrase in value for phrase in ANALYST_AUDIT_PHRASES)


def _framework_hint(value: str) -> str | None:
    if "framework" in value or "cpt" in value:
        return "Use the evaluation lens the user requested; name it if the raw answer supports it."
    return None


def _has_research_evidence(
    tool_names: set[str],
    evidence: Sequence[ResponseEvidence],
) -> bool:
    if "web_search" in tool_names:
        return True
    if any(tool.startswith("composio_") for tool in tool_names):
        return True
    return any(item.urls for item in evidence)


def _suggested_next_actions(
    response_mode: ResponseMode,
    failures: Sequence[ResponseFailure],
    artifacts: Sequence[ResponseArtifact],
) -> list[str]:
    if response_mode is ResponseMode.failure_recovery and failures:
        return ["retry with corrected input", "use an alternate path", "ask for access"]
    if response_mode is ResponseMode.artifact_delivery and artifacts:
        return ["review the artifact", "request a revision"]
    if response_mode is ResponseMode.research_summary:
        return ["deepen the comparison", "turn findings into a brief"]
    return []


COMPARISON_PHRASES = frozenset(
    {
        "compare",
        "tradeoff",
        "tradeoffs",
        "which one",
        "which two",
        "what would you choose",
        "recommend",
        "recommendation",
        "pick",
        "best option",
        "pros and cons",
    }
)

ANALYST_AUDIT_PHRASES = frozenset(
    {
        "audit",
        "review",
        "critique",
        "analyze",
        "analyse",
        "assess",
        "evaluate",
        "framework",
        "cpt",
        "gaps",
        "reframe",
        "strategy",
        "competitive analysis",
        "website",
        "copy",
        "positioning",
    }
)


def _task_events(session: Session, task: Task) -> Sequence[TaskEvent]:
    return tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def _extract_urls(value: object) -> list[str]:
    urls: list[str] = []

    def walk(item: object) -> None:
        if len(urls) >= 8:
            return
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.append(url)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return list(dict.fromkeys(urls))


def _tool_result_summary(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("summary"), str):
        return _shorten(value["summary"], max_chars=240)
    if isinstance(value.get("message"), str):
        return _shorten(value["message"], max_chars=240)
    error = _tool_error_payload(value)
    if error is not None and isinstance(error.get("message"), str):
        return _shorten(error["message"], max_chars=240)
    return None


def _output_preview(value: object) -> str | None:
    if value is None:
        return None
    serialized = json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)
    return _shorten(serialized, max_chars=MAX_TRACE_OUTPUT_CHARS)


def _tool_error_payload(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        return error
    if isinstance(error, str) and error.strip():
        return {
            "message": error.strip(),
            "code": _string(value.get("status_code")),
            "recoverable": _optional_bool(value.get("recoverable")),
        }
    if value.get("successful") is False:
        message = _string(value.get("message"))
        data = value.get("data")
        if message is None and isinstance(data, dict):
            message = _string(data.get("message"))
        return {
            "message": message or "Tool reported an unsuccessful result.",
            "code": _string(value.get("status_code")),
            "recoverable": _optional_bool(value.get("recoverable")),
        }
    return None


def _response_status(failures: Sequence[ResponseFailure]) -> str:
    if failures:
        return "ready_to_post_with_caveats"
    return "ready_to_post"


def _shorten(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
