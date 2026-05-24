"""Context assembly for agent task execution."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent.thread_context import (
    ThreadTranscriptMessage,
    ThreadTranscriptProvider,
)
from kortny.db.models import Artifact, Task, TaskEvent, TaskEventType
from kortny.llm import ChatMessage
from kortny.memory import EpisodeService, Fact, RelevantEpisode, WorkspaceStateService
from kortny.observability import observe_task_event
from kortny.tasks import TaskService

DEFAULT_THREAD_CONTEXT_MAX_CHARS = 12_000
DEFAULT_THREAD_CONTEXT_RECENT_TASKS = 3
DEFAULT_THREAD_TRANSCRIPT_LIMIT = 30
DEFAULT_KNOWN_FACTS_MAX_CHARS = 4_000
DEFAULT_EPISODE_CONTEXT_MAX_CHARS = 4_000
DEFAULT_EPISODE_CONTEXT_LIMIT = 5
IMMEDIATE_PRIOR_INPUT_MAX_CHARS = 500
IMMEDIATE_PRIOR_RESULT_MAX_CHARS = 1_800
SLACK_FILES_BLOCK_RE = re.compile(r"<slack_files>\s*(.*?)\s*</slack_files>", re.S)
SLACK_FILE_ID_RE = re.compile(r"^\s*-\s+id:\s*(\S+)\s*$", re.M)
THREAD_CONTEXT_EVENT_TYPES = {
    TaskEventType.llm_call,
    TaskEventType.tool_call,
    TaskEventType.tool_result,
    TaskEventType.error,
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextFact:
    """A durable fact selected for the task context."""

    fact_id: uuid.UUID
    scope_type: str
    scope_id: str | None
    key: str


@dataclass(frozen=True, slots=True)
class ContextTask:
    """A prior task selected for thread context."""

    task_id: uuid.UUID
    status: str
    slack_channel_id: str
    slack_thread_ts: str | None


@dataclass(frozen=True, slots=True)
class ContextEpisode:
    """A prior episode selected for episodic context."""

    episode_id: uuid.UUID
    task_id: uuid.UUID
    relation: str
    outcome: str


@dataclass(frozen=True, slots=True)
class ContextArtifact:
    """An artifact selected for thread context."""

    artifact_id: uuid.UUID
    task_id: uuid.UUID
    filename: str
    slack_file_id: str | None
    mime_type: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class ContextAcknowledgement:
    """A visible Slack acknowledgement already posted for the task."""

    message_ts: str | None
    text: str


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Configured and actual context budget usage."""

    system_prompt_chars: int
    known_facts_max_chars: int
    known_facts_chars: int
    thread_context_max_chars: int
    prior_context_chars: int
    thread_context_recent_tasks: int
    thread_transcript_limit: int
    episode_context_max_chars: int
    episode_context_chars: int
    episode_context_limit: int


@dataclass(frozen=True, slots=True)
class ContextOmission:
    """Context omitted or compacted while building the prompt."""

    kind: str
    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class ContextPackage:
    """Messages plus structured metadata for a task context build."""

    messages: tuple[ChatMessage, ...]
    selected_facts: tuple[ContextFact, ...]
    selected_prior_tasks: tuple[ContextTask, ...]
    selected_episodes: tuple[ContextEpisode, ...]
    selected_artifacts: tuple[ContextArtifact, ...]
    acknowledgement: ContextAcknowledgement | None
    budget: ContextBudget
    omissions: tuple[ContextOmission, ...]


@dataclass(frozen=True, slots=True)
class _KnownFactsContext:
    content: str | None
    selected_facts: tuple[ContextFact, ...]
    omissions: tuple[ContextOmission, ...]


@dataclass(frozen=True, slots=True)
class _PriorContext:
    content: str | None
    selected_prior_tasks: tuple[ContextTask, ...]
    selected_artifacts: tuple[ContextArtifact, ...]
    omissions: tuple[ContextOmission, ...]


@dataclass(frozen=True, slots=True)
class _EpisodeContext:
    content: str | None
    selected_episodes: tuple[ContextEpisode, ...]
    omissions: tuple[ContextOmission, ...]


class ContextAssembler:
    """Builds the LLM message context for a task."""

    def __init__(
        self,
        *,
        session: Session,
        task_service: TaskService | None = None,
        system_prompt: str | None = None,
        thread_transcript_provider: ThreadTranscriptProvider | None = None,
        thread_context_max_chars: int = DEFAULT_THREAD_CONTEXT_MAX_CHARS,
        thread_context_recent_tasks: int = DEFAULT_THREAD_CONTEXT_RECENT_TASKS,
        thread_transcript_limit: int = DEFAULT_THREAD_TRANSCRIPT_LIMIT,
        known_facts_max_chars: int = DEFAULT_KNOWN_FACTS_MAX_CHARS,
        episode_context_max_chars: int = DEFAULT_EPISODE_CONTEXT_MAX_CHARS,
        episode_context_limit: int = DEFAULT_EPISODE_CONTEXT_LIMIT,
    ) -> None:
        if thread_context_max_chars < 1:
            raise ValueError("thread_context_max_chars must be at least 1")
        if thread_context_recent_tasks < 1:
            raise ValueError("thread_context_recent_tasks must be at least 1")
        if thread_transcript_limit < 0:
            raise ValueError("thread_transcript_limit cannot be negative")
        if known_facts_max_chars < 0:
            raise ValueError("known_facts_max_chars cannot be negative")
        if episode_context_max_chars < 0:
            raise ValueError("episode_context_max_chars cannot be negative")
        if episode_context_limit < 0:
            raise ValueError("episode_context_limit cannot be negative")

        self.session = session
        self.task_service = task_service or TaskService(session)
        self.system_prompt = system_prompt
        self.thread_transcript_provider = thread_transcript_provider
        self.thread_context_max_chars = thread_context_max_chars
        self.thread_context_recent_tasks = thread_context_recent_tasks
        self.thread_transcript_limit = thread_transcript_limit
        self.known_facts_max_chars = known_facts_max_chars
        self.episode_context_max_chars = episode_context_max_chars
        self.episode_context_limit = episode_context_limit
        self.workspace_state_service = WorkspaceStateService(
            session,
            task_service=self.task_service,
        )
        self.episode_service = EpisodeService(session, task_service=self.task_service)

    def build_for_task(self, task: Task) -> ContextPackage:
        """Build prompt messages and context-selection metadata."""

        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        acknowledgement = self._acknowledgement_context(task)
        if acknowledgement is not None:
            messages.append(
                ChatMessage(
                    role="system",
                    content=_render_acknowledgement_context(acknowledgement),
                )
            )

        known_facts = self._known_facts_context(task)
        if known_facts.content:
            messages.append(ChatMessage(role="system", content=known_facts.content))

        prior_context = self._prior_context(task)
        if prior_context.content:
            messages.append(ChatMessage(role="system", content=prior_context.content))

        episode_context = self._episode_context(task)
        if episode_context.content:
            messages.append(ChatMessage(role="system", content=episode_context.content))

        messages.append(ChatMessage(role="user", content=task.input))

        package = ContextPackage(
            messages=tuple(messages),
            selected_facts=known_facts.selected_facts,
            selected_prior_tasks=prior_context.selected_prior_tasks,
            selected_episodes=episode_context.selected_episodes,
            selected_artifacts=prior_context.selected_artifacts,
            acknowledgement=acknowledgement,
            budget=ContextBudget(
                system_prompt_chars=len(self.system_prompt or ""),
                known_facts_max_chars=self.known_facts_max_chars,
                known_facts_chars=len(known_facts.content or ""),
                thread_context_max_chars=self.thread_context_max_chars,
                prior_context_chars=len(prior_context.content or ""),
                thread_context_recent_tasks=self.thread_context_recent_tasks,
                thread_transcript_limit=self.thread_transcript_limit,
                episode_context_max_chars=self.episode_context_max_chars,
                episode_context_chars=len(episode_context.content or ""),
                episode_context_limit=self.episode_context_limit,
            ),
            omissions=(
                known_facts.omissions
                + prior_context.omissions
                + episode_context.omissions
            ),
        )
        self._record_context_assembled(task, package)
        return package

    def _record_context_assembled(
        self,
        task: Task,
        package: ContextPackage,
    ) -> None:
        observe_task_event(
            self.task_service,
            task,
            "context_assembled",
            logger=logger,
            message_count=len(package.messages),
            selected_fact_ids=[str(fact.fact_id) for fact in package.selected_facts],
            selected_fact_keys=[fact.key for fact in package.selected_facts],
            selected_episode_ids=[
                str(episode.episode_id) for episode in package.selected_episodes
            ],
            selected_episode_task_ids=[
                str(episode.task_id) for episode in package.selected_episodes
            ],
            selected_episode_relations=[
                episode.relation for episode in package.selected_episodes
            ],
            selected_prior_task_ids=[
                str(prior.task_id) for prior in package.selected_prior_tasks
            ],
            selected_artifact_ids=[
                str(artifact.artifact_id) for artifact in package.selected_artifacts
            ],
            acknowledgement_present=package.acknowledgement is not None,
            acknowledgement_message_ts=package.acknowledgement.message_ts
            if package.acknowledgement
            else None,
            context_chars=_context_chars(package),
            context_budget=_context_budget_payload(package.budget),
            context_omissions=[
                {
                    "kind": omission.kind,
                    "reason": omission.reason,
                    "count": omission.count,
                }
                for omission in package.omissions
            ],
        )

    def _acknowledgement_context(self, task: Task) -> ContextAcknowledgement | None:
        event = self.session.scalars(
            select(TaskEvent)
            .where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == TaskEventType.message_posted,
                TaskEvent.payload["purpose"].as_string() == "acknowledgement",
            )
            .order_by(TaskEvent.seq.desc())
            .limit(1)
        ).first()
        if event is None:
            return None

        text = event.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        message_ts = event.payload.get("message_ts")
        return ContextAcknowledgement(
            message_ts=message_ts if isinstance(message_ts, str) else None,
            text=text.strip(),
        )

    def _known_facts_context(self, task: Task) -> _KnownFactsContext:
        if self.known_facts_max_chars == 0:
            return _KnownFactsContext(
                content=None,
                selected_facts=(),
                omissions=(ContextOmission("known_facts", "budget_disabled", 0),),
            )

        facts = self._scoped_known_facts(task)
        if not facts:
            return _KnownFactsContext(content=None, selected_facts=(), omissions=())

        kept = list(facts)
        dropped = 0
        rendered = _render_known_facts(kept)
        while len(rendered) > self.known_facts_max_chars and kept:
            oldest = min(kept, key=lambda fact: (fact.created_at, fact.id))
            kept.remove(oldest)
            dropped += 1
            rendered = _render_known_facts(kept)

        omissions: list[ContextOmission] = []
        if dropped:
            omissions.append(
                ContextOmission(
                    "known_facts",
                    "budget_exceeded_drop_oldest",
                    dropped,
                )
            )
        if not kept or len(rendered) > self.known_facts_max_chars:
            omissions.append(
                ContextOmission(
                    "known_facts",
                    "budget_too_small_for_remaining_facts",
                    len(kept),
                )
            )
            return _KnownFactsContext(
                content=None,
                selected_facts=(),
                omissions=tuple(omissions),
            )

        return _KnownFactsContext(
            content=rendered,
            selected_facts=tuple(_context_fact(fact) for fact in kept),
            omissions=tuple(omissions),
        )

    def _scoped_known_facts(self, task: Task) -> list[Fact]:
        facts_by_key: dict[str, Fact] = {}
        for fact in self.workspace_state_service.list(
            task.installation_id,
            scope_type="workspace",
            scope_id=None,
        ):
            facts_by_key[fact.key] = fact

        if not _is_dm_channel(task.slack_channel_id):
            for fact in self.workspace_state_service.list(
                task.installation_id,
                scope_type="channel",
                scope_id=task.slack_channel_id,
            ):
                facts_by_key[fact.key] = fact

        for fact in self.workspace_state_service.list(
            task.installation_id,
            scope_type="user",
            scope_id=task.slack_user_id,
        ):
            facts_by_key[fact.key] = fact

        return list(facts_by_key.values())

    def _prior_context(self, task: Task) -> _PriorContext:
        thread_ts = task.slack_thread_ts
        if not thread_ts:
            return _PriorContext(
                content=None,
                selected_prior_tasks=(),
                selected_artifacts=(),
                omissions=(),
            )

        thread_tasks = self.task_service.list_by_thread(
            task.slack_channel_id, thread_ts
        )
        prior_tasks = _tasks_before(thread_tasks, task)
        if not prior_tasks:
            return _PriorContext(
                content=None,
                selected_prior_tasks=(),
                selected_artifacts=(),
                omissions=(),
            )

        transcript = self._fetch_thread_transcript(task)
        detailed = self._render_prior_context(
            prior_tasks,
            transcript=transcript,
            include_events=True,
            compacted=False,
        )
        selected_prior_tasks = tuple(
            _context_task(prior_task) for prior_task in prior_tasks
        )
        detailed_content = detailed.content or ""
        if len(detailed_content) <= self.thread_context_max_chars:
            return _PriorContext(
                content=detailed_content,
                selected_prior_tasks=selected_prior_tasks,
                selected_artifacts=detailed.selected_artifacts,
                omissions=detailed.omissions,
            )

        compact = self._render_prior_context(
            prior_tasks,
            transcript=transcript,
            include_events=False,
            compacted=True,
        )
        compacted_content = _fit_context_to_budget(
            compact.content or "",
            self.thread_context_max_chars,
        )
        omissions = compact.omissions + (
            ContextOmission("prior_context", "compacted_to_budget", len(prior_tasks)),
        )
        return _PriorContext(
            content=compacted_content,
            selected_prior_tasks=selected_prior_tasks,
            selected_artifacts=compact.selected_artifacts,
            omissions=omissions,
        )

    def _episode_context(self, task: Task) -> _EpisodeContext:
        if self.episode_context_max_chars == 0 or self.episode_context_limit == 0:
            return _EpisodeContext(
                content=None,
                selected_episodes=(),
                omissions=(ContextOmission("episodes", "budget_disabled", 0),),
            )

        episodes = list(
            self.episode_service.relevant_for_task(
                task,
                limit=self.episode_context_limit,
            )
        )
        if not episodes:
            return _EpisodeContext(content=None, selected_episodes=(), omissions=())

        dropped = 0
        rendered = _render_episode_context(episodes)
        while len(rendered) > self.episode_context_max_chars and episodes:
            episodes.pop()
            dropped += 1
            rendered = _render_episode_context(episodes)

        omissions: list[ContextOmission] = []
        if dropped:
            omissions.append(
                ContextOmission(
                    "episodes",
                    "budget_exceeded_drop_lowest_relevance",
                    dropped,
                )
            )
        if not episodes or len(rendered) > self.episode_context_max_chars:
            omissions.append(
                ContextOmission(
                    "episodes",
                    "budget_too_small_for_remaining_episodes",
                    len(episodes),
                )
            )
            return _EpisodeContext(
                content=None,
                selected_episodes=(),
                omissions=tuple(omissions),
            )

        return _EpisodeContext(
            content=rendered,
            selected_episodes=tuple(_context_episode(item) for item in episodes),
            omissions=tuple(omissions),
        )

    def _fetch_thread_transcript(
        self,
        task: Task,
    ) -> tuple[ThreadTranscriptMessage, ...]:
        if self.thread_transcript_provider is None or self.thread_transcript_limit == 0:
            return ()
        if not task.slack_thread_ts:
            return ()
        if _is_dm_conversation_context_key(task.slack_channel_id, task.slack_thread_ts):
            return ()

        try:
            return self.thread_transcript_provider.fetch_thread_messages(
                channel_id=task.slack_channel_id,
                thread_ts=task.slack_thread_ts,
                limit=self.thread_transcript_limit,
            )
        except Exception as exc:
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "thread_transcript_unavailable",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return ()

    def _render_prior_context(
        self,
        prior_tasks: Sequence[Task],
        *,
        transcript: Sequence[ThreadTranscriptMessage],
        include_events: bool,
        compacted: bool,
    ) -> _PriorContext:
        if compacted:
            lines = [
                "<prior_context>",
                "Compacted follow-up context. Resolve references from these "
                "summaries; reuse Slack file IDs with slack_file_read.",
            ]
        else:
            lines = [
                "<prior_context>",
                "This task is a follow-up in the same Slack thread. Use this context "
                'to resolve references like "it", "that", "the PDF", and '
                '"your source". If prior context includes Slack file IDs, you can '
                "reuse those IDs with slack_file_read. Do not treat this as "
                "cross-thread memory. For document revision requests, prefer the "
                "newest generated artifact over older original attachments. If "
                "the current message is a short reply to "
                "the immediately previous assistant question, treat it as the "
                "answer to that question and continue the pending task.",
            ]
        if compacted:
            lines.append(
                "Context was compacted to stay within the configured token budget; "
                "older task event details were omitted."
            )

        lines.extend(_immediate_previous_exchange_lines(prior_tasks[-1]))

        selected_artifacts: list[ContextArtifact] = []
        omissions: list[ContextOmission] = []
        if include_events:
            older_tasks = prior_tasks[: -self.thread_context_recent_tasks]
            recent_tasks = prior_tasks[-self.thread_context_recent_tasks :]
            if older_tasks:
                omissions.append(
                    ContextOmission(
                        "prior_task_details",
                        "older_tasks_summary_only",
                        len(older_tasks),
                    )
                )
                lines.append("")
                lines.append("Older prior task summaries:")
                for index, prior_task in enumerate(older_tasks, start=1):
                    lines.append(_task_summary_line(index, prior_task))
            lines.append("")
            lines.append("Recent prior task details:")
            start_index = len(older_tasks) + 1
            for index, prior_task in enumerate(recent_tasks, start=start_index):
                detail = self._prior_task_detail_lines(index, prior_task)
                lines.extend(detail.lines)
                selected_artifacts.extend(detail.selected_artifacts)
        else:
            omissions.append(
                ContextOmission(
                    "prior_task_events",
                    "compacted_context_omits_event_details",
                    len(prior_tasks),
                )
            )
            lines.append("")
            lines.append("Prior task summaries:")
            for index, prior_task in enumerate(prior_tasks, start=1):
                lines.append(_task_summary_line(index, prior_task))

        if transcript:
            lines.append("")
            lines.append("Slack thread transcript:")
            for message in transcript:
                lines.append(_transcript_line(message))

        lines.append("</prior_context>")
        return _PriorContext(
            content="\n".join(lines),
            selected_prior_tasks=tuple(_context_task(task) for task in prior_tasks),
            selected_artifacts=tuple(selected_artifacts),
            omissions=tuple(omissions),
        )

    def _prior_task_detail_lines(self, index: int, task: Task) -> _PriorTaskDetail:
        lines = [_task_summary_line(index, task)]
        artifact_lines = self._artifact_detail_lines(task, indent="  ")
        lines.extend(artifact_lines.lines)
        slack_files_block = _slack_files_block(task.input)
        if slack_files_block is not None:
            lines.append("  attached Slack files from original request:")
            for line in slack_files_block.splitlines():
                lines.append(f"  {line}")
        events = self._context_events(task)
        if events:
            lines.append("  events:")
            for event in events:
                lines.append(
                    f"  - {event.type.value}: {_shorten(_json_dumps(event.payload), max_chars=600)}"
                )
        return _PriorTaskDetail(
            lines=lines,
            selected_artifacts=artifact_lines.selected_artifacts,
        )

    def _context_events(self, task: Task) -> list[TaskEvent]:
        return list(
            self.session.scalars(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.type.in_(THREAD_CONTEXT_EVENT_TYPES),
                )
                .order_by(TaskEvent.seq)
            )
        )

    def _artifact_detail_lines(
        self,
        task: Task,
        *,
        indent: str = "",
    ) -> _ArtifactDetail:
        artifacts = self._artifacts(task)
        if not artifacts:
            return _ArtifactDetail(lines=[], selected_artifacts=())

        lines = [f"{indent}generated artifacts:"]
        for artifact in artifacts:
            details = [
                f"artifact_id={artifact.id}",
                f"filename={_quote(artifact.filename)}",
            ]
            if artifact.slack_file_id:
                details.append(f"slack_file_id={artifact.slack_file_id}")
            if artifact.mime_type:
                details.append(f"mime_type={_quote(artifact.mime_type)}")
            if artifact.size_bytes is not None:
                details.append(f"size_bytes={artifact.size_bytes}")
            lines.append(f"{indent}- " + " ".join(details))
        return _ArtifactDetail(
            lines=lines,
            selected_artifacts=tuple(
                _context_artifact(artifact) for artifact in artifacts
            ),
        )

    def _artifacts(self, task: Task) -> list[Artifact]:
        return list(
            self.session.scalars(
                select(Artifact)
                .where(Artifact.task_id == task.id)
                .order_by(Artifact.created_at)
            )
        )


@dataclass(frozen=True, slots=True)
class _PriorTaskDetail:
    lines: list[str]
    selected_artifacts: tuple[ContextArtifact, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactDetail:
    lines: list[str]
    selected_artifacts: tuple[ContextArtifact, ...]


def _context_fact(fact: Fact) -> ContextFact:
    return ContextFact(
        fact_id=fact.id,
        scope_type=fact.scope_type,
        scope_id=fact.scope_id,
        key=fact.key,
    )


def _context_task(task: Task) -> ContextTask:
    return ContextTask(
        task_id=task.id,
        status=task.status.value,
        slack_channel_id=task.slack_channel_id,
        slack_thread_ts=task.slack_thread_ts,
    )


def _context_episode(item: RelevantEpisode) -> ContextEpisode:
    return ContextEpisode(
        episode_id=item.episode.id,
        task_id=item.episode.task_id,
        relation=item.relation,
        outcome=item.episode.outcome,
    )


def _context_artifact(artifact: Artifact) -> ContextArtifact:
    return ContextArtifact(
        artifact_id=artifact.id,
        task_id=artifact.task_id,
        filename=artifact.filename,
        slack_file_id=artifact.slack_file_id,
        mime_type=artifact.mime_type,
        size_bytes=artifact.size_bytes,
    )


def _context_chars(package: ContextPackage) -> int:
    return sum(len(message.content or "") for message in package.messages)


def _context_budget_payload(budget: ContextBudget) -> dict[str, int]:
    return {
        "system_prompt_chars": budget.system_prompt_chars,
        "known_facts_max_chars": budget.known_facts_max_chars,
        "known_facts_chars": budget.known_facts_chars,
        "thread_context_max_chars": budget.thread_context_max_chars,
        "prior_context_chars": budget.prior_context_chars,
        "thread_context_recent_tasks": budget.thread_context_recent_tasks,
        "thread_transcript_limit": budget.thread_transcript_limit,
        "episode_context_max_chars": budget.episode_context_max_chars,
        "episode_context_chars": budget.episode_context_chars,
        "episode_context_limit": budget.episode_context_limit,
    }


def _render_acknowledgement_context(acknowledgement: ContextAcknowledgement) -> str:
    lines = [
        "<visible_acknowledgement>",
        "Kortny already posted this visible Slack acknowledgement for the current request:",
        f"- text: {_quote(acknowledgement.text)}",
    ]
    if acknowledgement.message_ts:
        lines.append(f"- message_ts: {acknowledgement.message_ts}")
    lines.extend(
        [
            "Use this as already-said context. Write the worker response as a "
            "natural continuation, without repeating the acknowledgement or "
            "restarting the conversation.",
            "If the acknowledgement was too broad or slightly imprecise, continue "
            "with the correct answer without mentioning internal handoff.",
            "</visible_acknowledgement>",
        ]
    )
    return "\n".join(lines)


def _render_known_facts(facts: Sequence[Fact]) -> str:
    grouped = {
        "workspace": [fact for fact in facts if fact.scope_type == "workspace"],
        "channel": [fact for fact in facts if fact.scope_type == "channel"],
        "user": [fact for fact in facts if fact.scope_type == "user"],
    }
    lines = [
        "<known_facts>",
        "Confirmed durable facts for this task. Use these facts before asking "
        "follow-up questions or calling external tools. If facts conflict, user "
        "facts override channel facts, and channel facts override workspace facts.",
    ]
    for scope_type in ("workspace", "channel", "user"):
        scoped_facts = sorted(
            grouped[scope_type],
            key=lambda fact: (fact.key, fact.created_at, str(fact.id)),
        )
        if not scoped_facts:
            continue
        lines.append("")
        lines.append(f"{_known_fact_scope_label(scope_type)}:")
        for fact in scoped_facts:
            lines.append(_known_fact_line(fact))
    lines.append("</known_facts>")
    return "\n".join(lines)


def _render_episode_context(episodes: Sequence[RelevantEpisode]) -> str:
    lines = [
        "<recent_episodes>",
        "Bounded episodic memory from prior Kortny tasks. Use this to resolve "
        "references to prior work, artifacts, sources, failures, or decisions. "
        "Do not treat these as confirmed user/workspace facts; confirmed facts "
        "are supplied separately in known_facts.",
    ]
    for index, item in enumerate(episodes, start=1):
        episode = item.episode
        tools = ", ".join(episode.tools_used) if episode.tools_used else "none"
        lines.append(
            f"- {index}. relation={item.relation} episode_id={episode.id} "
            f"task_id={episode.task_id} outcome={episode.outcome} "
            f"channel={episode.channel_id} thread_ts={episode.thread_ts or ''} "
            f"user={episode.user_id} tools={_quote(tools)} "
            f"summary={_quote(_shorten(episode.summary, max_chars=500))}"
        )
        if episode.artifacts_created:
            lines.append("  artifacts:")
            for artifact in episode.artifacts_created[:5]:
                lines.append(f"  - {_episode_artifact_line(artifact)}")
        if episode.source_refs:
            lines.append("  sources:")
            for source in episode.source_refs[:5]:
                lines.append(f"  - {_episode_source_line(source)}")
        if episode.error:
            lines.append(
                "  error: "
                f"{_quote(_shorten(_json_dumps(episode.error), max_chars=360))}"
            )
    lines.append("</recent_episodes>")
    return "\n".join(lines)


def _episode_artifact_line(artifact: dict[str, object]) -> str:
    details: list[str] = []
    filename = artifact.get("filename")
    if isinstance(filename, str):
        details.append(f"filename={_quote(filename)}")
    slack_file_id = artifact.get("slack_file_id")
    if isinstance(slack_file_id, str):
        details.append(f"slack_file_id={slack_file_id}")
    mime_type = artifact.get("mime_type")
    if isinstance(mime_type, str):
        details.append(f"mime_type={_quote(mime_type)}")
    size_bytes = artifact.get("size_bytes")
    if isinstance(size_bytes, int):
        details.append(f"size_bytes={size_bytes}")
    return " ".join(details) if details else _json_dumps(artifact)


def _episode_source_line(source: dict[str, object]) -> str:
    details: list[str] = []
    query = source.get("query")
    if isinstance(query, str):
        details.append(f"query={_quote(_shorten(query, max_chars=120))}")
    title = source.get("title")
    if isinstance(title, str):
        details.append(f"title={_quote(_shorten(title, max_chars=160))}")
    url = source.get("url")
    if isinstance(url, str):
        details.append(f"url={url}")
    return " ".join(details) if details else _json_dumps(source)


def _known_fact_scope_label(scope_type: str) -> str:
    if scope_type == "workspace":
        return "Workspace facts"
    if scope_type == "channel":
        return "Channel facts"
    if scope_type == "user":
        return "User facts"
    return "Facts"


def _known_fact_line(fact: Fact) -> str:
    value = _shorten(_known_fact_value(fact), max_chars=600)
    return f"- {fact.key} = {_quote(value)}"


def _known_fact_value(fact: Fact) -> str:
    if fact.value_text:
        return fact.value_text
    return _json_dumps(fact.value)


def _tasks_before(thread_tasks: Sequence[Task], current_task: Task) -> list[Task]:
    prior_tasks: list[Task] = []
    for task in thread_tasks:
        if task.id == current_task.id:
            return prior_tasks
        prior_tasks.append(task)
    return [task for task in thread_tasks if task.id != current_task.id]


def _immediate_previous_exchange_lines(task: Task) -> list[str]:
    lines = [
        "",
        "Immediate previous exchange:",
        f"- user: {_quote(_shorten(task.input, max_chars=IMMEDIATE_PRIOR_INPUT_MAX_CHARS))}",
        "- assistant: "
        f"{_quote(_shorten(task.result_summary or '(no result summary yet)', max_chars=IMMEDIATE_PRIOR_RESULT_MAX_CHARS))}",
    ]
    slack_file_ids = _slack_file_ids(task.input)
    if slack_file_ids:
        lines.append(f"- attached_slack_file_ids: {','.join(slack_file_ids)}")
    return lines


def _task_summary_line(index: int, task: Task) -> str:
    result = task.result_summary or "(no result summary yet)"
    line = (
        f"- {index}. task_id={task.id} status={task.status.value} input={_quote(_shorten(task.input, max_chars=240))} "
        f"result={_quote(_shorten(result, max_chars=360))} cost_usd={task.total_cost_usd}"
    )
    slack_file_ids = _slack_file_ids(task.input)
    if slack_file_ids:
        line = f"{line} slack_file_ids={','.join(slack_file_ids)}"
    error = _error_summary(task.error)
    if error:
        line = f"{line} error={_quote(_shorten(error, max_chars=240))}"
    return line


def _slack_files_block(input_text: str) -> str | None:
    match = SLACK_FILES_BLOCK_RE.search(input_text)
    if match is None:
        return None
    content = match.group(1).strip()
    if not content:
        return None
    return content


def _slack_file_ids(input_text: str) -> list[str]:
    block = _slack_files_block(input_text)
    if block is None:
        return []
    return SLACK_FILE_ID_RE.findall(block)


def _error_summary(error: dict | None) -> str | None:
    if not error:
        return None
    error_type = error.get("type")
    message = error.get("message")
    if isinstance(error_type, str) and isinstance(message, str):
        return f"{error_type}: {message}"
    return _json_dumps(error)


def _transcript_line(message: ThreadTranscriptMessage) -> str:
    speaker = message.user_id
    if speaker is None and message.bot_id is not None:
        speaker = f"bot:{message.bot_id}"
    if speaker is None:
        speaker = "unknown"
    return f"- [{message.ts}] {speaker}: {_shorten(_single_line(message.text), max_chars=500)}"


def _fit_context_to_budget(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    suffix = "\n[prior_context truncated at configured budget]\n</prior_context>"
    return content[: max(0, max_chars - len(suffix))].rstrip() + suffix


def _shorten(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _quote(value: str) -> str:
    return json.dumps(_single_line(value))


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _is_dm_conversation_context_key(
    channel_id: str | None, thread_ts: str | None
) -> bool:
    return bool(channel_id and channel_id.startswith("D") and thread_ts == channel_id)


def _is_dm_channel(channel_id: str | None) -> bool:
    return bool(channel_id and channel_id.startswith("D"))


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
