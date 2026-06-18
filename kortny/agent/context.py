"""Context assembly for agent task execution."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent.capabilities import CapabilityOverview, render_capability_overview
from kortny.agent.thread_context import (
    ThreadTranscriptMessage,
    ThreadTranscriptProvider,
)
from kortny.db.models import (
    Artifact,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.embeddings import (
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    EPISODE_EMBEDDING_KIND,
    FACT_EMBEDDING_KIND,
    KG_ENTITY_EMBEDDING_KIND,
    EmbeddingIndex,
    ranked_score,
)
from kortny.knowledge_graph import (
    DestinationSurface,
    GraphContextPack,
    GraphService,
    RetrievedGraphEdge,
    RetrievedGraphEntity,
    VisibilityScope,
)
from kortny.knowledge_graph.projects import project_anchors_and_scopes
from kortny.llm import ChatMessage
from kortny.memory import EpisodeService, Fact, RelevantEpisode, WorkspaceStateService
from kortny.observability import observe_task_event, set_span_attributes, start_span
from kortny.skills.embedding import SKILL_EMBEDDING_KIND, skill_embedding_text
from kortny.tasks import TaskService

if TYPE_CHECKING:
    from kortny.skills.service import EnabledSkill

DEFAULT_THREAD_CONTEXT_MAX_CHARS = 12_000
DEFAULT_THREAD_CONTEXT_RECENT_TASKS = 3
DEFAULT_THREAD_TRANSCRIPT_LIMIT = 30
DEFAULT_KNOWN_FACTS_MAX_CHARS = 4_000
DEFAULT_EPISODE_CONTEXT_MAX_CHARS = 4_000
DEFAULT_EPISODE_CONTEXT_LIMIT = 5
DEFAULT_GRAPH_CONTEXT_MAX_CHARS = 1_500
DEFAULT_GRAPH_CONTEXT_MAX_ITEMS = 12
DEFAULT_GRAPH_CONTEXT_MAX_HOPS = 2
DEFAULT_SKILLS_CONTEXT_MAX_CHARS = 4_000
# HIG-239: tighten the ranked index from 30 → 15. The curated pack pushes the
# enabled-skill count up; a smaller, sharper index keeps the L1 block focused
# and within the 4k char budget. Omissions beyond K are still recorded.
DEFAULT_SKILLS_CONTEXT_MAX_SKILLS = 15
DEFAULT_CAPABILITIES_CONTEXT_MAX_CHARS = 1_200
DEFAULT_SKILL_DIRECT_THRESHOLD = 0.60
RELEVANCE_BUDGET_OMISSION_REASON = "relevance_budget"
EPISODE_RELATION_TIERS = {"same_thread": 0, "same_channel": 1, "same_user": 2}
EXECUTION_HINT_SKILL_DIRECT = "skill_direct"
DEFAULT_CONTEXT_ENGINE_ID = "kortny.context_assembler"
DEFAULT_CONTEXT_ENGINE_NAME = "ContextAssembler"
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
class ContextGraphEntity:
    """A graph entity selected for task context."""

    entity_id: uuid.UUID
    entity_type: str
    canonical_key: str
    visibility_scope_type: str
    visibility_scope_id: str | None
    evidence_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True, slots=True)
class ContextGraphEdge:
    """A graph edge selected for task context."""

    edge_id: uuid.UUID
    relationship_type: str
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    visibility_scope_type: str
    visibility_scope_id: str | None
    evidence_ids: tuple[uuid.UUID, ...]


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
    graph_context_max_chars: int
    graph_context_chars: int
    graph_context_max_items: int
    graph_context_max_hops: int


@dataclass(frozen=True, slots=True)
class ContextSkill:
    """An enabled procedural skill surfaced in the L1 skills block."""

    skill_id: uuid.UUID
    version_id: uuid.UUID
    slug: str
    name: str
    description: str
    trust_level: str
    scope_type: str


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
    selected_graph_entities: tuple[ContextGraphEntity, ...]
    selected_graph_edges: tuple[ContextGraphEdge, ...]
    acknowledgement: ContextAcknowledgement | None
    budget: ContextBudget
    omissions: tuple[ContextOmission, ...]
    selected_skills: tuple[ContextSkill, ...] = ()
    context_engine_id: str = DEFAULT_CONTEXT_ENGINE_ID
    context_engine_name: str = DEFAULT_CONTEXT_ENGINE_NAME
    skill_similarities: tuple[tuple[str, float], ...] = ()
    execution_hint: str | None = None
    matched_skill_slug: str | None = None


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


@dataclass(frozen=True, slots=True)
class _GraphContext:
    content: str | None
    selected_entities: tuple[ContextGraphEntity, ...]
    selected_edges: tuple[ContextGraphEdge, ...]
    returned_scopes: tuple[VisibilityScope, ...]
    omissions: tuple[ContextOmission, ...]


@dataclass(frozen=True, slots=True)
class _SkillsContext:
    content: str | None
    selected_skills: tuple[ContextSkill, ...]
    omissions: tuple[ContextOmission, ...]
    skill_similarities: tuple[tuple[str, float], ...] = ()
    execution_hint: str | None = None
    matched_skill_slug: str | None = None


@dataclass(frozen=True, slots=True)
class _CapabilitiesContext:
    content: str | None
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
        graph_context_max_chars: int = DEFAULT_GRAPH_CONTEXT_MAX_CHARS,
        graph_context_max_items: int = DEFAULT_GRAPH_CONTEXT_MAX_ITEMS,
        graph_context_max_hops: int = DEFAULT_GRAPH_CONTEXT_MAX_HOPS,
        context_engine_id: str = DEFAULT_CONTEXT_ENGINE_ID,
        context_engine_name: str = DEFAULT_CONTEXT_ENGINE_NAME,
        capability_overview: CapabilityOverview | None = None,
        embedding_index: EmbeddingIndex | None = None,
        skill_direct_threshold: float = DEFAULT_SKILL_DIRECT_THRESHOLD,
        recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
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
        if graph_context_max_chars < 0:
            raise ValueError("graph_context_max_chars cannot be negative")
        if graph_context_max_items < 0:
            raise ValueError("graph_context_max_items cannot be negative")
        if graph_context_max_hops < 0:
            raise ValueError("graph_context_max_hops cannot be negative")
        if not context_engine_id:
            raise ValueError("context_engine_id must be non-empty")
        if not context_engine_name:
            raise ValueError("context_engine_name must be non-empty")

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
        self.graph_context_max_chars = graph_context_max_chars
        self.graph_context_max_items = graph_context_max_items
        self.graph_context_max_hops = graph_context_max_hops
        self.context_engine_id = context_engine_id
        self.context_engine_name = context_engine_name
        self.capability_overview = capability_overview
        self.embedding_index = embedding_index
        self.skill_direct_threshold = skill_direct_threshold
        self.recency_half_life_days = recency_half_life_days
        self.workspace_state_service = WorkspaceStateService(
            session,
            task_service=self.task_service,
        )
        self.episode_service = EpisodeService(session, task_service=self.task_service)

    def build_for_task(self, task: Task) -> ContextPackage:
        """Build prompt messages and context-selection metadata."""

        with start_span(
            "context.assemble",
            task=task,
            attributes={"openinference.span.kind": "CHAIN"},
        ):
            return self._build_for_task(task)

    def _build_for_task(self, task: Task) -> ContextPackage:
        """Build prompt messages and context-selection metadata inside a span."""

        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        capabilities_context = self._capabilities_context()
        if capabilities_context.content:
            messages.append(
                ChatMessage(role="system", content=capabilities_context.content)
            )

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

        graph_context = self._graph_context(task)
        if graph_context.content:
            messages.append(ChatMessage(role="system", content=graph_context.content))

        skills_context = self._skills_context(task)
        if skills_context.content:
            messages.append(ChatMessage(role="system", content=skills_context.content))

        messages.append(ChatMessage(role="user", content=task.input))

        package = ContextPackage(
            messages=tuple(messages),
            selected_facts=known_facts.selected_facts,
            selected_prior_tasks=prior_context.selected_prior_tasks,
            selected_episodes=episode_context.selected_episodes,
            selected_artifacts=prior_context.selected_artifacts,
            selected_graph_entities=graph_context.selected_entities,
            selected_graph_edges=graph_context.selected_edges,
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
                graph_context_max_chars=self.graph_context_max_chars,
                graph_context_chars=len(graph_context.content or ""),
                graph_context_max_items=self.graph_context_max_items,
                graph_context_max_hops=self.graph_context_max_hops,
            ),
            omissions=(
                capabilities_context.omissions
                + known_facts.omissions
                + prior_context.omissions
                + episode_context.omissions
                + graph_context.omissions
                + skills_context.omissions
            ),
            selected_skills=skills_context.selected_skills,
            context_engine_id=self.context_engine_id,
            context_engine_name=self.context_engine_name,
            skill_similarities=skills_context.skill_similarities,
            execution_hint=skills_context.execution_hint,
            matched_skill_slug=skills_context.matched_skill_slug,
        )
        self._record_context_assembled(task, package)
        set_span_attributes(
            {
                "context.engine_id": package.context_engine_id,
                "context.engine_name": package.context_engine_name,
                "context.message_count": len(package.messages),
                "context.selected_fact_count": len(package.selected_facts),
                "context.selected_episode_count": len(package.selected_episodes),
                "context.selected_prior_task_count": len(package.selected_prior_tasks),
                "context.selected_artifact_count": len(package.selected_artifacts),
                "context.selected_graph_entity_count": len(
                    package.selected_graph_entities
                ),
                "context.selected_graph_edge_count": len(package.selected_graph_edges),
                "context.selected_skill_count": len(package.selected_skills),
                "context.acknowledgement_present": package.acknowledgement is not None,
                "context.context_chars": _context_chars(package),
                "context.omission_count": len(package.omissions),
            }
        )
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
            context_engine_id=package.context_engine_id,
            context_engine_name=package.context_engine_name,
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
            selected_graph_entity_ids=[
                str(entity.entity_id) for entity in package.selected_graph_entities
            ],
            selected_graph_entity_keys=[
                entity.canonical_key for entity in package.selected_graph_entities
            ],
            selected_graph_edge_ids=[
                str(edge.edge_id) for edge in package.selected_graph_edges
            ],
            selected_skill_slugs=[skill.slug for skill in package.selected_skills],
            selected_skill_ids=[
                str(skill.skill_id) for skill in package.selected_skills
            ],
            skill_similarities={
                slug: round(score, 4) for slug, score in package.skill_similarities
            },
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

        fact_scores = self._memory_ranked_scores(
            FACT_EMBEDDING_KIND,
            task.input,
            [(str(fact.id), fact.updated_at) for fact in facts],
        )

        kept = list(facts)
        dropped = 0
        rendered = _render_known_facts(kept)
        while len(rendered) > self.known_facts_max_chars and kept:
            if fact_scores is None:
                drop = min(kept, key=lambda fact: (fact.created_at, fact.id))
            else:
                drop = min(
                    kept,
                    key=lambda fact: (
                        fact_scores.get(str(fact.id), 0.0),
                        fact.created_at,
                        str(fact.id),
                    ),
                )
            kept.remove(drop)
            dropped += 1
            rendered = _render_known_facts(kept)

        omissions: list[ContextOmission] = []
        if dropped:
            omissions.append(
                ContextOmission(
                    "known_facts",
                    "budget_exceeded_drop_oldest"
                    if fact_scores is None
                    else RELEVANCE_BUDGET_OMISSION_REASON,
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

        episode_scores = self._memory_ranked_scores(
            EPISODE_EMBEDDING_KIND,
            task.input,
            [(str(item.episode.id), item.episode.created_at) for item in episodes],
        )
        if episode_scores is not None:
            # Rank within the existing thread > channel > user precedence
            # tiers; the tier order itself stays a hard precedence.
            order = {id(item): index for index, item in enumerate(episodes)}
            episodes.sort(
                key=lambda item: (
                    EPISODE_RELATION_TIERS.get(item.relation, 99),
                    -episode_scores.get(str(item.episode.id), 0.0),
                    order[id(item)],
                )
            )

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
                    "budget_exceeded_drop_lowest_relevance"
                    if episode_scores is None
                    else RELEVANCE_BUDGET_OMISSION_REASON,
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

    def _graph_context(self, task: Task) -> _GraphContext:
        if self.graph_context_max_chars == 0 or self.graph_context_max_items == 0:
            return _GraphContext(
                content=None,
                selected_entities=(),
                selected_edges=(),
                returned_scopes=(),
                omissions=(ContextOmission("knowledge_graph", "budget_disabled", 0),),
            )

        destination = _destination_surface_for_task(self.session, task)
        anchor_keys = _graph_anchor_keys(task)
        # Project layer (HIG-276): if the current channel belongs to a project,
        # anchor on the project hub and widen the audience to the project's
        # PUBLIC member channels so retrieval synthesizes across them. Bumps hops
        # so BFS reaches project hub -> channel hub -> channel facts.
        project_anchor_keys, additional_scopes = _project_anchors_and_scopes(
            self.session, task
        )
        if project_anchor_keys:
            anchor_keys = anchor_keys + project_anchor_keys
        max_hops = self.graph_context_max_hops
        if project_anchor_keys:
            max_hops = max(max_hops, _PROJECT_GRAPH_MAX_HOPS)
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "kg_retrieval_started",
                "destination_surface_type": destination.surface_type,
                "destination_surface_id": destination.surface_id,
                "destination_user_id": destination.user_id,
                "anchor_keys": list(anchor_keys),
                "project_anchor_keys": list(project_anchor_keys),
                "additional_scope_count": len(additional_scopes),
                "max_items": self.graph_context_max_items,
                "max_hops": max_hops,
            },
        )

        pack = GraphService(self.session).retrieve_current_context(
            installation_id=task.installation_id,
            destination=destination,
            anchor_keys=anchor_keys,
            max_hops=max_hops,
            max_items=self.graph_context_max_items,
            additional_scopes=additional_scopes,
        )
        violations = GraphService.scope_guard_violations(
            pack, destination, additional_scopes
        )
        if violations:
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "kg_scope_guard_failed",
                    "destination_surface_type": destination.surface_type,
                    "destination_surface_id": destination.surface_id,
                    "violation_scopes": [_scope_payload(scope) for scope in violations],
                },
            )
            return _GraphContext(
                content=None,
                selected_entities=(),
                selected_edges=(),
                returned_scopes=(),
                omissions=(
                    ContextOmission(
                        "knowledge_graph",
                        "scope_guard_failed",
                        len(violations),
                    ),
                ),
            )

        entity_scores = self._memory_ranked_scores(
            KG_ENTITY_EMBEDDING_KIND,
            task.input,
            [(str(entity.id), entity.last_seen_at) for entity in pack.entities],
        )
        if entity_scores is not None and pack.entities:
            order = {entity.id: index for index, entity in enumerate(pack.entities)}
            pack = replace(
                pack,
                entities=tuple(
                    sorted(
                        pack.entities,
                        key=lambda entity: (
                            -entity_scores.get(str(entity.id), 0.0),
                            order[entity.id],
                        ),
                    )
                ),
            )

        rendered = _render_graph_context(pack)
        omissions = list(pack.omitted_reasons)
        if rendered and len(rendered) > self.graph_context_max_chars:
            rendered = _fit_context_to_budget(
                rendered,
                self.graph_context_max_chars,
                context_name="workspace_graph_context",
            )
            omissions.append("budget_compacted")

        selected_entities = tuple(
            _context_graph_entity(entity) for entity in pack.entities
        )
        selected_edges = tuple(_context_graph_edge(edge) for edge in pack.edges)
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "kg_retrieval_completed",
                "destination_surface_type": destination.surface_type,
                "destination_surface_id": destination.surface_id,
                "anchor_keys": list(anchor_keys),
                "entity_count": len(selected_entities),
                "edge_count": len(selected_edges),
                "returned_scopes": [
                    _scope_payload(scope) for scope in pack.returned_scopes
                ],
                "omitted_count": pack.omitted_count,
                "omitted_reasons": omissions,
                "context_chars": len(rendered or ""),
            },
        )

        if not rendered:
            return _GraphContext(
                content=None,
                selected_entities=(),
                selected_edges=(),
                returned_scopes=pack.returned_scopes,
                omissions=(),
            )

        return _GraphContext(
            content=rendered,
            selected_entities=selected_entities,
            selected_edges=selected_edges,
            returned_scopes=pack.returned_scopes,
            omissions=tuple(
                ContextOmission("knowledge_graph", reason, 1) for reason in omissions
            ),
        )

    def _capabilities_context(self) -> _CapabilitiesContext:
        """Render the installation capability card within its budget."""

        if self.capability_overview is None:
            return _CapabilitiesContext(content=None, omissions=())
        rendered = render_capability_overview(self.capability_overview)
        if not rendered:
            return _CapabilitiesContext(content=None, omissions=())
        if len(rendered) > DEFAULT_CAPABILITIES_CONTEXT_MAX_CHARS:
            rendered = _fit_context_to_budget(
                rendered,
                DEFAULT_CAPABILITIES_CONTEXT_MAX_CHARS,
                context_name="capabilities",
            )
            return _CapabilitiesContext(
                content=rendered,
                omissions=(ContextOmission("capabilities", "budget_compacted", 1),),
            )
        return _CapabilitiesContext(content=rendered, omissions=())

    def _skills_context(self, task: Task) -> _SkillsContext:
        """Build the L1 name+description block for skills enabled in scope."""

        from kortny.skills import SkillRegistryService

        try:
            enabled = SkillRegistryService(
                self.session, task_service=self.task_service
            ).enabled_skills_for_task(task)
        except Exception:  # pragma: no cover - defensive: skills never block a task
            logger.exception("skills context build failed for task %s", task.id)
            return _SkillsContext(content=None, selected_skills=(), omissions=())
        if not enabled:
            return _SkillsContext(content=None, selected_skills=(), omissions=())

        skill_similarities = self._rank_skills(task, enabled)
        if skill_similarities:
            similarity_by_slug = dict(skill_similarities)
            enabled = sorted(
                enabled,
                key=lambda item: -similarity_by_slug.get(item.slug, -1.0),
            )
        execution_hint: str | None = None
        matched_skill_slug: str | None = None
        if (
            skill_similarities
            and skill_similarities[0][1] >= self.skill_direct_threshold
        ):
            execution_hint = EXECUTION_HINT_SKILL_DIRECT
            matched_skill_slug = skill_similarities[0][0]

        omissions: list[ContextOmission] = []
        if len(enabled) > DEFAULT_SKILLS_CONTEXT_MAX_SKILLS:
            omissions.append(
                ContextOmission(
                    kind="skills",
                    reason="skills_context_max_skills",
                    count=len(enabled) - DEFAULT_SKILLS_CONTEXT_MAX_SKILLS,
                )
            )
            enabled = enabled[:DEFAULT_SKILLS_CONTEXT_MAX_SKILLS]

        selected: list[ContextSkill] = []
        lines = [
            "<available_skills>",
            "You have added skills available. If a skill's description matches "
            "the task, call the load_skill tool with its slug to get the full "
            "instructions BEFORE doing the work, then follow them. Use "
            "load_skill_resource(slug, path) to read a skill's bundled "
            "reference files.",
        ]
        used_chars = sum(len(line) + 1 for line in lines) + len("</available_skills>")
        for item in enabled:
            line = f"- {item.slug} [{item.scope_type}]: {item.description}"
            if used_chars + len(line) + 1 > DEFAULT_SKILLS_CONTEXT_MAX_CHARS:
                omissions.append(
                    ContextOmission(
                        kind="skills",
                        reason="skills_context_max_chars",
                        count=len(enabled) - len(selected),
                    )
                )
                break
            used_chars += len(line) + 1
            lines.append(line)
            selected.append(
                ContextSkill(
                    skill_id=item.skill_id,
                    version_id=item.version_id,
                    slug=item.slug,
                    name=item.name,
                    description=item.description,
                    trust_level=item.trust_level,
                    scope_type=item.scope_type,
                )
            )
        if not selected:
            return _SkillsContext(
                content=None,
                selected_skills=(),
                omissions=tuple(omissions),
                skill_similarities=skill_similarities,
            )
        if matched_skill_slug is not None and any(
            skill.slug == matched_skill_slug for skill in selected
        ):
            lines.append(
                f"Highly relevant skill for this task: {matched_skill_slug}. "
                "Load it with load_skill and follow it before doing the work "
                "yourself."
            )
        else:
            execution_hint = None
            matched_skill_slug = None
        lines.append("</available_skills>")
        return _SkillsContext(
            content="\n".join(lines),
            selected_skills=tuple(selected),
            omissions=tuple(omissions),
            skill_similarities=skill_similarities,
            execution_hint=execution_hint,
            matched_skill_slug=matched_skill_slug,
        )

    def _memory_ranked_scores(
        self,
        kind: str,
        query_text: str,
        refs: Sequence[tuple[str, object]],
    ) -> dict[str, float] | None:
        """Recency-weighted relevance scores for memory rows, or None.

        ``None`` means semantic ranking is unavailable (no embedding index, or
        ranking failed) and callers must fall back to the exact legacy
        behavior. Rows without an embedding score 0.0 and drop first.
        """

        if self.embedding_index is None or not refs:
            return None
        ranked = self.embedding_index.rank(
            kind,
            query_text,
            [ref_key for ref_key, _ in refs],
            top_k=len(refs),
        )
        if ranked is None:
            return None
        similarity_by_ref = dict(ranked)
        scores: dict[str, float] = {}
        for ref_key, last_seen in refs:
            similarity = similarity_by_ref.get(ref_key)
            if similarity is None:
                scores[ref_key] = 0.0
                continue
            scores[ref_key] = ranked_score(
                similarity,
                last_seen if isinstance(last_seen, datetime) else None,
                half_life_days=self.recency_half_life_days,
            )
        return scores

    def _rank_skills(
        self,
        task: Task,
        enabled: Sequence[EnabledSkill],
    ) -> tuple[tuple[str, float], ...]:
        """Rank enabled skills by similarity to the task input.

        Semantic ranking is preferred (the embedded text now includes intent
        tags + trigger phrases, mirroring tool-card embeddings). When no
        embedding index is wired, or ranking fails, fall back to lexical token
        overlap so a name/description match still surfaces — the ranker never
        returns empty while enabled skills exist.
        """

        if not enabled:
            return ()
        if self.embedding_index is None:
            return self._lexical_rank_skills(task, enabled)
        items = [
            (
                skill.slug,
                skill_embedding_text(
                    name=skill.name,
                    description=skill.description,
                    intent_tags=skill.intent_tags,
                    trigger_phrases=skill.trigger_phrases,
                ),
            )
            for skill in enabled
        ]
        self.embedding_index.ensure(SKILL_EMBEDDING_KIND, items)
        ranked = self.embedding_index.rank(
            SKILL_EMBEDDING_KIND,
            task.input,
            [slug for slug, _ in items],
            top_k=DEFAULT_SKILLS_CONTEXT_MAX_SKILLS,
        )
        if ranked is None:
            return self._lexical_rank_skills(task, enabled)
        return tuple(ranked)

    @staticmethod
    def _lexical_rank_skills(
        task: Task,
        enabled: Sequence[EnabledSkill],
    ) -> tuple[tuple[str, float], ...]:
        """Token-overlap fallback ranking over name + description + tags.

        Mirrors the tool-RAG lexical fill: score each skill by the overlap
        between task-input tokens and the skill's name/description/tags/trigger
        tokens, normalized to [0, 1]. Order is best-first; ties keep slug order.
        """

        query = _lexical_tokens(task.input)
        scored: list[tuple[str, float]] = []
        for skill in sorted(enabled, key=lambda item: item.slug):
            corpus_tokens = _lexical_tokens(
                " ".join(
                    (
                        skill.slug,
                        skill.name,
                        skill.description,
                        " ".join(skill.intent_tags),
                        " ".join(skill.trigger_phrases),
                    )
                )
            )
            if not query or not corpus_tokens:
                scored.append((skill.slug, 0.0))
                continue
            overlap = len(query & corpus_tokens)
            score = min(1.0, overlap * 0.08)
            scored.append((skill.slug, round(score, 4)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return tuple(scored[:DEFAULT_SKILLS_CONTEXT_MAX_SKILLS])

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


def _context_graph_entity(entity: RetrievedGraphEntity) -> ContextGraphEntity:
    return ContextGraphEntity(
        entity_id=entity.id,
        entity_type=entity.entity_type,
        canonical_key=entity.canonical_key,
        visibility_scope_type=entity.visibility_scope.scope_type,
        visibility_scope_id=entity.visibility_scope.scope_id,
        evidence_ids=entity.evidence_ids,
    )


def _context_graph_edge(edge: RetrievedGraphEdge) -> ContextGraphEdge:
    return ContextGraphEdge(
        edge_id=edge.id,
        relationship_type=edge.relationship_type,
        source_entity_id=edge.source_entity_id,
        target_entity_id=edge.target_entity_id,
        visibility_scope_type=edge.visibility_scope.scope_type,
        visibility_scope_id=edge.visibility_scope.scope_id,
        evidence_ids=edge.evidence_ids,
    )


def _destination_surface_for_task(session: Session, task: Task) -> DestinationSurface:
    if _is_dm_channel(task.slack_channel_id):
        return DestinationSurface.dm(task.slack_channel_id, user_id=task.slack_user_id)

    membership = session.scalar(
        select(SlackChannelMembership).where(
            SlackChannelMembership.installation_id == task.installation_id,
            SlackChannelMembership.channel_id == task.slack_channel_id,
        )
    )
    if _is_private_channel(task.slack_channel_id, membership):
        return DestinationSurface.private_channel(task.slack_channel_id)
    return DestinationSurface.channel(task.slack_channel_id)


def _is_private_channel(
    channel_id: str,
    membership: SlackChannelMembership | None,
) -> bool:
    if channel_id.startswith("G"):
        return True
    channel_type = (membership.channel_type or "").lower() if membership else ""
    return channel_type in {"group", "private_channel", "private"}


def _graph_anchor_keys(task: Task) -> tuple[str, ...]:
    keys = [f"slack_user:{task.slack_user_id}"]
    if _is_dm_channel(task.slack_channel_id):
        keys.insert(0, f"slack_dm:{task.slack_channel_id}")
    else:
        keys.insert(0, f"slack_channel:{task.slack_channel_id}")
    return tuple(keys)


# BFS depth for project-anchored retrieval: project hub -> channel hub (1) ->
# channel facts (2). Single-channel retrieval keeps the smaller configured hops.
_PROJECT_GRAPH_MAX_HOPS = 2


def _project_anchors_and_scopes(
    session: Session, task: Task
) -> tuple[tuple[str, ...], tuple[VisibilityScope, ...]]:
    """Project hub anchors + audience-safe extra scopes for the current channel.

    If the task's channel belongs to one or more projects (HIG-276), return the
    project hubs' canonical keys (to anchor BFS on the hub) and the projects'
    PUBLIC member-channel scopes (to authorize cross-channel synthesis). Empty
    when the channel is not part of a project, so non-project tasks are
    unaffected. Best-effort: never fails context assembly.
    """

    channel_id = task.slack_channel_id
    if not channel_id or _is_dm_channel(channel_id):
        return ((), ())
    try:
        return project_anchors_and_scopes(
            session, installation_id=task.installation_id, channel_id=channel_id
        )
    except Exception:
        logger.warning(
            "project anchor lookup failed task_id=%s", task.id, exc_info=True
        )
        return ((), ())


def _scope_payload(scope: VisibilityScope) -> dict[str, str | None]:
    return {"type": scope.scope_type, "id": scope.scope_id}


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
        "graph_context_max_chars": budget.graph_context_max_chars,
        "graph_context_chars": budget.graph_context_chars,
        "graph_context_max_items": budget.graph_context_max_items,
        "graph_context_max_hops": budget.graph_context_max_hops,
    }


def _render_acknowledgement_context(acknowledgement: ContextAcknowledgement) -> str:
    lines = [
        "<visible_acknowledgement>",
        "__AGENT_NAME__ already posted this visible Slack acknowledgement for the current request:",
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
        "Bounded episodic memory from prior __AGENT_NAME__ tasks. Use this to resolve "
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


def _render_graph_context(pack: GraphContextPack) -> str | None:
    if not pack.entities and not pack.edges:
        return None
    lines = [
        "<workspace_graph_context>",
        "Scope-safe current workspace graph context. These rows were filtered "
        "by the destination Slack surface before reaching the model. Use as "
        "background context only; do not treat candidate or unlisted graph facts "
        "as confirmed.",
    ]
    if pack.entities:
        lines.append("")
        lines.append("Entities:")
        for index, entity in enumerate(pack.entities, start=1):
            lines.append(_graph_entity_line(index, entity))
    if pack.edges:
        lines.append("")
        lines.append("Relationships:")
        for index, edge in enumerate(pack.edges, start=1):
            lines.append(_graph_edge_line(index, edge))
    lines.append("</workspace_graph_context>")
    return "\n".join(lines)


def _graph_entity_line(index: int, entity: RetrievedGraphEntity) -> str:
    details = [
        f"{index}.",
        f"entity_id={entity.id}",
        f"type={entity.entity_type}",
        f"key={_quote(entity.canonical_key)}",
        f"state={entity.lifecycle_state}",
        f"scope={_scope_label(entity.visibility_scope)}",
        f"confidence={entity.confidence_score}",
        f"evidence_ids={_uuid_csv(entity.evidence_ids)}",
    ]
    if entity.display_name:
        details.insert(4, f"name={_quote(entity.display_name)}")
    return "- " + " ".join(details)


def _graph_edge_line(index: int, edge: RetrievedGraphEdge) -> str:
    return (
        f"- {index}. edge_id={edge.id} relationship={edge.relationship_type} "
        f"source_entity_id={edge.source_entity_id} "
        f"target_entity_id={edge.target_entity_id} state={edge.lifecycle_state} "
        f"scope={_scope_label(edge.visibility_scope)} "
        f"confidence={edge.confidence_score} evidence_ids={_uuid_csv(edge.evidence_ids)}"
    )


def _scope_label(scope: VisibilityScope) -> str:
    if scope.scope_id is None:
        return scope.scope_type
    return f"{scope.scope_type}:{scope.scope_id}"


def _uuid_csv(values: Sequence[uuid.UUID]) -> str:
    if not values:
        return "none"
    return ",".join(str(value) for value in values)


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


def _fit_context_to_budget(
    content: str,
    max_chars: int,
    *,
    context_name: str = "prior_context",
) -> str:
    if len(content) <= max_chars:
        return content
    suffix = f"\n[{context_name} truncated at configured budget]\n</{context_name}>"
    return content[: max(0, max_chars - len(suffix))].rstrip() + suffix


def _shorten(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _lexical_tokens(text: str) -> set[str]:
    """Alphanumeric token set for lexical overlap scoring (slug-aware)."""

    return {
        "".join(char for char in raw.casefold() if char.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if raw.strip()
    } - {""}


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
