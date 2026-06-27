"""Pass N: skill induction -- turn successful multi-tool episodes into candidate
procedural skills (HIG-300 S3).

Induced skills are ALWAYS:
- trust_level="untrusted"
- NO SkillEnablement created (never auto-enabled)
- visibility="catalog" (catalog-only, invisible to the runtime until a human
  enables them from the dashboard)

This is the WRITE half of the self-learning loop.  The READ half
(context injection) is deferred to S3b.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Episode,
    ProceduralSkill,
    ProceduralSkillInvocation,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.embeddings import EmbeddingIndex
from kortny.llm import ChatMessage, LLMService
from kortny.skills.embedding import SKILL_EMBEDDING_KIND, skill_embedding_text
from kortny.skills.ingestion import SkillIngestionService
from kortny.tools.types import JsonObject

logger = logging.getLogger(__name__)

SKILL_INDUCTION_PROMPT_NAME = "kortny.consolidator_skill_induction"
SKILL_INDUCTION_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
SKILL_INDUCTION_CREATED_BY = "consolidator:skill_induction"
SKILL_INDUCTION_PROVENANCE_PREFIX = "agent_induced:episode"

# routing_quality values that indicate a successful, clean execution
_ELIGIBLE_QUALITY = frozenset({"clean", "recovered"})

# identity_kind values for synthetic/consolidator tasks -- skip these
_SYNTHETIC_KINDS = frozenset({"synthetic", "scheduled"})

# Minimum distinct non-skill tools required to qualify as a multi-tool workflow.
_MIN_DISTINCT_TOOLS = 2

# Maximum tool trajectory steps sent to the LLM (bounded for cost).
_MAX_TRAJECTORY_STEPS = 20

# Dedup similarity threshold: if closest existing skill is above this, skip.
_DEDUP_SIMILARITY_THRESHOLD = 0.80

_KEY_RE = re.compile(r"[^a-z0-9\-]+")


@dataclass(slots=True)
class SkillInductionCounters:
    """Counters from one skill induction pass."""

    episodes_scanned: int = 0
    proposed: int = 0
    created: int = 0
    deduped_skipped: int = 0
    noop: int = 0
    failed: int = 0
    # created_at of the last episode actually processed; next run resumes here.
    anchor: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "episodes_scanned": self.episodes_scanned,
            "proposed": self.proposed,
            "created": self.created,
            "deduped_skipped": self.deduped_skipped,
            "noop": self.noop,
            "failed": self.failed,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class SkillInductionProposal:
    """Parsed LLM output for one induction candidate."""

    action: str  # "CREATE" or "NOOP"
    name: str | None
    description: str | None
    allowed_tools: list[str]
    instructions_md: str | None
    confidence: float
    reason: str | None


class SkillInductionPass:
    """Scan recent successful episodes and induce candidate procedural skills."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMService | None,
        embedding_index: EmbeddingIndex | None,
        min_tool_calls: int = 3,
    ) -> None:
        self.session = session
        self.llm = llm
        self.embedding_index = embedding_index
        self.min_tool_calls = min_tool_calls

    def run(
        self,
        *,
        installation_id: uuid.UUID,
        task: Task,
        since: datetime | None,
        now: datetime | None = None,
        episode_cap: int = 50,
    ) -> SkillInductionCounters:
        counters = SkillInductionCounters(
            anchor=since.isoformat() if since is not None else None
        )
        if self.llm is None:
            return counters

        effective_now = now or datetime.now(UTC)
        episodes = self._eligible_episodes(
            installation_id=installation_id,
            since=since,
            cap=episode_cap,
        )
        if not episodes:
            return counters

        counters.anchor = episodes[-1].created_at.isoformat()

        for episode in episodes:
            counters.episodes_scanned += 1
            try:
                self._process_episode(
                    episode=episode,
                    installation_id=installation_id,
                    task=task,
                    now=effective_now,
                    counters=counters,
                )
            except Exception:
                logger.exception(
                    "skill_induction episode failed installation_id=%s episode_id=%s",
                    installation_id,
                    episode.id,
                )
                counters.failed += 1
                # Roll back partial writes from this episode; continue with
                # the others -- per-episode isolation mirrors the pass-level
                # isolation in the service.
                self.session.rollback()

        self.session.flush()
        return counters

    # -- gate ------------------------------------------------------------------

    def _eligible_episodes(
        self,
        *,
        installation_id: uuid.UUID,
        since: datetime | None,
        cap: int,
    ) -> list[Episode]:
        """Return succeeded episodes since the watermark that pass the scan gate."""
        predicates = [
            Episode.installation_id == installation_id,
            Episode.outcome == "succeeded",
        ]
        if since is not None:
            predicates.append(Episode.created_at > since)

        candidates = list(
            self.session.scalars(
                select(Episode)
                .join(Task, Task.id == Episode.task_id)
                .where(
                    *predicates,
                    # Only clean or recovered executions
                    Task.routing_quality.in_(list(_ELIGIBLE_QUALITY)),
                    # Exclude synthetic tasks (consolidator, scheduler, etc.)
                    (Task.identity_kind.is_(None))
                    | (Task.identity_kind.notin_(list(_SYNTHETIC_KINDS))),
                )
                .order_by(Episode.created_at, Episode.id)
                .limit(cap)
            )
        )

        eligible: list[Episode] = []
        for episode in candidates:
            if self._episode_passes_gate(episode, installation_id=installation_id):
                eligible.append(episode)
        return eligible

    def _episode_passes_gate(
        self,
        episode: Episode,
        *,
        installation_id: uuid.UUID,
    ) -> bool:
        """True if the episode's task has enough diverse non-skill tool calls."""
        task_id = episode.task_id

        # Count ordered tool_call events for this task
        tool_call_events = list(
            self.session.scalars(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task_id,
                    TaskEvent.type == TaskEventType.tool_call,
                )
                .order_by(TaskEvent.seq)
            )
        )
        if len(tool_call_events) < self.min_tool_calls:
            return False

        # Collect distinct non-skill tool names
        tool_names: list[str] = []
        for event in tool_call_events:
            tool = event.payload.get("tool")
            if not isinstance(tool, str):
                continue
            tool_names.append(tool)

        # Exclude skill tools (they begin with "skill:" or are procedural invocations)
        non_skill_tools = [t for t in tool_names if not t.startswith("skill:")]
        distinct_non_skill = set(non_skill_tools)

        if len(distinct_non_skill) < _MIN_DISTINCT_TOOLS:
            return False

        # Skip tasks that already invoked a procedural skill
        invoked_skill = self.session.scalar(
            select(ProceduralSkillInvocation.id)
            .where(ProceduralSkillInvocation.task_id == task_id)
            .limit(1)
        )
        return invoked_skill is None

    # -- induct ----------------------------------------------------------------

    def _process_episode(
        self,
        *,
        episode: Episode,
        installation_id: uuid.UUID,
        task: Task,
        now: datetime,
        counters: SkillInductionCounters,
    ) -> None:
        """Run the LLM call and store an induced skill if warranted."""
        task_row = self.session.get(Task, episode.task_id)
        if task_row is None:
            return

        trajectory = self._build_trajectory(episode.task_id)
        proposal = self._call_llm(
            task=task,
            episode=episode,
            task_row=task_row,
            trajectory=trajectory,
        )
        if proposal is None:
            counters.failed += 1
            return

        if proposal.action != "CREATE":
            counters.noop += 1
            return

        counters.proposed += 1

        # Dedup check against existing active skills
        if self._is_duplicate(proposal):
            counters.deduped_skipped += 1
            return

        # Store the induced skill
        self._store_skill(
            proposal=proposal,
            episode=episode,
            task_row=task_row,
            installation_id=installation_id,
            trajectory=trajectory,
        )
        counters.created += 1

    def _build_trajectory(self, task_id: uuid.UUID) -> list[dict[str, object]]:
        """Build a bounded, sanitized tool trajectory from task events."""
        events = list(
            self.session.scalars(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task_id,
                    TaskEvent.type == TaskEventType.tool_call,
                )
                .order_by(TaskEvent.seq)
                .limit(_MAX_TRAJECTORY_STEPS)
            )
        )
        steps: list[dict[str, object]] = []
        for event in events:
            payload = event.payload
            tool = payload.get("tool")
            if not isinstance(tool, str):
                continue
            args = payload.get("args") or payload.get("arguments") or {}
            arg_keys = list(args.keys()) if isinstance(args, dict) else []
            steps.append(
                {
                    "step_id": event.seq,
                    "tool": tool,
                    "arg_keys": arg_keys,
                    # We do NOT send arg values -- they may contain PII/IDs
                }
            )
        return steps

    def _call_llm(
        self,
        *,
        task: Task,
        episode: Episode,
        task_row: Task,
        trajectory: list[dict[str, object]],
    ) -> SkillInductionProposal | None:
        """Call the cheap-tier LLM to generalize the episode into a skill."""
        tools_used = [t for t in (episode.tools_used or []) if isinstance(t, str)]
        artifacts = episode.artifacts_created or []

        user_payload = {
            "task_input": task_row.input[:800] if task_row.input else "",
            "result_summary": (episode.summary or "")[:800],
            "tools_used": tools_used[:20],
            "artifact_count": len(artifacts),
            "tool_trajectory": trajectory,
        }

        if self.llm is None:
            return None

        completion = self.llm.complete(
            task_id=task.id,
            messages=(
                ChatMessage(
                    role="system",
                    content=(
                        "You are Kortny's skill induction engine. "
                        "Given a successful multi-tool task, decide whether it "
                        "represents a REPEATABLE WORKFLOW that could be captured "
                        "as a reusable procedural skill, or a one-off answer that "
                        "should not be crystallised. "
                        "RULES: "
                        "1. Strip all repo names, channel names, user names, exact "
                        "dates, one-off IDs, and object-specific constants from the "
                        "skill. Keep only the procedural pattern. "
                        "2. Only include tools actually used (or broad equivalent "
                        "classes). "
                        "3. NOOP is the default -- only use CREATE for a genuinely "
                        "repeatable cross-context workflow (e.g. 'search + "
                        "summarize + draft report', not 'answer this specific "
                        "question'). "
                        "4. confidence must be 0.0..1.0. Only CREATE when "
                        "confidence >= 0.70. "
                        "Return ONLY the JSON object -- no prose, markdown, or "
                        "comments. "
                        'Schema: {"action":"CREATE|NOOP","name":"short-slug-name",'
                        '"description":"one-sentence description",'
                        '"allowed_tools":["tool1","tool2"],'
                        '"instructions_md":"## Procedure\\n...",'
                        '"confidence":0.0,"reason":"why"}'
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        user_payload,
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    ),
                ),
            ),
            response_format=SKILL_INDUCTION_RESPONSE_FORMAT,
            prompt_name=SKILL_INDUCTION_PROMPT_NAME,
        )
        return parse_skill_induction_proposal(completion.content)

    def _is_duplicate(self, proposal: SkillInductionProposal) -> bool:
        """Return True if an existing active skill is close enough to skip."""
        if self.embedding_index is None or not proposal.description:
            return False
        # Build the candidate text the same way ingestion embeds it
        name = proposal.name or ""
        description = proposal.description or ""
        query_text = skill_embedding_text(
            name=name,
            description=description,
            intent_tags=["agent-induced"],
        )
        # Gather active skill slugs
        slugs = list(
            self.session.scalars(
                select(ProceduralSkill.slug).where(
                    ProceduralSkill.status == "active",
                )
            )
        )
        if not slugs:
            return False
        ranked = self.embedding_index.rank(
            SKILL_EMBEDDING_KIND,
            query_text,
            slugs,
            top_k=1,
        )
        if not ranked:
            return False
        _top_slug, similarity = ranked[0]
        return float(similarity) >= _DEDUP_SIMILARITY_THRESHOLD

    def _store_skill(
        self,
        *,
        proposal: SkillInductionProposal,
        episode: Episode,
        task_row: Task,
        installation_id: uuid.UUID,
        trajectory: list[dict[str, object]],
    ) -> None:
        """Persist the induced skill as untrusted + catalog-only with NO enablement."""
        name = proposal.name or f"induced-{episode.id}"
        description = proposal.description or "Agent-induced procedural skill."
        instructions_md = proposal.instructions_md or ""
        allowed_tools = proposal.allowed_tools or []

        # Build SKILL.md content
        tags_line = "agent-induced"
        skill_md = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"allowed_tools: {' '.join(allowed_tools)}\n"
            f"tags: {tags_line}\n"
            f"---\n\n"
            f"{instructions_md}\n"
        )

        # Build induction metadata that goes into metadata_json on the version
        induction_meta: dict[str, object] = {
            "induction_state": "candidate",
            "source_task_id": str(task_row.id),
            "source_episode_id": str(episode.id),
            "source_channel_id": episode.channel_id,
            "source_user_id": episode.user_id,
            "tool_call_count": len(trajectory),
            "confidence": proposal.confidence,
            "induction_reason": proposal.reason or "",
        }

        provenance = f"{SKILL_INDUCTION_PROVENANCE_PREFIX}:{episode.id}"

        ingestion = SkillIngestionService(
            self.session,
            embedding_index=self.embedding_index,
        )
        result = ingestion.ingest_markdown(
            skill_md,
            owner_type="workspace",
            owner_id=str(installation_id),
            provenance=provenance,
            trust_level="untrusted",
            created_by=SKILL_INDUCTION_CREATED_BY,
        )

        # Merge induction metadata onto the version's metadata_json
        existing_meta = dict(result.version.metadata_json or {})
        existing_meta.update(induction_meta)
        result.version.metadata_json = existing_meta
        self.session.flush()

        logger.info(
            "skill_induction created candidate skill slug=%s episode_id=%s "
            "installation_id=%s confidence=%.2f",
            result.skill.slug,
            episode.id,
            installation_id,
            proposal.confidence,
        )


# -- parsing -------------------------------------------------------------------


def parse_skill_induction_proposal(raw: str | None) -> SkillInductionProposal | None:
    """Parse and validate the LLM's JSON skill induction output."""
    if not raw:
        return None
    try:
        payload = json.loads(_extract_json_object(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    action = payload.get("action")
    if action not in ("CREATE", "NOOP"):
        return None

    name_raw = payload.get("name")
    name = (
        _slugify(str(name_raw))
        if isinstance(name_raw, str) and name_raw.strip()
        else None
    )

    description_raw = payload.get("description")
    description = (
        str(description_raw).strip()[:500]
        if isinstance(description_raw, str) and description_raw.strip()
        else None
    )

    allowed_tools_raw = payload.get("allowed_tools")
    allowed_tools: list[str] = []
    if isinstance(allowed_tools_raw, list):
        allowed_tools = [str(t) for t in allowed_tools_raw if isinstance(t, str)]

    instructions_raw = payload.get("instructions_md")
    instructions_md = (
        str(instructions_raw).strip()
        if isinstance(instructions_raw, str) and instructions_raw.strip()
        else None
    )

    confidence_raw = payload.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        confidence = 0.0

    reason_raw = payload.get("reason")
    reason = (
        str(reason_raw).strip()[:500]
        if isinstance(reason_raw, str) and reason_raw.strip()
        else None
    )

    # For CREATE, require name and description and minimum confidence
    if action == "CREATE" and (
        name is None or description is None or confidence < 0.70
    ):
        return SkillInductionProposal(
            action="NOOP",
            name=None,
            description=None,
            allowed_tools=[],
            instructions_md=None,
            confidence=0.0,
            reason="Proposal rejected: missing name/description or confidence < 0.70",
        )

    return SkillInductionProposal(
        action=str(action),
        name=name,
        description=description,
        allowed_tools=allowed_tools,
        instructions_md=instructions_md,
        confidence=confidence,
        reason=reason,
    )


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]


def _slugify(value: str) -> str:
    slug = _KEY_RE.sub("-", value.strip().lower()).strip("-")
    return slug[:80] if slug else f"induced-{uuid.uuid4().hex[:8]}"
