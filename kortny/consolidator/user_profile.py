"""User persona extraction (HIG-277 / Linear HIG-275).

An ambient consolidator pass that infers a USER's persona — their role and the
work surfaces their work lives in — from per-user observation evidence (Slack
title + channels they're active in + tools they use + connected integrations),
and *proposes* it as a user-scoped ``WorkspaceState`` fact for that user to
confirm. The user-level analog of the org-profile pass (HIG-271).

Why: "what's on my plate / what should I focus on" is persona-relative — a
developer's plate is issues + PRs, a CEO's is calendar + email. Capability
grounding (HIG-274) tells Kortny WHAT is connected; this tells it WHO is asking
so the request resolves to the right surfaces.

Trust (locked 2026-06-18): inferred persona is propose→confirm by default,
proposed only at confidence >= 0.6 (noise floor). High-confidence auto-activation
(>= 0.85) and Slack-title auto-trust land in follow-up slices. On confirm the
fact flows into the user's known-facts context injection automatically. The
persona is gated at request time (it only helps role-relative asks — PRISM), so
the producer just maintains the fact; the intent gate decides when to use it.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.db.models import ObservationEvent, Task, TaskEvent, TaskEventType
from kortny.llm import ChatMessage, LLMService
from kortny.memory.service import (
    PENDING_PROPOSAL_MESSAGE,
    ConfirmationPoster,
    WorkspaceStateService,
)
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity
from kortny.tools.types import JsonObject

logger = logging.getLogger(__name__)

USER_PROFILE_FACT_KEY = "user_profile"
USER_PROFILE_TASK_SOURCE = "user_profile_proposal"
USER_PROFILE_PROMPT_NAME = "kortny.user_profile_extractor"
USER_PROFILE_SOURCE_KIND = "observer_proposed"

# Locked thresholds (HIG-277): propose at >= 0.6; auto-activate at >= 0.85
# (higher than the scheduler's 0.72 — a wrong persona colours every answer).
DEFAULT_MIN_CONFIDENCE = Decimal("0.6")
DEFAULT_AUTO_ACTIVE_CONFIDENCE = Decimal("0.85")
DEFAULT_MIN_OBSERVED_EVENTS = 5
DEFAULT_REPROPOSE_COOLDOWN = timedelta(days=14)

_MAX_CHANNELS = 12
_MAX_TOOLS = 15

# Fixed work-surface vocabulary (HIG-277 locked) the persona maps to so intent /
# tool selection can use it deterministically.
WORK_SURFACES = (
    "issues",
    "prs",
    "calendar",
    "email",
    "docs",
    "pipeline",
    "accounts",
    "designs",
    "analytics",
)

USER_PROFILE_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
USER_PROFILE_SYSTEM_PROMPT = """\
You infer a concise PERSONA for one user of a Slack workspace, using only the \
activity evidence provided, so a coworker assistant can resolve "what's on my \
plate / what should I focus on" to the right tools for THIS person.

Return a single JSON object with these fields (omit or null when the evidence \
is too thin):
- role: a short role label grounded in the evidence (e.g. "Software Engineer", \
"CEO / Founder", "RevOps", "Designer"). Prefer the Slack title when present.
- work_surfaces: an array drawn ONLY from this fixed vocabulary — \
[issues, prs, calendar, email, docs, pipeline, accounts, designs, analytics] — \
the surfaces this user's work actually lives in, most important first.
- notes: array of other short, durable facts about how this user works.
- confidence: a number 0..1, your honest confidence this persona is correct for \
THIS user. Be conservative; if the evidence is thin, return confidence below \
0.4 and leave fields empty. Do NOT fabricate a role.

Output ONLY the JSON object, no prose."""


@dataclass(slots=True)
class UserProfileCounters:
    """Outcome of one user-profile pass run."""

    proposed: int = 0
    auto_activated: int = 0
    observed_events: int = 0
    skipped_reason: str | None = None
    failed: int = 0
    confidence: str | None = None

    def to_payload(self) -> JsonObject:
        payload: JsonObject = {
            "proposed": self.proposed,
            "auto_activated": self.auto_activated,
            "observed_events": self.observed_events,
            "failed": self.failed,
        }
        if self.skipped_reason is not None:
            payload["skipped_reason"] = self.skipped_reason
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


# Resolves a Slack user id to their DM ("D…") channel id, or None on failure.
DmChannelResolver = Callable[[str], str | None]
# Resolves a Slack user id to their profile title, or None.
SlackTitleResolver = Callable[[str], str | None]


class UserProfilePass:
    """Infer + propose a user persona fact (HIG-277)."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMService | None,
        poster: ConfirmationPoster | None,
        dm_channel_for_user: DmChannelResolver | None,
        slack_title_for_user: SlackTitleResolver | None = None,
        min_observed_events: int = DEFAULT_MIN_OBSERVED_EVENTS,
        min_confidence: Decimal = DEFAULT_MIN_CONFIDENCE,
        auto_active_confidence: Decimal = DEFAULT_AUTO_ACTIVE_CONFIDENCE,
        repropose_cooldown: timedelta = DEFAULT_REPROPOSE_COOLDOWN,
    ) -> None:
        self.session = session
        self.llm = llm
        self.poster = poster
        self.dm_channel_for_user = dm_channel_for_user
        self.slack_title_for_user = slack_title_for_user
        self.min_observed_events = min_observed_events
        self.min_confidence = min_confidence
        self.auto_active_confidence = auto_active_confidence
        self.repropose_cooldown = repropose_cooldown

    def run(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        task: Task,
        now: datetime | None = None,
    ) -> UserProfileCounters:
        effective_now = now or datetime.now(UTC)

        if self.llm is None or self.poster is None or self.dm_channel_for_user is None:
            return UserProfileCounters(skipped_reason="slack_or_llm_unavailable")

        # Already have an active persona for this user — nothing to do.
        if (
            WorkspaceStateService(self.session).get(
                installation_id, "user", user_id, USER_PROFILE_FACT_KEY
            )
            is not None
        ):
            return UserProfileCounters(skipped_reason="profile_exists")

        if self._recent_proposal_exists(installation_id, user_id, effective_now):
            return UserProfileCounters(skipped_reason="recent_proposal")

        observed_events = self._observed_event_count(installation_id, user_id)
        if observed_events < self.min_observed_events:
            return UserProfileCounters(
                skipped_reason="insufficient_evidence",
                observed_events=observed_events,
            )

        profile = self._extract(
            task=task, installation_id=installation_id, user_id=user_id
        )
        if profile is None:
            return UserProfileCounters(failed=1, observed_events=observed_events)

        confidence = _coerce_confidence(profile.get("confidence"))
        if confidence is None or confidence < self.min_confidence:
            return UserProfileCounters(
                skipped_reason="low_confidence",
                observed_events=observed_events,
                confidence=str(confidence) if confidence is not None else None,
            )

        profile["work_surfaces"] = _clean_surfaces(profile.get("work_surfaces"))

        # Locked trust model (HIG-277): high-confidence personas auto-activate
        # (no confirmation round-trip); mid-confidence ones propose→confirm.
        if confidence >= self.auto_active_confidence:
            self._auto_activate(
                installation_id=installation_id,
                user_id=user_id,
                profile=profile,
                confidence=confidence,
                task=task,
            )
            return UserProfileCounters(
                auto_activated=1,
                observed_events=observed_events,
                confidence=str(confidence),
            )

        proposed = self._propose(
            installation_id=installation_id,
            user_id=user_id,
            profile=profile,
            confidence=confidence,
            now=effective_now,
        )
        if not proposed:
            return UserProfileCounters(failed=1, observed_events=observed_events)
        return UserProfileCounters(
            proposed=1,
            observed_events=observed_events,
            confidence=str(confidence),
        )

    def _auto_activate(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        profile: JsonObject,
        confidence: Decimal,
        task: Task,
    ) -> None:
        memory_service = WorkspaceStateService(
            self.session, task_service=TaskService(self.session)
        )
        memory_service.set_active(
            installation_id,
            "user",
            user_id,
            USER_PROFILE_FACT_KEY,
            value=profile,
            source_task_id=task.id,
            value_text=_render_profile(profile),
            confidence_score=confidence,
            confidence_reason="High-confidence inference over observed activity.",
            source_kind=USER_PROFILE_SOURCE_KIND,
        )

    # -- internals ----------------------------------------------------------

    def _recent_proposal_exists(
        self, installation_id: uuid.UUID, user_id: str, now: datetime
    ) -> bool:
        cutoff = now - self.repropose_cooldown
        count = self.session.scalar(
            select(func.count())
            .select_from(TaskEvent)
            .where(
                TaskEvent.type == TaskEventType.log,
                TaskEvent.payload["message"].astext == PENDING_PROPOSAL_MESSAGE,
                TaskEvent.payload["key"].astext == USER_PROFILE_FACT_KEY,
                TaskEvent.payload["scope_id"].astext == user_id,
                TaskEvent.payload["installation_id"].astext == str(installation_id),
                TaskEvent.created_at >= cutoff,
            )
        )
        return bool(count)

    def _observed_event_count(self, installation_id: uuid.UUID, user_id: str) -> int:
        return (
            self.session.scalar(
                select(func.count())
                .select_from(ObservationEvent)
                .where(
                    ObservationEvent.installation_id == installation_id,
                    ObservationEvent.user_id == user_id,
                )
            )
            or 0
        )

    def _top_channels(
        self, installation_id: uuid.UUID, user_id: str
    ) -> list[dict[str, object]]:
        rows = self.session.execute(
            select(ObservationEvent.channel_id, func.count().label("n"))
            .where(
                ObservationEvent.installation_id == installation_id,
                ObservationEvent.user_id == user_id,
            )
            .group_by(ObservationEvent.channel_id)
            .order_by(func.count().desc())
            .limit(_MAX_CHANNELS)
        ).all()
        return [{"channel_id": cid, "events": n} for cid, n in rows]

    def _tools_used(self, installation_id: uuid.UUID, user_id: str) -> list[str]:
        # Bind the JSON-path expression once and reuse it in SELECT and GROUP BY;
        # repeating the accessor emits two distinct bind params and Postgres then
        # rejects the GROUP BY as not matching the select expression.
        tool_expr = TaskEvent.payload["tool"].astext
        rows = self.session.execute(
            select(tool_expr.label("tool"), func.count().label("n"))
            .select_from(TaskEvent)
            .join(Task, Task.id == TaskEvent.task_id)
            .where(
                Task.installation_id == installation_id,
                Task.slack_user_id == user_id,
                TaskEvent.type == TaskEventType.tool_call,
            )
            .group_by(tool_expr)
            .order_by(func.count().desc())
            .limit(_MAX_TOOLS)
        ).all()
        return [tool for tool, _ in rows if tool]

    def _extract(
        self, *, task: Task, installation_id: uuid.UUID, user_id: str
    ) -> JsonObject | None:
        assert self.llm is not None  # gated in run()
        title = (
            self.slack_title_for_user(user_id) if self.slack_title_for_user else None
        )
        evidence = {
            "slack_title": title,
            "active_channels": self._top_channels(installation_id, user_id),
            "tools_used": self._tools_used(installation_id, user_id),
        }
        try:
            completion = self.llm.complete(
                task_id=task.id,
                messages=(
                    ChatMessage(role="system", content=USER_PROFILE_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            evidence, separators=(",", ":"), sort_keys=True
                        ),
                    ),
                ),
                response_format=USER_PROFILE_RESPONSE_FORMAT,
                prompt_name=USER_PROFILE_PROMPT_NAME,
            )
            parsed = json.loads(completion.content or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.info(
                "user profile extraction failed task_id=%s error_type=%s error=%s",
                task.id,
                type(exc).__name__,
                exc,
            )
            return None
        return parsed if isinstance(parsed, dict) else None

    def _propose(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        profile: JsonObject,
        confidence: Decimal,
        now: datetime,
    ) -> bool:
        assert self.dm_channel_for_user is not None and self.poster is not None
        dm_channel_id = self.dm_channel_for_user(user_id)
        if not dm_channel_id:
            logger.info(
                "user profile proposal skipped: could not open DM user_id=%s", user_id
            )
            return False

        task_service = TaskService(self.session)
        synthetic_ts = f"userprofile-{user_id}-{int(now.timestamp())}"
        prompt_task = task_service.create_task(
            installation_id=installation_id,
            slack_channel_id=dm_channel_id,
            slack_user_id=user_id,
            slack_message_ts=synthetic_ts,
            input="Confirm Kortny's inferred profile of how you work.",
            identity=TaskIdentity.synthetic(
                source=USER_PROFILE_TASK_SOURCE,
                source_id=f"{user_id}:{synthetic_ts}",
                input_text="user_profile_proposal",
            ),
            source_surface=USER_PROFILE_TASK_SOURCE,
        )
        memory_service = WorkspaceStateService(
            self.session,
            task_service=task_service,
            poster=self.poster,
        )
        memory_service.propose(
            installation_id,
            "user",
            user_id,
            USER_PROFILE_FACT_KEY,
            value=profile,
            source_task_id=prompt_task.id,
            value_text=_render_profile(profile),
            proposed_reason="Inferred from your channel activity and tool use.",
            confidence_score=confidence,
            confidence_reason="LLM inference over your observed activity.",
            source_kind=USER_PROFILE_SOURCE_KIND,
        )
        return True


def _coerce_confidence(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            score = Decimal(str(value))
        except (ValueError, ArithmeticError):
            return None
        return score if 0 <= score <= 1 else None
    return None


def _clean_surfaces(value: object) -> list[str]:
    """Keep only known surfaces, in order, deduped — enforce the fixed vocab."""

    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        s = str(item).strip().lower()
        if s in WORK_SURFACES and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _render_profile(profile: JsonObject) -> str:
    """Render the persona into a compact human-readable user fact."""

    parts: list[str] = []
    role = profile.get("role")
    if isinstance(role, str) and role.strip():
        parts.append(f"Role: {role.strip()}")
    surfaces = profile.get("work_surfaces")
    if isinstance(surfaces, list) and surfaces:
        parts.append(f"Works in: {', '.join(str(s) for s in surfaces)}")
    notes = profile.get("notes")
    if isinstance(notes, list):
        items = [str(n).strip() for n in notes if str(n).strip()]
        if items:
            parts.append(f"Notes: {'; '.join(items)}")
    return ". ".join(parts) if parts else "User work persona."


__all__ = [
    "USER_PROFILE_FACT_KEY",
    "WORK_SURFACES",
    "UserProfileCounters",
    "UserProfilePass",
]
