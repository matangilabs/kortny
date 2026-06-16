"""Workspace org-profile extraction (HIG-271).

A dedicated ambient consolidator pass that infers the firm's identity — what
company this workspace belongs to, what it does, who its customers and
competitors are — from cross-channel observation evidence, and *proposes* it as
a workspace-scoped ``WorkspaceState`` fact for an admin to confirm.

Why a pass, not install-time: org identity is far clearer once Kortny has
observed several channels than from the workspace name alone. Why propose, not
auto-activate: the profile is *inferred*; a wrong guess colouring every future
answer is worse than one confirmation step. On confirm, the fact flows into
both known-facts context injection and the knowledge graph automatically
(``project_confirmed_facts``), so the agent stops asking "who is your company?".

Scope safety: the inferred fact is workspace-scoped, but it is only ever a
*proposal* until a human confirms it — it never auto-promotes channel/DM
evidence into an active workspace fact.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ObserveChannelProfile,
    Task,
    TaskEvent,
    TaskEventType,
)
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

ORG_PROFILE_FACT_KEY = "org_profile"
ORG_PROFILE_TASK_SOURCE = "org_profile_proposal"
ORG_PROFILE_PROMPT_NAME = "kortny.org_profile_extractor"
ORG_PROFILE_SOURCE_KIND = "observer_proposed"

DEFAULT_MIN_CHANNEL_PROFILES = 3
DEFAULT_MIN_CONFIDENCE = Decimal("0.6")
DEFAULT_REPROPOSE_COOLDOWN = timedelta(days=14)

_MAX_PROFILE_SUMMARY_CHARS = 1000
_MAX_PROFILES = 25

# Open profile schema: every dimension optional so the model only fills what the
# evidence supports. Competitors are one dimension among many — pricing asks need
# the business model, GTM asks need customers — so nothing is privileged.
ORG_PROFILE_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
ORG_PROFILE_SYSTEM_PROMPT = """\
You infer a concise profile of the COMPANY that owns a Slack workspace, using \
only the channel-activity evidence provided. This profile helps a coworker \
assistant answer questions like "compare us to our competitors" without asking \
the user who they even are.

Return a single JSON object with these OPTIONAL string/array fields — include a \
field ONLY when the evidence genuinely supports it, otherwise omit it or use \
null/[]:
- company_name: the org's name
- what_we_do: one sentence on the product / value proposition
- industry: the market/category
- business_model: how they make money (e.g. B2B SaaS, marketplace)
- target_customers: who they sell to
- competitors: array of named rival companies
- stage_or_funding: company stage / funding if evident
- other_notes: array of other firm-level facts worth remembering

Also return:
- confidence: a number 0..1 — your honest confidence that this profile is \
correct for THIS workspace. Be conservative; do NOT fabricate a company. If the \
evidence is too thin to identify the firm, return confidence below 0.4 and leave \
fields empty.

Output ONLY the JSON object, no prose."""


@dataclass(slots=True)
class OrgProfileCounters:
    """Outcome of one org-profile pass run."""

    proposed: int = 0
    channel_profiles: int = 0
    skipped_reason: str | None = None
    failed: int = 0
    confidence: str | None = None

    def to_payload(self) -> JsonObject:
        payload: JsonObject = {
            "proposed": self.proposed,
            "channel_profiles": self.channel_profiles,
            "failed": self.failed,
        }
        if self.skipped_reason is not None:
            payload["skipped_reason"] = self.skipped_reason
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


# Resolves a Slack user id to their DM ("D…") channel id, or None on failure.
# Production wraps conversations.open; tests inject a stub. The DM channel (not
# the raw user id) is required so the posting boundary treats it as a linear DM
# and does not try to thread the prompt under a synthetic ts.
DmChannelResolver = Callable[[str], str | None]


class OrgProfilePass:
    """Infer + propose a workspace org-profile fact (HIG-271)."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMService | None,
        poster: ConfirmationPoster | None,
        dm_channel_for_user: DmChannelResolver | None,
        min_channel_profiles: int = DEFAULT_MIN_CHANNEL_PROFILES,
        min_confidence: Decimal = DEFAULT_MIN_CONFIDENCE,
        repropose_cooldown: timedelta = DEFAULT_REPROPOSE_COOLDOWN,
    ) -> None:
        self.session = session
        self.llm = llm
        self.poster = poster
        self.dm_channel_for_user = dm_channel_for_user
        self.min_channel_profiles = min_channel_profiles
        self.min_confidence = min_confidence
        self.repropose_cooldown = repropose_cooldown

    def run(
        self,
        *,
        installation_id: uuid.UUID,
        task: Task,
        now: datetime | None = None,
    ) -> OrgProfileCounters:
        effective_now = now or datetime.now(UTC)

        if self.llm is None or self.poster is None or self.dm_channel_for_user is None:
            return OrgProfileCounters(skipped_reason="slack_or_llm_unavailable")

        installation = self.session.get(Installation, installation_id)
        admin_user_id = (
            installation.primary_admin_user_id if installation is not None else None
        )
        if not admin_user_id:
            return OrgProfileCounters(skipped_reason="no_primary_admin")

        # Already have an active org profile — nothing to do.
        if (
            WorkspaceStateService(self.session).get(
                installation_id, "workspace", None, ORG_PROFILE_FACT_KEY
            )
            is not None
        ):
            return OrgProfileCounters(skipped_reason="profile_exists")

        # A pending proposal not yet confirmed is not a WorkspaceState row, so
        # gate on the proposal audit event to avoid re-prompting every run.
        if self._recent_proposal_exists(installation_id, effective_now):
            return OrgProfileCounters(skipped_reason="recent_proposal")

        profiles = list(
            self.session.scalars(
                select(ObserveChannelProfile)
                .where(
                    ObserveChannelProfile.installation_id == installation_id,
                    ObserveChannelProfile.profile_status == "active",
                )
                .order_by(ObserveChannelProfile.channel_id)
                .limit(_MAX_PROFILES)
            )
        )
        if len(profiles) < self.min_channel_profiles:
            return OrgProfileCounters(
                skipped_reason="insufficient_evidence",
                channel_profiles=len(profiles),
            )

        profile = self._extract(task=task, installation=installation, profiles=profiles)
        if profile is None:
            return OrgProfileCounters(failed=1, channel_profiles=len(profiles))

        confidence = _coerce_confidence(profile.get("confidence"))
        if confidence is None or confidence < self.min_confidence:
            return OrgProfileCounters(
                skipped_reason="low_confidence",
                channel_profiles=len(profiles),
                confidence=str(confidence) if confidence is not None else None,
            )

        proposed = self._propose(
            installation_id=installation_id,
            admin_user_id=admin_user_id,
            profile=profile,
            confidence=confidence,
            now=effective_now,
        )
        if not proposed:
            return OrgProfileCounters(failed=1, channel_profiles=len(profiles))
        return OrgProfileCounters(
            proposed=1,
            channel_profiles=len(profiles),
            confidence=str(confidence),
        )

    # -- internals ----------------------------------------------------------

    def _recent_proposal_exists(
        self, installation_id: uuid.UUID, now: datetime
    ) -> bool:
        cutoff = now - self.repropose_cooldown
        count = self.session.scalar(
            select(func.count())
            .select_from(TaskEvent)
            .where(
                TaskEvent.type == TaskEventType.log,
                TaskEvent.payload["message"].astext == PENDING_PROPOSAL_MESSAGE,
                TaskEvent.payload["key"].astext == ORG_PROFILE_FACT_KEY,
                TaskEvent.payload["installation_id"].astext == str(installation_id),
                TaskEvent.created_at >= cutoff,
            )
        )
        return bool(count)

    def _extract(
        self,
        *,
        task: Task,
        installation: Installation | None,
        profiles: Sequence[ObserveChannelProfile],
    ) -> JsonObject | None:
        assert self.llm is not None  # gated in run()
        evidence = {
            "workspace_name": (installation.team_name if installation else None),
            "channels": [
                {
                    "channel_id": profile.channel_id,
                    "summary": (profile.summary or "")[:_MAX_PROFILE_SUMMARY_CHARS],
                }
                for profile in profiles
            ],
        }
        try:
            completion = self.llm.complete(
                task_id=task.id,
                messages=(
                    ChatMessage(role="system", content=ORG_PROFILE_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            evidence, separators=(",", ":"), sort_keys=True
                        ),
                    ),
                ),
                response_format=ORG_PROFILE_RESPONSE_FORMAT,
                prompt_name=ORG_PROFILE_PROMPT_NAME,
            )
            parsed = json.loads(completion.content or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.info(
                "org profile extraction failed task_id=%s error_type=%s error=%s",
                task.id,
                type(exc).__name__,
                exc,
            )
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _propose(
        self,
        *,
        installation_id: uuid.UUID,
        admin_user_id: str,
        profile: JsonObject,
        confidence: Decimal,
        now: datetime,
    ) -> bool:
        assert self.dm_channel_for_user is not None and self.poster is not None
        dm_channel_id = self.dm_channel_for_user(admin_user_id)
        if not dm_channel_id:
            logger.info(
                "org profile proposal skipped: could not open DM installation_id=%s",
                installation_id,
            )
            return False

        task_service = TaskService(self.session)
        # Anchor the proposal to the admin's DM so propose() posts the
        # confirmation prompt there. A synthetic message_ts satisfies the thread
        # helper; the "D…" channel makes the posting boundary keep it linear.
        synthetic_ts = f"orgprofile-{installation_id}-{int(now.timestamp())}"
        prompt_task = task_service.create_task(
            installation_id=installation_id,
            slack_channel_id=dm_channel_id,
            slack_user_id=admin_user_id,
            slack_message_ts=synthetic_ts,
            input="Confirm Kortny's inferred profile of your company.",
            identity=TaskIdentity.synthetic(
                source=ORG_PROFILE_TASK_SOURCE,
                source_id=f"{installation_id}:{synthetic_ts}",
                input_text="org_profile_proposal",
            ),
            source_surface=ORG_PROFILE_TASK_SOURCE,
        )
        memory_service = WorkspaceStateService(
            self.session,
            task_service=task_service,
            poster=self.poster,
        )
        memory_service.propose(
            installation_id,
            "workspace",
            None,
            ORG_PROFILE_FACT_KEY,
            value=profile,
            source_task_id=prompt_task.id,
            value_text=_render_profile(profile),
            proposed_reason="Inferred from this workspace's channel activity.",
            confidence_score=confidence,
            confidence_reason="LLM inference over channel profiles.",
            source_kind=ORG_PROFILE_SOURCE_KIND,
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
        if score < 0 or score > 1:
            return None
        return score
    return None


_PROFILE_LABELS: tuple[tuple[str, str], ...] = (
    ("company_name", "Company"),
    ("what_we_do", "What we do"),
    ("industry", "Industry"),
    ("business_model", "Business model"),
    ("target_customers", "Target customers"),
    ("competitors", "Key competitors"),
    ("stage_or_funding", "Stage/funding"),
    ("other_notes", "Notes"),
)


def _render_profile(profile: JsonObject) -> str:
    """Render the open profile into a compact human-readable workspace fact."""

    parts: list[str] = []
    for key, label in _PROFILE_LABELS:
        value = profile.get(key)
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                parts.append(f"{label}: {', '.join(items)}")
        elif isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")
    return ". ".join(parts) if parts else "Workspace organisation profile."


__all__ = [
    "ORG_PROFILE_FACT_KEY",
    "OrgProfileCounters",
    "OrgProfilePass",
]
