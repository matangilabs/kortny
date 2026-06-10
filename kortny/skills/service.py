"""Procedural skill registry service."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    ProceduralSkill,
    ProceduralSkillInvocation,
    ProceduralSkillVersion,
    SkillEnablement,
    SkillFile,
    Task,
    TaskEventType,
)
from kortny.skills.builtins import BUILTIN_SKILLS, BuiltInSkillDefinition
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject

SKILL_CATALOG_BUILT_MESSAGE = "procedural_skill_catalog_built"
SKILL_INVOKED_MESSAGE = "procedural_skill_invoked"
RESPONSE_HUMANIZER_INVOCATION = "response_humanizer"
EXECUTION_INVOCATION = "execution"

CURATED_SKILLS_DIR = Path(__file__).parent / "curated"
SKILL_SCOPE_TYPES = frozenset({"workspace", "channel", "user"})
_SCOPE_SPECIFICITY = {"workspace": 0, "channel": 1, "user": 2}


@dataclass(frozen=True, slots=True)
class EnabledSkill:
    """A skill enabled for a task's scope, with its latest active version."""

    skill_id: uuid.UUID
    version_id: uuid.UUID
    slug: str
    name: str
    version: str
    description: str
    trust_level: str
    scope_type: str
    scope_id: str | None
    has_references: bool
    has_scripts: bool


@dataclass(frozen=True, slots=True)
class SkillActivation:
    """Selected procedural skill and the exact version used for a task."""

    skill_id: uuid.UUID
    skill_version_id: uuid.UUID
    slug: str
    name: str
    version: str
    owner_type: str
    trust_level: str
    instructions_md: str
    selected_reason: str

    def to_response_payload(self) -> JsonObject:
        return {
            "slug": self.slug,
            "name": self.name,
            "version": self.version,
            "owner_type": self.owner_type,
            "trust_level": self.trust_level,
            "selected_reason": self.selected_reason,
            "instructions_md": self.instructions_md,
        }

    def to_trace_payload(self) -> JsonObject:
        return {
            "skill_id": str(self.skill_id),
            "skill_version_id": str(self.skill_version_id),
            "slug": self.slug,
            "name": self.name,
            "version": self.version,
            "owner_type": self.owner_type,
            "trust_level": self.trust_level,
            "selected_reason": self.selected_reason,
        }


class SkillRegistryService:
    """Application service for built-in procedural skills."""

    def __init__(
        self,
        session: Session,
        *,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task_service = task_service or TaskService(session)

    def ensure_builtin_skills(self) -> None:
        """Idempotently seed system-owned built-in skill definitions."""

        for definition in BUILTIN_SKILLS:
            self._ensure_builtin_skill(definition)
        self.session.flush()

    def ensure_curated_skills(self) -> None:
        """Idempotently seed the curated execution-time skill catalog."""

        from kortny.skills.ingestion import SkillIngestionService

        if not CURATED_SKILLS_DIR.is_dir():
            return
        ingestion = SkillIngestionService(self.session)
        for skill_dir in sorted(CURATED_SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            ingestion.ingest_directory(
                skill_dir,
                owner_type="system",
                owner_id=None,
                provenance="kortny",
                trust_level="trusted",
                created_by="system",
            )
        self.session.flush()

    def enable_skill(
        self,
        *,
        installation_id: uuid.UUID,
        skill_id: uuid.UUID,
        scope_type: str,
        scope_id: str | None,
        added_by: str,
    ) -> SkillEnablement:
        """Enable a skill for a scope; re-enables a disabled enablement."""

        _validate_skill_scope(scope_type, scope_id)
        existing = self.session.scalar(
            select(SkillEnablement).where(
                SkillEnablement.installation_id == installation_id,
                SkillEnablement.skill_id == skill_id,
                SkillEnablement.scope_type == scope_type,
                SkillEnablement.scope_id == scope_id
                if scope_id is not None
                else SkillEnablement.scope_id.is_(None),
            )
        )
        if existing is not None:
            existing.status = "enabled"
            existing.added_by = added_by
            self.session.flush()
            return existing
        enablement = SkillEnablement(
            installation_id=installation_id,
            skill_id=skill_id,
            scope_type=scope_type,
            scope_id=scope_id,
            status="enabled",
            added_by=added_by,
        )
        self.session.add(enablement)
        self.session.flush()
        return enablement

    def disable_skill(self, *, enablement_id: uuid.UUID, by: str) -> SkillEnablement:
        """Disable one enablement, keeping the row for audit."""

        enablement = self.session.get(SkillEnablement, enablement_id)
        if enablement is None:
            raise ValueError(f"Skill enablement {enablement_id} not found.")
        enablement.status = "disabled"
        enablement.added_by = by
        self.session.flush()
        return enablement

    def enabled_skills_for_task(self, task: Task) -> list[EnabledSkill]:
        """Resolve skills enabled for a task's workspace/channel/user scopes.

        One row per skill; when a skill is enabled at several scopes the most
        specific scope (user > channel > workspace) wins for attribution.
        """

        scope_filter = SkillEnablement.scope_type == "workspace"
        if task.slack_channel_id:
            scope_filter = scope_filter | (
                (SkillEnablement.scope_type == "channel")
                & (SkillEnablement.scope_id == task.slack_channel_id)
            )
        if task.slack_user_id:
            scope_filter = scope_filter | (
                (SkillEnablement.scope_type == "user")
                & (SkillEnablement.scope_id == task.slack_user_id)
            )
        rows = self.session.execute(
            select(SkillEnablement, ProceduralSkill, ProceduralSkillVersion)
            .join(ProceduralSkill, ProceduralSkill.id == SkillEnablement.skill_id)
            .join(
                ProceduralSkillVersion,
                ProceduralSkillVersion.skill_id == ProceduralSkill.id,
            )
            .where(
                SkillEnablement.installation_id == task.installation_id,
                SkillEnablement.status == "enabled",
                scope_filter,
                ProceduralSkill.status == "active",
                ProceduralSkillVersion.status == "active",
            )
            .order_by(ProceduralSkill.slug)
        )
        by_skill: dict[uuid.UUID, EnabledSkill] = {}
        for enablement, skill, version in rows:
            current = by_skill.get(skill.id)
            if (
                current is not None
                and _SCOPE_SPECIFICITY[current.scope_type]
                >= _SCOPE_SPECIFICITY[enablement.scope_type]
            ):
                continue
            file_kinds = set(
                self.session.scalars(
                    select(SkillFile.kind).where(
                        SkillFile.skill_version_id == version.id
                    )
                )
            )
            by_skill[skill.id] = EnabledSkill(
                skill_id=skill.id,
                version_id=version.id,
                slug=skill.slug,
                name=version.name,
                version=version.version,
                description=version.description,
                trust_level=skill.trust_level,
                scope_type=enablement.scope_type,
                scope_id=enablement.scope_id,
                has_references="reference" in file_kinds or "asset" in file_kinds,
                has_scripts="script" in file_kinds,
            )
        return sorted(by_skill.values(), key=lambda item: item.slug)

    def select_for_response(
        self,
        task: Task,
        *,
        response_mode: str,
        response_shape: str | None = None,
        invocation_kind: str = RESPONSE_HUMANIZER_INVOCATION,
    ) -> list[SkillActivation]:
        """Return active system skills for a response path and record selection."""

        self.ensure_builtin_skills()
        candidates = self._candidate_system_skills(response_mode=response_mode)
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": SKILL_CATALOG_BUILT_MESSAGE,
                "invocation_kind": invocation_kind,
                "response_mode": response_mode,
                "response_shape": response_shape,
                "candidate_count": len(candidates),
                "candidate_slugs": [candidate.slug for candidate in candidates],
            },
        )
        selected = self._select_candidates(
            candidates,
            invocation_kind=invocation_kind,
            response_shape=response_shape,
        )
        for activation in selected:
            self.record_invocation(
                task,
                activation=activation,
                invocation_kind=invocation_kind,
                response_mode=response_mode,
                response_shape=response_shape,
            )
        return selected

    def record_invocation(
        self,
        task: Task,
        *,
        activation: SkillActivation,
        invocation_kind: str,
        response_mode: str,
        response_shape: str | None = None,
    ) -> ProceduralSkillInvocation:
        """Persist a skill invocation and mirror it into task_events."""

        trace_payload = activation.to_trace_payload()
        if response_shape is not None:
            trace_payload["response_shape"] = response_shape
        invocation = ProceduralSkillInvocation(
            installation_id=task.installation_id,
            task_id=task.id,
            skill_id=activation.skill_id,
            skill_version_id=activation.skill_version_id,
            invocation_kind=invocation_kind,
            response_mode=response_mode,
            selected_reason=activation.selected_reason,
            payload=trace_payload,
        )
        self.session.add(invocation)
        self.session.flush()
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": SKILL_INVOKED_MESSAGE,
                "invocation_id": str(invocation.id),
                "invocation_kind": invocation_kind,
                "response_mode": response_mode,
                "response_shape": response_shape,
                **trace_payload,
            },
        )
        return invocation

    def _ensure_builtin_skill(self, definition: BuiltInSkillDefinition) -> None:
        skill = self.session.scalar(
            select(ProceduralSkill).where(
                ProceduralSkill.owner_type == "system",
                ProceduralSkill.owner_id.is_(None),
                ProceduralSkill.slug == definition.slug,
            )
        )
        if skill is None:
            skill = ProceduralSkill(
                slug=definition.slug,
                owner_type="system",
                owner_id=None,
                status="active",
                trust_level="trusted",
                visibility="catalog",
            )
            self.session.add(skill)
            self.session.flush()
        else:
            skill.status = "active"
            skill.trust_level = "trusted"
            skill.visibility = "catalog"

        content_hash = _content_sha256(definition)
        version = self.session.scalar(
            select(ProceduralSkillVersion).where(
                ProceduralSkillVersion.skill_id == skill.id,
                ProceduralSkillVersion.version == definition.version,
            )
        )
        if version is None:
            version = ProceduralSkillVersion(
                skill_id=skill.id,
                version=definition.version,
                status="active",
                name=definition.name,
                description=definition.description,
                instructions_md=definition.instructions_md,
                intent_tags=list(definition.intent_tags),
                response_modes=list(definition.response_modes),
                trigger_phrases=list(definition.trigger_phrases),
                allowed_tools=[],
                metadata_json=definition.metadata or {},
                content_sha256=content_hash,
                created_by="system",
                approved_by="system",
                published_at=datetime.now(UTC),
            )
            self.session.add(version)
            return

        version.status = "active"
        version.name = definition.name
        version.description = definition.description
        version.instructions_md = definition.instructions_md
        version.intent_tags = list(definition.intent_tags)
        version.response_modes = list(definition.response_modes)
        version.trigger_phrases = list(definition.trigger_phrases)
        version.allowed_tools = []
        version.metadata_json = definition.metadata or {}
        version.content_sha256 = content_hash
        version.created_by = "system"
        version.approved_by = "system"
        if version.published_at is None:
            version.published_at = datetime.now(UTC)

    def _candidate_system_skills(self, *, response_mode: str) -> list[SkillActivation]:
        rows = self.session.execute(
            select(ProceduralSkill, ProceduralSkillVersion)
            .join(
                ProceduralSkillVersion,
                ProceduralSkillVersion.skill_id == ProceduralSkill.id,
            )
            .where(
                ProceduralSkill.owner_type == "system",
                ProceduralSkill.status == "active",
                ProceduralSkill.visibility == "catalog",
                ProceduralSkillVersion.status == "active",
            )
            .order_by(ProceduralSkill.slug, ProceduralSkillVersion.version.desc())
        )
        candidates: list[SkillActivation] = []
        seen_slugs: set[str] = set()
        for skill, version in rows:
            if skill.slug in seen_slugs:
                continue
            modes = _string_set(version.response_modes)
            if response_mode not in modes and "all" not in modes:
                continue
            seen_slugs.add(skill.slug)
            candidates.append(
                SkillActivation(
                    skill_id=skill.id,
                    skill_version_id=version.id,
                    slug=skill.slug,
                    name=version.name,
                    version=version.version,
                    owner_type=skill.owner_type,
                    trust_level=skill.trust_level,
                    instructions_md=version.instructions_md,
                    selected_reason=f"matches response_mode={response_mode}",
                )
            )
        return candidates

    def _select_candidates(
        self,
        candidates: list[SkillActivation],
        *,
        invocation_kind: str,
        response_shape: str | None,
    ) -> list[SkillActivation]:
        by_slug = {candidate.slug: candidate for candidate in candidates}
        if invocation_kind == RESPONSE_HUMANIZER_INVOCATION:
            selected: list[SkillActivation] = []
            self._append_selected(
                selected,
                by_slug,
                "slack-humanizer",
                reason="built-in rendering skill for response humanizer",
            )
            if response_shape in {"analyst_audit", "comparison_memo"}:
                self._append_selected(
                    selected,
                    by_slug,
                    "analyst-grade-synthesis",
                    reason=f"matches analyst response_shape={response_shape}",
                )
            elif response_shape == "research_brief":
                self._append_selected(
                    selected,
                    by_slug,
                    "research-synthesis",
                    reason="matches research brief response shape",
                )
            elif response_shape == "status_recap":
                self._append_selected(
                    selected,
                    by_slug,
                    "status-recap",
                    reason="matches status recap response shape",
                )
            elif response_shape in {"document_delivery", "file_review"}:
                self._append_selected(
                    selected,
                    by_slug,
                    "document-iteration",
                    reason=f"matches document response_shape={response_shape}",
                )
            if selected:
                return selected
        return candidates[:1]

    def _append_selected(
        self,
        selected: list[SkillActivation],
        candidates: dict[str, SkillActivation],
        slug: str,
        *,
        reason: str,
    ) -> None:
        candidate = candidates.get(slug)
        if candidate is None:
            return
        selected.append(
            SkillActivation(
                skill_id=candidate.skill_id,
                skill_version_id=candidate.skill_version_id,
                slug=candidate.slug,
                name=candidate.name,
                version=candidate.version,
                owner_type=candidate.owner_type,
                trust_level=candidate.trust_level,
                instructions_md=candidate.instructions_md,
                selected_reason=reason,
            )
        )


def _content_sha256(definition: BuiltInSkillDefinition) -> str:
    payload: dict[str, Any] = {
        "slug": definition.slug,
        "name": definition.name,
        "version": definition.version,
        "description": definition.description,
        "instructions_md": definition.instructions_md,
        "intent_tags": list(definition.intent_tags),
        "response_modes": list(definition.response_modes),
        "trigger_phrases": list(definition.trigger_phrases),
        "metadata": definition.metadata or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_skill_scope(scope_type: str, scope_id: str | None) -> None:
    if scope_type not in SKILL_SCOPE_TYPES:
        raise ValueError(f"Invalid skill scope_type: {scope_type!r}")
    if scope_type == "workspace" and scope_id is not None:
        raise ValueError("workspace scope must not carry a scope_id")
    if scope_type in {"channel", "user"} and not scope_id:
        raise ValueError(f"{scope_type} scope requires a scope_id")


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}
