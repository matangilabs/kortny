"""Read queries for the dashboard skills directory."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    ProceduralSkill,
    ProceduralSkillInvocation,
    ProceduralSkillVersion,
    SkillEnablement,
    SkillFile,
)

SKILL_MD_FORMAT = "skill_md"
INVOCATION_WINDOW_DAYS = 30


@dataclass(frozen=True, slots=True)
class SkillScopeChip:
    enablement_id: uuid.UUID
    scope_type: str
    scope_id: str | None
    status: str


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    skill_id: uuid.UUID
    slug: str
    name: str
    description: str
    version: str
    provenance: str
    trust_level: str
    owner_type: str
    has_references: bool
    has_scripts: bool
    enabled_scopes: tuple[SkillScopeChip, ...]
    invocations_30d: int

    @property
    def is_enabled(self) -> bool:
        return any(chip.status == "enabled" for chip in self.enabled_scopes)


@dataclass(frozen=True, slots=True)
class SkillsDashboard:
    curated: tuple[SkillCatalogEntry, ...]
    custom: tuple[SkillCatalogEntry, ...]

    @property
    def enabled_count(self) -> int:
        return sum(1 for entry in (*self.curated, *self.custom) if entry.is_enabled)


@dataclass(frozen=True, slots=True)
class SkillFileRow:
    path: str
    kind: str
    size_bytes: int
    is_binary: bool


@dataclass(frozen=True, slots=True)
class SkillVersionRow:
    version: str
    status: str
    created_by: str
    published_at: datetime | None


@dataclass(frozen=True, slots=True)
class SkillInvocationRow:
    task_id: uuid.UUID
    invocation_kind: str
    selected_reason: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SkillDetail:
    entry: SkillCatalogEntry
    instructions_md: str
    allowed_tools: tuple[str, ...]
    files: tuple[SkillFileRow, ...]
    versions: tuple[SkillVersionRow, ...]
    invocations: tuple[SkillInvocationRow, ...] = field(default=())


def get_skills_dashboard(
    session: Session,
    installation_id: uuid.UUID | None,
) -> SkillsDashboard:
    """Catalog of curated skills plus this installation's custom skills."""

    curated: list[SkillCatalogEntry] = []
    custom: list[SkillCatalogEntry] = []
    for skill, version in _directory_rows(session, installation_id):
        entry = _catalog_entry(session, skill, version, installation_id)
        if skill.owner_type == "system":
            curated.append(entry)
        else:
            custom.append(entry)
    return SkillsDashboard(curated=tuple(curated), custom=tuple(custom))


def get_skill_detail(
    session: Session,
    installation_id: uuid.UUID | None,
    skill_id: uuid.UUID,
) -> SkillDetail | None:
    skill = session.get(ProceduralSkill, skill_id)
    if skill is None:
        return None
    if skill.owner_type != "system" and (
        installation_id is None or skill.owner_id != str(installation_id)
    ):
        return None
    version = _active_version(session, skill.id)
    if version is None or not _is_skill_md(version):
        return None
    entry = _catalog_entry(session, skill, version, installation_id)
    files = tuple(
        SkillFileRow(
            path=row.path,
            kind=row.kind,
            size_bytes=row.size_bytes,
            is_binary=row.content_text is None,
        )
        for row in session.scalars(
            select(SkillFile)
            .where(SkillFile.skill_version_id == version.id)
            .order_by(SkillFile.path)
        )
    )
    versions = tuple(
        SkillVersionRow(
            version=row.version,
            status=row.status,
            created_by=row.created_by,
            published_at=row.published_at,
        )
        for row in session.scalars(
            select(ProceduralSkillVersion)
            .where(ProceduralSkillVersion.skill_id == skill.id)
            .order_by(ProceduralSkillVersion.created_at.desc())
        )
    )
    invocation_filters = [ProceduralSkillInvocation.skill_id == skill.id]
    if installation_id is not None:
        invocation_filters.append(
            ProceduralSkillInvocation.installation_id == installation_id
        )
    invocations = tuple(
        SkillInvocationRow(
            task_id=row.task_id,
            invocation_kind=row.invocation_kind,
            selected_reason=row.selected_reason,
            created_at=row.created_at,
        )
        for row in session.scalars(
            select(ProceduralSkillInvocation)
            .where(*invocation_filters)
            .order_by(ProceduralSkillInvocation.created_at.desc())
            .limit(20)
        )
    )
    allowed_tools = tuple(
        item for item in version.allowed_tools if isinstance(item, str)
    )
    return SkillDetail(
        entry=entry,
        instructions_md=version.instructions_md,
        allowed_tools=allowed_tools,
        files=files,
        versions=versions,
        invocations=invocations,
    )


def _directory_rows(
    session: Session,
    installation_id: uuid.UUID | None,
) -> list[tuple[ProceduralSkill, ProceduralSkillVersion]]:
    owner_filter = ProceduralSkill.owner_type == "system"
    if installation_id is not None:
        owner_filter = owner_filter | (
            (ProceduralSkill.owner_type == "workspace")
            & (ProceduralSkill.owner_id == str(installation_id))
        )
    rows = session.execute(
        select(ProceduralSkill, ProceduralSkillVersion)
        .join(
            ProceduralSkillVersion,
            ProceduralSkillVersion.skill_id == ProceduralSkill.id,
        )
        .where(
            owner_filter,
            ProceduralSkill.status == "active",
            ProceduralSkillVersion.status == "active",
        )
        .order_by(ProceduralSkill.slug)
    )
    return [(skill, version) for skill, version in rows if _is_skill_md(version)]


def _catalog_entry(
    session: Session,
    skill: ProceduralSkill,
    version: ProceduralSkillVersion,
    installation_id: uuid.UUID | None,
) -> SkillCatalogEntry:
    file_kinds = set(
        session.scalars(
            select(SkillFile.kind).where(SkillFile.skill_version_id == version.id)
        )
    )
    chips: tuple[SkillScopeChip, ...] = ()
    invocations_30d = 0
    if installation_id is not None:
        chips = tuple(
            SkillScopeChip(
                enablement_id=row.id,
                scope_type=row.scope_type,
                scope_id=row.scope_id,
                status=row.status,
            )
            for row in session.scalars(
                select(SkillEnablement)
                .where(
                    SkillEnablement.installation_id == installation_id,
                    SkillEnablement.skill_id == skill.id,
                    SkillEnablement.status == "enabled",
                )
                .order_by(SkillEnablement.scope_type)
            )
        )
        window_start = datetime.now(UTC) - timedelta(days=INVOCATION_WINDOW_DAYS)
        invocations_30d = (
            session.scalar(
                select(func.count(ProceduralSkillInvocation.id)).where(
                    ProceduralSkillInvocation.skill_id == skill.id,
                    ProceduralSkillInvocation.installation_id == installation_id,
                    ProceduralSkillInvocation.created_at >= window_start,
                )
            )
            or 0
        )
    return SkillCatalogEntry(
        skill_id=skill.id,
        slug=skill.slug,
        name=version.name,
        description=version.description,
        version=version.version,
        provenance=skill.provenance,
        trust_level=skill.trust_level,
        owner_type=skill.owner_type,
        has_references="reference" in file_kinds or "asset" in file_kinds,
        has_scripts="script" in file_kinds,
        enabled_scopes=chips,
        invocations_30d=invocations_30d,
    )


def _active_version(
    session: Session,
    skill_id: uuid.UUID,
) -> ProceduralSkillVersion | None:
    return session.scalar(
        select(ProceduralSkillVersion)
        .where(
            ProceduralSkillVersion.skill_id == skill_id,
            ProceduralSkillVersion.status == "active",
        )
        .order_by(ProceduralSkillVersion.created_at.desc())
    )


def _is_skill_md(version: ProceduralSkillVersion) -> bool:
    metadata = version.metadata_json or {}
    return metadata.get("format") == SKILL_MD_FORMAT
