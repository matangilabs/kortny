"""Write actions for the dashboard skills directory."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from kortny.db.models import ProceduralSkill, SkillEnablement
from kortny.skills import SkillRegistryService
from kortny.skills.ingestion import (
    IngestedSkill,
    SkillIngestionError,
    SkillIngestionService,
)

CUSTOM_SKILL_TRUST_DEFAULT = "untrusted"
TRUST_LEVELS = ("trusted", "community", "untrusted", "quarantined")


def enable_skill_for_scope(
    session: Session,
    *,
    installation_id: uuid.UUID,
    skill_id: uuid.UUID,
    scope_type: str,
    scope_id: str | None,
    by_user: str,
) -> SkillEnablement:
    skill = session.get(ProceduralSkill, skill_id)
    if skill is None or skill.status != "active":
        raise ValueError("Skill not found or inactive.")
    return SkillRegistryService(session).enable_skill(
        installation_id=installation_id,
        skill_id=skill_id,
        scope_type=scope_type,
        scope_id=scope_id or None,
        added_by=by_user,
    )


def disable_skill_enablement(
    session: Session,
    *,
    enablement_id: uuid.UUID,
    by_user: str,
) -> SkillEnablement:
    return SkillRegistryService(session).disable_skill(
        enablement_id=enablement_id, by=by_user
    )


def upload_skill(
    session: Session,
    *,
    installation_id: uuid.UUID,
    data: bytes,
    filename: str,
    by_user: str,
) -> IngestedSkill:
    """Ingest an uploaded skill zip, or a single SKILL.md file."""

    ingestion = SkillIngestionService(session)
    kwargs = _custom_skill_kwargs(installation_id, by_user)
    if filename.lower().endswith(".md"):
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillIngestionError("Markdown upload must be UTF-8 encoded.") from exc
        return ingestion.ingest_markdown(content, **kwargs)
    return ingestion.ingest_zip(data, **kwargs)


def paste_skill_markdown(
    session: Session,
    *,
    installation_id: uuid.UUID,
    content: str,
    name: str | None,
    description: str | None,
    by_user: str,
) -> IngestedSkill:
    return SkillIngestionService(session).ingest_markdown(
        content,
        fallback_name=name or None,
        fallback_description=description or None,
        **_custom_skill_kwargs(installation_id, by_user),
    )


def set_skill_trust(
    session: Session,
    *,
    skill_id: uuid.UUID,
    trust_level: str,
    by_user: str,
) -> ProceduralSkill:
    if trust_level not in TRUST_LEVELS:
        raise ValueError(f"Invalid trust level: {trust_level!r}")
    skill = session.get(ProceduralSkill, skill_id)
    if skill is None:
        raise ValueError("Skill not found.")
    if skill.owner_type == "system":
        raise ValueError("Curated skills keep their managed trust level.")
    skill.trust_level = trust_level
    session.flush()
    del by_user  # audited via the dashboard notice/redirect trail for now
    return skill


def _custom_skill_kwargs(installation_id: uuid.UUID, by_user: str) -> dict[str, str]:
    return {
        "owner_type": "workspace",
        "owner_id": str(installation_id),
        "provenance": f"user:{by_user}",
        "trust_level": CUSTOM_SKILL_TRUST_DEFAULT,
        "created_by": f"dashboard:{by_user}",
    }
