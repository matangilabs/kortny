"""SKILL.md frontmatter + resource models (HIG-281).

In-house replacement for ``google.adk.skills.models`` so SKILL.md ingestion no
longer pulls Google ADK into every worker. The validation contract matches ADK's
so community Claude/Codex/ADK skills still parse verbatim:

* ``name`` — required, lowercase kebab-case (a-z, 0-9, single hyphens), ≤64 chars.
* ``description`` — required, non-empty, ≤1024 chars.
* ``compatibility`` — optional, ≤500 chars.
* ``allowed-tools`` — optional, aliased to ``allowed_tools`` (space-delimited).
* ``metadata`` — defaults to ``{}``; extra frontmatter keys are allowed.

ADK's extra ``metadata.adk_additional_tools`` must-be-a-list check is dropped on
purpose: it's ADK-runtime-specific and no Kortny skill uses it.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_CHARS = 64
_MAX_DESCRIPTION_CHARS = 1024
_MAX_COMPATIBILITY_CHARS = 500


class Frontmatter(BaseModel):
    """SKILL.md YAML frontmatter."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = Field(default=None, alias="allowed-tools")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if len(value) > _MAX_NAME_CHARS:
            raise ValueError(f"skill name must be at most {_MAX_NAME_CHARS} characters")
        if not _NAME_RE.fullmatch(value):
            raise ValueError(
                "skill name must be lowercase kebab-case (a-z, 0-9, single hyphens)"
            )
        return value

    @field_validator("description")
    @classmethod
    def _valid_description(cls, value: str) -> str:
        if not value:
            raise ValueError("skill description must not be empty")
        if len(value) > _MAX_DESCRIPTION_CHARS:
            raise ValueError(
                f"skill description must be at most {_MAX_DESCRIPTION_CHARS} characters"
            )
        return value

    @field_validator("compatibility")
    @classmethod
    def _valid_compatibility(cls, value: str | None) -> str | None:
        if value is not None and len(value) > _MAX_COMPATIBILITY_CHARS:
            raise ValueError(
                f"skill compatibility must be at most {_MAX_COMPATIBILITY_CHARS} chars"
            )
        return value


class Script(BaseModel):
    """A bundled skill script (``scripts/`` entry)."""

    src: str


class Resources(BaseModel):
    """Bundled skill resources, keyed by relative path."""

    references: dict[str, str | bytes] = Field(default_factory=dict)
    assets: dict[str, str | bytes] = Field(default_factory=dict)
    scripts: dict[str, Script] = Field(default_factory=dict)


class Skill(BaseModel):
    """A parsed skill: frontmatter + instruction body + bundled resources."""

    frontmatter: Frontmatter
    instructions: str
    resources: Resources = Field(default_factory=Resources)


__all__ = ["Frontmatter", "Resources", "Script", "Skill"]
