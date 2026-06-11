"""Canonical embedding texts for memory rows (HIG-225).

One place defines what text represents a fact / episode / graph entity in the
semantic index so embed-on-write hooks and the consolidator backfill stay
sha-consistent with each other.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kortny.db.models import Episode, KnowledgeGraphEntity, WorkspaceState

FACT_EMBEDDING_KIND = "fact"
EPISODE_EMBEDDING_KIND = "episode"
KG_ENTITY_EMBEDDING_KIND = "kg_entity"
_MAX_EMBEDDING_TEXT_CHARS = 2_000


def fact_embedding_text(state: WorkspaceState) -> str:
    """Render one workspace_state fact for embedding."""

    value = state.value_text or _compact_json(state.value_json)
    scope = (
        state.scope_type
        if state.scope_id is None
        else (f"{state.scope_type}:{state.scope_id}")
    )
    return _bounded(f"{state.key} ({scope}): {value}")


def episode_embedding_text(episode: Episode) -> str:
    """Render one task episode for embedding."""

    tools = ", ".join(item for item in episode.tools_used if isinstance(item, str))
    text = episode.summary or ""
    if tools:
        text = f"{text} [tools: {tools}]"
    return _bounded(text)


def kg_entity_embedding_text(entity: KnowledgeGraphEntity) -> str:
    """Render one knowledge graph entity for embedding."""

    parts = [f"{entity.entity_type}: {entity.display_name or entity.canonical_key}"]
    attrs = entity.attrs_json if isinstance(entity.attrs_json, dict) else {}
    for key in ("summary", "description", "value_text", "value"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break
    return _bounded(". ".join(parts))


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _bounded(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= _MAX_EMBEDDING_TEXT_CHARS:
        return normalized
    return normalized[:_MAX_EMBEDDING_TEXT_CHARS]
