"""Workspace knowledge graph query tools."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    SlackChannelMembership,
    SlackIdentity,
    Task,
)
from kortny.knowledge_graph.projects import (
    ProjectGraphService,
    project_anchors_and_scopes,
)
from kortny.knowledge_graph.scopes import DestinationSurface, VisibilityScope
from kortny.knowledge_graph.service import (
    EvidenceInput,
    GraphService,
    RetrievedGraphEdge,
    RetrievedGraphEntity,
)
from kortny.tools.types import JsonObject, JsonSchema, RecoverableToolError, ToolResult

# BFS depth for project-anchored retrieval (hub -> channel hub -> facts).
_PROJECT_GRAPH_MAX_HOPS = 2


class QueryWorkspaceGraphTool:
    """Query scope-safe workspace graph context for the current Slack task."""

    name = "query_workspace_graph"
    description = (
        "Queries Kortny's workspace knowledge graph for the current Slack task. "
        "Use this when the user asks what Kortny knows about a channel, person, "
        "project, workflow, recurring topic, relationship, or why Kortny believes "
        "something. The tool only returns current active/confirmed graph rows "
        "with evidence that are visible to this Slack surface."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Optional search text for graph keys, labels, source types, "
                    "relationship types, and attributes."
                ),
            },
            "anchor_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional canonical graph keys to traverse from, such as "
                    "slack_channel:C123 or project:kortny."
                ),
            },
            "max_hops": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "Traversal depth when anchor_keys are provided.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum entities and relationships to return.",
            },
            "include_evidence": {
                "type": "boolean",
                "description": "When true, include short evidence snippets.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, *, session: Session, task: Task) -> None:
        self.session = session
        self.task = task
        self.graph = GraphService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        query = _optional_string(args.get("query"))
        anchor_keys = _optional_string_list(args.get("anchor_keys"))
        if not query and not anchor_keys:
            inferred_anchor = _default_anchor_key(self.task)
            if inferred_anchor is not None:
                anchor_keys = (inferred_anchor,)
        max_hops = _bounded_int(
            args.get("max_hops", 1), default=1, minimum=0, maximum=3
        )
        limit = _bounded_int(args.get("limit", 20), default=20, minimum=1, maximum=50)
        include_evidence = _optional_bool(args.get("include_evidence", True))

        destination = _destination_for_task(self.session, self.task)
        # Project layer (HIG-276): if the current channel belongs to a project,
        # add the project hub anchor(s) + the project's PUBLIC member-channel
        # scopes so the tool synthesizes across the project's public channels,
        # matching the context-assembly path. Audience-safe (no private siblings).
        project_anchor_keys, additional_scopes = _project_anchors_and_scopes(
            self.session, self.task
        )
        if project_anchor_keys:
            anchor_keys = tuple(dict.fromkeys((*anchor_keys, *project_anchor_keys)))
            max_hops = max(max_hops, _PROJECT_GRAPH_MAX_HOPS)
        pack = self.graph.query_current_context(
            installation_id=self.task.installation_id,
            destination=destination,
            query=query,
            anchor_keys=anchor_keys,
            max_hops=max_hops,
            max_items=limit,
            additional_scopes=additional_scopes,
        )
        violations = self.graph.scope_guard_violations(
            pack, destination, additional_scopes
        )
        if violations:
            return ToolResult(
                output={
                    "successful": False,
                    "error": {
                        "code": "scope_guard_violation",
                        "message": (
                            "Graph retrieval returned rows outside this Slack "
                            "surface. The result was withheld."
                        ),
                        "recoverable": False,
                    },
                }
            )

        entity_labels = _entity_labels(
            self.session,
            [edge.source_entity_id for edge in pack.edges]
            + [edge.target_entity_id for edge in pack.edges],
        )
        evidence = (
            _evidence_by_id(
                self.session,
                [
                    evidence_id
                    for entity in pack.entities
                    for evidence_id in entity.evidence_ids
                ]
                + [
                    evidence_id
                    for edge in pack.edges
                    for evidence_id in edge.evidence_ids
                ],
            )
            if include_evidence
            else {}
        )
        return ToolResult(
            output={
                "successful": True,
                "destination": {
                    "surface_type": destination.surface_type,
                    "surface_id": destination.surface_id,
                    "user_id": destination.user_id,
                },
                "query": query,
                "anchor_keys": list(anchor_keys),
                "max_hops": max_hops,
                "limit": limit,
                "entity_count": len(pack.entities),
                "edge_count": len(pack.edges),
                "omitted_count": pack.omitted_count,
                "omitted_reasons": list(pack.omitted_reasons),
                "scope_note": (
                    "Candidate, stale, unbacked, and out-of-scope graph rows are "
                    "excluded from this runtime result."
                ),
                "entities": [
                    _entity_output(entity, evidence.get(entity.id, ()))
                    for entity in pack.entities
                ],
                "relationships": [
                    _edge_output(
                        edge,
                        source_label=entity_labels.get(edge.source_entity_id),
                        target_label=entity_labels.get(edge.target_entity_id),
                        evidence=evidence.get(edge.id, ()),
                    )
                    for edge in pack.edges
                ],
            }
        )


def _project_anchors_and_scopes(
    session: Session, task: Task
) -> tuple[tuple[str, ...], tuple[VisibilityScope, ...]]:
    channel_id = task.slack_channel_id
    if not channel_id or channel_id.startswith("D"):
        return ((), ())
    try:
        return project_anchors_and_scopes(
            session, installation_id=task.installation_id, channel_id=channel_id
        )
    except Exception:
        return ((), ())


class DeclareProjectTool:
    """Declare a project boundary: name + the channels that make it up (HIG-276).

    Lets the agent record "Project Apollo includes #apollo-eng and #apollo-design"
    so Kortny can later synthesize across those channels. Creates/updates a
    workspace-visible project hub and ``project_includes_channel`` edges; public
    channels become cross-channel-visible, private channels stay gated. Pass
    Slack channel IDs (resolve <#C123|name> mentions to the C123 id).
    """

    name = "declare_project"
    description = (
        "Declare or update a project and the Slack channels that belong to it, so "
        'Kortny can answer project-wide questions ("how is Apollo going?") by '
        "synthesizing across those channels. Use when a user says some channels "
        "are part of a named project. Pass the project name and the channel IDs "
        "(resolve channel mentions to their C... id). Re-declaring is safe and "
        "merges new channels in."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Human project name, e.g. 'Apollo'.",
            },
            "channel_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Slack channel IDs that belong to the project.",
            },
        },
        "required": ["name", "channel_ids"],
        "additionalProperties": False,
    }

    def __init__(self, *, session: Session, task: Task) -> None:
        self.session = session
        self.task = task
        self.projects = ProjectGraphService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RecoverableToolError(
                code="invalid_arguments",
                message="declare_project requires a non-empty 'name'.",
                hint="Pass the project's human name, e.g. 'Apollo'.",
            )
        channel_ids = _optional_string_list(args.get("channel_ids"))
        if not channel_ids:
            raise RecoverableToolError(
                code="invalid_arguments",
                message="declare_project requires at least one channel id.",
                hint="Pass the Slack channel IDs (C...) that belong to the project.",
            )
        result = self.projects.declare_project(
            installation_id=self.task.installation_id,
            name=name,
            channel_ids=channel_ids,
            evidence=EvidenceInput(
                source_type="user_explicit",
                extracted_by="declare_project_tool",
                source_task_id=self.task.id,
                source_slack_channel_id=self.task.slack_channel_id,
            ),
        )
        return ToolResult(
            output={
                "successful": True,
                "project": result.project.display_name or result.project.canonical_key,
                "canonical_key": result.project.canonical_key,
                "linked_channel_ids": list(result.linked_channel_ids),
                "skipped_channel_ids": list(result.skipped_channel_ids),
                "message": (
                    f"Project '{result.project.display_name}' now spans "
                    f"{len(result.linked_channel_ids)} channel(s). "
                    + (
                        f"{len(result.skipped_channel_ids)} channel(s) were skipped "
                        "because I don't have them in the workspace graph yet "
                        "(I may need to be a member first)."
                        if result.skipped_channel_ids
                        else "Ask me project-wide questions and I'll synthesize "
                        "across them."
                    )
                ),
            }
        )


def _destination_for_task(session: Session, task: Task) -> DestinationSurface:
    channel_id = task.slack_channel_id
    if channel_id.startswith("D"):
        return DestinationSurface.dm(channel_id, user_id=task.slack_user_id)

    if _is_private_channel(session, task):
        return DestinationSurface.private_channel(channel_id)
    return DestinationSurface.channel(channel_id)


def _is_private_channel(session: Session, task: Task) -> bool:
    channel_id = task.slack_channel_id
    if channel_id.startswith("G"):
        return True

    membership = session.scalar(
        select(SlackChannelMembership).where(
            SlackChannelMembership.installation_id == task.installation_id,
            SlackChannelMembership.channel_id == channel_id,
        )
    )
    channel_type = (membership.channel_type or "").lower() if membership else ""
    if channel_type in {"group", "private_channel", "private"}:
        return True

    identity = session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == task.installation_id,
            SlackIdentity.kind == "channel",
            SlackIdentity.slack_id == channel_id,
        )
    )
    return bool(identity and identity.is_private)


def _default_anchor_key(task: Task) -> str | None:
    if task.slack_channel_id.startswith(("C", "G")):
        return f"slack_channel:{task.slack_channel_id}"
    return None


def _entity_output(
    entity: RetrievedGraphEntity,
    evidence: Sequence[JsonObject],
) -> JsonObject:
    return {
        "id": str(entity.id),
        "entity_type": entity.entity_type,
        "canonical_key": entity.canonical_key,
        "display_name": entity.display_name,
        "source_type": entity.source_type,
        "visibility_scope": {
            "type": entity.visibility_scope.scope_type,
            "id": entity.visibility_scope.scope_id,
        },
        "lifecycle_state": entity.lifecycle_state,
        "confidence_score": str(entity.confidence_score),
        "confidence_reason": entity.confidence_reason,
        "provenance": {
            "extraction_kind": entity.provenance_kind,
            "label": entity.provenance_label,
            "review_status": entity.review_status,
        },
        "evidence_count": len(entity.evidence_ids),
        "evidence": list(evidence),
    }


def _edge_output(
    edge: RetrievedGraphEdge,
    *,
    source_label: str | None,
    target_label: str | None,
    evidence: Sequence[JsonObject],
) -> JsonObject:
    return {
        "id": str(edge.id),
        "source_entity_id": str(edge.source_entity_id),
        "source_label": source_label,
        "target_entity_id": str(edge.target_entity_id),
        "target_label": target_label,
        "relationship_type": edge.relationship_type,
        "source_type": edge.source_type,
        "visibility_scope": {
            "type": edge.visibility_scope.scope_type,
            "id": edge.visibility_scope.scope_id,
        },
        "lifecycle_state": edge.lifecycle_state,
        "confidence_score": str(edge.confidence_score),
        "confidence_reason": edge.confidence_reason,
        "provenance": {
            "extraction_kind": edge.provenance_kind,
            "label": edge.provenance_label,
            "review_status": edge.review_status,
        },
        "evidence_count": len(edge.evidence_ids),
        "evidence": list(evidence),
    }


def _entity_labels(
    session: Session,
    entity_ids: Iterable[uuid.UUID],
) -> dict[uuid.UUID, str]:
    ids = tuple({entity_id for entity_id in entity_ids})
    if not ids:
        return {}
    rows = session.scalars(
        select(KnowledgeGraphEntity).where(KnowledgeGraphEntity.id.in_(ids))
    )
    return {row.id: row.display_name or row.canonical_key for row in rows}


def _evidence_by_id(
    session: Session,
    evidence_ids: Iterable[uuid.UUID],
    *,
    limit_per_target: int = 2,
) -> dict[uuid.UUID, tuple[JsonObject, ...]]:
    ids = tuple({evidence_id for evidence_id in evidence_ids})
    if not ids:
        return {}
    grouped: dict[uuid.UUID, list[JsonObject]] = defaultdict(list)
    rows = session.scalars(
        select(KnowledgeGraphEvidence)
        .where(KnowledgeGraphEvidence.id.in_(ids))
        .order_by(
            KnowledgeGraphEvidence.created_at.desc(),
            KnowledgeGraphEvidence.id.desc(),
        )
    )
    for row in rows:
        bucket = grouped[row.target_id]
        if len(bucket) >= limit_per_target:
            continue
        bucket.append(
            {
                "id": str(row.id),
                "source_type": row.source_type,
                "extracted_by": row.extracted_by,
                "source_slack_channel_id": row.source_slack_channel_id,
                "source_slack_message_ts": row.source_slack_message_ts,
                "source_url": row.source_url,
                "confidence_score": _decimal_str(row.confidence_score),
                "confidence_reason": row.confidence_reason,
                "snippet": row.raw_snippet,
            }
        )
    return {target_id: tuple(items) for target_id, items in grouped.items()}


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("query must be a string")
    cleaned = " ".join(value.split())
    return cleaned or None


def _optional_string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("anchor_keys must be an array of strings")
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("anchor_keys must contain only strings")
        cleaned = item.strip()
        if cleaned:
            output.append(cleaned)
    return tuple(output[:10])


def _bounded_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("numeric graph query options must be integers")
    return min(max(value, minimum), maximum)


def _optional_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError("include_evidence must be a boolean")


def _decimal_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)
