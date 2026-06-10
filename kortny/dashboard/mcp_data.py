"""Read queries for the dashboard MCP servers admin page."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import McpServer, McpServerTool

_QUALITY_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class McpToolRow:
    tool_id: uuid.UUID
    name: str
    description: str
    read_only_hint: bool | None
    destructive_hint: bool | None
    enabled: bool
    # Description quality fields (HIG-215)
    description_quality_score: Decimal | None
    enriched_description: str | None

    @property
    def quality_badge(self) -> str:
        """CSS badge variant for the description quality indicator.

        Returns one of: "success" (ok, score >= 0.5, no enrichment),
        "accent" (enriched_description present),
        "warning" (score < 0.5 and no enrichment).
        """
        if self.enriched_description:
            return "accent"
        score = self.description_quality_score
        if score is None or float(score) < _QUALITY_THRESHOLD:
            return "warning"
        return "success"

    @property
    def quality_label(self) -> str:
        """Short human-readable label for the quality badge."""
        if self.enriched_description:
            return "enriched"
        score = self.description_quality_score
        if score is None or float(score) < _QUALITY_THRESHOLD:
            return "poor"
        return "ok"


@dataclass(frozen=True, slots=True)
class McpServerRow:
    server_id: uuid.UUID
    name: str
    transport: str
    status: str
    tool_count: int
    enabled_tool_count: int
    last_discovery_at: datetime | None
    last_discovery_error: str | None
    created_by: str
    created_at: datetime
    command: str | None
    args: list[object]
    url: str | None
    tools: tuple[McpToolRow, ...]


@dataclass(frozen=True, slots=True)
class McpDashboard:
    servers: tuple[McpServerRow, ...]

    @property
    def enabled_count(self) -> int:
        return sum(1 for s in self.servers if s.status == "enabled")


def get_mcp_dashboard(
    session: Session,
    installation_id: uuid.UUID | None,
) -> McpDashboard:
    """Return all registered MCP servers for this installation."""
    if installation_id is None:
        return McpDashboard(servers=())

    servers = session.scalars(
        select(McpServer)
        .where(McpServer.installation_id == installation_id)
        .order_by(McpServer.created_at.desc())
    ).all()

    rows: list[McpServerRow] = []
    for server in servers:
        tools = tuple(
            McpToolRow(
                tool_id=t.id,
                name=t.name,
                description=t.description,
                read_only_hint=t.read_only_hint,
                destructive_hint=t.destructive_hint,
                enabled=t.enabled,
                description_quality_score=t.description_quality_score,
                enriched_description=t.enriched_description,
            )
            for t in session.scalars(
                select(McpServerTool)
                .where(McpServerTool.server_id == server.id)
                .order_by(McpServerTool.name)
            )
        )
        total = len(tools)
        enabled = sum(1 for t in tools if t.enabled)
        rows.append(
            McpServerRow(
                server_id=server.id,
                name=server.name,
                transport=server.transport,
                status=server.status,
                tool_count=total,
                enabled_tool_count=enabled,
                last_discovery_at=server.last_discovery_at,
                last_discovery_error=server.last_discovery_error,
                created_by=server.created_by,
                created_at=server.created_at,
                command=server.command,
                args=list(server.args) if server.args else [],
                url=server.url,
                tools=tools,
            )
        )

    return McpDashboard(servers=tuple(rows))


def get_mcp_server_row(
    session: Session,
    installation_id: uuid.UUID,
    server_id: uuid.UUID,
) -> McpServerRow | None:
    """Return a single server row scoped to the installation."""
    server = session.get(McpServer, server_id)
    if server is None or server.installation_id != installation_id:
        return None
    tools = tuple(
        McpToolRow(
            tool_id=t.id,
            name=t.name,
            description=t.description,
            read_only_hint=t.read_only_hint,
            destructive_hint=t.destructive_hint,
            enabled=t.enabled,
            description_quality_score=t.description_quality_score,
            enriched_description=t.enriched_description,
        )
        for t in session.scalars(
            select(McpServerTool)
            .where(McpServerTool.server_id == server.id)
            .order_by(McpServerTool.name)
        )
    )
    total = len(tools)
    enabled = sum(1 for t in tools if t.enabled)
    return McpServerRow(
        server_id=server.id,
        name=server.name,
        transport=server.transport,
        status=server.status,
        tool_count=total,
        enabled_tool_count=enabled,
        last_discovery_at=server.last_discovery_at,
        last_discovery_error=server.last_discovery_error,
        created_by=server.created_by,
        created_at=server.created_at,
        command=server.command,
        args=list(server.args) if server.args else [],
        url=server.url,
        tools=tools,
    )
