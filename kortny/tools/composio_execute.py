"""Kortny tool adapter for schema-aware Composio runtime execution."""

from __future__ import annotations

import hashlib
import logging
import re

from sqlalchemy.orm import Session

from kortny.composio import ComposioClient, ComposioConnectionResolver, ComposioTool
from kortny.db.models import Task
from kortny.observability.events import log_observation
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

logger = logging.getLogger(__name__)
MAX_TOOL_NAME_LENGTH = 64


class ComposioExecuteTool:
    """Execute one approved Composio tool through a scoped connected account."""

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        client: ComposioClient,
        tool: ComposioTool,
        name: str | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.client = client
        self.resolver = ComposioConnectionResolver(session, task)
        self.tool = tool
        self.name = name or composio_runtime_tool_name(tool.toolkit_slug, tool.slug)
        self.description = _description(tool)
        self.parameters = _parameters(tool)

    @property
    def has_available_connection(self) -> bool:
        return (
            self.resolver.best_connection(toolkit_slug=self.tool.toolkit_slug) is not None
        )

    def invoke(self, args: JsonObject) -> ToolResult:
        arguments = dict(args)
        _validate_required_arguments(
            arguments,
            parameters=self.parameters,
            tool_name=self.name,
        )

        connection = self.resolver.best_connection(toolkit_slug=self.tool.toolkit_slug)
        if connection is None:
            raise ValueError(
                f"No active Composio {self.tool.toolkit_slug} connection is available "
                "for this Slack user/channel/workspace."
            )

        log_observation(
            logger,
            "composio_tool_execution_started",
            task=self.task,
            provider="composio",
            runtime_tool=self.name,
            toolkit_slug=self.tool.toolkit_slug,
            tool_slug=self.tool.slug,
            visibility_scope_type=connection.visibility_scope_type,
            argument_keys=sorted(arguments),
        )
        execution = self.client.execute_tool(
            tool_slug=self.tool.slug,
            user_id=connection.composio_user_id,
            connected_account_id=connection.connected_account_id,
            arguments=arguments,
            version=None,
        )
        log_observation(
            logger,
            "composio_tool_execution_completed",
            task=self.task,
            provider="composio",
            runtime_tool=self.name,
            toolkit_slug=self.tool.toolkit_slug,
            tool_slug=self.tool.slug,
            visibility_scope_type=connection.visibility_scope_type,
            successful=execution.successful,
            log_id=execution.log_id,
        )
        return ToolResult(
            output={
                "provider": "composio",
                "toolkit_slug": self.tool.toolkit_slug,
                "tool_slug": self.tool.slug,
                "successful": execution.successful,
                "data": execution.data,
                "error": execution.error,
                "log_id": execution.log_id,
                "scope": {
                    "type": connection.visibility_scope_type,
                    "id": connection.visibility_scope_id,
                },
                "connection": {
                    "display_name": connection.display_name,
                    "connected_account_id": connection.connected_account_id,
                },
            }
        )


def composio_runtime_tool_name(toolkit_slug: str, tool_slug: str | None = None) -> str:
    toolkit = _safe_identifier(toolkit_slug)
    if tool_slug is None:
        return _fit_tool_name(f"composio_{toolkit}_execute")

    raw_tool = tool_slug.casefold()
    prefix = f"{toolkit}_"
    if raw_tool.startswith(prefix):
        raw_tool = raw_tool[len(prefix) :]
    tool = _safe_identifier(raw_tool)
    return _fit_tool_name(f"composio_{toolkit}_{tool}")


def _description(tool: ComposioTool) -> str:
    required = ", ".join(_required_fields(tool.input_parameters)) or "none"
    return (
        f"Composio {tool.toolkit_slug} tool {tool.slug}. "
        f"{tool.description or tool.name} Required fields: {required}. "
        "Use only when the task has enough context to satisfy the required fields; "
        "otherwise use a broader discovery tool or ask a clarification."
    )


def _parameters(tool: ComposioTool) -> JsonSchema:
    schema = dict(tool.input_parameters or {})
    if schema.get("type") != "object":
        schema["type"] = "object"
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    return schema


def _validate_required_arguments(
    args: JsonObject,
    *,
    parameters: JsonSchema,
    tool_name: str,
) -> None:
    missing = [
        field
        for field in _required_fields(parameters)
        if field not in args or args[field] is None or args[field] == ""
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"{tool_name} is missing required argument(s): {joined}. "
            "Use a discovery tool first or ask the user for the missing context."
        )


def _required_fields(parameters: JsonSchema) -> tuple[str, ...]:
    required = parameters.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required if isinstance(item, str) and item)


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return safe or "tool"


def _fit_tool_name(name: str) -> str:
    if len(name) <= MAX_TOOL_NAME_LENGTH:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: MAX_TOOL_NAME_LENGTH - 9].rstrip('_')}_{digest}"
