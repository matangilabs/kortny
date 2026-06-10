"""Kortny tool adapter for schema-aware Composio runtime execution."""

from __future__ import annotations

import hashlib
import json
import logging
import re

from sqlalchemy.orm import Session

from kortny.composio import ComposioClient, ComposioConnectionResolver, ComposioTool
from kortny.db.models import Task
from kortny.observability.events import log_observation
from kortny.tools.result_budget import bound_tool_result
from kortny.tools.types import JsonObject, JsonSchema, RecoverableToolError, ToolResult

logger = logging.getLogger(__name__)
MAX_TOOL_NAME_LENGTH = 64
DEFAULT_RESULT_MAX_CHARS = 16000


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
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ) -> None:
        self.session = session
        self.task = task
        self.client = client
        self.resolver = ComposioConnectionResolver(session, task)
        self.tool = tool
        self.result_max_chars = result_max_chars
        self.name = name or composio_runtime_tool_name(tool.toolkit_slug, tool.slug)
        self.description = _description(tool)
        self.parameters = _parameters(tool)

    @property
    def has_available_connection(self) -> bool:
        return (
            self.resolver.best_connection(toolkit_slug=self.tool.toolkit_slug)
            is not None
        )

    def invoke(self, args: JsonObject) -> ToolResult:
        arguments = dict(args)
        _validate_required_arguments(
            arguments,
            parameters=self.parameters,
            tool_name=self.name,
        )
        arguments = _strip_blank_optional_arguments(
            arguments,
            parameters=self.parameters,
        )

        connection = self.resolver.best_connection(toolkit_slug=self.tool.toolkit_slug)
        if connection is None:
            raise RecoverableToolError(
                code="missing_connection",
                message=(
                    f"No active Composio {self.tool.toolkit_slug} connection is "
                    "available for this Slack user/channel/workspace."
                ),
                hint=(
                    "Use another available tool if the task can be completed without "
                    f"{self.tool.toolkit_slug}. Otherwise ask the user to connect "
                    "the integration in Kortny's Integrations dashboard."
                ),
                details={
                    "provider": "composio",
                    "toolkit_slug": self.tool.toolkit_slug,
                    "tool_slug": self.tool.slug,
                },
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
        output: JsonObject = {
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
        output = self._bound_output(output)
        return ToolResult(output=output)

    def _bound_output(self, output: JsonObject) -> JsonObject:
        bounded = bound_tool_result(
            output,
            max_chars=self.result_max_chars,
            hint=(
                f"Composio tool {self.tool.slug} ({self.tool.toolkit_slug}) "
                "returned a large payload; result truncated."
            ),
        )
        if bounded is not output:
            logger.info(
                "composio tool result truncated runtime_tool=%s original_chars=%s "
                "final_chars=%s max_chars=%s",
                self.name,
                bounded.get("original_chars"),
                len(json.dumps(bounded, default=str)),
                self.result_max_chars,
            )
        return bounded


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
        "otherwise use a broader discovery tool or ask a clarification. "
        "Omit unknown optional arguments; never pass blank strings for cursors or IDs."
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
        raise RecoverableToolError(
            code="missing_required_arguments",
            message=f"{tool_name} is missing required argument(s): {joined}.",
            hint=(
                "Use a discovery/list/search tool first if one can find the missing "
                "identifier. If no available tool can infer it, ask the user for the "
                "smallest missing piece of context."
            ),
            details={"missing_fields": missing},
        )


def _required_fields(parameters: JsonSchema) -> tuple[str, ...]:
    required = parameters.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required if isinstance(item, str) and item)


def _strip_blank_optional_arguments(
    args: JsonObject,
    *,
    parameters: JsonSchema,
) -> JsonObject:
    required_fields = set(_required_fields(parameters))
    return {
        key: value
        for key, value in args.items()
        if key in required_fields or not (isinstance(value, str) and not value.strip())
    }


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return safe or "tool"


def _fit_tool_name(name: str) -> str:
    if len(name) <= MAX_TOOL_NAME_LENGTH:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: MAX_TOOL_NAME_LENGTH - 9].rstrip('_')}_{digest}"
