"""Brave-backed web search tool."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, Protocol

import httpx

from kortny.config import Settings, load_settings
from kortny.db.models import TaskEventType
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 20


class TaskEventSink(Protocol):
    """Subset of TaskService needed for tool event emission."""

    def append_event(
        self,
        task: uuid.UUID,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> object:
        """Append an event for a task."""


class WebSearchTool:
    """Search the public web with Brave Search."""

    name = "web_search"
    description = "Searches the public web and returns structured search results."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            },
            "count": {
                "type": "integer",
                "description": "Number of web results to return.",
                "minimum": 1,
                "maximum": MAX_RESULT_COUNT,
                "default": DEFAULT_RESULT_COUNT,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        api_key: str,
        *,
        task_service: TaskEventSink | None = None,
        task_id: uuid.UUID | None = None,
        endpoint: str = BRAVE_WEB_SEARCH_ENDPOINT,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("BRAVE_SEARCH_API_KEY is required for web_search")
        if (task_service is None) != (task_id is None):
            raise ValueError("task_service and task_id must be provided together")

        self.api_key = api_key
        self.task_service = task_service
        self.task_id = task_id
        self.endpoint = endpoint
        self.timeout = timeout
        self.transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        **kwargs: Any,
    ) -> WebSearchTool:
        """Create the tool from application settings."""

        resolved_settings = settings or load_settings()
        api_key = resolved_settings.brave_search_api_key
        if api_key is None:
            raise ValueError("BRAVE_SEARCH_API_KEY is required for web_search")
        return cls(api_key=api_key, **kwargs)

    def invoke(self, args: JsonObject) -> ToolResult:
        query = _require_query(args)
        count = _require_count(args)
        request_payload = {"query": query, "count": count}

        self._append_event(TaskEventType.tool_call, request_payload)
        response_payload = self._search(query=query, count=count)
        results = _parse_results(response_payload)
        output = {
            "provider": "brave",
            "query": query,
            "results": results,
        }
        self._append_event(
            TaskEventType.tool_result,
            {
                "query": query,
                "result_count": len(results),
                "results": results,
            },
        )

        return ToolResult(output=output)

    def _search(self, *, query: str, count: int) -> JsonObject:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params: dict[str, str | int] = {
            "q": query,
            "count": count,
            "result_filter": "web",
        }

        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            response = client.get(self.endpoint, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError("Brave Search response must be a JSON object")
        return payload

    def _append_event(self, event_type: TaskEventType, payload: JsonObject) -> None:
        if self.task_service is None or self.task_id is None:
            return

        self.task_service.append_event(
            self.task_id,
            event_type,
            {"tool": self.name, **payload},
        )


def _require_query(args: Mapping[str, Any]) -> str:
    query = args.get("query")
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("web_search requires a non-empty string 'query' argument")
    return query.strip()


def _require_count(args: Mapping[str, Any]) -> int:
    count = args.get("count", DEFAULT_RESULT_COUNT)
    if not isinstance(count, int):
        raise ValueError("web_search 'count' must be an integer")
    if count < 1 or count > MAX_RESULT_COUNT:
        raise ValueError(f"web_search 'count' must be between 1 and {MAX_RESULT_COUNT}")
    return count


def _parse_results(payload: JsonObject) -> list[JsonObject]:
    web = payload.get("web", {})
    if not isinstance(web, dict):
        return []

    raw_results = web.get("results", [])
    if not isinstance(raw_results, list):
        return []

    results: list[JsonObject] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue

        title = _optional_string(raw_result.get("title"))
        url = _optional_string(raw_result.get("url"))
        snippet = _optional_string(raw_result.get("description"))
        if title is None or url is None:
            continue

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet or "",
            }
        )

    return results


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
