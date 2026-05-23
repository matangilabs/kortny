import uuid
from typing import Any

import httpx
import pytest

from kortny.db.models import TaskEventType
from kortny.tools import ToolResult, WebSearchTool


class RecordingTaskService:
    def __init__(self) -> None:
        self.events: list[tuple[uuid.UUID, TaskEventType | str, dict[str, Any]]] = []

    def append_event(
        self,
        task: uuid.UUID,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> object:
        self.events.append((task, event_type, payload or {}))
        return object()


def test_web_search_tool_calls_brave_and_returns_structured_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(
            "https://api.search.brave.com/res/v1/web/search?"
        )
        assert request.headers["X-Subscription-Token"] == "brave-key"
        assert request.url.params["q"] == "python tempfile"
        assert request.url.params["count"] == "2"
        assert request.url.params["result_filter"] == "web"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "tempfile",
                            "url": "https://docs.python.org/3/library/tempfile.html",
                            "description": "Temporary file and directory helpers.",
                        },
                        {
                            "title": "pathlib",
                            "url": "https://docs.python.org/3/library/pathlib.html",
                            "description": "Object-oriented filesystem paths.",
                        },
                    ]
                }
            },
        )

    tool = WebSearchTool(
        api_key="brave-key",
        transport=httpx.MockTransport(handler),
    )

    result = tool.invoke({"query": "python tempfile", "count": 2})

    assert result == ToolResult(
        output={
            "provider": "brave",
            "query": "python tempfile",
            "results": [
                {
                    "title": "tempfile",
                    "url": "https://docs.python.org/3/library/tempfile.html",
                    "snippet": "Temporary file and directory helpers.",
                },
                {
                    "title": "pathlib",
                    "url": "https://docs.python.org/3/library/pathlib.html",
                    "snippet": "Object-oriented filesystem paths.",
                },
            ],
        }
    )


def test_web_search_tool_emits_task_events() -> None:
    task_id = uuid.uuid4()
    task_service = RecordingTaskService()

    tool = WebSearchTool(
        api_key="brave-key",
        task_service=task_service,
        task_id=task_id,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Kortny",
                                "url": "https://example.com/kortny",
                                "description": "A result.",
                            }
                        ]
                    }
                },
            )
        ),
    )

    tool.invoke({"query": "kortny"})

    assert task_service.events == [
        (
            task_id,
            TaskEventType.tool_call,
            {"tool": "web_search", "query": "kortny", "count": 5},
        ),
        (
            task_id,
            TaskEventType.tool_result,
            {
                "tool": "web_search",
                "query": "kortny",
                "result_count": 1,
                "results": [
                    {
                        "title": "Kortny",
                        "url": "https://example.com/kortny",
                        "snippet": "A result.",
                    }
                ],
            },
        ),
    ]


def test_web_search_tool_validates_query() -> None:
    tool = WebSearchTool(
        api_key="brave-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(ValueError, match="non-empty"):
        tool.invoke({"query": "   "})


def test_web_search_tool_validates_count() -> None:
    tool = WebSearchTool(
        api_key="brave-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(ValueError, match="between 1 and 20"):
        tool.invoke({"query": "kortny", "count": 21})


def test_web_search_tool_requires_task_context_pair() -> None:
    with pytest.raises(ValueError, match="provided together"):
        WebSearchTool(api_key="brave-key", task_id=uuid.uuid4())


def test_web_search_tool_raises_for_http_errors() -> None:
    tool = WebSearchTool(
        api_key="brave-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": "unauthorized"})
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        tool.invoke({"query": "kortny"})
