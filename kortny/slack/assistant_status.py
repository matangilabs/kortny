"""Live assistant status narration (HIG-247 follow-up).

Drives Slack's ``assistant.threads.setStatus`` from the worker so the loading
indicator in the agent pane reflects what Kortny is *actually* doing — the tool
it is calling, the phase it is in — instead of a static cycling loop. The app
process sets the initial cycling ``loading_messages`` when the message arrives;
once the worker claims the task it takes over with real, activity-specific
status lines. ``setStatus`` is stateless and accepts the bot token from any
process, so the worker can drive it directly.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class AssistantStatusClient(Protocol):
    """Subset of the Slack WebClient used to push assistant status."""

    def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
        loading_messages: list[str] | None = None,
    ) -> Any:
        """Set the assistant loading status for a thread."""


class StatusReporter(Protocol):
    """Reports a human-readable activity status for the current task."""

    def report(self, status: str) -> None:
        """Surface ``status`` to the user (best-effort, never raises)."""


class NullStatusReporter:
    """No-op reporter for non-assistant tasks (the default everywhere)."""

    def report(self, status: str) -> None:  # noqa: D102 - protocol impl
        return None


class AssistantStatusReporter:
    """Pushes live status into a Slack assistant thread.

    Throttled to *changes*: repeating the current status is a no-op, so a tool
    that runs several times in a row doesn't re-issue identical ``setStatus``
    calls. Never raises — a status update must not fail the task.
    """

    def __init__(
        self,
        *,
        client: AssistantStatusClient,
        channel_id: str,
        thread_ts: str,
    ) -> None:
        self._client = client
        self._channel_id = channel_id
        self._thread_ts = thread_ts
        self._last: str | None = None

    def report(self, status: str) -> None:
        cleaned = status.strip()
        if not cleaned or cleaned == self._last:
            return
        setter = getattr(self._client, "assistant_threads_setStatus", None)
        if not callable(setter):
            return
        try:
            # Drive both the composer status line and the below-app-name line to
            # the current step. loading_messages must be a NON-EMPTY list — Slack
            # rejects [] with invalid_arguments ("must provide at least 1 items"),
            # and there is no way to clear it back to nothing — so we send the
            # single current step (one item = no rotation, replaces the app's
            # static intro loop). The native step-timeline (TaskUpdateChunk) that
            # makes the two lines distinct is HIG-252.
            setter(
                channel_id=self._channel_id,
                thread_ts=self._thread_ts,
                status=cleaned,
                loading_messages=[cleaned],
            )
            self._last = cleaned
        except Exception:
            logger.warning(
                "failed to set assistant status channel=%s thread_ts=%s status=%s",
                self._channel_id,
                self._thread_ts,
                cleaned,
                exc_info=True,
            )


# Phase statuses the coordinator reports outside of tool calls.
STATUS_GETTING_STARTED = "Getting up to speed…"
STATUS_WRITING = "Writing the response…"

_GENERIC_TOOL_STATUS = "Working through it…"

# Explicit, friendly verbs for the native tool surface. Anything not listed
# falls back to MCP/Composio derivation or the generic phrase below.
_TOOL_STATUS: dict[str, str] = {
    "web_search": "Searching the web…",
    "slack_channel_history": "Reading the channel…",
    "search_observed_slack_history": "Reading the channel…",
    "slack_actions": "Working in Slack…",
    "query_workspace_graph": "Searching your workspace knowledge…",
    "inspect_memory": "Checking what I remember…",
    "remember_fact": "Saving that to memory…",
    "forget_fact": "Updating memory…",
    "list_schedules": "Checking your schedules…",
    "create_schedule": "Setting up the schedule…",
    "cancel_schedule": "Updating your schedules…",
    "code_exec": "Running code…",
    "sandbox_bash": "Running code…",
    "sandbox_write_file": "Working in the sandbox…",
    "sandbox_read_file": "Working in the sandbox…",
    "pdf_generator": "Building your document…",
    "describe_tools": "Checking what I can do…",
    "load_skill": "Loading a skill…",
    "load_skill_resource": "Loading a skill…",
    "run_skill_script": "Running a skill…",
    "deploy_site": "Deploying…",
}


def _humanize_token(token: str) -> str:
    return token.replace("_", " ").replace("-", " ").strip().title()


def status_for_tool(tool_name: str, *, display_name: str | None = None) -> str:
    """Map a tool name to a human activity status for the assistant pane.

    Order: explicit native verb → MCP server (``mcp__<server>__<tool>``) →
    Composio toolkit prefix → provided ``display_name`` → generic phrase.
    """

    if tool_name in _TOOL_STATUS:
        return _TOOL_STATUS[tool_name]
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        server = _humanize_token(parts[1]) if len(parts) > 1 and parts[1] else None
        return f"Querying {server}…" if server else _GENERIC_TOOL_STATUS
    if tool_name.startswith("composio_"):
        toolkit = tool_name.removeprefix("composio_").split("_", 1)[0]
        return (
            f"Checking {_humanize_token(toolkit)}…" if toolkit else _GENERIC_TOOL_STATUS
        )
    if display_name:
        return f"Using {display_name}…"
    return _GENERIC_TOOL_STATUS
