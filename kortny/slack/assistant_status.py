"""Live assistant status narration and Claude-Tag-style progress checklist.

Drives Slack's ``assistant.threads.setStatus`` from the worker so the loading
indicator in the agent pane reflects what Kortny is *actually* doing — the tool
it is calling, the phase it is in — instead of a static cycling loop. The app
process sets the initial cycling ``loading_messages`` when the message arrives;
once the worker claims the task it takes over with real, activity-specific
status lines. ``setStatus`` is stateless and accepts the bot token from any
process, so the worker can drive it directly.

For channel tasks (app-mention / non-assistant), ``ChannelProgressReporter``
edits the ack message in place.  When the coordinator produces a real plan with
two or more non-internal steps, the reporter switches into checklist mode and
renders a Claude-Tag-style checklist (``✓``/``✱``/``○``) via ``chat.update``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Minimum seconds between channel ack edits — Slack rate-limits chat.update and
# rapid edits are noise. Phase transitions are coarse, so this rarely bites.
DEFAULT_PROGRESS_THROTTLE_SECONDS = 5.0


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

    def report(self, status: str, *, phase: str | None = None) -> None:
        """Surface ``status`` to the user (best-effort, never raises)."""


class NullStatusReporter:
    """No-op reporter for non-assistant tasks (the default everywhere)."""

    def report(self, status: str, *, phase: str | None = None) -> None:  # noqa: D102
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
        self._last: tuple[str, str] | None = None

    def report(self, status: str, *, phase: str | None = None) -> None:
        """Surface the current activity as a two-level status.

        ``status`` is the live, granular step (the tool being run) and drives the
        prominent ``loading_messages`` bubble below the app name. ``phase`` is the
        coarse macro-phase ("is researching…", "is putting it together…") and
        drives the small composer line — it changes only a few times per task, so
        the two lines complement rather than mirror each other. ``loading_messages``
        must be a NON-EMPTY list — Slack rejects ``[]`` with invalid_arguments
        ("must provide at least 1 items") and there is no way to clear it back to
        nothing — so we always send the single current step (one item = no
        rotation). The native step-timeline (TaskUpdateChunk) is HIG-252.
        """

        step = status.strip()
        if not step:
            return
        composer = (phase or "").strip() or step
        key = (step, composer)
        if key == self._last:
            return
        setter = getattr(self._client, "assistant_threads_setStatus", None)
        if not callable(setter):
            return
        try:
            setter(
                channel_id=self._channel_id,
                thread_ts=self._thread_ts,
                status=composer,
                loading_messages=[step],
            )
            self._last = key
        except Exception:
            logger.warning(
                "failed to set assistant status channel=%s thread_ts=%s status=%s",
                self._channel_id,
                self._thread_ts,
                step,
                exc_info=True,
            )


class MessageUpdateClient(Protocol):
    """Subset of the Slack WebClient used to edit a posted message."""

    def chat_update(self, *, channel: str, ts: str, text: str) -> Any:
        """Edit the text of an existing message."""


# ---------------------------------------------------------------------------
# Checklist helpers
# ---------------------------------------------------------------------------

# Step labels that are internal/system and should never appear in the checklist.
_INTERNAL_STEP_LABELS: frozenset[str] = frozenset(
    {
        "Handle the Slack request using available context and tools.",
        "Formatting/Synthesizing the answer",
        "Planning",
        "Finalizing",
        "Compiling",
    }
)

_INTERNAL_STEP_PREFIXES: tuple[str, ...] = (
    "handle the slack request",
    "format",
    "plan",
    "finaliz",
    "compil",
    "synthesiz",
)


def _is_internal_step(label: str) -> bool:
    """Return True if this step label is internal/system and should be hidden."""

    if label in _INTERNAL_STEP_LABELS:
        return True
    lower = label.lower().strip()
    return any(lower.startswith(p) for p in _INTERNAL_STEP_PREFIXES)


def _render_checklist(
    steps: list[str],
    current_idx: int,
    *,
    all_done: bool,
) -> str:
    """Render a checked/in-progress/pending checklist string.

    Glyphs: ``✓`` completed, ``✱`` in-progress, ``○`` pending.
    ``current_idx`` is the index of the in-progress step (-1 = none started).
    ``all_done`` marks every step completed regardless of ``current_idx``.
    """

    lines: list[str] = []
    for i, label in enumerate(steps):
        if all_done or i < current_idx:
            glyph = "✓"  # ✓
        elif i == current_idx:
            glyph = "✱"  # ✱
        else:
            glyph = "○"  # ○
        lines.append(f"{glyph} {label}")
    return "\n".join(lines)


def _checklist_text(
    base_text: str, steps: list[str], current_idx: int, *, all_done: bool
) -> str:
    """Compose the full message text for a checklist update."""

    checklist = _render_checklist(steps, current_idx, all_done=all_done)
    return f"{base_text}\n{checklist}" if base_text else checklist


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------


class ChannelProgressReporter:
    """Narrates progress by editing the channel acknowledgement message.

    Assistant-pane tasks get a native loading indicator (AssistantStatusReporter);
    channel / app-mention tasks have only the posted ack, so we edit it in place
    (chat.update) as the task moves through phases — "Checking Linear…",
    "Writing the reply…" — appended under the original ack text so it is never
    lost. Throttled to >= ``min_interval_seconds`` between edits and deduped on
    content; never raises (a progress edit must not fail the task).

    When the coordinator produces a real execution plan with two or more
    non-internal steps, the reporter switches to *checklist mode*.  In that mode
    ``notify_plan``/``notify_step_started``/``notify_completed`` drive structured
    ``✓``/``✱``/``○`` updates via ``chat.update`` and ``report()`` becomes a
    no-op (the checklist methods own the message from that point).
    """

    def __init__(
        self,
        *,
        client: MessageUpdateClient,
        channel_id: str,
        message_ts: str,
        base_text: str,
        min_interval_seconds: float = DEFAULT_PROGRESS_THROTTLE_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._channel_id = channel_id
        self._message_ts = message_ts
        self._base_text = base_text.strip()
        self._min_interval = min_interval_seconds
        self._clock = clock or time.monotonic
        self._last_line: str | None = None
        self._last_at: float | None = None
        # Checklist state — inactive until notify_plan() is called with ≥2 steps.
        self._checklist_mode: bool = False
        self._plan_steps: list[str] = []
        self._current_step_idx: int = -1
        self._plan_completed: bool = False

    def report(self, status: str, *, phase: str | None = None) -> None:
        # In checklist mode the structured notify_* methods own all updates.
        if self._checklist_mode:
            return
        line = (status or "").strip()
        if not line:
            return
        if line == self._last_line:
            return
        now = self._clock()
        if self._last_at is not None and (now - self._last_at) < self._min_interval:
            return
        updater = getattr(self._client, "chat_update", None)
        if not callable(updater):
            return
        text = f"{self._base_text}\n_{line}_" if self._base_text else f"_{line}_"
        try:
            updater(channel=self._channel_id, ts=self._message_ts, text=text)
            self._last_line = line
            self._last_at = now
        except Exception:
            logger.warning(
                "failed to update channel progress channel=%s ts=%s status=%s",
                self._channel_id,
                self._message_ts,
                line,
                exc_info=True,
            )

    def notify_plan(self, steps: list[str]) -> None:
        """Switch to checklist mode when the plan has two or more real steps.

        ``steps`` must already be filtered (internal steps removed).  If fewer
        than two steps remain after filtering, checklist mode is not activated
        and the reporter continues with single-line progress.
        """

        if len(steps) < 2:
            return
        self._checklist_mode = True
        self._plan_steps = list(steps)
        self._current_step_idx = -1
        self._plan_completed = False
        self._chat_update(
            _checklist_text(self._base_text, self._plan_steps, -1, all_done=False)
        )

    def notify_step_started(self, step_label: str) -> None:
        """Advance the checklist to mark ``step_label`` as in-progress."""

        if not self._checklist_mode:
            return
        try:
            idx = self._plan_steps.index(step_label)
        except ValueError:
            # Step not in the visible checklist (internal step or unknown) — skip.
            return
        self._current_step_idx = idx
        self._chat_update(
            _checklist_text(
                self._base_text,
                self._plan_steps,
                self._current_step_idx,
                all_done=False,
            )
        )

    def notify_completed(self) -> None:
        """Mark all checklist steps as completed."""

        if not self._checklist_mode:
            return
        self._plan_completed = True
        self._chat_update(
            _checklist_text(self._base_text, self._plan_steps, -1, all_done=True)
        )

    def _chat_update(self, text: str) -> None:
        updater = getattr(self._client, "chat_update", None)
        if not callable(updater):
            return
        try:
            updater(channel=self._channel_id, ts=self._message_ts, text=text)
        except Exception:
            logger.warning(
                "failed to update channel checklist channel=%s ts=%s",
                self._channel_id,
                self._message_ts,
                exc_info=True,
            )


# Phase statuses the coordinator reports outside of tool calls.
STATUS_GETTING_STARTED = "Getting up to speed…"
STATUS_WRITING = "Writing the response…"

# Coarse macro-phases for the small composer line. These change only a handful of
# times per task so they read as complementary to the granular step bubble rather
# than a duplicate of it. Phrased to follow the bot name ("Kortny is researching…").
PHASE_STARTING = "is getting started…"
PHASE_RESEARCHING = "is gathering what it needs…"
PHASE_WORKING = "is putting it together…"
PHASE_WRITING = "is writing the reply…"

# Tools whose work is gathering inputs/context rather than producing output.
_RESEARCH_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "slack_channel_history",
        "search_observed_slack_history",
        "query_workspace_graph",
        "inspect_memory",
        "recall_fact",
        "list_schedules",
        "get_schedule",
        "describe_tools",
        "list_integrations",
        "load_skill",
        "load_skill_resource",
        "slack_user_info",
        "slack_channel_info",
        "resolve_slack_identity",
        "slack_file_read",
    }
)

_GENERIC_TOOL_STATUS = "Working through it…"


def phase_for_tool(tool_name: str) -> str:
    """Map a tool to a coarse macro-phase for the composer status line.

    Read-only lookups (search, history, skill/memory reads, MCP queries) are the
    research phase; everything else — code, sandbox, writes, integrations — is the
    build/work phase. Deliberately coarse so the line stays stable across the many
    granular tool calls within a phase.
    """

    if tool_name in _RESEARCH_TOOLS or tool_name.startswith("mcp__"):
        return PHASE_RESEARCHING
    return PHASE_WORKING


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
