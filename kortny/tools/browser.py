"""Browser tools backed by Playwright-MCP (HIG-282).

All seven tools share ONE lazily-opened BrowserMcpSession per task via
_BrowserSessionHolder. The holder is created in native_runtime / agent_executor
and closed in the agent_executor finally block so browser sessions never leak
across tasks.

Call flow:
    tool.invoke(args)
        -> holder.get()   # opens session on first call
        -> session.call_tool(playwright_mcp_tool_name, mcp_args)  -> BrowserToolResult
        -> ToolResult(output={...}, artifacts=(...))

Error contract:
    BrowserSessionError  -> RecoverableToolError(code="browser_unavailable")
    result.is_error=True -> RecoverableToolError(code="browser_error")
    Any other exception  -> RecoverableToolError(code="browser_error")
"""

from __future__ import annotations

import contextlib
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

from kortny.browser.session import (
    BrowserMcpSession,
    BrowserSessionError,
    BrowserToolResult,
    open_browser_session,
)
from kortny.config import Settings
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    ToolArtifact,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Session holder
# ---------------------------------------------------------------------------


class _BrowserSessionHolder:
    """Lazily opens ONE BrowserMcpSession per task, shared across all browser tools.

    The holder is created at registry-build time. ``get()`` is called inside
    each tool's ``invoke`` — the session is opened on the first call and reused
    for the lifetime of the task. ``close()`` is called by the agent_executor
    finally block; it is best-effort and safe to call before the session is
    ever opened.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: BrowserMcpSession | None = None

    def get(self) -> BrowserMcpSession:
        """Return the open session, opening it lazily on first call.

        Raises RecoverableToolError when the browser is disabled (no MCP URL).
        """
        if self._session is not None:
            return self._session
        session = open_browser_session(self._settings)
        if session is None:
            raise RecoverableToolError(
                code="browser_unavailable",
                message="Browser is not enabled on this Kortny instance (KORTNY_BROWSER_MCP_URL is not set).",
                hint="Ask the workspace admin to configure the browser integration.",
            )
        session.open()
        self._session = session
        return self._session

    def close(self) -> None:
        """Best-effort close of the underlying session."""
        session = self._session
        self._session = None
        if session is None:
            return
        with contextlib.suppress(Exception):
            session.close()


# ---------------------------------------------------------------------------
# Shared call helper
# ---------------------------------------------------------------------------


def _call_browser(
    holder: _BrowserSessionHolder,
    mcp_tool: str,
    mcp_args: dict[str, Any],
) -> BrowserToolResult:
    """Call the Playwright-MCP tool and return a structured BrowserToolResult.

    Converts BrowserSessionError -> browser_unavailable and any other exception
    -> browser_error so no raw exceptions escape to the coordinator.
    """
    try:
        session = holder.get()
    except RecoverableToolError:
        raise
    except Exception as exc:
        raise RecoverableToolError(
            code="browser_error",
            message=f"Failed to open browser session: {exc}",
            hint="The browser session encountered an error. Try again or simplify the task.",
        ) from exc

    try:
        return session.call_tool(mcp_tool, mcp_args)
    except BrowserSessionError as exc:
        raise RecoverableToolError(
            code="browser_unavailable",
            message=f"Browser session error: {exc}",
            hint="The browser session encountered an error. Try again or simplify the task.",
        ) from exc
    except Exception as exc:
        raise RecoverableToolError(
            code="browser_error",
            message=f"Unexpected browser error: {exc}",
            hint="The browser session encountered an unexpected error. Try again or simplify the task.",
        ) from exc


def _raise_if_error(result: BrowserToolResult, tool_name: str) -> None:
    """Raise RecoverableToolError when the MCP server flagged result as an error."""
    if result.is_error:
        raise RecoverableToolError(
            code="browser_error",
            message=result.text or f"{tool_name} returned an error with no message",
            hint="The browser tool reported an error. Check the page state and try again.",
        )


def _images_to_artifacts(
    images: tuple[tuple[bytes, str], ...],
) -> tuple[ToolArtifact, ...]:
    """Write image bytes to temp files and return ToolArtifact references.

    Each image is written to a NamedTemporaryFile (delete=False) so the
    bytes survive until the coordinator consumes the artifact.  The caller
    is responsible for cleanup; in practice the files are small and the OS
    will reclaim them at process exit.
    """
    artifacts: list[ToolArtifact] = []
    for idx, (img_bytes, mime) in enumerate(images):
        filename = "screenshot.png" if idx == 0 else f"screenshot_{idx}.png"
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="kortny_screenshot_"
        ) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        artifacts.append(
            ToolArtifact(
                filename=filename,
                path=tmp_path,
                mime_type=mime,
                size_bytes=len(img_bytes),
            )
        )
    return tuple(artifacts)


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


class BrowserNavigateTool:
    """Navigate the browser to a URL."""

    name = "browser_navigate"
    description = (
        "Navigate the browser to a URL. Returns the page title / confirmation text "
        "from Playwright-MCP. Call browser_snapshot afterwards to read the page."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The fully-qualified URL to navigate to (e.g. https://example.com).",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        url = str(args["url"])
        result = _call_browser(self._holder, "browser_navigate", {"url": url})
        _raise_if_error(result, self.name)
        return ToolResult(
            output={"url": url, "result": result.text},
            cost_usd=Decimal("0"),
            artifacts=(),
        )


class BrowserSnapshotTool:
    """Capture the accessibility-tree snapshot of the current page."""

    name = "browser_snapshot"
    description = (
        "Return the accessibility-tree snapshot of the current browser page. "
        "The snapshot contains visible text and element refs (e.g. ref='a1b2') "
        "that you can pass to browser_click and browser_type. This is the main "
        "read/extract primitive — call it after navigating to a page."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        result = _call_browser(self._holder, "browser_snapshot", {})
        _raise_if_error(result, self.name)
        return ToolResult(
            output={"snapshot": result.text},
            cost_usd=Decimal("0"),
            artifacts=(),
        )


class BrowserClickTool:
    """Click a browser element identified by its ref from a snapshot."""

    name = "browser_click"
    description = (
        "Click a browser element. Obtain the element label and ref from "
        "browser_snapshot first. May trigger page navigation or form submission."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "element": {
                "type": "string",
                "description": "Human-readable label of the element (e.g. 'Submit button').",
            },
            "ref": {
                "type": "string",
                "description": "Element ref from the snapshot (e.g. 'a1b2').",
            },
        },
        "required": ["element", "ref"],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        element = str(args["element"])
        ref = str(args["ref"])
        result = _call_browser(
            self._holder,
            "browser_click",
            {"element": element, "ref": ref},
        )
        _raise_if_error(result, self.name)
        return ToolResult(
            output={"element": element, "ref": ref, "result": result.text},
            cost_usd=Decimal("0"),
            artifacts=(),
        )


class BrowserTypeTool:
    """Type text into a browser element identified by its ref from a snapshot."""

    name = "browser_type"
    description = (
        "Type text into a browser element. Obtain the element label and ref from "
        "browser_snapshot first. Set submit=true to press Enter after typing."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "element": {
                "type": "string",
                "description": "Human-readable label of the element (e.g. 'Search box').",
            },
            "ref": {
                "type": "string",
                "description": "Element ref from the snapshot (e.g. 'c3d4').",
            },
            "text": {
                "type": "string",
                "description": "Text to type into the element.",
            },
            "submit": {
                "type": "boolean",
                "description": "If true, press Enter after typing to submit the form.",
                "default": False,
            },
        },
        "required": ["element", "ref", "text"],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        element = str(args["element"])
        ref = str(args["ref"])
        text_input = str(args["text"])
        submit = bool(args.get("submit", False))
        result = _call_browser(
            self._holder,
            "browser_type",
            {"element": element, "ref": ref, "text": text_input, "submit": submit},
        )
        _raise_if_error(result, self.name)
        return ToolResult(
            output={"element": element, "result": result.text},
            cost_usd=Decimal("0"),
            artifacts=(),
        )


class BrowserTakeScreenshotTool:
    """Take a screenshot of the current browser page."""

    name = "browser_take_screenshot"
    description = (
        "Take a screenshot of the current browser page. Returns the PNG image "
        "as an artifact alongside any accompanying text from Playwright-MCP."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        result = _call_browser(self._holder, "browser_take_screenshot", {})
        _raise_if_error(result, self.name)

        if result.images:
            artifacts = _images_to_artifacts(result.images)
            output: JsonObject = {"screenshot_count": len(result.images)}
            if result.text:
                output["result"] = result.text
        else:
            artifacts = ()
            output = {
                "result": result.text or "Screenshot taken but no image was returned."
            }

        return ToolResult(
            output=output,
            cost_usd=Decimal("0"),
            artifacts=artifacts,
        )


class BrowserWaitForTool:
    """Wait for text to appear on the page or for a time duration."""

    name = "browser_wait_for"
    description = (
        "Wait for text to appear on the current browser page, or wait for a "
        "fixed time duration. At least one of 'text' or 'time' must be provided."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to wait for on the page.",
            },
            "time": {
                "type": "number",
                "description": "Time in seconds to wait.",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        text_arg: str | None = args.get("text")
        time_arg: float | None = args.get("time")
        if text_arg is None and time_arg is None:
            raise RecoverableToolError(
                code="browser_wait_invalid_args",
                message="browser_wait_for requires at least one of 'text' or 'time'.",
                hint="Provide 'text' (a string to wait for on the page) or 'time' (seconds to wait).",
            )
        mcp_args: dict[str, Any] = {}
        if text_arg is not None:
            mcp_args["text"] = str(text_arg)
        if time_arg is not None:
            mcp_args["time"] = float(time_arg)
        result = _call_browser(self._holder, "browser_wait_for", mcp_args)
        _raise_if_error(result, self.name)
        return ToolResult(
            output={"result": result.text},
            cost_usd=Decimal("0"),
            artifacts=(),
        )


class BrowserNavigateBackTool:
    """Navigate the browser back to the previous page."""

    name = "browser_navigate_back"
    description = "Navigate the browser back to the previous page in the history stack."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, holder: _BrowserSessionHolder) -> None:
        self._holder = holder

    def invoke(self, args: JsonObject) -> ToolResult:
        result = _call_browser(self._holder, "browser_navigate_back", {})
        _raise_if_error(result, self.name)
        return ToolResult(
            output={"result": result.text},
            cost_usd=Decimal("0"),
            artifacts=(),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_browser_tools(
    settings: Settings,
) -> tuple[tuple[Any, ...], _BrowserSessionHolder | None]:
    """Return (tools_tuple, holder) or ((), None) when browser is disabled."""
    if not settings.browser_enabled:
        return (), None
    holder = _BrowserSessionHolder(settings)
    tools: tuple[Any, ...] = (
        BrowserNavigateTool(holder),
        BrowserSnapshotTool(holder),
        BrowserClickTool(holder),
        BrowserTypeTool(holder),
        BrowserTakeScreenshotTool(holder),
        BrowserWaitForTool(holder),
        BrowserNavigateBackTool(holder),
    )
    return tools, holder


__all__ = [
    "_BrowserSessionHolder",
    "BrowserNavigateTool",
    "BrowserSnapshotTool",
    "BrowserClickTool",
    "BrowserTypeTool",
    "BrowserTakeScreenshotTool",
    "BrowserWaitForTool",
    "BrowserNavigateBackTool",
    "build_browser_tools",
]
