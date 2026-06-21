"""Tests for browser tool classes (HIG-282 chunk 2).

Unit tests — no network, no DB, no live Playwright-MCP. All MCP calls are
intercepted via FakeBrowserSession.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from kortny.browser.session import BrowserSessionError, BrowserToolResult
from kortny.config import Settings
from kortny.tools.browser import (
    BrowserClickTool,
    BrowserNavigateBackTool,
    BrowserNavigateTool,
    BrowserSnapshotTool,
    BrowserTakeScreenshotTool,
    BrowserTypeTool,
    BrowserWaitForTool,
    _BrowserSessionHolder,
    build_browser_tools,
)
from kortny.tools.types import RecoverableToolError, ToolArtifact

# ---------------------------------------------------------------------------
# Settings helpers (mirror test_browser_session.py)
# ---------------------------------------------------------------------------

_BASE_SETTINGS: dict[str, str] = {
    "SLACK_BOT_TOKEN": "xoxb-test",
    "SLACK_APP_TOKEN": "xapp-test",
    "SLACK_SIGNING_SECRET": "signing-secret",
    "LLM_PROVIDER": "openrouter",
    "LLM_API_KEY": "llm-key",
    "LLM_MODEL": "openai/gpt-4o",
    "COMPOSIO_API_KEY": "composio-key",
    "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
    "ENCRYPTION_KEY": "ci-only-test-key",
    "KORTNY_BROWSER_URL": "",
}


def make_settings(**overrides: str) -> Settings:
    return Settings.model_validate({**_BASE_SETTINGS, **overrides})


# ---------------------------------------------------------------------------
# Fake PNG bytes for screenshot tests
# ---------------------------------------------------------------------------

# Minimal 1x1 white PNG for testing (same constant as test_browser_session.py).
_FAKE_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# Fake BrowserMcpSession for unit tests
# ---------------------------------------------------------------------------


class FakeBrowserSession:
    """Fake BrowserMcpSession for unit tests.

    Returns BrowserToolResult from call_tool so the tool layer sees the same
    structured type as the real session does.
    """

    def __init__(
        self,
        text_result: str = "ok",
        images: tuple[tuple[bytes, str], ...] = (),
        is_error: bool = False,
        raise_error: Exception | None = None,
    ) -> None:
        self.text_result = text_result
        self.images = images
        self.is_error = is_error
        self.raise_error = raise_error
        self.open_count = 0
        self.close_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def open(self) -> None:
        self.open_count += 1

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
    ) -> BrowserToolResult:
        self.calls.append((name, arguments))
        if self.raise_error:
            raise self.raise_error
        return BrowserToolResult(
            text=self.text_result,
            images=self.images,
            is_error=self.is_error,
        )

    def close(self) -> None:
        self.close_count += 1


# ---------------------------------------------------------------------------
# Helper: build holder + tool with a pre-wired fake session
# ---------------------------------------------------------------------------


def _make_holder_with_fake(
    fake: FakeBrowserSession,
    url: str = "http://fake:8931/mcp",
) -> _BrowserSessionHolder:
    settings = make_settings(KORTNY_BROWSER_URL=url)
    with patch("kortny.tools.browser.open_browser_session", return_value=fake):
        holder = _BrowserSessionHolder(settings)
        # Eagerly open so the fake is wired in before the test calls invoke.
        holder.get()
    return holder


# ---------------------------------------------------------------------------
# 1. Each tool calls the right MCP tool name + args
# ---------------------------------------------------------------------------


def test_browser_navigate_calls_mcp_navigate() -> None:
    fake = FakeBrowserSession(text_result="Navigated to example.com")
    holder = _make_holder_with_fake(fake)

    result = BrowserNavigateTool(holder).invoke({"url": "https://example.com"})

    assert fake.calls == [("browser_navigate", {"url": "https://example.com"})]
    assert result.output["url"] == "https://example.com"
    assert result.output["result"] == "Navigated to example.com"


def test_browser_snapshot_calls_mcp_snapshot() -> None:
    fake = FakeBrowserSession(text_result="<accessibility tree>")
    holder = _make_holder_with_fake(fake)

    result = BrowserSnapshotTool(holder).invoke({})

    assert fake.calls == [("browser_snapshot", {})]
    assert result.output["snapshot"] == "<accessibility tree>"


def test_browser_click_calls_mcp_click() -> None:
    fake = FakeBrowserSession(text_result="clicked")
    holder = _make_holder_with_fake(fake)

    result = BrowserClickTool(holder).invoke(
        {"element": "Submit button", "ref": "a1b2"}
    )

    assert fake.calls == [
        ("browser_click", {"element": "Submit button", "ref": "a1b2"})
    ]
    assert result.output["element"] == "Submit button"
    assert result.output["ref"] == "a1b2"
    assert result.output["result"] == "clicked"


def test_browser_type_calls_mcp_type() -> None:
    fake = FakeBrowserSession(text_result="typed")
    holder = _make_holder_with_fake(fake)

    result = BrowserTypeTool(holder).invoke(
        {"element": "Search box", "ref": "c3d4", "text": "hello"}
    )

    assert fake.calls == [
        (
            "browser_type",
            {"element": "Search box", "ref": "c3d4", "text": "hello", "submit": False},
        )
    ]
    assert result.output["element"] == "Search box"
    assert result.output["result"] == "typed"


def test_browser_type_with_submit_true() -> None:
    fake = FakeBrowserSession(text_result="submitted")
    holder = _make_holder_with_fake(fake)

    BrowserTypeTool(holder).invoke(
        {"element": "Search box", "ref": "c3d4", "text": "hello", "submit": True}
    )

    assert fake.calls[0][1]["submit"] is True


def test_browser_take_screenshot_returns_artifact_with_png_bytes() -> None:
    """Screenshot tool returns a ToolArtifact whose file contains the PNG bytes."""
    fake = FakeBrowserSession(
        text_result="",
        images=((_FAKE_PNG_BYTES, "image/png"),),
    )
    holder = _make_holder_with_fake(fake)

    result = BrowserTakeScreenshotTool(holder).invoke({})

    assert fake.calls == [("browser_take_screenshot", {})]
    # Must have exactly one artifact for the screenshot.
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert isinstance(artifact, ToolArtifact)
    assert artifact.filename == "screenshot.png"
    assert artifact.mime_type == "image/png"
    assert artifact.size_bytes == len(_FAKE_PNG_BYTES)
    # The artifact path must point to a file that contains the PNG bytes.
    assert artifact.path is not None
    assert Path(artifact.path).read_bytes() == _FAKE_PNG_BYTES


def test_browser_take_screenshot_no_image_returns_text() -> None:
    """When no image comes back, result has no artifacts and a text message."""
    fake = FakeBrowserSession(text_result="", images=())
    holder = _make_holder_with_fake(fake)

    result = BrowserTakeScreenshotTool(holder).invoke({})

    assert result.artifacts == ()
    assert "result" in result.output


def test_browser_take_screenshot_multiple_images() -> None:
    """Multiple images produce multiple artifacts with indexed filenames."""
    fake = FakeBrowserSession(
        text_result="two screenshots",
        images=(
            (_FAKE_PNG_BYTES, "image/png"),
            (_FAKE_PNG_BYTES, "image/png"),
        ),
    )
    holder = _make_holder_with_fake(fake)

    result = BrowserTakeScreenshotTool(holder).invoke({})

    assert len(result.artifacts) == 2
    assert result.artifacts[0].filename == "screenshot.png"
    assert result.artifacts[1].filename == "screenshot_1.png"
    assert result.output["result"] == "two screenshots"


def test_browser_wait_for_text() -> None:
    fake = FakeBrowserSession(text_result="found")
    holder = _make_holder_with_fake(fake)

    result = BrowserWaitForTool(holder).invoke({"text": "Loading complete"})

    assert fake.calls == [("browser_wait_for", {"text": "Loading complete"})]
    # Ensure 'time' was NOT included when not provided
    assert "time" not in fake.calls[0][1]
    assert result.output["result"] == "found"


def test_browser_wait_for_time() -> None:
    fake = FakeBrowserSession(text_result="waited")
    holder = _make_holder_with_fake(fake)

    result = BrowserWaitForTool(holder).invoke({"time": 2.0})

    assert fake.calls == [("browser_wait_for", {"time": 2.0})]
    assert "text" not in fake.calls[0][1]
    assert result.output["result"] == "waited"


def test_browser_wait_for_requires_text_or_time() -> None:
    fake = FakeBrowserSession()
    holder = _make_holder_with_fake(fake)

    with pytest.raises(RecoverableToolError) as exc_info:
        BrowserWaitForTool(holder).invoke({})

    assert exc_info.value.code == "browser_wait_invalid_args"


def test_browser_navigate_back_calls_mcp_back() -> None:
    fake = FakeBrowserSession(text_result="navigated back")
    holder = _make_holder_with_fake(fake)

    result = BrowserNavigateBackTool(holder).invoke({})

    assert fake.calls == [("browser_navigate_back", {})]
    assert result.output["result"] == "navigated back"


# ---------------------------------------------------------------------------
# 2. Error handling
# ---------------------------------------------------------------------------


def test_browser_session_error_raises_recoverable() -> None:
    fake = FakeBrowserSession(raise_error=BrowserSessionError("conn failed"))
    holder = _make_holder_with_fake(fake)

    with pytest.raises(RecoverableToolError) as exc_info:
        BrowserNavigateTool(holder).invoke({"url": "https://example.com"})

    assert exc_info.value.code == "browser_unavailable"
    assert "conn failed" in exc_info.value.message


def test_unexpected_error_raises_recoverable() -> None:
    fake = FakeBrowserSession(raise_error=RuntimeError("boom"))
    holder = _make_holder_with_fake(fake)

    with pytest.raises(RecoverableToolError) as exc_info:
        BrowserSnapshotTool(holder).invoke({})

    assert exc_info.value.code == "browser_error"
    assert "boom" in exc_info.value.message


def test_is_error_true_raises_recoverable() -> None:
    """When BrowserToolResult.is_error=True, tools raise RecoverableToolError."""
    fake = FakeBrowserSession(text_result="nav failed", is_error=True)
    holder = _make_holder_with_fake(fake)

    with pytest.raises(RecoverableToolError) as exc_info:
        BrowserNavigateTool(holder).invoke({"url": "https://example.com"})

    assert exc_info.value.code == "browser_error"
    assert "nav failed" in exc_info.value.message


def test_screenshot_is_error_raises_recoverable() -> None:
    """Screenshot tool also raises RecoverableToolError on is_error=True."""
    fake = FakeBrowserSession(text_result="screenshot failed", is_error=True)
    holder = _make_holder_with_fake(fake)

    with pytest.raises(RecoverableToolError) as exc_info:
        BrowserTakeScreenshotTool(holder).invoke({})

    assert exc_info.value.code == "browser_error"
    assert "screenshot failed" in exc_info.value.message


# ---------------------------------------------------------------------------
# 3. Session lifecycle: shared holder opens session once
# ---------------------------------------------------------------------------


def test_session_opened_once_across_multiple_tools() -> None:
    settings = make_settings(KORTNY_BROWSER_URL="http://fake:8931/mcp")
    fake = FakeBrowserSession(text_result="nav-result")

    with patch("kortny.tools.browser.open_browser_session", return_value=fake):
        holder = _BrowserSessionHolder(settings)
        nav_tool = BrowserNavigateTool(holder)
        snap_tool = BrowserSnapshotTool(holder)

        nav_tool.invoke({"url": "https://example.com"})
        snap_tool.invoke({})

    # open called exactly once despite two tool invocations
    assert fake.open_count == 1
    # both tools made their calls
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# 4. Catalog: every browser tool has metadata
# ---------------------------------------------------------------------------


def test_browser_tools_in_catalog() -> None:
    from kortny.tools.catalog import NATIVE_TOOL_METADATA

    browser_tools = [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_take_screenshot",
        "browser_wait_for",
        "browser_navigate_back",
    ]
    for name in browser_tools:
        assert name in NATIVE_TOOL_METADATA, f"{name} missing from catalog"
        assert NATIVE_TOOL_METADATA[name].namespace == "native.browser", (
            f"{name} has wrong namespace"
        )
        assert NATIVE_TOOL_METADATA[name].runtime_registered is False, (
            f"{name} should have runtime_registered=False"
        )


# ---------------------------------------------------------------------------
# 5. Trifecta: browser tools arm the gate
# ---------------------------------------------------------------------------


def test_browser_tools_arm_trifecta() -> None:
    from kortny.agent.trifecta import TrifectaGateState, is_untrusted_origin_tool

    browser_tool_names = [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_take_screenshot",
        "browser_wait_for",
        "browser_navigate_back",
    ]
    for tool_name in browser_tool_names:
        assert is_untrusted_origin_tool(tool_name), f"{tool_name} should be untrusted"

    gate = TrifectaGateState()
    assert not gate.armed
    gate.note_tool_result("browser_navigate")
    assert gate.armed
    assert gate.armed_by == "browser_navigate"


# ---------------------------------------------------------------------------
# 6. Disabled/enabled path for build_browser_tools
# ---------------------------------------------------------------------------


def test_build_browser_tools_disabled_when_no_url() -> None:
    settings = make_settings()  # no KORTNY_BROWSER_URL
    tools, holder = build_browser_tools(settings)
    assert tools == ()
    assert holder is None


def test_build_browser_tools_enabled_when_url_set() -> None:
    settings = make_settings(KORTNY_BROWSER_URL="http://fake:8931/mcp")
    with patch("kortny.tools.browser.open_browser_session") as mock_open:
        mock_open.return_value = FakeBrowserSession()
        tools, holder = build_browser_tools(settings)
    assert len(tools) == 7
    assert holder is not None


# ---------------------------------------------------------------------------
# 7. Holder close lifecycle
# ---------------------------------------------------------------------------


def test_holder_close_closes_session() -> None:
    settings = make_settings(KORTNY_BROWSER_URL="http://fake:8931/mcp")
    fake = FakeBrowserSession()

    with patch("kortny.tools.browser.open_browser_session", return_value=fake):
        holder = _BrowserSessionHolder(settings)
        holder.get()
        holder.close()

    assert fake.close_count == 1


def test_holder_close_before_open_is_safe() -> None:
    settings = make_settings(KORTNY_BROWSER_URL="http://fake:8931/mcp")
    holder = _BrowserSessionHolder(settings)
    holder.close()  # should not raise even if session was never opened


def test_holder_close_is_idempotent() -> None:
    settings = make_settings(KORTNY_BROWSER_URL="http://fake:8931/mcp")
    fake = FakeBrowserSession()

    with patch("kortny.tools.browser.open_browser_session", return_value=fake):
        holder = _BrowserSessionHolder(settings)
        holder.get()
        holder.close()
        holder.close()  # second close should not raise or double-close

    assert fake.close_count == 1
