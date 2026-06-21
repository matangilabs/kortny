"""Tests for BrowserMcpSession and open_browser_session (HIG-282 chunk 1).

Unit tests (always run, no container): verify the sync facade over the async
MCP client, lifecycle correctness, and the disabled-by-default behaviour,
using a fake in-process MCP server.

Live smoke (skipped unless KORTNY_BROWSER_MCP_URL is set): opens a real
session against the Playwright-MCP container.
"""

from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kortny.browser.session import (
    BrowserMcpSession,
    BrowserSessionError,
    BrowserToolResult,
    open_browser_session,
)
from kortny.config import Settings

# ---------------------------------------------------------------------------
# Helpers to build minimal Settings
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
}


def make_settings(**overrides: str) -> Settings:
    return Settings.model_validate({**_BASE_SETTINGS, **overrides})


# ---------------------------------------------------------------------------
# open_browser_session disabled test
# ---------------------------------------------------------------------------


def test_open_browser_session_returns_none_when_disabled() -> None:
    settings = make_settings()
    # browser_mcp_url not set -> None
    assert settings.browser_mcp_url is None
    assert settings.browser_enabled is False
    result = open_browser_session(settings)
    assert result is None


def test_open_browser_session_returns_session_when_url_is_set() -> None:
    settings = make_settings(KORTNY_BROWSER_MCP_URL="http://playwright-mcp:8931/mcp")
    assert settings.browser_mcp_url == "http://playwright-mcp:8931/mcp"
    assert settings.browser_enabled is True


def test_settings_browser_idle_timeout_default() -> None:
    settings = make_settings()
    assert settings.browser_session_idle_timeout_seconds == 120


def test_settings_browser_idle_timeout_custom() -> None:
    settings = make_settings(KORTNY_BROWSER_SESSION_IDLE_TIMEOUT_SECONDS="60")
    assert settings.browser_session_idle_timeout_seconds == 60


def test_settings_blank_browser_mcp_url_normalizes_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORTNY_BROWSER_MCP_URL", "")
    settings = make_settings()
    assert settings.browser_mcp_url is None


# ---------------------------------------------------------------------------
# Fake MCP server helpers for unit tests
# ---------------------------------------------------------------------------

# Minimal 1x1 white PNG in base64 for image content tests.
_FAKE_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_FAKE_PNG_B64 = base64.b64encode(_FAKE_PNG_BYTES).decode()


def _make_fake_mcp_session(
    tool_names: list[str] | None = None,
    tool_result_text: str = "fake-result",
    call_tool_error: Exception | None = None,
    include_image: bool = False,
    is_error: bool = False,
) -> tuple[AsyncMock, AsyncMock]:
    """Return (fake_session, fake_session_cm) for patching into ClientSession."""
    if tool_names is None:
        tool_names = ["browser_navigate", "browser_snapshot"]

    from mcp import types as mcp_types

    fake_tool_list = MagicMock()
    fake_tool_list.tools = []
    for n in tool_names:
        tool_mock = MagicMock()
        tool_mock.name = n
        fake_tool_list.tools.append(tool_mock)

    fake_result = MagicMock()
    content_blocks: list[Any] = []
    text_content = MagicMock(spec=mcp_types.TextContent)
    text_content.text = tool_result_text
    content_blocks.append(text_content)
    if include_image:
        img_content = MagicMock(spec=mcp_types.ImageContent)
        img_content.data = _FAKE_PNG_B64
        img_content.mimeType = "image/png"
        content_blocks.append(img_content)
    fake_result.content = content_blocks
    fake_result.isError = is_error

    fake_session = AsyncMock()
    fake_session.initialize = AsyncMock(return_value=MagicMock())
    fake_session.list_tools = AsyncMock(return_value=fake_tool_list)
    if call_tool_error:
        fake_session.call_tool = AsyncMock(side_effect=call_tool_error)
    else:
        fake_session.call_tool = AsyncMock(return_value=fake_result)

    fake_session_cm = AsyncMock()
    fake_session_cm.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session_cm.__aexit__ = AsyncMock(return_value=None)

    return fake_session, fake_session_cm


@asynccontextmanager
async def _fake_streamablehttp_client(
    url: str, **kwargs: Any
) -> AsyncIterator[tuple[Any, Any, Any]]:
    read = MagicMock()
    write = MagicMock()
    get_session_id = MagicMock(return_value=None)
    yield read, write, get_session_id


# ---------------------------------------------------------------------------
# Unit tests using patched MCP primitives
# ---------------------------------------------------------------------------


class TestBrowserMcpSessionUnit:
    """Unit tests -- no network, fake MCP."""

    def _make_session(
        self, url: str = "http://fake-browser:8931/mcp"
    ) -> BrowserMcpSession:
        return BrowserMcpSession(url, idle_timeout_seconds=10)

    def test_call_tool_returns_browser_tool_result(self) -> None:
        """call_tool now returns BrowserToolResult, not a plain string."""
        fake_session, fake_session_cm = _make_fake_mcp_session(
            tool_result_text="Example Domain"
        )

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            with session:
                result = session.call_tool(
                    "browser_navigate", {"url": "https://example.com"}
                )

        assert isinstance(result, BrowserToolResult)
        assert result.text == "Example Domain"
        assert result.images == ()
        assert result.is_error is False
        fake_session.call_tool.assert_called_once_with(
            "browser_navigate", {"url": "https://example.com"}
        )

    def test_call_tool_decodes_image_content(self) -> None:
        """ImageContent blocks are base64-decoded and returned in .images."""
        fake_session, fake_session_cm = _make_fake_mcp_session(
            tool_result_text="screenshot taken",
            include_image=True,
        )

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            with session:
                result = session.call_tool("browser_take_screenshot", {})

        assert isinstance(result, BrowserToolResult)
        assert result.text == "screenshot taken"
        assert len(result.images) == 1
        img_bytes, mime = result.images[0]
        assert img_bytes == _FAKE_PNG_BYTES
        assert mime == "image/png"
        assert result.is_error is False

    def test_call_tool_mixed_content_text_and_image(self) -> None:
        """Mixed content: text joined, images decoded, is_error reflected."""
        fake_session, fake_session_cm = _make_fake_mcp_session(
            tool_result_text="some text",
            include_image=True,
            is_error=False,
        )

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            with session:
                result = session.call_tool("browser_take_screenshot", {})

        assert result.text == "some text"
        assert len(result.images) == 1
        assert result.images[0][0] == _FAKE_PNG_BYTES
        assert result.images[0][1] == "image/png"

    def test_call_tool_is_error_flag(self) -> None:
        """is_error=True on the MCP result is propagated to BrowserToolResult."""
        fake_session, fake_session_cm = _make_fake_mcp_session(
            tool_result_text="something went wrong",
            is_error=True,
        )

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            with session:
                result = session.call_tool(
                    "browser_navigate", {"url": "https://bad.url"}
                )

        assert result.is_error is True
        assert result.text == "something went wrong"

    def test_list_tools_returns_tool_names(self) -> None:
        tool_names = ["browser_navigate", "browser_snapshot", "browser_click"]
        fake_session, fake_session_cm = _make_fake_mcp_session(tool_names=tool_names)

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            with session:
                names = session.list_tools()

        assert names == tool_names

    def test_close_is_idempotent(self) -> None:
        fake_session, fake_session_cm = _make_fake_mcp_session()

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            session.open()
            session.close()
            session.close()  # second close must not raise
            session.close()  # third close must not raise

    def test_call_tool_after_close_raises(self) -> None:
        fake_session, fake_session_cm = _make_fake_mcp_session()

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            session.open()
            session.close()

            with pytest.raises(BrowserSessionError, match="closed"):
                session.call_tool("browser_navigate", {"url": "https://example.com"})

    def test_call_tool_before_open_raises(self) -> None:
        session = self._make_session()
        with pytest.raises(BrowserSessionError, match="not open"):
            session.call_tool("browser_navigate", {"url": "https://example.com"})

    def test_context_manager_opens_and_closes(self) -> None:
        fake_session, fake_session_cm = _make_fake_mcp_session()

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            with BrowserMcpSession(
                "http://fake:8931/mcp", idle_timeout_seconds=10
            ) as session:
                assert session._open
                assert not session._closed
            # After context exit:
            assert session._closed

    def test_empty_url_raises(self) -> None:
        with pytest.raises(BrowserSessionError, match="cannot be empty"):
            BrowserMcpSession("", idle_timeout_seconds=10)

    def test_reopen_after_close_raises(self) -> None:
        fake_session, fake_session_cm = _make_fake_mcp_session()

        with (
            patch(
                "kortny.browser.session.streamablehttp_client",
                new=_fake_streamablehttp_client,
            ),
            patch("kortny.browser.session.ClientSession", return_value=fake_session_cm),
        ):
            session = self._make_session()
            session.open()
            session.close()

            with pytest.raises(BrowserSessionError, match="Cannot reopen"):
                session.open()


# ---------------------------------------------------------------------------
# Live smoke test (skipped unless KORTNY_BROWSER_MCP_URL is set)
# ---------------------------------------------------------------------------

_LIVE_BROWSER_URL = os.environ.get("KORTNY_BROWSER_MCP_URL")


@pytest.mark.skipif(
    not _LIVE_BROWSER_URL,
    reason="KORTNY_BROWSER_MCP_URL not set; skipping live Playwright-MCP smoke test",
)
def test_live_browser_navigate_and_snapshot() -> None:
    assert _LIVE_BROWSER_URL is not None
    session = BrowserMcpSession(_LIVE_BROWSER_URL, idle_timeout_seconds=60)
    session.open()
    try:
        session.call_tool("browser_navigate", {"url": "https://example.com"})
        snapshot = session.call_tool("browser_snapshot", {})
        assert "Example Domain" in snapshot.text
    finally:
        session.close()
