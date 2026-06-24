"""Tests for the <connected_integrations> context block (capabilities.py)."""

from __future__ import annotations

from kortny.agent.capabilities import (
    TOOLKIT_APP_DESCRIPTIONS,
    CapabilityOverview,
    ConnectedToolkitSummary,
    render_connected_integrations,
)


def _make_overview(
    summaries: tuple[ConnectedToolkitSummary, ...],
) -> CapabilityOverview:
    return CapabilityOverview(
        native_categories=(),
        disabled_native=(),
        composio_toolkits=tuple(s.toolkit_slug for s in summaries),
        mcp_servers=(),
        connected_toolkits=summaries,
    )


def test_render_connected_integrations_basic() -> None:
    overview = _make_overview(
        (
            ConnectedToolkitSummary(
                toolkit_slug="alpha_vantage",
                app_description="alpha_vantage",
                tool_names=("GLOBAL_QUOTE", "TIME_SERIES_DAILY", "COMPANY_OVERVIEW"),
            ),
            ConnectedToolkitSummary(
                toolkit_slug="linear",
                app_description="linear",
                tool_names=("list_issues", "get_issue", "create_issue"),
            ),
        )
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    assert "<connected_integrations>" in rendered
    assert "</connected_integrations>" in rendered
    # Known toolkit description is injected from TOOLKIT_APP_DESCRIPTIONS.
    assert "Alpha Vantage" in rendered
    assert "Linear" in rendered
    # Tool names are present.
    assert "GLOBAL_QUOTE" in rendered
    assert "list_issues" in rendered


def test_render_connected_integrations_no_toolkits() -> None:
    overview = _make_overview(())
    result = render_connected_integrations(overview)
    assert result is None


def test_render_connected_integrations_empty_connected_toolkits() -> None:
    overview = CapabilityOverview(
        native_categories=("memory",),
        disabled_native=(),
        composio_toolkits=(),
        mcp_servers=(),
        connected_toolkits=(),
    )
    result = render_connected_integrations(overview)
    assert result is None


def test_app_description_map_known() -> None:
    desc = TOOLKIT_APP_DESCRIPTIONS["alpha_vantage"]
    # Should mention financial/stock data.
    assert (
        "stock" in desc.lower()
        or "financial" in desc.lower()
        or "market" in desc.lower()
    )


def test_app_description_map_fallback() -> None:
    assert "my_custom_app" not in TOOLKIT_APP_DESCRIPTIONS
    overview = _make_overview(
        (
            ConnectedToolkitSummary(
                toolkit_slug="my_custom_app",
                app_description="my_custom_app",
                tool_names=("do_thing",),
            ),
        )
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    # Humanized name — slug underscores become spaces, each word capitalized.
    assert "My Custom App" in rendered
    # Generic description.
    assert "Composio" in rendered or "integration" in rendered.lower()


def test_per_app_tool_cap() -> None:
    tool_names = tuple(f"TOOL_{i}" for i in range(50))
    overview = _make_overview(
        (
            ConnectedToolkitSummary(
                toolkit_slug="alpha_vantage",
                app_description="alpha_vantage",
                tool_names=tool_names,
            ),
        )
    )
    rendered = render_connected_integrations(overview, per_app_tool_cap=10)
    assert rendered is not None
    # Should contain the "...and 40 more." trailer.
    assert "...and 40 more." in rendered
    # Only the first 10 tool names should appear.
    for i in range(10):
        assert f"TOOL_{i}" in rendered
    # Tools beyond the cap should NOT appear as individual names.
    for i in range(10, 50):
        assert f"TOOL_{i}" not in rendered


def test_char_budget_truncation_omission() -> None:
    # Build a very large connected set exceeding max_chars; the algorithm
    # truncates tool lists until the block fits (or gives up if the minimum
    # size with just one tool already exceeds the budget — that is a code
    # boundary condition, not a test requirement).  Use a budget large enough
    # to exceed when all 100 tools are listed but small enough to force the
    # truncation loop to run several iterations.
    tool_names = tuple(f"VERY_LONG_TOOL_NAME_{i}" for i in range(100))
    overview = _make_overview(
        (
            ConnectedToolkitSummary(
                toolkit_slug="alpha_vantage",
                app_description="alpha_vantage",
                tool_names=tool_names,
            ),
        )
    )
    # Full render with 100 tools would be much larger than 700 chars.
    max_chars = 700
    rendered = render_connected_integrations(overview, max_chars=max_chars)
    assert rendered is not None
    assert len(rendered) <= max_chars
    # The result should contain a "...and N more." truncation trailer.
    assert "...and " in rendered and " more." in rendered


def test_render_connected_integrations_tools_without_names() -> None:
    """Toolkit with no tool names still renders the app description."""
    overview = _make_overview(
        (
            ConnectedToolkitSummary(
                toolkit_slug="notion",
                app_description="notion",
                tool_names=(),
            ),
        )
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    assert "Notion" in rendered
    assert "<connected_integrations>" in rendered
