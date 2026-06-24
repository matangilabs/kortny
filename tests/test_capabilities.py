"""Tests for the connected_integrations capability rendering (HIG-291).

Pure: no DB, no live API calls.
"""

from __future__ import annotations

from kortny.agent.capabilities import (
    TOOLKIT_TOOL_DISPLAY_CAP,
    CapabilityOverview,
    ConnectedToolkitSummary,
    render_capability_overview,
)


def _minimal_overview(
    connected_toolkits: tuple[ConnectedToolkitSummary, ...] = (),
) -> CapabilityOverview:
    return CapabilityOverview(
        native_categories=(),
        disabled_native=(),
        composio_toolkits=(),
        mcp_servers=(),
        connected_toolkits=connected_toolkits,
    )


def test_connected_integrations_block_rendered() -> None:
    av = ConnectedToolkitSummary(
        toolkit_slug="alpha_vantage",
        app_name="Alpha Vantage",
        app_description="stock data",
        tool_names=["GLOBAL_QUOTE", "TIME_SERIES_DAILY", "OVERVIEW"],
        total_tool_count=3,
    )
    alpaca = ConnectedToolkitSummary(
        toolkit_slug="alpaca",
        app_name="Alpaca",
        app_description="brokerage",
        tool_names=["get_snapshots_for_multiple_stock_symbols", "get_stock_bars"],
        total_tool_count=2,
    )
    td = ConnectedToolkitSummary(
        toolkit_slug="twelve_data",
        app_name="Twelve Data",
        app_description="market data",
        tool_names=["price", "eod", "time_series"],
        total_tool_count=3,
    )
    overview = _minimal_overview(connected_toolkits=(av, alpaca, td))
    rendered = render_capability_overview(overview)

    assert "<connected_integrations>" in rendered
    assert "Alpha Vantage" in rendered
    assert "Alpaca" in rendered
    assert "Twelve Data" in rendered
    assert "GLOBAL_QUOTE" in rendered
    assert "get_snapshots_for_multiple_stock_symbols" in rendered
    assert "price" in rendered
    assert "directly by name" in rendered
    assert "</connected_integrations>" in rendered


def test_giant_app_tool_cap() -> None:
    all_tools = [f"TOOL_{i:03d}" for i in range(55)]
    summary = ConnectedToolkitSummary(
        toolkit_slug="alpha_vantage",
        app_name="Alpha Vantage",
        app_description="stock data",
        tool_names=all_tools,
        total_tool_count=55,
    )
    overview = _minimal_overview(connected_toolkits=(summary,))
    rendered = render_capability_overview(overview)

    remainder = 55 - TOOLKIT_TOOL_DISPLAY_CAP  # 15
    assert f"and {remainder} more (use find_tools)" in rendered
    # First 40 tools should appear, not the 41st
    assert "TOOL_000" in rendered
    assert f"TOOL_{TOOLKIT_TOOL_DISPLAY_CAP - 1:03d}" in rendered
    assert f"TOOL_{TOOLKIT_TOOL_DISPLAY_CAP:03d}" not in rendered


def test_no_connections_no_block() -> None:
    overview = _minimal_overview()
    rendered = render_capability_overview(overview)

    assert "<connected_integrations>" not in rendered
    assert "</connected_integrations>" not in rendered


def test_coordinator_direct_call_wording() -> None:
    from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT

    assert "directly by name" in DEFAULT_SYSTEM_PROMPT
    assert "MUST call find_tools" not in DEFAULT_SYSTEM_PROMPT
