"""Test capabilities block uses learned profile summary (HIG-295 Part 2)."""

from kortny.agent.capabilities import (
    CapabilityOverview,
    ConnectedToolkitSummary,
    render_connected_integrations,
)


def test_render_connected_integrations_uses_profile_summary() -> None:
    """render_connected_integrations uses learned profile_summary for unknown slugs."""
    summary = ConnectedToolkitSummary(
        toolkit_slug="some_new_app",
        app_description="some_new_app",
        tool_names=(),
        profile_summary="Some New App manages widget inventory and orders.",
        capability_buckets=("inventory management", "order tracking"),
    )
    overview = CapabilityOverview(
        native_categories=(),
        disabled_native=(),
        composio_toolkits=("some_new_app",),
        mcp_servers=(),
        connected_toolkits=(summary,),
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    assert "manages widget inventory" in rendered
    assert "inventory management" in rendered  # capability_buckets


def test_render_connected_integrations_curated_description_wins() -> None:
    """Curated TOOLKIT_APP_DESCRIPTIONS takes priority over profile_summary."""
    summary = ConnectedToolkitSummary(
        toolkit_slug="notion",
        app_description="notion",
        tool_names=(),
        profile_summary="This should not appear — curated wins.",
        capability_buckets=(),
    )
    overview = CapabilityOverview(
        native_categories=(),
        disabled_native=(),
        composio_toolkits=("notion",),
        mcp_servers=(),
        connected_toolkits=(summary,),
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    assert "This should not appear" not in rendered
    assert "workspace pages" in rendered  # from TOOLKIT_APP_DESCRIPTIONS
