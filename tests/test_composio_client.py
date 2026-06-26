import json

import httpx
import pytest

from kortny.composio import ComposioClient, ComposioConnectionError


def test_composio_client_lists_toolkits_from_catalog_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/toolkits"
        assert request.headers["x-api-key"] == "test-key"
        assert request.url.params["search"] == "github"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "slug": "github",
                        "name": "GitHub",
                        "auth_schemes": ["oauth2", "api_key"],
                        "composio_managed_auth_schemes": ["oauth2"],
                        "no_auth": False,
                        "is_local_toolkit": False,
                        "meta": {
                            "description": "Manage GitHub repositories.",
                            "logo": "https://assets.example/github.png",
                            "app_url": "https://github.com",
                            "tools_count": 12,
                            "triggers_count": 5,
                            "categories": [
                                {"id": "developer-tools", "name": "Developer Tools"}
                            ],
                        },
                    }
                ],
                "total_items": 1043,
                "next_cursor": "cursor-1",
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    catalog = client.list_toolkits(search="github", limit=10)

    assert catalog.total_items == 1043
    assert catalog.next_cursor == "cursor-1"
    assert len(catalog.items) == 1
    toolkit = catalog.items[0]
    assert toolkit.slug == "github"
    assert toolkit.name == "GitHub"
    assert toolkit.description == "Manage GitHub repositories."
    assert toolkit.categories == ("Developer Tools",)
    assert toolkit.auth_schemes == ("oauth2", "api_key")
    assert toolkit.managed_auth_schemes == ("oauth2",)
    assert toolkit.tools_count == 12
    assert toolkit.triggers_count == 5


def test_composio_client_gets_toolkit_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/toolkits/github"
        return httpx.Response(
            200,
            json={
                "slug": "github",
                "name": "GitHub",
                "enabled": True,
                "auth_guide_url": "https://composio.dev/auth/github",
                "base_url": "https://api.github.com",
                "auth_schemes": ["oauth2"],
                "composio_managed_auth_schemes": ["oauth2"],
                "meta": {
                    "description": "Manage GitHub repositories.",
                    "tools_count": 12,
                    "triggers_count": 5,
                    "categories": [{"name": "Developer Tools"}],
                },
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    toolkit = client.get_toolkit("github")

    assert toolkit.slug == "github"
    assert toolkit.enabled is True
    assert toolkit.auth_guide_url == "https://composio.dev/auth/github"
    assert toolkit.base_url == "https://api.github.com"


def test_composio_client_derives_auth_scheme_from_auth_config_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/toolkits/firecrawl"
        return httpx.Response(
            200,
            json={
                "slug": "firecrawl",
                "name": "Firecrawl",
                "composio_managed_auth_schemes": [],
                "auth_config_details": [
                    {
                        "name": "firecrawl_api_key",
                        "mode": "API_KEY",
                        "fields": {
                            "auth_config_creation": {
                                "required": [],
                                "optional": [],
                            },
                            "connected_account_initiation": {
                                "required": [
                                    {"name": "full", "displayName": "Base URL"},
                                    {
                                        "name": "generic_api_key",
                                        "displayName": "API Key",
                                    },
                                ],
                                "optional": [],
                            },
                        },
                    }
                ],
                "meta": {"description": "Crawl websites.", "tools_count": 29},
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    toolkit = client.get_toolkit("firecrawl")

    assert toolkit.auth_schemes == ("API_KEY",)
    assert toolkit.managed_auth_schemes == ()


def test_composio_client_lists_tools_from_dynamic_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/tools"
        assert request.url.params["toolkit_slug"] == "firecrawl"
        assert request.url.params["query"] == "recent AI tooling"
        assert request.url.params["limit"] == "8"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "slug": "FIRECRAWL_SEARCH",
                        "name": "Search",
                        "description": "Search the public web.",
                        "toolkit": {"slug": "firecrawl"},
                        "input_parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                        "tags": ["readOnlyHint"],
                        "version": "latest",
                    }
                ]
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    tools = client.list_tools(
        toolkit_slug="firecrawl",
        query="recent AI tooling",
        limit=8,
    )

    assert len(tools) == 1
    assert tools[0].slug == "FIRECRAWL_SEARCH"
    assert tools[0].toolkit_slug == "firecrawl"
    assert tools[0].input_parameters["properties"]["q"]["type"] == "string"
    assert tools[0].tags == ("readOnlyHint",)


def test_composio_client_lists_auth_configs_for_toolkit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/auth_configs"
        assert request.url.params["toolkit_slug"] == "github"
        assert request.url.params["limit"] == "20"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "ac_123",
                        "name": "GitHub OAuth",
                        "toolkit": {"slug": "github"},
                        "auth_scheme": "OAUTH2",
                        "is_composio_managed": True,
                        "enabled": True,
                    }
                ]
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    auth_configs = client.list_auth_configs(toolkit_slug="github")

    assert len(auth_configs) == 1
    assert auth_configs[0].id == "ac_123"
    assert auth_configs[0].name == "GitHub OAuth"
    assert auth_configs[0].toolkit_slug == "github"
    assert auth_configs[0].auth_scheme == "OAUTH2"
    assert auth_configs[0].is_composio_managed is True
    assert auth_configs[0].enabled is True


def test_composio_client_creates_managed_auth_config() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/auth_configs"
        assert request.method == "POST"
        payload = json.loads(request.read().decode())
        assert payload["toolkit"] == {"slug": "github"}
        assert payload["auth_config"]["type"] == "use_composio_managed_auth"
        return httpx.Response(
            200,
            json={
                "auth_config": {
                    "id": "ac_managed",
                    "toolkit": {"slug": "github"},
                    "auth_scheme": "OAUTH2",
                    "is_composio_managed": True,
                    "enabled": True,
                }
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    auth_config = client.create_managed_auth_config(toolkit_slug="github")

    assert auth_config.id == "ac_managed"
    assert auth_config.toolkit_slug == "github"
    assert auth_config.is_composio_managed is True


def test_composio_client_creates_custom_auth_config() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/auth_configs"
        assert request.method == "POST"
        payload = json.loads(request.read().decode())
        assert payload["toolkit"] == {"slug": "firecrawl"}
        assert payload["auth_config"] == {
            "type": "use_custom_auth",
            "authScheme": "API_KEY",
            "credentials": {},
            "restrict_to_following_tools": [],
        }
        return httpx.Response(
            200,
            json={
                "auth_config": {
                    "id": "ac_firecrawl",
                    "toolkit": {"slug": "firecrawl"},
                    "auth_scheme": "API_KEY",
                    "is_composio_managed": False,
                    "enabled": True,
                }
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    auth_config = client.create_custom_auth_config(
        toolkit_slug="firecrawl",
        auth_scheme="API_KEY",
    )

    assert auth_config.id == "ac_firecrawl"
    assert auth_config.toolkit_slug == "firecrawl"
    assert auth_config.auth_scheme == "API_KEY"
    assert auth_config.is_composio_managed is False


def test_composio_client_includes_composio_error_detail() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Auth scheme is required",
                    "suggested_fix": "Pass auth_scheme for custom auth",
                }
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ComposioConnectionError) as exc_info:
        client.create_custom_auth_config(
            toolkit_slug="firecrawl",
            auth_scheme="API_KEY",
        )

    message = str(exc_info.value)
    assert "Auth scheme is required" in message
    assert "Pass auth_scheme for custom auth" in message


def test_composio_client_creates_connect_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/connected_accounts/link"
        assert request.method == "POST"
        payload = json.loads(request.read().decode())
        assert payload["user_id"] == "slack:installation:user"
        assert payload["auth_config_id"] == "ac_123"
        assert payload["callback_url"] == "https://kortny.example/composio/callback"
        return httpx.Response(
            200,
            json={
                "link_token": "ln_123",
                "redirect_url": "https://connect.composio.dev/auth",
                "connected_account_id": "ca_pending_123",
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    connect_request = client.create_connect_link(
        user_id="slack:installation:user",
        auth_config_id="ac_123",
        callback_url="https://kortny.example/composio/callback",
    )

    assert connect_request.id == "ln_123"
    assert connect_request.redirect_url == "https://connect.composio.dev/auth"
    assert connect_request.status == "pending"
    assert connect_request.connected_account_id == "ca_pending_123"


def test_composio_client_disables_connected_account() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/connected_accounts/ca_123/status"
        assert request.method == "PATCH"
        payload = json.loads(request.read().decode())
        assert payload == {"enabled": False}
        return httpx.Response(200, json={"success": True})

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.set_connected_account_enabled("ca_123", enabled=False) is True


def test_composio_client_executes_tool_with_connected_account() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/tools/execute/FIRECRAWL_SCRAPE"
        assert request.method == "POST"
        payload = json.loads(request.read().decode())
        assert payload == {
            "user_id": "slack:installation:user",
            "connected_account_id": "ca_firecrawl",
            "arguments": {"url": "https://example.com"},
            "version": "latest",
        }
        return httpx.Response(
            200,
            json={
                "data": {"markdown": "# Example"},
                "successful": True,
                "error": None,
                "log_id": "log_123",
                "session_info": {"session_id": "session_123"},
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    execution = client.execute_tool(
        tool_slug="FIRECRAWL_SCRAPE",
        user_id="slack:installation:user",
        connected_account_id="ca_firecrawl",
        arguments={"url": "https://example.com"},
        version="latest",
    )

    assert execution.successful is True
    assert execution.data == {"markdown": "# Example"}
    assert execution.error is None
    assert execution.log_id == "log_123"
    assert execution.session_info == {"session_id": "session_123"}


def test_composio_client_executes_tool_without_connected_account_id() -> None:
    """execute_tool with connected_account_id=None omits the key from the POST body.

    NO_AUTH toolkits (e.g. hackernews) need no connected account; Composio's
    execute endpoint accepts user_id alone and returns real data.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3.1/tools/execute/HACKERNEWS_GET_FRONTPAGE"
        assert request.method == "POST"
        payload = json.loads(request.read().decode())
        # connected_account_id must be absent — not None, not empty string
        assert "connected_account_id" not in payload
        assert payload == {
            "user_id": "slack:installation:user",
            "arguments": {},
        }
        return httpx.Response(
            200,
            json={
                "data": [{"title": "Show HN: something cool"}],
                "successful": True,
                "error": None,
                "log_id": "log_hn",
                "session_info": None,
            },
        )

    client = ComposioClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    execution = client.execute_tool(
        tool_slug="HACKERNEWS_GET_FRONTPAGE",
        user_id="slack:installation:user",
        connected_account_id=None,
        arguments={},
    )

    assert execution.successful is True
    assert execution.data == [{"title": "Show HN: something cool"}]
    assert execution.log_id == "log_hn"
