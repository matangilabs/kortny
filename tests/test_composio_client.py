import json

import httpx

from kortny.composio import ComposioClient


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
