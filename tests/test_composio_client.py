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
