"""Small REST client for Composio catalog APIs.

The first Composio slice uses REST directly so the self-hosted setup does not
need provider-specific SDK dependencies before runtime tool execution exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ComposioCatalogError(RuntimeError):
    """Raised when the Composio catalog cannot be fetched."""


class ComposioConnectionError(RuntimeError):
    """Raised when a Composio connection action fails."""


@dataclass(frozen=True)
class ComposioAuthConfig:
    id: str
    name: str
    toolkit_slug: str
    auth_scheme: str | None
    is_composio_managed: bool
    enabled: bool


@dataclass(frozen=True)
class ComposioConnectionRequest:
    id: str
    redirect_url: str
    status: str
    connected_account_id: str | None = None


@dataclass(frozen=True)
class ComposioToolkit:
    slug: str
    name: str
    description: str
    categories: tuple[str, ...]
    auth_schemes: tuple[str, ...]
    managed_auth_schemes: tuple[str, ...]
    tools_count: int
    triggers_count: int
    logo_url: str | None
    app_url: str | None
    auth_guide_url: str | None
    base_url: str | None
    enabled: bool
    no_auth: bool
    is_local_toolkit: bool


@dataclass(frozen=True)
class ComposioCatalog:
    items: tuple[ComposioToolkit, ...]
    total_items: int | None
    next_cursor: str | None


class ComposioClient:
    """Minimal Composio API wrapper used by the dashboard and future adapters."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://backend.composio.dev",
        timeout_seconds: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    def list_toolkits(
        self,
        *,
        search: str | None = None,
        category: str | None = None,
        limit: int = 60,
        cursor: str | None = None,
        sort_by: str = "usage",
    ) -> ComposioCatalog:
        params: dict[str, str | int] = {
            "limit": limit,
            "sort_by": sort_by,
        }
        if search:
            params["search"] = search
        if category:
            params["category"] = category
        if cursor:
            params["cursor"] = cursor

        response = self._get("/api/v3.1/toolkits", params=params)
        payload = response.json()
        return ComposioCatalog(
            items=tuple(
                _toolkit_from_payload(item) for item in payload.get("items", ())
            ),
            total_items=_optional_int(payload.get("total_items")),
            next_cursor=_optional_str(payload.get("next_cursor")),
        )

    def get_toolkit(self, slug: str) -> ComposioToolkit:
        response = self._get(f"/api/v3.1/toolkits/{slug}", params={})
        payload = response.json()
        if isinstance(payload.get("toolkit"), dict):
            payload = payload["toolkit"]
        return _toolkit_from_payload(payload)

    def list_auth_configs(
        self,
        *,
        toolkit_slug: str,
        limit: int = 20,
    ) -> tuple[ComposioAuthConfig, ...]:
        response = self._get(
            "/api/v3.1/auth_configs",
            params={"toolkit": toolkit_slug, "limit": limit},
        )
        payload = response.json()
        items = payload.get("items") or payload.get("data") or ()
        return tuple(_auth_config_from_payload(item) for item in items)

    def create_managed_auth_config(self, *, toolkit_slug: str) -> ComposioAuthConfig:
        response = self._post(
            "/api/v3.1/auth_configs",
            json_payload={
                "toolkit": {"slug": toolkit_slug},
                "auth_config": {
                    "type": "use_composio_managed_auth",
                    "credentials": {},
                    "restrict_to_following_tools": [],
                },
            },
        )
        payload = response.json()
        auth_config = payload.get("auth_config") or payload.get("data") or payload
        if not isinstance(auth_config, dict):
            raise ComposioConnectionError("Composio auth config response was invalid")
        normalized_payload = dict(auth_config)
        normalized_payload.setdefault("toolkit", toolkit_slug)
        normalized_payload.setdefault("name", f"{toolkit_slug} managed auth")
        normalized_payload.setdefault("is_composio_managed", True)
        return _auth_config_from_payload(normalized_payload)

    def create_connect_link(
        self,
        *,
        user_id: str,
        auth_config_id: str,
        callback_url: str,
    ) -> ComposioConnectionRequest:
        response = self._post(
            "/api/v3.1/connected_accounts/link",
            json_payload={
                "user_id": user_id,
                "auth_config_id": auth_config_id,
                "callback_url": callback_url,
            },
        )
        payload = response.json()
        redirect_url = _optional_str(
            payload.get("redirect_url") or payload.get("redirectUrl")
        )
        connected_account_id = _optional_str(
            payload.get("connected_account_id") or payload.get("connectedAccountId")
        )
        request_id = _optional_str(
            payload.get("id")
            or payload.get("link_token")
            or payload.get("connection_id")
            or connected_account_id
        )
        if not redirect_url or not request_id:
            raise ComposioConnectionError("Composio connect-link response was invalid")
        return ComposioConnectionRequest(
            id=request_id,
            redirect_url=redirect_url,
            status=str(payload.get("status") or "pending").lower(),
            connected_account_id=connected_account_id,
        )

    def _get(self, path: str, *, params: dict[str, str | int]) -> httpx.Response:
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.http_client is None
        try:
            response = client.get(
                f"{self.base_url}{path}",
                headers={"x-api-key": self.api_key},
                params=params,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise ComposioCatalogError(str(exc)) from exc
        finally:
            if close_client:
                client.close()

    def _post(self, path: str, *, json_payload: dict[str, Any]) -> httpx.Response:
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.http_client is None
        try:
            response = client.post(
                f"{self.base_url}{path}",
                headers={"x-api-key": self.api_key},
                json=json_payload,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise ComposioConnectionError(str(exc)) from exc
        finally:
            if close_client:
                client.close()


def _toolkit_from_payload(payload: dict[str, Any]) -> ComposioToolkit:
    raw_meta = payload.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    return ComposioToolkit(
        slug=str(payload.get("slug") or payload.get("id") or ""),
        name=str(payload.get("name") or payload.get("slug") or "Unknown toolkit"),
        description=str(meta.get("description") or payload.get("description") or ""),
        categories=_category_names(meta.get("categories")),
        auth_schemes=_string_tuple(payload.get("auth_schemes")),
        managed_auth_schemes=_string_tuple(payload.get("composio_managed_auth_schemes")),
        tools_count=_optional_int(meta.get("tools_count")) or 0,
        triggers_count=_optional_int(meta.get("triggers_count")) or 0,
        logo_url=_optional_str(meta.get("logo")),
        app_url=_optional_str(meta.get("app_url")),
        auth_guide_url=_optional_str(payload.get("auth_guide_url")),
        base_url=_optional_str(payload.get("base_url")),
        enabled=bool(payload.get("enabled", True)),
        no_auth=bool(payload.get("no_auth")),
        is_local_toolkit=bool(payload.get("is_local_toolkit")),
    )


def _auth_config_from_payload(payload: dict[str, Any]) -> ComposioAuthConfig:
    config_id = _optional_str(
        payload.get("id") or payload.get("nanoid") or payload.get("auth_config_id")
    )
    if not config_id:
        raise ComposioConnectionError("Composio auth config is missing an id")
    toolkit = payload.get("toolkit")
    if isinstance(toolkit, dict):
        toolkit_slug = _optional_str(toolkit.get("slug"))
    else:
        toolkit_slug = _optional_str(toolkit)
    return ComposioAuthConfig(
        id=config_id,
        name=str(payload.get("name") or config_id),
        toolkit_slug=toolkit_slug or "",
        auth_scheme=_optional_str(payload.get("auth_scheme") or payload.get("authScheme")),
        is_composio_managed=bool(
            payload.get("is_composio_managed")
            if "is_composio_managed" in payload
            else payload.get("isComposioManaged")
        )
        or payload.get("type") == "use_composio_managed_auth",
        enabled=bool(payload.get("enabled", True)),
    )


def _category_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = _optional_str(item.get("name") or item.get("id"))
        else:
            name = _optional_str(item)
        if name:
            names.append(name)
    return tuple(names)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
