"""Small REST client for Composio catalog APIs.

The first Composio slice uses REST directly so the self-hosted setup does not
need provider-specific SDK dependencies before runtime tool execution exists.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


class ComposioCatalogError(RuntimeError):
    """Raised when the Composio catalog cannot be fetched."""


class ComposioRateLimitError(ComposioCatalogError):
    """Raised when Composio returns HTTP 429 (rate limited).

    A subclass of :class:`ComposioCatalogError` so existing ``except
    ComposioCatalogError`` blocks keep treating it as a catalog failure, while
    the catalog sync can catch it specifically to back off and retry.
    """


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


@dataclass(frozen=True)
class ComposioTool:
    slug: str
    name: str
    description: str
    toolkit_slug: str
    input_parameters: dict[str, Any]
    tags: tuple[str, ...]
    version: str | None


@dataclass(frozen=True)
class ComposioToolExecution:
    data: dict[str, Any] | list[Any] | str | int | float | bool | None
    successful: bool
    error: Any | None
    log_id: str | None
    session_info: dict[str, Any] | None


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

    def list_tools(
        self,
        *,
        toolkit_slug: str | None = None,
        tool_slugs: tuple[str, ...] = (),
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ComposioTool, ...]:
        params: dict[str, str | int] = {"limit": limit}
        if toolkit_slug:
            params["toolkit_slug"] = toolkit_slug
        if tool_slugs:
            params["tool_slugs"] = ",".join(tool_slugs)
        if query:
            params["query"] = query

        response = self._get("/api/v3.1/tools", params=params)
        payload = response.json()
        items: Any
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = (
                payload.get("items")
                or payload.get("tools")
                or payload.get("data")
                or ()
            )
            if isinstance(items, dict):
                items = (
                    items.get("items")
                    or items.get("tools")
                    or items.get("schemas")
                    or ()
                )
        else:
            items = ()
        return tuple(
            _tool_from_payload(item) for item in items if isinstance(item, dict)
        )

    def list_tools_page(
        self,
        *,
        toolkit_slug: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[tuple[ComposioTool, ...], str | None]:
        """Return one page of a toolkit's full tool list plus the next cursor.

        Unlike :meth:`list_tools`, this is query-free (the full catalog, not a
        relevance-pruned slice) and surfaces the pagination cursor so the
        catalog sync can walk every tool. ``next_cursor`` is ``None`` on the
        last page.
        """

        params: dict[str, str | int] = {
            "toolkit_slug": toolkit_slug,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor

        response = self._get("/api/v3.1/tools", params=params)
        payload = response.json()
        next_cursor: str | None = None
        items: Any
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            next_cursor = _optional_str(
                payload.get("next_cursor") or payload.get("nextCursor")
            )
            items = (
                payload.get("items")
                or payload.get("tools")
                or payload.get("data")
                or ()
            )
            if isinstance(items, dict):
                next_cursor = next_cursor or _optional_str(
                    items.get("next_cursor") or items.get("nextCursor")
                )
                items = (
                    items.get("items")
                    or items.get("tools")
                    or items.get("schemas")
                    or ()
                )
        else:
            items = ()
        tools = tuple(
            _tool_from_payload(item) for item in items if isinstance(item, dict)
        )
        return tools, next_cursor

    def list_auth_configs(
        self,
        *,
        toolkit_slug: str,
        limit: int = 20,
    ) -> tuple[ComposioAuthConfig, ...]:
        response = self._get(
            "/api/v3.1/auth_configs",
            params={"toolkit_slug": toolkit_slug, "limit": limit},
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

    def create_custom_auth_config(
        self,
        *,
        toolkit_slug: str,
        auth_scheme: str,
    ) -> ComposioAuthConfig:
        response = self._post(
            "/api/v3.1/auth_configs",
            json_payload={
                "toolkit": {"slug": toolkit_slug},
                "auth_config": {
                    "type": "use_custom_auth",
                    "authScheme": auth_scheme,
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
        normalized_payload.setdefault("name", f"{toolkit_slug} {auth_scheme} auth")
        normalized_payload.setdefault("auth_scheme", auth_scheme)
        normalized_payload.setdefault("is_composio_managed", False)
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

    def create_connected_account(
        self,
        *,
        user_id: str,
        auth_config_id: str,
    ) -> ComposioConnectionRequest:
        """Create a connected account directly, without an OAuth redirect.

        Used for NO_AUTH toolkits: they carry no credentials and have no redirect
        flow, so the connected account becomes active immediately. Returns a
        ``ComposioConnectionRequest`` with an empty ``redirect_url`` and the new
        connected-account id. (Composio v3.1 ``POST /connected_accounts`` — the
        exact no-auth ``connection`` body is undocumented publicly; user_id-only
        is the minimal form and is verified live.)
        """

        response = self._post(
            "/api/v3.1/connected_accounts",
            json_payload={
                "auth_config": {"id": auth_config_id},
                "connection": {"user_id": user_id},
            },
        )
        payload = response.json()
        account = payload.get("connected_account") or payload.get("connectedAccount")
        account_dict = account if isinstance(account, dict) else payload
        connected_account_id = _optional_str(
            account_dict.get("id")
            or account_dict.get("connected_account_id")
            or account_dict.get("connectedAccountId")
        )
        if not connected_account_id:
            raise ComposioConnectionError(
                "Composio connected-account response had no id"
            )
        status_value = str(
            account_dict.get("status") or payload.get("status") or "active"
        ).lower()
        return ComposioConnectionRequest(
            id=connected_account_id,
            redirect_url="",
            status=status_value,
            connected_account_id=connected_account_id,
        )

    def set_connected_account_enabled(
        self,
        connected_account_id: str,
        *,
        enabled: bool,
    ) -> bool:
        response = self._patch(
            f"/api/v3.1/connected_accounts/{connected_account_id}/status",
            json_payload={"enabled": enabled},
        )
        payload = response.json()
        success = payload.get("success")
        return bool(success) if success is not None else True

    def execute_tool(
        self,
        *,
        tool_slug: str,
        user_id: str,
        connected_account_id: str,
        arguments: dict[str, Any],
        version: str | None = None,
    ) -> ComposioToolExecution:
        payload: dict[str, Any] = {
            "user_id": user_id,
            "connected_account_id": connected_account_id,
            "arguments": arguments,
        }
        if version:
            payload["version"] = version

        response = self._post(
            f"/api/v3.1/tools/execute/{tool_slug}",
            json_payload=payload,
        )
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ComposioConnectionError("Composio tool response was invalid")
        session_info = response_payload.get("session_info")
        return ComposioToolExecution(
            data=response_payload.get("data"),
            successful=bool(response_payload.get("successful", True)),
            error=response_payload.get("error"),
            log_id=_optional_str(response_payload.get("log_id")),
            session_info=session_info if isinstance(session_info, dict) else None,
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
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise ComposioRateLimitError(_http_error_summary(exc)) from exc
            raise ComposioCatalogError(_http_error_summary(exc)) from exc
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
        except httpx.HTTPStatusError as exc:
            raise ComposioConnectionError(_http_error_summary(exc)) from exc
        except httpx.HTTPError as exc:
            raise ComposioConnectionError(str(exc)) from exc
        finally:
            if close_client:
                client.close()

    def _patch(self, path: str, *, json_payload: dict[str, Any]) -> httpx.Response:
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.http_client is None
        try:
            response = client.patch(
                f"{self.base_url}{path}",
                headers={"x-api-key": self.api_key},
                json=json_payload,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise ComposioConnectionError(_http_error_summary(exc)) from exc
        except httpx.HTTPError as exc:
            raise ComposioConnectionError(str(exc)) from exc
        finally:
            if close_client:
                client.close()

    def _delete(self, path: str) -> httpx.Response:
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.http_client is None
        try:
            response = client.delete(
                f"{self.base_url}{path}",
                headers={"x-api-key": self.api_key},
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise ComposioConnectionError(_http_error_summary(exc)) from exc
        except httpx.HTTPError as exc:
            raise ComposioConnectionError(str(exc)) from exc
        finally:
            if close_client:
                client.close()

    def create_trigger(
        self,
        slug: str,
        user_id: str,
        trigger_config: dict[str, Any],
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a Composio trigger subscription for a user."""
        payload: dict[str, Any] = {
            "trigger_slug": slug,
            "user_id": user_id,
            "trigger_config": trigger_config,
        }
        if connected_account_id is not None:
            payload["connected_account_id"] = connected_account_id
        response = self._post("/api/v3.1/triggers", json_payload=payload)
        result: dict[str, Any] = response.json()
        return result

    def list_triggers(
        self,
        toolkit_slugs: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """List available trigger definitions, optionally filtered by toolkit."""
        params: dict[str, str | int] = {}
        if toolkit_slugs:
            params["toolkit_slugs"] = ",".join(toolkit_slugs)
        response = self._get("/api/v3.1/triggers", params=params)
        payload = response.json()
        items: Any = payload.get("items") or payload.get("triggers") or payload
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def list_active_triggers(
        self,
        trigger_ids: Sequence[str] = (),
        connected_account_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """List active trigger instances, optionally filtered by id or account."""
        params: dict[str, str | int] = {}
        if trigger_ids:
            params["trigger_ids"] = ",".join(trigger_ids)
        if connected_account_ids:
            params["connected_account_ids"] = ",".join(connected_account_ids)
        response = self._get("/api/v3.1/triggers/active", params=params)
        payload = response.json()
        items: Any = payload.get("items") or payload.get("triggers") or payload
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def disable_trigger(self, trigger_id: str) -> dict[str, Any]:
        """Disable an active trigger instance by its ti_* id."""
        response = self._patch(
            f"/api/v3.1/triggers/{trigger_id}/disable",
            json_payload={},
        )
        result: dict[str, Any] = response.json()
        return result

    def delete_trigger(self, trigger_id: str) -> None:
        """Delete a trigger instance by its ti_* id."""
        self._delete(f"/api/v3.1/triggers/{trigger_id}")


def _toolkit_from_payload(payload: dict[str, Any]) -> ComposioToolkit:
    raw_meta = payload.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    return ComposioToolkit(
        slug=str(payload.get("slug") or payload.get("id") or ""),
        name=str(payload.get("name") or payload.get("slug") or "Unknown toolkit"),
        description=str(meta.get("description") or payload.get("description") or ""),
        categories=_category_names(meta.get("categories")),
        auth_schemes=_auth_schemes_from_payload(payload),
        managed_auth_schemes=_string_tuple(
            payload.get("composio_managed_auth_schemes")
        ),
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


def _tool_from_payload(payload: dict[str, Any]) -> ComposioTool:
    toolkit = payload.get("toolkit")
    if isinstance(toolkit, dict):
        toolkit_slug = _optional_str(toolkit.get("slug") or toolkit.get("name"))
    else:
        toolkit_slug = _optional_str(payload.get("toolkit_slug") or toolkit)
    parameters = (
        payload.get("input_parameters")
        or payload.get("inputParameters")
        or payload.get("input_schema")
        or payload.get("schema")
        or {}
    )
    if not isinstance(parameters, dict):
        parameters = {}
    slug = _optional_str(payload.get("slug") or payload.get("name"))
    if slug is None:
        raise ComposioCatalogError("Composio tool payload is missing a slug")
    return ComposioTool(
        slug=slug.upper(),
        name=str(payload.get("name") or slug),
        description=str(payload.get("description") or ""),
        toolkit_slug=(toolkit_slug or "").lower(),
        input_parameters=parameters,
        tags=_tag_names(payload.get("tags")),
        version=_optional_str(payload.get("version")),
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
        auth_scheme=_optional_str(
            payload.get("auth_scheme") or payload.get("authScheme")
        ),
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


def _tag_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = _optional_str(item.get("name") or item.get("slug") or item.get("id"))
        else:
            name = _optional_str(item)
        if name:
            names.append(name)
    return tuple(dict.fromkeys(names))


def _auth_schemes_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    schemes = list(_string_tuple(payload.get("auth_schemes")))
    details = payload.get("auth_config_details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            mode = _optional_str(item.get("mode"))
            if mode:
                schemes.append(mode)
    return tuple(dict.fromkeys(schemes))


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


def _http_error_summary(exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    detail = _response_error_detail(response)
    if detail:
        return f"{exc}; Composio response: {detail}"
    return str(exc)


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500]

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts = [
                _optional_str(error.get("message")),
                _optional_str(error.get("suggested_fix")),
            ]
            errors = error.get("errors")
            if isinstance(errors, list):
                parts.extend(_optional_str(item) for item in errors)
            return "; ".join(part for part in parts if part)[:500]
        message = _optional_str(payload.get("message"))
        if message:
            return message[:500]
    return str(payload)[:500]


# ---------------------------------------------------------------------------
# Webhook verification and parsing
# ---------------------------------------------------------------------------


class TriggerWebhookError(Exception):
    """Base for all trigger webhook verification/parse errors."""


class TriggerSignatureError(TriggerWebhookError):
    """HMAC signature on the incoming webhook did not match."""


class TriggerTimestampError(TriggerWebhookError):
    """Webhook timestamp is outside the allowed replay-tolerance window."""


class TriggerParseError(TriggerWebhookError):
    """Webhook body could not be parsed as a valid Composio trigger envelope."""


@dataclass(frozen=True)
class ParsedTriggerEvent:
    """Decoded, verified Composio trigger webhook envelope."""

    id: str
    type: str
    trigger_slug: str
    trigger_id: str | None
    connected_account_id: str | None
    user_id: str | None
    data: dict[str, Any]
    timestamp: str  # ISO string from the envelope


def verify_and_parse_trigger_webhook(
    *,
    raw_body: bytes,
    headers: Mapping[str, str],
    secret: str,
    tolerance_seconds: int = 300,
) -> ParsedTriggerEvent:
    """Verify Composio webhook HMAC-SHA256 signature and parse the V3 envelope.

    HMAC mechanics:
    - The signed string is: ``{webhook-id}.{webhook-timestamp}.{body_as_str}``
    - The key is the raw shared-secret string encoded as UTF-8 (NOT base64-decoded).
    - The expected signature is HMAC-SHA256 over that string, base64-encoded.
    - The ``webhook-signature`` header is formatted ``v1,<base64sig>``; multiple
      comma-separated signatures may appear (any valid one passes).
    - Timestamps outside ``tolerance_seconds`` of now are rejected.

    Raises:
        TriggerSignatureError: signature does not match.
        TriggerTimestampError: timestamp skew exceeds tolerance.
        TriggerParseError: body is not valid JSON or missing required fields.
    """
    # Normalise header lookup to lowercase.
    lower_headers: dict[str, str] = {k.lower(): v for k, v in headers.items()}

    webhook_id = lower_headers.get("webhook-id", "")
    webhook_timestamp = lower_headers.get("webhook-timestamp", "")
    sig_header = lower_headers.get("webhook-signature", "")

    # --- Timestamp replay check ------------------------------------------------
    try:
        ts_int = int(webhook_timestamp)
    except ValueError as exc:
        raise TriggerTimestampError(
            f"webhook-timestamp is not a valid integer: {webhook_timestamp!r}"
        ) from exc
    now = int(time.time())
    if abs(now - ts_int) > tolerance_seconds:
        raise TriggerTimestampError(
            f"webhook-timestamp {ts_int} is {abs(now - ts_int)}s from now "
            f"(tolerance {tolerance_seconds}s)"
        )

    # --- HMAC verification ----------------------------------------------------
    # Signed string: "{webhook-id}.{webhook-timestamp}.{body_as_str}"
    body_str = raw_body.decode("utf-8", errors="replace")
    signed_string = f"{webhook_id}.{webhook_timestamp}.{body_str}"
    key = secret.encode("utf-8")
    expected_mac = hmac.new(key, signed_string.encode("utf-8"), hashlib.sha256).digest()
    expected_b64 = base64.b64encode(expected_mac).decode("ascii")

    # webhook-signature may contain multiple "v1,<sig>" entries separated by spaces
    accepted_sigs: list[str] = []
    for part in sig_header.split():
        if part.startswith("v1,"):
            accepted_sigs.append(part[3:])
        else:
            accepted_sigs.append(part)

    if not any(
        hmac.compare_digest(expected_b64, candidate) for candidate in accepted_sigs
    ):
        raise TriggerSignatureError("webhook-signature did not match HMAC-SHA256")

    # --- Parse envelope -------------------------------------------------------
    try:
        envelope: Any = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise TriggerParseError(f"webhook body is not valid JSON: {exc}") from exc

    if not isinstance(envelope, dict):
        raise TriggerParseError("webhook envelope must be a JSON object")

    try:
        event_id = str(envelope["id"])
        event_type = str(envelope.get("type", "composio.trigger.message"))
        metadata: dict[str, Any] = envelope.get("metadata") or {}
        data: dict[str, Any] = envelope.get("data") or {}
        trigger_slug = str(metadata.get("trigger_slug") or "")
        trigger_id = _optional_str(metadata.get("trigger_id"))
        connected_account_id = _optional_str(metadata.get("connected_account_id"))
        user_id = _optional_str(metadata.get("user_id"))
        timestamp = str(envelope.get("timestamp") or "")
    except (KeyError, TypeError) as exc:
        raise TriggerParseError(
            f"webhook envelope is missing required fields: {exc}"
        ) from exc

    if not trigger_slug:
        raise TriggerParseError("webhook envelope metadata.trigger_slug is empty")

    return ParsedTriggerEvent(
        id=event_id,
        type=event_type,
        trigger_slug=trigger_slug,
        trigger_id=trigger_id,
        connected_account_id=connected_account_id,
        user_id=user_id,
        data=data,
        timestamp=timestamp,
    )
