"""FastAPI app for the read-only Kortny cost dashboard."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from starlette.middleware.sessions import SessionMiddleware

from kortny.composio import (
    ComposioCatalogError,
    ComposioClient,
    ComposioConnectionError,
)
from kortny.config import Settings, SettingsError, load_settings
from kortny.dashboard.auth import (
    DashboardAuthError,
    DashboardPrincipal,
    SlackOpenIDClient,
    upsert_dashboard_user,
)
from kortny.dashboard.data import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    get_composio_catalog_dashboard,
    get_composio_toolkit_detail,
    get_dashboard_overview,
    get_integration_dashboard,
    get_memory_dashboard,
    get_system_health,
    get_task_detail,
    get_usage_aggregate,
    get_user_detail,
    list_tasks,
    list_users,
    parse_date_bound,
)
from kortny.dashboard.memory_actions import (
    dashboard_actor,
    forget_fact,
    supersede_fact,
)
from kortny.dashboard.settings import DashboardSettings, load_dashboard_settings
from kortny.db.models import (
    ComposioConnection,
    DashboardOAuthState,
    DashboardUser,
    Installation,
    SlackIdentity,
)
from kortny.db.session import make_session_factory

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
SESSION_USER_KEY = "dashboard_user"
SESSION_DASHBOARD_USER_ID_KEY = "dashboard_user_id"
SESSION_DASHBOARD_ROLE_KEY = "dashboard_role"
SESSION_DASHBOARD_SOURCE_KEY = "dashboard_source"
SESSION_DASHBOARD_INSTALLATION_ID_KEY = "dashboard_installation_id"
SESSION_DASHBOARD_SLACK_USER_ID_KEY = "dashboard_slack_user_id"

templates = Jinja2Templates(directory=TEMPLATE_DIR)


def create_app(
    settings: DashboardSettings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    """Create the dashboard app."""

    resolved_settings = settings or load_dashboard_settings()
    resolved_session_factory = session_factory or make_session_factory(
        database_url=resolved_settings.postgres_url
    )
    app = FastAPI(title="Kortny Dashboard", docs_url=None, redoc_url=None)
    app.state.dashboard_settings = resolved_settings
    app.state.session_factory = resolved_session_factory
    app.add_middleware(
        SessionMiddleware,
        secret_key=resolved_settings.session_secret,
        session_cookie="kortny_dashboard_session",
        same_site="lax",
        https_only=resolved_settings.secure_cookies,
        max_age=60 * 60 * 24 * 7,
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    templates.env.filters["money"] = _money
    templates.env.filters["number"] = _number
    templates.env.filters["datetime"] = _datetime
    templates.env.filters["json"] = _json

    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    """Register dashboard routes."""

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request) -> Response:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        next_path = _safe_next_path(request.query_params.get("next"))
        if _session_principal(request) is not None:
            return RedirectResponse(
                url=next_path, status_code=status.HTTP_303_SEE_OTHER
            )
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context=_login_context(
                settings=settings,
                error=request.query_params.get("error"),
                next_path=next_path,
            ),
        )

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request) -> Response:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        username = form.get("username", [""])[0]
        password = form.get("password", [""])[0]
        next_path = _safe_next_path(form.get("next", ["/"])[0])

        if not settings.bootstrap_login_enabled:
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context=_login_context(
                    settings=settings,
                    error="Password login is disabled for this dashboard.",
                    next_path=next_path,
                ),
                status_code=status.HTTP_403_FORBIDDEN,
            )

        username_ok = secrets.compare_digest(username, settings.username)
        password_ok = secrets.compare_digest(password, settings.password)
        if not (username_ok and password_ok):
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context=_login_context(
                    settings=settings,
                    error="The username or password is incorrect.",
                    next_path=next_path,
                ),
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        request.session.clear()
        _set_dashboard_session(
            request,
            DashboardPrincipal(
                display_name=settings.username,
                role="admin",
                source="bootstrap",
            ),
        )
        return RedirectResponse(url=next_path, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/auth/slack/start")
    def slack_login_start(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        next_path: Annotated[str | None, Query(alias="next")] = None,
    ) -> RedirectResponse:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        safe_next = _safe_next_path(next_path)
        if not settings.slack_login_enabled:
            return _login_redirect_with_error(
                "Slack login is not configured for this dashboard.",
                next_path=safe_next,
            )

        state_value = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        session.add(
            DashboardOAuthState(
                provider="slack",
                state=state_value,
                redirect_path=safe_next,
                expires_at=now
                + timedelta(minutes=settings.slack_oauth_state_ttl_minutes),
            )
        )
        session.commit()

        client = _slack_openid_client(settings)
        return RedirectResponse(
            url=client.authorize_url(state=state_value),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/auth/slack/callback")
    def slack_login_callback(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        code: Annotated[str | None, Query()] = None,
        state_value: Annotated[str | None, Query(alias="state")] = None,
        error: Annotated[str | None, Query()] = None,
    ) -> RedirectResponse:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        if error:
            return _login_redirect_with_error(f"Slack login failed: {error}")
        if not settings.slack_login_enabled:
            return _login_redirect_with_error(
                "Slack login is not configured for this dashboard."
            )
        if not code or not state_value:
            return _login_redirect_with_error("Slack login did not return a code.")

        oauth_state = session.scalar(
            select(DashboardOAuthState).where(
                DashboardOAuthState.provider == "slack",
                DashboardOAuthState.state == state_value,
            )
        )
        now = datetime.now(UTC)
        if oauth_state is None or oauth_state.used_at is not None:
            return _login_redirect_with_error("Slack login state is invalid.")
        if oauth_state.expires_at < now:
            return _login_redirect_with_error("Slack login state expired.")

        try:
            client = _slack_openid_client(settings)
            access_token = client.exchange_code(code=code)
            profile = client.user_info(access_token=access_token)
            dashboard_user = upsert_dashboard_user(session, profile=profile, now=now)
            oauth_state.used_at = now
            session.commit()
        except DashboardAuthError as exc:
            session.rollback()
            return _login_redirect_with_error(str(exc))

        _set_dashboard_session(
            request,
            DashboardPrincipal(
                dashboard_user_id=dashboard_user.id,
                installation_id=dashboard_user.installation_id,
                slack_user_id=dashboard_user.slack_user_id,
                display_name=dashboard_user.display_name,
                role=dashboard_user.role,
                source="slack",
            ),
        )
        redirect_path = oauth_state.redirect_path
        if dashboard_user.role != "admin" and redirect_path == "/":
            redirect_path = "/me"
        return RedirectResponse(
            url=redirect_path,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/logout")
    def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        runtime_settings, runtime_error = _load_runtime_settings()
        system_health = get_system_health(
            session,
            dashboard_settings=settings,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
        )
        overview = get_dashboard_overview(session, system_health=system_health)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                **_dashboard_context(principal, active_page="overview"),
                "overview": overview,
            },
        )

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> Response:
        task_page = list_tasks(session, page=page, page_size=page_size)
        return templates.TemplateResponse(
            request=request,
            name="tasks.html",
            context={
                **_dashboard_context(principal, active_page="tasks"),
                "task_page": task_page,
                "page_size": page_size,
            },
        )

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    def task_detail(
        request: Request,
        task_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        detail = get_task_detail(session, task_id)
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="task_detail.html",
            context={
                **_dashboard_context(principal, active_page="tasks"),
                "detail": detail,
            },
        )

    @app.get("/usage", response_class=HTMLResponse)
    def usage(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
    ) -> Response:
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        aggregate = get_usage_aggregate(session, start=start, end=end)
        return templates.TemplateResponse(
            request=request,
            name="usage.html",
            context={
                **_dashboard_context(principal, active_page="usage"),
                "aggregate": aggregate,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/users", response_class=HTMLResponse)
    def users(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
    ) -> Response:
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        directory = list_users(session, start=start, end=end)
        return templates.TemplateResponse(
            request=request,
            name="users.html",
            context={
                **_dashboard_context(principal, active_page="users"),
                "directory": directory,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/memory", response_class=HTMLResponse)
    def memory(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        view: Annotated[str, Query()] = "facts",
        q: Annotated[str | None, Query()] = None,
        scope: Annotated[str, Query()] = "all",
        status_filter: Annotated[str, Query(alias="status")] = "active",
        outcome: Annotated[str, Query()] = "all",
        sort: Annotated[str | None, Query()] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        memory_dashboard = get_memory_dashboard(
            session,
            view=view,
            query=q,
            scope_filter=scope,
            status_filter=status_filter,
            outcome_filter=outcome,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        return templates.TemplateResponse(
            request=request,
            name="memory.html",
            context={
                **_dashboard_context(principal, active_page="memory"),
                "memory": memory_dashboard,
                "memory_return_path": _request_path(request),
                "memory_base_path": "/memory",
                "memory_actions_enabled": True,
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get("/integrations", response_class=HTMLResponse)
    def integrations(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        composio_q: Annotated[str | None, Query(alias="composio_q")] = None,
    ) -> Response:
        runtime_settings, runtime_error = _load_runtime_settings()
        integration_dashboard = get_integration_dashboard(
            session=session,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
            composio_query=composio_q,
        )
        return templates.TemplateResponse(
            request=request,
            name="integrations.html",
            context={
                **_dashboard_context(principal, active_page="integrations"),
                "integrations": integration_dashboard,
                "composio_q": composio_q or "",
            },
        )

    @app.get("/composio", response_class=HTMLResponse)
    def composio_catalog(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        q: Annotated[str | None, Query()] = None,
    ) -> Response:
        runtime_settings, runtime_error = _load_runtime_settings()
        catalog = get_composio_catalog_dashboard(
            session,
            runtime_settings=runtime_settings,
            query=q,
        )
        return templates.TemplateResponse(
            request=request,
            name="composio.html",
            context={
                **_dashboard_context(principal, active_page="composio"),
                "catalog": catalog,
                "runtime_error": runtime_error,
                "q": q or "",
            },
        )

    @app.get("/composio/callback")
    def composio_callback(
        request: Request,
        connection_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        status_text: Annotated[str | None, Query(alias="status")] = None,
        connected_account_id: Annotated[str | None, Query()] = None,
        connected_account_id_camel: Annotated[
            str | None, Query(alias="connectedAccountId")
        ] = None,
        connection_token: Annotated[str | None, Query()] = None,
    ) -> RedirectResponse:
        connection = session.get(ComposioConnection, connection_id)
        if connection is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if not _can_manage_composio_connection(principal, connection):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

        callback_status = (status_text or "").strip().lower()
        callback_payload = dict(request.query_params)
        metadata = dict(connection.metadata_json or {})
        expected_token = metadata.get("callback_token")
        if (
            isinstance(expected_token, str)
            and expected_token
            and (
                not connection_token
                or not secrets.compare_digest(connection_token, expected_token)
            )
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        metadata["callback"] = callback_payload
        resolved_connected_account_id = (
            connected_account_id or connected_account_id_camel
        )
        if resolved_connected_account_id:
            connection.connected_account_id = resolved_connected_account_id
        if resolved_connected_account_id or callback_status in {
            "success",
            "active",
            "connected",
        }:
            connection.status = "active"
            notice = "Composio account connected."
            tone = "success"
        else:
            connection.status = "failed"
            notice = "Composio connection did not complete."
            tone = "danger"
        connection.metadata_json = metadata
        session.commit()

        base_path = (
            "/composio"
            if principal.role == "admin"
            else "/me/integrations"
        )
        return _redirect_with_notice(
            f"{base_path}/{quote(connection.toolkit_slug, safe='')}",
            notice,
            tone=tone,
        )

    @app.post("/composio/{toolkit_slug}/auth-configs")
    async def composio_create_auth_config(
        request: Request,
        toolkit_slug: str,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        del principal, session
        next_path = f"/composio/{quote(toolkit_slug.strip().lower(), safe='')}"
        runtime_settings, runtime_error = _load_runtime_settings()
        if runtime_error or runtime_settings is None:
            return _redirect_with_notice(
                next_path,
                "Runtime settings are invalid. Fix System configuration first.",
                tone="danger",
            )
        if not runtime_settings.composio_api_key:
            return _redirect_with_notice(
                next_path,
                "COMPOSIO_API_KEY is required before creating auth configs.",
                tone="danger",
            )

        client = ComposioClient(
            api_key=runtime_settings.composio_api_key,
            timeout_seconds=runtime_settings.composio_request_timeout_seconds,
        )
        try:
            auth_config = client.create_managed_auth_config(
                toolkit_slug=toolkit_slug.strip().lower()
            )
        except ComposioConnectionError as exc:
            return _redirect_with_notice(
                next_path,
                f"Could not create auth config: {str(exc)}",
                tone="danger",
            )
        return _redirect_with_notice(
            next_path,
            f"Created auth config {auth_config.id}.",
        )

    @app.post("/composio/{toolkit_slug}/connect")
    async def composio_start_connect(
        request: Request,
        toolkit_slug: str,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        normalized_slug = toolkit_slug.strip().lower()
        next_path = f"/composio/{quote(normalized_slug, safe='')}"
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        auth_config_id = _form_value(form, "auth_config_id")
        owner_slack_user_id = _form_value(form, "owner_slack_user_id")
        if not owner_slack_user_id:
            owner_slack_user_id = _default_slack_owner_id(session)
        display_name = _form_value(form, "display_name")
        scope_type = _form_value(form, "visibility_scope_type") or "user"
        channel_scope_id = _form_value(form, "channel_scope_id")
        if principal.role != "admin":
            if principal.installation_id is None or principal.slack_user_id is None:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
            auth_config_id = ""
            owner_slack_user_id = principal.slack_user_id or ""
            scope_type = "user"
            channel_scope_id = ""
            next_path = f"/me/integrations/{quote(normalized_slug, safe='')}"

        if not owner_slack_user_id:
            return _redirect_with_notice(
                next_path,
                "Choose the Slack user who owns this connection in Advanced options.",
                tone="danger",
            )
        if scope_type not in {"user", "channel", "workspace"}:
            return _redirect_with_notice(
                next_path,
                "Visibility scope must be personal, channel, or workspace.",
                tone="danger",
            )
        if scope_type == "channel" and not channel_scope_id:
            return _redirect_with_notice(
                next_path,
                "Choose a Slack channel for channel-scoped connections.",
                tone="danger",
            )

        if principal.role == "admin":
            installation = _installation_for_owner(session, owner_slack_user_id)
        else:
            installation = session.get(Installation, principal.installation_id)
        if installation is None:
            return _redirect_with_notice(
                next_path,
                "No Slack installation has been recorded yet.",
                tone="danger",
            )

        runtime_settings, runtime_error = _load_runtime_settings()
        if runtime_error or runtime_settings is None:
            return _redirect_with_notice(
                next_path,
                "Runtime settings are invalid. Fix System configuration first.",
                tone="danger",
            )
        if not runtime_settings.composio_api_key:
            return _redirect_with_notice(
                next_path,
                "COMPOSIO_API_KEY is required before creating Connect Links.",
                tone="danger",
            )

        scope_id = _composio_scope_id(
            scope_type=scope_type,
            owner_slack_user_id=owner_slack_user_id,
            channel_scope_id=channel_scope_id,
        )
        composio_user_id = f"slack:{installation.id}:{owner_slack_user_id}"
        connection = ComposioConnection(
            installation_id=installation.id,
            toolkit_slug=normalized_slug,
            auth_config_id=auth_config_id,
            composio_user_id=composio_user_id,
            owner_slack_user_id=owner_slack_user_id,
            visibility_scope_type=scope_type,
            visibility_scope_id=scope_id,
            status="pending",
            display_name=display_name or f"{normalized_slug} connection",
            metadata_json={
                "created_from": "dashboard",
                "dashboard_user": request.session.get(SESSION_USER_KEY),
                "dashboard_source": "member" if principal.role != "admin" else "admin",
            },
        )
        session.add(connection)
        session.flush()

        callback_token = secrets.token_urlsafe(24)
        callback_url = (
            f"{request.url_for('composio_callback')}?"
            + urlencode(
                {
                    "connection_id": str(connection.id),
                    "connection_token": callback_token,
                }
            )
        )
        connection.metadata_json = {
            **dict(connection.metadata_json or {}),
            "callback_token": callback_token,
        }
        client = ComposioClient(
            api_key=runtime_settings.composio_api_key,
            timeout_seconds=runtime_settings.composio_request_timeout_seconds,
        )
        try:
            auth_config_id, auth_config_source = _resolve_composio_auth_config_id(
                client,
                toolkit_slug=normalized_slug,
                override_auth_config_id=auth_config_id,
            )
        except (ComposioCatalogError, ComposioConnectionError) as exc:
            session.rollback()
            return _redirect_with_notice(
                next_path,
                f"Could not prepare Composio auth: {str(exc)}",
                tone="danger",
            )
        connection.auth_config_id = auth_config_id

        try:
            connect_request = client.create_connect_link(
                user_id=composio_user_id,
                auth_config_id=auth_config_id,
                callback_url=callback_url,
            )
        except ComposioConnectionError as exc:
            session.rollback()
            return _redirect_with_notice(
                next_path,
                f"Could not create Connect Link: {str(exc)}",
                tone="danger",
            )

        connection.connection_request_id = connect_request.id
        if connect_request.connected_account_id:
            connection.connected_account_id = connect_request.connected_account_id
        connection.metadata_json = {
            **dict(connection.metadata_json or {}),
            "auth_config_source": auth_config_source,
            "connect_link_status": connect_request.status,
            "redirect_url": connect_request.redirect_url,
            "connected_account_id_from_link": connect_request.connected_account_id,
        }
        session.commit()
        return RedirectResponse(
            url=connect_request.redirect_url,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/composio/{toolkit_slug}", response_class=HTMLResponse)
    def composio_detail(
        request: Request,
        toolkit_slug: str,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        runtime_settings, runtime_error = _load_runtime_settings()
        detail = get_composio_toolkit_detail(
            session,
            slug=toolkit_slug,
            runtime_settings=runtime_settings,
        )
        if detail.error and "404" in detail.error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="composio_detail.html",
            context={
                **_dashboard_context(principal, active_page="composio"),
                "detail": detail,
                "runtime_error": runtime_error,
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get("/admin/users", response_class=HTMLResponse)
    def admin_users(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        users = tuple(
            session.scalars(
                select(DashboardUser).order_by(
                    DashboardUser.role.asc(),
                    DashboardUser.display_name.asc(),
                    DashboardUser.created_at.asc(),
                )
            )
        )
        return templates.TemplateResponse(
            request=request,
            name="admin_users.html",
            context={
                **_dashboard_context(principal, active_page="admin_users"),
                "users": users,
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.post("/admin/users/{dashboard_user_id}/role")
    async def admin_update_user_role(
        request: Request,
        dashboard_user_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        role = _form_value(form, "role")
        if role not in {"admin", "member"}:
            return _redirect_with_notice(
                "/admin/users",
                "Role must be admin or member.",
                tone="danger",
            )
        user = session.get(DashboardUser, dashboard_user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if (
            role == "member"
            and principal.dashboard_user_id == user.id
            and _active_admin_count(session, installation_id=user.installation_id) <= 1
        ):
            return _redirect_with_notice(
                "/admin/users",
                "You cannot demote the only active admin.",
                tone="danger",
            )
        user.role = role
        session.commit()
        return _redirect_with_notice("/admin/users", "Dashboard user role updated.")

    @app.post("/admin/users/{dashboard_user_id}/status")
    async def admin_update_user_status(
        request: Request,
        dashboard_user_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        user_status = _form_value(form, "status")
        if user_status not in {"active", "disabled"}:
            return _redirect_with_notice(
                "/admin/users",
                "Status must be active or disabled.",
                tone="danger",
            )
        user = session.get(DashboardUser, dashboard_user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if user_status == "disabled" and principal.dashboard_user_id == user.id:
            return _redirect_with_notice(
                "/admin/users",
                "You cannot disable your own dashboard user.",
                tone="danger",
            )
        if (
            user_status == "disabled"
            and user.role == "admin"
            and _active_admin_count(session, installation_id=user.installation_id) <= 1
        ):
            return _redirect_with_notice(
                "/admin/users",
                "You cannot disable the only active admin.",
                tone="danger",
            )
        user.status = user_status
        session.commit()
        return _redirect_with_notice("/admin/users", "Dashboard user status updated.")

    @app.get("/me", response_class=HTMLResponse)
    def me_home(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        detail = get_user_detail(
            session,
            principal.slack_user_id,
            start=start,
            end=end,
            installation_id=principal.installation_id,
        )
        if detail is None:
            detail = get_user_detail(
                session,
                principal.slack_user_id,
                installation_id=principal.installation_id,
            )
        return templates.TemplateResponse(
            request=request,
            name="me.html",
            context={
                **_dashboard_context(principal, active_page="me"),
                "detail": detail,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/me/tasks", response_class=HTMLResponse)
    def me_tasks(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        task_page = list_tasks(
            session,
            page=page,
            page_size=page_size,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
        )
        return templates.TemplateResponse(
            request=request,
            name="tasks.html",
            context={
                **_dashboard_context(principal, active_page="me_tasks"),
                "task_page": task_page,
                "page_size": page_size,
            },
        )

    @app.get("/me/tasks/{task_id}", response_class=HTMLResponse)
    def me_task_detail(
        request: Request,
        task_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        detail = get_task_detail(
            session,
            task_id,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
        )
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="task_detail.html",
            context={
                **_dashboard_context(principal, active_page="me_tasks"),
                "detail": detail,
            },
        )

    @app.get("/me/usage", response_class=HTMLResponse)
    def me_usage(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        aggregate = get_usage_aggregate(
            session,
            start=start,
            end=end,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
        )
        return templates.TemplateResponse(
            request=request,
            name="usage.html",
            context={
                **_dashboard_context(principal, active_page="me_usage"),
                "aggregate": aggregate,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/me/memory", response_class=HTMLResponse)
    def me_memory(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        view: Annotated[str, Query()] = "facts",
        q: Annotated[str | None, Query()] = None,
        status_filter: Annotated[str, Query(alias="status")] = "active",
        outcome: Annotated[str, Query()] = "all",
        sort: Annotated[str | None, Query()] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        memory_dashboard = get_memory_dashboard(
            session,
            view=view,
            query=q,
            scope_filter="user",
            status_filter=status_filter,
            outcome_filter=outcome,
            sort=sort,
            page=page,
            page_size=page_size,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
            base_path="/me/memory",
        )
        return templates.TemplateResponse(
            request=request,
            name="memory.html",
            context={
                **_dashboard_context(principal, active_page="me_memory"),
                "memory": memory_dashboard,
                "memory_return_path": _request_path(request),
                "memory_base_path": "/me/memory",
                "memory_actions_enabled": False,
                "notice": None,
                "notice_tone": "success",
            },
        )

    @app.get("/me/integrations", response_class=HTMLResponse)
    def me_integrations(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        composio_q: Annotated[str | None, Query(alias="composio_q")] = None,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        runtime_settings, runtime_error = _load_runtime_settings()
        integration_dashboard = get_integration_dashboard(
            session=session,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
            composio_query=composio_q,
            installation_id=principal.installation_id,
            owner_slack_user_id=principal.slack_user_id,
        )
        return templates.TemplateResponse(
            request=request,
            name="integrations.html",
            context={
                **_dashboard_context(principal, active_page="me_integrations"),
                "integrations": integration_dashboard,
                "composio_q": composio_q or "",
            },
        )

    @app.get("/me/integrations/{toolkit_slug}", response_class=HTMLResponse)
    def me_composio_detail(
        request: Request,
        toolkit_slug: str,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        runtime_settings, runtime_error = _load_runtime_settings()
        detail = get_composio_toolkit_detail(
            session,
            slug=toolkit_slug,
            runtime_settings=runtime_settings,
            installation_id=principal.installation_id,
            owner_slack_user_id=principal.slack_user_id,
        )
        if detail.error and "404" in detail.error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="composio_detail.html",
            context={
                **_dashboard_context(principal, active_page="me_integrations"),
                "detail": detail,
                "runtime_error": runtime_error,
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
                "member_scope": True,
            },
        )

    @app.post("/memory/facts/{fact_id}/forget")
    async def memory_forget_fact(
        request: Request,
        fact_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/memory"])[0])
        try:
            forget_fact(
                session,
                fact_id,
                by_user_id=dashboard_actor(principal.display_name),
            )
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, "Memory fact forgotten.")

    @app.post("/memory/facts/{fact_id}/supersede")
    async def memory_supersede_fact(
        request: Request,
        fact_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/memory"])[0])
        value_text = form.get("value_text", [""])[0]
        try:
            supersede_fact(
                session,
                fact_id,
                value_text=value_text,
                by_user_id=dashboard_actor(principal.display_name),
            )
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, "Memory fact superseded.")

    @app.get("/users/{slack_user_id}", response_class=HTMLResponse)
    def user_detail(
        request: Request,
        slack_user_id: str,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
    ) -> Response:
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        detail = get_user_detail(
            session,
            slack_user_id,
            start=start,
            end=end,
        )
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="user_detail.html",
            context={
                **_dashboard_context(principal, active_page="users"),
                "detail": detail,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/system", response_class=HTMLResponse)
    def system(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        runtime_settings, runtime_error = _load_runtime_settings()
        system_health = get_system_health(
            session,
            dashboard_settings=settings,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
        )
        return templates.TemplateResponse(
            request=request,
            name="system.html",
            context={
                **_dashboard_context(principal, active_page="system"),
                "system": system_health,
            },
        )


def require_principal(
    request: Request,
) -> DashboardPrincipal:
    """Require a dashboard login session."""

    principal = _session_principal(request)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": _login_url_for(request)},
        )
    return principal


def require_admin(
    principal: Annotated[DashboardPrincipal, Depends(require_principal)],
) -> DashboardPrincipal:
    """Require an admin dashboard session."""

    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return principal


def require_user(request: Request) -> str:
    """Legacy helper for tests and small call sites that only need a name."""

    return require_principal(request).display_name


def get_session(request: Request) -> Iterator[Session]:
    """Yield a database session for dashboard requests."""

    factory = cast(sessionmaker[Session], request.app.state.session_factory)
    with factory() as session:
        yield session


def _form_value(form: dict[str, list[str]], name: str) -> str:
    value = form.get(name, [""])[0]
    return value.strip()


def _resolve_composio_auth_config_id(
    client: ComposioClient,
    *,
    toolkit_slug: str,
    override_auth_config_id: str,
) -> tuple[str, str]:
    if override_auth_config_id:
        return override_auth_config_id, "manual_override"

    auth_configs = client.list_auth_configs(toolkit_slug=toolkit_slug)
    for auth_config in auth_configs:
        if auth_config.enabled and auth_config.toolkit_slug == toolkit_slug:
            return auth_config.id, "existing"

    auth_config = client.create_managed_auth_config(toolkit_slug=toolkit_slug)
    return auth_config.id, "created_managed"


def _default_slack_owner_id(session: Session) -> str:
    identity = session.scalar(
        select(SlackIdentity.slack_id)
        .where(SlackIdentity.kind == "user", SlackIdentity.is_deleted.is_(False))
        .order_by(SlackIdentity.last_seen_at.desc(), SlackIdentity.updated_at.desc())
        .limit(1)
    )
    return identity or ""


def _installation_for_owner(
    session: Session,
    owner_slack_user_id: str,
) -> Installation | None:
    identity = session.scalar(
        select(SlackIdentity)
        .where(
            SlackIdentity.kind == "user",
            SlackIdentity.slack_id == owner_slack_user_id,
        )
        .order_by(SlackIdentity.last_seen_at.desc(), SlackIdentity.updated_at.desc())
        .limit(1)
    )
    if identity is not None:
        return session.get(Installation, identity.installation_id)
    return session.scalar(select(Installation).order_by(Installation.created_at.desc()))


def _composio_scope_id(
    *,
    scope_type: str,
    owner_slack_user_id: str,
    channel_scope_id: str,
) -> str | None:
    if scope_type == "workspace":
        return None
    if scope_type == "channel":
        return channel_scope_id
    return owner_slack_user_id


def _can_manage_composio_connection(
    principal: DashboardPrincipal,
    connection: ComposioConnection,
) -> bool:
    if principal.role == "admin":
        return True
    return (
        principal.installation_id == connection.installation_id
        and principal.slack_user_id == connection.owner_slack_user_id
    )


def _active_admin_count(
    session: Session,
    *,
    installation_id: UUID,
) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(DashboardUser)
            .where(
                DashboardUser.installation_id == installation_id,
                DashboardUser.role == "admin",
                DashboardUser.status == "active",
            )
        )
        or 0
    )


def _load_runtime_settings() -> tuple[Settings | None, str | None]:
    try:
        return load_settings(), None
    except SettingsError as exc:
        return None, str(exc)


def _session_principal(request: Request) -> DashboardPrincipal | None:
    display_name = request.session.get(SESSION_USER_KEY)
    if not isinstance(display_name, str) or not display_name:
        return None
    role = request.session.get(SESSION_DASHBOARD_ROLE_KEY)
    source = request.session.get(SESSION_DASHBOARD_SOURCE_KEY)
    dashboard_user_id = _session_uuid(request, SESSION_DASHBOARD_USER_ID_KEY)
    installation_id = _session_uuid(request, SESSION_DASHBOARD_INSTALLATION_ID_KEY)
    slack_user_id = request.session.get(SESSION_DASHBOARD_SLACK_USER_ID_KEY)
    role_value = role if isinstance(role, str) and role else "admin"
    if role_value == "owner":
        role_value = "admin"
    return DashboardPrincipal(
        dashboard_user_id=dashboard_user_id,
        installation_id=installation_id,
        slack_user_id=slack_user_id if isinstance(slack_user_id, str) else None,
        display_name=display_name,
        role=role_value,
        source=source if isinstance(source, str) and source else "bootstrap",
    )


def _dashboard_context(
    principal: DashboardPrincipal,
    *,
    active_page: str,
) -> dict[str, object]:
    return {
        "active_page": active_page,
        "dashboard_user": principal.display_name,
        "dashboard_role": principal.role,
        "dashboard_is_admin": principal.role == "admin",
        "dashboard_slack_user_id": principal.slack_user_id or "",
    }


def _set_dashboard_session(request: Request, principal: DashboardPrincipal) -> None:
    request.session.clear()
    request.session[SESSION_USER_KEY] = principal.display_name
    request.session[SESSION_DASHBOARD_ROLE_KEY] = principal.role
    request.session[SESSION_DASHBOARD_SOURCE_KEY] = principal.source
    if principal.dashboard_user_id is not None:
        request.session[SESSION_DASHBOARD_USER_ID_KEY] = str(
            principal.dashboard_user_id
        )
    if principal.installation_id is not None:
        request.session[SESSION_DASHBOARD_INSTALLATION_ID_KEY] = str(
            principal.installation_id
        )
    if principal.slack_user_id:
        request.session[SESSION_DASHBOARD_SLACK_USER_ID_KEY] = principal.slack_user_id


def _session_uuid(request: Request, key: str) -> UUID | None:
    raw_value = request.session.get(key)
    if not isinstance(raw_value, str) or not raw_value:
        return None
    try:
        return UUID(raw_value)
    except ValueError:
        return None


def _login_context(
    *,
    settings: DashboardSettings,
    error: str | None,
    next_path: str,
) -> dict[str, object]:
    login_error = error
    if (
        login_error is None
        and not settings.bootstrap_login_enabled
        and not settings.slack_login_enabled
    ):
        login_error = "No dashboard login method is configured."
    return {
        "error": login_error,
        "next_path": next_path,
        "slack_login_path": f"/auth/slack/start?next={quote(next_path, safe='')}",
        "bootstrap_login_enabled": settings.bootstrap_login_enabled,
        "slack_login_enabled": settings.slack_login_enabled,
    }


def _slack_openid_client(settings: DashboardSettings) -> SlackOpenIDClient:
    if not (
        settings.slack_client_id
        and settings.slack_client_secret
        and settings.slack_redirect_uri
    ):
        raise DashboardAuthError("Slack login is not configured.")
    return SlackOpenIDClient(
        client_id=settings.slack_client_id,
        client_secret=settings.slack_client_secret,
        redirect_uri=settings.slack_redirect_uri,
    )


def _login_redirect_with_error(
    error: str,
    *,
    next_path: str = "/",
) -> RedirectResponse:
    query = urlencode({"next": _safe_next_path(next_path), "error": error})
    return RedirectResponse(
        url=f"/login?{query}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


def _login_url_for(request: Request) -> str:
    next_path = request.url.path
    if request.url.query:
        next_path = f"{next_path}?{request.url.query}"
    return f"/login?next={quote(next_path, safe='')}"


def _request_path(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        return f"{path}?{request.url.query}"
    return path


def _safe_next_path(value: str | None) -> str:
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def _redirect_with_notice(
    next_path: str,
    notice: str,
    *,
    tone: str = "success",
) -> RedirectResponse:
    path = _path_with_notice(next_path, notice=notice, tone=tone)
    return RedirectResponse(url=path, status_code=status.HTTP_303_SEE_OTHER)


def _path_with_notice(next_path: str, *, notice: str, tone: str) -> str:
    parts = urlsplit(_safe_next_path(next_path))
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in {"notice", "notice_tone"}
    ]
    query_pairs.append(("notice", notice))
    query_pairs.append(("notice_tone", _notice_tone(tone)))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query_pairs), parts.fragment)
    )


def _notice_tone(value: str) -> str:
    if value in {"success", "warning", "danger", "neutral"}:
        return value
    return "success"


def _money(value: Decimal | int | float | str | None) -> str:
    if value is None:
        return "$0.000000"
    return f"${Decimal(value):,.6f}"


def _number(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{int(Decimal(str(value))):,}"
    except (InvalidOperation, ValueError):
        return str(value)


def _datetime(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace("+00:00", " UTC")


def _json(value: object) -> str:
    if value is None:
        return "{}"
    return json.dumps(value, indent=2, sort_keys=True, default=str)
