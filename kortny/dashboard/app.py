"""FastAPI app for the read-only Kortny cost dashboard."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker
from starlette.middleware.sessions import SessionMiddleware

from kortny.config import Settings, SettingsError, load_settings
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
from kortny.db.session import make_session_factory

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
SESSION_USER_KEY = "dashboard_user"

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
        next_path = _safe_next_path(request.query_params.get("next"))
        if _session_username(request) is not None:
            return RedirectResponse(
                url=next_path, status_code=status.HTTP_303_SEE_OTHER
            )
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": None, "next_path": next_path},
        )

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request) -> Response:
        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        username = form.get("username", [""])[0]
        password = form.get("password", [""])[0]
        next_path = _safe_next_path(form.get("next", ["/"])[0])

        username_ok = secrets.compare_digest(username, settings.username)
        password_ok = secrets.compare_digest(password, settings.password)
        if not (username_ok and password_ok):
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "error": "The username or password is incorrect.",
                    "next_path": next_path,
                },
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        request.session.clear()
        request.session[SESSION_USER_KEY] = settings.username
        return RedirectResponse(url=next_path, status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/logout")
    def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "overview",
                "dashboard_user": username,
                "overview": overview,
            },
        )

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks(
        request: Request,
        username: Annotated[str, Depends(require_user)],
        session: Annotated[Session, Depends(get_session)],
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> Response:
        task_page = list_tasks(session, page=page, page_size=page_size)
        return templates.TemplateResponse(
            request=request,
            name="tasks.html",
            context={
                "active_page": "tasks",
                "dashboard_user": username,
                "task_page": task_page,
                "page_size": page_size,
            },
        )

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    def task_detail(
        request: Request,
        task_id: UUID,
        username: Annotated[str, Depends(require_user)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        detail = get_task_detail(session, task_id)
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="task_detail.html",
            context={
                "active_page": "tasks",
                "dashboard_user": username,
                "detail": detail,
            },
        )

    @app.get("/usage", response_class=HTMLResponse)
    def usage(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "usage",
                "dashboard_user": username,
                "aggregate": aggregate,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/users", response_class=HTMLResponse)
    def users(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "users",
                "dashboard_user": username,
                "directory": directory,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/memory", response_class=HTMLResponse)
    def memory(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "memory",
                "dashboard_user": username,
                "memory": memory_dashboard,
                "memory_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get("/integrations", response_class=HTMLResponse)
    def integrations(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "integrations",
                "dashboard_user": username,
                "integrations": integration_dashboard,
                "composio_q": composio_q or "",
            },
        )

    @app.get("/composio", response_class=HTMLResponse)
    def composio_catalog(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "composio",
                "dashboard_user": username,
                "catalog": catalog,
                "runtime_error": runtime_error,
                "q": q or "",
            },
        )

    @app.get("/composio/{toolkit_slug}", response_class=HTMLResponse)
    def composio_detail(
        request: Request,
        toolkit_slug: str,
        username: Annotated[str, Depends(require_user)],
        session: Annotated[Session, Depends(get_session)],
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
                "active_page": "composio",
                "dashboard_user": username,
                "detail": detail,
                "runtime_error": runtime_error,
            },
        )

    @app.post("/memory/facts/{fact_id}/forget")
    async def memory_forget_fact(
        request: Request,
        fact_id: UUID,
        username: Annotated[str, Depends(require_user)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/memory"])[0])
        try:
            forget_fact(
                session,
                fact_id,
                by_user_id=dashboard_actor(username),
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
        username: Annotated[str, Depends(require_user)],
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
                by_user_id=dashboard_actor(username),
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
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "users",
                "dashboard_user": username,
                "detail": detail,
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/system", response_class=HTMLResponse)
    def system(
        request: Request,
        username: Annotated[str, Depends(require_user)],
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
                "active_page": "system",
                "dashboard_user": username,
                "system": system_health,
            },
        )


def require_user(
    request: Request,
) -> str:
    """Require a dashboard login session."""

    username = _session_username(request)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": _login_url_for(request)},
        )
    return username


def get_session(request: Request) -> Iterator[Session]:
    """Yield a database session for dashboard requests."""

    factory = cast(sessionmaker[Session], request.app.state.session_factory)
    with factory() as session:
        yield session


def _load_runtime_settings() -> tuple[Settings | None, str | None]:
    try:
        return load_settings(), None
    except SettingsError as exc:
        return None, str(exc)


def _session_username(request: Request) -> str | None:
    username = request.session.get(SESSION_USER_KEY)
    if isinstance(username, str) and username:
        return username
    return None


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
