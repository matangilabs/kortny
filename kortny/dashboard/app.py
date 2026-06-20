"""FastAPI app for the read-only Kortny cost dashboard."""

from __future__ import annotations

import json
import secrets
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slack_sdk import WebClient
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
from kortny.dashboard.autonomy_actions import (
    clear_channel_level,
    set_channel_level,
    set_workspace_level,
)
from kortny.dashboard.autonomy_data import get_autonomy_dashboard
from kortny.dashboard.data import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MODEL_CATALOG_PAGE_SIZE,
    TaskListPage,
    get_composio_catalog_dashboard,
    get_composio_toolkit_detail,
    get_consolidation_dashboard,
    get_dashboard_overview,
    get_integration_dashboard,
    get_knowledge_graph_dashboard,
    get_llm_model_config_dashboard,
    get_llm_provider_config_detail,
    get_llm_provider_model_catalog_page,
    get_memory_dashboard,
    get_system_health,
    get_task_detail,
    get_usage_aggregate,
    get_user_detail,
    get_witness_candidates_dashboard,
    get_witness_kpis,
    list_tasks,
    list_users,
    llm_tier_catalog_options,
    parse_date_bound,
)
from kortny.dashboard.knowledge_graph_actions import (
    archive_edge,
    archive_entity,
    confirm_edge,
    confirm_entity,
)
from kortny.dashboard.mcp_actions import (
    McpServerError,
    add_mcp_server,
    parse_kv_textarea,
    remove_mcp_server,
    repin_mcp_tool,
    set_mcp_trust_tier,
    toggle_mcp_server,
    toggle_mcp_tool,
)
from kortny.dashboard.mcp_data import get_mcp_dashboard
from kortny.dashboard.memory_actions import (
    dashboard_actor,
    forget_fact,
    supersede_fact,
)
from kortny.dashboard.schedules import (
    apply_schedule_action,
    get_schedule_dashboard,
    get_schedule_detail,
    update_schedule_from_dashboard,
)
from kortny.dashboard.settings import DashboardSettings, load_dashboard_settings
from kortny.dashboard.setup import (
    LLM_PROVIDER_CHOICES,
    ValidationOutcome,
    load_app_manifest,
    manifest_deep_link,
    render_env_block,
    settings_are_complete,
    settings_error_message,
    validate_llm_key,
    validate_slack_token,
)
from kortny.dashboard.skills_actions import (
    disable_skill_enablement,
    enable_skill_for_scope,
    paste_skill_markdown,
    set_skill_trust,
    upload_skill,
)
from kortny.dashboard.skills_data import get_skill_detail, get_skills_dashboard
from kortny.db.models import (
    ComposioConnection,
    DashboardOAuthState,
    DashboardUser,
    EncryptedSecret,
    Installation,
    LLMConfigAudit,
    LLMModelCatalog,
    LLMModelPricing,
    LLMProviderAccount,
    LLMTierAssignment,
    ObserveChannelProfile,
    SlackIdentity,
    Task,
    TaskEvent,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_session_factory
from kortny.embeddings import EmbeddingIndex, embedding_index_from_settings
from kortny.execution.preview import verify_preview_token
from kortny.knowledge_graph.refresh import KnowledgeGraphRefreshService
from kortny.llm.litellm_catalog import (
    LiteLLMModelCandidate,
    check_litellm_provider_key,
    default_probe_model,
    litellm_endpoint_model_candidates,
    litellm_model_candidates,
    litellm_provider_option,
    model_candidate_for_identifier,
)
from kortny.llm.provider_config import (
    CONFIG_TIERS,
    ENV_CREDENTIAL_SOURCE,
    SECRET_CREDENTIAL_SOURCE,
    bootstrap_llm_provider_config_from_env,
    secret_resolver_from_settings,
)
from kortny.observe.style_cards import reset_style_card, set_pinned_style
from kortny.secrets import SecretEncryptionError, encrypt_secret_value
from kortny.skills import SkillRegistryService
from kortny.skills.bootstrap import seed_skills_at_startup
from kortny.skills.ingestion import SkillIngestionError
from kortny.witness import (
    DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
    DEFAULT_WITNESS_SNOOZE,
    WitnessAutopilot,
    WitnessAutopilotRunResult,
    WitnessRunner,
    WitnessRunResult,
    accept_candidate,
    archive_candidate,
    dismiss_candidate,
    reactivate_candidate,
    send_private_suggestion,
    snooze_candidate,
)
from kortny.witness.automation import AutomationOutcome, materialize_acceptance
from kortny.witness.lifecycle import WitnessSlackClient

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
SESSION_USER_KEY = "dashboard_user"
SESSION_DASHBOARD_USER_ID_KEY = "dashboard_user_id"
SESSION_DASHBOARD_ROLE_KEY = "dashboard_role"
SESSION_DASHBOARD_SOURCE_KEY = "dashboard_source"
SESSION_DASHBOARD_INSTALLATION_ID_KEY = "dashboard_installation_id"
SESSION_DASHBOARD_SLACK_USER_ID_KEY = "dashboard_slack_user_id"
MODEL_PRICE_PER_MTOK_MIN = Decimal("0")
MODEL_PRICE_PER_MTOK_MAX = Decimal("999999.999999")
MODEL_TIER_VALUES: frozenset[str] = frozenset(tier.value for tier in CONFIG_TIERS)

templates = Jinja2Templates(directory=TEMPLATE_DIR)


def create_app(
    settings: DashboardSettings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    *,
    setup_mode: bool | None = None,
) -> FastAPI:
    """Create the dashboard app.

    HIG-209: when the full runtime ``Settings`` cannot load (a missing required
    Slack/LLM/Postgres/Composio field), the app can boot in SETUP-ONLY mode
    where every route serves the first-run wizard. ``setup_mode`` is passed
    explicitly by the service entrypoint (``create_service_app`` /
    ``__main__``); when ``None`` it defaults to ``False`` so programmatic callers
    (tests, embedders) always get the normal app unless they opt in.
    """

    resolved_settings = settings or load_dashboard_settings()
    resolved_session_factory = session_factory or make_session_factory(
        database_url=resolved_settings.postgres_url
    )
    resolved_setup_mode = bool(setup_mode)
    app = FastAPI(title="Kortny Dashboard", docs_url=None, redoc_url=None)
    app.state.dashboard_settings = resolved_settings
    app.state.session_factory = resolved_session_factory
    app.state.setup_mode = resolved_setup_mode
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

    register_setup_routes(app)
    if resolved_setup_mode:
        # SETUP-ONLY mode: a catch-all funnels every other route to the wizard
        # so the operator is never stranded on a 500 while .env is incomplete.
        register_setup_catch_all(app)
    else:
        register_routes(app)

        @app.on_event("startup")
        def _seed_skills_on_startup() -> None:
            # HIG-239: ensure a fresh install has the builtin + curated skill
            # catalog without waiting for an admin to open /skills. Failure-
            # isolated end to end; never blocks dashboard boot.
            runtime_settings, _error = _load_runtime_settings()
            if runtime_settings is None:
                return
            seed_skills_at_startup(resolved_session_factory, runtime_settings)

    return app


def create_service_app() -> FastAPI:
    """Service entrypoint factory (used by ``kortny.dashboard.__main__``).

    Derives SETUP-ONLY mode from whether the full runtime ``Settings`` load, so
    the operator gets the wizard when ``.env`` is incomplete and the normal
    dashboard once it is complete.
    """

    return create_app(setup_mode=not settings_are_complete())


def _setup_values_from_form(form: Mapping[str, list[str]]) -> dict[str, str]:
    """Pull wizard field values out of a parsed form into env-key form."""

    def first(key: str) -> str:
        values = form.get(key)
        return values[0].strip() if values and isinstance(values[0], str) else ""

    observability = "true" if form.get("observability_enabled") else ""
    return {
        "LLM_PROVIDER": first("llm_provider"),
        "LLM_API_KEY": first("llm_api_key"),
        "LLM_MODEL": first("llm_model"),
        "SLACK_APP_NAME": first("app_name"),
        "SLACK_BOT_TOKEN": first("slack_bot_token"),
        "SLACK_APP_TOKEN": first("slack_app_token"),
        "SLACK_SIGNING_SECRET": first("slack_signing_secret"),
        "COMPOSIO_API_KEY": first("composio_api_key"),
        "OBSERVABILITY_ENABLED": observability,
    }


def _setup_context(
    request: Request,
    *,
    values: dict[str, str],
    llm_result: ValidationOutcome | None = None,
    slack_result: ValidationOutcome | None = None,
    env_block: str | None = None,
    notice: str | None = None,
    notice_tone: str = "success",
) -> dict[str, object]:
    setup_mode = bool(getattr(request.app.state, "setup_mode", False))
    app_name = values.get("SLACK_APP_NAME") or "Kortny"
    manifest = load_app_manifest(app_name=app_name)
    return {
        "setup_mode": setup_mode,
        "settings_error": settings_error_message() if setup_mode else None,
        "values": values,
        "llm_provider_choices": LLM_PROVIDER_CHOICES,
        "manifest_deep_link": manifest_deep_link(manifest),
        "llm_result": llm_result,
        "slack_result": slack_result,
        "env_block": env_block,
        "notice": notice,
        "notice_tone": notice_tone,
    }


def register_setup_routes(app: FastAPI) -> None:
    """Register the first-run setup wizard (reachable in both app modes)."""

    @app.get("/setup", response_class=HTMLResponse)
    def setup_wizard(request: Request) -> Response:
        setup_mode = bool(getattr(request.app.state, "setup_mode", False))
        # When config is complete, /setup stays admin-only for re-validation.
        if not setup_mode:
            principal = _session_principal(request)
            if principal is None:
                return RedirectResponse(
                    url=_login_url_for(request),
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            if principal.role != "admin":
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        values = _default_setup_values(request)
        app_name = request.query_params.get("app_name")
        if app_name:
            values["SLACK_APP_NAME"] = app_name.strip()
        return templates.TemplateResponse(
            request=request,
            name="setup.html",
            context=_setup_context(request, values=values),
        )

    @app.post("/setup/validate-llm", response_class=HTMLResponse)
    async def setup_validate_llm(request: Request) -> Response:
        _require_setup_access(request)
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        values = _setup_values_from_form(form)
        result = validate_llm_key(
            provider=values["LLM_PROVIDER"],
            api_key=values["LLM_API_KEY"],
            model=values["LLM_MODEL"],
        )
        return templates.TemplateResponse(
            request=request,
            name="setup.html",
            context=_setup_context(request, values=values, llm_result=result),
        )

    @app.post("/setup/validate-slack", response_class=HTMLResponse)
    async def setup_validate_slack(request: Request) -> Response:
        _require_setup_access(request)
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        values = _setup_values_from_form(form)
        client_factory = getattr(request.app.state, "setup_slack_client_factory", None)
        result = validate_slack_token(
            bot_token=values["SLACK_BOT_TOKEN"],
            client_factory=client_factory,
        )
        return templates.TemplateResponse(
            request=request,
            name="setup.html",
            context=_setup_context(request, values=values, slack_result=result),
        )

    @app.post("/setup/render-env", response_class=HTMLResponse)
    async def setup_render_env(request: Request) -> Response:
        _require_setup_access(request)
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        values = _setup_values_from_form(form)
        env_block = render_env_block(values)
        notice = (
            "Copy these into .env, then restart the app, worker, and ambient "
            "services. The dashboard cannot apply them to those processes for you."
        )
        return templates.TemplateResponse(
            request=request,
            name="setup.html",
            context=_setup_context(
                request,
                values=values,
                env_block=env_block,
                notice=notice,
            ),
        )


def register_setup_catch_all(app: FastAPI) -> None:
    """In SETUP-ONLY mode, funnel all unmatched routes to the wizard."""

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def setup_catch_all(request: Request, full_path: str) -> Response:
        return RedirectResponse(url="/setup", status_code=status.HTTP_303_SEE_OTHER)


def _require_setup_access(request: Request) -> None:
    """Allow setup mutations in setup mode; otherwise require an admin."""

    if bool(getattr(request.app.state, "setup_mode", False)):
        return
    principal = _session_principal(request)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": _login_url_for(request)},
        )
    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


def _default_setup_values(request: Request) -> dict[str, str]:
    """Seed wizard fields from any runtime settings that already loaded."""

    runtime_settings, _error = _load_runtime_settings()
    if runtime_settings is None:
        return {
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "",
            "LLM_MODEL": "",
            "SLACK_APP_NAME": "Kortny",
            "SLACK_BOT_TOKEN": "",
            "SLACK_APP_TOKEN": "",
            "SLACK_SIGNING_SECRET": "",
            "COMPOSIO_API_KEY": "",
            "OBSERVABILITY_ENABLED": "",
        }
    return {
        "LLM_PROVIDER": runtime_settings.llm_provider.value,
        "LLM_API_KEY": "",
        "LLM_MODEL": runtime_settings.llm_model,
        "SLACK_APP_NAME": runtime_settings.slack_app_name,
        "SLACK_BOT_TOKEN": "",
        "SLACK_APP_TOKEN": "",
        "SLACK_SIGNING_SECRET": "",
        "COMPOSIO_API_KEY": "",
        "OBSERVABILITY_ENABLED": (
            "true" if runtime_settings.observability_enabled else ""
        ),
    }


def register_routes(app: FastAPI) -> None:
    """Register dashboard routes."""

    @app.get("/preview/{token}/{task_id}/{slug}/{file_path:path}")
    def preview_file(
        request: Request,
        token: str,
        task_id: str,
        slug: str,
        file_path: str,
    ) -> Response:
        """Serve one published sandbox preview file at a capability URL.

        Intentionally unauthenticated: the HMAC token in the path is the
        access control, so links pasted into Slack open without dashboard
        login.
        """

        settings = cast(DashboardSettings, request.app.state.dashboard_settings)
        if not settings.artifacts_dir or not settings.preview_signing_secret:
            raise HTTPException(status_code=404)
        if not verify_preview_token(
            settings.preview_signing_secret, task_id, slug, token
        ):
            raise HTTPException(status_code=404)

        base = (Path(settings.artifacts_dir) / task_id / slug).resolve()
        artifacts_root = Path(settings.artifacts_dir).resolve()
        if not base.is_relative_to(artifacts_root):
            raise HTTPException(status_code=404)
        target = (base / (file_path or "index.html")).resolve()
        if not target.is_relative_to(base):
            raise HTTPException(status_code=404)
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target)

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
        principal: Annotated[DashboardPrincipal, Depends(require_dashboard_home)],
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
        q: Annotated[str | None, Query()] = None,
        status_filter: Annotated[str | None, Query(alias="status")] = None,
        channel: Annotated[str | None, Query()] = None,
        user: Annotated[str | None, Query()] = None,
        model: Annotated[str | None, Query()] = None,
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> Response:
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        task_filters = _task_filter_values(
            q=q,
            status=status_filter,
            channel=channel,
            user=user,
            model=model,
            from_date=from_date,
            to_date=to_date,
        )
        task_page = list_tasks(
            session,
            page=page,
            page_size=page_size,
            start=start,
            end=end,
            query=q,
            status=status_filter,
            channel=channel,
            user=user,
            model=model,
        )
        task_query_params = _task_query_params(task_filters, page_size=page_size)
        return templates.TemplateResponse(
            request=request,
            name="tasks.html",
            context={
                **_dashboard_context(principal, active_page="tasks"),
                "task_page": task_page,
                "page_size": page_size,
                "task_filters": task_filters,
                "task_toolbar": _task_toolbar(
                    action="/tasks",
                    task_page=task_page,
                    task_filters=task_filters,
                    reset_url="/tasks",
                ),
                "task_previous_url": _page_url(
                    "/tasks", task_query_params, task_page.previous_page
                ),
                "task_next_url": _page_url(
                    "/tasks", task_query_params, task_page.next_page
                ),
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
        days: Annotated[int | None, Query(ge=1, le=366)] = None,
    ) -> Response:
        start: datetime | None
        end: datetime | None
        if days is not None:
            end = datetime.now(UTC)
            start = end - timedelta(days=days)
        else:
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
                "usage_toolbar": _date_toolbar(
                    action="/usage",
                    title="Usage Window",
                    count_label=_usage_count_label(aggregate.total_calls),
                    from_date=from_date,
                    to_date=to_date,
                    reset_url="/usage",
                ),
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
                "users_toolbar": _date_toolbar(
                    action="/users",
                    title="User Activity Window",
                    count_label=_row_count_label(len(directory.users), "user"),
                    from_date=from_date,
                    to_date=to_date,
                    reset_url="/users",
                ),
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

    @app.get("/knowledge-graph", response_class=HTMLResponse)
    def knowledge_graph(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        view: Annotated[str, Query()] = "entities",
        q: Annotated[str | None, Query()] = None,
        scope: Annotated[str, Query()] = "all",
        state: Annotated[str, Query()] = "current",
        kind: Annotated[str, Query()] = "all",
        sort: Annotated[str | None, Query()] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        graph = get_knowledge_graph_dashboard(
            session,
            view=view,
            query=q,
            scope_filter=scope,
            state_filter=state,
            kind_filter=kind,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        return templates.TemplateResponse(
            request=request,
            name="knowledge_graph.html",
            context={
                **_dashboard_context(principal, active_page="knowledge_graph"),
                "graph": graph,
                "graph_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get("/consolidation", response_class=HTMLResponse)
    def consolidation(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        dashboard = get_consolidation_dashboard(session)
        return templates.TemplateResponse(
            request=request,
            name="consolidation.html",
            context={
                **_dashboard_context(principal, active_page="consolidation"),
                "consolidation": dashboard,
            },
        )

    @app.post("/consolidation/style-cards/{profile_id}/reset")
    async def consolidation_style_card_reset(
        request: Request,
        profile_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/consolidation"])[0])
        profile = session.get(ObserveChannelProfile, profile_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        reset_style_card(profile, by=dashboard_actor(principal.display_name))
        session.commit()
        return _redirect_with_notice(
            next_path,
            "Style card reset; the consolidator will re-derive it.",
        )

    @app.post("/consolidation/style-cards/{profile_id}/pin")
    async def consolidation_style_card_pin(
        request: Request,
        profile_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/consolidation"])[0])
        pinned_style = form.get("pinned_style", [""])[0]
        profile = session.get(ObserveChannelProfile, profile_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        set_pinned_style(
            profile,
            pinned_style=pinned_style,
            by=dashboard_actor(principal.display_name),
        )
        session.commit()
        notice = (
            "Pinned style saved; it overrides the derived channel voice."
            if pinned_style.strip()
            else "Pinned style cleared; the derived card applies again."
        )
        return _redirect_with_notice(next_path, notice)

    @app.get("/witness", response_class=HTMLResponse)
    def witness_candidates(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        q: Annotated[str | None, Query()] = None,
        status_filter: Annotated[str, Query(alias="status")] = "candidate",
        type_filter: Annotated[str, Query(alias="type")] = "all",
        scope: Annotated[str, Query()] = "all",
        sort: Annotated[str | None, Query()] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        candidates = get_witness_candidates_dashboard(
            session,
            query=q,
            status_filter=status_filter,
            type_filter=type_filter,
            scope_filter=scope,
            sort=sort,
            page=page,
            page_size=page_size,
            installation_id=principal.installation_id,
        )
        kpis = get_witness_kpis(
            session,
            installation_id=principal.installation_id,
        )
        return templates.TemplateResponse(
            request=request,
            name="witness.html",
            context={
                **_dashboard_context(principal, active_page="witness"),
                "witness": candidates,
                "witness_kpis": kpis,
                "witness_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.post("/witness/run")
    async def witness_run_scan(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/witness"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Witness scan requires a selected workspace.",
                tone="danger",
            )
        try:
            runtime_settings = load_settings()
            if not runtime_settings.witness_enabled:
                return _redirect_with_notice(
                    next_path,
                    "Witness is disabled in runtime settings.",
                    tone="warning",
                )
            actor = principal.slack_user_id or dashboard_actor(principal.display_name)
            result = WitnessRunner(
                session,
                settings=runtime_settings,
                runner_id=f"dashboard:{actor}",
            ).run_once(
                installation_id=installation_id,
                profile_limit=runtime_settings.witness_profile_scan_limit,
                delivery_limit=0,
                deliver_private=False,
                autopilot_enabled=False,
                min_scan_interval=timedelta(seconds=0),
                use_advisory_lock=False,
            )
            session.commit()
        except (SettingsError, ValueError) as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, _witness_run_notice(result))

    @app.post("/witness/autopilot")
    async def witness_run_autopilot(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/witness"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Witness autopilot requires a selected workspace.",
                tone="danger",
            )
        try:
            runtime_settings = load_settings()
            if not runtime_settings.witness_enabled:
                return _redirect_with_notice(
                    next_path,
                    "Witness is disabled in runtime settings.",
                    tone="warning",
                )
            actor = principal.slack_user_id or dashboard_actor(principal.display_name)
            result = WitnessAutopilot(
                session,
                settings=runtime_settings,
                actor_id=f"dashboard:{actor}",
            ).run_once(
                installation_id=installation_id,
                limit=runtime_settings.witness_autopilot_limit,
                min_confidence=(
                    runtime_settings.witness_autopilot_min_confidence
                    or DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE
                ),
            )
            session.commit()
        except (SettingsError, ValueError) as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, _witness_autopilot_notice(result))

    @app.post("/witness/candidates/{candidate_id}/{action}")
    async def witness_candidate_action(
        request: Request,
        candidate_id: UUID,
        action: str,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/witness"])[0])
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Witness candidate actions require a selected workspace.",
                tone="danger",
            )
        try:
            if action == "dismiss":
                dismiss_candidate(
                    session,
                    candidate_id,
                    installation_id=installation_id,
                    by_user_id=actor,
                    reason=_form_value(form, "reason"),
                )
                notice = "Witness candidate dismissed."
                tone = "warning"
            elif action == "snooze":
                snooze_candidate(
                    session,
                    candidate_id,
                    installation_id=installation_id,
                    by_user_id=actor,
                    duration=DEFAULT_WITNESS_SNOOZE,
                )
                notice = "Witness candidate snoozed for 7 days."
                tone = "warning"
            elif action == "accept":
                accepted = accept_candidate(
                    session,
                    candidate_id,
                    installation_id=installation_id,
                    by_user_id=actor,
                )
                notice, tone = _witness_accept_notice(
                    _materialize_accepted_candidate(session, accepted, actor=actor)
                )
            elif action == "reactivate":
                reactivate_candidate(
                    session,
                    candidate_id,
                    installation_id=installation_id,
                    by_user_id=actor,
                )
                notice = "Witness candidate reactivated."
                tone = "success"
            elif action == "archive":
                archive_candidate(
                    session,
                    candidate_id,
                    installation_id=installation_id,
                    by_user_id=actor,
                )
                notice = "Witness candidate archived."
                tone = "warning"
            elif action == "send":
                runtime_settings = load_settings()
                delivery = send_private_suggestion(
                    session,
                    candidate_id,
                    installation_id=installation_id,
                    by_user_id=actor,
                    client=cast(
                        WitnessSlackClient,
                        WebClient(token=runtime_settings.slack_bot_token),
                    ),
                )
                notice = f"Witness suggestion sent in DM at {delivery.message_ts}."
                tone = "success"
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except (SettingsError, ValueError) as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, notice, tone=tone)

    @app.get("/skills", response_class=HTMLResponse)
    def skills(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        view: Annotated[str, Query()] = "library",
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        SkillRegistryService(
            session, embedding_index=_embedding_index_for(session)
        ).ensure_curated_skills()
        session.commit()
        installation_id = _dashboard_installation_id(session, principal)
        skills_dashboard = get_skills_dashboard(session, installation_id)
        return templates.TemplateResponse(
            request=request,
            name="skills.html",
            context={
                **_dashboard_context(principal, active_page="skills"),
                "skills": skills_dashboard,
                "skills_view": "installed" if view == "installed" else "library",
                "skills_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get("/skills/{skill_id}", response_class=HTMLResponse)
    def skill_detail(
        request: Request,
        skill_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        installation_id = _dashboard_installation_id(session, principal)
        detail = get_skill_detail(session, installation_id, skill_id)
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="skill_detail.html",
            context={
                **_dashboard_context(principal, active_page="skills"),
                "detail": detail,
                "skills_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.post("/skills/{skill_id}/enable")
    async def skill_enable(
        request: Request,
        skill_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/skills"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Enabling a skill requires a selected workspace.",
                tone="danger",
            )
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        try:
            enablement = enable_skill_for_scope(
                session,
                installation_id=installation_id,
                skill_id=skill_id,
                scope_type=_form_value(form, "scope_type") or "workspace",
                scope_id=_form_value(form, "scope_id"),
                by_user=actor,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"Skill enabled for {enablement.scope_type} scope.",
        )

    @app.post("/skills/enablements/{enablement_id}/disable")
    async def skill_disable(
        request: Request,
        enablement_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/skills"])[0])
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        try:
            disable_skill_enablement(
                session, enablement_id=enablement_id, by_user=actor
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, "Skill disabled.", tone="warning")

    @app.post("/skills/upload")
    async def skill_upload(
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        skill_file: Annotated[UploadFile, File(alias="skill_file")],
        scope_type: Annotated[str, Form()] = "workspace",
        scope_id: Annotated[str, Form()] = "",
        next_path: Annotated[str, Form(alias="next")] = "/skills",
    ) -> RedirectResponse:
        next_path = _safe_next_path(next_path)
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Uploading a skill requires a selected workspace.",
                tone="danger",
            )
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        data = await skill_file.read()
        try:
            result = upload_skill(
                session,
                installation_id=installation_id,
                data=data,
                filename=skill_file.filename or "skill.zip",
                by_user=actor,
                embedding_index=_embedding_index_for(session),
            )
            enable_skill_for_scope(
                session,
                installation_id=installation_id,
                skill_id=result.skill.id,
                scope_type=scope_type or "workspace",
                scope_id=scope_id or None,
                by_user=actor,
            )
            session.commit()
        except (SkillIngestionError, ValueError) as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"Skill '{result.skill.slug}' uploaded and enabled.",
        )

    @app.post("/skills/paste")
    async def skill_paste(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/skills"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Adding a skill requires a selected workspace.",
                tone="danger",
            )
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        try:
            result = paste_skill_markdown(
                session,
                installation_id=installation_id,
                content=_form_value(form, "content") or "",
                name=_form_value(form, "name"),
                description=_form_value(form, "description"),
                by_user=actor,
                embedding_index=_embedding_index_for(session),
            )
            enable_skill_for_scope(
                session,
                installation_id=installation_id,
                skill_id=result.skill.id,
                scope_type=_form_value(form, "scope_type") or "workspace",
                scope_id=_form_value(form, "scope_id"),
                by_user=actor,
            )
            session.commit()
        except (SkillIngestionError, ValueError) as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"Skill '{result.skill.slug}' added and enabled.",
        )

    @app.post("/skills/{skill_id}/trust")
    async def skill_trust(
        request: Request,
        skill_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", [f"/skills/{skill_id}"])[0])
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        try:
            skill = set_skill_trust(
                session,
                skill_id=skill_id,
                trust_level=_form_value(form, "trust_level") or "",
                by_user=actor,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"Trust level set to {skill.trust_level}.",
        )

    @app.get("/mcp", response_class=HTMLResponse)
    def mcp_servers(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        installation_id = _dashboard_installation_id(session, principal)
        mcp = get_mcp_dashboard(session, installation_id)
        return templates.TemplateResponse(
            request=request,
            name="mcp_servers.html",
            context={
                **_dashboard_context(principal, active_page="mcp"),
                "mcp": mcp,
                "mcp_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.post("/mcp/add")
    async def mcp_add(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Registering an MCP server requires a selected workspace.",
                tone="danger",
            )
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        runtime_settings, runtime_error = _load_runtime_settings()
        encryption_key = (
            runtime_settings.encryption_key if runtime_settings is not None else None
        )
        transport = _form_value(form, "transport") or "stdio"
        args_raw = _form_value(form, "args") or ""
        args_list = [line.strip() for line in args_raw.splitlines() if line.strip()]
        env_pairs = parse_kv_textarea(_form_value(form, "env") or "")
        header_pairs = parse_kv_textarea(_form_value(form, "headers") or "")
        secret_pairs = parse_kv_textarea(_form_value(form, "secrets") or "")
        try:
            server = add_mcp_server(
                session,
                installation_id=installation_id,
                name=_form_value(form, "name") or "",
                transport=transport,
                command=_form_value(form, "command"),
                args=args_list,
                url=_form_value(form, "url"),
                env_pairs=env_pairs,
                header_pairs=header_pairs,
                secret_pairs=secret_pairs,
                created_by=actor,
                encryption_key=encryption_key,
            )
            session.commit()
        except McpServerError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        # Attempt immediate discovery; keep server on failure.
        discovery_notice = _mcp_attempt_discovery(
            session,
            server_id=server.id,
            installation_id=installation_id,
            runtime_settings=runtime_settings,
        )
        session.commit()
        return _redirect_with_notice(
            next_path,
            f"MCP server '{server.name}' registered. {discovery_notice}".strip(),
            tone="success" if "Error" not in discovery_notice else "warning",
        )

    @app.post("/mcp/{server_id}/remove")
    async def mcp_remove(
        request: Request,
        server_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "No workspace scope available.",
                tone="danger",
            )
        try:
            server = remove_mcp_server(
                session,
                installation_id=installation_id,
                server_id=server_id,
            )
            session.commit()
        except McpServerError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"MCP server '{server.name}' removed.",
            tone="warning",
        )

    @app.post("/mcp/{server_id}/toggle")
    async def mcp_toggle(
        request: Request,
        server_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No workspace scope available.", tone="danger"
            )
        try:
            server = toggle_mcp_server(
                session,
                installation_id=installation_id,
                server_id=server_id,
            )
            session.commit()
        except McpServerError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        label = "enabled" if server.status == "enabled" else "disabled"
        return _redirect_with_notice(
            next_path,
            f"MCP server '{server.name}' {label}.",
            tone="success" if server.status == "enabled" else "warning",
        )

    @app.post("/mcp/{server_id}/trust")
    async def mcp_set_trust(
        request: Request,
        server_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        trust_tier = form.get("trust_tier", [""])[0]
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No workspace scope available.", tone="danger"
            )
        try:
            server = set_mcp_trust_tier(
                session,
                installation_id=installation_id,
                server_id=server_id,
                trust_tier=trust_tier,
            )
            session.commit()
        except McpServerError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"MCP server '{server.name}' trust tier set to {server.trust_tier}.",
            tone="success" if server.trust_tier == "trusted" else "warning",
        )

    @app.post("/mcp/{server_id}/tools/{tool_id}/repin")
    async def mcp_tool_repin(
        request: Request,
        server_id: UUID,
        tool_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No workspace scope available.", tone="danger"
            )
        try:
            tool = repin_mcp_tool(
                session,
                installation_id=installation_id,
                server_id=server_id,
                tool_id=tool_id,
                approved_by=dashboard_actor(principal.display_name),
            )
            session.commit()
        except McpServerError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            f"Re-pinned MCP tool '{tool.name}' as approved.",
            tone="success",
        )

    @app.post("/mcp/{server_id}/discover")
    async def mcp_discover(
        request: Request,
        server_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No workspace scope available.", tone="danger"
            )
        runtime_settings, _runtime_error = _load_runtime_settings()
        notice = _mcp_attempt_discovery(
            session,
            server_id=server_id,
            installation_id=installation_id,
            runtime_settings=runtime_settings,
        )
        session.commit()
        tone = "danger" if "Error" in notice else "success"
        return _redirect_with_notice(next_path, notice, tone=tone)

    @app.post("/mcp/{server_id}/tools/{tool_id}/toggle")
    async def mcp_tool_toggle(
        request: Request,
        server_id: UUID,
        tool_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/mcp"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No workspace scope available.", tone="danger"
            )
        try:
            tool = toggle_mcp_tool(
                session,
                installation_id=installation_id,
                server_id=server_id,
                tool_id=tool_id,
            )
            session.commit()
        except McpServerError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        label = "enabled" if tool.enabled else "disabled"
        return _redirect_with_notice(
            next_path,
            f"Tool '{tool.name}' {label}.",
            tone="success" if tool.enabled else "warning",
        )

    @app.post("/knowledge-graph/refresh")
    async def knowledge_graph_refresh(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/knowledge-graph"])[0])
        actor = principal.slack_user_id or dashboard_actor(principal.display_name)
        result = KnowledgeGraphRefreshService(session).queue_channel_profile_refresh(
            installation_id=principal.installation_id,
            requested_by_user_id=actor,
        )
        session.commit()
        if result.known_channel_count == 0:
            return _redirect_with_notice(
                next_path,
                "No known active channels to refresh yet.",
                tone="warning",
            )
        if result.queued_count == 0:
            return _redirect_with_notice(
                next_path,
                _graph_refresh_notice(result),
                tone="neutral",
            )
        return _redirect_with_notice(next_path, _graph_refresh_notice(result))

    @app.post("/knowledge-graph/entities/{entity_id}/confirm")
    async def knowledge_graph_confirm_entity(
        request: Request,
        entity_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/knowledge-graph"])[0])
        try:
            confirm_entity(
                session,
                entity_id,
                by_user_id=dashboard_actor(principal.display_name),
            )
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, "Graph entity confirmed.")

    @app.post("/knowledge-graph/entities/{entity_id}/archive")
    async def knowledge_graph_archive_entity(
        request: Request,
        entity_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/knowledge-graph"])[0])
        try:
            result = archive_entity(
                session,
                entity_id,
                by_user_id=dashboard_actor(principal.display_name),
            )
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        edge_suffix = (
            f" Archived {len(result.archived_edge_ids):,} connected edge"
            f"{'' if len(result.archived_edge_ids) == 1 else 's'}."
            if result.archived_edge_ids
            else ""
        )
        return _redirect_with_notice(
            next_path,
            f"Graph entity archived.{edge_suffix}",
            tone="warning",
        )

    @app.post("/knowledge-graph/edges/{edge_id}/confirm")
    async def knowledge_graph_confirm_edge(
        request: Request,
        edge_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/knowledge-graph"])[0])
        try:
            confirm_edge(
                session,
                edge_id,
                by_user_id=dashboard_actor(principal.display_name),
            )
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, "Graph relationship confirmed.")

    @app.post("/knowledge-graph/edges/{edge_id}/archive")
    async def knowledge_graph_archive_edge(
        request: Request,
        edge_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/knowledge-graph"])[0])
        try:
            archive_edge(
                session,
                edge_id,
                by_user_id=dashboard_actor(principal.display_name),
            )
            session.commit()
        except LookupError as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(
            next_path,
            "Graph relationship archived.",
            tone="warning",
        )

    @app.get("/integrations", response_class=HTMLResponse)
    def integrations(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        runtime_settings, runtime_error = _load_runtime_settings()
        integration_dashboard = get_integration_dashboard(
            session=session,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
        )
        return templates.TemplateResponse(
            request=request,
            name="integrations.html",
            context={
                **_dashboard_context(principal, active_page="integrations"),
                "integrations": integration_dashboard,
            },
        )

    @app.get("/admin/models", response_class=HTMLResponse)
    def model_config(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        runtime_settings, runtime_error = _load_runtime_settings()
        model_config_dashboard = get_llm_model_config_dashboard(
            session=session,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
            installation_id=_dashboard_installation_id(session, principal),
        )
        return templates.TemplateResponse(
            request=request,
            name="model_config.html",
            context={
                **_dashboard_context(principal, active_page="model_config"),
                "model_config": model_config_dashboard,
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get(
        "/admin/models/providers/{provider_account_id}", response_class=HTMLResponse
    )
    def model_config_provider_detail(
        request: Request,
        provider_account_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        provider_detail = get_llm_provider_config_detail(
            session=session,
            provider_account_id=provider_account_id,
            installation_id=_dashboard_installation_id(session, principal),
        )
        if provider_detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request=request,
            name="model_provider_detail.html",
            context={
                **_dashboard_context(principal, active_page="model_config"),
                "detail": provider_detail,
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get(
        "/admin/models/providers/{provider_account_id}/models",
        response_class=JSONResponse,
    )
    def model_config_provider_models(
        provider_account_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        q: Annotated[str | None, Query()] = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = MODEL_CATALOG_PAGE_SIZE,
    ) -> JSONResponse:
        model_page = get_llm_provider_model_catalog_page(
            session=session,
            provider_account_id=provider_account_id,
            installation_id=_dashboard_installation_id(session, principal),
            offset=offset,
            limit=limit,
            query=q,
        )
        if model_page is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        html = templates.env.get_template("_model_catalog_rows.html").render(
            model_rows=model_page.rows,
            provider_id=provider_account_id,
            tier_options=llm_tier_catalog_options(),
            next_path=f"/admin/models/providers/{provider_account_id}",
            empty_message=(
                "No models match this search."
                if q and q.strip()
                else "No models have been synced for this provider yet."
            ),
        )
        return JSONResponse(
            {
                "html": html,
                "total_count": model_page.total_count,
                "shown_count": model_page.offset + len(model_page.rows),
                "next_offset": model_page.next_offset,
                "has_more": model_page.has_more,
            }
        )

    @app.post("/admin/models/bootstrap")
    async def model_config_bootstrap(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        runtime_settings, runtime_error = _load_runtime_settings()
        if runtime_settings is None:
            return _redirect_with_notice(
                next_path,
                f"Runtime settings are not available: {runtime_error or 'unknown error'}",
                tone="danger",
            )
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "Kortny needs a Slack installation before model config can be bootstrapped.",
                tone="danger",
            )
        result = bootstrap_llm_provider_config_from_env(
            session,
            installation_id=installation_id,
            settings=runtime_settings,
        )
        pricing_count = 0
        if result.provider_account_id is not None:
            provider = session.get(LLMProviderAccount, result.provider_account_id)
            if provider is not None:
                pricing_count = _backfill_model_pricing_for_provider(
                    session,
                    provider=provider,
                    include_provider_catalog=True,
                )
        session.commit()
        if result.created:
            pricing_suffix = (
                f" Synced {pricing_count} pricing row{'' if pricing_count == 1 else 's'}."
                if pricing_count
                else ""
            )
            return _redirect_with_notice(
                next_path,
                f"Seeded model provider config from env.{pricing_suffix}",
            )
        pricing_suffix = (
            f" Backfilled {pricing_count} pricing row{'' if pricing_count == 1 else 's'}."
            if pricing_count
            else ""
        )
        return _redirect_with_notice(
            next_path,
            f"Bootstrap skipped: {result.skipped_reason or 'config already exists'}.{pricing_suffix}",
            tone="neutral",
        )

    @app.post("/admin/models/providers")
    async def model_config_create_provider(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "No installation scope is available for this provider.",
                tone="danger",
            )
        runtime_settings, runtime_error = _load_runtime_settings()
        if runtime_settings is None:
            return _redirect_with_notice(
                next_path,
                f"Runtime settings are not available: {runtime_error or 'unknown error'}",
                tone="danger",
            )
        if runtime_settings.encryption_key is None:
            return _redirect_with_notice(
                next_path,
                "Set ENCRYPTION_KEY before saving dashboard-managed provider keys.",
                tone="danger",
            )
        selected_provider_kind = _form_value(form, "provider_kind")
        provider_kind_source = (
            _form_value(form, "provider_kind_custom")
            if selected_provider_kind == "__custom__"
            else selected_provider_kind
        )
        provider_kind = _normalize_provider_kind(provider_kind_source)
        if provider_kind is None:
            return _redirect_with_notice(
                next_path,
                "Choose a provider or enter a custom LiteLLM provider name.",
                tone="danger",
            )
        provider_option = litellm_provider_option(provider_kind)
        api_key = _form_value(form, "api_key")
        if not api_key:
            return _redirect_with_notice(
                next_path,
                "API key is required for dashboard-managed providers.",
                tone="danger",
            )
        display_name = _form_value(form, "display_name") or (
            f"{provider_option.label} provider"
            if provider_option is not None
            else f"{provider_kind.replace('_', ' ').title()} provider"
        )
        base_url = _optional_form_value(form, "base_url")
        if base_url is None and provider_option is not None:
            base_url = provider_option.default_base_url
        if (
            base_url is None
            and provider_option is not None
            and provider_option.needs_base_url
        ):
            return _redirect_with_notice(
                next_path,
                f"{provider_option.label} needs a base URL before it can be saved.",
                tone="danger",
            )
        api_version = _optional_form_value(form, "api_version")
        try:
            secret = EncryptedSecret(
                installation_id=installation_id,
                secret_type=f"llm_provider:{provider_kind}:{uuid.uuid4().hex}",
                ciphertext=encrypt_secret_value(
                    api_key,
                    encryption_key=runtime_settings.encryption_key,
                ),
            )
        except SecretEncryptionError as exc:
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        session.add(secret)
        session.flush()
        provider = LLMProviderAccount(
            installation_id=installation_id,
            provider_kind=provider_kind,
            display_name=display_name,
            status="active",
            health_status="unknown",
            base_url=base_url,
            encrypted_secret_id=secret.id,
            metadata_json={
                "credential_source": SECRET_CREDENTIAL_SOURCE,
                "source": "dashboard",
                "litellm_provider": provider_kind,
                "api_version": api_version,
                "setup_version": "hig_186_slice_4b",
            },
        )
        session.add(provider)
        session.flush()
        candidate_limit = _model_discovery_limit(provider.provider_kind, default=24)
        candidates = list(
            litellm_model_candidates(provider.provider_kind, limit=candidate_limit)
        )
        discovery_error: str | None = None
        try:
            endpoint_candidates = litellm_endpoint_model_candidates(
                provider.provider_kind,
                api_key=api_key,
                api_base=provider.base_url,
                limit=candidate_limit,
            )
            candidates = _merge_model_candidates(endpoint_candidates, candidates)
        except Exception as exc:
            discovery_error = type(exc).__name__
        imported_count, pricing_count = _upsert_model_candidates(
            session,
            provider=provider,
            candidates=tuple(candidates),
        )
        _append_llm_config_audit(
            session,
            installation_id=installation_id,
            principal=principal,
            action="create",
            entity_type="llm_provider_account",
            entity_id=str(provider.id),
            previous_value=None,
            new_value={
                **_provider_account_audit_payload(provider),
                "operation": "create_provider",
                "imported_count": imported_count,
                "pricing_count": pricing_count,
                "discovery_error": discovery_error,
            },
        )
        session.commit()
        suffix = (
            f" Model discovery hit {discovery_error}; saved local catalog rows only."
            if discovery_error
            else ""
        )
        return _redirect_with_notice(
            next_path,
            f"Added {provider.display_name} and imported {imported_count} model row{'' if imported_count == 1 else 's'}.{suffix}",
            tone="warning" if discovery_error else "success",
        )

    @app.post("/admin/models/providers/{provider_account_id}/test")
    async def model_config_test_provider(
        request: Request,
        provider_account_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        installation_id = _dashboard_installation_id(session, principal)
        provider = _get_scoped_provider_account(
            session,
            provider_account_id=provider_account_id,
            installation_id=installation_id,
        )
        if provider is None:
            return _redirect_with_notice(
                next_path, "Provider not found.", tone="danger"
            )
        runtime_settings, runtime_error = _load_runtime_settings()
        if runtime_settings is None:
            return _redirect_with_notice(
                next_path,
                f"Runtime settings are not available: {runtime_error or 'unknown error'}",
                tone="danger",
            )
        api_key = _provider_api_key(session, provider, runtime_settings)
        if api_key is None:
            return _redirect_with_notice(
                next_path,
                "Provider credentials are not available for testing.",
                tone="danger",
            )
        model_identifier = _optional_form_value(
            form, "model_identifier"
        ) or _provider_probe_model(session, provider)
        previous_value = _provider_account_audit_payload(provider)
        try:
            ok = check_litellm_provider_key(
                provider_kind=provider.provider_kind,
                api_key=api_key,
                model=model_identifier,
                api_base=provider.base_url,
            )
        except Exception as exc:
            provider.health_status = "down"
            _append_llm_config_audit(
                session,
                installation_id=provider.installation_id,
                principal=principal,
                action="update",
                entity_type="llm_provider_account",
                entity_id=str(provider.id),
                previous_value=previous_value,
                new_value={
                    **_provider_account_audit_payload(provider),
                    "operation": "test_provider",
                    "test_model": model_identifier,
                    "test_result": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            session.commit()
            return _redirect_with_notice(
                next_path,
                f"{provider.display_name} test failed: {type(exc).__name__}.",
                tone="danger",
            )
        provider.health_status = "ok" if ok else "down"
        _append_llm_config_audit(
            session,
            installation_id=provider.installation_id,
            principal=principal,
            action="update",
            entity_type="llm_provider_account",
            entity_id=str(provider.id),
            previous_value=previous_value,
            new_value={
                **_provider_account_audit_payload(provider),
                "operation": "test_provider",
                "test_model": model_identifier,
                "test_result": "ok" if ok else "failed",
            },
        )
        session.commit()
        return _redirect_with_notice(
            next_path,
            f"{provider.display_name} test {'passed' if ok else 'failed'}.",
            tone="success" if ok else "danger",
        )

    @app.post("/admin/models/providers/{provider_account_id}/import-models")
    async def model_config_import_provider_models(
        request: Request,
        provider_account_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        limit = _positive_int(_form_value(form, "limit"), default=24, maximum=100)
        installation_id = _dashboard_installation_id(session, principal)
        provider = _get_scoped_provider_account(
            session,
            provider_account_id=provider_account_id,
            installation_id=installation_id,
        )
        if provider is None:
            return _redirect_with_notice(
                next_path, "Provider not found.", tone="danger"
            )
        runtime_settings, _runtime_error = _load_runtime_settings()
        api_key = (
            _provider_api_key(session, provider, runtime_settings)
            if runtime_settings is not None
            else None
        )
        candidate_limit = _model_discovery_limit(
            provider.provider_kind,
            default=limit,
        )
        candidates = list(
            litellm_model_candidates(provider.provider_kind, limit=candidate_limit)
        )
        discovery_error: str | None = None
        if api_key is not None:
            try:
                endpoint_candidates = litellm_endpoint_model_candidates(
                    provider.provider_kind,
                    api_key=api_key,
                    api_base=provider.base_url,
                    limit=candidate_limit,
                )
                candidates = _merge_model_candidates(endpoint_candidates, candidates)
            except Exception as exc:
                discovery_error = type(exc).__name__
        imported_count, pricing_count = _upsert_model_candidates(
            session,
            provider=provider,
            candidates=tuple(candidates),
        )
        pricing_count += _backfill_model_pricing_for_provider(
            session,
            provider=provider,
            include_provider_catalog=True,
        )
        _append_llm_config_audit(
            session,
            installation_id=provider.installation_id,
            principal=principal,
            action="update",
            entity_type="llm_provider_account",
            entity_id=str(provider.id),
            previous_value=None,
            new_value={
                "operation": "import_models",
                "provider_account_id": str(provider.id),
                "provider_kind": provider.provider_kind,
                "imported_count": imported_count,
                "pricing_count": pricing_count,
                "candidate_count": len(candidates),
                "discovery_error": discovery_error,
            },
        )
        session.commit()
        tone = "warning" if discovery_error else "success"
        suffix = (
            f" Endpoint discovery failed with {discovery_error}."
            if discovery_error
            else ""
        )
        return _redirect_with_notice(
            next_path,
            f"Imported {imported_count} model row{'' if imported_count == 1 else 's'} and {pricing_count} pricing row{'' if pricing_count == 1 else 's'} for {provider.display_name}.{suffix}",
            tone=tone,
        )

    @app.post("/admin/models/providers/{provider_account_id}/update-pricing")
    async def model_config_update_provider_pricing(
        request: Request,
        provider_account_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        installation_id = _dashboard_installation_id(session, principal)
        provider = _get_scoped_provider_account(
            session,
            provider_account_id=provider_account_id,
            installation_id=installation_id,
        )
        if provider is None:
            return _redirect_with_notice(
                next_path, "Provider not found.", tone="danger"
            )

        pricing_count = _backfill_model_pricing_for_provider(
            session,
            provider=provider,
            include_provider_catalog=True,
            missing_only=True,
        )
        _append_llm_config_audit(
            session,
            installation_id=provider.installation_id,
            principal=principal,
            action="update",
            entity_type="llm_provider_account",
            entity_id=str(provider.id),
            previous_value=None,
            new_value={
                "operation": "update_missing_pricing",
                "provider_account_id": str(provider.id),
                "provider_kind": provider.provider_kind,
                "pricing_count": pricing_count,
            },
        )
        session.commit()
        return _redirect_with_notice(
            next_path,
            (
                f"Updated pricing for {pricing_count} model row{'' if pricing_count == 1 else 's'}."
                if pricing_count
                else "No missing pricing could be updated for this provider."
            ),
            tone="success" if pricing_count else "neutral",
        )

    @app.post("/admin/models/catalog")
    async def model_config_add_model(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        installation_id = _dashboard_installation_id(session, principal)
        try:
            provider_account_id = UUID(_form_value(form, "provider_account_id"))
        except ValueError:
            return _redirect_with_notice(
                next_path,
                "Choose a valid provider for this model.",
                tone="danger",
            )
        provider = _get_scoped_provider_account(
            session,
            provider_account_id=provider_account_id,
            installation_id=installation_id,
        )
        if provider is None:
            return _redirect_with_notice(
                next_path, "Provider not found.", tone="danger"
            )
        model_identifier = _form_value(form, "model_identifier")
        if not model_identifier:
            return _redirect_with_notice(
                next_path,
                "Model identifier is required.",
                tone="danger",
            )
        existing = session.scalar(
            select(LLMModelCatalog).where(
                LLMModelCatalog.provider_account_id == provider.id,
                LLMModelCatalog.model_identifier == model_identifier,
            )
        )
        if existing is not None:
            return _redirect_with_notice(
                next_path,
                "That model already exists for this provider.",
                tone="warning",
            )
        display_name = _form_value(form, "display_name") or model_identifier
        model = LLMModelCatalog(
            provider_account_id=provider.id,
            model_identifier=model_identifier,
            display_name=display_name,
            is_enabled=True,
            source="manual",
            capabilities_json={},
            metadata_json={"source": "dashboard_manual"},
        )
        session.add(model)
        session.flush()
        candidate = _local_litellm_candidate(provider.provider_kind, model_identifier)
        pricing_created = 0
        if candidate is not None:
            model.capabilities_json = candidate.capabilities
            model.metadata_json = {
                **model.metadata_json,
                "litellm_metadata": candidate.metadata,
            }
            pricing_created = _upsert_pricing_from_candidate(
                session,
                provider=provider,
                candidate=candidate,
            )
        _append_llm_config_audit(
            session,
            installation_id=provider.installation_id,
            principal=principal,
            action="create",
            entity_type="llm_model_catalog",
            entity_id=str(model.id),
            previous_value=None,
            new_value={
                "provider_account_id": str(provider.id),
                "model_identifier": model.model_identifier,
                "display_name": model.display_name,
                "source": model.source,
                "pricing_created": pricing_created,
            },
        )
        session.commit()
        return _redirect_with_notice(
            next_path,
            f"Added model {model.display_name}.",
        )

    @app.post("/admin/models/tiers/{tier}")
    async def model_config_update_tier(
        request: Request,
        tier: str,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "No installation scope is available for this model tier.",
                tone="danger",
            )
        if tier not in MODEL_TIER_VALUES:
            return _redirect_with_notice(
                next_path,
                "Unknown model tier.",
                tone="danger",
            )
        try:
            model_catalog_id = UUID(_form_value(form, "model_catalog_id"))
        except ValueError:
            return _redirect_with_notice(
                next_path,
                "Choose a valid model for this tier.",
                tone="danger",
            )
        model = _get_scoped_model_catalog(
            session,
            model_catalog_id=model_catalog_id,
            installation_id=installation_id,
        )
        if model is None:
            return _redirect_with_notice(
                next_path,
                "That model is not available for this installation.",
                tone="danger",
            )
        provider = session.get(LLMProviderAccount, model.provider_account_id)
        if provider is None or provider.status != "active" or not model.is_enabled:
            return _redirect_with_notice(
                next_path,
                "Only enabled models on active providers can be assigned to a tier.",
                tone="danger",
            )
        priority = _positive_int(_form_value(form, "priority"), default=1, maximum=5)
        _assign_llm_model_tier(
            session,
            installation_id=installation_id,
            tier=tier,
            model=model,
            priority=priority,
            principal=principal,
        )
        session.commit()
        route_label = "primary" if priority == 1 else f"fallback P{priority}"
        return _redirect_with_notice(
            next_path,
            f"{tier.replace('_', ' ').title()} {route_label} now routes to {model.display_name}.",
        )

    @app.post("/admin/models/catalog/{model_catalog_id}/assign-tier")
    async def model_config_assign_catalog_tier(
        request: Request,
        model_catalog_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path,
                "No installation scope is available for this model tier.",
                tone="danger",
            )
        tier = _form_value(form, "tier")
        if tier not in MODEL_TIER_VALUES:
            return _redirect_with_notice(
                next_path,
                "Choose a valid tier for this model.",
                tone="danger",
            )
        model = _get_scoped_model_catalog(
            session,
            model_catalog_id=model_catalog_id,
            installation_id=installation_id,
        )
        if model is None:
            return _redirect_with_notice(
                next_path,
                "That model is not available for this installation.",
                tone="danger",
            )
        provider = session.get(LLMProviderAccount, model.provider_account_id)
        if provider is None or provider.status != "active" or not model.is_enabled:
            return _redirect_with_notice(
                next_path,
                "Only enabled models on active providers can be assigned to a tier.",
                tone="danger",
            )
        priority = _positive_int(_form_value(form, "priority"), default=1, maximum=5)
        _assign_llm_model_tier(
            session,
            installation_id=installation_id,
            tier=tier,
            model=model,
            priority=priority,
            principal=principal,
        )
        session.commit()
        route_label = "primary" if priority == 1 else f"fallback P{priority}"
        return _redirect_with_notice(
            next_path,
            f"{tier.replace('_', ' ').title()} {route_label} now routes to {model.display_name}.",
        )

    @app.post("/admin/models/providers/{provider_account_id}/status")
    async def model_config_update_provider_status(
        request: Request,
        provider_account_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        new_status = _form_value(form, "status")
        if new_status not in {"active", "disabled"}:
            return _redirect_with_notice(
                next_path,
                "Provider status must be active or disabled.",
                tone="danger",
            )
        installation_id = _dashboard_installation_id(session, principal)
        provider = _get_scoped_provider_account(
            session,
            provider_account_id=provider_account_id,
            installation_id=installation_id,
        )
        if provider is None:
            return _redirect_with_notice(
                next_path,
                "Provider account not found.",
                tone="danger",
            )
        previous_value: dict[str, object] = {
            "status": provider.status,
            "provider_kind": provider.provider_kind,
            "display_name": provider.display_name,
        }
        provider.status = new_status
        _append_llm_config_audit(
            session,
            installation_id=provider.installation_id,
            principal=principal,
            action="enable" if new_status == "active" else "disable",
            entity_type="llm_provider_account",
            entity_id=str(provider.id),
            previous_value=previous_value,
            new_value={
                "status": provider.status,
                "provider_kind": provider.provider_kind,
                "display_name": provider.display_name,
            },
        )
        session.commit()
        return _redirect_with_notice(
            next_path,
            f"{provider.display_name} marked {new_status}.",
            tone="success" if new_status == "active" else "warning",
        )

    @app.post("/admin/models/catalog/{model_catalog_id}/status")
    async def model_config_update_model_status(
        request: Request,
        model_catalog_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(_form_value(form, "next") or "/admin/models")
        enabled_value = _form_value(form, "is_enabled")
        if enabled_value not in {"true", "false"}:
            return _redirect_with_notice(
                next_path,
                "Model status must be enabled or disabled.",
                tone="danger",
            )
        installation_id = _dashboard_installation_id(session, principal)
        model = _get_scoped_model_catalog(
            session,
            model_catalog_id=model_catalog_id,
            installation_id=installation_id,
        )
        if model is None:
            return _redirect_with_notice(
                next_path,
                "Model not found.",
                tone="danger",
            )
        previous_value = {
            "is_enabled": model.is_enabled,
            "model_identifier": model.model_identifier,
            "display_name": model.display_name,
        }
        model.is_enabled = enabled_value == "true"
        _append_llm_config_audit(
            session,
            installation_id=installation_id or _model_installation_id(session, model),
            principal=principal,
            action="enable" if model.is_enabled else "disable",
            entity_type="llm_model_catalog",
            entity_id=str(model.id),
            previous_value=previous_value,
            new_value={
                "is_enabled": model.is_enabled,
                "model_identifier": model.model_identifier,
                "display_name": model.display_name,
            },
        )
        session.commit()
        return _redirect_with_notice(
            next_path,
            f"{model.display_name} {'enabled' if model.is_enabled else 'disabled'}.",
            tone="success" if model.is_enabled else "warning",
        )

    @app.get("/composio", response_class=HTMLResponse)
    def composio_catalog(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        q: Annotated[str | None, Query()] = None,
        cursor: Annotated[str | None, Query()] = None,
        page_size: Annotated[int, Query(ge=1, le=100)] = 60,
        view: Annotated[str | None, Query()] = None,
    ) -> Response:
        runtime_settings, runtime_error = _load_runtime_settings()
        catalog_view = "list" if (view or "").strip().lower() == "list" else "card"
        catalog = get_composio_catalog_dashboard(
            session,
            runtime_settings=runtime_settings,
            query=q,
            cursor=cursor,
            limit=page_size,
        )
        catalog_query = _clean_query_params(
            {"q": q, "page_size": catalog.page_size or page_size, "view": catalog_view}
        )
        return templates.TemplateResponse(
            request=request,
            name="composio.html",
            context={
                **_dashboard_context(principal, active_page="composio"),
                "catalog": catalog,
                "runtime_error": runtime_error,
                "q": q or "",
                "cursor": cursor or "",
                "page_size": catalog.page_size or page_size,
                "page_size_options": (24, 60, 100),
                "catalog_view": catalog_view,
                "catalog_card_view_url": _url_with_query(
                    "/composio", {**catalog_query, "view": "card"}
                ),
                "catalog_list_view_url": _url_with_query(
                    "/composio", {**catalog_query, "view": "list"}
                ),
                "catalog_clear_url": _url_with_query(
                    "/composio",
                    {"page_size": catalog.page_size or page_size, "view": catalog_view},
                ),
                "catalog_next_url": _url_with_query(
                    "/composio",
                    {
                        **catalog_query,
                        "cursor": catalog.next_cursor,
                    },
                )
                if catalog.next_cursor
                else None,
                "catalog_restart_url": _url_with_query("/composio", catalog_query),
                "composio_catalog_url": "/composio",
                "composio_detail_base_url": "/composio",
                "integration_registry_url": "/integrations",
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

        base_path = "/composio" if principal.role == "admin" else "/me/composio"
        return _redirect_with_notice(
            f"{base_path}/{quote(connection.toolkit_slug, safe='')}",
            notice,
            tone=tone,
        )

    @app.post("/composio/connections/{connection_id}/disconnect")
    async def composio_disconnect(
        request: Request,
        connection_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        default_next_path = "/composio" if principal.role == "admin" else "/me/composio"
        next_path = _safe_next_path(form.get("next", [default_next_path])[0])
        connection = session.get(ComposioConnection, connection_id)
        if connection is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if not _can_manage_composio_connection(principal, connection):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

        if connection.connected_account_id:
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
                    "COMPOSIO_API_KEY is required before disconnecting the account.",
                    tone="danger",
                )
            client = ComposioClient(
                api_key=runtime_settings.composio_api_key,
                timeout_seconds=runtime_settings.composio_request_timeout_seconds,
            )
            try:
                disabled = client.set_connected_account_enabled(
                    connection.connected_account_id,
                    enabled=False,
                )
            except ComposioConnectionError as exc:
                return _redirect_with_notice(
                    next_path,
                    f"Could not disconnect Composio account: {str(exc)}",
                    tone="danger",
                )
            if not disabled:
                return _redirect_with_notice(
                    next_path,
                    "Composio did not confirm the account was disconnected.",
                    tone="danger",
                )

        metadata = dict(connection.metadata_json or {})
        metadata["disconnected_at"] = datetime.now(UTC).isoformat()
        metadata["disconnected_by"] = principal.display_name
        connection.status = "disabled"
        connection.metadata_json = metadata
        session.commit()
        return _redirect_with_notice(next_path, "Composio account disconnected.")

    @app.post("/composio/connections/{connection_id}/scope")
    async def composio_update_scope(
        request: Request,
        connection_id: UUID,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        default_next_path = "/composio" if principal.role == "admin" else "/me/composio"
        next_path = _safe_next_path(form.get("next", [default_next_path])[0])
        connection = session.get(ComposioConnection, connection_id)
        if connection is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if not _can_manage_composio_connection(principal, connection):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

        scope_type = _form_value(form, "visibility_scope_type") or "user"
        channel_scope_id = _form_value(form, "channel_scope_id")
        if principal.role != "admin":
            if scope_type != "user":
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
            channel_scope_id = ""

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

        previous_scope = {
            "type": connection.visibility_scope_type,
            "id": connection.visibility_scope_id,
        }
        connection.visibility_scope_type = scope_type
        connection.visibility_scope_id = _composio_scope_id(
            scope_type=scope_type,
            owner_slack_user_id=connection.owner_slack_user_id,
            channel_scope_id=channel_scope_id,
        )
        metadata = dict(connection.metadata_json or {})
        metadata["visibility_updated_at"] = datetime.now(UTC).isoformat()
        metadata["visibility_updated_by"] = principal.display_name
        metadata["previous_visibility_scope"] = previous_scope
        connection.metadata_json = metadata
        session.commit()
        return _redirect_with_notice(next_path, "Connection visibility updated.")

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
            next_path = f"/me/composio/{quote(normalized_slug, safe='')}"

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
        callback_url = f"{request.url_for('composio_callback')}?" + urlencode(
            {
                "connection_id": str(connection.id),
                "connection_token": callback_token,
            }
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
                "access_toolbar": {
                    "label": "Dashboard access controls",
                    "title": "Access Roster",
                    "count": _row_count_label(len(users), "dashboard user"),
                    "fields": (),
                },
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
                "me_toolbar": _date_toolbar(
                    action="/me",
                    title="Dashboard Window",
                    count_label=(
                        f"Showing {detail.task_count:,} tasks"
                        if detail is not None
                        else "No tasks"
                    ),
                    from_date=from_date,
                    to_date=to_date,
                    reset_url="/me",
                ),
                "from_date": from_date or "",
                "to_date": to_date or "",
            },
        )

    @app.get("/me/tasks", response_class=HTMLResponse)
    def me_tasks(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        q: Annotated[str | None, Query()] = None,
        status_filter: Annotated[str | None, Query(alias="status")] = None,
        channel: Annotated[str | None, Query()] = None,
        model: Annotated[str | None, Query()] = None,
        from_date: Annotated[str | None, Query(alias="from")] = None,
        to_date: Annotated[str | None, Query(alias="to")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        start = parse_date_bound(from_date)
        end = parse_date_bound(to_date, inclusive_end=True)
        task_filters = _task_filter_values(
            q=q,
            status=status_filter,
            channel=channel,
            user=None,
            model=model,
            from_date=from_date,
            to_date=to_date,
        )
        task_page = list_tasks(
            session,
            page=page,
            page_size=page_size,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
            start=start,
            end=end,
            query=q,
            status=status_filter,
            channel=channel,
            model=model,
        )
        task_query_params = _task_query_params(task_filters, page_size=page_size)
        return templates.TemplateResponse(
            request=request,
            name="tasks.html",
            context={
                **_dashboard_context(principal, active_page="me_tasks"),
                "task_page": task_page,
                "page_size": page_size,
                "task_filters": task_filters,
                "task_toolbar": _task_toolbar(
                    action="/me/tasks",
                    task_page=task_page,
                    task_filters=task_filters,
                    reset_url="/me/tasks",
                    member_scope=True,
                ),
                "task_previous_url": _page_url(
                    "/me/tasks", task_query_params, task_page.previous_page
                ),
                "task_next_url": _page_url(
                    "/me/tasks", task_query_params, task_page.next_page
                ),
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
        days: Annotated[int | None, Query(ge=1, le=366)] = None,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        start: datetime | None
        end: datetime | None
        if days is not None:
            end = datetime.now(UTC)
            start = end - timedelta(days=days)
        else:
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
                "usage_toolbar": _date_toolbar(
                    action="/me/usage",
                    title="Usage Window",
                    count_label=_usage_count_label(aggregate.total_calls),
                    from_date=from_date,
                    to_date=to_date,
                    reset_url="/me/usage",
                ),
            },
        )

    @app.get("/me/schedules", response_class=HTMLResponse)
    def me_schedules(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        schedule_status: Annotated[str, Query(alias="status")] = "all",
        page: Annotated[int, Query(ge=1)] = 1,
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        schedule_page = get_schedule_dashboard(
            session,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
            is_admin=False,
            view="my",
            status=schedule_status,
            page=page,
            base_path="/me/schedules",
        )
        return templates.TemplateResponse(
            request=request,
            name="schedules.html",
            context={
                **_dashboard_context(principal, active_page="me_schedules"),
                "schedule_page": schedule_page,
                "schedules_base_path": "/me/schedules",
                "schedules_detail_base_path": "/me/schedules",
                "schedules_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
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
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        runtime_settings, runtime_error = _load_runtime_settings()
        integration_dashboard = get_integration_dashboard(
            session=session,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
            installation_id=principal.installation_id,
            owner_slack_user_id=principal.slack_user_id,
        )
        return templates.TemplateResponse(
            request=request,
            name="integrations.html",
            context={
                **_dashboard_context(principal, active_page="me_integrations"),
                "integrations": integration_dashboard,
            },
        )

    @app.get("/me/composio", response_class=HTMLResponse)
    def me_composio_catalog(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        session: Annotated[Session, Depends(get_session)],
        q: Annotated[str | None, Query()] = None,
        cursor: Annotated[str | None, Query()] = None,
        page_size: Annotated[int, Query(ge=1, le=100)] = 60,
        view: Annotated[str | None, Query()] = None,
    ) -> Response:
        if principal.installation_id is None or principal.slack_user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        runtime_settings, runtime_error = _load_runtime_settings()
        catalog_view = "list" if (view or "").strip().lower() == "list" else "card"
        catalog = get_composio_catalog_dashboard(
            session,
            runtime_settings=runtime_settings,
            query=q,
            cursor=cursor,
            limit=page_size,
            installation_id=principal.installation_id,
            owner_slack_user_id=principal.slack_user_id,
        )
        catalog_query = _clean_query_params(
            {"q": q, "page_size": catalog.page_size or page_size, "view": catalog_view}
        )
        return templates.TemplateResponse(
            request=request,
            name="composio.html",
            context={
                **_dashboard_context(principal, active_page="me_composio"),
                "catalog": catalog,
                "runtime_error": runtime_error,
                "q": q or "",
                "cursor": cursor or "",
                "page_size": catalog.page_size or page_size,
                "page_size_options": (24, 60, 100),
                "catalog_view": catalog_view,
                "catalog_card_view_url": _url_with_query(
                    "/me/composio", {**catalog_query, "view": "card"}
                ),
                "catalog_list_view_url": _url_with_query(
                    "/me/composio", {**catalog_query, "view": "list"}
                ),
                "catalog_clear_url": _url_with_query(
                    "/me/composio",
                    {"page_size": catalog.page_size or page_size, "view": catalog_view},
                ),
                "catalog_next_url": _url_with_query(
                    "/me/composio",
                    {
                        **catalog_query,
                        "cursor": catalog.next_cursor,
                    },
                )
                if catalog.next_cursor
                else None,
                "catalog_restart_url": _url_with_query("/me/composio", catalog_query),
                "composio_catalog_url": "/me/composio",
                "composio_detail_base_url": "/me/composio",
                "integration_registry_url": "/me/integrations",
                "member_scope": True,
            },
        )

    @app.get("/me/composio/{toolkit_slug}", response_class=HTMLResponse)
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
                **_dashboard_context(principal, active_page="me_composio"),
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
                "user_toolbar": _date_toolbar(
                    action=f"/users/{quote(slack_user_id, safe='')}",
                    title="User Inspection Window",
                    count_label=_row_count_label(detail.task_count, "task"),
                    from_date=from_date,
                    to_date=to_date,
                    reset_url=f"/users/{quote(slack_user_id, safe='')}",
                ),
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

    def _autonomy_default_level() -> str:
        runtime_settings, _ = _load_runtime_settings()
        if runtime_settings is None:
            return "balanced"
        return runtime_settings.autonomy_default_level

    @app.get("/autonomy", response_class=HTMLResponse)
    def autonomy(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        installation_id = _dashboard_installation_id(session, principal)
        dashboard = get_autonomy_dashboard(
            session,
            installation_id=installation_id,
            default_level=_autonomy_default_level(),
        )
        return templates.TemplateResponse(
            request=request,
            name="autonomy.html",
            context={
                **_dashboard_context(principal, active_page="autonomy"),
                "autonomy": dashboard,
                "autonomy_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.post("/autonomy/workspace")
    async def autonomy_set_workspace(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/autonomy"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No installation is available.", tone="danger"
            )
        try:
            set_workspace_level(
                session,
                installation_id=installation_id,
                level=form.get("level", [""])[0],
                by_user_id=dashboard_actor(principal.display_name),
            )
        except ValueError as exc:
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        session.commit()
        return _redirect_with_notice(next_path, "Workspace autonomy level updated.")

    @app.post("/autonomy/channel")
    async def autonomy_set_channel(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/autonomy"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No installation is available.", tone="danger"
            )
        try:
            set_channel_level(
                session,
                installation_id=installation_id,
                channel_id=form.get("channel_id", [""])[0],
                level=form.get("level", [""])[0],
                by_user_id=dashboard_actor(principal.display_name),
            )
        except ValueError as exc:
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        session.commit()
        return _redirect_with_notice(next_path, "Channel autonomy override saved.")

    @app.post("/autonomy/channel/clear")
    async def autonomy_clear_channel(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        next_path = _safe_next_path(form.get("next", ["/autonomy"])[0])
        installation_id = _dashboard_installation_id(session, principal)
        if installation_id is None:
            return _redirect_with_notice(
                next_path, "No installation is available.", tone="danger"
            )
        try:
            removed = clear_channel_level(
                session,
                installation_id=installation_id,
                channel_id=form.get("channel_id", [""])[0],
            )
        except ValueError as exc:
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        session.commit()
        notice = (
            "Channel override removed." if removed else "No channel override to remove."
        )
        return _redirect_with_notice(next_path, notice, tone="neutral")

    @app.get("/schedules", response_class=HTMLResponse)
    def schedules(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        view: Annotated[str, Query()] = "all",
        schedule_status: Annotated[str, Query(alias="status")] = "all",
        page: Annotated[int, Query(ge=1)] = 1,
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        schedule_page = get_schedule_dashboard(
            session,
            installation_id=principal.installation_id,
            slack_user_id=principal.slack_user_id,
            is_admin=True,
            view=view,
            status=schedule_status,
            page=page,
            base_path="/schedules",
        )
        return templates.TemplateResponse(
            request=request,
            name="schedules.html",
            context={
                **_dashboard_context(principal, active_page="schedules"),
                "schedule_page": schedule_page,
                "schedules_base_path": "/schedules",
                "schedules_detail_base_path": "/schedules",
                "schedules_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.get("/schedules/{schedule_id}", response_class=HTMLResponse)
    @app.get("/me/schedules/{schedule_id}", response_class=HTMLResponse)
    def schedule_detail(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        schedule_id: UUID,
        notice: Annotated[str | None, Query()] = None,
        notice_tone: Annotated[str, Query()] = "success",
    ) -> Response:
        is_admin = principal.role == "admin"
        detail_base_path = "/schedules" if is_admin else "/me/schedules"
        active_page = "schedules" if is_admin else "me_schedules"
        try:
            detail = get_schedule_detail(
                session,
                schedule_id=schedule_id,
                installation_id=principal.installation_id,
                slack_user_id=principal.slack_user_id,
                is_admin=is_admin,
            )
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        return templates.TemplateResponse(
            request=request,
            name="schedule_detail.html",
            context={
                **_dashboard_context(principal, active_page=active_page),
                "detail": detail,
                "schedules_base_path": detail_base_path,
                "schedule_edit_path": f"{detail_base_path}/{schedule_id}/edit",
                "schedules_return_path": _request_path(request),
                "notice": notice,
                "notice_tone": _notice_tone(notice_tone),
            },
        )

    @app.post("/schedules/{schedule_id}/edit")
    @app.post("/me/schedules/{schedule_id}/edit")
    async def schedule_edit(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        schedule_id: UUID,
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        is_admin = principal.role == "admin"
        detail_base_path = "/schedules" if is_admin else "/me/schedules"
        next_path = _safe_next_path(
            form.get("next", [f"{detail_base_path}/{schedule_id}"])[0]
        )
        try:
            notice = update_schedule_from_dashboard(
                session,
                schedule_id=schedule_id,
                installation_id=principal.installation_id,
                slack_user_id=principal.slack_user_id,
                is_admin=is_admin,
                actor=principal.display_name,
                title=_form_value(form, "title"),
                schedule_text=_form_value(form, "schedule_text"),
                task_input=_form_value(form, "task_input"),
                planned_cost_ceiling_usd=_form_value(form, "planned_cost_ceiling_usd"),
                delivery_kind=_form_value(form, "delivery_kind"),
                delivery_slack_user_id=_form_value(form, "delivery_slack_user_id"),
                delivery_slack_channel_id=_form_value(
                    form, "delivery_slack_channel_id"
                ),
                delivery_slack_thread_ts=_form_value(form, "delivery_slack_thread_ts"),
                artifact_delivery_policy=_form_value(form, "artifact_delivery_policy"),
            )
        except (LookupError, PermissionError) as exc:
            session.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, notice)

    @app.post("/schedules/{schedule_id}/{action}")
    async def schedule_action(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
        principal: Annotated[DashboardPrincipal, Depends(require_principal)],
        schedule_id: UUID,
        action: str,
    ) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
        default_next = "/schedules" if principal.role == "admin" else "/me/schedules"
        next_path = _safe_next_path(form.get("next", [default_next])[0])
        try:
            notice = apply_schedule_action(
                session,
                schedule_id=schedule_id,
                action=action,
                installation_id=principal.installation_id,
                slack_user_id=principal.slack_user_id,
                is_admin=principal.role == "admin",
            )
        except ValueError as exc:
            session.rollback()
            return _redirect_with_notice(next_path, str(exc), tone="danger")
        return _redirect_with_notice(next_path, notice)

    @app.get("/playground", response_class=HTMLResponse)
    def playground(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        tasks = list(
            session.scalars(
                select(Task)
                .where(
                    Task.slack_channel_id == "playground",
                    Task.identity_kind == "manual",
                )
                .order_by(Task.created_at.desc())
                .limit(20)
            ).all()
        )
        return templates.TemplateResponse(
            request=request,
            name="playground.html",
            context={
                **_dashboard_context(principal, active_page="playground"),
                "tasks": tasks,
            },
        )

    @app.post("/playground/run")
    async def playground_run(
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
        session: Annotated[Session, Depends(get_session)],
    ) -> Response:
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")
        params = parse_qs(body_str)
        prompt = params.get("prompt", [None])[0]
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            try:
                json_data = json.loads(body_str)
                prompt = json_data.get("prompt")
            except Exception:
                pass

        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required")

        prompt = prompt.strip()

        installation_id = principal.installation_id
        if installation_id is None:
            installation_id = session.scalar(
                select(Installation.id)
                .order_by(Installation.created_at.desc())
                .limit(1)
            )
            if installation_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="No workspace installation found in database. Please register Kortny to a Slack workspace first.",
                )

        slack_user_id = (
            principal.slack_user_id or _default_slack_owner_id(session) or "U00000000"
        )

        from kortny.tasks.repository import TaskRepository

        task_repo = TaskRepository(session, commit_after_write=True)
        task = task_repo.create_task(
            installation_id=installation_id,
            slack_channel_id="playground",
            slack_user_id=slack_user_id,
            input=prompt,
            slack_event_id=None,
            slack_thread_ts=None,
            slack_message_ts=None,
            source_surface="playground",
        )

        return JSONResponse(content={"task_id": str(task.id)})

    @app.get("/playground/{task_id}/stream")
    def playground_stream(
        task_id: UUID,
        request: Request,
        principal: Annotated[DashboardPrincipal, Depends(require_admin)],
    ) -> Response:
        import asyncio

        from fastapi.responses import StreamingResponse

        async def event_generator() -> AsyncIterator[str]:
            session_factory = request.app.state.session_factory
            last_seq = 0
            done = False

            yield f"data: {json.dumps({'message': 'connected'})}\n\n"

            while not done:

                def query_events(
                    current_last_seq: int,
                ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
                    with session_factory() as session:
                        task = session.get(Task, task_id)
                        if not task:
                            return None, []

                        events = list(
                            session.scalars(
                                select(TaskEvent)
                                .where(
                                    TaskEvent.task_id == task_id,
                                    TaskEvent.seq > current_last_seq,
                                )
                                .order_by(TaskEvent.seq.asc())
                            ).all()
                        )
                        event_data = [
                            {
                                "seq": event.seq,
                                "type": event.type.value,
                                "payload": event.payload,
                                "created_at": event.created_at.isoformat(),
                            }
                            for event in events
                        ]
                        return {
                            "status": task.status.value,
                            "total_cost": str(task.total_cost_usd),
                            "total_input_tokens": task.total_input_tokens,
                            "total_output_tokens": task.total_output_tokens,
                        }, event_data

                try:
                    task_info, new_events = await asyncio.to_thread(
                        query_events, last_seq
                    )
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                    break

                if task_info is None:
                    yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                    break

                for ev in new_events:
                    yield f"data: {json.dumps({'event': ev})}\n\n"
                    last_seq = ev["seq"]

                status = task_info["status"]
                if status in {"succeeded", "failed", "cancelled"}:
                    try:
                        _, final_events = await asyncio.to_thread(
                            query_events, last_seq
                        )
                        for ev in final_events:
                            yield f"data: {json.dumps({'event': ev})}\n\n"
                    except Exception:
                        pass

                    yield f"data: {json.dumps({'status': status, 'finished': True, 'cost': task_info['total_cost'], 'input_tokens': task_info['total_input_tokens'], 'output_tokens': task_info['total_output_tokens']})}\n\n"
                    done = True
                else:
                    yield f"data: {json.dumps({'status': status, 'finished': False, 'cost': task_info['total_cost'], 'input_tokens': task_info['total_input_tokens'], 'output_tokens': task_info['total_output_tokens']})}\n\n"
                    await asyncio.sleep(0.5)

        return StreamingResponse(event_generator(), media_type="text/event-stream")


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


def require_dashboard_home(
    principal: Annotated[DashboardPrincipal, Depends(require_principal)],
) -> DashboardPrincipal:
    """Route logged-in members from admin home to their dashboard."""

    if principal.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/me"},
        )
    return principal


def require_user(request: Request) -> str:
    """Legacy helper for tests and small call sites that only need a name."""

    return require_principal(request).display_name


def get_session(request: Request) -> Iterator[Session]:
    """Yield a database session for dashboard requests."""

    factory = cast(sessionmaker[Session], request.app.state.session_factory)
    with factory() as session:
        yield session


def _task_filter_values(
    *,
    q: str | None,
    status: str | None,
    channel: str | None,
    user: str | None,
    model: str | None,
    from_date: str | None,
    to_date: str | None,
) -> dict[str, str]:
    return {
        "q": (q or "").strip(),
        "status": (status or "").strip(),
        "channel": (channel or "").strip(),
        "user": (user or "").strip(),
        "model": (model or "").strip(),
        "from": (from_date or "").strip(),
        "to": (to_date or "").strip(),
    }


def _task_query_params(
    task_filters: dict[str, str],
    *,
    page_size: int,
) -> dict[str, str]:
    return _clean_query_params(
        {
            **task_filters,
            "page_size": page_size,
        }
    )


def _task_toolbar(
    *,
    action: str,
    task_page: TaskListPage,
    task_filters: dict[str, str],
    reset_url: str,
    member_scope: bool = False,
) -> dict[str, object]:
    fields: list[dict[str, object]] = [
        {
            "type": "search",
            "name": "q",
            "label": "Search",
            "value": task_filters["q"],
            "placeholder": "request, result, user, channel",
        },
        {
            "type": "select",
            "name": "status",
            "label": "Status",
            "value": task_filters["status"],
            "options": (
                {"value": "", "label": "All statuses"},
                *(
                    {
                        "value": task_status.value,
                        "label": task_status.value.replace("_", " ").title(),
                    }
                    for task_status in TaskStatus
                ),
            ),
        },
        {
            "type": "search",
            "name": "channel",
            "label": "Channel",
            "value": task_filters["channel"],
            "placeholder": "C123 or #ops",
        },
        {
            "type": "search",
            "name": "model",
            "label": "Model",
            "value": task_filters["model"],
            "placeholder": "gpt, claude, router",
        },
        {
            "type": "date",
            "name": "from",
            "label": "From",
            "value": task_filters["from"],
        },
        {
            "type": "date",
            "name": "to",
            "label": "To",
            "value": task_filters["to"],
        },
    ]
    if not member_scope:
        fields.insert(
            3,
            {
                "type": "search",
                "name": "user",
                "label": "User",
                "value": task_filters["user"],
                "placeholder": "U123 or name",
            },
        )
    return {
        "label": "Task filters",
        "title": "Task Search",
        "count": (
            f"Showing {task_page.first_item:,}-{task_page.last_item:,} "
            f"of {task_page.total_count:,} tasks"
        ),
        "action": action,
        "reset_url": reset_url,
        "submit_label": "Apply",
        "fields": tuple(fields),
    }


def _date_toolbar(
    *,
    action: str,
    title: str,
    count_label: str,
    from_date: str | None,
    to_date: str | None,
    reset_url: str,
) -> dict[str, object]:
    return {
        "label": title,
        "title": title,
        "count": count_label,
        "action": action,
        "reset_url": reset_url,
        "submit_label": "Apply",
        "fields": (
            {
                "type": "date",
                "name": "from",
                "label": "From",
                "value": (from_date or "").strip(),
            },
            {
                "type": "date",
                "name": "to",
                "label": "To",
                "value": (to_date or "").strip(),
            },
        ),
    }


def _search_toolbar(
    *,
    action: str,
    name: str,
    value: str | None,
    title: str,
    count_label: str,
    placeholder: str,
    reset_url: str,
) -> dict[str, object]:
    return {
        "label": title,
        "title": title,
        "count": count_label,
        "action": action,
        "reset_url": reset_url,
        "submit_label": "Search",
        "fields": (
            {
                "type": "search",
                "name": name,
                "label": "Search",
                "value": (value or "").strip(),
                "placeholder": placeholder,
            },
        ),
    }


def _page_url(
    path: str,
    params: Mapping[str, object],
    page: int | None,
) -> str | None:
    if page is None:
        return None
    return _url_with_query(path, {**params, "page": page})


def _url_with_query(path: str, params: Mapping[str, object]) -> str:
    cleaned = _clean_query_params(params)
    if not cleaned:
        return path
    return f"{path}?{urlencode(cleaned)}"


def _clean_query_params(params: Mapping[str, object]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        string_value = str(value).strip()
        if not string_value or string_value == "None":
            continue
        cleaned[key] = string_value
    return cleaned


def _row_count_label(count: int, noun: str) -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def _usage_count_label(count: int) -> str:
    return f"{count:,} LLM call{'' if count == 1 else 's'}"


def _catalog_count_label(visible_count: int, total_items: int | None) -> str:
    if total_items is None:
        return f"{visible_count:,} shown"
    return f"{visible_count:,} shown of {total_items:,}"


def _form_value(form: dict[str, list[str]], name: str) -> str:
    value = form.get(name, [""])[0]
    return value.strip()


def _optional_form_value(form: dict[str, list[str]], name: str) -> str | None:
    value = _form_value(form, name)
    return value or None


def _positive_int(value: str, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed < 1:
        return default
    return min(parsed, maximum)


def _model_discovery_limit(provider_kind: str, *, default: int) -> int | None:
    if provider_kind == "openrouter":
        return None
    return default


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

    toolkit = client.get_toolkit(toolkit_slug)
    managed_schemes = {
        _normalize_auth_scheme(scheme) for scheme in toolkit.managed_auth_schemes
    }
    if "oauth2" in managed_schemes:
        auth_config = client.create_managed_auth_config(toolkit_slug=toolkit_slug)
        return auth_config.id, "created_managed"

    custom_scheme = _hosted_custom_auth_scheme(toolkit.auth_schemes)
    if custom_scheme is None:
        raise ComposioConnectionError(
            "This toolkit does not expose a hosted API-key style auth flow. "
            "Create an auth config in Composio and select it from Advanced options."
        )
    auth_config = client.create_custom_auth_config(
        toolkit_slug=toolkit_slug,
        auth_scheme=custom_scheme,
    )
    source = (
        f"created_{_normalize_auth_scheme(auth_config.auth_scheme or custom_scheme)}"
    )
    return auth_config.id, source


def _normalize_auth_scheme(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _hosted_custom_auth_scheme(auth_schemes: tuple[str, ...]) -> str | None:
    priority = (
        "api_key",
        "bearer_token",
        "bearer",
        "basic",
        "basic_auth",
        "jwt",
    )
    normalized_by_scheme = {
        _normalize_auth_scheme(scheme): scheme.strip().upper()
        for scheme in auth_schemes
        if scheme.strip()
    }
    for candidate in priority:
        scheme = normalized_by_scheme.get(candidate)
        if scheme:
            return scheme
    return None


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


def _embedding_index_for(session: Session) -> EmbeddingIndex | None:
    """Build an embedding index from runtime settings, or None if unavailable.

    Thin dashboard wrapper over ``embedding_index_from_settings`` that resolves
    the runtime settings first. Failure-isolated: any missing config simply
    skips embedding (the lazy per-task ranker backstops).
    """

    settings, _error = _load_runtime_settings()
    if settings is None:
        return None
    return embedding_index_from_settings(session, settings)


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
        "dashboard_user_id": principal.dashboard_user_id,
    }


def _dashboard_installation_id(
    session: Session,
    principal: DashboardPrincipal,
) -> UUID | None:
    if principal.installation_id is not None:
        return principal.installation_id
    installation_ids = tuple(
        session.scalars(
            select(Installation.id).order_by(Installation.created_at).limit(2)
        )
    )
    if len(installation_ids) == 1:
        return installation_ids[0]
    return None


def _get_scoped_provider_account(
    session: Session,
    *,
    provider_account_id: UUID,
    installation_id: UUID | None,
) -> LLMProviderAccount | None:
    provider = session.get(LLMProviderAccount, provider_account_id)
    if provider is None:
        return None
    if installation_id is not None and provider.installation_id != installation_id:
        return None
    return provider


def _get_scoped_model_catalog(
    session: Session,
    *,
    model_catalog_id: UUID,
    installation_id: UUID | None,
) -> LLMModelCatalog | None:
    model = session.get(LLMModelCatalog, model_catalog_id)
    if model is None:
        return None
    provider = session.get(LLMProviderAccount, model.provider_account_id)
    if provider is None:
        return None
    if installation_id is not None and provider.installation_id != installation_id:
        return None
    return model


def _model_installation_id(session: Session, model: LLMModelCatalog) -> UUID:
    provider = session.get(LLMProviderAccount, model.provider_account_id)
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return provider.installation_id


def _normalize_provider_kind(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        return None
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    if any(char not in allowed_chars for char in normalized):
        return None
    return normalized


def _provider_api_key(
    session: Session,
    provider: LLMProviderAccount,
    settings: Settings | None,
) -> str | None:
    if settings is None:
        return None
    metadata = (
        provider.metadata_json if isinstance(provider.metadata_json, dict) else {}
    )
    credential_source = metadata.get("credential_source")
    if credential_source == ENV_CREDENTIAL_SOURCE:
        return settings.llm_api_key
    if provider.encrypted_secret_id is None:
        return None
    resolver = secret_resolver_from_settings(session, settings=settings)
    if resolver is None:
        return None
    try:
        return resolver(provider.encrypted_secret_id)
    except Exception:
        return None


def _provider_probe_model(session: Session, provider: LLMProviderAccount) -> str:
    existing_model = session.scalar(
        select(LLMModelCatalog.model_identifier)
        .where(
            LLMModelCatalog.provider_account_id == provider.id,
            LLMModelCatalog.is_enabled.is_(True),
        )
        .order_by(LLMModelCatalog.created_at.asc())
        .limit(1)
    )
    return default_probe_model(provider.provider_kind, fallback=existing_model)


def _merge_model_candidates(
    primary: tuple[LiteLLMModelCandidate, ...],
    fallback: list[LiteLLMModelCandidate],
) -> list[LiteLLMModelCandidate]:
    merged: list[LiteLLMModelCandidate] = []
    seen: set[str] = set()
    for candidate in (*primary, *fallback):
        if candidate.model_identifier in seen:
            continue
        seen.add(candidate.model_identifier)
        merged.append(candidate)
    return merged


def _upsert_model_candidates(
    session: Session,
    *,
    provider: LLMProviderAccount,
    candidates: tuple[LiteLLMModelCandidate, ...],
) -> tuple[int, int]:
    imported_count = 0
    pricing_count = 0
    for candidate in candidates:
        existing = session.scalar(
            select(LLMModelCatalog).where(
                LLMModelCatalog.provider_account_id == provider.id,
                LLMModelCatalog.model_identifier == candidate.model_identifier,
            )
        )
        if existing is None:
            existing = LLMModelCatalog(
                provider_account_id=provider.id,
                model_identifier=candidate.model_identifier,
                display_name=candidate.display_name,
                is_enabled=True,
                source=candidate.source,
                capabilities_json=candidate.capabilities,
                metadata_json={
                    "source": candidate.source,
                    "litellm_metadata": candidate.metadata,
                },
            )
            session.add(existing)
            imported_count += 1
        else:
            existing.capabilities_json = {
                **(
                    existing.capabilities_json
                    if isinstance(existing.capabilities_json, dict)
                    else {}
                ),
                **candidate.capabilities,
            }
            existing.metadata_json = {
                **(
                    existing.metadata_json
                    if isinstance(existing.metadata_json, dict)
                    else {}
                ),
                "litellm_metadata": candidate.metadata,
            }
        pricing_count += _upsert_pricing_from_candidate(
            session,
            provider=provider,
            candidate=candidate,
        )
    return imported_count, pricing_count


def _backfill_model_pricing_for_provider(
    session: Session,
    *,
    provider: LLMProviderAccount,
    include_provider_catalog: bool,
    missing_only: bool = False,
) -> int:
    pricing_count = 0
    models = session.scalars(
        select(LLMModelCatalog).where(
            LLMModelCatalog.provider_account_id == provider.id
        )
    ).all()
    for model in models:
        if missing_only and _model_has_pricing(
            session,
            provider=provider,
            model_identifier=model.model_identifier,
        ):
            continue
        candidate = model_candidate_for_identifier(
            provider.provider_kind,
            model.model_identifier,
            include_provider_catalog=include_provider_catalog,
        )
        if candidate is None:
            continue
        if candidate.display_name and model.display_name == model.model_identifier:
            model.display_name = candidate.display_name
        model.capabilities_json = {
            **(
                model.capabilities_json
                if isinstance(model.capabilities_json, dict)
                else {}
            ),
            **candidate.capabilities,
        }
        model.metadata_json = {
            **(model.metadata_json if isinstance(model.metadata_json, dict) else {}),
            "litellm_metadata": candidate.metadata,
        }
        pricing_count += _upsert_pricing_from_candidate(
            session,
            provider=provider,
            candidate=candidate,
        )
    return pricing_count


def _model_has_pricing(
    session: Session,
    *,
    provider: LLMProviderAccount,
    model_identifier: str,
) -> bool:
    return (
        session.scalar(
            select(LLMModelPricing.id)
            .where(
                LLMModelPricing.provider_account_id == provider.id,
                LLMModelPricing.model_identifier == model_identifier,
            )
            .limit(1)
        )
        is not None
    )


def _upsert_pricing_from_candidate(
    session: Session,
    *,
    provider: LLMProviderAccount,
    candidate: LiteLLMModelCandidate,
) -> int:
    input_price = _valid_model_price_per_mtok(candidate.input_price_per_mtok)
    output_price = _valid_model_price_per_mtok(candidate.output_price_per_mtok)
    if input_price is None and output_price is None:
        return 0
    existing_pricing = session.scalar(
        select(LLMModelPricing.id)
        .where(
            LLMModelPricing.provider_account_id == provider.id,
            LLMModelPricing.model_identifier == candidate.model_identifier,
            LLMModelPricing.pricing_source == "litellm_catalog",
        )
        .limit(1)
    )
    if existing_pricing is not None:
        return 0
    session.add(
        LLMModelPricing(
            provider_account_id=provider.id,
            model_identifier=candidate.model_identifier,
            input_price_per_mtok=input_price,
            output_price_per_mtok=output_price,
            currency="USD",
            pricing_source="litellm_catalog",
            metadata_json={
                "source": candidate.source,
                "litellm_metadata": candidate.metadata,
            },
        )
    )
    return 1


def _valid_model_price_per_mtok(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    if value < MODEL_PRICE_PER_MTOK_MIN or value > MODEL_PRICE_PER_MTOK_MAX:
        return None
    return value


def _local_litellm_candidate(
    provider_kind: str,
    model_identifier: str,
) -> LiteLLMModelCandidate | None:
    for candidate in litellm_model_candidates(provider_kind, limit=500):
        if candidate.model_identifier == model_identifier:
            return candidate
    return None


def _provider_account_audit_payload(
    provider: LLMProviderAccount,
) -> dict[str, object]:
    metadata = (
        provider.metadata_json if isinstance(provider.metadata_json, dict) else {}
    )
    return {
        "provider_kind": provider.provider_kind,
        "display_name": provider.display_name,
        "status": provider.status,
        "health_status": provider.health_status,
        "base_url_configured": provider.base_url is not None,
        "credential_source": metadata.get("credential_source"),
        "source": metadata.get("source"),
        "api_version_configured": bool(metadata.get("api_version")),
        "encrypted_secret_id": str(provider.encrypted_secret_id)
        if provider.encrypted_secret_id is not None
        else None,
    }


def _tier_assignment_audit_payload(
    assignment: LLMTierAssignment | None,
    session: Session,
) -> dict[str, object] | None:
    if assignment is None:
        return None
    model = session.get(LLMModelCatalog, assignment.model_catalog_id)
    provider = (
        session.get(LLMProviderAccount, model.provider_account_id)
        if model is not None
        else None
    )
    return {
        "tier": assignment.tier,
        "priority": assignment.priority,
        "is_active": assignment.is_active,
        "model_catalog_id": str(assignment.model_catalog_id),
        "model_identifier": model.model_identifier if model is not None else None,
        "provider_account_id": str(provider.id) if provider is not None else None,
        "provider_kind": provider.provider_kind if provider is not None else None,
    }


def _assign_llm_model_tier(
    session: Session,
    *,
    installation_id: UUID,
    tier: str,
    model: LLMModelCatalog,
    priority: int,
    principal: DashboardPrincipal,
) -> LLMTierAssignment:
    assignment = session.scalar(
        select(LLMTierAssignment).where(
            LLMTierAssignment.installation_id == installation_id,
            LLMTierAssignment.tier == tier,
            LLMTierAssignment.priority == priority,
        )
    )
    previous_value = _tier_assignment_audit_payload(assignment, session)
    if assignment is None:
        assignment = LLMTierAssignment(
            installation_id=installation_id,
            tier=tier,
            model_catalog_id=model.id,
            priority=priority,
            is_active=True,
        )
        session.add(assignment)
        action = "create"
    else:
        assignment.model_catalog_id = model.id
        assignment.is_active = True
        action = "update"
    session.flush()
    _append_llm_config_audit(
        session,
        installation_id=installation_id,
        principal=principal,
        action=action,
        entity_type="llm_tier_assignment",
        entity_id=str(assignment.id),
        previous_value=previous_value,
        new_value=_tier_assignment_audit_payload(assignment, session),
    )
    return assignment


def _append_llm_config_audit(
    session: Session,
    *,
    installation_id: UUID,
    principal: DashboardPrincipal,
    action: str,
    entity_type: str,
    entity_id: str,
    previous_value: dict[str, object] | None,
    new_value: dict[str, object] | None,
) -> None:
    session.add(
        LLMConfigAudit(
            installation_id=installation_id,
            actor_slack_user_id=principal.slack_user_id or principal.display_name,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            previous_value=previous_value,
            new_value=new_value,
        )
    )


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


def _graph_refresh_notice(result: object) -> str:
    queued_count = getattr(result, "queued_count", 0)
    skipped_count = getattr(result, "skipped_count", 0)
    known_channel_count = getattr(result, "known_channel_count", 0)
    deterministic_entity_count = getattr(result, "deterministic_entity_count", 0)
    deterministic_edge_count = getattr(result, "deterministic_edge_count", 0)
    pieces = [
        f"Queued {queued_count:,} graph refresh assessment"
        f"{'' if queued_count == 1 else 's'}"
    ]
    deterministic_count = deterministic_entity_count + deterministic_edge_count
    if deterministic_count:
        pieces.append(f"projected {deterministic_count:,} trusted Slack graph facts")
    if skipped_count:
        pieces.append(f"skipped {skipped_count:,} already active or recent")
    pieces.append(f"across {known_channel_count:,} known active channel")
    return "; ".join(pieces) + f"{'' if known_channel_count == 1 else 's'}."


def _witness_run_notice(result: WitnessRunResult) -> str:
    projected_count = len(result.projections)
    created_count = sum(outcome.created_count for outcome in result.projections)
    updated_count = sum(outcome.updated_count for outcome in result.projections)
    skipped_count = sum(outcome.skipped_count for outcome in result.projections)
    if projected_count == 0:
        return "Witness scan complete: no active channel profiles needed a refresh."
    return (
        "Witness scan complete: "
        f"scanned {projected_count:,} profile"
        f"{'' if projected_count == 1 else 's'}, "
        f"created {created_count:,} candidate"
        f"{'' if created_count == 1 else 's'}, "
        f"updated {updated_count:,}, skipped {skipped_count:,}."
    )


def _materialize_accepted_candidate(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    actor: str,
) -> AutomationOutcome:
    """Run HIG-224 materialization after a dashboard accept.

    Never undoes the acceptance: configuration problems degrade to a
    status-only accept instead of raising into the action handler.
    """

    try:
        runtime_settings = load_settings()
    except SettingsError:
        return AutomationOutcome(kind="disabled")
    if not runtime_settings.witness_automation_enabled:
        return AutomationOutcome(kind="disabled")
    return materialize_acceptance(
        session,
        runtime_settings,
        candidate,
        accepted_by=actor,
        slack_client=cast(
            WitnessSlackClient,
            WebClient(token=runtime_settings.slack_bot_token),
        ),
    )


def _witness_accept_notice(outcome: AutomationOutcome) -> tuple[str, str]:
    if outcome.kind == "one_shot" and outcome.task_id is not None:
        return ("Witness candidate accepted; Kortny is running it now.", "success")
    if outcome.kind == "recurring" and outcome.schedule_id is not None:
        return (
            "Witness candidate accepted; schedule drafted and waiting for "
            "Slack confirmation.",
            "success",
        )
    if outcome.failure_reason is not None:
        return (
            "Witness candidate marked useful, but automation drafting failed: "
            f"{outcome.failure_reason}",
            "warning",
        )
    return ("Witness candidate marked useful.", "success")


def _witness_autopilot_notice(result: WitnessAutopilotRunResult) -> str:
    if result.reviewed_count == 0:
        return "Witness autopilot complete: no due candidates met the review threshold."
    return (
        "Witness autopilot complete: "
        f"reviewed {result.reviewed_count:,}, "
        f"started {result.executed_count:,} proactive task"
        f"{'' if result.executed_count == 1 else 's'}, "
        f"deferred {result.deferred_count:,}, dismissed {result.dismissed_count:,}."
    )


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


def _mcp_attempt_discovery(
    session: Session,
    *,
    server_id: uuid.UUID,
    installation_id: uuid.UUID,
    runtime_settings: Settings | None,
) -> str:
    """Run discover_server_tools against a registered MCP server.

    Returns a human-readable notice string (success or error).
    Does NOT commit — caller must commit after.
    """
    from kortny.dashboard.mcp_actions import upsert_discovered_tools as _upsert
    from kortny.db.models import McpServer as _McpServerORM

    server = session.get(_McpServerORM, server_id)
    if server is None or server.installation_id != installation_id:
        return "MCP server not found."

    encryption_key = (
        runtime_settings.encryption_key if runtime_settings is not None else None
    )
    if encryption_key is None:
        _upsert(
            session,
            server=server,
            discovered=[],
            error="ENCRYPTION_KEY is required for discovery.",
        )
        return "Discovery skipped: ENCRYPTION_KEY is not configured."

    try:
        from kortny.mcp.client import discover_server_tools

        tools = discover_server_tools(
            server, encryption_key=encryption_key, timeout_seconds=30
        )
        count = _upsert(
            session, server=server, discovered=cast(list[object], tools), error=None
        )
        return f"Discovered {count} tool{'s' if count != 1 else ''}."
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        _upsert(session, server=server, discovered=[], error=error_str)
        return f"Discovery error: {error_str}"
