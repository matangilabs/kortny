"""App Home console surface registration (HIG-232).

The App Home tab is Kortny's self-serve operator console inside Slack: a
per-user view of recent usage, recent tasks, the skill catalog (with
per-skill enable/disable + an add-skill modal), connected Composio accounts,
and read-only MCP servers. The LLM never authors Block Kit here — every block
comes from the typed builders in ``kortny.slack.blockkit``.

``views.publish`` is a full replace, so every action and view submission
rebuilds the whole home view and republishes it. Errors never crash the Bolt
listener: each handler catches, logs, and republishes the view with a context
block carrying an error notice.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from slack_bolt import App
from sqlalchemy import func, literal_column, or_, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings
from kortny.dashboard.data import get_integration_dashboard, get_usage_aggregate
from kortny.dashboard.mcp_data import get_mcp_dashboard
from kortny.dashboard.skills_actions import (
    disable_skill_enablement,
    enable_skill_for_scope,
    paste_skill_markdown,
)
from kortny.dashboard.skills_data import SkillCatalogEntry, get_skills_dashboard
from kortny.db.models import (
    DashboardUser,
    Installation,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import session_scope
from kortny.slack import blockkit

# The home cards drive the same accept/dismiss flow as the message buttons;
# these are package-internal helpers shared between the two Slack surfaces.
from kortny.slack.witness_actions import _do_accept, _do_dismiss

logger = logging.getLogger(__name__)

T = TypeVar("T")

USAGE_WINDOW_DAYS = 30
RECENT_TASK_LIMIT = 5
SKILL_PANEL_LIMIT = 10
COMPOSIO_PANEL_LIMIT = 10
MCP_PANEL_LIMIT = 10

ADD_SKILL_ACTION = f"{blockkit.HOME_ACTION_PREFIX}add_skill"
ENABLE_SKILL_ACTION = f"{blockkit.HOME_ACTION_PREFIX}enable_skill"
DISABLE_SKILL_ACTION = f"{blockkit.HOME_ACTION_PREFIX}disable_skill"
MANAGE_MCP_ACTION = f"{blockkit.HOME_ACTION_PREFIX}manage_mcp"
OPEN_INTEGRATIONS_ACTION = f"{blockkit.HOME_ACTION_PREFIX}open_integrations"
ADD_SKILL_MODAL_CALLBACK = f"{blockkit.HOME_ACTION_PREFIX}add_skill_submit"
WITNESS_ACCEPT_ACTION = f"{blockkit.HOME_ACTION_PREFIX}witness_accept"
WITNESS_DISMISS_ACTION = f"{blockkit.HOME_ACTION_PREFIX}witness_dismiss"

WAITING_PANEL_CANDIDATE_LIMIT = 3

# Brand favicons for connected-account cards: slug -> vendor domain, served
# as 64px PNGs by Google's favicon service (Slack card icons need raster
# URLs; Composio's own logo_url only surfaces in the live catalog API).
_TOOLKIT_DOMAINS: dict[str, str] = {
    "alpaca": "alpaca.markets",
    "alpha_vantage": "alphavantage.co",
    "confluence": "atlassian.com",
    "exa": "exa.ai",
    "firecrawl": "firecrawl.dev",
    "github": "github.com",
    "gmail": "google.com",
    "linear": "linear.app",
    "notion": "notion.so",
    "serpapi": "serpapi.com",
    "slack": "slack.com",
    "supabase": "supabase.com",
    "twelve_data": "twelvedata.com",
    "vercel": "vercel.com",
}


def _toolkit_icon_url(slug: str) -> str:
    domain = _TOOLKIT_DOMAINS.get(slug, f"{slug.replace('_', '')}.com")
    return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"


TOP_MODEL_LIMIT = 3

# Slack profile avatars are stable URLs; cache per process to keep
# app_home_opened cheap (users.info is rate-limited Tier 4).
_AVATAR_CACHE: dict[str, str | None] = {}

SKILL_NAME_BLOCK_ID = "kortny_home_skill_name"
SKILL_NAME_ACTION_ID = "kortny_home_skill_name_input"
SKILL_MARKDOWN_BLOCK_ID = "kortny_home_skill_markdown"
SKILL_MARKDOWN_ACTION_ID = "kortny_home_skill_markdown_input"

_STATUS_EMOJI: dict[TaskStatus, str] = {
    TaskStatus.pending: ":hourglass_flowing_sand:",
    TaskStatus.running: ":gear:",
    TaskStatus.waiting_approval: ":raised_hand:",
    TaskStatus.succeeded: ":white_check_mark:",
    TaskStatus.failed: ":x:",
    TaskStatus.crashed: ":boom:",
    TaskStatus.cancelled: ":no_entry_sign:",
}


def register_app_home(
    app: App,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
) -> None:
    """Wire the App Home tab, its action router, and modal submissions."""

    @app.event("app_home_opened")
    def handle_app_home_opened(
        ack: Callable[[], None],
        event: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            if event.get("tab") != "home":
                return
            slack_user_id = event.get("user")
            if not isinstance(slack_user_id, str) or not slack_user_id:
                return
            _publish_home(
                client,
                session_factory=session_factory,
                settings=settings,
                slack_team_id=_event_team_id(event),
                slack_user_id=slack_user_id,
            )

        _ack_then(ack, handle)

    @app.action(re.compile(f"^{re.escape(blockkit.HOME_ACTION_PREFIX)}"))
    def handle_home_action(
        ack: Callable[[], None],
        body: dict[str, Any],
        action: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            action_id = action.get("action_id")
            slack_user_id = _action_user_id(body)
            slack_team_id = _action_team_id(body)
            if slack_user_id is None:
                return

            # url buttons (open integrations / manage MCP) need no server work.
            if action_id in {OPEN_INTEGRATIONS_ACTION, MANAGE_MCP_ACTION}:
                return

            # Open the add-skill modal BEFORE any DB work: trigger_id is valid
            # for ~3s and views.open must beat that window.
            if action_id == ADD_SKILL_ACTION:
                trigger_id = body.get("trigger_id")
                if isinstance(trigger_id, str) and trigger_id:
                    client.views_open(
                        trigger_id=trigger_id,
                        view=_build_add_skill_modal(slack_user_id, slack_team_id),
                    )
                return

            error_notice: str | None = None
            try:
                with session_scope(session_factory) as session:
                    installation = _installation_for_team(session, slack_team_id)
                    if installation is None:
                        return
                    if action_id in {WITNESS_ACCEPT_ACTION, WITNESS_DISMISS_ACTION}:
                        candidate_id = uuid.UUID(_action_value(action))
                        if action_id == WITNESS_ACCEPT_ACTION:
                            _do_accept(
                                session=session,
                                candidate_id=candidate_id,
                                installation_id=installation.id,
                                user_id=slack_user_id,
                                client=client,
                                settings=settings,
                            )
                        else:
                            _do_dismiss(
                                session=session,
                                candidate_id=candidate_id,
                                installation_id=installation.id,
                                user_id=slack_user_id,
                            )
                    elif action_id == ENABLE_SKILL_ACTION:
                        enable_skill_for_scope(
                            session,
                            installation_id=installation.id,
                            skill_id=uuid.UUID(_action_value(action)),
                            scope_type="user",
                            scope_id=slack_user_id,
                            by_user=f"slack:{slack_user_id}",
                        )
                    elif action_id == DISABLE_SKILL_ACTION:
                        disable_skill_enablement(
                            session,
                            enablement_id=uuid.UUID(_action_value(action)),
                            by_user=f"slack:{slack_user_id}",
                        )
            except Exception:
                logger.exception("app_home action failed action_id=%s", action_id)
                error_notice = "Something went wrong handling that action."

            _publish_home(
                client,
                session_factory=session_factory,
                settings=settings,
                slack_team_id=slack_team_id,
                slack_user_id=slack_user_id,
                error_notice=error_notice,
            )

        _ack_then(ack, handle)

    @app.view(re.compile(f"^{re.escape(blockkit.HOME_ACTION_PREFIX)}"))
    def handle_home_view_submission(
        ack: Callable[[], None],
        body: dict[str, Any],
        view: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        # Ack the submission immediately so the modal closes, then do the work.
        ack()
        if view.get("callback_id") != ADD_SKILL_MODAL_CALLBACK:
            return

        slack_user_id, slack_team_id = _modal_metadata(view)
        if slack_user_id is None:
            return

        name = _view_value(view, SKILL_NAME_BLOCK_ID, SKILL_NAME_ACTION_ID)
        markdown = _view_value(view, SKILL_MARKDOWN_BLOCK_ID, SKILL_MARKDOWN_ACTION_ID)

        error_notice: str | None = None
        try:
            with session_scope(session_factory) as session:
                installation = _installation_for_team(session, slack_team_id)
                if installation is None:
                    return
                paste_skill_markdown(
                    session,
                    installation_id=installation.id,
                    content=markdown or "",
                    name=name,
                    description=None,
                    by_user=f"slack:{slack_user_id}",
                )
        except Exception:
            logger.exception("app_home add-skill submission failed")
            error_notice = "I couldn't add that skill — check the markdown and retry."

        _publish_home(
            client,
            session_factory=session_factory,
            settings=settings,
            slack_team_id=slack_team_id,
            slack_user_id=slack_user_id,
            error_notice=error_notice,
        )


def resolve_dashboard_role(
    session: Session,
    installation_id: uuid.UUID,
    slack_user_id: str,
) -> str | None:
    """Resolve a Slack user's dashboard role (admin/member) or ``None``."""

    user = session.scalar(
        select(DashboardUser).where(
            DashboardUser.installation_id == installation_id,
            DashboardUser.slack_user_id == slack_user_id,
        )
    )
    if user is None or user.status != "active":
        return None
    return user.role


def build_home_view(
    session: Session,
    *,
    installation_id: uuid.UUID,
    slack_user_id: str,
    settings: Settings,
    error_notice: str | None = None,
    avatar_url: str | None = None,
) -> dict:
    """Build the App Home view for one user (pure, ≤100 blocks)."""

    role = resolve_dashboard_role(session, installation_id, slack_user_id)
    blocks: list[dict] = []

    if error_notice:
        blocks.append(blockkit.context(f":warning: {error_notice}"))

    blocks.extend(
        _usage_panel(
            session,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
            app_name=settings.agent_display_name,
            avatar_url=avatar_url,
        )
    )
    waiting = _waiting_panel(
        session, installation_id=installation_id, slack_user_id=slack_user_id
    )
    if waiting:
        blocks.append(blockkit.divider())
        blocks.extend(waiting)
    blocks.append(blockkit.divider())
    blocks.extend(
        _recent_tasks_panel(
            session, installation_id=installation_id, slack_user_id=slack_user_id
        )
    )
    blocks.append(blockkit.divider())
    blocks.extend(
        _skills_panel(
            session,
            installation_id=installation_id,
            app_name=settings.agent_display_name,
        )
    )
    blocks.append(blockkit.divider())
    blocks.extend(
        _composio_panel(
            session,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
            settings=settings,
        )
    )
    blocks.append(blockkit.divider())
    blocks.extend(
        _mcp_panel(
            session,
            installation_id=installation_id,
            settings=settings,
            role=role,
        )
    )
    blocks.append(blockkit.divider())
    dashboard_url = _dashboard_url(settings, "/")
    footer = "Reopen this tab to refresh."
    if dashboard_url is not None:
        footer += f"  ·  <{dashboard_url}|Open the full dashboard>"
    blocks.append(blockkit.context(footer))

    return blockkit.home_view(blocks)


# --- panels -----------------------------------------------------------------


def _usage_panel(
    session: Session,
    *,
    installation_id: uuid.UUID,
    slack_user_id: str,
    app_name: str,
    avatar_url: str | None = None,
) -> list[dict]:
    now = datetime.now(UTC)
    usage = get_usage_aggregate(
        session,
        start=now - timedelta(days=USAGE_WINDOW_DAYS),
        end=now,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    avg_cost = usage.total_cost_usd / usage.total_tasks if usage.total_tasks else 0.0

    greeting_text = (
        f"Hey <@{slack_user_id}> — your last {USAGE_WINDOW_DAYS} days at a glance."
    )
    greeting = (
        blockkit.context(blockkit.image_element(avatar_url, "you"), greeting_text)
        if avatar_url
        else blockkit.context(greeting_text)
    )

    # Two balanced columns: period stats | model mix with cost shares.
    this_period = (
        f"*This period*\n"
        f"Tasks run: *{usage.total_tasks:,}*\n"
        f"LLM spend: *${usage.total_cost_usd:.2f}*\n"
        f"Avg / task: *${avg_cost:.3f}*"
    )
    total_cost = usage.total_cost_usd or 1
    model_lines = [
        f"{row.display_key} · {float(row.cost_usd) / float(total_cost):.0%}"
        for row in usage.by_model[:TOP_MODEL_LIMIT]
    ]
    top_models = "*Top models*\n" + (
        "\n".join(model_lines) if model_lines else "No LLM calls yet"
    )
    return [
        blockkit.header(f"Your {app_name} console"),
        greeting,
        blockkit.section(fields=[this_period, top_models]),
    ]


def _waiting_panel(
    session: Session,
    *,
    installation_id: uuid.UUID,
    slack_user_id: str,
) -> list[dict]:
    """Actionable items: undecided suggestions + approval-paused tasks.

    Returns [] when there is nothing waiting so the panel disappears
    entirely — an empty to-do list is the best to-do list.
    """

    candidates = list(
        session.scalars(
            select(WitnessOpportunityCandidate)
            .where(
                WitnessOpportunityCandidate.installation_id == installation_id,
                WitnessOpportunityCandidate.status == "sent",
            )
            .order_by(WitnessOpportunityCandidate.created_at.desc())
            .limit(WAITING_PANEL_CANDIDATE_LIMIT)
        )
    )
    approvals = list(
        session.scalars(
            select(Task)
            .where(
                Task.installation_id == installation_id,
                Task.slack_user_id == slack_user_id,
                Task.status == TaskStatus.waiting_approval,
            )
            .order_by(Task.created_at.desc())
            .limit(RECENT_TASK_LIMIT)
        )
    )
    if not candidates and not approvals:
        return []

    blocks: list[dict] = [blockkit.section("*Waiting on you*")]
    for task in approvals:
        where = f" in <#{task.slack_channel_id}>" if task.slack_channel_id else ""
        blocks.append(
            blockkit.section(
                f":raised_hand: {_snippet(task.input)}\n"
                f"_paused for approval{where} · {_relative_time(task.created_at)} — "
                "react :white_check_mark: in the thread to approve_"
            )
        )
    for candidate in candidates:
        meta = candidate.candidate_type.replace("_", " ")
        if candidate.channel_id:
            meta += f" · <#{candidate.channel_id}>"
        blocks.append(
            blockkit.card(
                title=candidate.title,
                subtitle=meta,
                body=_snippet(candidate.summary, limit=180),
                actions=[
                    blockkit.button(
                        "Accept",
                        WITNESS_ACCEPT_ACTION,
                        value=str(candidate.id),
                        style="primary",
                    ),
                    blockkit.button(
                        "Dismiss",
                        WITNESS_DISMISS_ACTION,
                        value=str(candidate.id),
                    ),
                ],
            )
        )
    return blocks


def _recent_tasks_panel(
    session: Session,
    *,
    installation_id: uuid.UUID,
    slack_user_id: str,
) -> list[dict]:
    tasks = list(
        session.scalars(
            select(Task)
            .where(
                Task.installation_id == installation_id,
                Task.slack_user_id == slack_user_id,
                # The console shows what the user asked for. Ambient
                # plumbing (witness scans ride as slack_event, consolidator
                # as synthetic) is noise here — allowlist conversational
                # identities; NULL predates identities and is user work.
                or_(
                    Task.identity_kind.is_(None),
                    Task.identity_kind.in_(("slack_message", "scheduled", "manual")),
                ),
            )
            .order_by(Task.created_at.desc())
            .limit(RECENT_TASK_LIMIT)
        )
    )
    blocks: list[dict] = [blockkit.section("*Recent tasks*")]
    if not tasks:
        blocks.append(
            blockkit.section(
                "No tasks yet — mention me in a channel or send me a DM to get going."
            )
        )
        return blocks
    lines = []
    for task in tasks:
        emoji = _STATUS_EMOJI.get(task.status, ":grey_question:")
        snippet = _snippet(task.input)
        relative = _relative_time(task.created_at)
        lines.append(f"{emoji}  {snippet}  ·  _{relative}_")
    blocks.append(blockkit.section("\n".join(lines)))
    return blocks


def _skills_panel(
    session: Session,
    *,
    installation_id: uuid.UUID,
    app_name: str,
) -> list[dict]:
    dashboard = get_skills_dashboard(session, installation_id)
    entries = (*dashboard.curated, *dashboard.custom)
    enabled_count = sum(1 for entry in entries if entry.is_enabled)
    blocks: list[dict] = [
        blockkit.section(
            "*Skills*",
            accessory=blockkit.button("Add skill", ADD_SKILL_ACTION, style="primary"),
        ),
        blockkit.context(
            f"{enabled_count} of {len(entries)} enabled — playbooks {app_name} "
            "loads when a request matches."
            if entries
            else f"Playbooks {app_name} loads when a request matches."
        ),
    ]
    if not entries:
        blocks.append(
            blockkit.section(
                "No skills available yet. Use *Add skill* to paste a SKILL.md."
            )
        )
        return blocks

    shown = entries[:SKILL_PANEL_LIMIT]
    cards = [card for entry in shown for card in _skill_row(entry)]
    blocks.append(blockkit.carousel(*cards))
    if len(entries) > SKILL_PANEL_LIMIT:
        blocks.append(
            blockkit.context(
                f"Showing {SKILL_PANEL_LIMIT} of {len(entries)} skills — "
                "manage the rest on the dashboard."
            )
        )
    return blocks


def _skill_row(entry: SkillCatalogEntry) -> list[dict]:
    if entry.is_enabled:
        # Prefer the user-scoped enablement so disable targets the right row;
        # fall back to the first enabled scope.
        enablement_id = _user_enablement_id(entry) or next(
            chip.enablement_id
            for chip in entry.enabled_scopes
            if chip.status == "enabled"
        )
        status_dot = ":large_green_circle:"
        accessory = blockkit.button(
            "Disable",
            DISABLE_SKILL_ACTION,
            value=str(enablement_id),
        )
    else:
        status_dot = ":white_circle:"
        accessory = blockkit.button(
            "Enable",
            ENABLE_SKILL_ACTION,
            value=str(entry.skill_id),
        )
    description = _snippet(entry.description, limit=160)
    return [
        blockkit.card(
            title=f"{status_dot} {entry.name}",
            subtitle=_skill_meta(entry),
            body=description,
            actions=[accessory],
        )
    ]


def _skill_meta(entry: SkillCatalogEntry) -> str:
    trust_emoji = {
        "trusted": ":shield:",
        "community": ":busts_in_silhouette:",
        "untrusted": ":lock:",
        "quarantined": ":no_entry:",
    }.get(entry.trust_level, ":grey_question:")
    parts = [
        "enabled" if entry.is_enabled else "disabled",
        f"{trust_emoji} {entry.trust_level}",
    ]
    scopes = sorted(
        {chip.scope_type for chip in entry.enabled_scopes if chip.status == "enabled"}
    )
    if scopes:
        parts.append(" + ".join(scopes) + " scope")
    if entry.invocations_30d:
        parts.append(
            f"{entry.invocations_30d} use"
            f"{'s' if entry.invocations_30d != 1 else ''} this month"
        )
    if entry.has_scripts:
        parts.append("has scripts")
    return "  ·  ".join(parts)


def _user_enablement_id(entry: SkillCatalogEntry) -> uuid.UUID | None:
    for chip in entry.enabled_scopes:
        if chip.scope_type == "user" and chip.status == "enabled":
            return chip.enablement_id
    return None


def _composio_panel(
    session: Session,
    *,
    installation_id: uuid.UUID,
    slack_user_id: str,
    settings: Settings,
) -> list[dict]:
    dashboard = get_integration_dashboard(
        session=session,
        runtime_settings=settings,
        installation_id=installation_id,
        owner_slack_user_id=slack_user_id,
    )
    connections = dashboard.composio_catalog.connections
    integrations_url = _dashboard_url(settings, "/integrations")
    header_accessory = (
        blockkit.button(
            "Open integrations",
            OPEN_INTEGRATIONS_ACTION,
            url=integrations_url,
        )
        if integrations_url is not None
        else None
    )
    blocks: list[dict] = [
        blockkit.section("*Connected accounts*", accessory=header_accessory)
    ]

    if not connections:
        blocks.append(
            blockkit.section(
                "No connected accounts yet — connect apps from the dashboard "
                "integrations page."
            )
        )
        return blocks

    usage_by_slug = _composio_tool_usage(
        session,
        installation_id=installation_id,
        toolkit_slugs=[c.toolkit_slug for c in connections],
    )
    # Most-used first: the cards the user actually cares about lead the rail.
    ranked = sorted(
        connections,
        key=lambda c: usage_by_slug.get(c.toolkit_slug, _NO_USAGE).calls,
        reverse=True,
    )
    cards = []
    for connection in ranked[:COMPOSIO_PANEL_LIMIT]:
        usage = usage_by_slug.get(connection.toolkit_slug, _NO_USAGE)
        if usage.calls:
            tools = (
                f"{usage.distinct_tools} tool{'s' if usage.distinct_tools != 1 else ''}"
            )
            parts = [
                f"*{usage.calls:,}* call{'s' if usage.calls != 1 else ''}",
                tools,
            ]
            if usage.last_used is not None:
                parts.append(f"last used {_relative_time(usage.last_used)}")
            subtitle = " · ".join(parts)
        else:
            subtitle = "Not used in the last 30 days"
        # Status and scope only when they carry signal: problems and
        # non-personal scopes. "active", the owner's own name, and the
        # green dot are defaults — defaults stay silent.
        notes = []
        if connection.status != "active":
            notes.append(f":warning: {connection.status}")
        if connection.visibility_scope_type in {"workspace", "channel"}:
            notes.append(connection.scope_label)
        cards.append(
            blockkit.card(
                icon_url=_toolkit_icon_url(connection.toolkit_slug),
                title=_integration_title(connection),
                subtitle=subtitle,
                body=" · ".join(notes) if notes else None,
            )
        )
    blocks.append(blockkit.carousel(*cards))
    active_count = sum(1 for c in connections if c.status == "active")
    summary = f"{active_count} of {len(connections)} active"
    if len(connections) > COMPOSIO_PANEL_LIMIT:
        summary += f" · showing {COMPOSIO_PANEL_LIMIT} — see the dashboard for the rest"
    blocks.append(blockkit.context(summary))
    return blocks


_CONNECTION_SUFFIX_RE = re.compile(r"\s*connection\s*$", re.IGNORECASE)


def _integration_title(connection: Any) -> str:
    name = connection.display_name or connection.toolkit_slug
    return _CONNECTION_SUFFIX_RE.sub("", name).strip() or connection.toolkit_slug


@dataclass(frozen=True, slots=True)
class _ToolkitUsage:
    calls: int
    distinct_tools: int
    last_used: datetime | None


_NO_USAGE = _ToolkitUsage(calls=0, distinct_tools=0, last_used=None)


def _composio_tool_usage(
    session: Session,
    *,
    installation_id: uuid.UUID,
    toolkit_slugs: Sequence[str],
    window_days: int = USAGE_WINDOW_DAYS,
) -> dict[str, _ToolkitUsage]:
    """30-day per-toolkit usage: call count, distinct tools, last-used.

    Composio runtime tools are named ``composio_<toolkit>_<tool>`` and slugs
    may themselves contain underscores (alpha_vantage), so usage is
    attributed by longest-matching known slug rather than naive splitting.
    """

    if not toolkit_slugs:
        return {}
    since = datetime.now(UTC) - timedelta(days=window_days)
    # Label the JSONB expression once and group by the label — grouping by a
    # second rendering of payload->>'tool' parameterizes the key twice and
    # Postgres rejects it as a non-aggregated column.
    tool_expr = TaskEvent.payload["tool"].astext.label("tool_name")
    rows = session.execute(
        select(tool_expr, func.count(), func.max(TaskEvent.created_at))
        .join(Task, Task.id == TaskEvent.task_id)
        .where(
            Task.installation_id == installation_id,
            TaskEvent.type == TaskEventType.tool_call,
            TaskEvent.created_at >= since,
            TaskEvent.payload["tool"].astext.like("composio_%"),
        )
        .group_by(literal_column("tool_name"))
    ).all()
    ordered_slugs = sorted(toolkit_slugs, key=len, reverse=True)
    calls: dict[str, int] = {}
    tools: dict[str, int] = {}
    last_used: dict[str, datetime] = {}
    for tool_name, count, latest in rows:
        if not isinstance(tool_name, str):
            continue
        for slug in ordered_slugs:
            if tool_name.startswith(f"composio_{slug}_"):
                calls[slug] = calls.get(slug, 0) + int(count)
                tools[slug] = tools.get(slug, 0) + 1
                if latest is not None and (
                    slug not in last_used or latest > last_used[slug]
                ):
                    last_used[slug] = latest
                break
    return {
        slug: _ToolkitUsage(
            calls=calls[slug],
            distinct_tools=tools.get(slug, 0),
            last_used=last_used.get(slug),
        )
        for slug in calls
    }


def _mcp_panel(
    session: Session,
    *,
    installation_id: uuid.UUID,
    settings: Settings,
    role: str | None,
) -> list[dict]:
    dashboard = get_mcp_dashboard(session, installation_id)
    servers = dashboard.servers
    manage_url = _dashboard_url(settings, "/mcp") if role == "admin" else None
    header_accessory = (
        blockkit.button("Manage on dashboard", MANAGE_MCP_ACTION, url=manage_url)
        if manage_url is not None
        else None
    )
    blocks: list[dict] = [blockkit.section("*MCP servers*", accessory=header_accessory)]

    if not servers:
        blocks.append(
            blockkit.section("No MCP servers registered for this workspace yet.")
        )
        return blocks

    for server in servers[:MCP_PANEL_LIMIT]:
        dot = ":large_green_circle:" if server.status == "enabled" else ":white_circle:"
        blocks.append(
            blockkit.card(
                title=f"{dot} {server.name}",
                subtitle=(
                    f"{server.status} · "
                    f"{server.enabled_tool_count}/{server.tool_count} tools enabled"
                ),
            )
        )
    if len(servers) > MCP_PANEL_LIMIT:
        blocks.append(
            blockkit.context(f"Showing {MCP_PANEL_LIMIT} of {len(servers)} servers.")
        )
    return blocks


# --- modal ------------------------------------------------------------------


def _build_add_skill_modal(slack_user_id: str, slack_team_id: str | None) -> dict:
    blocks = [
        blockkit.input_block(
            "Skill name",
            blockkit.plain_text_input(
                SKILL_NAME_ACTION_ID,
                placeholder="e.g. Weekly recap",
            ),
            block_id=SKILL_NAME_BLOCK_ID,
            optional=True,
            hint="Optional when the markdown has YAML frontmatter.",
        ),
        blockkit.input_block(
            "SKILL.md markdown",
            blockkit.plain_text_input(
                SKILL_MARKDOWN_ACTION_ID,
                multiline=True,
                placeholder="Paste the SKILL.md contents here.",
            ),
            block_id=SKILL_MARKDOWN_BLOCK_ID,
        ),
    ]
    return blockkit.modal(
        "Add a skill",
        blocks,
        callback_id=ADD_SKILL_MODAL_CALLBACK,
        submit="Add",
        private_metadata=f"{slack_user_id}|{slack_team_id or ''}",
    )


# --- helpers ----------------------------------------------------------------


def _publish_home(
    client: Any,
    *,
    session_factory: sessionmaker[Session] | None,
    settings: Settings,
    slack_team_id: str | None,
    slack_user_id: str,
    error_notice: str | None = None,
) -> None:
    try:
        with session_scope(session_factory) as session:
            installation = _installation_for_team(session, slack_team_id)
            if installation is None:
                return
            view = build_home_view(
                session,
                installation_id=installation.id,
                slack_user_id=slack_user_id,
                settings=settings,
                error_notice=error_notice,
                avatar_url=_avatar_url(client, slack_user_id),
            )
        client.views_publish(user_id=slack_user_id, view=view)
    except Exception:
        logger.exception("app_home publish failed slack_user_id=%s", slack_user_id)
        # Never leave a stale view masquerading as current: a visible
        # failure beats a silently outdated console (bit us twice).
        try:
            client.views_publish(
                user_id=slack_user_id,
                view=blockkit.home_view(
                    [
                        blockkit.header(f"Your {settings.agent_display_name} console"),
                        blockkit.section(
                            ":warning: The console failed to render. The "
                            "error is in the app logs — reopen this tab "
                            "after it's fixed."
                        ),
                    ]
                ),
            )
        except Exception:
            logger.exception(
                "app_home fallback publish failed slack_user_id=%s", slack_user_id
            )


def _avatar_url(client: Any, slack_user_id: str) -> str | None:
    """Best-effort profile avatar; cached per process, never fails the view."""

    if slack_user_id in _AVATAR_CACHE:
        return _AVATAR_CACHE[slack_user_id]
    url: str | None = None
    try:
        response = client.users_info(user=slack_user_id)
        profile = (response.get("user") or {}).get("profile") or {}
        candidate = profile.get("image_48") or profile.get("image_72")
        if isinstance(candidate, str) and candidate:
            url = candidate
    except Exception:
        logger.debug("avatar lookup failed slack_user_id=%s", slack_user_id)
    _AVATAR_CACHE[slack_user_id] = url
    return url


def _installation_for_team(
    session: Session,
    slack_team_id: str | None,
) -> Installation | None:
    if not slack_team_id:
        return session.scalar(select(Installation).limit(1))
    return session.scalar(
        select(Installation).where(Installation.slack_team_id == slack_team_id)
    )


def _ack_then(ack: Callable[[], None], handler: Callable[[], T]) -> T:
    ack()
    return handler()


def _event_team_id(event: dict[str, Any]) -> str | None:
    team = event.get("team")
    return team if isinstance(team, str) else None


def _action_team_id(body: dict[str, Any]) -> str | None:
    team = body.get("team")
    if isinstance(team, dict):
        team_id = team.get("id")
        if isinstance(team_id, str):
            return team_id
    team_id = body.get("team_id")
    return team_id if isinstance(team_id, str) else None


def _action_user_id(body: dict[str, Any]) -> str | None:
    user = body.get("user")
    if isinstance(user, dict):
        user_id = user.get("id")
        if isinstance(user_id, str):
            return user_id
    return None


def _action_value(action: dict[str, Any]) -> str:
    value = action.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError("action is missing a value")
    return value


def _modal_metadata(view: dict[str, Any]) -> tuple[str | None, str | None]:
    metadata = view.get("private_metadata")
    if not isinstance(metadata, str) or "|" not in metadata:
        return None, None
    slack_user_id, _, slack_team_id = metadata.partition("|")
    return (slack_user_id or None), (slack_team_id or None)


def _view_value(view: dict[str, Any], block_id: str, action_id: str) -> str | None:
    state = view.get("state")
    if not isinstance(state, dict):
        return None
    values = state.get("values")
    if not isinstance(values, dict):
        return None
    block = values.get(block_id)
    if not isinstance(block, dict):
        return None
    element = block.get(action_id)
    if not isinstance(element, dict):
        return None
    value = element.get("value")
    return value if isinstance(value, str) and value else None


def _snippet(text: str, *, limit: int = 80) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed or "(empty)"
    return collapsed[: limit - 1].rstrip() + "…"


def _relative_time(when: datetime) -> str:
    now = datetime.now(UTC)
    moment = when if when.tzinfo is not None else when.replace(tzinfo=UTC)
    delta = now - moment
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _dashboard_url(settings: Settings, path: str) -> str | None:
    base = settings.public_base_url
    if not base:
        return None
    return f"{base.rstrip('/')}{path}"
