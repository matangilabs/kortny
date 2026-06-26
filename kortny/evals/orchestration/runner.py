"""Live runner for the cross-app orchestration eval.

Builds a ``RunFn`` backed by the real ``AgentTaskExecutor`` and scores the seed
dataset against a live install. Needs a running Postgres DB, valid LLM API key,
and the connected integrations configured in the target install.

The runner is **on-demand only** — it creates real tasks, runs the agent, and
reads the resulting ``TaskEvent`` rows to derive the called-apps set. It is
never run in CI.

Toolkit-slug derivation
-----------------------
The authoritative source for the called-app slug is the ``tool_result``
``TaskEvent`` row. ``ComposioExecuteTool.invoke`` (kortny/tools/composio_execute.py)
writes its output as ``{"provider": "composio", "toolkit_slug": <slug>,
"tool_slug": ..., "successful": ..., ...}`` and the coordinator spreads that
output under ``payload["output"]`` of the ``tool_result`` event. So the real,
canonical Composio toolkit slug is read directly from
``payload["output"]["toolkit_slug"]`` when ``payload["output"]["provider"] ==
"composio"`` — no name-parsing required. This is correct for multi-underscore
toolkits (e.g. ``twelve_data``, ``alpha_vantage``) that name-parsing would
mangle.

Only successful Composio executions count toward the called-apps set
(``payload["output"]["successful"] is True``); a failed call did not actually
reach the integration's data.

``_toolkit_slug_from_tool_name`` survives as a *fallback only*: it is used when
a ``tool_result`` output is missing the ``toolkit_slug`` field. It splits a
``composio_{toolkit}_{tool}`` runtime name on ``_`` and takes index 1, which is
wrong for multi-underscore toolkits — hence it is never the primary path.

``any_tool_called`` counts every ``tool_call``/``tool_result`` row (Composio,
native, MCP) so the ``must_use_tools`` context-leak guard fires whenever no tool
ran at all — including a write that paused at the approval gate before its
result was recorded.

Side-effect-free egress
-----------------------
The eval must not post to Slack: the synthetic eval channel does not exist, so a
real ``chat.postMessage`` from the post-completion hook crashes the run. The
runner injects a ``_NoOpSlackClient`` whose ``chat_postMessage`` /
``files_upload_v2`` are no-ops returning a benign ``ok`` response with a
synthetic ``ts``, so ``_post_outputs`` and the approval-prompt post complete
without reaching Slack. The agent's LLM + tool execution still runs for real —
only the Slack egress is stubbed. Write tools (e.g. "file a ticket") still pause
at the approval gate; with no Slack approver the task simply parks on
``waiting_approval`` and no real artifact is created, while the eval still
captures that the right app's tool was reached.

Usage::

    uv run python -m kortny.evals.orchestration.runner
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.composio.runtime import connected_toolkit_slugs
from kortny.config import load_settings
from kortny.db import make_session_factory
from kortny.db.models import (
    ComposioConnection,
    Installation,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.evals.orchestration.cases import SEED_ORCHESTRATION_CASES, OrchestrationCase
from kortny.evals.orchestration.scoring import (
    OrchestrationReport,
    RunFn,
    RunResult,
    score_orchestration,
)
from kortny.slack.posting import SlackPostingClient
from kortny.tasks import TaskIdentity, TaskService
from kortny.worker.agent_executor import AgentTaskExecutor

# Sentinel channel used when creating an eval task — it never routes to Slack
# because the injected _NoOpSlackClient stubs all Slack egress.
# The default channel can be overridden via KORTNY_EVAL_SCOPE_CHANNEL_ID for
# channel-scoped connections; user-scoped connections key on slack_user_id.
_EVAL_CHANNEL_ID = "EVAL_ORCHESTRATION"
_EVAL_THREAD_TS = "0.000000"


class _NoOpSlackClient:
    """A Slack client stub that makes every egress call a no-op.

    Satisfies the ``SlackPostingClient`` protocol (``chat_postMessage`` /
    ``files_upload_v2``) and tolerates the reaction calls the executor probes
    via ``getattr`` (``reactions_add`` / ``reactions_remove``). Each method
    returns a benign ``{"ok": True, ...}`` response with a synthetic ``ts`` so
    the SlackPoster reads a message ts and never raises, and nothing is ever
    sent to Slack. The agent's LLM + tool execution is untouched — only egress
    is stubbed.
    """

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        unfurl_links: bool = True,
        unfurl_media: bool = True,
    ) -> Mapping[str, Any]:
        return {"ok": True, "ts": _synthetic_ts(), "channel": channel}

    def files_upload_v2(
        self,
        *,
        file: str,
        filename: str | None = None,
        title: str | None = None,
        channel: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> Mapping[str, Any]:
        return {"ok": True, "file": {"id": "FEVAL_NOOP"}, "ts": _synthetic_ts()}

    def reactions_add(
        self, *, channel: str, name: str, timestamp: str
    ) -> Mapping[str, Any]:
        return {"ok": True}

    def reactions_remove(
        self, *, channel: str, name: str, timestamp: str
    ) -> Mapping[str, Any]:
        return {"ok": True}


def _synthetic_ts() -> str:
    """A unique Slack-message-ts-shaped string for no-op post responses."""
    return f"1710000000.{uuid.uuid4().hex[:6]}"


# Env overrides for the connection scope owner.
_SCOPE_USER_ENV = "KORTNY_EVAL_SCOPE_USER_ID"
_SCOPE_CHANNEL_ENV = "KORTNY_EVAL_SCOPE_CHANNEL_ID"


def _toolkit_slug_from_tool_name(tool_name: str) -> str | None:
    """Fallback: extract a Composio toolkit slug from a runtime tool name.

    NOT the primary path. ``composio_runtime_tool_name`` produces
    ``composio_{toolkit}_{tool_slug}`` and this splits on ``_`` and takes index
    1 — which mangles multi-underscore toolkits (``composio_twelve_data_...`` →
    ``"twelve"``). Used only when a ``tool_result`` output is missing the
    authoritative ``toolkit_slug`` field. Returns None for non-Composio tools.
    """
    if not tool_name.startswith("composio_"):
        return None
    parts = tool_name.split("_", 2)
    if len(parts) < 2:
        return None
    return parts[1]


def _toolkit_slug_from_tool_result(payload: Mapping[str, object]) -> str | None:
    """Derive the authoritative Composio toolkit slug from a tool_result payload.

    Reads ``payload["output"]["toolkit_slug"]`` when the output marks a
    successful Composio execution (``provider == "composio"`` and
    ``successful is True``). This is the real canonical slug written by
    ``ComposioExecuteTool.invoke`` — no name-parsing, correct for
    multi-underscore toolkits.

    Falls back to name-parsing the runtime tool name only if the authoritative
    ``toolkit_slug`` field is absent. Returns None for non-Composio,
    unsuccessful, or non-tool outputs.
    """
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    if output.get("provider") != "composio":
        return None
    if output.get("successful") is not True:
        return None
    slug = output.get("toolkit_slug")
    if isinstance(slug, str) and slug:
        return slug
    # Authoritative field absent — fall back to name-parsing.
    tool_name = payload.get("tool")
    if isinstance(tool_name, str):
        return _toolkit_slug_from_tool_name(tool_name)
    return None


def _called_apps_from_events(
    session: Session,
    task_id: uuid.UUID,
) -> tuple[frozenset[str], bool]:
    """Read tool TaskEvent rows and return the called-apps set.

    Authoritative source: the canonical ``toolkit_slug`` carried in each
    successful Composio ``tool_result`` output. To also count writes that
    *reached* an integration but paused at the approval gate before a
    ``tool_result`` was ever recorded, this additionally unions in Composio
    toolkit slugs name-parsed from ``tool_call`` rows. A write that paused
    pre-result therefore still registers as "reached for that app" — which is
    exactly the orchestration signal the eval wants (the agent picked the right
    integration), independent of whether a real artifact was created.

    Returns:
        A tuple of (called_toolkit_slugs, any_tool_called) where
        called_toolkit_slugs is the set of Composio toolkit slugs whose tools
        were reached (completed or paused at approval) and any_tool_called is
        True if at least one tool of any kind was invoked (Composio, native, or
        MCP — including a call that paused before its result, for the
        must_use_tools guard).
    """
    result_rows = list(
        session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task_id,
                TaskEvent.type == TaskEventType.tool_result,
            )
        )
    )
    call_rows = list(
        session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task_id,
                TaskEvent.type == TaskEventType.tool_call,
            )
        )
    )
    # A tool that paused at the approval gate emits a `tool_approval_required`
    # log event carrying the pending runtime tool name in request["tool"] — and
    # may never emit a `tool_call` row (the gate fires at dispatch). Without this
    # the eval undercounts: the agent reached the right app but the paused tool
    # is invisible (observed: a Linear write-capable read paused here, scoring 0
    # despite correct reaching). Credit it.
    approval_rows = [
        row
        for row in session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task_id,
                TaskEvent.type == TaskEventType.log,
            )
        )
        if row.payload.get("message") == "tool_approval_required"
    ]

    # A tool was invoked if any call/result exists OR a tool paused at approval.
    any_tool_called = (
        len(result_rows) > 0 or len(call_rows) > 0 or len(approval_rows) > 0
    )

    slugs: set[str] = set()
    # Authoritative: real toolkit_slug from successful Composio results.
    for row in result_rows:
        slug = _toolkit_slug_from_tool_result(row.payload)
        if slug:
            slugs.add(slug)
    # Supplemental: name-parsed Composio slug from every tool_call, so an
    # app reached by a write that paused pre-result still counts.
    for row in call_rows:
        tool_name = row.payload.get("tool")
        if isinstance(tool_name, str):
            slug = _toolkit_slug_from_tool_name(tool_name)
            if slug:
                slugs.add(slug)
    # Tools paused at approval (request["tool"]) — the app was reached even with
    # no tool_call row.
    for row in approval_rows:
        request = row.payload.get("request")
        tool_name = request.get("tool") if isinstance(request, dict) else None
        if isinstance(tool_name, str):
            slug = _toolkit_slug_from_tool_name(tool_name)
            if slug:
                slugs.add(slug)
    return frozenset(slugs), any_tool_called


def _first_installation_id(session: Session) -> uuid.UUID:
    """Return the ID of the first installation in the DB.

    For a single-tenant install there is exactly one row. Raise if none found.
    """
    result = session.scalars(select(Installation).limit(1)).first()
    if result is None:
        raise RuntimeError(
            "No installation found in the database. "
            "Start the app and complete Slack install before running the eval."
        )
    return result.id


def _resolve_scope_user_id(
    session: Session,
    installation_id: uuid.UUID,
) -> str:
    """Resolve the Slack user that owns the install's Composio connections.

    Composio connections are commonly ``user``-scoped: the runtime resolver
    (``ComposioConnectionResolver._allows_task``) only surfaces a ``user``
    connection when ``visibility_scope_id == task.slack_user_id``. A synthetic
    task with no real owner therefore resolves ZERO connections and the agent
    sees no connected tools. To make the eval behave like a real Slack message,
    scope the task to the connection owner.

    Resolution order:
      1. ``KORTNY_EVAL_SCOPE_USER_ID`` env override, if set.
      2. The dominant ``visibility_scope_id`` among active ``user``-scoped
         ``composio_connections`` for this installation (the slack_user_id that
         owns the most connections).

    Raises if neither yields an owner (no user-scoped connections to ground on).
    """
    override = os.environ.get(_SCOPE_USER_ENV)
    if override:
        return override

    row = session.execute(
        select(
            ComposioConnection.visibility_scope_id,
            func.count().label("n"),
        )
        .where(
            ComposioConnection.installation_id == installation_id,
            ComposioConnection.status == "active",
            ComposioConnection.visibility_scope_type == "user",
            ComposioConnection.visibility_scope_id.is_not(None),
        )
        .group_by(ComposioConnection.visibility_scope_id)
        .order_by(func.count().desc(), ComposioConnection.visibility_scope_id)
        .limit(1)
    ).first()
    if row is None or row[0] is None:
        raise RuntimeError(
            "No active user-scoped Composio connections found to scope the eval "
            f"task to. Set {_SCOPE_USER_ENV} to the owning Slack user id, or "
            "connect integrations for a user in the install."
        )
    return str(row[0])


def _resolve_scope_channel_id() -> str:
    """Resolve the channel to scope the eval task to.

    Uses ``KORTNY_EVAL_SCOPE_CHANNEL_ID`` if set (for channel-scoped
    connections), otherwise the synthetic eval channel. User-scoped connections
    do not key on the channel, so the synthetic default is fine for them.
    """
    return os.environ.get(_SCOPE_CHANNEL_ENV) or _EVAL_CHANNEL_ID


def build_live_run_fn(
    session: Session,
    executor: AgentTaskExecutor,
    installation_id: uuid.UUID,
    *,
    scope_user_id: str,
    scope_channel_id: str,
) -> RunFn:
    """Build a RunFn that executes one case through the real agent.

    Creates a ``manual`` Task with the case's request text **scoped to the
    connection owner**, runs ``AgentTaskExecutor.execute()`` synchronously, then
    reads the task's ``tool_result`` ``TaskEvent`` rows to derive the
    called-apps set.

    The task is created with ``slack_user_id=scope_user_id`` and
    ``slack_channel_id=scope_channel_id`` so the Composio connection resolver
    (``ComposioConnectionResolver._allows_task``) surfaces the install's
    user-scoped connections exactly as a real Slack message from that user
    would — without this, the resolver matches zero connections and the agent
    has no connected tools.

    Requires:
    - A live Postgres session connected to the target install.
    - A configured ``AgentTaskExecutor`` (LLM provider, settings).
    - The ``installation_id`` of the target Kortny install.
    - ``scope_user_id``: the owning Slack user id for user-scoped connections.
    - ``scope_channel_id``: the channel id for channel-scoped connections.

    The executor is called directly so no Slack messages are posted (no
    slack_client is wired in by default).
    """
    task_service = TaskService(session)

    def run(case: OrchestrationCase) -> RunResult:
        # Each case gets a unique identity so the dedup logic never collapses
        # distinct runs into the same task row.
        unique_ts = f"{uuid.uuid4().int % 10**9}.{uuid.uuid4().int % 10**6}"
        # source_surface="assistant" makes the executor run the REAL cheap-tier
        # intent classifier on this task (AgentTaskExecutor._ensure_intent_
        # decision), exactly as a live Slack message would. Without it the eval
        # skipped intent entirely, so the intent->toolkit_affinity->retrieval
        # boost (HIG-274) — the production tool-disambiguation prior — never
        # fired and one app's tools could dominate the prewarm. This makes the
        # eval faithful to the full production path (intent + coordinator).
        identity = TaskIdentity.manual(
            channel_id=scope_channel_id,
            thread_ts=_EVAL_THREAD_TS,
            user_id=scope_user_id,
            input_text=f"{case.request}::{unique_ts}",
            source_surface="assistant",
        )
        task: Task = task_service.create_task(
            installation_id=installation_id,
            slack_channel_id=scope_channel_id,
            slack_user_id=scope_user_id,
            slack_thread_ts=_EVAL_THREAD_TS,
            slack_message_ts=unique_ts,
            input=case.request,
            identity=identity,
        )
        session.commit()

        # Confirm the scope actually resolves connections before running, so a
        # zero here clearly signals a scope misconfiguration vs an agent miss.
        resolved = connected_toolkit_slugs(session, task)
        print(
            f"  scope check: user={scope_user_id} channel={scope_channel_id} "
            f"resolved {len(resolved)} connected toolkit(s): {sorted(resolved)!r}"
        )

        # Skip cases whose required toolkits are not connected in this install.
        if case.requires_toolkits:
            missing_required = [
                slug for slug in case.requires_toolkits if slug not in resolved
            ]
            if missing_required:
                reason = f"requires {missing_required!r} not in connected set"
                print(f"  SKIP: {reason}")
                return RunResult(
                    called_apps=frozenset(),
                    any_tool_called=False,
                    answer="",
                    skipped=True,
                    skip_reason=reason,
                )

        # A write tool (e.g. case 1 "file a ticket", case 9 "create Linear
        # issues") pauses on the approval gate; execute() catches
        # ToolApprovalRequired internally and returns a result with the task
        # parked on waiting_approval — it does not raise. We still want the
        # tool_call events captured. Any *other* exception (a genuine mid-case
        # crash) is contained here so one bad case doesn't abort the whole run:
        # the transaction is rolled back and whatever events were already
        # committed are read back below.
        answer = ""
        try:
            result = executor.execute(
                session=session,
                task=task,
                task_service=task_service,
            )
            session.commit()
            answer = result.result_summary or ""
        except Exception as exc:  # noqa: BLE001 — eval must survive one bad case
            session.rollback()
            answer = f"<execute raised: {type(exc).__name__}: {exc}>"
            print(f"  ! case execution raised (continuing): {answer}")

        # Read whatever tool events exist — a task that paused at approval still
        # recorded its tool_call rows, so the app it reached still counts.
        called_apps, any_tool_called = _called_apps_from_events(session, task.id)
        return RunResult(
            called_apps=called_apps,
            any_tool_called=any_tool_called,
            answer=answer,
        )

    return run


def run() -> OrchestrationReport:
    """Run the full seed eval against the live install and return the report."""
    settings = load_settings()
    session_factory = make_session_factory(database_url=settings.postgres_url)
    with session_factory() as session:
        installation_id = _first_installation_id(session)
        scope_user_id = _resolve_scope_user_id(session, installation_id)
        scope_channel_id = _resolve_scope_channel_id()
        print(
            f"Eval scope: installation={installation_id} "
            f"user={scope_user_id} channel={scope_channel_id}"
        )
        # Inject a no-op Slack client so the post-completion hook never reaches
        # Slack (the synthetic eval channel does not exist). LLM + tools still
        # run for real; only egress is stubbed.
        executor = AgentTaskExecutor(
            settings=settings,
            slack_client=cast(SlackPostingClient, _NoOpSlackClient()),
        )
        run_fn = build_live_run_fn(
            session,
            executor,
            installation_id,
            scope_user_id=scope_user_id,
            scope_channel_id=scope_channel_id,
        )
        return score_orchestration(SEED_ORCHESTRATION_CASES, run_fn)


def _main() -> None:
    report = run()
    print(f"\nORCHESTRATION EVAL: {report.summary_line()}\n")
    for result in report.results:
        if result.skipped:
            print(f"  [SKIP] case {result.case_id}: {result.request!r}")
            print(f"         reason: {result.skip_reason}")
            continue
        status = "PASS" if result.passed else "FAIL"
        apps_label = (
            f"called={sorted(result.called_apps)!r} "
            f"expected={sorted(result.expected_apps)!r}"
        )
        print(f"  [{status}] case {result.case_id}: {result.request!r}")
        print(f"         {apps_label}")
        for failure in result.failures:
            print(f"         ! {failure}")
    print()
    if not report.failures:
        print("All non-skipped cases passed.")
    else:
        print(f"{report.failed} case(s) failed — see above for details.")


if __name__ == "__main__":
    _main()
