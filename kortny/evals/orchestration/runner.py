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

``any_tool_called`` counts every ``tool_result`` row (Composio, native, MCP) so
the ``must_use_tools`` context-leak guard fires whenever no tool ran at all.

Usage::

    uv run python -m kortny.evals.orchestration.runner
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.config import load_settings
from kortny.db import make_session_factory
from kortny.db.models import Installation, Task, TaskEvent, TaskEventType
from kortny.evals.orchestration.cases import SEED_ORCHESTRATION_CASES, OrchestrationCase
from kortny.evals.orchestration.scoring import (
    OrchestrationReport,
    RunFn,
    RunResult,
    score_orchestration,
)
from kortny.tasks import TaskIdentity, TaskService
from kortny.worker.agent_executor import AgentTaskExecutor

# Sentinel values used when creating an eval task — they never route to Slack
# because AgentTaskExecutor.execute() is called directly (no Slack client).
_EVAL_CHANNEL_ID = "EVAL_ORCHESTRATION"
_EVAL_USER_ID = "EVAL_RUNNER"
_EVAL_THREAD_TS = "0.000000"


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
    """Read tool_result TaskEvent rows and return the called-apps set.

    Derives the called-apps set from the authoritative ``toolkit_slug`` carried
    in each successful Composio ``tool_result`` output, falling back to the
    runtime-name parse only when that field is absent.

    Returns:
        A tuple of (called_toolkit_slugs, any_tool_called) where
        called_toolkit_slugs is the set of Composio toolkit slugs whose tools
        completed successfully and any_tool_called is True if at least one tool
        of any kind produced a result (including native/MCP tools, for the
        must_use_tools guard).
    """
    rows = list(
        session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task_id,
                TaskEvent.type == TaskEventType.tool_result,
            )
        )
    )
    any_tool_called = len(rows) > 0
    slugs: set[str] = set()
    for row in rows:
        slug = _toolkit_slug_from_tool_result(row.payload)
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


def build_live_run_fn(
    session: Session,
    executor: AgentTaskExecutor,
    installation_id: uuid.UUID,
) -> RunFn:
    """Build a RunFn that executes one case through the real agent.

    Creates a ``manual`` Task with the case's request text, runs
    ``AgentTaskExecutor.execute()`` synchronously, then reads the task's
    ``TaskEvent`` rows of type ``tool_call`` to derive the called-apps set.

    Requires:
    - A live Postgres session connected to the target install.
    - A configured ``AgentTaskExecutor`` (LLM provider, settings).
    - The ``installation_id`` of the target Kortny install.

    The task uses dummy Slack identifiers (EVAL_ORCHESTRATION channel,
    EVAL_RUNNER user) — the executor is called directly so no Slack messages
    are posted (no slack_client is wired in by default).
    """
    task_service = TaskService(session)

    def run(case: OrchestrationCase) -> RunResult:
        # Each case gets a unique identity so the dedup logic never collapses
        # distinct runs into the same task row.
        unique_ts = f"{uuid.uuid4().int % 10**9}.{uuid.uuid4().int % 10**6}"
        identity = TaskIdentity.manual(
            channel_id=_EVAL_CHANNEL_ID,
            thread_ts=_EVAL_THREAD_TS,
            user_id=_EVAL_USER_ID,
            input_text=f"{case.request}::{unique_ts}",
        )
        task: Task = task_service.create_task(
            installation_id=installation_id,
            slack_channel_id=_EVAL_CHANNEL_ID,
            slack_user_id=_EVAL_USER_ID,
            slack_thread_ts=_EVAL_THREAD_TS,
            slack_message_ts=unique_ts,
            input=case.request,
            identity=identity,
        )
        session.commit()

        result = executor.execute(
            session=session,
            task=task,
            task_service=task_service,
        )
        session.commit()

        called_apps, any_tool_called = _called_apps_from_events(session, task.id)
        answer = result.result_summary or ""
        return called_apps, any_tool_called, answer

    return run


def run() -> OrchestrationReport:
    """Run the full seed eval against the live install and return the report."""
    settings = load_settings()
    session_factory = make_session_factory(database_url=settings.postgres_url)
    with session_factory() as session:
        installation_id = _first_installation_id(session)
        executor = AgentTaskExecutor(settings=settings)
        run_fn = build_live_run_fn(session, executor, installation_id)
        return score_orchestration(SEED_ORCHESTRATION_CASES, run_fn)


def _main() -> None:
    report = run()
    print(f"\nORCHESTRATION EVAL: {report.summary_line()}\n")
    for result in report.results:
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
        print("All cases passed.")
    else:
        print(f"{report.failed} case(s) failed — see above for details.")


if __name__ == "__main__":
    _main()
