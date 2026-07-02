"""Tests for HIG-209 Part 3: in-thread Composio OAuth connect + auto-resume."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.composio.client import (
    ComposioAuthConfig,
    ComposioClient,
    ComposioConnectedAccountDetails,
    ComposioConnectionError,
    ComposioConnectionRequest,
    ComposioToolExecution,
)
from kortny.composio.connect import (
    WAIT_AUTH_RECOVERY_ACTION,
    ComposioConnectionRequired,
    connect_prompt_text,
    initiate_connect_for_task,
    park_payload,
    resume_parked_connect_tasks,
)
from kortny.db.models import (
    ComposioConnection,
    Installation,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.models import TaskStatus as DbTaskStatus
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.queue import TaskQueue
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for composio connect tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (
        ComposioConnection,
        SlackSideEffect,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def _create_task(
    session: Session,
    *,
    channel_id: str = "CResearch",
    user_id: str = "UAnalyst",
) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1779660000.000001",
        slack_message_ts="1779660000.000001",
        slack_user_id=user_id,
        input="Use Notion to find the launch doc",
    )


class FakeConnectClient(ComposioClient):
    """Fake Composio client for the connect-initiate flow."""

    def __init__(self) -> None:
        super().__init__(api_key="fake")
        self.auth_config_calls: list[str] = []
        self.connect_calls: list[dict[str, Any]] = []

    def create_managed_auth_config(self, *, toolkit_slug: str) -> ComposioAuthConfig:
        self.auth_config_calls.append(toolkit_slug)
        return ComposioAuthConfig(
            id=f"ac_{toolkit_slug}",
            name=f"{toolkit_slug} managed auth",
            toolkit_slug=toolkit_slug,
            auth_scheme="oauth2",
            is_composio_managed=True,
            enabled=True,
        )

    def create_connect_link(
        self,
        *,
        user_id: str,
        auth_config_id: str,
        callback_url: str,
    ) -> ComposioConnectionRequest:
        self.connect_calls.append(
            {
                "user_id": user_id,
                "auth_config_id": auth_config_id,
                "callback_url": callback_url,
            }
        )
        return ComposioConnectionRequest(
            id="ln_connect_1",
            redirect_url="https://connect.composio.dev/redirect/abc",
            status="pending",
            connected_account_id=None,
        )


class FakeImmediateConnectClient(ComposioClient):
    """Fake client whose connect link hands back an immediate account id.

    Mirrors the rare no_auth-shaped case: most OAuth toolkits only learn
    ``connected_account_id`` once the provider redirects back through the
    dashboard's ``/composio/callback`` route, but a handful resolve it
    synchronously from the connect-link response itself. Both paths run
    external-identity capture; this fake exercises the immediate one.
    """

    def __init__(
        self,
        *,
        connected_account_id: str = "ca_immediate",
        details: ComposioConnectedAccountDetails | None = None,
        fetch_error: Exception | None = None,
        whoami_execution: ComposioToolExecution | None = None,
    ) -> None:
        super().__init__(api_key="fake")
        self._connected_account_id = connected_account_id
        self._details = details
        self._fetch_error = fetch_error
        self._whoami_execution = whoami_execution
        self.get_connected_account_calls: list[str] = []
        self.execute_calls: list[dict[str, Any]] = []

    def create_managed_auth_config(self, *, toolkit_slug: str) -> ComposioAuthConfig:
        return ComposioAuthConfig(
            id=f"ac_{toolkit_slug}",
            name=f"{toolkit_slug} managed auth",
            toolkit_slug=toolkit_slug,
            auth_scheme="oauth2",
            is_composio_managed=True,
            enabled=True,
        )

    def create_connect_link(
        self,
        *,
        user_id: str,
        auth_config_id: str,
        callback_url: str,
    ) -> ComposioConnectionRequest:
        return ComposioConnectionRequest(
            id="ln_connect_immediate",
            redirect_url="https://connect.composio.dev/redirect/immediate",
            status="active",
            connected_account_id=self._connected_account_id,
        )

    def get_connected_account(
        self, connected_account_id: str
    ) -> ComposioConnectedAccountDetails:
        self.get_connected_account_calls.append(connected_account_id)
        if self._fetch_error is not None:
            raise self._fetch_error
        assert self._details is not None
        return self._details

    def execute_tool(
        self,
        *,
        tool_slug: str,
        user_id: str,
        connected_account_id: str | None = None,
        arguments: dict[str, Any],
        version: str | None = None,
    ) -> ComposioToolExecution:
        self.execute_calls.append(
            {
                "tool_slug": tool_slug,
                "user_id": user_id,
                "connected_account_id": connected_account_id,
                "arguments": arguments,
            }
        )
        if self._whoami_execution is None:
            raise AssertionError("unexpected whoami execution in this test")
        return self._whoami_execution


def _task_events(session: Session, task: Task) -> list[TaskEvent]:
    session.flush()
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


# --- Initiate connect -------------------------------------------------------


def test_initiate_connect_creates_pending_user_scoped_connection(
    db_session: Session,
) -> None:
    task = _create_task(db_session)
    client = FakeConnectClient()

    prompt = initiate_connect_for_task(
        db_session,
        task=task,
        toolkit_slug="Notion",
        client=client,
        callback_url="http://localhost:8080/composio/callback",
    )
    db_session.commit()

    assert prompt.redirect_url == "https://connect.composio.dev/redirect/abc"
    assert prompt.scope_type == "user"
    assert prompt.scope_id == "UAnalyst"
    assert client.auth_config_calls == ["notion"]
    assert client.connect_calls[0]["auth_config_id"] == "ac_notion"

    connection = db_session.scalar(
        select(ComposioConnection).where(ComposioConnection.id == prompt.connection_id)
    )
    assert connection is not None
    assert connection.status == "pending"
    assert connection.toolkit_slug == "notion"
    assert connection.visibility_scope_type == "user"
    assert connection.visibility_scope_id == "UAnalyst"
    assert connection.connection_request_id == "ln_connect_1"


def test_initiate_connect_captures_external_label_when_account_id_immediate(
    db_session: Session,
) -> None:
    task = _create_task(db_session)
    client = FakeImmediateConnectClient(
        connected_account_id="ca_immediate",
        details=ComposioConnectedAccountDetails(
            id="ca_immediate",
            status="active",
            toolkit_slug="notion",
            state_val={"login": "notion-user-x"},
        ),
    )

    prompt = initiate_connect_for_task(
        db_session,
        task=task,
        toolkit_slug="Notion",
        client=client,
        callback_url="http://localhost:8080/composio/callback",
    )
    db_session.commit()

    assert client.get_connected_account_calls == ["ca_immediate"]
    connection = db_session.scalar(
        select(ComposioConnection).where(ComposioConnection.id == prompt.connection_id)
    )
    assert connection is not None
    assert connection.external_account_label == "notion-user-x"


def test_initiate_connect_captures_label_via_whoami_fallback(
    db_session: Session,
) -> None:
    """Details payload lacks a handle -> the toolkit's whoami tool supplies it."""

    task = _create_task(db_session)
    client = FakeImmediateConnectClient(
        connected_account_id="ca_immediate",
        details=ComposioConnectedAccountDetails(
            id="ca_immediate",
            status="active",
            toolkit_slug="github",
            state_val={"oauth_token": "gho_test"},
        ),
        whoami_execution=ComposioToolExecution(
            data={"login": "octocat-connect"},
            successful=True,
            error=None,
            log_id=None,
            session_info=None,
        ),
    )

    prompt = initiate_connect_for_task(
        db_session,
        task=task,
        toolkit_slug="GitHub",
        client=client,
        callback_url="http://localhost:8080/composio/callback",
    )
    db_session.commit()

    connection = db_session.scalar(
        select(ComposioConnection).where(ComposioConnection.id == prompt.connection_id)
    )
    assert connection is not None
    assert connection.external_account_label == "octocat-connect"
    assert client.execute_calls == [
        {
            "tool_slug": "GITHUB_GET_THE_AUTHENTICATED_USER",
            "user_id": connection.composio_user_id,
            "connected_account_id": "ca_immediate",
            "arguments": {},
        }
    ]


def test_initiate_connect_leaves_label_none_when_identity_fetch_raises(
    db_session: Session,
) -> None:
    task = _create_task(db_session)
    client = FakeImmediateConnectClient(
        connected_account_id="ca_immediate",
        fetch_error=ComposioConnectionError("composio is down"),
    )

    prompt = initiate_connect_for_task(
        db_session,
        task=task,
        toolkit_slug="Notion",
        client=client,
        callback_url="http://localhost:8080/composio/callback",
    )
    db_session.commit()

    connection = db_session.scalar(
        select(ComposioConnection).where(ComposioConnection.id == prompt.connection_id)
    )
    assert connection is not None
    assert connection.connected_account_id == "ca_immediate"
    assert connection.external_account_label is None


def test_initiate_connect_leaves_label_none_when_payload_lacks_handle(
    db_session: Session,
) -> None:
    task = _create_task(db_session)
    client = FakeImmediateConnectClient(
        connected_account_id="ca_immediate",
        details=ComposioConnectedAccountDetails(
            id="ca_immediate",
            status="active",
            toolkit_slug="notion",
            state_val={"instanceName": "my-instance"},
        ),
    )

    prompt = initiate_connect_for_task(
        db_session,
        task=task,
        toolkit_slug="Notion",
        client=client,
        callback_url="http://localhost:8080/composio/callback",
    )
    db_session.commit()

    connection = db_session.scalar(
        select(ComposioConnection).where(ComposioConnection.id == prompt.connection_id)
    )
    assert connection is not None
    assert connection.external_account_label is None


def test_initiate_connect_skips_capture_when_no_immediate_account_id(
    db_session: Session,
) -> None:
    """The common OAuth case: no connected_account_id yet, so no fetch happens."""

    task = _create_task(db_session)
    client = FakeConnectClient()

    initiate_connect_for_task(
        db_session,
        task=task,
        toolkit_slug="Notion",
        client=client,
        callback_url="http://localhost:8080/composio/callback",
    )
    db_session.commit()

    # FakeConnectClient does not implement get_connected_account; if capture
    # had been attempted this would raise AttributeError instead of the
    # commit above succeeding cleanly.


def test_connect_prompt_text_includes_link() -> None:
    text = connect_prompt_text(
        toolkit_slug="notion", redirect_url="https://connect/abc"
    )
    assert "notion" in text
    assert "https://connect/abc" in text


# --- Park (waiting_approval + wait_auth marker) -----------------------------


def test_park_writes_wait_auth_marker_and_parks_task(db_session: Session) -> None:
    task = _create_task(db_session)
    db_session.flush()
    service = TaskService(db_session)

    request = park_payload(
        toolkit_slug="notion",
        tool_name="composio_notion_search",
        connection_id=uuid.uuid4(),
        scope_type="user",
        scope_id="UAnalyst",
        prompt_message_ts="1779660001.000001",
    )
    # Mirror the worker: write the approval-required marker, then park.
    from kortny.approvals import TOOL_APPROVAL_REQUIRED_MESSAGE

    service.append_event(
        task,
        TaskEventType.log,
        {"message": TOOL_APPROVAL_REQUIRED_MESSAGE, "request": request},
    )
    service.mark_waiting_for_tool_approval(
        task, request=request, prompt_message_ts="1779660001.000001"
    )
    db_session.commit()

    assert task.status is DbTaskStatus.waiting_approval
    pending = service.latest_pending_tool_approval(task)
    assert pending is not None
    assert pending["recovery_action"] == WAIT_AUTH_RECOVERY_ACTION
    assert pending["toolkit_slug"] == "notion"
    assert pending["approval_key"] == "composio-connect:notion"


# --- Resume (ambient tick requeues when connected) --------------------------


def _park_connect_task(
    session: Session,
    task: Task,
    *,
    toolkit_slug: str = "notion",
    scope_type: str = "user",
    scope_id: str | None = "UAnalyst",
) -> None:
    from kortny.approvals import TOOL_APPROVAL_REQUIRED_MESSAGE

    service = TaskService(session)
    request = park_payload(
        toolkit_slug=toolkit_slug,
        tool_name=f"composio_{toolkit_slug}_search",
        connection_id=uuid.uuid4(),
        scope_type=scope_type,
        scope_id=scope_id,
        prompt_message_ts="1779660001.000001",
    )
    service.append_event(
        task,
        TaskEventType.log,
        {"message": TOOL_APPROVAL_REQUIRED_MESSAGE, "request": request},
    )
    service.mark_waiting_for_tool_approval(
        task, request=request, prompt_message_ts="1779660001.000001"
    )


def test_resume_requeues_parked_task_when_connected(db_session: Session) -> None:
    task = _create_task(db_session)
    _park_connect_task(db_session, task)
    db_session.commit()
    assert task.status is DbTaskStatus.waiting_approval

    # No active connection yet -> resume is a no-op.
    result_before = resume_parked_connect_tasks(
        db_session, installation_id=task.installation_id
    )
    db_session.commit()
    assert result_before.requeued_task_ids == ()
    assert task.status is DbTaskStatus.waiting_approval

    # The user now connects their Notion account in scope.
    db_session.add(
        ComposioConnection(
            installation_id=task.installation_id,
            toolkit_slug="notion",
            auth_config_id="ac_notion",
            connected_account_id="ca_notion_user",
            connection_request_id="ln_connect_1",
            composio_user_id=f"slack:{task.installation_id}:UAnalyst",
            owner_slack_user_id="UAnalyst",
            visibility_scope_type="user",
            visibility_scope_id="UAnalyst",
            status="active",
        )
    )
    db_session.commit()

    result = resume_parked_connect_tasks(
        db_session, installation_id=task.installation_id
    )
    db_session.commit()

    assert result.requeued_task_ids == (task.id,)
    assert result.resumed_toolkits == ("notion",)
    db_session.refresh(task)
    assert task.status is DbTaskStatus.pending

    # The requeued task is claimable by a worker.
    claimed = TaskQueue(db_session).claim_next(worker_id="worker-1")
    assert claimed is not None
    assert claimed.id == task.id

    # A resume log marker was written.
    messages = [
        event.payload.get("message") for event in _task_events(db_session, task)
    ]
    assert "composio_connect_resumed" in messages


def test_resume_ignores_parked_task_when_connection_out_of_scope(
    db_session: Session,
) -> None:
    task = _create_task(db_session, user_id="UAnalyst")
    _park_connect_task(db_session, task, scope_type="user", scope_id="UAnalyst")
    # A connection exists, but for a DIFFERENT user scope.
    db_session.add(
        ComposioConnection(
            installation_id=task.installation_id,
            toolkit_slug="notion",
            auth_config_id="ac_notion",
            connected_account_id="ca_notion_other",
            composio_user_id=f"slack:{task.installation_id}:UOther",
            owner_slack_user_id="UOther",
            visibility_scope_type="user",
            visibility_scope_id="UOther",
            status="active",
        )
    )
    db_session.commit()

    result = resume_parked_connect_tasks(
        db_session, installation_id=task.installation_id
    )
    db_session.commit()

    assert result.requeued_task_ids == ()
    db_session.refresh(task)
    assert task.status is DbTaskStatus.waiting_approval


def test_composio_connection_required_carries_toolkit() -> None:
    exc = ComposioConnectionRequired(
        toolkit_slug="notion", tool_name="composio_notion_search"
    )
    assert exc.toolkit_slug == "notion"
    assert exc.tool_name == "composio_notion_search"


# --- Worker park (posts connect link once + parks) --------------------------


class FakePostingSlackClient:
    """Fake Slack client capturing chat_postMessage calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.calls.append({"channel": channel, "text": text, "thread_ts": thread_ts})
        return {
            "ok": True,
            "channel": channel,
            "ts": f"1779660002.{len(self.calls):06d}",
        }


def _connect_settings() -> Any:
    from kortny.config.settings import Settings

    assert TEST_POSTGRES_URL is not None
    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "sign",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "test-llm-key",
            "LLM_MODEL": "openai/gpt-4o-mini",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": TEST_POSTGRES_URL,
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
        }
    )


def test_worker_parks_for_connect_and_posts_link_once(db_session: Session) -> None:
    from kortny.worker import AgentTaskExecutor

    task = _create_task(db_session)
    db_session.commit()
    service = TaskService(db_session)
    service.transition(task, DbTaskStatus.running)
    db_session.commit()

    composio_client = FakeConnectClient()
    slack_client = FakePostingSlackClient()
    executor = AgentTaskExecutor(
        settings=_connect_settings(),
        composio_client=composio_client,
        slack_client=slack_client,  # type: ignore[arg-type]
    )

    connect = ComposioConnectionRequired(
        toolkit_slug="notion", tool_name="composio_notion_search"
    )
    executor._park_for_composio_connect(
        settings=_connect_settings(),
        session=db_session,
        task=task,
        task_service=service,
        connect=connect,
    )
    db_session.commit()

    # Task parked on the existing waiting_approval status.
    db_session.refresh(task)
    assert task.status is DbTaskStatus.waiting_approval

    # The connect link was posted exactly once with the redirect URL.
    connect_posts = [
        call for call in slack_client.calls if "connect.composio.dev" in call["text"]
    ]
    assert len(connect_posts) == 1

    # A pending user-scoped connection row exists for the toolkit.
    connection = db_session.scalar(
        select(ComposioConnection).where(
            ComposioConnection.installation_id == task.installation_id,
            ComposioConnection.toolkit_slug == "notion",
        )
    )
    assert connection is not None
    assert connection.status == "pending"
    assert connection.visibility_scope_type == "user"

    # The wait_auth marker is discoverable for the resume scan.
    pending = service.latest_pending_tool_approval(task)
    assert pending is not None
    assert pending["recovery_action"] == WAIT_AUTH_RECOVERY_ACTION
    assert pending["toolkit_slug"] == "notion"
