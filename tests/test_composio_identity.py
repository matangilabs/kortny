"""Tests for external-identity capture on Composio connections.

Covers the read-side resolver (``resolve_external_identities``), the
best-effort handle extraction (``extract_external_handle`` /
``extract_whoami_handle``), and the non-blocking two-stage capture call
(``capture_external_identity``: details payload, then whoami-execution
fallback for registered toolkits).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.composio.client import (
    ComposioCatalogError,
    ComposioClient,
    ComposioConnectedAccountDetails,
    ComposioConnectionError,
    ComposioToolExecution,
)
from kortny.composio.identity import (
    WHOAMI_REGISTRY,
    capture_external_identity,
    extract_external_handle,
    extract_whoami_handle,
    resolve_external_identities,
)
from kortny.db.models import ComposioConnection, Installation
from kortny.db.session import make_engine, make_session_factory, normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for composio identity tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


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
    for model in (ComposioConnection, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def _connection(
    session: Session,
    installation: Installation,
    *,
    toolkit_slug: str = "github",
    owner_slack_user_id: str = "UDev",
    connected_account_id: str | None = "ca_1",
    status: str = "active",
    external_account_label: str | None = None,
) -> ComposioConnection:
    connection = ComposioConnection(
        installation_id=installation.id,
        toolkit_slug=toolkit_slug,
        auth_config_id="ac_1",
        connected_account_id=connected_account_id,
        composio_user_id=f"slack:{installation.id}:{owner_slack_user_id}",
        owner_slack_user_id=owner_slack_user_id,
        visibility_scope_type="user",
        visibility_scope_id=owner_slack_user_id,
        status=status,
        external_account_label=external_account_label,
    )
    session.add(connection)
    session.flush()
    return connection


# --- extract_external_handle -------------------------------------------------


def test_extract_external_handle_prefers_username_over_email() -> None:
    handle = extract_external_handle(
        {"login": "octocat-x", "email": "octocat-x@example.test"}
    )
    assert handle == "octocat-x"


def test_extract_external_handle_falls_back_to_email() -> None:
    handle = extract_external_handle({"account_email": "octocat-x@example.test"})
    assert handle == "octocat-x@example.test"


def test_extract_external_handle_returns_none_for_unrelated_keys() -> None:
    # These are the kind of dynamic connection-parameter keys Composio's
    # connected-account state.val payload actually carries (verified against
    # the live API reference): none of them are identity-shaped.
    handle = extract_external_handle(
        {
            "oauth_token": "gho_abcdef",
            "instanceName": "my-instance",
            "subdomain": "acme",
        }
    )
    assert handle is None


def test_extract_external_handle_truncates_to_max_length() -> None:
    handle = extract_external_handle({"login": "x" * 500})
    assert handle is not None
    assert len(handle) == 120


# --- extract_whoami_handle ----------------------------------------------------


def test_whoami_registry_covers_verified_toolkits_only() -> None:
    assert WHOAMI_REGISTRY["github"].tool_slug == "GITHUB_GET_THE_AUTHENTICATED_USER"
    assert WHOAMI_REGISTRY["linear"].tool_slug == "LINEAR_GET_CURRENT_USER"
    assert WHOAMI_REGISTRY["gmail"].tool_slug == "GMAIL_GET_PROFILE"
    # No verified whoami tool exists for googlecalendar — deliberately absent.
    assert "googlecalendar" not in WHOAMI_REGISTRY


def test_extract_whoami_handle_github_login() -> None:
    assert extract_whoami_handle("github", {"login": "octocat-x"}) == "octocat-x"


def test_extract_whoami_handle_github_nested_details() -> None:
    payload = {"details": {"login": "octocat-x", "id": 12345}}
    assert extract_whoami_handle("github", payload) == "octocat-x"


def test_extract_whoami_handle_linear_prefers_name_then_email() -> None:
    assert (
        extract_whoami_handle(
            "linear", {"name": "Dev X", "email": "dev-x@example.test"}
        )
        == "Dev X"
    )
    assert (
        extract_whoami_handle("linear", {"viewer": {"email": "dev-x@example.test"}})
        == "dev-x@example.test"
    )


def test_extract_whoami_handle_gmail_email_address() -> None:
    payload = {"emailAddress": "dev-x@example.test", "messagesTotal": 100}
    assert extract_whoami_handle("gmail", payload) == "dev-x@example.test"


def test_extract_whoami_handle_unknown_toolkit_returns_none() -> None:
    assert extract_whoami_handle("notion", {"login": "someone"}) is None


def test_extract_whoami_handle_missing_fields_returns_none() -> None:
    assert extract_whoami_handle("github", {"id": "12345", "type": "User"}) is None


# --- capture_external_identity -----------------------------------------------


class _FakeClient(ComposioClient):
    """Configurable identity-capture fake: stubs BOTH capture stages.

    ``execute_tool`` is always stubbed (never inherits the real REST call) so
    no test can accidentally reach the network when the whoami fallback fires.
    """

    def __init__(
        self,
        *,
        details: ComposioConnectedAccountDetails | None = None,
        details_error: Exception | None = None,
        whoami_execution: ComposioToolExecution | None = None,
        whoami_error: Exception | None = None,
    ) -> None:
        super().__init__(api_key="fake")
        self._details = details
        self._details_error = details_error
        self._whoami_execution = whoami_execution
        self._whoami_error = whoami_error
        self.calls: list[str] = []
        self.execute_calls: list[dict[str, Any]] = []

    def get_connected_account(
        self, connected_account_id: str
    ) -> ComposioConnectedAccountDetails:
        self.calls.append(connected_account_id)
        if self._details_error is not None:
            raise self._details_error
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
        if self._whoami_error is not None:
            raise self._whoami_error
        if self._whoami_execution is None:
            raise AssertionError("unexpected whoami execution in this test")
        return self._whoami_execution


def _details(
    toolkit_slug: str,
    state_val: dict[str, Any],
    account_id: str = "ca_1",
) -> ComposioConnectedAccountDetails:
    return ComposioConnectedAccountDetails(
        id=account_id,
        status="active",
        toolkit_slug=toolkit_slug,
        state_val=state_val,
    )


def test_capture_external_identity_sets_label_on_success(db_session: Session) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _FakeClient(details=_details("github", {"login": "octocat-x"}, "ca_gh"))

    capture_external_identity(client, connection)

    assert client.calls == ["ca_gh"]
    # Details stage resolved the handle; the whoami fallback never fires.
    assert client.execute_calls == []
    assert connection.external_account_label == "octocat-x"


def test_capture_external_identity_leaves_none_on_client_error(
    db_session: Session,
) -> None:
    # notion has no whoami registry entry, so a details failure is terminal.
    installation = _installation(db_session)
    connection = _connection(
        db_session,
        installation,
        toolkit_slug="notion",
        connected_account_id="ca_no",
    )
    client = _FakeClient(details_error=ComposioConnectionError("boom"))

    capture_external_identity(client, connection)

    assert client.execute_calls == []
    assert connection.external_account_label is None


def test_capture_external_identity_leaves_none_on_catalog_error(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(
        db_session,
        installation,
        toolkit_slug="notion",
        connected_account_id="ca_no",
    )
    client = _FakeClient(details_error=ComposioCatalogError("rate limited"))

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_external_identity_leaves_none_when_payload_lacks_handle(
    db_session: Session,
) -> None:
    # notion: details payload lacks identity fields and no whoami fallback
    # exists -> label stays None and no execution is attempted.
    installation = _installation(db_session)
    connection = _connection(
        db_session,
        installation,
        toolkit_slug="notion",
        connected_account_id="ca_no",
    )
    client = _FakeClient(details=_details("notion", {"oauth_token": "gho_abcdef"}))

    capture_external_identity(client, connection)

    assert client.execute_calls == []
    assert connection.external_account_label is None


def test_capture_external_identity_noop_without_connected_account_id(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id=None)
    client = _FakeClient(details=_details("github", {"login": "octocat-x"}))

    capture_external_identity(client, connection)

    assert client.calls == []
    assert client.execute_calls == []
    assert connection.external_account_label is None


# --- capture: whoami-execution fallback ---------------------------------------


def test_capture_falls_back_to_whoami_when_details_lack_handle(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _FakeClient(
        details=_details("github", {"oauth_token": "gho_abcdef"}, "ca_gh"),
        whoami_execution=ComposioToolExecution(
            data={"login": "octocat-x", "id": 12345},
            successful=True,
            error=None,
            log_id="log_whoami",
            session_info=None,
        ),
    )

    capture_external_identity(client, connection)

    assert connection.external_account_label == "octocat-x"
    assert client.execute_calls == [
        {
            "tool_slug": "GITHUB_GET_THE_AUTHENTICATED_USER",
            "user_id": connection.composio_user_id,
            "connected_account_id": "ca_gh",
            "arguments": {},
        }
    ]


def test_capture_falls_back_to_whoami_when_details_fetch_fails(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(
        db_session,
        installation,
        toolkit_slug="gmail",
        connected_account_id="ca_gm",
    )
    client = _FakeClient(
        details_error=ComposioConnectionError("details endpoint down"),
        whoami_execution=ComposioToolExecution(
            data={"emailAddress": "dev-x@example.test"},
            successful=True,
            error=None,
            log_id=None,
            session_info=None,
        ),
    )

    capture_external_identity(client, connection)

    assert connection.external_account_label == "dev-x@example.test"
    assert client.execute_calls[0]["tool_slug"] == "GMAIL_GET_PROFILE"


def test_capture_whoami_failure_leaves_label_none(db_session: Session) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _FakeClient(
        details=_details("github", {}, "ca_gh"),
        whoami_error=ComposioConnectionError("execute failed"),
    )

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_whoami_unsuccessful_execution_leaves_label_none(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(
        db_session,
        installation,
        toolkit_slug="linear",
        connected_account_id="ca_ln",
    )
    client = _FakeClient(
        details=_details("linear", {}, "ca_ln"),
        whoami_execution=ComposioToolExecution(
            data={"name": "should-not-be-used"},
            successful=False,
            error="token expired",
            log_id=None,
            session_info=None,
        ),
    )

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_whoami_non_mapping_data_leaves_label_none(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _FakeClient(
        details=_details("github", {}, "ca_gh"),
        whoami_execution=ComposioToolExecution(
            data=["not", "a", "mapping"],
            successful=True,
            error=None,
            log_id=None,
            session_info=None,
        ),
    )

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_unknown_toolkit_never_attempts_whoami(db_session: Session) -> None:
    installation = _installation(db_session)
    connection = _connection(
        db_session,
        installation,
        toolkit_slug="notion",
        connected_account_id="ca_no",
    )
    client = _FakeClient(details=_details("notion", {}))

    capture_external_identity(client, connection)

    assert client.execute_calls == []
    assert connection.external_account_label is None


# --- resolve_external_identities ----------------------------------------------


def test_resolve_external_identities_returns_only_callers_active_labeled_connections(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    _connection(
        db_session,
        installation,
        toolkit_slug="github",
        owner_slack_user_id="UDev",
        connected_account_id="ca_dev_github",
        status="active",
        external_account_label="octocat-x",
    )
    # Different toolkit, same user, no label captured yet -> excluded.
    _connection(
        db_session,
        installation,
        toolkit_slug="notion",
        owner_slack_user_id="UDev",
        connected_account_id="ca_dev_notion",
        status="active",
        external_account_label=None,
    )
    # Same user, pending (not yet active) -> excluded even though labeled.
    _connection(
        db_session,
        installation,
        toolkit_slug="linear",
        owner_slack_user_id="UDev",
        connected_account_id="ca_dev_linear",
        status="pending",
        external_account_label="dev-x",
    )
    # Different user, active + labeled -> excluded.
    _connection(
        db_session,
        installation,
        toolkit_slug="github",
        owner_slack_user_id="UOther",
        connected_account_id="ca_other_github",
        status="active",
        external_account_label="other-x",
    )
    db_session.commit()

    identities = resolve_external_identities(db_session, installation.id, "UDev")

    assert identities == {"github": "octocat-x"}


def test_resolve_external_identities_empty_when_no_labels(db_session: Session) -> None:
    installation = _installation(db_session)
    _connection(
        db_session,
        installation,
        owner_slack_user_id="UDev",
        status="active",
        external_account_label=None,
    )
    db_session.commit()

    identities = resolve_external_identities(db_session, installation.id, "UDev")

    assert identities == {}
