"""Tests for external-identity capture on Composio connections.

Covers the read-side resolver (``resolve_external_identities``), the
best-effort handle extraction (``extract_external_handle``), and the
non-blocking capture call (``capture_external_identity``).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

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
)
from kortny.composio.identity import (
    capture_external_identity,
    extract_external_handle,
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


# --- capture_external_identity -----------------------------------------------


class _FakeClient(ComposioClient):
    def __init__(self, details: ComposioConnectedAccountDetails) -> None:
        super().__init__(api_key="fake")
        self._details = details
        self.calls: list[str] = []

    def get_connected_account(
        self, connected_account_id: str
    ) -> ComposioConnectedAccountDetails:
        self.calls.append(connected_account_id)
        return self._details


class _RaisingClient(ComposioClient):
    def __init__(self, exc: Exception) -> None:
        super().__init__(api_key="fake")
        self._exc = exc
        self.calls: list[str] = []

    def get_connected_account(
        self, connected_account_id: str
    ) -> ComposioConnectedAccountDetails:
        self.calls.append(connected_account_id)
        raise self._exc


def test_capture_external_identity_sets_label_on_success(db_session: Session) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _FakeClient(
        ComposioConnectedAccountDetails(
            id="ca_gh",
            status="active",
            toolkit_slug="github",
            state_val={"login": "octocat-x"},
        )
    )

    capture_external_identity(client, connection)

    assert client.calls == ["ca_gh"]
    assert connection.external_account_label == "octocat-x"


def test_capture_external_identity_leaves_none_on_client_error(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _RaisingClient(ComposioConnectionError("boom"))

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_external_identity_leaves_none_on_catalog_error(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _RaisingClient(ComposioCatalogError("rate limited"))

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_external_identity_leaves_none_when_payload_lacks_handle(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id="ca_gh")
    client = _FakeClient(
        ComposioConnectedAccountDetails(
            id="ca_gh",
            status="active",
            toolkit_slug="github",
            state_val={"oauth_token": "gho_abcdef"},
        )
    )

    capture_external_identity(client, connection)

    assert connection.external_account_label is None


def test_capture_external_identity_noop_without_connected_account_id(
    db_session: Session,
) -> None:
    installation = _installation(db_session)
    connection = _connection(db_session, installation, connected_account_id=None)
    client = _FakeClient(
        ComposioConnectedAccountDetails(
            id="ca_gh",
            status="active",
            toolkit_slug="github",
            state_val={"login": "octocat-x"},
        )
    )

    capture_external_identity(client, connection)

    assert client.calls == []
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
