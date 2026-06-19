"""DB-backed tests for the interactive-action lifecycle + security (HIG-255 s2).

The security model is the point here: the opaque key is only the lookup; a click
is honored only when actor + workspace + route match and the action is live,
under a row lock, idempotently. Wrong-actor/forged/expired clicks must NOT
consume the action.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import Installation, InteractiveAction
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.slack.interactions import (
    STATUS_CONSUMED,
    STATUS_CONSUMING,
    STATUS_EXPIRED,
    STATUS_SENT,
    STATUS_SUPERSEDED,
    ClaimStatus,
    InteractiveActionService,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for interactive-action tests",
)

NOW = datetime(2026, 6, 19, 12, 0, 0, tzinfo=UTC)
OWNER = "U_OWNER"
TEAM = "T_ACME"
CHANNEL = "C_MAIN"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "heads")
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
    from kortny.db.models import Artifact, Task, TaskEvent

    for model in (InteractiveAction, Artifact, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=TEAM, team_name="Acme")
    session.add(installation)
    session.flush()
    return installation


def _service(session: Session) -> InteractiveActionService:
    return InteractiveActionService(session, signing_key="test-signing-key")


def _mint_sent(
    session: Session,
    service: InteractiveActionService,
    installation: Installation,
    *,
    target_id: str = "approval:abc",
    allowed_user_id: str | None = OWNER,
    route: str = "kortny:v1:approval.approve",
    ttl: timedelta = timedelta(hours=1),
) -> str:
    minted = service.mint(
        installation_id=installation.id,
        action_kind="approve",
        route=route,
        target_type="approval",
        target_id=target_id,
        allowed_user_id=allowed_user_id,
        slack_team_id=TEAM,
        allowed_channel_id=CHANNEL,
        ttl=ttl,
        now=NOW,
    )
    service.mark_sent(
        minted.action,
        channel_id=CHANNEL,
        message_ts="1781000000.0001",
        block_id="b1",
        slack_action_id=route,
        now=NOW,
    )
    return minted.raw_key


def test_control_deck_mints_buttons_for_every_action(db_session: Session) -> None:
    import uuid

    from kortny.slack.document_decisions import (
        RERENDER_FORMATS,
        ROUTE_DOC_RERENDER,
        TARGET_DOCUMENT,
        render_control_deck,
    )

    installation = _installation(db_session)
    service = _service(db_session)
    group = uuid.uuid4()
    blocks, minted = render_control_deck(
        service,
        installation_id=installation.id,
        task_id=None,
        doc_group_id=group,
        doc_version=2,
        current_format="pdf",
        themes=["editorial", "minimal"],
        allowed_user_id=OWNER,
        allowed_channel_id=CHANNEL,
        slack_team_id=TEAM,
    )
    # 4 formats + 2 themes + 3 edits + 1 revert (v>1).
    assert len(minted) == len(RERENDER_FORMATS) + 2 + 3 + 1
    # Every button is a live row scoped to this document group.
    rows = db_session.scalars(
        select(InteractiveAction).where(
            InteractiveAction.target_type == TARGET_DOCUMENT,
            InteractiveAction.target_id == str(group),
        )
    ).all()
    assert len(rows) == len(minted)
    assert all(r.payload_json["base_version"] == 2 for r in rows)
    # Format buttons carry the rerender route + the target format.
    fmt_rows = [r for r in rows if r.action_kind == "format"]
    assert {r.payload_json["value"] for r in fmt_rows} == set(RERENDER_FORMATS)
    assert all(r.route == ROUTE_DOC_RERENDER for r in fmt_rows)
    # Block Kit shape: a section + three actions rows.
    assert blocks[0]["type"] == "section"
    assert [b["type"] for b in blocks[1:]] == ["actions", "actions", "actions"]


def test_control_deck_omits_revert_for_v1(db_session: Session) -> None:
    import uuid

    from kortny.slack.document_decisions import render_control_deck

    installation = _installation(db_session)
    service = _service(db_session)
    _, minted = render_control_deck(
        service,
        installation_id=installation.id,
        task_id=None,
        doc_group_id=uuid.uuid4(),
        doc_version=1,
        current_format="pdf",
        themes=["editorial"],
        allowed_user_id=OWNER,
        allowed_channel_id=CHANNEL,
        slack_team_id=TEAM,
    )
    assert not any(m.action.action_kind == "revert" for m in minted)


def test_process_document_action_spawns_rerender_child_in_thread(
    db_session: Session,
) -> None:
    import uuid

    from kortny.slack.document_decisions import (
        ROUTE_DOC_RERENDER,
        TARGET_DOCUMENT,
        process_document_action,
    )
    from kortny.tasks import TaskService
    from kortny.tasks.identity import TaskIdentity

    installation = _installation(db_session)
    service = _service(db_session)
    task_service = TaskService(db_session)
    parent = task_service.create_task(
        installation_id=installation.id,
        slack_channel_id=CHANNEL,
        slack_user_id=OWNER,
        input="make a report",
        slack_thread_ts="1781000000.0001",
        identity=TaskIdentity.manual(
            channel_id=CHANNEL,
            thread_ts="1781000000.0001",
            user_id=OWNER,
            input_text="make a report",
        ),
    )
    group = uuid.uuid4()
    minted = service.mint(
        installation_id=installation.id,
        action_kind="format",
        route=ROUTE_DOC_RERENDER,
        target_type=TARGET_DOCUMENT,
        target_id=str(group),
        task_id=parent.id,
        payload={
            "doc_group_id": str(group),
            "base_version": 1,
            "mode": "format",
            "value": "pptx",
        },
        allowed_user_id=OWNER,
        slack_team_id=TEAM,
        allowed_channel_id=CHANNEL,
        now=NOW,
    )

    child_id = process_document_action(
        db_session, minted.action, actor_user_id=OWNER, task_service=task_service
    )

    assert child_id is not None
    from kortny.db.models import Task

    child = db_session.get(Task, child_id)
    assert child is not None
    assert child.identity_payload["kind"] == "document_rerender"
    assert child.identity_payload["mode"] == "format"
    assert child.identity_payload["value"] == "pptx"
    assert child.parent_task_id == parent.id
    # Output lands in the original document's thread.
    assert child.slack_thread_ts == "1781000000.0001"


def test_process_document_action_edit_embeds_spec_and_lineage(
    db_session: Session,
) -> None:
    import uuid

    from kortny.db.models import Artifact, Task
    from kortny.slack.document_decisions import (
        ROUTE_DOC_EDIT,
        TARGET_DOCUMENT,
        process_document_action,
    )
    from kortny.tasks import TaskService
    from kortny.tasks.identity import TaskIdentity

    installation = _installation(db_session)
    service = _service(db_session)
    task_service = TaskService(db_session)
    parent = task_service.create_task(
        installation_id=installation.id,
        slack_channel_id=CHANNEL,
        slack_user_id=OWNER,
        input="make a report",
        slack_thread_ts="1781000000.0001",
        identity=TaskIdentity.manual(
            channel_id=CHANNEL,
            thread_ts="1781000000.0001",
            user_id=OWNER,
            input_text="make a report",
        ),
    )
    group = uuid.uuid4()
    db_session.add(
        Artifact(
            task_id=parent.id,
            filename="report.pdf",
            mime_type="application/pdf",
            doc_group_id=group,
            doc_version=1,
            spec_json={
                "title": "Q2 Report",
                "blocks": [{"type": "prose", "text": "hi"}],
            },
        )
    )
    db_session.flush()
    minted = service.mint(
        installation_id=installation.id,
        action_kind="edit",
        route=ROUTE_DOC_EDIT,
        target_type=TARGET_DOCUMENT,
        target_id=str(group),
        task_id=parent.id,
        payload={"doc_group_id": str(group), "base_version": 1, "value": "shorten"},
        allowed_user_id=OWNER,
        slack_team_id=TEAM,
        allowed_channel_id=CHANNEL,
        now=NOW,
    )

    child_id = process_document_action(
        db_session, minted.action, actor_user_id=OWNER, task_service=task_service
    )

    assert child_id is not None
    child = db_session.get(Task, child_id)
    assert child is not None
    assert child.identity_payload["kind"] == "document_edit"
    assert child.identity_payload["edit_kind"] == "shorten"
    # The agent gets the stored spec + explicit lineage so the edit stays the
    # same document's next version.
    assert "Q2 Report" in child.input
    assert str(group) in child.input
    assert "base_version=1" in child.input
    # The edit must keep the format the user is on (pdf here), not default away.
    assert 'format="pdf"' in child.input


def test_mint_stores_hash_not_raw_key(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    minted = service.mint(
        installation_id=installation.id,
        action_kind="approve",
        route="kortny:v1:approval.approve",
        target_type="approval",
        target_id="approval:abc",
        now=NOW,
    )
    assert minted.raw_key.startswith("iact_v1_")
    # The DB stores only the HMAC hash, never the bearer token.
    assert minted.action.action_key_hash != minted.raw_key
    assert minted.action.status == "pending_send"


def test_claim_ok_then_complete(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    raw = _mint_sent(db_session, service, installation)

    result = service.claim(
        raw, actor_user_id=OWNER, team_id=TEAM, channel_id=CHANNEL, now=NOW
    )
    assert result.status is ClaimStatus.ok
    assert result.action is not None
    assert result.action.status == STATUS_CONSUMING

    service.complete(result.action, consumed_by_user_id=OWNER, now=NOW)
    assert result.action.status == STATUS_CONSUMED
    assert result.action.consumed_by_user_id == OWNER


def test_second_claim_after_consume_is_already_handled(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    raw = _mint_sent(db_session, service, installation)

    first = service.claim(
        raw, actor_user_id=OWNER, team_id=TEAM, channel_id=CHANNEL, now=NOW
    )
    assert first.status is ClaimStatus.ok
    assert first.action is not None
    service.complete(first.action, consumed_by_user_id=OWNER, now=NOW)

    second = service.claim(
        raw, actor_user_id=OWNER, team_id=TEAM, channel_id=CHANNEL, now=NOW
    )
    assert second.status is ClaimStatus.already_handled


def test_wrong_user_denied_and_action_stays_usable(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    raw = _mint_sent(db_session, service, installation)

    denied = service.claim(
        raw, actor_user_id="U_INTRUDER", team_id=TEAM, channel_id=CHANNEL, now=NOW
    )
    assert denied.status is ClaimStatus.denied
    assert denied.action is not None
    assert denied.action.denied_count == 1
    assert denied.action.status == STATUS_SENT  # still usable

    # The legitimate owner can still claim it.
    ok = service.claim(
        raw, actor_user_id=OWNER, team_id=TEAM, channel_id=CHANNEL, now=NOW
    )
    assert ok.status is ClaimStatus.ok


def test_forged_key_is_not_found(db_session: Session) -> None:
    _installation(db_session)
    service = _service(db_session)
    result = service.claim(
        "iact_v1_totally-made-up", actor_user_id=OWNER, team_id=TEAM, channel_id=CHANNEL
    )
    assert result.status is ClaimStatus.not_found


def test_expired_action_is_rejected(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    raw = _mint_sent(db_session, service, installation, ttl=timedelta(minutes=5))
    later = NOW + timedelta(hours=1)
    result = service.claim(
        raw, actor_user_id=OWNER, team_id=TEAM, channel_id=CHANNEL, now=later
    )
    assert result.status is ClaimStatus.expired
    assert result.action is not None
    assert result.action.status == STATUS_EXPIRED


def test_route_mismatch_denied(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    raw = _mint_sent(db_session, service, installation)
    result = service.claim(
        raw,
        actor_user_id=OWNER,
        team_id=TEAM,
        channel_id=CHANNEL,
        route="kortny:v1:task.retry",  # wrong route for an approval action
        now=NOW,
    )
    assert result.status is ClaimStatus.denied


def test_cross_workspace_denied(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    raw = _mint_sent(db_session, service, installation)
    result = service.claim(
        raw, actor_user_id=OWNER, team_id="T_OTHER", channel_id=CHANNEL, now=NOW
    )
    assert result.status is ClaimStatus.denied


def test_supersede_siblings_retires_the_others(db_session: Session) -> None:
    installation = _installation(db_session)
    service = _service(db_session)
    approve = service.mint(
        installation_id=installation.id,
        action_kind="approve",
        route="kortny:v1:approval.approve",
        target_type="approval",
        target_id="approval:abc",
        now=NOW,
    )
    reject = service.mint(
        installation_id=installation.id,
        action_kind="reject",
        route="kortny:v1:approval.reject",
        target_type="approval",
        target_id="approval:abc",
        now=NOW,
    )
    superseded = service.supersede_siblings(
        installation_id=installation.id,
        target_type="approval",
        target_id="approval:abc",
        keep_id=approve.action.id,
    )
    assert superseded == 1
    db_session.refresh(reject.action)
    db_session.refresh(approve.action)
    assert reject.action.status == STATUS_SUPERSEDED
    assert approve.action.status != STATUS_SUPERSEDED
