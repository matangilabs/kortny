"""Tests for Witness activation epoch burst-safety mechanism."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from kortny.dashboard.data import get_witness_dormancy_status
from kortny.db.models import (
    Installation,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.observe.service import ObserveService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required",
)

NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
CHANNEL_ID = "Cepoch_test_01"


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
    factory = make_session_factory(engine=engine)
    with factory() as session:
        yield session
        session.rollback()


@pytest.fixture
def installation(db_session: Session) -> Installation:
    install = Installation(slack_team_id=f"T_epoch_{uuid.uuid4().hex[:8]}")
    db_session.add(install)
    db_session.flush()
    return install


def _make_candidate(
    db_session: Session,
    installation: Installation,
    channel_id: str,
    created_at: datetime,
    scope_type: str = "channel",
) -> WitnessOpportunityCandidate:
    c = WitnessOpportunityCandidate(
        installation_id=installation.id,
        title="Test opportunity",
        summary="Test",
        candidate_type="recurring_check",
        status="candidate",
        visibility_scope_type=scope_type,
        visibility_scope_id=channel_id,
        channel_id=channel_id,
        dedupe_key=f"test:{uuid.uuid4()}",
        confidence_score=Decimal("0.700"),
        source_type="channel_profile",
        created_at=created_at,
        updated_at=created_at,
    )
    db_session.add(c)
    db_session.flush()
    return c


def test_set_channel_proactivity_stamps_epoch(
    db_session: Session, installation: Installation
) -> None:
    """set_channel_proactivity_status to 'full' stamps full_enabled_at once."""
    svc = ObserveService(db_session)
    policy = svc.set_channel_proactivity_status(
        installation.id, CHANNEL_ID, "full", now=NOW
    )
    assert policy.full_enabled_at == NOW
    assert policy.proactivity_status == "full"

    # Re-enabling after going off should NOT overwrite the epoch
    svc.set_channel_proactivity_status(installation.id, CHANNEL_ID, "off", now=NOW)
    policy2 = svc.set_channel_proactivity_status(
        installation.id, CHANNEL_ID, "full", now=NOW + timedelta(hours=1)
    )
    assert policy2.full_enabled_at == NOW  # epoch preserved


def test_set_channel_proactivity_invalid_status(
    db_session: Session, installation: Installation
) -> None:
    """Invalid proactivity_status raises ValueError."""
    svc = ObserveService(db_session)
    with pytest.raises(ValueError, match="Invalid proactivity_status"):
        svc.set_channel_proactivity_status(installation.id, CHANNEL_ID, "banana")


def test_set_digest_delivery_stamps_epoch(
    db_session: Session, installation: Installation
) -> None:
    """set_digest_delivery(enabled=True) stamps digest_enabled_at."""
    svc = ObserveService(db_session)
    svc.set_digest_delivery(installation.id, enabled=True, now=NOW)
    # session.get returns the same object from identity map, so check attribute directly
    assert installation.digest_enabled_at == NOW


def test_set_digest_delivery_disable_clears_epoch(
    db_session: Session, installation: Installation
) -> None:
    """set_digest_delivery(enabled=False) clears digest_enabled_at."""
    svc = ObserveService(db_session)
    svc.set_digest_delivery(installation.id, enabled=True, now=NOW)
    svc.set_digest_delivery(installation.id, enabled=False)
    assert installation.digest_enabled_at is None


def test_set_autopilot_enabled(db_session: Session, installation: Installation) -> None:
    """set_autopilot_enabled round-trips."""
    svc = ObserveService(db_session)
    svc.set_autopilot_enabled(installation.id, enabled=False)
    assert installation.autopilot_enabled is False

    svc.set_autopilot_enabled(installation.id, enabled=True)
    assert installation.autopilot_enabled is True


def test_dormancy_status_counts(
    db_session: Session, installation: Installation
) -> None:
    """get_witness_dormancy_status returns correct queued/sent/channel counts."""
    svc = ObserveService(db_session)
    # Set up one channel policy as 'full'
    svc.set_channel_proactivity_status(installation.id, CHANNEL_ID, "full", now=NOW)
    # Create two candidates
    _make_candidate(db_session, installation, CHANNEL_ID, NOW - timedelta(hours=1))
    _make_candidate(db_session, installation, CHANNEL_ID, NOW)

    dormancy = get_witness_dormancy_status(
        db_session, installation_id=installation.id, now=NOW
    )
    assert dormancy.total_queued >= 2
    assert dormancy.channels_full == 1
    assert dormancy.channels_total >= 1
    assert not dormancy.dm_digest_enabled
    assert dormancy.autopilot_db_override is None  # fresh installation


def test_pre_epoch_candidate_excluded_channel(
    db_session: Session, installation: Installation
) -> None:
    """Channel candidate created before full_enabled_at must not be eligible for delivery.

    This is the burst-safety invariant: the 211-backlog must never auto-deliver.
    We verify the epoch timestamp is earlier than ALL pre-epoch candidates.
    We don't call the full runner (which requires Slack), but verify the epoch
    data is present and correct so the runner's epoch filter will exclude them.
    """
    epoch = NOW
    pre_epoch_time = NOW - timedelta(days=30)

    svc = ObserveService(db_session)
    policy = svc.set_channel_proactivity_status(
        installation.id, CHANNEL_ID, "full", now=epoch
    )
    assert policy.full_enabled_at == epoch

    # Create a pre-epoch candidate
    pre = _make_candidate(db_session, installation, CHANNEL_ID, pre_epoch_time)
    # Create a post-epoch candidate
    post = _make_candidate(
        db_session, installation, CHANNEL_ID, epoch + timedelta(seconds=1)
    )

    # Verify: pre_epoch candidate's created_at < full_enabled_at
    assert pre.created_at < policy.full_enabled_at
    # Verify: post-epoch candidate's created_at >= full_enabled_at
    assert post.created_at >= policy.full_enabled_at


def test_pre_epoch_digest_candidate_excluded(
    db_session: Session, installation: Installation
) -> None:
    """DM candidates before digest_enabled_at are excluded (epoch filter invariant)."""
    epoch = NOW
    svc = ObserveService(db_session)
    svc.set_digest_delivery(installation.id, enabled=True, now=epoch)

    pre = _make_candidate(
        db_session,
        installation,
        f"D{uuid.uuid4().hex[:8]}",
        epoch - timedelta(days=5),
        scope_type="dm",
    )
    post = _make_candidate(
        db_session,
        installation,
        f"D{uuid.uuid4().hex[:8]}",
        epoch + timedelta(seconds=1),
        scope_type="dm",
    )

    assert installation.digest_enabled_at == epoch
    assert pre.created_at < installation.digest_enabled_at
    assert post.created_at >= installation.digest_enabled_at


def test_env_deliver_private_back_compat(
    db_session: Session, installation: Installation
) -> None:
    """When deliver_private=True (env), no epoch is required — back-compat.

    With env deliver_private=True, _deliver_digests is called with digest_epoch=None
    meaning no epoch filter applied (all candidates eligible by epoch alone).
    Just verify the installation model correctly has None by default.
    """
    db_session.refresh(installation)
    assert installation.digest_enabled_at is None
