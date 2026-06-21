"""Integration tests for the visual-revision post-completion hook (HIG-244 slice 2-ii).

Tests call ``AgentTaskExecutor._maybe_revise_documents`` directly against a real
Postgres DB so the JSONB query and session lifecycle are exercised end-to-end.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMUsage,
    ModelPricing,
    ObservationEvent,
    ObserveChannelProfile,
    Schedule,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.documents.critique import VisualCritique
from kortny.documents.ir import DocKind, DocumentSpec, Prose
from kortny.documents.revision import RevisionEvent, RevisionOutcome
from kortny.tasks import TaskService
from kortny.worker import AgentTaskExecutor

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for visual revision hook tests",
)


# ---------------------------------------------------------------------------
# DB fixtures (self-contained; same pattern as test_worker.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(cfg, "head")

    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    factory = make_session_factory(engine=engine)
    with factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup(session: Session) -> None:
    for model in (
        WitnessOpportunityCandidate,
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        Episode,
        ObserveChannelProfile,
        ObservationEvent,
        SlackChannelMembership,
        Artifact,
        LLMUsage,
        TaskEvent,
        SlackSideEffect,
        Task,
        Schedule,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def _make_installation(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(inst)
    session.flush()
    return inst


def _make_task(session: Session, *, event_id: str) -> Task:
    inst = _make_installation(session)
    return TaskService(session).create_task(
        installation_id=inst.id,
        slack_event_id=event_id,
        slack_channel_id="C123",
        slack_thread_ts=event_id,
        slack_message_ts=event_id,
        slack_user_id="U123",
        input=f"task {event_id}",
    )


def _task_events(session: Session, task: Task) -> list[TaskEvent]:
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def _make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": SettingsLLMProvider.openrouter,
        "LLM_API_KEY": "openrouter-key",
        "LLM_MODEL": "openai/gpt-4o-mini",
        "AGENT_RUNTIME": "custom",
        "KORTNY_WORKFLOW_BACKEND": "inline",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        "KORTNY_EMBEDDINGS_BACKEND": "disabled",
        "COMPOSIO_API_KEY": "composio-key",
    }
    base.update(overrides)
    return Settings.model_validate(base)


def _minimal_spec() -> DocumentSpec:
    return DocumentSpec(
        doc_kind=DocKind.report,
        title="Test Report",
        blocks=[Prose(type="prose", text="Hello world.")],
    )


def _add_artifact_with_critique(
    session: Session,
    task: Task,
    *,
    pdf_path: Path,
    score: int,
    doc_version: int = 1,
) -> Artifact:
    """Create an Artifact row and its matching artifact_created TaskEvent."""
    spec = _minimal_spec()
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=len(pdf_path.read_bytes()),
        storage_path=str(pdf_path),
        doc_group_id=uuid.uuid4(),
        doc_version=doc_version,
        spec_json=spec.model_dump(mode="json"),
    )
    session.add(artifact)
    session.flush()

    critique_payload: dict[str, Any] = {
        "overall_score": score,
        "summary": f"Score {score} fixture critique.",
        "issues": [],
    }
    TaskService(session).append_event(
        task,
        TaskEventType.artifact_created,
        {
            "artifact_id": str(artifact.id),
            "filename": artifact.filename,
            "mime_type": artifact.mime_type,
            "size_bytes": artifact.size_bytes,
            "storage_path": str(pdf_path),
            "visual_critique": critique_payload,
            "doc_version": doc_version,
        },
    )
    return artifact


def _make_executor() -> AgentTaskExecutor:
    from kortny.slack.posting import SlackPostingClient

    class _FakeClient:
        def files_upload_v2(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "files": [{"id": "F000001"}]}

    from typing import cast

    return AgentTaskExecutor(slack_client=cast(SlackPostingClient, _FakeClient()))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_visual_revision_disabled_no_op(db_session: Session, tmp_path: Path) -> None:
    """When the flag is off the hook returns immediately without touching the DB."""
    settings = (
        _make_settings()
    )  # KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED defaults False
    task = _make_task(db_session, event_id="EvRevDisabled")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=4)
    db_session.flush()

    before_events = _task_events(db_session, task)

    executor = _make_executor()
    executor._maybe_revise_documents(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        working_dir=tmp_path,
    )

    after_events = _task_events(db_session, task)
    # No new events appended
    assert len(after_events) == len(before_events)

    # No v2 artifact created
    v2 = db_session.scalars(
        select(Artifact).where(
            Artifact.task_id == task.id,
            Artifact.doc_version == 2,
        )
    ).all()
    assert v2 == []


def test_visual_revision_high_score_no_revision(
    db_session: Session, tmp_path: Path
) -> None:
    """An artifact with score >= 7 is skipped — no revision attempted."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvHighScore")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=8)
    db_session.flush()

    before_event_count = len(_task_events(db_session, task))

    # The critic should never be called when score is >= 7, but we need the
    # critic-builder to return a non-None critic so the guard doesn't short-circuit.
    mock_critic = MagicMock(
        return_value=VisualCritique(overall_score=8, summary="", issues=[])
    )

    executor = _make_executor()
    with patch(
        "kortny.worker.agent_executor._build_document_visual_critic",
        return_value=mock_critic,
    ):
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=TaskService(db_session),
            working_dir=tmp_path,
        )

    # No revision events emitted — only the pre-existing task_created event
    after_events = _task_events(db_session, task)
    assert len(after_events) == before_event_count

    v2 = db_session.scalars(
        select(Artifact).where(
            Artifact.task_id == task.id,
            Artifact.doc_version == 2,
        )
    ).all()
    assert v2 == []


def test_visual_revision_idempotency_v2_not_revised(
    db_session: Session, tmp_path: Path
) -> None:
    """A doc_version=2 artifact is never re-revised (idempotency guard)."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvIdempotent")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    _add_artifact_with_critique(
        db_session, task, pdf_path=pdf_file, score=3, doc_version=2
    )
    db_session.flush()

    before_event_count = len(_task_events(db_session, task))

    mock_critic = MagicMock()

    executor = _make_executor()
    with patch(
        "kortny.worker.agent_executor._build_document_visual_critic",
        return_value=mock_critic,
    ):
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=TaskService(db_session),
            working_dir=tmp_path,
        )

    # Critic should not have been called at all; no qualifying artifacts (doc_version != 1)
    mock_critic.assert_not_called()
    after_events = _task_events(db_session, task)
    assert len(after_events) == before_event_count

    # No v3 created
    v3 = db_session.scalars(
        select(Artifact).where(
            Artifact.task_id == task.id,
            Artifact.doc_version == 3,
        )
    ).all()
    assert v3 == []


def test_visual_revision_hook_error_doesnt_propagate(
    db_session: Session, tmp_path: Path
) -> None:
    """An error inside the hook body is swallowed and logged as a TaskEvent."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvHookError")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=3)
    db_session.flush()

    task_svc = TaskService(db_session)
    executor = _make_executor()

    # Cause _build_document_visual_critic to raise inside the try block
    with patch(
        "kortny.worker.agent_executor._build_document_visual_critic",
        side_effect=RuntimeError("critic build exploded"),
    ):
        # Must not raise
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=task_svc,
            working_dir=tmp_path,
        )

    events = _task_events(db_session, task)
    error_events = [
        e for e in events if e.payload.get("message") == "visual_revision_hook_failed"
    ]
    assert len(error_events) == 1
    assert error_events[0].payload["error_type"] == "RuntimeError"
    assert "critic build exploded" in error_events[0].payload["error"]


def test_visual_revision_rejected_no_v2(db_session: Session, tmp_path: Path) -> None:
    """When attempt_visual_revision returns rejected, no v2 artifact is created."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvRejected")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    artifact = _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=3)
    db_session.flush()

    mock_critic = MagicMock(
        return_value=VisualCritique(overall_score=3, summary="bad", issues=[])
    )
    rejected_outcome = RevisionOutcome(
        status="rejected",
        reason="no improvement",
        events=[
            RevisionEvent(
                kind="visual_revision_started",
                detail="Starting visual revision (score=3)",
                old_score=3,
            ),
            RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail="no improvement: old=3 new=3",
                old_score=3,
                new_score=3,
            ),
        ],
    )

    task_svc = TaskService(db_session)
    executor = _make_executor()

    with (
        patch(
            "kortny.worker.agent_executor._build_document_visual_critic",
            return_value=mock_critic,
        ),
        patch(
            "kortny.worker.agent_executor.attempt_visual_revision",
            return_value=rejected_outcome,
        ),
    ):
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=task_svc,
            working_dir=tmp_path,
        )

    v2 = db_session.scalars(
        select(Artifact).where(
            Artifact.task_id == task.id,
            Artifact.doc_version == 2,
        )
    ).all()
    assert v2 == []

    events = _task_events(db_session, task)
    rev_events = [
        e for e in events if e.payload.get("message", "").startswith("visual_revision_")
    ]
    assert len(rev_events) == 2
    assert rev_events[0].payload["message"] == "visual_revision_started"
    assert rev_events[1].payload["message"] == "visual_revision_candidate_rejected"
    # Both events should reference the original artifact
    for ev in rev_events:
        assert ev.payload["artifact_id"] == str(artifact.id)


def test_visual_revision_enabled_low_score_accepted(
    db_session: Session, tmp_path: Path
) -> None:
    """Full accepted path: v2 artifact is created, Slack upload called, events recorded."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvAccepted")

    pdf_file = tmp_path / "v1_report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake-v1")
    artifact = _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=4)
    db_session.flush()

    revised_spec = _minimal_spec()
    revised_pdf_bytes = b"%PDF-1.4 fake-v2"

    mock_critic = MagicMock(
        return_value=VisualCritique(overall_score=4, summary="poor", issues=[])
    )
    accepted_outcome = RevisionOutcome(
        status="accepted",
        revised_spec=revised_spec,
        revised_pdf=revised_pdf_bytes,
        new_critique=VisualCritique(
            overall_score=8,
            summary="improved",
            issues=[],
        ),
        reason="accepted: old=4 new=8",
        events=[
            RevisionEvent(
                kind="visual_revision_started",
                detail="Starting visual revision (score=4)",
                old_score=4,
            ),
            RevisionEvent(
                kind="visual_revision_accepted",
                detail="accepted: old=4 new=8",
                old_score=4,
                new_score=8,
            ),
        ],
    )

    uploads: list[dict[str, Any]] = []

    class _FakeClient:
        def files_upload_v2(self, **kwargs: Any) -> dict[str, Any]:
            uploads.append(kwargs)
            return {"ok": True, "files": [{"id": "F999"}]}

    from typing import cast

    from kortny.slack.posting import SlackPostingClient

    executor = AgentTaskExecutor(slack_client=cast(SlackPostingClient, _FakeClient()))
    task_svc = TaskService(db_session)

    with (
        patch(
            "kortny.worker.agent_executor._build_document_visual_critic",
            return_value=mock_critic,
        ),
        patch(
            "kortny.worker.agent_executor.attempt_visual_revision",
            return_value=accepted_outcome,
        ),
    ):
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=task_svc,
            working_dir=tmp_path,
        )

    # v2 Artifact should exist
    v2_artifacts = db_session.scalars(
        select(Artifact).where(
            Artifact.task_id == task.id,
            Artifact.doc_version == 2,
        )
    ).all()
    assert len(v2_artifacts) == 1
    v2 = v2_artifacts[0]
    assert v2.filename == artifact.filename
    assert v2.mime_type == "application/pdf"
    assert v2.size_bytes == len(revised_pdf_bytes)
    assert v2.doc_group_id == artifact.doc_group_id
    assert v2.spec_json is not None

    # v2 PDF written to disk
    assert v2.storage_path is not None
    v2_path = Path(v2.storage_path)
    assert v2_path.exists()
    assert v2_path.read_bytes() == revised_pdf_bytes

    # Slack upload was called; filename is the v2 on-disk name (contains the artifact id),
    # but the artifact.filename (clean name) is embedded within it.
    assert len(uploads) == 1
    assert artifact.filename in uploads[0]["filename"]

    # Events include revision events + artifact_created for v2
    events = _task_events(db_session, task)
    rev_events = [
        e for e in events if e.payload.get("message", "").startswith("visual_revision_")
    ]
    assert len(rev_events) == 2
    assert rev_events[0].payload["message"] == "visual_revision_started"
    assert rev_events[1].payload["message"] == "visual_revision_accepted"

    artifact_events = [e for e in events if e.type == TaskEventType.artifact_created]
    # One for v1 (seeded by helper), one for v2
    assert len(artifact_events) == 2
    v2_artifact_event = next(
        e for e in artifact_events if e.payload.get("doc_version") == 2
    )
    assert v2_artifact_event.payload["revision_of"] == str(artifact.id)
    assert v2_artifact_event.payload["visual_critique"]["overall_score"] == 8


def test_llm_proposer_wired_calls_llm_propose_fn(
    db_session: Session, tmp_path: Path
) -> None:
    """_build_revision_patch_proposer result is passed as llm_propose_fn to attempt_visual_revision."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvProposerWired")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=4)
    db_session.flush()

    mock_critic = MagicMock(
        return_value=VisualCritique(overall_score=4, summary="poor", issues=[])
    )
    mock_proposer = MagicMock(return_value=None)
    noop_outcome = RevisionOutcome(
        status="noop",
        reason="no actionable deterministic fix",
        events=[
            RevisionEvent(
                kind="visual_revision_noop",
                detail="no actionable deterministic fix",
                old_score=4,
            )
        ],
    )

    captured: dict[str, Any] = {}

    def _capture_attempt_visual_revision(*args: Any, **kwargs: Any) -> RevisionOutcome:
        captured.update(kwargs)
        return noop_outcome

    task_svc = TaskService(db_session)
    executor = _make_executor()

    with (
        patch(
            "kortny.worker.agent_executor._build_document_visual_critic",
            return_value=mock_critic,
        ),
        patch(
            "kortny.worker.agent_executor._build_revision_patch_proposer",
            return_value=mock_proposer,
        ),
        patch(
            "kortny.worker.agent_executor.attempt_visual_revision",
            side_effect=_capture_attempt_visual_revision,
        ),
    ):
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=task_svc,
            working_dir=tmp_path,
        )

    assert "llm_propose_fn" in captured
    assert captured["llm_propose_fn"] is mock_proposer


def test_llm_proposer_none_still_works(db_session: Session, tmp_path: Path) -> None:
    """When _build_revision_patch_proposer returns None, _maybe_revise_documents does not raise."""
    settings = _make_settings(KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED=True)
    task = _make_task(db_session, event_id="EvProposerNone")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    _add_artifact_with_critique(db_session, task, pdf_path=pdf_file, score=4)
    db_session.flush()

    mock_critic = MagicMock(
        return_value=VisualCritique(overall_score=4, summary="poor", issues=[])
    )
    noop_outcome = RevisionOutcome(
        status="noop",
        reason="no actionable deterministic fix",
        events=[
            RevisionEvent(
                kind="visual_revision_noop",
                detail="no actionable deterministic fix",
                old_score=4,
            )
        ],
    )

    captured: dict[str, Any] = {}

    def _capture_attempt_visual_revision(*args: Any, **kwargs: Any) -> RevisionOutcome:
        captured.update(kwargs)
        return noop_outcome

    task_svc = TaskService(db_session)
    executor = _make_executor()

    with (
        patch(
            "kortny.worker.agent_executor._build_document_visual_critic",
            return_value=mock_critic,
        ),
        patch(
            "kortny.worker.agent_executor._build_revision_patch_proposer",
            return_value=None,
        ),
        patch(
            "kortny.worker.agent_executor.attempt_visual_revision",
            side_effect=_capture_attempt_visual_revision,
        ),
    ):
        # Must not raise
        executor._maybe_revise_documents(
            settings=settings,
            session=db_session,
            task=task,
            task_service=task_svc,
            working_dir=tmp_path,
        )

    assert "llm_propose_fn" in captured
    assert captured["llm_propose_fn"] is None
