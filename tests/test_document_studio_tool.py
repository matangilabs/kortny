"""Tests for the Document Studio tool (HIG-244 Phase 1)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pypdf import PdfReader
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.documents import typst_available
from kortny.execution import task_workspace
from kortny.tasks import TaskService
from kortny.tools import DocumentStudioTool, RecoverableToolError

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")
_TYPST_MISSING = not typst_available()

_SPEC_ARGS = {
    "doc_kind": "report",
    "title": "Test Report",
    "blocks": [
        {
            "type": "cover_header",
            "title": "Test Report",
            "subtitle": "A small fixture",
            "meta": ["Kortny"],
        },
        {"type": "heading", "text": "Overview"},
        {"type": "prose", "text": "First.\n\nSecond."},
        {
            "type": "stat_cards",
            "cards": [{"value": "$1B", "label": "Raise"}],
        },
        {
            "type": "table",
            "columns": ["A", "B"],
            "rows": [["1", "2"]],
        },
    ],
}


def _args(**overrides: object) -> dict[str, object]:
    return {**_SPEC_ARGS, **overrides}


# --------------------------------------------------------------------------- #
# Validation / error paths (no binary needed)
# --------------------------------------------------------------------------- #


def test_invalid_spec_raises_recoverable(tmp_path: Path) -> None:
    tool = DocumentStudioTool(working_dir=tmp_path)
    with pytest.raises(RecoverableToolError) as exc:
        tool.invoke({"title": "x", "blocks": [{"type": "not_a_block"}]})
    assert exc.value.code == "invalid_document_spec"
    assert "errors" in exc.value.details


def test_missing_required_field_raises_recoverable(tmp_path: Path) -> None:
    tool = DocumentStudioTool(working_dir=tmp_path)
    with pytest.raises(RecoverableToolError) as exc:
        # cover_header requires a title.
        tool.invoke({"title": "x", "blocks": [{"type": "cover_header"}]})
    assert exc.value.code == "invalid_document_spec"


def test_typst_unavailable_raises_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate the compiler package being absent.
    monkeypatch.setattr("kortny.documents.render._typst", None)
    tool = DocumentStudioTool(working_dir=tmp_path)
    with pytest.raises(RecoverableToolError) as exc:
        tool.invoke(_args())
    assert exc.value.code == "typst_unavailable"
    assert exc.value.hint is not None


def test_chart_render_error_raises_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a chart that fails to render (bad spec / vl-convert failure)
    # must surface as a recoverable tool error the model can correct, not bubble
    # as a raw ChartRenderError that crashes the tool call.
    from kortny.documents import ChartRenderError

    def _boom(*_args: object, **_kwargs: object) -> bytes:
        raise ChartRenderError("chart render failed: bad encoding")

    monkeypatch.setattr("kortny.tools.document_studio.render_spec_pdf", _boom)
    tool = DocumentStudioTool(working_dir=tmp_path)
    with pytest.raises(RecoverableToolError) as exc:
        tool.invoke(_args())
    assert exc.value.code == "chart_render_failed"
    assert exc.value.hint is not None


# --------------------------------------------------------------------------- #
# Rendering (real typst binary)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(_TYPST_MISSING, reason="typst binary not installed")
def test_renders_pdf_and_returns_artifact(tmp_path: Path) -> None:
    result = DocumentStudioTool(working_dir=tmp_path).invoke(
        _args(filename="../My Report")
    )
    output_path = Path(result.output["path"])

    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"%PDF")
    # Filename sanitised + .pdf enforced, kept inside the working dir.
    assert result.output["filename"] == "My_Report.pdf"
    assert output_path.parent == tmp_path
    assert result.output["mime_type"] == "application/pdf"
    assert result.output["doc_kind"] == "report"
    assert result.output["block_count"] == 5
    assert result.output["artifact_id"] is None
    assert len(result.artifacts) == 1
    assert len(PdfReader(str(output_path)).pages) >= 1


def test_renders_pptx_format(tmp_path: Path) -> None:
    result = DocumentStudioTool(working_dir=tmp_path).invoke(
        _args(format="pptx", filename="deck")
    )
    output_path = Path(result.output["path"])
    assert output_path.exists()
    assert output_path.read_bytes()[:2] == b"PK"
    assert result.output["filename"] == "deck.pptx"
    assert result.output["format"] == "pptx"
    assert "presentationml" in result.output["mime_type"]


def test_renders_docx_format(tmp_path: Path) -> None:
    result = DocumentStudioTool(working_dir=tmp_path).invoke(
        _args(format="docx", filename="brief.pdf")  # wrong ext -> corrected
    )
    output_path = Path(result.output["path"])
    assert output_path.exists()
    assert output_path.read_bytes()[:2] == b"PK"
    # User-supplied .pdf extension is stripped and the docx extension enforced.
    assert result.output["filename"] == "brief.docx"
    assert result.output["format"] == "docx"
    assert "wordprocessingml" in result.output["mime_type"]


def test_renders_xlsx_format(tmp_path: Path) -> None:
    result = DocumentStudioTool(working_dir=tmp_path).invoke(
        _args(format="xlsx", filename="data")
    )
    output_path = Path(result.output["path"])
    assert output_path.exists()
    assert output_path.read_bytes()[:2] == b"PK"
    assert result.output["filename"] == "data.xlsx"
    assert result.output["format"] == "xlsx"
    assert "spreadsheetml" in result.output["mime_type"]


def test_invalid_format_raises_recoverable(tmp_path: Path) -> None:
    with pytest.raises(RecoverableToolError) as exc:
        DocumentStudioTool(working_dir=tmp_path).invoke(_args(format="rtf"))
    assert exc.value.code == "invalid_document_format"


def test_critique_autofixes_ragged_table_and_reports(tmp_path: Path) -> None:
    ragged = {"type": "table", "columns": ["A", "B", "C"], "rows": [["1"]]}
    result = DocumentStudioTool(working_dir=tmp_path).invoke(
        _args(blocks=[{"type": "prose", "text": "body"}, ragged])
    )
    assert Path(result.output["path"]).exists()
    critique = result.output["critique"]
    assert critique["autofixes"] >= 1
    assert "ragged_table_rows" in critique["codes"]


def test_empty_document_raises_needs_revision(tmp_path: Path) -> None:
    with pytest.raises(RecoverableToolError) as exc:
        DocumentStudioTool(working_dir=tmp_path).invoke(
            _args(blocks=[{"type": "prose", "text": "   "}])
        )
    assert exc.value.code == "document_spec_needs_revision"


# --------------------------------------------------------------------------- #
# DB-backed artifact recording
# --------------------------------------------------------------------------- #


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


@pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for DB-backed tests",
)
@pytest.mark.skipif(_TYPST_MISSING, reason="typst binary not installed")
def test_records_artifact_row_and_event(db_session: Session, tmp_path: Path) -> None:
    task = _create_task(db_session)
    task_service = TaskService(db_session)

    with task_workspace(task.id, base_dir=tmp_path) as workspace:
        result = DocumentStudioTool(
            working_dir=workspace.path,
            session=db_session,
            task_id=task.id,
            task_service=task_service,
        ).invoke(_args())

        artifact = db_session.scalar(
            select(Artifact).where(Artifact.task_id == task.id)
        )
        event = db_session.scalar(
            select(TaskEvent).where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == TaskEventType.artifact_created,
            )
        )

    assert artifact is not None
    assert artifact.mime_type == "application/pdf"
    assert result.output["artifact_id"] == str(artifact.id)
    assert event is not None
    assert event.payload["artifact_id"] == str(artifact.id)


def _cleanup(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def _create_task(session: Session) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts="1716400000.000001",
        slack_user_id="U123",
        input="make a document",
    )
