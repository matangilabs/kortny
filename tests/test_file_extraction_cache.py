"""Tests for content-addressed file extraction cache (HIG-279 slice 3b-1)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.db.models import FileExtractionCache
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tools.file_extraction_cache import FileExtractionCacheRepository
from kortny.tools.slack_file_read import SlackFileReadTool, TextExtraction

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for file extraction cache tests",
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
    session.execute(delete(FileExtractionCache))


# ---------------------------------------------------------------------------
# Repository unit tests (DB-backed)
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(db_session: Session) -> None:
    repo = FileExtractionCacheRepository(db_session)
    result = repo.get("a" * 64)
    assert result is None


def test_put_then_get_round_trip(db_session: Session) -> None:
    content = b"hello world"
    sha = hashlib.sha256(content).hexdigest()
    extraction = TextExtraction(
        supported=True,
        text="hello world",
        truncated=False,
        backend="text",
        warnings=(),
    )

    repo = FileExtractionCacheRepository(db_session)
    repo.put(sha, extraction, byte_size=len(content))
    db_session.flush()

    cached = repo.get(sha)
    assert cached is not None
    assert cached.supported is True
    assert cached.text == "hello world"
    assert cached.truncated is False
    assert cached.backend == "text"
    assert cached.warnings == ()


def test_put_then_get_preserves_warnings(db_session: Session) -> None:
    content = b"docx bytes"
    sha = hashlib.sha256(content).hexdigest()
    extraction = TextExtraction(
        supported=False,
        text=None,
        truncated=False,
        backend="docx",
        warnings=("docx_parse_error",),
    )

    repo = FileExtractionCacheRepository(db_session)
    repo.put(sha, extraction, byte_size=len(content))
    db_session.flush()

    cached = repo.get(sha)
    assert cached is not None
    assert cached.supported is False
    assert cached.warnings == ("docx_parse_error",)


def test_different_content_yields_separate_rows(db_session: Session) -> None:
    content_a = b"file content A"
    content_b = b"file content B"
    sha_a = hashlib.sha256(content_a).hexdigest()
    sha_b = hashlib.sha256(content_b).hexdigest()

    extraction_a = TextExtraction(supported=True, text="A", backend="text")
    extraction_b = TextExtraction(supported=True, text="B", backend="text")

    repo = FileExtractionCacheRepository(db_session)
    repo.put(sha_a, extraction_a, byte_size=len(content_a))
    repo.put(sha_b, extraction_b, byte_size=len(content_b))
    db_session.flush()

    cached_a = repo.get(sha_a)
    cached_b = repo.get(sha_b)
    assert cached_a is not None and cached_a.text == "A"
    assert cached_b is not None and cached_b.text == "B"


def test_put_twice_same_hash_is_idempotent(db_session: Session) -> None:
    content = b"idempotent content"
    sha = hashlib.sha256(content).hexdigest()
    extraction = TextExtraction(supported=True, text="idempotent", backend="text")

    repo = FileExtractionCacheRepository(db_session)
    # First write
    repo.put(sha, extraction, byte_size=len(content))
    db_session.flush()
    # Second write — ON CONFLICT DO NOTHING should not raise
    repo.put(sha, extraction, byte_size=len(content))
    db_session.flush()

    cached = repo.get(sha)
    assert cached is not None
    assert cached.text == "idempotent"


# ---------------------------------------------------------------------------
# Integration: SlackFileReadTool with session (cache miss then hit)
# ---------------------------------------------------------------------------


class _FakeSlackFilesClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[str] = []

    def files_info(self, *, file: str) -> dict[str, Any]:
        self.calls.append(file)
        return self.response


def _download_transport(content: bytes, content_type: str) -> Any:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={
                "content-type": content_type,
                "content-length": str(len(content)),
            },
            request=request,
        )

    return httpx.MockTransport(handler)


def _file_info(
    *,
    file_id: str,
    name: str,
    mimetype: str,
    size: int,
    url: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "file": {
            "id": file_id,
            "name": name,
            "mimetype": mimetype,
            "size": size,
            "url_private_download": url,
        },
    }


def test_tool_cache_miss_then_hit(db_session: Session, tmp_path: Path) -> None:
    text_bytes = b"cached plain text content"
    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FCACHE1",
            name="notes.txt",
            mimetype="text/plain",
            size=len(text_bytes),
            url="https://files.slack.com/files-pri/T1-FCACHE1/notes.txt",
        )
    )

    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(text_bytes, "text/plain"),
        session=db_session,
    )

    # First call: cache miss — extraction runs and is stored
    result1 = tool.invoke({"file_id": "FCACHE1"})
    db_session.flush()
    assert result1.output["extraction_cache"] == "miss"
    assert result1.output["extracted_text"] == text_bytes.decode()

    # Second call: cache hit — no re-extraction needed
    result2 = tool.invoke({"file_id": "FCACHE1"})
    assert result2.output["extraction_cache"] == "hit"
    assert result2.output["extracted_text"] == text_bytes.decode()


def test_tool_without_session_omits_cache_key(tmp_path: Path) -> None:
    text_bytes = b"no cache session"
    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FNOCACHE",
            name="data.txt",
            mimetype="text/plain",
            size=len(text_bytes),
            url="https://files.slack.com/files-pri/T1-FNOCACHE/data.txt",
        )
    )

    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(text_bytes, "text/plain"),
        # no session= kwarg
    )

    result = tool.invoke({"file_id": "FNOCACHE"})
    # When no session is provided, extraction_cache key must not appear
    assert "extraction_cache" not in result.output
    assert result.output["extracted_text"] == text_bytes.decode()
