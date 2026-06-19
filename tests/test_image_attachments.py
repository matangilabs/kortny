"""HIG-279 slice 2A: Slack image attachment resolution and context injection.

Tests are deliberately split into pure-unit layers:

1. ``_slack_image_file_pairs`` parsing — no I/O, no DB.
2. ``SlackImageAttachmentResolver`` — fake Slack client + httpx MockTransport.
3. ``ContextAssembler._build_user_message`` — fake resolver injected, no DB.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.agent.context import ContextAssembler, _slack_image_file_pairs
from kortny.agent.image_attachments import SlackImageAttachmentResolver
from kortny.db.models import Installation, Task, TaskEvent
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm.types import ImagePart
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

CHANNEL = "C_IMG_TEST"
USER = "U_IMG_TEST"

# ---------------------------------------------------------------------------
# Pure-unit: <slack_files> image pair parser
# ---------------------------------------------------------------------------


def _block(entries: str) -> str:
    """Wrap entries in a task.input with a slack_files block."""
    return f"check this out\n\n<slack_files>\n{entries}\n</slack_files>"


def test_parse_image_pair_single_image() -> None:
    text = _block(
        "- id: F111\n  name: photo.png\n  mimetype: image/png\n  size_bytes: 1234"
    )
    pairs = _slack_image_file_pairs(text)
    assert pairs == [("F111", "image/png")]


def test_parse_image_pair_multiple_images() -> None:
    text = _block(
        "- id: FA1\n  mimetype: image/jpeg\n- id: FB2\n  mimetype: image/webp\n"
    )
    pairs = _slack_image_file_pairs(text)
    assert pairs == [("FA1", "image/jpeg"), ("FB2", "image/webp")]


def test_parse_image_pair_skips_non_image_mime() -> None:
    text = _block(
        "- id: FDOC\n  mimetype: application/pdf\n- id: FIMG\n  mimetype: image/png\n"
    )
    pairs = _slack_image_file_pairs(text)
    assert pairs == [("FIMG", "image/png")]


def test_parse_image_pair_skips_entry_without_mimetype() -> None:
    text = _block("- id: FNOMIME\n  name: mystery.bin\n")
    pairs = _slack_image_file_pairs(text)
    assert pairs == []


def test_parse_image_pair_empty_block() -> None:
    assert _slack_image_file_pairs("just plain text") == []


def test_parse_image_pair_no_slack_files_block() -> None:
    assert _slack_image_file_pairs("hello world") == []


def test_parse_image_pair_mixed_entries_preserves_order() -> None:
    text = _block(
        "- id: F1\n  mimetype: image/png\n"
        "- id: F2\n  mimetype: text/plain\n"
        "- id: F3\n  mimetype: image/jpeg\n"
    )
    pairs = _slack_image_file_pairs(text)
    assert pairs == [("F1", "image/png"), ("F3", "image/jpeg")]


# ---------------------------------------------------------------------------
# Pure-unit: SlackImageAttachmentResolver
# ---------------------------------------------------------------------------

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20  # fake PNG header


def _fake_files_info(
    *,
    file_id: str,
    mime: str,
    download_url: str,
    size: int = 100,
) -> dict[str, Any]:
    return {
        "ok": True,
        "file": {
            "id": file_id,
            "mimetype": mime,
            "size": size,
            "url_private_download": download_url,
        },
    }


class _FakeSlackClient:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses  # file_id → files_info response
        self.calls: list[str] = []

    def files_info(self, *, file: str) -> dict[str, Any]:
        self.calls.append(file)
        if file not in self._responses:
            return {"ok": False, "error": "file_not_found"}
        return self._responses[file]


def _download_transport(
    content: bytes,
    content_type: str = "image/png",
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" in request.headers
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": content_type},
            request=request,
        )

    return httpx.MockTransport(handler)


def _make_resolver(
    *,
    client: _FakeSlackClient,
    transport: httpx.BaseTransport,
    max_image_bytes: int = 10 * 1024 * 1024,
    allowed_mimes: frozenset[str] = frozenset(
        {"image/png", "image/jpeg", "image/webp"}
    ),
) -> SlackImageAttachmentResolver:
    return SlackImageAttachmentResolver(
        client=client,
        bot_token="xoxb-test",
        max_image_bytes=max_image_bytes,
        allowed_mimes=allowed_mimes,
        transport=transport,
    )


def test_resolver_returns_image_part_for_allowed_mime() -> None:
    img_bytes = _PNG_BYTES
    client = _FakeSlackClient(
        {
            "F001": _fake_files_info(
                file_id="F001",
                mime="image/png",
                download_url="https://files.slack.com/F001/img.png",
            )
        }
    )
    resolver = _make_resolver(
        client=client,
        transport=_download_transport(img_bytes),
    )
    parts = resolver([("F001", "image/png")])
    assert len(parts) == 1
    assert parts[0].mime == "image/png"
    assert parts[0].source == "slack_file:F001"
    assert parts[0].data == img_bytes


def test_resolver_skips_disallowed_mime() -> None:
    client = _FakeSlackClient({})  # no files_info calls expected
    resolver = _make_resolver(
        client=client,
        transport=_download_transport(b""),
        allowed_mimes=frozenset({"image/png"}),
    )
    parts = resolver([("F999", "image/bmp")])  # bmp not in allowed set
    assert parts == ()
    assert client.calls == []  # never hit Slack API


def test_resolver_skips_oversized_file() -> None:
    huge_bytes = b"X" * 200
    client = _FakeSlackClient(
        {
            "F002": _fake_files_info(
                file_id="F002",
                mime="image/jpeg",
                download_url="https://files.slack.com/F002/big.jpg",
            )
        }
    )
    resolver = _make_resolver(
        client=client,
        transport=_download_transport(huge_bytes, "image/jpeg"),
        max_image_bytes=100,  # less than 200 bytes
    )
    parts = resolver([("F002", "image/jpeg")])
    assert parts == ()


def test_resolver_skips_on_download_error_no_raise() -> None:
    def _bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    client = _FakeSlackClient(
        {
            "F003": _fake_files_info(
                file_id="F003",
                mime="image/png",
                download_url="https://files.slack.com/F003/secret.png",
            )
        }
    )
    resolver = _make_resolver(
        client=client,
        transport=httpx.MockTransport(_bad_handler),
    )
    # Must NOT raise — returns empty tuple
    parts = resolver([("F003", "image/png")])
    assert parts == ()


def test_resolver_skips_on_files_info_error_no_raise() -> None:
    client = _FakeSlackClient({"F004": {"ok": False, "error": "file_not_found"}})
    resolver = _make_resolver(
        client=client,
        transport=_download_transport(b""),
    )
    parts = resolver([("F004", "image/png")])
    assert parts == ()


def test_resolver_continues_after_per_file_failure() -> None:
    """Failure on the first file must not abort subsequent files."""
    good_bytes = _PNG_BYTES

    def _selective_handler(request: httpx.Request) -> httpx.Response:
        if "F_BAD" in str(request.url):
            return httpx.Response(500, request=request)
        return httpx.Response(
            200,
            content=good_bytes,
            headers={"content-type": "image/png"},
            request=request,
        )

    client = _FakeSlackClient(
        {
            "F_BAD": _fake_files_info(
                file_id="F_BAD",
                mime="image/png",
                download_url="https://files.slack.com/F_BAD/bad.png",
            ),
            "F_OK": _fake_files_info(
                file_id="F_OK",
                mime="image/png",
                download_url="https://files.slack.com/F_OK/good.png",
            ),
        }
    )
    resolver = _make_resolver(
        client=client,
        transport=httpx.MockTransport(_selective_handler),
    )
    parts = resolver([("F_BAD", "image/png"), ("F_OK", "image/png")])
    assert len(parts) == 1
    assert parts[0].source == "slack_file:F_OK"


def test_resolver_returns_multiple_parts() -> None:
    png_a = b"PNG_A" * 10
    png_b = b"PNG_B" * 10
    url_a = "https://files.slack.com/FA/a.png"
    url_b = "https://files.slack.com/FB/b.png"

    def _handler(request: httpx.Request) -> httpx.Response:
        content = png_a if "FA" in str(request.url) else png_b
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "image/png"},
            request=request,
        )

    client = _FakeSlackClient(
        {
            "FA": _fake_files_info(file_id="FA", mime="image/png", download_url=url_a),
            "FB": _fake_files_info(file_id="FB", mime="image/png", download_url=url_b),
        }
    )
    resolver = _make_resolver(
        client=client,
        transport=httpx.MockTransport(_handler),
    )
    parts = resolver([("FA", "image/png"), ("FB", "image/png")])
    assert len(parts) == 2
    assert {p.source for p in parts} == {"slack_file:FA", "slack_file:FB"}


# ---------------------------------------------------------------------------
# ContextAssembler._build_user_message (no DB needed — uses a fake session)
# ---------------------------------------------------------------------------


def _fake_image_resolver(
    parts: tuple[ImagePart, ...],
) -> SlackImageAttachmentResolver:
    """Return a callable that always produces ``parts`` regardless of input."""

    def _resolver(
        pairs: Any,
    ) -> tuple[ImagePart, ...]:
        return parts

    return _resolver  # type: ignore[return-value]


def _make_assembler_no_db(
    *,
    image_resolver: Any = None,
) -> ContextAssembler:
    """Build a ContextAssembler with a mock session (no real DB ops)."""
    mock_session = MagicMock(spec=Session)
    return ContextAssembler(
        session=mock_session,
        image_resolver=image_resolver,
    )


def test_user_message_no_resolver_text_only() -> None:
    assembler = _make_assembler_no_db()
    msg = assembler._build_user_message("hello world")
    assert msg.role == "user"
    assert msg.content == "hello world"
    assert msg.images == ()


def test_user_message_resolver_none_text_with_slack_files_block() -> None:
    """Without a resolver, even a message with a <slack_files> block is unchanged."""
    text = _block("- id: FXYZ\n  mimetype: image/png\n")
    assembler = _make_assembler_no_db(image_resolver=None)
    msg = assembler._build_user_message(text)
    assert msg.images == ()
    assert msg.content == text


def test_user_message_resolver_injects_images() -> None:
    part = ImagePart(data=b"\x89PNG", mime="image/png", source="slack_file:F_TEST")
    text = _block("- id: F_TEST\n  mimetype: image/png\n")
    assembler = _make_assembler_no_db(image_resolver=_fake_image_resolver((part,)))
    msg = assembler._build_user_message(text)
    assert len(msg.images) == 1
    assert msg.images[0].source == "slack_file:F_TEST"
    assert msg.content == text  # text block left intact (ADDITIVE)


def test_user_message_resolver_text_only_input_unchanged() -> None:
    """If input has no <slack_files> block, resolver is not called and images=()."""

    called = []

    def _tracking_resolver(pairs: Any) -> tuple[ImagePart, ...]:
        called.append(pairs)
        return ()

    assembler = _make_assembler_no_db(image_resolver=_tracking_resolver)
    msg = assembler._build_user_message("plain text, no files block")
    assert msg.images == ()
    assert called == []  # resolver must NOT be called for text-only input


def test_user_message_resolver_non_image_slack_files_unchanged() -> None:
    """A <slack_files> block with only non-image entries → images=()."""
    # The resolver gets an empty pairs list → returns ()
    assembler = _make_assembler_no_db(image_resolver=_fake_image_resolver(()))
    text = _block("- id: FDOC\n  mimetype: application/pdf\n")
    msg = assembler._build_user_message(text)
    assert msg.images == ()
    assert msg.content == text


def test_user_message_resolver_exception_continues_text_only() -> None:
    """If the resolver raises unexpectedly, the task continues text-only."""

    def _exploding_resolver(pairs: Any) -> tuple[ImagePart, ...]:
        raise RuntimeError("network timeout")

    text = _block("- id: FCHAOS\n  mimetype: image/png\n")
    assembler = _make_assembler_no_db(image_resolver=_exploding_resolver)
    msg = assembler._build_user_message(text)  # must NOT raise
    assert msg.images == ()
    assert msg.content == text


# ---------------------------------------------------------------------------
# DB-backed: ContextAssembler.build_for_task with injected image resolver
# ---------------------------------------------------------------------------

pytestmark_db = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for DB-backed image context tests",
)


@pytest.fixture(scope="session")
def img_engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(cfg, "heads")
    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def img_db_session(img_engine: Engine) -> Iterator[Session]:
    factory = make_session_factory(engine=img_engine)
    with factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (TaskEvent, Task, Installation):
        session.execute(delete(model))


@pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for DB-backed image context tests",
)
def test_build_for_task_injects_images_from_resolver(
    img_db_session: Session,
) -> None:
    """User ChatMessage gets ImagePart when resolver + image entry present."""
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    img_db_session.add(installation)
    img_db_session.flush()

    image_text = (
        "check this screenshot\n\n"
        "<slack_files>\n"
        "- id: FSCREEN\n"
        "  name: screen.png\n"
        "  mimetype: image/png\n"
        "  size_bytes: 28\n"
        "</slack_files>"
    )
    task = TaskService(img_db_session).create_task(
        installation_id=installation.id,
        slack_channel_id=CHANNEL,
        slack_user_id=USER,
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        input=image_text,
        identity=TaskIdentity.manual(
            channel_id=CHANNEL,
            thread_ts=None,
            user_id=USER,
            input_text=image_text + uuid.uuid4().hex,
        ),
    )

    expected_part = ImagePart(
        data=b"\x89PNG\r\n\x1a\n",
        mime="image/png",
        source="slack_file:FSCREEN",
    )

    def _fake_resolver(
        pairs: Any,
    ) -> tuple[ImagePart, ...]:
        assert pairs == [("FSCREEN", "image/png")]
        return (expected_part,)

    assembler = ContextAssembler(
        session=img_db_session,
        image_resolver=_fake_resolver,
    )
    package = assembler.build_for_task(task)

    user_messages = [m for m in package.messages if m.role == "user"]
    assert len(user_messages) == 1
    user_msg = user_messages[0]
    assert len(user_msg.images) == 1
    assert user_msg.images[0].source == "slack_file:FSCREEN"
    assert user_msg.images[0].mime == "image/png"
    # Original text is ADDITIVE — must not be stripped
    assert user_msg.content == image_text


@pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for DB-backed image context tests",
)
def test_build_for_task_text_only_unchanged(
    img_db_session: Session,
) -> None:
    """A text-only task must produce an unchanged user message (images=())."""
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    img_db_session.add(installation)
    img_db_session.flush()

    plain_input = "what is the weather like today?"
    task = TaskService(img_db_session).create_task(
        installation_id=installation.id,
        slack_channel_id=CHANNEL,
        slack_user_id=USER,
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        input=plain_input,
        identity=TaskIdentity.manual(
            channel_id=CHANNEL,
            thread_ts=None,
            user_id=USER,
            input_text=plain_input + uuid.uuid4().hex,
        ),
    )

    called: list[Any] = []

    def _tracking_resolver(pairs: Any) -> tuple[ImagePart, ...]:
        called.append(pairs)
        return ()

    assembler = ContextAssembler(
        session=img_db_session,
        image_resolver=_tracking_resolver,
    )
    package = assembler.build_for_task(task)

    user_messages = [m for m in package.messages if m.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].images == ()
    assert user_messages[0].content == plain_input
    assert called == []  # resolver never called for text-only
