"""Tests for vision-OCR path in slack_file_read (HIG-279 slice 3b-2).

Covers:
- Scanned PDF (empty text layer) + fake pdf_ocr → OCR text returned, cached
- Second invoke on same content → cache HIT, pdf_ocr NOT called again
- Scanned PDF + pdf_ocr=None → recoverable warning, no crash
- Native text-layer PDF → OCR not attempted, pdf_ocr not called
- Page cap: pages beyond limit → truncation warning
- pypdfium2 rasterisation unit test: real small PDF → ≥1 PNG bytes

Tests that exercise the cache (FileExtractionCache) require Postgres and are
skipped when KORTNY_TEST_POSTGRES_URL is unset.

PDF creation in tests uses pypdfium2 directly (not reportlab) because some
pdfium builds cannot parse older/variant PDF formats produced by other generators.
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import kortny.tools.slack_file_read as sfr
from kortny.tools.slack_file_read import (
    SCANNED_PDF_TEXT_THRESHOLD,
    SlackFileReadTool,
    TextExtraction,
    _is_scanned_pdf,
    _rasterize_pdf_pages,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSlackFilesClient:
    """Minimal Slack WebClient stub for file download tests."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def files_info(self, *, file: str) -> dict[str, Any]:
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


def _make_blank_pdf_bytes(num_pages: int = 1) -> bytes:
    """Create a valid blank PDF using pypdfium2 (pdfium-compatible)."""
    import pypdfium2

    doc = pypdfium2.PdfDocument.new()
    for _ in range(num_pages):
        doc.new_page(width=595, height=842)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_pdf_bytes_with_text_layer(text: str) -> bytes:
    """Create a PDF with embedded text using pypdfium2.

    We embed the text using a font object on the page so pypdf can extract it.
    Falls back to using reportlab if available (more reliable text extraction).
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas

        # reportlab PDFs may not open with all pdfium builds, but that's fine
        # here — we only need pypdf to extract the text layer from them.
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=letter)
        c.drawString(72, 720, text)
        c.save()
        return buf.getvalue()
    except Exception:
        # If reportlab fails too, return a blank pypdfium2 PDF;
        # the test that calls this should then check OCR is NOT triggered
        # only if extraction returns real text — otherwise skip gracefully.
        return _make_blank_pdf_bytes()


# ---------------------------------------------------------------------------
# Unit: _is_scanned_pdf helper
# ---------------------------------------------------------------------------


def test_is_scanned_pdf_empty_text() -> None:
    extraction = TextExtraction(supported=True, text="", backend="pdf_textlayer")
    assert _is_scanned_pdf(extraction) is True


def test_is_scanned_pdf_whitespace_only() -> None:
    extraction = TextExtraction(
        supported=True, text="   \n\t ", backend="pdf_textlayer"
    )
    assert _is_scanned_pdf(extraction) is True


def test_is_scanned_pdf_below_threshold() -> None:
    # Just under the threshold
    short = "x" * (SCANNED_PDF_TEXT_THRESHOLD - 1)
    extraction = TextExtraction(supported=True, text=short, backend="pdf_textlayer")
    assert _is_scanned_pdf(extraction) is True


def test_is_scanned_pdf_at_threshold() -> None:
    # Exactly at the threshold — not scanned
    text = "x" * SCANNED_PDF_TEXT_THRESHOLD
    extraction = TextExtraction(supported=True, text=text, backend="pdf_textlayer")
    assert _is_scanned_pdf(extraction) is False


def test_is_scanned_pdf_none_text() -> None:
    extraction = TextExtraction(supported=False, text=None, backend="pdf_textlayer")
    assert _is_scanned_pdf(extraction) is True


# ---------------------------------------------------------------------------
# Unit: _rasterize_pdf_pages (requires pypdfium2)
# ---------------------------------------------------------------------------


def test_rasterize_pdf_produces_png_bytes(tmp_path: Path) -> None:
    """A real minimal PDF (pypdfium2-created) rasterises to ≥1 non-empty PNG bytes."""
    pdf_bytes = _make_blank_pdf_bytes(num_pages=1)
    pdf_path = tmp_path / "blank.pdf"
    pdf_path.write_bytes(pdf_bytes)

    pages = _rasterize_pdf_pages(pdf_path, max_pages=5)
    assert len(pages) >= 1
    for png_bytes in pages:
        # PNG magic bytes: \x89PNG
        assert png_bytes[:4] == b"\x89PNG", "Expected PNG output"
        assert len(png_bytes) > 100, "PNG is suspiciously small"


def test_rasterize_pdf_respects_max_pages(tmp_path: Path) -> None:
    """With max_pages=1, only 1 page is returned even if PDF has 3 pages."""
    pdf_bytes = _make_blank_pdf_bytes(num_pages=3)
    pdf_path = tmp_path / "three_pages.pdf"
    pdf_path.write_bytes(pdf_bytes)

    pages = _rasterize_pdf_pages(pdf_path, max_pages=1)
    assert len(pages) == 1


# ---------------------------------------------------------------------------
# Integration: scanned PDF OCR (monkeypatched pdf_ocr, no real LLM)
# ---------------------------------------------------------------------------


def test_scanned_pdf_with_ocr_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scanned PDF + fake pdf_ocr → supported TextExtraction with OCR text."""
    pdf_bytes = _make_blank_pdf_bytes()
    ocr_call_count = 0

    def fake_ocr(pages: Sequence[bytes]) -> str:
        nonlocal ocr_call_count
        ocr_call_count += 1
        return "OCR TEXT"

    # Patch both _extract_pdf_text (returns empty) and _rasterize_pdf_pages
    # (returns 1 fake PNG) to decouple the test from the real PDF pipeline.
    monkeypatch.setattr(sfr, "_extract_pdf_text", lambda path: "")
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    monkeypatch.setattr(sfr, "_rasterize_pdf_pages", lambda path, max_pages: [fake_png])

    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FOCR001",
            name="scan.pdf",
            mimetype="application/pdf",
            size=len(pdf_bytes),
            url="https://files.slack.com/files-pri/T1-FOCR001/scan.pdf",
        )
    )
    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(pdf_bytes, "application/pdf"),
        pdf_ocr=fake_ocr,
        pdf_ocr_max_pages=20,
        # No session: cache disabled, simplest path
    )

    result = tool.invoke({"file_id": "FOCR001"})
    assert result.output["extraction_supported"] is True
    assert result.output["backend"] == "vision_ocr"
    assert result.output["extracted_text"] == "OCR TEXT"
    assert ocr_call_count == 1


def test_scanned_pdf_no_ocr_callable_returns_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scanned PDF + pdf_ocr=None → recoverable unsupported result, no crash."""
    pdf_bytes = _make_blank_pdf_bytes()
    monkeypatch.setattr(sfr, "_extract_pdf_text", lambda path: "")

    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FOCR002",
            name="scan.pdf",
            mimetype="application/pdf",
            size=len(pdf_bytes),
            url="https://files.slack.com/files-pri/T1-FOCR002/scan.pdf",
        )
    )
    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(pdf_bytes, "application/pdf"),
        pdf_ocr=None,  # No OCR callable
    )

    result = tool.invoke({"file_id": "FOCR002"})
    assert result.output["extraction_supported"] is False
    assert "scanned_pdf_needs_vision_model" in (result.output.get("warnings") or [])


def test_native_pdf_skips_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF that returns real text → pdf_ocr callable is NOT called."""
    pdf_bytes = _make_blank_pdf_bytes()
    # Patch _extract_pdf_text to return real text (simulates native text layer)
    real_text = "This is real extracted text with enough characters."
    monkeypatch.setattr(sfr, "_extract_pdf_text", lambda path: real_text)

    ocr_call_count = 0

    def fake_ocr(pages: Sequence[bytes]) -> str:
        nonlocal ocr_call_count
        ocr_call_count += 1
        return "SHOULD NOT BE CALLED"

    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FOCR003",
            name="native.pdf",
            mimetype="application/pdf",
            size=len(pdf_bytes),
            url="https://files.slack.com/files-pri/T1-FOCR003/native.pdf",
        )
    )
    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(pdf_bytes, "application/pdf"),
        pdf_ocr=fake_ocr,
    )

    result = tool.invoke({"file_id": "FOCR003"})
    # OCR should NOT have been called — text layer was non-empty
    assert ocr_call_count == 0
    assert result.output["extraction_supported"] is True
    assert result.output["backend"] == "pdf_textlayer"


def test_scanned_pdf_page_cap_adds_truncation_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-page scanned PDF beyond cap → only cap pages OCR'd + truncation warning."""
    # 3-page PDF but cap is 2
    pdf_bytes = _make_blank_pdf_bytes(num_pages=3)
    monkeypatch.setattr(sfr, "_extract_pdf_text", lambda path: "")

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    def fake_rasterize(path: Path, max_pages: int) -> list[bytes]:
        # Simulate returning only max_pages pages
        return [fake_png] * max_pages

    monkeypatch.setattr(sfr, "_rasterize_pdf_pages", fake_rasterize)

    page_counts_seen: list[int] = []

    def counting_ocr(pages: Sequence[bytes]) -> str:
        page_counts_seen.append(len(pages))
        return "OCR"

    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FOCR004",
            name="long_scan.pdf",
            mimetype="application/pdf",
            size=len(pdf_bytes),
            url="https://files.slack.com/files-pri/T1-FOCR004/long_scan.pdf",
        )
    )
    # Cap at 2 pages; PDF has 3
    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(pdf_bytes, "application/pdf"),
        pdf_ocr=counting_ocr,
        pdf_ocr_max_pages=2,
    )

    result = tool.invoke({"file_id": "FOCR004"})
    assert result.output["extraction_supported"] is True
    assert result.output["backend"] == "vision_ocr"
    # Only 2 pages should have been sent to OCR
    assert sum(page_counts_seen) == 2
    # Truncation warning should appear
    warnings = result.output.get("warnings") or []
    assert any("pdf_ocr_truncated_at_2_pages" in w for w in warnings)


# ---------------------------------------------------------------------------
# Integration with cache (requires Postgres)
# ---------------------------------------------------------------------------

pytestmark_db = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for cache integration tests",
)


@pytest.fixture
def db_session() -> Any:
    """Real Postgres session for cache integration tests."""
    from sqlalchemy import delete

    from kortny.db.models import FileExtractionCache
    from kortny.db.session import make_engine, make_session_factory

    assert TEST_POSTGRES_URL is not None
    engine = make_engine(TEST_POSTGRES_URL)
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        session.execute(delete(FileExtractionCache))
        session.commit()
        yield session
        session.rollback()
        session.execute(delete(FileExtractionCache))
        session.commit()
    engine.dispose()


@pytestmark_db
def test_ocr_result_is_cached_and_not_reocrd(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First invoke OCRs the scanned PDF; second invoke is a cache HIT (OCR not called)."""
    pdf_bytes = _make_blank_pdf_bytes()
    monkeypatch.setattr(sfr, "_extract_pdf_text", lambda path: "")
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    monkeypatch.setattr(sfr, "_rasterize_pdf_pages", lambda path, max_pages: [fake_png])

    ocr_call_count = 0

    def fake_ocr(pages: Sequence[bytes]) -> str:
        nonlocal ocr_call_count
        ocr_call_count += 1
        return "OCR TEXT"

    slack_client = _FakeSlackFilesClient(
        _file_info(
            file_id="FOCRCACHE",
            name="cached_scan.pdf",
            mimetype="application/pdf",
            size=len(pdf_bytes),
            url="https://files.slack.com/files-pri/T1-FOCRCACHE/cached_scan.pdf",
        )
    )
    tool = SlackFileReadTool(
        client=slack_client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=_download_transport(pdf_bytes, "application/pdf"),
        session=db_session,
        pdf_ocr=fake_ocr,
        pdf_ocr_max_pages=20,
    )

    # First call — cache MISS, OCR runs
    result1 = tool.invoke({"file_id": "FOCRCACHE"})
    db_session.flush()
    assert result1.output["extraction_cache"] == "miss"
    assert result1.output["backend"] == "vision_ocr"
    assert result1.output["extracted_text"] == "OCR TEXT"
    assert ocr_call_count == 1

    # Second call — cache HIT, OCR not called again
    result2 = tool.invoke({"file_id": "FOCRCACHE"})
    assert result2.output["extraction_cache"] == "hit"
    assert result2.output["backend"] == "vision_ocr"
    assert result2.output["extracted_text"] == "OCR TEXT"
    assert ocr_call_count == 1  # still 1 — not called again
