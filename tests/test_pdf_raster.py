"""Tests for the shared PDF rasterisation utility (HIG-244)."""

from __future__ import annotations

import io
from pathlib import Path

from kortny.pdf_raster import rasterize_pdf_pages


def _make_blank_pdf_bytes(num_pages: int = 1) -> bytes:
    import pypdfium2

    doc = pypdfium2.PdfDocument.new()
    for _ in range(num_pages):
        doc.new_page(width=595, height=842)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.read()


def test_rasterize_returns_png_bytes(tmp_path: Path) -> None:
    pdf_bytes = _make_blank_pdf_bytes(2)
    path = tmp_path / "test.pdf"
    path.write_bytes(pdf_bytes)

    pages = rasterize_pdf_pages(path, max_pages=10)
    assert len(pages) == 2
    for png in pages:
        assert isinstance(png, bytes)
        assert len(png) > 0
        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_rasterize_respects_max_pages(tmp_path: Path) -> None:
    pdf_bytes = _make_blank_pdf_bytes(5)
    path = tmp_path / "test.pdf"
    path.write_bytes(pdf_bytes)

    pages = rasterize_pdf_pages(path, max_pages=2)
    assert len(pages) == 2


def test_rasterize_single_page(tmp_path: Path) -> None:
    pdf_bytes = _make_blank_pdf_bytes(1)
    path = tmp_path / "test.pdf"
    path.write_bytes(pdf_bytes)

    pages = rasterize_pdf_pages(path, max_pages=20)
    assert len(pages) == 1
    assert pages[0][:8] == b"\x89PNG\r\n\x1a\n"
