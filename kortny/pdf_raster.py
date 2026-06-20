"""Shared PDF rasterisation utility (HIG-244).

A leaf module with no kortny dependencies so it can be imported by both
``kortny.tools.slack_file_read`` and ``kortny.documents.critique`` without
creating import cycles.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

# pypdfium2 render scale: ~150 DPI (72 DPI native × scale 2.0 ≈ 144 DPI).
PDF_RASTERIZE_SCALE = 2.0


def rasterize_pdf_pages(
    path: Path, *, max_pages: int, scale: float = PDF_RASTERIZE_SCALE
) -> list[bytes]:
    """Render up to *max_pages* pages of a PDF to PNG bytes.

    Uses pypdfium2 at *scale* (default ≈150 DPI) for legible output.
    Returns a list of raw PNG bytes, one entry per rendered page.
    """
    import pypdfium2  # noqa: PLC0415

    doc = pypdfium2.PdfDocument(str(path))
    page_count = len(doc)
    pages_to_render = min(page_count, max_pages)
    result: list[bytes] = []
    try:
        for i in range(pages_to_render):
            page = doc[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            result.append(buf.getvalue())
            page.close()
    finally:
        doc.close()
    return result


__all__ = ["PDF_RASTERIZE_SCALE", "rasterize_pdf_pages"]
