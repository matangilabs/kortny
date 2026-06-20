"""Tests for VisualCritique models and visual_critique function (HIG-244)."""

from __future__ import annotations

import io
from collections.abc import Sequence

import pytest

from kortny.documents.critique import (
    VisualCritique,
    VisualIssue,
    visual_critique,
)


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


def _fake_critic(pages: Sequence[bytes]) -> VisualCritique:
    return VisualCritique(
        overall_score=7,
        summary="Looks mostly good.",
        issues=[
            VisualIssue(
                page=1,
                category="whitespace",
                severity="warning",
                message="Slight excess whitespace at bottom of page 1.",
            )
        ],
    )


def test_visual_critique_with_fake_critic() -> None:
    pdf_bytes = _make_blank_pdf_bytes(1)
    result = visual_critique(pdf_bytes, _fake_critic, max_pages=8)
    assert result is not None
    assert result.overall_score == 7
    assert result.summary == "Looks mostly good."
    assert len(result.issues) == 1
    assert result.issues[0].page == 1
    assert result.issues[0].category == "whitespace"


def test_visual_critique_with_none_critic() -> None:
    pdf_bytes = _make_blank_pdf_bytes(1)
    result = visual_critique(pdf_bytes, None, max_pages=8)
    assert result is None


def test_visual_critique_critic_raises_returns_none() -> None:
    def bad_critic(pages: Sequence[bytes]) -> VisualCritique:
        raise RuntimeError("LLM exploded")

    pdf_bytes = _make_blank_pdf_bytes(1)
    result = visual_critique(pdf_bytes, bad_critic, max_pages=8)
    assert result is None


def test_visual_issue_model_validation() -> None:
    issue = VisualIssue(
        page=2, category="overflow", severity="error", message="Text overflows margin"
    )
    assert issue.page == 2
    assert issue.category == "overflow"
    assert issue.severity == "error"


def test_visual_critique_model_validation() -> None:
    critique = VisualCritique(overall_score=10, summary="Perfect.", issues=[])
    assert critique.overall_score == 10
    assert critique.issues == []


def test_visual_issue_category_literal() -> None:
    from pydantic import ValidationError  # noqa: PLC0415

    bad_category: str = "not_a_category"
    with pytest.raises(ValidationError):
        VisualIssue(
            page=1,
            category=bad_category,  # type: ignore[arg-type]
            severity="warning",
            message="x",
        )


def test_visual_critique_score_range() -> None:
    # Valid: 0-10
    c = VisualCritique(overall_score=0, summary="Broken.", issues=[])
    assert c.overall_score == 0
    c2 = VisualCritique(overall_score=10, summary="Perfect.", issues=[])
    assert c2.overall_score == 10
