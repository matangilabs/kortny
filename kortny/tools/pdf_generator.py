"""ReportLab-backed PDF generation tool."""

from __future__ import annotations

import html
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, Task, TaskEventType
from kortny.tools.types import JsonObject, JsonSchema, ToolArtifact, ToolResult

PDF_MIME_TYPE = "application/pdf"
DEFAULT_FILENAME = "report.pdf"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
SLACK_FILES_BLOCK_RE = re.compile(r"<slack_files>\s*(.*?)\s*</slack_files>", re.S)
SLACK_FILE_NAME_RE = re.compile(r"^\s*name:\s*(.+?)\s*$", re.M)
PDF_VERSION_RE = re.compile(r"^(?P<base>.+?)_v(?P<version>[0-9]+)$", re.I)
REVISION_TERMS = (
    "enhance",
    "extend",
    "elaborate",
    "revise",
    "update",
    "expand",
    "improve",
    "make it",
    "make this",
    "add ",
)


class TaskEventSink(Protocol):
    """Subset of TaskService needed for artifact event emission."""

    def append_event(
        self,
        task: uuid.UUID,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> object:
        """Append an event for a task."""


class PdfGeneratorTool:
    """Generate a structured report PDF in a task working directory."""

    name = "pdf_generator"
    description = "Generates a PDF report from structured title and section content."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The report title.",
            },
            "sections": {
                "type": "array",
                "description": "Ordered report sections.",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {
                            "type": "string",
                            "description": "Section heading.",
                        },
                        "body": {
                            "type": "string",
                            "description": "Section body text.",
                        },
                        "bullets": {
                            "type": "array",
                            "description": "Optional bullet points for the section.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["heading"],
                    "additionalProperties": False,
                },
            },
            "filename": {
                "type": "string",
                "description": "Optional output filename. The .pdf suffix is enforced.",
                "default": DEFAULT_FILENAME,
            },
            "min_pages": {
                "type": "integer",
                "description": (
                    "Optional minimum page count requested by the user. "
                    "If the generated PDF has fewer pages, the tool returns "
                    "a recoverable error instead of an artifact."
                ),
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["title", "sections"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        working_dir: str | Path,
        session: Session | None = None,
        task_id: uuid.UUID | None = None,
        task_service: TaskEventSink | None = None,
    ) -> None:
        if (session is None) != (task_id is None):
            raise ValueError("session and task_id must be provided together")

        self.working_dir = Path(working_dir)
        self.session = session
        self.task_id = task_id
        self.task_service = task_service

    def invoke(self, args: JsonObject) -> ToolResult:
        """Generate a PDF and return artifact metadata."""

        title = _required_string(args, "title")
        sections = _parse_sections(args.get("sections"))
        min_pages = _optional_min_pages(args.get("min_pages"))
        filename = self._output_filename(args.get("filename"))
        self.working_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.working_dir / filename

        _build_pdf(output_path, title=title, sections=sections)
        page_count = _pdf_page_count(output_path)
        if min_pages is not None and page_count < min_pages:
            output_path.unlink(missing_ok=True)
            return ToolResult(
                output={
                    "filename": filename,
                    "path": str(output_path),
                    "mime_type": PDF_MIME_TYPE,
                    "page_count": page_count,
                    "min_pages": min_pages,
                    "error": {
                        "code": "min_pages_not_met",
                        "message": (
                            f"Generated PDF has {page_count} page(s), "
                            f"but the user requested at least {min_pages}."
                        ),
                        "recoverable": True,
                    },
                },
            )

        size_bytes = output_path.stat().st_size
        artifact = ToolArtifact(
            filename=filename,
            path=str(output_path),
            mime_type=PDF_MIME_TYPE,
            size_bytes=size_bytes,
        )

        artifact_id: str | None = None
        if self.session is not None and self.task_id is not None:
            artifact_id = str(
                self._record_artifact(
                    filename=filename,
                    path=output_path,
                    size_bytes=size_bytes,
                ).id
            )

        return ToolResult(
            output={
                "filename": filename,
                "path": str(output_path),
                "mime_type": PDF_MIME_TYPE,
                "size_bytes": size_bytes,
                "page_count": page_count,
                "artifact_id": artifact_id,
            },
            artifacts=(artifact,),
        )

    def _output_filename(self, raw_filename: object) -> str:
        requested_filename = _safe_pdf_filename(raw_filename)
        revision_context = self._revision_context()
        if revision_context is None:
            return requested_filename
        return revision_context.next_filename

    def _record_artifact(
        self,
        *,
        filename: str,
        path: Path,
        size_bytes: int,
    ) -> Artifact:
        assert self.session is not None
        assert self.task_id is not None

        artifact = Artifact(
            task_id=self.task_id,
            filename=filename,
            mime_type=PDF_MIME_TYPE,
            size_bytes=size_bytes,
            storage_path=str(path),
        )
        self.session.add(artifact)
        self.session.flush()

        if self.task_service is not None:
            self.task_service.append_event(
                self.task_id,
                TaskEventType.artifact_created,
                {
                    "artifact_id": str(artifact.id),
                    "filename": filename,
                    "mime_type": PDF_MIME_TYPE,
                    "size_bytes": size_bytes,
                    "storage_path": str(path),
                },
            )

        return artifact

    def _revision_context(self) -> RevisionContext | None:
        if self.session is None or self.task_id is None:
            return None

        task = self.session.scalar(select(Task).where(Task.id == self.task_id))
        if task is None or not _looks_like_revision_request(task.input):
            return None

        prior_tasks = self._prior_thread_tasks(task)
        prior_artifacts = self._prior_artifacts(prior_tasks)
        source_filename = (
            _first_pdf_filename_from_task_input(task.input)
            or _latest_artifact_filename(prior_artifacts)
            or _latest_prior_input_pdf_filename(prior_tasks)
        )
        if source_filename is None:
            return None

        base_stem = _base_pdf_stem(source_filename)
        next_version = _next_pdf_version(base_stem, prior_artifacts)
        return RevisionContext(
            source_filename=source_filename,
            next_filename=f"{base_stem}_v{next_version}.pdf",
        )

    def _prior_thread_tasks(self, task: Task) -> list[Task]:
        assert self.session is not None
        if not task.slack_thread_ts:
            return []
        current_created_at = task.created_at
        return list(
            self.session.scalars(
                select(Task)
                .where(
                    Task.slack_channel_id == task.slack_channel_id,
                    Task.slack_thread_ts == task.slack_thread_ts,
                    Task.created_at < current_created_at,
                )
                .order_by(Task.created_at, Task.id)
            )
        )

    def _prior_artifacts(self, prior_tasks: list[Task]) -> list[Artifact]:
        assert self.session is not None
        prior_task_ids = [task.id for task in prior_tasks]
        if not prior_task_ids:
            return []
        return list(
            self.session.scalars(
                select(Artifact)
                .where(Artifact.task_id.in_(prior_task_ids))
                .order_by(Artifact.created_at)
            )
        )


@dataclass(frozen=True, slots=True)
class RevisionContext:
    source_filename: str
    next_filename: str


def _build_pdf(
    output_path: Path,
    *,
    title: str,
    sections: list[dict[str, Any]],
) -> None:
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph(_paragraph_text(title), styles["Title"]),
        Spacer(1, 0.25 * inch),
    ]

    for section in sections:
        story.append(Paragraph(_paragraph_text(section["heading"]), styles["Heading2"]))
        body = section.get("body")
        if isinstance(body, str) and body.strip():
            for paragraph in _split_paragraphs(body):
                story.append(Paragraph(_paragraph_text(paragraph), styles["BodyText"]))
                story.append(Spacer(1, 0.12 * inch))

        bullets = section.get("bullets")
        if isinstance(bullets, list):
            for bullet in bullets:
                if isinstance(bullet, str) and bullet.strip():
                    story.append(
                        Paragraph(f"- {_paragraph_text(bullet)}", styles["BodyText"])
                    )
                    story.append(Spacer(1, 0.08 * inch))

        story.append(Spacer(1, 0.18 * inch))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=title,
    )
    document.build(story)


def _parse_sections(raw_sections: object) -> list[dict[str, Any]]:
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("pdf_generator requires a non-empty 'sections' array")

    sections: list[dict[str, Any]] = []
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            raise ValueError("Each PDF section must be an object")
        heading = raw_section.get("heading")
        if not isinstance(heading, str) or not heading.strip():
            raise ValueError("Each PDF section requires a non-empty heading")
        body = raw_section.get("body")
        if body is not None and not isinstance(body, str):
            raise ValueError("PDF section body must be a string when provided")
        bullets = raw_section.get("bullets")
        if bullets is not None and (
            not isinstance(bullets, list)
            or not all(isinstance(bullet, str) for bullet in bullets)
        ):
            raise ValueError("PDF section bullets must be an array of strings")

        sections.append(
            {
                "heading": heading.strip(),
                "body": body,
                "bullets": bullets,
            }
        )

    return sections


def _required_string(args: JsonObject, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pdf_generator requires a non-empty {key!r} argument")
    return value.strip()


def _optional_min_pages(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("pdf_generator 'min_pages' must be an integer")
    if value < 1 or value > 50:
        raise ValueError("pdf_generator 'min_pages' must be between 1 and 50")
    return value


def _safe_pdf_filename(raw_filename: object) -> str:
    filename = raw_filename if isinstance(raw_filename, str) else DEFAULT_FILENAME
    safe_name = SAFE_FILENAME_RE.sub("_", Path(filename).name.strip())
    if safe_name in ("", ".", ".."):
        safe_name = DEFAULT_FILENAME
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    return safe_name


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _paragraph_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


def _pdf_page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def _looks_like_revision_request(input_text: str) -> bool:
    lowered = input_text.lower()
    return any(term in lowered for term in REVISION_TERMS)


def _first_pdf_filename_from_task_input(input_text: str) -> str | None:
    block = _slack_files_block(input_text)
    if block is None:
        return None
    for raw_name in SLACK_FILE_NAME_RE.findall(block):
        if Path(raw_name.strip()).suffix.lower() == ".pdf":
            return _safe_pdf_filename(raw_name)
    return None


def _slack_files_block(input_text: str) -> str | None:
    match = SLACK_FILES_BLOCK_RE.search(input_text)
    if match is None:
        return None
    content = match.group(1).strip()
    return content or None


def _latest_artifact_filename(artifacts: list[Artifact]) -> str | None:
    for artifact in reversed(artifacts):
        if artifact.filename.lower().endswith(".pdf"):
            return artifact.filename
    return None


def _latest_prior_input_pdf_filename(prior_tasks: list[Task]) -> str | None:
    for task in reversed(prior_tasks):
        filename = _first_pdf_filename_from_task_input(task.input)
        if filename is not None:
            return filename
    return None


def _base_pdf_stem(filename: str) -> str:
    stem = Path(_safe_pdf_filename(filename)).stem
    version_match = PDF_VERSION_RE.match(stem)
    if version_match is not None:
        return version_match.group("base")
    return stem


def _next_pdf_version(base_stem: str, artifacts: list[Artifact]) -> int:
    versions = [1]
    for artifact in artifacts:
        artifact_stem = Path(_safe_pdf_filename(artifact.filename)).stem
        if artifact_stem == base_stem:
            versions.append(1)
            continue
        version_match = PDF_VERSION_RE.match(artifact_stem)
        if version_match is not None and version_match.group("base") == base_stem:
            versions.append(int(version_match.group("version")))
    return max(versions) + 1
