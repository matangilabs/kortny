"""Document Studio tool — editorial-grade PDF from a structured IR (HIG-244).

The agent passes a structured document spec (intent + themed blocks); the tool
validates it, renders it to PDF through the Typst beauty engine, writes the
artifact into the task working dir, and records it. Unlike ``pdf_generator``
(freeform sections -> ReportLab), this produces themed, paginated, editorial
output from a typed block vocabulary.

Validation is strict and recoverable: a malformed spec raises
``RecoverableToolError`` with the validation detail so the model can fix the
shape and retry, rather than failing the task.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, TaskEventType
from kortny.documents import (
    DocumentRenderError,
    DocumentSpec,
    TypstNotAvailableError,
    render_spec_pdf,
    theme_names,
)
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    ToolArtifact,
    ToolResult,
)

PDF_MIME_TYPE = "application/pdf"
DEFAULT_FILENAME = "document.pdf"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_BLOCK_GUIDE = (
    "Ordered content blocks. Each block is an object with a 'type' field and "
    "type-specific fields:\n"
    "- cover_header: title (req), eyebrow, subtitle, accent_tail (a trailing "
    "fragment of the title to accent-colour), meta (string list).\n"
    "- section_divider: title (req), index (e.g. '01'), label, subtitle. Renders "
    "as its own full-page break.\n"
    "- heading: text (req).\n"
    "- prose: text (req); blank lines separate paragraphs.\n"
    "- stat_cards: cards (req, 1-4) of {value, label, note}.\n"
    "- table: columns (req string list), rows (list of string lists), caption.\n"
    "- callout: text (req), label.\n"
    "- pull_quote: text (req), attribution.\n"
    "- cta: label (req), text."
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


class DocumentStudioTool:
    """Render a structured document spec to an editorial-grade PDF."""

    name = "document_studio"
    description = (
        "Generate a beautiful, themed, multi-page PDF from a structured document "
        "spec. Use for reports, briefs, and pitch documents that should look "
        "editorial-grade (cover, section dividers, stat cards, tables, callouts, "
        "pull quotes). Prefer this over pdf_generator for any polished deliverable."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "doc_kind": {
                "type": "string",
                "enum": ["pitch", "report", "brief", "memo"],
                "description": (
                    "Document archetype. Drives density and default theme: "
                    "'pitch' = sparse + very beautiful; 'report'/'brief' = "
                    "information-dense; 'memo' = light. Defaults to 'report'."
                ),
            },
            "title": {
                "type": "string",
                "description": "Document title (used for metadata + footer).",
            },
            "theme": {
                "type": "string",
                "description": (
                    "Optional theme name. Defaults from doc_kind. "
                    f"Available: {', '.join(theme_names())}."
                ),
            },
            "blocks": {
                "type": "array",
                "description": _BLOCK_GUIDE,
                "items": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}},
                    "required": ["type"],
                    "additionalProperties": True,
                },
                "minItems": 1,
            },
            "filename": {
                "type": "string",
                "description": "Optional output filename; the .pdf suffix is enforced.",
                "default": DEFAULT_FILENAME,
            },
        },
        "required": ["title", "blocks"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        working_dir: str | Path,
        font_paths: Sequence[str] = (),
        session: Session | None = None,
        task_id: uuid.UUID | None = None,
        task_service: TaskEventSink | None = None,
    ) -> None:
        if (session is None) != (task_id is None):
            raise ValueError("session and task_id must be provided together")
        self.working_dir = Path(working_dir)
        self.font_paths = tuple(font_paths)
        self.session = session
        self.task_id = task_id
        self.task_service = task_service

    def invoke(self, args: JsonObject) -> ToolResult:
        """Validate the spec, render the PDF, and record the artifact."""

        spec = _parse_spec(args)
        filename = _safe_pdf_filename(args.get("filename"))

        try:
            pdf = render_spec_pdf(spec, font_paths=self.font_paths)
        except TypstNotAvailableError as exc:
            raise RecoverableToolError(
                code="typst_unavailable",
                message="The document renderer (Typst) is not available.",
                hint=(
                    "Fall back to the pdf_generator tool for a plain PDF, or "
                    "inform the user that document rendering is not configured."
                ),
            ) from exc
        except DocumentRenderError as exc:
            raise RecoverableToolError(
                code="document_render_failed",
                message="Failed to render the document.",
                hint="Check the block fields; simplify content and retry.",
                details={"stderr": exc.stderr[:2000]},
            ) from exc

        self.working_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.working_dir / filename
        output_path.write_bytes(pdf)
        size_bytes = len(pdf)

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
                    filename=filename, path=output_path, size_bytes=size_bytes
                ).id
            )

        return ToolResult(
            output={
                "filename": filename,
                "path": str(output_path),
                "mime_type": PDF_MIME_TYPE,
                "size_bytes": size_bytes,
                "doc_kind": spec.doc_kind.value,
                "block_count": len(spec.blocks),
                "artifact_id": artifact_id,
            },
            artifacts=(artifact,),
        )

    def _record_artifact(
        self, *, filename: str, path: Path, size_bytes: int
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


def _parse_spec(args: JsonObject) -> DocumentSpec:
    payload = {k: v for k, v in args.items() if k != "filename"}
    try:
        return DocumentSpec.model_validate(payload)
    except ValidationError as exc:
        raise RecoverableToolError(
            code="invalid_document_spec",
            message="The document spec failed validation.",
            hint="Fix the reported fields and retry. See the tool schema.",
            details={"errors": exc.errors(include_url=False, include_input=False)},
        ) from exc


def _safe_pdf_filename(raw_filename: object) -> str:
    filename = raw_filename if isinstance(raw_filename, str) else DEFAULT_FILENAME
    safe_name = SAFE_FILENAME_RE.sub("_", Path(filename).name.strip())
    if safe_name in ("", ".", ".."):
        safe_name = DEFAULT_FILENAME
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    return safe_name
