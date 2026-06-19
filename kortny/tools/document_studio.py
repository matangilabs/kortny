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
    ChartRenderError,
    DocumentRenderError,
    DocumentSpec,
    TypstNotAvailableError,
    render_docx,
    render_pptx,
    render_spec_pdf,
    render_xlsx,
    theme_names,
    xlsx_is_poor_fit,
)
from kortny.documents.critique import (
    DocumentIssue,
    critique_and_fix,
    validate_render,
)
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    ToolArtifact,
    ToolResult,
)

# format -> (mime, extension)
_FORMATS: dict[str, tuple[str, str]] = {
    "pdf": ("application/pdf", "pdf"),
    "pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pptx",
    ),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx",
    ),
    "xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
    ),
}
DEFAULT_FORMAT = "pdf"
DEFAULT_STEM = "document"
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
    "- cta: label (req), text.\n"
    "- chart: chart_type (bar|line|area|pie|scatter), title, x_label, y_label, "
    "caption, series (req, 1-8) of {name, points:[{x,y}]}. Use for real data "
    "visualizations; x is a category label or a number, y is numeric.\n"
    "Composition: a section_divider takes its own full page, so put dividers "
    "BETWEEN major sections — never immediately after the cover, and never strand "
    "a single small block (e.g. stat_cards alone) right before a divider, which "
    "leaves a near-empty page. Open the body with a heading + intro prose and the "
    "headline stat_cards together."
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
    """Render a structured document spec to an editorial-grade PDF/PPTX/DOCX."""

    name = "document_studio"
    description = (
        "Generate a beautiful, themed document from a structured spec, as PDF, "
        "PowerPoint (pptx), or Word (docx). This is THE tool for any polished "
        "deliverable — reports, research notes, briefs, one-pagers, decks, "
        "leave-behinds — with cover, section dividers, stat cards, tables, "
        "charts, callouts, and pull quotes. Choose 'pptx' for a slide deck, "
        "'docx' for an editable Word doc the user will keep editing, 'pdf' for a "
        "finished deliverable (default). Prefer this over pdf_generator for any "
        "polished output. Exercise judgment about presenting data: when the "
        "content has comparisons, trends, or proportions, add a chart block "
        "(bar=compare categories, line/area=trend over time, pie=share with <=6 "
        "slices, scatter=correlation) instead of burying numbers in prose — the "
        "user will rarely ask for a chart explicitly; decide for them."
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
            "format": {
                "type": "string",
                "enum": ["pdf", "pptx", "docx", "xlsx"],
                "description": (
                    "Output format. 'pdf' = finished deliverable (default), "
                    "'pptx' = slide deck, 'docx' = editable Word document, "
                    "'xlsx' = spreadsheet DATA EXPORT — use only when the answer "
                    "is mostly tables/metrics/chart data the user will analyze, "
                    "never for a prose narrative (use pdf/docx for those)."
                ),
            },
            "filename": {
                "type": "string",
                "description": (
                    "Optional output filename; the correct extension for the "
                    "chosen format is enforced."
                ),
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
        """Validate the spec, render the chosen format, record the artifact."""

        spec = _parse_spec(args)
        fmt = _parse_format(args.get("format"))
        mime_type, extension = _FORMATS[fmt]
        filename = _safe_filename(args.get("filename"), extension)

        # Deterministic critique: lint + semantics-preserving auto-fix before
        # render (HIG-244). A doc that lints down to nothing is recoverable.
        critique = critique_and_fix(spec)
        if critique.has_errors:
            raise RecoverableToolError(
                code="document_spec_needs_revision",
                message="The document spec has defects that can't be auto-fixed.",
                hint="Address the reported issues (e.g. add content) and retry.",
                details={"issues": [i.model_dump() for i in critique.issues]},
            )
        spec = critique.spec
        issues = list(critique.issues)
        if fmt == "xlsx" and xlsx_is_poor_fit(spec):
            issues.append(
                DocumentIssue(
                    code="xlsx_poor_fit",
                    severity="warning",
                    message=(
                        "This document is mostly prose; a spreadsheet is a weak "
                        "fit. Consider pdf or docx."
                    ),
                )
            )

        try:
            data = self._render(spec, fmt)
        except TypstNotAvailableError as exc:
            raise RecoverableToolError(
                code="typst_unavailable",
                message="The document renderer (Typst) is not available.",
                hint=(
                    "Fall back to the pdf_generator tool for a plain PDF, or "
                    "inform the user that document rendering is not configured."
                ),
            ) from exc
        except ChartRenderError as exc:
            raise RecoverableToolError(
                code="chart_render_failed",
                message="Failed to render a chart in the document.",
                hint=(
                    "Check the chart block's data and encoding (valid numbers, "
                    "matching series), simplify the chart, or drop it and retry."
                ),
                details={"error": str(exc)[:2000]},
            ) from exc
        except DocumentRenderError as exc:
            raise RecoverableToolError(
                code="document_render_failed",
                message="Failed to render the document.",
                hint="Check the block fields; simplify content and retry.",
                details={"stderr": exc.stderr[:2000]},
            ) from exc

        # Post-render validation: the bytes must be a real file of the right kind.
        render_issues = validate_render(data, fmt)
        issues.extend(render_issues)
        if any(issue.severity == "error" for issue in render_issues):
            raise RecoverableToolError(
                code="document_render_invalid",
                message="The rendered document failed validation.",
                hint="Simplify the content and retry.",
                details={"issues": [i.model_dump() for i in render_issues]},
            )

        self.working_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.working_dir / filename
        output_path.write_bytes(data)
        size_bytes = len(data)

        artifact = ToolArtifact(
            filename=filename,
            path=str(output_path),
            mime_type=mime_type,
            size_bytes=size_bytes,
        )

        artifact_id: str | None = None
        if self.session is not None and self.task_id is not None:
            artifact_id = str(
                self._record_artifact(
                    filename=filename,
                    path=output_path,
                    size_bytes=size_bytes,
                    mime_type=mime_type,
                ).id
            )

        return ToolResult(
            output={
                "filename": filename,
                "path": str(output_path),
                "mime_type": mime_type,
                "format": fmt,
                "size_bytes": size_bytes,
                "doc_kind": spec.doc_kind.value,
                "block_count": len(spec.blocks),
                "artifact_id": artifact_id,
                "critique": {
                    "autofixes": sum(1 for i in issues if i.autofix == "applied"),
                    "warnings": sum(1 for i in issues if i.severity == "warning"),
                    "codes": sorted({i.code for i in issues}),
                },
            },
            artifacts=(artifact,),
        )

    def _render(self, spec: DocumentSpec, fmt: str) -> bytes:
        if fmt == "pptx":
            return render_pptx(spec)
        if fmt == "docx":
            return render_docx(spec)
        if fmt == "xlsx":
            return render_xlsx(spec)
        return render_spec_pdf(spec, font_paths=self.font_paths)

    def _record_artifact(
        self, *, filename: str, path: Path, size_bytes: int, mime_type: str
    ) -> Artifact:
        assert self.session is not None
        assert self.task_id is not None

        artifact = Artifact(
            task_id=self.task_id,
            filename=filename,
            mime_type=mime_type,
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
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "storage_path": str(path),
                },
            )
        return artifact


# The IR fields live alongside the tool-only knobs in args; strip the knobs
# before validating the spec.
_NON_SPEC_KEYS = frozenset({"filename", "format"})


def _parse_spec(args: JsonObject) -> DocumentSpec:
    payload = {k: v for k, v in args.items() if k not in _NON_SPEC_KEYS}
    try:
        return DocumentSpec.model_validate(payload)
    except ValidationError as exc:
        raise RecoverableToolError(
            code="invalid_document_spec",
            message="The document spec failed validation.",
            hint="Fix the reported fields and retry. See the tool schema.",
            details={"errors": exc.errors(include_url=False, include_input=False)},
        ) from exc


def _parse_format(raw_format: object) -> str:
    if raw_format is None:
        return DEFAULT_FORMAT
    if isinstance(raw_format, str) and raw_format in _FORMATS:
        return raw_format
    raise RecoverableToolError(
        code="invalid_document_format",
        message=f"Unsupported document format {raw_format!r}.",
        hint=f"Use one of: {', '.join(_FORMATS)}.",
    )


def _safe_filename(raw_filename: object, extension: str) -> str:
    stem_source = raw_filename if isinstance(raw_filename, str) else DEFAULT_STEM
    safe_name = SAFE_FILENAME_RE.sub("_", Path(stem_source).name.strip())
    if safe_name in ("", ".", ".."):
        safe_name = DEFAULT_STEM
    # Strip any user-supplied extension, then enforce the format's extension.
    stem = Path(safe_name).stem if "." in safe_name else safe_name
    return f"{stem}.{extension}"
