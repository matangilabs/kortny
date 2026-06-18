"""Compile Typst source to PDF bytes (HIG-244 Phase 1).

Uses the official ``typst`` PyPI package, which bundles the compiler as a wheel
— so the engine rides in via ``uv sync`` with no system binary or Docker
change, and runs identically in dev, CI, and the worker image. Font
directories can be supplied so a deployment can ship deterministic theme fonts
instead of relying on whatever system fonts exist.

Charts compile to side-car SVG assets the source references by filename; when
present we compile from a temp directory (with it as the Typst ``root``) so
those ``#image`` references resolve. With no assets we compile straight from
stdin bytes.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

try:  # pragma: no cover - import guard
    import typst as _typst
except ImportError:  # pragma: no cover - the package is a hard dependency
    _typst = None  # type: ignore[assignment]

from kortny.documents.ir import DocumentSpec
from kortny.documents.typst_writer import build_typst


class TypstNotAvailableError(RuntimeError):
    """The ``typst`` compiler package could not be imported."""


class DocumentRenderError(RuntimeError):
    """Typst failed to compile the document; ``stderr`` holds the diagnostics."""

    def __init__(self, message: str, *, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def typst_available() -> bool:
    """Return whether the Typst compiler is importable."""

    return _typst is not None


def render_typst_pdf(
    source: str,
    *,
    font_paths: Sequence[str] = (),
    assets: Mapping[str, bytes] | None = None,
) -> bytes:
    """Compile Typst ``source`` to PDF bytes.

    ``assets`` maps filename -> bytes for side-car files (e.g. chart SVGs) the
    source references; when given, compilation happens in a temp dir so the
    references resolve. Raises ``TypstNotAvailableError`` if the compiler is
    missing and ``DocumentRenderError`` if compilation fails.
    """

    if _typst is None:
        raise TypstNotAvailableError("the typst compiler package is not installed")

    fonts = list(font_paths)
    try:
        if assets:
            with tempfile.TemporaryDirectory(prefix="kortny-doc-") as tmp:
                root = Path(tmp)
                for name, data in assets.items():
                    (root / name).write_bytes(data)
                main = root / "main.typ"
                main.write_text(source, encoding="utf-8")
                result = _typst.compile(
                    str(main), root=str(root), font_paths=fonts, format="pdf"
                )
        else:
            result = _typst.compile(
                source.encode("utf-8"),
                format="pdf",
                font_paths=fonts,  # type: ignore[call-overload]
            )
    except Exception as exc:  # typst raises its own TypstError on bad source
        raise DocumentRenderError("typst compile failed", stderr=str(exc)) from exc

    if not isinstance(result, (bytes, bytearray)):
        raise DocumentRenderError("typst compile returned no PDF bytes")
    return bytes(result)


def render_spec_pdf(
    spec: DocumentSpec,
    *,
    font_paths: Sequence[str] = (),
) -> bytes:
    """Render an IR ``spec`` straight to PDF bytes."""

    source, assets = build_typst(spec)
    return render_typst_pdf(source, font_paths=font_paths, assets=assets)
