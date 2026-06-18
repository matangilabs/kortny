"""Compile Typst source to PDF bytes (HIG-244 Phase 1).

Uses the official ``typst`` PyPI package, which bundles the compiler as a wheel
— so the engine rides in via ``uv sync`` with no system binary or Docker
change, and runs identically in dev, CI, and the worker image. Font
directories can be supplied so a deployment can ship deterministic theme fonts
instead of relying on whatever system fonts exist.
"""

from __future__ import annotations

from collections.abc import Sequence

try:  # pragma: no cover - import guard
    import typst as _typst
except ImportError:  # pragma: no cover - the package is a hard dependency
    _typst = None  # type: ignore[assignment]

from kortny.documents.ir import DocumentSpec
from kortny.documents.typst_writer import render_document


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
) -> bytes:
    """Compile Typst ``source`` to PDF bytes.

    Raises ``TypstNotAvailableError`` if the compiler is missing and
    ``DocumentRenderError`` if compilation fails.
    """

    if _typst is None:
        raise TypstNotAvailableError("the typst compiler package is not installed")

    try:
        # The package's overload binds font_paths' element type to the input
        # type (bytes source would force list[bytes] paths), but str paths are
        # what fontdb wants at runtime — so the overload is overly strict here.
        result = _typst.compile(
            source.encode("utf-8"),
            format="pdf",
            font_paths=list(font_paths),  # type: ignore[call-overload]
        )
    except Exception as exc:  # typst raises its own TypstError on bad source
        raise DocumentRenderError("typst compile failed", stderr=str(exc)) from exc

    # The package returns bytes when no output path is given.
    if not isinstance(result, (bytes, bytearray)):
        raise DocumentRenderError("typst compile returned no PDF bytes")
    return bytes(result)


def render_spec_pdf(
    spec: DocumentSpec,
    *,
    font_paths: Sequence[str] = (),
) -> bytes:
    """Render an IR ``spec`` straight to PDF bytes."""

    return render_typst_pdf(render_document(spec), font_paths=font_paths)
