"""Compile Typst source to PDF bytes (HIG-244 Phase 1).

Thin wrapper over the ``typst`` binary. Uses stdin -> stdout (``typst compile
- -``) so no temp files are needed. Font directories can be supplied so a
worker image can ship the theme fonts instead of relying on system fonts.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

from kortny.documents.ir import DocumentSpec
from kortny.documents.typst_writer import render_document

DEFAULT_TYPST_BIN = "typst"


class TypstNotAvailableError(RuntimeError):
    """The ``typst`` binary could not be found on PATH / at the given path."""


class DocumentRenderError(RuntimeError):
    """Typst failed to compile the document; ``stderr`` holds the diagnostics."""

    def __init__(self, message: str, *, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def typst_available(typst_bin: str = DEFAULT_TYPST_BIN) -> bool:
    """Return whether the Typst binary is resolvable."""

    return shutil.which(typst_bin) is not None


def render_typst_pdf(
    source: str,
    *,
    typst_bin: str = DEFAULT_TYPST_BIN,
    font_paths: Sequence[str] = (),
    timeout_seconds: float = 60.0,
) -> bytes:
    """Compile Typst ``source`` to PDF bytes.

    Raises ``TypstNotAvailableError`` if the binary is missing and
    ``DocumentRenderError`` if compilation fails.
    """

    if shutil.which(typst_bin) is None:
        raise TypstNotAvailableError(f"typst binary {typst_bin!r} not found on PATH")

    cmd = [typst_bin, "compile"]
    for path in font_paths:
        cmd += ["--font-path", path]
    cmd += ["-", "-"]  # stdin -> stdout

    try:
        proc = subprocess.run(
            cmd,
            input=source.encode("utf-8"),
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise DocumentRenderError("typst compile timed out", stderr=str(exc)) from exc

    if proc.returncode != 0:
        raise DocumentRenderError(
            "typst compile failed",
            stderr=proc.stderr.decode("utf-8", errors="replace"),
        )
    return proc.stdout


def render_spec_pdf(
    spec: DocumentSpec,
    *,
    typst_bin: str = DEFAULT_TYPST_BIN,
    font_paths: Sequence[str] = (),
    timeout_seconds: float = 60.0,
) -> bytes:
    """Render an IR ``spec`` straight to PDF bytes."""

    return render_typst_pdf(
        render_document(spec),
        typst_bin=typst_bin,
        font_paths=font_paths,
        timeout_seconds=timeout_seconds,
    )
