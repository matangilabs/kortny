"""Slack image attachment resolution for vision-capable tasks (HIG-279 slice 2A).

Resolves ``(file_id, mime)`` pairs from the ``<slack_files>`` block into
``ImagePart`` objects by downloading each image with the bot token.  The
resolver is intentionally best-effort: individual download failures are logged
and skipped so a task never crashes because an image cannot be fetched.

``parse_image_attachment_pairs`` is re-exported from the leaf module
``kortny.agent.attachment_parsing`` so callers that already import this module
(e.g. ``kortny.agent.context``) can access the shared parser without needing a
second import.  The leaf module is what ``kortny.llm.routing`` imports directly
to avoid a package-level import cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx

from kortny.agent.attachment_parsing import parse_image_attachment_pairs
from kortny.llm.types import ImagePart

__all__ = [
    "parse_image_attachment_pairs",
    "ImageAttachmentResolver",
    "SlackImageAttachmentResolver",
]

logger = logging.getLogger(__name__)

# Public type alias: callable from ``(file_id, mime)`` pairs → ImageParts.
ImageAttachmentResolver = Callable[
    [Sequence[tuple[str, str]]],
    tuple[ImagePart, ...],
]


class _SlackFilesInfoClient(Protocol):
    """Minimal Slack client subset needed to look up a file's download URL."""

    def files_info(self, *, file: str) -> Any:
        """Fetch Slack file metadata."""


class SlackImageAttachmentResolver:
    """Download Slack image files and return ``ImagePart`` objects.

    Parameters
    ----------
    client:
        Any Slack client that implements ``files_info(file=...)`` — typically
        a ``slack_sdk.WebClient`` instance.
    bot_token:
        The Slack bot token used as the ``Authorization`` header when streaming
        private download URLs.
    max_image_bytes:
        Per-image size cap.  Files larger than this are skipped (the hard
        ``LLMService`` guard will still fire for anything that sneaks through).
    allowed_mimes:
        Frozenset of allowed MIME types.  Files with other MIME types are
        silently skipped — they are not images the model can process.
    timeout:
        HTTP timeout in seconds for the download request.
    transport:
        Optional httpx transport override (injected in tests to avoid real
        network calls).
    """

    def __init__(
        self,
        *,
        client: _SlackFilesInfoClient,
        bot_token: str,
        max_image_bytes: int,
        allowed_mimes: frozenset[str],
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = client
        self._bot_token = bot_token
        self._max_image_bytes = max_image_bytes
        self._allowed_mimes = allowed_mimes
        self._timeout = timeout
        self._transport = transport

    def __call__(
        self,
        file_pairs: Sequence[tuple[str, str]],
    ) -> tuple[ImagePart, ...]:
        """Resolve ``(file_id, mime)`` pairs into ``ImagePart`` objects.

        Each file is downloaded independently.  Failures (network errors, API
        errors, size violations, disallowed MIME types) are logged at WARNING
        level and skipped — the task continues text-only for that file.
        """
        parts: list[ImagePart] = []
        for file_id, mime in file_pairs:
            try:
                part = self._resolve_one(file_id, mime)
            except Exception as exc:
                logger.warning(
                    "image_attachment: skipping file_id=%s mime=%s reason=%s",
                    file_id,
                    mime,
                    exc,
                )
                continue
            if part is not None:
                parts.append(part)
        return tuple(parts)

    def _resolve_one(self, file_id: str, mime: str) -> ImagePart | None:
        """Return an ``ImagePart`` for ``file_id``, or ``None`` to skip."""

        if mime not in self._allowed_mimes:
            logger.debug(
                "image_attachment: skipping file_id=%s — mime %s not in allowed set",
                file_id,
                mime,
            )
            return None

        download_url = self._fetch_download_url(file_id)
        if download_url is None:
            logger.warning(
                "image_attachment: skipping file_id=%s — no download URL in files.info",
                file_id,
            )
            return None

        data = self._download(file_id, download_url)
        if data is None:
            return None

        if len(data) > self._max_image_bytes:
            logger.warning(
                "image_attachment: skipping file_id=%s — %d bytes exceeds limit %d",
                file_id,
                len(data),
                self._max_image_bytes,
            )
            return None

        return ImagePart(data=data, mime=mime, source=f"slack_file:{file_id}")

    def _fetch_download_url(self, file_id: str) -> str | None:
        """Call ``files.info`` and return the best download URL or None."""
        try:
            response = self._client.files_info(file=file_id)
        except Exception as exc:
            raise RuntimeError(f"files.info failed: {exc}") from exc

        # Handle both dict-like and object responses from the SDK.
        payload: Any
        if isinstance(response, dict):
            payload = response
        else:
            payload = getattr(response, "data", None) or response

        if isinstance(payload, dict) and payload.get("ok") is False:
            error = payload.get("error", "unknown_error")
            raise RuntimeError(f"files.info API error: {error}")

        raw_file: Any = None
        if isinstance(payload, dict):
            raw_file = payload.get("file")

        if not isinstance(raw_file, dict):
            return None

        url = raw_file.get("url_private_download") or raw_file.get("url_private")
        if not isinstance(url, str) or not url.strip():
            return None
        return url.strip()

    def _download(self, file_id: str, url: str) -> bytes | None:
        """Stream ``url`` with the bot token and return raw bytes."""
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        try:
            chunks: list[bytes] = []
            total = 0
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=self._timeout,
                    follow_redirects=True,
                ) as http,
                http.stream("GET", url, headers=headers) as resp,
            ):
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > self._max_image_bytes:
                        logger.warning(
                            "image_attachment: skipping file_id=%s — "
                            "download exceeded %d bytes mid-stream",
                            file_id,
                            self._max_image_bytes,
                        )
                        return None
                    chunks.append(chunk)
            return b"".join(chunks)
        except Exception as exc:
            raise RuntimeError(f"download failed: {exc}") from exc
