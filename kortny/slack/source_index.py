"""Server-built index of citable sources for the Block Kit ``sources`` element.

The trust rule (HIG-255): the humanizer LLM may *choose* which sources to surface
(by ``source_ref``) but must never author the URL — a hallucinated or phishing
link must be impossible. So URLs come only from here: the index is built
deterministically from the response record's evidence (tool results that carried
URLs), exposed to the LLM as ``available_sources`` (ref + domain + snippet) so it
can reference them, and resolved back to the real URL at render time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from kortny.tools.types import JsonObject

# Render caps (codex): a handful of sources, never a wall.
MAX_SOURCES = 10
_SNIPPET_CHARS = 160


class _EvidenceLike(Protocol):
    @property
    def urls(self) -> list[str] | None: ...

    @property
    def preview(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class RenderSource:
    """A resolved, renderable source — the URL is server-owned, never LLM copy."""

    ref: str
    url: str
    domain: str
    snippet: str | None


class SourceIndex:
    """Maps a ``source_ref`` to a server-owned :class:`RenderSource`."""

    def __init__(self, sources: Sequence[RenderSource]) -> None:
        self._ordered = tuple(sources)
        self._by_ref = {source.ref: source for source in sources}

    def __bool__(self) -> bool:
        return bool(self._ordered)

    def resolve(self, ref: str) -> RenderSource | None:
        return self._by_ref.get(ref)

    def available(self) -> list[JsonObject]:
        """The catalog shown to the LLM so it knows which refs it may cite."""

        return [
            {"ref": s.ref, "domain": s.domain, "snippet": s.snippet or ""}
            for s in self._ordered
        ]


def build_source_index(evidence: Sequence[_EvidenceLike]) -> SourceIndex:
    """Flatten evidence URLs (in order) into ``source:0``, ``source:1``, …

    Only http(s) URLs become sources. The same deterministic ordering is used to
    expose ``available_sources`` to the LLM and to resolve refs at render time,
    so the indices the model sees always line up with what the renderer binds.
    """

    sources: list[RenderSource] = []
    for item in evidence:
        snippet = (item.preview or "").strip() or None
        if snippet and len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS].rstrip() + "…"
        for url in item.urls or []:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            ref = f"source:{len(sources)}"
            sources.append(
                RenderSource(
                    ref=ref,
                    url=url,
                    domain=_domain(url),
                    snippet=snippet,
                )
            )
            if len(sources) >= MAX_SOURCES:
                return SourceIndex(sources)
    return SourceIndex(sources)


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc or url


__all__ = ["MAX_SOURCES", "RenderSource", "SourceIndex", "build_source_index"]
