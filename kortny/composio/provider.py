"""Provider-neutral Composio external tool adapter.

HIG-222: when an embedding index is available, candidate listing is a pure
semantic rank over the synced ``composio_tool_cards`` catalog (kind
``tool_card``) — no hot-path Composio HTTP. Full runtime schemas are fetched
lazily, only for the tools that survive selection. When embeddings are disabled
(``KORTNY_EMBEDDINGS_BACKEND=disabled``) the provider falls back to the original
per-task Composio search, byte-compatible with the pre-HIG-222 behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.composio.catalog_sync import (
    TOOL_CARD_EMBEDDING_KIND,
    ComposioCatalogSyncService,
)
from kortny.composio.client import ComposioCatalogError, ComposioClient, ComposioTool
from kortny.composio.runtime import (
    ComposioConnectionResolver,
    RuntimeComposioConnection,
)
from kortny.composio.tool_cards import synced_tool_card, tool_card_for
from kortny.db.models import ComposioToolCard, Task
from kortny.embeddings import EmbeddingIndex
from kortny.observability.events import log_observation
from kortny.tool_selection import tool_card_embedding_text
from kortny.tool_selection.models import ToolCard
from kortny.tools.composio_execute import (
    DEFAULT_RESULT_MAX_CHARS,
    ComposioExecuteTool,
    composio_runtime_tool_name,
)
from kortny.tools.types import Tool

logger = logging.getLogger(__name__)

# Top-K cap for synced-catalog candidate listing (HIG-222 spec: stays 15).
DEFAULT_TOOL_RETRIEVAL_TOP_K = 15
# Per intent-named connected toolkit, how many of its best tools to force past
# the top_k catalog-ranking cap (HIG-274 reachability floor).
FORCED_TOOLS_PER_INTENT_TOOLKIT = 3


@dataclass(frozen=True, slots=True)
class _ToolkitCatalog:
    connection: RuntimeComposioConnection
    tools: tuple[ComposioTool, ...]


@dataclass(frozen=True, slots=True)
class _SyncedCandidate:
    connection: RuntimeComposioConnection
    tool_slug: str
    card: ToolCard


class ComposioExternalToolProvider:
    """Expose scoped Composio connections as selectable external tools."""

    provider_name = "composio"

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        client: ComposioClient,
        per_toolkit_limit: int = 8,
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
        embedding_index: EmbeddingIndex | None = None,
        top_k: int = DEFAULT_TOOL_RETRIEVAL_TOP_K,
        forced_toolkits: tuple[str, ...] = (),
    ) -> None:
        self.session = session
        self.task = task
        self.client = client
        self.per_toolkit_limit = per_toolkit_limit
        self.result_max_chars = result_max_chars
        self.embedding_index = embedding_index
        self.top_k = top_k
        self.forced_toolkits = tuple(
            dict.fromkeys(slug.casefold() for slug in forced_toolkits if slug)
        )
        self.resolver = ComposioConnectionResolver(session, task)
        self._catalog: tuple[_ToolkitCatalog, ...] | None = None
        self._synced_candidates: tuple[_SyncedCandidate, ...] | None = None

    def tool_cards(self) -> tuple[ToolCard, ...]:
        if self.embedding_index is not None:
            return tuple(item.card for item in self._load_synced_candidates())
        cards: list[ToolCard] = []
        for entry in self._load_catalog():
            cards.extend(_tool_cards(entry))
        return tuple(cards)

    def runtime_tools(self) -> tuple[Tool, ...]:
        if self.embedding_index is not None:
            return self._synced_runtime_tools()
        tools: list[Tool] = []
        for entry in self._load_catalog():
            for tool in entry.tools:
                tools.append(self._execute_tool(entry.connection, tool))
        return tuple(tools)

    # --- Synced-catalog path (HIG-222) --------------------------------------

    def _synced_runtime_tools(self) -> tuple[Tool, ...]:
        """Lazily fetch full schemas ONLY for the surviving candidates.

        ``_load_synced_candidates`` has already ranked the synced catalog down
        to <= top_k cards; here we resolve their full input schemas with a
        bounded Composio fetch (per toolkit), so the latency win is real: we
        never fetch schemas for the whole catalog.
        """

        candidates = self._load_synced_candidates()
        if not candidates:
            return ()
        slugs_by_toolkit: dict[str, set[str]] = {}
        connections: dict[str, RuntimeComposioConnection] = {}
        for item in candidates:
            slugs_by_toolkit.setdefault(item.connection.toolkit_slug, set()).add(
                item.tool_slug
            )
            connections.setdefault(item.connection.toolkit_slug, item.connection)
        tools: list[Tool] = []
        for toolkit_slug, slugs in slugs_by_toolkit.items():
            connection = connections[toolkit_slug]
            try:
                fetched = self.client.list_tools(
                    toolkit_slug=toolkit_slug,
                    tool_slugs=tuple(sorted(slugs)),
                    limit=len(slugs),
                )
            except ComposioCatalogError as exc:
                log_observation(
                    logger,
                    "composio_schema_fetch_failed",
                    task=self.task,
                    provider="composio",
                    toolkit_slug=toolkit_slug,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            for tool in fetched:
                if tool.slug in slugs:
                    tools.append(self._execute_tool(connection, tool))
        return tuple(tools)

    def _load_synced_candidates(self) -> tuple[_SyncedCandidate, ...]:
        if self._synced_candidates is not None:
            return self._synced_candidates

        connections = {
            connection.toolkit_slug: connection
            for connection in _best_connections_by_toolkit(
                self.resolver.allowed_connections()
            )
        }
        rows = self._load_synced_rows(set(connections))
        # On-demand: a connected toolkit with zero synced cards triggers a
        # bounded (one-toolkit) sync so a fresh connection works this task.
        missing = [
            slug
            for slug in connections
            if slug not in {row.toolkit_slug for row in rows}
        ]
        if missing:
            self._on_demand_sync(missing[0])
            rows = self._load_synced_rows(set(connections))

        candidates: list[_SyncedCandidate] = []
        by_ref: dict[str, _SyncedCandidate] = {}
        for row in rows:
            connection = connections.get(row.toolkit_slug)
            if connection is None:
                continue
            card = synced_tool_card(
                connection=connection,
                tool_slug=row.tool_slug,
                name=row.name,
                description=row.description,
                side_effect=row.side_effect,
            )
            candidate = _SyncedCandidate(
                connection=connection,
                tool_slug=row.tool_slug,
                card=card,
            )
            candidates.append(candidate)
            by_ref[card.registry_name] = candidate

        selected = self._rank_candidates(candidates, by_ref)
        self._synced_candidates = selected
        log_observation(
            logger,
            "composio_synced_catalog_ranked",
            task=self.task,
            provider="composio",
            candidate_count=len(candidates),
            selected_count=len(selected),
            toolkit_slugs=sorted({item.connection.toolkit_slug for item in selected}),
        )
        return selected

    def _rank_candidates(
        self,
        candidates: list[_SyncedCandidate],
        by_ref: dict[str, _SyncedCandidate],
    ) -> tuple[_SyncedCandidate, ...]:
        if not candidates:
            return ()
        assert self.embedding_index is not None
        # Embeddings are sha-gated + idempotent; ensure keeps the index fresh
        # even if a card slipped through without an embedding row.
        self.embedding_index.ensure(
            TOOL_CARD_EMBEDDING_KIND,
            [
                (item.card.registry_name, tool_card_embedding_text(item.card))
                for item in candidates
            ],
        )
        ref_keys = [item.card.registry_name for item in candidates]
        ranked = self.embedding_index.rank(
            TOOL_CARD_EMBEDDING_KIND,
            self.task.input,
            ref_keys,
            top_k=self.top_k,
        )
        forced = _explicitly_named_refs(self.task.input, candidates)
        if ranked is None:
            # Embedding rank failed: keep explicit names, then fill by stored
            # order up to top_k so the task still has candidates.
            kept_refs = list(dict.fromkeys(forced))
            for ref in ref_keys:
                if len(kept_refs) >= self.top_k:
                    break
                if ref not in kept_refs:
                    kept_refs.append(ref)
        else:
            kept_refs = [ref for ref, _ in ranked]
            for ref in forced:
                if ref not in kept_refs:
                    kept_refs.append(ref)
        # Intent-grounded reachability floor (HIG-274): the grounded intent
        # classifier names connected toolkits the user implied (toolkit_affinity,
        # e.g. "linear" for "what's on my plate"). Embedding rank against the raw
        # query can bury those toolkits under more lexically-similar ones (a
        # finance-heavy catalog drowned out Linear for task c65e7b2f). Force a
        # connected, intent-named toolkit's best tools in when ranking dropped it.
        if self.forced_toolkits:
            kept_refs = self._apply_intent_toolkit_floor(kept_refs, candidates)
        return tuple(by_ref[ref] for ref in kept_refs if ref in by_ref)

    def _apply_intent_toolkit_floor(
        self,
        kept_refs: list[str],
        candidates: list[_SyncedCandidate],
    ) -> list[str]:
        ref_toolkit = {
            item.card.registry_name: (item.connection.toolkit_slug or "").casefold()
            for item in candidates
        }
        kept = list(kept_refs)
        represented = {ref_toolkit.get(ref) for ref in kept}
        for toolkit in self.forced_toolkits:
            if toolkit in represented:
                continue
            toolkit_refs = [
                item.card.registry_name
                for item in candidates
                if (item.connection.toolkit_slug or "").casefold() == toolkit
            ]
            if not toolkit_refs:
                continue
            for ref in self._rank_within_toolkit(toolkit_refs):
                if ref not in kept:
                    kept.append(ref)
            represented.add(toolkit)
        return kept

    def _rank_within_toolkit(self, refs: list[str]) -> list[str]:
        """Best ``FORCED_TOOLS_PER_INTENT_TOOLKIT`` tools of one toolkit."""

        if self.embedding_index is not None:
            ranked = self.embedding_index.rank(
                TOOL_CARD_EMBEDDING_KIND,
                self.task.input,
                refs,
                top_k=FORCED_TOOLS_PER_INTENT_TOOLKIT,
            )
            if ranked is not None:
                return [ref for ref, _ in ranked]
        return refs[:FORCED_TOOLS_PER_INTENT_TOOLKIT]

    def _load_synced_rows(self, toolkit_slugs: set[str]) -> list[ComposioToolCard]:
        if not toolkit_slugs:
            return []
        return list(
            self.session.scalars(
                select(ComposioToolCard)
                .where(
                    ComposioToolCard.installation_id == self.task.installation_id,
                    ComposioToolCard.toolkit_slug.in_(sorted(toolkit_slugs)),
                )
                .order_by(ComposioToolCard.toolkit_slug, ComposioToolCard.tool_slug)
            )
        )

    def _on_demand_sync(self, toolkit_slug: str) -> None:
        try:
            ComposioCatalogSyncService(
                self.session,
                client=self.client,
                embedding_index=self.embedding_index,
            ).sync_toolkit(self.task.installation_id, toolkit_slug)
        except Exception:
            logger.warning(
                "composio on-demand catalog sync failed installation_id=%s toolkit=%s",
                self.task.installation_id,
                toolkit_slug,
                exc_info=True,
            )

    # --- Degraded (embeddings disabled) path --------------------------------

    def _execute_tool(
        self,
        connection: RuntimeComposioConnection,
        tool: ComposioTool,
    ) -> ComposioExecuteTool:
        return ComposioExecuteTool(
            session=self.session,
            task=self.task,
            client=self.client,
            tool=tool,
            name=composio_runtime_tool_name(connection.toolkit_slug, tool.slug),
            result_max_chars=self.result_max_chars,
        )

    def _load_catalog(self) -> tuple[_ToolkitCatalog, ...]:
        if self._catalog is not None:
            return self._catalog

        entries: list[_ToolkitCatalog] = []
        for connection in _best_connections_by_toolkit(
            self.resolver.allowed_connections()
        ):
            try:
                raw_tools = self.client.list_tools(
                    toolkit_slug=connection.toolkit_slug,
                    query=self.task.input,
                    limit=self.per_toolkit_limit,
                )
                fallback_tools = self.client.list_tools(
                    toolkit_slug=connection.toolkit_slug,
                    limit=self.per_toolkit_limit,
                )
                # HIG-223: surface write tools at all autonomy levels. The
                # approval gate (kortny.approvals) — not this provider — decides
                # whether a write needs explicit approval, so we no longer drop
                # non-read-only tools here.
                allowed_tools = _merge_tools(raw_tools, fallback_tools)
            except ComposioCatalogError as exc:
                log_observation(
                    logger,
                    "composio_catalog_lookup_failed",
                    task=self.task,
                    provider="composio",
                    toolkit_slug=connection.toolkit_slug,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue

            if allowed_tools:
                entries.append(
                    _ToolkitCatalog(connection=connection, tools=allowed_tools)
                )

        self._catalog = tuple(entries)
        log_observation(
            logger,
            "composio_catalog_lookup_completed",
            task=self.task,
            provider="composio",
            toolkit_count=len(self._catalog),
            tool_count=sum(len(entry.tools) for entry in self._catalog),
            toolkit_slugs=[entry.connection.toolkit_slug for entry in self._catalog],
        )
        return self._catalog


def _best_connections_by_toolkit(
    connections: tuple[RuntimeComposioConnection, ...],
) -> tuple[RuntimeComposioConnection, ...]:
    best: dict[str, RuntimeComposioConnection] = {}
    for connection in connections:
        best.setdefault(connection.toolkit_slug, connection)
    return tuple(best.values())


def _merge_tools(
    *tool_groups: tuple[ComposioTool, ...],
) -> tuple[ComposioTool, ...]:
    tools_by_slug: dict[str, ComposioTool] = {}
    for group in tool_groups:
        for tool in group:
            tools_by_slug.setdefault(tool.slug, tool)
    return tuple(tools_by_slug.values())


def _tool_cards(entry: _ToolkitCatalog) -> tuple[ToolCard, ...]:
    return tuple(
        tool_card_for(connection=entry.connection, tool=tool) for tool in entry.tools
    )


def _explicitly_named_refs(
    task_input: str,
    candidates: list[_SyncedCandidate],
) -> list[str]:
    """Force-include candidates whose toolkit the user named verbatim.

    Mirrors the selector's explicit-naming forced-include so the top_k cap
    never drops a tool from an integration the user asked for by name.
    """

    words = _task_words(task_input)
    if not words:
        return []
    forced: list[str] = []
    for item in candidates:
        slug = (item.connection.toolkit_slug or "").casefold()
        if slug and slug in words and item.card.registry_name not in forced:
            forced.append(item.card.registry_name)
    return forced


def _task_words(text: str) -> set[str]:
    return {
        "".join(char for char in raw.casefold() if char.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if raw.strip()
    } - {""}
