"""Full-catalog Composio embedding sync (HIG-222).

Pulls each *connected* toolkit's FULL tool list from Composio (paginated, not
query-pruned), builds tool cards, persists them in ``composio_tool_cards``, and
embeds them into ``tool_embeddings`` (kind ``tool_card``) so per-task tool
retrieval is a pure semantic rank with no hot-path Composio HTTP for candidate
listing.

Tombstoning: a tool that vanished from a toolkit, or whose toolkit is no longer
connected, has its card row and embedding row deleted (embeddings by
``ref_key``). The sync is idempotent and sha-gated end to end: re-running with
unchanged descriptions re-embeds nothing.

Triggers (both wire into this module):
* periodic reconcile — the ``composio_catalog_sync`` ambient loop, ~6h.
* on-demand — the provider calls :meth:`sync_toolkit` for a single connected
  toolkit with zero synced cards, so a fresh connection works within one task.

Capability profiling (enriched_description per tool card) is handled by a
separate background loop (``CapabilityProfilerWorker`` in
``kortny.integration_learning.profiler_worker``).  The sync path is now a pure
card-upsert + embed pipeline: no LLM calls, no blocking on profiling.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from kortny.composio.client import (
    ComposioCatalogError,
    ComposioClient,
    ComposioRateLimitError,
    ComposioTool,
)
from kortny.composio.tool_cards import card_sha, side_effect_for_tool
from kortny.config import Settings, load_settings
from kortny.db.models import (
    ComposioConnection,
    ComposioToolCard,
    Installation,
)
from kortny.db.session import make_session_factory
from kortny.embeddings import EmbeddingIndex, create_embedding_backend
from kortny.tool_selection import tool_card_embedding_text
from kortny.tools.composio_execute import composio_runtime_tool_name
from kortny.tools.pinning import ToolPinService, compute_tool_fingerprint

logger = logging.getLogger(__name__)

TOOL_CARD_EMBEDDING_KIND = "tool_card"
EMBED_CHUNK_SIZE = 100
MAX_PAGES_PER_TOOLKIT = 50
DEFAULT_RATE_LIMIT_RETRIES = 3
DEFAULT_SYNC_ADVISORY_LOCK_KEY = 759340222
DEFAULT_SYNC_INTERVAL_SECONDS = 6 * 60 * 60.0


@dataclass(frozen=True, slots=True)
class ToolkitSyncResult:
    """Outcome of syncing one toolkit's catalog for one installation."""

    toolkit_slug: str
    tool_count: int
    upserted: int
    embedded: int
    tombstoned: int


@dataclass(frozen=True, slots=True)
class InstallationSyncResult:
    """Outcome of syncing every connected toolkit for one installation."""

    installation_id: object
    toolkits: tuple[ToolkitSyncResult, ...]
    disconnected_tombstoned: int


class ComposioCatalogSyncService:
    """Sync connected toolkit catalogs into cards + embeddings.

    This service is now a pure card-upsert + embed pipeline.  Capability
    profiling (enriched_description, KG entities) runs in the separate
    ``CapabilityProfilerWorker`` background loop.
    """

    def __init__(
        self,
        session: Session,
        *,
        client: ComposioClient,
        embedding_index: EmbeddingIndex | None,
        page_size: int = 20,
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        sleep: object = time.sleep,
    ) -> None:
        self.session = session
        self.client = client
        self.embedding_index = embedding_index
        self.page_size = max(1, page_size)
        self.rate_limit_retries = max(0, rate_limit_retries)
        self._sleep = sleep

    def connected_toolkits(self, installation_id: object) -> tuple[str, ...]:
        """Distinct active, connected toolkit slugs for an installation."""

        rows = self.session.scalars(
            select(ComposioConnection.toolkit_slug)
            .where(
                ComposioConnection.installation_id == installation_id,
                ComposioConnection.status == "active",
                # No-auth toolkits are "connected" without a connected account
                # (their tools run directly) — include them or their tools never
                # get synced into the catalog and stay invisible to retrieval.
                or_(
                    ComposioConnection.no_auth.is_(True),
                    ComposioConnection.connected_account_id.is_not(None),
                ),
            )
            .distinct()
        )
        return tuple(sorted({slug for slug in rows if slug}))

    def sync_installation(self, installation_id: object) -> InstallationSyncResult:
        """Sync every connected toolkit and tombstone disconnected ones."""

        connected = set(self.connected_toolkits(installation_id))
        results: list[ToolkitSyncResult] = []
        for toolkit_slug in sorted(connected):
            try:
                results.append(self.sync_toolkit(installation_id, toolkit_slug))
            except ComposioCatalogError as exc:
                logger.warning(
                    "composio catalog sync failed installation_id=%s toolkit=%s "
                    "error=%s",
                    installation_id,
                    toolkit_slug,
                    exc,
                )
        disconnected = self._tombstone_disconnected_toolkits(
            installation_id, connected=connected
        )
        return InstallationSyncResult(
            installation_id=installation_id,
            toolkits=tuple(results),
            disconnected_tombstoned=disconnected,
        )

    def sync_toolkit(
        self,
        installation_id: object,
        toolkit_slug: str,
    ) -> ToolkitSyncResult:
        """Pull the full catalog for one toolkit and reconcile cards/embeddings.

        Bounded: one toolkit. Used both by the periodic reconcile (per toolkit)
        and the provider's on-demand sync of a freshly connected toolkit.

        No LLM calls are made here. Capability profiling (enriched_description)
        is handled by the separate CapabilityProfilerWorker background loop.
        """

        tools = self._fetch_full_toolkit(toolkit_slug)
        upserted = self._upsert_cards(
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            tools=tools,
        )
        tombstoned = self._tombstone_removed_tools(
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            present_tool_slugs={tool.slug for tool in tools},
        )
        embedded = self._embed_cards(
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
        )
        logger.info(
            "composio catalog synced installation_id=%s toolkit=%s tools=%s "
            "upserted=%s embedded=%s tombstoned=%s",
            installation_id,
            toolkit_slug,
            len(tools),
            upserted,
            embedded,
            tombstoned,
        )
        return ToolkitSyncResult(
            toolkit_slug=toolkit_slug,
            tool_count=len(tools),
            upserted=upserted,
            embedded=embedded,
            tombstoned=tombstoned,
        )

    def _fetch_full_toolkit(self, toolkit_slug: str) -> tuple[ComposioTool, ...]:
        collected: dict[str, ComposioTool] = {}
        cursor: str | None = None
        for _ in range(MAX_PAGES_PER_TOOLKIT):
            page, cursor = self._fetch_page(toolkit_slug, cursor)
            for tool in page:
                collected.setdefault(tool.slug, tool)
            if not cursor or not page:
                break
        return tuple(collected.values())

    def _fetch_page(
        self,
        toolkit_slug: str,
        cursor: str | None,
    ) -> tuple[tuple[ComposioTool, ...], str | None]:
        attempt = 0
        while True:
            try:
                return self.client.list_tools_page(
                    toolkit_slug=toolkit_slug,
                    limit=self.page_size,
                    cursor=cursor,
                )
            except ComposioRateLimitError:
                if attempt >= self.rate_limit_retries:
                    raise
                backoff = float(2**attempt)
                logger.info(
                    "composio rate limited toolkit=%s; retrying in %.1fs "
                    "(attempt %s/%s)",
                    toolkit_slug,
                    backoff,
                    attempt + 1,
                    self.rate_limit_retries,
                )
                self._sleep(backoff)  # type: ignore[operator]
                attempt += 1

    def _upsert_cards(
        self,
        *,
        installation_id: object,
        toolkit_slug: str,
        tools: Sequence[ComposioTool],
    ) -> int:
        if not tools:
            return 0
        existing_rows = self.session.execute(
            select(
                ComposioToolCard.tool_slug,
                ComposioToolCard.card_sha,
                ComposioToolCard.input_schema_json,
            ).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
            )
        ).all()
        existing: dict[str, str] = {
            row.tool_slug: row.card_sha for row in existing_rows
        }
        existing_schemas: dict[str, dict[str, object]] = {
            row.tool_slug: (row.input_schema_json or {}) for row in existing_rows
        }
        # HIG-169 P0.3: drift-check every tool on every refresh — NOT only the
        # ones whose card_sha changed. card_sha omits inputSchema, so a
        # silent inputSchema rug-pull would otherwise pass the card_sha gate
        # unseen. The fingerprint includes inputSchema and pins independently.
        self._pin_composio_tools(
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            tools=tools,
        )
        rows: list[dict[str, object]] = []
        for tool in tools:
            side_effect = side_effect_for_tool(tool)
            description = tool.description or tool.name
            sha = card_sha(
                name=tool.name,
                description=description,
                side_effect=side_effect,
            )
            # Skip only if sha matches AND we already have a non-empty schema
            # cached. This ensures schema backfill on rows that pre-date the
            # input_schema_json column.
            if existing.get(tool.slug) == sha and existing_schemas.get(tool.slug):
                continue
            rows.append(
                {
                    "installation_id": installation_id,
                    "toolkit_slug": toolkit_slug,
                    "tool_slug": tool.slug,
                    "name": tool.name,
                    "description": description,
                    "side_effect": side_effect,
                    "card_sha": sha,
                    "input_schema_json": tool.input_parameters or {},
                }
            )
        if not rows:
            return 0
        statement = pg_insert(ComposioToolCard).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="uq_composio_tool_cards_installation_tool",
            set_={
                "toolkit_slug": statement.excluded.toolkit_slug,
                "name": statement.excluded.name,
                "description": statement.excluded.description,
                "side_effect": statement.excluded.side_effect,
                "card_sha": statement.excluded.card_sha,
                "input_schema_json": statement.excluded.input_schema_json,
                "synced_at": statement.excluded.synced_at,
            },
        )
        self.session.execute(statement)
        self.session.flush()
        return len(rows)

    def _pin_composio_tools(
        self,
        *,
        installation_id: object,
        toolkit_slug: str,
        tools: Sequence[ComposioTool],
    ) -> None:
        """Pin each Composio tool's full-schema fingerprint; flag drift.

        Composio is a single admin-connected provider trusted at the provider
        level, so the threat here is drift, not a rogue server: a toolkit
        silently changing a tool's ``input_parameters`` after approval. Pinning
        catches that. Failures never fail the sync.
        """

        if not isinstance(installation_id, uuid.UUID):
            return
        pin_service = ToolPinService(self.session)
        for tool in tools:
            try:
                fingerprint = compute_tool_fingerprint(
                    name=tool.name,
                    description=tool.description or tool.name,
                    input_schema=tool.input_parameters,
                )
                result = pin_service.check_and_pin(
                    installation_id=installation_id,
                    provider="composio",
                    server_ref=toolkit_slug,
                    tool_name=tool.slug,
                    fingerprint=fingerprint,
                )
                if result.drifted:
                    logger.warning(
                        "composio_tool_schema_drift toolkit=%s tool=%s "
                        "prior_fingerprint=%s new_fingerprint=%s",
                        toolkit_slug,
                        tool.slug,
                        result.prior_fingerprint,
                        result.fingerprint,
                    )
            except Exception:
                logger.exception(
                    "composio_tool_pin_failed toolkit=%s tool=%s",
                    toolkit_slug,
                    tool.slug,
                )

    def _embed_cards(
        self,
        *,
        installation_id: object,
        toolkit_slug: str,
    ) -> int:
        if self.embedding_index is None:
            return 0
        cards = self.session.execute(
            select(
                ComposioToolCard.toolkit_slug,
                ComposioToolCard.tool_slug,
                ComposioToolCard.name,
                ComposioToolCard.description,
                ComposioToolCard.side_effect,
                ComposioToolCard.enriched_description,
            ).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
            )
        ).all()
        items = [
            (
                composio_runtime_tool_name(row.toolkit_slug, row.tool_slug),
                _embedding_text(
                    toolkit_slug=row.toolkit_slug,
                    tool_slug=row.tool_slug,
                    name=row.name,
                    description=row.description,
                    side_effect=row.side_effect,
                    enriched_description=getattr(row, "enriched_description", None),
                ),
            )
            for row in cards
        ]
        embedded = 0
        for start in range(0, len(items), EMBED_CHUNK_SIZE):
            chunk = items[start : start + EMBED_CHUNK_SIZE]
            embedded += self.embedding_index.ensure(TOOL_CARD_EMBEDDING_KIND, chunk)
        return embedded

    def _tombstone_removed_tools(
        self,
        *,
        installation_id: object,
        toolkit_slug: str,
        present_tool_slugs: set[str],
    ) -> int:
        stored = self.session.execute(
            select(ComposioToolCard.tool_slug).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
            )
        ).all()
        removed = [
            row.tool_slug for row in stored if row.tool_slug not in present_tool_slugs
        ]
        return self._delete_cards(
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            tool_slugs=removed,
        )

    def _tombstone_disconnected_toolkits(
        self,
        installation_id: object,
        *,
        connected: set[str],
    ) -> int:
        stored_toolkits = set(
            self.session.scalars(
                select(ComposioToolCard.toolkit_slug)
                .where(ComposioToolCard.installation_id == installation_id)
                .distinct()
            )
        )
        deleted = 0
        for toolkit_slug in sorted(stored_toolkits - connected):
            tool_slugs = list(
                self.session.scalars(
                    select(ComposioToolCard.tool_slug).where(
                        ComposioToolCard.installation_id == installation_id,
                        ComposioToolCard.toolkit_slug == toolkit_slug,
                    )
                )
            )
            deleted += self._delete_cards(
                installation_id=installation_id,
                toolkit_slug=toolkit_slug,
                tool_slugs=tool_slugs,
            )
        return deleted

    def _delete_cards(
        self,
        *,
        installation_id: object,
        toolkit_slug: str,
        tool_slugs: Sequence[str],
    ) -> int:
        if not tool_slugs:
            return 0
        ref_keys = [
            composio_runtime_tool_name(toolkit_slug, tool_slug)
            for tool_slug in tool_slugs
        ]
        self.session.execute(
            delete(ComposioToolCard).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
                ComposioToolCard.tool_slug.in_(list(tool_slugs)),
            )
        )
        self.session.flush()
        if self.embedding_index is not None:
            self.embedding_index.delete(TOOL_CARD_EMBEDDING_KIND, ref_keys)
        return len(tool_slugs)


def _embedding_text(
    *,
    toolkit_slug: str,
    tool_slug: str,
    name: str,
    description: str,
    side_effect: str,
    enriched_description: str | None = None,
) -> str:
    """Embedding text for a synced card, matching the runtime card's text.

    Mirrors ``tool_card_embedding_text`` over the same ToolCard fields the
    provider builds at selection time so the sha gate and ranking stay
    consistent across the sync and the hot path.

    When ``enriched_description`` is provided (HIG-295), it is injected onto
    the card so ``tool_card_embedding_text`` prefers it over the raw description.
    """

    from dataclasses import replace as _replace

    from kortny.composio.runtime import RuntimeComposioConnection
    from kortny.composio.tool_cards import synced_tool_card

    card = synced_tool_card(
        connection=RuntimeComposioConnection(
            toolkit_slug=toolkit_slug,
            connected_account_id="",
            composio_user_id="",
            visibility_scope_type="workspace",
            visibility_scope_id=None,
            display_name=None,
        ),
        tool_slug=tool_slug,
        name=name,
        description=description,
        side_effect=side_effect,
    )
    if enriched_description:
        card = _replace(card, enriched_description=enriched_description)
    return tool_card_embedding_text(card)


class ComposioCatalogSyncWorker:
    """Poll-loop that reconciles every installation's Composio catalog.

    Mirrors the consolidator worker: a dedicated connection per tick, an
    advisory lock for single-leader execution, and a ``run_forever`` body the
    ambient supervisor hosts as a supervised thread.

    Capability profiling is now handled by the separate CapabilityProfilerWorker;
    this worker is a pure sync + embed pipeline with no LLM calls.
    """

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        settings: Settings | None = None,
        client: ComposioClient | None = None,
        poll_interval_seconds: float | None = None,
        advisory_lock_key: int | None = None,
        use_advisory_lock: bool = True,
    ) -> None:
        self.session_factory = session_factory or make_session_factory()
        self.settings = settings
        self._client = client
        self.poll_interval_seconds = poll_interval_seconds or (
            settings.composio_sync_interval_hours * 3600.0
            if settings is not None
            else DEFAULT_SYNC_INTERVAL_SECONDS
        )
        self.advisory_lock_key = advisory_lock_key or (
            settings.composio_sync_advisory_lock_key
            if settings is not None
            else DEFAULT_SYNC_ADVISORY_LOCK_KEY
        )
        self.use_advisory_lock = use_advisory_lock

    def run_once(self) -> tuple[InstallationSyncResult, ...]:
        engine = self.session_factory.kw["bind"]
        with (
            engine.connect() as connection,
            Session(bind=connection, expire_on_commit=False) as session,
        ):
            if self.use_advisory_lock and not self._try_advisory_lock(session):
                return ()
            try:
                results = self._sync_all(session)
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                if self.use_advisory_lock:
                    self._release_advisory_lock(session)
            return results

    def _sync_all(self, session: Session) -> tuple[InstallationSyncResult, ...]:
        service = ComposioCatalogSyncService(
            session,
            client=self._resolve_client(),
            embedding_index=self._embedding_index(session),
            page_size=self._settings.composio_sync_page_size,
        )
        results: list[InstallationSyncResult] = []
        for installation_id in session.scalars(
            select(Installation.id).order_by(Installation.created_at)
        ):
            # HIG-209 Part 3: cheap pre-pass — requeue any connect-parked tasks
            # whose toolkit is now connected in scope, then sync that toolkit so
            # the resumed task's tool resolves on re-run.
            self._resume_parked_connect_tasks(session, service, installation_id)
            try:
                results.append(service.sync_installation(installation_id))
            except Exception:
                logger.exception(
                    "composio catalog sync failed installation_id=%s",
                    installation_id,
                )
        return tuple(results)

    def _resume_parked_connect_tasks(
        self,
        session: Session,
        service: ComposioCatalogSyncService,
        installation_id: object,
    ) -> None:
        from kortny.composio.connect import resume_parked_connect_tasks

        try:
            resume = resume_parked_connect_tasks(
                session, installation_id=installation_id
            )
        except Exception:
            logger.exception(
                "composio connect resume failed installation_id=%s",
                installation_id,
            )
            return
        for toolkit_slug in resume.resumed_toolkits:
            try:
                service.sync_toolkit(installation_id, toolkit_slug)
            except Exception:
                logger.exception(
                    "composio connect resume toolkit sync failed "
                    "installation_id=%s toolkit=%s",
                    installation_id,
                    toolkit_slug,
                )

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("composio catalog sync tick failed; continuing")
            time.sleep(self.poll_interval_seconds)

    def _embedding_index(self, session: Session) -> EmbeddingIndex | None:
        backend = create_embedding_backend(self._settings)
        if backend is None:
            return None
        return EmbeddingIndex(session, backend)

    def _resolve_client(self) -> ComposioClient:
        if self._client is not None:
            return self._client
        self._client = ComposioClient(
            api_key=self._settings.composio_api_key,
            timeout_seconds=self._settings.composio_request_timeout_seconds,
        )
        return self._client

    def _try_advisory_lock(self, session: Session) -> bool:
        return bool(
            session.scalar(select(func.pg_try_advisory_lock(self.advisory_lock_key)))
        )

    def _release_advisory_lock(self, session: Session) -> None:
        session.execute(select(func.pg_advisory_unlock(self.advisory_lock_key)))

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings
