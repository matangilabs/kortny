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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select
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
from kortny.db.models import ComposioConnection, ComposioToolCard, Installation
from kortny.db.session import make_session_factory
from kortny.embeddings import EmbeddingIndex, create_embedding_backend
from kortny.tool_selection import tool_card_embedding_text
from kortny.tools.composio_execute import composio_runtime_tool_name

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
    """Sync connected toolkit catalogs into cards + embeddings."""

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
                ComposioConnection.connected_account_id.is_not(None),
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
        existing: dict[str, str] = {
            tool_slug: sha
            for tool_slug, sha in self.session.execute(
                select(ComposioToolCard.tool_slug, ComposioToolCard.card_sha).where(
                    ComposioToolCard.installation_id == installation_id,
                    ComposioToolCard.toolkit_slug == toolkit_slug,
                )
            ).all()
        }
        rows: list[dict[str, object]] = []
        for tool in tools:
            side_effect = side_effect_for_tool(tool)
            description = tool.description or tool.name
            sha = card_sha(
                name=tool.name,
                description=description,
                side_effect=side_effect,
            )
            if existing.get(tool.slug) == sha:
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
                "synced_at": statement.excluded.synced_at,
            },
        )
        self.session.execute(statement)
        self.session.flush()
        return len(rows)

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
) -> str:
    """Embedding text for a synced card, matching the runtime card's text.

    Mirrors ``tool_card_embedding_text`` over the same ToolCard fields the
    provider builds at selection time so the sha gate and ranking stay
    consistent across the sync and the hot path.
    """

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
    return tool_card_embedding_text(card)


class ComposioCatalogSyncWorker:
    """Poll-loop that reconciles every installation's Composio catalog.

    Mirrors the consolidator worker: a dedicated connection per tick, an
    advisory lock for single-leader execution, and a ``run_forever`` body the
    ambient supervisor hosts as a supervised thread.
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
            try:
                results.append(service.sync_installation(installation_id))
            except Exception:
                logger.exception(
                    "composio catalog sync failed installation_id=%s",
                    installation_id,
                )
        return tuple(results)

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
