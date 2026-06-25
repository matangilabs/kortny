"""Backfill capability profiles for all connected toolkits (HIG-295 Step A).

Run once post-deploy to populate existing connections:
    python -m kortny.integration_learning.backfill

Idempotent and resumable: toolkits that already have enriched descriptions are
re-profiled (profiles improve idempotently; no harm in re-running).
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from kortny.composio.client import ComposioClient
    from kortny.llm import LLMService

logger = logging.getLogger(__name__)


def backfill_capability_profiles(
    session: Session,
    installation_id: uuid.UUID,
    *,
    llm: LLMService | None = None,
    task_id: uuid.UUID | None = None,
    client: ComposioClient | None = None,
) -> None:
    """Profile + re-embed all connected toolkits for one installation.

    ``llm`` and ``task_id`` are required for the LLM profile pass. When not
    provided (e.g. called from a standalone script), they are constructed from
    settings. ``client`` overrides the Composio HTTP client (useful in tests).
    """

    from kortny.composio.catalog_sync import ComposioCatalogSyncService
    from kortny.composio.client import ComposioClient as _ComposioClient
    from kortny.config import load_settings
    from kortny.embeddings import EmbeddingIndex, create_embedding_backend
    from kortny.integration_learning.profiles import build_capability_profile

    settings = load_settings()

    resolved_client: ComposioClient
    if client is None:
        resolved_client = _ComposioClient(
            api_key=settings.composio_api_key,
            timeout_seconds=settings.composio_request_timeout_seconds,
        )
    else:
        resolved_client = client

    embedding_backend = create_embedding_backend(settings)
    embedding_index = (
        EmbeddingIndex(session, embedding_backend)
        if embedding_backend is not None
        else None
    )

    sync_service = ComposioCatalogSyncService(
        session,
        client=resolved_client,
        embedding_index=embedding_index,
    )
    toolkit_slugs = sync_service.connected_toolkits(installation_id)

    if llm is None or task_id is None:
        logger.warning(
            "backfill_capability_profiles: llm/task_id not provided; "
            "skipping LLM profile pass for installation_id=%s",
            installation_id,
        )
        # Still re-embed to pick up any existing enriched_description values.
        for slug in toolkit_slugs:
            sync_service._embed_cards(  # noqa: SLF001
                installation_id=installation_id,
                toolkit_slug=slug,
            )
        return

    for slug in toolkit_slugs:
        try:
            toolkit_meta: dict[str, Any] | None = None
            try:
                tk = resolved_client.get_toolkit(slug)
                toolkit_meta = {
                    "name": tk.name,
                    "description": tk.description,
                    "categories": list(tk.categories),
                    "auth_schemes": list(tk.auth_schemes),
                }
            except Exception as exc:
                logger.debug(
                    "backfill get_toolkit failed toolkit=%s error=%s", slug, exc
                )

            build_capability_profile(
                session,
                installation_id=installation_id,
                toolkit_slug=slug,
                llm=llm,
                task_id=task_id,
                toolkit_metadata=toolkit_meta,
            )
            # Re-embed so embeddings pick up the new enriched_description.
            sync_service._embed_cards(  # noqa: SLF001
                installation_id=installation_id,
                toolkit_slug=slug,
            )
            session.flush()
        except Exception as exc:
            logger.warning("backfill failed toolkit=%s error=%s", slug, exc)


if __name__ == "__main__":
    import sys

    from sqlalchemy import select as _select

    from kortny.config import load_settings
    from kortny.db.models import Installation
    from kortny.db.session import make_session_factory

    logging.basicConfig(level=logging.INFO)
    _settings = load_settings()
    _session_factory = make_session_factory()

    with _session_factory() as _session:
        _installation_ids = list(_session.scalars(_select(Installation.id)))

    if not _installation_ids:
        print("No installations found.", file=sys.stderr)
        sys.exit(0)

    for _inst_id in _installation_ids:
        logger.info("backfilling installation_id=%s", _inst_id)
        with _session_factory() as _session:
            backfill_capability_profiles(_session, _inst_id)
            _session.commit()
