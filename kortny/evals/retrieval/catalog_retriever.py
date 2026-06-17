"""Run the CURRENT Composio catalog retriever against the eval set.

This is the increment-1 "measure first" step of find_tools (Linear HIG-269): it
adapts the live embedding-over-synced-cards retriever (the one the provider uses
today) into the `retrieve_fn` seam the eval expects, so we can put a real
Recall@k / nDCG@k number on OUR catalog before building anything. Per ToolRet,
off-the-shelf retrieval is weak and caps agent success, so this baseline tells us
how much query enrichment / re-ranking find_tools actually needs.

Needs the live DB + a real embedding backend; run it in the worker container, not
CI. The pure scoring it feeds (scoring.py) is what CI covers.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.composio.catalog_sync import TOOL_CARD_EMBEDDING_KIND
from kortny.composio.runtime import RuntimeComposioConnection
from kortny.composio.tool_cards import synced_tool_card
from kortny.config import load_settings
from kortny.db.models import ComposioConnection, ComposioToolCard, Installation
from kortny.db.session import make_session_factory
from kortny.embeddings import EmbeddingIndex, create_embedding_backend
from kortny.evals.retrieval.cases import SEED_RETRIEVAL_CASES
from kortny.evals.retrieval.scoring import RetrieveFn, score_retrieval
from kortny.tool_selection import tool_card_embedding_text


def connected_toolkit_slugs_for_installation(
    session: Session, installation_id: object
) -> tuple[str, ...]:
    """Distinct active Composio toolkits for an installation."""

    rows = session.scalars(
        select(ComposioConnection.toolkit_slug)
        .where(
            ComposioConnection.installation_id == installation_id,
            ComposioConnection.status == "active",
            ComposioConnection.connected_account_id.is_not(None),
        )
        .distinct()
    ).all()
    return tuple(sorted({slug for slug in rows if slug}))


def _eval_connection(toolkit_slug: str) -> RuntimeComposioConnection:
    return RuntimeComposioConnection(
        toolkit_slug=toolkit_slug,
        connected_account_id="eval",
        composio_user_id="eval",
        visibility_scope_type="workspace",
        visibility_scope_id=None,
        display_name=None,
    )


def build_catalog_retrieve_fn(
    session: Session,
    *,
    toolkit_slugs: Sequence[str],
    embedding_index: EmbeddingIndex,
) -> RetrieveFn:
    """Build a retrieve_fn that ranks synced tool cards by semantic similarity.

    Mirrors the provider's retrieval (same embedding kind, same registry_name
    ref keys), but ranks per arbitrary query so the eval can drive it. Returns
    tool slugs (best first) to match the eval's ground truth.
    """

    slug_set = {slug for slug in toolkit_slugs if slug}
    rows = list(
        session.scalars(
            select(ComposioToolCard)
            .where(
                ComposioToolCard.toolkit_slug.in_(sorted(slug_set)),
            )
            .order_by(ComposioToolCard.toolkit_slug, ComposioToolCard.tool_slug)
        )
    )
    ref_to_slug: dict[str, str] = {}
    embed_items: list[tuple[str, str]] = []
    for row in rows:
        card = synced_tool_card(
            connection=_eval_connection(row.toolkit_slug),
            tool_slug=row.tool_slug,
            name=row.name,
            description=row.description,
            side_effect=row.side_effect,
        )
        ref_to_slug[card.registry_name] = row.tool_slug
        embed_items.append((card.registry_name, tool_card_embedding_text(card)))

    embedding_index.ensure(TOOL_CARD_EMBEDDING_KIND, embed_items)
    ref_keys = [ref for ref, _ in embed_items]

    def retrieve(query: str) -> list[str]:
        ranked = embedding_index.rank(
            TOOL_CARD_EMBEDDING_KIND, query, ref_keys, top_k=len(ref_keys)
        )
        if ranked is None:
            return []
        return [ref_to_slug[ref] for ref, _ in ranked if ref in ref_to_slug]

    return retrieve


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tool-retrieval baseline eval")
    parser.add_argument(
        "--installation-id",
        default=None,
        help="Installation UUID (defaults to the only installation)",
    )
    args = parser.parse_args()

    settings = load_settings()
    backend = create_embedding_backend(settings)
    if backend is None:
        raise SystemExit("embeddings backend is disabled; cannot run retrieval eval")
    with make_session_factory()() as session:
        if args.installation_id:
            installation_id: object = args.installation_id
        else:
            installs = session.scalars(select(Installation.id)).all()
            if len(installs) != 1:
                raise SystemExit(
                    f"expected exactly 1 installation, found {len(installs)}; "
                    "pass --installation-id"
                )
            installation_id = installs[0]

        toolkits = connected_toolkit_slugs_for_installation(session, installation_id)
        print(f"connected toolkits ({len(toolkits)}): {', '.join(toolkits)}")
        retrieve_fn = build_catalog_retrieve_fn(
            session,
            toolkit_slugs=toolkits,
            embedding_index=EmbeddingIndex(session, backend),
        )
        report = score_retrieval(SEED_RETRIEVAL_CASES, retrieve_fn)

    print(f"\nBASELINE (current catalog retriever): {report.summary_line()}\n")
    for score in report.scores:
        rank = "miss"
        for index, slug in enumerate(score.ranked):
            if slug in set(score.expected):
                rank = f"#{index + 1} ({slug})"
                break
        print(f"  [{'HIT ' if score.hit else 'MISS'}] rank={rank:<28} {score.query}")


if __name__ == "__main__":
    _main()
