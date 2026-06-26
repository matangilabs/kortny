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
from collections.abc import Callable, Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from kortny.composio.catalog_sync import TOOL_CARD_EMBEDDING_KIND
from kortny.composio.runtime import RuntimeComposioConnection
from kortny.composio.tool_cards import synced_tool_card
from kortny.config import load_settings
from kortny.db.models import ComposioConnection, ComposioToolCard, Installation
from kortny.db.session import make_session_factory
from kortny.embeddings import EmbeddingIndex, create_embedding_backend
from kortny.evals.retrieval.cases import SEED_RETRIEVAL_CASES
from kortny.evals.retrieval.scoring import (
    RetrievalReport,
    RetrievalScore,
    RetrieveFn,
    ndcg_at_k,
    recall_at_k,
    score_retrieval,
)
from kortny.tool_selection import ToolCard, tool_card_embedding_text


def connected_toolkit_slugs_for_installation(
    session: Session, installation_id: object
) -> tuple[str, ...]:
    """Distinct active Composio toolkits for an installation."""

    rows = session.scalars(
        select(ComposioConnection.toolkit_slug)
        .where(
            ComposioConnection.installation_id == installation_id,
            ComposioConnection.status == "active",
            # Include no-auth toolkits (connected, no account) so their tools are
            # retrievable by find_tools / the prewarm.
            or_(
                ComposioConnection.no_auth.is_(True),
                ComposioConnection.connected_account_id.is_not(None),
            ),
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


# Score blending weights for hybrid retrieval (HIG-269 increment 3).
LEXICAL_WEIGHT = 0.3
GROUNDING_BOOST = 0.5

# A query rewriter (raw user/agent text -> a tool-capability description).
QueryEnricher = Callable[[str], str]


def _words(text: str) -> set[str]:
    return {
        "".join(ch for ch in raw.casefold() if ch.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if raw.strip()
    } - {""}


def build_catalog_retrieve_fn(
    session: Session,
    *,
    toolkit_slugs: Sequence[str],
    embedding_index: EmbeddingIndex,
    enrich: QueryEnricher | None = None,
    boost_toolkits: frozenset[str] = frozenset(),
    extra_cards: Sequence[ToolCard] = (),
) -> RetrieveFn:
    """Build a retrieve_fn that ranks synced tool cards for a query.

    Mirrors the provider's retrieval (same embedding kind, same registry_name
    ref keys) but adds the increment-3 quality levers:
    - ``enrich``: rewrite the query into a tool-capability description before
      embedding (ToolRet's low-lexical-overlap fix). Identity when None.
    - hybrid lexical blend: add term overlap with the card text to the semantic
      score so exact toolkit/verb mentions rank up.
    - ``boost_toolkits``: the grounding prior (intent toolkit_affinity) gives a
      score bonus to those connected toolkits' tools.
    - ``extra_cards``: non-Composio tool cards (e.g. MCP) ranked in the SAME
      index so find_tools surfaces every connected provider, not just Composio
      (HIG-269). Their ``registry_name`` doubles as the returned slug, so the
      caller's loader dispatches them by name.
    Returns tool slugs (best first) to match the eval's ground truth.
    """

    slug_set = {slug for slug in toolkit_slugs if slug}
    boost = {slug.casefold() for slug in boost_toolkits if slug}
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
    ref_to_toolkit: dict[str, str] = {}
    ref_to_words: dict[str, set[str]] = {}
    embed_items: list[tuple[str, str]] = []
    for row in rows:
        card = synced_tool_card(
            connection=_eval_connection(row.toolkit_slug),
            tool_slug=row.tool_slug,
            name=row.name,
            description=row.description,
            side_effect=row.side_effect,
        )
        text = tool_card_embedding_text(card)
        ref_to_slug[card.registry_name] = row.tool_slug
        ref_to_toolkit[card.registry_name] = row.toolkit_slug.casefold()
        ref_to_words[card.registry_name] = _words(text)
        embed_items.append((card.registry_name, text))

    for card in extra_cards:
        text = tool_card_embedding_text(card)
        # Non-Composio providers (MCP) are loaded by runtime name, so the slug
        # the retriever returns IS the registry_name.
        ref_to_slug[card.registry_name] = card.registry_name
        ref_to_toolkit[card.registry_name] = (card.toolkit_slug or "").casefold()
        ref_to_words[card.registry_name] = _words(text)
        embed_items.append((card.registry_name, text))

    embedding_index.ensure(TOOL_CARD_EMBEDDING_KIND, embed_items)
    ref_keys = [ref for ref, _ in embed_items]

    def retrieve(query: str) -> list[str]:
        embed_query = enrich(query) if enrich is not None else query
        ranked = embedding_index.rank(
            TOOL_CARD_EMBEDDING_KIND, embed_query, ref_keys, top_k=len(ref_keys)
        )
        if ranked is None:
            return []
        query_words = _words(query) | _words(embed_query)
        rescored: list[tuple[float, str]] = []
        for ref, semantic in ranked:
            card_words = ref_to_words.get(ref, set())
            overlap = len(query_words & card_words)
            lexical = min(1.0, overlap * 0.1)
            score = semantic + LEXICAL_WEIGHT * lexical
            if ref_to_toolkit.get(ref) in boost:
                score += GROUNDING_BOOST
            rescored.append((score, ref))
        rescored.sort(key=lambda item: -item[0])
        return [ref_to_slug[ref] for _, ref in rescored if ref in ref_to_slug]

    return retrieve


def _grounded_report(
    session: Session,
    *,
    toolkits: Sequence[str],
    embedding_index: EmbeddingIndex,
) -> RetrievalReport:
    """Score the improved retriever (hybrid + per-case grounding boost).

    Each case is retrieved with its own implied-toolkit boost, mirroring what
    the grounded intent supplies to find_tools at runtime.
    """

    scores: list[RetrievalScore] = []
    ks = (1, 3, 5, 10)
    for case in SEED_RETRIEVAL_CASES:
        retrieve_fn = build_catalog_retrieve_fn(
            session,
            toolkit_slugs=toolkits,
            embedding_index=embedding_index,
            boost_toolkits=frozenset(case.implies_toolkits),
        )
        ranked = tuple(retrieve_fn(case.query))
        scores.append(
            RetrievalScore(
                query=case.query,
                expected=case.expected_tool_slugs,
                ranked=ranked,
                recall_at_k={
                    k: recall_at_k(ranked, case.expected_tool_slugs, k) for k in ks
                },
                ndcg_at_k={
                    k: ndcg_at_k(ranked, case.expected_tool_slugs, k) for k in ks
                },
            )
        )
    n = len(scores)
    return RetrievalReport(
        case_count=n,
        ks=ks,
        mean_recall_at_k={k: sum(s.recall_at_k[k] for s in scores) / n for k in ks},
        mean_ndcg_at_k={k: sum(s.ndcg_at_k[k] for s in scores) / n for k in ks},
        hit_rate=sum(1 for s in scores if s.hit) / n,
        scores=tuple(scores),
    )


def _print_report(label: str, report: RetrievalReport) -> None:
    print(f"\n{label}: {report.summary_line()}\n")
    for score in report.scores:
        rank = "miss"
        for index, slug in enumerate(score.ranked):
            if slug in set(score.expected):
                rank = f"#{index + 1} ({slug})"
                break
        print(f"  [{'HIT ' if score.hit else 'MISS'}] rank={rank:<32} {score.query}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tool-retrieval eval")
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
        index = EmbeddingIndex(session, backend)
        baseline = score_retrieval(
            SEED_RETRIEVAL_CASES,
            build_catalog_retrieve_fn(
                session, toolkit_slugs=toolkits, embedding_index=index
            ),
        )
        improved = _grounded_report(session, toolkits=toolkits, embedding_index=index)

    _print_report("BASELINE (hybrid only)", baseline)
    _print_report("IMPROVED (hybrid + grounding boost)", improved)


if __name__ == "__main__":
    _main()
