"""Postgres/pgvector-backed embedding index for tool cards and skills."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from kortny.db.models import ToolEmbedding
from kortny.embeddings.backends import EmbeddingBackend

logger = logging.getLogger(__name__)

_RANK_SQL = text(
    "SELECT ref_key, 1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity "
    "FROM tool_embeddings "
    "WHERE kind = :kind AND model = :model AND ref_key = ANY(:ref_keys) "
    "ORDER BY similarity DESC "
    "LIMIT :top_k"
)


class EmbeddingIndex:
    """Sha-gated upsert + cosine-similarity ranking over ``tool_embeddings``.

    Every public method is failure-isolated: any exception is logged and turned
    into a no-op (``ensure``) or ``None`` (``rank``) so embedding problems can
    never fail a task.
    """

    def __init__(self, session: Session, backend: EmbeddingBackend) -> None:
        self.session = session
        self.backend = backend

    def ensure(self, kind: str, items: Sequence[tuple[str, str]]) -> int:
        """Embed and upsert new/changed items; skip unchanged ones (sha gate).

        Returns the number of items embedded (0 when everything was already
        up to date or on failure).
        """

        try:
            return self._ensure(kind, items)
        except Exception:
            logger.warning(
                "embedding ensure failed kind=%s model=%s item_count=%s",
                kind,
                self.backend.model_name,
                len(items),
                exc_info=True,
            )
            return 0

    def rank(
        self,
        kind: str,
        query_text: str,
        ref_keys: Sequence[str],
        top_k: int,
    ) -> list[tuple[str, float]] | None:
        """Return (ref_key, cosine similarity) pairs, best first, or None on failure."""

        try:
            if not ref_keys or top_k < 1:
                return []
            query_vector = self.backend.embed_query(query_text)
            rows = self.session.execute(
                _RANK_SQL,
                {
                    "query_vector": _vector_literal(query_vector),
                    "kind": kind,
                    "model": self.backend.model_name,
                    "ref_keys": list(ref_keys),
                    "top_k": top_k,
                },
            ).all()
            return [(str(ref_key), float(similarity)) for ref_key, similarity in rows]
        except Exception:
            logger.warning(
                "embedding rank failed kind=%s model=%s candidate_count=%s",
                kind,
                self.backend.model_name,
                len(ref_keys),
                exc_info=True,
            )
            return None

    def _ensure(self, kind: str, items: Sequence[tuple[str, str]]) -> int:
        deduped = dict(items)
        if not deduped:
            return 0

        existing_shas: dict[str, str] = {
            ref_key: content_sha256
            for ref_key, content_sha256 in self.session.execute(
                select(ToolEmbedding.ref_key, ToolEmbedding.content_sha256).where(
                    ToolEmbedding.kind == kind,
                    ToolEmbedding.model == self.backend.model_name,
                    ToolEmbedding.ref_key.in_(deduped),
                )
            ).all()
        }
        changed: list[tuple[str, str, str]] = []
        for ref_key, content in deduped.items():
            sha = _sha256(content)
            if existing_shas.get(ref_key) == sha:
                continue
            changed.append((ref_key, content, sha))
        if not changed:
            return 0

        vectors = self.backend.embed_passages([content for _, content, _ in changed])
        statement = pg_insert(ToolEmbedding).values(
            [
                {
                    "kind": kind,
                    "ref_key": ref_key,
                    "model": self.backend.model_name,
                    "dim": len(vector),
                    "content_sha256": sha,
                    "embedding": vector,
                }
                for (ref_key, _, sha), vector in zip(changed, vectors, strict=True)
            ]
        )
        statement = statement.on_conflict_do_update(
            constraint="uq_tool_embeddings_kind_ref_key_model",
            set_={
                "dim": statement.excluded.dim,
                "content_sha256": statement.excluded.content_sha256,
                "embedding": statement.excluded.embedding,
                "updated_at": text("now()"),
            },
        )
        self.session.execute(statement)
        self.session.flush()
        return len(changed)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"
