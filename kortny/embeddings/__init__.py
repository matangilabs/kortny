"""Local embedding backends and the pgvector-backed semantic index."""

from kortny.embeddings.backends import (
    EmbeddingBackend,
    FastembedBackend,
    create_embedding_backend,
)
from kortny.embeddings.index import EmbeddingIndex
from kortny.embeddings.memory_texts import (
    EPISODE_EMBEDDING_KIND,
    FACT_EMBEDDING_KIND,
    KG_ENTITY_EMBEDDING_KIND,
    episode_embedding_text,
    fact_embedding_text,
    kg_entity_embedding_text,
)
from kortny.embeddings.ranking import (
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    ranked_score,
    recency_decay,
)

__all__ = [
    "DEFAULT_RECENCY_HALF_LIFE_DAYS",
    "EPISODE_EMBEDDING_KIND",
    "EmbeddingBackend",
    "EmbeddingIndex",
    "FACT_EMBEDDING_KIND",
    "FastembedBackend",
    "KG_ENTITY_EMBEDDING_KIND",
    "create_embedding_backend",
    "episode_embedding_text",
    "fact_embedding_text",
    "kg_entity_embedding_text",
    "ranked_score",
    "recency_decay",
]
