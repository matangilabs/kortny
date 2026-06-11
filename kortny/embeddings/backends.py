"""Embedding backends for the capability fabric (HIG-219).

The only real backend in this slice is :class:`FastembedBackend`, which runs a
small ONNX model locally via ``fastembed``. The fastembed import lives lazily
inside the backend so importing this module (or constructing the backend) never
pulls in onnxruntime or downloads a model — important for tests, which use a
deterministic fake backend instead.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from kortny.config import Settings

logger = logging.getLogger(__name__)

# One loaded fastembed model per process, keyed by model name. Model load is
# expensive (ONNX session + tokenizer), so the worker reuses it across tasks.
_FASTEMBED_MODELS: dict[str, Any] = {}
_UNAVAILABLE_WARNED = False

# fastembed's default internal batch is 256 documents. ONNX materializes the
# attention tensors for the whole batch (batch x heads x seq^2 floats), which
# peaks at ~2.4GB for a few hundred long passages — enough to get the process
# OOM-killed on a small Docker VM. 16 passages bounds the peak under ~200MB.
PASSAGE_EMBED_BATCH_SIZE = 16


class EmbeddingBackend(Protocol):
    """Minimal embedding contract used by the EmbeddingIndex."""

    model_name: str

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents/passages."""


class FastembedBackend:
    """Local fastembed (ONNX) text-embedding backend.

    ``query_embed``/``passage_embed`` are used (not plain ``embed``) so models
    with asymmetric prefixes — e.g. bge's "query:"/"passage:" handling — embed
    queries and documents correctly.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _model(self) -> Any:
        cached = _FASTEMBED_MODELS.get(self.model_name)
        if cached is None:
            from fastembed import TextEmbedding  # lazy: heavy import + model load

            cached = TextEmbedding(model_name=self.model_name)
            _FASTEMBED_MODELS[self.model_name] = cached
        return cached

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model().query_embed([text])))
        return [float(value) for value in vector]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(value) for value in vector]
            for vector in self._model().passage_embed(
                list(texts), batch_size=PASSAGE_EMBED_BATCH_SIZE
            )
        ]


def create_embedding_backend(settings: Settings) -> EmbeddingBackend | None:
    """Return the configured embedding backend, or None when unavailable.

    Returns None when the backend is disabled or fastembed is not importable.
    Embedding must never fail a task, so callers treat None as "skip semantic
    retrieval and fall back to lexical behavior".
    """

    global _UNAVAILABLE_WARNED
    if settings.embeddings_backend == "disabled":
        return None
    try:
        if importlib.util.find_spec("fastembed") is None:
            raise ImportError("fastembed is not installed")
        return FastembedBackend(settings.embeddings_model)
    except Exception:
        if not _UNAVAILABLE_WARNED:
            _UNAVAILABLE_WARNED = True
            logger.warning(
                "embedding backend unavailable; semantic retrieval disabled",
                exc_info=True,
            )
        return None
