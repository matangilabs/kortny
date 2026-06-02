"""Workspace knowledge graph primitives for Kortny."""

from kortny.knowledge_graph.extraction import (
    KG_CHANNEL_PROFILE_PROJECTED_MESSAGE,
    KnowledgeGraphExtractionService,
    KnowledgeGraphProjectionResult,
)
from kortny.knowledge_graph.scopes import (
    DestinationSurface,
    VisibilityScope,
    compatible_scope_predicate,
    is_scope_compatible,
)
from kortny.knowledge_graph.service import (
    EvidenceInput,
    GraphContextPack,
    GraphService,
    GraphStalenessResult,
    RetrievedGraphEdge,
    RetrievedGraphEntity,
)

__all__ = [
    "DestinationSurface",
    "EvidenceInput",
    "GraphContextPack",
    "GraphService",
    "GraphStalenessResult",
    "KG_CHANNEL_PROFILE_PROJECTED_MESSAGE",
    "KnowledgeGraphExtractionService",
    "KnowledgeGraphProjectionResult",
    "RetrievedGraphEdge",
    "RetrievedGraphEntity",
    "VisibilityScope",
    "compatible_scope_predicate",
    "is_scope_compatible",
]
