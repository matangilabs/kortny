"""Workspace knowledge graph primitives for Kortny.

Keep this package initializer light. Importing `kortny.knowledge_graph.scopes`
must not pull in LLM or tool modules while `kortny.llm` is still initializing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kortny.knowledge_graph.reinforcement import (
    KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE,
    RuntimeGraphReinforcementResult,
    RuntimeGraphReinforcementService,
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

_LAZY_EXPORTS = {
    "GRAPH_REFRESH_HISTORY_LIMIT": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_REFRESH_HISTORY_LOADED_MESSAGE": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_REFRESH_PIPELINE_COMPLETED_MESSAGE": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_REFRESH_PIPELINE_STARTED_MESSAGE": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_REFRESH_SEMANTIC_FALLBACK_MESSAGE": "kortny.knowledge_graph.channel_refresh",
    "ChannelGraphRefreshPipeline": "kortny.knowledge_graph.channel_refresh",
    "ChannelGraphRefreshPipelineResult": "kortny.knowledge_graph.channel_refresh",
    "is_dashboard_graph_refresh_task": "kortny.knowledge_graph.channel_refresh",
    "KG_CHANNEL_PROFILE_PROJECTED_MESSAGE": "kortny.knowledge_graph.extraction",
    "KnowledgeGraphDeterministicProjectionResult": "kortny.knowledge_graph.extraction",
    "KnowledgeGraphExtractionService": "kortny.knowledge_graph.extraction",
    "KnowledgeGraphProjectionResult": "kortny.knowledge_graph.extraction",
    "KG_CHANNEL_REFRESH_REQUESTED_MESSAGE": "kortny.knowledge_graph.refresh",
    "KG_REFRESH_SOURCE": "kortny.knowledge_graph.refresh",
    "KnowledgeGraphRefreshResult": "kortny.knowledge_graph.refresh",
    "KnowledgeGraphRefreshService": "kortny.knowledge_graph.refresh",
}

__all__ = [
    "DestinationSurface",
    "EvidenceInput",
    "GRAPH_REFRESH_HISTORY_LIMIT",
    "GraphContextPack",
    "GraphService",
    "GraphStalenessResult",
    "KG_CHANNEL_REFRESH_HISTORY_LOADED_MESSAGE",
    "KG_CHANNEL_REFRESH_PIPELINE_COMPLETED_MESSAGE",
    "KG_CHANNEL_REFRESH_PIPELINE_STARTED_MESSAGE",
    "KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE",
    "KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE",
    "KG_CHANNEL_REFRESH_SEMANTIC_FALLBACK_MESSAGE",
    "KG_CHANNEL_REFRESH_REQUESTED_MESSAGE",
    "KG_CHANNEL_PROFILE_PROJECTED_MESSAGE",
    "KG_REFRESH_SOURCE",
    "KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE",
    "ChannelGraphRefreshPipeline",
    "ChannelGraphRefreshPipelineResult",
    "KnowledgeGraphRefreshResult",
    "KnowledgeGraphRefreshService",
    "KnowledgeGraphDeterministicProjectionResult",
    "KnowledgeGraphExtractionService",
    "KnowledgeGraphProjectionResult",
    "RetrievedGraphEdge",
    "RetrievedGraphEntity",
    "RuntimeGraphReinforcementResult",
    "RuntimeGraphReinforcementService",
    "VisibilityScope",
    "compatible_scope_predicate",
    "is_dashboard_graph_refresh_task",
    "is_scope_compatible",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
