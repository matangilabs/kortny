"""Sleep-time memory consolidation service (HIG-225).

The consolidator is the single background brain that promotes task episodes
into durable graph knowledge, adjudicates stuck candidates, merges duplicates,
ages stale rows, reconciles user-confirmed facts into the graph, and runs
retention hygiene — all on the cheap LLM tier, batched, off the request path.
"""

from kortny.consolidator.service import (
    CONSOLIDATOR_EXTRACTOR,
    ConsolidationOutcome,
    ConsolidationService,
)
from kortny.consolidator.trigger import TriggerDecision, evaluate_trigger

__all__ = [
    "CONSOLIDATOR_EXTRACTOR",
    "ConsolidationOutcome",
    "ConsolidationService",
    "TriggerDecision",
    "evaluate_trigger",
]
