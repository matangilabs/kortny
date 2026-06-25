"""On-connect CapabilityProfile builder (HIG-295 Step A).

One cheap JSON LLM pass per toolkit produces clean enriched_description text for
each tool card (the #1 retrieval-quality lever per Gorilla/EASYTOOL research) and
a workspace-scoped KG entity (entity_type="integration") carrying the app-level
profile summary, capability buckets, and cross-app affinity hints.

The profiler runs inside sync_toolkit() AFTER _upsert_cards() so enriched
descriptions are written before _embed_cards() sees them.  Re-embedding happens
automatically because _embed_cards() is called after the profile step.

Backfill: call backfill_capability_profiles(session, installation_id) or run
python -m kortny.integration_learning.backfill to populate existing connections.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from kortny.db.models import ComposioToolCard, KnowledgeGraphEntity
from kortny.knowledge_graph.scopes import VisibilityScope
from kortny.knowledge_graph.service import EvidenceInput, GraphService
from kortny.llm import ChatMessage, LLMService

logger = logging.getLogger(__name__)

CAPABILITY_PROFILER_PROMPT_NAME = "kortny.integration_learning.capability_profiler"
_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

# Batch size for the LLM profiling pass.  Each batch is one LLM call so that
# large toolkits (e.g. 97-tool twelve_data) are fully covered instead of
# silently dropping the tail.
_PROFILER_BATCH_SIZE = 30

# Confidence for capability_profile KG entities — seeded on-connect so we
# start at active not candidate (profile is immediately useful).
_PROFILE_CONFIDENCE = Decimal("0.700")


@dataclass(frozen=True)
class CapabilityProfile:
    """Parsed output of the LLM profiling pass."""

    summary: str
    capability_buckets: list[str]
    per_tool: list[dict[str, str]]  # [{tool_slug, enriched_description}, ...]
    cross_app_affinity_hints: list[str]


def build_capability_profile(
    session: Session,
    *,
    installation_id: uuid.UUID,
    toolkit_slug: str,
    llm: LLMService,
    task_id: uuid.UUID,
    toolkit_metadata: dict[str, Any] | None = None,
) -> CapabilityProfile | None:
    """Run batched cheap LLM passes to profile a toolkit and write enriched descriptions.

    Iterates over all tool cards in batches of _PROFILER_BATCH_SIZE so that
    large toolkits (e.g. 97 tools) are fully covered.  App-level fields
    (summary, capability_buckets, cross_app_affinity_hints) come from the first
    successfully-parsed batch; per_tool entries are merged across all batches.

    Returns the parsed profile, or None if there are no cards or every batch
    fails (failures are logged, never propagate — caller must not hard-fail).

    Side effects:
    - Writes ``enriched_description`` on each ``ComposioToolCard`` row.
    - Creates or upserts a KG entity with entity_type="integration" and the
      app-level profile summary in attrs_json.
    """

    cards = list(
        session.execute(
            select(
                ComposioToolCard.tool_slug,
                ComposioToolCard.name,
                ComposioToolCard.description,
                ComposioToolCard.side_effect,
                ComposioToolCard.input_schema_json,
            ).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
            )
        ).all()
    )
    if not cards:
        logger.debug(
            "capability_profiler no cards toolkit=%s installation_id=%s",
            toolkit_slug,
            installation_id,
        )
        return None

    # Build toolkit metadata dict once — shared across all batch payloads.
    meta: dict[str, Any] = {"toolkit_slug": toolkit_slug}
    if toolkit_metadata:
        meta.update(
            {
                k: toolkit_metadata.get(k)
                for k in ("name", "description", "categories", "auth_schemes")
                if toolkit_metadata.get(k)
            }
        )

    system_prompt = (
        "You are a tool-description enrichment engine. "
        "Given a Composio toolkit's metadata and its raw tool list, produce a "
        "clean, accurate capability profile as JSON. "
        "Rules: (1) Never invent tools or capabilities not in the input. "
        "(2) enriched_description must be one crisp sentence that names the "
        "tool's DISTINCTIVE capability — include the canonical measurement name "
        "or action concept in the first few words so retrieval can distinguish "
        "similar tools (e.g. 'Fetch Average True Range (ATR) indicator values "
        "for a symbol over a time range' NOT 'Get technical indicator data'). "
        "(3) No em dashes. Use commas, colons, or parentheses instead. "
        "(4) summary: 2-3 sentences, what the app does and for whom. "
        "(5) capability_buckets: 3-8 short phrases (e.g. 'historical OHLCV bars', "
        "'technical indicators', 'real-time quotes'). "
        "(6) cross_app_affinity_hints: 0-4 short phrases naming complementary "
        "apps (e.g. 'pairs well with Alpaca for automated trading'). "
        "Return ONLY JSON matching exactly: "
        '{"summary": "...", "capability_buckets": [...], '
        '"per_tool": [{"tool_slug": "...", "enriched_description": "..."}, ...], '
        '"cross_app_affinity_hints": [...]}'
    )

    parsed: CapabilityProfile | None = None
    enriched_by_slug: dict[str, str] = {}

    for start in range(0, len(cards), _PROFILER_BATCH_SIZE):
        batch = cards[start : start + _PROFILER_BATCH_SIZE]
        batch_tool_list = [
            {
                "tool_slug": row.tool_slug,
                "name": row.name,
                "description": row.description or row.name,
                "side_effect": row.side_effect,
                "required_fields": [
                    k
                    for k, v in (row.input_schema_json or {})
                    .get("properties", {})
                    .items()
                    if k in (row.input_schema_json or {}).get("required", [])
                ],
            }
            for row in batch
        ]
        payload_json = json.dumps(
            {"toolkit": meta, "tools": batch_tool_list},
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        try:
            completion = llm.complete(
                task_id=task_id,
                messages=(
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=payload_json),
                ),
                response_format=_RESPONSE_FORMAT,
                prompt_name=CAPABILITY_PROFILER_PROMPT_NAME,
            )
            batch_parsed = _parse_profile(
                completion.content or "", toolkit_slug=toolkit_slug
            )
        except Exception as exc:
            logger.warning(
                "capability_profiler batch failed toolkit=%s batch_start=%s error=%s",
                toolkit_slug,
                start,
                exc,
            )
            continue
        if batch_parsed is None:
            logger.warning(
                "capability_profiler batch parse failed toolkit=%s batch_start=%s",
                toolkit_slug,
                start,
            )
            continue
        if parsed is None:
            parsed = batch_parsed  # first successful batch provides app-level fields
        for item in batch_parsed.per_tool:
            slug = item.get("tool_slug")
            desc = item.get("enriched_description")
            if slug and desc:
                enriched_by_slug[slug] = desc

    # Persist enriched_description onto each card.
    if enriched_by_slug:
        _write_enriched_descriptions(
            session,
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            enriched_by_slug=enriched_by_slug,
        )

    if parsed is None:
        return None

    # Upsert the KG capability_profile entity.
    _upsert_kg_profile_entity(
        session,
        installation_id=installation_id,
        toolkit_slug=toolkit_slug,
        task_id=task_id,
        profile=parsed,
    )

    logger.info(
        "capability_profiler done toolkit=%s tools=%s enriched=%s",
        toolkit_slug,
        len(cards),
        len(enriched_by_slug),
    )
    return parsed


def _parse_profile(content: str, *, toolkit_slug: str) -> CapabilityProfile | None:
    """Parse the LLM JSON response into a CapabilityProfile."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "capability_profiler bad json toolkit=%s error=%s", toolkit_slug, exc
        )
        return None

    summary = str(data.get("summary") or "").strip()
    if not summary:
        logger.warning("capability_profiler empty summary toolkit=%s", toolkit_slug)
        return None

    capability_buckets = [
        str(b).strip()
        for b in (data.get("capability_buckets") or [])
        if b and str(b).strip()
    ]
    cross_app_affinity_hints = [
        str(h).strip()
        for h in (data.get("cross_app_affinity_hints") or [])
        if h and str(h).strip()
    ]
    per_tool: list[dict[str, str]] = []
    for item in data.get("per_tool") or []:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("tool_slug") or "").strip()
        desc = str(item.get("enriched_description") or "").strip()
        if slug and desc:
            per_tool.append({"tool_slug": slug, "enriched_description": desc})

    return CapabilityProfile(
        summary=summary,
        capability_buckets=capability_buckets,
        per_tool=per_tool,
        cross_app_affinity_hints=cross_app_affinity_hints,
    )


def _write_enriched_descriptions(
    session: Session,
    *,
    installation_id: uuid.UUID,
    toolkit_slug: str,
    enriched_by_slug: dict[str, str],
) -> None:
    """Bulk-update enriched_description for each card."""
    for tool_slug, enriched in enriched_by_slug.items():
        session.execute(
            update(ComposioToolCard)
            .where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
                ComposioToolCard.tool_slug == tool_slug,
            )
            .values(enriched_description=enriched)
        )
    session.flush()


def _upsert_kg_profile_entity(
    session: Session,
    *,
    installation_id: uuid.UUID,
    toolkit_slug: str,
    task_id: uuid.UUID,
    profile: CapabilityProfile,
) -> None:
    """Create or replace the KG capability_profile entity for a toolkit."""
    canonical_key = f"composio_app:{toolkit_slug}"

    existing = session.scalars(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == installation_id,
            KnowledgeGraphEntity.canonical_key == canonical_key,
        )
    ).first()

    attrs: dict[str, Any] = {
        "kind": "capability_profile",
        "summary": profile.summary,
        "capability_buckets": profile.capability_buckets,
        "cross_app_affinity_hints": profile.cross_app_affinity_hints,
    }

    if existing is not None:
        existing.attrs_json = attrs
        existing.lifecycle_state = "active"
        existing.confidence_score = _PROFILE_CONFIDENCE
        session.flush()
        return

    graph = GraphService(session)
    graph.create_entity(
        installation_id=installation_id,
        entity_type="integration",
        canonical_key=canonical_key,
        display_name=toolkit_slug,
        visibility_scope=VisibilityScope.workspace(),
        source_type="onboarding_scan",
        attrs_json=attrs,
        lifecycle_state="active",
        confidence_score=_PROFILE_CONFIDENCE,
        evidence=EvidenceInput(
            source_type="onboarding_scan",
            extracted_by=CAPABILITY_PROFILER_PROMPT_NAME,
            source_task_id=task_id,
            source_url=f"composio://toolkit/{toolkit_slug}",
        ),
    )
