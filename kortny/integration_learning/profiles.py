"""On-connect CapabilityProfile builder (HIG-295 Step A).

One cheap JSON LLM pass per toolkit produces clean enriched_description text for
each tool card (the #1 retrieval-quality lever per Gorilla/EASYTOOL research) and
a workspace-scoped KG entity (entity_type="integration") carrying the app-level
profile summary, capability buckets, and cross-app affinity hints.

The profiler now runs in a dedicated background loop (CapabilityProfilerWorker)
rather than inline inside sync_toolkit().  build_capability_profile() accepts an
optional max_tools budget so the background loop can process toolkits
incrementally across ticks without stalling the sync path.

Backfill: call backfill_capability_profiles(session, installation_id) or run
python -m kortny.integration_learning.backfill to populate existing connections.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypedDict

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

# Per-cycle card budget for the background profiler loop.  The loop processes at
# most this many unenriched tools per tick (spread across all pending toolkits).
# Deferred upgrade: usage-ranked tool prioritization (process the most-retrieved
# tools first within a toolkit).
_DEFAULT_PROFILE_CARD_BUDGET = 120

# Confidence for capability_profile KG entities — seeded on-connect so we
# start at active not candidate (profile is immediately useful).
_PROFILE_CONFIDENCE = Decimal("0.700")


class ProfileResult(TypedDict):
    """Counts returned by build_capability_profile when max_tools is used."""

    processed: int
    remaining_unenriched: int


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
    max_tools: int | None = None,
) -> CapabilityProfile | None:
    """Run batched cheap LLM passes to profile a toolkit and write enriched descriptions.

    When ``max_tools`` is set (background loop mode):
    - Only cards lacking ``enriched_description`` are considered.
    - At most ``max_tools`` unenriched cards are processed this call.
    - The app-level KG entity (summary/buckets) is written on the first run
      (when no profile entity exists yet); subsequent top-up runs skip that step
      if the entity already exists and only add per-tool enrichment.
    - Once ALL unenriched cards have been enriched (remaining_unenriched == 0),
      the ``generated_from.card_sha_digest`` is stamped on the KG entity so the
      stale-gate skips this toolkit on the next cycle.

    When ``max_tools`` is None (legacy/backfill mode), the full set of cards is
    processed as before — app-level fields are always written.

    Returns the parsed profile from the first successful batch (app-level
    fields), or None if there are no cards to process or every batch fails.

    Side effects:
    - Writes ``enriched_description`` on each ``ComposioToolCard`` row processed.
    - Creates or upserts a KG entity with entity_type="integration" and the
      app-level profile summary in attrs_json (first run / when no entity exists).
    - When all cards are enriched, stamps card_sha_digest on the KG entity.
    """

    # Fetch all cards to compute counts and the digest for gate stamping.
    all_card_rows = list(
        session.execute(
            select(
                ComposioToolCard.tool_slug,
                ComposioToolCard.name,
                ComposioToolCard.description,
                ComposioToolCard.side_effect,
                ComposioToolCard.input_schema_json,
                ComposioToolCard.card_sha,
                ComposioToolCard.enriched_description,
            ).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
            )
        ).all()
    )
    if not all_card_rows:
        logger.debug(
            "capability_profiler no cards toolkit=%s installation_id=%s",
            toolkit_slug,
            installation_id,
        )
        return None

    if max_tools is not None:
        # Background loop mode: only process unenriched cards, capped to budget.
        unenriched = [r for r in all_card_rows if r.enriched_description is None]
        cards_to_process = unenriched[:max_tools]
    else:
        # Legacy / backfill mode: process all cards.
        unenriched = [r for r in all_card_rows if r.enriched_description is None]
        cards_to_process = list(all_card_rows)

    if not cards_to_process:
        logger.debug(
            "capability_profiler nothing to process toolkit=%s installation_id=%s",
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

    # Determine whether a KG entity exists already (for first-run detection).
    canonical_key = f"composio_app:{toolkit_slug}"
    existing_entity = session.scalars(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == installation_id,
            KnowledgeGraphEntity.canonical_key == canonical_key,
        )
    ).first()
    profile_entity_exists = existing_entity is not None

    for start in range(0, len(cards_to_process), _PROFILER_BATCH_SIZE):
        batch = cards_to_process[start : start + _PROFILER_BATCH_SIZE]
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

    # In max_tools mode, write app-level KG entity only on the first run (when
    # no entity exists yet).  Subsequent top-up runs skip this to avoid
    # overwriting a previously-written summary with a partial batch summary.
    should_write_kg = max_tools is None or not profile_entity_exists

    if should_write_kg:
        _upsert_kg_profile_entity(
            session,
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            task_id=task_id,
            profile=parsed,
        )

    # When all cards are now enriched, stamp the card_sha_digest on the KG
    # entity so the stale-gate skips this toolkit on the next cycle.
    if max_tools is not None:
        newly_enriched = set(enriched_by_slug.keys())
        # Count remaining unenriched after this pass: unenriched cards that were
        # not processed this call (tail beyond budget) and processed cards that
        # the LLM failed to return an enriched_description for.
        processed_slugs = {r.tool_slug for r in cards_to_process}
        tail_unenriched = [r for r in unenriched if r.tool_slug not in processed_slugs]
        processed_but_missing = [
            r for r in cards_to_process if r.tool_slug not in newly_enriched
        ]
        remaining_unenriched = len(tail_unenriched) + len(processed_but_missing)

        if remaining_unenriched == 0:
            _stamp_digest(
                session,
                installation_id=installation_id,
                toolkit_slug=toolkit_slug,
                all_card_rows=all_card_rows,
                newly_enriched=newly_enriched,
                canonical_key=canonical_key,
            )

    logger.info(
        "capability_profiler done toolkit=%s tools=%s enriched=%s",
        toolkit_slug,
        len(all_card_rows),
        len(enriched_by_slug),
    )
    return parsed


def _stamp_digest(
    session: Session,
    *,
    installation_id: uuid.UUID,
    toolkit_slug: str,
    all_card_rows: list[Any],
    newly_enriched: set[str],
    canonical_key: str,
) -> None:
    """Stamp the card_sha_digest on the KG entity once all cards are enriched."""
    sorted_shas = "".join(
        row.card_sha for row in sorted(all_card_rows, key=lambda r: r.tool_slug)
    )
    card_sha_digest = hashlib.sha256(sorted_shas.encode()).hexdigest()

    stamped_entity = session.scalars(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == installation_id,
            KnowledgeGraphEntity.canonical_key == canonical_key,
        )
    ).first()
    if stamped_entity is not None:
        attrs = dict(stamped_entity.attrs_json or {})
        attrs["generated_from"] = {"card_sha_digest": card_sha_digest}
        stamped_entity.attrs_json = attrs
        session.flush()
        logger.debug(
            "capability_profiler digest stamped toolkit=%s digest=%s",
            toolkit_slug,
            card_sha_digest,
        )


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
