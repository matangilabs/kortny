"""Normalized provenance helpers for workspace graph rows."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

PROVENANCE_OBSERVED = "observed"
PROVENANCE_EXTRACTED = "extracted"
PROVENANCE_INFERRED = "inferred"
PROVENANCE_AMBIGUOUS = "ambiguous"

VALID_PROVENANCE_KINDS = frozenset(
    {
        PROVENANCE_OBSERVED,
        PROVENANCE_EXTRACTED,
        PROVENANCE_INFERRED,
        PROVENANCE_AMBIGUOUS,
    }
)

OBSERVED_SOURCE_TYPES = frozenset(
    {
        "slack_authoritative",
        "user_explicit",
        "workspace_state",
        "admin_import",
    }
)
EXTRACTED_SOURCE_TYPES = frozenset(
    {
        "onboarding_scan",
        "task_summary",
        "integration_result",
    }
)
INFERRED_SOURCE_TYPES = frozenset({"agent_inferred"})

PROVENANCE_LABELS = {
    PROVENANCE_OBSERVED: "Observed",
    PROVENANCE_EXTRACTED: "Extracted",
    PROVENANCE_INFERRED: "Inferred",
    PROVENANCE_AMBIGUOUS: "Ambiguous",
}


def provenance_kind(source_type: str, attrs_json: dict | None = None) -> str:
    """Return the normalized provenance kind for a graph row."""

    stored_kind = _stored_provenance_kind(attrs_json)
    if stored_kind is not None:
        return stored_kind
    if source_type in OBSERVED_SOURCE_TYPES:
        return PROVENANCE_OBSERVED
    if source_type in EXTRACTED_SOURCE_TYPES:
        return PROVENANCE_EXTRACTED
    if source_type in INFERRED_SOURCE_TYPES:
        return PROVENANCE_INFERRED
    return PROVENANCE_AMBIGUOUS


def provenance_label(kind: str) -> str:
    return PROVENANCE_LABELS.get(kind, PROVENANCE_LABELS[PROVENANCE_AMBIGUOUS])


def review_status(attrs_json: dict | None, lifecycle_state: str) -> str:
    """Return the operator review status for a graph row."""

    if isinstance(attrs_json, dict):
        nested = attrs_json.get("provenance")
        if isinstance(nested, dict):
            stored = nested.get("review_status")
            if isinstance(stored, str) and stored.strip():
                return stored.strip()
        stored = attrs_json.get("review_status")
        if isinstance(stored, str) and stored.strip():
            return stored.strip()
    if lifecycle_state in {"candidate", "stale"}:
        return "needs_review"
    if lifecycle_state == "confirmed":
        return "confirmed"
    if lifecycle_state in {"contradicted", "archived", "forgotten", "superseded"}:
        return lifecycle_state
    return "auto"


def review_reason(attrs_json: dict | None) -> str | None:
    if not isinstance(attrs_json, dict):
        return None
    nested = attrs_json.get("provenance")
    if isinstance(nested, dict):
        reason = nested.get("review_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    reason = attrs_json.get("review_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return None


def with_provenance_attrs(
    attrs_json: dict | None,
    *,
    source_type: str,
    lifecycle_state: str,
    confidence_score: Decimal | float | None = None,
) -> dict:
    """Merge normalized provenance metadata into graph row attrs."""

    attrs = dict(attrs_json or {})
    existing = attrs.get("provenance")
    provenance = dict(existing) if isinstance(existing, dict) else {}
    kind = provenance_kind(source_type, attrs)
    provenance.setdefault("extraction_kind", kind)
    provenance.setdefault("source_type", source_type)
    provenance.setdefault("review_status", review_status(attrs, lifecycle_state))
    reason = review_reason(attrs)
    if reason is not None:
        provenance.setdefault("review_reason", reason)
    if confidence_score is not None:
        provenance.setdefault("confidence_score", str(confidence_score))
    attrs["provenance"] = provenance
    return attrs


def provenance_output(
    *,
    source_type: str,
    lifecycle_state: str,
    attrs_json: dict | None,
) -> dict[str, Any]:
    kind = provenance_kind(source_type, attrs_json)
    return {
        "extraction_kind": kind,
        "label": provenance_label(kind),
        "source_type": source_type,
        "review_status": review_status(attrs_json, lifecycle_state),
        "review_reason": review_reason(attrs_json),
    }


def _stored_provenance_kind(attrs_json: dict | None) -> str | None:
    if not isinstance(attrs_json, dict):
        return None
    nested = attrs_json.get("provenance")
    if isinstance(nested, dict):
        kind = nested.get("extraction_kind")
        if isinstance(kind, str) and kind in VALID_PROVENANCE_KINDS:
            return kind
    kind = attrs_json.get("extraction_kind")
    if isinstance(kind, str) and kind in VALID_PROVENANCE_KINDS:
        return kind
    return None
