"""Runtime adapters for DB-managed LLM model configuration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.llm.litellm_provider import create_litellm_provider
from kortny.llm.provider_config import (
    ModelConfigService,
    ResolvedLLMModel,
    ResolvedLLMModelChain,
    secret_resolver_from_settings,
)
from kortny.llm.routing import ModelRoute
from kortny.llm.types import LLMProvider


@dataclass(frozen=True, slots=True)
class RuntimeModelSelection:
    """Resolved model/provider data for one task-bound model route."""

    model_route: ModelRoute
    chain: ResolvedLLMModelChain
    model: ResolvedLLMModel
    provider_name: DbLLMProvider

    @property
    def event_payload(self) -> dict[str, object]:
        return {
            "model_config_source": self.chain.source,
            "model_config_fallback_reason": self.chain.fallback_reason,
            "model_config_skipped_candidate_count": self.chain.skipped_candidate_count,
            "provider_kind": self.model.provider_kind,
            "provider_account_id": str(self.model.provider_account_id)
            if self.model.provider_account_id is not None
            else None,
            "model_catalog_id": str(self.model.model_catalog_id)
            if self.model.model_catalog_id is not None
            else None,
            "tier_assignment_id": str(self.model.tier_assignment_id)
            if self.model.tier_assignment_id is not None
            else None,
            "credential_source": self.model.credential_source,
        }


def select_runtime_model(
    *,
    session: Session,
    settings: Settings,
    installation_id: uuid.UUID,
    model_route: ModelRoute,
    model_config_service: ModelConfigService | None = None,
) -> RuntimeModelSelection:
    """Resolve one internal model route through DB config plus env fallback."""

    service = model_config_service or ModelConfigService(
        session,
        settings=settings,
        secret_resolver=secret_resolver_from_settings(session, settings=settings),
    )
    chain = service.resolve_model_chain(
        installation_id=installation_id,
        tier=model_route.tier,
    )
    model = chain.primary
    return RuntimeModelSelection(
        model_route=ModelRoute(
            tier=model_route.tier,
            model=model.model,
            reason=model_route.reason,
        ),
        chain=chain,
        model=model,
        provider_name=db_provider_name(
            model.provider_kind,
            fallback=settings.llm_provider.value,
        ),
    )


def create_provider_for_selection(
    *,
    settings: Settings,
    selection: RuntimeModelSelection,
) -> LLMProvider:
    """Create the direct LLM provider for a resolved runtime model."""

    return create_litellm_provider(
        settings,
        **selection.model.litellm_provider_kwargs,
    )


def db_provider_name(provider_kind: str, *, fallback: str) -> DbLLMProvider:
    """Map string provider identity to the legacy usage enum while it exists."""

    try:
        return DbLLMProvider(provider_kind)
    except ValueError:
        return DbLLMProvider(fallback)
