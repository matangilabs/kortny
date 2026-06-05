"""DB-backed LLM provider configuration helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from types import MappingProxyType
from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import (
    EncryptedSecret,
    LLMConfigAudit,
    LLMModelCatalog,
    LLMProviderAccount,
    LLMTierAssignment,
)
from kortny.llm.routing import ModelRouter, ModelRouteTier
from kortny.secrets import decrypt_secret_value

ENV_BOOTSTRAP_SOURCE = "env_bootstrap"
ENV_CREDENTIAL_SOURCE = "env"
SECRET_CREDENTIAL_SOURCE = "encrypted_secret"
ResolutionSource = Literal["db", "env_bootstrap", "env_fallback", "env_forced"]
SecretResolver = Callable[[uuid.UUID], str]
CONFIG_TIERS: tuple[ModelRouteTier, ...] = (
    ModelRouteTier.cheap_fast,
    ModelRouteTier.standard,
    ModelRouteTier.analysis,
    ModelRouteTier.document,
    ModelRouteTier.high_reasoning,
    ModelRouteTier.humanizer,
)


class LLMModelConfigError(RuntimeError):
    """Raised when model configuration cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class LLMProviderBootstrapResult:
    """Result from an env-to-DB provider configuration bootstrap attempt."""

    created: bool
    skipped_reason: str | None
    provider_account_id: uuid.UUID | None
    model_count: int
    tier_assignment_count: int


@dataclass(frozen=True, slots=True)
class ResolvedLLMModel:
    """Provider/model config ready for ADK or direct provider construction."""

    tier: ModelRouteTier
    provider_kind: str
    model: str
    api_key: str
    provider_account_id: uuid.UUID | None
    model_catalog_id: uuid.UUID | None
    tier_assignment_id: uuid.UUID | None
    priority: int
    credential_source: str
    base_url: str | None = None
    api_version: str | None = None
    extra_headers: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def litellm_model(self) -> str:
        """Return the model name shape ADK/LiteLLM expects."""

        if self.provider_kind == "openrouter" and not self.model.startswith(
            "openrouter/"
        ):
            return f"openrouter/{self.model}"
        return self.model

    @property
    def adk_litellm_kwargs(self) -> dict[str, object]:
        """Return constructor kwargs for ADK's `LiteLlm` wrapper."""

        kwargs: dict[str, object] = {
            "model": self.litellm_model,
            "api_key": self.api_key,
        }
        if self.base_url is not None:
            kwargs["api_base"] = self.base_url
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if self.extra_headers:
            kwargs["extra_headers"] = dict(self.extra_headers)
        return kwargs

    @property
    def direct_provider_kwargs(self) -> dict[str, object]:
        """Return kwargs for Kortny's direct provider adapters."""

        return self.litellm_provider_kwargs

    @property
    def litellm_provider_kwargs(self) -> dict[str, object]:
        """Return kwargs for Kortny's direct LiteLLM provider adapter."""

        kwargs: dict[str, object] = {
            "api_key": self.api_key,
            "model": self.litellm_model,
        }
        if self.base_url is not None:
            kwargs["api_base"] = self.base_url
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if self.extra_headers:
            kwargs["extra_headers"] = dict(self.extra_headers)
        return kwargs


@dataclass(frozen=True, slots=True)
class ResolvedLLMModelChain:
    """Ordered provider/model candidates for an internal model tier."""

    installation_id: uuid.UUID
    tier: ModelRouteTier
    source: ResolutionSource
    models: tuple[ResolvedLLMModel, ...]
    fallback_reason: str | None = None
    skipped_candidate_count: int = 0

    @property
    def primary(self) -> ResolvedLLMModel:
        if not self.models:
            raise LLMModelConfigError("Model resolution produced no candidates")
        return self.models[0]


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    chain: ResolvedLLMModelChain


class ModelConfigService:
    """Resolve DB-managed model-tier config with deterministic env fallback."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings,
        secret_resolver: SecretResolver | None = None,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        self.session = session
        self.settings = settings
        self.secret_resolver = secret_resolver
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[tuple[uuid.UUID, str, bool], _CacheEntry] = {}

    def resolve_model(
        self,
        *,
        installation_id: uuid.UUID,
        tier: ModelRouteTier,
    ) -> ResolvedLLMModel:
        """Return the primary resolved model for a tier."""

        return self.resolve_model_chain(
            installation_id=installation_id,
            tier=tier,
        ).primary

    def resolve_model_chain(
        self,
        *,
        installation_id: uuid.UUID,
        tier: ModelRouteTier,
    ) -> ResolvedLLMModelChain:
        """Return ordered DB model candidates, bootstrapping/falling back to env."""

        cache_key = (installation_id, tier.value, self.settings.llm_config_force_env)
        cached = self._cache.get(cache_key)
        now = monotonic()
        if cached is not None and cached.expires_at > now:
            return cached.chain

        chain = self._resolve_model_chain_uncached(
            installation_id=installation_id,
            tier=tier,
        )
        if self.cache_ttl_seconds > 0:
            self._cache[cache_key] = _CacheEntry(
                expires_at=now + self.cache_ttl_seconds,
                chain=chain,
            )
        return chain

    def invalidate_cache(
        self,
        *,
        installation_id: uuid.UUID | None = None,
        tier: ModelRouteTier | None = None,
    ) -> None:
        """Invalidate cached resolutions after provider config changes."""

        if installation_id is None and tier is None:
            self._cache.clear()
            return

        for key in list(self._cache):
            key_installation_id, key_tier, _force_env = key
            if installation_id is not None and key_installation_id != installation_id:
                continue
            if tier is not None and key_tier != tier.value:
                continue
            self._cache.pop(key, None)

    def _resolve_model_chain_uncached(
        self,
        *,
        installation_id: uuid.UUID,
        tier: ModelRouteTier,
    ) -> ResolvedLLMModelChain:
        if self.settings.llm_config_force_env:
            return self._env_chain(
                installation_id=installation_id,
                tier=tier,
                source="env_forced",
                fallback_reason="force_env_enabled",
            )

        rows = self._active_db_rows(installation_id=installation_id, tier=tier)
        if not rows:
            bootstrap = bootstrap_llm_provider_config_from_env(
                self.session,
                installation_id=installation_id,
                settings=self.settings,
            )
            rows = self._active_db_rows(installation_id=installation_id, tier=tier)
            if rows:
                return self._db_chain(
                    installation_id=installation_id,
                    tier=tier,
                    rows=rows,
                    source="env_bootstrap" if bootstrap.created else "db",
                )
            return self._env_chain(
                installation_id=installation_id,
                tier=tier,
                source="env_fallback",
                fallback_reason=bootstrap.skipped_reason or "db_tier_missing",
            )

        return self._db_chain(
            installation_id=installation_id,
            tier=tier,
            rows=rows,
            source="db",
        )

    def _active_db_rows(
        self,
        *,
        installation_id: uuid.UUID,
        tier: ModelRouteTier,
    ) -> list[tuple[LLMTierAssignment, LLMModelCatalog, LLMProviderAccount]]:
        statement = (
            select(LLMTierAssignment, LLMModelCatalog, LLMProviderAccount)
            .join(
                LLMModelCatalog,
                LLMTierAssignment.model_catalog_id == LLMModelCatalog.id,
            )
            .join(
                LLMProviderAccount,
                LLMModelCatalog.provider_account_id == LLMProviderAccount.id,
            )
            .where(
                LLMTierAssignment.installation_id == installation_id,
                LLMTierAssignment.tier == tier.value,
                LLMTierAssignment.is_active.is_(True),
                LLMModelCatalog.is_enabled.is_(True),
                LLMProviderAccount.status == "active",
            )
            .order_by(
                LLMTierAssignment.priority.asc(), LLMTierAssignment.created_at.asc()
            )
        )
        return [
            (assignment, catalog, provider)
            for assignment, catalog, provider in self.session.execute(statement).all()
        ]

    def _db_chain(
        self,
        *,
        installation_id: uuid.UUID,
        tier: ModelRouteTier,
        rows: list[tuple[LLMTierAssignment, LLMModelCatalog, LLMProviderAccount]],
        source: ResolutionSource,
    ) -> ResolvedLLMModelChain:
        skipped_count = 0
        models: list[ResolvedLLMModel] = []
        for assignment, catalog, provider in rows:
            try:
                api_key = self._resolve_api_key(provider)
            except Exception:
                api_key = None
            if api_key is None:
                skipped_count += 1
                continue
            models.append(
                ResolvedLLMModel(
                    tier=tier,
                    provider_kind=provider.provider_kind,
                    model=catalog.model_identifier,
                    api_key=api_key,
                    provider_account_id=provider.id,
                    model_catalog_id=catalog.id,
                    tier_assignment_id=assignment.id,
                    priority=assignment.priority,
                    credential_source=self._credential_source(provider),
                    base_url=provider.base_url,
                    api_version=_api_version_from_metadata(provider.metadata_json),
                    extra_headers=_extra_headers_from_metadata(provider.metadata_json),
                )
            )

        if models:
            return ResolvedLLMModelChain(
                installation_id=installation_id,
                tier=tier,
                source=source,
                models=tuple(models),
                skipped_candidate_count=skipped_count,
            )

        return self._env_chain(
            installation_id=installation_id,
            tier=tier,
            source="env_fallback",
            fallback_reason="db_candidates_missing_credentials",
            skipped_candidate_count=skipped_count,
        )

    def _env_chain(
        self,
        *,
        installation_id: uuid.UUID,
        tier: ModelRouteTier,
        source: ResolutionSource,
        fallback_reason: str | None,
        skipped_candidate_count: int = 0,
    ) -> ResolvedLLMModelChain:
        route = ModelRouter(self.settings).route_for_tier(tier, reason="env_fallback")
        return ResolvedLLMModelChain(
            installation_id=installation_id,
            tier=tier,
            source=source,
            fallback_reason=fallback_reason,
            skipped_candidate_count=skipped_candidate_count,
            models=(
                ResolvedLLMModel(
                    tier=tier,
                    provider_kind=self.settings.llm_provider.value,
                    model=route.model,
                    api_key=self.settings.llm_api_key,
                    provider_account_id=None,
                    model_catalog_id=None,
                    tier_assignment_id=None,
                    priority=1,
                    credential_source=ENV_CREDENTIAL_SOURCE,
                ),
            ),
        )

    def _resolve_api_key(self, provider: LLMProviderAccount) -> str | None:
        credential_source = self._credential_source(provider)
        if credential_source == ENV_CREDENTIAL_SOURCE:
            api_key: str = self.settings.llm_api_key
            return api_key
        if credential_source == SECRET_CREDENTIAL_SOURCE:
            if provider.encrypted_secret_id is None or self.secret_resolver is None:
                return None
            return self.secret_resolver(provider.encrypted_secret_id)
        return None

    def _credential_source(self, provider: LLMProviderAccount) -> str:
        value = provider.metadata_json.get("credential_source")
        if isinstance(value, str) and value:
            return value
        if provider.encrypted_secret_id is not None:
            return SECRET_CREDENTIAL_SOURCE
        return ENV_CREDENTIAL_SOURCE


def bootstrap_llm_provider_config_from_env(
    session: Session,
    *,
    installation_id: uuid.UUID,
    settings: Settings,
    actor_slack_user_id: str | None = None,
) -> LLMProviderBootstrapResult:
    """Seed provider/model/tier config from env when no DB config exists.

    This intentionally records only that credentials come from env. It does not
    copy the API key into the database; dashboard-managed secrets need the
    dedicated secret service planned for the next slice.
    """

    if settings.llm_config_force_env:
        return LLMProviderBootstrapResult(
            created=False,
            skipped_reason="force_env_enabled",
            provider_account_id=None,
            model_count=0,
            tier_assignment_count=0,
        )

    existing = session.scalar(
        select(LLMProviderAccount.id)
        .where(LLMProviderAccount.installation_id == installation_id)
        .limit(1)
    )
    if existing is not None:
        return LLMProviderBootstrapResult(
            created=False,
            skipped_reason="existing_provider_config",
            provider_account_id=existing,
            model_count=0,
            tier_assignment_count=0,
        )

    provider_kind = settings.llm_provider.value
    seeded_at = datetime.now(UTC).isoformat()
    routes = [
        ModelRouter(settings).route_for_tier(tier, reason="env_bootstrap")
        for tier in CONFIG_TIERS
    ]
    model_tiers: dict[str, list[str]] = {}
    for route in routes:
        model_tiers.setdefault(route.model, []).append(route.tier.value)

    provider = LLMProviderAccount(
        installation_id=installation_id,
        provider_kind=provider_kind,
        display_name=f"{provider_kind.title()} env provider",
        status="active",
        health_status="unknown",
        encrypted_secret_id=None,
        metadata_json={
            "credential_source": ENV_CREDENTIAL_SOURCE,
            "seeded_from_env": True,
            "seeded_at": seeded_at,
        },
    )
    session.add(provider)
    session.flush()

    catalog_by_model: dict[str, LLMModelCatalog] = {}
    for model_identifier, tiers in model_tiers.items():
        catalog = LLMModelCatalog(
            provider_account_id=provider.id,
            model_identifier=model_identifier,
            display_name=model_identifier,
            is_enabled=True,
            capabilities_json={},
            source=ENV_BOOTSTRAP_SOURCE,
            metadata_json={
                "credential_source": ENV_CREDENTIAL_SOURCE,
                "env_tiers": tiers,
                "seeded_from_env": True,
            },
        )
        session.add(catalog)
        catalog_by_model[model_identifier] = catalog
    session.flush()

    assignments: list[LLMTierAssignment] = []
    for route in routes:
        assignment = LLMTierAssignment(
            installation_id=installation_id,
            tier=route.tier.value,
            model_catalog_id=catalog_by_model[route.model].id,
            priority=1,
            is_active=True,
            routing_json={
                "source": ENV_BOOTSTRAP_SOURCE,
                "reason": route.reason,
            },
        )
        session.add(assignment)
        assignments.append(assignment)

    audit = LLMConfigAudit(
        installation_id=installation_id,
        actor_slack_user_id=actor_slack_user_id,
        action="bootstrap",
        entity_type="llm_provider_config",
        entity_id=str(provider.id),
        previous_value=None,
        new_value={
            "provider_account_id": str(provider.id),
            "provider_kind": provider.provider_kind,
            "credential_source": ENV_CREDENTIAL_SOURCE,
            "tiers": {route.tier.value: route.model for route in routes},
        },
    )
    session.add(audit)
    session.flush()

    return LLMProviderBootstrapResult(
        created=True,
        skipped_reason=None,
        provider_account_id=provider.id,
        model_count=len(catalog_by_model),
        tier_assignment_count=len(assignments),
    )


def secret_resolver_from_settings(
    session: Session,
    *,
    settings: Settings,
) -> SecretResolver | None:
    """Build a DB secret resolver when encrypted provider keys are enabled."""

    if settings.encryption_key is None:
        return None

    def resolve(secret_id: uuid.UUID) -> str:
        secret = session.get(EncryptedSecret, secret_id)
        if secret is None:
            raise LLMModelConfigError("Encrypted secret was not found")
        return decrypt_secret_value(
            bytes(secret.ciphertext),
            encryption_key=cast(str, settings.encryption_key),
        )

    return resolve


def _extra_headers_from_metadata(metadata: Mapping[str, object]) -> Mapping[str, str]:
    raw_headers = metadata.get("extra_headers")
    if not isinstance(raw_headers, Mapping):
        return MappingProxyType({})
    headers = {
        key: value
        for key, value in raw_headers.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    return MappingProxyType(headers)


def _api_version_from_metadata(metadata: Mapping[str, object]) -> str | None:
    value = metadata.get("api_version")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
