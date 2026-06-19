import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import (
    EncryptedSecret,
    Installation,
    LLMBudgetPolicy,
    LLMConfigAudit,
    LLMModelCatalog,
    LLMModelPricing,
    LLMProviderAccount,
    LLMTierAssignment,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm.litellm_provider import LiteLLMProvider
from kortny.llm.provider_config import (
    ModelConfigService,
    bootstrap_llm_provider_config_from_env,
    secret_resolver_from_settings,
)
from kortny.llm.routing import ModelRoute, ModelRouteTier
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.secrets import encrypt_secret_value

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for provider config tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")

    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        cleanup_database(session)
        session.commit()
        yield session
        session.rollback()
        cleanup_database(session)
        session.commit()


def test_env_bootstrap_seeds_provider_models_tiers_and_audit(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    settings = build_settings()

    result = bootstrap_llm_provider_config_from_env(
        db_session,
        installation_id=installation.id,
        settings=settings,
        actor_slack_user_id="UAdmin",
    )

    provider = db_session.scalar(select(LLMProviderAccount))
    models = db_session.scalars(select(LLMModelCatalog)).all()
    assignments = db_session.scalars(select(LLMTierAssignment)).all()
    audit = db_session.scalar(select(LLMConfigAudit))

    assert result.created is True
    assert result.skipped_reason is None
    assert result.provider_account_id is not None
    assert result.model_count == 6
    assert result.tier_assignment_count == 7
    assert provider is not None
    assert provider.provider_kind == "openrouter"
    assert provider.status == "active"
    assert provider.encrypted_secret_id is None
    assert provider.metadata_json["credential_source"] == "env"
    assert provider.metadata_json["seeded_from_env"] is True
    assert {model.model_identifier for model in models} == {
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.8",
        "openai/gpt-5.1",
        "fallback/model",
    }
    assert all(model.source == "env_bootstrap" for model in models)
    assert all(model.is_enabled for model in models)

    model_by_id = {model.id: model.model_identifier for model in models}
    tiers = {item.tier: model_by_id[item.model_catalog_id] for item in assignments}
    assert tiers == {
        "cheap_fast": "deepseek/deepseek-v4-flash",
        "standard": "deepseek/deepseek-v4-pro",
        "analysis": "anthropic/claude-sonnet-4.6",
        "document": "openai/gpt-5.1",
        "high_reasoning": "anthropic/claude-opus-4.8",
        # HIG-268: the humanizer tier now falls back to cheap_fast before
        # standard (it is a stylistic rewrite, the cheapest cognitive task), so
        # an unset LLM_HUMANIZER_MODEL seeds the cheap model, not standard.
        "humanizer": "deepseek/deepseek-v4-flash",
        # HIG-279: vision tier falls back to llm_model when LLM_VISION_MODEL is
        # unset; in this test fixture that is "fallback/model".
        "vision": "fallback/model",
    }
    assert audit is not None
    assert audit.action == "bootstrap"
    assert audit.actor_slack_user_id == "UAdmin"
    assert audit.new_value is not None
    assert audit.new_value["credential_source"] == "env"
    assert audit.new_value["tiers"]["analysis"] == "anthropic/claude-sonnet-4.6"
    assert "secret-llm-key" not in str(provider.metadata_json)
    assert "secret-llm-key" not in str(audit.new_value)


def test_env_bootstrap_skips_when_provider_config_exists(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    existing = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="Existing provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "test"},
    )
    db_session.add(existing)
    db_session.flush()

    result = bootstrap_llm_provider_config_from_env(
        db_session,
        installation_id=installation.id,
        settings=build_settings(),
    )

    provider_count = db_session.scalar(
        select(func.count()).select_from(LLMProviderAccount)
    )
    model_count = db_session.scalar(select(func.count()).select_from(LLMModelCatalog))

    assert result.created is False
    assert result.skipped_reason == "existing_provider_config"
    assert result.provider_account_id == existing.id
    assert provider_count == 1
    assert model_count == 0


def test_env_bootstrap_respects_force_env_escape_hatch(db_session: Session) -> None:
    installation = create_installation(db_session)

    result = bootstrap_llm_provider_config_from_env(
        db_session,
        installation_id=installation.id,
        settings=build_settings(LLM_CONFIG_FORCE_ENV=True),
    )

    provider_count = db_session.scalar(
        select(func.count()).select_from(LLMProviderAccount)
    )

    assert result.created is False
    assert result.skipped_reason == "force_env_enabled"
    assert result.provider_account_id is None
    assert provider_count == 0


def test_model_config_service_bootstraps_and_resolves_env_seeded_config(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    service = ModelConfigService(db_session, settings=build_settings())

    chain = service.resolve_model_chain(
        installation_id=installation.id,
        tier=ModelRouteTier.analysis,
    )

    provider_count = db_session.scalar(
        select(func.count()).select_from(LLMProviderAccount)
    )

    assert chain.source == "env_bootstrap"
    assert chain.fallback_reason is None
    assert chain.primary.model == "anthropic/claude-sonnet-4.6"
    assert chain.primary.provider_kind == "openrouter"
    assert chain.primary.provider_account_id is not None
    assert chain.primary.api_key == "secret-llm-key"
    assert chain.primary.litellm_model == "openrouter/anthropic/claude-sonnet-4.6"
    assert provider_count == 1


def test_model_config_service_resolves_db_chain_in_priority_order(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    primary_model = create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="anthropic/claude-sonnet-4.6",
        tier=ModelRouteTier.analysis,
        priority=1,
        metadata_json={"credential_source": "env", "extra_headers": {"X-Test": "ok"}},
    )
    fallback_model = create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="deepseek/deepseek-v4-pro",
        tier=ModelRouteTier.analysis,
        priority=2,
    )
    db_session.flush()
    service = ModelConfigService(db_session, settings=build_settings())

    chain = service.resolve_model_chain(
        installation_id=installation.id,
        tier=ModelRouteTier.analysis,
    )

    assert chain.source == "db"
    assert [model.model for model in chain.models] == [
        "anthropic/claude-sonnet-4.6",
        "deepseek/deepseek-v4-pro",
    ]
    assert chain.primary.model_catalog_id == primary_model.id
    assert chain.models[1].model_catalog_id == fallback_model.id
    assert chain.primary.api_key == "secret-llm-key"
    assert chain.primary.credential_source == "env"
    assert chain.primary.litellm_kwargs == {
        "model": "openrouter/anthropic/claude-sonnet-4.6",
        "api_key": "secret-llm-key",
        "extra_headers": {"X-Test": "ok"},
    }
    assert chain.primary.litellm_provider_kwargs == {
        "api_key": "secret-llm-key",
        "model": "openrouter/anthropic/claude-sonnet-4.6",
        "extra_headers": {"X-Test": "ok"},
    }
    assert chain.primary.direct_provider_kwargs == chain.primary.litellm_provider_kwargs


def test_model_config_service_force_env_bypasses_db_config(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="anthropic/claude-opus-4.8",
        tier=ModelRouteTier.analysis,
        priority=1,
    )
    db_session.flush()
    service = ModelConfigService(
        db_session,
        settings=build_settings(
            LLM_CONFIG_FORCE_ENV=True,
            LLM_ANALYSIS_MODEL="deepseek/deepseek-v4-flash",
        ),
    )

    chain = service.resolve_model_chain(
        installation_id=installation.id,
        tier=ModelRouteTier.analysis,
    )

    assert chain.source == "env_forced"
    assert chain.fallback_reason == "force_env_enabled"
    assert chain.primary.model == "deepseek/deepseek-v4-flash"
    assert chain.primary.provider_account_id is None


def test_model_config_service_cache_can_be_invalidated(db_session: Session) -> None:
    installation = create_installation(db_session)
    catalog = create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="old/model",
        tier=ModelRouteTier.standard,
        priority=1,
    )
    db_session.flush()
    service = ModelConfigService(
        db_session,
        settings=build_settings(),
        cache_ttl_seconds=60,
    )

    first = service.resolve_model(
        installation_id=installation.id,
        tier=ModelRouteTier.standard,
    )
    catalog.model_identifier = "new/model"
    catalog.display_name = "new/model"
    db_session.flush()
    cached = service.resolve_model(
        installation_id=installation.id,
        tier=ModelRouteTier.standard,
    )
    service.invalidate_cache(
        installation_id=installation.id,
        tier=ModelRouteTier.standard,
    )
    refreshed = service.resolve_model(
        installation_id=installation.id,
        tier=ModelRouteTier.standard,
    )

    assert first.model == "old/model"
    assert cached.model == "old/model"
    assert refreshed.model == "new/model"


def test_model_config_service_skips_secret_backed_candidate_without_resolver(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    encrypted_secret = EncryptedSecret(
        installation_id=installation.id,
        secret_type="llm_provider:openrouter",
        ciphertext=b"ciphertext",
    )
    db_session.add(encrypted_secret)
    db_session.flush()
    create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="secret/model",
        tier=ModelRouteTier.standard,
        priority=1,
        encrypted_secret_id=encrypted_secret.id,
        metadata_json={"credential_source": "encrypted_secret"},
    )
    db_session.flush()
    service = ModelConfigService(
        db_session,
        settings=build_settings(LLM_STANDARD_MODEL="env/model"),
    )

    chain = service.resolve_model_chain(
        installation_id=installation.id,
        tier=ModelRouteTier.standard,
    )

    assert chain.source == "env_fallback"
    assert chain.fallback_reason == "db_candidates_missing_credentials"
    assert chain.skipped_candidate_count == 1
    assert chain.primary.model == "env/model"
    assert chain.primary.provider_account_id is None


def test_model_config_service_resolves_secret_backed_provider_credentials(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    encryption_key = "test-encryption-key"
    encrypted_secret = EncryptedSecret(
        installation_id=installation.id,
        secret_type="llm_provider:azure:test",
        ciphertext=encrypt_secret_value(
            "db-provider-key",
            encryption_key=encryption_key,
        ),
    )
    db_session.add(encrypted_secret)
    db_session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="azure",
        display_name="Azure dashboard provider",
        status="active",
        health_status="ok",
        base_url="https://example.openai.azure.com",
        encrypted_secret_id=encrypted_secret.id,
        metadata_json={
            "credential_source": "encrypted_secret",
            "api_version": "2024-10-21",
        },
    )
    db_session.add(provider)
    db_session.flush()
    model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="azure/gpt-4o-mini",
        display_name="Azure GPT-4o Mini",
        is_enabled=True,
        capabilities_json={},
        source="manual",
        metadata_json={},
    )
    db_session.add(model)
    db_session.flush()
    db_session.add(
        LLMTierAssignment(
            installation_id=installation.id,
            tier=ModelRouteTier.standard.value,
            model_catalog_id=model.id,
            priority=1,
            is_active=True,
            routing_json={},
        )
    )
    db_session.flush()
    settings = build_settings(ENCRYPTION_KEY=encryption_key)
    service = ModelConfigService(
        db_session,
        settings=settings,
        secret_resolver=secret_resolver_from_settings(
            db_session,
            settings=settings,
        ),
    )

    chain = service.resolve_model_chain(
        installation_id=installation.id,
        tier=ModelRouteTier.standard,
    )

    assert chain.source == "db"
    assert chain.primary.provider_kind == "azure"
    assert chain.primary.api_key == "db-provider-key"
    assert chain.primary.litellm_kwargs == {
        "model": "azure/gpt-4o-mini",
        "api_key": "db-provider-key",
        "api_base": "https://example.openai.azure.com",
        "api_version": "2024-10-21",
    }


def test_runtime_selection_builds_provider_from_db_model_config(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="deepseek/deepseek-db-standard",
        tier=ModelRouteTier.standard,
        priority=1,
    )
    db_session.flush()
    settings = build_settings(
        LLM_API_KEY="env-key",
        LLM_STANDARD_MODEL="deepseek/deepseek-env-standard",
    )

    selection = select_runtime_model(
        session=db_session,
        settings=settings,
        installation_id=installation.id,
        model_route=ModelRoute(
            tier=ModelRouteTier.standard,
            model="deepseek/deepseek-env-standard",
            reason="test_route",
        ),
    )
    provider = create_provider_for_selection(settings=settings, selection=selection)

    assert selection.chain.source == "db"
    assert selection.model_route.model == "deepseek/deepseek-db-standard"
    assert selection.model_route.reason == "test_route"
    assert selection.event_payload["model_config_source"] == "db"
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openrouter/deepseek/deepseek-db-standard"
    assert provider.api_key == "env-key"


def test_runtime_selection_uses_litellm_prefixed_model_for_direct_provider(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_provider_model_assignment(
        db_session,
        installation_id=installation.id,
        model_identifier="openrouter/qwen/qwen3.5-flash-02-23",
        tier=ModelRouteTier.cheap_fast,
        priority=1,
    )
    db_session.flush()
    settings = build_settings(
        LLM_API_KEY="env-key",
        LLM_CHEAP_MODEL="deepseek/deepseek-env-fast",
    )

    selection = select_runtime_model(
        session=db_session,
        settings=settings,
        installation_id=installation.id,
        model_route=ModelRoute(
            tier=ModelRouteTier.cheap_fast,
            model="deepseek/deepseek-env-fast",
            reason="test_route",
        ),
    )
    provider = create_provider_for_selection(settings=settings, selection=selection)

    assert selection.chain.source == "db"
    assert selection.model.model == "openrouter/qwen/qwen3.5-flash-02-23"
    assert selection.model.litellm_model == "openrouter/qwen/qwen3.5-flash-02-23"
    assert selection.model.litellm_provider_kwargs["model"] == (
        "openrouter/qwen/qwen3.5-flash-02-23"
    )
    assert provider.model == "openrouter/qwen/qwen3.5-flash-02-23"


def cleanup_database(session: Session) -> None:
    for model in (
        LLMConfigAudit,
        LLMBudgetPolicy,
        LLMTierAssignment,
        LLMModelPricing,
        LLMModelCatalog,
        LLMProviderAccount,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(
        slack_team_id=f"T{uuid.uuid4().hex}",
        team_name="Highbrow",
        bot_user_id="UKortny",
    )
    session.add(installation)
    session.flush()
    return installation


def create_provider_model_assignment(
    session: Session,
    *,
    installation_id: uuid.UUID,
    model_identifier: str,
    tier: ModelRouteTier,
    priority: int,
    encrypted_secret_id: uuid.UUID | None = None,
    metadata_json: dict[str, object] | None = None,
) -> LLMModelCatalog:
    provider = LLMProviderAccount(
        installation_id=installation_id,
        provider_kind="openrouter",
        display_name=f"OpenRouter provider {priority}",
        status="active",
        health_status="ok",
        encrypted_secret_id=encrypted_secret_id,
        metadata_json=metadata_json or {"credential_source": "env"},
    )
    session.add(provider)
    session.flush()
    catalog = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier=model_identifier,
        display_name=model_identifier,
        is_enabled=True,
        capabilities_json={},
        source="manual",
        metadata_json={},
    )
    session.add(catalog)
    session.flush()
    assignment = LLMTierAssignment(
        installation_id=installation_id,
        tier=tier.value,
        model_catalog_id=catalog.id,
        priority=priority,
        is_active=True,
        routing_json={},
    )
    session.add(assignment)
    return catalog


def build_settings(**overrides: Any) -> Settings:
    assert TEST_POSTGRES_URL is not None
    values: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test-token",
        "SLACK_APP_TOKEN": "xapp-test-token",
        "SLACK_SIGNING_SECRET": "test-signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "secret-llm-key",
        "LLM_MODEL": "fallback/model",
        "LLM_CHEAP_MODEL": "deepseek/deepseek-v4-flash",
        "LLM_STANDARD_MODEL": "deepseek/deepseek-v4-pro",
        "LLM_ANALYSIS_MODEL": "anthropic/claude-sonnet-4.6",
        "LLM_DOCUMENT_MODEL": "openai/gpt-5.1",
        "LLM_HIGH_REASONING_MODEL": "anthropic/claude-opus-4.8",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": TEST_POSTGRES_URL,
    }
    values.update(overrides)
    return Settings(**values)
