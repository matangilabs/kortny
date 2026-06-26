"""Tests: profiler model tier resolution + new background profiler worker (HIG-295)."""

from __future__ import annotations

from unittest.mock import MagicMock

from kortny.config import Settings
from kortny.llm.routing import ModelRouter, ModelRouteTier

# ---- Tier resolution --------------------------------------------------------


def _settings(**extra: str) -> Settings:
    base: dict[str, str] = dict(
        SLACK_BOT_TOKEN="xoxb-test",
        SLACK_APP_TOKEN="xapp-test",
        SLACK_SIGNING_SECRET="sec",
        LLM_PROVIDER="openai",
        LLM_API_KEY="sk-test",
        LLM_MODEL="openai/gpt-4o",
        POSTGRES_URL="postgresql://x:x@localhost/x",
        COMPOSIO_API_KEY="test-composio-key",
        ENCRYPTION_KEY="ci-only-test-key",
    )
    base.update(extra)
    return Settings.model_validate(base)


def test_profiler_tier_uses_llm_profiler_model() -> None:
    s = _settings(
        LLM_PROFILER_MODEL="openai/gpt-4.1-mini", LLM_STANDARD_MODEL="openai/gpt-4.1"
    )
    router = ModelRouter(s)
    route = router.route_for_tier(ModelRouteTier.profiler, reason="test")
    assert route.model == "openai/gpt-4.1-mini"
    assert route.tier == ModelRouteTier.profiler


def test_profiler_tier_falls_back_to_standard() -> None:
    s = _settings(LLM_STANDARD_MODEL="openai/gpt-4.1")
    router = ModelRouter(s)
    route = router.route_for_tier(ModelRouteTier.profiler, reason="test")
    assert route.model == "openai/gpt-4.1"


def test_profiler_tier_falls_back_to_llm_model() -> None:
    # Explicitly nullify all optional models so OS env leaks don't interfere.
    s = _settings(
        LLM_PROFILER_MODEL="",
        LLM_STANDARD_MODEL="",
        LLM_CHEAP_MODEL="",
        LLM_ANALYSIS_MODEL="",
        LLM_DOCUMENT_MODEL="",
        LLM_HIGH_REASONING_MODEL="",
        LLM_HUMANIZER_MODEL="",
        LLM_VISION_MODEL="",
    )
    router = ModelRouter(s)
    route = router.route_for_tier(ModelRouteTier.profiler, reason="test")
    assert route.model == "openai/gpt-4o"


def test_profiler_model_none_when_not_set() -> None:
    s = _settings(LLM_PROFILER_MODEL="")
    assert s.llm_profiler_model is None


def test_profiler_model_field_set() -> None:
    s = _settings(LLM_PROFILER_MODEL="openai/gpt-4.1-mini")
    assert s.llm_profiler_model == "openai/gpt-4.1-mini"


def test_profiler_tier_in_model_route_tier() -> None:
    assert ModelRouteTier.profiler == "profiler"


# ---- Settings flags ---------------------------------------------------------


def test_profiler_enabled_default_true() -> None:
    s = _settings()
    assert s.profiler_enabled is True


def test_profiler_enabled_can_be_disabled() -> None:
    s = _settings(KORTNY_PROFILER_ENABLED="false")
    assert s.profiler_enabled is False


def test_profiler_poll_interval_default() -> None:
    s = _settings()
    assert s.profiler_poll_interval_seconds == 60


def test_profiler_poll_interval_override() -> None:
    s = _settings(KORTNY_PROFILER_POLL_INTERVAL_SECONDS="120")
    assert s.profiler_poll_interval_seconds == 120


# ---- Sync path no longer calls LLM -----------------------------------------


def test_sync_toolkit_has_no_llm_attributes() -> None:
    """ComposioCatalogSyncService no longer has set_profiler or llm attributes."""
    from kortny.composio.catalog_sync import ComposioCatalogSyncService

    service = ComposioCatalogSyncService(
        MagicMock(),
        client=MagicMock(),
        embedding_index=None,
    )
    assert not hasattr(service, "set_profiler")
    assert not hasattr(service, "llm")
    assert not hasattr(service, "_profile_task_id")
    assert not hasattr(service, "_profile_toolkit")


def test_catalog_sync_worker_has_no_profiler_factory_param() -> None:
    """ComposioCatalogSyncWorker no longer accepts profiler_factory."""
    import inspect

    from kortny.composio.catalog_sync import ComposioCatalogSyncWorker

    sig = inspect.signature(ComposioCatalogSyncWorker.__init__)
    assert "profiler_factory" not in sig.parameters


# ---- build_default_loops wires capability_profiler --------------------------


def _supervisor_settings(**overrides: object):  # type: ignore[no-untyped-def]
    from kortny.config.settings import LLMProvider, Settings

    kwargs: dict[str, object] = {
        "SLACK_BOT_TOKEN": "xoxb-test-token",
        "SLACK_APP_TOKEN": "xapp-test-token",
        "SLACK_SIGNING_SECRET": "test-signing-secret",
        "LLM_PROVIDER": LLMProvider.openrouter,
        "LLM_API_KEY": "test-llm-key",
        "LLM_MODEL": "openai/gpt-5.4-mini",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost:5432/kortny_test",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_build_default_loops_includes_composio_sync_when_configured() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {loop.name: loop for loop in build_default_loops(_supervisor_settings())}
    assert "composio_catalog_sync" in loops
    assert loops["composio_catalog_sync"].enabled is True


def test_build_default_loops_disables_composio_sync_when_catalog_off() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {
        loop.name: loop
        for loop in build_default_loops(
            _supervisor_settings(COMPOSIO_CATALOG_ENABLED=False)
        )
    }
    assert loops["composio_catalog_sync"].enabled is False


def test_build_default_loops_includes_capability_profiler_when_configured() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {loop.name: loop for loop in build_default_loops(_supervisor_settings())}
    assert "capability_profiler" in loops
    assert loops["capability_profiler"].enabled is True


def test_build_default_loops_disables_profiler_when_catalog_off() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {
        loop.name: loop
        for loop in build_default_loops(
            _supervisor_settings(COMPOSIO_CATALOG_ENABLED=False)
        )
    }
    assert loops["capability_profiler"].enabled is False


def test_build_default_loops_disables_profiler_when_profiler_flag_off() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {
        loop.name: loop
        for loop in build_default_loops(
            _supervisor_settings(KORTNY_PROFILER_ENABLED=False)
        )
    }
    assert loops["capability_profiler"].enabled is False


# ---- CapabilityProfilerWorker unit tests ------------------------------------


def test_profiler_worker_has_correct_advisory_lock_key() -> None:
    from kortny.integration_learning.profiler_worker import (
        PROFILER_ADVISORY_LOCK_KEY,
        CapabilityProfilerWorker,
    )

    worker = CapabilityProfilerWorker(
        session_factory=MagicMock(),
        settings=_settings(),
        use_advisory_lock=False,
    )
    assert worker.advisory_lock_key == PROFILER_ADVISORY_LOCK_KEY
    # Lock key must not clash with catalog sync (759340222).
    assert PROFILER_ADVISORY_LOCK_KEY != 759340222


def test_profiler_worker_default_budget() -> None:
    from kortny.integration_learning.profiler_worker import CapabilityProfilerWorker
    from kortny.integration_learning.profiles import _DEFAULT_PROFILE_CARD_BUDGET

    worker = CapabilityProfilerWorker(
        session_factory=MagicMock(),
        settings=_settings(),
        use_advisory_lock=False,
    )
    assert worker.card_budget == _DEFAULT_PROFILE_CARD_BUDGET


def test_profiler_worker_custom_budget() -> None:
    from kortny.integration_learning.profiler_worker import CapabilityProfilerWorker

    worker = CapabilityProfilerWorker(
        session_factory=MagicMock(),
        settings=_settings(),
        use_advisory_lock=False,
        card_budget=50,
    )
    assert worker.card_budget == 50
