"""Tests: profiler model tier resolution + catalog sync stale gate + ambient wiring."""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import MagicMock

import pytest

from kortny.composio.catalog_sync import ComposioCatalogSyncService
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


# ---- Stale gate -------------------------------------------------------------


def _make_catalog_sync_service(session: MagicMock) -> ComposioCatalogSyncService:
    service = ComposioCatalogSyncService(
        session,
        client=MagicMock(),
        embedding_index=None,
    )
    return service


def _make_card_row(
    tool_slug: str, card_sha: str, enriched: str | None = "desc"
) -> MagicMock:
    row = MagicMock()
    row.tool_slug = tool_slug
    row.card_sha = card_sha
    row.enriched_description = enriched
    return row


def _digest(rows: list[MagicMock]) -> str:
    sorted_shas = "".join(r.card_sha for r in sorted(rows, key=lambda r: r.tool_slug))
    return hashlib.sha256(sorted_shas.encode()).hexdigest()


def test_profiler_skips_when_cards_enriched_and_digest_matches() -> None:
    """Gate: all cards enriched + digest matches -> skip (no LLM call)."""
    installation_id = uuid.uuid4()
    toolkit_slug = "github"

    session = MagicMock()
    service = _make_catalog_sync_service(session)
    llm = MagicMock()
    task_id = uuid.uuid4()
    service.set_profiler(llm, task_id)

    rows = [
        _make_card_row("GITHUB_CREATE_ISSUE", "sha1", "Creates a GitHub issue."),
        _make_card_row("GITHUB_LIST_REPOS", "sha2", "Lists GitHub repositories."),
    ]
    digest = _digest(rows)

    # card_sha_rows query
    card_result = MagicMock()
    card_result.all.return_value = rows
    # KG entity query
    entity = MagicMock()
    entity.attrs_json = {
        "kind": "capability_profile",
        "generated_from": {"card_sha_digest": digest},
    }
    entity_result = MagicMock()
    entity_result.first.return_value = entity

    session.execute.return_value = card_result
    session.scalars.return_value = entity_result

    service._profile_toolkit(installation_id=installation_id, toolkit_slug=toolkit_slug)

    llm.complete.assert_not_called()


def test_profiler_runs_when_card_missing_enriched_description() -> None:
    """Gate: at least one card lacks enriched_description -> run profiler."""
    installation_id = uuid.uuid4()
    toolkit_slug = "github"

    session = MagicMock()
    service = _make_catalog_sync_service(session)
    llm = MagicMock()
    task_id = uuid.uuid4()
    service.set_profiler(llm, task_id)

    rows = [
        _make_card_row("GITHUB_CREATE_ISSUE", "sha1", None),  # missing enriched
        _make_card_row("GITHUB_LIST_REPOS", "sha2", "Lists GitHub repositories."),
    ]
    digest = _digest(rows)

    card_result = MagicMock()
    card_result.all.return_value = rows
    entity = MagicMock()
    entity.attrs_json = {"generated_from": {"card_sha_digest": digest}}
    entity_result = MagicMock()
    entity_result.first.return_value = entity

    # Second scalars call (re-fetch for stamp) returns None entity
    second_entity_result = MagicMock()
    second_entity_result.first.return_value = None

    session.execute.return_value = card_result
    session.scalars.side_effect = [entity_result, second_entity_result]

    with pytest.MonkeyPatch().context() as mp:
        called: list[bool] = []

        def _fake_build(*args: object, **kwargs: object) -> None:
            called.append(True)

        mp.setattr(
            "kortny.integration_learning.profiles.build_capability_profile",
            _fake_build,
        )
        service._profile_toolkit(
            installation_id=installation_id, toolkit_slug=toolkit_slug
        )

    assert called, "build_capability_profile was not called"


def test_profiler_runs_when_digest_changed() -> None:
    """Gate: digest mismatch (cards changed) -> run profiler."""
    installation_id = uuid.uuid4()
    toolkit_slug = "slack"

    session = MagicMock()
    service = _make_catalog_sync_service(session)
    llm = MagicMock()
    task_id = uuid.uuid4()
    service.set_profiler(llm, task_id)

    rows = [
        _make_card_row("SLACK_SEND_MESSAGE", "new_sha1", "Sends a Slack message."),
    ]

    card_result = MagicMock()
    card_result.all.return_value = rows
    entity = MagicMock()
    entity.attrs_json = {"generated_from": {"card_sha_digest": "old_digest"}}
    entity_result = MagicMock()
    entity_result.first.return_value = entity

    second_entity_result = MagicMock()
    second_entity_result.first.return_value = None

    session.execute.return_value = card_result
    session.scalars.side_effect = [entity_result, second_entity_result]

    with pytest.MonkeyPatch().context() as mp:
        called: list[bool] = []

        def _fake_build(*args: object, **kwargs: object) -> None:
            called.append(True)

        mp.setattr(
            "kortny.integration_learning.profiles.build_capability_profile",
            _fake_build,
        )
        service._profile_toolkit(
            installation_id=installation_id, toolkit_slug=toolkit_slug
        )

    assert called, "build_capability_profile was not called"


def test_profiler_skips_when_no_llm_set() -> None:
    """When set_profiler was not called, _profile_toolkit is a no-op."""
    installation_id = uuid.uuid4()
    session = MagicMock()
    service = _make_catalog_sync_service(session)
    # no set_profiler call

    with pytest.MonkeyPatch().context() as mp:
        called: list[bool] = []

        def _fake_build(*args: object, **kwargs: object) -> None:
            called.append(True)

        mp.setattr(
            "kortny.integration_learning.profiles.build_capability_profile",
            _fake_build,
        )
        service._profile_toolkit(installation_id=installation_id, toolkit_slug="github")

    assert not called
    session.execute.assert_not_called()


# ---- Ambient loop wiring seam -----------------------------------------------


def test_catalog_sync_worker_profiler_factory_is_called() -> None:
    """ComposioCatalogSyncWorker._wire_profiler calls the factory and sets profiler."""
    installation_id = uuid.uuid4()
    llm = MagicMock()
    task_id = uuid.uuid4()

    factory_calls: list[tuple[object, uuid.UUID]] = []

    def _factory(session: object, inst_id: uuid.UUID) -> tuple[MagicMock, uuid.UUID]:
        factory_calls.append((session, inst_id))
        return llm, task_id

    from kortny.composio.catalog_sync import ComposioCatalogSyncWorker

    worker = ComposioCatalogSyncWorker(
        session_factory=MagicMock(),
        settings=_settings(),
        profiler_factory=_factory,
        use_advisory_lock=False,
    )

    session = MagicMock()
    service = MagicMock()
    worker._wire_profiler(session, service, installation_id)

    assert len(factory_calls) == 1
    assert factory_calls[0][1] == installation_id
    service.set_profiler.assert_called_once_with(llm, task_id)


def test_catalog_sync_worker_profiler_factory_none_is_noop() -> None:
    """When profiler_factory is None, _wire_profiler is a no-op."""
    from kortny.composio.catalog_sync import ComposioCatalogSyncWorker

    worker = ComposioCatalogSyncWorker(
        session_factory=MagicMock(),
        settings=_settings(),
        profiler_factory=None,
        use_advisory_lock=False,
    )
    session = MagicMock()
    service = MagicMock()
    worker._wire_profiler(session, service, uuid.uuid4())
    service.set_profiler.assert_not_called()
