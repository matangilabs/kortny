"""Bolt Socket Mode app for Slack ingress."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, TypeVar

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.models import Installation
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.session import session_scope
from kortny.intent import LLMIntentClassifier, should_classify_channel_message
from kortny.llm import (
    LLMProvider,
    LLMService,
    ModelRoute,
    ModelRouter,
    ModelRouteTier,
    create_litellm_provider,
)
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.logging_config import configure_logging
from kortny.observability import configure_tracing, record_span_exception, start_span
from kortny.scheduler import LLMScheduleParser
from kortny.slack.acknowledgement import LLMAcknowledgementGenerator
from kortny.slack.app_home import register_app_home
from kortny.slack.assistant import register_assistant
from kortny.slack.ingress import SlackIngress, is_bare_app_mention
from kortny.slack.outbox import SlackSideEffectOutbox
from kortny.slack.schedule_blocks import SCHEDULE_ACTION_PREFIX
from kortny.slack.witness_actions import register_witness_actions

T = TypeVar("T")
logger = logging.getLogger(__name__)


def acknowledge_then_handle(ack: Callable[[], None], handler: Callable[[], T]) -> T:
    """Ack Slack immediately, then run application work."""

    ack()
    return handler()


def create_bolt_app(
    settings: Settings | None = None,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> App:
    """Create a Bolt app with Kortny event listeners registered."""

    resolved_settings = settings or load_settings()
    acknowledgement_generator = LLMAcknowledgementGenerator(settings=resolved_settings)
    app = App(
        token=resolved_settings.slack_bot_token,
        signing_secret=resolved_settings.slack_signing_secret,
    )
    _recover_stale_side_effects_on_start(session_factory)

    @app.event("app_mention")
    def handle_app_mention(
        ack: Callable[[], None],
        body: dict[str, Any],
        event: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            with start_span(
                "slack.ingress.app_mention",
                attributes=_slack_event_attributes(body, event),
            ):
                try:
                    with session_scope(session_factory) as session:
                        ingress = SlackIngress(
                            session=session,
                            client=client,
                            acknowledgement_generator=acknowledgement_generator,
                            intent_classifier=_intent_classifier(
                                resolved_settings,
                                session,
                                slack_team_id=body.get("team_id"),
                            ),
                            schedule_fallback_parser=_schedule_fallback_parser(
                                resolved_settings,
                                session,
                                slack_team_id=body.get("team_id"),
                            ),
                        )
                        onboarding_result = (
                            ingress.ensure_channel_onboarding_from_mention(
                                body=body,
                                event=event,
                            )
                        )
                        if onboarding_result.observed and is_bare_app_mention(event):
                            logger.info(
                                "Skipped bare app_mention task after channel onboarding event_id=%s channel=%s",
                                body.get("event_id"),
                                event.get("channel"),
                            )
                            return
                        ingress.handle_app_mention(
                            body=body,
                            event=event,
                        )
                except Exception as exc:
                    record_span_exception(exc)
                    logger.exception("Failed to process Slack app_mention event")
                    raise

        acknowledge_then_handle(ack, handle)

    @app.event("message")
    def handle_message(
        ack: Callable[[], None],
        body: dict[str, Any],
        event: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            with start_span(
                "slack.ingress.message",
                attributes=_slack_event_attributes(body, event),
            ):
                is_dm = event.get("channel_type") == "im"
                is_soft_mention_candidate = should_classify_channel_message(
                    event,
                    app_name=resolved_settings.slack_app_name,
                )

                try:
                    with session_scope(session_factory) as session:
                        if is_dm:
                            ingress = SlackIngress(
                                session=session,
                                client=client,
                                acknowledgement_generator=acknowledgement_generator,
                                intent_classifier=_intent_classifier(
                                    resolved_settings,
                                    session,
                                    slack_team_id=body.get("team_id"),
                                ),
                                schedule_fallback_parser=_schedule_fallback_parser(
                                    resolved_settings,
                                    session,
                                    slack_team_id=body.get("team_id"),
                                ),
                            )
                            ingress.handle_dm(
                                body=body,
                                event=event,
                            )
                        else:
                            ingress = SlackIngress(
                                session=session,
                                client=client,
                                acknowledgement_generator=acknowledgement_generator,
                                intent_classifier=_pre_task_intent_classifier(
                                    resolved_settings,
                                    session,
                                    slack_team_id=body.get("team_id"),
                                )
                                if is_soft_mention_candidate
                                else None,
                                schedule_fallback_parser=_schedule_fallback_parser(
                                    resolved_settings,
                                    session,
                                    slack_team_id=body.get("team_id"),
                                )
                                if is_soft_mention_candidate
                                else None,
                            )
                            ingress.observe_channel_message(
                                body=body,
                                event=event,
                            )
                            if not is_soft_mention_candidate:
                                return
                            ingress.handle_channel_message(
                                body=body,
                                event=event,
                                app_name=resolved_settings.slack_app_name,
                            )
                except Exception as exc:
                    record_span_exception(exc)
                    logger.exception("Failed to process Slack message event")
                    raise

        acknowledge_then_handle(ack, handle)

    @app.event("member_joined_channel")
    def handle_member_joined_channel(
        ack: Callable[[], None],
        body: dict[str, Any],
        event: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            with start_span(
                "slack.ingress.member_joined_channel",
                attributes=_slack_event_attributes(body, event),
            ):
                try:
                    with session_scope(session_factory) as session:
                        SlackIngress(
                            session=session,
                            client=client,
                        ).handle_member_joined_channel(
                            body=body,
                            event=event,
                        )
                except Exception as exc:
                    record_span_exception(exc)
                    logger.exception(
                        "Failed to process Slack member_joined_channel event"
                    )
                    raise

        acknowledge_then_handle(ack, handle)

    @app.event("reaction_added")
    def handle_reaction_added(
        ack: Callable[[], None],
        body: dict[str, Any],
        event: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            with start_span(
                "slack.ingress.reaction_added",
                attributes=_slack_event_attributes(body, event),
            ):
                try:
                    with session_scope(session_factory) as session:
                        SlackIngress(
                            session=session,
                            client=client,
                        ).handle_reaction_added(
                            body=body,
                            event=event,
                        )
                except Exception as exc:
                    record_span_exception(exc)
                    logger.exception("Failed to process Slack reaction_added event")
                    raise

        acknowledge_then_handle(ack, handle)

    @app.action(re.compile(f"^{re.escape(SCHEDULE_ACTION_PREFIX)}"))
    def handle_schedule_action(
        ack: Callable[[], None],
        body: dict[str, Any],
        action: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            with start_span(
                "slack.ingress.schedule_action",
                attributes={
                    "slack.action_id": action.get("action_id"),
                    "slack.team_id": _action_body_team_id(body),
                },
            ):
                try:
                    with session_scope(session_factory) as session:
                        SlackIngress(
                            session=session,
                            client=client,
                        ).handle_schedule_action(
                            body=body,
                            action=action,
                        )
                except Exception as exc:
                    record_span_exception(exc)
                    logger.exception("Failed to process Slack schedule action")
                    raise

        acknowledge_then_handle(ack, handle)

    if resolved_settings.app_home_enabled:
        register_app_home(
            app, settings=resolved_settings, session_factory=session_factory
        )
    if resolved_settings.assistant_enabled:
        register_assistant(
            app, settings=resolved_settings, session_factory=session_factory
        )
    register_witness_actions(
        app, settings=resolved_settings, session_factory=session_factory
    )

    return app


def _cheap_tier_llm(
    settings: Settings,
    session: Session,
    *,
    slack_team_id: str | None,
    reason: str,
) -> tuple[LLMProvider, ModelRoute, DbLLMProvider | str]:
    """Resolve the cheap tier through dashboard model config, env as fallback.

    The worker resolves every call through ``select_runtime_model`` so the
    dashboard tier assignments apply; ingress must do the same or operators
    end up with two different models serving the cheap tier.
    """

    model_route = ModelRouter(settings).route_for_tier(
        ModelRouteTier.cheap_fast,
        reason=reason,
    )
    if slack_team_id:
        installation = session.scalar(
            select(Installation).where(Installation.slack_team_id == slack_team_id)
        )
        if installation is not None:
            try:
                selection = select_runtime_model(
                    session=session,
                    settings=settings,
                    installation_id=installation.id,
                    model_route=model_route,
                )
            except Exception:
                logger.exception(
                    "runtime model selection failed at ingress; using env fallback"
                )
            else:
                return (
                    create_provider_for_selection(
                        settings=settings, selection=selection
                    ),
                    selection.model_route,
                    selection.provider_name,
                )
    return (
        create_litellm_provider(settings, model=model_route.model),
        model_route,
        DbLLMProvider(settings.llm_provider),
    )


def _intent_classifier(
    settings: Settings,
    session: Session,
    *,
    slack_team_id: str | None,
) -> LLMIntentClassifier:
    provider, model_route, provider_name = _cheap_tier_llm(
        settings,
        session,
        slack_team_id=slack_team_id,
        reason="intent_classification",
    )
    return LLMIntentClassifier(
        llm=LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            model_route=model_route,
        )
    )


def _pre_task_intent_classifier(
    settings: Settings,
    session: Session,
    *,
    slack_team_id: str | None,
) -> LLMIntentClassifier:
    provider, _model_route, _provider_name = _cheap_tier_llm(
        settings,
        session,
        slack_team_id=slack_team_id,
        reason="intent_classification",
    )
    return LLMIntentClassifier(provider=provider)


def _schedule_fallback_parser(
    settings: Settings,
    session: Session,
    *,
    slack_team_id: str | None,
) -> LLMScheduleParser:
    provider, model_route, provider_name = _cheap_tier_llm(
        settings,
        session,
        slack_team_id=slack_team_id,
        reason="schedule_parsing",
    )
    return LLMScheduleParser(
        llm=LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            model_route=model_route,
        )
    )


def _action_body_team_id(body: dict[str, Any]) -> str | None:
    team = body.get("team")
    if isinstance(team, dict):
        team_id = team.get("id")
        if isinstance(team_id, str):
            return team_id
    team_id = body.get("team_id")
    return team_id if isinstance(team_id, str) else None


def _recover_stale_side_effects_on_start(
    session_factory: sessionmaker[Session] | None,
) -> None:
    with session_scope(session_factory) as session:
        result = SlackSideEffectOutbox(session).recover_stale_in_progress()
        if result.recovered_count:
            logger.warning(
                "slack app recovered stale slack side effects side_effect_ids=%s",
                ",".join(str(id_) for id_ in result.recovered_ids),
            )


def run_socket_mode(settings: Settings | None = None) -> None:
    """Run the Slack Bolt app in Socket Mode."""

    configure_logging()
    resolved_settings = settings or load_settings()
    configure_tracing(resolved_settings)
    app = create_bolt_app(resolved_settings)
    SocketModeHandler(app, resolved_settings.slack_app_token).start()


def _slack_event_attributes(
    body: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "slack.event_id": body.get("event_id"),
        "slack.event_type": event.get("type"),
        "slack.event_subtype": event.get("subtype"),
        "slack.channel_id": event.get("channel"),
        "slack.channel_type": event.get("channel_type"),
        "slack.message_ts": event.get("ts"),
        "slack.thread_ts": event.get("thread_ts"),
        "slack.user_id": event.get("user"),
    }
