"""Bolt Socket Mode app for Slack ingress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.session import session_scope
from kortny.intent import LLMIntentClassifier, should_classify_channel_message
from kortny.llm import LLMService, create_llm_provider
from kortny.logging_config import configure_logging
from kortny.slack.acknowledgement import LLMAcknowledgementGenerator
from kortny.slack.ingress import SlackIngress

T = TypeVar("T")


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

    @app.event("app_mention")
    def handle_app_mention(
        ack: Callable[[], None],
        body: dict[str, Any],
        event: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        def handle() -> None:
            try:
                with session_scope(session_factory) as session:
                    SlackIngress(
                        session=session,
                        client=client,
                        acknowledgement_generator=acknowledgement_generator,
                        intent_classifier=_intent_classifier(
                            resolved_settings,
                            session,
                        ),
                    ).handle_app_mention(
                        body=body,
                        event=event,
                    )
            except Exception:
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
            is_dm = event.get("channel_type") == "im"
            is_soft_mention_candidate = should_classify_channel_message(
                event,
                app_name=resolved_settings.slack_app_name,
            )
            if not is_dm and not is_soft_mention_candidate:
                return

            try:
                with session_scope(session_factory) as session:
                    ingress = SlackIngress(
                        session=session,
                        client=client,
                        acknowledgement_generator=acknowledgement_generator,
                        intent_classifier=_intent_classifier(
                            resolved_settings,
                            session,
                        )
                        if is_dm
                        else _pre_task_intent_classifier(resolved_settings),
                    )
                    if is_dm:
                        ingress.handle_dm(
                            body=body,
                            event=event,
                        )
                    else:
                        ingress.handle_channel_message(
                            body=body,
                            event=event,
                            app_name=resolved_settings.slack_app_name,
                        )
            except Exception:
                logger.exception("Failed to process Slack message event")
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
            try:
                with session_scope(session_factory) as session:
                    SlackIngress(
                        session=session,
                        client=client,
                    ).handle_reaction_added(
                        body=body,
                        event=event,
                    )
            except Exception:
                logger.exception("Failed to process Slack reaction_added event")
                raise

        acknowledge_then_handle(ack, handle)

    return app


def _intent_classifier(settings: Settings, session: Session) -> LLMIntentClassifier:
    return LLMIntentClassifier(
        llm=LLMService(
            session=session,
            provider=create_llm_provider(settings),
            provider_name=DbLLMProvider(settings.llm_provider),
        )
    )


def _pre_task_intent_classifier(settings: Settings) -> LLMIntentClassifier:
    return LLMIntentClassifier(provider=create_llm_provider(settings))


def run_socket_mode(settings: Settings | None = None) -> None:
    """Run the Slack Bolt app in Socket Mode."""

    configure_logging()
    resolved_settings = settings or load_settings()
    app = create_bolt_app(resolved_settings)
    SocketModeHandler(app, resolved_settings.slack_app_token).start()
