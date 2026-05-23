"""Bolt Socket Mode app for Slack ingress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.session import session_scope
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
            # Non-DM message subscriptions are reserved for the V1.1 ambient
            # observer. For now, only explicit app mentions and DMs create tasks.
            if event.get("channel_type") != "im":
                return

            try:
                with session_scope(session_factory) as session:
                    SlackIngress(
                        session=session,
                        client=client,
                        acknowledgement_generator=acknowledgement_generator,
                    ).handle_dm(
                        body=body,
                        event=event,
                    )
            except Exception:
                logger.exception("Failed to process Slack message event")
                raise

        acknowledge_then_handle(ack, handle)

    return app


def run_socket_mode(settings: Settings | None = None) -> None:
    """Run the Slack Bolt app in Socket Mode."""

    configure_logging()
    resolved_settings = settings or load_settings()
    app = create_bolt_app(resolved_settings)
    SocketModeHandler(app, resolved_settings.slack_app_token).start()
