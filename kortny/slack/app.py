"""Bolt Socket Mode app for Slack ingress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.session import session_scope
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
                    SlackIngress(session=session, client=client).handle_app_mention(
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
    ) -> None:
        ack()

    return app


def run_socket_mode(settings: Settings | None = None) -> None:
    """Run the Slack Bolt app in Socket Mode."""

    resolved_settings = settings or load_settings()
    app = create_bolt_app(resolved_settings)
    SocketModeHandler(app, resolved_settings.slack_app_token).start()
