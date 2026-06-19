"""The single grounded chokepoint for intent classification (HIG-187).

Intent classification happens at three surfaces — app mention, soft channel
mention, and the DM/assistant pane — across two processes. Each used to build
its own ``IntentRequest`` and attach grounding (the connected integrations the
classifier needs to route well) separately, so every cross-cutting concern had
to be wired three times and drifted: capability grounding shipped into one site,
then had to be retro-fitted into the DM path and the soft-mention path in
follow-up fixes.

This service is the one path all three funnel through. It attaches grounding
ONCE — derived from ``(installation_id, channel_id, user_id)`` so it works
whether or not a Task row exists yet — wraps the call in the standard span, and
returns the decision. A new surface (or a new grounding prior, e.g. persona or
project for the World Model) is added here, once, instead of in N call sites.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from kortny.composio.runtime import (
    IngressConnectionScope,
    connected_toolkit_slugs_for_scope,
)
from kortny.db.models import Task
from kortny.intent.classifier import IntentClassifier
from kortny.intent.models import IntentDecision, IntentRequest
from kortny.intent.persona_gate import persona_relevant_for_text
from kortny.observability import set_span_attributes, start_span


@dataclass(frozen=True, slots=True)
class IntentScope:
    """Slack context used to ground classification, Task or not.

    A Task satisfies this (``task.installation_id`` etc.), and so does a pre-task
    ingress event — so the soft-mention path (which classifies before a Task
    exists) grounds through the same code as the post-task surfaces.
    """

    installation_id: uuid.UUID
    channel_id: str | None
    user_id: str | None


class IntentClassificationService:
    """Ground + classify in one place for every surface."""

    def __init__(self, session: Session, classifier: IntentClassifier) -> None:
        self.session = session
        self.classifier = classifier

    def classify(
        self,
        *,
        request: IntentRequest,
        scope: IntentScope,
        task_id: uuid.UUID | None = None,
        span_task: Task | None = None,
        span_attributes: dict[str, object] | None = None,
    ) -> IntentDecision:
        """Classify ``request``, grounding it from ``scope`` first.

        ``connected_integrations`` is always (re)derived here from the scope, so
        callers never have to remember to attach it — the drift that grounding
        used to suffer is structurally impossible from this path.
        """

        grounded = request.model_copy(
            update={"connected_integrations": self._grounding(scope)}
        )
        attributes: dict[str, object] = {
            "intent.surface": request.surface.value,
            **(span_attributes or {}),
        }
        with start_span("intent.classify", task=span_task, attributes=attributes):
            decision = self.classifier.classify(task_id=task_id, request=grounded)
            # HIG-277: gate persona injection on request shape. Take the
            # classifier's signal OR the deterministic heuristic, so a clear
            # role-relative ask ("my plate") always activates the persona even if
            # the model didn't flag it; factual lookups stay on the neutral path.
            persona_relevant = decision.persona_relevant or persona_relevant_for_text(
                request.text
            )
            if persona_relevant != decision.persona_relevant:
                decision = decision.model_copy(
                    update={"persona_relevant": persona_relevant}
                )
            set_span_attributes(
                {
                    "intent.classification": decision.classification.value,
                    "intent.confidence": decision.confidence,
                    "intent.addressed_to_kortny": decision.addressed_to_kortny,
                    "intent.should_create_task": decision.should_create_task,
                    "intent.model_tier": decision.model_tier.value,
                    "intent.persona_relevant": decision.persona_relevant,
                }
            )
        return decision

    def _grounding(self, scope: IntentScope) -> tuple[str, ...]:
        return connected_toolkit_slugs_for_scope(
            self.session,
            IngressConnectionScope(
                installation_id=scope.installation_id,
                slack_channel_id=scope.channel_id,
                slack_user_id=scope.user_id,
            ),
        )
