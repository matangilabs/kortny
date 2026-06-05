"""Witness candidate primitives for proactive Kortny behavior."""

from kortny.witness.extractor import (
    WITNESS_CHANNEL_PROFILE_EXTRACTOR_PROMPT_NAME,
    WITNESS_CHANNEL_PROFILE_EXTRACTOR_RESPONSE_FORMAT,
    WITNESS_TASK_RESPONSE_EXTRACTOR_PROMPT_NAME,
    WITNESS_TASK_RESPONSE_EXTRACTOR_RESPONSE_FORMAT,
    WitnessChannelProfileExtractor,
    WitnessTaskResponseExtraction,
    WitnessTaskResponseExtractor,
    parse_witness_channel_profile_extraction,
    parse_witness_task_response_extraction,
)
from kortny.witness.lifecycle import (
    DEFAULT_WITNESS_SNOOZE,
    WITNESS_SUGGESTION_PURPOSE,
    WitnessDeliveryResult,
    accept_candidate,
    archive_candidate,
    dismiss_candidate,
    reactivate_candidate,
    send_private_suggestion,
    snooze_candidate,
)
from kortny.witness.opportunities import (
    ALLOWED_CANDIDATE_TYPES,
    WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
    WitnessOpportunityCandidateInput,
    WitnessOpportunityCandidateResult,
    WitnessOpportunityService,
)

__all__ = [
    "ALLOWED_CANDIDATE_TYPES",
    "WITNESS_CHANNEL_PROFILE_EXTRACTOR_PROMPT_NAME",
    "WITNESS_CHANNEL_PROFILE_EXTRACTOR_RESPONSE_FORMAT",
    "WITNESS_SUGGESTION_PURPOSE",
    "WITNESS_TASK_RESPONSE_EXTRACTOR_PROMPT_NAME",
    "WITNESS_TASK_RESPONSE_EXTRACTOR_RESPONSE_FORMAT",
    "WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE",
    "DEFAULT_WITNESS_SNOOZE",
    "WitnessDeliveryResult",
    "WitnessOpportunityCandidateInput",
    "WitnessOpportunityCandidateResult",
    "WitnessOpportunityService",
    "WitnessChannelProfileExtractor",
    "WitnessTaskResponseExtraction",
    "WitnessTaskResponseExtractor",
    "accept_candidate",
    "archive_candidate",
    "dismiss_candidate",
    "parse_witness_channel_profile_extraction",
    "parse_witness_task_response_extraction",
    "reactivate_candidate",
    "send_private_suggestion",
    "snooze_candidate",
]
