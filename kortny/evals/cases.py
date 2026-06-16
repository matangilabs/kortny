"""Replayable eval cases (HIG-258 slice 1: intent classification).

A case is the typed input to a cheap_fast consumer plus the expected label. The
JSON fixture is validated into the real production types at load time, so a
malformed case fails fast (and the well-formedness test catches it in CI without
any network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kortny.intent.models import IntentClassification, IntentRequest

DEFAULT_CASES_PATH = Path(__file__).parent / "fixtures" / "cheap_fast_cases.json"


@dataclass(frozen=True, slots=True)
class IntentCase:
    """One intent-classification case: a request and its expected label."""

    id: str
    request: IntentRequest
    expected_classification: IntentClassification


@dataclass(frozen=True, slots=True)
class EvalCaseFile:
    """A loaded, validated case file."""

    version: int
    intent_cases: tuple[IntentCase, ...]


def load_cases(path: Path | None = None) -> EvalCaseFile:
    """Load and validate the eval case file, raising on malformed content."""

    raw = json.loads((path or DEFAULT_CASES_PATH).read_text(encoding="utf-8"))
    version = int(raw["version"])
    cases: list[IntentCase] = []
    seen: set[str] = set()
    for item in raw.get("intent_cases", []):
        case_id = str(item["id"])
        if case_id in seen:
            raise ValueError(f"duplicate eval case id: {case_id!r}")
        seen.add(case_id)
        cases.append(
            IntentCase(
                id=case_id,
                request=IntentRequest.model_validate(item["request"]),
                expected_classification=IntentClassification(
                    item["expected_classification"]
                ),
            )
        )
    if not cases:
        raise ValueError("case file has no intent_cases")
    return EvalCaseFile(version=version, intent_cases=tuple(cases))
