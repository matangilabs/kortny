"""Pure scoring for the response-pipeline substance-drop eval (HIG-287).

No DB, LLM, or rendering. Given a sequence of ResponsePipelineCase instances,
scores each by:
  1. Running _is_substance_dropped_prerender() (pre-render colon guard).
  2. Evaluating the post-render zero-blocks guard (mirrors agent_executor.py
     logic: presentation had elements but rendered_blocks is None).
  3. Computing the final posted text (raw fallback when either guard fires,
     humanized text otherwise) and checking that all key_tokens appear in it.

The scorer is intentionally dumb — it assumes the guard replaces text with
sanitize_humanized_response(None, fallback=raw_answer) when it fires, which
mirrors what synthesize_response() + agent_executor.py do after the fix.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from kortny.evals.response_pipeline.cases import ResponsePipelineCase
from kortny.slack.humanizer import (
    _is_substance_dropped_prerender,
    sanitize_humanized_response,
)

# Minimum raw answer length for the post-render guard to fire — mirrors the
# constant used in agent_executor.py (same 200-char threshold as the pre-render
# guard's _SUBSTANCE_DROP_MIN_RAW_CHARS).
_POST_RENDER_MIN_RAW_CHARS = 200


def _is_zero_blocks_from_nonempty_presentation(case: ResponsePipelineCase) -> bool:
    """Mirror the post-render guard logic from agent_executor.py.

    Fires when presentation had elements (presentation_element_count > 0)
    but rendered_blocks is None (all elements dropped), and the raw answer
    is long enough to carry real substance.  Period-terminated intros that
    the colon guard misses are caught here.
    """
    return (
        case.presentation_element_count > 0
        and case.rendered_blocks is None
        and len(case.raw_answer.strip()) >= _POST_RENDER_MIN_RAW_CHARS
    )


@dataclass(frozen=True, slots=True)
class ResponsePipelineResult:
    case_name: str
    guard_fired: bool
    post_render_guard_fired: bool
    expects_guard_trigger: bool
    expects_post_render_guard_trigger: bool
    final_text: str
    missing_tokens: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when both guards behaved correctly AND all key tokens are present."""
        return (
            self.guard_fired == self.expects_guard_trigger
            and self.post_render_guard_fired == self.expects_post_render_guard_trigger
            and not self.missing_tokens
        )


@dataclass(frozen=True, slots=True)
class ResponsePipelineReport:
    results: tuple[ResponsePipelineResult, ...]

    @property
    def discrimination(self) -> float:
        """Fraction of cases where the pre-render guard_fired == expects_guard_trigger."""
        if not self.results:
            return 0.0
        correct = sum(
            1 for r in self.results if r.guard_fired == r.expects_guard_trigger
        )
        return correct / len(self.results)

    @property
    def post_render_discrimination(self) -> float:
        """Fraction of cases where post_render_guard_fired == expects_post_render_guard_trigger."""
        if not self.results:
            return 0.0
        correct = sum(
            1
            for r in self.results
            if r.post_render_guard_fired == r.expects_post_render_guard_trigger
        )
        return correct / len(self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases where passed == True."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def summary_line(self) -> str:
        return (
            f"discrimination={self.discrimination:.3f} "
            f"post_render_discrimination={self.post_render_discrimination:.3f} "
            f"pass_rate={self.pass_rate:.3f} "
            f"cases={len(self.results)}"
        )


def _substance_in_output(final_text: str, rendered_blocks: list[dict] | None) -> str:
    """Combine text and any block JSON into a single searchable string."""
    parts = [final_text]
    if rendered_blocks is not None:
        parts.append(json.dumps(rendered_blocks))
    return " ".join(parts)


def _score_one(case: ResponsePipelineCase) -> ResponsePipelineResult:
    guard_fired = _is_substance_dropped_prerender(
        humanized_text=case.humanized_text,
        raw_answer=case.raw_answer,
        presentation_element_count=case.presentation_element_count,
    )
    post_render_guard_fired = _is_zero_blocks_from_nonempty_presentation(case)

    if guard_fired or post_render_guard_fired:
        # Either guard replaces text with the sanitized raw answer.
        # When the pre-render guard fires, presentation is cleared before
        # render_blocks runs. When the post-render guard fires, render_blocks
        # already returned None and agent_executor replaces response_text.
        final_text = sanitize_humanized_response(None, fallback=case.raw_answer)
        effective_blocks: list[dict] | None = None
    else:
        final_text = case.humanized_text
        effective_blocks = case.rendered_blocks

    searchable = _substance_in_output(final_text, effective_blocks)
    missing = [tok for tok in case.key_tokens if tok not in searchable]

    return ResponsePipelineResult(
        case_name=case.name,
        guard_fired=guard_fired,
        post_render_guard_fired=post_render_guard_fired,
        expects_guard_trigger=case.expects_guard_trigger,
        expects_post_render_guard_trigger=case.expects_post_render_guard_trigger,
        final_text=final_text,
        missing_tokens=missing,
    )


def score_response_pipeline(
    cases: Sequence[ResponsePipelineCase],
) -> ResponsePipelineReport:
    """Score every case and return an aggregate report.

    PURE: contains zero rendering or LLM calls.
    """
    results = tuple(_score_one(c) for c in cases)
    return ResponsePipelineReport(results=results)
