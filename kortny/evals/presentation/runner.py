"""Offline runner for the presentation eval (HIG-280).

Renders each seed case via Typst, runs IR critique + optional visual scoring,
and reports discrimination / pass-rate / mean visual score. Needs Typst and
optionally a vision-capable model -- runs on demand, never in CI.

    uv run python -m kortny.evals.presentation.runner
"""

from __future__ import annotations

import logging

from kortny.documents.critique import (
    DocumentVisualCritic,
    critique_and_fix,
    validate_render,
    visual_critique,
)
from kortny.documents.render import render_spec_pdf
from kortny.evals.presentation.cases import SEED_PRESENTATION_CASES, PresentationCase
from kortny.evals.presentation.scoring import (
    EvaluateFn,
    PresentationReport,
    PresentationScore,
    score_presentation,
)

logger = logging.getLogger(__name__)


def build_live_evaluate_fn(
    font_paths: tuple[str, ...] = (),
    visual_critic: DocumentVisualCritic | None = None,
    visual_critic_max_pages: int = 8,
) -> EvaluateFn:
    """Build an EvaluateFn backed by real rendering (Typst + optional vision critic).

    *visual_critic* is the ``DocumentVisualCritic`` callable from
    ``_build_document_visual_critic``; pass ``None`` to skip vision scoring
    (visual_score will be -1 on every case).
    """

    def evaluate(case: PresentationCase) -> PresentationScore:
        try:
            # 1. IR-level critique (deterministic lint + auto-fix)
            critique_result = critique_and_fix(case.spec)
            issue_count = len(critique_result.issues)

            # 2. Render to PDF
            try:
                pdf_bytes = render_spec_pdf(critique_result.spec, font_paths=font_paths)
            except Exception:
                logger.exception("runner: render failed for %s", case.name)
                return PresentationScore(
                    case_name=case.name,
                    visual_score=-1,
                    issue_count=issue_count + 1,  # count render failure as an issue
                    expects_defects=case.expects_defects,
                )

            # 3. Post-render validation
            render_issues = validate_render(pdf_bytes, "pdf")
            issue_count += len(render_issues)

            # 4. Vision scoring (optional)
            visual_score = -1
            if visual_critic is not None:
                result = visual_critique(
                    pdf_bytes,
                    visual_critic,
                    max_pages=visual_critic_max_pages,
                )
                if result is not None:
                    visual_score = result.overall_score

        except Exception:
            logger.exception("runner: unexpected error for %s", case.name)
            return PresentationScore(
                case_name=case.name,
                visual_score=-1,
                issue_count=1,
                expects_defects=case.expects_defects,
            )

        return PresentationScore(
            case_name=case.name,
            visual_score=visual_score,
            issue_count=issue_count,
            expects_defects=case.expects_defects,
        )

    return evaluate


def run(
    font_paths: tuple[str, ...] = (),
    visual_critic: DocumentVisualCritic | None = None,
    visual_critic_max_pages: int = 8,
) -> PresentationReport:
    evaluate_fn = build_live_evaluate_fn(
        font_paths=font_paths,
        visual_critic=visual_critic,
        visual_critic_max_pages=visual_critic_max_pages,
    )
    return score_presentation(SEED_PRESENTATION_CASES, evaluate_fn)


def _main() -> None:
    report = run()
    print(f"\nPRESENTATION EVAL: {report.summary_line}\n")
    for score in report.scores:
        status = "FLAGGED" if score.flagged else "PASS"
        vision = (
            f"visual={score.visual_score}" if score.visual_score >= 0 else "vision=n/a"
        )
        expected = "expects_defects" if score.expects_defects else "expects_clean"
        print(
            f"  [{status}] {score.case_name:<35} issues={score.issue_count} {vision} ({expected})"
        )
    if report.discrimination < 1.0:
        mismatches = [s for s in report.scores if s.flagged != s.expects_defects]
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches:
            print(
                f"  {m.case_name}: flagged={m.flagged}, expects_defects={m.expects_defects}"
            )


if __name__ == "__main__":
    _main()
