from __future__ import annotations

import pytest

from prscope.planning.runtime.orchestration_support.chat_flow import RuntimeChatFlow
from prscope.planning.runtime.reasoning import (
    IssueReferenceSignals,
    OpenQuestionResolutionSignals,
    ReasoningContext,
    RefinementReasoner,
)


def test_extract_refinement_message_signals_marks_ambiguous_security_request() -> None:
    signals = RuntimeChatFlow._extract_refinement_message_signals("add security hardening and RBAC")

    assert signals.intent == "refine"
    assert signals.small_refinement is False
    assert signals.heuristic_route == "full_refine"


@pytest.mark.asyncio
async def test_refinement_reasoner_prefers_confident_model_route() -> None:
    reasoner = RefinementReasoner()
    signals = RuntimeChatFlow._extract_refinement_message_signals(
        "actually only update the testing section",
        model_route={"route": "lightweight_refine", "confidence": "high"},
    )

    decision = await reasoner.decide(ReasoningContext(signals=signals))

    assert decision.route == "lightweight_refine"
    assert decision.confidence >= 0.8


def test_refinement_reasoner_preserves_unanswered_open_questions() -> None:
    reasoner = RefinementReasoner()

    decision = reasoner.resolve_open_questions(
        ReasoningContext(
            signals=OpenQuestionResolutionSignals(
                user_message="We should use Postgres",
                current_items=[
                    "- Which database should we use?",
                    "- What auth model should we use?",
                ],
                proposed_items=["- None."],
            )
        )
    )

    assert decision.resolved_action == "preserve_unanswered"
    assert decision.resulting_open_questions == "- What auth model should we use?"


def test_refinement_reasoner_resolves_single_targeted_issue_reference() -> None:
    reasoner = RefinementReasoner()

    decision = reasoner.resolve_issue_references(
        ReasoningContext(
            signals=IssueReferenceSignals(
                user_message="address issue_auth_missing by adding auth requirements",
                issues=[
                    {"id": "issue_auth_missing", "description": "Authentication requirements are missing"},
                    {"id": "issue_retry", "description": "Retry behavior is underspecified"},
                ],
            ),
            session_metadata={"scenario": "issue_resolution"},
        )
    )

    assert decision.issue_resolution == ["issue_auth_missing"]
    assert decision.confidence >= 0.75
