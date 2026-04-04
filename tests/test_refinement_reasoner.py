from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prscope.config import IssueDedupeConfig
from prscope.planning.runtime.orchestration_support.chat_flow import RuntimeChatFlow
from prscope.planning.runtime.pipeline.adversarial_loop import AdversarialPlanningLoop
from prscope.planning.runtime.pipeline.round_context import PlanningRoundContext
from prscope.planning.runtime.reasoning import (
    IssueReferenceSignals,
    OpenQuestionResolutionSignals,
    ReasoningContext,
    RefinementReasoner,
)
from prscope.planning.runtime.review import IssueGraphTracker, IssueSimilarityService


def test_extract_refinement_message_signals_marks_ambiguous_security_request() -> None:
    signals = RuntimeChatFlow._extract_refinement_message_signals("add security hardening and RBAC")

    assert signals.intent == "refine"
    assert signals.small_refinement is False
    assert signals.heuristic_route == "full_refine"


def test_plan_panel_issue_followup_is_small_despite_length_and_security_keywords() -> None:
    """Add-all-to-chat prompts exceed 280 chars and often mention security — must not force full_refine."""
    msg = (
        "Please update the plan to address these review notes:\n\n"
        "1. Implement Bearer token authentication for all API routes.\n"
        "2. Add structured logging for authentication failures.\n\n"
        "Tracked issue ids: `issue_12`, `issue_13` — include fixed ids in resolved_issues.\n"
        "Adjust the approach, tasks, dependencies, and success checks where needed.\n" + ("padding " * 80)
    )
    assert RefinementReasoner.looks_like_plan_panel_issue_followup(msg)
    assert RefinementReasoner.is_small_request(msg)
    signals = RuntimeChatFlow._extract_refinement_message_signals(msg)
    assert signals.small_refinement is True
    assert signals.heuristic_route == "lightweight_refine"


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
    assert decision.investigation is not None
    assert decision.investigation.should_refresh is False


@pytest.mark.asyncio
async def test_refinement_reasoner_triggers_investigation_for_pressure_without_anchors() -> None:
    reasoner = RefinementReasoner()
    signals = RuntimeChatFlow._extract_refinement_message_signals(
        "We should revisit the architecture tradeoff for auth ownership and the source of truth.",
    )

    decision = await reasoner.decide(
        ReasoningContext(
            signals=signals,
            revision_metadata={
                "known_anchors": [],
                "reconsideration_candidates": [{"decision_id": "architecture.auth", "reason": "high_pressure_cluster"}],
                "evidence_confidence": 0.1,
            },
        )
    )

    assert decision.route == "full_refine"
    assert decision.investigation is not None
    assert decision.investigation.should_refresh is True
    assert decision.investigation.reason == "decision_graph_conflict"


@pytest.mark.asyncio
async def test_refinement_reasoner_skips_investigation_for_small_grounded_edit() -> None:
    reasoner = RefinementReasoner()
    signals = RuntimeChatFlow._extract_refinement_message_signals("Please update the test strategy section wording.")

    decision = await reasoner.decide(
        ReasoningContext(
            signals=signals,
            revision_metadata={
                "known_anchors": [
                    "src/prscope/web/frontend/src/components/PlanPanel.tsx",
                    "tests/test_web_api_models.py",
                ],
                "reconsideration_candidates": [],
                "evidence_confidence": 0.92,
            },
        )
    )

    assert decision.route == "lightweight_refine"
    assert decision.investigation is not None
    assert decision.investigation.should_refresh is False


@pytest.mark.asyncio
async def test_refinement_reasoner_skips_preserve_existing_owner_wording_when_grounded() -> None:
    reasoner = RefinementReasoner()
    signals = RuntimeChatFlow._extract_refinement_message_signals(
        "Keep the existing owner choice explicit in Files Changed and Architecture.",
    )

    decision = await reasoner.decide(
        ReasoningContext(
            signals=signals,
            revision_metadata={
                "known_anchors": [
                    "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                    "src/prscope/web/frontend/src/components/PlanPanel.tsx",
                ],
                "reconsideration_candidates": [],
                "evidence_confidence": 0.9,
            },
        )
    )

    assert decision.investigation is not None
    assert decision.investigation.should_refresh is False


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


def test_refinement_reasoner_resolves_batch_tracked_issue_ids() -> None:
    reasoner = RefinementReasoner()

    decision = reasoner.resolve_issue_references(
        ReasoningContext(
            signals=IssueReferenceSignals(
                user_message=(
                    "Please update the plan to address these review notes:\n\n"
                    "1. First note.\n"
                    "2. Second note.\n\n"
                    "Tracked issue ids: `issue_19`, `issue_20`, `issue_21`, `issue_22` — "
                    "include fixed ids in resolved_issues.\n"
                ),
                issues=[
                    {"id": "issue_19", "description": "A"},
                    {"id": "issue_20", "description": "B"},
                    {"id": "issue_21", "description": "C"},
                    {"id": "issue_22", "description": "D"},
                ],
            ),
            session_metadata={"scenario": "issue_resolution"},
        )
    )

    assert decision.issue_resolution == ["issue_19", "issue_20", "issue_21", "issue_22"]
    assert decision.confidence >= 0.8


def test_resolve_plan_panel_tracked_issue_ids_when_validation_skipped_closes_named_issues() -> None:
    similarity = IssueSimilarityService(
        IssueDedupeConfig(
            embeddings_enabled="false",
            embedding_model="unused",
            similarity_threshold=0.95,
            fallback_mode="none",
        )
    )
    tracker = IssueGraphTracker(similarity=similarity, max_nodes=50, max_edges=100)
    tracker.add_issue("a", 1, preferred_id="issue_19")
    tracker.add_issue("b", 1, preferred_id="issue_20")

    ctx = PlanningRoundContext(
        core=MagicMock(),
        session_id="s",
        round_number=3,
        requirements=(
            "Please update the plan to address these review notes:\n\n"
            "Tracked issue ids: `issue_19`, `issue_20` — include fixed ids in resolved_issues.\n"
        ),
        state=MagicMock(),
        issue_tracker=tracker,
    )
    AdversarialPlanningLoop._resolve_plan_panel_tracked_issue_ids_when_validation_skipped(ctx)

    assert [i.id for i in tracker.open_issues()] == []
