from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prscope.config import IssueDedupeConfig, PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.critic import ReviewResult
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.pipeline.stages import PlanningStages
from prscope.planning.runtime.review import IssueGraphTracker, IssueSimilarityService
from prscope.store import Store


def _issue_tracker() -> IssueGraphTracker:
    similarity = IssueSimilarityService(
        IssueDedupeConfig(
            embeddings_enabled="false",
            embedding_model="unused",
            similarity_threshold=0.95,
            fallback_mode="none",
        )
    )
    return IssueGraphTracker(similarity=similarity, max_nodes=50, max_edges=100)


def test_compose_prior_critique_without_compact() -> None:
    ctx = MagicMock()
    ctx.state.review = None
    ctx.issue_tracker.distilled_context.return_value = "- a: one"
    ctx.state.working_summary = ""
    assert PlanningStages._compose_prior_critique(ctx) == "- a: one"


def test_compose_prior_critique_with_compact() -> None:
    ctx = MagicMock()
    ctx.state.review = None
    ctx.issue_tracker.distilled_context.return_value = "- a: one"
    ctx.state.working_summary = "WORKING SUMMARY (compact prior rounds)\n\nx"
    out = PlanningStages._compose_prior_critique(ctx)
    assert "- a: one" in out
    assert "Prior rounds (compact)" in out
    assert "WORKING SUMMARY" in out


def test_compose_prior_critique_compact_only() -> None:
    ctx = MagicMock()
    ctx.state.review = None
    ctx.issue_tracker.distilled_context.return_value = "(none)"
    ctx.state.working_summary = "summary only"
    out = PlanningStages._compose_prior_critique(ctx)
    assert out.startswith("## Prior rounds (compact)")
    assert "summary only" in out


def test_open_tracked_issues_block_for_validation_lists_ids() -> None:
    tracker = _issue_tracker()
    tracker.add_issue("JWT middleware missing", 1, preferred_id="issue_1")
    block = PlanningStages._open_tracked_issues_block_for_validation(tracker)
    assert block is not None
    assert "Open tracked issues" in block
    assert "resolved_issues" in block
    assert "`issue_1`" in block
    assert "JWT middleware missing" in block


def test_open_tracked_issues_block_for_validation_none_when_no_open_issues() -> None:
    tracker = _issue_tracker()
    assert PlanningStages._open_tracked_issues_block_for_validation(tracker) is None


def test_merge_explicit_issue_ids_adds_resolution_when_not_blocked() -> None:
    tracker = _issue_tracker()
    tracker.add_issue("JWT middleware detail", 1, preferred_id="issue_1")
    ctx = MagicMock()
    ctx.requirements = "User input:\nPlease fix issue_1"
    ctx.issue_tracker = tracker
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=7.0,
        confidence="medium",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )
    PlanningStages._merge_explicit_issue_ids_into_validation_resolved(ctx, review)
    assert review.resolved_issues == ["issue_1"]


def test_merge_explicit_issue_ids_skips_when_blocking_repeats_description() -> None:
    tracker = _issue_tracker()
    tracker.add_issue("JWT middleware missing", 1, preferred_id="issue_1")
    ctx = MagicMock()
    ctx.requirements = "User input:\nissue_1"
    ctx.issue_tracker = tracker
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=["JWT middleware missing still"],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=4.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )
    PlanningStages._merge_explicit_issue_ids_into_validation_resolved(ctx, review)
    assert review.resolved_issues == []


def test_merge_open_issues_when_plan_covers_logging_constraint() -> None:
    tracker = _issue_tracker()
    tracker.add_issue(
        "Logging must comply with HARD_CONSTRAINT_001 regarding secret handling.",
        1,
        preferred_id="issue_26",
    )
    ctx = MagicMock()
    ctx.issue_tracker = tracker
    plan = (
        "## Implementation\n"
        "We will use winston and comply with HARD_CONSTRAINT_001; "
        "sensitive tokens are never logged.\n"
    )
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=6.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )
    PlanningStages._merge_open_issues_when_plan_covers_description(ctx, review, plan)
    assert "issue_26" in review.resolved_issues


def test_merge_open_issues_skips_when_blocking_repeats_issue() -> None:
    tracker = _issue_tracker()
    desc = "Logging must comply with HARD_CONSTRAINT_001 regarding secret handling."
    tracker.add_issue(desc, 1, preferred_id="issue_26")
    ctx = MagicMock()
    ctx.issue_tracker = tracker
    plan = "We comply with HARD_CONSTRAINT_001 and never log secrets."
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[desc[:80]],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=5.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )
    PlanningStages._merge_open_issues_when_plan_covers_description(ctx, review, plan)
    assert review.resolved_issues == []


def test_plan_text_covers_issue_description_rate_limiting() -> None:
    desc = "Rate limiting thresholds need to be aligned with existing standards."
    plan = (
        "Implement rate limiting with express-rate-limit; 100 requests per IP per 15-minute window per team standards."
    )
    assert PlanningStages._plan_text_covers_issue_description(plan.lower(), desc)


@pytest.mark.asyncio
async def test_prepare_adversarial_compaction_sets_working_summary(tmp_path) -> None:
    store = Store(tmp_path / "s.db")
    cfg = PrscopeConfig(local_repo=str(tmp_path), planning=PlanningConfig())
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=cfg, repo=repo)
    session = store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    core = runtime._core(session.id)  # noqa: SLF001
    for _ in range(4):
        core.add_turn("critic", "critique text " * 50, round_number=1)
    st = runtime._state(session.id, session)  # noqa: SLF001
    st.max_prompt_tokens = 500_000
    ctx = MagicMock()
    ctx.session_id = session.id
    ctx.round_number = 2
    ctx.requirements = "do thing"
    ctx.core = core
    ctx.state = st
    ctx.event_callback = None
    plan_content = "x" * 15_000
    await runtime._prepare_adversarial_compaction_context(  # noqa: SLF001
        ctx=ctx,
        plan_content=plan_content,
    )
    assert st.working_summary
    assert "WORKING SUMMARY" in st.working_summary or "Objective" in st.working_summary
