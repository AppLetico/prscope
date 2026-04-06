"""Tests for critic plan-text coherence and design_review mode selection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.critic import ReviewResult
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.pipeline.plan_fingerprint import plan_content_fingerprint
from prscope.planning.runtime.pipeline.round_context import PlanningRoundContext
from prscope.planning.runtime.pipeline.stages import PlanningStages
from prscope.planning.runtime.review.critic_plan_coherence import coherence_adjust_review
from prscope.store import Store


def _review(
    *,
    blocking_issues: list[str],
    primary_issue: str | None = None,
) -> ReviewResult:
    return ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=blocking_issues,
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=5.0,
        confidence="medium",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=primary_issue,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )


def test_coherence_drops_stale_endpoint_blocking_when_routes_in_plan() -> None:
    plan = "## API\nUse `POST /oauth/token` for the callback.\n"
    rev = _review(blocking_issues=["No concrete endpoint definitions; add routes."])
    out = coherence_adjust_review(plan, rev)
    assert out.blocking_issues == []


def test_coherence_keeps_endpoint_blocking_when_plan_has_no_routes() -> None:
    plan = "## Approach\nWe will improve reliability.\n"
    line = "No concrete endpoint definitions; add routes."
    rev = _review(blocking_issues=[line])
    out = coherence_adjust_review(plan, rev)
    assert out.blocking_issues == [line]


def test_coherence_drops_stale_test_blocking_when_test_section_present() -> None:
    plan = "## Test Strategy\n- pytest for `tests/test_api.py`\n" + ("x" * 500)
    line = "Missing test strategy; no tests named."
    rev = _review(blocking_issues=[line], primary_issue=line)
    out = coherence_adjust_review(plan, rev)
    assert out.blocking_issues == []
    assert out.primary_issue is None


def test_coherence_drops_stale_evidence_when_multiple_backtick_paths() -> None:
    plan = "Touch `src/a.py` and `src/b.py` for wiring.\n"
    line = "Lack of evidence linking the plan to the existing codebase."
    rev = _review(blocking_issues=[line])
    out = coherence_adjust_review(plan, rev)
    assert out.blocking_issues == []


def test_select_design_review_mode_initial_when_fingerprint_missing() -> None:
    ctx = MagicMock()
    ctx.state.last_critic_turn_plan_fingerprint = ""
    ctx.state.review_score_history = [5.0, 5.0]
    assert PlanningStages._select_design_review_mode(ctx, "any plan") == "initial"


def test_select_design_review_mode_initial_when_plan_changed() -> None:
    ctx = MagicMock()
    ctx.state.last_critic_turn_plan_fingerprint = plan_content_fingerprint("old")
    ctx.state.review_score_history = [5.0, 5.0]
    assert PlanningStages._select_design_review_mode(ctx, "new plan body") == "initial"


def test_select_design_review_mode_stabilization_same_fp_stagnant_scores() -> None:
    body = "unchanged plan"
    fp = plan_content_fingerprint(body)
    ctx = MagicMock()
    ctx.state.last_critic_turn_plan_fingerprint = fp
    ctx.state.review_score_history = [5.0, 5.0]
    assert PlanningStages._select_design_review_mode(ctx, body) == "stabilization"


def test_select_design_review_mode_initial_when_scores_not_stagnant() -> None:
    body = "unchanged plan"
    fp = plan_content_fingerprint(body)
    ctx = MagicMock()
    ctx.state.last_critic_turn_plan_fingerprint = fp
    ctx.state.review_score_history = [5.0, 8.0]
    assert PlanningStages._select_design_review_mode(ctx, body) == "initial"


@pytest.mark.asyncio
async def test_design_review_passes_mode_from_selection(tmp_path) -> None:
    store = Store(tmp_path / "coh.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)
    session = store.create_planning_session(
        repo_name=repo.name,
        title="m",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    state = runtime._state(session.id)
    ctx = PlanningRoundContext(
        core=runtime._core(session.id),
        session_id=session.id,
        round_number=2,
        requirements="r",
        state=state,
        issue_tracker=state.issue_tracker,
        selected_author_model="gpt-4o-mini",
        selected_critic_model="gpt-4o-mini",
        event_callback=None,
    )
    captured: dict[str, object] = {}

    async def fake_run_design_review(**kwargs):  # type: ignore[no-untyped-def]
        captured["mode"] = kwargs.get("mode")
        return ReviewResult(
            strengths=[],
            architectural_concerns=[],
            risks=[],
            simplification_opportunities=[],
            blocking_issues=[],
            reviewer_questions=[],
            recommended_changes=[],
            design_quality_score=6.0,
            confidence="medium",
            review_complete=True,
            simplest_possible_design=None,
            primary_issue=None,
            resolved_issues=[],
            constraint_violations=[],
            issue_priority=[],
            prose="",
        )

    runtime.critic.run_design_review = fake_run_design_review  # type: ignore[method-assign]

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs

    plan_a = "plan version a"
    # Fingerprint change vs stored → initial
    state.last_critic_turn_plan_fingerprint = plan_content_fingerprint("older")
    await runtime._stages.design_review(  # noqa: SLF001
        ctx=ctx,
        current_plan_content=plan_a,
        emit_tool=emit_tool,
    )
    assert captured.get("mode") == "initial"

    # Same body as last critic fp + stagnant scores → stabilization
    fp = plan_content_fingerprint(plan_a)
    state.last_critic_turn_plan_fingerprint = fp
    state.review_score_history = [5.0, 5.0]
    await runtime._stages.design_review(  # noqa: SLF001
        ctx=ctx,
        current_plan_content=plan_a,
        emit_tool=emit_tool,
        same_round_repeat=True,
    )
    assert captured.get("mode") == "stabilization"
