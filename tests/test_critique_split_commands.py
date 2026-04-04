"""Tests for split critique vs author application (run_critique / apply_critique)."""

from prscope.planning.core import PlanningCore
from prscope.planning.runtime.critic import (
    ReviewResult,
    hydrate_review_result,
    skipped_validation_review_placeholder,
    synthetic_review_for_chat_refinement,
)
from prscope.store import Store


def test_valid_commands_include_run_critique_and_apply_critique():
    assert "run_critique" in PlanningCore.VALID_COMMANDS["refining"]
    assert "apply_critique" in PlanningCore.VALID_COMMANDS["refining"]
    assert "run_critique" in PlanningCore.VALID_COMMANDS["converged"]
    assert "apply_critique" in PlanningCore.VALID_COMMANDS["converged"]


def test_hydrate_review_result_roundtrip():
    r = ReviewResult(
        strengths=["a"],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=["b"],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=5.0,
        confidence="medium",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue="p",
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
        parse_error=None,
    )
    from dataclasses import asdict

    back = hydrate_review_result(asdict(r))
    assert back is not None
    assert back.design_quality_score == 5.0
    assert back.primary_issue == "p"


class _FakeIssue:
    def __init__(self, description: str) -> None:
        self.description = description


class _FakeTracker:
    def __init__(self, items: list[str]) -> None:
        self._items = items

    def open_issues(self):  # noqa: ANN201
        return [_FakeIssue(t) for t in self._items]


def test_synthetic_review_prefers_open_issues():
    r = synthetic_review_for_chat_refinement(
        _FakeTracker(["alpha", "beta"]),
        user_input="ignored when issues exist",
    )
    assert r.blocking_issues == ["alpha", "beta"]
    assert r.primary_issue == "alpha"


def test_synthetic_review_falls_back_to_user_message():
    r = synthetic_review_for_chat_refinement(_FakeTracker([]), user_input="Please fix the thing")
    assert r.blocking_issues == ["Please fix the thing"]


def test_skipped_validation_placeholder_is_conservative_for_convergence():
    p = skipped_validation_review_placeholder()
    assert p.design_quality_score == 0.0
    assert p.review_complete is False
    assert p.blocking_issues == []
    assert "deferred" in p.prose.lower()


def test_critique_pending_apply_column_migrates(tmp_path):
    db_path = tmp_path / "t.db"
    store = Store(db_path=db_path)
    session = store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="chat",
        status="refining",
    )
    updated = store.update_planning_session(session.id, critique_pending_apply=1)
    assert int(getattr(updated, "critique_pending_apply", 0) or 0) == 1
