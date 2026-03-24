from __future__ import annotations

import pytest

from prscope.planning.runtime.acceptance_contract import (
    acceptance_criteria_structurally_valid,
    acceptance_evidence_missing_criteria,
    acceptance_satisfied_from_plan,
    acceptance_structurally_invalid_criteria,
    build_convergence_debug_payload,
    compute_harness_convergence_gates,
    harness_failed_gate_label,
    harness_failure_delta,
    verify_convergence_postcondition,
)
from prscope.planning.runtime.critic import ReviewResult


def test_acceptance_structural_rejects_vague_criteria() -> None:
    assert acceptance_criteria_structurally_valid(["The API section lists every new route with HTTP method and path"])
    assert not acceptance_criteria_structurally_valid(["The plan is clear and robust"])


def test_acceptance_satisfied_vacuous_when_empty() -> None:
    assert acceptance_satisfied_from_plan("# any", [])


def test_acceptance_evidence_missing_lists_unmatched() -> None:
    plan = "# Plan\nWe define API routes GET /users and POST /users."
    crit = ["List failure modes for the rollout"]
    missing = acceptance_evidence_missing_criteria(plan, crit)
    assert missing == ["List failure modes for the rollout"]


def test_acceptance_structurally_invalid_criteria() -> None:
    bad = "too short"
    vague = "The plan is clear and robust with good structure"
    assert bad in acceptance_structurally_invalid_criteria([bad, vague])
    assert vague in acceptance_structurally_invalid_criteria([bad, vague])


def _minimal_review(**kwargs: object) -> ReviewResult:
    base = dict(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=8.0,
        confidence="high",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
        plan_rubric=dict.fromkeys(
            ("specificity", "testability", "coherence", "evidence_alignment"),
            8.0,
        ),
        rubric_incomplete=False,
        blocking_categories=[],
        blocking_categories_invalid=False,
        acceptance_criteria=[],
    )
    base.update(kwargs)
    return ReviewResult(**base)  # type: ignore[arg-type]


def test_harness_failed_gate_label_priority() -> None:
    g_ok_acceptance = {"acceptance_structurally_valid": True, "acceptance_satisfied": True}
    g_inc = {**g_ok_acceptance, "rubric_incomplete": True, "rubric_floor_ok": False, "blockers_ok": False}
    assert harness_failed_gate_label(g_inc) == "rubric_incomplete"
    g_floor = {**g_ok_acceptance, "rubric_incomplete": False, "rubric_floor_ok": False, "blockers_ok": True}
    assert harness_failed_gate_label(g_floor) == "rubric_floor"
    g_block = {**g_ok_acceptance, "rubric_incomplete": False, "rubric_floor_ok": True, "blockers_ok": False}
    assert harness_failed_gate_label(g_block) == "blockers"


def test_build_convergence_debug_payload() -> None:
    review = _minimal_review(
        plan_rubric=dict.fromkeys(
            ("specificity", "testability", "coherence", "evidence_alignment"),
            6.0,
        ),
        blocking_categories=["testability"],
        acceptance_criteria=["Document the rollback procedure step by step"],
    )
    gates = compute_harness_convergence_gates(review, "# empty", rubric_floor=7.25)
    payload = build_convergence_debug_payload(gates, review, "# empty", converged=False, reason="rubric_below_floor")
    assert payload["failed_gate"] == "rubric_floor"
    assert payload["min_rubric"] == 6.0
    assert payload["blocking_categories"] == ["testability"]
    assert "Document the rollback procedure step by step" in payload["acceptance_missing"]
    assert payload["failure_delta"] == {
        "acceptance_missing_count": 1,
        "rubric_floor": 1.25,
    }


def test_harness_failure_delta_rubric_only_when_below_floor() -> None:
    gates_pass_rubric = {
        "floor": 7.25,
        "min_rubric": 8.0,
        "rubric_floor_ok": True,
    }
    assert harness_failure_delta(gates_pass_rubric, []) == {"acceptance_missing_count": 0}
    gates_fail = {**gates_pass_rubric, "min_rubric": 7.0, "rubric_floor_ok": False}
    assert harness_failure_delta(gates_fail, []) == {
        "acceptance_missing_count": 0,
        "rubric_floor": 0.25,
    }


def test_compute_harness_gates_rubric_floor() -> None:
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=9.0,
        confidence="high",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
        plan_rubric={
            "specificity": 8.0,
            "testability": 6.0,
            "coherence": 8.0,
            "evidence_alignment": 8.0,
        },
        rubric_incomplete=False,
        blocking_categories=[],
        blocking_categories_invalid=False,
        acceptance_criteria=[],
    )
    g = compute_harness_convergence_gates(review, "# plan", rubric_floor=7.25)
    assert g["rubric_floor_ok"] is False
    assert g["min_rubric"] == 6.0


def test_verify_convergence_postcondition_passes() -> None:
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=8.0,
        confidence="high",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
        plan_rubric=dict.fromkeys(
            ("specificity", "testability", "coherence", "evidence_alignment"),
            8.0,
        ),
        rubric_incomplete=False,
        blocking_categories=[],
        blocking_categories_invalid=False,
        acceptance_criteria=[],
    )
    verify_convergence_postcondition(True, review, "# plan", rubric_floor=7.25)


def test_verify_convergence_postcondition_raises() -> None:
    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=8.0,
        confidence="high",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
        plan_rubric=dict.fromkeys(
            ("specificity", "testability", "coherence", "evidence_alignment"),
            5.0,
        ),
        rubric_incomplete=False,
        blocking_categories=[],
        blocking_categories_invalid=False,
        acceptance_criteria=[],
    )
    with pytest.raises(RuntimeError, match="postcondition"):
        verify_convergence_postcondition(True, review, "# plan", rubric_floor=7.25)
