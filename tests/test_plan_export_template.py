"""Tests for plan export PRD template rendering."""

from __future__ import annotations

import json

from prscope.config import RepoProfile
from prscope.planning.render import _critic_acceptance_bullets_from_plan_json, render_prd
from prscope.store.models import PlanningSession, PlanVersion


def _session() -> PlanningSession:
    return PlanningSession(
        id="sess-1",
        repo_name="demo",
        title="Test Plan",
        requirements="Do the thing.",
        author_model="gpt-4o-mini",
        critic_model="gpt-4o-mini",
        status="converged",
        seed_type="requirements",
        seed_ref=None,
        current_round=1,
        no_recall=0,
        created_at="",
        updated_at="",
    )


def _plan(plan_content: str, plan_json: str | None) -> PlanVersion:
    return PlanVersion(
        id=1,
        session_id="sess-1",
        round=1,
        plan_content=plan_content,
        plan_json=plan_json,
        decision_graph_json=None,
        followups_json=None,
        plan_sha="abc",
        created_at="",
    )


def _repo() -> RepoProfile:
    return RepoProfile(name="demo", path="/tmp/demo", upstream=[])


def test_render_prd_includes_handoff_and_verification() -> None:
    text = render_prd(_session(), _plan("# Summary\n\nHello", None), _repo())
    assert "## After approval (handoff)" in text
    assert "does not modify your repository" in text.lower()
    assert "## Suggested verification" in text
    assert "make check" in text


def test_critic_acceptance_from_plan_json() -> None:
    payload = json.dumps({"acceptance_criteria": ["Must pass tests", "No new deps"]})
    assert _critic_acceptance_bullets_from_plan_json(payload) == ["Must pass tests", "No new deps"]
    lines = render_prd(_session(), _plan("x", payload), _repo())
    assert "Must pass tests" in lines
    assert "Critic-stated criteria" in lines


def test_critic_acceptance_empty_when_no_json() -> None:
    assert _critic_acceptance_bullets_from_plan_json(None) == []
    lines = render_prd(_session(), _plan("x", None), _repo())
    assert "Critic-stated criteria" not in lines
