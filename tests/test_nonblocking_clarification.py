from __future__ import annotations

import asyncio

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.author import AuthorResult
from prscope.planning.runtime.critic import CriticResult
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.store import Store


@pytest.mark.asyncio
async def test_round_returns_quickly_when_clarification_requested(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="t",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial", round_number=1)

    async def fake_run_critic(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return CriticResult(
            major_issues_remaining=1,
            minor_issues_remaining=0,
            hard_constraint_violations=["HARD_CONSTRAINT_001"],
            critique_complete=False,
            failure_modes=[],
            design_tradeoff_risks=[],
            unsupported_claims=[],
            missing_evidence=[],
            critic_confidence=0.2,
            operational_readiness=False,
            vagueness_score=0.0,
            citation_count=0,
            clarification_questions=["Which environment should rollout target first?"],
            prose="Need clarification before refining.",
            parse_error=None,
        )

    async def fake_author_loop(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AuthorResult(plan="# Plan\n\nUnexpected", unverified_references=set(), accessed_paths=set())

    runtime.critic.run_critic = fake_run_critic  # type: ignore[method-assign]
    runtime.author.author_loop = fake_author_loop  # type: ignore[method-assign]

    events: list[dict[str, object]] = []

    async def capture_event(event: dict[str, object]) -> None:
        events.append(event)

    critic_result, author_result, _ = await asyncio.wait_for(
        runtime.run_adversarial_round(session.id, event_callback=capture_event),
        timeout=2.0,
    )

    assert critic_result.clarification_questions == ["Which environment should rollout target first?"]
    assert author_result.plan == "# Plan\n\nInitial"
    assert any(event.get("type") == "clarification_needed" for event in events)
    assert any(
        event.get("type") == "complete" and event.get("message") == "Clarification requested"
        for event in events
    )
