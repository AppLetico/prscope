from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.author import AuthorResult
from prscope.planning.runtime.critic import CriticResult
from prscope.planning.runtime.discovery import DiscoveryQuestion, DiscoveryTurnResult, QuestionOption
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
            missing_evidence=[
                "Security controls for `prscope/web/api.py` endpoint `/api/health` "
                "are not specified in the Architecture section."
            ],
            critic_confidence=0.2,
            operational_readiness=False,
            vagueness_score=0.0,
            citation_count=0,
            clarification_questions=["Which environment should rollout target first?"],
            prose="Need clarification before refining.",
            parse_error=None,
        )

    async def fake_author_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
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
    # In web/API mode, clarification is surfaced but the round continues best-effort.
    assert author_result.plan == "# Plan\n\nUnexpected"
    assert any(event.get("type") == "clarification_needed" for event in events)
    assert any(
        event.get("type") == "warning" and "best-effort assumptions" in str(event.get("message", ""))
        for event in events
    )


@pytest.mark.asyncio
async def test_chat_with_author_in_refining_does_not_run_critique(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="chat",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    async def fake_llm_call(messages, **kwargs):  # type: ignore[no-untyped-def]
        del messages, kwargs
        return (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="Absolutely. We can simplify rollout sequencing."))
                ]
            ),
            "gpt-4o-mini",
        )

    runtime.author._llm_call = fake_llm_call  # type: ignore[method-assign]  # noqa: SLF001

    events: list[dict[str, object]] = []

    async def capture_event(event: dict[str, object]) -> None:
        events.append(event)

    reply = await runtime.chat_with_author(
        session_id=session.id,
        user_message="Can we simplify rollout?",
        event_callback=capture_event,
    )
    assert "simplify rollout" in reply.lower()
    assert any(event.get("type") == "complete" for event in events)

    core = runtime._core(session.id)
    turns = core.get_conversation()
    assert turns[-2].role == "user"
    assert turns[-2].content == "Can we simplify rollout?"
    assert turns[-1].role == "author"
    assert "simplify rollout" in turns[-1].content.lower()
    assert core.get_session().status == "refining"


@pytest.mark.asyncio
async def test_discovery_turn_preserves_question_text_for_follow_up_context(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="discovery",
        requirements="",
        seed_type="chat",
        status="draft",
    )

    async def fake_discovery_handle_turn(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return DiscoveryTurnResult(
            reply=(
                "Q1: Which backend framework are we using?\n"
                "A) FastAPI\nB) Flask\nC) Django\nD) Other — describe your preference"
            ),
            complete=False,
            summary=None,
            questions=[
                DiscoveryQuestion(
                    index=1,
                    text="Which backend framework are we using?",
                    options=[
                        QuestionOption(letter="A", text="FastAPI", is_other=False),
                        QuestionOption(letter="B", text="Flask", is_other=False),
                        QuestionOption(letter="C", text="Django", is_other=False),
                        QuestionOption(letter="D", text="Other — describe your preference", is_other=True),
                    ],
                )
            ],
        )

    runtime.discovery.handle_turn = fake_discovery_handle_turn  # type: ignore[method-assign]

    result = await runtime.handle_discovery_turn(
        session_id=session.id,
        user_message="Add a health endpoint",
    )
    assert result.complete is False

    turns = runtime._core(session.id).get_conversation()
    assert turns[-1].role == "author"
    assert "which backend framework" in turns[-1].content.lower()


@pytest.mark.asyncio
async def test_discovery_turn_emits_complete_event_for_tool_state_cleanup(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="discovery-complete",
        requirements="",
        seed_type="chat",
        status="draft",
    )

    async def fake_discovery_handle_turn(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return DiscoveryTurnResult(
            reply="Q1: Choose scope?\nA) Small\nB) Medium\nC) Large\nD) Other — describe your preference",
            complete=False,
            summary=None,
            questions=[
                DiscoveryQuestion(
                    index=1,
                    text="Choose scope?",
                    options=[
                        QuestionOption(letter="A", text="Small", is_other=False),
                        QuestionOption(letter="B", text="Medium", is_other=False),
                        QuestionOption(letter="C", text="Large", is_other=False),
                        QuestionOption(letter="D", text="Other — describe your preference", is_other=True),
                    ],
                )
            ],
        )

    runtime.discovery.handle_turn = fake_discovery_handle_turn  # type: ignore[method-assign]
    events: list[dict[str, object]] = []

    async def capture_event(event: dict[str, object]) -> None:
        events.append(event)

    await runtime.handle_discovery_turn(
        session_id=session.id,
        user_message="Add endpoint enhancements",
        event_callback=capture_event,
    )

    assert any(
        event.get("type") == "complete" and "discovery turn complete" in str(event.get("message", "")).lower()
        for event in events
    )
