from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.author import RepairPlan, RevisionResult
from prscope.planning.runtime.authoring.models import PlanDocument, render_markdown
from prscope.planning.runtime.critic import CriticResult, ReviewResult
from prscope.planning.runtime.discovery import DiscoveryQuestion, DiscoveryTurnResult, QuestionOption
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.pipeline.round_context import PlanningRoundContext
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
            strengths=[],
            architectural_concerns=["Security controls for `/api/health` are underspecified."],
            risks=["Deployment target ambiguity could block rollout sequencing."],
            simplification_opportunities=[],
            blocking_issues=["Specify rollout environment before refinement."],
            reviewer_questions=["Which environment should rollout target first?"],
            recommended_changes=["Clarify target environment and threat model assumptions."],
            design_quality_score=3.0,
            confidence="low",
            review_complete=False,
            simplest_possible_design=None,
            primary_issue="Missing rollout environment",
            resolved_issues=[],
            constraint_violations=["HARD_CONSTRAINT_001"],
            issue_priority=["Missing rollout environment"],
            prose="Need clarification before refining.",
            parse_error=None,
        )

    async def fake_plan_repair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return RepairPlan(
            problem_understanding="Need clarification before deep changes.",
            accepted_issues=["Specify rollout environment before refinement."],
            rejected_issues=[],
            root_causes=["Missing deployment context"],
            repair_strategy="Apply best-effort assumptions and mark open question",
            target_sections=["open_questions"],
            revision_plan="Add unresolved clarification explicitly.",
        )

    async def fake_revise_plan(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return RevisionResult(
            problem_understanding="Proceeding with best-effort assumptions.",
            updates={"open_questions": "- Which environment should rollout target first?"},
            justification={"open_questions": "Carries reviewer clarification request forward."},
            review_prediction="Reviewer should request explicit answer next round.",
        )

    runtime.critic.run_design_review = fake_run_critic  # type: ignore[method-assign]
    runtime.author.plan_repair = fake_plan_repair  # type: ignore[method-assign]
    runtime.author.revise_plan = fake_revise_plan  # type: ignore[method-assign]

    events: list[dict[str, object]] = []

    async def capture_event(event: dict[str, object]) -> None:
        events.append(event)

    critic_result, author_result, _ = await asyncio.wait_for(
        runtime.run_adversarial_round(session.id, event_callback=capture_event),
        timeout=2.0,
    )

    assert critic_result.reviewer_questions == ["Which environment should rollout target first?"]
    latest_plan = store.get_plan_versions(session.id, limit=1)[0]
    assert "Which environment should rollout target first?" not in latest_plan.plan_content
    followups = json.loads(latest_plan.followups_json or "{}")
    assert followups["questions"][0]["question"] == "Which environment should rollout target first?"
    assert any(event.get("type") == "plan_ready" for event in events)
    assert any(event.get("type") == "complete" for event in events)


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
async def test_refinement_message_routes_question_to_chat(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="chat-intent",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return "Here's the rationale."

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow.chat_with_author = fake_chat  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="What is the rationale for this section?",
    )

    assert mode == "author_chat"
    assert "rationale" in (reply or "").lower()
    assert round_calls == 0


@pytest.mark.asyncio
async def test_refinement_message_routes_edit_request_to_round(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="refine-intent",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0
    lightweight_calls = 0
    chat_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal chat_calls
        del args, kwargs
        chat_calls += 1
        return "chat fallback"

    async def fake_lightweight(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal lightweight_calls
        del args, kwargs
        lightweight_calls += 1
        return None

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow.chat_with_author = fake_chat  # type: ignore[method-assign]  # noqa: SLF001
    runtime._chat_flow._apply_lightweight_plan_edit = fake_lightweight  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="Please update the plan to remove per-call health logging.",
    )

    assert mode == "refine_round"
    assert reply is None
    assert round_calls == 0
    assert lightweight_calls == 1
    assert chat_calls == 0


@pytest.mark.asyncio
async def test_refinement_message_routes_should_statement_to_round(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="should-intent",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0
    lightweight_calls = 0
    chat_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal chat_calls
        del args, kwargs
        chat_calls += 1
        return "chat fallback"

    async def fake_lightweight(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal lightweight_calls
        del args, kwargs
        lightweight_calls += 1
        return None

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow.chat_with_author = fake_chat  # type: ignore[method-assign]  # noqa: SLF001
    runtime._chat_flow._apply_lightweight_plan_edit = fake_lightweight  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="You should log response every time the endpoint is called",
    )

    assert mode == "refine_round"
    assert reply is None
    assert round_calls == 0
    assert lightweight_calls == 1
    assert chat_calls == 0


@pytest.mark.asyncio
async def test_refinement_message_routes_followup_suggestion_to_lightweight_edit(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="followup-suggestion-intent",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0
    lightweight_calls = 0
    chat_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal chat_calls
        del args, kwargs
        chat_calls += 1
        return "chat fallback"

    async def fake_lightweight(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal lightweight_calls
        del args, kwargs
        lightweight_calls += 1
        return None

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow.chat_with_author = fake_chat  # type: ignore[method-assign]  # noqa: SLF001
    runtime._chat_flow._apply_lightweight_plan_edit = fake_lightweight  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="Add how we'll verify the new behavior works.",
    )

    assert mode == "refine_round"
    assert reply is None
    assert round_calls == 0
    assert lightweight_calls == 1
    assert chat_calls == 0


@pytest.mark.asyncio
async def test_refinement_message_yes_answer_routes_to_lightweight_edit(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="yes-answer-intent",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0
    lightweight_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_lightweight(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal lightweight_calls
        del args, kwargs
        lightweight_calls += 1
        return None

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow._apply_lightweight_plan_edit = fake_lightweight  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="yes response time should be logged",
    )

    assert mode == "refine_round"
    assert reply is None
    assert lightweight_calls == 1
    assert round_calls == 0


@pytest.mark.asyncio
async def test_refinement_message_ambiguous_statement_defaults_to_full_round(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="ambiguous-intent",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0
    chat_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal chat_calls
        del args, kwargs
        chat_calls += 1
        return "chat fallback"

    async def fake_classifier(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return None

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow.chat_with_author = fake_chat  # type: ignore[method-assign]  # noqa: SLF001
    runtime._chat_flow._classify_ambiguous_refinement_message = fake_classifier  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="Actually this should be async.",
    )

    assert mode == "refine_round"
    assert reply is None
    assert round_calls == 1
    assert chat_calls == 0


@pytest.mark.asyncio
async def test_lightweight_edit_resolves_single_targeted_issue_in_snapshot(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="targeted-issue",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    plan = PlanDocument(
        title="Health endpoint improvements",
        summary="Improve the existing health endpoint.",
        goals="- Add clearer health reporting.",
        non_goals="- Do not add a new endpoint.",
        files_changed="- `src/prscope/web/api.py`",
        architecture="Keep the existing FastAPI route and extend its response carefully.",
        implementation_steps="1. Update the existing handler.\n2. Return clearer health information.",
        test_strategy="Add tests for healthy and unhealthy responses.",
        rollback_plan="Revert the handler to the previous response shape if needed.",
        open_questions="",
    )
    runtime._core(session.id).save_plan_version(
        render_markdown(plan),
        round_number=1,
        plan_document=plan,
    )
    runtime.store.update_planning_session(session.id, current_round=1)

    state = runtime._state(session.id, session)  # noqa: SLF001
    tracker = state.issue_tracker
    assert tracker is not None
    tracker.add_issue(
        "Lack of a clear strategy for error handling in the new health checks.",
        1,
        preferred_id="issue_2",
    )
    tracker.add_issue(
        "Lack of clear error handling for failed health checks could lead to misleading health statuses.",
        1,
        preferred_id="issue_5",
    )
    tracker.add_issue(
        "Error handling strategy needs to be clearly defined to avoid misleading health statuses.",
        1,
        preferred_id="issue_8",
    )
    tracker.add_edge("issue_5", "issue_2", "causes")
    tracker.add_edge("issue_5", "issue_8", "causes")

    async def fake_llm_call(messages, **kwargs):  # type: ignore[no-untyped-def]
        del messages, kwargs
        payload = {
            "problem_understanding": "Clarify failure handling for health checks.",
            "updates": {
                "implementation_steps": (
                    "1. Update the existing handler.\n"
                    "2. Return clear failure details when a health check fails.\n"
                    "3. Avoid reporting a healthy status when any required check fails."
                ),
                "test_strategy": (
                    "Add tests for healthy and unhealthy responses, including failed dependency checks "
                    "that must surface a non-healthy status."
                ),
            },
            "assistant_reply": "Updated the plan to clarify failed health check error handling.",
        }
        return (
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]),
            "gpt-4o-mini",
        )

    runtime.author._llm_call = fake_llm_call  # type: ignore[method-assign]  # noqa: SLF001
    runtime._attach_plan_version_artifacts = lambda **kwargs: None  # type: ignore[method-assign]  # noqa: SLF001
    snapshot_seen_at_plan_ready: dict[str, object] = {}

    async def capture_event(event: dict[str, object]) -> None:
        if event.get("type") != "plan_ready":
            return
        snapshot = runtime.read_state_snapshot(session.id)
        assert snapshot is not None
        graph = snapshot.get("issue_graph")
        assert isinstance(graph, dict)
        snapshot_seen_at_plan_ready["open_total"] = graph["summary"]["open_total"]
        snapshot_seen_at_plan_ready["issue_5"] = {node["id"]: node["status"] for node in graph["nodes"]}["issue_5"]
        snapshot_seen_at_plan_ready["issue_5_resolution_source"] = {
            node["id"]: node.get("resolution_source") for node in graph["nodes"]
        }["issue_5"]

    await runtime._chat_flow._apply_lightweight_plan_edit(  # noqa: SLF001
        session_id=session.id,
        user_message=(
            "Please update the plan to address Lack of clear error handling for failed health checks "
            "could lead to misleading health statuses. Adjust the approach, tasks, dependencies, "
            "and success checks if needed."
        ),
        event_callback=capture_event,
    )

    snapshot = runtime.read_state_snapshot(session.id)
    assert snapshot is not None
    graph = snapshot.get("issue_graph")
    assert isinstance(graph, dict)
    assert snapshot_seen_at_plan_ready["open_total"] == 2
    assert snapshot_seen_at_plan_ready["issue_5"] == "resolved"
    assert snapshot_seen_at_plan_ready["issue_5_resolution_source"] == "lightweight"
    assert graph["summary"]["open_total"] == 2
    resolved = {node["id"]: node["status"] for node in graph["nodes"]}
    resolution_sources = {node["id"]: node.get("resolution_source") for node in graph["nodes"]}
    assert resolved["issue_2"] == "open"
    assert resolved["issue_5"] == "resolved"
    assert resolved["issue_8"] == "open"
    assert resolution_sources["issue_2"] is None
    assert resolution_sources["issue_5"] == "lightweight"
    assert resolution_sources["issue_8"] is None


@pytest.mark.asyncio
async def test_followup_answer_updates_decision_graph_before_refreshing_sections(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="followup-decision",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    plan = PlanDocument(
        title="Plan",
        summary="Add persistence.",
        goals="Support durable state.",
        non_goals="Do not change the external API.",
        files_changed="`src/app.py` - wire persistence.",
        architecture="Keep the storage layer abstract until the database decision is confirmed.",
        implementation_steps="1. Add the storage abstraction.",
        test_strategy="Add integration tests for persistence.",
        rollback_plan="Revert the storage integration if rollout fails.",
        open_questions="- Which database should store the primary application data?",
    )
    version = runtime._core(session.id).save_plan_version(
        render_markdown(plan),
        round_number=1,
        plan_document=plan,
    )
    runtime.store.update_plan_version_artifacts(
        version_id=version.id,
        decision_graph_json=json.dumps(
            {
                "nodes": {
                    "architecture.database": {
                        "id": "architecture.database",
                        "description": "Which database should store the primary application data?",
                        "value": "PostgreSQL",
                        "section": "architecture",
                    }
                },
                "edges": [],
            }
        ),
        followups_json=json.dumps({"questions": []}),
    )
    runtime.store.update_planning_session(session.id, current_round=1)

    async def fake_llm_call(messages, **kwargs):  # type: ignore[no-untyped-def]
        del messages, kwargs
        payload = {
            "problem_understanding": "Apply the chosen database to the architecture section.",
            "updates": {
                "architecture": (
                    "Use PostgreSQL for the primary application database while keeping the storage "
                    "layer abstraction stable for the rest of the system."
                )
            },
            "assistant_reply": "Updated the architecture to reflect the PostgreSQL decision.",
        }
        return (
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]),
            "gpt-4o-mini",
        )

    runtime.author._llm_call = fake_llm_call  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.apply_followup_answer(
        session_id=session.id,
        followup_id="architecture.database",
        followup_answer="PostgreSQL",
        target_sections=["architecture"],
    )

    latest = store.get_plan_versions(session.id, limit=1)[0]
    plan_payload = json.loads(latest.plan_json or "{}")
    graph_payload = json.loads(latest.decision_graph_json or "{}")
    followups_payload = json.loads(latest.followups_json or "{}")

    assert mode == "decision_refine"
    assert "PostgreSQL" in (reply or "")
    assert "PostgreSQL" in latest.plan_content
    assert plan_payload["open_questions"] == "- None."
    assert graph_payload["nodes"]["architecture.database"]["value"] == "PostgreSQL"
    assert followups_payload["questions"] == []


@pytest.mark.asyncio
async def test_full_revision_reconciles_decision_graph_before_persisting(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="revision-decision",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    plan = PlanDocument(
        title="Plan",
        summary="Add persistence.",
        goals="Support durable state.",
        non_goals="Do not change the external API.",
        files_changed="`src/app.py` - wire persistence.",
        architecture="Keep the storage layer abstract until the database decision is confirmed.",
        implementation_steps="1. Add the storage abstraction.",
        test_strategy="Add integration tests for persistence.",
        rollback_plan="Revert the storage integration if rollout fails.",
        open_questions="- Which database should store the primary application data?",
    )
    version = runtime._core(session.id).save_plan_version(
        render_markdown(plan),
        round_number=1,
        plan_document=plan,
    )
    runtime.store.update_plan_version_artifacts(
        version_id=version.id,
        decision_graph_json=json.dumps(
            {
                "nodes": {
                    "architecture.database": {
                        "id": "architecture.database",
                        "description": "Which database should store the primary application data?",
                        "value": "PostgreSQL",
                        "section": "architecture",
                    }
                },
                "edges": [],
            }
        ),
        followups_json=json.dumps({"questions": []}),
    )
    runtime.store.update_planning_session(session.id, current_round=1)

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
    review_result = ReviewResult(
        strengths=[],
        architectural_concerns=["Database choice remains underspecified for persistence rollout."],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=6.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue="Database choice remains underspecified for persistence rollout.",
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="Database choice remains underspecified for persistence rollout.",
        parse_error=None,
    )
    repair_plan = RepairPlan(
        problem_understanding="Resolve the database decision.",
        accepted_issues=["Database choice remains underspecified for persistence rollout."],
        rejected_issues=[],
        root_causes=["Missing persistence decision"],
        repair_strategy="Choose the database and update the architecture narrative.",
        target_sections=["architecture", "open_questions"],
        revision_plan="Resolve the database decision in the architecture section.",
    )

    async def fake_revise_plan(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return RevisionResult(
            problem_understanding="Choose PostgreSQL for persistence.",
            updates={
                "architecture": (
                    "Use PostgreSQL for the primary application database while keeping the storage "
                    "layer abstraction stable for the rest of the system."
                )
            },
            justification={"architecture": "Resolves the missing persistence decision."},
            review_prediction="Reviewer should accept the clarified persistence choice.",
        )

    runtime.author.revise_plan = fake_revise_plan  # type: ignore[method-assign]

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs

    updated_markdown, changed_sections, _ = await runtime._stages.revise_plan(  # noqa: SLF001
        ctx=ctx,
        current_plan_doc=plan,
        repair_plan=repair_plan,
        review_result=review_result,
        emit_tool=emit_tool,
    )

    latest = store.get_plan_versions(session.id, limit=1)[0]
    plan_payload = json.loads(latest.plan_json or "{}")
    graph_payload = json.loads(latest.decision_graph_json or "{}")

    assert "PostgreSQL" in updated_markdown
    assert "PostgreSQL" in latest.plan_content
    assert "architecture" in changed_sections
    assert "open_questions" in changed_sections
    assert plan_payload["open_questions"] == "- None."
    assert graph_payload["nodes"]["architecture.database"]["value"] == "PostgreSQL"


def test_top_reconsideration_candidates_falls_back_to_medium_pressure_decisions(tmp_path) -> None:
    runtime = PlanningRuntime(
        store=Store(tmp_path / "test.db"),
        config=PrscopeConfig(
            local_repo=str(tmp_path),
            planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
        ),
        repo=RepoProfile(name="repo", path=str(tmp_path)),
    )
    ctx = SimpleNamespace(
        session_id="session-1",
        issue_tracker=SimpleNamespace(
            graph_snapshot=lambda: {
                "nodes": [
                    {
                        "id": "issue_1",
                        "description": "Database scaling limits remain unresolved.",
                        "status": "open",
                        "raised_round": 1,
                        "severity": "major",
                        "issue_type": "architecture",
                        "related_decision_ids": ["architecture.database"],
                        "tags": [],
                    },
                    {
                        "id": "issue_2",
                        "description": "Database throughput may bottleneck rollout traffic.",
                        "status": "open",
                        "raised_round": 1,
                        "severity": "minor",
                        "issue_type": "performance",
                        "related_decision_ids": ["architecture.database"],
                        "tags": [],
                    },
                ],
                "edges": [{"source": "issue_1", "target": "issue_2", "relation": "causes"}],
                "duplicate_alias": {},
            }
        ),
        core=SimpleNamespace(
            store=SimpleNamespace(get_plan_versions=lambda session_id, limit=2: [])
        ),
    )

    candidates = runtime._stages._top_reconsideration_candidates(  # type: ignore[attr-defined]  # noqa: SLF001
        ctx=ctx,
        decision_graph_json=json.dumps(
            {
                "nodes": {
                    "architecture.database": {
                        "id": "architecture.database",
                        "description": "Which database should store the primary application data?",
                        "value": "PostgreSQL",
                        "section": "architecture",
                    }
                }
            }
        ),
    )

    assert len(candidates) == 1
    assert candidates[0]["decision_id"] == "architecture.database"
    assert candidates[0]["reason"] == "pressure_guidance"
    assert candidates[0]["decision_pressure"] == 4
    assert candidates[0]["dominant_cluster"]["root_issue"] == "Database scaling limits remain unresolved."


@pytest.mark.asyncio
async def test_revise_plan_receives_reconsideration_candidates_from_impact_view(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="revision-pressure",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    plan = PlanDocument(
        title="Plan",
        summary="Add persistence caching.",
        goals="Support faster reads.",
        non_goals="Do not change the external API.",
        files_changed="`src/app.py` - add cache reads.",
        architecture="Use PostgreSQL as the canonical database and add an in-memory cache for repeated reads.",
        implementation_steps="1. Add cache lookup before SQLite reads.",
        test_strategy="Add tests for cache hits and invalidation.",
        rollback_plan="Disable caching if rollout fails.",
        open_questions="- None.",
    )
    version = runtime._core(session.id).save_plan_version(
        render_markdown(plan),
        round_number=1,
        plan_document=plan,
    )
    runtime.store.update_plan_version_artifacts(
        version_id=version.id,
        decision_graph_json=json.dumps(
            {
                "nodes": {
                    "architecture.database": {
                        "id": "architecture.database",
                        "description": "Which database should store the primary application data?",
                        "value": "PostgreSQL",
                        "section": "architecture",
                    }
                },
                "edges": [],
            }
        ),
        followups_json=json.dumps({"questions": []}),
    )
    runtime.store.update_planning_session(session.id, current_round=1)

    state = runtime._state(session.id)
    tracker = state.issue_tracker
    issue_a = tracker.add_issue(
        "Cache invalidation strategy is underspecified and may serve stale data.",
        1,
        severity="major",
        source="critic",
        issue_type="architecture",
    )
    tracker.link_issue_to_decisions(issue_a.id, ["architecture.database"], relation="conflict")
    issue_b = tracker.add_issue(
        "Source-of-truth ambiguity between SQLite and cache remains unresolved.",
        1,
        severity="major",
        source="critic",
        issue_type="correctness",
    )
    tracker.link_issue_to_decisions(issue_b.id, ["architecture.database"], relation="conflict")
    tracker.add_edge(issue_a.id, issue_b.id, "causes")

    ctx = PlanningRoundContext(
        core=runtime._core(session.id),
        session_id=session.id,
        round_number=2,
        requirements="r",
        state=state,
        issue_tracker=tracker,
        selected_author_model="gpt-4o-mini",
        selected_critic_model="gpt-4o-mini",
        event_callback=None,
    )
    review_result = ReviewResult(
        strengths=[],
        architectural_concerns=["Cache invalidation strategy remains underspecified."],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=5.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue="Cache invalidation strategy remains underspecified.",
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="Cache invalidation strategy remains underspecified.",
        parse_error=None,
    )
    repair_plan = RepairPlan(
        problem_understanding="Clarify cache invalidation and source-of-truth ownership.",
        accepted_issues=["Cache invalidation strategy remains underspecified."],
        rejected_issues=[],
        root_causes=["Caching decision is under pressure from stale-data risks."],
        repair_strategy="Clarify decision boundaries and invalidation ownership.",
        target_sections=["architecture"],
        revision_plan="Update architecture to make cache invalidation explicit.",
    )
    runtime._stages._top_reconsideration_candidates = (  # type: ignore[attr-defined]  # noqa: SLF001
        lambda **kwargs: [
            {
                "decision_id": "architecture.database",
                "reason": "high_pressure_cluster",
                "decision_pressure": 6,
                "suggested_action": "reconsider architecture",
                "recently_changed": False,
                "dominant_cluster": {
                    "root_issue_id": "issue_1",
                    "root_issue": "Cache invalidation strategy is underspecified and may serve stale data.",
                    "severity": "major",
                    "affected_plan_sections": ["architecture"],
                    "suggested_action": "reconsider architecture",
                },
            }
        ]
    )
    captured: dict[str, object] = {}

    async def fake_revise_plan(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return RevisionResult(
            problem_understanding="Clarify cache invalidation and SQLite ownership.",
            updates={
                "architecture": (
                    "Keep SQLite canonical, scope the cache to read-through behavior, and define explicit invalidation "
                    "when session snapshots change."
                )
            },
            justification={"architecture": "Addresses the pressured cache strategy decision."},
            review_prediction="Reviewer should accept the clarified cache ownership decision.",
        )

    runtime.author.revise_plan = fake_revise_plan  # type: ignore[method-assign]

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs

    await runtime._stages.revise_plan(  # noqa: SLF001
        ctx=ctx,
        current_plan_doc=plan,
        repair_plan=repair_plan,
        review_result=review_result,
        emit_tool=emit_tool,
    )

    candidates = captured.get("reconsideration_candidates")
    assert isinstance(candidates, list)
    assert candidates
    top = candidates[0]
    assert top["decision_id"] == "architecture.database"
    assert top["reason"] == "high_pressure_cluster"
    assert top["suggested_action"] == "reconsider architecture"
    assert top["dominant_cluster"]["root_issue"]


def test_open_question_guard_keeps_unanswered_items(tmp_path) -> None:
    runtime = PlanningRuntime(
        store=Store(tmp_path / "test.db"),
        config=PrscopeConfig(
            local_repo=str(tmp_path),
            planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
        ),
        repo=RepoProfile(name="repo", path=str(tmp_path)),
    )
    current = "- Should response time be logged every call?\n- What response field format should be used?"
    guarded = runtime._chat_flow._guard_open_question_resolution(  # type: ignore[attr-defined]  # noqa: SLF001
        user_message="yes response time should be logged",
        current_open_questions=current,
        proposed_open_questions="- None.",
    )
    assert guarded == "- What response field format should be used?"


def test_open_question_guard_allows_clear_when_user_says_all(tmp_path) -> None:
    runtime = PlanningRuntime(
        store=Store(tmp_path / "test.db"),
        config=PrscopeConfig(
            local_repo=str(tmp_path),
            planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
        ),
        repo=RepoProfile(name="repo", path=str(tmp_path)),
    )
    current = "- Should response time be logged every call?\n- What response field format should be used?"
    guarded = runtime._chat_flow._guard_open_question_resolution(  # type: ignore[attr-defined]  # noqa: SLF001
        user_message="resolve all questions: yes and keep existing schema",
        current_open_questions=current,
        proposed_open_questions="- None.",
    )
    assert guarded == "- None."


@pytest.mark.asyncio
async def test_refinement_message_falls_back_to_full_round_when_lightweight_fails(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="lightweight-fallback",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\nInitial draft", round_number=1)

    round_calls = 0
    lightweight_calls = 0

    async def fake_round(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal round_calls
        del args, kwargs
        round_calls += 1
        return None

    async def fake_lightweight(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal lightweight_calls
        del args, kwargs
        lightweight_calls += 1
        raise ValueError("bad lightweight payload")

    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]
    runtime._chat_flow._apply_lightweight_plan_edit = fake_lightweight  # type: ignore[method-assign]  # noqa: SLF001

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="Please update the plan wording in summary.",
    )

    assert mode == "refine_round"
    assert reply is None
    assert lightweight_calls == 1
    assert round_calls == 1


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
