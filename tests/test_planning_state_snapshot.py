from __future__ import annotations

import json

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.store import Store


def test_persist_state_snapshot_writes_session_json(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)
    session = store.create_planning_session(
        repo_name=repo.name,
        title="snapshot",
        requirements="Add a better planning state",
        seed_type="requirements",
        status="draft",
    )

    state = runtime._state(session.id, session)  # noqa: SLF001
    state.plan_markdown = "# Plan\n\nStateful planning."
    state.accessed_paths = {"prscope/planning/runtime/orchestration.py"}
    state.session_cost_usd = 1.25
    state.max_prompt_tokens = 4096

    runtime._persist_state_snapshot(session.id)  # noqa: SLF001

    snapshot_path = tmp_path / ".prscope" / "sessions" / f"{session.id}.json"
    assert snapshot_path.exists()

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == session.id
    assert payload["requirements"] == "Add a better planning state"
    assert payload["plan_markdown"] == "# Plan\n\nStateful planning."
    assert payload["accessed_paths"] == ["prscope/planning/runtime/orchestration.py"]
    assert payload["session_cost_usd"] == 1.25
    assert payload["max_prompt_tokens"] == 4096


def test_current_state_snapshot_uses_live_issue_graph(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)
    session = store.create_planning_session(
        repo_name=repo.name,
        title="live-snapshot",
        requirements="Expose review notes from live state",
        seed_type="requirements",
        status="refining",
    )

    state = runtime._state(session.id, session)  # noqa: SLF001
    state.issue_tracker.add_issue("Missing rollback plan", 1, severity="major", source="critic")

    payload = runtime.current_state_snapshot(session.id)

    assert payload is not None
    assert payload["session_id"] == session.id
    assert payload["issue_graph"]["summary"]["open_total"] == 1
    assert payload["issue_graph"]["nodes"][0]["description"] == "Missing rollback plan"


def test_current_state_snapshot_recovers_issue_graph_from_turns_when_snapshot_is_empty(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)
    session = store.create_planning_session(
        repo_name=repo.name,
        title="recovered-snapshot",
        requirements="Recover review notes after refresh",
        seed_type="requirements",
        status="refining",
    )

    runtime._persist_state_snapshot(session.id)  # noqa: SLF001
    runtime._states.pop(session.id, None)  # noqa: SLF001
    store.add_planning_turn(
        session.id,
        "critic",
        (
            "Design review: score 7.0/10, confidence=medium.\n\n"
            "Primary issue: Clarify the integration with session or diagnostics APIs.\n\n"
            "Recommended changes: Include a basic error handling strategy."
        ),
        round_number=3,
    )
    store.add_planning_turn(
        session.id,
        "user",
        "Please update the plan to address Clarify the integration with session or diagnostics APIs.",
        round_number=4,
    )

    payload = runtime.current_state_snapshot(session.id)

    assert payload is not None
    assert payload["issue_graph"]["summary"]["open_total"] == 1
    assert payload["issue_graph"]["summary"]["resolved_total"] == 1
    assert any(
        node["description"] == "Clarify the integration with session or diagnostics APIs."
        and node["status"] == "resolved"
        and node["resolution_source"] == "lightweight"
        for node in payload["issue_graph"]["nodes"]
    )
