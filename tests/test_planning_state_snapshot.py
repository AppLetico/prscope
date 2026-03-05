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
