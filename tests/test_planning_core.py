from __future__ import annotations

from prscope.config import PlanningConfig, RepoProfile
from prscope.memory import MemoryStore
from prscope.planning.core import PlanningCore
from prscope.store import Store


def test_memory_constraints_parse_and_extends_stub(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifesto = repo_root / ".prscope" / "manifesto.md"
    manifesto.parent.mkdir(parents=True, exist_ok=True)
    manifesto.write_text(
        """
# Project Manifesto

# extends: org-default
constraints:
  - id: C-001
    text: "No synchronous I/O"
    severity: hard
  - id: C-002
    text: "Prefer stdlib"
    severity: soft
    optional: true
        """.strip()
    )

    repo = RepoProfile(name="alpha", path=str(repo_root))
    memory = MemoryStore(repo=repo, config=PlanningConfig())
    constraints = memory.load_constraints(manifesto)

    assert len(constraints) == 2
    assert constraints[0].id == "C-001"
    assert constraints[0].extends == "org-default"
    assert constraints[1].optional is True


def test_planning_core_convergence_semantic_gate(tmp_path):
    store = Store(db_path=tmp_path / "prscope.db")
    session = store.create_planning_session(
        repo_name="alpha",
        title="Plan",
        requirements="Req",
        seed_type="requirements",
    )
    core = PlanningCore(store=store, session_id=session.id, config=PlanningConfig(convergence_threshold=0.2))

    plan_v1 = "# Plan\n\n## Goals\n- A\n\n## Non-Goals\n- B\n\n## TODOs\n- [ ] one\n"
    plan_v2 = "# Plan\n\n## Goals\n- A\n\n## Non-Goals\n- B\n\n## TODOs\n- [ ] one\n"
    core.save_plan_version(plan_v1, round_number=0)
    core.add_turn("critic", "critic", 1, major_issues_remaining=1, minor_issues_remaining=0)
    core.save_plan_version(plan_v2, round_number=1)

    result = core.check_convergence()
    assert result.converged is False
    assert result.reason == "major_issues_open"

    core.add_turn("critic", "critic done", 2, major_issues_remaining=0, minor_issues_remaining=0)
    core.save_plan_version(plan_v2 + "\n", round_number=2)
    result2 = core.check_convergence()
    assert result2.reason in {"identical", "below_threshold"}
