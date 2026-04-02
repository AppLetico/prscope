from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.pipeline.stages import PlanningStages
from prscope.store import Store


def test_compose_prior_critique_without_compact() -> None:
    ctx = MagicMock()
    ctx.issue_tracker.distilled_context.return_value = "- a: one"
    ctx.state.working_summary = ""
    assert PlanningStages._compose_prior_critique(ctx) == "- a: one"


def test_compose_prior_critique_with_compact() -> None:
    ctx = MagicMock()
    ctx.issue_tracker.distilled_context.return_value = "- a: one"
    ctx.state.working_summary = "WORKING SUMMARY (compact prior rounds)\n\nx"
    out = PlanningStages._compose_prior_critique(ctx)
    assert "- a: one" in out
    assert "Prior rounds (compact)" in out
    assert "WORKING SUMMARY" in out


def test_compose_prior_critique_compact_only() -> None:
    ctx = MagicMock()
    ctx.issue_tracker.distilled_context.return_value = "(none)"
    ctx.state.working_summary = "summary only"
    out = PlanningStages._compose_prior_critique(ctx)
    assert out.startswith("## Prior rounds (compact)")
    assert "summary only" in out


@pytest.mark.asyncio
async def test_prepare_adversarial_compaction_sets_working_summary(tmp_path) -> None:
    store = Store(tmp_path / "s.db")
    cfg = PrscopeConfig(local_repo=str(tmp_path), planning=PlanningConfig())
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=cfg, repo=repo)
    session = store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    core = runtime._core(session.id)  # noqa: SLF001
    for _ in range(4):
        core.add_turn("critic", "critique text " * 50, round_number=1)
    st = runtime._state(session.id, session)  # noqa: SLF001
    st.max_prompt_tokens = 500_000
    ctx = MagicMock()
    ctx.session_id = session.id
    ctx.round_number = 2
    ctx.requirements = "do thing"
    ctx.core = core
    ctx.state = st
    ctx.event_callback = None
    plan_content = "x" * 15_000
    await runtime._prepare_adversarial_compaction_context(  # noqa: SLF001
        ctx=ctx,
        plan_content=plan_content,
    )
    assert st.working_summary
    assert "WORKING SUMMARY" in st.working_summary or "Objective" in st.working_summary
