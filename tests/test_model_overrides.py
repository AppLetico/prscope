from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, RepoProfile
from prscope.memory import ParsedConstraint
from prscope.planning.runtime.author import AuthorAgent
from prscope.planning.runtime.critic import CriticAgent
from prscope.planning.runtime.tools import ToolExecutor


@pytest.mark.asyncio
async def test_author_loop_forwards_model_override(tmp_path):
    executor = ToolExecutor(tmp_path)
    agent = AuthorAgent(config=PlanningConfig(author_model="gpt-4o"), tool_executor=executor)

    captured: dict[str, str | None] = {"model": None}

    async def fake_llm_call(messages, *, allow_tools=True, max_output_tokens=None, model_override=None):
        del messages, allow_tools, max_output_tokens
        captured["model"] = model_override
        content = (
            "# Plan Title\n\n## Goals\n- G\n\n## Non-Goals\n- N\n\n## Files Changed\n- `a.py`\n\n## Architecture\n- A\n"
        )
        message = SimpleNamespace(content=content, tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)]), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    await agent.author_loop(
        [{"role": "user", "content": "plan this"}],
        require_tool_calls=False,
        max_attempts=1,
        draft_phase="planner",
        model_override="gpt-4o-mini",
    )
    assert captured["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_critic_run_forwards_model_override(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o"), repo=repo)

    captured: dict[str, str | None] = {"model": None}

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature
        captured["model"] = model_override
        raw = (
            '{"strengths":["good decomposition"],'
            '"architectural_concerns":[],"risks":[],"simplification_opportunities":[],'
            '"blocking_issues":[],"reviewer_questions":[],"recommended_changes":[],'
            '"design_quality_score":9.0,"confidence":"high","review_complete":true,'
            '"simplest_possible_design":"","primary_issue":"","resolved_issues":[],'
            '"constraint_violations":[],"issue_priority":[]}'
        )
        response = SimpleNamespace(usage=None)
        return raw, response, "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="# P",
        manifesto="M",
        architecture="A",
        constraints=[ParsedConstraint(id="C-001", text="X", severity="hard", optional=False)],
        model_override="gpt-4o-mini",
    )
    assert captured["model"] == "gpt-4o-mini"
    assert result.review_complete is True
