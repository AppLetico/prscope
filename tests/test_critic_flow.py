from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, RepoProfile
from prscope.memory import ParsedConstraint
from prscope.planning.runtime.critic import CriticAgent


@pytest.mark.asyncio
async def test_critic_tolerates_unknown_ids_when_no_constraints(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = (
            '{"major_issues_remaining":1,"minor_issues_remaining":0,'
            '"hard_constraint_violations":["HARD_CONSTRAINT_001"],'
            '"critique_complete":false,"failure_modes":[],'
            '"design_tradeoff_risks":[],"unsupported_claims":[],'
            '"missing_evidence":[],"critic_confidence":0.5,'
            '"operational_readiness":false,"clarification_questions":[]}'
        )
        return raw, SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="P",
        manifesto="M",
        architecture="A",
        constraints=[],
    )
    assert result.hard_constraint_violations == []


@pytest.mark.asyncio
async def test_critic_keeps_strict_unknown_id_validation_with_known_ids(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = (
            '{"major_issues_remaining":1,"minor_issues_remaining":0,'
            '"hard_constraint_violations":["UNKNOWN_ID"],'
            '"critique_complete":false,"failure_modes":[],'
            '"design_tradeoff_risks":[],"unsupported_claims":[],'
            '"missing_evidence":[],"critic_confidence":0.5,'
            '"operational_readiness":false,"clarification_questions":[]}'
        )
        return raw, SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="P",
        manifesto="M",
        architecture="A",
        constraints=[ParsedConstraint(id="HARD_CONSTRAINT_001", text="T", severity="hard", optional=False)],
        max_retries=0,
    )
    assert result.parse_error is not None
