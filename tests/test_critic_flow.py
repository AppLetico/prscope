from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, RepoProfile
from prscope.memory import ParsedConstraint
from prscope.planning.runtime.critic import CriticAgent, ImplementabilityResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HC001 = ParsedConstraint(
    id="HARD_CONSTRAINT_001",
    text="No secrets.",
    severity="hard",
    evidence_keywords=["secret", "api_key", "token", "password"],
)
_HC002 = ParsedConstraint(
    id="HARD_CONSTRAINT_002",
    text="No destructive ops.",
    severity="hard",
    evidence_keywords=["drop table", "truncate", "rm -rf", "shutil.rmtree"],
)


def _make_agent(tmp_path: Path) -> CriticAgent:
    repo = RepoProfile(name="alpha", path=str(tmp_path))
    return CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)


def _review_json(
    *,
    score: float = 7.5,
    confidence: str = "medium",
    include_optional: bool = True,
) -> str:
    payload: dict[str, object] = {
        "strengths": ["Clear decomposition"],
        "architectural_concerns": ["Coupling in orchestration"],
        "risks": ["Retry storm under partial failure"],
        "simplification_opportunities": ["Merge two persistence layers"],
        "blocking_issues": ["No rollback guard for schema migration"],
        "reviewer_questions": ["How is idempotency guaranteed?"],
        "recommended_changes": ["Add migration rollback playbook"],
        "design_quality_score": score,
        "confidence": confidence,
        "review_complete": False,
        "resolved_issues": [],
    }
    if include_optional:
        payload["simplest_possible_design"] = "Single orchestrator with derived state"
        payload["primary_issue"] = "Missing rollback strategy"
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_run_critic_parses_new_review_schema(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = _review_json() + "\nStrong design direction, but rollout needs hardening."
        return raw, SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="P",
        manifesto="M",
        architecture="A",
        constraints=[],
    )
    assert result.parse_error is None
    assert result.design_quality_score == 7.5
    assert result.confidence == "medium"
    assert result.primary_issue == "Missing rollback strategy"
    assert result.prose == "Strong design direction, but rollout needs hardening."


@pytest.mark.asyncio
async def test_run_critic_defaults_optional_fields_to_none(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = _review_json(include_optional=False)
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
    assert result.parse_error is None
    assert result.simplest_possible_design is None
    assert result.primary_issue is None


@pytest.mark.asyncio
async def test_run_critic_normalizes_confidence_to_lowercase(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = _review_json(confidence="HIGH")
        return raw, SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content='Store api_key = "sk-..." in config.py',
        manifesto="M",
        architecture="A",
        constraints=[ParsedConstraint(id="HARD_CONSTRAINT_001", text="T", severity="hard", optional=False)],
        max_retries=0,
    )
    assert result.parse_error is None
    assert result.confidence == "high"


@pytest.mark.asyncio
async def test_run_critic_returns_parse_error_on_missing_required_field(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        payload = json.loads(_review_json())
        del payload["strengths"]
        raw = json.dumps(payload)
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
    assert "Missing required field: strengths" in result.parse_error


@pytest.mark.asyncio
async def test_run_critic_returns_parse_error_on_out_of_range_score(tmp_path):
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return _review_json(score=10.5), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="Add a public /health endpoint to prscope/web/api.py",
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        max_retries=0,
    )
    assert result.parse_error is not None
    assert "design_quality_score must be in [0,10]" in result.parse_error


@pytest.mark.asyncio
async def test_run_design_review_implementability_mode_parses_contract(tmp_path):
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        payload = {
            "implementable": True,
            "missing_details": [],
            "implementation_risks": ["Edge case in retry sequencing"],
            "suggested_additions": ["Add idempotency note to execution_flow"],
        }
        return (
            json.dumps(payload) + "\nImplementation-ready with minor caveats.",
            SimpleNamespace(usage=None),
            "gpt-4o-mini",
        )

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_design_review(
        requirements="R",
        plan_content="P",
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        mode="implementability",
    )
    assert isinstance(result, ImplementabilityResult)
    assert result.parse_error is None
    assert result.implementable is True
    assert result.implementation_risks == ["Edge case in retry sequencing"]
