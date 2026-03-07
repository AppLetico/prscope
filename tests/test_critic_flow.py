from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, RepoProfile
from prscope.memory import ParsedConstraint
from prscope.planning.runtime.critic import (
    REVIEWER_SYSTEM_PROMPT,
    CriticAgent,
    CriticContractError,
    ImplementabilityResult,
)

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


def test_reviewer_prompt_preserves_scope_for_simple_health_endpoints() -> None:
    assert "Preserve the user's requested scope" in REVIEWER_SYSTEM_PROMPT
    assert "A public `/health` endpoint is acceptable by default" in REVIEWER_SYSTEM_PROMPT
    assert 'Add a lightweight /health endpoint and tests for it' in REVIEWER_SYSTEM_PROMPT
    assert "logging, monitoring, telemetry, or documentation work" in REVIEWER_SYSTEM_PROMPT


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
async def test_run_critic_coerces_optional_text_fields_from_lists(tmp_path):
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        payload = json.loads(_review_json())
        payload["simplest_possible_design"] = [
            "Keep the existing `/health` route",
            "Add lightweight timing and counters only",
        ]
        payload["primary_issue"] = ["No observability baseline"]
        return json.dumps(payload), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="P",
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        max_retries=0,
    )

    assert result.parse_error is None
    assert result.simplest_possible_design == (
        "Keep the existing `/health` route; Add lightweight timing and counters only"
    )
    assert result.primary_issue == "No observability baseline"


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
async def test_run_critic_filters_scope_creep_for_lightweight_health_requests(tmp_path):
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        payload = json.loads(_review_json())
        payload["blocking_issues"] = [
            "Lack of error handling for the /health endpoint.",
            "Lack of dependency checks for the /health endpoint.",
        ]
        payload["recommended_changes"] = [
            "Implement graceful error handling for unexpected failures.",
            "Add database and external-service dependency checks.",
            "Add authentication to the /health endpoint.",
            "Add logging and monitoring for every health request.",
        ]
        payload["architectural_concerns"] = [
            "Endpoint may raise an unhandled exception.",
            "Missing authentication for the /health endpoint.",
        ]
        payload["risks"] = [
            "Unhandled exceptions may return 500 responses.",
            "Missing dependency checks may misreport service health.",
        ]
        payload["issue_priority"] = [
            "Lack of dependency checks for the /health endpoint.",
            "Lack of error handling for the /health endpoint.",
        ]
        payload["primary_issue"] = "Lack of dependency checks for the /health endpoint."
        return json.dumps(payload), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="Add a lightweight /health endpoint and tests for it.",
        plan_content="Add `/health` to `src/prscope/web/api.py` and tests in `tests/test_web_api_models.py`.",
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        max_retries=0,
    )

    assert result.primary_issue == "Lack of error handling for the /health endpoint."
    assert result.blocking_issues == ["Lack of error handling for the /health endpoint."]
    assert result.recommended_changes == ["Implement graceful error handling for unexpected failures."]
    assert result.architectural_concerns == ["Endpoint may raise an unhandled exception."]
    assert result.risks == ["Unhandled exceptions may return 500 responses."]


@pytest.mark.asyncio
async def test_run_critic_raises_on_missing_required_field(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        payload = json.loads(_review_json())
        del payload["strengths"]
        raw = json.dumps(payload)
        return raw, SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    with pytest.raises(CriticContractError, match="Missing required field: strengths"):
        await agent.run_critic(
            requirements="R",
            plan_content="P",
            manifesto="M",
            architecture="A",
            constraints=[ParsedConstraint(id="HARD_CONSTRAINT_001", text="T", severity="hard", optional=False)],
            max_retries=0,
        )


@pytest.mark.asyncio
async def test_run_critic_raises_on_out_of_range_score(tmp_path):
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return _review_json(score=10.5), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    with pytest.raises(CriticContractError, match="design_quality_score must be in"):
        await agent.run_critic(
            requirements="R",
            plan_content="Add a public /health endpoint to prscope/web/api.py",
            manifesto="M",
            architecture="A",
            constraints=[_HC001],
            max_retries=0,
        )


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


@pytest.mark.asyncio
async def test_run_design_review_skips_multi_perspective_for_small_plan(tmp_path):
    agent = _make_agent(tmp_path)
    perspective_calls = {"count": 0}

    async def fake_run_perspective(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        perspective_calls["count"] += 1
        return "unused"

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return _review_json(include_optional=False), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._run_perspective = fake_run_perspective  # type: ignore[method-assign]
    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_design_review(
        requirements="R",
        plan_content=(
            "# Plan\n\n## Files Changed\n- `prscope/web/api.py`\n\n"
            "## Changes\n- Add lightweight observability to the health endpoint."
        ),
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        round_number=1,
        mode="initial",
    )

    assert result.parse_error is None
    assert perspective_calls["count"] == 0
