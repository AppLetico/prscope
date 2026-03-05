from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig, RepoProfile
from prscope.memory import ParsedConstraint
from prscope.planning.runtime.critic import CriticAgent

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


def _critic_json(violations: list[str], major: int = 1) -> str:
    ids = ", ".join(f'"{v}"' for v in violations)
    return (
        f'{{"major_issues_remaining":{major},"minor_issues_remaining":0,'
        f'"hard_constraint_violations":[{ids}],'
        '"critique_complete":false,"failure_modes":[],'
        '"design_tradeoff_risks":[],"unsupported_claims":[],'
        '"missing_evidence":[],"critic_confidence":0.5,'
        '"operational_readiness":false,"clarification_questions":[]}}'
    )


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


@pytest.mark.asyncio
async def test_critic_accepts_object_style_constraint_violations(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = (
            '{"major_issues_remaining":1,"minor_issues_remaining":0,'
            '"hard_constraint_violations":[{"constraint_id":"HARD_CONSTRAINT_001","evidence":"`api_key` in config"}],'
            '"critique_complete":false,"failure_modes":[],'
            '"design_tradeoff_risks":[],"unsupported_claims":[],'
            '"missing_evidence":[],"critic_confidence":0.5,'
            '"operational_readiness":false,"clarification_questions":[]}'
        )
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
    assert result.hard_constraint_violations == ["HARD_CONSTRAINT_001"]


@pytest.mark.asyncio
async def test_critic_rejects_object_violation_without_constraint_id(tmp_path):
    repo = RepoProfile(name="alpha", path=str(Path(tmp_path)))
    agent = CriticAgent(config=PlanningConfig(critic_model="gpt-4o-mini"), repo=repo)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        raw = (
            '{"major_issues_remaining":1,"minor_issues_remaining":0,'
            '"hard_constraint_violations":[{"evidence":"missing id"}],'
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


# ---------------------------------------------------------------------------
# Evidence-gate filter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppresses_violation_when_no_evidence_in_plan(tmp_path):
    """Violation with evidence_keywords but no matching text in plan → suppressed."""
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return _critic_json(["HARD_CONSTRAINT_001"]), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="Add a public /health endpoint to prscope/web/api.py",
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        session_id="s1",
        round_number=1,
    )
    assert result.hard_constraint_violations == []
    assert result.suppressed_violations == ["HARD_CONSTRAINT_001"]
    assert result.major_issues_remaining == 0


@pytest.mark.asyncio
async def test_filter_keeps_violation_when_keyword_found_in_plan(tmp_path):
    """Violation with a matching evidence keyword in plan text → kept."""
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return _critic_json(["HARD_CONSTRAINT_001"]), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content='Store api_key = "sk-..." in config.py',
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        session_id="s1",
        round_number=1,
    )
    assert "HARD_CONSTRAINT_001" in result.hard_constraint_violations
    assert result.suppressed_violations == []


@pytest.mark.asyncio
async def test_filter_passes_through_unknown_constraint_id(tmp_path):
    """Custom constraint with no evidence_keywords → always kept (safe default)."""
    agent = _make_agent(tmp_path)
    custom = ParsedConstraint(id="MY_CONSTRAINT_XYZ", text="Custom rule.", severity="hard")

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return _critic_json(["MY_CONSTRAINT_XYZ"]), SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="Some plan without matching keywords.",
        manifesto="M",
        architecture="A",
        constraints=[custom],
        session_id="s1",
        round_number=1,
    )
    assert "MY_CONSTRAINT_XYZ" in result.hard_constraint_violations
    assert result.suppressed_violations == []


@pytest.mark.asyncio
async def test_filter_mixed_violations_keeps_evidenced_suppresses_other(tmp_path):
    """HC001 evidenced (api_key in plan), HC002 not evidenced → HC001 kept, HC002 suppressed."""
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        return (
            _critic_json(["HARD_CONSTRAINT_001", "HARD_CONSTRAINT_002"], major=2),
            SimpleNamespace(usage=None),
            "gpt-4o-mini",
        )

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content='Store api_key = "sk-..." in config.py',
        manifesto="M",
        architecture="A",
        constraints=[_HC001, _HC002],
        session_id="s1",
        round_number=2,
    )
    assert result.hard_constraint_violations == ["HARD_CONSTRAINT_001"]
    assert result.suppressed_violations == ["HARD_CONSTRAINT_002"]
    # major_issues_remaining is NOT zeroed — HC001 is real, so LLM count is left intact
    assert result.major_issues_remaining == 2


@pytest.mark.asyncio
async def test_filter_quoted_evidence_override(tmp_path):
    """No keyword in plan text, but LLM prose quotes `api_key` in backticks → kept."""
    agent = _make_agent(tmp_path)

    def fake_llm_call(messages, temperature, model_override=None):
        del messages, temperature, model_override
        # Plan has no keyword, but prose contains a backtick-quoted api_key reference.
        raw = (
            _critic_json(["HARD_CONSTRAINT_001"])
            + "\nThe plan at step 3 stores `api_key` in the config which violates the constraint."
        )
        return raw, SimpleNamespace(usage=None), "gpt-4o-mini"

    agent._llm_call = fake_llm_call  # type: ignore[method-assign]
    result = await agent.run_critic(
        requirements="R",
        plan_content="Add a public /health endpoint. No credentials involved.",
        manifesto="M",
        architecture="A",
        constraints=[_HC001],
        session_id="s1",
        round_number=3,
    )
    assert "HARD_CONSTRAINT_001" in result.hard_constraint_violations
    assert result.suppressed_violations == []
