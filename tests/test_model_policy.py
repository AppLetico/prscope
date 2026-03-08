from __future__ import annotations

from prscope.config import PlanningConfig
from prscope.planning.runtime.model_policy import RuntimeModelPolicyResolver


def test_stage_model_policy_uses_stage_overrides() -> None:
    resolver = RuntimeModelPolicyResolver(
        PlanningConfig(
            author_model="gpt-4o-mini",
            critic_model="gpt-4o-mini",
            discovery_model="claude-3-haiku-20240307",
            initial_draft_model="gemini-2.5-flash",
            author_refine_model="gpt-4o-mini",
            critic_review_model="gpt-4o-mini",
            structured_output_fallback_model="gpt-4o-mini",
        )
    )

    policy = resolver.resolve(
        session_author_model=None,
        session_critic_model=None,
        author_model_override=None,
        critic_model_override=None,
    )

    assert policy.discovery.primary_model == "claude-3-haiku-20240307"
    assert policy.initial_draft.primary_model == "gemini-2.5-flash"
    assert policy.author_refine.primary_model == "gpt-4o-mini"
    assert policy.critic_review.primary_model == "gpt-4o-mini"


def test_stage_model_policy_adds_structured_output_fallback_for_google_json_stages() -> None:
    resolver = RuntimeModelPolicyResolver(
        PlanningConfig(
            author_model="gemini-2.5-flash",
            critic_model="gemini-2.5-flash",
            structured_output_fallback_model="gpt-4o-mini",
        )
    )

    policy = resolver.resolve(
        session_author_model=None,
        session_critic_model=None,
        author_model_override=None,
        critic_model_override=None,
    )

    assert policy.initial_draft.fallback_models == ()
    assert policy.author_refine.fallback_models == ("gpt-4o-mini",)
    assert policy.critic_review.fallback_models == ("gpt-4o-mini",)
    assert policy.author_refine.prefers_compact_json is True
    assert policy.critic_review.elevated_json_contract_risk is True
