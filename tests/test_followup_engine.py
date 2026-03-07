from prscope.planning.runtime.followups.decision_graph import DecisionGraph
from prscope.planning.runtime.followups.engine import FollowupEngine


def test_followup_engine_omits_verification_suggestion_once_covered() -> None:
    engine = FollowupEngine()
    followups = engine.generate(
        current_graph=DecisionGraph(),
        plan_content=(
            "# Plan\n\n"
            "## Summary\nEnhance the /health endpoint.\n\n"
            "## Test Strategy\n"
            "Verify the new behavior with integration tests and performance benchmarks.\n\n"
            "## Rollback Plan\n"
            "Revert the endpoint if rollout issues appear.\n"
        ),
        plan_version_id=3,
    )

    suggestion_ids = [suggestion.id for suggestion in followups.suggestions]
    assert "add_api_contract_tests" not in suggestion_ids


def test_followup_engine_keeps_other_suggestions_when_one_is_addressed() -> None:
    engine = FollowupEngine()
    followups = engine.generate(
        current_graph=DecisionGraph(),
        plan_content=(
            "# Plan\n\n"
            "## Summary\nEnhance the /health endpoint with a schema migration.\n\n"
            "## Architecture\n"
            "Keep the new schema backward compatible.\n\n"
            "## Test Strategy\n"
            "Verify the new behavior with endpoint tests.\n"
        ),
        plan_version_id=4,
    )

    suggestion_ids = [suggestion.id for suggestion in followups.suggestions]
    assert "add_api_contract_tests" not in suggestion_ids
    assert "generate_migration_steps" in suggestion_ids


def test_followup_engine_omits_observability_suggestion_once_monitoring_is_described() -> None:
    engine = FollowupEngine()
    followups = engine.generate(
        current_graph=DecisionGraph(),
        plan_content=(
            "# Plan\n\n"
            "## Summary\nEnhance the /health endpoint with logs and metrics.\n\n"
            "## Architecture\n"
            "Track metrics and add monitoring alerts for failed checks.\n"
        ),
        plan_version_id=5,
    )

    suggestion_ids = [suggestion.id for suggestion in followups.suggestions]
    assert "expand_observability" not in suggestion_ids
