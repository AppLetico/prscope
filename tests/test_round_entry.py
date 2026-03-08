from __future__ import annotations

from types import SimpleNamespace

from prscope.planning.runtime.orchestration_support.round_entry import RuntimeRoundEntry


def test_effective_requirements_include_recent_refinement_user_guidance() -> None:
    core = SimpleNamespace(
        get_conversation=lambda: [
            SimpleNamespace(role="author", round=0, content="draft"),
            SimpleNamespace(role="user", round=1, content="Keep the response simple."),
            SimpleNamespace(role="author", round=1, content="updated"),
            SimpleNamespace(role="user", round=2, content="Limit tests to the happy-path 200 response."),
        ]
    )
    session = SimpleNamespace(requirements="Add a lightweight /health endpoint and tests for it.")

    result = RuntimeRoundEntry._effective_requirements(core, session, user_input=None)

    assert result.startswith("Add a lightweight /health endpoint and tests for it.")
    assert "Latest user guidance:" in result
    assert "- Keep the response simple." in result
    assert "- Limit tests to the happy-path 200 response." in result


def test_effective_requirements_prefer_fresh_user_input_when_present() -> None:
    core = SimpleNamespace(get_conversation=lambda: [SimpleNamespace(role="user", round=1, content="older guidance")])
    session = SimpleNamespace(requirements="Base requirements")

    result = RuntimeRoundEntry._effective_requirements(core, session, user_input="New critique focus")

    assert result == "Base requirements\n\nUser input:\nNew critique focus"
