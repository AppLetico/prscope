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


def test_effective_requirements_short_followup_includes_prior_assistant_message() -> None:
    core = SimpleNamespace(
        get_conversation=lambda: [
            SimpleNamespace(role="author", round=0, content="I can add token checks to /api/foo next."),
            SimpleNamespace(role="critic", round=0, content="Some review text."),
        ]
    )
    session = SimpleNamespace(requirements="Secure the API.")

    result = RuntimeRoundEntry._effective_requirements(core, session, user_input="Yes do it")

    assert "Secure the API." in result
    assert "Reply context" in result
    assert "I can add token checks to /api/foo next." in result
    assert "User input:\nYes do it" in result


def test_effective_requirements_short_followup_falls_back_to_critic_when_no_author() -> None:
    core = SimpleNamespace(
        get_conversation=lambda: [
            SimpleNamespace(role="critic", round=0, content="Please tighten the error handling."),
        ]
    )
    session = SimpleNamespace(requirements="Hardening.")

    result = RuntimeRoundEntry._effective_requirements(core, session, user_input="go ahead")

    assert "Please tighten the error handling." in result
    assert "User input:\ngo ahead" in result


def test_effective_requirements_short_followup_uses_latest_author_not_older_critic() -> None:
    core = SimpleNamespace(
        get_conversation=lambda: [
            SimpleNamespace(role="author", round=0, content="First draft."),
            SimpleNamespace(role="critic", round=0, content="Older critic feedback."),
            SimpleNamespace(role="author", round=1, content="If you want, I can wire OAuth next."),
        ]
    )
    session = SimpleNamespace(requirements="Auth work.")

    result = RuntimeRoundEntry._effective_requirements(core, session, user_input="ok")

    assert "If you want, I can wire OAuth next." in result
    assert "Older critic feedback." not in result
