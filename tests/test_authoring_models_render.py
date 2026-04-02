from prscope.planning.runtime.authoring.models import (
    PlanDocument,
    open_questions_visible_in_markdown,
    render_markdown,
)


def test_open_questions_visible_in_markdown() -> None:
    assert not open_questions_visible_in_markdown("")
    assert not open_questions_visible_in_markdown("- None.")
    assert not open_questions_visible_in_markdown("- none")
    assert open_questions_visible_in_markdown("- Which logging library should we use?")


def test_render_markdown_includes_open_questions_when_meaningful() -> None:
    plan = PlanDocument(
        title="T",
        summary="S",
        goals="G",
        non_goals="N",
        files_changed="F",
        architecture="A",
        implementation_steps="I",
        test_strategy="T",
        rollback_plan="R",
        open_questions="- None.",
    )
    out = render_markdown(plan)
    assert "Open Questions" not in out

    plan2 = PlanDocument(
        title="T",
        summary="S",
        goals="G",
        non_goals="N",
        files_changed="F",
        architecture="A",
        implementation_steps="I",
        test_strategy="T",
        rollback_plan="R",
        open_questions="- Confirm bcrypt cost factor with security review.",
    )
    out2 = render_markdown(plan2)
    assert "## Open Questions" in out2
    assert "bcrypt" in out2
