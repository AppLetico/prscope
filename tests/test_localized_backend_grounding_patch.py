from prscope.planning.runtime.authoring.models import PlanDocument, RepoUnderstanding, render_markdown
from prscope.planning.runtime.authoring.validation import (
    AuthorValidationService,
    patch_plan_document_localized_backend_grounding,
)
from prscope.planning.runtime.pipeline.stages import PlanningStages


def test_patch_plan_document_localized_backend_grounding_fills_missing_refs() -> None:
    plan = PlanDocument(
        title="T",
        summary="S",
        goals="- g",
        non_goals="- n",
        files_changed="- `src/prscope/web/frontend/src/pages/PlanningView.tsx`",
        architecture="If the payload shape needs a small adjustment, keep the backend response change localized.",
        implementation_steps="1. Do the thing.",
        test_strategy="- t",
        rollback_plan="- r",
        open_questions="- What payload fields need to be added?",
    )
    ru = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=["src/prscope/web/api.py"],
        relevant_modules=["src/prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="",
        risks=[],
        file_contents={},
    )
    requirements = (
        "FastAPI localized change: adjust session snapshot payload and response serialization for PlanningView."
    )
    md_before = render_markdown(plan)
    assert AuthorValidationService.localized_backend_grounding_failures(md_before, ru, requirements)

    patched = patch_plan_document_localized_backend_grounding(plan, ru, requirements)
    md_after = render_markdown(patched)
    assert not AuthorValidationService.localized_backend_grounding_failures(md_after, ru, requirements)
    assert "src/prscope/web/api.py" in md_after
    assert "test_web_api_models.py" in md_after


def test_localized_backend_patch_requires_impl_restore_for_refinement_validation() -> None:
    """Patch appends Files Changed only; pipeline must re-sync Implementation Steps."""
    plan = PlanDocument(
        title="T",
        summary="S",
        goals="- g",
        non_goals="- n",
        files_changed="- `src/prscope/web/frontend/src/pages/PlanningView.tsx`",
        architecture="If the payload shape needs a small adjustment, keep the backend response change localized.",
        implementation_steps="1. Do the thing.",
        test_strategy="- t",
        rollback_plan="- r",
        open_questions="- What payload fields need to be added?",
    )
    ru = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=["src/prscope/web/api.py"],
        relevant_modules=["src/prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="",
        risks=[],
        file_contents={},
    )
    requirements = (
        "FastAPI localized change: adjust session snapshot payload and response serialization for PlanningView."
    )
    patched = patch_plan_document_localized_backend_grounding(plan, ru, requirements)
    md_patched = render_markdown(patched)
    validator = AuthorValidationService(None)
    before = validator.validate_refinement_result(
        plan_content=md_patched,
        repo_understanding=ru,
        verified_paths_extra=set(),
        requirements_text=requirements,
    )
    assert before.failure_messages
    assert any("Files Changed entries missing from Implementation Steps" in m for m in before.failure_messages)

    restored = PlanningStages._restore_missing_implementation_refs(plan=patched, current_plan=plan)
    md_ok = render_markdown(restored)
    after = validator.validate_refinement_result(
        plan_content=md_ok,
        repo_understanding=ru,
        verified_paths_extra=set(),
        requirements_text=requirements,
    )
    assert not any("Files Changed entries missing from Implementation Steps" in m for m in after.failure_messages)
