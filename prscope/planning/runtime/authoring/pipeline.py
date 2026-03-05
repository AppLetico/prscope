from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..tools import extract_file_references
from .models import ArchitectureDesign, AuthorResult


class AuthorPlannerPipeline:
    def __init__(
        self,
        *,
        tool_executor: Any,
        scan_repo_candidates: Callable[..., Any],
        explore_repo: Callable[..., Any],
        classify_complexity: Callable[..., Any],
        design_architecture: Callable[..., Awaitable[Any]],
        draft_plan: Callable[..., Awaitable[str]],
        validate_draft: Callable[..., Any],
        design_record_from_architecture: Callable[[Any], Any],
    ) -> None:
        self._tool_executor = tool_executor
        self._scan_repo_candidates = scan_repo_candidates
        self._explore_repo = explore_repo
        self._classify_complexity = classify_complexity
        self._design_architecture = design_architecture
        self._draft_plan = draft_plan
        self._validate_draft = validate_draft
        self._design_record_from_architecture = design_record_from_architecture

    async def run(
        self,
        *,
        requirements: str,
        min_grounding_ratio: float | None,
        grounding_paths: set[str] | None,
        model_override: str | None,
        rejection_counts: dict[str, int],
        rejection_reasons: list[dict[str, str]],
    ) -> AuthorResult:
        mental_model = ""
        if callable(self._tool_executor.memory_block_callback):
            try:
                mental_payload = self._tool_executor.memory_block_callback("mental_model")
                mental_model = str(mental_payload.get("content", "")).strip()
            except Exception:  # noqa: BLE001
                mental_model = ""
        candidates = self._scan_repo_candidates(mental_model=mental_model)
        repo_understanding = self._explore_repo(
            requirements=requirements,
            candidates=candidates,
            mental_model=mental_model,
        )
        complexity = self._classify_complexity(
            requirements=requirements,
            repo_understanding=repo_understanding,
        )
        architecture: ArchitectureDesign | None = None
        if complexity in {"moderate", "complex"}:
            architecture = await self._design_architecture(
                requirements=requirements,
                repo_understanding=repo_understanding,
                model_override=model_override,
            )
        plan_content = await self._draft_plan(
            requirements=requirements,
            repo_understanding=repo_understanding,
            architecture=architecture,
            model_override=model_override,
        )
        validation_failures = self._validate_draft(
            plan_content=plan_content,
            repo_understanding=repo_understanding,
            draft_phase="planner",
            min_grounding_ratio=min_grounding_ratio,
        )
        if validation_failures:
            rejection_reasons.append(
                {
                    "reason": "COMPLETION_CONSTRAINT",
                    "details": "; ".join(validation_failures),
                }
            )
            plan_content = await self._draft_plan(
                requirements=requirements,
                repo_understanding=repo_understanding,
                architecture=architecture,
                model_override=model_override,
                revision_hints=validation_failures,
            )
        referenced = extract_file_references(plan_content)
        verified_paths = (
            set(repo_understanding.file_contents.keys())
            | set(repo_understanding.entrypoints)
            | set(repo_understanding.core_modules)
            | set(repo_understanding.relevant_modules)
            | set(repo_understanding.relevant_tests)
            | set(grounding_paths or set())
            | set(self._tool_executor.read_history.keys())
        )
        return AuthorResult(
            plan=plan_content,
            unverified_references=referenced - verified_paths,
            accessed_paths=verified_paths,
            design_record=(
                self._design_record_from_architecture(architecture).__dict__ if architecture is not None else None
            ),
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
        )
