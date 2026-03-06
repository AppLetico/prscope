from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..tools import extract_file_references
from .models import AuthorResult

logger = logging.getLogger(__name__)


class AuthorPlannerPipeline:
    def __init__(
        self,
        *,
        tool_executor: Any,
        scan_repo_candidates: Callable[..., Any],
        explore_repo: Callable[..., Any],
        classify_complexity: Callable[..., Any],
        draft_plan: Callable[..., Awaitable[str]],
        validate_draft: Callable[..., Any],
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._scan_repo_candidates = scan_repo_candidates
        self._explore_repo = explore_repo
        self._classify_complexity = classify_complexity
        self._draft_plan = draft_plan
        self._validate_draft = validate_draft
        self._emit = event_emitter

    async def _emit_progress(
        self,
        *,
        stage: str,
        step: str,
        **extra: Any,
    ) -> None:
        if self._emit is None:
            return
        payload: dict[str, Any] = {"type": "setup_progress", "step": step, "draft_stage": stage}
        payload.update(extra)
        await self._emit(payload)

    async def run(
        self,
        *,
        requirements: str,
        min_grounding_ratio: float | None,
        grounding_paths: set[str] | None,
        model_override: str | None,
        rejection_counts: dict[str, int],
        rejection_reasons: list[dict[str, str]],
        timeout_seconds_override: int | Callable[[], int] | None,
    ) -> AuthorResult:
        pipeline_start = time.perf_counter()
        mental_model = ""
        if callable(self._tool_executor.memory_block_callback):
            try:
                mental_payload = self._tool_executor.memory_block_callback("mental_model")
                mental_model = str(mental_payload.get("content", "")).strip()
            except Exception:  # noqa: BLE001
                mental_model = ""

        seeded_paths = {path for path in (grounding_paths or set()) if str(path).strip()}
        scan_step = (
            "Draft: reusing discovery evidence..." if seeded_paths else "Draft: scanning repository candidates..."
        )
        await self._emit_progress(stage="planner_scan", step=scan_step)
        t0 = time.perf_counter()
        candidates = self._scan_repo_candidates(mental_model=mental_model, seed_paths=seeded_paths or None)
        t1 = time.perf_counter()
        await self._emit_progress(stage="planner_explore", step="Draft: reading the most relevant files...")
        repo_understanding = self._explore_repo(
            requirements=requirements,
            candidates=candidates,
            mental_model=mental_model,
        )
        t2 = time.perf_counter()
        await self._emit_progress(stage="planner_classify", step="Draft: sizing implementation complexity...")
        complexity = self._classify_complexity(
            requirements=requirements,
            repo_understanding=repo_understanding,
        )
        t3 = time.perf_counter()

        await self._emit_progress(
            stage="planner_draft",
            step="Draft: writing the first implementation plan...",
            complexity=complexity,
        )
        t4 = time.perf_counter()
        plan_content = await self._draft_plan(
            requirements=requirements,
            repo_understanding=repo_understanding,
            architecture=None,
            draft_phase="planner",
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
        )
        draft_elapsed = time.perf_counter() - t4

        validation_failures = self._validate_draft(
            plan_content=plan_content,
            repo_understanding=repo_understanding,
            draft_phase="planner",
            min_grounding_ratio=min_grounding_ratio,
            verified_paths_extra=set(grounding_paths or set()),
        )
        redraft_elapsed = 0.0
        if validation_failures:
            logger.warning(
                "planner_pipeline redraft_requested failures=%s",
                "; ".join(validation_failures),
            )
            await self._emit_progress(
                stage="planner_redraft",
                step="Draft: revising outline to satisfy planner guardrails...",
                failures=validation_failures,
            )
            redraft_started = time.perf_counter()
            plan_content = await self._draft_plan(
                requirements=requirements,
                repo_understanding=repo_understanding,
                architecture=None,
                draft_phase="planner",
                model_override=model_override,
                revision_hints=validation_failures,
                timeout_seconds_override=timeout_seconds_override,
            )
            redraft_elapsed = time.perf_counter() - redraft_started
            validation_failures = self._validate_draft(
                plan_content=plan_content,
                repo_understanding=repo_understanding,
                draft_phase="planner",
                min_grounding_ratio=min_grounding_ratio,
                verified_paths_extra=set(grounding_paths or set()),
            )
            if validation_failures:
                rejection_reasons.append(
                    {
                        "reason": "COMPLETION_CONSTRAINT",
                        "details": "; ".join(validation_failures),
                    }
                )
                logger.warning(
                    "planner_pipeline skip_redraft failures=%s",
                    "; ".join(validation_failures),
                )

        total = time.perf_counter() - pipeline_start
        logger.info(
            "planner_pipeline total=%.1fs scan=%.2fs explore=%.2fs classify=%.2fs "
            "arch=%.2fs draft=%.2fs redraft=%.2fs complexity=%s",
            total,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            0.0,
            draft_elapsed,
            redraft_elapsed,
            complexity,
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
            design_record=None,
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
        )
