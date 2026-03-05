"""
Author runtime: plan drafting/refinement with tool-use enforcement.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable
from typing import Any, Callable, Literal

from ...config import PlanningConfig
from .authoring.discovery import (
    AuthorDesignService,
    AuthorDiscoveryService,
    extract_paths_from_mental_model,
    is_entrypoint_like,
    is_non_trivial_source,
    is_test_or_config,
    path_tokens,
    requirements_keywords,
)
from .authoring.models import (
    ArchitectureDesign,
    AuthorResult,
    DesignRecord,
    PlanDocument,
    RepairPlan,
    RepoCandidates,
    RepoUnderstanding,
    RevisionResult,
)
from .authoring.models import (
    apply_section_updates as _apply_section_updates,
)
from .authoring.models import (
    render_markdown as _render_markdown,
)
from .authoring.pipeline import AuthorPlannerPipeline
from .authoring.repair import AuthorRepairService, extract_first_json_object, parse_plan_document
from .authoring.validation import AuthorValidationService
from .context import TokenBudgetManager, estimate_tokens
from .critic import ReviewResult
from .tools import ToolExecutor, extract_file_references
from .transport import AuthorLLMClient

apply_section_updates = _apply_section_updates
render_markdown = _render_markdown


class StageRunner:
    def __init__(
        self,
        llm_caller: Callable[..., Awaitable[tuple[Any, str]]],
        tool_executor: ToolExecutor,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        self._llm_call = llm_caller
        self._tool_executor = tool_executor
        self._emit = event_emitter

    def set_llm_caller(self, llm_caller: Callable[..., Awaitable[tuple[Any, str]]]) -> None:
        self._llm_call = llm_caller

    async def execute_tool_calls(
        self,
        *,
        stage: str,
        conversation: list[dict[str, Any]],
        content: str,
        tool_calls: list[Any],
        clarification_handler: Callable[[str, str], Awaitable[list[str]]] | None = None,
    ) -> tuple[int, bool, list[float]]:
        conversation.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": getattr(tc, "id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", "{}"),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        asked_clarification = False
        timestamps: list[float] = []
        for tc in tool_calls:
            timestamps.append(asyncio.get_running_loop().time())
            tool_name = getattr(getattr(tc, "function", None), "name", "")
            raw_args = getattr(getattr(tc, "function", None), "arguments", "{}") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
            except json.JSONDecodeError:
                parsed_args = {}
            path_hint = parsed_args.get("path") or parsed_args.get("relative_path")
            query_hint = parsed_args.get("query") or parsed_args.get("pattern")
            await self._emit(
                {
                    "type": "tool_call",
                    "name": tool_name,
                    "session_stage": stage,
                    "path": str(path_hint) if isinstance(path_hint, str) else None,
                    "query": str(query_hint) if isinstance(query_hint, str) else None,
                }
            )
            if tool_name == "ask_clarification":
                asked_clarification = True
                await self._emit(
                    {
                        "type": "clarification_needed",
                        "question": str(parsed_args.get("question", "")),
                        "context": str(parsed_args.get("context", "")),
                        "source": stage,
                    }
                )
            try:
                tool_started = asyncio.get_running_loop().time()
                if tool_name == "ask_clarification" and clarification_handler is not None:
                    answers = await clarification_handler(
                        str(parsed_args.get("question", "")),
                        str(parsed_args.get("context", "")),
                    )
                    result = {
                        "tool_call_id": getattr(tc, "id", ""),
                        "name": tool_name,
                        "result": {
                            "question": str(parsed_args.get("question", "")),
                            "context": str(parsed_args.get("context", "")),
                            "answers": answers,
                            "timed_out": len(answers) == 0,
                        },
                    }
                else:
                    result = await asyncio.to_thread(self._tool_executor.execute, tc)
                tool_elapsed_ms = (asyncio.get_running_loop().time() - tool_started) * 1000.0
            except Exception as exc:  # noqa: BLE001
                result = {
                    "tool_call_id": getattr(tc, "id", ""),
                    "name": tool_name,
                    "result": {"error": str(exc)},
                }
                tool_elapsed_ms = 0.0
            await self._emit(
                {
                    "type": "tool_result",
                    "name": tool_name,
                    "session_stage": stage,
                    "duration_ms": round(tool_elapsed_ms, 2),
                }
            )
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": json.dumps(result["result"]),
                }
            )
        return len(tool_calls), asked_clarification, timestamps

    async def run_stage(
        self,
        stage: str,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = False,
        max_tool_calls: int = 5,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
    ) -> str:
        conversation = list(messages)
        tool_call_count = 0
        while True:
            response, _ = await self._llm_call(
                conversation,
                allow_tools=allow_tools and tool_call_count < max_tool_calls,
                max_output_tokens=max_output_tokens,
                model_override=model_override,
            )
            message = response.choices[0].message
            content = str(getattr(message, "content", None) or "")
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return content.strip()
            executed_count, _, _ = await self.execute_tool_calls(
                stage=stage,
                conversation=conversation,
                content=content,
                tool_calls=tool_calls,
            )
            tool_call_count += executed_count


AUTHOR_SYSTEM_PROMPT = """You are an expert software architect creating implementation plans.

Your job is to produce a deeply practical, strategically aware plan grounded in the real codebase.
The output must look like a Cursor-quality implementation plan, not a lightweight outline.

Non-negotiable rules:
1. Verify assumptions against the repository using tools before finalizing.
2. If an assumption cannot be verified, mark it as an explicit risk.
3. Reference concrete file paths in backticks where relevant.
4. Prefer specific, actionable steps over abstract guidance.
5. Think 6-12 months ahead: call out lock-in risks, migration impacts, and maintenance cost.

Required markdown sections and order:
- # <Relevant plan title>
- ## Summary
- ## Goals
- ## Non-Goals
- ## Changes
- ## Files Changed
- ## To-dos In Order
- ## Architecture
- ## Mermaid Diagram
- ## Implementation Steps
- ## Test Strategy
- ## Rollback Plan
- ## Example Code Snippets
- ## Open Questions
- ## Design Decision Records
- ## User Stories

Additional format requirements:
- "Files Changed" must list concrete file paths and a short rationale per file.
- "To-dos In Order" must be a numbered list in execution order.
- "Example Code Snippets" must contain fenced code blocks that are relevant to planned changes.
- "Mermaid Diagram" should include a mermaid code block when architecture has meaningful flow/components.
- If a mermaid diagram is truly unnecessary, explicitly state why in that section.

Legacy quality requirements (still required):
- Goals
- Non-Goals
- User Stories
- Architecture
- Implementation Steps
- Test Strategy
- Rollback Plan
- TODOs
- Open Questions
- Design Decision Records

Implementation Steps quality bar:
- For each step, specify: what changes, where it changes (exact file paths), why that file is touched, and expected interface/signature impacts.
- Include operational details where applicable (observability, alerting, migration, rollout/rollback).
- Include explicit acceptance criteria.
"""


class AuthorAgent:
    def __init__(
        self,
        config: PlanningConfig,
        tool_executor: ToolExecutor,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        clarification_handler: Callable[[str, str], Awaitable[list[str]]] | None = None,
    ):
        self.config = config
        self.tool_executor = tool_executor
        self.event_callback = event_callback
        self.clarification_handler = clarification_handler
        self._llm_client = AuthorLLMClient(self.config, self._emit)
        self.stage_runner = StageRunner(self._llm_client.call, self.tool_executor, self._emit)
        self._discovery_service = AuthorDiscoveryService(self.tool_executor)
        self._validation_service = AuthorValidationService(self.tool_executor)
        self._design_service = AuthorDesignService(self.stage_runner, extract_first_json_object)
        self._planner_pipeline = AuthorPlannerPipeline(
            tool_executor=self.tool_executor,
            scan_repo_candidates=self.scan_repo_candidates,
            explore_repo=self.explore_repo,
            classify_complexity=self.classify_complexity,
            design_architecture=self.design_architecture,
            draft_plan=self.draft_plan,
            validate_draft=self.validate_draft,
            design_record_from_architecture=self._design_record_from_architecture,
        )

    @staticmethod
    def _excerpt(text: str, max_lines: int = 40) -> str:
        lines = text.splitlines()
        return "\n".join(lines[:max_lines]).strip()

    @staticmethod
    def _requirements_vagueness_score(text: str) -> float:
        tokens = [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]
        if not tokens:
            return 0.0
        vague_markers = {
            "maybe",
            "possibly",
            "some",
            "improve",
            "better",
            "stuff",
            "things",
            "etc",
            "roughly",
            "around",
        }
        marker_hits = sum(1 for token in tokens if token in vague_markers)
        return marker_hits / float(len(tokens))

    @staticmethod
    def _initial_draft_policy(requirements: str) -> tuple[int, int, float, int]:
        """Tune startup discovery budget from requirement complexity."""
        tokens = [token for token in re.split(r"[^a-z0-9]+", requirements.lower()) if token]
        uniq_tokens = {token for token in tokens if len(token) >= 4}
        path_mentions = re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]+", requirements)
        complexity_score = len(uniq_tokens) + (3 * len(path_mentions))
        if complexity_score >= 45:
            return 3, 4, 0.55, 1800
        if complexity_score >= 25:
            return 2, 3, 0.45, 1400
        return 2, 2, 0.35, 1200

    async def run_initial_draft(
        self,
        *,
        requirements: str,
        manifesto: str,
        manifesto_path: str,
        skills_block: str,
        recall_block: str,
        context_index: str,
        grounding_paths: set[str],
        model_override: str | None = None,
    ) -> AuthorResult:
        manifesto_excerpt = self._excerpt(manifesto, max_lines=40)
        budget = TokenBudgetManager(context_window=128_000, max_completion_tokens=4000)
        budget.enforce_required([requirements, f"Manifesto:\n{manifesto_excerpt}"])
        remaining = budget.available_prompt_tokens - estimate_tokens(requirements)
        manifesto_block, used_manifesto = budget.allocate(
            f"PROJECT MANIFESTO (excerpt):\n{manifesto_excerpt}\n\n", remaining
        )
        remaining = max(0, remaining - used_manifesto)
        skills_allocated, used_skills = budget.allocate(skills_block, remaining)
        remaining = max(0, remaining - used_skills)
        recall_allocated, used_recall = budget.allocate(recall_block, remaining)
        remaining = max(0, remaining - used_recall)
        context_index_block, _ = budget.allocate(
            f"CONTEXT INDEX:\n{context_index}\n\n",
            remaining,
        )
        initial_ratio = budget.injection_ratio(
            [requirements, manifesto_block, skills_allocated, recall_allocated, context_index_block]
        )
        if initial_ratio > budget.enforce_ratio:
            target_tokens = int(budget.context_window * budget.enforce_ratio)
            remaining = max(0, target_tokens - estimate_tokens(requirements))
            manifesto_block, used_manifesto = budget.allocate(manifesto_block, remaining)
            remaining = max(0, remaining - used_manifesto)
            skills_allocated, used_skills = budget.allocate(skills_allocated, remaining)
            remaining = max(0, remaining - used_skills)
            recall_allocated, used_recall = budget.allocate(recall_allocated, remaining)
            remaining = max(0, remaining - used_recall)
            context_index_block, _ = budget.allocate(context_index_block, remaining)
        messages = [
            {
                "role": "user",
                "content": (
                    "You are planning changes to this repository.\n\n"
                    f"REQUIREMENTS:\n{requirements}\n\n"
                    f"{manifesto_block}"
                    f"Full manifesto available at: {manifesto_path}\n\n"
                    f"{skills_allocated}"
                    f"{recall_allocated}"
                    f"{context_index_block}"
                    "IMPORTANT:\n"
                    "- Do not assume file paths.\n"
                    "- Search before referencing files.\n"
                    "- Read files before proposing changes.\n"
                    "- Pull additional memory only when needed with get_memory_block(key).\n"
                ),
            }
        ]
        require_clarification_first = self._requirements_vagueness_score(requirements) > 0.25
        max_attempts, max_tool_calls, min_grounding_ratio, max_output_tokens = self._initial_draft_policy(requirements)
        return await self.author_loop(
            messages,
            require_tool_calls=True,
            max_attempts=min(max_attempts, self.config.author_tool_rounds),
            max_tool_calls=max_tool_calls,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths,
            draft_phase="planner",
            require_clarification_first=require_clarification_first and self.clarification_handler is not None,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
            reset_tool_history=False,
        )

    @staticmethod
    def _requirements_keywords(text: str) -> set[str]:
        return requirements_keywords(text)

    @staticmethod
    def _path_tokens(path: str) -> set[str]:
        return path_tokens(path)

    @staticmethod
    def _is_non_trivial_source(path: str) -> bool:
        return is_non_trivial_source(path)

    @staticmethod
    def _is_entrypoint_like(path: str) -> bool:
        return is_entrypoint_like(path)

    @staticmethod
    def _is_test_or_config(path: str) -> bool:
        return is_test_or_config(path)

    @staticmethod
    def _extract_paths_from_mental_model(mental_model: str) -> set[str]:
        return extract_paths_from_mental_model(mental_model)

    def scan_repo_candidates(
        self,
        *,
        mental_model: str | None = None,
        max_entries_per_dir: int = 250,
    ) -> RepoCandidates:
        return self._discovery_service.scan_repo_candidates(
            mental_model=mental_model,
            max_entries_per_dir=max_entries_per_dir,
        )

    def explore_repo(
        self,
        *,
        requirements: str,
        candidates: RepoCandidates,
        mental_model: str | None = None,
        max_file_reads: int = 5,
    ) -> RepoUnderstanding:
        return self._discovery_service.explore_repo(
            requirements=requirements,
            candidates=candidates,
            mental_model=mental_model,
            max_file_reads=max_file_reads,
        )

    def classify_complexity(
        self,
        *,
        requirements: str,
        repo_understanding: RepoUnderstanding,
    ) -> Literal["simple", "moderate", "complex"]:
        return self._discovery_service.classify_complexity(
            requirements=requirements,
            repo_understanding=repo_understanding,
        )

    async def design_architecture(
        self,
        *,
        requirements: str,
        repo_understanding: RepoUnderstanding,
        model_override: str | None = None,
    ) -> ArchitectureDesign:
        return await self._design_service.design_architecture(
            requirements=requirements,
            repo_understanding=repo_understanding,
            model_override=model_override,
        )

    @staticmethod
    def _design_record_from_architecture(architecture: ArchitectureDesign) -> DesignRecord:
        return AuthorDesignService.design_record_from_architecture(architecture)

    async def draft_plan(
        self,
        *,
        requirements: str,
        repo_understanding: RepoUnderstanding,
        architecture: ArchitectureDesign | None = None,
        model_override: str | None = None,
        revision_hints: list[str] | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": AUTHOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Repository Understanding\n{json.dumps(repo_understanding.__dict__, indent=2)}\n\n"
                    f"## Architecture Design\n{json.dumps(architecture.__dict__, indent=2) if architecture else '(none)'}\n\n"
                    f"## Revision Hints\n{json.dumps(revision_hints or [], indent=2)}\n\n"
                    "Produce a complete markdown implementation plan."
                ),
            },
        ]
        return await self.stage_runner.run_stage(
            "draft_plan",
            messages,
            allow_tools=False,
            max_output_tokens=2600,
            model_override=model_override,
        )

    def validate_draft(
        self,
        *,
        plan_content: str,
        repo_understanding: RepoUnderstanding,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        min_grounding_ratio: float | None = None,
    ) -> list[str]:
        return self._validation_service.validate_draft(
            plan_content=plan_content,
            repo_understanding=repo_understanding,
            draft_phase=draft_phase,
            min_grounding_ratio=min_grounding_ratio,
        )

    def _explorer_gate_failures(self, requirements_text: str) -> list[str]:
        return self._validation_service.explorer_gate_failures(requirements_text)

    @staticmethod
    def _extract_section(content: str, heading: str) -> str:
        return AuthorValidationService.extract_section(content, heading)

    def _grounding_failures(
        self,
        plan_content: str,
        verified_paths: set[str],
        min_grounding_ratio: float,
        draft_phase: Literal["planner", "refiner"],
    ) -> tuple[list[str], set[str], float]:
        return self._validation_service.grounding_failures(
            plan_content=plan_content,
            verified_paths=verified_paths,
            min_grounding_ratio=min_grounding_ratio,
            draft_phase=draft_phase,
        )

    def _phase_failures(self, plan_content: str, draft_phase: Literal["planner", "refiner"]) -> list[str]:
        return AuthorValidationService.phase_failures(plan_content, draft_phase)

    def _completion_failures(self, plan_content: str) -> list[str]:
        return self._validation_service.completion_failures(plan_content)

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        maybe = self.event_callback(event)
        if asyncio.iscoroutine(maybe):
            await maybe

    def _fallback_plan(self, user_context: str) -> str:
        return (
            "# Plan Draft: Implementation Strategy\n\n"
            "## Summary\n"
            "- This is a fallback plan because the authoring model was unavailable.\n\n"
            "## Goals\n- Define implementation steps aligned with requirements.\n\n"
            "## Non-Goals\n- No code changes in planning phase.\n\n"
            "## Changes\n"
            "1. Establish architecture-level approach.\n"
            "2. Define implementation phases and testing strategy.\n\n"
            "## Files Changed\n"
            "- `src/path/to/module.py` — placeholder for concrete implementation changes.\n\n"
            "## To-dos In Order\n"
            "1. Inspect target modules and interfaces.\n"
            "2. Draft changes by file path.\n"
            "3. Define tests and rollout checks.\n\n"
            "## Architecture\n- Reuse existing project structure with targeted additions.\n\n"
            "## Mermaid Diagram\n"
            "```mermaid\n"
            "flowchart TD\n"
            "  requirements[Requirements] --> planDraft[PlanDraft]\n"
            "  planDraft --> implementation[Implementation]\n"
            "```\n\n"
            "## Implementation Steps\n- [ ] Inspect target modules and interfaces\n"
            "- [ ] Draft changes by file path\n"
            "- [ ] Define tests and rollout\n\n"
            "## Test Strategy\n- Add unit/integration tests for changed modules.\n\n"
            "## Rollback Plan\n- Revert changes by module and restore previous release artifact.\n\n"
            "## Example Code Snippets\n"
            "```python\n"
            "def example_change() -> None:\n"
            "    pass\n"
            "```\n\n"
            "## TODOs\n- [ ] Replace fallback with model-generated plan once LLM is available\n\n"
            "## Open Questions\n- Clarify acceptance criteria.\n\n"
            "## Design Decision Records\n"
            "- Decision: Keep fallback lightweight and deterministic.\n\n"
            "## User Stories\n"
            "- As an engineer, I can execute a deterministic fallback plan when LLM is unavailable.\n\n"
            f"Context summary:\n{user_context[:1000]}"
        )

    def _missing_required_sections(self, plan_content: str, draft_phase: Literal["planner", "refiner"]) -> list[str]:
        return AuthorValidationService.missing_required_sections(plan_content, draft_phase)

    async def _llm_call(
        self,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = True,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
    ):
        return await self._llm_client.call(
            messages=messages,
            allow_tools=allow_tools,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
        )

    @staticmethod
    def _extract_first_json_object(raw: str) -> tuple[str, str]:
        return extract_first_json_object(raw)

    @staticmethod
    def _parse_plan_document(raw: str) -> PlanDocument:
        return parse_plan_document(raw)

    async def plan_repair(
        self,
        review: ReviewResult,
        plan: PlanDocument,
        requirements: str,
        design_record: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> RepairPlan:
        return await AuthorRepairService(self._llm_call).plan_repair(
            review=review,
            plan=plan,
            requirements=requirements,
            design_record=design_record,
            model_override=model_override,
        )

    async def update_design_record(
        self,
        *,
        design_record: dict[str, Any],
        review: ReviewResult,
        requirements: str,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        return await AuthorRepairService(self._llm_call).update_design_record(
            design_record=design_record,
            review=review,
            requirements=requirements,
            model_override=model_override,
        )

    async def revise_plan(
        self,
        repair_plan: RepairPlan,
        current_plan: PlanDocument,
        requirements: str,
        design_record: dict[str, Any] | None = None,
        revision_budget: int = 3,
        model_override: str | None = None,
        simplest_possible_design: str | None = None,
    ) -> RevisionResult:
        return await AuthorRepairService(self._llm_call).revise_plan(
            repair_plan=repair_plan,
            current_plan=current_plan,
            requirements=requirements,
            design_record=design_record,
            revision_budget=revision_budget,
            model_override=model_override,
            simplest_possible_design=simplest_possible_design,
        )

    async def _run_planner_pipeline(
        self,
        *,
        requirements: str,
        min_grounding_ratio: float | None,
        grounding_paths: set[str] | None,
        model_override: str | None,
        rejection_counts: dict[str, int],
        rejection_reasons: list[dict[str, str]],
    ) -> AuthorResult:
        return await self._planner_pipeline.run(
            requirements=requirements,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths,
            model_override=model_override,
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
        )

    async def _run_legacy_authoring_loop(
        self,
        *,
        conversation: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        requirements_text: str,
        require_tool_calls: bool,
        max_attempts: int,
        max_tool_calls: int | None,
        min_grounding_ratio: float | None,
        grounding_paths: set[str] | None,
        draft_phase: Literal["planner", "refiner"],
        require_clarification_first: bool,
        max_output_tokens: int | None,
        model_override: str | None,
        rejection_counts: dict[str, int],
        rejection_reasons: list[dict[str, str]],
    ) -> tuple[AuthorResult | None, str, int]:
        asked_clarification = False
        total_tool_calls = 0
        best_non_empty_content = ""
        tool_call_timestamps: list[float] = []

        try:
            for attempt in range(max_attempts):
                if attempt == 0:
                    await self._emit({"type": "thinking", "message": "Drafting plan..."})
                allow_tools = attempt < max_attempts - 1
                if not allow_tools:
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool-use phase is complete. Do not call any tools now. "
                                "Produce your best complete draft using the evidence already gathered."
                            ),
                        }
                    )
                response, _ = await self._llm_call(
                    conversation,
                    allow_tools=allow_tools,
                    max_output_tokens=max_output_tokens,
                    model_override=model_override,
                )
                message = response.choices[0].message
                content = str(getattr(message, "content", None) or "")
                tool_calls = getattr(message, "tool_calls", None) or []
                if content.strip():
                    best_non_empty_content = content.strip()

                if tool_calls:
                    if max_tool_calls is not None and total_tool_calls >= max_tool_calls:
                        conversation.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": [
                                    {
                                        "id": getattr(tc, "id", None),
                                        "type": "function",
                                        "function": {
                                            "name": getattr(tc.function, "name", ""),
                                            "arguments": getattr(tc.function, "arguments", "{}"),
                                        },
                                    }
                                    for tc in tool_calls
                                ],
                            }
                        )
                        for tc in tool_calls:
                            conversation.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": getattr(tc, "id", ""),
                                    "name": getattr(getattr(tc, "function", None), "name", ""),
                                    "content": json.dumps(
                                        {
                                            "error": (
                                                "Tool-call budget reached for this draft. "
                                                "Use existing findings and produce the best plan now."
                                            )
                                        }
                                    ),
                                }
                            )
                        rejection_reasons.append(
                            {
                                "reason": "BUDGET_EXCEEDED",
                                "details": (f"tool-call budget reached at {total_tool_calls}/{max_tool_calls}"),
                            }
                        )
                        rejection_counts["rejected_for_budget"] += 1
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Tool-call budget reached. Do not call more tools. "
                                    "Produce your best complete draft now using gathered evidence."
                                ),
                            }
                        )
                        continue
                    executed_count, asked_now, timestamps = await self.stage_runner.execute_tool_calls(
                        stage="author",
                        conversation=conversation,
                        content=content,
                        tool_calls=tool_calls,
                        clarification_handler=self.clarification_handler,
                    )
                    total_tool_calls += executed_count
                    asked_clarification = asked_clarification or asked_now
                    tool_call_timestamps.extend(timestamps)
                    continue

                if require_tool_calls and attempt == 0 and not self.tool_executor.read_history:
                    rejection_counts["rejected_for_no_discovery"] += 1
                    rejection_reasons.append(
                        {
                            "reason": "NO_DISCOVERY",
                            "details": "initial draft attempted without read_file calls",
                        }
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "You must verify your assumptions against real files. "
                                "Call `read_file` on relevant files before finalizing."
                            ),
                        }
                    )
                    continue
                if require_tool_calls:
                    failures = self._explorer_gate_failures(requirements_text)
                    if failures:
                        rejection_counts["rejected_for_no_discovery"] += 1
                        rejection_reasons.append(
                            {
                                "reason": "NO_DISCOVERY",
                                "details": "; ".join(failures),
                            }
                        )
                        read_paths = sorted(self.tool_executor.read_history.keys())
                        if attempt < max_attempts - 1:
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Explorer gate not satisfied. Continue exploring before drafting.\n"
                                        f"Missing: {', '.join(failures)}.\n"
                                        f"Files read so far: {', '.join(read_paths) if read_paths else '(none)'}."
                                    ),
                                }
                            )
                            continue
                        await self._emit(
                            {
                                "type": "warning",
                                "message": (
                                    "Explorer gate still failing at final attempt; accepting best-effort draft."
                                ),
                            }
                        )

                plan_content = content.strip()
                if not plan_content:
                    # One extra no-tool synthesis attempt before declaring fallback.
                    if attempt == max_attempts - 1:
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Final attempt: return a complete, non-empty markdown plan now. Do not call tools."
                                ),
                            }
                        )
                        retry_response, _ = await self._llm_call(
                            conversation,
                            allow_tools=False,
                            max_output_tokens=max_output_tokens,
                            model_override=model_override,
                        )
                        retry_message = retry_response.choices[0].message
                        retry_content = str(getattr(retry_message, "content", None) or "").strip()
                        if retry_content:
                            plan_content = retry_content
                            best_non_empty_content = retry_content
                        else:
                            plan_content = ""
                    rejection_reasons.append(
                        {
                            "reason": "COMPLETION_CONSTRAINT",
                            "details": "empty draft content returned by model",
                        }
                    )
                    if plan_content:
                        # Continue validation pipeline with recovered non-empty content.
                        pass
                    elif attempt < max_attempts - 1:
                        conversation.append(
                            {
                                "role": "user",
                                "content": "Your previous response was empty. Return a complete non-empty plan draft now.",
                            }
                        )
                        continue
                    rejection_reasons.append(
                        {
                            "reason": "AUTHOR_FALLBACK",
                            "details": (
                                f"author exhausted attempts with empty final draft after {total_tool_calls} tool calls"
                            ),
                        }
                    )
                    break
                if require_clarification_first and not asked_clarification and attempt < max_attempts - 1:
                    rejection_reasons.append(
                        {
                            "reason": "NO_DISCOVERY",
                            "details": "requirements ambiguous; call ask_clarification before drafting",
                        }
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Requirements are ambiguous. Before drafting, call "
                                "`ask_clarification(question, context)` and use the answer."
                            ),
                        }
                    )
                    continue
                phase_failures = self._phase_failures(plan_content, draft_phase=draft_phase)
                if phase_failures and attempt < max_attempts - 1:
                    rejection_reasons.append(
                        {
                            "reason": "PHASE_VIOLATION",
                            "details": "; ".join(phase_failures),
                        }
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Phase boundary violation.\n"
                                f"Fix: {', '.join(phase_failures)}.\n"
                                "In planner phase, keep output high-level: goals/non-goals/files/architecture only."
                            ),
                        }
                    )
                    continue
                missing_sections = self._missing_required_sections(plan_content, draft_phase=draft_phase)
                if missing_sections and attempt < max_attempts - 1:
                    rejection_reasons.append(
                        {
                            "reason": "COMPLETION_CONSTRAINT",
                            "details": f"missing required sections: {', '.join(missing_sections)}",
                        }
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous draft is missing required Cursor-style structure. "
                                "Revise and include all required sections/content exactly. "
                                f"Missing: {', '.join(missing_sections)}"
                            ),
                        }
                    )
                    continue
                verified_paths = set(self.tool_executor.read_history.keys()) | set(grounding_paths or set())
                grounding_failures: list[str] = []
                unverified: set[str] = set()
                grounding_ratio: float | None = None
                if min_grounding_ratio is not None:
                    grounding_failures, unverified, grounding_ratio = self._grounding_failures(
                        plan_content=plan_content,
                        verified_paths=verified_paths,
                        min_grounding_ratio=min_grounding_ratio,
                        draft_phase=draft_phase,
                    )
                    if grounding_failures:
                        rejection_counts["rejected_for_grounding"] += 1
                        rejection_reasons.append(
                            {
                                "reason": "INSUFFICIENT_GROUNDING",
                                "details": "; ".join(grounding_failures),
                            }
                        )
                        if attempt < max_attempts - 1:
                            unread = sorted(unverified)
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Grounding validation failed.\n"
                                        f"Reasons: {', '.join(grounding_failures)}.\n"
                                        f"Files referenced but not read: {', '.join(unread) if unread else '(none)'}.\n"
                                        "Read the missing files before finalizing."
                                    ),
                                }
                            )
                            continue
                        await self._emit(
                            {
                                "type": "warning",
                                "message": (
                                    "Grounding validation still failing at final attempt; "
                                    "accepting best-effort draft with unverified references."
                                ),
                            }
                        )
                if min_grounding_ratio is None:
                    referenced = extract_file_references(plan_content)
                    unverified = referenced - verified_paths
                if draft_phase == "refiner":
                    completion_failures = self._completion_failures(plan_content)
                    if completion_failures and attempt < max_attempts - 1:
                        rejection_reasons.append(
                            {
                                "reason": "COMPLETION_CONSTRAINT",
                                "details": "; ".join(completion_failures),
                            }
                        )
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Completion constraints failed.\n"
                                    f"Fix: {', '.join(completion_failures)}.\n"
                                    "Return a complete plan with no placeholders."
                                ),
                            }
                        )
                        continue
                avg_read_depth = None
                if self.tool_executor.read_history:
                    avg_read_depth = sum(
                        int(meta.get("line_count", 0)) for meta in self.tool_executor.read_history.values()
                    ) / float(len(self.tool_executor.read_history))
                avg_call_spacing = None
                if len(tool_call_timestamps) > 1:
                    deltas = [
                        tool_call_timestamps[i] - tool_call_timestamps[i - 1]
                        for i in range(1, len(tool_call_timestamps))
                    ]
                    avg_call_spacing = sum(deltas) / float(len(deltas))
                return (
                    AuthorResult(
                        plan=plan_content,
                        unverified_references=unverified,
                        accessed_paths=verified_paths.copy(),
                        grounding_ratio=grounding_ratio,
                        rejection_counts=rejection_counts,
                        rejection_reasons=rejection_reasons,
                        average_read_depth=avg_read_depth,
                        average_time_between_tool_calls=avg_call_spacing,
                    ),
                    best_non_empty_content,
                    total_tool_calls,
                )
        except Exception as exc:  # noqa: BLE001
            rejection_reasons.append(
                {
                    "reason": "AUTHOR_FALLBACK",
                    "details": str(exc),
                }
            )
            await self._emit(
                {
                    "type": "warning",
                    "message": f"Author fallback used: {exc}",
                }
            )
            fallback = self._fallback_plan("\n".join(m.get("content", "") for m in messages))
            refs = extract_file_references(fallback)
            return (
                AuthorResult(
                    plan=fallback,
                    unverified_references=refs - self.tool_executor.accessed_paths,
                    accessed_paths=self.tool_executor.accessed_paths.copy(),
                    rejection_counts=rejection_counts,
                    rejection_reasons=rejection_reasons,
                ),
                best_non_empty_content,
                total_tool_calls,
            )
        return None, best_non_empty_content, total_tool_calls

    async def author_loop(
        self,
        messages: list[dict[str, Any]],
        require_tool_calls: bool = True,
        max_attempts: int | None = None,
        max_tool_calls: int | None = None,
        min_grounding_ratio: float | None = None,
        grounding_paths: set[str] | None = None,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        require_clarification_first: bool = False,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
        reset_tool_history: bool = True,
    ) -> AuthorResult:
        if max_attempts is None:
            max_attempts = self.config.author_tool_rounds
        if reset_tool_history:
            self.tool_executor.accessed_paths.clear()
            self.tool_executor.read_history.clear()
        # Preserve test/runtime monkeypatch seam: StageRunner follows AuthorAgent `_llm_call`.
        self.stage_runner.set_llm_caller(self._llm_call)
        conversation = [
            {"role": "system", "content": AUTHOR_SYSTEM_PROMPT},
            *messages,
        ]
        rejection_counts = {
            "rejected_for_no_discovery": 0,
            "rejected_for_grounding": 0,
            "rejected_for_budget": 0,
        }
        rejection_reasons: list[dict[str, str]] = []
        requirements_text = "\n".join(
            str(message.get("content", "")) for message in messages if str(message.get("role", "")) == "user"
        )

        llm_call_is_overridden = getattr(self._llm_call, "__func__", None) is not AuthorAgent._llm_call
        try:
            import litellm  # noqa: F401
        except ImportError:
            if not llm_call_is_overridden:
                fallback = self._fallback_plan("\n".join(m.get("content", "") for m in messages))
                refs = extract_file_references(fallback)
                return AuthorResult(
                    plan=fallback,
                    unverified_references=refs - self.tool_executor.accessed_paths,
                    accessed_paths=self.tool_executor.accessed_paths.copy(),
                    rejection_counts=rejection_counts,
                )

        if draft_phase == "planner":
            try:
                return await self._run_planner_pipeline(
                    requirements=requirements_text,
                    min_grounding_ratio=min_grounding_ratio,
                    grounding_paths=grounding_paths,
                    model_override=model_override,
                    rejection_counts=rejection_counts,
                    rejection_reasons=rejection_reasons,
                )
            except Exception as exc:  # noqa: BLE001
                rejection_reasons.append(
                    {
                        "reason": "PIPELINE_FALLBACK",
                        "details": f"planner pipeline failed, using legacy loop: {exc}",
                    }
                )

        legacy_result, best_non_empty_content, total_tool_calls = await self._run_legacy_authoring_loop(
            conversation=conversation,
            messages=messages,
            requirements_text=requirements_text,
            require_tool_calls=require_tool_calls,
            max_attempts=max_attempts,
            max_tool_calls=max_tool_calls,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths,
            draft_phase=draft_phase,
            require_clarification_first=require_clarification_first,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
        )
        if legacy_result is not None:
            return legacy_result

        if best_non_empty_content:
            recovered = best_non_empty_content
            referenced = extract_file_references(recovered)
            verified_paths = set(self.tool_executor.read_history.keys()) | set(grounding_paths or set())
            rejection_reasons.append(
                {
                    "reason": "BEST_EFFORT_RECOVERY",
                    "details": "recovered non-empty draft after fallback path",
                }
            )
            return AuthorResult(
                plan=recovered,
                unverified_references=referenced - verified_paths,
                accessed_paths=verified_paths.copy(),
                rejection_counts=rejection_counts,
                rejection_reasons=rejection_reasons,
            )

        if not any(item.get("reason") == "AUTHOR_FALLBACK" for item in rejection_reasons):
            rejection_reasons.append(
                {
                    "reason": "AUTHOR_FALLBACK",
                    "details": (
                        f"author exhausted attempts without producing a finalized draft (tool_calls={total_tool_calls})"
                    ),
                }
            )
        fallback = self._fallback_plan("")
        refs = extract_file_references(fallback)
        return AuthorResult(
            plan=fallback,
            unverified_references=refs - self.tool_executor.accessed_paths,
            accessed_paths=self.tool_executor.accessed_paths.copy(),
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
        )
