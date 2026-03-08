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
    AttemptContext,
    ArchitectureDesign,
    AuthorResult,
    DesignRecord,
    EvidenceBundle,
    PlanDocument,
    RepairPlan,
    RepoCandidates,
    RepoUnderstanding,
    RevisionResult,
    ValidationResult,
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
from .tools import ToolExecutor
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
                tool_elapsed_ms = (asyncio.get_running_loop().time() - tool_started) * 1000.0
                result = {
                    "tool_call_id": getattr(tc, "id", ""),
                    "name": tool_name,
                    "result": {"error": str(exc)},
                }
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
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ) -> str:
        conversation = list(messages)
        tool_call_count = 0
        while True:
            response, _ = await self._llm_call(
                conversation,
                allow_tools=allow_tools and tool_call_count < max_tool_calls,
                max_output_tokens=max_output_tokens,
                model_override=model_override,
                timeout_seconds_override=timeout_seconds_override,
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


PLANNER_SYSTEM_PROMPT = """You are an expert software architect creating the first grounded planning draft.

Your job is to produce a concise, high-signal outline grounded in the real codebase.
This planner draft is an intermediate artifact, not the final implementation document.

Non-negotiable rules:
1. Verify assumptions against the repository evidence already provided.
2. If an assumption cannot be verified, mark it as an explicit risk or open question.
3. Reference only concrete file paths in backticks.
4. Keep the draft concise and avoid speculative implementation detail.
5. Do not include code fences, example snippets, or mermaid diagrams in this phase.
6. If the repository evidence shows the requested route, feature, or integration point already exists, plan to modify the existing implementation instead of creating a parallel one.
7. For endpoint or route changes, mention both the expected success response and minimal failure/error handling, but do not broaden the design into dependency checks, authentication, or platform work unless the requirements require that.
8. For lightweight endpoint requests, keep observability, logging, and documentation notes proportional; do not turn them into explicit workstreams unless the user asked for them.
9. For localized UI or API-wiring requests, prefer the owning page/component and existing client/helper files already shown in repository evidence. Do not pull in planning runtime, discovery, or unrelated backend modules unless the evidence directly links them to the requested behavior.
10. When tests are requested for a localized UI change, prefer an existing adjacent page/container test over an unrelated component test. If no verified test file is clearly related, do not invent a new path; surface the missing test target as a risk or open question instead.
11. For localized UI or API-wiring requests, do not add observability, logging, telemetry, rollout controls, or platform notes unless the requirements or verified repository evidence explicitly require them.

Required markdown sections and order:
- # <Relevant plan title>
- ## Summary
- ## Goals
- ## Non-Goals
- ## Changes
- ## Files Changed
- ## Architecture
- ## Open Questions

Additional format requirements:
- "Files Changed" must list only verified concrete file paths and a short rationale per file.
- "Architecture" should explain the minimal design direction, key interfaces, and observability/runtime concerns when relevant.
- "Open Questions" should only include unresolved decisions that cannot be answered from the codebase.

Strict exclusions for this phase:
- Do NOT include "Implementation Steps".
- Do NOT include "Test Strategy".
- Do NOT include "Rollback Plan".
- Do NOT include "Example Code Snippets".
- Do NOT include detailed numbered execution steps.
"""


REFINER_SYSTEM_PROMPT = """You are an expert software architect creating implementation plans.

Your job is to produce a deeply practical, strategically aware plan grounded in the real codebase.
The output must look like a Cursor-quality implementation plan, not a lightweight outline.

Non-negotiable rules:
1. Verify assumptions against the repository using tools before finalizing.
2. If an assumption cannot be verified, mark it as an explicit risk.
3. Reference concrete file paths in backticks where relevant.
4. Prefer specific, actionable steps over abstract guidance.
5. Think 6-12 months ahead: call out lock-in risks, migration impacts, and maintenance cost.
6. If the repository evidence shows the requested route, feature, or integration point already exists, revise the existing implementation instead of creating a duplicate one unless the requirements explicitly demand a new parallel path.
7. For localized UI or API-wiring requests, keep the plan anchored to the owning page/component and existing client/helper files already shown in repository evidence. Do not introduce planning runtime modules, new service layers, or state-management machinery unless the requirements or verified evidence explicitly require them.
8. When tests are requested for a localized UI change, prefer an existing adjacent page/container test over an unrelated component test. If no verified test file is clearly related, call that out explicitly instead of inventing a new test path.
9. For localized UI or API-wiring requests, keep observability, logging, telemetry, rollout controls, and platform hardening proportional. Do not add them as workstreams or architecture notes unless the requirements explicitly ask for them.

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
            draft_plan=self.draft_plan,
            validate_draft=self.validate_draft_result,
            self_review_draft=self.self_review_draft,
            event_emitter=self._emit,
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
        timeout_seconds_override: int | Callable[[], int] | None = None,
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
        _, _, min_grounding_ratio, _ = self._initial_draft_policy(requirements)
        return await self.author_loop(
            messages,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
            requirements_text_override=requirements,
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
        seed_paths: set[str] | None = None,
        max_entries_per_dir: int = 250,
    ) -> RepoCandidates:
        return self._discovery_service.scan_repo_candidates(
            mental_model=mental_model,
            seed_paths=seed_paths,
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
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ) -> ArchitectureDesign:
        return await self._design_service.design_architecture(
            requirements=requirements,
            repo_understanding=repo_understanding,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
        )

    @staticmethod
    def _design_record_from_architecture(architecture: ArchitectureDesign) -> DesignRecord:
        return AuthorDesignService.design_record_from_architecture(architecture)

    async def draft_plan(
        self,
        *,
        requirements: str,
        repo_understanding: RepoUnderstanding,
        evidence_bundle: EvidenceBundle | None = None,
        attempt_context: AttemptContext | None = None,
        architecture: ArchitectureDesign | None = None,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        model_override: str | None = None,
        revision_hints: list[str] | None = None,
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ) -> str:
        system_prompt = PLANNER_SYSTEM_PROMPT if draft_phase == "planner" else REFINER_SYSTEM_PROMPT
        prioritized_verified_paths: list[str] = []
        for group in (
            repo_understanding.relevant_modules,
            repo_understanding.relevant_tests,
            list(repo_understanding.file_contents.keys()),
            repo_understanding.entrypoints,
            repo_understanding.core_modules,
        ):
            for path in group:
                normalized = str(path).strip()
                if normalized and normalized not in prioritized_verified_paths:
                    prioritized_verified_paths.append(normalized)
        verified_paths_block = (
            "\n".join(f"- `{path}`" for path in prioritized_verified_paths[:40]) or "- (none captured)"
        )
        evidence_payload = {
            "relevant_files": list((evidence_bundle.relevant_files if evidence_bundle else ())[:12]),
            "existing_components": list((evidence_bundle.existing_components if evidence_bundle else ())[:12]),
            "test_targets": list((evidence_bundle.test_targets if evidence_bundle else ())[:8]),
            "related_modules": list((evidence_bundle.related_modules if evidence_bundle else ())[:8]),
            "existing_routes_or_helpers": list((evidence_bundle.existing_routes_or_helpers if evidence_bundle else ())[:10]),
            "evidence_notes": list((evidence_bundle.evidence_notes if evidence_bundle else ())[:8]),
        }
        attempt_payload = {
            "attempt_number": attempt_context.attempt_number if attempt_context else 1,
            "previous_failures": list(attempt_context.previous_failures if attempt_context else ()),
            "revision_hints": list(attempt_context.revision_hints if attempt_context else tuple(revision_hints or [])),
            "elapsed_ms": attempt_context.elapsed_ms if attempt_context else 0,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Verified File Paths\n{verified_paths_block}\n\n"
                    f"## Structured Evidence\n{json.dumps(evidence_payload, indent=2)}\n\n"
                    f"## Attempt Context\n{json.dumps(attempt_payload, indent=2)}\n\n"
                    f"## Repository Understanding\n{json.dumps(repo_understanding.__dict__, indent=2)}\n\n"
                    f"## Architecture Design\n{json.dumps(architecture.__dict__, indent=2) if architecture else '(none)'}\n\n"
                    f"## Revision Hints\n{json.dumps(revision_hints or [], indent=2)}\n\n"
                    "## Grounding Rules\n"
                    "- Use only exact spellings from Verified File Paths when naming files.\n"
                    "- Do not invent new test filenames or modules.\n"
                    "- If the provided evidence already shows the requested behavior exists, plan to extend that implementation rather than adding a duplicate.\n"
                    "- If existing evidence names the same route, endpoint, handler, or test target, do not describe the change as introducing a brand-new feature.\n"
                    "- Treat Structured Evidence as the authoritative summary of relevant files, helpers, and tests for this draft attempt.\n"
                    "- Reuse the same Structured Evidence across retries instead of inventing new repository facts.\n"
                    "- When Structured Evidence lists existing routes or helpers, reference those exact reuse points explicitly instead of describing generic API access.\n"
                    "- When Structured Evidence lists existing helper names for export, download, snapshot, or diagnostics flows, mention those exact helper names in the plan instead of saying only 'existing helpers'.\n"
                    "- When Structured Evidence lists existing test_targets for a localized UI/API change, name at least one of those exact test files in the plan unless the requirements explicitly exclude tests.\n"
                    "- For localized UI/API work, do not add observability, logging, telemetry, rollout controls, or platform notes unless the requirements explicitly ask for them.\n"
                    "- If the requirements only say to keep an existing component behavior intact during rollout, treat that component as a compatibility constraint or test target unless verified evidence shows it must be edited.\n"
                    "- For localized UI/API work, keep file references focused on the owning component/page and existing client helpers already present in Verified File Paths.\n"
                    "- Do not reference planning runtime or discovery modules for frontend wiring work unless those modules are explicitly part of the verified evidence for the requested behavior.\n\n"
                    + (
                        "Produce a concise grounded planner draft that satisfies the planner-phase constraints. "
                        "When you reference files, use the exact spellings from Verified File Paths."
                        if draft_phase == "planner"
                        else "Produce a complete markdown implementation plan. "
                        "When you reference files, use the exact spellings from Verified File Paths."
                    )
                ),
            },
        ]
        return await self.stage_runner.run_stage(
            "draft_plan",
            messages,
            allow_tools=False,
            max_output_tokens=2600,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
        )

    def validate_draft(
        self,
        *,
        plan_content: str,
        repo_understanding: RepoUnderstanding,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        min_grounding_ratio: float | None = None,
        verified_paths_extra: set[str] | None = None,
        requirements_text: str | None = None,
    ) -> list[str]:
        return list(
            self.validate_draft_result(
                plan_content=plan_content,
                repo_understanding=repo_understanding,
                draft_phase=draft_phase,
                min_grounding_ratio=min_grounding_ratio,
                verified_paths_extra=verified_paths_extra,
                requirements_text=requirements_text,
            ).failure_messages
        )

    def validate_draft_result(
        self,
        *,
        plan_content: str,
        repo_understanding: RepoUnderstanding,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        min_grounding_ratio: float | None = None,
        verified_paths_extra: set[str] | None = None,
        requirements_text: str | None = None,
    ) -> ValidationResult:
        return self._validation_service.validate_draft_result(
            plan_content=plan_content,
            repo_understanding=repo_understanding,
            draft_phase=draft_phase,
            min_grounding_ratio=min_grounding_ratio,
            verified_paths_extra=verified_paths_extra,
            requirements_text=requirements_text,
        )

    @staticmethod
    def _parse_revision_hints(raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            json_block, _ = extract_first_json_object(text)
            payload = json.loads(json_block)
            if isinstance(payload, dict):
                items = payload.get("revision_hints", [])
            elif isinstance(payload, list):
                items = payload
            else:
                items = []
            hints = [str(item).strip() for item in items if str(item).strip()]
            if hints:
                return hints[:4]
        except Exception:  # noqa: BLE001
            pass
        parsed = [
            line.lstrip("-*0123456789. ").strip()
            for line in text.splitlines()
            if line.strip().startswith(("-", "*")) or re.match(r"^\s*\d+\.\s+", line)
        ]
        return [hint for hint in parsed if hint][:4]

    async def self_review_draft(
        self,
        *,
        requirements: str,
        plan_content: str,
        evidence_bundle: EvidenceBundle,
        validation_result: ValidationResult,
        attempt_context: AttemptContext,
        model_override: str | None = None,
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ) -> list[str]:
        if validation_result.ok:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are reviewing a draft implementation plan.\n"
                    "Return JSON with one key: revision_hints.\n"
                    "revision_hints must be a list of at most 4 concise strings.\n"
                    "Focus only on these defect classes: grounding errors, missing sections, missing test targets, "
                    "failure to reuse verified implementation, and overspecified architecture for localized changes.\n"
                    "Do not rewrite the plan. Do not introduce new architecture. Do not invent new repository facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Draft Plan\n{plan_content}\n\n"
                    f"## Validation Failures\n{json.dumps(list(validation_result.failure_messages), indent=2)}\n\n"
                    f"## Structured Evidence\n{json.dumps(evidence_bundle.__dict__, indent=2)}\n\n"
                    f"## Attempt Context\n{json.dumps(attempt_context.__dict__, indent=2)}"
                ),
            },
        ]
        raw = await self.stage_runner.run_stage(
            "self_review_draft",
            messages,
            allow_tools=False,
            max_output_tokens=400,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
        )
        return self._parse_revision_hints(raw)

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

    def _missing_required_sections(self, plan_content: str, draft_phase: Literal["planner", "refiner"]) -> list[str]:
        return AuthorValidationService.missing_required_sections(plan_content, draft_phase)

    async def _llm_call(
        self,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = True,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ):
        return await self._llm_client.call(
            messages=messages,
            allow_tools=allow_tools,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
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
        reconsideration_candidates: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
    ) -> RepairPlan:
        return await AuthorRepairService(self._llm_call).plan_repair(
            review=review,
            plan=plan,
            requirements=requirements,
            design_record=design_record,
            reconsideration_candidates=reconsideration_candidates,
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
        revision_hints: list[str] | None = None,
        reconsideration_candidates: list[dict[str, Any]] | None = None,
    ) -> RevisionResult:
        return await AuthorRepairService(self._llm_call).revise_plan(
            repair_plan=repair_plan,
            current_plan=current_plan,
            requirements=requirements,
            design_record=design_record,
            revision_budget=revision_budget,
            model_override=model_override,
            simplest_possible_design=simplest_possible_design,
            revision_hints=revision_hints,
            reconsideration_candidates=reconsideration_candidates,
        )

    def incremental_grounding_failures(
        self,
        *,
        previous_plan_content: str,
        updated_plan_content: str,
        verified_paths: set[str],
    ) -> list[str]:
        return self._validation_service.incremental_grounding_failures(
            previous_plan_content=previous_plan_content,
            updated_plan_content=updated_plan_content,
            verified_paths=verified_paths,
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
        timeout_seconds_override: int | Callable[[], int] | None,
    ) -> AuthorResult:
        return await self._planner_pipeline.run(
            requirements=requirements,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths,
            model_override=model_override,
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
            timeout_seconds_override=timeout_seconds_override,
        )

    async def author_loop(
        self,
        messages: list[dict[str, Any]],
        require_tool_calls: bool = True,
        max_attempts: int | None = None,
        max_tool_calls: int | None = None,
        min_grounding_ratio: float | None = None,
        grounding_paths: set[str] | None = None,
        draft_phase: Literal["planner", "refiner"] = "planner",
        require_clarification_first: bool = False,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
        timeout_seconds_override: int | Callable[[], int] | None = None,
        requirements_text_override: str | None = None,
        reset_tool_history: bool = True,
    ) -> AuthorResult:
        if reset_tool_history:
            self.tool_executor.accessed_paths.clear()
            self.tool_executor.read_history.clear()
        self.stage_runner.set_llm_caller(self._llm_call)
        rejection_counts: dict[str, int] = {
            "rejected_for_no_discovery": 0,
            "rejected_for_grounding": 0,
            "rejected_for_budget": 0,
        }
        rejection_reasons: list[dict[str, str]] = []
        requirements_text = str(requirements_text_override or "").strip() or "\n".join(
            str(message.get("content", "")) for message in messages if str(message.get("role", "")) == "user"
        )

        pipeline_started = asyncio.get_running_loop().time()
        result = await self._run_planner_pipeline(
            requirements=requirements_text,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths,
            model_override=model_override,
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
            timeout_seconds_override=timeout_seconds_override,
        )
        pipeline_elapsed = asyncio.get_running_loop().time() - pipeline_started
        await self._emit({"type": "thinking", "message": f"Plan drafted via pipeline ({pipeline_elapsed:.1f}s)."})
        return result
