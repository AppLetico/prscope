"""
Chat-first requirements discovery flow.

On first user message the LLM scans the codebase via tools before responding,
then asks only questions that cannot be answered from code.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, Callable

from ...config import PlanningConfig
from ...memory import MemoryStore
from .discovery_support import (
    BOOTSTRAP_ROUTE_REGEX,
    CODE_SIGNALS,
    FRAMEWORKS,
    CodeSignal,
    DiscoveryBootstrapService,
    DiscoveryLLMClient,
    DiscoveryQuestion,
    DiscoveryTurnResult,
    Evidence,
    FeatureIntent,
    Framework,
    IndexedMatch,
    QuestionOption,
    SignalIndex,
    aggregate_evidence,
    build_existing_endpoint_deep_summary,
    build_existing_feature_enhancement_summary,
    build_signal_index,
    detect_architecture,
    detect_code_signals,
    detect_framework,
    drop_redundant_framework_questions,
    existing_feature_evidence_lines,
    extract_feature_intent,
    format_evidence_line,
    format_grep_matches,
    functional_summary_from_snippet,
    is_concrete_enhancement_request,
    location_score,
    merge_feature_evidence,
    parse_evidence_reference,
    parse_existing_endpoint_followup_choice,
    parse_questions,
    route_file_score,
    select_scan_directories,
    should_bootstrap_scan,
    summarize_endpoint_snippet,
    try_extract_completion,
)
from .tools import ToolExecutor

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM_PROMPT = """You are a planning assistant helping scope a software implementation plan.

## Your Process

**Step 1 — Research first (ALWAYS on the first user message):**
Use list_files, read_file, and grep_code to understand the project before asking anything.
Read: README, key source files, package manifests, existing patterns relevant to the request.
Do not skip this step. Hallucinating project structure is worse than asking.
If the request mentions endpoints/routes/APIs, inspect backend route handlers (not only frontend files) before asking.

**Step 2 — Ask only what code can't tell you:**
After scanning, ask ONLY 2-3 questions that require a human decision:
- Priorities and trade-offs
- Acceptance criteria or definition of done
- Business constraints not visible in code
- Architectural preferences where multiple valid approaches exist

Never ask questions whose answers are already in the codebase.
If code scan evidence clearly identifies the backend/API framework (for example FastAPI imports
or route decorators), do NOT ask which backend/framework is being used.

## Question Format (REQUIRED)

You MUST format every question with lettered multiple-choice options:

**Q1: <your question>**
A) <concrete option — the most common sensible default>
B) <concrete alternative>
C) <another concrete alternative>
D) Other — describe your preference

Rules:
- Always provide A–C or A–D options covering the realistic choices for this specific codebase.
- Always end with an "Other" option so the user can type a custom answer.
- Options must be specific and actionable (e.g. "A) Add a `lastSeen` column to the `users` table", not "A) Use the database").
- Do NOT ask open-ended questions without options.

**Step 3 — Complete discovery:**
After 1-2 exchanges when you have enough context, return ONLY this JSON (no extra text after it):
{"discovery":"complete","summary":"<comprehensive requirements summary including relevant file paths and constraints>"}

## Rules
- Research first, questions second.
- Max 3 questions per turn.
- Batch questions in one turn: when discovery is still open, ask all unresolved questions together
  in a single response (usually 3), not one-by-one across multiple turns.
- Do not ask "Q2" or "Q3" in later turns if those could have been asked in the prior question batch.
- Only ask a follow-up single question when a prior answer is genuinely ambiguous or contradictory.
- Never repeat a question the user already answered in a previous turn.
- If the user's message already answers all open questions, complete immediately.
- Reference specific file paths you found when relevant.
"""


class DiscoveryManager:
    def __init__(
        self,
        config: PlanningConfig,
        tool_executor: ToolExecutor,
        memory: MemoryStore,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ):
        self.config = config
        self.tool_executor = tool_executor
        self.memory = memory
        self.turn_counts_by_session: dict[str, int] = {}
        self.bootstrap_insights_by_session: dict[str, dict[str, Any]] = {}
        self._active_discovery_session_id: str | None = None
        self.event_callback = event_callback
        self._bootstrap_service = DiscoveryBootstrapService(self)
        self._llm_client = DiscoveryLLMClient(self)

    def reset_session(self, session_id: str) -> None:
        self.turn_counts_by_session[session_id] = 0

    def clear_session(self, session_id: str) -> None:
        self.turn_counts_by_session.pop(session_id, None)
        if hasattr(self, "bootstrap_insights_by_session"):
            self.bootstrap_insights_by_session.pop(session_id, None)
        if getattr(self, "_active_discovery_session_id", None) == session_id:
            self._active_discovery_session_id = None

    def bootstrap_seed_paths(self, session_id: str) -> set[str]:
        insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        matched_paths = {str(path).strip() for path in insights.get("matched_paths", []) if str(path).strip()}
        for line in insights.get("matched_evidence", []):
            parsed = self._parse_evidence_reference(str(line))
            if parsed is not None:
                matched_paths.add(parsed[0])
        return matched_paths

    def _bootstrap(self) -> DiscoveryBootstrapService:
        service = getattr(self, "_bootstrap_service", None)
        if service is None:
            service = DiscoveryBootstrapService(self)
            self._bootstrap_service = service
        return service

    def _llm(self) -> DiscoveryLLMClient:
        client = getattr(self, "_llm_client", None)
        if client is None:
            client = DiscoveryLLMClient(self)
            self._llm_client = client
        return client

    def _next_turn_count(self, session_id: str) -> int:
        current = int(self.turn_counts_by_session.get(session_id, 0))
        next_count = current + 1
        self.turn_counts_by_session[session_id] = next_count
        return next_count

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        maybe = self.event_callback(event)
        if asyncio.iscoroutine(maybe):
            await maybe

    @staticmethod
    def _normalize_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map internal role names to OpenAI-accepted roles (author → assistant)."""
        role_map = {"author": "assistant"}
        return [{**m, "role": role_map.get(m.get("role", ""), m.get("role", ""))} for m in messages]

    def _build_memory_context(self) -> str:
        """Inject pre-built memory blocks so LLM has architecture overview upfront."""
        blocks = self.memory.load_all_blocks()
        manifesto = self.memory.load_manifesto()
        parts = []
        if manifesto:
            parts.append(f"## Project Manifesto\n{manifesto}")
        for name in ("architecture", "modules", "patterns", "entrypoints", "context"):
            block = blocks.get(name, "").strip()
            if block:
                parts.append(f"## {name.title()}\n{block}")
        return "\n\n".join(parts)

    @staticmethod
    def _latest_user_message(conversation: list[dict[str, Any]]) -> str:
        for message in reversed(conversation):
            if str(message.get("role", "")).strip() == "user":
                return str(message.get("content", "")).strip()
        return ""

    @classmethod
    def _extract_feature_intent(cls, user_message: str) -> FeatureIntent | None:
        return extract_feature_intent(user_message)

    @classmethod
    def _should_bootstrap_scan(cls, user_message: str) -> bool:
        return should_bootstrap_scan(user_message)

    @staticmethod
    def _parse_existing_endpoint_followup_choice(user_message: str) -> str | None:
        return parse_existing_endpoint_followup_choice(user_message)

    @staticmethod
    def _is_concrete_enhancement_request(user_message: str) -> bool:
        return is_concrete_enhancement_request(user_message)

    @staticmethod
    def _detect_framework(index: SignalIndex) -> str | None:
        return detect_framework(index)

    @classmethod
    def _drop_redundant_framework_questions(
        cls,
        parsed: DiscoveryTurnResult,
        inferred_framework: str | None,
    ) -> DiscoveryTurnResult:
        return drop_redundant_framework_questions(parsed, inferred_framework)

    @staticmethod
    def _route_file_score(path: str) -> int:
        return route_file_score(path)

    @staticmethod
    def _location_score(path: str, file_line_count: int | None = None) -> int:
        return location_score(path, file_line_count=file_line_count)

    @staticmethod
    def _detect_code_signals(index: SignalIndex) -> dict[str, int]:
        return detect_code_signals(index)

    @staticmethod
    def _detect_architecture(signal_scores: dict[str, int]) -> str | None:
        return detect_architecture(signal_scores)

    @staticmethod
    def _build_signal_index(matches: list[dict[str, Any]]) -> SignalIndex:
        return build_signal_index(matches)

    @staticmethod
    def _aggregate_evidence(evidence_list: list[Evidence]) -> list[Evidence]:
        return aggregate_evidence(evidence_list)

    @classmethod
    def _select_scan_directories(cls, entries: list[dict[str, Any]]) -> list[str]:
        return select_scan_directories(entries)

    @staticmethod
    def _format_grep_matches(matches: list[dict[str, Any]], limit: int = 14) -> str:
        return format_grep_matches(matches, limit=limit)

    def _existing_feature_evidence_lines(self, session_id: str) -> list[str]:
        insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        return existing_feature_evidence_lines(insights, self._route_file_score)

    @staticmethod
    def _format_evidence_line(item: Evidence) -> str:
        return format_evidence_line(item)

    @staticmethod
    def _parse_evidence_reference(line: str) -> tuple[str, int] | None:
        return parse_evidence_reference(line)

    @staticmethod
    def _summarize_endpoint_snippet(snippet: str) -> str | None:
        return summarize_endpoint_snippet(snippet)

    @staticmethod
    def _functional_summary_from_snippet(snippet: str) -> str | None:
        return functional_summary_from_snippet(snippet)

    async def _build_existing_endpoint_deep_summary(self, session_id: str) -> tuple[str | None, str | None]:
        evidence_lines = self._existing_feature_evidence_lines(session_id)
        if not hasattr(self, "tool_executor"):
            return None, None
        return await build_existing_endpoint_deep_summary(
            evidence_lines=evidence_lines,
            emit=self._emit,
            read_file=self.tool_executor.read_file,
        )

    async def _build_existing_feature_enhancement_summary(
        self,
        session_id: str,
        feature_label: str,
        requested_improvement: str | None = None,
    ) -> str:
        insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        return await build_existing_feature_enhancement_summary(
            insights=insights,
            feature_label=feature_label,
            requested_improvement=requested_improvement,
            route_file_score=self._route_file_score,
            deep_summary_loader=lambda: self._build_existing_endpoint_deep_summary(session_id),
        )

    def _merge_feature_evidence(
        self,
        *,
        session_id: str,
        feature: FeatureIntent,
        candidate_paths: list[str],
        evidence_lines: list[str] | None = None,
    ) -> None:
        if not hasattr(self, "bootstrap_insights_by_session"):
            self.bootstrap_insights_by_session = {}
        merge_feature_evidence(
            self.bootstrap_insights_by_session,
            session_id=session_id,
            feature=feature,
            candidate_paths=candidate_paths,
            evidence_lines=evidence_lines,
        )

    async def _maybe_read_more_context(self, path: str, line_number: int) -> str:
        return await self._bootstrap().maybe_read_more_context(path, line_number)

    async def _ingest_feature_evidence_from_tool(
        self,
        *,
        session_id: str,
        feature: FeatureIntent | None,
        tool_name: str,
        parsed_args: dict[str, Any],
        tool_result_payload: dict[str, Any],
    ) -> list[Evidence]:
        return await self._bootstrap().ingest_feature_evidence_from_tool(
            session_id=session_id,
            feature=feature,
            tool_name=tool_name,
            parsed_args=parsed_args,
            tool_result_payload=tool_result_payload,
        )

    async def _run_bootstrap_tool(
        self,
        *,
        tool_name: str,
        path: str | None = None,
        pattern: str | None = None,
        max_entries: int = 120,
        max_results: int = 80,
    ) -> dict[str, Any] | None:
        return await self._bootstrap().run_bootstrap_tool(
            tool_name=tool_name,
            path=path,
            pattern=pattern,
            max_entries=max_entries,
            max_results=max_results,
        )

    @staticmethod
    def _extract_feature_evidence_from_content(
        path: str,
        content: str,
        feature: FeatureIntent,
        limit: int = 3,
    ) -> list[str]:
        return DiscoveryBootstrapService.extract_feature_evidence_from_content(path, content, feature, limit=limit)

    async def _verify_feature_in_candidate_files(
        self,
        *,
        session_id: str,
        feature: FeatureIntent,
        candidate_paths: list[str],
    ) -> list[Evidence]:
        return await self._bootstrap().verify_feature_in_candidate_files(
            session_id=session_id,
            feature=feature,
            candidate_paths=candidate_paths,
        )

    async def _build_first_turn_bootstrap_context(
        self,
        session_id: str,
        conversation: list[dict[str, Any]],
        turn_count: int,
    ) -> tuple[str, str | None]:
        return await self._bootstrap().build_first_turn_bootstrap_context(session_id, conversation, turn_count)

    async def _llm_call_with_tools(
        self,
        messages: list[dict[str, Any]],
        max_tool_rounds: int = 6,
        model_override: str | None = None,
    ) -> str:
        return await self._llm().llm_call_with_tools(
            messages,
            max_tool_rounds=max_tool_rounds,
            model_override=model_override,
        )

    async def _llm_call(self, messages: list[dict[str, Any]], model_override: str | None = None) -> str:
        return await self._llm().llm_call(messages, model_override=model_override)

    async def _safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1200,
        model_override: str | None = None,
    ) -> Any:
        return await self._llm().safe_completion_call(
            litellm=litellm,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            model_override=model_override,
        )

    def opening_prompt(self) -> str:
        return (
            "Tell me what you want to plan. "
            "I'll scan the codebase first, then ask only about decisions and constraints "
            "that aren't already answered by the code."
        )

    def _try_extract_completion(self, text: str) -> DiscoveryTurnResult:
        return try_extract_completion(text)

    async def _handle_existing_feature_followup(
        self,
        *,
        session_id: str,
        latest_user_message: str,
        prior_insights: dict[str, Any],
        feature_label: str,
    ) -> DiscoveryTurnResult | None:
        if bool(prior_insights.get("enhance_existing")) and self._is_concrete_enhancement_request(latest_user_message):
            return DiscoveryTurnResult(
                reply="Discovery complete — drafting plan now.",
                complete=True,
                summary=await self._build_existing_feature_enhancement_summary(
                    session_id,
                    feature_label,
                    requested_improvement=latest_user_message,
                ),
                questions=[],
            )
        choice = self._parse_existing_endpoint_followup_choice(latest_user_message)
        if choice == "A":
            evidence_lines = self._existing_feature_evidence_lines(session_id)
            evidence = "\n".join(f"- {line}" for line in evidence_lines)
            deep_summary, functional_summary = await self._build_existing_endpoint_deep_summary(session_id)
            matched_paths = [str(path).strip() for path in prior_insights.get("matched_paths", []) if str(path).strip()]
            runtime_paths = [
                path
                for path in sorted(matched_paths, key=lambda p: (-self._route_file_score(p), p))
                if self._route_file_score(path) > 0
            ][:3]
            primary_impl = runtime_paths[0] if runtime_paths else None
            if functional_summary:
                overview = functional_summary
            elif evidence_lines:
                overview = (
                    f"I found existing implementation evidence for {feature_label} with concrete references below."
                )
            else:
                overview = (
                    f"I found signs of an existing {feature_label}, but current citations are weak. "
                    "I should run a targeted route/handler scan before giving a definitive summary."
                )
            files_block = ""
            if primary_impl:
                files_block = f"- Primary implementation: `{primary_impl}`\n"
            elif runtime_paths:
                files_block = (
                    "Likely implementation files:\n" + "\n".join(f"- `{path}`" for path in runtime_paths) + "\n"
                )
            details_block = f"Implementation notes:\n{deep_summary}\n\n" if deep_summary else ""
            return DiscoveryTurnResult(
                reply=(
                    "Summary of what already exists:\n"
                    f"- Functional overview: {overview}\n"
                    f"{files_block}"
                    "Supporting references:\n"
                    f"{evidence}\n\n"
                    f"{details_block}"
                    f"If you want, I can propose targeted enhancements next for the existing {feature_label}, "
                    "but I won't create a duplicate implementation."
                ),
                complete=False,
                summary=None,
                questions=[],
            )
        if choice == "B":
            prior_insights["enhance_existing"] = True
            return DiscoveryTurnResult(
                reply="Tell me which area to prioritize first.",
                complete=False,
                summary=None,
                questions=[],
            )
        if choice == "C":
            prior_insights["enhance_existing"] = False
            return DiscoveryTurnResult(
                reply=(
                    f"Understood — we'll leave the existing {feature_label} unchanged. "
                    "No planning draft will be generated."
                ),
                complete=False,
                summary=None,
                questions=[],
            )
        return None

    async def _maybe_existing_feature_first_turn_result(
        self,
        *,
        turn_count: int,
        latest_user_message: str,
        session_id: str,
    ) -> DiscoveryTurnResult | None:
        if turn_count != 1:
            return None
        insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        if not bool(insights.get("existing_feature")):
            return None
        if self._extract_feature_intent(latest_user_message) is None:
            return None
        current_feature_label = str(insights.get("feature_label", "feature")).strip() or "feature"
        evidence_lines = self._existing_feature_evidence_lines(session_id)
        evidence_block = (
            "\n".join(f"- {line}" for line in evidence_lines[:5]) or "- Found existing implementation in the codebase"
        )
        deep_summary, functional_summary = await self._build_existing_endpoint_deep_summary(session_id)
        details_block = f"\n\nWhat it currently does:\n{deep_summary}" if deep_summary else ""
        overview_line = f"Functional overview: {functional_summary}\n\n" if functional_summary else ""
        return DiscoveryTurnResult(
            reply=(
                f"The {current_feature_label} already appears to exist in the codebase, so I won't draft a new creation plan.\n\n"
                f"{overview_line}"
                f"Evidence:\n{evidence_block}"
                f"{details_block}\n\n"
                f"Do you want me to review and propose enhancements to the existing {current_feature_label} instead?"
            ),
            complete=False,
            summary=None,
            questions=[
                DiscoveryQuestion(
                    index=1,
                    text=f"What should we do with the existing {current_feature_label}?",
                    options=[
                        QuestionOption("A", "Review current behavior and summarize it only."),
                        QuestionOption(
                            "B", "Propose targeted enhancements without creating a duplicate implementation."
                        ),
                        QuestionOption("C", "Leave it unchanged; no planning needed."),
                        QuestionOption("D", "Other — describe your preference", is_other=True),
                    ],
                )
            ],
        )

    async def handle_turn(
        self,
        conversation: list[dict[str, Any]],
        session_id: str = "default",
        model_override: str | None = None,
        extra_context: str = "",
    ) -> DiscoveryTurnResult:
        turn_count = self._next_turn_count(session_id)

        if turn_count >= self.config.discovery_max_turns:
            summary = await self.force_summary(conversation)
            return DiscoveryTurnResult(
                reply=("I have enough context to start drafting your plan now."),
                complete=True,
                summary=summary,
            )

        try:
            import litellm  # noqa: F401
        except ImportError:
            return DiscoveryTurnResult(
                reply="LLM unavailable; proceeding with provided requirements context.",
                complete=True,
                summary="\n".join(m.get("content", "") for m in conversation if m.get("role") == "user"),
            )

        latest_user_message = self._latest_user_message(conversation)
        prior_insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        feature_label = str(prior_insights.get("feature_label", "feature")).strip() or "feature"
        if turn_count > 1 and bool(prior_insights.get("existing_feature")):
            followup = await self._handle_existing_feature_followup(
                session_id=session_id,
                latest_user_message=latest_user_message,
                prior_insights=prior_insights,
                feature_label=feature_label,
            )
            if followup is not None:
                return followup

        memory_context = self._build_memory_context()
        system_content = DISCOVERY_SYSTEM_PROMPT
        if memory_context:
            system_content += f"\n\n## Pre-built Codebase Memory\n{memory_context}"
        try:
            bootstrap_context, inferred_framework = await self._build_first_turn_bootstrap_context(
                session_id,
                conversation,
                turn_count,
            )
        except TypeError:
            # Backwards-compatible fallback for tests/mocks that still provide the
            # older method signature (_conversation, _turn_count).
            bootstrap_context, inferred_framework = await self._build_first_turn_bootstrap_context(  # type: ignore[misc]
                conversation,
                turn_count,
            )
        if bootstrap_context:
            system_content += f"\n\n{bootstrap_context}"
        if extra_context:
            system_content += f"\n\n{extra_context}"

        messages = [{"role": "system", "content": system_content}, *conversation]
        await self._emit({"type": "thinking", "message": "Refining questions from available context..."})
        self._active_discovery_session_id = session_id
        try:
            response = await self._llm_call_with_tools(
                messages,
                max_tool_rounds=self.config.discovery_tool_rounds,
                model_override=model_override,
            )
        finally:
            self._active_discovery_session_id = None
        parsed = self._drop_redundant_framework_questions(
            self._try_extract_completion(response),
            inferred_framework,
        )
        existing_first_turn = await self._maybe_existing_feature_first_turn_result(
            turn_count=turn_count,
            latest_user_message=latest_user_message,
            session_id=session_id,
        )
        if existing_first_turn is not None:
            return existing_first_turn
        if not parsed.complete and len(parsed.questions) == 0:
            # If discovery returns prose without questions, treat it as implicit
            # completion rather than paying extra LLM calls just to restate shape.
            summary = parsed.reply[:500] if parsed.reply else "Requirements gathered from conversation."
            return DiscoveryTurnResult(
                reply=parsed.reply or "Discovery complete — drafting plan now.",
                complete=True,
                summary=summary,
                questions=[],
            )
        return parsed

    async def force_summary(
        self,
        conversation: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> str:
        try:
            import litellm  # noqa: F401
        except ImportError:
            return "\n".join(m.get("content", "") for m in conversation if m.get("role") == "user")[:2000]

        messages = [
            {
                "role": "system",
                "content": "Summarize the discovered planning requirements in concise markdown.",
            },
            *conversation,
        ]
        return await self._llm_call(messages, model_override=model_override)


__all__ = [
    "BOOTSTRAP_ROUTE_REGEX",
    "CODE_SIGNALS",
    "DISCOVERY_SYSTEM_PROMPT",
    "FRAMEWORKS",
    "CodeSignal",
    "DiscoveryManager",
    "DiscoveryQuestion",
    "DiscoveryTurnResult",
    "Evidence",
    "FeatureIntent",
    "Framework",
    "IndexedMatch",
    "QuestionOption",
    "SignalIndex",
    "parse_questions",
]
