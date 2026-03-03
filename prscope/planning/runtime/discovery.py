"""
Chat-first requirements discovery flow.

On first user message the LLM scans the codebase via tools before responding,
then asks only questions that cannot be answered from code.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ...config import PlanningConfig
from ...pricing import MODEL_CONTEXT_WINDOWS
from ...memory import MemoryStore
from .tools import CODEBASE_TOOLS, ToolExecutor
from .telemetry import completion_telemetry


DISCOVERY_SYSTEM_PROMPT = """You are a planning assistant helping scope a software implementation plan.

## Your Process

**Step 1 — Research first (ALWAYS on the first user message):**
Use list_files, read_file, and grep_code to understand the project before asking anything.
Read: README, key source files, package manifests, existing patterns relevant to the request.
Do not skip this step. Hallucinating project structure is worse than asking.

**Step 2 — Ask only what code can't tell you:**
After scanning, ask ONLY 2-3 questions that require a human decision:
- Priorities and trade-offs
- Acceptance criteria or definition of done
- Business constraints not visible in code
- Architectural preferences where multiple valid approaches exist

Never ask questions whose answers are already in the codebase.

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


@dataclass
class QuestionOption:
    letter: str   # "A", "B", "C", "D"
    text: str
    is_other: bool = False


@dataclass
class DiscoveryQuestion:
    index: int    # 1-based question number shown in the UI
    text: str
    options: list[QuestionOption]

    def option_text(self, letter: str) -> str:
        for opt in self.options:
            if opt.letter == letter:
                return opt.text
        return ""


def parse_questions(reply: str) -> list[DiscoveryQuestion]:
    """
    Parse structured Q&A blocks from an LLM reply.

    Recognises:
      **Q1: question text**    or    **Q: text**    or    **1. text** (bold)
    followed by lines:
      A) option text
      B) option text
      ...
    """
    questions: list[DiscoveryQuestion] = []
    lines = reply.splitlines()
    i = 0

    # Accept both bold and plain headers:
    #   **Q1: text** / Q1: text / Question 1: text / 1. text
    q_re = re.compile(
        r"^(?:Q\s*\d*|Question\s+\d+|\d+)\s*[:\.\)\-]\s*(.+)$",
        re.IGNORECASE,
    )
    # Accept option prefixes with markdown bullets and separators:
    #   A) text / A. text / - A) text / * B. text
    opt_re = re.compile(r"^(?:[-*]\s*)?([A-D])[\)\.\-:]\s*(.+)$", re.IGNORECASE)

    def normalize_line(line: str) -> str:
        normalized = line.strip()
        # Strip simple markdown wrappers and bullets.
        normalized = re.sub(r"^[-*]\s+", "", normalized)
        normalized = normalized.replace("**", "").strip()
        return normalized

    while i < len(lines):
        line = normalize_line(lines[i])
        q_match = q_re.match(line)
        if q_match:
            q_text = q_match.group(1).strip().rstrip(":?").strip() + "?"
            options: list[QuestionOption] = []
            i += 1
            while i < len(lines):
                stripped = normalize_line(lines[i])
                opt_match = opt_re.match(stripped)
                if opt_match:
                    letter = opt_match.group(1).upper()
                    text = opt_match.group(2).strip()
                    is_other = bool(re.match(r"other", text, re.IGNORECASE))
                    options.append(QuestionOption(letter=letter, text=text, is_other=is_other))
                    i += 1
                elif not stripped and options:
                    # blank line after options block — stop collecting
                    break
                elif stripped and not opt_match:
                    # non-option text after collecting at least one option — stop
                    if options:
                        break
                    i += 1  # pre-option prose — keep scanning
                else:
                    i += 1
            if options:
                questions.append(
                    DiscoveryQuestion(
                        index=len(questions) + 1,
                        text=q_text,
                        options=options,
                    )
                )
        else:
            i += 1

    return questions


@dataclass
class DiscoveryTurnResult:
    reply: str
    complete: bool
    summary: str | None = None
    questions: list[DiscoveryQuestion] = field(default_factory=list)


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
        self.event_callback = event_callback

    def reset_session(self, session_id: str) -> None:
        self.turn_counts_by_session[session_id] = 0

    def clear_session(self, session_id: str) -> None:
        self.turn_counts_by_session.pop(session_id, None)

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
        return [
            {**m, "role": role_map.get(m.get("role", ""), m.get("role", ""))}
            for m in messages
        ]

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

    async def _llm_call_with_tools(
        self,
        messages: list[dict[str, Any]],
        max_tool_rounds: int = 6,
        model_override: str | None = None,
    ) -> str:
        """LLM call loop that executes tool calls until LLM produces a text response."""
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        conversation = list(messages)
        announced_scanning = False

        for _ in range(max_tool_rounds):
            response = await self._safe_completion_call(
                litellm=litellm,
                messages=self._normalize_roles(conversation),
                tools=CODEBASE_TOOLS,
                max_tokens=1800,
                model_override=model_override,
            )
            message = response.choices[0].message
            content = str(getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                if not announced_scanning:
                    announced_scanning = True
                    await self._emit(
                        {"type": "thinking", "message": "Scanning codebase and refining questions..."}
                    )
                # Execute all tool calls and append results
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
                            "session_stage": "discovery",
                            "path": str(path_hint) if isinstance(path_hint, str) else None,
                            "query": str(query_hint) if isinstance(query_hint, str) else None,
                        }
                    )
                    try:
                        tool_started = asyncio.get_running_loop().time()
                        result = await asyncio.to_thread(self.tool_executor.execute, tc)
                        tool_elapsed_ms = (asyncio.get_running_loop().time() - tool_started) * 1000.0
                    except Exception as exc:  # noqa: BLE001
                        result = {"tool_call_id": getattr(tc, "id", ""), "name": "", "result": {"error": str(exc)}}
                        tool_elapsed_ms = 0.0
                    await self._emit(
                        {
                            "type": "tool_result",
                            "name": tool_name,
                            "session_stage": "discovery",
                            "duration_ms": round(tool_elapsed_ms, 2),
                        }
                    )
                    raw_content = json.dumps(result["result"])
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "name": result["name"],
                            "content": raw_content,
                        }
                    )
                continue

            return content

        return "I've scanned the codebase. What aspect of this would you like to focus on first?"

    async def _llm_call(self, messages: list[dict[str, Any]], model_override: str | None = None) -> str:
        """Simple LLM call without tools (used for force_summary)."""
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        response = await self._safe_completion_call(
            litellm=litellm,
            messages=self._normalize_roles(messages),
            max_tokens=900,
            model_override=model_override,
        )
        return str(response.choices[0].message.content or "").strip()

    async def _safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1200,
        model_override: str | None = None,
    ) -> Any:
        """Call chat completion with graceful fallback for non-chat models."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        primary_model = model_override or self.config.author_model
        fallback_model = "gpt-4o-mini"
        models_to_try = [primary_model]
        if fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                llm_started = asyncio.get_running_loop().time()
                response = await asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    **kwargs,
                )
                llm_elapsed_ms = (asyncio.get_running_loop().time() - llm_started) * 1000.0
                telemetry = completion_telemetry(response, model=model)
                context_window = MODEL_CONTEXT_WINDOWS.get(model)
                await self._emit(
                    {
                        "type": "token_usage",
                        "session_stage": "discovery",
                        "model": model,
                        "prompt_tokens": telemetry.usage.prompt_tokens,
                        "completion_tokens": telemetry.usage.completion_tokens,
                        "call_cost_usd": telemetry.cost.total_cost_usd,
                        "llm_call_latency_ms": round(llm_elapsed_ms, 2),
                        "context_window_tokens": context_window,
                        "context_usage_ratio": (
                            round(float(telemetry.usage.prompt_tokens) / float(context_window), 4)
                            if context_window and context_window > 0
                            else None
                        ),
                    }
                )
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                err_text = str(exc).lower()
                non_chat_model = (
                    "not a chat model" in err_text
                    or "v1/chat/completions" in err_text
                    or "did you mean to use v1/completions" in err_text
                )
                # Retry only for known model-compatibility issues.
                if not non_chat_model or idx == len(models_to_try) - 1:
                    break

        if last_error is not None:
            raise RuntimeError(
                "Configured planning model is incompatible with chat completions. "
                "Update planning.author_model or use a chat-capable model."
            ) from last_error
        raise RuntimeError("Unknown completion failure during discovery.")

    def opening_prompt(self) -> str:
        return (
            "Tell me what you want to plan. "
            "I'll scan the codebase first, then ask only about decisions and constraints "
            "that aren't already answered by the code."
        )

    @staticmethod
    def _strip_json_comments(raw: str) -> str:
        # Best-effort cleanup for model outputs that include JS-style comments.
        no_block = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
        no_line = re.sub(r"//[^\n\r]*", "", no_block)
        return no_line.strip()

    @classmethod
    def _parse_discovery_complete_candidate(cls, candidate: str) -> dict[str, Any] | None:
        for attempt in (candidate, cls._strip_json_comments(candidate)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and str(parsed.get("discovery", "")).strip().lower() == "complete":
                return parsed
        return None

    @staticmethod
    def _extract_json_code_blocks(text: str) -> list[str]:
        blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        return [block.strip() for block in blocks if block.strip()]

    @staticmethod
    def _extract_balanced_json_objects(text: str) -> list[str]:
        candidates: list[str] = []
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            end = -1
            for idx in range(start, len(text)):
                ch = text[idx]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == "\"":
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            if end != -1:
                candidate = text[start : end + 1].strip()
                if candidate:
                    candidates.append(candidate)
                start = text.find("{", end + 1)
            else:
                break
        return candidates

    def _extract_completion_payload(self, text: str) -> tuple[dict[str, Any], str] | None:
        candidates = self._extract_json_code_blocks(text)
        candidates.extend(self._extract_balanced_json_objects(text))
        for candidate in candidates:
            parsed = self._parse_discovery_complete_candidate(candidate)
            if parsed is not None:
                return parsed, candidate
        return None

    def _try_extract_completion(self, text: str) -> DiscoveryTurnResult:
        payload = self._extract_completion_payload(text)
        if payload is None:
            questions = parse_questions(text)
            return DiscoveryTurnResult(reply=text, complete=False, summary=None, questions=questions)
        parsed, payload_text = payload

        summary_raw = parsed.get("summary", "")
        if isinstance(summary_raw, str):
            summary = summary_raw.strip()
        elif summary_raw:
            summary = json.dumps(summary_raw, ensure_ascii=False)
        else:
            summary = ""
        # Remove machine-readable completion payload from chat-visible prose.
        prose = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)
        prose = prose.replace(payload_text, "", 1).strip()
        return DiscoveryTurnResult(
            reply=prose or "Discovery complete — drafting plan now.",
            complete=True,
            summary=summary or None,
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
                reply=(
                    "I have enough context to start drafting your plan now."
                ),
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

        memory_context = self._build_memory_context()
        system_content = DISCOVERY_SYSTEM_PROMPT
        if memory_context:
            system_content += f"\n\n## Pre-built Codebase Memory\n{memory_context}"
        if extra_context:
            system_content += f"\n\n{extra_context}"

        messages = [{"role": "system", "content": system_content}, *conversation]
        await self._emit({"type": "thinking", "message": "Refining questions from available context..."})
        response = await self._llm_call_with_tools(
            messages,
            max_tool_rounds=self.config.discovery_tool_rounds,
            model_override=model_override,
        )
        parsed = self._try_extract_completion(response)
        # Guardrail: discovery should ask a batched question set (2-3), not one-by-one.
        if not parsed.complete and len(parsed.questions) == 1:
            expanded = await self._expand_question_batch(
                original_response=response,
                model_override=model_override,
            )
            expanded_parsed = self._try_extract_completion(expanded)
            if not expanded_parsed.complete and len(expanded_parsed.questions) >= 2:
                return expanded_parsed
            # Retry once more to reduce one-by-one drift from weaker model replies.
            expanded_retry = await self._expand_question_batch(
                original_response=expanded,
                model_override=model_override,
            )
            expanded_retry_parsed = self._try_extract_completion(expanded_retry)
            if not expanded_retry_parsed.complete and len(expanded_retry_parsed.questions) >= 2:
                return expanded_retry_parsed
        # Guardrail: if the model returned prose (no questions, not complete) it
        # likely decided discovery is done but forgot to emit the completion JSON.
        # Ask it to either emit the JSON payload or provide more questions.
        if not parsed.complete and len(parsed.questions) == 0:
            reformatted = await self._force_completion_or_questions(
                original_response=response,
                model_override=model_override,
            )
            reformatted_parsed = self._try_extract_completion(reformatted)
            if reformatted_parsed.complete or len(reformatted_parsed.questions) > 0:
                return reformatted_parsed
            # If still empty after reformat, treat as implicit completion so the
            # session can proceed rather than hanging with no questions and no draft.
            summary = parsed.reply[:500] if parsed.reply else "Requirements gathered from conversation."
            return DiscoveryTurnResult(
                reply=parsed.reply or "Discovery complete — drafting plan now.",
                complete=True,
                summary=summary,
                questions=[],
            )
        return parsed

    async def _force_completion_or_questions(self, original_response: str, model_override: str | None = None) -> str:
        """Ask the model to reformat a prose-only response into either the completion JSON or questions."""
        prompt = (
            "Your previous response contained neither discovery questions nor the required completion JSON.\n"
            "You must do one of the following:\n\n"
            "Option A — If you have all the information needed, emit ONLY this JSON (no prose):\n"
            '{"discovery":"complete","summary":"<comprehensive requirements summary>"}\n\n'
            "Option B — If you need more information, emit 2-3 discovery questions in this exact format:\n"
            "Q1: ...\nA) ...\nB) ...\nC) ...\nD) Other — describe your preference\n\n"
            "Do not include any other text. Choose exactly one option.\n\n"
            f"Your previous response was:\n{original_response}"
        )
        return await self._llm_call(
            [
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model_override=model_override,
        )

    async def _expand_question_batch(self, original_response: str, model_override: str | None = None) -> str:
        """Rewrite a singleton question into a full 2-3 question discovery batch."""
        prompt = (
            "Rewrite the discovery questions as a single batched response with 2-3 questions.\n"
            "Return only question blocks in this exact format:\n"
            "Q1: ...\nA) ...\nB) ...\nC) ...\nD) Other — describe your preference\n\n"
            "Do not include any explanatory prose before or after questions.\n"
            "Do not ask one question at a time.\n\n"
            f"Original response:\n{original_response}"
        )
        return await self._llm_call(
            [
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model_override=model_override,
        )

    async def force_summary(
        self,
        conversation: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> str:
        try:
            import litellm  # noqa: F401
        except ImportError:
            return "\n".join(
                m.get("content", "")
                for m in conversation
                if m.get("role") == "user"
            )[:2000]

        messages = [
            {
                "role": "system",
                "content": "Summarize the discovered planning requirements in concise markdown.",
            },
            *conversation,
        ]
        return await self._llm_call(messages, model_override=model_override)
