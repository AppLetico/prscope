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
from ...memory import MemoryStore
from .tools import CODEBASE_TOOLS, ToolExecutor


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

    # Regex for bold question headers in a few common formats
    q_re = re.compile(
        r"\*\*\s*(?:Q\s*\d*\s*[:\.]?\s*|Question\s+\d+\s*[:\.]?\s*|\d+\s*[:\.]?\s*)(.+?)\*\*",
        re.IGNORECASE,
    )
    opt_re = re.compile(r"^([A-D])\)\s*(.+)", re.IGNORECASE)

    while i < len(lines):
        q_match = q_re.search(lines[i])
        if q_match:
            q_text = q_match.group(1).strip().rstrip(":?").strip() + "?"
            options: list[QuestionOption] = []
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
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
        self.turn_count = 0
        self.event_callback = event_callback

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

    async def _llm_call_with_tools(self, messages: list[dict[str, Any]], max_tool_rounds: int = 6) -> str:
        """LLM call loop that executes tool calls until LLM produces a text response."""
        import litellm

        litellm.drop_params = True
        conversation = list(messages)

        for _ in range(max_tool_rounds):
            response = await self._safe_completion_call(
                litellm=litellm,
                messages=self._normalize_roles(conversation),
                tools=CODEBASE_TOOLS,
                max_tokens=1800,
            )
            message = response.choices[0].message
            content = str(getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
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
                    await self._emit(
                        {
                            "type": "tool_call",
                            "name": getattr(getattr(tc, "function", None), "name", ""),
                            "session_stage": "discovery",
                        }
                    )
                    try:
                        result = self.tool_executor.execute(tc)
                    except Exception as exc:  # noqa: BLE001
                        result = {"tool_call_id": getattr(tc, "id", ""), "name": "", "result": {"error": str(exc)}}
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

    async def _llm_call(self, messages: list[dict[str, Any]]) -> str:
        """Simple LLM call without tools (used for force_summary)."""
        import litellm

        litellm.drop_params = True
        response = await self._safe_completion_call(
            litellm=litellm,
            messages=self._normalize_roles(messages),
            max_tokens=900,
        )
        return str(response.choices[0].message.content or "").strip()

    async def _safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1200,
    ) -> Any:
        """Call chat completion with graceful fallback for non-chat models."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        primary_model = self.config.author_model
        fallback_model = "gpt-4o-mini"
        models_to_try = [primary_model]
        if fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                return await asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    **kwargs,
                )
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

    def _try_extract_completion(self, text: str) -> DiscoveryTurnResult:
        match = re.search(r"\{[^{}]*\"discovery\"\s*:\s*\"complete\"[^{}]*\}", text, re.DOTALL)
        if not match:
            questions = parse_questions(text)
            return DiscoveryTurnResult(reply=text, complete=False, summary=None, questions=questions)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            questions = parse_questions(text)
            return DiscoveryTurnResult(reply=text, complete=False, summary=None, questions=questions)

        if parsed.get("discovery") == "complete":
            summary = str(parsed.get("summary", "")).strip()
            prose = (text[: match.start()] + text[match.end() :]).strip()
            return DiscoveryTurnResult(
                reply=prose or "Discovery complete — drafting plan now.",
                complete=True,
                summary=summary or None,
            )
        questions = parse_questions(text)
        return DiscoveryTurnResult(reply=text, complete=False, summary=None, questions=questions)

    async def handle_turn(self, conversation: list[dict[str, Any]]) -> DiscoveryTurnResult:
        self.turn_count += 1

        if self.turn_count >= self.config.discovery_max_turns:
            summary = await self.force_summary(conversation)
            return DiscoveryTurnResult(
                reply=(
                    f"Discovery limit reached ({self.config.discovery_max_turns} turns). "
                    "Proceeding with draft."
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

        messages = [{"role": "system", "content": system_content}, *conversation]
        await self._emit({"type": "thinking", "message": "Scanning codebase and refining questions..."})
        response = await self._llm_call_with_tools(
            messages,
            max_tool_rounds=self.config.discovery_tool_rounds,
        )
        return self._try_extract_completion(response)

    async def force_summary(self, conversation: list[dict[str, Any]]) -> str:
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
        return await self._llm_call(messages)
