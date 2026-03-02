"""
Author runtime: plan drafting/refinement with tool-use enforcement.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from ...config import PlanningConfig
from ...pricing import MODEL_CONTEXT_WINDOWS
from .telemetry import completion_telemetry
from .tools import CODEBASE_TOOLS, ToolExecutor, extract_file_references


@dataclass
class AuthorResult:
    plan: str
    unverified_references: set[str]
    accessed_paths: set[str]
    grounding_ratio: float | None = None
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejection_reasons: list[dict[str, str]] = field(default_factory=list)
    average_read_depth: float | None = None
    average_time_between_tool_calls: float | None = None


@dataclass
class PlanSchema:
    required_sections: dict[str, str] = field(default_factory=dict)
    requires_mermaid: bool = True
    requires_code_blocks: bool = True
    requires_ordered_todos: bool = True


@dataclass
class _FunctionCall:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _FunctionCall


@dataclass
class _Message:
    content: str
    tool_calls: list[_ToolCall]


@dataclass
class _Choice:
    message: _Message


@dataclass
class _ChatLikeResponse:
    choices: list[_Choice]
    usage: Any | None = None


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

DEFAULT_REQUIRED_PLAN_SECTION_PATTERNS: dict[str, str] = {
    "title": r"^#\s+.+",
    "summary": r"^##\s+summary\b",
    "goals": r"^##\s+goals\b",
    "non_goals": r"^##\s+non-goals\b",
    "changes": r"^##\s+changes\b",
    "files_changed": r"^##\s+files\s+changed\b",
    "todos_in_order": r"^##\s+to-?dos?\s+in\s+order\b",
    "architecture": r"^##\s+architecture\b",
    "mermaid_diagram": r"^##\s+mermaid\s+diagram\b",
    "implementation_steps": r"^##\s+implementation\s+steps\b",
    "test_strategy": r"^##\s+test\s+strategy\b",
    "rollback_plan": r"^##\s+rollback\s+plan\b",
    "example_code_snippets": r"^##\s+example\s+code\s+snippets\b",
    "open_questions": r"^##\s+open\s+questions\b",
    "design_decision_records": r"^##\s+design\s+decision\s+records\b",
    "user_stories": r"^##\s+user\s+stories\b",
}

REQUIREMENT_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under",
    "mode", "plan", "planning", "project", "repo", "repository", "should", "must",
    "will", "can", "about", "after", "before", "while", "when", "where", "what",
    "then", "than", "use", "using", "add", "make", "more", "less", "only",
}

NON_TRIVIAL_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb",
    ".yaml", ".yml", ".toml", ".json", ".sql", ".sh",
}

TRIVIAL_FILENAMES = {"readme", "license", ".gitignore"}


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
        self.plan_schema = PlanSchema(required_sections=DEFAULT_REQUIRED_PLAN_SECTION_PATTERNS.copy())

    @staticmethod
    def _requirements_keywords(text: str) -> set[str]:
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        return {
            token
            for token in tokens
            if len(token) >= 3 and token not in REQUIREMENT_STOPWORDS
        }

    @staticmethod
    def _path_tokens(path: str) -> set[str]:
        return {token for token in re.split(r"[^a-z0-9]+", path.lower()) if token}

    @staticmethod
    def _is_non_trivial_source(path: str) -> bool:
        lower = path.lower()
        base = lower.rsplit("/", 1)[-1]
        stem = base.split(".", 1)[0]
        if stem in TRIVIAL_FILENAMES or lower == ".prscope/manifesto.md":
            return False
        dot = base.rfind(".")
        ext = base[dot:] if dot >= 0 else ""
        return ext in NON_TRIVIAL_EXTENSIONS

    @staticmethod
    def _is_entrypoint_like(path: str) -> bool:
        lower = path.lower()
        base = lower.rsplit("/", 1)[-1]
        return (
            base in {"main.py", "app.py", "server.py", "index.ts", "index.tsx", "index.js"}
            or "/cli" in lower
            or "/cmd/" in lower
            or "/bin/" in lower
            or "/server/" in lower
            or "/api/" in lower
        )

    @staticmethod
    def _is_test_or_config(path: str) -> bool:
        lower = path.lower()
        base = lower.rsplit("/", 1)[-1]
        if any(token in base for token in (".test.", ".spec.")):
            return True
        if "/tests/" in lower or "/test/" in lower:
            return True
        return base in {
            "pyproject.toml",
            "package.json",
            "package-lock.json",
            "tsconfig.json",
            "vite.config.ts",
            "dockerfile",
            "docker-compose.yml",
            ".github/workflows/ci.yml",
        } or lower.endswith((".yml", ".yaml", ".toml", ".json"))

    def _explorer_gate_failures(self, requirements_text: str) -> list[str]:
        read_history = self.tool_executor.read_history
        read_paths = set(read_history.keys())
        failures: list[str] = []
        if len(read_paths) < 3:
            failures.append(f"need at least 3 unique `read_file` calls (currently {len(read_paths)})")
        if not any(self._is_non_trivial_source(path) for path in read_paths):
            failures.append("need at least 1 non-trivial source/config file read")
        keywords = self._requirements_keywords(requirements_text)
        if keywords and not any(self._path_tokens(path) & keywords for path in read_paths):
            failures.append("need at least 1 requirement-relevant file read")
        if keywords and not any(
            (self._path_tokens(path) & keywords)
            and (
                int(meta.get("line_count", 0)) >= 20
                or int(meta.get("file_size_bytes", 0)) >= 1000
            )
            for path, meta in read_history.items()
        ):
            failures.append("need at least 1 requirement-relevant substantive read")
        if not any(
            int(meta.get("line_count", 0)) >= 30 or int(meta.get("file_size_bytes", 0)) >= 1500
            for meta in read_history.values()
        ):
            failures.append("need at least 1 substantive read (>=30 lines OR >=1.5KB)")
        if not any(self._is_entrypoint_like(path) for path in read_paths):
            failures.append("need at least 1 entrypoint/runtime file read")
        if not any(self._is_test_or_config(path) for path in read_paths):
            failures.append("need at least 1 test or config file read")
        return failures

    @staticmethod
    def _extract_section(content: str, heading: str) -> str:
        pattern = re.compile(
            rf"^##\s+{re.escape(heading)}\b(.*?)(?=^##\s+|\Z)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(content)
        return match.group(1).strip() if match else ""

    def _grounding_failures(
        self,
        plan_content: str,
        verified_paths: set[str],
        min_grounding_ratio: float,
        draft_phase: Literal["planner", "refiner"],
    ) -> tuple[list[str], set[str], float]:
        referenced = extract_file_references(plan_content)
        if not referenced:
            return [], set(), 1.0
        unverified = referenced - verified_paths
        grounding_ratio = (len(referenced) - len(unverified)) / float(len(referenced))
        failures: list[str] = []
        if grounding_ratio < min_grounding_ratio:
            failures.append(
                f"grounding ratio {grounding_ratio:.2f} below required {min_grounding_ratio:.2f}"
            )
        # Cross-checking Files Changed vs Implementation Steps only applies once
        # we're in refiner mode where implementation detail sections are required.
        if draft_phase == "refiner":
            files_changed = extract_file_references(self._extract_section(plan_content, "Files Changed"))
            implementation = extract_file_references(
                self._extract_section(plan_content, "Implementation Steps")
            )
            missing_impl_refs = sorted(files_changed - implementation)
            if missing_impl_refs:
                failures.append(
                    "Files Changed entries missing from Implementation Steps: "
                    + ", ".join(missing_impl_refs)
                )
        return failures, unverified, grounding_ratio

    def _phase_failures(self, plan_content: str, draft_phase: Literal["planner", "refiner"]) -> list[str]:
        if draft_phase != "planner":
            return []
        failures: list[str] = []
        fence_count = len(re.findall(r"```", plan_content))
        if fence_count > 2:
            failures.append("planner draft has too many code fences")
        if re.search(r"^##\s+Implementation\s+Steps\b", plan_content, re.IGNORECASE | re.MULTILINE):
            failures.append("planner draft must not include Implementation Steps section")
        if re.search(r"^##\s+Test\s+Strategy\b", plan_content, re.IGNORECASE | re.MULTILINE):
            failures.append("planner draft must not include Test Strategy section")
        if re.search(r"^##\s+Rollback\s+Plan\b", plan_content, re.IGNORECASE | re.MULTILINE):
            failures.append("planner draft must not include Rollback Plan section")
        numbered_items = re.findall(
            r"^\s*\d+\.\s+.*\b(modify|add|update|delete|replace|rename|refactor)\b.*$",
            plan_content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if len(numbered_items) > 5:
            failures.append("planner draft contains detailed implementation step list")
        return failures

    def _completion_failures(self, plan_content: str) -> list[str]:
        failures: list[str] = []
        lower = plan_content.lower()
        if re.search(r"\b(todo|tbd|placeholder)\b", lower):
            failures.append("final draft contains TODO/TBD/placeholder markers")
        required_non_empty = [
            "Goals",
            "Non-Goals",
            "Files Changed",
            "Architecture",
            "Implementation Steps",
            "Test Strategy",
            "Rollback Plan",
        ]
        for heading in required_non_empty:
            if not self._extract_section(plan_content, heading):
                failures.append(f"required section is empty: {heading}")
        files_changed = extract_file_references(self._extract_section(plan_content, "Files Changed"))
        if not files_changed:
            failures.append("Files Changed section is empty")
        total_references = extract_file_references(plan_content)
        if len(files_changed) == 1 and len(total_references) > 1:
            failures.append("under-scoped draft: one file in Files Changed but multiple referenced files")
        implementation = self._extract_section(plan_content, "Implementation Steps").lower()
        if implementation and not any(token in implementation for token in ("interface", "signature", "contract", "api shape")):
            failures.append("Implementation Steps missing interface/signature impact notes")
        test_strategy = self._extract_section(plan_content, "Test Strategy").lower()
        if test_strategy and not any(
            token in test_strategy
            for token in ("assert", "expects", "status code", "error path", "failure", "regression")
        ):
            failures.append("Test Strategy lacks concrete assertions or failure-path checks")
        rollback = self._extract_section(plan_content, "Rollback Plan").lower()
        if rollback and not any(token in rollback for token in ("trigger", "if ", "on ", "when ")):
            failures.append("Rollback Plan missing rollback trigger conditions")
        if rollback and not any(token in rollback for token in ("revert", "disable", "restore", "rollback action")):
            failures.append("Rollback Plan missing explicit rollback actions")
        architecture = self._extract_section(plan_content, "Architecture").lower()
        if architecture and not any(token in architecture for token in ("metric", "log", "alert", "observe", "monitor")):
            failures.append("Architecture missing observability/monitoring specifics")
        return failures

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

    def _missing_required_sections(
        self, plan_content: str, draft_phase: Literal["planner", "refiner"]
    ) -> list[str]:
        missing: list[str] = []
        if draft_phase == "planner":
            planner_required = {
                "title",
                "goals",
                "non_goals",
                "files_changed",
                "architecture",
            }
            for section_name, pattern in self.plan_schema.required_sections.items():
                if section_name not in planner_required:
                    continue
                if not re.search(pattern, plan_content, re.IGNORECASE | re.MULTILINE):
                    missing.append(section_name)
            return missing
        planner_forbidden = {
            "implementation_steps",
            "test_strategy",
            "rollback_plan",
            "example_code_snippets",
            "user_stories",
        }
        for section_name, pattern in self.plan_schema.required_sections.items():
            if draft_phase == "planner" and section_name in planner_forbidden:
                continue
            if not re.search(pattern, plan_content, re.IGNORECASE | re.MULTILINE):
                missing.append(section_name)

        if self.plan_schema.requires_mermaid:
            has_mermaid_block = "```mermaid" in plan_content.lower()
            has_explicit_mermaid_waiver = (
                "mermaid diagram is unnecessary" in plan_content.lower()
                or "no mermaid diagram needed" in plan_content.lower()
            )
            if not has_mermaid_block and not has_explicit_mermaid_waiver:
                missing.append("mermaid_content")

        if self.plan_schema.requires_code_blocks and draft_phase == "refiner":
            has_any_code_fence = bool(re.search(r"```[a-zA-Z0-9_-]*\n", plan_content))
            if not has_any_code_fence:
                missing.append("example_code_fence")

        if self.plan_schema.requires_ordered_todos:
            has_ordered_todos = bool(re.search(r"^\s*1\.\s+.+", plan_content, re.MULTILINE))
            if not has_ordered_todos:
                missing.append("ordered_todos")

        return missing

    async def _llm_call(
        self,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = True,
        max_output_tokens: int | None = None,
    ):
        import litellm

        litellm.drop_params = True  # gpt-5 models don't support all params (e.g. temperature)
        return await self._safe_completion_call(
            litellm=litellm,
            messages=messages,
            allow_tools=allow_tools,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _prefer_responses_api(model: str) -> bool:
        return model.startswith("gpt-5")

    @staticmethod
    def _responses_tools() -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for entry in CODEBASE_TOOLS:
            fn = entry.get("function", {})
            tools.append(
                {
                    "type": "function",
                    "name": str(fn.get("name", "")),
                    "description": str(fn.get("description", "")),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return tools

    @staticmethod
    def _as_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            if role == "tool":
                payload.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id", "")),
                        "output": str(message.get("content", "")),
                    }
                )
                continue
            if role == "assistant":
                content = str(message.get("content", ""))
                if content:
                    payload.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                tool_calls = message.get("tool_calls", []) or []
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        payload.append(
                            {
                                "type": "function_call",
                                "call_id": str(tc.get("id", "")) if isinstance(tc, dict) else "",
                                "name": str(fn.get("name", "")),
                                "arguments": str(fn.get("arguments", "{}")),
                            }
                        )
                continue
            payload.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": str(message.get("content", ""))}],
                }
            )
        return payload

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        output = getattr(response, "output", None)
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        return ""

    @staticmethod
    def _extract_responses_tool_calls(response: Any) -> list[_ToolCall]:
        tool_calls: list[_ToolCall] = []
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return tool_calls
        for item in output:
            item_type = str(getattr(item, "type", "") or "")
            if item_type != "function_call":
                continue
            name = str(getattr(item, "name", "") or "")
            arguments = str(getattr(item, "arguments", "") or "{}")
            call_id = str(
                getattr(item, "call_id", None)
                or getattr(item, "id", None)
                or f"call_{len(tool_calls)}"
            )
            if not name:
                continue
            tool_calls.append(
                _ToolCall(
                    id=call_id,
                    function=_FunctionCall(name=name, arguments=arguments),
                )
            )
        return tool_calls

    async def _safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        allow_tools: bool,
        max_output_tokens: int | None,
    ) -> tuple[Any, str]:
        """Call completion with fallback for non-chat model misconfiguration."""
        fallback_model = "gpt-4o"
        per_call_timeout_seconds = 20
        output_cap = max(512, min(4000, int(max_output_tokens or 4000)))
        models_to_try = [self.config.author_model]
        if fallback_model != self.config.author_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                if self._prefer_responses_api(model):
                    from openai import OpenAI

                    client = OpenAI()
                    responses_kwargs: dict[str, Any] = {
                        "model": model,
                        "input": self._as_responses_input(messages),
                        "max_output_tokens": output_cap,
                    }
                    if allow_tools:
                        responses_kwargs["tools"] = self._responses_tools()
                        responses_kwargs["tool_choice"] = "auto"
                    response = await asyncio.wait_for(
                        asyncio.to_thread(client.responses.create, **responses_kwargs),
                        timeout=per_call_timeout_seconds,
                    )
                    text = self._extract_responses_text(response)
                    tool_calls = self._extract_responses_tool_calls(response)
                    chat_like = _ChatLikeResponse(
                        choices=[
                            _Choice(
                                message=_Message(
                                    content=text,
                                    tool_calls=tool_calls,
                                )
                            )
                        ],
                        usage=getattr(response, "usage", None),
                    )
                    telemetry = completion_telemetry(response, model=model)
                    context_window = MODEL_CONTEXT_WINDOWS.get(model)
                    if model not in MODEL_CONTEXT_WINDOWS:
                        await self._emit(
                            {
                                "type": "warning",
                                "message": f"Unknown model '{model}' - context window tracking disabled",
                            }
                        )
                    if model not in MODEL_CONTEXT_WINDOWS:
                        await self._emit(
                            {
                                "type": "warning",
                                "message": f"Unknown model '{model}' - cost tracking disabled for this call",
                            }
                        )
                    await self._emit(
                        {
                            "type": "token_usage",
                            "session_stage": "author",
                            "model": model,
                            "prompt_tokens": telemetry.usage.prompt_tokens,
                            "completion_tokens": telemetry.usage.completion_tokens,
                            "call_cost_usd": telemetry.cost.total_cost_usd,
                        }
                    )
                    if context_window and telemetry.usage.prompt_tokens > int(context_window * 0.75):
                        await self._emit(
                            {
                                "type": "warning",
                                "message": (
                                    f"Prompt tokens {telemetry.usage.prompt_tokens} exceed "
                                    f"75% of context window ({context_window}) for {model}"
                                ),
                            }
                        )
                    return chat_like, model
                completion_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": output_cap,
                }
                if allow_tools:
                    completion_kwargs["tools"] = CODEBASE_TOOLS
                    completion_kwargs["tool_choice"] = "auto"
                response = await asyncio.wait_for(
                    asyncio.to_thread(litellm.completion, **completion_kwargs),
                    timeout=per_call_timeout_seconds,
                )
                telemetry = completion_telemetry(response, model=model)
                context_window = MODEL_CONTEXT_WINDOWS.get(model)
                if model not in MODEL_CONTEXT_WINDOWS:
                    await self._emit(
                        {
                            "type": "warning",
                            "message": f"Unknown model '{model}' - context window tracking disabled",
                        }
                    )
                if model not in MODEL_CONTEXT_WINDOWS:
                    await self._emit(
                        {
                            "type": "warning",
                            "message": f"Unknown model '{model}' - cost tracking disabled for this call",
                        }
                    )
                await self._emit(
                    {
                        "type": "token_usage",
                        "session_stage": "author",
                        "model": model,
                        "prompt_tokens": telemetry.usage.prompt_tokens,
                        "completion_tokens": telemetry.usage.completion_tokens,
                        "call_cost_usd": telemetry.cost.total_cost_usd,
                    }
                )
                if context_window and telemetry.usage.prompt_tokens > int(context_window * 0.75):
                    await self._emit(
                        {
                            "type": "warning",
                            "message": (
                                f"Prompt tokens {telemetry.usage.prompt_tokens} exceed "
                                f"75% of context window ({context_window}) for {model}"
                            ),
                        }
                    )
                return response, model
            except asyncio.TimeoutError as exc:
                last_error = RuntimeError(
                    f"Model '{model}' timed out after {per_call_timeout_seconds}s"
                )
                await self._emit(
                    {
                        "type": "warning",
                        "message": (
                            f"Author call timeout on {model} after "
                            f"{per_call_timeout_seconds}s; trying fallback model."
                        ),
                    }
                )
                if idx == len(models_to_try) - 1:
                    raise RuntimeError(
                        "Configured planning author model timed out and fallback also timed out."
                    ) from exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                err_text = str(exc).lower()
                non_chat_model = (
                    "not a chat model" in err_text
                    or "v1/chat/completions" in err_text
                    or "did you mean to use v1/completions" in err_text
                )
                if not non_chat_model or idx == len(models_to_try) - 1:
                    break

        if last_error is not None:
            raise RuntimeError(
                "Configured planning author model failed. "
                f"Last error: {last_error}"
            ) from last_error
        raise RuntimeError("Unknown completion failure during authoring.")

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
        quick_start_mode: bool = False,
    ) -> AuthorResult:
        if max_attempts is None:
            max_attempts = self.config.author_tool_rounds
        self.tool_executor.accessed_paths.clear()
        self.tool_executor.read_history.clear()
        conversation = [
            {"role": "system", "content": AUTHOR_SYSTEM_PROMPT},
            *messages,
        ]
        if draft_phase == "planner" and quick_start_mode:
            if max_output_tokens is None:
                max_output_tokens = 1400
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Startup quick-draft mode is active. Return a concise but grounded starter plan.\n"
                        "Strict constraints:\n"
                        "- Keep to planner-phase sections only (title, goals, non-goals, files changed, architecture).\n"
                        "- No code fences.\n"
                        "- Keep total response under ~900 words.\n"
                        "- Prefer concrete file paths and short rationale over broad prose."
                    ),
                }
            )
        rejection_counts = {
            "rejected_for_no_discovery": 0,
            "rejected_for_grounding": 0,
            "rejected_for_budget": 0,
        }
        rejection_reasons: list[dict[str, str]] = []
        tool_call_timestamps: list[float] = []
        asked_clarification = False
        total_tool_calls = 0
        best_non_empty_content: str = ""
        requirements_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")) == "user"
        )

        try:
            import litellm  # noqa: F401
        except ImportError:
            fallback = self._fallback_plan("\n".join(m.get("content", "") for m in messages))
            refs = extract_file_references(fallback)
            return AuthorResult(
                plan=fallback,
                unverified_references=refs - self.tool_executor.accessed_paths,
                accessed_paths=self.tool_executor.accessed_paths.copy(),
                rejection_counts=rejection_counts,
            )

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
                                "details": (
                                    f"tool-call budget reached at {total_tool_calls}/"
                                    f"{max_tool_calls}"
                                ),
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
                        total_tool_calls += 1
                        tool_call_timestamps.append(asyncio.get_running_loop().time())
                        tool_name = getattr(getattr(tc, "function", None), "name", "")
                        raw_args = getattr(getattr(tc, "function", None), "arguments", "{}") or "{}"
                        try:
                            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
                        except json.JSONDecodeError:
                            parsed_args = {}
                        await self._emit(
                            {
                                "type": "tool_call",
                                "name": tool_name,
                                "session_stage": "author",
                            }
                        )
                        if tool_name == "ask_clarification":
                            asked_clarification = True
                            await self._emit(
                                {
                                    "type": "clarification_needed",
                                    "question": str(parsed_args.get("question", "")),
                                    "context": str(parsed_args.get("context", "")),
                                    "source": "author",
                                }
                            )
                        try:
                            if tool_name == "ask_clarification" and self.clarification_handler is not None:
                                answers = await self.clarification_handler(
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
                                result = self.tool_executor.execute(tc)
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "tool_call_id": getattr(tc, "id", ""),
                                "name": getattr(getattr(tc, "function", None), "name", ""),
                                "result": {"error": str(exc)},
                            }
                        conversation.append(
                            {
                                "role": "tool",
                                "tool_call_id": result["tool_call_id"],
                                "name": result["name"],
                                "content": json.dumps(result["result"]),
                            }
                        )
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
                                    "Explorer gate still failing at final attempt; "
                                    "accepting best-effort draft."
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
                                    "Final attempt: return a complete, non-empty markdown plan now. "
                                    "Do not call tools."
                                ),
                            }
                        )
                        retry_response, _ = await self._llm_call(
                            conversation,
                            allow_tools=False,
                            max_output_tokens=max_output_tokens,
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
                                "content": (
                                    "Your previous response was empty. "
                                    "Return a complete non-empty plan draft now."
                                ),
                            }
                        )
                        continue
                    rejection_reasons.append(
                        {
                            "reason": "AUTHOR_FALLBACK",
                            "details": (
                                "author exhausted attempts with empty final draft "
                                f"after {total_tool_calls} tool calls"
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
                        int(meta.get("line_count", 0))
                        for meta in self.tool_executor.read_history.values()
                    ) / float(len(self.tool_executor.read_history))
                avg_call_spacing = None
                if len(tool_call_timestamps) > 1:
                    deltas = [
                        tool_call_timestamps[i] - tool_call_timestamps[i - 1]
                        for i in range(1, len(tool_call_timestamps))
                    ]
                    avg_call_spacing = sum(deltas) / float(len(deltas))
                return AuthorResult(
                    plan=plan_content,
                    unverified_references=unverified,
                    accessed_paths=verified_paths.copy(),
                    grounding_ratio=grounding_ratio,
                    rejection_counts=rejection_counts,
                    rejection_reasons=rejection_reasons,
                    average_read_depth=avg_read_depth,
                    average_time_between_tool_calls=avg_call_spacing,
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
                    "type": "error",
                    "message": f"Author fallback used: {exc}",
                }
            )
            fallback = self._fallback_plan("\n".join(m.get("content", "") for m in messages))
            refs = extract_file_references(fallback)
            return AuthorResult(
                plan=fallback,
                unverified_references=refs - self.tool_executor.accessed_paths,
                accessed_paths=self.tool_executor.accessed_paths.copy(),
                rejection_counts=rejection_counts,
                rejection_reasons=rejection_reasons,
            )

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
                        "author exhausted attempts without producing a finalized draft "
                        f"(tool_calls={total_tool_calls})"
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
