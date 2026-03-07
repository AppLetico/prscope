"""
Reviewer runtime with strict JSON contract validation.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from ...config import PlanningConfig, RepoProfile
from ...memory import ParsedConstraint
from ...pricing import MODEL_CONTEXT_WINDOWS
from .telemetry import completion_telemetry

PERSPECTIVE_TIMEOUT_SECONDS = 20.0
PERSPECTIVE_SYNTHESIS_BUDGET_SECONDS = 45.0

SCOPE_EXPANSION_PATTERNS = (
    re.compile(r"\bauthentication\b|\bauthorization\b|\bunauthorized\b", re.IGNORECASE),
    re.compile(r"\bdependency checks?\b|\bcritical dependencies\b|\bexternal services?\b|\bdatabase checks?\b", re.IGNORECASE),
    re.compile(r"\bconcurrency\b|\bhigh[- ]load\b|\brace conditions?\b", re.IGNORECASE),
    re.compile(r"\blogging\b|\bmonitoring\b|\btelemetry\b|\bobservability\b", re.IGNORECASE),
    re.compile(r"\bstartup/shutdown\b|\bdocument(?:ation)?\b", re.IGNORECASE),
)


class CriticParseError(RuntimeError):
    """Raised on malformed reviewer contract responses."""


class CriticContractError(RuntimeError):
    """Raised when reviewer contract validation fails."""


@dataclass
class CriticContractSchema:
    required_fields: dict[str, type[Any]] = field(
        default_factory=lambda: {
            "major_issues_remaining": int,
            "minor_issues_remaining": int,
            "hard_constraint_violations": list,
            "critique_complete": bool,
        }
    )
    optional_fields: dict[str, tuple[type[Any], Any]] = field(
        default_factory=lambda: {
            "failure_modes": (list, []),
            "design_tradeoff_risks": (list, []),
            "unsupported_claims": (list, []),
            "missing_evidence": (list, []),
            "critic_confidence": (float, 0.0),
            "operational_readiness": (bool, False),
            "clarification_questions": (list, []),
        }
    )


@dataclass
class ReviewContractSchema:
    required_fields: dict[str, type[Any]] = field(
        default_factory=lambda: {
            "strengths": list,
            "architectural_concerns": list,
            "risks": list,
            "simplification_opportunities": list,
            "blocking_issues": list,
            "reviewer_questions": list,
            "recommended_changes": list,
            "design_quality_score": float,
            "confidence": str,
            "review_complete": bool,
            "resolved_issues": list,
        }
    )
    optional_fields: dict[str, tuple[type[Any], Any]] = field(
        default_factory=lambda: {
            "simplest_possible_design": (str, ""),
            "primary_issue": (str, ""),
            "constraint_violations": (list, []),
            "issue_priority": (list, []),
        }
    )


@dataclass
class ReviewResult:
    strengths: list[str]
    architectural_concerns: list[str]
    risks: list[str]
    simplification_opportunities: list[str]
    blocking_issues: list[str]
    reviewer_questions: list[str]
    recommended_changes: list[str]
    design_quality_score: float
    confidence: str
    review_complete: bool
    simplest_possible_design: str | None
    primary_issue: str | None
    resolved_issues: list[str]
    constraint_violations: list[str]
    issue_priority: list[str]
    prose: str
    parse_error: str | None = None


@dataclass
class ImplementabilityResult:
    implementable: bool
    missing_details: list[str]
    implementation_risks: list[str]
    suggested_additions: list[str]
    prose: str
    parse_error: str | None = None


# Backwards-compatibility aliases during migration.
CriticResult = ReviewResult


REVIEWER_SYSTEM_PROMPT = """You are a senior staff engineer reviewing a design proposal.
Prioritize identifying architectural flaws over minor improvements.
Your goal is to improve the design, not simply criticize it.

Scope discipline rules:
- Preserve the user's requested scope unless broader changes are clearly required by repository evidence or explicit constraints.
- Do not recommend authentication, authorization, cross-service dependency checks, or major contract expansion for a simple health/status endpoint unless the requirements explicitly ask for them or the design would otherwise expose sensitive data.
- A public `/health` endpoint is acceptable by default; treat it as a problem only if the plan exposes secrets, private diagnostics, or privileged controls.
- Prefer tightening observability and failure handling within the stated scope over inventing new platform/security requirements.
- Example: if the request is "Add a lightweight /health endpoint and tests for it", do not escalate to database checks, external-service checks, authentication, or concurrency-control work unless the request or verified evidence explicitly requires that broader behavior.
- For that same lightweight `/health` example, do not turn the plan into logging, monitoring, telemetry, or documentation work unless the user explicitly asks for those deliverables.

First perform structured analysis using these headings:

### Problem Reconstruction
Restate the problem in your own words. What is actually being solved?

### Architecture Model
Reconstruct the proposed architecture. What are the key components, interfaces, and data flows?

### Failure Simulation
Simulate 2-3 realistic failure scenarios. What breaks under concurrency, partial failure, or edge cases?

### Simplification Opportunities
Can any component be removed entirely? Can mechanisms be merged? Can derived state replace stored state?

Then output the JSON review exactly as specified below.
Do NOT wrap JSON in markdown fences.

Required JSON fields:
- strengths: list[str]
- architectural_concerns: list[str]
- risks: list[str]
- simplification_opportunities: list[str]
- blocking_issues: list[str]
- reviewer_questions: list[str]
- recommended_changes: list[str]
- design_quality_score: number in [0, 10]
- confidence: "low" | "medium" | "high"
- review_complete: bool
- simplest_possible_design: string or null
- primary_issue: string or null
- resolved_issues: list[str]
- constraint_violations: list[str] (constraint IDs violated)
- issue_priority: list[str] (issues ranked highest impact first)

Review process (perform in order):
1) Understand the problem
2) Reconstruct the solution architecture
3) Identify key mechanisms
4) Identify assumptions
5) Decompose the plan into execution stages
6) Simulate normal execution path
7) Simulate failure scenarios
8) Explore 2-3 alternative architectures
9) Compare proposed design vs alternatives
10) Perspective analysis (architecture, operations, scalability, failure, simplicity)
11) Evaluate strengths
12) Identify architectural concerns (top 5)
13) Identify risks
14) Look for simplification opportunities
15) Ask reviewer questions
16) Recommend concrete improvements
17) Evaluate constraints and rank issue priority

If a significantly simpler architecture can solve the problem, set simplest_possible_design.
Otherwise set it to null.

Prefer improvements that build on the current design.
Only propose a completely different architecture if the current design has fundamental flaws.

After listing issues, identify the single issue that would most improve the design if fixed.
Return it as primary_issue (or null if no serious issue exists).

Focus primarily on solution quality and architectural clarity.
"""

ARCHITECTURE_PERSPECTIVE_PROMPT = """Focus only on architecture quality.
Analyze component boundaries, responsibilities, coupling, data flow, and scaling implications.
Return concise bullet points with concrete weaknesses and proposed architectural improvements.
"""

OPERATIONS_PERSPECTIVE_PROMPT = """Focus only on operational readiness.
Analyze observability, deployment/rollback, migration safety, and runtime operability risks.
Return concise bullet points with concrete missing controls and mitigations.
"""

FAILURE_PERSPECTIVE_PROMPT = """Focus only on failure simulation.
Simulate 2-3 realistic failure scenarios (concurrency, retries, partial failures, stale state).
Return concise bullet points with failure mode, impact, and mitigation.
"""

SIMPLIFICATION_PERSPECTIVE_PROMPT = """Focus only on simplification opportunities.
Identify components/mechanisms that can be removed, merged, or replaced with simpler alternatives.
Return concise bullet points and one strongest simplified design option.
"""


class CriticAgent:
    def __init__(
        self,
        config: PlanningConfig,
        repo: RepoProfile,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ):
        self.config = config
        self.repo = repo
        self.event_callback = event_callback
        self.schema = ReviewContractSchema()

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        maybe = self.event_callback(event)
        if asyncio.iscoroutine(maybe):
            await maybe

    async def _call_with_telemetry(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        model_override: str | None = None,
    ) -> tuple[str, str]:
        raw, response, model = await asyncio.to_thread(
            self._llm_call,
            messages,
            temperature,
            model_override,
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
            await self._emit(
                {
                    "type": "warning",
                    "message": f"Unknown model '{model}' - cost tracking disabled for this call",
                }
            )
        await self._emit(
            {
                "type": "token_usage",
                "session_stage": "reviewer",
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
        return raw, model

    async def _run_perspective(
        self,
        *,
        perspective_name: str,
        perspective_prompt: str,
        context_blob: str,
        model_override: str | None,
    ) -> str:
        messages = [
            {"role": "system", "content": perspective_prompt},
            {
                "role": "user",
                "content": (
                    f"Perspective: {perspective_name}\n\n{context_blob}\n\nKeep output concise and evidence-oriented."
                ),
            },
        ]
        raw, _ = await self._call_with_telemetry(
            messages=messages,
            temperature=0.1,
            model_override=model_override,
        )
        return raw.strip()

    @staticmethod
    def _is_non_chat_model_error(exc: Exception) -> bool:
        err_text = str(exc).lower()
        return (
            "not a chat model" in err_text
            or "v1/chat/completions" in err_text
            or "did you mean to use v1/completions" in err_text
        )

    @staticmethod
    def _prefer_responses_api(model: str) -> bool:
        # GPT-5 variants are frequently exposed via responses/completions contracts
        # rather than legacy chat-completions semantics.
        return model.startswith("gpt-5")

    @staticmethod
    def _as_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            payload.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
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
    def _count_file_references(plan_content: str) -> int:
        return len(set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan_content or "")))

    def _should_run_multi_perspective(
        self,
        *,
        mode: Literal["initial", "validation", "stabilization", "implementability"],
        plan_content: str,
        round_number: int,
    ) -> bool:
        if mode != "initial":
            return False
        if round_number > 1:
            return False
        file_refs = self._count_file_references(plan_content)
        normalized_plan = (plan_content or "").strip()
        if file_refs <= 2 and len(normalized_plan) <= 2200:
            return False
        return True

    def _llm_call(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        model_override: str | None = None,
    ) -> tuple[str, Any, str]:
        import litellm

        litellm.drop_params = True  # gpt-5 models don't support all params (e.g. temperature)
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        fallback_model = "gpt-4o-mini"
        primary_model = model_override or self.config.critic_model
        models_to_try = [primary_model]
        if fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                if self._prefer_responses_api(model):
                    from openai import OpenAI

                    client = OpenAI()
                    response = client.responses.create(
                        model=model,
                        input=self._as_responses_input(messages),
                        max_output_tokens=3000,
                    )
                    text = self._extract_responses_text(response)
                    if text:
                        return text, response, model
                    raise RuntimeError("Empty response text from OpenAI Responses API")
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens=3000,
                    temperature=temperature,
                )
                return str(response.choices[0].message.content or "").strip(), response, model
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                non_chat_model = self._is_non_chat_model_error(exc)
                if non_chat_model:
                    try:
                        from openai import OpenAI

                        client = OpenAI()
                        response = client.responses.create(
                            model=model,
                            input=self._as_responses_input(messages),
                            max_output_tokens=3000,
                        )
                        text = self._extract_responses_text(response)
                        if text:
                            return text, response, model
                        raise RuntimeError("Empty response text from OpenAI Responses API")
                    except Exception as response_exc:  # noqa: BLE001
                        last_error = response_exc
                if idx == len(models_to_try) - 1:
                    break

        if last_error is not None:
            raise RuntimeError(
                "Configured planning critic model is incompatible with chat completions."
            ) from last_error
        raise RuntimeError("Unknown completion failure during critique.")

    @staticmethod
    def _extract_first_json_object(raw: str) -> tuple[str, str]:
        start = raw.find("{")
        if start < 0:
            raise CriticParseError("No JSON block found in critic response")
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end < 0:
            raise CriticParseError("Unterminated JSON object in critic response")
        return raw[start:end], raw[end:].strip()

    def _parse_review_response(
        self,
        raw: str,
    ) -> ReviewResult:
        json_text, prose = self._extract_first_json_object(raw)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise CriticParseError(f"Malformed review JSON: {exc}") from exc

        for field_name, expected_type in self.schema.required_fields.items():
            if field_name not in data:
                raise CriticParseError(f"Missing required field: {field_name}")
            if field_name == "design_quality_score":
                if not isinstance(data[field_name], (int, float)):
                    raise CriticParseError(f"Wrong type for {field_name}: expected number")
            elif not isinstance(data[field_name], expected_type):
                raise CriticParseError(f"Wrong type for {field_name}: expected {expected_type.__name__}")

        optional_values: dict[str, Any] = {}
        for field_name, (expected_type, default_value) in self.schema.optional_fields.items():
            if field_name not in data:
                optional_values[field_name] = default_value
                continue
            value = data[field_name]
            if field_name in {"simplest_possible_design", "primary_issue"}:
                optional_values[field_name] = self._coerce_optional_text(value)
                continue
            if not isinstance(value, expected_type):
                raise CriticParseError(f"Wrong type for {field_name}: expected {expected_type.__name__}")
            optional_values[field_name] = value

        score = float(data["design_quality_score"])
        if score < 0.0 or score > 10.0:
            raise CriticParseError("design_quality_score must be in [0,10]")
        confidence = str(data["confidence"]).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            raise CriticParseError("confidence must be one of: low, medium, high")

        simplest = str(optional_values.get("simplest_possible_design", "")).strip() or None
        primary_issue = str(optional_values.get("primary_issue", "")).strip() or None
        constraint_violations = [str(item) for item in optional_values.get("constraint_violations", [])]
        issue_priority = [str(item) for item in optional_values.get("issue_priority", [])]

        return ReviewResult(
            strengths=[str(item) for item in data["strengths"]],
            architectural_concerns=[str(item) for item in data["architectural_concerns"]],
            risks=[str(item) for item in data["risks"]],
            simplification_opportunities=[str(item) for item in data["simplification_opportunities"]],
            blocking_issues=[str(item) for item in data["blocking_issues"]],
            reviewer_questions=[str(item) for item in data["reviewer_questions"]],
            recommended_changes=[str(item) for item in data["recommended_changes"]],
            design_quality_score=score,
            confidence=confidence,
            review_complete=bool(data["review_complete"]),
            simplest_possible_design=simplest,
            primary_issue=primary_issue,
            resolved_issues=[str(item) for item in data["resolved_issues"]],
            constraint_violations=constraint_violations,
            issue_priority=issue_priority,
            prose=prose,
        )

    @staticmethod
    def _is_lightweight_health_request(requirements: str) -> bool:
        lowered = str(requirements or "").lower()
        if "/health" not in lowered:
            return False
        return any(token in lowered for token in ("lightweight", "simple", "basic"))

    @staticmethod
    def _is_scope_expansion_feedback(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in SCOPE_EXPANSION_PATTERNS)

    def _apply_scope_discipline(self, requirements: str, review: ReviewResult) -> ReviewResult:
        if not self._is_lightweight_health_request(requirements):
            return review

        def _filter(items: list[str]) -> list[str]:
            return [item for item in items if not self._is_scope_expansion_feedback(item)]

        blocking_issues = _filter(review.blocking_issues)
        recommended_changes = _filter(review.recommended_changes)
        architectural_concerns = _filter(review.architectural_concerns)
        risks = _filter(review.risks)
        reviewer_questions = _filter(review.reviewer_questions)
        issue_priority = _filter(review.issue_priority)
        primary_issue = review.primary_issue
        if primary_issue and self._is_scope_expansion_feedback(primary_issue):
            primary_issue = issue_priority[0] if issue_priority else None

        return ReviewResult(
            strengths=review.strengths,
            architectural_concerns=architectural_concerns,
            risks=risks,
            simplification_opportunities=review.simplification_opportunities,
            blocking_issues=blocking_issues,
            reviewer_questions=reviewer_questions,
            recommended_changes=recommended_changes,
            design_quality_score=review.design_quality_score,
            confidence=review.confidence,
            review_complete=review.review_complete,
            simplest_possible_design=review.simplest_possible_design,
            primary_issue=primary_issue,
            resolved_issues=review.resolved_issues,
            constraint_violations=review.constraint_violations,
            issue_priority=issue_priority,
            prose=review.prose,
            parse_error=review.parse_error,
        )

    @staticmethod
    def _coerce_optional_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return "; ".join(parts)
        if isinstance(value, dict):
            normalized = {str(key): value_for_key for key, value_for_key in value.items() if str(key).strip()}
            return json.dumps(normalized, sort_keys=True) if normalized else ""
        if isinstance(value, (int, float, bool)):
            return str(value)
        raise CriticParseError("Wrong type for optional text field")

    def _parse_implementability_response(self, raw: str) -> ImplementabilityResult:
        json_text, prose = self._extract_first_json_object(raw)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise CriticParseError(f"Malformed implementability JSON: {exc}") from exc
        required = {
            "implementable": bool,
            "missing_details": list,
            "implementation_risks": list,
            "suggested_additions": list,
        }
        for key, expected in required.items():
            if key not in data:
                raise CriticParseError(f"Missing required field: {key}")
            if not isinstance(data[key], expected):
                raise CriticParseError(f"Wrong type for {key}: expected {expected.__name__}")
        return ImplementabilityResult(
            implementable=bool(data["implementable"]),
            missing_details=[str(item) for item in data["missing_details"]],
            implementation_risks=[str(item) for item in data["implementation_risks"]],
            suggested_additions=[str(item) for item in data["suggested_additions"]],
            prose=prose,
        )

    @staticmethod
    def _mode_prompt(mode: Literal["initial", "validation", "stabilization", "implementability"]) -> str:
        if mode == "validation":
            return (
                "Validation mode:\n"
                "Evaluate whether recent section updates resolved previously identified issues.\n"
                "Do not introduce new critiques unless they represent serious defects.\n"
                "Focus on confirming or rejecting fixes.\n"
                "Keep the score stable when the revision meaningfully addresses the prior issue set.\n"
                "Do not reduce the score for low-severity ambiguity that can be handled by existing repo conventions."
            )
        if mode == "stabilization":
            return (
                "Stabilization mode:\n"
                "The design has changed significantly across rounds without improvement.\n"
                "Focus on stabilizing the architecture and refining the current design.\n"
                "Avoid proposing brand-new architectural directions unless required.\n"
                "Prefer incremental closure of already-known issues over surfacing new minor concerns.\n"
                "Only reduce the design score if you observe a concrete regression or still-open major issue."
            )
        if mode == "implementability":
            return (
                "Implementability check mode:\n"
                "Return JSON fields:\n"
                "- implementable (bool)\n"
                "- missing_details (list[str])\n"
                "- implementation_risks (list[str])\n"
                "- suggested_additions (list[str])\n\n"
                "Evaluate whether the plan can be implemented without additional design work.\n"
                "Check: concrete steps, component boundaries, code locations, test strategy, rollback/migration."
            )
        return "Initial review mode: perform full structured design review."

    async def run_design_review(
        self,
        *,
        requirements: str,
        plan_content: str,
        architecture: str,
        design_record: str = "",
        modules: str = "",
        patterns: str = "",
        constraints: list[ParsedConstraint],
        manifesto: str = "",
        prior_critique: str | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        strict_mode: bool = False,
        model_override: str | None = None,
        session_id: str = "",
        round_number: int = 0,
        mode: Literal["initial", "validation", "stabilization", "implementability"] = "initial",
    ) -> ReviewResult | ImplementabilityResult:
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise RuntimeError("litellm is required for critic reviews but is not installed")

        prior_critique_blob = prior_critique.strip() if prior_critique else "None"
        constraints_block = (
            "\n".join(f"- {item.id} ({item.severity}): {item.text}" for item in constraints)
            if constraints
            else "- (none)"
        )
        manifesto_excerpt = (manifesto or "").strip()[:4000]
        context_blob = (
            f"## Mode\n{mode}\n\n"
            f"{self._mode_prompt(mode)}\n\n"
            f"## Requirements\n{requirements}\n\n"
            f"## Planning Constraints\n{constraints_block}\n\n"
            f"## Manifesto (excerpt)\n{manifesto_excerpt or '(none)'}\n\n"
            f"## Architecture\n{architecture}\n\n"
            f"## Design Record\n{design_record or '(none)'}\n\n"
            f"## Module Structure\n{modules}\n\n"
            f"## Patterns\n{patterns}\n\n"
            f"## Prior Review\n{prior_critique_blob}\n\n"
            f"## Current Plan\n{plan_content}\n\n"
            "Evaluate constraints explicitly in your JSON output.\n"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": context_blob},
        ]

        if self._should_run_multi_perspective(
            mode=mode,
            plan_content=plan_content,
            round_number=round_number,
        ):
            try:
                started = time.perf_counter()
                perspective_results = await asyncio.gather(
                    asyncio.wait_for(
                        self._run_perspective(
                            perspective_name="architecture",
                            perspective_prompt=ARCHITECTURE_PERSPECTIVE_PROMPT,
                            context_blob=context_blob,
                            model_override=model_override,
                        ),
                        timeout=PERSPECTIVE_TIMEOUT_SECONDS,
                    ),
                    asyncio.wait_for(
                        self._run_perspective(
                            perspective_name="operations",
                            perspective_prompt=OPERATIONS_PERSPECTIVE_PROMPT,
                            context_blob=context_blob,
                            model_override=model_override,
                        ),
                        timeout=PERSPECTIVE_TIMEOUT_SECONDS,
                    ),
                    asyncio.wait_for(
                        self._run_perspective(
                            perspective_name="failure",
                            perspective_prompt=FAILURE_PERSPECTIVE_PROMPT,
                            context_blob=context_blob,
                            model_override=model_override,
                        ),
                        timeout=PERSPECTIVE_TIMEOUT_SECONDS,
                    ),
                    asyncio.wait_for(
                        self._run_perspective(
                            perspective_name="simplification",
                            perspective_prompt=SIMPLIFICATION_PERSPECTIVE_PROMPT,
                            context_blob=context_blob,
                            model_override=model_override,
                        ),
                        timeout=PERSPECTIVE_TIMEOUT_SECONDS,
                    ),
                    return_exceptions=True,
                )
                failures = sum(1 for result in perspective_results if isinstance(result, Exception))
                elapsed = time.perf_counter() - started
                if failures >= 2 or elapsed > PERSPECTIVE_SYNTHESIS_BUDGET_SECONDS:
                    raise RuntimeError(f"perspective guardrail triggered (failures={failures}, elapsed={elapsed:.1f}s)")
                architecture_review, operations_review, failure_review, simplification_review = [
                    result if isinstance(result, str) else "(perspective unavailable)" for result in perspective_results
                ]
                messages = [
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{context_blob}\n\n"
                            "## Perspective Inputs\n\n"
                            f"### Architecture Perspective\n{architecture_review}\n\n"
                            f"### Operations Perspective\n{operations_review}\n\n"
                            f"### Failure Perspective\n{failure_review}\n\n"
                            f"### Simplification Perspective\n{simplification_review}\n\n"
                            "Synthesize these into the final review contract."
                        ),
                    },
                ]
            except Exception as exc:  # noqa: BLE001
                await self._emit(
                    {
                        "type": "warning",
                        "message": (
                            "Multi-perspective review fell back to single-pass "
                            f"due to timeout/failure guardrails: {exc}"
                        ),
                    }
                )

        temp = 0.0 if temperature is None else temperature
        for attempt in range(max_retries + 1):
            try:
                raw, _ = await self._call_with_telemetry(
                    messages=messages,
                    temperature=temp,
                    model_override=model_override,
                )
            except Exception as exc:
                raise CriticContractError(f"Reviewer call failed: {exc}") from exc
            try:
                if mode == "implementability":
                    return self._parse_implementability_response(raw)
                parsed = self._parse_review_response(raw)
                return self._apply_scope_discipline(requirements, parsed)
            except CriticParseError as exc:
                if strict_mode and attempt >= max_retries:
                    raise CriticContractError(f"Reviewer contract parse failure: {exc}") from exc
                if attempt < max_retries:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Formatting error: {exc}. Respond again with valid JSON first, then prose analysis."
                            ),
                        }
                    )
                    continue
                raise CriticContractError(f"Reviewer parse failed after {max_retries + 1} attempts: {exc}") from exc

        raise CriticContractError("Reviewer exhausted retries without producing a valid contract")

    async def run_critic(
        self,
        *,
        requirements: str,
        plan_content: str,
        manifesto: str,
        architecture: str,
        constraints: list[ParsedConstraint],
        prior_critique: str | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        strict_mode: bool = False,
        model_override: str | None = None,
        session_id: str = "",
        round_number: int = 0,
    ) -> ReviewResult:
        result = await self.run_design_review(
            requirements=requirements,
            plan_content=plan_content,
            manifesto=manifesto,
            architecture=architecture,
            constraints=constraints,
            prior_critique=prior_critique,
            max_retries=max_retries,
            temperature=temperature,
            strict_mode=strict_mode,
            model_override=model_override,
            session_id=session_id,
            round_number=round_number,
            mode="initial",
        )
        if isinstance(result, ImplementabilityResult):
            raise CriticContractError("run_critic alias returned implementability payload unexpectedly")
        return result

    async def validate_headless(
        self,
        *,
        session_id: str,
        round_number: int,
        plan_sha: str,
        requirements: str,
        plan_content: str,
        manifesto: str,
        architecture: str,
        constraints: list[ParsedConstraint],
    ) -> ReviewResult:
        result = await self.run_design_review(
            requirements=requirements,
            plan_content=plan_content,
            manifesto=manifesto,
            architecture=architecture,
            constraints=constraints,
            prior_critique=None,
            max_retries=2,
            temperature=self.config.validate_temperature,
            strict_mode=True,
            mode="validation",
        )
        if isinstance(result, ImplementabilityResult):
            raise CriticContractError("validate_headless received implementability payload unexpectedly")

        if self.config.validate_audit_log:
            audit_dir = Path.home() / ".prscope" / "repos" / self.repo.name / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            audit_path = audit_dir / f"{session_id}-validate-{stamp}.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "round": round_number,
                        "plan_sha": plan_sha,
                        "review_result": asdict(result),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result
