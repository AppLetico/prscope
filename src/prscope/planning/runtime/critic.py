"""
Reviewer runtime with strict JSON contract validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from ...config import PlanningConfig, RepoProfile
from ...memory import ParsedConstraint
from ...model_catalog import (
    litellm_model_name,
    model_has_elevated_json_contract_risk,
    model_prefers_compact_json,
    model_provider,
)
from ...pricing import MODEL_CONTEXT_WINDOWS
from .telemetry import completion_telemetry

_LOG = logging.getLogger(__name__)

PERSPECTIVE_TIMEOUT_SECONDS = 20.0
PERSPECTIVE_SYNTHESIS_BUDGET_SECONDS = 45.0

SCOPE_EXPANSION_PATTERNS = (
    re.compile(r"\bauthentication\b|\bauthorization\b|\bunauthorized\b", re.IGNORECASE),
    re.compile(
        r"\bdependency checks?\b|\bcritical dependencies\b|\bexternal services?\b|\bdatabase checks?\b", re.IGNORECASE
    ),
    re.compile(r"\bconcurrency\b|\bhigh[- ]load\b|\brace conditions?\b", re.IGNORECASE),
    re.compile(r"\blogging\b|\bmonitoring\b|\btelemetry\b|\bobservability\b", re.IGNORECASE),
    re.compile(r"\bstartup/shutdown\b|\bdocument(?:ation)?\b", re.IGNORECASE),
    re.compile(r"\bpartial failures?\b|\bdependency failures?\b|\btimeouts?\b", re.IGNORECASE),
    re.compile(r"\baccurate health status\b|\bactual service health\b|\boverall system health\b", re.IGNORECASE),
    re.compile(r"\bother services?\b|\bcritical service\b|\bservice health\b", re.IGNORECASE),
    re.compile(r"\bdetailed health checks?\b|\bhealth check logic\b", re.IGNORECASE),
    re.compile(r"\bmislead users about (?:application|service) status\b", re.IGNORECASE),
)

LOCALIZED_REUSE_SCOPE_PATTERNS = (
    re.compile(
        r"\bservice layer\b|\bshared utility module\b|\bstate management\b|\bcentralized state management\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brefactor common .* logic\b|\breduce coupling\b|\btight integration\b|\bcoupling between .* may increase complexity\b|"
        r"\bcoupling between .* may hinder reusability\b|\bdry principles\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpage actions\b|\bsession management controls\b", re.IGNORECASE),
    re.compile(r"\bextensible for future formats\b|\bfuture formats\b|\bextensib(?:le|ility)\b", re.IGNORECASE),
    re.compile(r"\blogging\b|\btelemetry\b|\bobservability\b|\buat\b|\buser acceptance testing\b", re.IGNORECASE),
    re.compile(
        r"\bdedicated export (?:handler|service)\b|\babstract api calls\b|\bversioning for the frontend components\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfeature flags?\b|\broll out changes incrementally\b", re.IGNORECASE),
    re.compile(
        r"\bdedicated .* context\b|\bdedicated .* hook\b|\bcentralized state management solution\b", re.IGNORECASE
    ),
)

# Plan harness: rubric axes, blocking categories (closed vocabulary), missing rubric fill.
PLAN_RUBRIC_AXIS_KEYS: tuple[str, ...] = (
    "specificity",
    "testability",
    "coherence",
    "evidence_alignment",
    "intent_alignment",
)
VALID_BLOCKING_CATEGORIES: frozenset[str] = frozenset({"testability", "evidence", "vagueness", "scope"})
MISSING_RUBRIC_SCORE: float = 0.0


def _default_plan_rubric() -> dict[str, float]:
    return dict.fromkeys(PLAN_RUBRIC_AXIS_KEYS, 8.0)


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
    # Harness (Tier A): enforced at convergence; permissive defaults for tests / manual construction.
    plan_rubric: dict[str, float] = field(default_factory=_default_plan_rubric)
    rubric_incomplete: bool = False
    blocking_categories: list[str] = field(default_factory=list)
    blocking_categories_invalid: bool = False
    acceptance_criteria: list[str] = field(default_factory=list)

    @property
    def blockers_ok(self) -> bool:
        return len(self.blocking_categories) == 0 and not self.blocking_categories_invalid

    def min_plan_rubric_score(self) -> float:
        if not self.plan_rubric:
            return 0.0
        return float(min(self.plan_rubric.values()))


def synthetic_review_for_chat_refinement(
    issue_tracker: Any,
    *,
    user_input: str,
) -> ReviewResult:
    """When chat drives a refine round but no stored critique exists, avoid running design_review.

    Repair/revise still need a ReviewResult; we derive blocking issues from the issue graph and/or
    the user's message. A full scored critique only runs via run_critique (Review button).
    """
    issues: list[str] = []
    try:
        for node in issue_tracker.open_issues()[:32]:
            desc = str(getattr(node, "description", "") or "").strip()
            if desc:
                issues.append(desc)
    except Exception:  # noqa: BLE001 — best-effort; empty issues still ok
        issues = []
    ui = user_input.strip()
    if not issues and ui:
        issues = [ui[:8000]]
    primary = issues[0] if issues else None
    prose = (
        "Chat refinement context (no automatic critic design review). "
        "Use Review when you want a fresh scored critique of the plan."
    )
    return ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=issues,
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=5.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue=primary,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=issues[:8],
        prose=prose,
        parse_error=None,
    )


def skipped_validation_review_placeholder() -> ReviewResult:
    """Synthetic review result when validation is deferred (e.g. after apply_critique).

    Used for convergence/metrics only — no critic call. Keeps convergence conservative
    (design_quality_score 0, review_complete False) so we do not auto-converge.
    """
    return ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=0.0,
        confidence="n/a",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="Validation deferred — run Review for a fresh critic pass on the updated plan.",
        parse_error=None,
    )


def hydrate_review_result(raw: Any) -> ReviewResult | None:
    """Restore ReviewResult from a persisted state snapshot dict (best-effort)."""
    if not isinstance(raw, dict):
        return None
    try:
        pr = raw.get("plan_rubric")
        plan_rubric: dict[str, float] = dict(pr) if isinstance(pr, dict) else _default_plan_rubric()
        return ReviewResult(
            strengths=list(raw.get("strengths") or []),
            architectural_concerns=list(raw.get("architectural_concerns") or []),
            risks=list(raw.get("risks") or []),
            simplification_opportunities=list(raw.get("simplification_opportunities") or []),
            blocking_issues=list(raw.get("blocking_issues") or []),
            reviewer_questions=list(raw.get("reviewer_questions") or []),
            recommended_changes=list(raw.get("recommended_changes") or []),
            design_quality_score=float(raw.get("design_quality_score") or 0.0),
            confidence=str(raw.get("confidence") or "medium"),
            review_complete=bool(raw.get("review_complete", True)),
            simplest_possible_design=raw.get("simplest_possible_design"),
            primary_issue=raw.get("primary_issue"),
            resolved_issues=list(raw.get("resolved_issues") or []),
            constraint_violations=list(raw.get("constraint_violations") or []),
            issue_priority=list(raw.get("issue_priority") or []),
            prose=str(raw.get("prose") or ""),
            parse_error=raw.get("parse_error"),
            plan_rubric=plan_rubric,
            rubric_incomplete=bool(raw.get("rubric_incomplete", False)),
            blocking_categories=list(raw.get("blocking_categories") or []),
            blocking_categories_invalid=bool(raw.get("blocking_categories_invalid", False)),
            acceptance_criteria=list(raw.get("acceptance_criteria") or []),
        )
    except (TypeError, ValueError):
        return None


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

_HARNESS_JSON_SENTINEL = object()

REVIEWER_CALIBRATION_FEWSHOTS = """
## Required calibration examples (follow this strictness)

### BAD (too lenient — do NOT emulate)
Plan: "We will improve the API and add tests." Reviewer gives design_quality_score 8, empty blocking_issues, vague praise.
This is INVALID: vagueness must be a blocking issue; testability and evidence_alignment rubric scores must be low; blocking_categories should include vagueness and testability.

### GOOD (appropriate gatekeeper)
Same plan. Reviewer: blocking_issues include "No concrete routes, files, or acceptance tests named"; design_quality_score <= 6; plan_rubric testability <= 4; evidence_alignment <= 4; blocking_categories: ["vagueness","testability"]; acceptance_criteria rewritten to falsifiable bullets only.

You must behave like the GOOD example, not the BAD one.
"""

REVIEWER_SYSTEM_PROMPT = (
    """You are an adversarial staff engineer acting as a GATEKEEPER, not a polite collaborator.
Optimize for correctness, testability, and evidence — not for agreement or a smooth conversation.
"Seems fine" is never sufficient justification. If you label an issue minor, you must state explicitly why it is NOT major.

GLOBAL RULE (non-negotiable): Mode-specific user hints (validation, stabilization) must NEVER override blocking standards.
If the plan is vague, untestable, or makes claims without repo path / constraint / manifesto / prior-decision linkage, that remains BLOCKING regardless of mode.

Gatekeeper standards:
- Vague plans → blocking_issues (not buried in minor concerns).
- Missing or non-actionable test strategy → blocking; use blocking_categories "testability" when applicable.
- Architectural or behavioral claims without evidence (backtick file paths, constraints, manifesto, decision graph) → blocking; use "evidence".
- Bare filenames alone (for example a lone `api.py`) are weak evidence unless tied to a verified repo-relative path from exploration or the decision graph; prefer concrete paths seen in tool history.
- If the plan names a module as an HTTP client to the web server (benchmarks, CLI, tests), require evidence tying it to real `fetch`/`httpx`/`TestClient`/`request.get` usage; otherwise blocking with "evidence" when the citation looks invented.
- When a 'Repository paths explored this session' section is present, treat those paths as the session's primary file-evidence set for grounding checks (in addition to the plan text and memory blocks).
- Prefer false positives over silent misses on *subtle* gaps, but never claim something is missing when it is plainly present in ## Current Plan (that wastes users' time). Before emitting blocking_issues, scan ## Current Plan for HTTP methods and paths, a test strategy section, and backtick file paths; do not use boilerplate "no endpoints / no tests / no evidence" language if that category of detail already appears — instead cite what is still incomplete, unverifiable, or weakly grounded.
- Never downgrade a serious gap to "architectural_concern" to avoid conflict.

Plan-text fidelity (mandatory):
- Read ## Current Plan before writing blocking_issues. If the plan already names concrete routes, tests, or repo paths, your blocking_issues must reflect *remaining* risk (weak linkage, wrong files, missing edge cases), not blanket absence.

Scope drift vs original requirements:
- Explicitly compare the plan to the user's stated requirements. If the plan adds major work not asked for (unrelated subsystems, broad refactors, extra deliverables) or omits a core part of the ask without evidence-backed justification, call it out in blocking_issues or recommended_changes.
- Use blocking_categories "scope" when the mismatch is over- or under-shooting the user's goals. Lower intent_alignment when the plan diverges from what was requested.

Scope discipline rules (still apply when they do not contradict the gatekeeper standards above):
- Preserve the user's requested scope unless broader changes are clearly required by repository evidence or explicit constraints.
- Do not recommend authentication, authorization, cross-service dependency checks, or major contract expansion for a simple health/status endpoint unless the requirements explicitly ask for them or the design would otherwise expose sensitive data.
- A public `/health` endpoint is acceptable by default; treat it as a problem only if the plan exposes secrets, private diagnostics, or privileged controls.
- Prefer tightening observability and failure handling within the stated scope over inventing new platform/security requirements.
- Example: if the request is "Add a lightweight /health endpoint and tests for it", do not escalate to database checks, external-service checks, authentication, or concurrency-control work unless the request or verified evidence explicitly requires that broader behavior.
- For that same lightweight `/health` example, do not turn the plan into logging, monitoring, telemetry, or documentation work unless the user explicitly asks for those deliverables.
- If the user narrows scope further with guidance like "keep the response simple", "keep it public", or "limit tests to the happy-path 200 response", do not criticize the plan for lacking dependency-failure, timeout, partial-failure, or overall-system-health semantics.
- For a lightweight `/health` request, avoid treating the absence of "detailed health checks" as a blocking flaw when the plan already returns a simple public 200 response and explicitly excludes dependency checks.
- For localized UI or API-wiring requests that explicitly say to reuse existing helpers/endpoints and avoid new endpoints, do not escalate into new service layers, shared utility modules, state-management rewrites, or broad page-action redesigns unless verified repository evidence shows the current structure cannot support the request.
- For localized UI requests that simply say to show the latest result/status, do not treat unspecified display formatting as a blocking flaw; a simple success/failure presentation is acceptable unless the requirements or verified repository evidence call for richer formatting.

"""
    + REVIEWER_CALIBRATION_FEWSHOTS
    + """

Acceptance criteria (per round):
- Emit acceptance_criteria: list[str] of short, FALSIFIABLE bullets for what "done" means this round (observable in the plan markdown).
- Reject subjective fluff: do NOT use criteria like "plan is clear", "architecture is solid", "robust", "good enough".
- Each criterion should be checkable from the plan text (sections, file paths, named tests, enumerated behaviors).

plan_rubric object (required): scores 0–10 for EACH key exactly:
- specificity: actionable detail vs generic boilerplate
- testability: verifiable steps / tests / acceptance
- coherence: the plan hangs together
- evidence_alignment: claims tied to repo paths, constraints, manifesto, or recorded decisions; low if claims lack linkage
- intent_alignment: plan matches the user's stated goals and scope; low when the plan drifts, gold-plates, or misses the core ask

blocking_categories (required): list[str], each MUST be one of: testability | evidence | vagueness | scope
Use [] only when no category applies. Do NOT invent new category strings.

First perform structured analysis using these headings:

### Problem Reconstruction
### Architecture Model
### Failure Simulation
### Simplification Opportunities

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
- plan_rubric: object with keys specificity, testability, coherence, evidence_alignment, intent_alignment (numbers [0,10])
- blocking_categories: list[str] (subset of testability|evidence|vagueness|scope, or [])
- acceptance_criteria: list[str] (falsifiable bullets; use [] if none this round)

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
17b) Compare plan scope to stated requirements; assign intent_alignment and scope-related blocking_categories when drift exists
18) Assign plan_rubric, blocking_categories, acceptance_criteria consistent with blocking_issues

If a significantly simpler architecture can solve the problem, set simplest_possible_design.
Otherwise set it to null.

Prefer improvements that build on the current design.
Only propose a completely different architecture if the current design has fundamental flaws.

After listing issues, identify the single issue that would most improve the design if fixed.
Return it as primary_issue (or null if no serious issue exists).
"""
)

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
            _LOG.warning("Unknown model '%s' - context window tracking disabled", model)
            _LOG.warning("Unknown model '%s' - cost tracking disabled for this call", model)
        await self._emit(
            {
                "type": "token_usage",
                "session_stage": "reviewer",
                "model": model,
                "model_provider": model_provider(model),
                "prompt_tokens": telemetry.usage.prompt_tokens,
                "completion_tokens": telemetry.usage.completion_tokens,
                "call_cost_usd": telemetry.cost.total_cost_usd,
            }
        )
        if context_window and telemetry.usage.prompt_tokens > int(context_window * 0.75):
            _LOG.warning(
                "Prompt tokens %s exceed 75%% of context window (%s) for %s",
                telemetry.usage.prompt_tokens,
                context_window,
                model,
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
            litellm_model = litellm_model_name(model)
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
                    model=litellm_model,
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

    def _extract_review_json_object(self, raw: str) -> tuple[str, str]:
        """Extract the review JSON, skipping prose-embedded objects like {\"status\": \"healthy\"}."""
        required_keys = set(self.schema.required_fields)
        fallback: tuple[str, str] | None = None
        search_start = 0
        while True:
            start = raw.find("{", search_start)
            if start < 0:
                if fallback is not None:
                    return fallback
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
            json_text = raw[start:end]
            try:
                data = json.loads(json_text)
                if isinstance(data, dict):
                    if required_keys.issubset(data.keys()):
                        return json_text, raw[end:].strip()
                    if fallback is None:
                        fallback = (json_text, raw[end:].strip())
            except json.JSONDecodeError:
                pass
            search_start = end

    @staticmethod
    def _parse_plan_rubric_field(data: dict[str, Any]) -> tuple[dict[str, float], bool]:
        raw = data.get("plan_rubric", _HARNESS_JSON_SENTINEL)
        incomplete = False
        if raw is _HARNESS_JSON_SENTINEL:
            incomplete = True
            return dict.fromkeys(PLAN_RUBRIC_AXIS_KEYS, MISSING_RUBRIC_SCORE), incomplete
        if not isinstance(raw, dict):
            raise CriticParseError("plan_rubric must be an object")
        rubric: dict[str, float] = {}
        for axis in PLAN_RUBRIC_AXIS_KEYS:
            if axis not in raw:
                incomplete = True
                rubric[axis] = MISSING_RUBRIC_SCORE
                continue
            val = raw[axis]
            if not isinstance(val, (int, float)):
                raise CriticParseError(f"plan_rubric.{axis} must be a number")
            score = float(val)
            if score < 0.0 or score > 10.0:
                raise CriticParseError(f"plan_rubric.{axis} must be in [0,10]")
            rubric[axis] = score
        return rubric, incomplete

    @staticmethod
    def _parse_blocking_categories_field(data: dict[str, Any]) -> tuple[list[str], bool]:
        raw = data.get("blocking_categories", _HARNESS_JSON_SENTINEL)
        if raw is _HARNESS_JSON_SENTINEL:
            return [], True
        if not isinstance(raw, list):
            raise CriticParseError("blocking_categories must be a list")
        normalized: list[str] = []
        invalid = False
        for item in raw:
            token = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            if token in VALID_BLOCKING_CATEGORIES:
                if token not in normalized:
                    normalized.append(token)
            else:
                invalid = True
        return normalized, invalid

    @staticmethod
    def _parse_acceptance_criteria_field(data: dict[str, Any]) -> list[str]:
        raw = data.get("acceptance_criteria", [])
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise CriticParseError("acceptance_criteria must be a list")
        return [str(x).strip() for x in raw if str(x).strip()]

    def _parse_review_response(
        self,
        raw: str,
    ) -> ReviewResult:
        json_text, prose = self._extract_review_json_object(raw)
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

        plan_rubric, rubric_incomplete = CriticAgent._parse_plan_rubric_field(data)
        blocking_categories, blocking_categories_invalid = CriticAgent._parse_blocking_categories_field(data)
        acceptance_criteria = CriticAgent._parse_acceptance_criteria_field(data)

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
            plan_rubric=plan_rubric,
            rubric_incomplete=rubric_incomplete,
            blocking_categories=blocking_categories,
            blocking_categories_invalid=blocking_categories_invalid,
            acceptance_criteria=acceptance_criteria,
        )

    @staticmethod
    def _is_lightweight_health_request(requirements: str) -> bool:
        lowered = str(requirements or "").lower()
        if "/health" not in lowered:
            return False
        return any(token in lowered for token in ("lightweight", "simple", "basic"))

    @staticmethod
    def _is_localized_reuse_request(requirements: str) -> bool:
        lowered = str(requirements or "").lower()
        if not any(token in lowered for token in ("reuse", "existing", "keep the current", "during rollout")):
            return False
        if not any(
            token in lowered
            for token in ("actionbar", "planpanel", "planningview", "frontend", "react", "component", "ui")
        ):
            return False
        return any(
            token in lowered
            for token in (
                "instead of creating new endpoints",
                "do not add new endpoints",
                "without adding backend endpoints",
            )
        )

    @staticmethod
    def _is_scope_expansion_feedback(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in SCOPE_EXPANSION_PATTERNS)

    @staticmethod
    def _is_localized_reuse_scope_expansion_feedback(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in LOCALIZED_REUSE_SCOPE_PATTERNS)

    def _apply_scope_discipline(self, requirements: str, review: ReviewResult) -> ReviewResult:
        # Hard rule: never strip issues when category blockers are present (avoids critic→strip→converge ghosts).
        if review.blocking_categories or review.blocking_categories_invalid:
            return review
        apply_health_filter = self._is_lightweight_health_request(requirements)
        apply_localized_filter = self._is_localized_reuse_request(requirements)
        if not apply_health_filter and not apply_localized_filter:
            return review

        def _filter(items: list[str]) -> list[str]:
            filtered: list[str] = []
            for item in items:
                if apply_health_filter and self._is_scope_expansion_feedback(item):
                    continue
                if apply_localized_filter and self._is_localized_reuse_scope_expansion_feedback(item):
                    continue
                filtered.append(item)
            return filtered

        blocking_issues = _filter(review.blocking_issues)
        recommended_changes = _filter(review.recommended_changes)
        architectural_concerns = _filter(review.architectural_concerns)
        risks = _filter(review.risks)
        reviewer_questions = _filter(review.reviewer_questions)
        issue_priority = _filter(review.issue_priority)
        primary_issue = review.primary_issue
        if primary_issue and (
            (apply_health_filter and self._is_scope_expansion_feedback(primary_issue))
            or (apply_localized_filter and self._is_localized_reuse_scope_expansion_feedback(primary_issue))
        ):
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
            plan_rubric=dict(review.plan_rubric),
            rubric_incomplete=review.rubric_incomplete,
            blocking_categories=list(review.blocking_categories),
            blocking_categories_invalid=review.blocking_categories_invalid,
            acceptance_criteria=list(review.acceptance_criteria),
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
                "You must still apply the global gatekeeper standards: vagueness, missing testability, and "
                "ungrounded claims remain BLOCKING even when confirming fixes.\n"
                "You may close issues that are truly fixed, but do not soften rubric scores or "
                "blocking_categories to 'move on' if defects remain.\n"
                "Update plan_rubric, blocking_categories, and acceptance_criteria to match the current plan text."
            )
        if mode == "stabilization":
            return (
                "Stabilization mode (incremental review):\n"
                "The plan text is unchanged since the last critic pass and prior review scores were stagnant.\n"
                "Compare ## Current Plan to ## Prior Review and ## Previous design review (if present). Credit fixes "
                "already reflected in the plan; focus blocking_issues on what still blocks shipping or verification.\n"
                "Global gatekeeper standards still apply: do not trade honesty for closure.\n"
                "Do not repeat verbatim blocking themes that the plan text has already addressed unless you explain "
                "why that text is still insufficient."
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
        fallback_model_override: str | None = None,
        session_id: str = "",
        round_number: int = 0,
        mode: Literal["initial", "validation", "stabilization", "implementability"] = "initial",
        open_tracked_issues_block: str | None = None,
        exploration_paths_block: str | None = None,
    ) -> ReviewResult | ImplementabilityResult:
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise RuntimeError("litellm is required for critic reviews but is not installed")

        prior_critique_blob = prior_critique.strip() if prior_critique else "None"
        open_issues_section = ""
        if open_tracked_issues_block and str(open_tracked_issues_block).strip():
            open_issues_section = f"{str(open_tracked_issues_block).strip()}\n\n"
        constraints_block = (
            "\n".join(f"- {item.id} ({item.severity}): {item.text}" for item in constraints)
            if constraints
            else "- (none)"
        )
        manifesto_excerpt = (manifesto or "").strip()[:4000]
        exploration_section = ""
        if exploration_paths_block and str(exploration_paths_block).strip():
            exploration_section = (
                f"## Repository paths explored this session (from tools)\n{str(exploration_paths_block).strip()}\n\n"
            )
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
            f"{open_issues_section}"
            f"{exploration_section}"
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
                _LOG.warning(
                    "Multi-perspective review fell back to single-pass due to timeout/failure guardrails: %s",
                    exc,
                )

        temp = 0.0 if temperature is None else temperature
        models_to_try = [model_override]
        if (
            fallback_model_override
            and fallback_model_override != model_override
            and model_override
            and model_has_elevated_json_contract_risk(model_override)
        ):
            models_to_try.append(fallback_model_override)

        last_parse_error: CriticParseError | None = None
        for model_index, active_model in enumerate(models_to_try):
            active_messages = list(messages)
            for attempt in range(max_retries + 1):
                try:
                    raw, _ = await self._call_with_telemetry(
                        messages=active_messages,
                        temperature=temp,
                        model_override=active_model,
                    )
                except Exception as exc:
                    raise CriticContractError(f"Reviewer call failed: {exc}") from exc
                try:
                    if mode == "implementability":
                        return self._parse_implementability_response(raw)
                    parsed = self._parse_review_response(raw)
                    if parsed.rubric_incomplete:
                        _LOG.warning(
                            "Reviewer omitted or partially omitted plan_rubric axes; "
                            "defaulted missing scores to %s (rubric_incomplete).",
                            MISSING_RUBRIC_SCORE,
                        )
                    if parsed.blocking_categories_invalid:
                        _LOG.warning(
                            "blocking_categories contained unknown tokens or was omitted; "
                            "convergence will fail closed until fixed."
                        )
                    from .review.critic_plan_coherence import coherence_adjust_review

                    parsed = coherence_adjust_review(plan_content, parsed)
                    return self._apply_scope_discipline(requirements, parsed)
                except CriticParseError as exc:
                    last_parse_error = exc
                    await self._emit(
                        {
                            "type": "structured_output_retry",
                            "model_stage": "critic_review",
                            "model": active_model,
                            "failure_category": "json_contract",
                        }
                    )
                    if strict_mode and attempt >= max_retries:
                        raise CriticContractError(f"Reviewer contract parse failure: {exc}") from exc
                    if attempt < max_retries:
                        retry_instruction = "Respond again with valid JSON first, then prose analysis."
                        if active_model and model_prefers_compact_json(active_model):
                            retry_instruction = (
                                "Respond with exactly one compact JSON object first. "
                                "No markdown fences, no prose before the JSON, and no trailing text after the closing brace."
                            )
                        active_messages.append(
                            {
                                "role": "user",
                                "content": f"Formatting error: {exc}. {retry_instruction}",
                            }
                        )
                        continue
                    if model_index < len(models_to_try) - 1:
                        await self._emit(
                            {
                                "type": "structured_output_fallback",
                                "model_stage": "critic_review",
                                "model": active_model,
                                "fallback_model": models_to_try[model_index + 1],
                                "failure_category": "json_contract",
                            }
                        )
                        _LOG.warning(
                            "Reviewer JSON contract failed on %s; retrying stage with fallback model %s.",
                            active_model,
                            models_to_try[model_index + 1],
                        )
                        break
                    raise CriticContractError(f"Reviewer parse failed after {max_retries + 1} attempts: {exc}") from exc

        if last_parse_error is not None:
            raise CriticContractError(
                "Reviewer exhausted retries without producing a valid contract"
            ) from last_parse_error
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
        fallback_model_override: str | None = None,
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
            fallback_model_override=fallback_model_override,
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
