"""
Critic runtime with strict JSON contract validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ...config import PlanningConfig, RepoProfile
from ...memory import ParsedConstraint
from ...pricing import MODEL_CONTEXT_WINDOWS
from .telemetry import completion_telemetry

logger = logging.getLogger(__name__)


class CriticParseError(RuntimeError):
    """Raised on malformed critic contract responses."""


class CriticContractError(RuntimeError):
    """Raised when critic contract validation fails."""


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
class CriticResult:
    major_issues_remaining: int
    minor_issues_remaining: int
    hard_constraint_violations: list[str]
    critique_complete: bool
    failure_modes: list[dict[str, str]]
    design_tradeoff_risks: list[dict[str, str]]
    unsupported_claims: list[str]
    missing_evidence: list[str]
    critic_confidence: float
    operational_readiness: bool
    vagueness_score: float
    citation_count: int
    clarification_questions: list[str]
    prose: str
    parse_error: str | None = None
    # Internal only — never persisted to DB or emitted over SSE.
    # Tracks violations suppressed by the evidence gate for analytics.
    suppressed_violations: list[str] = field(default_factory=list)


CRITIC_SYSTEM_PROMPT = """You are an adversarial senior reviewer for implementation plans.

You are not checking boxes. You are trying to break the plan before production does.

Return a JSON block FIRST, then prose critique.
Required JSON fields (all required):
- major_issues_remaining (int)
- minor_issues_remaining (int)
- hard_constraint_violations (list[str])
- critique_complete (bool)
- failure_modes (list[object])
- design_tradeoff_risks (list[object])
- unsupported_claims (list[str])
- missing_evidence (list[str])
- critic_confidence (float in [0, 1])
- operational_readiness (bool)
- clarification_questions (list[str], optional)

Rules:
- Evaluate hard constraints FIRST. Any hard constraint violation must be listed and must count as a major issue.
- major_issues_remaining must be > 0 if hard constraints are violated.
- critique_complete must be true only when major_issues_remaining is 0.
- critique_complete must be false if unsupported_claims or missing_evidence is non-empty.
- Use only valid hard-constraint IDs.
- hard_constraint_violations must be an array of string IDs only, e.g.
  ["HARD_CONSTRAINT_001"]. Do not emit objects in this field.
- operational_readiness is true only when observability, rollback, and test strategy are concrete.
- Every failure_modes/design_tradeoff_risks entry must include a "source" field pointing to a plan heading or file path.
- Do NOT wrap JSON in markdown fences. Return a raw JSON object first.
- Before listing a hard_constraint_violations entry, quote the specific plan line that proves the
  violation using backticks. The quoted text must come directly from the plan, not from your own
  analysis or reasoning. If you cannot quote a specific plan line, do NOT report that constraint
  as violated.
- If the plan text explicitly states an action is safe or excluded by the constraint
  (e.g. a public /health endpoint is fine per HARD_CONSTRAINT_001, or adding a new route is
  not restricted per HARD_CONSTRAINT_002), do NOT report a violation.

Each entry in "failure_modes" MUST be a JSON object with exactly these fields:
{
  "condition": "<specific triggering condition>",
  "blast_radius": "<who or what is affected and how severely>",
  "mitigation_missing": "<what is absent from the plan that would prevent this>",
  "instrumentation": "<what metric, log, or alert would detect this in production>",
  "source": "<plan section heading or file path where this risk originates>"
}

Each entry in "design_tradeoff_risks" MUST be a JSON object with:
{
  "risk": "<tradeoff risk>",
  "impact": "<expected impact if unaddressed>",
  "source": "<plan section heading or file path>"
}

Internal review process (perform in order before scoring):
1) Constraint compliance and policy risks
2) Failure-mode simulation: if this ships tomorrow, what breaks first?
3) Hidden complexity and unstated assumptions
4) Tradeoff quality and long-term lock-in risk
5) Operational readiness and rollout safety
6) If prior critique exists, verify claimed fixes are real (not cosmetic)
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
        self.schema = CriticContractSchema()
        # Lazily populated by _filter_spurious_violations on first call.
        self._constraint_pattern_cache: dict[str, re.Pattern[str]] = {}

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        maybe = self.event_callback(event)
        if asyncio.iscoroutine(maybe):
            await maybe

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
    def _require_object_fields(payload: Any, required_fields: set[str], *, field_name: str) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise CriticParseError(f"{field_name} entries must be objects")
        missing = [key for key in sorted(required_fields) if key not in payload]
        if missing:
            raise CriticParseError(f"{field_name} entry missing required fields: {missing}")
        return {k: str(payload[k]) for k in payload.keys()}

    @staticmethod
    def _vagueness_score(prose: str) -> float:
        sentence_count = len([s for s in re.split(r"[.!?]\s+", prose) if s.strip()])
        if sentence_count == 0:
            return 0.0
        vague_phrases = [
            "consider",
            "could",
            "might",
            "perhaps",
            "may want to",
            "potentially",
            "it is possible",
        ]
        lower = prose.lower()
        count = sum(lower.count(phrase) for phrase in vague_phrases)
        return float(count) / float(sentence_count)

    @staticmethod
    def _citation_count(prose: str) -> int:
        file_paths = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", prose))
        headings = set(re.findall(r"##\s+[A-Za-z0-9 _-]+", prose))
        return len(file_paths | headings)

    def _filter_spurious_violations(
        self,
        violations: list[str],
        constraints: list[ParsedConstraint],
        plan_text: str,
        prose: str,
        session_id: str = "",
        round_number: int = 0,
    ) -> tuple[list[str], list[str]]:
        """Return (kept, suppressed) after evidence-gating each violation.

        A violation is kept when:
        - The constraint has no evidence_keywords (safe default — never filter custom constraints).
        - At least one evidence keyword appears in the plan text (keyword match).
        - The LLM's prose contains a backtick-quoted excerpt with an evidence keyword
          AND plan_text is non-empty (quoted-evidence override).

        Deduplicates the input list before filtering to handle LLMs that repeat IDs.
        """
        # Deduplicate while preserving order — LLMs occasionally return the same ID twice.
        violations = list(dict.fromkeys(violations))

        keyword_map: dict[str, list[str]] = {c.id: c.evidence_keywords for c in constraints if c.evidence_keywords}
        # Build pattern cache lazily on first call; reused across all rounds.
        # \b word boundaries prevent partial-token matches (e.g. api_key_manager vs api_key).
        if not self._constraint_pattern_cache:
            self._constraint_pattern_cache = {
                vid: re.compile(
                    r"`[^`\n]*\b(" + "|".join(map(re.escape, kws)) + r")\b[^`\n]*`",
                    re.IGNORECASE,
                )
                for vid, kws in keyword_map.items()
            }

        kept: list[str] = []
        suppressed: list[str] = []
        text_lower = plan_text.lower()

        for vid in violations:
            keywords = keyword_map.get(vid)
            if not keywords:
                # No keywords defined for this constraint — always keep.
                kept.append(vid)
                continue
            keyword_found = any(kw.lower() in text_lower for kw in keywords)
            # Quoted-evidence override: the LLM is required by prompt to quote the
            # offending plan line in backticks. Accept the violation when that quoted
            # excerpt contains an evidence keyword — even if the plan text doesn't match
            # directly. Guard: only apply when plan_text is non-empty to avoid accepting
            # hallucinated quotes against a missing or malformed plan.
            quoted_found = bool(
                plan_text
                and self._constraint_pattern_cache.get(vid)
                and self._constraint_pattern_cache[vid].search(prose)
            )
            if keyword_found or quoted_found:
                kept.append(vid)
            else:
                suppressed.append(vid)
                logger.info(
                    "[constraint_filter] suppressed_violation",
                    extra={
                        "session_id": session_id,
                        "round": round_number,
                        "constraint_id": vid,
                        "keyword_found": keyword_found,
                        "quoted_found": quoted_found,
                    },
                )
        return kept, suppressed

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

    @staticmethod
    def _normalize_constraint_violations(raw_violations: list[Any]) -> list[str]:
        violations: list[str] = []
        for entry in raw_violations:
            if isinstance(entry, str):
                constraint_id = entry.strip()
            elif isinstance(entry, dict):
                candidate = entry.get("constraint_id") or entry.get("id")
                if not isinstance(candidate, str):
                    raise CriticParseError(
                        "hard_constraint_violations object entries must include a string constraint_id"
                    )
                constraint_id = candidate.strip()
            else:
                raise CriticParseError(
                    "hard_constraint_violations entries must be strings or objects with constraint_id"
                )
            if not constraint_id:
                raise CriticParseError("hard_constraint_violations entries cannot be empty")
            violations.append(constraint_id)
        return violations

    def _parse_critic_response(
        self,
        raw: str,
        valid_constraint_ids: set[str],
        constraints: list[ParsedConstraint] | None = None,
        plan_content: str = "",
        session_id: str = "",
        round_number: int = 0,
    ) -> CriticResult:
        json_text, prose = self._extract_first_json_object(raw)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise CriticParseError(f"Malformed critic JSON: {exc}") from exc

        for field_name, expected_type in self.schema.required_fields.items():
            if field_name not in data:
                raise CriticParseError(f"Missing required field: {field_name}")
            if not isinstance(data[field_name], expected_type):
                raise CriticParseError(f"Wrong type for {field_name}: expected {expected_type.__name__}")

        violations = self._normalize_constraint_violations(data["hard_constraint_violations"])
        unknown = set(violations) - valid_constraint_ids
        if unknown:
            # If no hard constraints are configured, tolerate invented IDs and
            # normalize to an empty list instead of degrading the critique flow.
            if len(valid_constraint_ids) == 0:
                violations = []
            else:
                raise CriticParseError(f"Unknown constraint IDs in violations: {sorted(unknown)}")

        # Evidence gate: filter out violations that have no evidence in the plan text.
        suppressed_violations: list[str] = []
        if constraints:
            violations, suppressed_violations = self._filter_spurious_violations(
                violations,
                constraints,
                plan_content,
                prose,
                session_id=session_id,
                round_number=round_number,
            )
            if suppressed_violations and not violations:
                # Every violation was suppressed — zero out the LLM's major issue count.
                # Never decrement arithmetically; the count may include non-constraint issues.
                data["major_issues_remaining"] = 0

        optional_values: dict[str, Any] = {}
        for field_name, (expected_type, default_value) in self.schema.optional_fields.items():
            if field_name not in data:
                optional_values[field_name] = default_value
                continue
            value = data[field_name]
            if field_name == "critic_confidence":
                if not isinstance(value, (int, float)):
                    raise CriticParseError("Wrong type for critic_confidence: expected float")
                if value < 0.0 or value > 1.0:
                    raise CriticParseError("critic_confidence must be in [0,1]")
                optional_values[field_name] = float(value)
                continue
            if not isinstance(value, expected_type):
                raise CriticParseError(f"Wrong type for {field_name}: expected {expected_type.__name__}")
            optional_values[field_name] = value

        failure_modes: list[dict[str, str]] = []
        for entry in optional_values["failure_modes"]:
            parsed = self._require_object_fields(
                entry,
                {"condition", "blast_radius", "mitigation_missing", "instrumentation", "source"},
                field_name="failure_modes",
            )
            failure_modes.append(parsed)

        design_tradeoff_risks: list[dict[str, str]] = []
        for entry in optional_values["design_tradeoff_risks"]:
            parsed = self._require_object_fields(
                entry,
                {"risk", "impact", "source"},
                field_name="design_tradeoff_risks",
            )
            design_tradeoff_risks.append(parsed)

        clarification_questions = [str(item) for item in optional_values["clarification_questions"]]
        unsupported_claims = [str(item) for item in optional_values["unsupported_claims"]]
        missing_evidence = [str(item) for item in optional_values["missing_evidence"]]
        critique_complete = bool(data["critique_complete"])
        if unsupported_claims or missing_evidence:
            critique_complete = False
        return CriticResult(
            major_issues_remaining=int(data["major_issues_remaining"]),
            minor_issues_remaining=int(data["minor_issues_remaining"]),
            hard_constraint_violations=violations,
            critique_complete=critique_complete,
            failure_modes=failure_modes,
            design_tradeoff_risks=design_tradeoff_risks,
            unsupported_claims=unsupported_claims,
            missing_evidence=missing_evidence,
            critic_confidence=float(optional_values["critic_confidence"]),
            operational_readiness=bool(optional_values["operational_readiness"]),
            vagueness_score=self._vagueness_score(prose),
            citation_count=self._citation_count(prose),
            clarification_questions=clarification_questions,
            prose=prose,
            suppressed_violations=suppressed_violations,
        )

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
    ) -> CriticResult:
        try:
            import litellm  # noqa: F401
        except ImportError:
            return CriticResult(
                major_issues_remaining=0,
                minor_issues_remaining=1,
                hard_constraint_violations=[],
                critique_complete=True,
                failure_modes=[],
                design_tradeoff_risks=[],
                unsupported_claims=[],
                missing_evidence=[],
                critic_confidence=0.0,
                operational_readiness=False,
                vagueness_score=0.0,
                citation_count=0,
                clarification_questions=[],
                prose="LiteLLM unavailable; critic fallback used.",
            )

        valid_ids = {c.id for c in constraints if c.severity == "hard" and not c.optional}
        valid_ids_text = ", ".join(sorted(valid_ids)) if valid_ids else "(none configured)"
        constraints_blob = "\n".join(
            f"- {c.id}: {c.text} (severity={c.severity}, optional={c.optional})" for c in constraints
        )
        prior_critique_blob = prior_critique.strip() if prior_critique else "None"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Manifesto\n{manifesto}\n\n"
                    f"## Constraints\n{constraints_blob}\n\n"
                    f"## Architecture\n{architecture}\n\n"
                    f"## Requirements\n{requirements}\n\n"
                    f"## Prior Critique (check whether fixes are real)\n{prior_critique_blob}\n\n"
                    f"## Current Plan\n{plan_content}\n\n"
                    "## Constraint-ID Rules\n"
                    f"Valid hard constraint IDs: {valid_ids_text}\n"
                    "- Use ONLY IDs from the list above.\n"
                    "- If no valid hard IDs are configured, return hard_constraint_violations as []."
                ),
            },
        ]

        temp = 0.0 if temperature is None else temperature
        for attempt in range(max_retries + 1):
            try:
                raw, response, model = await asyncio.to_thread(
                    self._llm_call,
                    messages,
                    temp,
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
                        "session_stage": "critic",
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
            except Exception as exc:  # noqa: BLE001
                if strict_mode:
                    raise CriticContractError(f"Critic failed in strict mode: {exc}") from exc
                return CriticResult(
                    major_issues_remaining=0,
                    minor_issues_remaining=1,
                    hard_constraint_violations=[],
                    critique_complete=True,
                    failure_modes=[],
                    design_tradeoff_risks=[],
                    unsupported_claims=[],
                    missing_evidence=[],
                    critic_confidence=0.0,
                    operational_readiness=False,
                    vagueness_score=0.0,
                    citation_count=0,
                    clarification_questions=[],
                    prose=f"Critic fallback used due to model/runtime error: {exc}",
                    parse_error=str(exc),
                )
            try:
                return self._parse_critic_response(
                    raw,
                    valid_ids,
                    constraints=constraints,
                    plan_content=plan_content,
                    session_id=session_id,
                    round_number=round_number,
                )
            except CriticParseError as exc:
                if strict_mode and attempt >= max_retries:
                    raise CriticContractError(f"Critic contract parse failure: {exc}") from exc
                if attempt < max_retries:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Formatting error: {exc}. Respond again with JSON first, then prose critique. "
                                "Important: hard_constraint_violations must be an array of constraint ID strings only."
                            ),
                        }
                    )
                    continue
                await self._emit(
                    {
                        "type": "warning",
                        "message": f"Critic parse degraded after retries: {exc}",
                    }
                )
                return CriticResult(
                    major_issues_remaining=1,
                    minor_issues_remaining=0,
                    hard_constraint_violations=[],
                    critique_complete=False,
                    failure_modes=[],
                    design_tradeoff_risks=[],
                    unsupported_claims=[],
                    missing_evidence=[],
                    critic_confidence=0.0,
                    operational_readiness=False,
                    vagueness_score=0.0,
                    citation_count=0,
                    clarification_questions=[],
                    prose=raw,
                    parse_error=str(exc),
                )

        return CriticResult(
            major_issues_remaining=1,
            minor_issues_remaining=0,
            hard_constraint_violations=[],
            critique_complete=False,
            failure_modes=[],
            design_tradeoff_risks=[],
            unsupported_claims=[],
            missing_evidence=[],
            critic_confidence=0.0,
            operational_readiness=False,
            vagueness_score=0.0,
            citation_count=0,
            clarification_questions=[],
            prose="Critic failed to return a valid contract.",
            parse_error="unreachable retry guard",
        )

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
    ) -> CriticResult:
        result = await self.run_critic(
            requirements=requirements,
            plan_content=plan_content,
            manifesto=manifesto,
            architecture=architecture,
            constraints=constraints,
            prior_critique=None,
            max_retries=2,
            temperature=self.config.validate_temperature,
            strict_mode=True,
        )

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
                        "critic_result": asdict(result),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result
