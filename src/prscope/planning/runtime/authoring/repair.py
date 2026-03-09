from __future__ import annotations

import json
import re
from collections.abc import Awaitable
from typing import Any, Callable

from ....model_catalog import (
    model_has_elevated_json_contract_risk,
    model_prefers_compact_json,
    model_provider,
)
from .discovery import is_localized_frontend_request
from .models import PlanDocument, RepairPlan, RevisionResult
from .validation import localized_request_explicit_payload_change

LlmCaller = Callable[..., Awaitable[tuple[Any, str]]]


def extract_first_json_object(raw: str) -> tuple[str, str]:
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON block found in author response")
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
        raise ValueError("Unterminated JSON object in author response")
    return raw[start:end], raw[end:].strip()


def load_json_object(raw: str) -> dict[str, Any]:
    """Parse JSON object with lightweight repair for common LLM syntax slips."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Common malformed output from LLMs: trailing commas in objects/arrays.
        repaired = re.sub(r",(\s*[}\]])", r"\1", raw)
        payload = json.loads(repaired)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object payload")
    return payload


def parse_plan_document(raw: str) -> PlanDocument:
    json_text, _ = extract_first_json_object(raw)
    payload = load_json_object(json_text)
    required = [
        "title",
        "summary",
        "goals",
        "non_goals",
        "files_changed",
        "architecture",
        "implementation_steps",
        "test_strategy",
        "rollback_plan",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Missing required PlanDocument fields: {missing}")
    plan_payload = {field: str(payload[field]) for field in required}
    plan_payload["open_questions"] = str(payload.get("open_questions", ""))
    return PlanDocument(**plan_payload)


class AuthorRepairService:
    def __init__(
        self,
        llm_call: LlmCaller,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._event_callback = event_callback

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        maybe = self._event_callback(event)
        if maybe is not None:
            await maybe

    @staticmethod
    def _compact_json_retry_instruction(model_override: str | None) -> str:
        base = "Return compact JSON only. No markdown fences, no commentary."
        if model_override and (
            model_prefers_compact_json(model_override) or model_provider(model_override) == "anthropic"
        ):
            return base + " Return exactly one JSON object and nothing after the closing brace."
        return base

    @staticmethod
    def _json_contract_messages(
        *,
        system_prompt: str,
        user_message: dict[str, Any],
        model_override: str | None,
    ) -> list[list[dict[str, Any]]]:
        retry_prompt = (
            system_prompt
            + "\n\n"
            + AuthorRepairService._compact_json_retry_instruction(model_override)
            + " Keep each list concise and grounded."
        )
        return [
            [{"role": "system", "content": system_prompt}, user_message],
            [{"role": "system", "content": retry_prompt}, user_message],
        ]

    async def _call_json_object(
        self,
        *,
        system_prompt: str,
        user_message: dict[str, Any],
        max_output_tokens: int,
        model_override: str | None,
        fallback_model_override: str | None,
        model_stage: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        first_error: Exception | None = None
        models_to_try: list[str | None] = [model_override]
        if (
            fallback_model_override
            and fallback_model_override != model_override
            and model_override
            and model_has_elevated_json_contract_risk(model_override)
        ):
            models_to_try.append(fallback_model_override)

        for active_model in models_to_try:
            for attempt_messages in self._json_contract_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                model_override=active_model,
            ):
                response, _ = await self._llm_call(
                    attempt_messages,
                    allow_tools=False,
                    max_output_tokens=max_output_tokens,
                    model_override=active_model,
                )
                raw = str(getattr(response.choices[0].message, "content", None) or "")
                try:
                    json_text, _ = extract_first_json_object(raw)
                    payload = load_json_object(json_text)
                    break
                except ValueError as exc:
                    await self._emit(
                        {
                            "type": "structured_output_retry",
                            "model_stage": model_stage,
                            "model": active_model,
                            "failure_category": "json_contract",
                        }
                    )
                    if first_error is None:
                        first_error = exc
                    continue
            if payload is not None:
                break
            if active_model and payload is None and active_model != models_to_try[-1]:
                await self._emit(
                    {
                        "type": "structured_output_fallback",
                        "model_stage": model_stage,
                        "model": active_model,
                        "fallback_model": models_to_try[models_to_try.index(active_model) + 1],
                        "failure_category": "json_contract",
                    }
                )
        if payload is None:
            raise first_error or ValueError("No JSON block found in author response")
        return payload

    @staticmethod
    def _requirements_forbid_frontend_state_abstractions(requirements: str) -> bool:
        lowered = str(requirements or "").lower()
        return any(
            phrase in lowered
            for phrase in (
                "do not introduce new hooks",
                "do not introduce hooks",
                "avoid new hooks",
                "do not introduce new contexts",
                "avoid new contexts",
                "do not introduce shared state",
                "shared state layers",
                "do not introduce background polling",
                "background polling",
                "keep the change localized",
            )
        )

    @classmethod
    def _localized_ui_drift_patterns(cls, requirements: str) -> tuple[re.Pattern[str], ...]:
        lowered = str(requirements or "").lower()
        patterns = []
        if not any(token in lowered for token in ("observability", "telemetry", "logging", "monitoring")):
            patterns.extend(
                [
                    r"\bobservability\b",
                    r"\btelemetry\b",
                    r"\blogging\b",
                    r"\bmonitoring\b",
                    r"\bseparation of concerns\b",
                ]
            )
        if cls._requirements_forbid_frontend_state_abstractions(requirements):
            patterns.extend(
                [
                    r"\b(new|dedicated)\s+hooks?\b",
                    r"\bhooks?/contexts?\b",
                    r"\buseState\b",
                    r"\buseEffect\b",
                    r"\buseRef\b",
                    r"\buseCallback\b",
                    r"\buseMemo\b",
                    r"\bisExporting\b",
                    r"\blastExportResult\b",
                    r"\bcontext\s+provider\b",
                    r"\blocal state\b",
                    r"\bshared state\b",
                    r"\bcentralized state\b",
                    r"\bsingle state object\b",
                    r"\bui state object\b",
                    r"\bstate management\b",
                    r"\bseparation of state and logic\b",
                    r"\bseparation of ui and business logic\b",
                    r"\bmanage concurrency\b",
                    r"\bconcurrency proactively\b",
                    r"\bconcurrent(?:ly)?\s+exports?\b",
                    r"\bbackground polling\b",
                    r"\bpolling\b",
                    r"\bself-contained state\b",
                    r"\bservice layer\b",
                    r"\babstraction layer\b",
                    r"\breact hook patterns?\b",
                    r"\bstate initialization\b",
                    r"\bmount tracking\b",
                    r"\bunmount(?:ed)?\b",
                    r"\bunmount guard\b",
                    r"\bisMountedRef\b",
                    r"\bPromise\.race\b",
                    r"\bsetTimeout\b",
                    r"\btimeout recovery\b",
                    r"\bdouble-click prevention\b",
                    r"\btype\s+[A-Z][A-Za-z0-9_]*\s*=",
                ]
            )
        if not localized_request_explicit_payload_change(requirements):
            patterns.extend(
                [
                    r"\bget_session\b",
                    r"\bgetsession\b",
                    r"\bplanningsession\b",
                    r"\bis_exporting\b",
                    r"\blast_export_result\b",
                    r"\blast_export_at\b",
                    r"\bbackend api responses?\b",
                    r"\bbackend api contract\b",
                    r"\bapi contract changes?\b",
                    r"\bapi model regression test\b",
                    r"\btests/test_web_api_models\.py\b",
                    r"\bsrc/prscope/web/api\.py\b",
                    r"\bresponse shape\b",
                    r"\bpayload shape\b",
                ]
            )
        return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)

    @classmethod
    def _sanitize_localized_ui_update(
        cls,
        *,
        section_id: str,
        content: str,
        requirements: str,
        current_plan: PlanDocument,
    ) -> str:
        if section_id not in {
            "summary",
            "goals",
            "non_goals",
            "changes",
            "files_changed",
            "architecture",
            "implementation_steps",
            "test_strategy",
        }:
            return str(content or "").strip()
        if not is_localized_frontend_request(requirements):
            return str(content or "").strip()
        patterns = cls._localized_ui_drift_patterns(requirements)
        if not patterns:
            return str(content or "").strip()
        text = str(content or "").strip()
        if not text:
            return text
        if section_id == "architecture":
            text = re.sub(r"```[\s\S]*?```", "", text).strip()
        filtered_lines = [line for line in text.splitlines() if not any(pattern.search(line) for pattern in patterns)]
        filtered = "\n".join(line for line in filtered_lines if line.strip()).strip()
        if filtered:
            return filtered
        current_value = str(getattr(current_plan, section_id, "") or "").strip()
        if current_value and not any(pattern.search(current_value) for pattern in patterns):
            return current_value
        fallback_map = {
            "summary": "Keep the change localized to the existing UI flow and verified helper wiring.",
            "goals": (
                "- Keep the change localized to the existing UI flow.\n"
                "- Reuse the verified helper wiring already present in the codebase.\n"
                "- Cover the requested user-visible behavior with focused frontend regression tests."
            ),
            "non_goals": (
                "- Do not introduce new hooks, contexts, shared state layers, or polling behavior.\n"
                "- Do not broaden the work into backend contract or session-state changes unless explicitly required."
            ),
            "changes": "- Update the existing UI component and helper wiring directly without adding new frontend state abstractions.",
            "files_changed": "\n".join(
                line for line in str(getattr(current_plan, "files_changed", "") or "").splitlines() if line.strip()
            ).strip(),
            "architecture": (
                "Keep the change localized to the existing component and verified helper wiring. "
                "Do not add new hooks, contexts, shared state layers, or polling behavior."
            ),
            "implementation_steps": (
                "1. Update the existing component to reflect the requested UI state.\n"
                "2. Reuse the verified helper wiring already present in the codebase.\n"
                "3. Add focused regression coverage for the localized flow."
            ),
            "test_strategy": (
                "- Assert the requested user-visible behavior in the focused frontend regression test.\n"
                "- Assert the localized UI uses the existing helper wiring without introducing new state abstractions."
            ),
        }
        return fallback_map.get(section_id, text)

    @staticmethod
    def _pressure_guidance_block(reconsideration_candidates: list[dict[str, Any]] | None) -> str:
        items = reconsideration_candidates or []
        if not items:
            return "(none)"
        lines: list[str] = []
        for candidate in items[:2]:
            if not isinstance(candidate, dict):
                continue
            decision_id = str(candidate.get("decision_id", "")).strip() or "(unknown decision)"
            reason = str(candidate.get("reason", "")).strip() or "pressure_guidance"
            pressure = int(candidate.get("decision_pressure", 0) or 0)
            action = str(candidate.get("suggested_action", "")).strip() or "clarify pressured decision"
            cluster = candidate.get("dominant_cluster")
            cluster_payload = cluster if isinstance(cluster, dict) else {}
            root_issue = str(cluster_payload.get("root_issue", "")).strip() or "unspecified root issue"
            lines.append(
                f"- `{decision_id}` pressure={pressure} reason={reason}; root issue: {root_issue}; suggested action: {action}"
            )
        return "\n".join(lines) if lines else "(none)"

    async def plan_repair(
        self,
        review: Any,
        plan: PlanDocument,
        requirements: str,
        design_record: dict[str, Any] | None = None,
        reconsideration_candidates: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
        fallback_model_override: str | None = None,
    ) -> RepairPlan:
        prompt = (
            "You are revising a design after a technical review.\n\n"
            "Before editing the plan:\n"
            "1. Restate the most important reviewer concern.\n"
            "2. Decide which reviewer issues you accept or reject.\n"
            "3. Identify the root causes of the accepted issues.\n"
            "4. Describe the repair strategy.\n"
            "5. Identify which sections of the plan must change.\n"
            "6. When the primary issue is a lack of clarity or missing answer that corresponds to an item in open_questions, include 'open_questions' in target_sections so the revision can remove the addressed question.\n\n"
            "Focus on fixing root causes rather than applying superficial edits.\n"
            "Do not blindly accept all feedback.\n\n"
            "Reject reviewer issues that materially expand scope beyond the user requirements, the current plan's non-goals, "
            "or verified repository evidence.\n"
            "For lightweight endpoint work, prefer minimal changes over broader platform features unless they are explicitly required.\n\n"
            "For localized UI or API-wiring work that explicitly reuses existing helpers/endpoints, reject feedback that introduces "
            "service layers, dedicated hooks/contexts, centralized state/error handling, single state objects, concurrency-management rhetoric, observability/telemetry work, broad architecture refactors, or code-level React prescriptions like typed state objects, `useRef`, `useCallback`, `Promise.race`, timeout guards, or unmount guards unless the requirements or verified "
            "repository evidence explicitly require those structures.\n\n"
            "If the requirements say backend payload or contract work is needed only conditionally (for example, "
            '"only if the response shape must change"), treat backend response fields, session-state plumbing, API model tests, '
            "and backend file changes as out of scope unless the review proves a concrete payload change is required.\n\n"
            "If cross-graph reconsideration candidates are provided, treat them as architectural pressure signals. "
            "Explicitly decide whether the top pressured decision should be clarified, narrowed, reconsidered, or defended with a concrete rationale in the revised plan.\n"
            "Do not ignore the top pressure signal when one is provided.\n\n"
            "When the requirements name a source of truth, preserve that exact source-of-truth decision explicitly.\n"
            "For localized caching or invalidation work, name only the concrete invalidation triggers required by the request. "
            "Do not broaden the plan with generic comprehensive invalidation lists, manual admin actions, expiration policies, or new wrapper layers unless the requirements explicitly ask for them.\n\n"
            "For localized cache-wiring requests, reject speculative concurrency or abstraction drift. "
            "Do not add concurrency management, synchronization schemes, locks, background invalidation workers, shared cache modules, or separation-of-concerns rhetoric unless the requirements explicitly ask for them.\n\n"
            "For localized UI requests that simply say to show the latest result/status, assume a simple success/failure presentation by default. "
            "Do not create open questions or architecture churn about display formatting/details unless the requirements or verified evidence make that choice genuinely ambiguous.\n\n"
            "Example: for a request to add or verify a lightweight `/health` endpoint, reject suggestions to add database "
            "checks, external-service checks, authentication, or concurrency-control machinery unless the requirements explicitly ask for them.\n\n"
            "Return JSON with fields:\n"
            "- problem_understanding: str\n"
            "- accepted_issues: list[str]\n"
            "- rejected_issues: list[str]\n"
            "- root_causes: list[str]\n"
            "- repair_strategy: str\n"
            "- target_sections: list[str]\n"
            "- revision_plan: str"
        )
        user_message = {
            "role": "user",
            "content": (
                f"## Requirements\n{requirements}\n\n"
                f"## Current Plan JSON\n{json.dumps(plan.__dict__, indent=2)}\n\n"
                f"## Review Result\n{json.dumps(review.__dict__, indent=2)}\n\n"
                f"## Architectural Pressure Guidance\n{self._pressure_guidance_block(reconsideration_candidates)}\n\n"
                f"## Design Record\n{json.dumps(design_record or {}, indent=2)}\n\n"
                f"## Reconsideration Candidates\n{json.dumps(reconsideration_candidates or [], indent=2)}"
            ),
        }
        payload = await self._call_json_object(
            system_prompt=prompt,
            user_message=user_message,
            max_output_tokens=2200,
            model_override=model_override,
            fallback_model_override=fallback_model_override,
            model_stage="author_refine",
        )
        return RepairPlan(
            problem_understanding=str(payload.get("problem_understanding", "")),
            accepted_issues=[str(item) for item in payload.get("accepted_issues", [])],
            rejected_issues=[str(item) for item in payload.get("rejected_issues", [])],
            root_causes=[str(item) for item in payload.get("root_causes", [])],
            repair_strategy=str(payload.get("repair_strategy", "")),
            target_sections=[str(item) for item in payload.get("target_sections", [])],
            revision_plan=str(payload.get("revision_plan", "")),
        )

    async def update_design_record(
        self,
        *,
        design_record: dict[str, Any],
        review: Any,
        requirements: str,
        model_override: str | None = None,
        fallback_model_override: str | None = None,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are updating a technical design record after review feedback.\n"
            "Return JSON with fields:\n"
            "- problem_summary: str\n"
            "- constraints: list[str]\n"
            "- architecture: str\n"
            "- alternatives_considered: list[str]\n"
            "- tradeoffs: list[str]\n"
            "- chosen_design: str\n"
            "- assumptions: list[str]\n"
            "- potential_failure_modes: list[str]\n"
            "Prioritize updates that address the highest-priority reviewer issues."
        )
        user_message = {
            "role": "user",
            "content": (
                f"## Requirements\n{requirements}\n\n"
                f"## Current Design Record\n{json.dumps(design_record, indent=2)}\n\n"
                f"## Review Result\n{json.dumps(review.__dict__, indent=2)}"
            ),
        }
        payload = await self._call_json_object(
            system_prompt=system_prompt,
            user_message=user_message,
            max_output_tokens=1600,
            model_override=model_override,
            fallback_model_override=fallback_model_override,
            model_stage="author_refine",
        )
        return {
            "problem_summary": str(payload.get("problem_summary", "")),
            "constraints": [str(item) for item in payload.get("constraints", [])],
            "architecture": str(payload.get("architecture", "")),
            "alternatives_considered": [str(item) for item in payload.get("alternatives_considered", [])],
            "tradeoffs": [str(item) for item in payload.get("tradeoffs", [])],
            "chosen_design": str(payload.get("chosen_design", "")),
            "assumptions": [str(item) for item in payload.get("assumptions", [])],
            "potential_failure_modes": [str(item) for item in payload.get("potential_failure_modes", [])],
        }

    async def revise_plan(
        self,
        repair_plan: RepairPlan,
        current_plan: PlanDocument,
        requirements: str,
        design_record: dict[str, Any] | None = None,
        revision_budget: int = 3,
        model_override: str | None = None,
        fallback_model_override: str | None = None,
        simplest_possible_design: str | None = None,
        revision_hints: list[str] | None = None,
        reconsideration_candidates: list[dict[str, Any]] | None = None,
        supplemental_evidence: dict[str, Any] | None = None,
    ) -> RevisionResult:
        simplification_hint = (
            "\nIf a simplification proposal is provided and sound, prefer it over incremental fixes."
            if simplest_possible_design
            else ""
        )
        system_prompt = (
            "You are revising a structured design document.\n"
            "Only modify sections necessary to address reviewer concerns.\n"
            f"You may update at most {max(1, revision_budget)} sections.\n"
            "Do not rewrite the entire plan.\n"
            "Do not add new subsystems, dependency checks, concurrency machinery, or security controls unless they are "
            "required by the accepted issues and still fit the original scope.\n"
            "Keep lightweight requests lightweight.\n"
            "For a lightweight `/health` endpoint request, that means preserving a simple status response and focused tests "
            "unless the requirements explicitly ask for broader checks.\n"
            "For localized UI or API-wiring requests that reuse existing helpers/endpoints, do not add dedicated handler layers, "
            "dedicated hooks/contexts, centralized state/error handling, single state objects, concurrency-management rhetoric, observability/telemetry work, broad component architecture changes, or code-level React prescriptions like typed state objects, `useRef`, `useCallback`, `Promise.race`, timeout guards, or unmount guards unless the accepted issues explicitly "
            "require them and the repository evidence supports that added structure.\n"
            "If the requirements say backend payload or contract work is needed only conditionally (for example, "
            '"only if the response shape must change"), do not add backend response fields, session-state plumbing, '
            "backend files, or API model regression tests unless the accepted issues prove a concrete payload change is required.\n"
            "Preserve exact grounded file paths, helper names, and endpoint names already present in the current plan "
            "unless the requirements or accepted issues explicitly replace them with another verified spelling.\n"
            "Preserve already-grounded owner files and focused regression targets from the current plan unless the accepted "
            "issues explicitly prove those references are out of scope.\n"
            "When reviewer feedback or pressure signals raise ownership ambiguity, choose exactly one owner for the affected UI state "
            "and keep Files Changed, Architecture, and Implementation Steps consistent about that choice. "
            "Do not say both that state is passed from the parent and that the child keeps the same state internally.\n"
            "If the request explicitly says to preserve current `PlanPanel` behavior, do not move export-state ownership into "
            "`PlanPanel.tsx`; keep `PlanningView.tsx` as the existing owner and limit `PlanPanel.tsx` updates to localized rendering "
            "or button-state behavior.\n"
            "Do not shorten, normalize, or rename existing paths or symbols "
            "(for example, keep `src/prscope/web/frontend/src/lib/api.ts` and `exportSession` exactly as written).\n"
            "Prefer surgical edits over collapsing precise implementation or test detail into broader summaries.\n"
            "Do not replace concrete wiring or assertion detail with generic wording if that would make the plan less actionable.\n"
            "If reconsideration candidates are provided, update the relevant plan sections so the highest-pressure "
            "decision is explicitly clarified, constrained, reconsidered, or defended with concrete rationale rather than left as implicit pressure.\n"
            "When architectural pressure guidance is present, make a visible plan change that addresses that pressure unless you can justify preserving the current decision.\n"
            "If bounded refinement evidence is provided, prefer those verified anchors and adjacent tests over inventing new files, owners, or abstractions.\n"
            "When the requirements name a source of truth, keep that source-of-truth statement explicit in the revised plan.\n"
            "For localized caching or invalidation work, name only the requested invalidation triggers. "
            "Do not introduce generic comprehensive invalidation lists, manual admin actions, expiration policies, or wrapper-layer abstractions unless the requirements explicitly ask for them.\n"
            "For localized cache-wiring requests, reject speculative concurrency or abstraction drift. "
            "Do not add concurrency management, synchronization schemes, locks, background invalidation workers, shared cache modules, or generic separation-of-concerns/module rhetoric unless the requirements explicitly ask for them.\n"
            "When your revisions address a question that appears in the current plan's open_questions section (e.g. by adding concrete content that answers it), you MUST include open_questions in your updates and remove that question from the list. If all questions are now addressed, set open_questions to '- None.'.\n"
            "Before generating updates:\n"
            "Step 1: Restate the primary concern.\n"
            "Step 2: Explain how revisions resolve it.\n"
            "Step 3: Predict reviewer reaction.\n"
            f"{simplification_hint}\n\n"
            "Return JSON with:\n"
            "- problem_understanding: str\n"
            "- updates: object {section_id: new_content}\n"
            "- justification: object {section_id: why_changed}\n"
            "- what_changed: object {section_id: one_sentence_concrete_description}. For each updated section, describe in one sentence WHAT was added or changed (e.g. 'Added explicit dependency checks for DB, Redis, and external API health'). Be concrete and specific. Do not write meta-commentary like 'the reviewer will appreciate...'.\n"
            "- review_prediction: str"
        )
        user_message = {
            "role": "user",
            "content": (
                f"## Requirements\n{requirements}\n\n"
                f"## Repair Plan\n{json.dumps(repair_plan.__dict__, indent=2)}\n\n"
                f"## Current Plan JSON\n{json.dumps(current_plan.__dict__, indent=2)}\n\n"
                f"## Architectural Pressure Guidance\n{self._pressure_guidance_block(reconsideration_candidates)}\n\n"
                f"## Design Record\n{json.dumps(design_record or {}, indent=2)}\n\n"
                f"## Simplest Possible Design\n{simplest_possible_design or '(none)'}\n\n"
                f"## Revision Hints\n{json.dumps(revision_hints or [], indent=2)}\n\n"
                f"## Reconsideration Candidates\n{json.dumps(reconsideration_candidates or [], indent=2)}\n\n"
                f"## Bounded Refinement Evidence\n{json.dumps(supplemental_evidence or {}, indent=2)}"
            ),
        }
        payload = await self._call_json_object(
            system_prompt=system_prompt,
            user_message=user_message,
            max_output_tokens=2800,
            model_override=model_override,
            fallback_model_override=fallback_model_override,
            model_stage="author_refine",
        )
        updates_raw = payload.get("updates", {})
        justification_raw = payload.get("justification", {})
        updates = {str(k): str(v) for k, v in updates_raw.items() if isinstance(k, str)}
        updates = {
            key: self._sanitize_localized_ui_update(
                section_id=key,
                content=value,
                requirements=requirements,
                current_plan=current_plan,
            )
            for key, value in updates.items()
        }
        limited_updates = dict(list(updates.items())[: max(1, revision_budget)])
        justification = {
            str(k): str(v) for k, v in justification_raw.items() if isinstance(k, str) and str(k) in limited_updates
        }
        what_changed_raw = payload.get("what_changed", {})
        what_changed = {
            str(k): str(v)
            for k, v in what_changed_raw.items()
            if isinstance(k, str) and isinstance(v, str) and str(k) in limited_updates
        }
        return RevisionResult(
            problem_understanding=str(payload.get("problem_understanding", "")),
            updates=limited_updates,
            justification=justification,
            what_changed=what_changed,
            review_prediction=str(payload.get("review_prediction", "")),
        )
