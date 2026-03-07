from __future__ import annotations

import json
import re
from collections.abc import Awaitable
from typing import Any, Callable

from .models import PlanDocument, RepairPlan, RevisionResult

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
    def __init__(self, llm_call: LlmCaller) -> None:
        self._llm_call = llm_call

    async def plan_repair(
        self,
        review: Any,
        plan: PlanDocument,
        requirements: str,
        design_record: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> RepairPlan:
        prompt = (
            "You are revising a design after a technical review.\n\n"
            "Before editing the plan:\n"
            "1. Restate the most important reviewer concern.\n"
            "2. Decide which reviewer issues you accept or reject.\n"
            "3. Identify the root causes of the accepted issues.\n"
            "4. Describe the repair strategy.\n"
            "5. Identify which sections of the plan must change.\n\n"
            "Focus on fixing root causes rather than applying superficial edits.\n"
            "Do not blindly accept all feedback.\n\n"
            "Reject reviewer issues that materially expand scope beyond the user requirements, the current plan's non-goals, "
            "or verified repository evidence.\n"
            "For lightweight endpoint work, prefer minimal changes over broader platform features unless they are explicitly required.\n\n"
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
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Current Plan JSON\n{json.dumps(plan.__dict__, indent=2)}\n\n"
                    f"## Review Result\n{json.dumps(review.__dict__, indent=2)}\n\n"
                    f"## Design Record\n{json.dumps(design_record or {}, indent=2)}"
                ),
            },
        ]
        response, _ = await self._llm_call(
            messages, allow_tools=False, max_output_tokens=1800, model_override=model_override
        )
        raw = str(getattr(response.choices[0].message, "content", None) or "")
        json_text, _ = extract_first_json_object(raw)
        payload = load_json_object(json_text)
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
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
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
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Current Design Record\n{json.dumps(design_record, indent=2)}\n\n"
                    f"## Review Result\n{json.dumps(review.__dict__, indent=2)}"
                ),
            },
        ]
        response, _ = await self._llm_call(
            messages,
            allow_tools=False,
            max_output_tokens=1600,
            model_override=model_override,
        )
        raw = str(getattr(response.choices[0].message, "content", None) or "")
        json_text, _ = extract_first_json_object(raw)
        payload = load_json_object(json_text)
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
        simplest_possible_design: str | None = None,
    ) -> RevisionResult:
        simplification_hint = (
            "\nIf a simplification proposal is provided and sound, prefer it over incremental fixes."
            if simplest_possible_design
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are revising a structured design document.\n"
                    "Only modify sections necessary to address reviewer concerns.\n"
                    f"You may update at most {max(1, revision_budget)} sections.\n"
                    "Do not rewrite the entire plan.\n"
                    "Do not add new subsystems, dependency checks, concurrency machinery, or security controls unless they are "
                    "required by the accepted issues and still fit the original scope.\n"
                    "Keep lightweight requests lightweight.\n"
                    "For a lightweight `/health` endpoint request, that means preserving a simple status response and focused tests "
                    "unless the requirements explicitly ask for broader checks.\n"
                    "Before generating updates:\n"
                    "Step 1: Restate the primary concern.\n"
                    "Step 2: Explain how revisions resolve it.\n"
                    "Step 3: Predict reviewer reaction.\n"
                    f"{simplification_hint}\n\n"
                    "Return JSON with:\n"
                    "- problem_understanding: str\n"
                    "- updates: object {section_id: new_content}\n"
                    "- justification: object {section_id: why_changed}\n"
                    "- review_prediction: str"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Repair Plan\n{json.dumps(repair_plan.__dict__, indent=2)}\n\n"
                    f"## Current Plan JSON\n{json.dumps(current_plan.__dict__, indent=2)}\n\n"
                    f"## Design Record\n{json.dumps(design_record or {}, indent=2)}\n\n"
                    f"## Simplest Possible Design\n{simplest_possible_design or '(none)'}"
                ),
            },
        ]
        response, _ = await self._llm_call(
            messages, allow_tools=False, max_output_tokens=2200, model_override=model_override
        )
        raw = str(getattr(response.choices[0].message, "content", None) or "")
        json_text, _ = extract_first_json_object(raw)
        payload = load_json_object(json_text)
        updates_raw = payload.get("updates", {})
        justification_raw = payload.get("justification", {})
        updates = {str(k): str(v) for k, v in updates_raw.items() if isinstance(k, str)}
        limited_updates = dict(list(updates.items())[: max(1, revision_budget)])
        justification = {
            str(k): str(v) for k, v in justification_raw.items() if isinstance(k, str) and str(k) in limited_updates
        }
        return RevisionResult(
            problem_understanding=str(payload.get("problem_understanding", "")),
            updates=limited_updates,
            justification=justification,
            review_prediction=str(payload.get("review_prediction", "")),
        )
