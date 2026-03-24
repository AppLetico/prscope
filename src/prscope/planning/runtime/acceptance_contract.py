"""
Acceptance criteria validation and harness convergence gate helpers.

Keeps opinion out of acceptance satisfaction: structural checks + shallow plan evidence.
"""

from __future__ import annotations

import re
from typing import Any

from .critic import ReviewResult

# Subjective fluff that must not pass as a falsifiable acceptance criterion.
ACCEPTANCE_BANNED_SUBSTRINGS: tuple[str, ...] = (
    "clear",
    "robust",
    "good ",
    " good",
    "solid",
    "nice ",
    "better ",
    "appropriate",
    "reasonable",
    "seems fine",
    "looks good",
)

_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "must",
        "shall",
        "should",
        "could",
        "would",
        "each",
        "every",
        "plan",
        "section",
        "list",
        "include",
    }
)


def acceptance_criteria_structurally_valid(criteria: list[str]) -> bool:
    if not criteria:
        return True
    for c in criteria:
        s = str(c).strip()
        if len(s) < 12:
            return False
        lowered = s.lower()
        if any(b in lowered for b in ACCEPTANCE_BANNED_SUBSTRINGS):
            return False
    return True


def _significant_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9_-]{3,}", text.lower()) if t not in _STOPWORDS]


def _criterion_satisfied_in_plan(plan_markdown: str, criterion: str) -> bool:
    plan_lower = plan_markdown.lower()
    c_lower = str(criterion).lower().strip()
    chunk = c_lower[: min(40, len(c_lower))]
    if chunk and chunk in plan_lower:
        return True
    tokens = _significant_tokens(c_lower)
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in plan_lower)
    return hits >= max(1, min(2, len(tokens) // 2 + 1))


def acceptance_structurally_invalid_criteria(criteria: list[str]) -> list[str]:
    """Criteria that fail length/banned-substring checks (for tuning / debug)."""
    bad: list[str] = []
    for c in criteria:
        s = str(c).strip()
        if len(s) < 12:
            bad.append(str(c))
            continue
        lowered = s.lower()
        if any(b in lowered for b in ACCEPTANCE_BANNED_SUBSTRINGS):
            bad.append(str(c))
    return bad


def acceptance_evidence_missing_criteria(plan_markdown: str, criteria: list[str]) -> list[str]:
    """Structurally valid criteria with no shallow match in plan text."""
    return [c for c in criteria if not _criterion_satisfied_in_plan(plan_markdown, c)]


def acceptance_satisfied_from_plan(plan_markdown: str, criteria: list[str]) -> bool:
    """
    Shallow evidence: criterion text or its tokens must appear in the plan.
    Vacuously true when criteria is empty.
    """
    if not criteria:
        return True
    return len(acceptance_evidence_missing_criteria(plan_markdown, criteria)) == 0


def harness_failed_gate_label(gates: dict[str, Any]) -> str | None:
    """Single primary harness failure for logs (priority order)."""
    if gates.get("rubric_incomplete"):
        return "rubric_incomplete"
    if not gates.get("rubric_floor_ok", False):
        return "rubric_floor"
    if not gates.get("blockers_ok", False):
        return "blockers"
    if not gates.get("acceptance_structurally_valid", False):
        return "acceptance_structural"
    if not gates.get("acceptance_satisfied", False):
        return "acceptance_evidence"
    return None


def harness_failure_delta(
    gates: dict[str, Any],
    acceptance_missing: list[str],
) -> dict[str, float | int]:
    """
    Scalar gaps for tuning: how far below rubric floor (when failing), and how many criteria miss plan evidence.
    Omits rubric_floor key when the min axis already clears the floor.
    """
    delta: dict[str, float | int] = {"acceptance_missing_count": len(acceptance_missing)}
    if not gates.get("rubric_floor_ok", True):
        floor = float(gates.get("floor", 0.0))
        min_r = float(gates.get("min_rubric", 0.0))
        gap = floor - min_r
        if gap > 0:
            delta["rubric_floor"] = round(gap, 4)
    return delta


def build_convergence_debug_payload(
    gates: dict[str, Any],
    review: ReviewResult,
    plan_markdown: str,
    *,
    converged: bool,
    reason: str,
) -> dict[str, Any]:
    """Structured DEBUG line for tuning: primary failed gate, rubric min, blockers, acceptance gaps."""
    failed: str | None = None if converged else harness_failed_gate_label(gates)
    if not converged and failed is None:
        failed = "legacy"

    acceptance_missing: list[str] = []
    if not gates.get("acceptance_structurally_valid", True):
        acceptance_missing = acceptance_structurally_invalid_criteria(review.acceptance_criteria)
    elif not gates.get("acceptance_satisfied", True) and review.acceptance_criteria:
        acceptance_missing = acceptance_evidence_missing_criteria(plan_markdown, review.acceptance_criteria)

    return {
        **gates,
        "converged": converged,
        "reason": reason,
        "failed_gate": failed,
        "failure_delta": harness_failure_delta(gates, acceptance_missing),
        "blocking_categories": list(review.blocking_categories),
        "blocking_categories_invalid": bool(review.blocking_categories_invalid),
        "acceptance_missing": acceptance_missing,
    }


def compute_harness_convergence_gates(
    review: ReviewResult,
    plan_markdown: str,
    *,
    rubric_floor: float,
) -> dict[str, Any]:
    """Single source of truth for Tier-A harness gates (used by ConvergenceSignals + postcondition)."""
    axis_values = list(review.plan_rubric.values()) if review.plan_rubric else []
    min_rubric = min(axis_values) if axis_values else 0.0
    rubric_floor_ok = min_rubric >= float(rubric_floor)
    blockers_ok = len(review.blocking_categories) == 0 and not review.blocking_categories_invalid
    struct_ok = acceptance_criteria_structurally_valid(review.acceptance_criteria)
    acceptance_ok = struct_ok and acceptance_satisfied_from_plan(plan_markdown, review.acceptance_criteria)
    return {
        "min_rubric": min_rubric,
        "floor": float(rubric_floor),
        "rubric_floor_ok": rubric_floor_ok,
        "rubric_incomplete": bool(review.rubric_incomplete),
        "blockers_ok": blockers_ok,
        "acceptance_structurally_valid": struct_ok,
        "acceptance_satisfied": acceptance_ok,
    }


def verify_convergence_postcondition(
    converged: bool,
    review: ReviewResult,
    plan_markdown: str,
    *,
    rubric_floor: float,
) -> None:
    if not converged:
        return
    g = compute_harness_convergence_gates(review, plan_markdown, rubric_floor=rubric_floor)
    if not (
        g["rubric_floor_ok"]
        and g["blockers_ok"]
        and not g["rubric_incomplete"]
        and g["acceptance_structurally_valid"]
        and g["acceptance_satisfied"]
    ):
        raise RuntimeError(f"convergence postcondition violated: {g}")
