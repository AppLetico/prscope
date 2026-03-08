from __future__ import annotations

from typing import Literal

IssueType = Literal["architecture", "ambiguity", "correctness", "performance"]

_AMBIGUITY_MARKERS = (
    "ambiguous",
    "ambiguity",
    "unclear",
    "underspecified",
    "unspecified",
    "not specified",
    "missing",
    "needs clarification",
    "open question",
    "undecided",
    "unresolved",
)

_PERFORMANCE_MARKERS = (
    "latency",
    "slow",
    "throughput",
    "performance",
    "scaling",
    "scale",
    "capacity",
    "bottleneck",
)

_ARCHITECTURE_MARKERS = (
    "architecture",
    "architectural",
    "layering",
    "layer",
    "boundary",
    "module",
    "dependency",
    "database",
    "cache",
    "protocol",
    "schema",
)


def infer_issue_type(
    issue_text: str,
    *,
    source_kind: str | None = None,
    decision_relation: str | None = None,
) -> IssueType:
    lowered = str(issue_text or "").lower()
    normalized_kind = str(source_kind or "").strip().lower()
    normalized_relation = str(decision_relation or "").strip().lower()

    if normalized_kind in {"constraint_violation", "blocking_issue", "validation_blocking_issue"}:
        return "performance" if _contains_any(lowered, _PERFORMANCE_MARKERS) else "correctness"
    if normalized_kind in {"architectural_concern", "validation_architectural_concern"}:
        return "performance" if _contains_any(lowered, _PERFORMANCE_MARKERS) else "architecture"

    if normalized_relation == "missing" or _contains_any(lowered, _AMBIGUITY_MARKERS):
        return "ambiguity"
    if _contains_any(lowered, _PERFORMANCE_MARKERS):
        return "performance"
    if normalized_relation == "conflict" or _contains_any(lowered, _ARCHITECTURE_MARKERS):
        return "architecture"
    return "correctness"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(marker in text for marker in needles)
