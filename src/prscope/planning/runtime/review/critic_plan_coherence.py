"""
Post-parse adjustment: drop blocking lines that contradict obvious ## Current Plan content.

Conservative heuristics only — prefer prompt/mode fixes; this is a safety net.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any

_LOG = logging.getLogger(__name__)

# HTTP route shape in plan prose (method + path).
_ROUTE_PATTERN = re.compile(
    r"\b(?:POST|GET|PUT|PATCH|DELETE)\s+/[^\s\)`]+",
    re.IGNORECASE,
)
# Backtick-wrapped file paths (weak but common in plans).
_FILE_BACKTICK_PATTERN = re.compile(r"`([^`]+\.[a-z0-9]+)`", re.IGNORECASE)


def _plan_has_concrete_routes(plan: str) -> bool:
    if _ROUTE_PATTERN.search(plan):
        return True
    lowered = plan.lower()
    return "/oauth" in lowered and ("post " in lowered or "`post " in lowered)


def _plan_has_test_strategy_section(plan: str) -> bool:
    lowered = plan.lower()
    if "## test strategy" in lowered or "## testing" in lowered:
        return True
    if "unit test" in lowered and "test" in lowered:
        # Require some substance beyond a single word
        return len(plan) > 400 and ("tests/" in lowered or "pytest" in lowered or "test_" in lowered)
    return False


def _plan_has_multiple_file_refs(plan: str) -> bool:
    paths = _FILE_BACKTICK_PATTERN.findall(plan)
    return len(set(paths)) >= 2


def _is_stale_endpoint_blocking(line: str, plan: str) -> bool:
    low = line.lower()
    if not any(
        p in low
        for p in (
            "no concrete endpoint",
            "endpoint definitions",
            "no endpoint",
            "routes provided",
            "oauth2 routes",
        )
    ):
        return False
    return _plan_has_concrete_routes(plan)


def _is_stale_test_blocking(line: str, plan: str) -> bool:
    low = line.lower()
    if "test" not in low:
        return False
    if not any(p in low for p in ("missing test", "no test strategy", "test strategy", "testing strategy")):
        return False
    return _plan_has_test_strategy_section(plan)


def _is_stale_evidence_blocking(line: str, plan: str) -> bool:
    low = line.lower()
    if "evidence" not in low and "linking" not in low and "codebase" not in low:
        return False
    if not any(p in low for p in ("lack of evidence", "linking the plan", "existing codebase", "architecture")):
        return False
    return _plan_has_multiple_file_refs(plan)


def coherence_adjust_review(plan_content: str, review: Any) -> Any:
    """Remove blocking lines that are inconsistent with obvious plan content."""
    from ..critic import ReviewResult

    if not isinstance(review, ReviewResult):
        return review
    plan = plan_content or ""
    if not plan.strip():
        return review

    kept: list[str] = []
    dropped: list[str] = []
    for line in review.blocking_issues:
        s = str(line).strip()
        if not s:
            continue
        if _is_stale_endpoint_blocking(s, plan):
            dropped.append(s)
            continue
        if _is_stale_test_blocking(s, plan):
            dropped.append(s)
            continue
        if _is_stale_evidence_blocking(s, plan):
            dropped.append(s)
            continue
        kept.append(s)

    if not dropped:
        return review

    for item in dropped:
        _LOG.debug("Coherence prune: dropped blocking line: %s", item[:160])

    primary = review.primary_issue
    pstrip = str(primary).strip() if primary else ""
    dropped_set = {d.strip() for d in dropped}
    if pstrip and (pstrip in dropped_set or any(pstrip == d.strip() for d in dropped)):
        primary = kept[0] if kept else None

    return replace(
        review,
        blocking_issues=kept,
        primary_issue=primary,
    )
