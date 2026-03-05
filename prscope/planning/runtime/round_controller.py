"""
Round orchestration helpers extracted from PlanningRuntime.
"""

from __future__ import annotations

from dataclasses import dataclass

from .critic import CriticResult


@dataclass
class PlanDelta:
    issues_resolved: int
    issues_introduced: int
    net_improvement: int


def compute_plan_delta(previous: CriticResult | None, current: CriticResult) -> PlanDelta:
    if previous is None:
        return PlanDelta(issues_resolved=0, issues_introduced=0, net_improvement=0)
    previous_total = len(getattr(previous, "blocking_issues", []) or []) + len(
        getattr(previous, "architectural_concerns", []) or []
    )
    current_total = len(getattr(current, "blocking_issues", []) or []) + len(
        getattr(current, "architectural_concerns", []) or []
    )
    if current_total <= previous_total:
        resolved = previous_total - current_total
        return PlanDelta(
            issues_resolved=resolved,
            issues_introduced=0,
            net_improvement=resolved,
        )
    introduced = current_total - previous_total
    return PlanDelta(
        issues_resolved=0,
        issues_introduced=introduced,
        net_improvement=-introduced,
    )
