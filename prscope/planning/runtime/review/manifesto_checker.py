from __future__ import annotations

from dataclasses import dataclass

from ....memory import ParsedConstraint


@dataclass
class ManifestoCheckResult:
    violations: list[str]
    warnings: list[str]


class ManifestoChecker:
    """Lightweight post-refinement manifesto validation pass."""

    def validate(self, plan_content: str, constraints: list[ParsedConstraint]) -> ManifestoCheckResult:
        text = (plan_content or "").lower()
        violations: list[str] = []
        warnings: list[str] = []
        for constraint in constraints:
            keywords = [kw.lower() for kw in (constraint.evidence_keywords or []) if kw]
            if not keywords:
                continue
            if any(keyword in text for keyword in keywords):
                if constraint.severity == "hard" and not constraint.optional:
                    violations.append(constraint.id)
                else:
                    warnings.append(constraint.id)
        return ManifestoCheckResult(violations=violations, warnings=warnings)
