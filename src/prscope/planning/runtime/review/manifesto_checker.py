from __future__ import annotations

import re
from dataclasses import dataclass

from ....memory import ParsedConstraint


@dataclass
class ManifestoCheckResult:
    violations: list[str]
    warnings: list[str]


class ManifestoChecker:
    """Lightweight post-refinement manifesto validation pass."""

    _SAFE_CONTEXT_PATTERNS = (
        r"\bdo not\b",
        r"\bdon't\b",
        r"\bnever\b",
        r"\bavoid\b",
        r"\bwithout\b",
        r"\bredact(?:ed)?\b",
        r"\bmask(?:ed|ing)?\b",
        r"\bomit(?:ted)?\b",
        r"\bscrub(?:bed|bing)?\b",
    )
    _RISK_ACTION_PATTERNS = (
        r"\blog\b",
        r"\bprint\b",
        r"\breturn\b",
        r"\bstore\b",
        r"\bsave\b",
        r"\bpersist\b",
        r"\bwrite\b",
        r"\bsend\b",
        r"\btransmit\b",
        r"\bexpose\b",
        r"\bhard[- ]?code\b",
        r"\bset\b",
        r"\bassign\b",
        r"\binclude\b",
    )
    _ROLLBACK_PATTERNS = (
        r"\brollback\b",
        r"\brevert\b",
        r"\brestore\b",
        r"\bbackout\b",
        r"\bundo\b",
    )

    @classmethod
    def _keyword_pattern(cls, keyword: str) -> str:
        normalized = re.escape(keyword.strip().lower()).replace("_", r"[_\s-]+")
        return rf"\b{normalized}s?\b"

    @classmethod
    def _matching_windows(cls, text: str, keyword: str) -> list[str]:
        windows: list[str] = []
        keyword_pattern = cls._keyword_pattern(keyword)
        window_pattern = re.compile(rf".{{0,48}}(?:{keyword_pattern})|(?:{keyword_pattern}).{{0,48}}")
        for match in window_pattern.finditer(text):
            windows.append(match.group(0))
        return windows

    @classmethod
    def _has_safe_context(cls, text: str, keyword: str) -> bool:
        for snippet in cls._matching_windows(text, keyword):
            if any(re.search(pattern, snippet) for pattern in cls._SAFE_CONTEXT_PATTERNS):
                return True
        return False

    @classmethod
    def _looks_like_secret_handling_risk(cls, text: str, keyword: str) -> bool:
        if cls._has_safe_context(text, keyword):
            return False
        keyword_pattern = cls._keyword_pattern(keyword)
        patterns = (
            rf"(?:{'|'.join(cls._RISK_ACTION_PATTERNS)}).{{0,32}}(?:{keyword_pattern})",
            rf"(?:{keyword_pattern}).{{0,32}}(?:{'|'.join(cls._RISK_ACTION_PATTERNS)})",
            rf"(?:{keyword_pattern})\s*(?:=|:)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @classmethod
    def _looks_like_destructive_plan_without_rollback(cls, text: str, keyword: str) -> bool:
        if cls._has_safe_context(text, keyword):
            return False
        if not re.search(cls._keyword_pattern(keyword), text):
            return False
        has_rollback = any(re.search(pattern, text) for pattern in cls._ROLLBACK_PATTERNS)
        return not has_rollback

    def validate(self, plan_content: str, constraints: list[ParsedConstraint]) -> ManifestoCheckResult:
        text = (plan_content or "").lower()
        violations: list[str] = []
        warnings: list[str] = []
        for constraint in constraints:
            keywords = [kw.lower() for kw in (constraint.evidence_keywords or []) if kw]
            if not keywords:
                continue
            matched = False
            for keyword in keywords:
                if "constraint_001" in constraint.id.lower():
                    matched = self._looks_like_secret_handling_risk(text, keyword)
                elif "constraint_002" in constraint.id.lower():
                    matched = self._looks_like_destructive_plan_without_rollback(text, keyword)
                else:
                    matched = bool(re.search(self._keyword_pattern(keyword), text)) and not self._has_safe_context(
                        text, keyword
                    )
                if matched:
                    break
            if matched:
                if constraint.severity == "hard" and not constraint.optional:
                    violations.append(constraint.id)
                else:
                    warnings.append(constraint.id)
        return ManifestoCheckResult(violations=violations, warnings=warnings)
