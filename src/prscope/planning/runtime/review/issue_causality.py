"""
Rule-based causal edge inference from critic prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....config import IssueGraphConfig
    from ..critic import ReviewResult
    from .issue_graph import IssueGraphTracker


@dataclass
class IssueCausalityExtractionResult:
    scanned_sentences: int = 0
    extracted_pairs: int = 0
    accepted_edges: int = 0
    rejected_pairs: int = 0


class IssueCausalityExtractor:
    def __init__(self, config: IssueGraphConfig) -> None:
        self._config = config

    def extract_edges(
        self,
        *,
        graph: IssueGraphTracker,
        review: ReviewResult,
        round_number: int,
    ) -> IssueCausalityExtractionResult:
        result = IssueCausalityExtractionResult()
        if not self._config.causality_extraction_enabled:
            return result

        candidate_sentences = self._candidate_sentences(review)
        seen_edges: set[tuple[str, str]] = set()
        max_edges = max(0, int(self._config.causality_max_edges_per_review))
        for sentence in candidate_sentences:
            result.scanned_sentences += 1
            pair = self._extract_cause_effect(sentence)
            if pair is None:
                continue
            result.extracted_pairs += 1
            cause_text, effect_text = pair
            if not self._is_actionable_issue_text(cause_text) or not self._is_actionable_issue_text(effect_text):
                result.rejected_pairs += 1
                continue
            if max_edges and result.accepted_edges >= max_edges:
                break
            cause_issue = graph.add_issue(
                cause_text,
                round_number,
                severity="minor",
                source="inference",
            )
            effect_issue = graph.add_issue(
                effect_text,
                round_number,
                severity="minor",
                source="inference",
            )
            if not cause_issue.id or not effect_issue.id:
                result.rejected_pairs += 1
                continue
            key = (graph.canonical_issue_id(cause_issue.id), graph.canonical_issue_id(effect_issue.id))
            if key in seen_edges or key[0] == key[1]:
                continue
            seen_edges.add(key)
            graph.add_edge(key[0], key[1], "causes")
            result.accepted_edges += 1
        return result

    def _candidate_sentences(self, review: ReviewResult) -> list[str]:
        sentences: list[str] = []
        if review.primary_issue:
            sentences.append(str(review.primary_issue))
        sentences.extend([str(item) for item in review.blocking_issues if str(item).strip()])
        sentences.extend([str(item) for item in review.recommended_changes if str(item).strip()])
        prose = str(review.prose or "").strip()
        if prose:
            for part in re.split(r"[.!?]\s+|\n+", prose):
                text = str(part).strip()
                if text:
                    sentences.append(text)
        return sentences

    def _extract_cause_effect(self, sentence: str) -> tuple[str, str] | None:
        text = " ".join(str(sentence).split())
        if not text:
            return None
        lowered = text.lower()
        if any(pattern in lowered for pattern in self._config.causality_negative_patterns):
            return None
        if self._config.causality_patterns and not any(marker in lowered for marker in self._config.causality_patterns):
            return None

        pairs: list[tuple[str, str]] = []
        if " because " in lowered:
            effect, cause = self._split_once(text, "because")
            pairs.append((cause, effect))
        if " due to " in lowered:
            effect, cause = self._split_once(text, "due to")
            pairs.append((cause, effect))
        if " caused by " in lowered:
            effect, cause = self._split_once(text, "caused by")
            pairs.append((cause, effect))
        if " leads to " in lowered:
            cause, effect = self._split_once(text, "leads to")
            pairs.append((cause, effect))
        if " results in " in lowered:
            cause, effect = self._split_once(text, "results in")
            pairs.append((cause, effect))
        if " therefore " in lowered:
            cause, effect = self._split_once(text, "therefore")
            pairs.append((cause, effect))

        for cause, effect in pairs:
            cause_clean = self._clean_fragment(cause)
            effect_clean = self._clean_fragment(effect)
            if cause_clean and effect_clean:
                return cause_clean, effect_clean
        return None

    @staticmethod
    def _split_once(text: str, marker: str) -> tuple[str, str]:
        pattern = re.compile(re.escape(marker), flags=re.IGNORECASE)
        parts = pattern.split(text, maxsplit=1)
        if len(parts) != 2:
            return text, ""
        return parts[0], parts[1]

    @staticmethod
    def _clean_fragment(value: str) -> str:
        cleaned = value.strip(" .,:;-")
        cleaned = re.sub(r"^(this|that|it)\s+", "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def _is_actionable_issue_text(self, value: str) -> bool:
        text = " ".join(value.strip().split())
        min_len = max(1, int(self._config.causality_min_text_len))
        if len(text) < min_len:
            return False
        if any(pattern in text.lower() for pattern in self._config.causality_negative_patterns):
            return False
        # Noun-like heuristic: must include at least one alphabetic token with >=4 chars.
        if re.search(r"\b[a-zA-Z]{4,}\b", text) is None:
            return False
        return True
