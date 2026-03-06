"""
Issue similarity helpers for deduping critic findings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....config import IssueDedupeConfig

logger = logging.getLogger(__name__)


@dataclass
class IssueSimilarityService:
    config: IssueDedupeConfig
    stopwords: set[str] = field(
        default_factory=lambda: {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "with",
            "is",
            "are",
            "be",
            "that",
            "this",
            "it",
            "as",
            "on",
        }
    )
    _embedding_cache: dict[str, list[float]] = field(default_factory=dict)

    def find_duplicate(
        self,
        description: str,
        open_issues: list[tuple[str, str]],
    ) -> str | None:
        candidate = description.strip()
        if not candidate:
            return None

        if self._embeddings_enabled():
            duplicate = self._find_embedding_duplicate(candidate, open_issues)
            if duplicate is not None:
                return duplicate

        if self.config.fallback_mode == "none":
            return None
        return self._find_lexical_duplicate(candidate, open_issues)

    def _embeddings_enabled(self) -> bool:
        mode = self.config.embeddings_enabled
        if mode == "true":
            return True
        if mode == "false":
            return False
        return bool(self.config.embedding_model)

    def _get_embedding(self, text: str) -> list[float] | None:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        try:
            import litellm

            response = litellm.embedding(model=self.config.embedding_model, input=[text[:4000]])
            vector = response.data[0]["embedding"]
            if isinstance(vector, list):
                self._embedding_cache[text] = vector
                return vector
        except Exception as exc:  # noqa: BLE001
            logger.debug("Issue dedupe embedding unavailable; falling back to lexical: %s", exc)
        return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a_norm = sum(v * v for v in a) ** 0.5
        b_norm = sum(v * v for v in b) ** 0.5
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (a_norm * b_norm)

    def _find_embedding_duplicate(self, candidate: str, open_issues: list[tuple[str, str]]) -> str | None:
        candidate_vector = self._get_embedding(candidate)
        if candidate_vector is None:
            return None

        for issue_id, issue_text in open_issues:
            existing_vector = self._get_embedding(issue_text)
            if existing_vector is None:
                continue
            similarity = self._cosine_similarity(candidate_vector, existing_vector)
            if similarity >= float(self.config.similarity_threshold):
                return issue_id
        return None

    def _normalize_tokens(self, text: str) -> set[str]:
        parts = re.split(r"[^a-z0-9]+", text.lower())
        return {part for part in parts if len(part) > 2 and part not in self.stopwords}

    def _find_lexical_duplicate(self, candidate: str, open_issues: list[tuple[str, str]]) -> str | None:
        candidate_tokens = self._normalize_tokens(candidate)
        if not candidate_tokens:
            return None
        for issue_id, issue_text in open_issues:
            existing_tokens = self._normalize_tokens(issue_text)
            if not existing_tokens:
                continue
            overlap = len(candidate_tokens & existing_tokens) / max(len(candidate_tokens | existing_tokens), 1)
            if overlap >= 0.5:
                return issue_id
        return None
