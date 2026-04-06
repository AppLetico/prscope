"""
Issue similarity helpers for deduping critic findings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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

    # Near-identical prose (SequenceMatcher) — complements token Jaccard for long sentences.
    _SEQUENCE_RATIO_DUPLICATE = 0.88
    # Shorter normalized string must be at least this fraction of the longer to count as containment duplicate.
    _SUBSTRING_LENGTH_RATIO_MIN = 0.6
    # Avoid trivial substring matches on very short strings.
    _SUBSTRING_MIN_CHARS = 12

    def _find_duplicate_in_pool(self, candidate: str, issues: list[tuple[str, str]]) -> str | None:
        """Match candidate against one pool (open or resolved) using the same pipeline as before."""
        if not issues:
            return None
        embeddings_on = self._embeddings_enabled()
        if embeddings_on:
            duplicate = self._find_embedding_duplicate(candidate, issues)
            if duplicate is not None:
                return duplicate
            if self.config.fallback_mode == "none":
                return self._find_text_duplicate(candidate, issues)
            return self._find_lexical_duplicate(candidate, issues)

        # Embeddings off + "none": strict text match only (Jaccard disabled). Old bug: returned None
        # before any text compare, so identical prose never matched resolved issues.
        if self.config.fallback_mode == "none":
            return self._find_text_duplicate(candidate, issues)
        return self._find_lexical_duplicate(candidate, issues)

    def find_duplicate(
        self,
        description: str,
        open_issues: list[tuple[str, str]],
        resolved_issues: list[tuple[str, str]] | None = None,
    ) -> str | None:
        """Return a duplicate issue id: **open** matches first, then **resolved** (same text rules).

        Resolved matches prevent spawning a new open issue when the critic repeats prose for an
        already-closed note.
        """
        candidate = description.strip()
        if not candidate:
            return None
        resolved = resolved_issues or []
        for pool in (open_issues, resolved):
            dup = self._find_duplicate_in_pool(candidate, pool)
            if dup is not None:
                return dup
        return None

    @staticmethod
    def _normalize_for_text_compare(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _issue_text_duplicate(self, a: str, b: str) -> bool:
        """True if normalized strings match closely (ratio), are equal, or one contains the other."""
        na = self._normalize_for_text_compare(a)
        nb = self._normalize_for_text_compare(b)
        if not na or not nb:
            return False
        if na == nb:
            return True
        if SequenceMatcher(None, na, nb).ratio() >= self._SEQUENCE_RATIO_DUPLICATE:
            return True
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(longer) < self._SUBSTRING_MIN_CHARS or len(shorter) < self._SUBSTRING_MIN_CHARS:
            return False
        if shorter not in longer:
            return False
        return len(shorter) >= int(len(longer) * self._SUBSTRING_LENGTH_RATIO_MIN)

    def _find_text_duplicate(self, candidate: str, open_issues: list[tuple[str, str]]) -> str | None:
        for issue_id, issue_text in open_issues:
            if self._issue_text_duplicate(candidate, issue_text):
                return issue_id
        return None

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

    # When cosine is below `similarity_threshold` but still in this band, embeddings are inconclusive:
    # run lexical duplicate check so near-paraphrases (e.g. performance wording) still merge.
    _EMBEDDING_LEXICAL_TIEBREAK_LOW = 0.75

    def _find_embedding_duplicate(self, candidate: str, open_issues: list[tuple[str, str]]) -> str | None:
        candidate_vector = self._get_embedding(candidate)
        if candidate_vector is None:
            return None

        threshold = float(self.config.similarity_threshold)
        tiebreak_low = self._EMBEDDING_LEXICAL_TIEBREAK_LOW
        candidate_tokens = self._normalize_tokens(candidate)

        for issue_id, issue_text in open_issues:
            existing_vector = self._get_embedding(issue_text)
            if existing_vector is None:
                continue
            similarity = self._cosine_similarity(candidate_vector, existing_vector)
            if similarity >= threshold:
                return issue_id
            if tiebreak_low <= similarity < threshold and candidate_tokens:
                existing_tokens = self._normalize_tokens(issue_text)
                if self._lexical_duplicate_pair(candidate_tokens, existing_tokens):
                    return issue_id
            if self._issue_text_duplicate(candidate, issue_text):
                return issue_id
        return None

    def _normalize_tokens(self, text: str) -> set[str]:
        parts = re.split(r"[^a-z0-9]+", text.lower())
        return {part for part in parts if len(part) > 2 and part not in self.stopwords}

    @staticmethod
    def _lexical_duplicate_pair(candidate_tokens: set[str], existing_tokens: set[str]) -> bool:
        """Return True if two issue descriptions are duplicates for lexical fallback.

        - Classic Jaccard >= 0.5 (unchanged).
        - Narrow paraphrase: subset-style overlap (high recall, modest Jaccard) with at most
          three shared tokens. This catches performance-monitoring paraphrases (three-token core)
          without merging distinct health/error issues that share four topical tokens.
        """
        if not candidate_tokens or not existing_tokens:
            return False
        inter = candidate_tokens & existing_tokens
        inter_len = len(inter)
        union = len(candidate_tokens | existing_tokens)
        jacc = inter_len / union if union else 0.0
        smaller = min(len(candidate_tokens), len(existing_tokens))
        recall = inter_len / smaller if smaller else 0.0
        if jacc >= 0.5:
            return True
        if recall >= 0.5 and 0.28 <= jacc <= 0.50 and inter_len <= 3:
            return True
        return False

    def _find_lexical_duplicate(self, candidate: str, open_issues: list[tuple[str, str]]) -> str | None:
        candidate_tokens = self._normalize_tokens(candidate)
        for issue_id, issue_text in open_issues:
            if self._issue_text_duplicate(candidate, issue_text):
                return issue_id
            if not candidate_tokens:
                continue
            existing_tokens = self._normalize_tokens(issue_text)
            if not existing_tokens:
                continue
            if self._lexical_duplicate_pair(candidate_tokens, existing_tokens):
                return issue_id
        return None
