from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..tools import ToolExecutor, ToolSafetyError


def _is_test_path(path: str) -> bool:
    normalized = str(path).lower()
    return normalized.startswith("tests/") or ".test." in normalized or normalized.endswith("_test.py")


@dataclass(frozen=True)
class RefinementEvidenceRefreshResult:
    reason: str
    search_queries: list[str] = field(default_factory=list)
    anchor_paths: list[str] = field(default_factory=list)
    adjacent_tests: list[str] = field(default_factory=list)
    evidence_notes: list[str] = field(default_factory=list)
    read_paths: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    reused_known_anchors: bool = False

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "search_queries": list(self.search_queries),
            "anchor_paths": list(self.anchor_paths),
            "adjacent_tests": list(self.adjacent_tests),
            "evidence_notes": list(self.evidence_notes),
            "read_paths": list(self.read_paths),
            "elapsed_ms": self.elapsed_ms,
            "reused_known_anchors": self.reused_known_anchors,
        }

    def all_paths(self) -> list[str]:
        return list(dict.fromkeys([*self.anchor_paths, *self.adjacent_tests, *self.read_paths]))


class RefinementEvidenceRefresh:
    _ALLOWED_SUFFIXES = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".sql",
        ".sh",
    }
    _STOPWORDS = {
        "about",
        "across",
        "address",
        "after",
        "against",
        "also",
        "because",
        "between",
        "change",
        "changes",
        "component",
        "decision",
        "explicitly",
        "followup",
        "issue",
        "latest",
        "message",
        "missing",
        "plan",
        "pressure",
        "refine",
        "review",
        "round",
        "section",
        "should",
        "tests",
        "their",
        "there",
        "these",
        "this",
        "update",
        "user",
        "with",
    }

    def __init__(self, tool_executor: ToolExecutor):
        self._tool_executor = tool_executor

    @classmethod
    def _focus_terms(cls, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in re.findall(r"[a-zA-Z0-9_./-]+", str(text or "")):
            token = raw.strip().strip("`").lower()
            if len(token) < 4 or token in cls._STOPWORDS or token.startswith("issue_") or token in tokens:
                continue
            tokens.append(token)
        return tokens[:8]

    @staticmethod
    def _search_patterns(
        *,
        user_message: str,
        known_anchor_paths: list[str],
        reason: str,
    ) -> list[str]:
        focus_terms = RefinementEvidenceRefresh._focus_terms(user_message)
        basename_terms = [
            path.rsplit("/", 1)[-1].split(".", 1)[0].lower() for path in known_anchor_paths if str(path).strip()
        ]
        basename_terms = [term for term in basename_terms if len(term) >= 4]
        prioritized_terms = list(dict.fromkeys([*basename_terms, *focus_terms]))
        patterns: list[str] = []
        if prioritized_terms:
            patterns.append("|".join(re.escape(term) for term in prioritized_terms[:2]))
        if len(prioritized_terms) >= 4:
            patterns.append("|".join(re.escape(term) for term in prioritized_terms[2:4]))
        if reason in {"missing_tests", "missing_grounding"} and prioritized_terms:
            patterns.append("|".join(re.escape(term) for term in prioritized_terms[:1]))
        return patterns[:3]

    @staticmethod
    def _snippet_note(snippet: dict[str, Any]) -> str:
        content = str(snippet.get("content", "")).splitlines()
        preview = next((line.strip() for line in content if line.strip()), "")
        if len(preview) > 120:
            preview = preview[:117].rstrip() + "..."
        path = str(snippet.get("path", "")).strip()
        start = int(snippet.get("start_line", 1) or 1)
        return f"{path}:{start} {preview}".strip()

    @classmethod
    def _should_keep_match_path(cls, path: str) -> bool:
        normalized = str(path).strip()
        if not normalized or normalized.startswith(".") or "/." in normalized:
            return False
        lower = normalized.lower()
        if lower.startswith(("docs/", "plans/", ".pytest_cache/", ".prscope/")):
            return False
        suffix = ""
        if "." in normalized.rsplit("/", 1)[-1]:
            suffix = "." + normalized.rsplit(".", 1)[-1].lower()
        return suffix in cls._ALLOWED_SUFFIXES or _is_test_path(normalized)

    @staticmethod
    def _rank_path(path: str, *, known_anchor_paths: list[str]) -> tuple[int, int, str]:
        path_lower = path.lower()
        direct_anchor = 0 if path in known_anchor_paths else 1
        test_bias = 1 if _is_test_path(path_lower) else 0
        source_bias = (
            0
            if path_lower.startswith(("src/", "tests/"))
            else 1
            if "/src/" in path_lower or "/tests/" in path_lower
            else 2
        )
        return (direct_anchor, source_bias, test_bias, path_lower)

    def refresh(
        self,
        *,
        user_message: str,
        reason: str,
        known_anchor_paths: list[str] | None = None,
        max_search_queries: int = 3,
        max_files_read: int = 8,
        max_wall_clock_seconds: float = 5.0,
    ) -> RefinementEvidenceRefreshResult:
        started = time.perf_counter()
        known_anchor_paths = [path for path in (known_anchor_paths or []) if str(path).strip()][:8]
        candidate_paths: list[str] = []
        evidence_notes: list[str] = []
        read_paths: list[str] = []
        search_queries: list[str] = []
        reused_known_anchors = False

        for path in known_anchor_paths[: min(3, max_files_read)]:
            try:
                snippet = self._tool_executor.read_file(path=path, max_lines=80)
            except (ToolSafetyError, OSError):
                continue
            candidate_paths.append(path)
            read_paths.append(path)
            evidence_notes.append(self._snippet_note(snippet))
            reused_known_anchors = True
            if (time.perf_counter() - started) >= max_wall_clock_seconds:
                break

        for pattern in self._search_patterns(
            user_message=user_message,
            known_anchor_paths=known_anchor_paths,
            reason=reason,
        )[: max(1, max_search_queries)]:
            if (time.perf_counter() - started) >= max_wall_clock_seconds:
                break
            try:
                result = self._tool_executor.grep_code(pattern=pattern, max_results=20)
            except (ToolSafetyError, OSError, re.error):
                continue
            search_queries.append(pattern)
            for match in result.get("results", []):
                match_path = str(match.get("path", "")).strip()
                if self._should_keep_match_path(match_path) and match_path not in candidate_paths:
                    candidate_paths.append(match_path)

        ranked_candidates = sorted(
            candidate_paths,
            key=lambda path: self._rank_path(path, known_anchor_paths=known_anchor_paths),
        )
        for path in ranked_candidates:
            if path in read_paths:
                continue
            if len(read_paths) >= max_files_read or (time.perf_counter() - started) >= max_wall_clock_seconds:
                break
            try:
                snippet = self._tool_executor.read_file(path=path, max_lines=80)
            except (ToolSafetyError, OSError):
                continue
            read_paths.append(path)
            evidence_notes.append(self._snippet_note(snippet))

        anchor_paths = [path for path in read_paths if not _is_test_path(path)][:3]
        adjacent_tests = [path for path in read_paths if _is_test_path(path)][:3]
        if not anchor_paths and known_anchor_paths:
            anchor_paths = known_anchor_paths[:3]
        if not evidence_notes:
            evidence_notes.append("No stronger repo anchors were found within the bounded refresh budget.")
        return RefinementEvidenceRefreshResult(
            reason=reason,
            search_queries=search_queries,
            anchor_paths=anchor_paths,
            adjacent_tests=adjacent_tests,
            evidence_notes=evidence_notes[:6],
            read_paths=read_paths[:max_files_read],
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            reused_known_anchors=reused_known_anchors,
        )
