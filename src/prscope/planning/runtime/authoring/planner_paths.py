"""Planner draft: shared verified path ordering for prompts and Files Changed validation."""

from __future__ import annotations

from .models import RepoUnderstanding


def planner_verified_file_paths(repo_understanding: RepoUnderstanding, *, limit: int = 40) -> list[str]:
    """Paths shown under Verified File Paths in planner/refiner draft prompts (deduped, stable order)."""
    prioritized: list[str] = []
    for group in (
        repo_understanding.relevant_modules,
        repo_understanding.relevant_tests,
        list(repo_understanding.file_contents.keys()),
        repo_understanding.entrypoints,
        repo_understanding.core_modules,
    ):
        for path in group:
            normalized = str(path).strip()
            if normalized and normalized not in prioritized:
                prioritized.append(normalized)
    return prioritized[:limit]
