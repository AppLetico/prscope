"""
Context and memory assembly helpers for planning runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from ....memory import load_skills

if TYPE_CHECKING:
    from ....config import PlanningConfig, RepoProfile
    from ....memory import MemoryStore
    from ....store import Store
    from ..state import PlanningState

logger = logging.getLogger(__name__)


class ContextAssembler:
    def __init__(
        self,
        *,
        repo: RepoProfile,
        planning_config: PlanningConfig,
        memory: MemoryStore,
        store: Store,
        state_getter: Callable[[str], PlanningState],
        truncate_memory_block: Callable[[str, str], str],
        recall_disabled: Callable[[str], bool],
    ) -> None:
        self.repo = repo
        self.planning_config = planning_config
        self.memory = memory
        self.store = store
        self._state = state_getter
        self._truncate_memory_block = truncate_memory_block
        self._recall_disabled = recall_disabled

    def repo_memory(self, state: PlanningState) -> dict[str, str]:
        if not state.repo_memory:
            state.repo_memory = self.memory.load_all_blocks()
        return state.repo_memory

    def skills_context(self, session_id: str) -> str:
        state = self._state(session_id)
        if state.skills_context:
            return state.skills_context
        skills_body = load_skills(self.repo.skills_dir, self.planning_config.skills_max_chars)
        if not skills_body:
            state.skills_context = ""
            return ""
        wrapped = (
            "## Team Skills & Patterns\n"
            "The following represent stable team patterns and engineering defaults.\n"
            "Apply them unless explicitly overridden by requirements.\n\n"
            "---\n"
            f"{skills_body}\n"
        )
        state.skills_context = wrapped
        return wrapped

    def build_recall_context(self, session_id: str, query: str) -> str:
        if not self.planning_config.recall_prior_sessions:
            return ""
        if self._recall_disabled(session_id):
            return ""
        if len(query.split()) < 6:
            logger.info("[recall] Skipped - insufficient query signal")
            return ""

        ranked = self.store.search_sessions(
            query=query,
            repo_name=self.repo.name,
            limit=self.planning_config.recall_top_k,
        )
        if not ranked:
            return ""

        header = (
            "## Prior Planning Context\nHistorical context may be outdated. Validate against current requirements.\n\n"
        )
        body_blocks: list[str] = []
        total = len(header)
        max_chars = max(0, self.planning_config.recall_max_chars)
        for item in ranked:
            block = (
                f"### Session: {item['title']} ({item['created_at'][:10]})\n"
                f"Repo: {item['repo_name']}\n"
                f"Relevance Score: {float(item['score']):.2f}\n\n"
                "Key Summary:\n"
                f"- {item['summary_snippet']}\n"
            )
            if max_chars > 0 and (total + len(block) > max_chars):
                break
            body_blocks.append(block)
            total += len(block)
        if not body_blocks:
            return ""
        return header + "\n".join(body_blocks)

    @staticmethod
    def build_context_index(blocks: dict[str, str]) -> str:
        descriptions = {
            "architecture": "high-level system structure and dependencies",
            "modules": "module responsibilities and file mapping",
            "patterns": "cross-cutting implementation patterns",
            "entrypoints": "runtime and CLI entrypoints",
            "context": "scanner-generated rich repository context",
            "mental_model": "synthesized repository mental model (key abstractions, dependency flow, critical paths)",
        }
        lines: list[str] = []
        for key, value in blocks.items():
            if key == "manifesto":
                continue
            if not value:
                continue
            desc = descriptions.get(key, "repo memory block")
            lines.append(f"- {key} memory ({len(value)} chars): {desc}")
        lines.append("- Entire repo via grep_code, list_files, read_file")
        return "\n".join(lines)

    def memory_block_for_tool(self, key: str, allowed_keys: set[str]) -> dict[str, Any]:
        if key not in allowed_keys:
            allowed = ", ".join(sorted(allowed_keys))
            raise ValueError(f"Unsupported memory block '{key}'. Allowed: {allowed}")
        raw = self.memory.load_block(key)
        truncated = self._truncate_memory_block(key, raw)
        return {
            "key": key,
            "truncated": truncated != raw,
            "content": truncated,
        }
