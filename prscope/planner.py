"""
Backward-compatible planning engine shim.

This preserves the earlier `PlanningEngine` shape while delegating to the
new `planning/runtime` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PrscopeConfig, RepoProfile
from .planning.runtime import PlanningRuntime
from .store import Store


class PlanningEngine:
    """Compatibility wrapper around `PlanningRuntime`."""

    def __init__(self, store: Store, config: PrscopeConfig, repo: RepoProfile):
        self.runtime = PlanningRuntime(store=store, config=config, repo=repo)

    async def author_loop(self, session_id: str, user_context: str) -> dict[str, Any]:
        critic, author, convergence = await self.runtime.run_adversarial_round(
            session_id=session_id,
            user_input=user_context,
        )
        return {
            "plan": author.plan,
            "unverified_references": sorted(author.unverified_references),
            "convergence": {
                "converged": convergence.converged,
                "reason": convergence.reason,
                "change_pct": convergence.change_pct,
            },
            "critic": {
                "blocking_issues": critic.blocking_issues,
                "architectural_concerns": critic.architectural_concerns,
                "design_quality_score": critic.design_quality_score,
            },
        }

    async def adversarial_round(self, session_id: str, user_input: str | None = None) -> dict[str, Any]:
        critic, author, convergence = await self.runtime.run_adversarial_round(
            session_id=session_id,
            user_input=user_input,
        )
        return {
            "critic": critic,
            "author": author,
            "convergence": convergence,
        }

    def add_user_input(self, session_id: str, content: str) -> None:
        core = self.runtime._core(session_id)  # noqa: SLF001
        session = core.get_session()
        core.add_turn("user", content, round_number=session.current_round)

    def export(self, session_id: str, output_dir: Path | None = None):
        return self.runtime.export(session_id=session_id, output_dir=output_dir)
