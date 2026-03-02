"""
Pure planning core logic: state machine, versioning, and convergence checks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
import difflib
from difflib import SequenceMatcher

from ..config import PlanningConfig
from ..store import PlanVersion, PlanningTurn, Store


@dataclass
class PlanStructure:
    todo_count: int
    file_path_count: int
    has_goals: bool
    has_non_goals: bool
    has_todos_section: bool


@dataclass
class ConvergenceResult:
    converged: bool
    reason: str
    change_pct: float
    regression: str | None
    major_issues: int | None = None


class ApprovalBlockedError(RuntimeError):
    """Raised when approval is blocked by strict verified-reference mode."""


class InvalidTransitionError(RuntimeError):
    """Raised when a state transition is not allowed."""


class PlanningCore:
    """Deterministic planning state and convergence logic."""

    ALLOWED_TRANSITIONS: dict[str, set[str]] = {
        "created": {"discovering"},
        "discovering": {"drafting"},
        "discovery": {"drafting"},  # backward-compatible persisted value
        "drafting": {"refining"},
        "refining": {"refining", "converged"},
        "converged": {"approved", "refining"},
        "approved": {"exported"},
        "exported": set(),
    }

    def __init__(self, store: Store, session_id: str, config: PlanningConfig):
        self.store = store
        self.session_id = session_id
        self.config = config

    def get_session(self):
        session = self.store.get_planning_session(self.session_id)
        if session is None:
            raise ValueError(f"Planning session not found: {self.session_id}")
        return session

    def get_current_plan(self) -> PlanVersion | None:
        versions = self.store.get_plan_versions(self.session_id, limit=1)
        return versions[0] if versions else None

    def save_plan_version(self, content: str, round_number: int) -> PlanVersion:
        plan_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        previous = self.store.get_plan_version(self.session_id, max(round_number - 1, 0))
        diff_from_previous: str | None = None
        if previous is not None:
            diff_from_previous = "\n".join(
                difflib.unified_diff(
                    previous.plan_content.splitlines(),
                    content.splitlines(),
                    fromfile=f"round-{previous.round}",
                    tofile=f"round-{round_number}",
                    lineterm="",
                )
            )
        version = self.store.save_plan_version(
            session_id=self.session_id,
            round_number=round_number,
            plan_content=content,
            plan_sha=plan_sha,
            diff_from_previous=diff_from_previous,
        )
        self.store.update_planning_session(self.session_id, current_round=round_number)
        return version

    def add_turn(
        self,
        role: str,
        content: str,
        round_number: int,
        major_issues_remaining: int | None = None,
        minor_issues_remaining: int | None = None,
        hard_constraint_violations: list[str] | None = None,
        parse_error: str | None = None,
    ) -> PlanningTurn:
        return self.store.add_planning_turn(
            session_id=self.session_id,
            role=role,
            content=content,
            round_number=round_number,
            major_issues_remaining=major_issues_remaining,
            minor_issues_remaining=minor_issues_remaining,
            hard_constraint_violations=hard_constraint_violations,
            parse_error=parse_error,
        )

    def get_conversation(self, round_number: int | None = None) -> list[PlanningTurn]:
        return self.store.get_planning_turns(self.session_id, round_number=round_number)

    def advance_round(self) -> int:
        session = self.get_session()
        next_round = session.current_round + 1
        self.store.update_planning_session(self.session_id, current_round=next_round)
        return next_round

    def transition(self, new_status: str) -> None:
        session = self.get_session()
        allowed = self.ALLOWED_TRANSITIONS.get(session.status, set())
        if new_status not in allowed and new_status != session.status:
            raise InvalidTransitionError(f"Invalid transition {session.status} -> {new_status}")
        self.store.update_planning_session(self.session_id, status=new_status)

    def approve(self, unverified_references: set[str] | None = None) -> None:
        if self.config.require_verified_file_references and unverified_references:
            count = len(unverified_references)
            raise ApprovalBlockedError(
                f"Approval blocked: {count} file references unverified. "
                "Read them first or disable require_verified_file_references."
            )
        session = self.get_session()
        if session.status not in {"converged", "approved"}:
            raise InvalidTransitionError(f"Cannot approve from status: {session.status}")
        self.store.update_planning_session(self.session_id, status="approved")

    def mark_exported(self) -> None:
        self.transition("exported")

    def _extract_structure(self, content: str) -> PlanStructure:
        todo_count = content.count("- [ ]") + content.count("TODO")
        file_paths = re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", content)
        return PlanStructure(
            todo_count=todo_count,
            file_path_count=len(set(file_paths)),
            has_goals=("## Goals" in content or "## Goal" in content),
            has_non_goals=("## Non-Goals" in content or "## Non-Goal" in content),
            has_todos_section=("## TODO" in content or "## Implementation" in content),
        )

    @staticmethod
    def _compute_change_pct(previous: str, current: str) -> float:
        ratio = SequenceMatcher(None, previous, current).ratio()
        return max(0.0, min(1.0, 1.0 - ratio))

    def check_convergence(self) -> ConvergenceResult:
        versions = self.store.get_plan_versions(self.session_id, limit=2)
        if len(versions) < 2:
            return ConvergenceResult(converged=False, reason="none", change_pct=1.0, regression=None)

        current, previous = versions[0], versions[1]
        if current.plan_sha == previous.plan_sha:
            latest_critic = self.store.get_latest_critic_turn(self.session_id)
            major = latest_critic.major_issues_remaining if latest_critic else None
            if major is not None and major > 0:
                return ConvergenceResult(
                    converged=False,
                    reason="major_issues_open",
                    change_pct=0.0,
                    regression=None,
                    major_issues=major,
                )
            return ConvergenceResult(
                converged=True,
                reason="identical",
                change_pct=0.0,
                regression=None,
                major_issues=0,
            )

        prev_s = self._extract_structure(previous.plan_content)
        curr_s = self._extract_structure(current.plan_content)
        regressions: list[str] = []
        if prev_s.todo_count > 0 and curr_s.todo_count < prev_s.todo_count * 0.8:
            regressions.append(f"TODOs dropped {prev_s.todo_count} -> {curr_s.todo_count}")
        if prev_s.file_path_count > 0 and curr_s.file_path_count < prev_s.file_path_count * 0.8:
            regressions.append(
                f"file references dropped {prev_s.file_path_count} -> {curr_s.file_path_count}"
            )
        if prev_s.has_goals and not curr_s.has_goals:
            regressions.append("Goals section removed")
        if prev_s.has_non_goals and not curr_s.has_non_goals:
            regressions.append("Non-Goals section removed")
        if prev_s.has_todos_section and not curr_s.has_todos_section:
            regressions.append("TODO/Implementation section removed")

        if regressions:
            return ConvergenceResult(
                converged=False,
                reason="regression",
                change_pct=1.0,
                regression="; ".join(regressions),
            )

        change_pct = self._compute_change_pct(previous.plan_content, current.plan_content)
        latest_critic = self.store.get_latest_critic_turn(self.session_id)
        major = latest_critic.major_issues_remaining if latest_critic else None
        if major is not None and major > 0:
            return ConvergenceResult(
                converged=False,
                reason="major_issues_open",
                change_pct=change_pct,
                regression=None,
                major_issues=major,
            )

        if change_pct < self.config.convergence_threshold:
            return ConvergenceResult(
                converged=True,
                reason="below_threshold",
                change_pct=change_pct,
                regression=None,
                major_issues=0 if major is None else major,
            )

        session = self.get_session()
        if session.current_round >= self.config.max_adversarial_rounds:
            return ConvergenceResult(
                converged=True,
                reason="max_rounds",
                change_pct=change_pct,
                regression=None,
                major_issues=major,
            )

        return ConvergenceResult(
            converged=False,
            reason="none",
            change_pct=change_pct,
            regression=None,
            major_issues=major,
        )
