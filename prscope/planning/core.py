"""
Pure planning core logic: state machine, versioning, and convergence checks.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from ..config import PlanningConfig
from ..store import PlanningSession, PlanningTurn, PlanVersion, Store


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


class InvalidCommandError(RuntimeError):
    """Raised when a command is invalid for current state."""


_UNSET = object()


class PlanningCore:
    @staticmethod
    def _render_plan_markdown_from_payload(payload: dict[str, Any]) -> str:
        title = str(payload.get("title", "Plan")).strip() or "Plan"
        lines = [f"# {title}", ""]
        sections = (
            ("summary", "Summary"),
            ("goals", "Goals"),
            ("non_goals", "Non-Goals"),
            ("files_changed", "Files Changed"),
            ("architecture", "Architecture"),
            ("implementation_steps", "Implementation Steps"),
            ("test_strategy", "Test Strategy"),
            ("rollback_plan", "Rollback Plan"),
            ("open_questions", "Open Questions"),
        )
        for key, label in sections:
            content = str(payload.get(key, "") or "").strip()
            if not content:
                continue
            lines.append(f"## {label}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    """Deterministic planning state and convergence logic."""

    ALLOWED_TRANSITIONS: dict[str, set[str]] = {
        "draft": {"draft", "refining", "error"},
        "refining": {"refining", "converged", "error"},
        "converged": {"approved", "refining", "error"},
        "approved": {"approved", "error"},
        "error": {"draft"},
    }
    VALID_COMMANDS: dict[str, set[str]] = {
        "draft": {"message", "export"},
        "refining": {"message", "run_round", "export"},
        "converged": {"run_round", "approve", "export"},
        "approved": {"export"},
        "error": {"reset"},
    }
    WORK_STATES: set[str] = {"draft", "refining"}

    def __init__(self, store: Store, session_id: str, config: PlanningConfig):
        self.store = store
        self.session_id = session_id
        self.config = config

    @staticmethod
    def _normalize_status(status: str) -> str:
        legacy_to_canonical = {
            "created": "draft",
            "preparing": "draft",
            "discovery": "draft",
            "discovering": "draft",
            "drafting": "draft",
            "exported": "approved",
        }
        return legacy_to_canonical.get(status, status)

    def get_session(self):
        session = self.store.get_planning_session(self.session_id)
        if session is None:
            raise ValueError(f"Planning session not found: {self.session_id}")
        return session

    def get_current_plan(self) -> PlanVersion | None:
        versions = self.store.get_plan_versions(self.session_id, limit=1)
        if not versions:
            return None
        current = versions[0]
        if current.plan_json and (not current.plan_content or not current.plan_content.strip()):
            try:
                payload = json.loads(current.plan_json)
                if isinstance(payload, dict):
                    current.plan_content = self._render_plan_markdown_from_payload(payload)
            except Exception:  # noqa: BLE001
                pass
        return current

    def save_plan_version(
        self,
        content: str,
        round_number: int,
        *,
        plan_document: Any | None = None,
        changed_sections: list[str] | None = None,
    ) -> PlanVersion:
        if plan_document is not None:
            payload = dict(getattr(plan_document, "__dict__", {}))
            plan_json = json.dumps(payload)
            content = self._render_plan_markdown_from_payload(payload)
        else:
            plan_json = None
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
            plan_json=plan_json,
            plan_sha=plan_sha,
            changed_sections=json.dumps(changed_sections or []),
            diff_from_previous=diff_from_previous,
        )
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

    @classmethod
    def allowed_commands_for(cls, status: str) -> list[str]:
        return sorted(cls.VALID_COMMANDS.get(status, set()))

    def validate_command(self, command_type: str, session: PlanningSession | None = None) -> None:
        current = session or self.get_session()
        allowed = self.VALID_COMMANDS.get(current.status, set())
        if command_type not in allowed:
            raise InvalidCommandError(
                f"Command '{command_type}' is invalid for status '{current.status}'. Allowed: {sorted(allowed)}"
            )

    @staticmethod
    def _parse_json(raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def _build_snapshot(self, session: PlanningSession) -> dict[str, Any]:
        pending_questions = self._parse_json(session.pending_questions_json, None)
        if pending_questions is not None and not isinstance(pending_questions, list):
            pending_questions = None
        active_tool_calls = self._parse_json(session.active_tool_calls_json, [])
        if not isinstance(active_tool_calls, list):
            active_tool_calls = []
        completed_groups_raw = self._parse_json(getattr(session, "completed_tool_call_groups_json", None), [])
        completed_groups: list[Any] = []
        if isinstance(completed_groups_raw, list):
            for idx, group in enumerate(completed_groups_raw):
                if isinstance(group, dict) and "tools" in group:
                    completed_groups.append(group)
                elif isinstance(group, list):
                    tools = [e for e in group if isinstance(e, dict)]
                    first_ts = str(tools[0].get("created_at", "")) if tools else ""
                    completed_groups.append(
                        {
                            "sequence": idx,
                            "created_at": first_ts,
                            "tools": tools,
                        }
                    )
        return {
            "type": "session_state",
            "v": 1,
            "status": session.status,
            "phase_message": session.phase_message,
            "is_processing": bool(session.is_processing),
            "current_round": session.current_round,
            "pending_questions": pending_questions,
            "active_tool_calls": active_tool_calls,
            "completed_tool_call_groups": completed_groups,
        }

    def transition_and_snapshot(
        self,
        new_status: str,
        *,
        phase_message: str | None = None,
        pending_questions_json: str | None | object = _UNSET,
        active_tool_calls_json: str | None | object = _UNSET,
        current_round: int | None = None,
        allow_round_stability: bool = False,
    ) -> dict[str, Any]:
        with self.store._connect() as conn:  # noqa: SLF001
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (self.session_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Planning session not found: {self.session_id}")
            session = PlanningSession(**dict(row))
            old_status = self._normalize_status(session.status)
            new_status = self._normalize_status(new_status)
            allowed = self.ALLOWED_TRANSITIONS.get(old_status, set())
            if new_status != old_status and new_status not in allowed:
                raise InvalidTransitionError(f"Invalid transition {old_status} -> {new_status}")

            if old_status == "refining" and new_status == "refining":
                if allow_round_stability:
                    current_round = session.current_round
                elif current_round is None or current_round <= session.current_round:
                    raise InvalidTransitionError("refining -> refining requires current_round > existing round")
            elif current_round is not None and current_round != session.current_round:
                raise InvalidTransitionError("current_round can only change on refining -> refining transitions")

            next_pending_questions_json = session.pending_questions_json
            if pending_questions_json is not _UNSET:
                next_pending_questions_json = pending_questions_json
            if new_status != "draft":
                next_pending_questions_json = None
            if new_status == "draft":
                if next_pending_questions_json is not None and phase_message is not None:
                    raise InvalidTransitionError("discovery coherence violated: cannot show questions while processing")
                if phase_message is not None:
                    next_pending_questions_json = None

            next_active_tool_calls_json = session.active_tool_calls_json
            if active_tool_calls_json is not _UNSET:
                next_active_tool_calls_json = active_tool_calls_json

            next_round = session.current_round if current_round is None else current_round
            is_processing = int(new_status in self.WORK_STATES and phase_message is not None)
            was_processing = bool(session.is_processing)
            now = datetime.now(timezone.utc).isoformat()
            if not was_processing and is_processing:
                processing_started_at = now
            elif was_processing and not is_processing:
                processing_started_at = None
            else:
                processing_started_at = session.processing_started_at

            updated_at = datetime.utcnow().isoformat() + "Z"
            conn.execute(
                """
                UPDATE planning_sessions
                SET status = ?,
                    phase_message = ?,
                    is_processing = ?,
                    pending_questions_json = ?,
                    active_tool_calls_json = ?,
                    processing_started_at = ?,
                    current_round = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    new_status,
                    phase_message,
                    is_processing,
                    next_pending_questions_json,
                    next_active_tool_calls_json,
                    processing_started_at,
                    next_round,
                    updated_at,
                    self.session_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (self.session_id,),
            ).fetchone()
            if updated is None:
                raise ValueError(f"Planning session not found: {self.session_id}")
            return self._build_snapshot(PlanningSession(**dict(updated)))

    def transition(self, new_status: str) -> None:
        self.transition_and_snapshot(new_status, phase_message=None)

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
        self.transition_and_snapshot("approved", phase_message=None)

    @classmethod
    def reconcile_stuck_sessions(
        cls,
        store: Store,
        config: PlanningConfig,
        timeout_seconds: int = 300,
    ) -> int:
        now = datetime.now(timezone.utc)
        reconciled = 0
        sessions = store.list_planning_sessions(limit=10_000)
        for session in sessions:
            if not bool(session.is_processing):
                continue
            if session.status not in cls.WORK_STATES:
                continue
            if not session.processing_started_at:
                continue
            try:
                started = datetime.fromisoformat(session.processing_started_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if (now - started).total_seconds() < timeout_seconds:
                continue
            core = PlanningCore(store, session.id, config)
            core.transition_and_snapshot("error", phase_message=None)
            reconciled += 1
        return reconciled

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
            regressions.append(f"file references dropped {prev_s.file_path_count} -> {curr_s.file_path_count}")
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
