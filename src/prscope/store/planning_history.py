from __future__ import annotations

import json
import os
import sqlite3

from .models import PlanningRoundMetrics, PlanningTurn, PlanVersion


class StorePlanningHistoryMixin:
    # =========================================================================
    # Planning turns / versions / round metrics
    # =========================================================================

    def _row_to_planning_turn(self, row: sqlite3.Row) -> PlanningTurn:
        data = dict(row)
        violations_raw = data.get("hard_constraint_violations")
        violations: list[str] | None = None
        if violations_raw:
            try:
                loaded = json.loads(violations_raw)
                if isinstance(loaded, list):
                    violations = [str(v) for v in loaded]
            except json.JSONDecodeError:
                violations = None

        return PlanningTurn(
            id=data.get("id"),
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            round=data["round"],
            major_issues_remaining=data.get("major_issues_remaining"),
            minor_issues_remaining=data.get("minor_issues_remaining"),
            hard_constraint_violations=violations,
            parse_error=data.get("parse_error"),
            created_at=data.get("created_at", ""),
            sequence=data.get("sequence"),
        )

    def add_round_metrics(
        self,
        *,
        session_id: str,
        repo_name: str | None = None,
        round_number: int,
        author_prompt_tokens: int | None,
        author_completion_tokens: int | None,
        critic_prompt_tokens: int | None,
        critic_completion_tokens: int | None,
        max_prompt_tokens: int | None,
        major_issues: int | None,
        minor_issues: int | None,
        critic_confidence: float | None,
        vagueness_score: float | None,
        citation_count: int | None,
        constraint_violations: list[str] | None,
        resolved_since_last_round: list[str] | None,
        clarifications_this_round: int | None,
        call_cost_usd: float | None,
        issues_resolved: int | None = None,
        issues_introduced: int | None = None,
        net_improvement: int | None = None,
        model_costs: dict[str, float] | None = None,
        time_to_first_tool_call: int | None = None,
        grounding_ratio: float | None = None,
        static_injection_tokens_pct: float | None = None,
        rejected_for_no_discovery: int | None = None,
        rejected_for_grounding: int | None = None,
        rejected_for_budget: int | None = None,
        average_read_depth_per_round: float | None = None,
        time_between_tool_calls: float | None = None,
        rejection_reasons: list[dict[str, str]] | None = None,
        plan_quality_score: float | None = None,
        unsupported_claims_count: int | None = None,
        missing_evidence_count: int | None = None,
    ) -> PlanningRoundMetrics:
        stamp = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_round_metrics (
                    session_id, round, timestamp,
                    author_prompt_tokens, author_completion_tokens,
                    critic_prompt_tokens, critic_completion_tokens,
                    max_prompt_tokens, major_issues, minor_issues,
                    critic_confidence, vagueness_score, citation_count,
                    constraint_violations_json, resolved_since_last_round_json,
                    clarifications_this_round, call_cost_usd,
                    issues_resolved, issues_introduced, net_improvement,
                    time_to_first_tool_call, grounding_ratio, static_injection_tokens_pct,
                    rejected_for_no_discovery, rejected_for_grounding, rejected_for_budget,
                    average_read_depth_per_round, time_between_tool_calls, rejection_reasons_json,
                    plan_quality_score, unsupported_claims_count, missing_evidence_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    round_number,
                    stamp,
                    author_prompt_tokens,
                    author_completion_tokens,
                    critic_prompt_tokens,
                    critic_completion_tokens,
                    max_prompt_tokens,
                    major_issues,
                    minor_issues,
                    critic_confidence,
                    vagueness_score,
                    citation_count,
                    json.dumps(constraint_violations or []),
                    json.dumps(resolved_since_last_round or []),
                    clarifications_this_round,
                    call_cost_usd,
                    issues_resolved,
                    issues_introduced,
                    net_improvement,
                    time_to_first_tool_call,
                    grounding_ratio,
                    static_injection_tokens_pct,
                    rejected_for_no_discovery,
                    rejected_for_grounding,
                    rejected_for_budget,
                    average_read_depth_per_round,
                    time_between_tool_calls,
                    json.dumps(rejection_reasons or []),
                    plan_quality_score,
                    unsupported_claims_count,
                    missing_evidence_count,
                ),
            )
            row = conn.execute("SELECT * FROM planning_round_metrics WHERE id = last_insert_rowid()").fetchone()
            metrics = PlanningRoundMetrics(**dict(row))
        if repo_name:
            # Atomic append for per-round analytics log.
            log_path = self._rounds_log_path(repo_name)
            line = (
                json.dumps(
                    {
                        "session_id": session_id,
                        "round": round_number,
                        "timestamp": stamp,
                        "author_prompt_tokens": author_prompt_tokens,
                        "author_completion_tokens": author_completion_tokens,
                        "critic_prompt_tokens": critic_prompt_tokens,
                        "critic_completion_tokens": critic_completion_tokens,
                        "max_prompt_tokens": max_prompt_tokens,
                        "major_issues": major_issues,
                        "minor_issues": minor_issues,
                        "critic_confidence": critic_confidence,
                        "vagueness_score": vagueness_score,
                        "citation_count": citation_count,
                        "constraint_violations": constraint_violations or [],
                        "resolved_since_last_round": resolved_since_last_round or [],
                        "clarifications_this_round": clarifications_this_round,
                        "call_cost_usd": call_cost_usd,
                        "issues_resolved": issues_resolved,
                        "issues_introduced": issues_introduced,
                        "net_improvement": net_improvement,
                        "model_costs": model_costs or {},
                        "time_to_first_tool_call": time_to_first_tool_call,
                        "grounding_ratio": grounding_ratio,
                        "static_injection_tokens_pct": static_injection_tokens_pct,
                        "rejected_for_no_discovery": rejected_for_no_discovery,
                        "rejected_for_grounding": rejected_for_grounding,
                        "rejected_for_budget": rejected_for_budget,
                        "average_read_depth_per_round": average_read_depth_per_round,
                        "time_between_tool_calls": time_between_tool_calls,
                        "rejection_reasons": rejection_reasons or [],
                        "plan_quality_score": plan_quality_score,
                        "unsupported_claims_count": unsupported_claims_count,
                        "missing_evidence_count": missing_evidence_count,
                    }
                )
                + "\n"
            )
            fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        return metrics

    def get_round_metrics(self, session_id: str) -> list[PlanningRoundMetrics]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM planning_round_metrics WHERE session_id = ? ORDER BY round ASC, id ASC",
                (session_id,),
            ).fetchall()
            return [PlanningRoundMetrics(**dict(row)) for row in rows]

    def increment_event_seq(self, session_id: str) -> int:
        """Atomically increment and return the next event sequence number for a session."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE planning_sessions SET event_seq = event_seq + 1 WHERE id = ?",
                (session_id,),
            )
            row = conn.execute(
                "SELECT event_seq FROM planning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return int(row["event_seq"]) if row else 1

    def add_planning_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        round_number: int,
        major_issues_remaining: int | None = None,
        minor_issues_remaining: int | None = None,
        hard_constraint_violations: list[str] | None = None,
        parse_error: str | None = None,
        sequence: int | None = None,
    ) -> PlanningTurn:
        """Insert a planning conversation turn."""
        violations_json = json.dumps(hard_constraint_violations or [])
        if sequence is None:
            sequence = self.increment_event_seq(session_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_turns
                    (session_id, role, content, round, major_issues_remaining,
                     minor_issues_remaining, hard_constraint_violations, parse_error, created_at, sequence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    round_number,
                    major_issues_remaining,
                    minor_issues_remaining,
                    violations_json,
                    parse_error,
                    self._now(),
                    sequence,
                ),
            )
            row = conn.execute("SELECT * FROM planning_turns WHERE id = last_insert_rowid()").fetchone()
            return self._row_to_planning_turn(row)

    def get_planning_turns(self, session_id: str, round_number: int | None = None) -> list[PlanningTurn]:
        """Fetch planning turns for a session."""
        with self._connect() as conn:
            if round_number is None:
                rows = conn.execute(
                    """
                    SELECT * FROM planning_turns
                    WHERE session_id = ?
                    ORDER BY round ASC, id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM planning_turns
                    WHERE session_id = ? AND round = ?
                    ORDER BY id ASC
                    """,
                    (session_id, round_number),
                ).fetchall()
            return [self._row_to_planning_turn(row) for row in rows]

    def get_latest_critic_turn(self, session_id: str) -> PlanningTurn | None:
        """Get the latest critic turn for semantic convergence checks."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM planning_turns
                WHERE session_id = ? AND role = 'critic'
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return self._row_to_planning_turn(row) if row else None

    def save_plan_version(
        self,
        session_id: str,
        round_number: int,
        plan_content: str,
        plan_sha: str,
        plan_json: str | None = None,
        changed_sections: str | None = None,
        diff_from_previous: str | None = None,
        convergence_score: float | None = None,
    ) -> PlanVersion:
        """Persist a plan snapshot."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plan_versions
                    (
                        session_id,
                        round,
                        plan_content,
                        plan_json,
                        plan_sha,
                        changed_sections,
                        diff_from_previous,
                        convergence_score,
                        created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    round_number,
                    plan_content,
                    plan_json,
                    plan_sha,
                    changed_sections,
                    diff_from_previous,
                    convergence_score,
                    self._now(),
                ),
            )
            row = conn.execute("SELECT * FROM plan_versions WHERE id = last_insert_rowid()").fetchone()
            return PlanVersion(**dict(row))

    def update_plan_version_convergence(self, session_id: str, round_number: int, convergence_score: float) -> None:
        """Update the convergence score of an existing plan version (after check_convergence)."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE plan_versions
                SET convergence_score = ?
                WHERE session_id = ? AND round = ?
                """,
                (convergence_score, session_id, round_number),
            )

    def get_plan_versions(self, session_id: str, limit: int = 20) -> list[PlanVersion]:
        """Get latest plan versions for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM plan_versions
                WHERE session_id = ?
                ORDER BY round DESC, id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [PlanVersion(**dict(row)) for row in rows]

    def get_plan_version(self, session_id: str, round_number: int) -> PlanVersion | None:
        """Get a specific plan version by round."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM plan_versions
                WHERE session_id = ? AND round = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id, round_number),
            ).fetchone()
            return PlanVersion(**dict(row)) if row else None
