from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .models import PlanningCommand, PlanningSession


class StorePlanningCommandsMixin:
    # =========================================================================
    # Planning commands (executor command log)
    # =========================================================================

    def _row_to_planning_command(self, row: sqlite3.Row | None) -> PlanningCommand | None:
        if row is None:
            return None
        return PlanningCommand(**dict(row))

    def create_planning_command(
        self,
        session_id: str,
        command_id: str,
        payload_json: str,
        *,
        command: str | None = None,
    ) -> PlanningCommand:
        """Create a queued command-log row for a session command."""
        jid = str(uuid4())
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_commands
                    (id, session_id, command, command_id, status, payload_json, result_snapshot_json,
                     started_at, completed_at, last_error, attempt_count, worker_id,
                     lease_expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', ?, NULL, NULL, NULL, NULL, 0, NULL, NULL, ?, ?)
                """,
                (jid, session_id, command, command_id, payload_json, now, now),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (jid,)).fetchone()
            command_row = self._row_to_planning_command(row)
            if command_row is None:
                raise ValueError("Failed to create planning command")
            return command_row

    def get_planning_command(self, command_row_id: str) -> PlanningCommand | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def get_planning_command_by_command_id(self, session_id: str, command_id: str) -> PlanningCommand | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ? AND command_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, command_id),
            ).fetchone()
            return self._row_to_planning_command(row)

    def get_active_planning_command(self, session_id: str) -> PlanningCommand | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ? AND status IN ('queued', 'running', 'finalizing')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return self._row_to_planning_command(row)

    def get_live_running_planning_command(self, session_id: str) -> PlanningCommand | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ?
                  AND status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, now),
            ).fetchone()
            return self._row_to_planning_command(row)

    def list_planning_commands(self, session_id: str, limit: int = 50) -> list[PlanningCommand]:
        capped_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, capped_limit),
            ).fetchall()
            return [PlanningCommand(**dict(row)) for row in rows]

    def claim_next_planning_command(self, worker_id: str, lease_seconds: int) -> PlanningCommand | None:
        """Claim oldest queued job atomically and increment attempt_count."""
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_expires = (now_dt + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    UPDATE planning_commands
                    SET status = 'running',
                        lease_expires_at = ?,
                        worker_id = ?,
                        attempt_count = attempt_count + 1,
                        updated_at = ?
                    WHERE id = (
                        SELECT id
                        FROM planning_commands
                        WHERE status = 'queued'
                        ORDER BY created_at ASC
                        LIMIT 1
                    )
                      AND status = 'queued'
                    RETURNING *
                    """,
                    (lease_expires, worker_id, now),
                ).fetchone()
                if row is not None:
                    return PlanningCommand(**dict(row))
                return None
            except sqlite3.OperationalError as exc:
                # Older SQLite builds may not support RETURNING.
                if "RETURNING" not in str(exc).upper():
                    raise
                cur = conn.execute(
                    """
                    UPDATE planning_commands
                    SET status = 'running',
                        lease_expires_at = ?,
                        worker_id = ?,
                        attempt_count = attempt_count + 1,
                        updated_at = ?
                    WHERE id = (
                        SELECT id
                        FROM planning_commands
                        WHERE status = 'queued'
                        ORDER BY created_at ASC
                        LIMIT 1
                    )
                      AND status = 'queued'
                    """,
                    (lease_expires, worker_id, now),
                )
                if int(cur.rowcount or 0) != 1:
                    return None
                row = conn.execute(
                    """
                    SELECT * FROM planning_commands
                    WHERE status = 'running' AND worker_id = ? AND updated_at = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (worker_id, now),
                ).fetchone()
                return self._row_to_planning_command(row)

    def complete_planning_command_row(self, command_row_id: str) -> PlanningCommand | None:
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE planning_commands
                SET status = 'completed',
                    lease_expires_at = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, command_row_id),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def fail_planning_command_row(self, command_row_id: str) -> PlanningCommand | None:
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE planning_commands
                SET status = 'failed',
                    lease_expires_at = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, command_row_id),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def requeue_stale_planning_commands(self, max_attempts: int = 3) -> int:
        """
        Requeue expired running jobs (or fail when attempts exhausted).

        NOTE: attempt_count increments only on claim, never on requeue.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                UPDATE planning_commands
                SET status = CASE
                        WHEN attempt_count >= ? THEN 'failed'
                        ELSE 'queued'
                    END,
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (max(1, int(max_attempts)), now, now),
            )
            return int(cur.rowcount or 0)

    def reserve_planning_command(
        self,
        *,
        session_id: str,
        command: str,
        command_id: str,
        payload_json: str,
        lease_seconds: int,
        allowed_commands_by_status: dict[str, set[str]] | None = None,
    ) -> tuple[PlanningCommand | None, PlanningCommand | None, PlanningCommand | None, str | None]:
        """
        Reserve a command row atomically.

        Returns (reserved, replay, active_conflict, reason).
        """
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_expires = (now_dt + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            replay_row = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ? AND command_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, command_id),
            ).fetchone()
            if replay_row is not None:
                return None, PlanningCommand(**dict(replay_row)), None, "duplicate_command"

            session_row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                return None, None, None, "session_not_found"
            session = PlanningSession(**dict(session_row))
            if allowed_commands_by_status is not None:
                allowed = allowed_commands_by_status.get(session.status, set())
                if command not in allowed and not (command == "cancel" and bool(session.current_command_id)):
                    return None, None, None, "invalid_status"

            active_row = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ?
                  AND status IN ('running', 'finalizing')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, now),
            ).fetchone()
            if active_row is not None:
                if command != "cancel":
                    return None, None, PlanningCommand(**dict(active_row)), "processing_lock"

            jid = str(uuid4())
            conn.execute(
                """
                INSERT INTO planning_commands
                    (id, session_id, command, command_id, status, payload_json,
                     result_snapshot_json, started_at, completed_at, last_error,
                     attempt_count, worker_id, lease_expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'running', ?, NULL, ?, NULL, NULL, 0, NULL, ?, ?, ?)
                """,
                (jid, session_id, command, command_id, payload_json, now, lease_expires, now, now),
            )
            conn.execute(
                "UPDATE planning_sessions SET current_command_id = ?, updated_at = ? WHERE id = ?",
                (jid, now, session_id),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (jid,)).fetchone()
            return self._row_to_planning_command(row), None, None, None

    def renew_planning_command_lease(self, command_row_id: str, lease_seconds: int) -> PlanningCommand | None:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        lease_expires = (now_dt + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE planning_commands
                SET lease_expires_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status IN ('running', 'finalizing')
                """,
                (lease_expires, now, command_row_id),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def begin_planning_command_finalize(self, command_row_id: str) -> PlanningCommand | None:
        now = self._now()
        with self._connect() as conn:
            try:
                row = conn.execute(
                    """
                    UPDATE planning_commands
                    SET status = 'finalizing',
                        updated_at = ?
                    WHERE id = ?
                      AND status = 'running'
                    RETURNING *
                    """,
                    (now, command_row_id),
                ).fetchone()
                if row is not None:
                    return PlanningCommand(**dict(row))
            except sqlite3.OperationalError as exc:
                if "RETURNING" not in str(exc).upper():
                    raise
                conn.execute(
                    """
                    UPDATE planning_commands
                    SET status = 'finalizing',
                        updated_at = ?
                    WHERE id = ?
                      AND status = 'running'
                    """,
                    (now, command_row_id),
                )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def complete_planning_command(
        self,
        *,
        command_row_id: str,
        result_snapshot_json: str,
    ) -> PlanningCommand | None:
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE planning_commands
                SET status = 'completed',
                    result_snapshot_json = ?,
                    lease_expires_at = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (result_snapshot_json, now, now, command_row_id),
            )
            conn.execute(
                """
                UPDATE planning_sessions
                SET current_command_id = NULL,
                    updated_at = ?
                WHERE current_command_id = ?
                """,
                (now, command_row_id),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def fail_planning_command(self, command_row_id: str, *, reason: str | None = None) -> PlanningCommand | None:
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE planning_commands
                SET status = 'failed',
                    last_error = ?,
                    lease_expires_at = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status IN ('running', 'finalizing')
                """,
                (reason, now, now, command_row_id),
            )
            conn.execute(
                """
                UPDATE planning_sessions
                SET current_command_id = NULL,
                    updated_at = ?
                WHERE current_command_id = ?
                """,
                (now, command_row_id),
            )
            row = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (command_row_id,)).fetchone()
            return self._row_to_planning_command(row)

    def cancel_active_planning_command(self, session_id: str) -> PlanningCommand | None:
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM planning_commands
                WHERE session_id = ?
                  AND status = 'running'
                  AND command != 'cancel'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["id"])
            conn.execute(
                """
                UPDATE planning_commands
                SET status = 'cancelled',
                    last_error = 'cancelled',
                    lease_expires_at = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )
            conn.execute(
                """
                UPDATE planning_sessions
                SET current_command_id = NULL,
                    updated_at = ?
                WHERE current_command_id = ?
                """,
                (now, job_id),
            )
            updated = conn.execute("SELECT * FROM planning_commands WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_planning_command(updated)

    def fail_expired_running_planning_commands(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id FROM planning_commands
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            if not rows:
                return 0
            ids = [str(row["id"]) for row in rows]
            placeholders = ", ".join(["?"] * len(ids))
            conn.execute(
                f"""
                UPDATE planning_commands
                SET status = 'failed',
                    last_error = 'timeout',
                    lease_expires_at = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                [now, now, *ids],
            )
            conn.execute(
                f"""
                UPDATE planning_sessions
                SET current_command_id = NULL,
                    updated_at = ?
                WHERE current_command_id IN ({placeholders})
                """,
                [now, *ids],
            )
            return len(ids)
