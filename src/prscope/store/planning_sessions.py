from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import PlanningSession

_UNSET = object()
PROTECTED_SESSION_FIELDS = {
    "status",
    "is_processing",
    "pending_questions_json",
    "phase_message",
    "active_tool_calls_json",
    "completed_tool_call_groups_json",
    "processing_started_at",
}


class StorePlanningSessionsMixin:
    # =========================================================================
    # Planning sessions
    # =========================================================================

    def create_planning_session(
        self,
        repo_name: str,
        title: str,
        requirements: str,
        seed_type: str,
        author_model: str | None = None,
        critic_model: str | None = None,
        seed_ref: str | None = None,
        no_recall: bool = False,
        status: str = "draft",
        session_id: str | None = None,
    ) -> PlanningSession:
        """Create and persist a planning session."""
        sid = session_id or str(uuid4())
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_sessions
                    (id, repo_name, title, requirements, author_model, critic_model, status, seed_type, seed_ref,
                     current_round, pending_questions_json, phase_message, is_processing,
                     processing_started_at, last_commands_json, active_tool_calls_json,
                     completed_tool_call_groups_json, current_command_id,
                     session_total_cost_usd, max_prompt_tokens, confidence_trend,
                     converged_early, no_recall,
                     clarifications_log_json, diagnostics_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    repo_name,
                    title,
                    requirements,
                    author_model,
                    critic_model,
                    status,
                    seed_type,
                    seed_ref,
                    0,
                    None,
                    None,
                    0,
                    None,
                    "{}",
                    "[]",
                    "[]",
                    None,
                    0.0,
                    0,
                    0.0,
                    0,
                    int(bool(no_recall)),
                    "[]",
                    "{}",
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            return PlanningSession(**dict(row))

    def get_planning_session(self, session_id: str) -> PlanningSession | None:
        """Get a planning session by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return PlanningSession(**dict(row)) if row else None

    def delete_planning_session(self, session_id: str) -> bool:
        """Delete a session and all its turns and plan versions. Returns True if found."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM planning_sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM planning_turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM plan_versions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM planning_sessions WHERE id = ?", (session_id,))
            return True

    def clear_planning_session_history(self, session_id: str) -> bool:
        """Delete user-visible planning artifacts while keeping the session row."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM planning_sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM planning_turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM plan_versions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM planning_round_metrics WHERE session_id = ?", (session_id,))
            return True

    def delete_all_planning_sessions(self, repo_name: str | None = None) -> int:
        """Delete all sessions (optionally filtered by repo). Returns count deleted."""
        with self._connect() as conn:
            if repo_name:
                ids = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM planning_sessions WHERE repo_name = ?", (repo_name,)
                    ).fetchall()
                ]
            else:
                ids = [row[0] for row in conn.execute("SELECT id FROM planning_sessions").fetchall()]
            for sid in ids:
                conn.execute("DELETE FROM planning_turns WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM plan_versions WHERE session_id = ?", (sid,))
            if repo_name:
                conn.execute("DELETE FROM planning_sessions WHERE repo_name = ?", (repo_name,))
            else:
                conn.execute("DELETE FROM planning_sessions")
            return len(ids)

    def list_planning_sessions(
        self,
        repo_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[PlanningSession]:
        """List planning sessions with optional filters."""
        query = "SELECT * FROM planning_sessions WHERE 1=1"
        params: list[Any] = []
        if repo_name is not None:
            query += " AND repo_name = ?"
            params.append(repo_name)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [PlanningSession(**dict(row)) for row in rows]

    def search_sessions(
        self,
        query: str,
        repo_name: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search historical planning sessions with BM25 + recency boost."""
        if len(query.split()) < 6:
            return []
        if limit <= 0:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return []

        sql = """
            SELECT
                ps.id AS session_id,
                ps.repo_name AS repo_name,
                ps.title AS title,
                ps.requirements AS requirements,
                ps.created_at AS created_at,
                COALESCE(pv.plan_content, '') AS plan_summary
            FROM planning_sessions ps
            LEFT JOIN plan_versions pv
                ON pv.id = (
                    SELECT pv2.id
                    FROM plan_versions pv2
                    WHERE pv2.session_id = ps.id
                    ORDER BY pv2.round DESC, pv2.id DESC
                    LIMIT 1
                )
            WHERE (? IS NULL OR ps.repo_name = ?)
            ORDER BY ps.created_at ASC
        """

        with self._connect() as conn:
            rows = conn.execute(sql, (repo_name, repo_name)).fetchall()
        if not rows:
            return []

        def _parse_created(value: str) -> datetime:
            raw = (value or "").strip()
            if not raw:
                return datetime.utcnow()
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return datetime.utcnow()
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed

        corpus_tokens: list[list[str]] = []
        row_payloads: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            title = str(payload.get("title") or "")
            requirements = str(payload.get("requirements") or "")
            plan_summary = str(payload.get("plan_summary") or "")
            document = f"{title} {requirements} {plan_summary}".strip().lower()
            tokens = document.split()
            corpus_tokens.append(tokens if tokens else [""])
            payload["created_dt"] = _parse_created(str(payload.get("created_at") or ""))
            row_payloads.append(payload)

        query_tokens = query.lower().split()
        if not query_tokens:
            return []

        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        now = datetime.utcnow()

        ranked: list[dict[str, Any]] = []
        for payload, bm25_score in zip(row_payloads, scores):
            created_dt: datetime = payload["created_dt"]
            days_old = (now - created_dt).days
            if days_old < 30:
                multiplier = 1.2
            elif days_old < 90:
                multiplier = 1.1
            else:
                multiplier = 1.0
            base_score = max(float(bm25_score), 0.0)
            final_score = base_score * multiplier
            summary_raw = str(payload.get("plan_summary") or "")
            snippet = summary_raw.replace("\n", " ")[:300]
            if len(summary_raw) > 300:
                snippet += "..."
            ranked.append(
                {
                    "session_id": str(payload.get("session_id") or ""),
                    "title": str(payload.get("title") or ""),
                    "repo_name": str(payload.get("repo_name") or ""),
                    "created_at": str(payload.get("created_at") or ""),
                    "score": final_score,
                    "summary_snippet": snippet,
                    "plan_summary": summary_raw,
                    "_created_dt": created_dt,
                }
            )

        ranked.sort(key=lambda item: (float(item["score"]), item["_created_dt"]), reverse=True)
        for item in ranked:
            item.pop("_created_dt", None)
        return ranked[:limit]

    def update_planning_session(
        self,
        session_id: str,
        *,
        _bypass_protection: bool = False,
        status: str | None = None,
        requirements: str | None = None,
        author_model: str | None = None,
        critic_model: str | None = None,
        current_round: int | None = None,
        seed_ref: str | None = None,
        session_total_cost_usd: float | None = None,
        max_prompt_tokens: int | None = None,
        confidence_trend: float | None = None,
        converged_early: int | None = None,
        clarifications_log_json: str | None = None,
        diagnostics_json: str | None | object = _UNSET,
        pending_questions_json: str | None | object = _UNSET,
        phase_message: str | None | object = _UNSET,
        is_processing: int | bool | object = _UNSET,
        processing_started_at: str | None | object = _UNSET,
        last_commands_json: str | None | object = _UNSET,
        active_tool_calls_json: str | None | object = _UNSET,
        completed_tool_call_groups_json: str | None | object = _UNSET,
        current_command_id: str | None | object = _UNSET,
    ) -> PlanningSession:
        """Update mutable fields on a planning session."""
        if not _bypass_protection:
            protected_updates: set[str] = set()
            if status is not None:
                protected_updates.add("status")
            if pending_questions_json is not _UNSET:
                protected_updates.add("pending_questions_json")
            if phase_message is not _UNSET:
                protected_updates.add("phase_message")
            if is_processing is not _UNSET:
                protected_updates.add("is_processing")
            if processing_started_at is not _UNSET:
                protected_updates.add("processing_started_at")
            if active_tool_calls_json is not _UNSET:
                protected_updates.add("active_tool_calls_json")
            if completed_tool_call_groups_json is not _UNSET:
                protected_updates.add("completed_tool_call_groups_json")
            if protected_updates:
                raise RuntimeError(
                    "Direct write to protected fields: "
                    f"{sorted(protected_updates)}. Use transition_and_snapshot() instead."
                )
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if requirements is not None:
            updates.append("requirements = ?")
            params.append(requirements)
        if author_model is not None:
            updates.append("author_model = ?")
            params.append(author_model)
        if critic_model is not None:
            updates.append("critic_model = ?")
            params.append(critic_model)
        if current_round is not None:
            updates.append("current_round = ?")
            params.append(current_round)
        if seed_ref is not None:
            updates.append("seed_ref = ?")
            params.append(seed_ref)
        if session_total_cost_usd is not None:
            updates.append("session_total_cost_usd = ?")
            params.append(session_total_cost_usd)
        if max_prompt_tokens is not None:
            updates.append("max_prompt_tokens = ?")
            params.append(max_prompt_tokens)
        if confidence_trend is not None:
            updates.append("confidence_trend = ?")
            params.append(confidence_trend)
        if converged_early is not None:
            updates.append("converged_early = ?")
            params.append(converged_early)
        if clarifications_log_json is not None:
            updates.append("clarifications_log_json = ?")
            params.append(clarifications_log_json)
        if diagnostics_json is not _UNSET:
            updates.append("diagnostics_json = ?")
            params.append(diagnostics_json)
        if pending_questions_json is not _UNSET:
            updates.append("pending_questions_json = ?")
            params.append(pending_questions_json)
        if phase_message is not _UNSET:
            updates.append("phase_message = ?")
            params.append(phase_message)
        if is_processing is not _UNSET:
            updates.append("is_processing = ?")
            params.append(int(bool(is_processing)))
        if processing_started_at is not _UNSET:
            updates.append("processing_started_at = ?")
            params.append(processing_started_at)
        if last_commands_json is not _UNSET:
            updates.append("last_commands_json = ?")
            params.append(last_commands_json)
        if active_tool_calls_json is not _UNSET:
            updates.append("active_tool_calls_json = ?")
            params.append(active_tool_calls_json)
        if completed_tool_call_groups_json is not _UNSET:
            updates.append("completed_tool_call_groups_json = ?")
            params.append(completed_tool_call_groups_json)
        if current_command_id is not _UNSET:
            updates.append("current_command_id = ?")
            params.append(current_command_id)

        updates.append("updated_at = ?")
        params.append(self._now())
        params.append(session_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE planning_sessions SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            row = conn.execute(
                "SELECT * FROM planning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Planning session not found: {session_id}")
            return PlanningSession(**dict(row))
